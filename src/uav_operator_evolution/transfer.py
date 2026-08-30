"""UAV-owned evidence projection into the domain-neutral mechanism protocol."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np

from operator_evolution_core.diagnosis import OperatorDiagnoser, OperatorProfile
from operator_evolution_core.evolution import population_fingerprint
from operator_evolution_core.memory import (
    AbstractMechanismContextV1,
    ExpectedMechanismEffectV1,
    MechanismBankV1,
    MechanismRecordV1,
    create_mechanism_bank_v1,
    create_mechanism_record_v1,
)
from operator_evolution_core.proposal import proposal_hash

from .diagnosis.features import UAV_FEATURE_CATALOG
from .domain.adapters import UAV_DOMAIN_ID
from .domain.uav_kit import UAVDomainKit
from .environment.environment import Environment2D
from .operators.catalog import manual_operator_specs
from .operators.registry import default_manual_operators
from .path.evaluator import PathEvaluator
from .reproducibility import derive_seed
from .search.executor import SearchExecutor
from .trajectory import TrajectoryRecorder


@dataclass(frozen=True, slots=True)
class UAVMechanismBankConfig:
    train_calls: int = 400
    validation_calls: int = 240
    train_instances: int = 60
    validation_instances: int = 40
    initializer_grid_resolution: float = 4.0

    def __post_init__(self) -> None:
        if min(
            self.train_calls,
            self.validation_calls,
            self.train_instances,
            self.validation_instances,
        ) <= 0:
            raise ValueError("UAV mechanism-bank budgets must be positive")
        if self.initializer_grid_resolution <= 0:
            raise ValueError("initializer_grid_resolution must be positive")


def _mechanism_tags(operator_id: str) -> tuple[str, ...]:
    if operator_id in {"waypoint_perturb", "segment_shift", "insert_waypoint"}:
        return ("diversify",)
    if operator_id in {"delete_waypoint", "shortcut", "smooth_segment"}:
        return ("intensify",)
    if operator_id == "obstacle_detour":
        return ("repair", "rollback")
    return ("repair", "diversify", "rollback")


def _ordinal_rate(value: float) -> str:
    if value < 0.10:
        return "low"
    if value < 0.30:
        return "medium"
    return "high"


def _abstract_context(profile: OperatorProfile) -> AbstractMechanismContextV1:
    feasibility = profile.feasibility_rate
    return AbstractMechanismContextV1(
        constraint_pressure=(
            "high"
            if profile.failure_modes or (feasibility is not None and feasibility < 0.95)
            else "low"
        ),
        stagnation=(
            "high"
            if profile.success_rate < 0.10
            else "medium" if profile.success_rate < 0.30 else "low"
        ),
        diversity=_ordinal_rate(profile.acceptance_rate),
        feasibility=(
            "unknown"
            if feasibility is None
            else "feasible" if feasibility >= 0.999 else "mixed"
        ),
        phase="unknown",
    )


def _expected_effect(tags: tuple[str, ...]) -> ExpectedMechanismEffectV1:
    if "repair" in tags:
        return ExpectedMechanismEffectV1(
            feasibility="improve",
            cost="preserve",
            diversity="increase" if "diversify" in tags else "preserve",
            locality="local",
        )
    if tags == ("diversify",):
        return ExpectedMechanismEffectV1(
            feasibility="risk",
            cost="improve",
            diversity="increase",
            locality="mixed",
        )
    return ExpectedMechanismEffectV1(
        feasibility="preserve",
        cost="improve",
        diversity="decrease",
        locality="local",
    )


def _failure_modes(profile: OperatorProfile) -> tuple[str, ...]:
    modes: list[str] = []
    if profile.failure_modes:
        modes.append("inapplicable-context")
    if profile.success_rate < 0.10:
        modes.append("low-yield")
    if profile.feasibility_gain_rate is not None and profile.feasibility_gain_rate < 0.0:
        modes.append("feasibility-regression")
    return tuple(modes)


def _evidence_strength(profile: OperatorProfile) -> float:
    sample_confidence = 1.0 - math.exp(-profile.attempts / 32.0)
    outcome_signal = 0.5 + 0.5 * profile.success_rate
    return min(1.0, sample_confidence * outcome_signal)


def _record_from_profile(
    profile: OperatorProfile,
    *,
    bank_seed: int,
    source_code_commit: str,
    source_population_fingerprint: str,
) -> MechanismRecordV1:
    tags = _mechanism_tags(profile.operator_id)
    return create_mechanism_record_v1(
        source_domain_id=UAV_DOMAIN_ID,
        mechanism_tags=tags,
        trigger_context=_abstract_context(profile),
        expected_effect=_expected_effect(tags),
        failure_modes=_failure_modes(profile),
        evidence_refs=(
            "profile:" + proposal_hash(profile.model_dump(mode="json")),
            "split:train",
            "split:validation",
        ),
        evidence_strength=_evidence_strength(profile),
        evidence_sample_count=profile.attempts,
        evidence_splits=("train", "validation"),
        bank_run_id=f"uav-mechanism-bank-{bank_seed}",
        bank_master_seed=bank_seed,
        source_operator_id=profile.operator_id,
        source_code_commit=source_code_commit,
        source_population_fingerprint=source_population_fingerprint,
    )


def _assert_disjoint(
    train: Sequence[Environment2D],
    validation: Sequence[Environment2D],
) -> None:
    for field_name in (
        "content_hash",
        "terminal_hash",
        "obstacle_layout_hash",
        "geometry_hash",
    ):
        train_values = {getattr(item, field_name) for item in train}
        validation_values = {getattr(item, field_name) for item in validation}
        if train_values & validation_values:
            raise ValueError(f"UAV bank splits overlap by {field_name}")


def build_uav_mechanism_bank(
    train_environments: Sequence[Environment2D],
    validation_environments: Sequence[Environment2D],
    *,
    bank_master_seeds: tuple[int, ...],
    source_code_commit: str,
    config: UAVMechanismBankConfig | None = None,
) -> MechanismBankV1:
    """Build an abstract bank from explicit train/validation capabilities only."""

    if not bank_master_seeds or len(set(bank_master_seeds)) != len(bank_master_seeds):
        raise ValueError("bank_master_seeds must be unique and non-empty")
    active = config or UAVMechanismBankConfig()
    train = tuple(train_environments[: active.train_instances])
    validation = tuple(validation_environments[: active.validation_instances])
    if not train or not validation:
        raise ValueError("UAV mechanism bank requires train and validation instances")
    _assert_disjoint(train, validation)
    specs = manual_operator_specs()
    kit = UAVDomainKit()
    fingerprint = population_fingerprint(tuple(specs), specs, kit)
    records: list[MechanismRecordV1] = []
    for bank_seed in bank_master_seeds:
        with TrajectoryRecorder(":memory:") as recorder:
            run_ids: list[str] = []
            for split_name, environments, calls in (
                ("train", train, active.train_calls),
                ("validation", validation, active.validation_calls),
            ):
                run_id = f"uav-bank-{bank_seed}-{split_name}"
                run_ids.append(run_id)
                for index, environment in enumerate(environments):
                    seed = derive_seed(
                        bank_seed,
                        "mechanism-bank",
                        split_name,
                        index,
                        environment.map_id,
                    )
                    SearchExecutor(
                        default_manual_operators(),
                        PathEvaluator(),
                        max_iterations=calls,
                        initializer_grid_resolution=active.initializer_grid_resolution,
                        recorder=recorder,
                    ).run(
                        environment,
                        np.random.default_rng(seed),
                        run_id=run_id,
                    )
                recorder.update_delayed_rewards((5, 10, 20), run_id=run_id)
            traces = [
                trace
                for run_id in run_ids
                for trace in recorder.list_traces(run_id=run_id)
            ]
        profiles = OperatorDiagnoser(
            feature_catalog=UAV_FEATURE_CATALOG,
            minimum_context_samples=2,
        ).diagnose(traces)
        records.extend(
            _record_from_profile(
                profile,
                bank_seed=bank_seed,
                source_code_commit=source_code_commit,
                source_population_fingerprint=fingerprint,
            )
            for profile in profiles
        )
    return create_mechanism_bank_v1(
        source_domain_id=UAV_DOMAIN_ID,
        bank_master_seeds=bank_master_seeds,
        source_code_commit=source_code_commit,
        records=tuple(records),
    )


__all__ = ["UAVMechanismBankConfig", "build_uav_mechanism_bank"]
