"""JSSP-owned evidence projection into the domain-neutral mechanism protocol."""

from __future__ import annotations

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
from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    SearchBudget,
    SimulatedAnnealingAcceptance,
)
from operator_evolution_core.trajectory import TrajectoryRecorder

from .adapter import JSSP_DOMAIN_ID, create_jssp_domain_adapter
from .data import JSSPDatasetSplits
from .features import JSSP_FEATURE_CATALOG
from .operators import JSSPDomainKit, initial_operator_population, initial_operator_specs
from .trajectory import JSSPTrajectorySink
from .validation import derive_jssp_seed


@dataclass(frozen=True, slots=True)
class JSSPMechanismBankConfig:
    train_calls: int = 400
    validation_calls: int = 240
    train_instances: int = 60
    validation_instances: int = 41

    def __post_init__(self) -> None:
        if min(
            self.train_calls,
            self.validation_calls,
            self.train_instances,
            self.validation_instances,
        ) <= 0:
            raise ValueError("JSSP mechanism-bank budgets must be positive")
        if self.train_instances > 60 or self.validation_instances > 41:
            raise ValueError("JSSP mechanism-bank split limits exceed registration")


def _mechanism_tags(operator_id: str) -> tuple[str, ...]:
    if operator_id.startswith("random-"):
        return ("diversify",)
    if operator_id.startswith("critical-block"):
        return ("intensify",)
    if operator_id in {"bottleneck-block-insertion", "high-idle-gap-relocation"}:
        return ("repair", "intensify", "rollback")
    return ("diversify", "intensify")


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
            diversity="preserve",
            locality="local",
        )
    if tags == ("diversify",):
        return ExpectedMechanismEffectV1(
            feasibility="preserve",
            cost="improve",
            diversity="increase",
            locality="mixed",
        )
    return ExpectedMechanismEffectV1(
        feasibility="preserve",
        cost="improve",
        diversity="decrease" if tags == ("intensify",) else "preserve",
        locality="local",
    )


def _failure_modes(profile: OperatorProfile) -> tuple[str, ...]:
    modes: list[str] = []
    if profile.failure_modes:
        modes.append("inapplicable-context")
    if profile.success_rate < 0.10:
        modes.append("low-yield")
    if profile.acceptance_rate >= 0.30 and profile.success_rate < 0.10:
        modes.append("disruptive-change")
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
    profile_ref = "profile:" + proposal_hash(profile.model_dump(mode="json"))
    return create_mechanism_record_v1(
        source_domain_id=JSSP_DOMAIN_ID,
        mechanism_tags=tags,
        trigger_context=_abstract_context(profile),
        expected_effect=_expected_effect(tags),
        failure_modes=_failure_modes(profile),
        evidence_refs=(profile_ref, "split:train", "split:validation"),
        evidence_strength=_evidence_strength(profile),
        evidence_sample_count=profile.attempts,
        evidence_splits=("train", "validation"),
        bank_run_id=f"jssp-mechanism-bank-{bank_seed}",
        bank_master_seed=bank_seed,
        source_operator_id=profile.operator_id,
        source_code_commit=source_code_commit,
        source_population_fingerprint=source_population_fingerprint,
    )


def build_jssp_mechanism_bank(
    splits: JSSPDatasetSplits,
    *,
    bank_master_seeds: tuple[int, ...],
    source_code_commit: str,
    config: JSSPMechanismBankConfig | None = None,
) -> MechanismBankV1:
    """Build a bank without requesting the sealed JSSP test capability."""

    if not bank_master_seeds or len(set(bank_master_seeds)) != len(bank_master_seeds):
        raise ValueError("bank_master_seeds must be unique and non-empty")
    active = config or JSSPMechanismBankConfig()
    train = splits.open_train()[: active.train_instances]
    validation = splits.open_validation()[: active.validation_instances]
    adapter = create_jssp_domain_adapter()
    population = initial_operator_population()
    specs = initial_operator_specs()
    kit = JSSPDomainKit()
    fingerprint = population_fingerprint(
        [spec.operator_id for spec in specs],
        {spec.operator_id: spec for spec in specs},
        kit,
    )
    records: list[MechanismRecordV1] = []
    for bank_seed in bank_master_seeds:
        with TrajectoryRecorder(":memory:") as recorder:
            run_ids: list[str] = []
            for split_name, instances, calls in (
                ("train", train, active.train_calls),
                ("validation", validation, active.validation_calls),
            ):
                run_id = f"jssp-bank-{bank_seed}-{split_name}"
                run_ids.append(run_id)
                for index, instance in enumerate(instances):
                    seed = derive_jssp_seed(
                        bank_seed,
                        f"mechanism-bank-{split_name}",
                        index,
                        instance.instance_id,
                    )
                    GenericSearchKernel(
                        adapter=adapter,
                        operators=population,
                        scheduler=BlockRandomRoundRobinScheduler(),
                        acceptance=SimulatedAnnealingAcceptance(),
                        budget=SearchBudget(max_iterations=calls),
                    ).run(
                        instance,
                        np.random.default_rng(seed),
                        on_step=JSSPTrajectorySink(
                            recorder,
                            run_id=run_id,
                            instance=instance,
                            seed=seed,
                        ),
                    )
                recorder.update_delayed_rewards((5, 10, 20), run_id=run_id)
            traces = [
                trace
                for run_id in run_ids
                for trace in recorder.list_traces(run_id=run_id)
            ]
        profiles = OperatorDiagnoser(
            feature_catalog=JSSP_FEATURE_CATALOG,
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
        source_domain_id=JSSP_DOMAIN_ID,
        bank_master_seeds=bank_master_seeds,
        source_code_commit=source_code_commit,
        records=tuple(records),
    )


__all__ = [
    "JSSPMechanismBankConfig",
    "build_jssp_mechanism_bank",
]
