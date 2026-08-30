"""Validation-only UAV arms for the bidirectional mechanism-transfer study."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from operator_evolution_core.evolution import (
    TransferArmLifecycleV1,
    TransferArmV1,
    TransferCandidateLifecycleV1,
    select_transfer_evidence_v1,
    transfer_candidate_context_v1,
)
from operator_evolution_core.memory import MechanismBankV1

from .config import ExperimentConfig
from .domain.adapters import UAV_DOMAIN_ID
from .domain.uav_kit import UAVDomainKit, UAVSmokeFixture
from .environment import Environment2D
from .evolution.candidate_validator import FixedBudgetCandidateValidator
from .operators.compiler import OperatorCompiler
from .operators.registry import default_manual_operators
from .path import ObjectiveWeights, PathEvaluator, initialize_path
from .transfer_design import design_uav_operator_from_mechanisms


@dataclass(frozen=True, slots=True)
class UAVTransferArmConfig:
    search_calls: int = 64
    generations: int = 1
    candidates_per_generation: int = 1
    validation_instances: int = 1
    runtime_repetitions: int = 2

    def __post_init__(self) -> None:
        for field_name in (
            "search_calls",
            "generations",
            "candidates_per_generation",
            "validation_instances",
            "runtime_repetitions",
        ):
            if getattr(self, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.runtime_repetitions % 2:
            raise ValueError("UAV ABBA timing repetitions must be even")


def _validation_config(
    base: ExperimentConfig,
    active: UAVTransferArmConfig,
    master_seed: int,
) -> ExperimentConfig:
    config = base.model_copy(deep=True)
    config.seed = master_seed
    config.search.validation_iterations = active.search_calls
    config.evolution.runtime_validation_repetitions = active.runtime_repetitions
    return config


def run_uav_transfer_arm(
    train_environments: Sequence[Environment2D],
    validation_environments: Sequence[Environment2D],
    *,
    arm: TransferArmV1,
    master_seed: int,
    same_domain_bank: MechanismBankV1,
    cross_domain_bank: MechanismBankV1,
    base_config: ExperimentConfig,
    config: UAVTransferArmConfig | None = None,
) -> TransferArmLifecycleV1:
    """Run one equal-budget UAV arm with explicit train/validation capabilities."""

    if not train_environments:
        raise ValueError("at least one training environment is required")
    active = config or UAVTransferArmConfig()
    validation = tuple(validation_environments[: active.validation_instances])
    if not validation:
        raise ValueError("at least one validation environment is required")
    experiment = _validation_config(base_config, active, master_seed)
    evaluator = PathEvaluator(
        ObjectiveWeights.model_validate(experiment.objective.model_dump())
    )
    kit = UAVDomainKit(OperatorCompiler(experiment.dsl))
    fixture_environment = train_environments[0]
    fixture = UAVSmokeFixture(
        fixture_environment,
        initialize_path(
            fixture_environment,
            grid_resolution=experiment.maps.grid_resolution,
        ),
    )
    population = list(default_manual_operators())
    initial_ids = tuple(str(operator.name) for operator in population)
    records_by_id = {
        record.mechanism_id: record
        for bank in (same_domain_bank, cross_domain_bank)
        for record in bank.records
    }
    validator = FixedBudgetCandidateValidator(experiment, evaluator)
    candidates: list[TransferCandidateLifecycleV1] = []
    for generation in range(active.generations):
        for candidate_index in range(active.candidates_per_generation):
            flat_index = generation * active.candidates_per_generation + candidate_index
            slot = flat_index % len(population)
            parent = population[slot]
            context = transfer_candidate_context_v1(
                generation,
                candidate_index,
                generations=active.generations,
            )
            evidence = select_transfer_evidence_v1(
                arm=arm,
                target_domain_id=UAV_DOMAIN_ID,
                same_domain_bank=same_domain_bank,
                cross_domain_bank=cross_domain_bank,
                context=context,
            )
            records = tuple(records_by_id[item] for item in evidence.mechanism_ids)
            spec = design_uav_operator_from_mechanisms(
                records,
                master_seed=master_seed,
                candidate_index=flat_index,
            )
            parsed = kit.parse_ir(spec)
            smoke = kit.smoke(parsed, fixture)
            candidate = kit.compile(parsed)
            report = validator.validate(
                population,
                str(parent.name),
                candidate,
                validation,
                generation=generation,
                candidate_index=candidate_index,
                recorder=None,
                root_run_id=f"mechanism-transfer-v1-uav-{arm}-{master_seed}",
            )
            retained = smoke.smoke_passed and report.retained
            reasons = tuple(report.retention_reasons)
            if not smoke.smoke_passed:
                reasons = ("failed target-domain smoke gate", *smoke.failures)
            if retained:
                population[slot] = candidate
            candidates.append(
                TransferCandidateLifecycleV1(
                    generation=generation,
                    candidate_index=candidate_index,
                    parent_operator_id=str(parent.name),
                    candidate_operator_id=str(candidate.name),
                    evidence=evidence,
                    smoke_passed=smoke.smoke_passed,
                    validation_outcomes=len(report.outcomes),
                    mean_gain=report.mean_gain,
                    parent_feasibility_rate=report.parent_feasibility_rate,
                    candidate_feasibility_rate=report.candidate_feasibility_rate,
                    candidate_effective_call_rate=report.candidate_effective_call_rate,
                    candidate_acceptance_rate=(
                        report.candidate_operator_acceptance_rate
                    ),
                    retained=retained,
                    retention_reasons=reasons,
                )
            )
    return TransferArmLifecycleV1(
        target_domain_id=UAV_DOMAIN_ID,
        arm=arm,
        master_seed=master_seed,
        search_calls=active.search_calls,
        generations=active.generations,
        candidates_per_generation=active.candidates_per_generation,
        validation_instances=len(validation),
        runtime_repetitions=active.runtime_repetitions,
        initial_population_ids=initial_ids,
        final_population_ids=tuple(str(operator.name) for operator in population),
        candidates=tuple(candidates),
    )


__all__ = ["UAVTransferArmConfig", "run_uav_transfer_arm"]
