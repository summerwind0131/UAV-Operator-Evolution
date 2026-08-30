"""Validation-only JSSP arms for the bidirectional mechanism-transfer study."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from operator_evolution_core.evolution import (
    TransferArmLifecycleV1,
    TransferArmV1,
    TransferCandidateLifecycleV1,
    select_transfer_evidence_v1,
    transfer_candidate_context_v1,
)
from operator_evolution_core.memory import MechanismBankV1

from .adapter import JSSP_DOMAIN_ID, create_jssp_domain_adapter
from .data import JSSPDatasetSplits
from .operators import (
    JSSPDomainKit,
    JSSPSmokeFixture,
    initial_operator_population,
)
from .transfer_design import design_jssp_operator_from_mechanisms
from .validation import JSSPCandidateValidator


@dataclass(frozen=True, slots=True)
class JSSPTransferArmConfig:
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


def run_jssp_transfer_arm(
    splits: JSSPDatasetSplits,
    *,
    arm: TransferArmV1,
    master_seed: int,
    same_domain_bank: MechanismBankV1,
    cross_domain_bank: MechanismBankV1,
    config: JSSPTransferArmConfig | None = None,
) -> TransferArmLifecycleV1:
    """Run one equal-budget JSSP arm without acquiring a test capability."""

    active = config or JSSPTransferArmConfig()
    validation = splits.open_validation()[: active.validation_instances]
    training_fixture = splits.open_train()[0]
    kit = JSSPDomainKit()
    adapter = create_jssp_domain_adapter()
    fixture_solution = adapter.initializer.initialize(
        training_fixture,
        np.random.default_rng(master_seed),
    )
    fixture = JSSPSmokeFixture(training_fixture, fixture_solution)
    population = list(initial_operator_population())
    initial_ids = tuple(operator.operator_id for operator in population)
    records_by_id = {
        record.mechanism_id: record
        for bank in (same_domain_bank, cross_domain_bank)
        for record in bank.records
    }
    validator = JSSPCandidateValidator(
        search_calls=active.search_calls,
        master_seed=master_seed,
        runtime_repetitions=active.runtime_repetitions,
    )
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
                target_domain_id=JSSP_DOMAIN_ID,
                same_domain_bank=same_domain_bank,
                cross_domain_bank=cross_domain_bank,
                context=context,
            )
            records = tuple(records_by_id[item] for item in evidence.mechanism_ids)
            spec = design_jssp_operator_from_mechanisms(
                records,
                master_seed=master_seed,
                candidate_index=flat_index,
            )
            parsed = kit.parse_ir(spec)
            smoke = kit.smoke(parsed, fixture)
            candidate = kit.compile(parsed)
            report = validator.validate(
                validation,
                population,
                parent.operator_id,
                candidate,
                generation=generation,
                candidate_index=candidate_index,
                root_run_id=f"mechanism-transfer-v1-jssp-{arm}-{master_seed}",
                safety_failures=smoke.failures,
            )
            if report.retained:
                population[slot] = candidate
            candidates.append(
                TransferCandidateLifecycleV1(
                    generation=generation,
                    candidate_index=candidate_index,
                    parent_operator_id=parent.operator_id,
                    candidate_operator_id=candidate.operator_id,
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
                    retained=report.retained,
                    retention_reasons=tuple(report.retention_reasons),
                )
            )
    return TransferArmLifecycleV1(
        target_domain_id=JSSP_DOMAIN_ID,
        arm=arm,
        master_seed=master_seed,
        search_calls=active.search_calls,
        generations=active.generations,
        candidates_per_generation=active.candidates_per_generation,
        validation_instances=len(validation),
        runtime_repetitions=active.runtime_repetitions,
        initial_population_ids=initial_ids,
        final_population_ids=tuple(operator.operator_id for operator in population),
        candidates=tuple(candidates),
    )


__all__ = ["JSSPTransferArmConfig", "run_jssp_transfer_arm"]
