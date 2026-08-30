"""Registered full-budget JSSP cross-domain qualification workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict

from operator_evolution_core.diagnosis import OperatorDiagnoser, OperatorProfile
from operator_evolution_core.evolution import PopulationFreezeReceipt
from operator_evolution_core.memory import MechanismMemory
from operator_evolution_core.proposal import proposal_hash
from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    SearchBudget,
    SimulatedAnnealingAcceptance,
)
from operator_evolution_core.trajectory import OperatorTrace, TrajectoryRecorder

from .adapter import create_jssp_domain_adapter
from .data import JSSPDatasetSplits
from .evolution import (
    JSSPEvolutionSmokeConfig,
    JSSPEvolutionSmokeOutcome,
    run_offline_evolution_smoke,
)
from .features import JSSP_FEATURE_CATALOG
from .models import JobShopInstance, JobShopSolution
from .operators import (
    CompiledJSSPOperator,
    JSSPOperatorSpec,
    initial_operator_population,
    initial_operator_specs,
)
from .trajectory import JSSPTrajectorySink
from .validation import derive_jssp_seed


class _QualificationModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TrainingDiagnosticsSummary(_QualificationModel):
    instances: int
    search_calls_per_instance: int
    total_search_calls: int
    trace_count: int
    profile_count: int
    synergy_count: int
    mean_best_makespan: float
    feasibility_rate: float
    mean_acceptance_rate: float
    mean_effective_call_rate: float
    parent_slot_order: list[int]
    evidence_refs_by_parent: dict[str, list[str]]


class PopulationInstanceMetric(_QualificationModel):
    instance_id: str
    source_family: str
    jobs: int
    machines: int
    p0_best_makespan: float
    pn_best_makespan: float
    p0_feasible: bool
    pn_feasible: bool
    p0_runtime_ms: float
    pn_runtime_ms: float
    p0_effective_call_rate: float
    pn_effective_call_rate: float
    p0_acceptance_rate: float
    pn_acceptance_rate: float

    @property
    def relative_gain(self) -> float:
        return (self.p0_best_makespan - self.pn_best_makespan) / max(
            abs(self.p0_best_makespan), 1e-12
        )


class PopulationAggregate(_QualificationModel):
    mean_best_makespan: float
    median_best_makespan: float
    feasibility_rate: float
    median_runtime_ms: float
    mean_effective_call_rate: float
    mean_acceptance_rate: float


class FrozenPopulationComparison(_QualificationModel):
    test_instances: int
    search_calls_per_arm: int
    p0: PopulationAggregate
    pn: PopulationAggregate
    mean_relative_gain: float
    win_rate: float
    tie_rate: float
    outcomes: list[PopulationInstanceMetric]


class JSSPFormalQualificationReport(_QualificationModel):
    schema_version: str = "jssp-cross-domain-qualification-v1"
    configuration: dict[str, int]
    training: TrainingDiagnosticsSummary
    evolution: dict[str, object]
    frozen_test: FrozenPopulationComparison
    freeze_receipt_id: str
    initial_population_ids: list[str]
    final_population_ids: list[str]


@dataclass(frozen=True, slots=True)
class JSSPFormalQualificationConfig:
    master_seed: int = 20260823
    population_slots: int = 8
    train_calls: int = 400
    validation_calls: int = 240
    test_calls: int = 400
    generations: int = 3
    candidates_per_generation: int = 3
    train_instances: int = 60
    validation_instances: int = 41
    test_instances: int = 41
    runtime_repetitions: int = 4

    def __post_init__(self) -> None:
        for name, value in self.as_dict().items():
            if name != "master_seed" and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.population_slots != 8:
            raise ValueError("jssp-v1 qualification requires the registered 8 slots")
        if self.train_instances > 60 or self.validation_instances > 41 or self.test_instances > 41:
            raise ValueError("qualification instance limits exceed registered splits")

    def as_dict(self) -> dict[str, int]:
        return {
            "master_seed": self.master_seed,
            "population_slots": self.population_slots,
            "train_calls": self.train_calls,
            "validation_calls": self.validation_calls,
            "test_calls": self.test_calls,
            "generations": self.generations,
            "candidates_per_generation": self.candidates_per_generation,
            "train_instances": self.train_instances,
            "validation_instances": self.validation_instances,
            "test_instances": self.test_instances,
            "runtime_repetitions": self.runtime_repetitions,
        }


@dataclass(frozen=True, slots=True)
class TrainingDiagnostics:
    summary: TrainingDiagnosticsSummary
    profiles: tuple[OperatorProfile, ...]


def _register_population_memory(
    memory: MechanismMemory,
    specs: tuple[JSSPOperatorSpec, ...],
) -> None:
    for spec in specs:
        memory.add_mechanism(
            spec.operator_id,
            spec.model_dump(mode="json"),
            name=spec.name,
            description=spec.description,
            tags=["jssp", "operator", "generation-0"],
            metadata={
                "domain_id": "jssp",
                "ir_version": "jssp-v1",
                "generation": 0,
            },
        )


def _persist_diagnostics(
    memory: MechanismMemory,
    traces: list[OperatorTrace],
    profiles: list[OperatorProfile],
    synergies: list[object],
) -> None:
    for trace in traces:
        memory.record_operator_history(trace, mechanism_id=trace.operator_id)
    for profile in profiles:
        payload = profile.model_dump(mode="json")
        memory.add_operator_profile(
            payload,
            operator_id=profile.operator_id,
            run_id="jssp-formal-training",
            generation=0,
        )
        for mode, count in profile.failure_modes.items():
            memory.add_failure_mode(
                mode,
                mechanism_id=profile.operator_id,
                operator_id=profile.operator_id,
                count=count,
                evidence=profile.representative_failure_ids,
                metadata={"domain_id": "jssp", "generation": 0},
            )
    for relation in synergies:
        memory.add_synergy(
            relation.first_operator,
            relation.second_operator,
            relation.synergy,
            sample_count=relation.occurrences,
            context=relation.context,
            metadata={"domain_id": "jssp", "generation": 0},
        )


def run_training_diagnostics(
    splits: JSSPDatasetSplits,
    recorder: TrajectoryRecorder,
    memory: MechanismMemory,
    *,
    config: JSSPFormalQualificationConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> TrainingDiagnostics:
    adapter = create_jssp_domain_adapter()
    population = initial_operator_population()
    specs = initial_operator_specs()
    _register_population_memory(memory, specs)
    best_costs: list[float] = []
    feasible: list[bool] = []
    acceptance_rates: list[float] = []
    effective_rates: list[float] = []
    training = splits.open_train()[: config.train_instances]
    for index, instance in enumerate(training):
        seed = derive_jssp_seed(
            config.master_seed,
            "formal-training",
            index,
            instance.instance_id,
        )
        result = GenericSearchKernel(
            adapter=adapter,
            operators=population,
            scheduler=BlockRandomRoundRobinScheduler(),
            acceptance=SimulatedAnnealingAcceptance(),
            budget=SearchBudget(max_iterations=config.train_calls),
        ).run(
            instance,
            np.random.default_rng(seed),
            on_step=JSSPTrajectorySink(
                recorder,
                run_id="jssp-formal-training",
                instance=instance,
                seed=seed,
            ),
        )
        best_costs.append(float(result.best_evaluation.scalar_cost))
        feasible.append(bool(result.best_evaluation.feasible))
        acceptance_rates.append(result.acceptance_rate)
        effective_rates.append(
            sum(step.operator_outcome.success for step in result.steps)
            / max(1, result.iterations)
        )
        if progress_callback is not None:
            progress_callback(
                f"training instance {index + 1}/{len(training)} "
                f"best={result.best_evaluation.scalar_cost:.0f}"
            )
    traces = recorder.update_delayed_rewards(
        (5, 10, 20),
        run_id="jssp-formal-training",
    )
    diagnoser = OperatorDiagnoser(
        feature_catalog=JSSP_FEATURE_CATALOG,
        minimum_context_samples=2,
    )
    profiles = diagnoser.diagnose(traces)
    synergies = diagnoser.analyze_synergies(traces, min_samples=2)
    _persist_diagnostics(memory, traces, profiles, synergies)
    profile_by_id = {profile.operator_id: profile for profile in profiles}
    ranked_ids = sorted(
        (spec.operator_id for spec in specs),
        key=lambda operator_id: (
            profile_by_id[operator_id].mean_immediate_reward
            if operator_id in profile_by_id
            and profile_by_id[operator_id].mean_immediate_reward is not None
            else float("-inf"),
            operator_id,
        ),
    )
    index_by_id = {
        operator.operator_id: index for index, operator in enumerate(population)
    }
    evidence = {
        profile.operator_id: [
            "profile_" + proposal_hash(profile.model_dump(mode="json"))[:24]
        ]
        for profile in profiles
    }
    return TrainingDiagnostics(
        summary=TrainingDiagnosticsSummary(
            instances=len(training),
            search_calls_per_instance=config.train_calls,
            total_search_calls=len(training) * config.train_calls,
            trace_count=len(traces),
            profile_count=len(profiles),
            synergy_count=len(synergies),
            mean_best_makespan=float(np.mean(best_costs)),
            feasibility_rate=float(np.mean(feasible)),
            mean_acceptance_rate=float(np.mean(acceptance_rates)),
            mean_effective_call_rate=float(np.mean(effective_rates)),
            parent_slot_order=[index_by_id[operator_id] for operator_id in ranked_ids],
            evidence_refs_by_parent=evidence,
        ),
        profiles=tuple(profiles),
    )


def _aggregate(
    outcomes: list[PopulationInstanceMetric],
    prefix: str,
) -> PopulationAggregate:
    costs = [getattr(outcome, f"{prefix}_best_makespan") for outcome in outcomes]
    return PopulationAggregate(
        mean_best_makespan=float(np.mean(costs)),
        median_best_makespan=float(np.median(costs)),
        feasibility_rate=float(
            np.mean([getattr(outcome, f"{prefix}_feasible") for outcome in outcomes])
        ),
        median_runtime_ms=float(
            np.median([getattr(outcome, f"{prefix}_runtime_ms") for outcome in outcomes])
        ),
        mean_effective_call_rate=float(
            np.mean(
                [
                    getattr(outcome, f"{prefix}_effective_call_rate")
                    for outcome in outcomes
                ]
            )
        ),
        mean_acceptance_rate=float(
            np.mean(
                [getattr(outcome, f"{prefix}_acceptance_rate") for outcome in outcomes]
            )
        ),
    )


def compare_frozen_populations(
    splits: JSSPDatasetSplits,
    receipt: PopulationFreezeReceipt,
    initial_population: tuple[CompiledJSSPOperator, ...],
    final_population: tuple[CompiledJSSPOperator, ...],
    *,
    config: JSSPFormalQualificationConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> FrozenPopulationComparison:
    adapter = create_jssp_domain_adapter()
    instances = splits.open_test(receipt)[: config.test_instances]
    outcomes: list[PopulationInstanceMetric] = []
    for index, instance in enumerate(instances):
        initial_seed = derive_jssp_seed(
            config.master_seed, "formal-test-initial", index, instance.instance_id
        )
        search_seed = derive_jssp_seed(
            config.master_seed, "formal-test-search", index, instance.instance_id
        )
        initial = adapter.initializer.initialize(
            instance, np.random.default_rng(initial_seed)
        )
        measurements: dict[str, tuple[object, float, float]] = {}
        for arm, population in (
            ("p0", initial_population),
            ("pn", final_population),
        ):
            started = perf_counter()
            result = GenericSearchKernel(
                adapter=adapter,
                operators=population,
                scheduler=BlockRandomRoundRobinScheduler(),
                acceptance=SimulatedAnnealingAcceptance(),
                budget=SearchBudget(max_iterations=config.test_calls),
            ).run(
                instance,
                np.random.default_rng(search_seed),
                initial_solution=initial,
            )
            runtime_ms = (perf_counter() - started) * 1000.0
            effective_rate = sum(
                step.operator_outcome.success for step in result.steps
            ) / max(1, result.iterations)
            measurements[arm] = (result, runtime_ms, effective_rate)
        p0_result, p0_runtime, p0_effective = measurements["p0"]
        pn_result, pn_runtime, pn_effective = measurements["pn"]
        outcomes.append(
            PopulationInstanceMetric(
                instance_id=instance.instance_id,
                source_family=instance.source_family,
                jobs=instance.job_count,
                machines=instance.machines,
                p0_best_makespan=float(p0_result.best_evaluation.scalar_cost),
                pn_best_makespan=float(pn_result.best_evaluation.scalar_cost),
                p0_feasible=bool(p0_result.best_evaluation.feasible),
                pn_feasible=bool(pn_result.best_evaluation.feasible),
                p0_runtime_ms=p0_runtime,
                pn_runtime_ms=pn_runtime,
                p0_effective_call_rate=p0_effective,
                pn_effective_call_rate=pn_effective,
                p0_acceptance_rate=p0_result.acceptance_rate,
                pn_acceptance_rate=pn_result.acceptance_rate,
            )
        )
        if progress_callback is not None:
            progress_callback(
                f"frozen test instance {index + 1}/{len(instances)} "
                f"p0={outcomes[-1].p0_best_makespan:.0f} "
                f"pn={outcomes[-1].pn_best_makespan:.0f}"
            )
    gains = [outcome.relative_gain for outcome in outcomes]
    return FrozenPopulationComparison(
        test_instances=len(outcomes),
        search_calls_per_arm=config.test_calls,
        p0=_aggregate(outcomes, "p0"),
        pn=_aggregate(outcomes, "pn"),
        mean_relative_gain=float(np.mean(gains)),
        win_rate=float(np.mean([gain > 0 for gain in gains])),
        tie_rate=float(np.mean([gain == 0 for gain in gains])),
        outcomes=outcomes,
    )


def run_formal_qualification(
    splits: JSSPDatasetSplits,
    recorder: TrajectoryRecorder,
    memory: MechanismMemory,
    *,
    config: JSSPFormalQualificationConfig | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[JSSPFormalQualificationReport, JSSPEvolutionSmokeOutcome]:
    active = config or JSSPFormalQualificationConfig()
    training = run_training_diagnostics(
        splits,
        recorder,
        memory,
        config=active,
        progress_callback=progress_callback,
    )
    evolution = run_offline_evolution_smoke(
        splits,
        config=JSSPEvolutionSmokeConfig(
            master_seed=active.master_seed,
            search_calls=active.validation_calls,
            generations=active.generations,
            candidates_per_generation=active.candidates_per_generation,
            validation_instances=active.validation_instances,
            runtime_repetitions=active.runtime_repetitions,
        ),
        parent_slot_order=training.summary.parent_slot_order,
        evidence_refs_by_parent=training.summary.evidence_refs_by_parent,
        persist_validation_evidence=False,
        progress_callback=progress_callback,
    )
    frozen = compare_frozen_populations(
        splits,
        evolution.freeze_receipt,
        initial_operator_population(),
        evolution.final_population,
        config=active,
        progress_callback=progress_callback,
    )
    report = JSSPFormalQualificationReport(
        configuration=active.as_dict(),
        training=training.summary,
        evolution=evolution.report.model_dump(mode="json"),
        frozen_test=frozen,
        freeze_receipt_id=evolution.freeze_receipt.receipt_id,
        initial_population_ids=[
            operator.operator_id for operator in initial_operator_population()
        ],
        final_population_ids=[
            operator.operator_id for operator in evolution.final_population
        ],
    )
    return report, evolution


__all__ = [
    "FrozenPopulationComparison",
    "JSSPFormalQualificationConfig",
    "JSSPFormalQualificationReport",
    "PopulationAggregate",
    "PopulationInstanceMetric",
    "TrainingDiagnosticsSummary",
    "compare_frozen_populations",
    "run_formal_qualification",
    "run_training_diagnostics",
]
