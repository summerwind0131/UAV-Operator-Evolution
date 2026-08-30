"""Domain-independent fixed-budget paired candidate validation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import AbstractSet, Generic, TypeVar

import numpy as np

from ..contracts import DomainAdapter
from ..search import SearchBudget
from .fitness import FitnessPolicy
from .paired import PairedOutcome, RetentionConfig, ValidationReport, decide_retention
from .schedule import (
    abba_timing_order,
    build_crn_seed_schedule,
    replace_population_slot,
)

InstanceT = TypeVar("InstanceT")
SolutionT = TypeVar("SolutionT")
OperatorT = TypeVar("OperatorT")


@dataclass(frozen=True, slots=True)
class ArmMeasurement:
    best_cost: float
    feasible: bool
    total_runtime_ms: float
    operator_runtime_ms: float
    operator_call_count: int
    operator_changed_call_count: int
    operator_accepted_call_count: int


ArmRunner = Callable[
    [Sequence[OperatorT], InstanceT, SolutionT, int, str, int, str, bool],
    ArmMeasurement,
]
SeedDeriver = Callable[..., int]


class GenericPairedCandidateValidator(Generic[InstanceT, SolutionT, OperatorT]):
    """Compare one slot replacement using validation-only instances.

    The constructor receives capabilities rather than a dataset dictionary, so
    a caller cannot accidentally make a retention decision with a test split.
    """

    def __init__(
        self,
        *,
        adapter: DomainAdapter[InstanceT, SolutionT],
        budget: SearchBudget,
        retention_config: RetentionConfig,
        master_seed: int,
        seed_deriver: SeedDeriver,
        arm_runner: ArmRunner[InstanceT, SolutionT, OperatorT],
        operator_id: Callable[[OperatorT], str],
        instance_id: Callable[[InstanceT], str],
        context_label: Callable[[InstanceT], str],
        runtime_repetitions: int,
        fitness_policy: FitnessPolicy | str = FitnessPolicy.DETERMINISTIC_V2,
        specialist_contexts: AbstractSet[str] = frozenset(),
    ) -> None:
        self.adapter = adapter
        self.budget = budget
        self.retention_config = retention_config
        self.master_seed = int(master_seed)
        self.seed_deriver = seed_deriver
        self.arm_runner = arm_runner
        self.operator_id = operator_id
        self.instance_id = instance_id
        self.context_label = context_label
        self.runtime_repetitions = int(runtime_repetitions)
        self.fitness_policy = FitnessPolicy(fitness_policy)
        self.specialist_contexts = frozenset(specialist_contexts)
        # Validate once instead of discovering an invalid value mid-study.
        abba_timing_order(self.runtime_repetitions)

    def validate(
        self,
        validation_instances: Sequence[InstanceT],
        operator_population: Sequence[OperatorT],
        parent_operator_id: str,
        candidate_operator: OperatorT,
        *,
        generation: int,
        candidate_index: int,
        root_run_id: str,
        safety_failures: Sequence[str] = (),
        persist_evidence: bool = False,
    ) -> ValidationReport:
        candidate_id = self.operator_id(candidate_operator)
        failures = list(safety_failures)
        replacement = replace_population_slot(
            operator_population,
            parent_operator_id,
            candidate_operator,
            operator_id=self.operator_id,
        )
        if replacement is None:
            failures.append("parent slot not present in population")
            return decide_retention(
                parent_operator_id,
                candidate_id,
                [],
                self.retention_config,
                safety_passed=False,
                safety_failures=failures,
                fitness_policy=self.fitness_policy,
                specialist_contexts=self.specialist_contexts,
            )

        outcomes = self.compare(
            validation_instances,
            operator_population,
            replacement.population,
            parent_operator_id=parent_operator_id,
            candidate_operator_id=candidate_id,
            generation=generation,
            candidate_index=candidate_index,
            root_run_id=root_run_id,
            persist_evidence=persist_evidence,
        )
        return decide_retention(
            parent_operator_id,
            candidate_id,
            outcomes,
            self.retention_config,
            safety_passed=not failures,
            safety_failures=failures,
            bootstrap_seed=self.seed_deriver(
                self.master_seed, "bootstrap", generation, candidate_index
            ),
            fitness_policy=self.fitness_policy,
            specialist_contexts=self.specialist_contexts,
        )

    def compare(
        self,
        validation_instances: Sequence[InstanceT],
        parent_population: Sequence[OperatorT],
        candidate_population: Sequence[OperatorT],
        *,
        parent_operator_id: str,
        candidate_operator_id: str,
        generation: int,
        candidate_index: int,
        root_run_id: str,
        persist_evidence: bool,
    ) -> list[PairedOutcome]:
        ids = [self.instance_id(instance) for instance in validation_instances]
        schedule = build_crn_seed_schedule(
            ids,
            lambda index, instance_id: self.seed_deriver(
                self.master_seed,
                "paired",
                "validation",
                generation,
                index,
                instance_id,
            ),
        )
        orders = abba_timing_order(self.runtime_repetitions)
        outcomes: list[PairedOutcome] = []
        for instance, seed_entry in zip(
            validation_instances, schedule, strict=True
        ):
            initial_seed = self.seed_deriver(
                self.master_seed,
                "validation-initial",
                generation,
                seed_entry.instance_index,
                seed_entry.instance_id,
            )
            initial = self.adapter.initializer.initialize(
                instance, np.random.default_rng(initial_seed)
            )
            run_prefix = (
                f"{root_run_id}-validation-g{generation}-c{candidate_index}-"
                f"m{seed_entry.instance_index}"
            )
            parent_measurements: list[ArmMeasurement] = []
            candidate_measurements: list[ArmMeasurement] = []
            for repetition, order in enumerate(orders):
                arms = (
                    (
                        "parent",
                        parent_population,
                        parent_operator_id,
                        parent_measurements,
                    ),
                    (
                        "candidate",
                        candidate_population,
                        candidate_operator_id,
                        candidate_measurements,
                    ),
                )
                if order == "candidate_first":
                    arms = (arms[1], arms[0])
                for arm_name, population, target_id, measurements in arms:
                    measurements.append(
                        self.arm_runner(
                            population,
                            instance,
                            initial,
                            seed_entry.seed,
                            target_id,
                            generation,
                            f"{run_prefix}-timing-r{repetition}-{arm_name}",
                            False,
                        )
                    )

            if persist_evidence:
                self.arm_runner(
                    parent_population,
                    instance,
                    initial,
                    seed_entry.seed,
                    parent_operator_id,
                    generation,
                    f"{run_prefix}-evidence-parent",
                    True,
                )
                self.arm_runner(
                    candidate_population,
                    instance,
                    initial,
                    seed_entry.seed,
                    candidate_operator_id,
                    generation,
                    f"{run_prefix}-evidence-candidate",
                    True,
                )

            outcomes.append(
                _paired_outcome(
                    instance_id=seed_entry.instance_id,
                    context_label=self.context_label(instance),
                    parent=parent_measurements,
                    candidate=candidate_measurements,
                    orders=orders,
                )
            )
        return outcomes


def _paired_outcome(
    *,
    instance_id: str,
    context_label: str,
    parent: Sequence[ArmMeasurement],
    candidate: Sequence[ArmMeasurement],
    orders: Sequence[str],
) -> PairedOutcome:
    parent_runtime = [item.total_runtime_ms for item in parent]
    candidate_runtime = [item.total_runtime_ms for item in candidate]
    parent_operator_runtime = [item.operator_runtime_ms for item in parent]
    candidate_operator_runtime = [item.operator_runtime_ms for item in candidate]
    return PairedOutcome(
        instance_id=instance_id,
        context_label=context_label,
        parent_best_cost=parent[0].best_cost,
        candidate_best_cost=candidate[0].best_cost,
        parent_feasible=parent[0].feasible,
        candidate_feasible=candidate[0].feasible,
        parent_runtime_ms=float(np.median(parent_runtime)),
        candidate_runtime_ms=float(np.median(candidate_runtime)),
        runtime_repetitions=len(orders),
        parent_runtime_samples_ms=parent_runtime,
        candidate_runtime_samples_ms=candidate_runtime,
        timing_order=list(orders),
        parent_operator_runtime_ms=float(np.median(parent_operator_runtime)),
        candidate_operator_runtime_ms=float(np.median(candidate_operator_runtime)),
        parent_operator_runtime_samples_ms=parent_operator_runtime,
        candidate_operator_runtime_samples_ms=candidate_operator_runtime,
        parent_operator_call_count=sum(item.operator_call_count for item in parent),
        candidate_operator_call_count=sum(
            item.operator_call_count for item in candidate
        ),
        parent_operator_changed_call_count=sum(
            item.operator_changed_call_count for item in parent
        ),
        candidate_operator_changed_call_count=sum(
            item.operator_changed_call_count for item in candidate
        ),
        parent_operator_accepted_call_count=sum(
            item.operator_accepted_call_count for item in parent
        ),
        candidate_operator_accepted_call_count=sum(
            item.operator_accepted_call_count for item in candidate
        ),
    )


__all__ = ["ArmMeasurement", "GenericPairedCandidateValidator"]
