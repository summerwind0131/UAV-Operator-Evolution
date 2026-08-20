"""Reusable fixed-budget paired validator for generated operators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..environment import Environment2D
from ..operators.base import PathOperator
from ..path.evaluator import PathEvaluator
from ..path.initializer import initialize_path
from ..reproducibility import derive_seed
from ..search.context import SearchContext
from ..search.executor import SearchExecutor, SearchResult
from ..trajectory import TrajectoryRecorder
from .validation import PairedOutcome, ValidationReport, decide_retention, validate_operator_contract


@dataclass(frozen=True, slots=True)
class _ArmMeasurement:
    result: SearchResult
    total_runtime_ms: float
    operator_runtime_ms: float
    operator_call_count: int
    operator_changed_call_count: int
    operator_accepted_call_count: int


class FixedBudgetCandidateValidator:
    """Apply the existing contract and CRN validation protocol outside an agent.

    The API intentionally accepts only a sequence already identified as the
    validation split.  It has no dataset dictionary and therefore cannot read a
    held-out test split while making a retention decision.
    """

    def __init__(self, config: ExperimentConfig, evaluator: PathEvaluator) -> None:
        self.config = config
        self.evaluator = evaluator

    def _executor(
        self,
        operators: list[PathOperator],
        recorder: TrajectoryRecorder | None,
    ) -> SearchExecutor:
        search = self.config.search
        return SearchExecutor(
            operators,
            self.evaluator,
            max_iterations=search.validation_iterations,
            temperature_start_ratio=search.temperature_start_ratio,
            temperature_end_ratio=search.temperature_end_ratio,
            recent_window=search.recent_window,
            initializer_grid_resolution=self.config.maps.grid_resolution,
            recorder=recorder,
        )

    def validate(
        self,
        population: Sequence[PathOperator],
        parent_name: str,
        candidate: PathOperator,
        validation_environments: Sequence[Environment2D],
        *,
        generation: int,
        candidate_index: int,
        recorder: TrajectoryRecorder | None,
        root_run_id: str,
    ) -> ValidationReport:
        if not validation_environments:
            return decide_retention(
                parent_name, str(candidate.name), [], self.config.evolution
            )

        contract_failures = self.contract_failures(
            candidate,
            validation_environments[0],
            generation=generation,
            candidate_index=candidate_index,
        )
        parent_pool = list(population)
        candidate_pool = list(population)
        replacement_index = next(
            (
                index
                for index, operator in enumerate(candidate_pool)
                if str(operator.name) == parent_name
            ),
            None,
        )
        if replacement_index is None:
            contract_failures.append("parent slot not present in population")
            return decide_retention(
                parent_name,
                str(candidate.name),
                [],
                self.config.evolution,
                safety_passed=False,
                safety_failures=contract_failures,
            )
        else:
            candidate_pool[replacement_index] = candidate

        outcomes = self.compare(
            parent_pool,
            candidate_pool,
            validation_environments,
            parent_name=parent_name,
            generation=generation,
            candidate_index=candidate_index,
            recorder=recorder,
            root_run_id=root_run_id,
        )
        return decide_retention(
            parent_name,
            str(candidate.name),
            outcomes,
            self.config.evolution,
            safety_passed=not contract_failures,
            safety_failures=contract_failures,
            bootstrap_seed=derive_seed(
                self.config.seed, "bootstrap", generation, candidate_index
            ),
        )

    def contract_failures(
        self,
        candidate: PathOperator,
        environment: Environment2D,
        *,
        generation: int,
        candidate_index: int,
    ) -> list[str]:
        initial = initialize_path(
            environment, grid_resolution=self.config.maps.grid_resolution
        )
        initial_eval = self.evaluator.evaluate(initial, environment)
        context = SearchContext(
            iteration=0,
            max_iterations=self.config.search.validation_iterations,
            current_evaluation=initial_eval,
            best_evaluation=initial_eval,
        )
        return validate_operator_contract(
            candidate,
            initial,
            environment,
            context,
            [
                derive_seed(
                    self.config.seed,
                    "contract",
                    generation,
                    candidate_index,
                    index,
                )
                for index in range(3)
            ],
            self.config.dsl.max_waypoints,
            self.config.dsl.deadline_ms,
        )

    def compare(
        self,
        parent_population: Sequence[PathOperator],
        candidate_population: Sequence[PathOperator],
        validation_environments: Sequence[Environment2D],
        *,
        parent_name: str,
        generation: int,
        candidate_index: int,
        recorder: TrajectoryRecorder | None,
        root_run_id: str,
    ) -> list[PairedOutcome]:
        outcomes: list[PairedOutcome] = []
        replacement_index = next(
            (
                index
                for index, operator in enumerate(parent_population)
                if str(operator.name) == parent_name
            ),
            None,
        )
        if replacement_index is None or replacement_index >= len(candidate_population):
            raise ValueError("paired populations do not contain the requested parent slot")
        candidate_name = str(candidate_population[replacement_index].name)
        for map_index, environment in enumerate(validation_environments):
            seed = derive_seed(
                self.config.seed,
                "paired",
                "validation",
                generation,
                map_index,
                environment.map_id,
            )
            initial = initialize_path(
                environment, grid_resolution=self.config.maps.grid_resolution
            )
            run_prefix = (
                f"{root_run_id}-validation-g{generation}-c{candidate_index}-m{map_index}"
            )
            parent_measurements: list[_ArmMeasurement] = []
            candidate_measurements: list[_ArmMeasurement] = []
            timing_order: list[str] = []
            repetitions = self.config.evolution.runtime_validation_repetitions
            for repetition in range(repetitions):
                order = self._abba_order(repetition)
                timing_order.append(order)
                arms = (
                    (
                        "parent",
                        list(parent_population),
                        parent_name,
                        parent_measurements,
                    ),
                    (
                        "candidate",
                        list(candidate_population),
                        candidate_name,
                        candidate_measurements,
                    ),
                )
                if order == "candidate_first":
                    arms = (arms[1], arms[0])
                for arm_name, population, target_name, measurements in arms:
                    result, runtime_ms = self._run_arm(
                        population,
                        environment,
                        initial,
                        seed,
                        None,
                        f"{run_prefix}-timing-r{repetition}-{arm_name}",
                        generation,
                    )
                    measurements.append(
                        self._measurement(result, runtime_ms, target_name)
                    )

            parent_result = parent_measurements[0].result
            candidate_result = candidate_measurements[0].result
            parent_runtime_samples = [
                measurement.total_runtime_ms for measurement in parent_measurements
            ]
            candidate_runtime_samples = [
                measurement.total_runtime_ms for measurement in candidate_measurements
            ]
            parent_operator_runtime_samples = [
                measurement.operator_runtime_ms for measurement in parent_measurements
            ]
            candidate_operator_runtime_samples = [
                measurement.operator_runtime_ms for measurement in candidate_measurements
            ]

            # Persist one deterministic evidence pair, but keep recorder I/O out
            # of the repeated timing samples used by the runtime retention gate.
            if recorder is not None:
                self._run_arm(
                    list(parent_population),
                    environment,
                    initial,
                    seed,
                    recorder,
                    f"{run_prefix}-evidence-parent",
                    generation,
                )
                self._run_arm(
                    list(candidate_population),
                    environment,
                    initial,
                    seed,
                    recorder,
                    f"{run_prefix}-evidence-candidate",
                    generation,
                )

            outcomes.append(
                PairedOutcome(
                    map_id=environment.map_id,
                    difficulty=environment.difficulty,
                    parent_best_cost=float(parent_result.best_evaluation.total_cost),
                    candidate_best_cost=float(candidate_result.best_evaluation.total_cost),
                    parent_feasible=bool(parent_result.best_evaluation.feasible),
                    candidate_feasible=bool(candidate_result.best_evaluation.feasible),
                    parent_runtime_ms=float(np.median(parent_runtime_samples)),
                    candidate_runtime_ms=float(np.median(candidate_runtime_samples)),
                    runtime_repetitions=repetitions,
                    parent_runtime_samples_ms=parent_runtime_samples,
                    candidate_runtime_samples_ms=candidate_runtime_samples,
                    timing_order=timing_order,
                    parent_operator_runtime_ms=float(
                        np.median(parent_operator_runtime_samples)
                    ),
                    candidate_operator_runtime_ms=float(
                        np.median(candidate_operator_runtime_samples)
                    ),
                    parent_operator_runtime_samples_ms=parent_operator_runtime_samples,
                    candidate_operator_runtime_samples_ms=candidate_operator_runtime_samples,
                    parent_operator_call_count=sum(
                        measurement.operator_call_count
                        for measurement in parent_measurements
                    ),
                    candidate_operator_call_count=sum(
                        measurement.operator_call_count
                        for measurement in candidate_measurements
                    ),
                    parent_operator_changed_call_count=sum(
                        measurement.operator_changed_call_count
                        for measurement in parent_measurements
                    ),
                    candidate_operator_changed_call_count=sum(
                        measurement.operator_changed_call_count
                        for measurement in candidate_measurements
                    ),
                    parent_operator_accepted_call_count=sum(
                        measurement.operator_accepted_call_count
                        for measurement in parent_measurements
                    ),
                    candidate_operator_accepted_call_count=sum(
                        measurement.operator_accepted_call_count
                        for measurement in candidate_measurements
                    ),
                )
            )
        return outcomes

    @staticmethod
    def _abba_order(repetition: int) -> str:
        return (
            "parent_first"
            if repetition % 4 in {0, 3}
            else "candidate_first"
        )

    @staticmethod
    def _measurement(
        result: SearchResult,
        total_runtime_ms: float,
        operator_name: str,
    ) -> _ArmMeasurement:
        calls = [
            step for step in result.steps if step.operator_name == operator_name
        ]
        changed_calls = sum(
            step.operator_result.success
            and step.candidate_path != step.path_before
            for step in calls
        )
        return _ArmMeasurement(
            result=result,
            total_runtime_ms=total_runtime_ms,
            operator_runtime_ms=sum(step.runtime_ms for step in calls),
            operator_call_count=len(calls),
            operator_changed_call_count=changed_calls,
            operator_accepted_call_count=sum(step.accepted for step in calls),
        )

    def _run_arm(
        self,
        population: list[PathOperator],
        environment: Environment2D,
        initial: list[tuple[float, float]],
        seed: int,
        recorder: TrajectoryRecorder | None,
        run_id: str,
        generation: int,
    ) -> tuple[Any, float]:
        import time

        started = time.perf_counter()
        result = self._executor(population, recorder).run(
            environment,
            np.random.default_rng(seed),
            initial_path=initial,
            recorder=recorder,
            run_id=run_id,
            generation=generation,
        )
        return result, (time.perf_counter() - started) * 1_000.0


__all__ = ["FixedBudgetCandidateValidator"]
