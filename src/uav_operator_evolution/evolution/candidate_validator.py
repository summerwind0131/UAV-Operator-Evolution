"""Reusable fixed-budget paired validator for generated operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from operator_evolution_core.search import SearchBudget
from operator_evolution_core.validation import (
    ArmMeasurement,
    FitnessPolicy,
    GenericPairedCandidateValidator,
    abba_timing_order,
)

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
        return self._generic_validator(recorder).validate(
            validation_environments,
            population,
            parent_name,
            candidate,
            generation=generation,
            candidate_index=candidate_index,
            root_run_id=root_run_id,
            safety_failures=contract_failures,
            persist_evidence=recorder is not None,
        )

    def _generic_validator(
        self,
        recorder: TrajectoryRecorder | None,
    ) -> GenericPairedCandidateValidator[Environment2D, list[tuple[float, float]], PathOperator]:
        # Lazy import avoids initializing the domain/operator/search cycle while
        # the public evolution package itself is still importing.
        from ..domain.uav_adapter import UAVDomainAdapter

        def run_arm(
            population: Sequence[PathOperator],
            environment: Environment2D,
            initial: list[tuple[float, float]],
            seed: int,
            target_name: str,
            generation: int,
            run_id: str,
            persist_evidence: bool,
        ) -> ArmMeasurement:
            result, runtime_ms = self._run_arm(
                list(population),
                environment,
                initial,
                seed,
                recorder if persist_evidence else None,
                run_id,
                generation,
            )
            return self._measurement(result, runtime_ms, target_name)

        search = self.config.search
        return GenericPairedCandidateValidator(
            adapter=UAVDomainAdapter(
                self.evaluator,
                initializer_grid_resolution=self.config.maps.grid_resolution,
            ),
            budget=SearchBudget(
                max_iterations=search.validation_iterations,
                recent_window=search.recent_window,
            ),
            retention_config=self.config.evolution,
            master_seed=self.config.seed,
            seed_deriver=derive_seed,
            arm_runner=run_arm,
            operator_id=lambda operator: str(operator.name),
            instance_id=lambda environment: environment.map_id,
            context_label=lambda environment: environment.difficulty,
            runtime_repetitions=(
                self.config.evolution.runtime_validation_repetitions
            ),
            fitness_policy=FitnessPolicy.UAV_LEGACY_V1,
            specialist_contexts=frozenset({"dense", "corridor"}),
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
        return self._generic_validator(recorder).compare(
            validation_environments,
            parent_population,
            candidate_population,
            parent_operator_id=parent_name,
            candidate_operator_id=candidate_name,
            generation=generation,
            candidate_index=candidate_index,
            root_run_id=root_run_id,
            persist_evidence=recorder is not None,
        )

    @staticmethod
    def _abba_order(repetition: int) -> str:
        return abba_timing_order(repetition + 1)[-1]

    @staticmethod
    def _measurement(
        result: SearchResult,
        total_runtime_ms: float,
        operator_name: str,
    ) -> ArmMeasurement:
        calls = [
            step for step in result.steps if step.operator_name == operator_name
        ]
        changed_calls = sum(
            step.operator_result.success
            and step.candidate_path != step.path_before
            for step in calls
        )
        return ArmMeasurement(
            best_cost=float(result.best_evaluation.total_cost),
            feasible=bool(result.best_evaluation.feasible),
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
