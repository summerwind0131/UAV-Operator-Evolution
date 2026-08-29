"""Fixed iterative local search with complete per-call state capture."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import exp
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from operator_evolution_core.search import (
    GenericSearchKernel,
    SearchBudget,
    SearchStep as CoreSearchStep,
)

from ..domain.adapters import objective_to_evaluation_result
from ..domain.uav_adapter import UAVDomainAdapter
from ..environment.environment import Environment2D, extract_environment_features
from ..operators.base import OperatorResult, PathOperator, copied_path
from ..path.evaluator import PathEvaluator
from ..path.features import extract_path_features
from ..path.models import EvaluationResult, Path
from .acceptance import SimulatedAnnealingAcceptance
from .context import SearchContext
from .core_adapter import (
    UAVSchedulerFacade,
    UAVSearchOperatorFacade,
    core_context_to_uav,
    outcome_to_uav_result,
    sanitize_uav_operator_result,
    validate_uav_initial_path,
)
from .scheduler import BlockRandomRoundRobinScheduler, OperatorScheduler

if TYPE_CHECKING:
    from ..trajectory.recorder import TrajectoryRecorder


@dataclass(frozen=True, slots=True)
class SearchStep:
    """Complete in-memory record of one proposed search transition."""

    iteration: int
    operator_id: str
    operator_name: str
    path_before: Path
    candidate_path: Path
    current_path_after: Path
    evaluation_before: EvaluationResult
    candidate_evaluation: EvaluationResult
    current_evaluation_after: EvaluationResult
    best_evaluation_before: EvaluationResult
    best_evaluation_after: EvaluationResult
    context_before: SearchContext
    context_after: SearchContext
    operator_result: OperatorResult
    accepted: bool
    created_new_best: bool
    temperature: float
    runtime_ms: float

    @property
    def immediate_reward(self) -> float:
        return float(self.evaluation_before.total_cost - self.candidate_evaluation.total_cost)

    @property
    def path_after(self) -> Path:
        """Alias for the direct operator output, even when SA rejects it."""

        return self.candidate_path


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Immutable summary of one completed fixed-budget search run."""

    initial_path: Path
    final_path: Path
    best_path: Path
    initial_evaluation: EvaluationResult
    final_evaluation: EvaluationResult
    best_evaluation: EvaluationResult
    steps: tuple[SearchStep, ...]
    accepted_count: int

    @property
    def iterations(self) -> int:
        return len(self.steps)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_count / len(self.steps) if self.steps else 0.0

    @property
    def cost_history(self) -> tuple[float, ...]:
        return (
            float(self.initial_evaluation.total_cost),
            *(float(step.current_evaluation_after.total_cost) for step in self.steps),
        )

    @property
    def best_cost_history(self) -> tuple[float, ...]:
        return (
            float(self.initial_evaluation.total_cost),
            *(float(step.best_evaluation_after.total_cost) for step in self.steps),
        )


StepCallback = Callable[[SearchStep], None]


class SearchExecutor:
    """Run a fixed operator schedule and fixed simulated-annealing rule."""

    def __init__(
        self,
        operators: Sequence[PathOperator],
        evaluator: PathEvaluator | None = None,
        *,
        max_iterations: int = 100,
        temperature_start_ratio: float = 0.05,
        temperature_end_ratio: float = 0.001,
        recent_window: int = 10,
        initializer_grid_resolution: float = 4.0,
        scheduler: OperatorScheduler | None = None,
        recorder: "TrajectoryRecorder | None" = None,
    ) -> None:
        if not operators:
            raise ValueError("SearchExecutor requires at least one operator")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if recent_window <= 0:
            raise ValueError("recent_window must be positive")
        if initializer_grid_resolution <= 0:
            raise ValueError("initializer_grid_resolution must be positive")
        self.operators = tuple(operators)
        self.evaluator = evaluator or PathEvaluator()
        self.max_iterations = int(max_iterations)
        self.recent_window = int(recent_window)
        self.initializer_grid_resolution = float(initializer_grid_resolution)
        self.scheduler = scheduler or BlockRandomRoundRobinScheduler()
        self.acceptance = SimulatedAnnealingAcceptance(
            start_temperature_ratio=temperature_start_ratio,
            end_temperature_ratio=temperature_end_ratio,
        )
        self.recorder = recorder

    def run(
        self,
        environment: Environment2D,
        rng: np.random.Generator,
        initial_path: Path | None = None,
        *,
        recorder: "TrajectoryRecorder | None" = None,
        on_step: StepCallback | None = None,
        run_id: str = "run",
        generation: int = 0,
    ) -> SearchResult:
        """Execute exactly ``max_iterations`` operator calls.

        A caller-provided path is copied before evaluation.  Operators receive a
        second private copy, so even a non-conforming implementation cannot
        mutate the executor's current solution.
        """

        active_recorder = recorder if recorder is not None else self.recorder
        adapter = UAVDomainAdapter(
            self.evaluator,
            initializer_grid_resolution=self.initializer_grid_resolution,
        )
        facades = tuple(UAVSearchOperatorFacade(operator) for operator in self.operators)
        kernel = GenericSearchKernel(
            adapter=adapter,
            operators=facades,
            scheduler=UAVSchedulerFacade(self.scheduler),
            acceptance=self.acceptance,
            budget=SearchBudget(
                max_iterations=self.max_iterations,
                recent_window=self.recent_window,
            ),
            clock=perf_counter,
        )
        steps: list[SearchStep] = []

        def handle_core_step(
            core_step: CoreSearchStep[Path],
            facade: UAVSearchOperatorFacade,
        ) -> None:
            step = self._from_core_step(core_step)
            steps.append(step)
            if active_recorder is not None:
                self._record_step(
                    active_recorder,
                    step,
                    facade.native_operator,
                    environment,
                    run_id,
                    generation,
                )
            if on_step is not None:
                on_step(step)

        core_result = kernel.run(
            environment,
            rng,
            initial_solution=initial_path,
            on_step=handle_core_step,
        )

        return SearchResult(
            initial_path=copied_path(core_result.initial_solution),
            final_path=copied_path(core_result.final_solution),
            best_path=copied_path(core_result.best_solution),
            initial_evaluation=objective_to_evaluation_result(
                core_result.initial_evaluation
            ),
            final_evaluation=objective_to_evaluation_result(
                core_result.final_evaluation
            ),
            best_evaluation=objective_to_evaluation_result(
                core_result.best_evaluation
            ),
            steps=tuple(steps),
            accepted_count=core_result.accepted_count,
        )

    @staticmethod
    def _validate_initial_path(path: Path, environment: Environment2D) -> None:
        validate_uav_initial_path(path, environment)

    @classmethod
    def _sanitize_result(
        cls,
        result: object,
        parent: Path,
        environment: Environment2D,
    ) -> OperatorResult:
        return sanitize_uav_operator_result(result, parent, environment)

    @staticmethod
    def _from_core_step(core_step: CoreSearchStep[Path]) -> SearchStep:
        return SearchStep(
            iteration=core_step.iteration,
            operator_id=core_step.operator_id,
            operator_name=core_step.operator_name,
            path_before=copied_path(core_step.solution_before),
            candidate_path=copied_path(core_step.candidate_solution),
            current_path_after=copied_path(core_step.current_solution_after),
            evaluation_before=objective_to_evaluation_result(
                core_step.evaluation_before
            ),
            candidate_evaluation=objective_to_evaluation_result(
                core_step.candidate_evaluation
            ),
            current_evaluation_after=objective_to_evaluation_result(
                core_step.current_evaluation_after
            ),
            best_evaluation_before=objective_to_evaluation_result(
                core_step.best_evaluation_before
            ),
            best_evaluation_after=objective_to_evaluation_result(
                core_step.best_evaluation_after
            ),
            context_before=core_context_to_uav(core_step.context_before),
            context_after=core_context_to_uav(core_step.context_after),
            operator_result=outcome_to_uav_result(core_step.operator_outcome),
            accepted=core_step.accepted,
            created_new_best=core_step.created_new_best,
            temperature=core_step.temperature,
            runtime_ms=core_step.runtime_ms,
        )

    def _record_step(
        self,
        recorder: "TrajectoryRecorder",
        step: SearchStep,
        operator: PathOperator,
        environment: Environment2D,
        run_id: str,
        generation: int,
    ) -> None:
        from ..trajectory.models import OperatorTrace

        path_features_before = _model_mapping(
            extract_path_features(step.path_before, environment, evaluator=self.evaluator)
        )
        path_features_after = _model_mapping(
            extract_path_features(step.candidate_path, environment, evaluator=self.evaluator)
        )
        path_features_accepted = _model_mapping(
            extract_path_features(step.current_path_after, environment, evaluator=self.evaluator)
        )
        environment_features = _model_mapping(extract_environment_features(environment))
        search_features_before = step.context_before.as_features()
        search_features_after = step.context_after.as_features()
        before_components = _evaluation_components(step.evaluation_before)
        candidate_components = _evaluation_components(step.candidate_evaluation)
        accepted_components = _evaluation_components(step.current_evaluation_after)
        before_state = _state_snapshot(
            step.path_before,
            path_features_before,
            search_features_before,
            step.evaluation_before,
        )
        candidate_state = _state_snapshot(
            step.candidate_path,
            path_features_after,
            search_features_after,
            step.candidate_evaluation,
        )
        accepted_state = _state_snapshot(
            step.current_path_after,
            path_features_accepted,
            search_features_after,
            step.current_evaluation_after,
        )
        delta = float(step.candidate_evaluation.total_cost - step.evaluation_before.total_cost)
        if not step.operator_result.success:
            acceptance_reason = "operator_failed"
            acceptance_probability = 0.0
        elif delta <= 0.0:
            acceptance_reason = "non_worsening"
            acceptance_probability = 1.0
        else:
            acceptance_probability = exp(-delta / max(step.temperature, 1e-12))
            acceptance_reason = (
                "simulated_annealing" if step.accepted else "simulated_annealing_rejected"
            )
        parent_operator_ids = list(getattr(operator, "parent_operator_ids", ()))
        operator_info = _json_value(step.operator_result.info)
        payload = {
            "run_id": run_id,
            "map_id": environment.map_id,
            "map_difficulty": environment.difficulty,
            "iteration": step.iteration,
            # Older serialized maps may contain an unsigned 64-bit seed.  Keep
            # trajectory persistence valid for SQLite's signed INTEGER range.
            "seed": int(environment.seed) & ((1 << 63) - 1),
            "operator_id": step.operator_id,
            "operator_family": type(operator).__module__.rsplit(".", 1)[-1],
            "operator_params": operator_info,
            "context": {
                "search_features": search_features_before,
                "environment_features": environment_features,
            },
            "before_state": before_state,
            "candidate_state": candidate_state,
            "accepted_state": accepted_state,
            "before_objective": float(step.evaluation_before.total_cost),
            "candidate_objective": float(step.candidate_evaluation.total_cost),
            "accepted_objective": float(step.current_evaluation_after.total_cost),
            "before_components": before_components,
            "candidate_components": candidate_components,
            "accepted_components": accepted_components,
            "before_feasible": bool(step.evaluation_before.feasible),
            "candidate_feasible": bool(step.candidate_evaluation.feasible),
            "accepted_feasible": bool(step.current_evaluation_after.feasible),
            "immediate_reward": step.immediate_reward,
            "accepted": step.accepted,
            "acceptance_reason": acceptance_reason,
            "acceptance_probability": acceptance_probability,
            "temperature": float(step.temperature),
            "runtime_ms": float(step.runtime_ms),
            "metadata": {
                "operator_name": step.operator_name,
                "parent_operator_ids": parent_operator_ids,
                "generation": int(getattr(operator, "generation", generation)),
                "operator_success": step.operator_result.success,
                "failure_reason": step.operator_result.failure_reason,
                "modified_indices": list(step.operator_result.modified_indices),
                "best_cost_before": float(step.best_evaluation_before.total_cost),
                "best_cost_after": float(step.best_evaluation_after.total_cost),
                "created_new_best": step.created_new_best,
            },
            # Explicit compatibility fields preserve the user-facing trace schema.
            "path_before": [list(point) for point in step.path_before],
            "path_after": [list(point) for point in step.candidate_path],
            "path_features_before": path_features_before,
            "path_features_after": path_features_after,
            "search_features_before": search_features_before,
            "search_features_after": search_features_after,
            "environment_features": environment_features,
            "operator_info": operator_info,
            "cost_before": float(step.evaluation_before.total_cost),
            "cost_after": float(step.candidate_evaluation.total_cost),
            "feasible_before": bool(step.evaluation_before.feasible),
            "feasible_after": bool(step.candidate_evaluation.feasible),
            "collision_count_before": int(step.evaluation_before.collision_count),
            "collision_count_after": int(step.candidate_evaluation.collision_count),
            "best_cost_before": float(step.best_evaluation_before.total_cost),
            "best_cost_after": float(step.best_evaluation_after.total_cost),
            "created_new_best": step.created_new_best,
            "parent_operator_ids": parent_operator_ids,
            "generation": int(getattr(operator, "generation", generation)),
        }
        trace = OperatorTrace(**payload)
        recorder.record(trace)


def _model_mapping(value: object) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return dict(value.model_dump(mode="json"))
    if hasattr(value, "__dict__"):
        return {key: _json_value(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    raise TypeError(f"feature extractor returned unsupported type: {type(value).__name__}")


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _evaluation_components(evaluation: EvaluationResult) -> dict[str, float]:
    return {
        "path_length": float(evaluation.path_length),
        "collision_penalty": float(evaluation.collision_penalty),
        "smoothness_penalty": float(evaluation.smoothness_penalty),
        "risk_penalty": float(evaluation.risk_penalty),
        "waypoint_penalty": float(evaluation.waypoint_penalty),
    }


def _state_snapshot(
    path: Path,
    path_features: Mapping[str, Any],
    search_features: Mapping[str, Any],
    evaluation: EvaluationResult,
) -> dict[str, Any]:
    """Build one canonical JSON state for the trajectory recorder."""

    return {
        "path": [list(point) for point in path],
        "path_features": _json_value(path_features),
        "search_features": _json_value(search_features),
        "objective": float(evaluation.total_cost),
        "objective_components": _evaluation_components(evaluation),
        "feasible": bool(evaluation.feasible),
        "collision_count": int(evaluation.collision_count),
        "minimum_clearance": float(evaluation.minimum_clearance),
    }
