"""Fixed iterative local search with complete per-call state capture."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, isfinite
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np

from ..environment.environment import Environment2D, extract_environment_features
from ..operators.base import OperatorResult, PathOperator, copied_path, unchanged_result
from ..path.evaluator import PathEvaluator
from ..path.features import extract_path_features
from ..path.initializer import initialize_path
from ..path.models import EvaluationResult, Path
from .acceptance import SimulatedAnnealingAcceptance
from .context import SearchContext
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

        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        stream_seeds = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=4,
            dtype=np.uint64,
        )
        initializer_rng = np.random.default_rng(stream_seeds[0])
        scheduler_rng = np.random.default_rng(stream_seeds[1])
        operator_seed_rng = np.random.default_rng(stream_seeds[2])
        acceptance_rng = np.random.default_rng(stream_seeds[3])
        source = (
            initialize_path(
                environment,
                rng=initializer_rng,
                grid_resolution=self.initializer_grid_resolution,
            )
            if initial_path is None
            else initial_path
        )
        initial = copied_path(source)
        self._validate_initial_path(initial, environment)
        initial_evaluation = self.evaluator.evaluate(initial, environment)
        current_path = copied_path(initial)
        current_evaluation = initial_evaluation
        best_path = copied_path(initial)
        best_evaluation = initial_evaluation
        cost_scale = max(abs(float(initial_evaluation.total_cost)), 1.0)
        recent_improvements: list[float] = []
        recent_acceptances: list[bool] = []
        stagnation_count = 0
        last_created_new_best = False
        steps: list[SearchStep] = []
        active_recorder = recorder if recorder is not None else self.recorder
        reset = getattr(self.scheduler, "reset", None)
        if callable(reset):
            reset()

        for iteration in range(self.max_iterations):
            context_before = SearchContext(
                iteration=iteration,
                max_iterations=self.max_iterations,
                current_evaluation=current_evaluation,
                best_evaluation=best_evaluation,
                stagnation_count=stagnation_count,
                recent_improvements=tuple(recent_improvements),
                recent_acceptances=tuple(recent_acceptances),
                last_created_new_best=last_created_new_best,
            )
            operator = self.scheduler.select(self.operators, iteration, scheduler_rng)
            operator_name = str(operator.name)
            operator_id = str(getattr(operator, "operator_id", operator_name))
            path_before = copied_path(current_path)
            operator_input = copied_path(current_path)
            operator_rng = np.random.default_rng(
                operator_seed_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            start_time = perf_counter()
            try:
                result = operator.apply(operator_input, environment, operator_rng, context_before)
            except Exception as exc:  # safe operator boundary is intentional
                result = unchanged_result(
                    path_before,
                    "operator raised an exception",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            runtime_ms = (perf_counter() - start_time) * 1000.0
            result = self._sanitize_result(result, path_before, environment)
            try:
                candidate_evaluation = self.evaluator.evaluate(result.path, environment)
            except Exception as exc:  # malformed numerical output is a safe no-op
                result = unchanged_result(
                    path_before,
                    "candidate evaluation failed",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                candidate_evaluation = current_evaluation

            temperature = self.acceptance.temperature(iteration, self.max_iterations, cost_scale)
            accepted = bool(
                result.success
                and self.acceptance.accept(
                    float(current_evaluation.total_cost),
                    float(candidate_evaluation.total_cost),
                    temperature,
                    acceptance_rng,
                )
            )
            best_before = best_evaluation
            if accepted:
                current_path = copied_path(result.path)
                current_evaluation = candidate_evaluation
            created_new_best = bool(
                accepted
                and float(current_evaluation.total_cost) < float(best_evaluation.total_cost) - 1e-12
            )
            if created_new_best:
                best_path = copied_path(current_path)
                best_evaluation = current_evaluation
                stagnation_count = 0
            else:
                stagnation_count += 1
            immediate_reward = float(
                context_before.current_evaluation.total_cost - candidate_evaluation.total_cost
            )
            recent_improvements.append(immediate_reward)
            recent_acceptances.append(accepted)
            del recent_improvements[: max(0, len(recent_improvements) - self.recent_window)]
            del recent_acceptances[: max(0, len(recent_acceptances) - self.recent_window)]
            context_after = SearchContext(
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
                current_evaluation=current_evaluation,
                best_evaluation=best_evaluation,
                stagnation_count=stagnation_count,
                recent_improvements=tuple(recent_improvements),
                recent_acceptances=tuple(recent_acceptances),
                last_created_new_best=created_new_best,
            )
            step = SearchStep(
                iteration=iteration,
                operator_id=operator_id,
                operator_name=operator_name,
                path_before=path_before,
                candidate_path=copied_path(result.path),
                current_path_after=copied_path(current_path),
                evaluation_before=context_before.current_evaluation,
                candidate_evaluation=candidate_evaluation,
                current_evaluation_after=current_evaluation,
                best_evaluation_before=best_before,
                best_evaluation_after=best_evaluation,
                context_before=context_before,
                context_after=context_after,
                operator_result=result,
                accepted=accepted,
                created_new_best=created_new_best,
                temperature=temperature,
                runtime_ms=runtime_ms,
            )
            steps.append(step)
            if active_recorder is not None:
                self._record_step(active_recorder, step, operator, environment, run_id, generation)
            if on_step is not None:
                on_step(step)
            last_created_new_best = created_new_best

        return SearchResult(
            initial_path=copied_path(initial),
            final_path=copied_path(current_path),
            best_path=copied_path(best_path),
            initial_evaluation=initial_evaluation,
            final_evaluation=current_evaluation,
            best_evaluation=best_evaluation,
            steps=tuple(steps),
            accepted_count=sum(step.accepted for step in steps),
        )

    @staticmethod
    def _validate_initial_path(path: Path, environment: Environment2D) -> None:
        if len(path) < 2:
            raise ValueError("initial path must contain at least start and goal")
        if path[0] != environment.start or path[-1] != environment.goal:
            raise ValueError("initial path endpoints must equal environment start and goal")
        if not all(
            len(point) == 2
            and all(isfinite(float(coordinate)) for coordinate in point)
            and environment.in_bounds(point)
            for point in path
        ):
            raise ValueError("initial path waypoints must be finite and in bounds")

    @classmethod
    def _sanitize_result(
        cls,
        result: object,
        parent: Path,
        environment: Environment2D,
    ) -> OperatorResult:
        if not isinstance(result, OperatorResult):
            return unchanged_result(parent, "operator returned an invalid result type")
        candidate = copied_path(result.path)
        valid = (
            len(candidate) >= 2
            and candidate[0] == parent[0]
            and candidate[-1] == parent[-1]
            and all(
                len(point) == 2
                and all(isfinite(float(coordinate)) for coordinate in point)
                and environment.in_bounds(point)
                for point in candidate
            )
        )
        if not valid:
            return unchanged_result(parent, "operator returned an invalid path")
        return OperatorResult(
            path=candidate,
            modified_indices=tuple(int(index) for index in result.modified_indices),
            success=bool(result.success),
            info=dict(result.info),
            failure_reason=result.failure_reason,
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
