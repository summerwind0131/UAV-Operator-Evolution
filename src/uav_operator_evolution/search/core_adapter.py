"""Compatibility facades between UAV v1 search objects and the core kernel."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from operator_evolution_core.search import (
    OperatorOutcome,
    SearchContext as CoreSearchContext,
)

from ..domain.adapters import objective_to_evaluation_result
from ..environment.environment import Environment2D
from ..operators.base import OperatorResult, PathOperator, copied_path, unchanged_result
from ..path.models import Path
from .context import SearchContext
from .scheduler import OperatorScheduler


def core_context_to_uav(context: CoreSearchContext) -> SearchContext:
    current = (
        None
        if context.current_evaluation is None
        else objective_to_evaluation_result(context.current_evaluation)
    )
    best = (
        None
        if context.best_evaluation is None
        else objective_to_evaluation_result(context.best_evaluation)
    )
    return SearchContext(
        iteration=context.iteration,
        max_iterations=context.max_iterations,
        current_evaluation=current,
        best_evaluation=best,
        stagnation_count=context.stagnation_count,
        recent_improvements=context.recent_improvements,
        recent_acceptances=context.recent_acceptances,
        last_created_new_best=context.last_created_new_best,
    )


def validate_uav_initial_path(path: Path, environment: Environment2D) -> None:
    """Preserve the exact validation messages exposed by UAV SearchExecutor."""

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


def sanitize_uav_operator_result(
    result: object,
    parent: Path,
    environment: Environment2D,
) -> OperatorResult:
    """Apply the historical UAV result boundary before entering the core."""

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


def outcome_to_uav_result(outcome: OperatorOutcome[Path]) -> OperatorResult:
    return OperatorResult(
        path=copied_path(outcome.solution),
        modified_indices=tuple(int(index) for index in outcome.changed_items),
        success=bool(outcome.success),
        info=dict(outcome.metadata),
        failure_reason=outcome.failure_reason,
    )


@dataclass(slots=True)
class UAVSearchOperatorFacade:
    """Expose an existing PathOperator through the generic operator contract."""

    native_operator: PathOperator

    @property
    def name(self) -> str:
        return str(self.native_operator.name)

    @property
    def operator_id(self) -> str:
        return str(getattr(self.native_operator, "operator_id", self.name))

    def apply(
        self,
        solution: Path,
        instance: Environment2D,
        rng: np.random.Generator,
        context: CoreSearchContext,
    ) -> OperatorOutcome[Path]:
        result = self.native_operator.apply(
            solution,
            instance,
            rng,
            core_context_to_uav(context),
        )
        sanitized = sanitize_uav_operator_result(result, solution, instance)
        return OperatorOutcome(
            solution=copied_path(sanitized.path),
            changed_items=tuple(sanitized.modified_indices),
            success=bool(sanitized.success),
            metadata=dict(sanitized.info),
            failure_reason=sanitized.failure_reason,
        )


class UAVSchedulerFacade:
    """Let a legacy scheduler continue to observe the native operator objects."""

    def __init__(self, scheduler: OperatorScheduler) -> None:
        self.native_scheduler = scheduler

    def reset(self) -> None:
        reset = getattr(self.native_scheduler, "reset", None)
        if callable(reset):
            reset()

    def select(
        self,
        operators: Sequence[UAVSearchOperatorFacade],
        iteration: int,
        rng: np.random.Generator,
    ) -> UAVSearchOperatorFacade:
        native_operators = tuple(operator.native_operator for operator in operators)
        selected = self.native_scheduler.select(native_operators, iteration, rng)
        for facade in operators:
            if facade.native_operator is selected:
                return facade
        raise ValueError("UAV scheduler returned an operator outside the population")


__all__ = [
    "UAVSchedulerFacade",
    "UAVSearchOperatorFacade",
    "core_context_to_uav",
    "outcome_to_uav_result",
    "sanitize_uav_operator_result",
    "validate_uav_initial_path",
]
