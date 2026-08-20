"""Shared contracts, budgets, counters, and trusted validation for planners."""

from __future__ import annotations

import math
import time
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..environment.environment import Environment2D
from ..path.evaluator import PathEvaluator
from ..path.models import EvaluationResult, Path, copy_and_validate_path

PlannerStatus = Literal[
    "success",
    "no_path",
    "timeout",
    "budget_exhausted",
    "invalid_path",
    "error",
]


class PlanningBudget(BaseModel):
    """Two independent hard limits shared by every planner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    time_limit_seconds: float = Field(1.0, gt=0)
    max_objective_evaluations: int = Field(2_000, ge=1)


class PlannerResult(BaseModel):
    """Planner output before and after trusted runner-side validation."""

    model_config = ConfigDict(extra="forbid")

    planner: str
    status: PlannerStatus
    path: Path | None = None
    message: str = ""
    objective_evaluations: int = Field(0, ge=0)
    collision_checks: int = Field(0, ge=0)
    node_expansions: int = Field(0, ge=0)
    elapsed_seconds: float = Field(0.0, ge=0)
    diagnostics: dict[str, Any] = Field(default_factory=dict)
    trusted_evaluation: EvaluationResult | None = None


class PlanningLimitReached(RuntimeError):
    """Base class for cooperative benchmark limits."""


class PlanningTimeout(PlanningLimitReached):
    """Raised once the wall-clock limit is reached."""


class ObjectiveBudgetExhausted(PlanningLimitReached):
    """Raised before an objective evaluation would exceed the limit."""


class BudgetedEvaluator:
    """Single metered gateway for objective and collision queries."""

    def __init__(
        self,
        environment: Environment2D,
        evaluator: PathEvaluator,
        budget: PlanningBudget,
    ) -> None:
        self.environment = environment
        self.evaluator = evaluator
        self.budget = budget
        self.started_at = time.perf_counter()
        self.objective_evaluations = 0
        self.collision_checks = 0
        self.node_expansions = 0

    @property
    def elapsed_seconds(self) -> float:
        return time.perf_counter() - self.started_at

    @property
    def remaining_evaluations(self) -> int:
        return max(
            0,
            self.budget.max_objective_evaluations - self.objective_evaluations,
        )

    def check_time(self) -> None:
        if self.elapsed_seconds >= self.budget.time_limit_seconds:
            raise PlanningTimeout("wall-clock planning budget exhausted")

    def evaluate(self, path: Path) -> EvaluationResult:
        self.check_time()
        if self.objective_evaluations >= self.budget.max_objective_evaluations:
            raise ObjectiveBudgetExhausted("objective-evaluation budget exhausted")
        self.objective_evaluations += 1
        return self.evaluator.evaluate(path, self.environment)

    def point_is_free(self, point: tuple[float, float]) -> bool:
        self.check_time()
        self.collision_checks += 1
        return self.environment.point_is_collision_free(point)

    def segment_is_free(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
    ) -> bool:
        self.check_time()
        self.collision_checks += 1
        return self.environment.segment_is_collision_free(start, end)

    def expand(self, count: int = 1) -> None:
        self.check_time()
        self.node_expansions += count

    def record_external_counts(
        self,
        *,
        objective_evaluations: int,
        collision_checks: int,
        node_expansions: int,
    ) -> None:
        """Import audited counters from a bounded generated-solver subprocess."""

        values = (objective_evaluations, collision_checks, node_expansions)
        if any(isinstance(value, bool) or int(value) != value or value < 0 for value in values):
            raise ValueError("external planner counters must be non-negative integers")
        if self.objective_evaluations + objective_evaluations > self.budget.max_objective_evaluations:
            raise ObjectiveBudgetExhausted("external solver exceeded objective-evaluation budget")
        self.objective_evaluations += int(objective_evaluations)
        self.collision_checks += int(collision_checks)
        self.node_expansions += int(node_expansions)

    def result(
        self,
        planner: str,
        status: PlannerStatus,
        *,
        path: Path | None = None,
        message: str = "",
        diagnostics: dict[str, Any] | None = None,
    ) -> PlannerResult:
        return PlannerResult(
            planner=planner,
            status=status,
            path=path,
            message=message,
            objective_evaluations=self.objective_evaluations,
            collision_checks=self.collision_checks,
            node_expansions=self.node_expansions,
            elapsed_seconds=self.elapsed_seconds,
            diagnostics=diagnostics or {},
        )


class Planner(Protocol):
    """Minimal planner interface used by the independent benchmark runner."""

    name: str
    stochastic: bool
    research_claim_eligible: bool

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        """Return a best-so-far path or an explicit failure status."""


def simplify_path(path: Path, problem: BudgetedEvaluator) -> Path:
    """Greedy continuous line-of-sight simplification with metered checks."""

    canonical = copy_and_validate_path(path)
    if len(canonical) <= 2:
        return canonical
    simplified: Path = [canonical[0]]
    anchor = 0
    while anchor < len(canonical) - 1:
        candidate = len(canonical) - 1
        while candidate > anchor + 1:
            if problem.segment_is_free(canonical[anchor], canonical[candidate]):
                break
            candidate -= 1
        simplified.append(canonical[candidate])
        anchor = candidate
    return simplified


def path_hash(path: Path | None) -> str | None:
    """Hash rounded path coordinates for duplicate and reproducibility checks."""

    if path is None:
        return None
    from ..reproducibility import stable_hash

    return stable_hash([[round(x, 10), round(y, 10)] for x, y in path])


def run_with_trusted_validation(
    planner: Planner,
    environment: Environment2D,
    evaluator: PathEvaluator,
    budget: PlanningBudget,
    rng: np.random.Generator,
) -> PlannerResult:
    """Execute a planner and independently validate any returned path."""

    problem = BudgetedEvaluator(environment, evaluator, budget)
    try:
        result = planner.plan(problem, budget, rng)
    except PlanningTimeout as exc:
        result = problem.result(planner.name, "timeout", message=str(exc))
    except ObjectiveBudgetExhausted as exc:
        result = problem.result(planner.name, "budget_exhausted", message=str(exc))
    except Exception as exc:  # pragma: no cover - defensive experiment boundary
        result = problem.result(
            planner.name,
            "error",
            message=f"{type(exc).__name__}: {exc}",
        )

    result.objective_evaluations = problem.objective_evaluations
    result.collision_checks = problem.collision_checks
    result.node_expansions = problem.node_expansions
    result.elapsed_seconds = problem.elapsed_seconds
    if not math.isfinite(result.elapsed_seconds):
        result.status = "error"
        result.message = "non-finite elapsed time"
        result.path = None
    elif (
        result.elapsed_seconds >= budget.time_limit_seconds
        and result.status == "success"
    ):
        result.status = "timeout"
        result.message = (
            result.message + "; " if result.message else ""
        ) + "planner completed beyond the trusted wall-clock boundary"

    if result.path is not None:
        try:
            # This runner-side check is intentionally outside the optimization
            # evaluation budget: algorithms cannot forge feasibility or cost.
            trusted = evaluator.evaluate(result.path, environment)
            result.trusted_evaluation = trusted
            if not trusted.feasible and result.status == "success":
                result.status = "invalid_path"
                result.message = "returned path failed trusted hard-constraint validation"
        except Exception as exc:
            result.status = "invalid_path"
            result.message = f"trusted validation failed: {type(exc).__name__}: {exc}"
            result.trusted_evaluation = None
    elif result.status == "success":
        result.status = "no_path"
        result.message = "planner reported success without a path"
    return result


__all__ = [
    "BudgetedEvaluator",
    "ObjectiveBudgetExhausted",
    "Planner",
    "PlannerResult",
    "PlannerStatus",
    "PlanningBudget",
    "PlanningLimitReached",
    "PlanningTimeout",
    "path_hash",
    "run_with_trusted_validation",
    "simplify_path",
]
