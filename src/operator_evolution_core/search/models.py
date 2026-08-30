"""Domain-independent data models for one fixed-budget search run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, TypeVar

from pydantic import JsonValue

from ..contracts import ObjectiveEvaluation

SolutionT = TypeVar("SolutionT")


@dataclass(frozen=True, slots=True)
class SearchBudget:
    """Deterministic call budget and bounded recent-history window."""

    max_iterations: int
    recent_window: int = 10

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if self.recent_window <= 0:
            raise ValueError("recent_window must be positive")


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Read-only core state passed to an operator before one call."""

    iteration: int = 0
    max_iterations: int = 1
    current_evaluation: ObjectiveEvaluation | None = None
    best_evaluation: ObjectiveEvaluation | None = None
    stagnation_count: int = 0
    recent_improvements: tuple[float, ...] = ()
    recent_acceptances: tuple[bool, ...] = ()
    last_created_new_best: bool = False
    domain_features: dict[str, JsonValue] = field(default_factory=dict)

    @property
    def iteration_ratio(self) -> float:
        return min(
            max(float(self.iteration) / max(1, int(self.max_iterations)), 0.0),
            1.0,
        )

    @property
    def current_cost(self) -> float:
        if self.current_evaluation is None:
            return float("inf")
        return float(self.current_evaluation.scalar_cost)

    @property
    def best_cost(self) -> float:
        if self.best_evaluation is None:
            return float("inf")
        return float(self.best_evaluation.scalar_cost)

    @property
    def current_best_gap(self) -> float:
        if self.current_evaluation is None or self.best_evaluation is None:
            return 0.0
        scale = max(abs(self.best_cost), 1e-12)
        return max(0.0, (self.current_cost - self.best_cost) / scale)

    @property
    def best_cost_gap(self) -> float:
        return self.current_best_gap

    @property
    def recent_improvement_rate(self) -> float:
        if not self.recent_improvements:
            return 0.0
        return sum(value > 0.0 for value in self.recent_improvements) / len(
            self.recent_improvements
        )

    @property
    def recent_acceptance_rate(self) -> float:
        if not self.recent_acceptances:
            return 0.0
        return sum(self.recent_acceptances) / len(self.recent_acceptances)

    def as_features(self) -> dict[str, JsonValue]:
        return {
            "iteration_ratio": self.iteration_ratio,
            "stagnation_count": int(self.stagnation_count),
            "best_cost_gap": self.current_best_gap,
            "recent_improvement_rate": self.recent_improvement_rate,
            "recent_acceptance_rate": self.recent_acceptance_rate,
            "last_created_new_best": bool(self.last_created_new_best),
        }


@dataclass(slots=True)
class OperatorOutcome(Generic[SolutionT]):
    """Candidate returned by a domain operator through the core boundary."""

    solution: SolutionT
    changed_items: tuple[str | int, ...] = ()
    success: bool = True
    metadata: dict[str, JsonValue] = field(default_factory=dict)
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class SearchStep(Generic[SolutionT]):
    iteration: int
    operator_id: str
    operator_name: str
    solution_before: SolutionT
    candidate_solution: SolutionT
    current_solution_after: SolutionT
    evaluation_before: ObjectiveEvaluation
    candidate_evaluation: ObjectiveEvaluation
    current_evaluation_after: ObjectiveEvaluation
    best_evaluation_before: ObjectiveEvaluation
    best_evaluation_after: ObjectiveEvaluation
    context_before: SearchContext
    context_after: SearchContext
    operator_outcome: OperatorOutcome[SolutionT]
    accepted: bool
    created_new_best: bool
    temperature: float
    runtime_ms: float

    @property
    def immediate_reward(self) -> float:
        return float(
            self.evaluation_before.scalar_cost
            - self.candidate_evaluation.scalar_cost
        )


@dataclass(frozen=True, slots=True)
class SearchResult(Generic[SolutionT]):
    initial_solution: SolutionT
    final_solution: SolutionT
    best_solution: SolutionT
    initial_evaluation: ObjectiveEvaluation
    final_evaluation: ObjectiveEvaluation
    best_evaluation: ObjectiveEvaluation
    steps: tuple[SearchStep[SolutionT], ...]
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
            float(self.initial_evaluation.scalar_cost),
            *(
                float(step.current_evaluation_after.scalar_cost)
                for step in self.steps
            ),
        )

    @property
    def best_cost_history(self) -> tuple[float, ...]:
        return (
            float(self.initial_evaluation.scalar_cost),
            *(float(step.best_evaluation_after.scalar_cost) for step in self.steps),
        )


__all__ = [
    "OperatorOutcome",
    "SearchBudget",
    "SearchContext",
    "SearchResult",
    "SearchStep",
    "SolutionT",
]

