"""Read-only search state passed to path operators."""

from __future__ import annotations

from dataclasses import dataclass

from ..path.models import EvaluationResult


@dataclass(frozen=True, slots=True)
class SearchContext:
    """Snapshot of fixed-search state immediately before an operator call."""

    iteration: int = 0
    max_iterations: int = 1
    current_evaluation: EvaluationResult | None = None
    best_evaluation: EvaluationResult | None = None
    stagnation_count: int = 0
    recent_improvements: tuple[float, ...] = ()
    recent_acceptances: tuple[bool, ...] = ()
    last_created_new_best: bool = False

    @property
    def iteration_ratio(self) -> float:
        return min(max(float(self.iteration) / max(1, int(self.max_iterations)), 0.0), 1.0)

    @property
    def current_cost(self) -> float:
        return float(self.current_evaluation.total_cost) if self.current_evaluation is not None else float("inf")

    @property
    def best_cost(self) -> float:
        return float(self.best_evaluation.total_cost) if self.best_evaluation is not None else float("inf")

    @property
    def current_best_gap(self) -> float:
        if self.current_evaluation is None or self.best_evaluation is None:
            return 0.0
        scale = max(abs(self.best_cost), 1e-12)
        return max(0.0, (self.current_cost - self.best_cost) / scale)

    @property
    def best_cost_gap(self) -> float:
        """Compatibility alias used by diagnostics."""

        return self.current_best_gap

    @property
    def recent_improvement_rate(self) -> float:
        if not self.recent_improvements:
            return 0.0
        return sum(value > 0.0 for value in self.recent_improvements) / len(self.recent_improvements)

    @property
    def recent_acceptance_rate(self) -> float:
        if not self.recent_acceptances:
            return 0.0
        return sum(self.recent_acceptances) / len(self.recent_acceptances)

    def as_features(self) -> dict[str, float | int | bool]:
        """Return the JSON-compatible search features required by traces."""

        return {
            "iteration_ratio": self.iteration_ratio,
            "stagnation_count": int(self.stagnation_count),
            "best_cost_gap": self.current_best_gap,
            "recent_improvement_rate": self.recent_improvement_rate,
            "recent_acceptance_rate": self.recent_acceptance_rate,
            "last_created_new_best": bool(self.last_created_new_best),
        }
