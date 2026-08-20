"""A deliberately small counterfactual evaluator for diagnostic spot checks."""

from __future__ import annotations

import copy
import math
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..trajectory import OperatorTrace


class CounterfactualResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    state_index: int = Field(ge=0)
    source_trace_id: int | None = None
    operator_id: str
    before_objective: float | None = None
    candidate_objective: float | None = None
    reward: float | None = None
    advantage: float | None = None
    feasible: bool | None = None
    runtime_ms: float = Field(ge=0.0)
    candidate_state: dict[str, Any] | list[Any] | None = None
    error: str | None = None

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def _default_objective(state: Any) -> float | None:
    if isinstance(state, tuple) and len(state) == 2:
        candidate, explicit = state
        if isinstance(explicit, (int, float)) and not isinstance(explicit, bool):
            return float(explicit)
        state = candidate
    if isinstance(state, Mapping):
        for key in ("objective", "cost", "score", "total_cost"):
            value = state.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                result = float(value)
                return result if math.isfinite(result) else None
        metrics = state.get("metrics")
        if isinstance(metrics, Mapping):
            return _default_objective(metrics)
    return None


def _operator_items(
    operators: Mapping[str, Any] | Iterable[Any],
) -> list[tuple[str, Any]]:
    if isinstance(operators, Mapping):
        return [(str(name), operator) for name, operator in operators.items()]
    result: list[tuple[str, Any]] = []
    for index, operator in enumerate(operators):
        name = getattr(operator, "operator_id", None) or getattr(
            operator, "name", None
        )
        if not name:
            name = getattr(operator, "__name__", f"operator_{index}")
        result.append((str(name), operator))
    return result


class CounterfactualEvaluator:
    """Apply a handful of operators to identical copied states.

    This is intended for diagnosis, not for running the main search.  Sampling is
    deterministic for a given seed and exceptions become result rows so a single
    incompatible operator does not discard the comparison.
    """

    def __init__(
        self,
        objective: Callable[[Any], float | None] | None = None,
        *,
        max_states: int = 4,
        seed: int = 0,
    ) -> None:
        if max_states < 0:
            raise ValueError("max_states cannot be negative")
        self.objective = objective or _default_objective
        self.max_states = int(max_states)
        self.seed = int(seed)

    @staticmethod
    def _apply(operator: Any, state: Any) -> Any:
        if callable(operator):
            return operator(state)
        for method_name in ("apply", "propose", "transform"):
            method = getattr(operator, method_name, None)
            if callable(method):
                return method(state)
        raise TypeError("counterfactual operator is not callable and has no apply method")

    def evaluate(
        self,
        states: Sequence[Any] | Iterable[Any],
        operators: Mapping[str, Any] | Iterable[Any],
    ) -> list[CounterfactualResult]:
        """Evaluate every operator on the same deterministically sampled states."""

        supplied = list(states)
        if self.max_states == 0 or not supplied:
            return []
        indexes = list(range(len(supplied)))
        if len(indexes) > self.max_states:
            indexes = sorted(
                np.random.default_rng(self.seed).choice(
                    indexes, size=self.max_states, replace=False
                ).tolist()
            )
        operator_items = _operator_items(operators)
        results: list[CounterfactualResult] = []
        for state_index in indexes:
            source = supplied[state_index]
            trace = source if isinstance(source, OperatorTrace) else None
            state = trace.before_state if trace is not None else source
            before = (
                trace.before_objective
                if trace is not None and trace.before_objective is not None
                else self.objective(state)
            )
            for operator_id, operator in operator_items:
                started = time.perf_counter()
                candidate: Any = None
                error: str | None = None
                try:
                    returned = self._apply(operator, copy.deepcopy(state))
                    candidate = returned[0] if isinstance(returned, tuple) and returned else returned
                    candidate_objective = self.objective(returned)
                    feasible_raw = (
                        candidate.get("feasible")
                        if isinstance(candidate, Mapping)
                        else None
                    )
                    feasible = None if feasible_raw is None else bool(feasible_raw)
                except Exception as exc:  # Diagnostic probes deliberately isolate failures.
                    candidate_objective = None
                    feasible = None
                    error = f"{type(exc).__name__}: {exc}"
                runtime_ms = (time.perf_counter() - started) * 1000.0
                reward = (
                    before - candidate_objective
                    if before is not None and candidate_objective is not None
                    else None
                )
                serializable_candidate = (
                    candidate
                    if isinstance(candidate, (dict, list))
                    else None
                )
                results.append(
                    CounterfactualResult(
                        state_index=state_index,
                        source_trace_id=trace.trace_id if trace is not None else None,
                        operator_id=operator_id,
                        before_objective=before,
                        candidate_objective=candidate_objective,
                        reward=reward,
                        feasible=feasible,
                        runtime_ms=runtime_ms,
                        candidate_state=serializable_candidate,
                        error=error,
                    )
                )
        return self._with_advantages(results)

    @staticmethod
    def _with_advantages(results: list[CounterfactualResult]) -> list[CounterfactualResult]:
        """Attach reward-minus-other-operators advantage within each state."""

        updated: list[CounterfactualResult] = []
        for result in results:
            others = [
                item.reward
                for item in results
                if item.state_index == result.state_index
                and item.operator_id != result.operator_id
                and item.reward is not None
            ]
            advantage = (
                result.reward - float(np.mean(others))
                if result.reward is not None and others
                else None
            )
            updated.append(result.model_copy(update={"advantage": advantage}))
        return updated

    def evaluate_path_state(
        self,
        path: list[tuple[float, float]],
        environment: Any,
        context: Any,
        evaluator: Any,
        operators: Mapping[str, Any] | Iterable[Any],
        *,
        seed: int | None = None,
    ) -> list[CounterfactualResult]:
        """Apply PathOperators to identical path/map/context/common random numbers."""

        before = evaluator.evaluate(path, environment)
        common_seed = self.seed if seed is None else int(seed)
        results: list[CounterfactualResult] = []
        for operator_id, operator in _operator_items(operators):
            started = time.perf_counter()
            error: str | None = None
            candidate_state: dict[str, Any] | None = None
            try:
                operator_result = operator.apply(
                    copy.deepcopy(path),
                    environment,
                    np.random.default_rng(common_seed),
                    context,
                )
                candidate = list(operator_result.path)
                evaluation = evaluator.evaluate(candidate, environment)
                candidate_objective = float(evaluation.total_cost)
                feasible = bool(evaluation.feasible)
                candidate_state = {
                    "path": [list(point) for point in candidate],
                    "objective": candidate_objective,
                    "feasible": feasible,
                }
            except Exception as exc:
                candidate_objective = None
                feasible = None
                error = f"{type(exc).__name__}: {exc}"
            results.append(
                CounterfactualResult(
                    state_index=0,
                    operator_id=operator_id,
                    before_objective=float(before.total_cost),
                    candidate_objective=candidate_objective,
                    reward=(
                        float(before.total_cost) - candidate_objective
                        if candidate_objective is not None
                        else None
                    ),
                    feasible=feasible,
                    runtime_ms=(time.perf_counter() - started) * 1000.0,
                    candidate_state=candidate_state,
                    error=error,
                )
            )
        return self._with_advantages(results)

    evaluate_traces = evaluate
