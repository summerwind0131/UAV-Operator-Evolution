"""UAV compatibility surface for generic paired retention and path smoke."""

from __future__ import annotations

import math
import time
from typing import AbstractSet, Any

import numpy as np

from operator_evolution_core.validation.paired import (
    PairedOutcome,
    RetentionConfig,
    ValidationReport,
    decide_retention as _decide_retention,
    paired_bootstrap_ci,
)

from .fitness import FitnessPolicy


def decide_retention(
    parent_operator: str,
    candidate_operator: str,
    outcomes: list[PairedOutcome],
    config: RetentionConfig,
    *,
    safety_passed: bool = True,
    safety_failures: list[str] | None = None,
    bootstrap_seed: int = 0,
    fitness_policy: FitnessPolicy | str = FitnessPolicy.UAV_LEGACY_V1,
    specialist_contexts: AbstractSet[str] = frozenset({"dense", "corridor"}),
) -> ValidationReport:
    """Preserve UAV v1 retention semantics over the generic core gate."""

    return _decide_retention(
        parent_operator,
        candidate_operator,
        outcomes,
        config,
        safety_passed=safety_passed,
        safety_failures=safety_failures,
        bootstrap_seed=bootstrap_seed,
        fitness_policy=fitness_policy,
        specialist_contexts=specialist_contexts,
    )


def validate_operator_contract(
    operator: Any,
    path: list[tuple[float, float]],
    environment: Any,
    context: Any,
    seeds: list[int],
    max_waypoints: int,
    deadline_ms: float,
) -> list[str]:
    """Exercise UAV operators against the historical path safety invariants."""

    failures: list[str] = []
    for seed in seeds:
        original = list(path)
        started = time.perf_counter()
        try:
            result = operator.apply(
                list(path), environment, np.random.default_rng(seed), context
            )
        except Exception as exc:
            failures.append(f"seed {seed}: exception {type(exc).__name__}: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate = list(getattr(result, "path", []))
        if path != original:
            failures.append(f"seed {seed}: mutated input path")
        if len(candidate) < 2 or len(candidate) > max_waypoints:
            failures.append(f"seed {seed}: invalid waypoint count {len(candidate)}")
        elif candidate[0] != original[0] or candidate[-1] != original[-1]:
            failures.append(f"seed {seed}: changed endpoint")
        if any(
            not math.isfinite(float(value))
            for point in candidate
            for value in point
        ):
            failures.append(f"seed {seed}: non-finite coordinate")
        if elapsed_ms > max(deadline_ms * 3.0, deadline_ms + 10.0):
            failures.append(
                f"seed {seed}: exceeded contract runtime ({elapsed_ms:.3f} ms)"
            )
    return failures


__all__ = [
    "PairedOutcome",
    "RetentionConfig",
    "ValidationReport",
    "decide_retention",
    "paired_bootstrap_ci",
    "validate_operator_contract",
]
