"""Within-generation rank-normalized operator fitness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Any

import pandas as pd


class FitnessPolicy(StrEnum):
    """Versioned parent-ranking semantics."""

    UAV_LEGACY_V1 = "uav-legacy-v1"
    DETERMINISTIC_V2 = "deterministic-v2"


FITNESS_WEIGHTS = {
    "cost": -0.45,
    "feasible": 0.25,
    "delayed": 0.15,
    "worst_context": 0.10,
    "runtime": -0.05,
}

DETERMINISTIC_FITNESS_WEIGHTS = {
    "cost": -0.50,
    "feasible": 0.25,
    "delayed": 0.15,
    "worst_context": 0.10,
}


def compute_fitness(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy: FitnessPolicy | str = FitnessPolicy.UAV_LEGACY_V1,
) -> dict[str, float]:
    """Rank operators without allowing a single metric's scale to dominate."""

    if not rows:
        return {}
    selected = FitnessPolicy(policy)
    weights = (
        FITNESS_WEIGHTS
        if selected is FitnessPolicy.UAV_LEGACY_V1
        else DETERMINISTIC_FITNESS_WEIGHTS
    )
    frame = pd.DataFrame(rows).copy()
    required = {"operator_name", *weights}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fitness rows missing columns: {sorted(missing)}")
    n = max(len(frame) - 1, 1)
    scores = pd.Series(0.0, index=frame.index)
    for column, weight in weights.items():
        numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        percentile = (numeric.rank(method="average") - 1.0) / n
        scores += weight * percentile
    return {
        str(name): float(score)
        for name, score in zip(frame["operator_name"].tolist(), scores.tolist(), strict=True)
    }


__all__ = [
    "DETERMINISTIC_FITNESS_WEIGHTS",
    "FITNESS_WEIGHTS",
    "FitnessPolicy",
    "compute_fitness",
]
