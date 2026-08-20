"""Within-generation rank-normalized operator fitness."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


FITNESS_WEIGHTS = {
    "cost": -0.45,
    "feasible": 0.25,
    "delayed": 0.15,
    "worst_context": 0.10,
    "runtime": -0.05,
}


def compute_fitness(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Rank operators without allowing a single metric's scale to dominate."""

    if not rows:
        return {}
    frame = pd.DataFrame(rows).copy()
    required = {"operator_name", *FITNESS_WEIGHTS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"fitness rows missing columns: {sorted(missing)}")
    n = max(len(frame) - 1, 1)
    scores = pd.Series(0.0, index=frame.index)
    for column, weight in FITNESS_WEIGHTS.items():
        numeric = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
        percentile = (numeric.rank(method="average") - 1.0) / n
        scores += weight * percentile
    return {
        str(name): float(score)
        for name, score in zip(frame["operator_name"].tolist(), scores.tolist(), strict=True)
    }

