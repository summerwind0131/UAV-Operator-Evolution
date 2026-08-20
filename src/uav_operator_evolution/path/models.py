"""Core path aliases and evaluation result models."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict, Field

Waypoint: TypeAlias = tuple[float, float]
Path: TypeAlias = list[Waypoint]


class ObjectiveWeights(BaseModel):
    """Non-negative coefficients for the fixed objective function."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    length: float = Field(default=1.0, ge=0)
    collision: float = Field(default=1000.0, ge=0)
    smoothness: float = Field(default=5.0, ge=0)
    risk: float = Field(default=10.0, ge=0)
    waypoint: float = Field(default=0.5, ge=0)


class EvaluationResult(BaseModel):
    """A decomposed, serializable path objective evaluation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    total_cost: float
    path_length: float
    collision_penalty: float
    smoothness_penalty: float
    risk_penalty: float
    waypoint_penalty: float
    feasible: bool
    collision_count: int = Field(ge=0)
    minimum_clearance: float


def copy_and_validate_path(path: Sequence[Sequence[float]]) -> Path:
    """Return a detached canonical path or raise for malformed coordinates."""

    if len(path) < 2:
        raise ValueError("a path must contain at least a start and goal waypoint")
    result: Path = []
    for waypoint in path:
        if len(waypoint) != 2:
            raise ValueError("each waypoint must contain exactly two coordinates")
        point = (float(waypoint[0]), float(waypoint[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("waypoint coordinates must be finite")
        result.append(point)
    return result


__all__ = [
    "EvaluationResult",
    "ObjectiveWeights",
    "Path",
    "Waypoint",
    "copy_and_validate_path",
]
