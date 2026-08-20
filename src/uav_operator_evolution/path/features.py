"""Deterministic path-state feature extraction."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from ..environment.environment import Environment2D
from ..environment.geometry import turn_angles
from .evaluator import PathEvaluator


class PathFeatures(BaseModel):
    """Compact features used by trajectory diagnosis and operator conditions."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path_length: float = Field(ge=0)
    waypoint_count: int = Field(ge=2)
    collision_segment_count: int = Field(ge=0)
    minimum_clearance: float
    max_turn_angle: float = Field(ge=0)
    mean_turn_angle: float = Field(ge=0)
    smoothness: float = Field(ge=0, le=1)
    smoothness_penalty: float = Field(ge=0)
    feasible: bool


def extract_path_features(
    path: Sequence[Sequence[float]],
    environment: Environment2D,
    evaluator: PathEvaluator | None = None,
) -> PathFeatures:
    """Extract path features using the same geometry as the objective evaluator."""

    active_evaluator = evaluator or PathEvaluator()
    result = active_evaluator.evaluate(path, environment)
    canonical_angles = turn_angles([(float(point[0]), float(point[1])) for point in path])
    maximum = max(canonical_angles, default=0.0)
    mean = sum(canonical_angles) / len(canonical_angles) if canonical_angles else 0.0
    return PathFeatures(
        path_length=result.path_length,
        waypoint_count=len(path),
        collision_segment_count=result.collision_count,
        minimum_clearance=result.minimum_clearance,
        max_turn_angle=maximum,
        mean_turn_angle=mean,
        smoothness=1.0 / (1.0 + result.smoothness_penalty),
        smoothness_penalty=result.smoothness_penalty,
        feasible=result.feasible,
    )


__all__ = ["PathFeatures", "extract_path_features"]
