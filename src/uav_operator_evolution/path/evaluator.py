"""Fixed decomposed objective function for UAV paths."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ..environment.environment import Environment2D
from ..environment.geometry import (
    EPSILON,
    euclidean_distance,
    path_length,
    segment_obstacle_clearance,
    turn_angles,
)
from .models import EvaluationResult, ObjectiveWeights, Path, copy_and_validate_path


class PathEvaluator:
    """Evaluate paths with fixed length, safety, smoothness, risk, and size terms."""

    def __init__(self, weights: ObjectiveWeights | Mapping[str, float] | Any | None = None) -> None:
        if weights is None:
            self.weights = ObjectiveWeights()
        elif isinstance(weights, ObjectiveWeights):
            self.weights = weights
        elif hasattr(weights, "model_dump"):
            self.weights = ObjectiveWeights.model_validate(weights.model_dump())
        else:
            self.weights = ObjectiveWeights.model_validate(weights)

    def evaluate(self, path: Sequence[Sequence[float]], environment: Environment2D) -> EvaluationResult:
        """Return an objective decomposition without mutating the input path."""

        canonical = copy_and_validate_path(path)
        length_term = path_length(canonical)
        collision_term, collision_count, minimum_clearance, feasible = self._collision_terms(
            canonical, environment
        )
        angles = turn_angles(canonical)
        smoothness_term = sum((angle / math.pi) ** 2 for angle in angles)
        risk_term = environment.risk_exposure(canonical)
        waypoint_term = float(max(0, len(canonical) - 2))
        total = (
            self.weights.length * length_term
            + self.weights.collision * collision_term
            + self.weights.smoothness * smoothness_term
            + self.weights.risk * risk_term
            + self.weights.waypoint * waypoint_term
        )
        return EvaluationResult(
            total_cost=total,
            path_length=length_term,
            collision_penalty=collision_term,
            smoothness_penalty=smoothness_term,
            risk_penalty=risk_term,
            waypoint_penalty=waypoint_term,
            feasible=feasible,
            collision_count=collision_count,
            minimum_clearance=minimum_clearance,
        )

    def __call__(self, path: Sequence[Sequence[float]], environment: Environment2D) -> EvaluationResult:
        return self.evaluate(path, environment)

    @staticmethod
    def _collision_terms(path: Path, environment: Environment2D) -> tuple[float, int, float, bool]:
        penalty = 0.0
        colliding_segments = 0
        minimum_clearance = environment.diagonal
        endpoints_valid = (
            euclidean_distance(path[0], environment.start) <= 1e-7
            and euclidean_distance(path[-1], environment.goal) <= 1e-7
        )
        if euclidean_distance(path[0], environment.start) > 1e-7:
            penalty += 1.0
        if euclidean_distance(path[-1], environment.goal) > 1e-7:
            penalty += 1.0

        points_in_bounds = True
        for point in path:
            if environment.in_bounds(point):
                continue
            points_in_bounds = False
            dx = max(-point[0], 0.0, point[0] - environment.width)
            dy = max(-point[1], 0.0, point[1] - environment.height)
            penalty += 1.0 + math.hypot(dx, dy) / max(environment.diagonal, EPSILON)

        all_segments_clear = True
        for start, end in zip(path, path[1:]):
            segment_violates = not environment.in_bounds(start) or not environment.in_bounds(end)
            for obstacle in environment.obstacles:
                clearance = segment_obstacle_clearance(start, end, obstacle)
                minimum_clearance = min(minimum_clearance, clearance)
                if clearance <= environment.safety_distance + EPSILON:
                    segment_violates = True
                    scale = max(environment.safety_distance, 1.0)
                    penalty += 1.0 + max(0.0, environment.safety_distance - clearance) / scale
            if segment_violates:
                colliding_segments += 1
                all_segments_clear = False
        feasible = endpoints_valid and points_in_bounds and all_segments_clear
        return penalty, colliding_segments, minimum_clearance, feasible


__all__ = ["PathEvaluator"]
