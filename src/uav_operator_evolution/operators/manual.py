"""Eight deterministic-by-RNG baseline operators for the research MVP."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import ClassVar

import numpy as np

from ..environment.environment import Environment2D
from ..path.models import Path
from ..search.context import SearchContext
from .base import OperatorResult, copied_path, unchanged_result
from .primitives import (
    clamp_point,
    delete_waypoint,
    generate_obstacle_detour,
    insert_waypoint,
    perturb_waypoint,
    point_segment_distance,
    reconstruct_segment,
    select_collision_segment,
    select_high_curvature_waypoint,
    select_long_segment,
    select_random_waypoint,
    shift_segment,
    shortcut_segment,
    smooth_segment,
)


def _random_displacement(rng: np.random.Generator, scale: float) -> tuple[float, float]:
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    distance = float(rng.uniform(0.25, 1.0)) * max(float(scale), 0.0)
    return (distance * float(np.cos(angle)), distance * float(np.sin(angle)))


def _changed(before: Path, after: Path) -> bool:
    return len(before) != len(after) or any(a != b for a, b in zip(before, after))


@dataclass(slots=True)
class WaypointPerturbOperator:
    """Randomly displace one interior waypoint."""

    name: ClassVar[str] = "waypoint_perturb"
    max_displacement: float = 8.0

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        index = select_random_waypoint(path, rng)
        if index is None:
            return unchanged_result(path, "no interior waypoint")
        displacement = _random_displacement(rng, self.max_displacement)
        candidate = perturb_waypoint(path, index, displacement, environment)
        if not _changed(path, candidate):
            return unchanged_result(path, "perturbation was clipped to the original point", index=index)
        return OperatorResult(
            candidate,
            (index,),
            info={"index": index, "displacement": displacement},
        )


@dataclass(slots=True)
class SegmentShiftOperator:
    """Translate a contiguous interior waypoint block by one shared vector."""

    name: ClassVar[str] = "segment_shift"
    max_displacement: float = 6.0
    max_segment_points: int = 4

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        interior_count = len(path) - 2
        if interior_count <= 0:
            return unchanged_result(path, "no interior waypoint")
        maximum = min(interior_count, max(1, int(self.max_segment_points)))
        minimum = 2 if maximum >= 2 else 1
        count = int(rng.integers(minimum, maximum + 1))
        start = int(rng.integers(1, len(path) - count))
        end = start + count - 1
        displacement = _random_displacement(rng, self.max_displacement)
        candidate = shift_segment(path, start, end, displacement, environment)
        indices = tuple(range(start, end + 1))
        if not _changed(path, candidate):
            return unchanged_result(path, "segment shift was clipped to the original points", indices=indices)
        return OperatorResult(
            candidate,
            indices,
            info={"start_index": start, "end_index": end, "displacement": displacement},
        )


@dataclass(slots=True)
class InsertWaypointOperator:
    """Insert a point near the midpoint of a longest path segment."""

    name: ClassVar[str] = "insert_waypoint"
    perpendicular_jitter_ratio: float = 0.15

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        segment = select_long_segment(path, rng)
        if segment is None:
            return unchanged_result(path, "path has no segment")
        start = np.asarray(path[segment], dtype=float)
        end = np.asarray(path[segment + 1], dtype=float)
        direction = end - start
        length = float(np.linalg.norm(direction))
        midpoint = 0.5 * (start + end)
        if length > 1e-12 and self.perpendicular_jitter_ratio > 0:
            normal = np.asarray((-direction[1], direction[0]), dtype=float) / length
            jitter = float(rng.uniform(-1.0, 1.0)) * length * self.perpendicular_jitter_ratio
            midpoint += jitter * normal
        point = clamp_point((float(midpoint[0]), float(midpoint[1])), environment)
        candidate = insert_waypoint(path, segment, point)
        inserted_index = segment + 1
        return OperatorResult(
            candidate,
            (inserted_index,),
            info={
                "segment_index": segment,
                "inserted_index": inserted_index,
                "inserted_point": point,
                "segment_length": length,
            },
        )


@dataclass(slots=True)
class DeleteWaypointOperator:
    """Delete the geometrically most redundant interior waypoint."""

    name: ClassVar[str] = "delete_waypoint"

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del environment, context
        if len(path) <= 2:
            return unchanged_result(path, "no interior waypoint")
        redundancy = [
            point_segment_distance(path[index], path[index - 1], path[index + 1])
            for index in range(1, len(path) - 1)
        ]
        minimum = min(redundancy)
        ties = [i + 1 for i, value in enumerate(redundancy) if np.isclose(value, minimum)]
        index = ties[int(rng.integers(0, len(ties)))]
        candidate = delete_waypoint(path, index)
        return OperatorResult(
            candidate,
            (index,),
            info={"deleted_index": index, "redundancy_distance": minimum},
        )


@dataclass(slots=True)
class ShortcutOperator:
    """Replace a multi-segment subpath with one collision-free segment."""

    name: ClassVar[str] = "shortcut"
    max_span: int = 12

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        if len(path) <= 2:
            return unchanged_result(path, "no shortcuttable waypoint")
        pairs = [
            (start, end)
            for start in range(len(path) - 2)
            for end in range(start + 2, min(len(path), start + max(2, self.max_span) + 1))
            if environment.segment_is_collision_free(path[start], path[end])
        ]
        if not pairs:
            return unchanged_result(path, "no collision-free shortcut")
        start, end = pairs[int(rng.integers(0, len(pairs)))]
        candidate = shortcut_segment(path, start, end)
        removed = tuple(range(start + 1, end))
        return OperatorResult(
            candidate,
            removed,
            info={"start_index": start, "end_index": end, "removed_count": len(removed)},
        )


@dataclass(slots=True)
class SmoothSegmentOperator:
    """Smooth a local region centred on a high-curvature waypoint."""

    name: ClassVar[str] = "smooth_segment"
    radius: int = 2
    weight: float = 0.6

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        centre = select_high_curvature_waypoint(path, rng)
        if centre is None:
            return unchanged_result(path, "no interior waypoint")
        start = max(1, centre - max(0, int(self.radius)))
        end = min(len(path) - 2, centre + max(0, int(self.radius)))
        candidate = smooth_segment(path, start, end, self.weight, environment)
        indices = tuple(index for index in range(start, end + 1) if candidate[index] != path[index])
        if not indices:
            return unchanged_result(path, "selected segment is already locally straight", centre=centre)
        return OperatorResult(
            candidate,
            indices,
            info={"start_index": start, "end_index": end, "weight": self.weight},
        )


@dataclass(slots=True)
class ObstacleDetourOperator:
    """Insert one or two waypoints around a colliding path segment."""

    name: ClassVar[str] = "obstacle_detour"
    clearance_scale: float = 1.0
    max_attempts: int = 8

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        segment = select_collision_segment(path, environment, rng)
        if segment is None:
            return unchanged_result(path, "path has no colliding segment")
        candidate = generate_obstacle_detour(
            path,
            segment,
            environment,
            rng,
            clearance_scale=self.clearance_scale,
            max_attempts=self.max_attempts,
        )
        if not _changed(path, candidate):
            return unchanged_result(path, "no bounded collision-free detour found", segment_index=segment)
        added_count = len(candidate) - len(path)
        indices = tuple(range(segment + 1, segment + 1 + added_count))
        return OperatorResult(
            candidate,
            indices,
            info={
                "segment_index": segment,
                "inserted_indices": indices,
                "inserted_count": added_count,
            },
        )


@dataclass(slots=True)
class PartialReconstructionOperator:
    """Destroy and rebuild a randomly chosen local subpath."""

    name: ClassVar[str] = "partial_reconstruction"
    max_span: int = 8
    attempts: int = 6

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        del context
        if len(path) <= 2:
            return unchanged_result(path, "no reconstructable subpath")
        pairs = [
            (start, end)
            for start in range(len(path) - 2)
            for end in range(start + 2, min(len(path), start + max(2, self.max_span) + 1))
        ]
        if not pairs:
            return unchanged_result(path, "no reconstructable subpath")
        order = rng.permutation(len(pairs))
        for pair_index in order[: max(1, int(self.attempts))]:
            start, end = pairs[int(pair_index)]
            candidate = reconstruct_segment(path, start, end, environment, rng)
            if _changed(path, candidate):
                old_length = sum(
                    hypot(b[0] - a[0], b[1] - a[1])
                    for a, b in zip(path[start : end + 1], path[start + 1 : end + 1])
                )
                new_local = candidate[start : start + max(2, len(candidate) - len(path) + end - start + 1)]
                new_length = sum(
                    hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(new_local, new_local[1:])
                )
                return OperatorResult(
                    candidate,
                    tuple(range(start + 1, end)),
                    info={
                        "start_index": start,
                        "end_index": end,
                        "removed_count": end - start - 1,
                        "inserted_count": len(candidate) - (len(path) - (end - start - 1)),
                        "old_local_length": old_length,
                        "new_local_length": new_length,
                    },
                )
        return unchanged_result(path, "local reconstruction attempts found no valid replacement")


ManualOperator = (
    WaypointPerturbOperator
    | SegmentShiftOperator
    | InsertWaypointOperator
    | DeleteWaypointOperator
    | ShortcutOperator
    | SmoothSegmentOperator
    | ObstacleDetourOperator
    | PartialReconstructionOperator
)
