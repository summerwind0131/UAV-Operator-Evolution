"""Small, reusable and side-effect-free path transformation primitives."""

from __future__ import annotations

from collections.abc import Callable
from math import atan2, hypot, pi
from typing import TypeAlias

import numpy as np

from ..environment.environment import Environment2D
from ..path.models import Path, Waypoint
from .base import copied_path

IndexRange: TypeAlias = tuple[int, int]


def clamp_point(point: Waypoint, environment: Environment2D) -> Waypoint:
    """Clamp a point to the closed map boundary."""

    return (
        float(np.clip(point[0], 0.0, float(environment.width))),
        float(np.clip(point[1], 0.0, float(environment.height))),
    )


def point_segment_distance(point: Waypoint, start: Waypoint, end: Waypoint) -> float:
    """Euclidean distance from ``point`` to a finite line segment."""

    p = np.asarray(point, dtype=float)
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    direction = b - a
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-15:
        return float(np.linalg.norm(p - a))
    fraction = float(np.clip(np.dot(p - a, direction) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + fraction * direction)))


def turning_angle(previous: Waypoint, current: Waypoint, following: Waypoint) -> float:
    """Return the unsigned heading change in radians."""

    incoming = atan2(current[1] - previous[1], current[0] - previous[0])
    outgoing = atan2(following[1] - current[1], following[0] - current[0])
    return abs((outgoing - incoming + pi) % (2.0 * pi) - pi)


def select_random_waypoint(path: Path, rng: np.random.Generator) -> int | None:
    """Select an interior waypoint uniformly."""

    if len(path) <= 2:
        return None
    return int(rng.integers(1, len(path) - 1))


def _choose_tie(indices: list[int], rng: np.random.Generator | None) -> int | None:
    if not indices:
        return None
    if rng is None or len(indices) == 1:
        return indices[0]
    return indices[int(rng.integers(0, len(indices)))]


def select_high_curvature_waypoint(
    path: Path, rng: np.random.Generator | None = None
) -> int | None:
    """Select an interior waypoint with maximum turning angle."""

    if len(path) <= 2:
        return None
    angles = [turning_angle(path[i - 1], path[i], path[i + 1]) for i in range(1, len(path) - 1)]
    maximum = max(angles)
    ties = [i + 1 for i, angle in enumerate(angles) if np.isclose(angle, maximum)]
    return _choose_tie(ties, rng)


def select_long_segment(path: Path, rng: np.random.Generator | None = None) -> int | None:
    """Select the start index of a longest segment."""

    if len(path) < 2:
        return None
    lengths = [hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    maximum = max(lengths)
    ties = [i for i, length in enumerate(lengths) if np.isclose(length, maximum)]
    return _choose_tie(ties, rng)


def _segment_is_free(environment: Environment2D, start: Waypoint, end: Waypoint) -> bool:
    return bool(environment.segment_is_collision_free(start, end))


def select_collision_segment(
    path: Path,
    environment: Environment2D,
    rng: np.random.Generator | None = None,
) -> int | None:
    """Select the start index of a colliding segment, if one exists."""

    collisions = [
        i for i, (start, end) in enumerate(zip(path, path[1:])) if not _segment_is_free(environment, start, end)
    ]
    return _choose_tie(collisions, rng)


def select_low_clearance_segment(
    path: Path,
    environment: Environment2D,
    rng: np.random.Generator | None = None,
) -> int | None:
    """Select the segment with the lowest reported obstacle clearance."""

    if len(path) < 2:
        return None
    clearances = [float(environment.segment_clearance(a, b)) for a, b in zip(path, path[1:])]
    minimum = min(clearances)
    ties = [i for i, clearance in enumerate(clearances) if np.isclose(clearance, minimum)]
    return _choose_tie(ties, rng)


def select_continuous_collision_region(
    path: Path,
    environment: Environment2D,
    rng: np.random.Generator | None = None,
) -> IndexRange | None:
    """Return waypoint endpoints spanning one maximal run of collisions."""

    segments = [
        i for i, (start, end) in enumerate(zip(path, path[1:])) if not _segment_is_free(environment, start, end)
    ]
    if not segments:
        return None
    runs: list[IndexRange] = []
    run_start = run_end = segments[0]
    for index in segments[1:]:
        if index == run_end + 1:
            run_end = index
        else:
            runs.append((run_start, run_end + 1))
            run_start = run_end = index
    runs.append((run_start, run_end + 1))
    maximum_span = max(end - start for start, end in runs)
    ties = [run for run in runs if run[1] - run[0] == maximum_span]
    if rng is None or len(ties) == 1:
        return ties[0]
    return ties[int(rng.integers(0, len(ties)))]


def perturb_waypoint(
    path: Path,
    index: int,
    displacement: Waypoint,
    environment: Environment2D | None = None,
) -> Path:
    """Return a copy with one interior waypoint displaced."""

    candidate = copied_path(path)
    if index <= 0 or index >= len(candidate) - 1:
        return candidate
    point = (candidate[index][0] + displacement[0], candidate[index][1] + displacement[1])
    candidate[index] = clamp_point(point, environment) if environment is not None else point
    return candidate


def shift_segment(
    path: Path,
    start_index: int,
    end_index: int,
    displacement: Waypoint,
    environment: Environment2D | None = None,
) -> Path:
    """Translate a contiguous range of interior waypoints together."""

    candidate = copied_path(path)
    start = max(1, int(start_index))
    end = min(len(candidate) - 2, int(end_index))
    for index in range(start, end + 1):
        point = (candidate[index][0] + displacement[0], candidate[index][1] + displacement[1])
        candidate[index] = clamp_point(point, environment) if environment is not None else point
    return candidate


def insert_waypoint(path: Path, segment_index: int, point: Waypoint) -> Path:
    """Insert ``point`` after a valid segment start."""

    candidate = copied_path(path)
    if 0 <= segment_index < len(candidate) - 1:
        candidate.insert(segment_index + 1, (float(point[0]), float(point[1])))
    return candidate


def delete_waypoint(path: Path, index: int) -> Path:
    """Delete one interior waypoint."""

    candidate = copied_path(path)
    if 0 < index < len(candidate) - 1:
        del candidate[index]
    return candidate


def shortcut_segment(path: Path, start_index: int, end_index: int) -> Path:
    """Remove all waypoints strictly between two retained endpoints."""

    candidate = copied_path(path)
    start = max(0, int(start_index))
    end = min(len(candidate) - 1, int(end_index))
    if end - start >= 2:
        candidate = candidate[: start + 1] + candidate[end:]
    return candidate


def smooth_segment(
    path: Path,
    start_index: int,
    end_index: int,
    weight: float = 0.5,
    environment: Environment2D | None = None,
) -> Path:
    """Apply one Jacobi smoothing pass to an interior waypoint range."""

    source = copied_path(path)
    candidate = copied_path(path)
    blend = float(np.clip(weight, 0.0, 1.0))
    start = max(1, int(start_index))
    end = min(len(source) - 2, int(end_index))
    for index in range(start, end + 1):
        midpoint = (
            0.5 * (source[index - 1][0] + source[index + 1][0]),
            0.5 * (source[index - 1][1] + source[index + 1][1]),
        )
        point = (
            (1.0 - blend) * source[index][0] + blend * midpoint[0],
            (1.0 - blend) * source[index][1] + blend * midpoint[1],
        )
        candidate[index] = clamp_point(point, environment) if environment is not None else point
    return candidate


def _detour_candidates(
    start: Waypoint,
    end: Waypoint,
    environment: Environment2D,
    rng: np.random.Generator,
    clearance_scale: float,
    max_attempts: int,
) -> list[list[Waypoint]]:
    direction = np.asarray(end, dtype=float) - np.asarray(start, dtype=float)
    length = float(np.linalg.norm(direction))
    if length <= 1e-12:
        return []
    normal = np.asarray((-direction[1], direction[0]), dtype=float) / length
    midpoint = 0.5 * (np.asarray(start, dtype=float) + np.asarray(end, dtype=float))
    base = max(
        1.0,
        float(environment.safety_distance) * 1.5,
        0.08 * length,
        0.015 * min(float(environment.width), float(environment.height)),
    ) * max(float(clearance_scale), 0.1)
    side_order = [1.0, -1.0]
    if bool(rng.integers(0, 2)):
        side_order.reverse()
    candidates: list[list[Waypoint]] = []
    for attempt in range(1, max(1, int(max_attempts)) + 1):
        offset = base * attempt
        for side in side_order:
            one = clamp_point(tuple(midpoint + side * offset * normal), environment)
            if _segment_is_free(environment, start, one) and _segment_is_free(environment, one, end):
                candidates.append([one])
            first = clamp_point(
                tuple(np.asarray(start) + direction / 3.0 + side * offset * normal), environment
            )
            second = clamp_point(
                tuple(np.asarray(start) + 2.0 * direction / 3.0 + side * offset * normal), environment
            )
            if (
                _segment_is_free(environment, start, first)
                and _segment_is_free(environment, first, second)
                and _segment_is_free(environment, second, end)
            ):
                candidates.append([first, second])
        if candidates:
            break
    return candidates


def detour_points_for_segment(
    start: Waypoint,
    end: Waypoint,
    environment: Environment2D,
    rng: np.random.Generator,
    clearance_scale: float = 1.0,
    max_attempts: int = 8,
) -> list[Waypoint] | None:
    """Find a short one- or two-point collision-free detour."""

    candidates = _detour_candidates(
        start, end, environment, rng, clearance_scale=clearance_scale, max_attempts=max_attempts
    )
    if not candidates:
        return None

    def added_length(points: list[Waypoint]) -> float:
        chain = [start, *points, end]
        return sum(hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(chain, chain[1:]))

    return min(candidates, key=added_length)


def generate_obstacle_detour(
    path: Path,
    segment_index: int,
    environment: Environment2D,
    rng: np.random.Generator,
    clearance_scale: float = 1.0,
    max_attempts: int = 8,
) -> Path:
    """Insert collision-free detour points around one segment."""

    candidate = copied_path(path)
    if not (0 <= segment_index < len(candidate) - 1):
        return candidate
    points = detour_points_for_segment(
        candidate[segment_index],
        candidate[segment_index + 1],
        environment,
        rng,
        clearance_scale=clearance_scale,
        max_attempts=max_attempts,
    )
    if points is None:
        return candidate
    return candidate[: segment_index + 1] + points + candidate[segment_index + 1 :]


def try_alternative_side(
    path: Path,
    segment_index: int,
    environment: Environment2D,
    rng: np.random.Generator,
    clearance_scale: float = 1.5,
) -> Path:
    """Retry the detour primitive with a larger clearance envelope."""

    return generate_obstacle_detour(
        path,
        segment_index,
        environment,
        rng,
        clearance_scale=clearance_scale,
        max_attempts=12,
    )


def reconstruct_segment(
    path: Path,
    start_index: int,
    end_index: int,
    environment: Environment2D,
    rng: np.random.Generator,
) -> Path:
    """Replace a local interior region with a direct link or safe detour."""

    source = copied_path(path)
    start = max(0, int(start_index))
    end = min(len(source) - 1, int(end_index))
    if end - start < 2:
        return source
    left, right = source[start], source[end]
    if _segment_is_free(environment, left, right):
        replacement: list[Waypoint] = []
    else:
        replacement = detour_points_for_segment(left, right, environment, rng) or []
        if not replacement:
            return source
    return source[: start + 1] + replacement + source[end:]


def rollback_on_failure(
    original: Path, candidate: Path, environment: Environment2D
) -> Path:
    """Return ``original`` when a candidate is malformed or colliding."""

    valid = len(candidate) >= 2 and candidate[0] == original[0] and candidate[-1] == original[-1]
    if valid:
        valid = bool(environment.path_is_collision_free(candidate))
    return copied_path(candidate if valid else original)


def repeat_until_feasible(
    path: Path,
    transform: Callable[[Path], Path],
    environment: Environment2D,
    max_repeats: int = 3,
) -> Path:
    """Repeatedly apply a pure transform, retaining the first feasible result."""

    current = copied_path(path)
    for _ in range(max(1, int(max_repeats))):
        candidate = copied_path(transform(current))
        if bool(environment.path_is_collision_free(candidate)):
            return candidate
        current = candidate
    return copied_path(path)
