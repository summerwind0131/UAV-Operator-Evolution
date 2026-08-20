"""Numerically robust continuous geometry helpers used throughout the project."""

from __future__ import annotations

import math
from collections.abc import Sequence

from .obstacles import CircleObstacle, Obstacle, Point, RectangleObstacle, RiskZone

EPSILON = 1e-9


def euclidean_distance(first: Point, second: Point) -> float:
    """Return the Euclidean distance between two points."""

    return math.hypot(second[0] - first[0], second[1] - first[1])


def path_length(path: Sequence[Point]) -> float:
    """Return total polyline length."""

    return sum(euclidean_distance(first, second) for first, second in zip(path, path[1:]))


def point_to_segment_distance(point: Point, start: Point, end: Point) -> float:
    """Return the shortest distance between a point and a closed segment."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    squared_length = dx * dx + dy * dy
    if squared_length <= EPSILON * EPSILON:
        return euclidean_distance(point, start)
    fraction = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / squared_length
    fraction = min(1.0, max(0.0, fraction))
    projection = (start[0] + fraction * dx, start[1] + fraction * dy)
    return euclidean_distance(point, projection)


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (
        second[1] - first[1]
    ) * (third[0] - first[0])


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    return (
        abs(_orientation(start, end, point)) <= EPSILON
        and min(start[0], end[0]) - EPSILON <= point[0] <= max(start[0], end[0]) + EPSILON
        and min(start[1], end[1]) - EPSILON <= point[1] <= max(start[1], end[1]) + EPSILON
    )


def segments_intersect(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> bool:
    """Return whether two closed segments intersect, including tangencies."""

    o1 = _orientation(first_start, first_end, second_start)
    o2 = _orientation(first_start, first_end, second_end)
    o3 = _orientation(second_start, second_end, first_start)
    o4 = _orientation(second_start, second_end, first_end)
    if ((o1 > EPSILON and o2 < -EPSILON) or (o1 < -EPSILON and o2 > EPSILON)) and (
        (o3 > EPSILON and o4 < -EPSILON) or (o3 < -EPSILON and o4 > EPSILON)
    ):
        return True
    return (
        (abs(o1) <= EPSILON and _point_on_segment(second_start, first_start, first_end))
        or (abs(o2) <= EPSILON and _point_on_segment(second_end, first_start, first_end))
        or (abs(o3) <= EPSILON and _point_on_segment(first_start, second_start, second_end))
        or (abs(o4) <= EPSILON and _point_on_segment(first_end, second_start, second_end))
    )


def segment_to_segment_distance(
    first_start: Point,
    first_end: Point,
    second_start: Point,
    second_end: Point,
) -> float:
    """Return the shortest distance between two closed segments."""

    if segments_intersect(first_start, first_end, second_start, second_end):
        return 0.0
    return min(
        point_to_segment_distance(first_start, second_start, second_end),
        point_to_segment_distance(first_end, second_start, second_end),
        point_to_segment_distance(second_start, first_start, first_end),
        point_to_segment_distance(second_end, first_start, first_end),
    )


def point_in_rectangle(point: Point, rectangle: RectangleObstacle | RiskZone) -> bool:
    """Return whether a point lies in a closed axis-aligned rectangle."""

    return (
        rectangle.min_x - EPSILON <= point[0] <= rectangle.max_x + EPSILON
        and rectangle.min_y - EPSILON <= point[1] <= rectangle.max_y + EPSILON
    )


def segment_rectangle_interval(
    start: Point,
    end: Point,
    rectangle: RectangleObstacle | RiskZone,
) -> tuple[float, float] | None:
    """Return the segment parameter interval inside a rectangle using Liang-Barsky."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    lower = 0.0
    upper = 1.0
    checks = (
        (-dx, start[0] - rectangle.min_x),
        (dx, rectangle.max_x - start[0]),
        (-dy, start[1] - rectangle.min_y),
        (dy, rectangle.max_y - start[1]),
    )
    for coefficient, offset in checks:
        if abs(coefficient) <= EPSILON:
            if offset < -EPSILON:
                return None
            continue
        ratio = offset / coefficient
        if coefficient < 0:
            lower = max(lower, ratio)
        else:
            upper = min(upper, ratio)
        if lower - upper > EPSILON:
            return None
    return (max(0.0, lower), min(1.0, upper))


def segment_intersects_rectangle(
    start: Point,
    end: Point,
    rectangle: RectangleObstacle | RiskZone,
) -> bool:
    """Return whether a segment intersects a closed rectangle."""

    return segment_rectangle_interval(start, end, rectangle) is not None


def segment_intersects_circle(start: Point, end: Point, circle: CircleObstacle) -> bool:
    """Return whether a segment intersects a closed circle."""

    return point_to_segment_distance(circle.center, start, end) <= circle.radius + EPSILON


def point_obstacle_clearance(point: Point, obstacle: Obstacle) -> float:
    """Return signed clearance to an obstacle (negative for interior points)."""

    if isinstance(obstacle, CircleObstacle):
        return euclidean_distance(point, obstacle.center) - obstacle.radius
    if point_in_rectangle(point, obstacle):
        return -min(
            point[0] - obstacle.min_x,
            obstacle.max_x - point[0],
            point[1] - obstacle.min_y,
            obstacle.max_y - point[1],
        )
    dx = max(obstacle.min_x - point[0], 0.0, point[0] - obstacle.max_x)
    dy = max(obstacle.min_y - point[1], 0.0, point[1] - obstacle.max_y)
    return math.hypot(dx, dy)


def segment_obstacle_clearance(start: Point, end: Point, obstacle: Obstacle) -> float:
    """Return the non-positive/positive clearance of a segment to an obstacle."""

    if isinstance(obstacle, CircleObstacle):
        return point_to_segment_distance(obstacle.center, start, end) - obstacle.radius
    if segment_intersects_rectangle(start, end, obstacle):
        return min(0.0, point_obstacle_clearance(start, obstacle), point_obstacle_clearance(end, obstacle))
    bottom_left = (obstacle.min_x, obstacle.min_y)
    bottom_right = (obstacle.max_x, obstacle.min_y)
    top_right = (obstacle.max_x, obstacle.max_y)
    top_left = (obstacle.min_x, obstacle.max_y)
    edges = (
        (bottom_left, bottom_right),
        (bottom_right, top_right),
        (top_right, top_left),
        (top_left, bottom_left),
    )
    return min(segment_to_segment_distance(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


def obstacle_contains_point(point: Point, obstacle: Obstacle, clearance: float = 0.0) -> bool:
    """Return whether a point violates an obstacle clearance."""

    return point_obstacle_clearance(point, obstacle) <= clearance + EPSILON


def segment_intersects_obstacle(
    start: Point,
    end: Point,
    obstacle: Obstacle,
    clearance: float = 0.0,
) -> bool:
    """Return whether a segment violates an obstacle clearance."""

    return segment_obstacle_clearance(start, end, obstacle) <= clearance + EPSILON


def segment_risk_exposure(start: Point, end: Point, zone: RiskZone) -> float:
    """Return weighted segment length lying inside a rectangular risk zone."""

    interval = segment_rectangle_interval(start, end, zone)
    if interval is None:
        return 0.0
    return euclidean_distance(start, end) * max(0.0, interval[1] - interval[0]) * zone.weight


def turn_angles(path: Sequence[Point]) -> list[float]:
    """Return unsigned turning angles in radians for interior waypoints."""

    angles: list[float] = []
    for previous, current, following in zip(path, path[1:], path[2:]):
        first = (previous[0] - current[0], previous[1] - current[1])
        second = (following[0] - current[0], following[1] - current[1])
        first_norm = math.hypot(*first)
        second_norm = math.hypot(*second)
        if first_norm <= EPSILON or second_norm <= EPSILON:
            angles.append(math.pi)
            continue
        cosine = (first[0] * second[0] + first[1] * second[1]) / (first_norm * second_norm)
        interior = math.acos(min(1.0, max(-1.0, cosine)))
        angles.append(math.pi - interior)
    return angles


__all__ = [
    "EPSILON",
    "euclidean_distance",
    "obstacle_contains_point",
    "path_length",
    "point_in_rectangle",
    "point_obstacle_clearance",
    "point_to_segment_distance",
    "segment_intersects_circle",
    "segment_intersects_obstacle",
    "segment_intersects_rectangle",
    "segment_obstacle_clearance",
    "segment_rectangle_interval",
    "segment_risk_exposure",
    "segment_to_segment_distance",
    "segments_intersect",
    "turn_angles",
]
