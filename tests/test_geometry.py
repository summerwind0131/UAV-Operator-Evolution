"""Unit tests for continuous geometry and environment queries."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uav_operator_evolution.environment import (
    CircleObstacle,
    Environment2D,
    RectangleObstacle,
    RiskZone,
    point_obstacle_clearance,
    point_to_segment_distance,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
    segments_intersect,
)
from uav_operator_evolution.environment.geometry import segment_risk_exposure


def test_segment_geometry_handles_intersection_and_tangency() -> None:
    assert point_to_segment_distance((1.0, 1.0), (0.0, 0.0), (2.0, 0.0)) == pytest.approx(1.0)
    assert segments_intersect((0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0))
    assert segments_intersect((0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (2.0, 0.0))


def test_circle_and_rectangle_clearance_are_continuous() -> None:
    circle = CircleObstacle(center=(5.0, 5.0), radius=2.0)
    rectangle = RectangleObstacle(min_x=4.0, min_y=4.0, max_x=6.0, max_y=7.0)
    assert point_obstacle_clearance((8.0, 5.0), circle) == pytest.approx(1.0)
    assert point_obstacle_clearance((5.0, 5.0), circle) == pytest.approx(-2.0)
    assert segment_obstacle_clearance((0.0, 8.0), (10.0, 8.0), circle) == pytest.approx(1.0)
    assert segment_intersects_obstacle((0.0, 7.0), (10.0, 7.0), circle)
    assert point_obstacle_clearance((7.0, 5.0), rectangle) == pytest.approx(1.0)
    assert segment_obstacle_clearance((0.0, 3.0), (10.0, 3.0), rectangle) == pytest.approx(1.0)
    assert segment_intersects_obstacle((0.0, 5.0), (10.0, 5.0), rectangle)


def test_environment_queries_apply_configured_safety_distance() -> None:
    environment = Environment2D(
        width=20.0,
        height=20.0,
        start=(1.0, 1.0),
        goal=(19.0, 19.0),
        obstacles=[CircleObstacle(center=(10.0, 10.0), radius=2.0)],
        safety_distance=1.0,
    )
    assert environment.point_is_collision_free((10.0, 13.5))
    assert not environment.point_is_collision_free((10.0, 13.0))
    assert not environment.segment_is_collision_free((1.0, 10.0), (19.0, 10.0))
    assert environment.segment_is_collision_free((1.0, 14.0), (19.0, 14.0))
    assert environment.colliding_segment_indices([(1.0, 10.0), (19.0, 10.0)]) == [0]


def test_risk_exposure_uses_exact_clipped_length() -> None:
    zone = RiskZone(min_x=4.0, min_y=0.0, max_x=6.0, max_y=3.0, weight=2.0)
    assert segment_risk_exposure((0.0, 1.0), (10.0, 1.0), zone) == pytest.approx(4.0)
    assert segment_risk_exposure((0.0, 4.0), (10.0, 4.0), zone) == 0.0


def test_environment_rejects_out_of_bounds_geometry() -> None:
    with pytest.raises(ValidationError):
        Environment2D(
            width=10.0,
            height=10.0,
            start=(1.0, 1.0),
            goal=(9.0, 9.0),
            obstacles=[CircleObstacle(center=(9.0, 5.0), radius=2.0)],
        )
