from __future__ import annotations

import numpy as np
import pytest

from uav_operator_evolution.environment import CircleObstacle, Environment2D
from uav_operator_evolution.operators import (
    DeleteWaypointOperator,
    InsertWaypointOperator,
    ObstacleDetourOperator,
    PartialReconstructionOperator,
    SegmentShiftOperator,
    ShortcutOperator,
    SmoothSegmentOperator,
    WaypointPerturbOperator,
    default_manual_operators,
)
from uav_operator_evolution.search import SearchContext


def open_environment() -> Environment2D:
    return Environment2D(
        map_id="open",
        width=100.0,
        height=100.0,
        start=(10.0, 10.0),
        goal=(90.0, 90.0),
        obstacles=[],
        safety_distance=1.0,
        difficulty="sparse",
    )


def bent_path() -> list[tuple[float, float]]:
    return [
        (10.0, 10.0),
        (25.0, 42.0),
        (43.0, 24.0),
        (64.0, 68.0),
        (90.0, 90.0),
    ]


@pytest.mark.parametrize(
    "operator",
    [
        WaypointPerturbOperator(),
        SegmentShiftOperator(),
        InsertWaypointOperator(),
        DeleteWaypointOperator(),
        ShortcutOperator(),
        SmoothSegmentOperator(),
        PartialReconstructionOperator(),
    ],
)
def test_non_obstacle_manual_operators_are_pure_and_keep_endpoints(operator: object) -> None:
    environment = open_environment()
    path = bent_path()
    snapshot = list(path)

    result = operator.apply(path, environment, np.random.default_rng(123), SearchContext())

    assert path == snapshot
    assert result.path is not path
    assert result.success
    assert result.path[0] == path[0]
    assert result.path[-1] == path[-1]
    assert result.modified_indices


def test_operator_specific_waypoint_count_changes() -> None:
    environment = open_environment()
    path = bent_path()
    context = SearchContext()

    inserted = InsertWaypointOperator().apply(path, environment, np.random.default_rng(1), context)
    deleted = DeleteWaypointOperator().apply(path, environment, np.random.default_rng(1), context)
    shortcut = ShortcutOperator().apply(path, environment, np.random.default_rng(1), context)

    assert len(inserted.path) == len(path) + 1
    assert len(deleted.path) == len(path) - 1
    assert len(shortcut.path) < len(path)


def test_obstacle_detour_repairs_selected_collision_without_mutating_input() -> None:
    environment = Environment2D(
        map_id="circle",
        width=100.0,
        height=100.0,
        start=(10.0, 50.0),
        goal=(90.0, 50.0),
        obstacles=[CircleObstacle(center=(50.0, 50.0), radius=8.0)],
        safety_distance=2.0,
        difficulty="medium",
    )
    path = [environment.start, environment.goal]
    snapshot = list(path)

    result = ObstacleDetourOperator(max_attempts=10).apply(
        path, environment, np.random.default_rng(8), SearchContext()
    )

    assert path == snapshot
    assert result.success
    assert len(result.path) > len(path)
    assert environment.path_is_collision_free(result.path)
    assert result.path[0] == environment.start
    assert result.path[-1] == environment.goal


def test_operators_return_safe_copy_when_operation_is_not_applicable() -> None:
    environment = open_environment()
    path = [environment.start, environment.goal]

    for operator in (WaypointPerturbOperator(), DeleteWaypointOperator(), SmoothSegmentOperator()):
        result = operator.apply(path, environment, np.random.default_rng(5), SearchContext())
        assert not result.success
        assert result.path == path
        assert result.path is not path
        assert result.failure_reason


def test_registry_exposes_exactly_eight_unique_manual_operators() -> None:
    operators = default_manual_operators()
    assert len(operators) == 8
    assert len({operator.name for operator in operators}) == 8
