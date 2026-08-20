"""Tests for objective decomposition and path features."""

from __future__ import annotations

import pytest

from uav_operator_evolution.environment import CircleObstacle, Environment2D, RiskZone
from uav_operator_evolution.path import ObjectiveWeights, PathEvaluator, extract_path_features


def test_straight_path_has_only_length_cost_in_empty_map() -> None:
    environment = Environment2D(
        width=10.0,
        height=10.0,
        start=(0.0, 1.0),
        goal=(10.0, 1.0),
        difficulty="sparse",
    )
    result = PathEvaluator().evaluate([environment.start, environment.goal], environment)
    assert result.feasible
    assert result.total_cost == pytest.approx(10.0)
    assert result.path_length == pytest.approx(10.0)
    assert result.collision_penalty == 0.0
    assert result.smoothness_penalty == 0.0
    assert result.waypoint_penalty == 0.0
    assert result.minimum_clearance == pytest.approx(environment.diagonal)


def test_collision_and_safe_detour_are_reported_separately() -> None:
    environment = Environment2D(
        width=12.0,
        height=10.0,
        start=(1.0, 5.0),
        goal=(11.0, 5.0),
        obstacles=[CircleObstacle(center=(6.0, 5.0), radius=1.0)],
        safety_distance=1.0,
    )
    evaluator = PathEvaluator()
    direct = evaluator.evaluate([environment.start, environment.goal], environment)
    detour = evaluator.evaluate([environment.start, (6.0, 8.0), environment.goal], environment)
    assert not direct.feasible
    assert direct.collision_count == 1
    assert direct.collision_penalty > 0
    assert direct.minimum_clearance < 0
    assert detour.feasible
    assert detour.collision_count == 0
    assert detour.waypoint_penalty == 1.0
    assert detour.smoothness_penalty > 0


def test_risk_and_objective_weights_are_decomposed() -> None:
    environment = Environment2D(
        width=10.0,
        height=5.0,
        start=(0.0, 1.0),
        goal=(10.0, 1.0),
        risk_zones=[RiskZone(min_x=4.0, min_y=0.0, max_x=6.0, max_y=2.0, weight=2.0)],
    )
    evaluator = PathEvaluator(
        ObjectiveWeights(length=0.0, collision=0.0, smoothness=0.0, risk=3.0, waypoint=0.0)
    )
    result = evaluator.evaluate([environment.start, environment.goal], environment)
    assert result.risk_penalty == pytest.approx(4.0)
    assert result.total_cost == pytest.approx(12.0)


def test_endpoint_and_boundary_violations_make_path_infeasible() -> None:
    environment = Environment2D(
        width=10.0,
        height=10.0,
        start=(1.0, 1.0),
        goal=(9.0, 9.0),
    )
    result = PathEvaluator().evaluate([(2.0, 1.0), (11.0, 9.0)], environment)
    assert not result.feasible
    assert result.collision_penalty >= 3.0


def test_evaluation_does_not_mutate_input_and_features_match() -> None:
    environment = Environment2D(
        width=10.0,
        height=10.0,
        start=(1.0, 1.0),
        goal=(9.0, 9.0),
    )
    path = [environment.start, (4.0, 6.0), environment.goal]
    snapshot = list(path)
    evaluator = PathEvaluator()
    result = evaluator(path, environment)
    features = extract_path_features(path, environment, evaluator)
    assert path == snapshot
    assert features.path_length == pytest.approx(result.path_length)
    assert features.waypoint_count == 3
    assert features.collision_segment_count == result.collision_count
    assert features.feasible == result.feasible
    assert 0.0 < features.smoothness <= 1.0


@pytest.mark.parametrize("path", [[], [(0.0, 0.0)], [(0.0, 0.0), (float("nan"), 1.0)]])
def test_malformed_paths_are_rejected(path: list[tuple[float, float]]) -> None:
    environment = Environment2D(width=10.0, height=10.0, start=(1.0, 1.0), goal=(9.0, 9.0))
    with pytest.raises(ValueError):
        PathEvaluator().evaluate(path, environment)
