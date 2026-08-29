from __future__ import annotations

import math

import numpy as np
import pytest

from operator_evolution_core.contracts import (
    DomainAdapter,
    Evaluator,
    FeatureExtractor,
    Initializer,
    SolutionCodec,
    SolutionGuard,
    TraceEncoder,
)
from uav_operator_evolution.domain import (
    UAV_DOMAIN_ID,
    UAVDomainAdapter,
    objective_to_evaluation_result,
)
from uav_operator_evolution.environment import (
    CircleObstacle,
    Environment2D,
    RectangleObstacle,
    RiskZone,
)
from uav_operator_evolution.path import (
    ObjectiveWeights,
    PathEvaluator,
    extract_path_features,
    initialize_path,
)
from uav_operator_evolution.reproducibility import stable_hash
from uav_operator_evolution.search import SearchContext, SearchExecutor


def _environment() -> Environment2D:
    return Environment2D(
        map_id="domain-adapter-map",
        width=48.0,
        height=40.0,
        start=(2.0, 2.0),
        goal=(46.0, 38.0),
        obstacles=[
            CircleObstacle(center=(19.0, 19.0), radius=5.0),
            RectangleObstacle(min_x=29.0, min_y=4.0, max_x=34.0, max_y=25.0),
        ],
        risk_zones=[
            RiskZone(min_x=4.0, min_y=27.0, max_x=17.0, max_y=36.0, weight=1.5)
        ],
        safety_distance=1.25,
        difficulty="mixed",
        seed=20260820,
    )


def _legacy_state_snapshot(path, path_features, search_features, evaluation):
    """Frozen Step-2 payload used to prove the new encoder is identical."""

    return {
        "path": [list(point) for point in path],
        "path_features": path_features,
        "search_features": search_features,
        "objective": float(evaluation.total_cost),
        "objective_components": {
            "path_length": float(evaluation.path_length),
            "collision_penalty": float(evaluation.collision_penalty),
            "smoothness_penalty": float(evaluation.smoothness_penalty),
            "risk_penalty": float(evaluation.risk_penalty),
            "waypoint_penalty": float(evaluation.waypoint_penalty),
        },
        "feasible": bool(evaluation.feasible),
        "collision_count": int(evaluation.collision_count),
        "minimum_clearance": float(evaluation.minimum_clearance),
    }


def _weights() -> ObjectiveWeights:
    return ObjectiveWeights(
        length=1.3,
        collision=850.0,
        smoothness=4.0,
        risk=7.0,
        waypoint=0.75,
    )


def test_uav_domain_adapter_composes_all_runtime_contracts() -> None:
    adapter = UAVDomainAdapter(PathEvaluator(_weights()), initializer_grid_resolution=3.0)

    assert isinstance(adapter, DomainAdapter)
    assert adapter.domain_id == UAV_DOMAIN_ID
    assert isinstance(adapter.initializer, Initializer)
    assert isinstance(adapter.evaluator, Evaluator)
    assert isinstance(adapter.features, FeatureExtractor)
    assert isinstance(adapter.codec, SolutionCodec)
    assert isinstance(adapter.guard, SolutionGuard)
    assert isinstance(adapter.trace_encoder, TraceEncoder)


def test_initializer_is_a_field_exact_wrapper_without_rng_drift() -> None:
    environment = _environment()
    adapter_rng = np.random.default_rng(105)
    legacy_rng = np.random.default_rng(105)
    adapter = UAVDomainAdapter(initializer_grid_resolution=3.0)

    actual = adapter.initializer.initialize(environment, adapter_rng)
    expected = initialize_path(environment, legacy_rng, grid_resolution=3.0)

    assert actual == expected
    assert adapter_rng.integers(0, 2**32) == legacy_rng.integers(0, 2**32)
    with pytest.raises(TypeError, match="numpy.random.Generator"):
        adapter.initializer.initialize(environment, object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "path",
    [
        [(2.0, 2.0), (8.0, 31.0), (25.0, 36.0), (46.0, 38.0)],
        [(2.0, 2.0), (19.0, 19.0), (46.0, 38.0)],
    ],
)
def test_evaluation_and_features_match_existing_uav_functions_field_for_field(path) -> None:
    environment = _environment()
    native_evaluator = PathEvaluator(_weights())
    adapter = UAVDomainAdapter(native_evaluator)

    expected_evaluation = native_evaluator.evaluate(path, environment)
    objective = adapter.evaluator.evaluate(path, environment)
    actual_evaluation = objective_to_evaluation_result(objective)
    expected_features = extract_path_features(
        path,
        environment,
        evaluator=native_evaluator,
    ).model_dump(mode="json")

    assert actual_evaluation.model_dump() == expected_evaluation.model_dump()
    assert adapter.features.extract(path, environment, objective) == expected_features


def test_codec_reproduces_copy_canonical_json_and_stable_hash_semantics() -> None:
    adapter = UAVDomainAdapter()
    path = [(2.0, 2.0), (8.5, 31.25), (46.0, 38.0)]

    cloned = adapter.codec.clone(path)
    canonical = adapter.codec.canonicalize([[2, 2], [8.5, 31.25], [46, 38]])
    encoded = adapter.codec.to_json(path)

    assert cloned == path
    assert cloned is not path
    assert canonical == path
    assert encoded == [[2.0, 2.0], [8.5, 31.25], [46.0, 38.0]]
    assert adapter.codec.stable_hash(path) == stable_hash(encoded)
    with pytest.raises(ValueError, match="at least a start and goal"):
        adapter.codec.canonicalize([[2.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        adapter.codec.canonicalize([[2.0, 2.0], [math.nan, 38.0]])


@pytest.mark.parametrize(
    "invalid_path",
    [
        [(3.0, 2.0), (46.0, 38.0)],
        [(2.0, 2.0), (45.0, 38.0)],
        [(2.0, 2.0), (49.0, 20.0), (46.0, 38.0)],
        [(2.0, 2.0)],
        [(2.0, 2.0), (math.inf, 38.0)],
    ],
)
def test_guard_rejects_every_path_rejected_by_legacy_structure_check(invalid_path) -> None:
    environment = _environment()
    adapter = UAVDomainAdapter()

    with pytest.raises(ValueError):
        SearchExecutor._validate_initial_path(invalid_path, environment)
    assert adapter.guard.validate_structure(invalid_path, environment)


def test_guard_preserves_legacy_separation_of_structure_and_feasibility() -> None:
    environment = _environment()
    colliding_path = [environment.start, (19.0, 19.0), environment.goal]
    adapter = UAVDomainAdapter()

    SearchExecutor._validate_initial_path(colliding_path, environment)
    assert adapter.guard.validate_structure(colliding_path, environment) == []
    assert not adapter.evaluator.evaluate(colliding_path, environment).feasible


def test_trace_snapshot_matches_existing_uav_payload_field_for_field() -> None:
    environment = _environment()
    path = [(2.0, 2.0), (8.0, 31.0), (25.0, 36.0), (46.0, 38.0)]
    native_evaluator = PathEvaluator(_weights())
    native_evaluation = native_evaluator.evaluate(path, environment)
    adapter = UAVDomainAdapter(native_evaluator)
    objective = adapter.evaluator.evaluate(path, environment)
    context = SearchContext(
        iteration=4,
        max_iterations=12,
        current_evaluation=native_evaluation,
        best_evaluation=native_evaluation,
        stagnation_count=2,
        recent_improvements=(1.0, -0.25, 0.0),
        recent_acceptances=(True, False, True),
        last_created_new_best=False,
    )
    legacy_features = extract_path_features(
        path,
        environment,
        evaluator=native_evaluator,
    ).model_dump(mode="json")
    expected = _legacy_state_snapshot(
        path,
        legacy_features,
        context.as_features(),
        native_evaluation,
    )

    actual = adapter.trace_encoder.snapshot(path, environment, objective, context)

    assert actual == expected
    assert stable_hash(actual) == stable_hash(expected)


def test_trace_encoder_rejects_nonfinite_context_payloads() -> None:
    class NonFiniteContext:
        def as_features(self):
            return {"bad": np.float64(math.nan)}

    environment = _environment()
    path = [environment.start, environment.goal]
    adapter = UAVDomainAdapter()
    objective = adapter.evaluator.evaluate(path, environment)

    with pytest.raises(ValueError, match="non-finite"):
        adapter.trace_encoder.snapshot(
            path,
            environment,
            objective,
            NonFiniteContext(),  # type: ignore[arg-type]
        )
