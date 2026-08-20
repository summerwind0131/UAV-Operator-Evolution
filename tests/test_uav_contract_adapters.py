from __future__ import annotations

import math

import pytest

from operator_evolution_core.contracts import InstanceRef, ObjectiveEvaluation
from uav_operator_evolution.domain import (
    UAV_DOMAIN_ID,
    environment_matches_instance_ref,
    environment_to_instance_ref,
    evaluation_result_to_objective,
    objective_to_evaluation_result,
)
from uav_operator_evolution.environment import (
    CircleObstacle,
    Environment2D,
    RectangleObstacle,
    RiskZone,
)
from uav_operator_evolution.path import PathEvaluator
from uav_operator_evolution.reproducibility import stable_hash


def _environment() -> Environment2D:
    return Environment2D(
        map_id="adapter-map",
        width=50.0,
        height=40.0,
        start=(2.0, 2.0),
        goal=(48.0, 38.0),
        obstacles=[
            CircleObstacle(center=(20.0, 20.0), radius=4.0),
            RectangleObstacle(min_x=30.0, min_y=5.0, max_x=34.0, max_y=20.0),
        ],
        risk_zones=[
            RiskZone(min_x=5.0, min_y=25.0, max_x=18.0, max_y=35.0, weight=2.0)
        ],
        safety_distance=1.5,
        difficulty="mixed",
        seed=20260820,
    )


def test_environment_projects_to_stable_instance_reference_without_full_payload() -> None:
    environment = _environment()

    reference = environment_to_instance_ref(environment, "train")
    restored = InstanceRef.model_validate_json(reference.model_dump_json())

    assert reference.domain_id == UAV_DOMAIN_ID
    assert reference.instance_id == environment.map_id
    assert reference.content_hash == environment.content_hash
    assert reference.difficulty == environment.difficulty
    assert reference.metadata["geometry_hash"] == environment.geometry_hash
    assert "obstacles" not in reference.metadata
    assert restored == reference
    assert environment_matches_instance_ref(environment, restored)
    assert stable_hash(reference.model_dump(mode="json")) == stable_hash(
        restored.model_dump(mode="json")
    )


def test_environment_reference_detects_content_or_domain_mismatch() -> None:
    environment = _environment()
    reference = environment_to_instance_ref(environment, "validation")

    changed = environment.model_copy(update={"safety_distance": 2.0})
    foreign = reference.model_copy(update={"domain_id": "job-shop-scheduling"})

    assert not environment_matches_instance_ref(changed, reference)
    assert not environment_matches_instance_ref(environment, foreign)


@pytest.mark.parametrize(
    "path",
    [
        [(2.0, 2.0), (10.0, 30.0), (25.0, 35.0), (48.0, 38.0)],
        [(2.0, 2.0), (20.0, 20.0), (48.0, 38.0)],
    ],
)
def test_uav_evaluation_round_trips_through_core_objective(path) -> None:
    environment = _environment()
    original = PathEvaluator().evaluate(path, environment)

    core = evaluation_result_to_objective(original)
    restored = objective_to_evaluation_result(core)

    assert restored == original
    assert ObjectiveEvaluation.model_validate_json(core.model_dump_json()) == core
    assert stable_hash(core.model_dump(mode="json")) == stable_hash(
        evaluation_result_to_objective(restored).model_dump(mode="json")
    )
    assert core.scalar_cost == original.total_cost
    assert core.violations["collision_count"] == original.collision_count


def test_uav_objective_reverse_adapter_rejects_foreign_or_incomplete_payloads() -> None:
    original = PathEvaluator().evaluate(
        [(2.0, 2.0), (10.0, 30.0), (48.0, 38.0)], _environment()
    )
    core = evaluation_result_to_objective(original)

    with pytest.raises(ValueError, match="does not belong"):
        objective_to_evaluation_result(
            core.model_copy(
                update={"metadata": {**core.metadata, "domain_id": "job-shop-scheduling"}}
            )
        )
    with pytest.raises(ValueError, match="missing components"):
        objective_to_evaluation_result(
            core.model_copy(
                update={
                    "components": {
                        key: value
                        for key, value in core.components.items()
                        if key != "path_length"
                    }
                }
            )
        )
    with pytest.raises(ValueError, match="finite integer"):
        objective_to_evaluation_result(
            core.model_copy(
                update={"violations": {**core.violations, "collision_count": 1.5}}
            )
        )


def test_uav_adapter_never_emits_nonfinite_values() -> None:
    core = evaluation_result_to_objective(
        PathEvaluator().evaluate(
            [(2.0, 2.0), (10.0, 30.0), (48.0, 38.0)], _environment()
        )
    )
    assert math.isfinite(core.scalar_cost)
    assert all(math.isfinite(value) for value in core.components.values())
    assert all(math.isfinite(value) for value in core.violations.values())
