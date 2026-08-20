from __future__ import annotations

import pytest
from pydantic import ValidationError

from uav_operator_evolution.operators.specs import OperatorSpec


def valid_payload() -> dict:
    return {
        "name": "DetourThenSmooth",
        "description": "bounded composite",
        "parent_operators": ["ObstacleDetourOperator"],
        "applicability_conditions": [],
        "selection_strategy": {"kind": "select_collision_segment"},
        "transformations": [
            {"kind": "generate_obstacle_detour", "clearance_factor": 1.5},
            {"kind": "smooth_segment", "strength": 0.5},
        ],
        "repair_strategy": None,
        "fallback_strategy": {"kind": "rollback_on_failure"},
        "parameters": {},
        "expected_mechanism": "repair then smooth",
        "target_failure_modes": ["jagged_detour"],
    }


def test_operator_spec_accepts_whitelisted_composite() -> None:
    spec = OperatorSpec.model_validate(valid_payload())
    assert [step.kind for step in spec.transformations] == [
        "generate_obstacle_detour",
        "smooth_segment",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload.update(extra_code="exec('bad')"),
        lambda payload: payload["transformations"].append({"kind": "shell"}),
        lambda payload: payload["transformations"].__setitem__(0, {"kind": "smooth_segment", "strength": float("nan")}),
        lambda payload: payload["transformations"].__setitem__(0, {"kind": "smooth_segment", "repeat": 4}),
    ],
)
def test_operator_spec_rejects_unsafe_or_unbounded_values(mutation) -> None:
    payload = valid_payload()
    mutation(payload)
    with pytest.raises(ValidationError):
        OperatorSpec.model_validate(payload)

