from __future__ import annotations

from uav_operator_evolution.agents.heuristic_designer import HeuristicDesigner
from uav_operator_evolution.operators.specs import OperatorSpec


def parent_spec() -> OperatorSpec:
    return OperatorSpec.model_validate(
        {
            "name": "ObstacleDetourOperator",
            "description": "manual parent",
            "selection_strategy": {"kind": "select_collision_segment"},
            "transformations": [{"kind": "generate_obstacle_detour"}],
            "expected_mechanism": "detour",
        }
    )


def test_heuristic_designer_uses_profile_evidence() -> None:
    profile = {
        "operator_name": "ObstacleDetourOperator",
        "total_calls": 12,
        "average_immediate_reward": 2.0,
        "average_delayed_reward": 5.0,
        "feasibility_rate": 0.7,
        "failure_contexts": ["dense collision"],
        "synergy_relations": [
            {"second_operator": "SmoothSegmentOperator", "reward_delta": 3.0, "relation": "positive"}
        ],
    }
    proposal = HeuristicDesigner().propose("dense maps", [parent_spec()], [profile], [], [], [])
    assert proposal.spec.parent_operators == ["ObstacleDetourOperator"]
    assert len(proposal.spec.transformations) >= 2
    assert any("calls=12" in item for item in proposal.evidence_used)
    assert proposal.evidence_level == "computed"

