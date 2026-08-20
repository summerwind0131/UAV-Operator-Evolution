"""Data-only specifications corresponding to the eight manual baselines."""

from __future__ import annotations

from .specs import OperatorSpec


def manual_operator_specs() -> dict[str, OperatorSpec]:
    """Return reproducible parent specifications for lineage and design input."""

    payloads = [
        ("waypoint_perturb", "select_random_waypoint", [{"kind": "perturb_waypoint", "scale": 8.0}], "local stochastic exploration"),
        ("segment_shift", "select_random_waypoint", [{"kind": "shift_segment", "scale": 6.0}], "move a local block coherently"),
        ("insert_waypoint", "select_long_segment", [{"kind": "insert_waypoint", "offset_scale": 2.0}], "add local path flexibility"),
        ("delete_waypoint", "select_high_curvature_waypoint", [{"kind": "delete_waypoint"}], "remove a potentially redundant point"),
        ("shortcut", "select_long_segment", [{"kind": "shortcut_segment"}], "replace a multi-edge subpath by line of sight"),
        ("smooth_segment", "select_high_curvature_waypoint", [{"kind": "smooth_segment", "strength": 0.6}], "reduce local curvature"),
        ("obstacle_detour", "select_collision_segment", [{"kind": "generate_obstacle_detour"}], "repair a collision with bounded detour points"),
        ("partial_reconstruction", "select_continuous_collision_region", [{"kind": "reconstruct_segment"}], "destroy and rebuild a local region"),
    ]
    return {
        name: OperatorSpec.model_validate(
            {
                "name": name,
                "description": f"Manual baseline: {mechanism}.",
                "parent_operators": [],
                "applicability_conditions": [],
                "selection_strategy": {"kind": selection},
                "transformations": transformations,
                "repair_strategy": None,
                "fallback_strategy": {"kind": "rollback_on_failure"},
                "parameters": {},
                "expected_mechanism": mechanism,
                "target_failure_modes": [],
            }
        )
        for name, selection, transformations, mechanism in payloads
    }


__all__ = ["manual_operator_specs"]

