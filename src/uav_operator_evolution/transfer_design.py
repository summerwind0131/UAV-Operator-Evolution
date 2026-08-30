"""Deterministic UAV redesign from domain-neutral mechanism records."""

from __future__ import annotations

from collections.abc import Sequence

from operator_evolution_core.memory import MechanismRecordV1
from operator_evolution_core.proposal import proposal_hash

from .operators.specs import OperatorSpec


def _primary_tag(records: Sequence[MechanismRecordV1], seed: int) -> str:
    if not records:
        return ("diversify", "intensify", "repair")[seed % 3]
    scores = {tag: 0.0 for tag in ("repair", "diversify", "intensify")}
    for record in records:
        for tag in scores:
            if tag in record.mechanism_tags:
                scores[tag] += record.evidence_strength
    return min(scores, key=lambda tag: (-scores[tag], tag))


def design_uav_operator_from_mechanisms(
    records: Sequence[MechanismRecordV1],
    *,
    master_seed: int,
    candidate_index: int,
) -> OperatorSpec:
    """Realize semantic evidence using only target-domain ``uav-v1`` capabilities."""

    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    records = tuple(records)
    primary = _primary_tag(records, master_seed + candidate_index)
    variant = (master_seed + candidate_index) % 2
    if primary == "repair":
        selection = (
            "select_collision_segment"
            if variant == 0
            else "select_continuous_collision_region"
        )
        transformations = [
            {
                "kind": (
                    "generate_obstacle_detour" if variant == 0 else "reconstruct_segment"
                )
            }
        ]
    elif primary == "intensify":
        selection = "select_high_curvature_waypoint" if variant == 0 else "select_long_segment"
        transformations = [
            {"kind": "smooth_segment", "strength": 0.6}
            if variant == 0
            else {"kind": "shortcut_segment"}
        ]
    else:
        selection = "select_random_waypoint"
        transformations = [
            {"kind": "perturb_waypoint", "scale": 8.0}
            if variant == 0
            else {"kind": "shift_segment", "scale": 6.0}
        ]
    evidence_ids = tuple(record.mechanism_id for record in records)
    identity = proposal_hash(
        {
            "schema": "uav-transfer-redesign-v1",
            "primary": primary,
            "variant": variant,
            "evidence": evidence_ids,
            "master_seed": master_seed,
            "candidate_index": candidate_index,
        }
    )
    return OperatorSpec.model_validate(
        {
            "name": f"transfer_{primary}_{identity[:12]}",
            "description": (
                "Deterministic uav-v1 realization of abstract mechanism evidence; "
                "source-domain IR and capabilities are not available to this designer."
            ),
            "parent_operators": list(evidence_ids),
            "applicability_conditions": [],
            "selection_strategy": {"kind": selection},
            "transformations": transformations,
            "repair_strategy": None,
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "parameters": {},
            "expected_mechanism": (
                f"Target-domain {primary} realization from abstract mechanism records."
            ),
            "target_failure_modes": [
                mode for record in records for mode in record.failure_modes
            ][:8],
        }
    )


__all__ = ["design_uav_operator_from_mechanisms"]
