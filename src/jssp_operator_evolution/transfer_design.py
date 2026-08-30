"""Deterministic JSSP redesign from domain-neutral mechanism records."""

from __future__ import annotations

from collections.abc import Sequence

from operator_evolution_core.memory import MechanismRecordV1
from operator_evolution_core.proposal import proposal_hash

from .operators import JSSPOperatorSpec


def _primary_tag(records: Sequence[MechanismRecordV1], seed: int) -> str:
    if not records:
        return ("diversify", "intensify", "repair")[seed % 3]
    scores = {tag: 0.0 for tag in ("repair", "diversify", "intensify")}
    for record in records:
        for tag in scores:
            if tag in record.mechanism_tags:
                scores[tag] += record.evidence_strength
    return min(scores, key=lambda tag: (-scores[tag], tag))


def design_jssp_operator_from_mechanisms(
    records: Sequence[MechanismRecordV1],
    *,
    master_seed: int,
    candidate_index: int,
) -> JSSPOperatorSpec:
    """Realize semantic evidence using only the target-domain ``jssp-v1`` IR."""

    if candidate_index < 0:
        raise ValueError("candidate_index must be non-negative")
    records = tuple(records)
    primary = _primary_tag(records, master_seed + candidate_index)
    variant = (master_seed + candidate_index) % 2
    if primary == "repair":
        selector = "high_idle_gap" if variant == 0 else "bottleneck_block"
        transform = "insert"
    elif primary == "intensify":
        selector = (
            "critical_block_adjacent" if variant == 0 else "critical_block_endpoints"
        )
        transform = "swap"
    else:
        selector = "random_pair" if variant == 0 else "bounded_pair"
        transform = "swap" if variant == 0 else "reverse"
    evidence_ids = tuple(record.mechanism_id for record in records)
    identity = proposal_hash(
        {
            "schema": "jssp-transfer-redesign-v1",
            "primary": primary,
            "variant": variant,
            "evidence": evidence_ids,
            "master_seed": master_seed,
            "candidate_index": candidate_index,
        }
    )
    return JSSPOperatorSpec.model_validate(
        {
            "operator_id": f"transfer-{primary}-{identity[:12]}",
            "name": f"Target-domain {primary} redesign",
            "description": (
                "Deterministic jssp-v1 realization of abstract mechanism evidence; "
                "source-domain IR and capabilities are not available to this designer."
            ),
            "parent_ids": list(evidence_ids),
            "selector": {
                "kind": selector,
                "max_distance": 16,
                "max_attempts": 8,
            },
            "transform": {"kind": transform, "max_segment_length": 32},
            "repair": {"kind": "multiplicity_guard"},
        }
    )


__all__ = ["design_jssp_operator_from_mechanisms"]
