"""Deterministic, evidence-grounded composite operator designer."""

from __future__ import annotations

import re
from typing import Any

from ..operators.specs import (
    CollisionSegmentSelection,
    FallbackSpec,
    LongSegmentSelection,
    ObstacleDetourSpec,
    OperatorSpec,
    ReconstructSegmentSpec,
    RepairSpec,
    ShortcutSegmentSpec,
    SmoothSegmentSpec,
)
from .designer_base import OperatorProposal


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", value)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_") or "Parent"
    if not cleaned[0].isalpha():
        cleaned = f"Operator_{cleaned}"
    return cleaned


class HeuristicDesigner:
    """Generate one bounded composite from computed profiles and synergy evidence."""

    def propose(
        self,
        problem_description: str,
        parent_specs: list[OperatorSpec],
        parent_profiles: list[Any],
        memory_context: list[Any],
        success_cases: list[dict[str, Any]],
        failure_cases: list[dict[str, Any]],
    ) -> OperatorProposal:
        if not parent_specs:
            raise ValueError("HeuristicDesigner requires at least one parent specification")

        profiles_by_name = {
            str(_value(profile, "operator_name", "")): profile for profile in parent_profiles
        }
        ranked = sorted(
            parent_specs,
            key=lambda spec: (
                float(_value(profiles_by_name.get(spec.name, {}), "average_delayed_reward", 0.0) or 0.0),
                float(_value(profiles_by_name.get(spec.name, {}), "average_immediate_reward", 0.0) or 0.0),
            ),
            reverse=True,
        )
        parent = ranked[0]
        profile = profiles_by_name.get(parent.name, parent_profiles[0] if parent_profiles else {})
        total_calls = int(_value(profile, "total_calls", 0) or 0)
        delayed = float(_value(profile, "average_delayed_reward", 0.0) or 0.0)
        immediate = float(_value(profile, "average_immediate_reward", 0.0) or 0.0)
        feasibility = float(_value(profile, "feasibility_rate", 0.0) or 0.0)
        failure_contexts = list(_value(profile, "failure_contexts", []) or [])
        synergy_relations = list(_value(profile, "synergy_relations", []) or [])

        positive_smooth = False
        synergy_evidence = ""
        for relation in synergy_relations:
            label = str(_value(relation, "relation", _value(relation, "label", ""))).lower()
            second = str(
                _value(relation, "second_operator", _value(relation, "operator_j", ""))
            ).lower()
            delta = float(_value(relation, "reward_delta", _value(relation, "delta", 0.0)) or 0.0)
            if ("positive" in label or delta > 0) and "smooth" in second:
                positive_smooth = True
                synergy_evidence = f"computed synergy with {second}: reward delta={delta:.6g}"
                break

        parent_lower = parent.name.lower()
        targets_collision = "obstacle" in parent_lower or "detour" in parent_lower or any(
            "collision" in str(context).lower() for context in failure_contexts
        )
        if targets_collision or positive_smooth:
            selection = CollisionSegmentSelection()
            transformations = [ObstacleDetourSpec(), SmoothSegmentSpec(strength=0.55)]
            mechanism = "repair a colliding segment, then remove detour-induced curvature"
            changes = ["compose obstacle detour with a subsequent local smoothing step"]
            target_failures = ["collision", "jagged_detour"]
        else:
            selection = LongSegmentSelection()
            transformations = [ShortcutSegmentSpec(), SmoothSegmentSpec(strength=0.45)]
            mechanism = "shorten a long segment and smooth the surviving local geometry"
            changes = ["compose long-segment shortcutting with local smoothing"]
            target_failures = ["path_redundancy", "excess_curvature"]

        repair = None
        if feasibility < 0.8 or failure_contexts:
            repair = RepairSpec(
                transformations=[ReconstructSegmentSpec(max_points=8)], max_attempts=2
            )
            changes.append("add bounded reconstruction repair for the observed failure contexts")

        evidence = [
            f"{parent.name}: calls={total_calls}",
            f"{parent.name}: mean immediate reward={immediate:.6g}",
            f"{parent.name}: mean delayed reward={delayed:.6g}",
            f"{parent.name}: feasibility rate={feasibility:.3f}",
        ]
        if synergy_evidence:
            evidence.append(synergy_evidence)
        if failure_contexts:
            evidence.append(f"computed failure contexts={failure_contexts[:3]}")
        if memory_context:
            first = memory_context[0]
            evidence.append(f"memory insight={_value(first, 'insight_type', 'available')}")

        spec = OperatorSpec(
            name=f"{_safe_name(parent.name)}_Composite",
            description=(
                "Deterministic composite generated from measured immediate/delayed rewards, "
                "feasibility and pairwise synergy."
            ),
            parent_operators=[parent.name],
            applicability_conditions=[],
            selection_strategy=selection,
            transformations=transformations,
            repair_strategy=repair,
            fallback_strategy=FallbackSpec(),
            parameters={"evidence_calls": total_calls},
            expected_mechanism=mechanism,
            target_failure_modes=target_failures,
        )
        return OperatorProposal(
            specification=spec,
            design_rationale=(
                f"Selected {parent.name} from computed profiles and added a structurally distinct "
                "bounded follow-up/repair sequence."
            ),
            evidence_used=evidence,
            target_failure_modes=target_failures,
            changes_from_parents=changes,
            expected_contexts=["contexts reported by the parent profile", problem_description[:200]],
            potential_risks=["extra transformations may increase runtime", "small samples are associative evidence"],
            evidence_level="computed" if total_calls >= 3 else "insufficient_evidence",
        )
