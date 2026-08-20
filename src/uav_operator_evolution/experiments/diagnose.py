"""Trajectory diagnosis and mechanism-memory update workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..diagnosis.diagnoser import OperatorDiagnoser
from ..memory import MechanismMemory
from ..operators.catalog import manual_operator_specs
from ..trajectory import OperatorTrace, TrajectoryRecorder
from ..visualization.diagnostics import generate_diagnostic_figures
from .common import write_json


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _annotate_context_bins(traces: list[OperatorTrace]) -> list[OperatorTrace]:
    smooth_values = [
        float(_nested(trace.before_state, "path_features", "smoothness", default=0.0) or 0.0)
        for trace in traces
    ]
    q1, q2 = (np.quantile(smooth_values, [1 / 3, 2 / 3]) if smooth_values else (0.0, 0.0))
    annotated: list[OperatorTrace] = []
    for trace in traces:
        env = trace.context.get("environment_features", {})
        search = trace.context.get("search_features", {})
        density = float(env.get("obstacle_density", 0.0) or 0.0)
        ratio = float(search.get("iteration_ratio", 0.0) or 0.0)
        stagnation = int(search.get("stagnation_count", 0) or 0)
        collisions = int(trace.before_state.get("collision_count", 0) or 0)
        smoothness = float(_nested(trace.before_state, "path_features", "smoothness", default=0.0) or 0.0)
        analysis = {
            "map_type": trace.map_difficulty or "unknown",
            "obstacle_density": "low" if density < 0.05 else "medium" if density < 0.12 else "high",
            "search_phase": "early" if ratio < 1 / 3 else "middle" if ratio < 2 / 3 else "late",
            "stagnation": "low" if stagnation < 5 else "medium" if stagnation < 15 else "high",
            "feasible_before": bool(trace.before_feasible),
            "collision_count": "zero" if collisions == 0 else "one" if collisions == 1 else "multiple",
            "smoothness": "low" if smoothness <= q1 else "medium" if smoothness <= q2 else "high",
        }
        annotated.append(trace.model_copy(update={"context": {**trace.context, "analysis": analysis}}))
    return annotated


def run_diagnosis_workflow(
    config: ExperimentConfig,
    run_dir: str | Path,
    *,
    figure_dir: str | Path | None = None,
) -> dict[str, Any]:
    directory = Path(run_dir)
    database = directory / "experiment.sqlite"
    if not database.exists():
        raise FileNotFoundError(f"trajectory database not found: {database}")
    with TrajectoryRecorder(database) as recorder:
        recorder.update_delayed_rewards(config.diagnostics.delayed_horizons)
        traces = recorder.list_traces()
    annotated = _annotate_context_bins(traces)
    diagnoser = OperatorDiagnoser(
        minimum_context_samples=1,
        representative_cases=config.diagnostics.representative_cases,
    )
    global_profiles = diagnoser.diagnose(annotated)
    group_names = [
        "map_type",
        "obstacle_density",
        "search_phase",
        "stagnation",
        "feasible_before",
        "collision_count",
        "smoothness",
    ]
    grouped: dict[str, list[Any]] = {}
    for group in group_names:
        grouped[group] = OperatorDiagnoser(
            minimum_context_samples=config.diagnostics.minimum_context_samples,
            representative_cases=config.diagnostics.representative_cases,
        ).diagnose(annotated, group_by=f"context.analysis.{group}")
    synergies = diagnoser.analyze_synergies(
        annotated,
        min_samples=config.diagnostics.minimum_context_samples,
        reward_horizon=min(config.diagnostics.delayed_horizons),
    )
    global_by_name = {profile.operator_id: profile for profile in global_profiles}
    profile_rows: list[dict[str, Any]] = []
    for profile in global_profiles:
        baseline = float(profile.mean_immediate_reward or 0.0)
        contexts = [item for values in grouped.values() for item in values if item.operator_id == profile.operator_id]
        effective = [
            {**item.context, "average_reward": item.mean_immediate_reward, "calls": item.attempts}
            for item in contexts
            if item.mean_immediate_reward is not None and item.mean_immediate_reward > baseline
        ]
        failure = [
            {**item.context, "average_reward": item.mean_immediate_reward, "calls": item.attempts}
            for item in contexts
            if item.mean_immediate_reward is not None and item.mean_immediate_reward < 0
        ]
        relations = [
            relation.model_dump(mode="json")
            for relation in synergies
            if relation.first_operator == profile.operator_id
        ]
        row = profile.model_dump(mode="json")
        row["effective_contexts"] = effective[:10]
        row["failure_contexts"] = failure[:10]
        row["synergy_relations"] = relations[:10]
        row["evidence_status"] = "computed" if profile.attempts >= config.diagnostics.minimum_context_samples else "insufficient_evidence"
        profile_rows.append(row)

    with MechanismMemory(database) as memory:
        specs = manual_operator_specs()
        for row in profile_rows:
            name = row["operator_name"]
            spec = specs.get(name)
            existing = memory.get_mechanism(name)
            definition = (
                spec.model_dump(mode="json")
                if spec is not None
                else existing.definition
                if existing is not None
                else {"name": name}
            )
            existing_metadata = dict(existing.metadata) if existing is not None else {}
            memory.add_mechanism(
                name,
                definition,
                name=name,
                description=(spec.description if spec else existing.description if existing else "Profiled operator"),
                status=existing.status if existing is not None else "active",
                score=float(row.get("average_delayed_reward") or row.get("average_immediate_reward") or 0.0),
                evidence_count=int(row["total_calls"]),
                success_rate=float(row["immediate_improvement_rate"]),
                tags=sorted(set([*(existing.tags if existing else []), "operator", "profiled"])),
                metadata={
                    **existing_metadata,
                    "profile": row,
                    "evidence_type": "association",
                },
            )
            memory.add_operator_profile(
                row,
                operator_id=name,
                run_id="diagnose",
                generation=int(_nested(row, "context", "generation", default=0) or 0),
            )
            confidence = min(1.0, int(row["total_calls"]) / max(1, config.diagnostics.minimum_context_samples * 4))
            if row.get("effective_contexts"):
                memory.add_insight(
                    operator_id=name,
                    insight_type="effective_mechanism",
                    evidence={
                        "average_immediate_reward": row.get("average_immediate_reward"),
                        "average_delayed_reward": row.get("average_delayed_reward"),
                        "trace_ids": row.get("representative_success_cases", []),
                    },
                    confidence=confidence,
                    applicable_context=row["effective_contexts"][0],
                    source_profile_id=f"diagnosis:{name}",
                )
            if row.get("failure_contexts") or row.get("failure_modes"):
                memory.add_insight(
                    operator_id=name,
                    insight_type="failure_mode",
                    evidence={
                        "failure_modes": row.get("failure_modes", {}),
                        "trace_ids": row.get("representative_failure_cases", []),
                    },
                    confidence=confidence,
                    failure_context=(row.get("failure_contexts") or [{}])[0],
                    source_profile_id=f"diagnosis:{name}",
                )
            for mode, count in (row.get("failure_modes") or {}).items():
                memory.add_failure_mode(
                    mode,
                    mechanism_id=name,
                    operator_id=name,
                    count=int(count),
                    evidence=row.get("representative_failure_cases", []),
                    metadata={"evidence_type": "association"},
                )
        for relation in synergies:
            memory.add_synergy(
                relation.first_operator,
                relation.second_operator,
                relation.synergy,
                sample_count=relation.occurrences,
                context=relation.context,
                metadata={"evidence_type": "association"},
            )
            memory.add_insight(
                operator_id=relation.first_operator,
                insight_type="synergy",
                evidence=relation.model_dump(mode="json"),
                confidence=min(1.0, relation.occurrences / max(1, config.diagnostics.minimum_context_samples * 4)),
                applicable_context=relation.context,
                source_profile_id=f"synergy:{relation.first_operator}->{relation.second_operator}",
            )

    report = {
        "trace_count": len(traces),
        "operator_profiles": profile_rows,
        "grouped_profiles": {
            name: [profile.model_dump(mode="json") for profile in profiles]
            for name, profiles in grouped.items()
        },
        "synergies": [relation.model_dump(mode="json") for relation in synergies],
    }
    write_json(directory / "diagnosis.json", report)
    target_figures = Path(figure_dir) if figure_dir else directory / "figures"
    generate_diagnostic_figures(
        traces,
        profile_rows,
        report["synergies"],
        [],
        [],
        target_figures,
    )
    return report
