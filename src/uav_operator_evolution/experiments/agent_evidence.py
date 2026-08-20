"""Evidence preparation shared by Phase-8 CLI and orchestration workflows."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..agents.evidence import DesignBudget, EvidenceBundleBuilder, OperatorEvidenceBundle
from ..config import ExperimentConfig
from ..diagnosis.counterfactual import CounterfactualEvaluator, CounterfactualResult
from ..environment import Environment2D
from ..memory import MechanismMemory
from ..operators.registry import OperatorRegistry, build_manual_operator_registry
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..reproducibility import derive_seed
from ..search.context import SearchContext
from ..trajectory import TrajectoryRecorder


def select_evidence_parents(
    memory: MechanismMemory,
    registry: OperatorRegistry,
    *,
    limit: int = 1,
) -> list[str]:
    """Choose deterministic profiled parents, preferring actual failure evidence."""

    ranked: list[tuple[tuple[float, ...], str]] = []
    for name in registry.names():
        profiles = memory.get_operator_profiles(name, limit=1)
        profile = profiles[0].profile if profiles else {}
        failures = memory.get_failure_modes(name, limit=32)
        attempts = float(
            profile.get("attempts", profile.get("total_calls", 0)) or 0
        )
        delayed = profile.get("mean_delayed_rewards", {}) or {}
        if isinstance(delayed, dict):
            delayed_score = max(
                [float(value) for value in delayed.values() if value is not None]
                or [float(profile.get("mean_immediate_reward", 0.0) or 0.0)]
            )
        else:
            delayed_score = float(
                profile.get("average_delayed_reward", 0.0)
                or profile.get("mean_immediate_reward", 0.0)
                or 0.0
            )
        failure_samples = float(sum(item.count for item in failures))
        ranked.append(
            (
                (
                    1.0 if failures else 0.0,
                    1.0 if profiles else 0.0,
                    attempts,
                    delayed_score,
                    failure_samples,
                ),
                name,
            )
        )
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected = [name for _, name in ranked[: max(1, int(limit))]]
    if not selected:
        raise ValueError("no registered parent operators are available")
    return selected


def _profile_trace_id(memory: MechanismMemory, operator_id: str) -> int | None:
    rows = memory.get_operator_profiles(operator_id, limit=1)
    if not rows:
        return None
    profile = rows[0].profile
    for key in ("representative_failure_ids", "representative_success_ids"):
        values = profile.get(key)
        if isinstance(values, list):
            for value in values:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    return None


def _counterfactual_results(
    config: ExperimentConfig,
    memory: MechanismMemory,
    recorder: TrajectoryRecorder,
    registry: OperatorRegistry,
    parent_id: str,
    maps: Sequence[Environment2D],
) -> tuple[list[CounterfactualResult], int | None]:
    if config.diagnostics.counterfactual_states <= 0 or not maps:
        return [], None
    trace_id = _profile_trace_id(memory, parent_id)
    trace = recorder.get_trace(trace_id) if trace_id is not None else None
    if trace is None:
        return [], None
    environment = next((item for item in maps if item.map_id == trace.map_id), None)
    if environment is None:
        return [], None
    raw_path = trace.before_state.get("path")
    if not isinstance(raw_path, list) or len(raw_path) < 2:
        return [], None
    try:
        path = [(float(point[0]), float(point[1])) for point in raw_path]
    except (TypeError, ValueError, IndexError):
        return [], None

    evaluator = PathEvaluator(
        ObjectiveWeights.model_validate(config.objective.model_dump())
    )
    before = evaluator.evaluate(path, environment)
    context = SearchContext(
        iteration=trace.iteration,
        max_iterations=config.search.train_iterations,
        current_evaluation=before,
        best_evaluation=before,
        stagnation_count=int(
            trace.context.get("search_features", {}).get("stagnation_count", 0)
            if isinstance(trace.context.get("search_features"), dict)
            else 0
        ),
    )
    operator_names = [parent_id]
    operator_names.extend(
        name for name in registry.names() if name != parent_id
    )
    operator_names = operator_names[: config.diagnostics.counterfactual_operators]
    operators = {name: registry.get(name) for name in operator_names}
    seed = derive_seed(config.seed, "phase8-counterfactual", trace.trace_id, trace.map_id)
    results = CounterfactualEvaluator(
        max_states=1,
        seed=seed,
    ).evaluate_path_state(
        path,
        environment,
        context,
        evaluator,
        operators,
        seed=seed,
    )
    return [
        item.model_copy(
            update={"source_trace_id": trace.trace_id, "candidate_state": None}
        )
        for item in results
    ], seed


def build_evidence_for_run(
    config: ExperimentConfig,
    database: str | Path,
    *,
    parent_operator_ids: Sequence[str] | None = None,
    train_maps: Sequence[Environment2D] = (),
    problem_summary: str = (
        "Improve fixed-budget UAV path search using computed trajectory, "
        "mechanism-memory, and bounded counterfactual evidence."
    ),
    registry: OperatorRegistry | None = None,
) -> OperatorEvidenceBundle:
    """Build one compact bundle without loading the full trajectory table."""

    operator_registry = registry or build_manual_operator_registry()
    with MechanismMemory(database) as memory, TrajectoryRecorder(database) as recorder:
        parents = list(parent_operator_ids or select_evidence_parents(memory, operator_registry))
        counterfactual, counterfactual_seed = _counterfactual_results(
            config,
            memory,
            recorder,
            operator_registry,
            parents[0],
            train_maps,
        )
        budget = DesignBudget.model_validate(
            config.agent.design_budget.model_dump(mode="python")
        )
        return EvidenceBundleBuilder(
            memory,
            operator_registry,
            recorder=recorder,
            minimum_reliable_samples=config.diagnostics.minimum_context_samples,
        ).build(
            problem_summary,
            parents,
            budget,
            counterfactual_results=counterfactual,
            counterfactual_seed=counterfactual_seed,
        )


__all__ = ["build_evidence_for_run", "select_evidence_parents"]
