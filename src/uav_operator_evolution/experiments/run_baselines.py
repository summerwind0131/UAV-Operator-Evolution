"""Six-arm baseline comparison under a shared test budget."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..evolution.manager import OperatorEvolutionManager
from ..operators.compiler import OperatorCompiler
from ..operators.registry import default_manual_operators
from ..operators.specs import OperatorSpec
from ..path.evaluator import PathEvaluator
from ..path.initializer import initialize_path
from ..path.models import ObjectiveWeights
from ..reproducibility import derive_seed
from ..runtime import RunPaths
from ..search.executor import SearchExecutor
from .common import write_csv, write_json


def _compiled(config: ExperimentConfig, name: str, mode: str):
    if mode == "random":
        selection = {"kind": "select_random_waypoint"}
        transformations = [{"kind": "perturb_waypoint", "scale": 6.0}, {"kind": "smooth_segment"}]
        mechanism = "uninformed random composite"
    else:
        selection = {"kind": "select_long_segment"}
        transformations = [{"kind": "shortcut_segment"}, {"kind": "smooth_segment"}]
        mechanism = "aggregate-score-only shortening composite"
    spec = OperatorSpec.model_validate(
        {
            "name": name,
            "description": mechanism,
            "selection_strategy": selection,
            "transformations": transformations,
            "fallback_strategy": {"kind": "rollback_on_failure"},
            "expected_mechanism": mechanism,
        }
    )
    return OperatorCompiler(config.dsl).compile(spec)


def _evaluate_arm(
    config: ExperimentConfig,
    arm: str,
    operators: list[Any],
    maps: list[Environment2D],
) -> list[dict[str, Any]]:
    evaluator = PathEvaluator(ObjectiveWeights.model_validate(config.objective.model_dump()))
    rows: list[dict[str, Any]] = []
    for index, environment in enumerate(maps):
        seed = derive_seed(config.seed, "baseline", index, environment.map_id)
        initial = initialize_path(environment, grid_resolution=config.maps.grid_resolution)
        started = time.perf_counter()
        result = SearchExecutor(
            operators,
            evaluator,
            max_iterations=config.search.test_iterations,
            temperature_start_ratio=config.search.temperature_start_ratio,
            temperature_end_ratio=config.search.temperature_end_ratio,
            recent_window=config.search.recent_window,
            initializer_grid_resolution=config.maps.grid_resolution,
        ).run(environment, np.random.default_rng(seed), initial_path=initial)
        rows.append(
            {
                "arm": arm,
                "map_id": environment.map_id,
                "difficulty": environment.difficulty,
                "best_cost": result.best_evaluation.total_cost,
                "feasible": result.best_evaluation.feasible,
                "runtime_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    return rows


def run_baselines_workflow(
    config: ExperimentConfig,
    paths: RunPaths,
    datasets: dict[str, list[Environment2D]],
) -> dict[str, Any]:
    manager = OperatorEvolutionManager(config, paths.database)
    evolution = manager.run(datasets, paths.run_id)
    manual = default_manual_operators()
    random_pool = list(manual)
    random_pool[0] = _compiled(config, "RandomCompositeBaseline", "random")
    score_pool = list(manual)
    score_pool[4] = _compiled(config, "ScoreOnlyCompositeBaseline", "score")
    final_pool = manager.last_population or manual
    parent_isolated = [manual[4] for _ in range(8)]
    evolved_candidates = [operator for operator in final_pool if str(operator.name).startswith("G")]
    best_evolved = evolved_candidates[0] if evolved_candidates else final_pool[4]
    evolved_isolated = [best_evolved for _ in range(8)]
    arms = {
        "initial_manual_operators": manual,
        "random_composite": random_pool,
        "score_only_composite": score_pool,
        "diagnosis_memory_composite": final_pool,
        "parent_operator": parent_isolated,
        "best_evolved_operator": evolved_isolated,
    }
    rows = [row for name, operators in arms.items() for row in _evaluate_arm(config, name, operators, datasets["test"])]
    summaries: list[dict[str, Any]] = []
    for name in arms:
        arm_rows = [row for row in rows if row["arm"] == name]
        summaries.append(
            {
                "arm": name,
                "mean_best_cost": float(np.mean([row["best_cost"] for row in arm_rows])) if arm_rows else None,
                "feasible_rate": float(np.mean([row["feasible"] for row in arm_rows])) if arm_rows else None,
                "mean_runtime_ms": float(np.mean([row["runtime_ms"] for row in arm_rows])) if arm_rows else None,
                "map_count": len(arm_rows),
            }
        )
    report = {
        "run_id": paths.run_id,
        "arms": summaries,
        "paired_rows": rows,
        "evolution_retained_candidates": evolution.retained_candidates,
    }
    write_json(paths.result_dir / "baseline_report.json", report)
    write_csv(paths.result_dir / "baseline_rows.csv", rows)
    return report

