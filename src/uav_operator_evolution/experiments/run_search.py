"""Fixed P0 search and trajectory collection workflow."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..operators.registry import default_manual_operators
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..reproducibility import derive_seed
from ..runtime import RunPaths
from ..search.executor import SearchExecutor
from ..trajectory import TrajectoryRecorder
from ..visualization.paths import plot_path_comparison
from .common import write_csv, write_json


def run_search_workflow(
    config: ExperimentConfig,
    paths: RunPaths,
    maps: list[Environment2D],
) -> dict[str, Any]:
    weights = ObjectiveWeights.model_validate(config.objective.model_dump())
    evaluator = PathEvaluator(weights)
    metrics: list[dict[str, Any]] = []
    first_result = None
    first_environment = None
    jsonl = paths.result_dir / "traces.jsonl" if config.output.export_jsonl else None
    with TrajectoryRecorder(paths.database, jsonl) as recorder:
        for map_index, environment in enumerate(maps):
            seed = derive_seed(config.seed, "run-search", map_index, environment.map_id)
            executor = SearchExecutor(
                default_manual_operators(),
                evaluator,
                max_iterations=config.search.train_iterations,
                temperature_start_ratio=config.search.temperature_start_ratio,
                temperature_end_ratio=config.search.temperature_end_ratio,
                recent_window=config.search.recent_window,
                initializer_grid_resolution=config.maps.grid_resolution,
                recorder=recorder,
            )
            started = time.perf_counter()
            result = executor.run(
                environment,
                np.random.default_rng(seed),
                run_id=paths.run_id,
                generation=0,
            )
            metrics.append(
                {
                    "map_id": environment.map_id,
                    "difficulty": environment.difficulty,
                    "initial_cost": result.initial_evaluation.total_cost,
                    "best_cost": result.best_evaluation.total_cost,
                    "final_cost": result.final_evaluation.total_cost,
                    "feasible": result.best_evaluation.feasible,
                    "acceptance_rate": result.acceptance_rate,
                    "iterations": result.iterations,
                    "runtime_ms": (time.perf_counter() - started) * 1000.0,
                }
            )
            if first_result is None:
                first_result, first_environment = result, environment
        recorder.update_delayed_rewards(config.diagnostics.delayed_horizons, run_id=paths.run_id)
        trace_count = len(recorder.list_traces(paths.run_id))
    if first_result is not None and first_environment is not None:
        figure = plot_path_comparison(
            first_environment,
            first_result.initial_path,
            first_result.best_path,
            before_evaluation=first_result.initial_evaluation,
            after_evaluation=first_result.best_evaluation,
            output_path=paths.figure_dir / "01_path_comparison.png",
        )
        import matplotlib.pyplot as plt

        plt.close(figure)
    summary = {
        "run_id": paths.run_id,
        "map_count": len(maps),
        "trace_count": trace_count,
        "metrics": metrics,
        "database": str(paths.database),
        "figure_dir": str(paths.figure_dir),
    }
    write_json(paths.result_dir / "search_summary.json", summary)
    write_csv(paths.result_dir / "search_metrics.csv", metrics)
    return summary

