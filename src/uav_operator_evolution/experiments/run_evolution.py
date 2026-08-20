"""Evolution, artifact export, and full demo workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..evolution.manager import EvolutionResult, OperatorEvolutionManager
from ..reproducibility import derive_seed
from ..runtime import RunPaths
from ..search.executor import SearchExecutor
from ..trajectory import TrajectoryRecorder
from ..visualization.diagnostics import generate_diagnostic_figures
from ..visualization.lineage import plot_lineage
from ..visualization.paths import plot_path_comparison
from .common import write_csv, write_json


def _lineage_rows(result: EvolutionResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {"operator_name": name, "generation": 0, "parent_operators": [], "retained": True}
        for name in result.initial_population
    ]
    for summary in result.generations:
        reports = {report.candidate_operator: report for report in summary.validations}
        for proposal in summary.proposals:
            report = reports.get(proposal.spec.name)
            rows.append(
                {
                    "operator_name": proposal.spec.name,
                    "generation": summary.generation + 1,
                    "parent_operators": proposal.spec.parent_operators,
                    "retained": bool(report.retained) if report else False,
                    "reason": report.retention_reasons if report else ["not evaluated"],
                }
            )
    return rows


def _path_figure(
    config: ExperimentConfig,
    paths: RunPaths,
    manager: OperatorEvolutionManager,
    environment: Environment2D,
) -> None:
    seed = derive_seed(config.seed, "demo-path", environment.map_id)
    executor = SearchExecutor(
        manager.last_population,
        manager.evaluator,
        max_iterations=config.search.test_iterations,
        temperature_start_ratio=config.search.temperature_start_ratio,
        temperature_end_ratio=config.search.temperature_end_ratio,
        recent_window=config.search.recent_window,
        initializer_grid_resolution=config.maps.grid_resolution,
    )
    result = executor.run(environment, np.random.default_rng(seed))
    figure = plot_path_comparison(
        environment,
        result.initial_path,
        result.best_path,
        before_evaluation=result.initial_evaluation,
        after_evaluation=result.best_evaluation,
        output_path=paths.figure_dir / "01_path_comparison.png",
    )
    import matplotlib.pyplot as plt

    plt.close(figure)


def run_evolution_workflow(
    config: ExperimentConfig,
    paths: RunPaths,
    datasets: dict[str, list[Environment2D]],
) -> tuple[EvolutionResult, list[Path]]:
    manager = OperatorEvolutionManager(
        config,
        paths.database,
        jsonl_path=(paths.result_dir / "traces.jsonl" if config.output.export_jsonl else None),
    )
    result = manager.run(datasets, paths.run_id)
    write_json(paths.result_dir / "evolution_summary.json", result)
    metric_rows = [metric.model_dump(mode="json") for metric in result.metrics]
    write_csv(paths.result_dir / "metrics.csv", metric_rows)
    write_json(paths.result_dir / "operator_profiles.json", result.profiles)
    write_json(
        paths.result_dir / "validation_reports.json",
        [
            report.model_dump(mode="json")
            for summary in result.generations
            for report in summary.validations
        ],
    )
    write_json(
        paths.result_dir / "candidate_specs.json",
        [
            proposal.model_dump(mode="json")
            for summary in result.generations
            for proposal in summary.proposals
        ],
    )
    lineage_rows = _lineage_rows(result)
    write_json(paths.result_dir / "lineage.json", lineage_rows)
    if datasets["test"]:
        _path_figure(config, paths, manager, datasets["test"][0])
    with TrajectoryRecorder(paths.database) as recorder:
        traces = recorder.list_traces()
    figures = generate_diagnostic_figures(
        traces,
        result.profiles,
        result.synergies,
        metric_rows,
        result.test_outcomes,
        paths.figure_dir,
    )
    figures.append(plot_lineage(lineage_rows, paths.figure_dir / "07_lineage.png"))
    if (paths.figure_dir / "01_path_comparison.png").exists():
        figures.insert(0, paths.figure_dir / "01_path_comparison.png")
    return result, figures

