from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from uav_operator_evolution.config import ExperimentConfig, OutputConfig, load_config
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.generator import generate_dataset
from uav_operator_evolution.environment.obstacles import RectangleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.planning_benchmarks import (
    EvolutionaryAFLUAVPlanner,
    PlanningBudget,
    build_planners,
    path_hash,
    run_with_trusted_validation,
)
from uav_operator_evolution.planning_benchmarks.runner import (
    _json_safe,
    run_planner_benchmark,
)


def _environment(*, obstacle: bool = False) -> Environment2D:
    return Environment2D(
        map_id="test-map",
        width=100.0,
        height=100.0,
        start=(5.0, 5.0),
        goal=(95.0, 95.0),
        obstacles=(
            [
                RectangleObstacle(
                    min_x=40.0,
                    min_y=20.0,
                    max_x=60.0,
                    max_y=80.0,
                )
            ]
            if obstacle
            else []
        ),
        safety_distance=2.0,
        difficulty="sparse",
        seed=7,
    )


def test_all_planners_return_trusted_paths_on_empty_map() -> None:
    environment = _environment()
    evaluator = PathEvaluator()
    budget = PlanningBudget(
        time_limit_seconds=0.5,
        max_objective_evaluations=50,
    )
    for planner in build_planners().values():
        result = run_with_trusted_validation(
            planner,
            environment,
            evaluator,
            budget,
            np.random.default_rng(123),
        )
        assert result.path is not None, (planner.name, result.status, result.message)
        assert result.trusted_evaluation is not None
        assert result.trusted_evaluation.feasible
        assert result.objective_evaluations <= 50


def test_non_finite_diagnostics_are_strict_json_safe() -> None:
    assert _json_safe({"best": float("inf"), "nested": [float("-inf")]}) == {
        "best": None,
        "nested": [None],
    }


def test_all_planners_solve_single_obstacle_map() -> None:
    environment = _environment(obstacle=True)
    evaluator = PathEvaluator()
    budget = PlanningBudget(
        time_limit_seconds=1.0,
        max_objective_evaluations=200,
    )
    planners = build_planners()
    for name, planner in planners.items():
        result = run_with_trusted_validation(
            planner,
            environment,
            evaluator,
            budget,
            np.random.default_rng(9),
        )
        assert result.path is not None, (name, result.status, result.message)
        assert result.trusted_evaluation is not None, name
        assert result.trusted_evaluation.feasible, name


def test_seeded_rrt_is_reproducible_and_seed_sensitive() -> None:
    environment = _environment(obstacle=True)
    evaluator = PathEvaluator()
    budget = PlanningBudget(
        time_limit_seconds=1.0,
        max_objective_evaluations=200,
    )
    planner = build_planners()["rrt"]
    first = run_with_trusted_validation(
        planner, environment, evaluator, budget, np.random.default_rng(23)
    )
    second = run_with_trusted_validation(
        planner, environment, evaluator, budget, np.random.default_rng(23)
    )
    different = run_with_trusted_validation(
        planner, environment, evaluator, budget, np.random.default_rng(24)
    )
    assert first.path is not None and second.path is not None and different.path is not None
    assert path_hash(first.path) == path_hash(second.path)
    assert path_hash(first.path) != path_hash(different.path)


def test_time_and_evaluation_limits_return_explicit_statuses() -> None:
    environment = _environment(obstacle=True)
    evaluator = PathEvaluator()
    planners = build_planners()
    evaluation_limited = run_with_trusted_validation(
        planners["ga"],
        environment,
        evaluator,
        PlanningBudget(
            time_limit_seconds=1.0,
            max_objective_evaluations=50,
        ),
        np.random.default_rng(4),
    )
    assert evaluation_limited.status == "budget_exhausted"
    assert evaluation_limited.objective_evaluations == 50
    assert evaluation_limited.path is not None
    assert evaluation_limited.trusted_evaluation is not None
    assert evaluation_limited.trusted_evaluation.feasible

    time_limited = run_with_trusted_validation(
        planners["rrt_star"],
        environment,
        evaluator,
        PlanningBudget(
            time_limit_seconds=1e-6,
            max_objective_evaluations=2_000,
        ),
        np.random.default_rng(4),
    )
    assert time_limited.status == "timeout"
    assert time_limited.objective_evaluations <= 2_000


def test_runner_reclassifies_success_returned_after_wall_clock_boundary() -> None:
    class LateSuccessPlanner:
        name = "late_success"
        stochastic = False
        research_claim_eligible = False

        def plan(self, problem, budget, rng):
            while problem.elapsed_seconds < 0.002:
                pass
            return problem.result(
                self.name,
                "success",
                path=[problem.environment.start, problem.environment.goal],
            )

    result = run_with_trusted_validation(
        LateSuccessPlanner(),
        _environment(),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=0.001, max_objective_evaluations=10),
        np.random.default_rng(1),
    )
    assert result.status == "timeout"
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert "trusted wall-clock boundary" in result.message


def test_evolutionary_afl_uav_is_offline_budgeted_and_quality_diverse() -> None:
    artifact = Path(
        "artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3"
    )
    planner = EvolutionaryAFLUAVPlanner(
        artifact,
        arm_id="offline_evolution_test",
        population_size=16,
        archive_size=6,
        max_generations=6,
        base_iteration_limit=16,
    )
    result = run_with_trusted_validation(
        planner,
        _environment(obstacle=True),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=1.0, max_objective_evaluations=500),
        np.random.default_rng(73),
    )
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert result.objective_evaluations <= 500
    assert result.elapsed_seconds <= 1.05
    assert result.diagnostics["llm_calls_during_planning"] == 0
    assert result.diagnostics["archive_unique_paths"] >= 2
    assert result.diagnostics["best_cost"] <= result.diagnostics["seed_cost"] + 1e-9
    assert set(result.diagnostics["operator_attempts"]) == {
        "insert",
        "delete",
        "move",
        "swap",
        "crossover",
    }
    assert all(
        result.diagnostics["operator_attempts"][name] > 0
        for name in ("insert", "delete", "move", "swap")
    )


def test_evolutionary_afl_uav_seed_reproduces_the_archive_trajectory() -> None:
    artifact = Path(
        "artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3"
    )

    def execute(seed: int):
        planner = EvolutionaryAFLUAVPlanner(
            artifact,
            arm_id="reproduction_test",
            population_size=12,
            archive_size=5,
            max_generations=4,
            base_iteration_limit=8,
        )
        return run_with_trusted_validation(
            planner,
            _environment(obstacle=True),
            PathEvaluator(),
            PlanningBudget(time_limit_seconds=2.0, max_objective_evaluations=400),
            np.random.default_rng(seed),
        )

    first = execute(91)
    repeated = execute(91)
    different = execute(92)
    assert path_hash(first.path) == path_hash(repeated.path)
    assert (
        first.diagnostics["archive_path_hashes"]
        == repeated.diagnostics["archive_path_hashes"]
    )
    assert (
        first.diagnostics["archive_path_hashes"]
        != different.diagnostics["archive_path_hashes"]
    )


def test_smoke_runner_writes_one_complete_row_per_execution_arm(
    tmp_path: Path,
) -> None:
    difficulties = [
        "sparse",
        "dense",
        "corridor",
        "clustered",
        "rooms_maze",
        "mixed",
    ]
    config = ExperimentConfig.model_validate(
        {
            "name": "planner-smoke",
            "seed": 20260725,
            "maps": {
                "grid_resolution": 2.0,
                "generation_attempts": 100,
                **{
                    split: {
                        "count": 6,
                        "difficulties": difficulties,
                        "width": 100.0,
                        "height": 100.0,
                        "safety_distance": 2.0,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "output": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "figures_dir": str(tmp_path / "figures"),
            },
        }
    )
    generate_dataset(config)
    report = run_planner_benchmark(
        config,
        split="test",
        maps_per_class=1,
        time_limit_seconds=0.25,
        max_objective_evaluations=50,
        repetitions=1,
        run_id="smoke",
    )
    assert report["selected_maps"] == 6
    assert report["records"] == 66
    destination = Path(report["run_dir"])
    with (destination / "benchmark_runs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 66
    assert len(
        {(row["planner"], row["map_id"], row["repetition"]) for row in rows}
    ) == 66
    metadata = json.loads(
        (destination / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["actual_records"] == metadata["expected_records"] == 66
    assert {
        "benchmark_runs.csv",
        "benchmark_paths.jsonl",
        "benchmark_summary.json",
        "benchmark_summary.csv",
        "benchmark_metadata.json",
    }.issubset({path.name for path in destination.iterdir()})


def test_sealed_dataset_is_rejected_before_results_are_created(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate(
        {
            "name": "sealed-hidden-test",
            "seed": 2026081701,
            "maps": {
                "grid_resolution": 2.0,
                "generation_attempts": 30,
                **{
                    split: {
                        "count": 1,
                        "difficulties": ["sparse"],
                        "width": 100.0,
                        "height": 100.0,
                        "safety_distance": 2.0,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "planning_benchmark": {"planners": ["astar"]},
            "output": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "figures_dir": str(tmp_path / "figures"),
            },
        }
    )
    generate_dataset(config)
    (config.output.data_dir / "SEALED.json").write_text(
        '{"status":"sealed_unrun"}\n', encoding="utf-8"
    )

    with pytest.raises(PermissionError, match="dataset is sealed"):
        run_planner_benchmark(config, split="test", planners=["astar"])

    assert not config.output.results_dir.exists()


def test_multi_artifact_arms_have_independent_record_keys(tmp_path: Path) -> None:
    difficulties = [
        "sparse",
        "dense",
        "corridor",
        "clustered",
        "rooms_maze",
        "mixed",
    ]
    config = ExperimentConfig.model_validate(
        {
            "name": "multi-artifact-smoke",
            "seed": 20260725,
            "maps": {
                "grid_resolution": 2.0,
                "generation_attempts": 100,
                **{
                    split: {
                        "count": 6,
                        "difficulties": difficulties,
                        "width": 100.0,
                        "height": 100.0,
                        "safety_distance": 2.0,
                    }
                    for split in ("train", "validation", "test")
                },
            },
            "output": {
                "data_dir": str(tmp_path / "data"),
                "results_dir": str(tmp_path / "results"),
                "figures_dir": str(tmp_path / "figures"),
            },
        }
    )
    generate_dataset(config)
    artifact = Path(
        "artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3"
    )
    report = run_planner_benchmark(
        config,
        split="validation",
        planners=["astar", "afl_uav", "evolutionary_afl_uav"],
        maps_per_class=1,
        time_limit_seconds=0.25,
        max_objective_evaluations=20,
        repetitions=1,
        afl_artifacts={"openai_gpt41": artifact, "gemini_25pro": artifact},
        evolutionary_afl_artifacts={"offline_evolution": artifact},
        run_id="multi-artifact",
    )
    assert report["records"] == 24
    destination = Path(report["run_dir"])
    with (destination / "benchmark_runs.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert {row["arm_id"] for row in rows if row["planner"] == "afl_uav"} == {
        "openai_gpt41",
        "gemini_25pro",
    }
    assert {
        row["arm_id"]
        for row in rows
        if row["planner"] == "evolutionary_afl_uav"
    } == {"offline_evolution"}
    assert len(
        {
            (row["planner"], row["arm_id"], row["map_id"], row["seed"])
            for row in rows
        }
    ) == 24
    metadata = json.loads(
        (destination / "benchmark_metadata.json").read_text(encoding="utf-8")
    )
    assert [item["arm_id"] for item in metadata["afl_uav_artifacts"]] == [
        "gemini_25pro",
        "openai_gpt41",
        "offline_evolution",
    ]
    evolutionary_metadata = next(
        item
        for item in metadata["afl_uav_artifacts"]
        if item["planner"] == "evolutionary_afl_uav"
    )
    assert evolutionary_metadata["evolutionary_parameters"]["llm_calls_during_planning"] == 0
    assert metadata["afl_generation_summary"]["artifacts"] == 1
    summary = json.loads(
        (destination / "benchmark_summary.json").read_text(encoding="utf-8")
    )
    assert {item["arm_id"] for item in summary["ranking"]} >= {
        "openai_gpt41",
        "gemini_25pro",
    }


def test_frozen_afl_artifacts_cannot_access_test_split(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml").model_copy(
        update={
            "output": OutputConfig(
                data_dir=tmp_path / "data",
                results_dir=tmp_path / "results",
                figures_dir=tmp_path / "figures",
            )
        }
    )
    with pytest.raises(ValueError, match="restricted to Train/Validation"):
        run_planner_benchmark(
            config,
            split="test",
            planners=["afl_uav"],
            afl_artifacts={"openai_gpt41": "not-opened"},
        )
    with pytest.raises(ValueError, match="Train/Validation"):
        run_planner_benchmark(
            config,
            split="test",
            planners=["evolutionary_afl_uav"],
            evolutionary_afl_artifacts={"offline_evolution": "not-opened"},
        )
