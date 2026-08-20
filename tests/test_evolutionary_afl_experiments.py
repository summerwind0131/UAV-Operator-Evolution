from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from uav_operator_evolution.config import OutputConfig, load_config
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.obstacles import RectangleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.planning_benchmarks import (
    PlanningBudget,
    run_with_trusted_validation,
)
from uav_operator_evolution.planning_benchmarks.evolutionary_afl_experiments import (
    EvolutionaryAFLExperimentPlanner,
)
from uav_operator_evolution.planning_benchmarks.runner import run_planner_benchmark


CORE_HASH = "79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f"
OFFLINE_ARTIFACT = Path(
    "artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3"
)


def _environment() -> Environment2D:
    return Environment2D(
        map_id="ablation-unit-map",
        width=100.0,
        height=100.0,
        start=(5.0, 5.0),
        goal=(95.0, 95.0),
        obstacles=[
            RectangleObstacle(min_x=40.0, min_y=20.0, max_x=60.0, max_y=80.0)
        ],
        safety_distance=2.0,
        difficulty="rooms_maze",
        layout_subtype="rooms",
        seed=17,
    )


def test_frozen_evolutionary_v1_core_hash_is_unchanged() -> None:
    source = Path(
        "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == CORE_HASH


@pytest.mark.parametrize(
    "variant",
    [
        "no_quality_diversity_archive",
        "no_crossover",
        "move_only",
        "no_rooms_maze_strategy",
        "fixed_length_population",
    ],
)
def test_ablation_variants_are_offline_bounded_and_feasible(variant: str) -> None:
    planner = EvolutionaryAFLExperimentPlanner(
        OFFLINE_ARTIFACT,
        arm_id=f"unit_{variant}",
        variant=variant,
        population_size=8,
        archive_size=4,
        max_generations=1,
        base_iteration_limit=4,
    )
    result = run_with_trusted_validation(
        planner,
        _environment(),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=1.0, max_objective_evaluations=200),
        np.random.default_rng(301),
    )
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert result.objective_evaluations <= 200
    assert result.diagnostics["llm_calls_during_planning"] == 0
    assert result.diagnostics["algorithm"]["experiment_variant"] == variant
    assert not result.diagnostics["algorithm"]["frozen_v1_core_modified"]
    if variant == "move_only":
        assert result.diagnostics["operator_successes"]["insert"] == 0
        assert result.diagnostics["operator_successes"]["delete"] == 0
        assert result.diagnostics["operator_successes"]["swap"] == 0
        assert result.diagnostics["operator_successes"]["crossover"] == 0


def test_planner_override_evolutionary_arm_is_rejected_before_test_access(
    tmp_path: Path,
) -> None:
    config = load_config("configs/smoke.yaml").model_copy(
        update={
            "output": OutputConfig(
                data_dir=tmp_path / "not-opened",
                results_dir=tmp_path / "results",
                figures_dir=tmp_path / "figures",
            )
        }
    )
    planner = EvolutionaryAFLExperimentPlanner(
        OFFLINE_ARTIFACT,
        arm_id="test_guard",
        variant="no_crossover",
        population_size=8,
        archive_size=4,
        max_generations=1,
        base_iteration_limit=4,
    )
    with pytest.raises(ValueError, match="Train/Validation"):
        run_planner_benchmark(
            config,
            split="test",
            planners=["evolutionary_afl_uav:test_guard"],
            planner_overrides={"evolutionary_afl_uav:test_guard": planner},
        )
    assert not config.output.data_dir.exists()


def test_short_time_sensitivity_arm_keeps_finalization_inside_budget() -> None:
    planner = EvolutionaryAFLExperimentPlanner(
        OFFLINE_ARTIFACT,
        arm_id="short_time_guard",
        variant="full_v1",
        population_size=8,
        archive_size=4,
        max_generations=2,
        base_iteration_limit=4,
    )
    result = run_with_trusted_validation(
        planner,
        _environment(),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=0.25, max_objective_evaluations=200),
        np.random.default_rng(911),
    )
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert result.elapsed_seconds <= 0.25
    assert result.diagnostics["advertised_time_limit_seconds"] == 0.25
    assert result.diagnostics["cooperative_time_limit_seconds"] == pytest.approx(0.225)
