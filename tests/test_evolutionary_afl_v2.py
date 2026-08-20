from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pytest

from uav_operator_evolution.config import OutputConfig, load_config
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.obstacles import RectangleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.planning_benchmarks.core import (
    BudgetedEvaluator,
    PlanningBudget,
    PlanningTimeout,
    run_with_trusted_validation,
)
from uav_operator_evolution.planning_benchmarks.evolutionary_afl_v2 import (
    EvolutionaryAFLUAVV2Planner,
    _DeadlineEvaluatorView,
    _extract_wall_portals,
)
from uav_operator_evolution.planning_benchmarks.runner import run_planner_benchmark


V1_HASH = "79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f"
OFFLINE_ARTIFACT = Path(
    "artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3"
)


def _rooms_environment() -> Environment2D:
    return Environment2D(
        map_id="v2-rooms-unit",
        width=100.0,
        height=100.0,
        start=(5.0, 5.0),
        goal=(95.0, 95.0),
        obstacles=[
            RectangleObstacle(min_x=40.0, min_y=0.0, max_x=41.2, max_y=43.0),
            RectangleObstacle(min_x=40.0, min_y=57.0, max_x=41.2, max_y=100.0),
        ],
        safety_distance=2.0,
        difficulty="rooms_maze",
        layout_subtype="rooms",
        seed=19,
    )


def _planner() -> EvolutionaryAFLUAVV2Planner:
    return EvolutionaryAFLUAVV2Planner(
        OFFLINE_ARTIFACT,
        arm_id="v2_unit",
        population_size=32,
        archive_size=8,
        max_generations=3,
        base_iteration_limit=8,
    )


def test_v2_does_not_modify_frozen_v1() -> None:
    source = Path(
        "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py"
    )
    assert hashlib.sha256(source.read_bytes()).hexdigest() == V1_HASH


def test_v2_extracts_room_door_center() -> None:
    assert _extract_wall_portals(_rooms_environment()) == [(40.6, 50.0)]


def test_v2_operation_start_guard_stops_new_expensive_work() -> None:
    problem = BudgetedEvaluator(
        _rooms_environment(),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=1.0, max_objective_evaluations=500),
    )
    problem.started_at = time.perf_counter() - 0.83
    guarded = _DeadlineEvaluatorView(
        problem,
        deadline_seconds=0.88,
        start_guard_seconds=0.06,
    )
    with pytest.raises(PlanningTimeout, match="operation start guard"):
        guarded.check_time()
    assert guarded.local_timeout_triggered is True


def test_v2_is_offline_feasible_multisource_and_finishes_inside_budget() -> None:
    result = run_with_trusted_validation(
        _planner(),
        _rooms_environment(),
        PathEvaluator(),
        PlanningBudget(time_limit_seconds=1.0, max_objective_evaluations=500),
        np.random.default_rng(1203),
    )
    assert result.status != "timeout"
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert result.elapsed_seconds < 1.0
    assert result.diagnostics["v2_finalization_reserve_seconds"] > 0.05
    assert result.diagnostics["portal_centers_detected"] == 1
    assert result.diagnostics["llm_calls_during_planning"] == 0
    assert result.diagnostics["algorithm"]["version"].startswith(
        "evolutionary-afl-uav-v2"
    )
    source_counts = result.diagnostics["population_source_counts"]
    assert sum(source_counts.values()) == 32
    assert source_counts["afl"] >= 1
    assert any(source_counts[name] > 0 for name in ("astar", "theta_star", "prm"))
    assert result.diagnostics["research_claim_eligible"] is False


def test_v2_inherits_existing_test_split_guard(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml").model_copy(
        update={
            "output": OutputConfig(
                data_dir=tmp_path / "must-not-open",
                results_dir=tmp_path / "results",
                figures_dir=tmp_path / "figures",
            )
        }
    )
    planner = _planner()
    with pytest.raises(ValueError, match="restricted to Train/Validation"):
        run_planner_benchmark(
            config,
            split="test",
            planners=["evolutionary_afl_uav:v2_unit"],
            planner_overrides={"evolutionary_afl_uav:v2_unit": planner},
        )
    assert not config.output.data_dir.exists()
