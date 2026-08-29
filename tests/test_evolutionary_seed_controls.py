from __future__ import annotations

import hashlib
import statistics
from pathlib import Path

import numpy as np
import pytest

from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.obstacles import RectangleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.planning_benchmarks import (
    PlanningBudget,
    run_with_trusted_validation,
)
from uav_operator_evolution.planning_benchmarks.evolutionary_afl import (
    EvolutionaryAFLUAVPlanner,
)
from uav_operator_evolution.planning_benchmarks.evolutionary_seed_controls import (
    FROZEN_V1_CORE_SHA256,
    SeedSourceEvolutionaryControlPlanner,
)
from scripts.run_evolutionary_seed_source_controls import _validate_frozen_inputs


def _environment() -> Environment2D:
    return Environment2D(
        map_id="seed-control-unit-map",
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
        seed=19,
    )


def test_seed_controls_leave_frozen_v1_core_unchanged() -> None:
    source = Path("src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == FROZEN_V1_CORE_SHA256


def test_dedicated_seed_control_entry_rejects_test_split() -> None:
    with pytest.raises(ValueError, match="Train/Validation-only"):
        _validate_frozen_inputs({"split": "test"})


@pytest.mark.parametrize(
    "method_name",
    [
        "plan",
        "_initialize_population",
        "_apply_operator",
        "_crossover",
        "_update_archive",
        "_select_survivors",
    ],
)
def test_seed_controls_inherit_the_exact_frozen_evolution_method(method_name: str) -> None:
    assert getattr(SeedSourceEvolutionaryControlPlanner, method_name) is getattr(
        EvolutionaryAFLUAVPlanner, method_name
    )


@pytest.mark.parametrize(
    "seed_source",
    ["astar", "theta_star", "handcrafted_destroy_repair"],
)
def test_seed_controls_are_bounded_reproducible_and_feasible(seed_source: str) -> None:
    results = []
    for _ in range(2):
        planner = SeedSourceEvolutionaryControlPlanner(
            arm_id=f"unit_{seed_source}",
            seed_source=seed_source,
            population_size=8,
            archive_size=4,
            max_generations=1,
            base_iteration_limit=4,
        )
        results.append(
            run_with_trusted_validation(
                planner,
                _environment(),
                PathEvaluator(),
                # Leave enough margin for both repeats to execute the same
                # fixed operator count; exact wall-clock cutoffs can otherwise
                # differ by one operation under OS scheduling jitter.
                PlanningBudget(time_limit_seconds=5.0, max_objective_evaluations=200),
                np.random.default_rng(701),
            )
        )
    first, second = results
    assert first.path is not None
    assert first.trusted_evaluation is not None
    assert first.trusted_evaluation.feasible
    assert first.objective_evaluations <= 200
    assert first.path == second.path
    assert first.diagnostics["algorithm"]["seed_source"] == seed_source
    assert first.diagnostics["algorithm"]["shared_evolution_core_sha256"] == (
        FROZEN_V1_CORE_SHA256
    )
    assert first.diagnostics["llm_calls_during_planning"] == 0


@pytest.mark.performance
def test_handcrafted_seed_control_median_terminates_at_one_second_boundary() -> None:
    elapsed_samples: list[float] = []
    for repetition in range(5):
        planner = SeedSourceEvolutionaryControlPlanner(
            arm_id=f"unit_manual_boundary_r{repetition}",
            seed_source="handcrafted_destroy_repair",
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
            np.random.default_rng(702),
        )
        elapsed_samples.append(result.elapsed_seconds)
        assert result.objective_evaluations <= 200
        assert result.status in {"success", "timeout"}
    assert statistics.median(elapsed_samples) < 1.05
