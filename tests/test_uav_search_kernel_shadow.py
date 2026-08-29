"""Step-for-step shadow gate captured before routing UAV through the core."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_operator_evolution.environment.generator import load_dataset_split
from uav_operator_evolution.operators.registry import default_manual_operators
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.reproducibility import stable_hash
from uav_operator_evolution.search.executor import SearchExecutor


ROOT = Path(__file__).resolve().parents[1]
SEARCH_SEED = 20_260_820
LEGACY_SHADOW_HASH = (
    "d26477dd9d64bc581dfa4855c1a623369626403abaf98c07d95c2d7bd5f4a820"
)


def test_generic_uav_loop_matches_frozen_legacy_loop_step_for_step() -> None:
    environment = load_dataset_split(ROOT / "data" / "multi-agent-smoke", "train")[0]
    rng = np.random.default_rng(SEARCH_SEED)
    result = SearchExecutor(
        default_manual_operators(),
        evaluator=PathEvaluator(),
        max_iterations=16,
        initializer_grid_resolution=4.0,
    ).run(environment, rng, run_id="uav-step3-shadow")

    best_path = result.initial_path
    steps = []
    for step in result.steps:
        if step.created_new_best:
            best_path = step.current_path_after
        steps.append(
            {
                "operator_id": step.operator_id,
                "candidate_hash": stable_hash(step.candidate_path),
                "current_hash": stable_hash(step.current_path_after),
                "best_hash": stable_hash(best_path),
                "evaluation_before": step.evaluation_before.model_dump(mode="json"),
                "candidate_evaluation": step.candidate_evaluation.model_dump(
                    mode="json"
                ),
                "current_evaluation_after": step.current_evaluation_after.model_dump(
                    mode="json"
                ),
                "best_evaluation_after": step.best_evaluation_after.model_dump(
                    mode="json"
                ),
                "accepted": step.accepted,
                "created_new_best": step.created_new_best,
                "temperature": step.temperature,
                "stagnation_before": step.context_before.stagnation_count,
                "stagnation_after": step.context_after.stagnation_count,
            }
        )

    shadow_projection = {
        "steps": steps,
        "final_rng_state_hash": stable_hash(rng.bit_generator.state),
        "initial_hash": stable_hash(result.initial_path),
        "final_hash": stable_hash(result.final_path),
        "best_hash": stable_hash(result.best_path),
    }
    assert stable_hash(shadow_projection) == LEGACY_SHADOW_HASH

