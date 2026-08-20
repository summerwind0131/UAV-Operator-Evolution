from __future__ import annotations

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment.generator import MapGenerator
from uav_operator_evolution.experiments.run_baselines import run_baselines_workflow
from uav_operator_evolution.runtime import RunPaths


def test_six_baseline_arms_execute_with_shared_budget(tmp_path) -> None:
    config = load_config("configs/smoke.yaml").model_copy(deep=True)
    config.search.train_iterations = 4
    config.search.validation_iterations = 3
    config.search.test_iterations = 3
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    config.output.results_dir = tmp_path / "results"
    config.output.figures_dir = tmp_path / "figures"
    generator = MapGenerator(config.seed, grid_resolution=5, generation_attempts=20)
    datasets = {
        "train": [generator.generate_map("train", 0, "sparse", width=45, height=45, safety_distance=1)],
        "validation": [generator.generate_map("validation", 0, "medium", width=45, height=45, safety_distance=1)],
        "test": [generator.generate_map("test", 0, "dense", width=50, height=50, safety_distance=1)],
    }
    paths = RunPaths.create(config, "baseline", run_id="tiny", run_dir=tmp_path / "run")
    report = run_baselines_workflow(config, paths, datasets)
    assert len(report["arms"]) == 6
    assert {row["arm"] for row in report["paired_rows"]} == {
        "initial_manual_operators",
        "random_composite",
        "score_only_composite",
        "diagnosis_memory_composite",
        "parent_operator",
        "best_evolved_operator",
    }

