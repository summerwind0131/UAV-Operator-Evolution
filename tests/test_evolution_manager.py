from __future__ import annotations

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment.generator import MapGenerator
from uav_operator_evolution.evolution.manager import OperatorEvolutionManager


def test_tiny_evolution_closes_the_generate_compile_validate_loop(tmp_path) -> None:
    config = load_config("configs/smoke.yaml").model_copy(deep=True)
    config.search.train_iterations = 8
    config.search.validation_iterations = 6
    config.search.test_iterations = 6
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    generator = MapGenerator(config.seed, grid_resolution=5.0, generation_attempts=20)
    datasets = {
        "train": [generator.generate_map("train", 0, "sparse", width=50, height=50, safety_distance=1)],
        "validation": [generator.generate_map("validation", 0, "medium", width=50, height=50, safety_distance=1)],
        "test": [generator.generate_map("test", 0, "dense", width=55, height=55, safety_distance=1)],
    }
    manager = OperatorEvolutionManager(config, tmp_path / "experiment.sqlite")
    result = manager.run(datasets, "tiny")
    # Training plus both arms of validation and held-out test are all recorded.
    assert result.trace_count == 8 + (2 * 6) + (2 * 6)
    assert len(result.generations) == 1
    assert len(result.generations[0].proposals) == 1
    assert len(result.generations[0].validations) == 1
    assert len(result.final_population) == 8
