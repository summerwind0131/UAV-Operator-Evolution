"""Tests for reproducible maps, A* initialization, manifests, and path figures."""

from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment import (
    DatasetManifest,
    Environment2D,
    MapGenerator,
    MapManifestEntry,
    extract_environment_features,
    generate_dataset,
    load_dataset,
    load_dataset_split,
)
from uav_operator_evolution.path import PathEvaluator, initialize_path_astar
from uav_operator_evolution.reproducibility import stable_hash
from uav_operator_evolution.visualization.paths import plot_path, plot_path_comparison


DIFFICULTIES = [
    "sparse",
    "medium",
    "dense",
    "corridor",
    "clustered",
    "rooms_maze",
    "mixed",
]


@pytest.mark.parametrize("difficulty", DIFFICULTIES)
def test_all_map_types_are_reproducible_and_navigable(difficulty: str) -> None:
    generator = MapGenerator(1234, grid_resolution=4.0, generation_attempts=20)
    index = DIFFICULTIES.index(difficulty)
    first = generator.generate_map("train", index, difficulty)  # type: ignore[arg-type]
    second = generator.generate_map("train", index, difficulty)  # type: ignore[arg-type]
    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert first.difficulty == difficulty
    path = initialize_path_astar(first, grid_resolution=4.0)
    assert path[0] == first.start
    assert path[-1] == first.goal
    assert first.path_is_collision_free(path)
    assert PathEvaluator().evaluate(path, first).feasible
    features = extract_environment_features(first)
    assert features.obstacle_count == len(first.obstacles)
    assert 0.0 <= features.obstacle_density <= 1.0


def test_environment_json_round_trip(tmp_path: Path) -> None:
    environment = MapGenerator(8).generate_map("validation", 0, "mixed")
    destination = tmp_path / "map.json"
    environment.save_json(destination)
    restored = Environment2D.load_json(destination)
    assert restored == environment
    assert restored.content_hash == environment.content_hash


def test_dataset_manifest_separates_splits_and_verifies_hashes(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml")
    maps = config.maps.model_copy(
        update={
            "train": config.maps.train.model_copy(update={"count": 2}),
            "validation": config.maps.validation.model_copy(update={"count": 2}),
            "test": config.maps.test.model_copy(update={"count": 2}),
        }
    )
    small_config = config.model_copy(update={"maps": maps})
    manifest = generate_dataset(small_config, tmp_path)
    loaded = load_dataset(tmp_path)
    assert len(manifest.maps) == 6
    assert {split: len(environments) for split, environments in loaded.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    ids = [entry.map_id for entry in manifest.maps]
    hashes = [entry.content_hash for entry in manifest.maps]
    assert len(ids) == len(set(ids))
    assert len(hashes) == len(set(hashes))
    for field_name in ("terminal_hash", "obstacle_layout_hash", "geometry_hash"):
        semantic_hashes = [getattr(entry, field_name) for entry in manifest.maps]
        assert all(semantic_hashes)
        assert len(semantic_hashes) == len(set(semantic_hashes))
    repeated = generate_dataset(small_config, tmp_path)
    assert repeated == manifest


def test_split_loader_does_not_open_other_split_map_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config("configs/smoke.yaml")
    maps = config.maps.model_copy(
        update={
            "train": config.maps.train.model_copy(update={"count": 2}),
            "validation": config.maps.validation.model_copy(update={"count": 2}),
            "test": config.maps.test.model_copy(update={"count": 2}),
        }
    )
    generate_dataset(config.model_copy(update={"maps": maps}), tmp_path)
    opened: list[Path] = []
    original = Environment2D.load_json

    def tracked(path: str | Path) -> Environment2D:
        opened.append(Path(path))
        return original(path)

    monkeypatch.setattr(Environment2D, "load_json", staticmethod(tracked))
    loaded = load_dataset_split(tmp_path, "validation")
    assert len(loaded) == 2
    assert len(opened) == 2
    assert all("validation" in path.parts for path in opened)


def test_v2_manifest_hashes_remain_loadable(tmp_path: Path) -> None:
    environment = MapGenerator(9).generate_map("train", 0, "medium")
    legacy_payload = environment.model_dump(mode="json")
    legacy_payload.pop("layout_subtype", None)
    relative_path = Path("train") / "legacy.json"
    destination = tmp_path / relative_path
    destination.parent.mkdir(parents=True)
    destination.write_text(json.dumps(legacy_payload), encoding="utf-8")
    manifest = DatasetManifest(
        master_seed=9,
        config_hash="legacy",
        generator_version="2",
        maps=[
            MapManifestEntry(
                map_id=environment.map_id,
                split="train",
                difficulty="medium",
                seed=environment.seed,
                relative_path=relative_path.as_posix(),
                content_hash=stable_hash(legacy_payload),
            )
        ],
    )
    (tmp_path / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    loaded = load_dataset(tmp_path)
    assert loaded["train"][0].map_id == environment.map_id


def test_benchmark_split_balances_rooms_and_maze_with_safe_random_terminals() -> None:
    config = load_config("configs/uav_benchmark_v1.yaml")
    split = config.maps.test.model_copy(update={"count": 12})
    maps = MapGenerator(
        config.seed,
        grid_resolution=config.maps.grid_resolution,
        generation_attempts=config.maps.generation_attempts,
    ).generate_split("test", split)
    assert Counter(environment.difficulty for environment in maps) == {
        "sparse": 2,
        "dense": 2,
        "corridor": 2,
        "clustered": 2,
        "rooms_maze": 2,
        "mixed": 2,
    }
    rooms_maze = [
        environment
        for environment in maps
        if environment.difficulty == "rooms_maze"
    ]
    assert Counter(environment.layout_subtype for environment in rooms_maze) == {
        "rooms": 1,
        "maze": 1,
    }
    assert all(
        math.dist(environment.start, environment.goal)
        >= 0.65 * environment.diagonal
        for environment in maps
    )
    assert all(
        environment.point_is_collision_free(environment.start)
        and environment.point_is_collision_free(environment.goal)
        for environment in maps
    )


def test_corridor_requires_detour_and_line_of_sight_simplification() -> None:
    environment = MapGenerator(99, grid_resolution=4.0).generate_map("test", 0, "corridor")
    assert not environment.segment_is_collision_free(environment.start, environment.goal)
    path = initialize_path_astar(environment, grid_resolution=4.0)
    assert len(path) >= 3
    assert environment.path_is_collision_free(path)


def test_path_figures_are_saved(tmp_path: Path) -> None:
    environment = MapGenerator(17).generate_map("train", 0, "medium")
    initial = [environment.start, environment.goal]
    planned = initialize_path_astar(environment)
    evaluator = PathEvaluator()
    output = tmp_path / "path.png"
    comparison = tmp_path / "comparison.png"
    figure = plot_path(
        environment,
        planned,
        initial_path=initial,
        evaluation=evaluator.evaluate(planned, environment),
        output_path=output,
    )
    comparison_figure = plot_path_comparison(
        environment,
        initial,
        planned,
        before_evaluation=evaluator.evaluate(initial, environment),
        after_evaluation=evaluator.evaluate(planned, environment),
        output_path=comparison,
    )
    assert output.stat().st_size > 0
    assert comparison.stat().st_size > 0
    plt.close(figure)
    plt.close(comparison_figure)
