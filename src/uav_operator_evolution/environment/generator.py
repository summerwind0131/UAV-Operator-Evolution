"""Reproducible map generation and dataset serialization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..reproducibility import derive_seed, stable_hash
from .environment import Difficulty, Environment2D, LayoutSubtype
from .geometry import point_obstacle_clearance
from .obstacles import CircleObstacle, Obstacle, RectangleObstacle, RiskZone

if TYPE_CHECKING:
    from ..config import ExperimentConfig, MapSplitConfig

SplitName = Literal["train", "validation", "test"]
GENERATOR_VERSION = "3"


class MapManifestEntry(BaseModel):
    """One deterministic map entry in a generated dataset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    map_id: str
    split: SplitName
    difficulty: Difficulty
    seed: int
    relative_path: str
    content_hash: str
    layout_subtype: LayoutSubtype | None = None
    terminal_hash: str | None = None
    obstacle_layout_hash: str | None = None
    geometry_hash: str | None = None


class DatasetManifest(BaseModel):
    """A complete manifest for train, validation, and test maps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    master_seed: int = Field(ge=0)
    config_hash: str
    benchmark_id: str | None = None
    generator_version: str = GENERATOR_VERSION
    maps: list[MapManifestEntry]

    @property
    def manifest_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))

    def entries_for(self, split: SplitName) -> list[MapManifestEntry]:
        return [entry for entry in self.maps if entry.split == split]


class MapGenerator:
    """Generate connected continuous maps from independent semantic seeds."""

    def __init__(
        self,
        master_seed: int,
        *,
        grid_resolution: float = 4.0,
        generation_attempts: int = 30,
    ) -> None:
        if master_seed < 0:
            raise ValueError("master_seed must be non-negative")
        if grid_resolution <= 0:
            raise ValueError("grid_resolution must be positive")
        if generation_attempts < 1:
            raise ValueError("generation_attempts must be positive")
        self.master_seed = master_seed
        self.grid_resolution = float(grid_resolution)
        self.generation_attempts = generation_attempts

    def generate_map(
        self,
        split: SplitName,
        index: int,
        difficulty: Difficulty,
        *,
        width: float = 100.0,
        height: float = 100.0,
        safety_distance: float = 2.0,
        layout_subtype: LayoutSubtype | None = None,
    ) -> Environment2D:
        """Generate one connected map without consuming shared random state."""

        if index < 0:
            raise ValueError("index must be non-negative")
        if width <= 10 or height <= 10:
            raise ValueError("map dimensions must exceed 10")
        for attempt in range(self.generation_attempts):
            seed = derive_seed(
                self.master_seed,
                "map",
                split,
                index,
                difficulty,
                layout_subtype or "default",
                attempt,
            )
            rng = np.random.default_rng(seed)
            environment = self._build_candidate(
                split,
                index,
                difficulty,
                width,
                height,
                safety_distance,
                seed,
                rng,
                layout_subtype,
            )
            if not environment.point_is_collision_free(environment.start):
                continue
            if not environment.point_is_collision_free(environment.goal):
                continue
            if self._is_connected(environment):
                return environment
        raise RuntimeError(
            f"unable to generate connected {difficulty!r} map after "
            f"{self.generation_attempts} attempts"
        )

    def generate_split(
        self,
        split: SplitName,
        config: "MapSplitConfig",
    ) -> list[Environment2D]:
        """Generate a balanced, deterministically ordered map split."""

        if not config.difficulties:
            raise ValueError("map split must contain at least one difficulty")
        maps: list[Environment2D] = []
        occurrence: dict[Difficulty, int] = {}
        for index in range(config.count):
            difficulty = config.difficulties[index % len(config.difficulties)]
            occurrence_index = occurrence.get(difficulty, 0)
            occurrence[difficulty] = occurrence_index + 1
            layout_subtype: LayoutSubtype | None = None
            if difficulty == "rooms_maze":
                layout_subtype = "rooms" if occurrence_index % 2 == 0 else "maze"
            maps.append(
                self.generate_map(
                    split,
                    index,
                    difficulty,
                    width=config.width,
                    height=config.height,
                    safety_distance=config.safety_distance,
                    layout_subtype=layout_subtype,
                )
            )
        return maps

    def _build_candidate(
        self,
        split: SplitName,
        index: int,
        difficulty: Difficulty,
        width: float,
        height: float,
        safety_distance: float,
        seed: int,
        rng: np.random.Generator,
        layout_subtype: LayoutSubtype | None,
    ) -> Environment2D:
        if difficulty == "corridor":
            obstacles = self._corridor_obstacles(width, height, safety_distance, rng)
            start, goal = self._sample_terminals(
                width,
                height,
                safety_distance,
                obstacles,
                rng,
                corridor=True,
            )
            risks = [
                RiskZone(
                    min_x=0.46 * width,
                    min_y=0.43 * height,
                    max_x=0.62 * width,
                    max_y=0.57 * height,
                    weight=1.25,
                    name="corridor_crosswind",
                )
            ]
        elif difficulty == "rooms_maze":
            subtype = layout_subtype or "rooms"
            layout_subtype = subtype
            obstacles = self._rooms_maze_obstacles(
                subtype,
                width,
                height,
                safety_distance,
                rng,
            )
            start, goal = self._sample_terminals(
                width,
                height,
                safety_distance,
                obstacles,
                rng,
            )
            risks = self._risk_zones(difficulty, width, height, rng)
        else:
            start, goal = self._sample_terminals(
                width,
                height,
                safety_distance,
                [],
                rng,
            )
            obstacles = self._random_obstacles(
                difficulty,
                width,
                height,
                safety_distance,
                start,
                goal,
                rng,
            )
            risks = self._risk_zones(difficulty, width, height, rng)
        map_stub = f"{split}-{index:03d}-{difficulty}"
        environment = Environment2D(
            map_id=map_stub,
            width=width,
            height=height,
            start=start,
            goal=goal,
            obstacles=obstacles,
            risk_zones=risks,
            safety_distance=safety_distance,
            difficulty=difficulty,
            layout_subtype=layout_subtype,
            seed=seed,
        )
        return environment.model_copy(
            update={"map_id": f"{map_stub}-{environment.geometry_hash[:10]}"}
        )

    def _sample_terminals(
        self,
        width: float,
        height: float,
        safety_distance: float,
        obstacles: list[Obstacle],
        rng: np.random.Generator,
        *,
        corridor: bool = False,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Sample safe, well-separated terminals from opposite map regions."""

        terminal_clearance = safety_distance + self.grid_resolution
        minimum_distance = 0.65 * math.hypot(width, height)
        for _ in range(500):
            reverse = bool(rng.integers(0, 2))
            if corridor:
                left = (
                    float(rng.uniform(0.04, 0.10) * width),
                    float(rng.uniform(0.06, 0.20) * height),
                )
                right = (
                    float(rng.uniform(0.90, 0.96) * width),
                    float(rng.uniform(0.80, 0.94) * height),
                )
            else:
                left = (
                    float(rng.uniform(0.05, 0.15) * width),
                    float(rng.uniform(0.05, 0.15) * height),
                )
                right = (
                    float(rng.uniform(0.85, 0.95) * width),
                    float(rng.uniform(0.85, 0.95) * height),
                )
            start, goal = (right, left) if reverse else (left, right)
            if math.dist(start, goal) < minimum_distance:
                continue
            if all(
                point_obstacle_clearance(start, obstacle) > terminal_clearance
                and point_obstacle_clearance(goal, obstacle) > terminal_clearance
                for obstacle in obstacles
            ):
                return start, goal
        raise RuntimeError("failed to sample safe, well-separated terminals")

    def _random_obstacles(
        self,
        difficulty: Difficulty,
        width: float,
        height: float,
        safety_distance: float,
        start: tuple[float, float],
        goal: tuple[float, float],
        rng: np.random.Generator,
    ) -> list[Obstacle]:
        counts = {
            "sparse": (3, 5),
            "medium": (6, 9),
            "dense": (11, 15),
            "clustered": (9, 13),
            "mixed": (7, 11),
        }
        lower, upper = counts[difficulty]
        target = int(rng.integers(lower, upper + 1))
        obstacles: list[Obstacle] = []
        scale = min(width, height)
        clusters: list[tuple[float, float]] = []
        if difficulty == "clustered":
            clusters = [
                (float(rng.uniform(0.28, 0.45) * width), float(rng.uniform(0.35, 0.7) * height)),
                (float(rng.uniform(0.58, 0.75) * width), float(rng.uniform(0.3, 0.65) * height)),
            ]
        attempts = 0
        while len(obstacles) < target and attempts < target * 30:
            attempts += 1
            if clusters:
                cluster = clusters[len(obstacles) % len(clusters)]
                center_x = float(np.clip(rng.normal(cluster[0], 0.09 * width), 0.1 * width, 0.9 * width))
                center_y = float(np.clip(rng.normal(cluster[1], 0.09 * height), 0.1 * height, 0.9 * height))
            else:
                center_x = float(rng.uniform(0.1, 0.9) * width)
                center_y = float(rng.uniform(0.1, 0.9) * height)
            use_circle = difficulty != "mixed" or len(obstacles) % 2 == 0
            if use_circle:
                radius = float(rng.uniform(0.025, 0.052 if difficulty != "dense" else 0.045) * scale)
                center_x = float(np.clip(center_x, radius, width - radius))
                center_y = float(np.clip(center_y, radius, height - radius))
                candidate: Obstacle = CircleObstacle(center=(center_x, center_y), radius=radius)
            else:
                obstacle_width = float(rng.uniform(0.045, 0.10) * width)
                obstacle_height = float(rng.uniform(0.04, 0.10) * height)
                min_x = float(np.clip(center_x - obstacle_width / 2, 0.0, width - obstacle_width))
                min_y = float(np.clip(center_y - obstacle_height / 2, 0.0, height - obstacle_height))
                candidate = RectangleObstacle(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=min_x + obstacle_width,
                    max_y=min_y + obstacle_height,
                )
            terminal_buffer = safety_distance + self.grid_resolution
            if point_obstacle_clearance(start, candidate) <= terminal_buffer:
                continue
            if point_obstacle_clearance(goal, candidate) <= terminal_buffer:
                continue
            obstacles.append(candidate)
        if len(obstacles) < target:
            raise RuntimeError("failed to place requested number of obstacles")
        return obstacles

    def _rooms_maze_obstacles(
        self,
        layout_subtype: LayoutSubtype,
        width: float,
        height: float,
        safety_distance: float,
        rng: np.random.Generator,
    ) -> list[Obstacle]:
        """Create room walls or a deterministic recursive-division maze."""

        thickness = max(0.012 * min(width, height), 0.5 * self.grid_resolution)
        door_width = max(
            0.14 * min(width, height),
            2.0 * safety_distance + 3.0 * self.grid_resolution,
        )
        obstacles: list[Obstacle] = []

        def vertical_wall(
            x: float,
            low: float,
            high: float,
            door_center: float,
        ) -> None:
            door_low = max(low, door_center - door_width / 2.0)
            door_high = min(high, door_center + door_width / 2.0)
            if door_low - low > 1e-6:
                obstacles.append(
                    RectangleObstacle(
                        min_x=x - thickness / 2.0,
                        min_y=low,
                        max_x=x + thickness / 2.0,
                        max_y=door_low,
                    )
                )
            if high - door_high > 1e-6:
                obstacles.append(
                    RectangleObstacle(
                        min_x=x - thickness / 2.0,
                        min_y=door_high,
                        max_x=x + thickness / 2.0,
                        max_y=high,
                    )
                )

        def horizontal_wall(
            y: float,
            low: float,
            high: float,
            door_center: float,
        ) -> None:
            door_low = max(low, door_center - door_width / 2.0)
            door_high = min(high, door_center + door_width / 2.0)
            if door_low - low > 1e-6:
                obstacles.append(
                    RectangleObstacle(
                        min_x=low,
                        min_y=y - thickness / 2.0,
                        max_x=door_low,
                        max_y=y + thickness / 2.0,
                    )
                )
            if high - door_high > 1e-6:
                obstacles.append(
                    RectangleObstacle(
                        min_x=door_high,
                        min_y=y - thickness / 2.0,
                        max_x=high,
                        max_y=y + thickness / 2.0,
                    )
                )

        if layout_subtype == "rooms":
            for fraction in (0.34, 0.66):
                vertical_wall(
                    fraction * width,
                    0.0,
                    height,
                    float(rng.uniform(0.22, 0.78) * height),
                )
            for fraction in (0.36, 0.64):
                horizontal_wall(
                    fraction * height,
                    0.0,
                    width,
                    float(rng.uniform(0.22, 0.78) * width),
                )
            return obstacles

        min_span = max(2.5 * door_width, 0.25 * min(width, height))

        def divide(
            min_x: float,
            min_y: float,
            max_x: float,
            max_y: float,
            depth: int,
        ) -> None:
            span_x = max_x - min_x
            span_y = max_y - min_y
            if depth <= 0 or max(span_x, span_y) < min_span:
                return
            vertical = span_x >= span_y
            if vertical and span_x >= min_span:
                wall_x = float(rng.uniform(min_x + 0.38 * span_x, min_x + 0.62 * span_x))
                door_center = float(
                    rng.uniform(min_y + 0.22 * span_y, min_y + 0.78 * span_y)
                )
                vertical_wall(wall_x, min_y, max_y, door_center)
                divide(min_x, min_y, wall_x, max_y, depth - 1)
                divide(wall_x, min_y, max_x, max_y, depth - 1)
            elif span_y >= min_span:
                wall_y = float(rng.uniform(min_y + 0.38 * span_y, min_y + 0.62 * span_y))
                door_center = float(
                    rng.uniform(min_x + 0.22 * span_x, min_x + 0.78 * span_x)
                )
                horizontal_wall(wall_y, min_x, max_x, door_center)
                divide(min_x, min_y, max_x, wall_y, depth - 1)
                divide(min_x, wall_y, max_x, max_y, depth - 1)

        divide(0.0, 0.0, width, height, 3)
        return obstacles

    def _corridor_obstacles(
        self,
        width: float,
        height: float,
        safety_distance: float,
        rng: np.random.Generator,
    ) -> list[Obstacle]:
        gap_height = max(0.17 * height, 2 * safety_distance + 2.5 * self.grid_resolution)
        wall_width = max(0.055 * width, self.grid_resolution)
        first_center = float(rng.uniform(0.30, 0.37) * height)
        second_center = float(rng.uniform(0.63, 0.70) * height)
        obstacles: list[Obstacle] = []
        for x_center, gap_center in ((0.42 * width, first_center), (0.66 * width, second_center)):
            gap_low = max(0.05 * height, gap_center - gap_height / 2)
            gap_high = min(0.95 * height, gap_center + gap_height / 2)
            obstacles.extend(
                [
                    RectangleObstacle(
                        min_x=x_center - wall_width / 2,
                        min_y=0.0,
                        max_x=x_center + wall_width / 2,
                        max_y=gap_low,
                    ),
                    RectangleObstacle(
                        min_x=x_center - wall_width / 2,
                        min_y=gap_high,
                        max_x=x_center + wall_width / 2,
                        max_y=height,
                    ),
                ]
            )
        return obstacles

    @staticmethod
    def _risk_zones(
        difficulty: Difficulty,
        width: float,
        height: float,
        rng: np.random.Generator,
    ) -> list[RiskZone]:
        count = 2 if difficulty == "mixed" else int(rng.random() < 0.45)
        zones: list[RiskZone] = []
        for zone_index in range(count):
            zone_width = float(rng.uniform(0.12, 0.22) * width)
            zone_height = float(rng.uniform(0.10, 0.20) * height)
            min_x = float(rng.uniform(0.15 * width, 0.85 * width - zone_width))
            min_y = float(rng.uniform(0.15 * height, 0.85 * height - zone_height))
            zones.append(
                RiskZone(
                    min_x=min_x,
                    min_y=min_y,
                    max_x=min_x + zone_width,
                    max_y=min_y + zone_height,
                    weight=float(rng.uniform(0.75, 1.5)),
                    name=f"risk_{zone_index}",
                )
            )
        return zones

    def _is_connected(self, environment: Environment2D) -> bool:
        # Local import avoids making the environment data model depend on path search.
        from ..path.initializer import PathInitializationError, initialize_path_astar

        try:
            initialize_path_astar(
                environment,
                grid_resolution=self.grid_resolution,
                max_nodes=max(20_000, math.ceil(environment.area / self.grid_resolution**2) * 8),
            )
        except PathInitializationError:
            return False
        return True


def generate_dataset(
    config: "ExperimentConfig",
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> DatasetManifest:
    """Generate all configured splits and write a content-addressed manifest."""

    root = Path(output_dir) if output_dir is not None else config.output.data_dir
    generator = MapGenerator(
        config.seed,
        grid_resolution=config.maps.grid_resolution,
        generation_attempts=config.maps.generation_attempts,
    )
    entries: list[MapManifestEntry] = []
    for split in ("train", "validation", "test"):
        split_config = getattr(config.maps, split)
        maps = generator.generate_split(split, split_config)
        for environment in maps:
            relative_path = Path(split) / f"{environment.map_id}.json"
            destination = root / relative_path
            if destination.exists() and not overwrite:
                existing = Environment2D.load_json(destination)
                if existing.content_hash != environment.content_hash:
                    raise FileExistsError(
                        f"generated map differs from existing file: {destination}; use overwrite=True"
                    )
            else:
                environment.save_json(destination)
            entries.append(
                MapManifestEntry(
                    map_id=environment.map_id,
                    split=split,
                    difficulty=environment.difficulty,
                    seed=environment.seed,
                    relative_path=relative_path.as_posix(),
                    content_hash=environment.content_hash,
                    layout_subtype=environment.layout_subtype,
                    terminal_hash=environment.terminal_hash,
                    obstacle_layout_hash=environment.obstacle_layout_hash,
                    geometry_hash=environment.geometry_hash,
                )
            )
    hash_fields = ("terminal_hash", "obstacle_layout_hash", "geometry_hash")
    for field_name in hash_fields:
        seen: dict[str, SplitName] = {}
        for entry in entries:
            value = getattr(entry, field_name)
            if value is None:
                continue
            if value in seen:
                raise ValueError(
                    f"dataset contains duplicate {field_name}: "
                    f"{seen[value]} and {entry.split}"
                )
            seen[value] = entry.split
    manifest = DatasetManifest(
        master_seed=config.seed,
        config_hash=config.config_hash,
        benchmark_id=config.planning_benchmark.benchmark_id,
        maps=entries,
    )
    manifest_path = root / "manifest.json"
    payload = json.dumps(
        manifest.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if manifest_path.exists() and not overwrite:
        existing_payload = manifest_path.read_text(encoding="utf-8")
        if existing_payload != payload:
            raise FileExistsError(f"dataset manifest already exists and differs: {manifest_path}")
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(payload, encoding="utf-8")
    return manifest


def load_dataset(path: str | Path) -> dict[SplitName, list[Environment2D]]:
    """Load all maps named by a dataset manifest and verify content hashes."""

    source = Path(path)
    manifest_path = source / "manifest.json" if source.is_dir() else source
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    result: dict[SplitName, list[Environment2D]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    seen_semantic_hashes: dict[str, dict[str, SplitName]] = {
        "terminal_hash": {},
        "obstacle_layout_hash": {},
        "geometry_hash": {},
    }
    for entry in manifest.maps:
        environment = Environment2D.load_json(root / entry.relative_path)
        actual_content_hash = environment.content_hash
        if manifest.generator_version == "2":
            legacy_payload = environment.model_dump(mode="json")
            legacy_payload.pop("layout_subtype", None)
            actual_content_hash = stable_hash(legacy_payload)
        if environment.map_id != entry.map_id or actual_content_hash != entry.content_hash:
            raise ValueError(f"map does not match manifest entry: {entry.relative_path}")
        for field_name in seen_semantic_hashes:
            expected = getattr(entry, field_name)
            actual = getattr(environment, field_name)
            if expected is not None and actual != expected:
                raise ValueError(
                    f"map {field_name} does not match manifest: {entry.relative_path}"
                )
            if expected is not None:
                previous_split = seen_semantic_hashes[field_name].get(expected)
                if previous_split is not None:
                    raise ValueError(
                        f"dataset contains duplicate {field_name} in "
                        f"{previous_split} and {entry.split}"
                    )
                seen_semantic_hashes[field_name][expected] = entry.split
        if entry.map_id in seen_ids or entry.content_hash in seen_hashes:
            raise ValueError("dataset contains duplicate map ids or contents")
        seen_ids.add(entry.map_id)
        seen_hashes.add(entry.content_hash)
        result[entry.split].append(environment)
    return result


def load_dataset_split(
    path: str | Path,
    split: SplitName,
) -> list[Environment2D]:
    """Load and verify one split without opening map files from other splits."""

    source = Path(path)
    manifest_path = source / "manifest.json" if source.is_dir() else source
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    semantic_hashes = (
        "terminal_hash",
        "obstacle_layout_hash",
        "geometry_hash",
    )
    seen_semantic: dict[str, set[str]] = {
        field_name: set() for field_name in semantic_hashes
    }
    for entry in manifest.maps:
        if entry.map_id in seen_ids or entry.content_hash in seen_hashes:
            raise ValueError("dataset manifest contains duplicate map ids or contents")
        seen_ids.add(entry.map_id)
        seen_hashes.add(entry.content_hash)
        for field_name in semantic_hashes:
            value = getattr(entry, field_name)
            if value is not None and value in seen_semantic[field_name]:
                raise ValueError(
                    f"dataset manifest contains duplicate {field_name}"
                )
            if value is not None:
                seen_semantic[field_name].add(value)

    environments: list[Environment2D] = []
    root = manifest_path.parent
    for entry in manifest.entries_for(split):
        environment = Environment2D.load_json(root / entry.relative_path)
        actual_content_hash = environment.content_hash
        if manifest.generator_version == "2":
            legacy_payload = environment.model_dump(mode="json")
            legacy_payload.pop("layout_subtype", None)
            actual_content_hash = stable_hash(legacy_payload)
        if environment.map_id != entry.map_id or actual_content_hash != entry.content_hash:
            raise ValueError(f"map does not match manifest entry: {entry.relative_path}")
        for field_name in semantic_hashes:
            expected = getattr(entry, field_name)
            if expected is not None and getattr(environment, field_name) != expected:
                raise ValueError(
                    f"map {field_name} does not match manifest: {entry.relative_path}"
                )
        environments.append(environment)
    return environments


__all__ = [
    "DatasetManifest",
    "GENERATOR_VERSION",
    "MapGenerator",
    "MapManifestEntry",
    "SplitName",
    "generate_dataset",
    "load_dataset",
    "load_dataset_split",
]
