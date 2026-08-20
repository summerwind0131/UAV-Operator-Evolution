"""Serializable continuous two-dimensional planning environments."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..reproducibility import stable_hash
from .geometry import (
    euclidean_distance,
    point_obstacle_clearance,
    segment_obstacle_clearance,
    segment_risk_exposure,
)
from .obstacles import CircleObstacle, Obstacle, Point, RectangleObstacle, RiskZone

Difficulty = Literal[
    "sparse",
    "medium",
    "dense",
    "corridor",
    "clustered",
    "rooms_maze",
    "mixed",
]
LayoutSubtype = Literal["rooms", "maze"]


class Environment2D(BaseModel):
    """A bounded continuous map with static obstacles and optional risk zones."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    map_id: str = "map"
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    start: Point
    goal: Point
    obstacles: list[Obstacle] = Field(default_factory=list)
    risk_zones: list[RiskZone] = Field(default_factory=list)
    safety_distance: float = Field(default=0.0, ge=0)
    difficulty: Difficulty = "medium"
    layout_subtype: LayoutSubtype | None = None
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_map_geometry(self) -> "Environment2D":
        values = (*self.start, *self.goal, self.width, self.height, self.safety_distance)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("map coordinates and dimensions must be finite")
        if not self.in_bounds(self.start) or not self.in_bounds(self.goal):
            raise ValueError("start and goal must lie inside map bounds")
        if euclidean_distance(self.start, self.goal) <= 1e-9:
            raise ValueError("start and goal must be distinct")
        for obstacle in self.obstacles:
            if isinstance(obstacle, CircleObstacle):
                if (
                    obstacle.center[0] - obstacle.radius < -1e-9
                    or obstacle.center[1] - obstacle.radius < -1e-9
                    or obstacle.center[0] + obstacle.radius > self.width + 1e-9
                    or obstacle.center[1] + obstacle.radius > self.height + 1e-9
                ):
                    raise ValueError("circle obstacle must lie inside map bounds")
            elif (
                obstacle.min_x < -1e-9
                or obstacle.min_y < -1e-9
                or obstacle.max_x > self.width + 1e-9
                or obstacle.max_y > self.height + 1e-9
            ):
                raise ValueError("rectangle obstacle must lie inside map bounds")
        for zone in self.risk_zones:
            if (
                zone.min_x < -1e-9
                or zone.min_y < -1e-9
                or zone.max_x > self.width + 1e-9
                or zone.max_y > self.height + 1e-9
            ):
                raise ValueError("risk zone must lie inside map bounds")
        return self

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def diagonal(self) -> float:
        return math.hypot(self.width, self.height)

    @property
    def content_hash(self) -> str:
        """Return a deterministic hash of the complete map payload."""

        return stable_hash(self.model_dump(mode="json"))

    @property
    def terminal_hash(self) -> str:
        """Hash only the semantic terminal placement, independent of map identity."""

        return stable_hash(
            {
                "width": self.width,
                "height": self.height,
                "start": self.start,
                "goal": self.goal,
            }
        )

    @property
    def obstacle_layout_hash(self) -> str:
        """Hash obstacle and risk geometry without terminals or identity fields."""

        return stable_hash(
            {
                "width": self.width,
                "height": self.height,
                "obstacles": [
                    obstacle.model_dump(mode="json") for obstacle in self.obstacles
                ],
                "risk_zones": [
                    zone.model_dump(mode="json") for zone in self.risk_zones
                ],
                "safety_distance": self.safety_distance,
                "difficulty": self.difficulty,
                "layout_subtype": self.layout_subtype,
            }
        )

    @property
    def geometry_hash(self) -> str:
        """Hash complete planning geometry without map id or random seed."""

        return stable_hash(
            {
                "width": self.width,
                "height": self.height,
                "start": self.start,
                "goal": self.goal,
                "obstacles": [
                    obstacle.model_dump(mode="json") for obstacle in self.obstacles
                ],
                "risk_zones": [
                    zone.model_dump(mode="json") for zone in self.risk_zones
                ],
                "safety_distance": self.safety_distance,
                "difficulty": self.difficulty,
                "layout_subtype": self.layout_subtype,
            }
        )

    def in_bounds(self, point: Point, margin: float = 0.0) -> bool:
        """Return whether a point lies within the map after applying a margin."""

        return (
            margin <= point[0] <= self.width - margin
            and margin <= point[1] <= self.height - margin
        )

    def point_clearance(self, point: Point) -> float:
        """Return minimum signed obstacle clearance at a point."""

        if not self.obstacles:
            return self.diagonal
        return min(point_obstacle_clearance(point, obstacle) for obstacle in self.obstacles)

    def segment_clearance(self, start: Point, end: Point) -> float:
        """Return minimum signed obstacle clearance along a segment."""

        if not self.obstacles:
            return self.diagonal
        return min(segment_obstacle_clearance(start, end, obstacle) for obstacle in self.obstacles)

    def point_is_collision_free(self, point: Point, clearance: float | None = None) -> bool:
        """Return whether a point is in bounds and respects obstacle clearance."""

        required = self.safety_distance if clearance is None else clearance
        return self.in_bounds(point) and self.point_clearance(point) > required + 1e-9

    def segment_is_collision_free(
        self,
        start: Point,
        end: Point,
        clearance: float | None = None,
    ) -> bool:
        """Return whether a segment stays in bounds and respects obstacle clearance."""

        required = self.safety_distance if clearance is None else clearance
        return (
            self.in_bounds(start)
            and self.in_bounds(end)
            and self.segment_clearance(start, end) > required + 1e-9
        )

    def path_is_collision_free(
        self,
        path: list[Point],
        clearance: float | None = None,
    ) -> bool:
        """Return whether every path segment is collision-free."""

        return len(path) >= 2 and all(
            self.segment_is_collision_free(start, end, clearance)
            for start, end in zip(path, path[1:])
        )

    def colliding_segment_indices(
        self,
        path: list[Point],
        clearance: float | None = None,
    ) -> list[int]:
        """Return indices of path segments violating bounds or clearance."""

        return [
            index
            for index, (start, end) in enumerate(zip(path, path[1:]))
            if not self.segment_is_collision_free(start, end, clearance)
        ]

    def risk_exposure(self, path: list[Point]) -> float:
        """Return weighted path length inside all risk zones."""

        return sum(
            segment_risk_exposure(start, end, zone)
            for start, end in zip(path, path[1:])
            for zone in self.risk_zones
        )

    def save_json(self, path: str | Path) -> Path:
        """Serialize the environment as deterministic, human-readable JSON."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                self.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        return destination

    @classmethod
    def load_json(cls, path: str | Path) -> "Environment2D":
        """Load and validate an environment JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(payload)


class EnvironmentFeatures(BaseModel):
    """Deterministic, lightweight summary of a map."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    obstacle_count: int
    obstacle_density: float
    average_obstacle_size: float
    clustering_index: float
    direct_distance: float
    difficulty: Difficulty
    risk_area_ratio: float


def extract_environment_features(environment: Environment2D) -> EnvironmentFeatures:
    """Compute map features without sampling or global state."""

    obstacle_areas = [obstacle.area for obstacle in environment.obstacles]
    obstacle_density = min(1.0, sum(obstacle_areas) / environment.area)
    average_size = (
        sum(math.sqrt(area) for area in obstacle_areas) / len(obstacle_areas)
        if obstacle_areas
        else 0.0
    )
    centers = [obstacle.center for obstacle in environment.obstacles]
    if len(centers) < 2:
        clustering = 0.0
    else:
        nearest = [
            min(
                euclidean_distance(center, other)
                for other_index, other in enumerate(centers)
                if other_index != center_index
            )
            for center_index, center in enumerate(centers)
        ]
        clustering = max(0.0, min(1.0, 1.0 - sum(nearest) / len(nearest) / environment.diagonal))
    risk_ratio = min(1.0, sum(zone.area for zone in environment.risk_zones) / environment.area)
    return EnvironmentFeatures(
        obstacle_count=len(environment.obstacles),
        obstacle_density=obstacle_density,
        average_obstacle_size=average_size,
        clustering_index=clustering,
        direct_distance=euclidean_distance(environment.start, environment.goal),
        difficulty=environment.difficulty,
        risk_area_ratio=risk_ratio,
    )


__all__ = [
    "Difficulty",
    "Environment2D",
    "EnvironmentFeatures",
    "LayoutSubtype",
    "extract_environment_features",
]
