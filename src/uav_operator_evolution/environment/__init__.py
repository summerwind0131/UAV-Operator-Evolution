"""Continuous UAV path-planning environments and map generation."""

from .environment import (
    Difficulty,
    Environment2D,
    EnvironmentFeatures,
    LayoutSubtype,
    extract_environment_features,
)
from .geometry import (
    euclidean_distance,
    path_length,
    point_obstacle_clearance,
    point_to_segment_distance,
    segment_intersects_obstacle,
    segment_obstacle_clearance,
    segments_intersect,
    turn_angles,
)
from .generator import (
    DatasetManifest,
    MapGenerator,
    MapManifestEntry,
    generate_dataset,
    load_dataset,
    load_dataset_split,
)
from .obstacles import CircleObstacle, Obstacle, Point, RectangleObstacle, RiskZone

__all__ = [
    "CircleObstacle",
    "Difficulty",
    "DatasetManifest",
    "Environment2D",
    "EnvironmentFeatures",
    "LayoutSubtype",
    "Obstacle",
    "MapGenerator",
    "MapManifestEntry",
    "Point",
    "RectangleObstacle",
    "RiskZone",
    "euclidean_distance",
    "extract_environment_features",
    "generate_dataset",
    "load_dataset",
    "load_dataset_split",
    "path_length",
    "point_obstacle_clearance",
    "point_to_segment_distance",
    "segment_intersects_obstacle",
    "segment_obstacle_clearance",
    "segments_intersect",
    "turn_angles",
]
