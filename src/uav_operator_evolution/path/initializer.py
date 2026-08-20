"""Deterministic grid A* path initialization with continuous visibility checks."""

from __future__ import annotations

import heapq
import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from ..environment.environment import Environment2D
from ..environment.geometry import euclidean_distance
from .models import Path, Waypoint, copy_and_validate_path


class PathInitializationError(RuntimeError):
    """Raised when no safe initial path can be found within the fixed budget."""


def simplify_path_line_of_sight(
    path: Sequence[Sequence[float]],
    environment: Environment2D,
    clearance: float | None = None,
) -> Path:
    """Greedily remove waypoints while preserving continuous collision clearance."""

    canonical = copy_and_validate_path(path)
    if len(canonical) <= 2:
        return canonical
    simplified: Path = [canonical[0]]
    anchor = 0
    while anchor < len(canonical) - 1:
        candidate = len(canonical) - 1
        while candidate > anchor + 1 and not environment.segment_is_collision_free(
            canonical[anchor], canonical[candidate], clearance
        ):
            candidate -= 1
        simplified.append(canonical[candidate])
        anchor = candidate
    return simplified


def initialize_path_astar(
    environment: Environment2D,
    grid_resolution: float = 4.0,
    *,
    max_nodes: int = 250_000,
) -> Path:
    """Find a safe path using 8-connected grid A* and continuous edge checks."""

    if grid_resolution <= 0:
        raise ValueError("grid_resolution must be positive")
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    if not environment.point_is_collision_free(environment.start):
        raise PathInitializationError("environment start violates obstacle clearance")
    if not environment.point_is_collision_free(environment.goal):
        raise PathInitializationError("environment goal violates obstacle clearance")
    if environment.segment_is_collision_free(environment.start, environment.goal):
        return [environment.start, environment.goal]

    x_count = math.ceil(environment.width / grid_resolution) + 1
    y_count = math.ceil(environment.height / grid_resolution) + 1
    x_coordinates = [min(index * grid_resolution, environment.width) for index in range(x_count)]
    y_coordinates = [min(index * grid_resolution, environment.height) for index in range(y_count)]

    def point(node: tuple[int, int]) -> Waypoint:
        return (x_coordinates[node[0]], y_coordinates[node[1]])

    traversable: dict[tuple[int, int], bool] = {}

    def is_traversable(node: tuple[int, int]) -> bool:
        if node not in traversable:
            traversable[node] = environment.point_is_collision_free(point(node))
        return traversable[node]

    all_nodes = ((x_index, y_index) for x_index in range(x_count) for y_index in range(y_count))
    start_node = _nearest_visible_node(environment.start, all_nodes, point, is_traversable, environment)
    all_nodes = ((x_index, y_index) for x_index in range(x_count) for y_index in range(y_count))
    goal_node = _nearest_visible_node(environment.goal, all_nodes, point, is_traversable, environment)
    if start_node is None or goal_node is None:
        raise PathInitializationError("start or goal cannot connect to the planning grid")

    frontier: list[tuple[float, float, int, int]] = []
    heapq.heappush(
        frontier,
        (euclidean_distance(point(start_node), point(goal_node)), 0.0, *start_node),
    )
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost: dict[tuple[int, int], float] = {start_node: 0.0}
    closed: set[tuple[int, int]] = set()
    expanded = 0
    directions = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while frontier:
        _, current_cost, x_index, y_index = heapq.heappop(frontier)
        current = (x_index, y_index)
        if current in closed:
            continue
        closed.add(current)
        expanded += 1
        if expanded > max_nodes:
            raise PathInitializationError("A* node budget exhausted")
        if current == goal_node:
            grid_path = _reconstruct_grid_path(parent, current, point)
            full_path: Path = [environment.start]
            full_path.extend(node for node in grid_path if euclidean_distance(full_path[-1], node) > 1e-9)
            if euclidean_distance(full_path[-1], environment.goal) > 1e-9:
                full_path.append(environment.goal)
            simplified = simplify_path_line_of_sight(full_path, environment)
            if not environment.path_is_collision_free(simplified):
                raise PathInitializationError("A* produced an invalid continuous path")
            return simplified
        for dx, dy in directions:
            neighbor = (x_index + dx, y_index + dy)
            if not (0 <= neighbor[0] < x_count and 0 <= neighbor[1] < y_count):
                continue
            if not is_traversable(neighbor):
                continue
            if not environment.segment_is_collision_free(point(current), point(neighbor)):
                continue
            tentative = current_cost + euclidean_distance(point(current), point(neighbor))
            if tentative + 1e-12 >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = tentative
            parent[neighbor] = current
            estimate = tentative + euclidean_distance(point(neighbor), point(goal_node))
            heapq.heappush(frontier, (estimate, tentative, *neighbor))
    raise PathInitializationError("no collision-free path exists on the configured grid")


def _nearest_visible_node(
    endpoint: Waypoint,
    nodes: Iterable[tuple[int, int]],
    point: Callable[[tuple[int, int]], Waypoint],
    is_traversable: Callable[[tuple[int, int]], bool],
    environment: Environment2D,
) -> tuple[int, int] | None:
    node_list = list(nodes)
    node_list.sort(key=lambda node: (euclidean_distance(endpoint, point(node)), node))
    for node in node_list:
        if not is_traversable(node):
            continue
        if environment.segment_is_collision_free(endpoint, point(node)):
            return node
    return None


def _reconstruct_grid_path(
    parent: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
    point: Callable[[tuple[int, int]], Waypoint],
) -> Path:
    nodes = [current]
    while current in parent:
        current = parent[current]
        nodes.append(current)
    nodes.reverse()
    return [point(node) for node in nodes]


def initialize_path(
    environment: Environment2D,
    rng: np.random.Generator | None = None,
    grid_resolution: float = 4.0,
) -> Path:
    """Uniform initializer interface; grid A* itself intentionally uses no randomness."""

    del rng
    return initialize_path_astar(environment, grid_resolution=grid_resolution)


line_of_sight_simplify = simplify_path_line_of_sight


__all__ = [
    "PathInitializationError",
    "initialize_path",
    "initialize_path_astar",
    "line_of_sight_simplify",
    "simplify_path_line_of_sight",
]
