"""Pre-registered ablations and sensitivity arms for frozen Evolutionary AFL-UAV v1.

This module is intentionally separate from ``evolutionary_afl.py`` so that
experiments cannot mutate the hash-frozen v1 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..path.models import Path as UAVPath, Waypoint
from .core import BudgetedEvaluator, PlannerResult, PlanningBudget
from .evolutionary_afl import (
    EvolutionaryAFLUAVPlanner,
    _Individual,
    _deduplicate_consecutive,
    _path_key,
)

ExperimentVariant = Literal[
    "full_v1",
    "no_quality_diversity_archive",
    "no_crossover",
    "move_only",
    "no_rooms_maze_strategy",
    "fixed_length_population",
]


@dataclass
class EvolutionaryAFLExperimentPlanner(EvolutionaryAFLUAVPlanner):
    """One explicitly labeled explanatory experiment arm."""

    variant: ExperimentVariant = "full_v1"

    def __post_init__(self) -> None:
        if self.variant not in {
            "full_v1",
            "no_quality_diversity_archive",
            "no_crossover",
            "move_only",
            "no_rooms_maze_strategy",
            "fixed_length_population",
        }:
            raise ValueError(f"unknown Evolutionary AFL-UAV experiment variant: {self.variant}")
        if self.variant in {"no_crossover", "move_only"}:
            self.crossover_probability = 0.0
        super().__post_init__()

    @property
    def algorithm_parameters(self) -> dict[str, object]:
        parameters = dict(super().algorithm_parameters)
        parameters.update(
            {
                "experiment_variant": self.variant,
                "frozen_v1_core_modified": False,
                "is_ablation": self.variant != "full_v1",
            }
        )
        if self.variant == "no_quality_diversity_archive":
            parameters["archive"] = "weighted_cost_elitism_only"
        elif self.variant == "move_only":
            parameters["operators"] = ["move"]
        elif self.variant == "fixed_length_population":
            parameters["representation"] = "fixed_seed_waypoint_count"
            parameters["operators"] = ["move", "swap", "same_cut_crossover"]
        elif self.variant == "no_rooms_maze_strategy":
            parameters["difficulty_generation_caps"] = "disabled"
            parameters["rooms_maze_geometry_adaptation"] = "disabled"
        return parameters

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        # Sub-second sensitivity arms reserve a larger deterministic finalization
        # margin.  The trusted BudgetedEvaluator still enforces the advertised
        # outer budget, so this can only use less time, never more.
        cooperative_budget = budget
        if budget.time_limit_seconds <= 0.5:
            cooperative_budget = budget.model_copy(
                update={"time_limit_seconds": 0.9 * budget.time_limit_seconds}
            )
        result = super().plan(problem, cooperative_budget, rng)
        result.diagnostics["advertised_time_limit_seconds"] = budget.time_limit_seconds
        result.diagnostics["cooperative_time_limit_seconds"] = (
            cooperative_budget.time_limit_seconds
        )
        return result

    def _apply_operator(
        self,
        name: str,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if self.variant == "move_only" and name != "move":
            return list(path), False
        if self.variant == "fixed_length_population" and name in {"insert", "delete"}:
            return list(path), False
        return super()._apply_operator(name, path, problem, rng)

    def _crossover(
        self,
        first: UAVPath,
        second: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if self.variant != "fixed_length_population":
            return super()._crossover(first, second, problem, rng)
        if len(first) != len(second) or len(first) < 3:
            return list(first), False
        for _ in range(10):
            cut = int(rng.integers(1, len(first) - 1))
            candidate = _deduplicate_consecutive(
                [*first[: cut + 1], *second[cut + 1 :]]
            )
            if len(candidate) != len(first):
                continue
            if not problem.segment_is_free(candidate[cut], candidate[cut + 1]):
                continue
            if _path_key(candidate) != _path_key(first):
                return candidate, True
        return list(first), False

    def _update_archive(
        self,
        candidates: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        if self.variant != "no_quality_diversity_archive":
            return super()._update_archive(candidates, diagonal)
        unique = self._unique_feasible(candidates)
        return sorted(unique, key=lambda item: item.evaluation.total_cost)[
            : self.archive_size
        ]

    def _select_survivors(
        self,
        candidates: list[_Individual],
        archive: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        if self.variant != "no_quality_diversity_archive":
            return super()._select_survivors(candidates, archive, diagonal)
        unique = self._unique_feasible(candidates)
        return sorted(unique, key=lambda item: item.evaluation.total_cost)[
            : self.population_size
        ]

    def _generation_cap(self, problem: BudgetedEvaluator) -> int:
        if self.variant == "no_rooms_maze_strategy":
            return self.max_generations
        return super()._generation_cap(problem)

    def _mutate_insert(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if self.variant != "no_rooms_maze_strategy":
            return super()._mutate_insert(path, problem, rng)
        if len(path) >= self.max_waypoints:
            return list(path), False
        indices = list(rng.permutation(len(path) - 1))
        diagonal = problem.environment.diagonal
        for segment_index in indices[: min(8, len(indices))]:
            start, end = path[segment_index], path[segment_index + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            perpendicular = (-dy / length, dx / length)
            base_scale = min(0.12 * length, 0.025 * diagonal)
            offsets = [
                float(rng.normal(0.0, max(base_scale, 0.15))),
                float(rng.normal(0.0, max(base_scale * 0.5, 0.08))),
                0.0,
            ]
            for offset in offsets:
                fraction = float(rng.uniform(0.25, 0.75))
                point = (
                    start[0] + fraction * dx + offset * perpendicular[0],
                    start[1] + fraction * dy + offset * perpendicular[1],
                )
                if not problem.environment.in_bounds(point):
                    continue
                if problem.segment_is_free(start, point) and problem.segment_is_free(
                    point, end
                ):
                    return [
                        *path[: segment_index + 1],
                        point,
                        *path[segment_index + 1 :],
                    ], True
        return list(path), False

    def _mutate_delete(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if self.variant != "no_rooms_maze_strategy":
            return super()._mutate_delete(path, problem, rng)
        if len(path) <= 2:
            return list(path), False
        for index in rng.permutation(np.arange(1, len(path) - 1)):
            waypoint_index = int(index)
            if problem.segment_is_free(path[waypoint_index - 1], path[waypoint_index + 1]):
                return [*path[:waypoint_index], *path[waypoint_index + 1 :]], True
        return list(path), False

    def _mutate_move(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if self.variant != "no_rooms_maze_strategy":
            return super()._mutate_move(path, problem, rng)
        if len(path) <= 2:
            return list(path), False
        environment = problem.environment
        indices = list(rng.permutation(np.arange(1, len(path) - 1)))
        for index in indices[: min(10, len(indices))]:
            previous, current, following = path[index - 1], path[index], path[index + 1]
            midpoint = (
                0.5 * (previous[0] + following[0]),
                0.5 * (previous[1] + following[1]),
            )
            proposals: list[Waypoint] = [
                (
                    current[0] + alpha * (midpoint[0] - current[0]),
                    current[1] + alpha * (midpoint[1] - current[1]),
                )
                for alpha in (0.15, 0.30, 0.50, 0.70)
            ]
            scale = 0.035 * environment.diagonal
            for factor in (1.0, 0.5, 0.25, 0.125):
                delta = rng.normal(0.0, max(0.08, scale * factor), size=2)
                proposals.append(
                    (current[0] + float(delta[0]), current[1] + float(delta[1]))
                )
            for point in proposals:
                if math.dist(point, current) <= 1e-9 or not environment.in_bounds(point):
                    continue
                if problem.segment_is_free(previous, point) and problem.segment_is_free(
                    point, following
                ):
                    candidate = list(path)
                    candidate[index] = point
                    return candidate, True
        return list(path), False


__all__ = ["EvolutionaryAFLExperimentPlanner", "ExperimentVariant"]
