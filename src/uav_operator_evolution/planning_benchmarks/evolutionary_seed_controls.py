"""Seed-source controls for the frozen Evolutionary AFL-UAV v1 layer.

The scientific question is whether the LLM-generated AFL seed contributes
anything beyond the shared quality-diversity evolution. This module therefore
changes only ``base_planner``. Population construction, variation operators,
archive maintenance, survivor selection, stopping, and final selection are
inherited verbatim from the hash-frozen v1 implementation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..path.models import Path as UAVPath, Waypoint
from ..reproducibility import stable_hash
from .core import (
    BudgetedEvaluator,
    ObjectiveBudgetExhausted,
    PlannerResult,
    PlanningBudget,
    PlanningTimeout,
)
from .evolutionary_afl import EvolutionaryAFLUAVPlanner, _deduplicate_consecutive
from .planners import GridPlanner

SeedSource = Literal["astar", "theta_star", "handcrafted_destroy_repair"]

FROZEN_V1_CORE_SHA256 = (
    "79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f"
)


@dataclass(frozen=True)
class _ControlArtifactMetadata:
    """The small metadata surface consumed by the inherited v1 planner."""

    seed_source: SeedSource
    max_waypoints: int
    artifact_id: str = field(init=False)
    solver_hash: str = field(init=False)
    provider: str = field(default="not_applicable", init=False)
    model: str = field(default="handcrafted_seed_control", init=False)

    def __post_init__(self) -> None:
        specification = {
            "kind": "evolutionary-afl-v1-seed-control",
            "seed_source": self.seed_source,
            "max_waypoints": self.max_waypoints,
            "frozen_v1_core_sha256": FROZEN_V1_CORE_SHA256,
        }
        identifier = stable_hash(specification)
        object.__setattr__(self, "artifact_id", f"seed-control-{identifier}")
        object.__setattr__(self, "solver_hash", identifier)


@dataclass
class HandcraftedDestroyRepairSeedPlanner:
    """A human-written A* plus bounded destroy-repair seed solver."""

    resolution: float = 2.0
    iteration_limit: int = 64
    time_fraction: float = 0.18
    name: str = field(default="handcrafted_destroy_repair_seed", init=False)
    stochastic: bool = field(default=True, init=False)
    research_claim_eligible: bool = field(default=True, init=False)

    @property
    def algorithm_parameters(self) -> dict[str, object]:
        return {
            "initialization": "astar",
            "grid_resolution": self.resolution,
            "iteration_limit": self.iteration_limit,
            "time_fraction": self.time_fraction,
            "acceptance": "strict_trusted_cost_improvement",
            "destroy": "contiguous_internal_waypoint_block",
            "repair": "direct_or_single_waypoint_reconnection",
            "llm_generated": False,
        }

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        diagnostics: dict[str, object] = {
            "algorithm": self.algorithm_parameters,
            "attempts": 0,
            "feasible_candidates": 0,
            "accepted_improvements": 0,
        }
        initial = GridPlanner("astar", self.resolution).plan(problem, budget, rng)
        diagnostics["astar_status"] = initial.status
        if initial.path is None:
            return problem.result(
                self.name,
                initial.status,
                message="handcrafted seed could not obtain its A* initialization",
                diagnostics=diagnostics,
            )

        best_path = _deduplicate_consecutive(list(initial.path))
        try:
            best_evaluation = problem.evaluate(best_path)
        except (PlanningTimeout, ObjectiveBudgetExhausted) as exc:
            status = "timeout" if isinstance(exc, PlanningTimeout) else "budget_exhausted"
            return problem.result(
                self.name,
                status,
                path=best_path,
                message="budget ended while evaluating the A* seed",
                diagnostics=diagnostics,
            )
        if not best_evaluation.feasible:
            return problem.result(
                self.name,
                "invalid_path",
                path=best_path,
                message="A* initialization failed the shared objective evaluator",
                diagnostics=diagnostics,
            )

        initial_cost = best_evaluation.total_cost
        local_deadline = self.time_fraction * budget.time_limit_seconds
        try:
            for _ in range(self.iteration_limit):
                problem.check_time()
                if problem.elapsed_seconds >= local_deadline:
                    diagnostics["stop_reason"] = "seed_time_fraction"
                    break
                if problem.remaining_evaluations <= 1:
                    diagnostics["stop_reason"] = "reserved_outer_evaluation"
                    break
                diagnostics["attempts"] = int(diagnostics["attempts"]) + 1
                candidate = self._destroy_and_repair(best_path, problem, rng)
                if candidate is None:
                    continue
                evaluation = problem.evaluate(candidate)
                if not evaluation.feasible:
                    continue
                diagnostics["feasible_candidates"] = (
                    int(diagnostics["feasible_candidates"]) + 1
                )
                if evaluation.total_cost + 1e-12 < best_evaluation.total_cost:
                    best_path = candidate
                    best_evaluation = evaluation
                    diagnostics["accepted_improvements"] = (
                        int(diagnostics["accepted_improvements"]) + 1
                    )
        except PlanningTimeout:
            diagnostics["stop_reason"] = "hard_time_limit"
        except ObjectiveBudgetExhausted:
            diagnostics["stop_reason"] = "evaluation_limit"

        diagnostics.setdefault("stop_reason", "iteration_limit")
        diagnostics["initial_cost"] = initial_cost
        diagnostics["best_cost"] = best_evaluation.total_cost
        diagnostics["absolute_improvement"] = initial_cost - best_evaluation.total_cost
        return problem.result(
            self.name,
            "success",
            path=best_path,
            diagnostics=diagnostics,
        )

    def _destroy_and_repair(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> UAVPath | None:
        if len(path) <= 2:
            return None
        internal_count = len(path) - 2
        remove_count = int(rng.integers(1, min(internal_count, 3) + 1))
        first = int(rng.integers(1, len(path) - remove_count))
        last = first + remove_count - 1
        left = path[first - 1]
        right = path[last + 1]

        if problem.segment_is_free(left, right):
            return _deduplicate_consecutive([*path[:first], *path[last + 1 :]])

        removed = path[first : last + 1]
        proposals: list[Waypoint] = list(removed)
        dx, dy = right[0] - left[0], right[1] - left[1]
        length = math.hypot(dx, dy)
        if length > 1e-12:
            normal = (-dy / length, dx / length)
            scale = min(0.10 * length, 0.025 * problem.environment.diagonal)
            for fraction in (0.25, 0.5, 0.75):
                for multiplier in (0.0, -1.0, 1.0, -0.5, 0.5):
                    jitter = multiplier * scale
                    proposals.append(
                        (
                            left[0] + fraction * dx + jitter * normal[0],
                            left[1] + fraction * dy + jitter * normal[1],
                        )
                    )
            current = removed[int(rng.integers(0, len(removed)))]
            midpoint = (0.5 * (left[0] + right[0]), 0.5 * (left[1] + right[1]))
            for alpha in (0.2, 0.4, 0.6, 0.8):
                proposals.append(
                    (
                        current[0] + alpha * (midpoint[0] - current[0]),
                        current[1] + alpha * (midpoint[1] - current[1]),
                    )
                )

        for index in rng.permutation(len(proposals)):
            point = proposals[int(index)]
            if not problem.environment.in_bounds(point):
                continue
            if not problem.point_is_free(point):
                continue
            if not problem.segment_is_free(left, point):
                continue
            if not problem.segment_is_free(point, right):
                continue
            candidate = _deduplicate_consecutive(
                [*path[:first], point, *path[last + 1 :]]
            )
            if candidate != path:
                return candidate
        return None


@dataclass
class SeedSourceEvolutionaryControlPlanner(EvolutionaryAFLUAVPlanner):
    """Frozen v1 evolution with exactly one non-AFL seed source."""

    artifact_path: str = "control://no-llm-artifact"
    seed_source: SeedSource = "astar"
    grid_resolution: float = 2.0
    manual_seed_time_fraction: float = 0.18
    name: str = field(default="evolutionary_seed_control", init=False)
    research_claim_eligible: bool = field(default=True, init=False)
    test_restricted: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if self.seed_source not in {
            "astar",
            "theta_star",
            "handcrafted_destroy_repair",
        }:
            raise ValueError(f"unknown seed-source control: {self.seed_source}")
        if self.population_size < 4:
            raise ValueError("seed control population_size must be at least 4")
        if not 2 <= self.archive_size <= self.population_size:
            raise ValueError("archive_size must be between 2 and population_size")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive")
        if self.grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")
        if not 0.0 < self.manual_seed_time_fraction < 0.88:
            raise ValueError("manual seed time fraction must be in (0, 0.88)")

        if self.seed_source in {"astar", "theta_star"}:
            self.base_planner = GridPlanner(self.seed_source, self.grid_resolution)
        else:
            self.base_planner = HandcraftedDestroyRepairSeedPlanner(
                resolution=self.grid_resolution,
                iteration_limit=self.base_iteration_limit,
                time_fraction=self.manual_seed_time_fraction,
            )
        self.artifact = _ControlArtifactMetadata(
            self.seed_source,
            self.max_waypoints,
        )

    @property
    def algorithm_parameters(self) -> dict[str, object]:
        parameters = dict(super().algorithm_parameters)
        parameters.update(
            {
                "experiment": "seed_source_control",
                "seed_source": self.seed_source,
                "grid_resolution": self.grid_resolution,
                "manual_seed_time_fraction": self.manual_seed_time_fraction,
                "shared_evolution_layer": "EvolutionaryAFLUAVPlanner",
                "shared_evolution_core_sha256": FROZEN_V1_CORE_SHA256,
                "frozen_v1_core_modified": False,
                "llm_generated_seed": False,
            }
        )
        if hasattr(self.base_planner, "algorithm_parameters"):
            parameters["seed_algorithm"] = self.base_planner.algorithm_parameters
        return parameters


__all__ = [
    "FROZEN_V1_CORE_SHA256",
    "HandcraftedDestroyRepairSeedPlanner",
    "SeedSource",
    "SeedSourceEvolutionaryControlPlanner",
]
