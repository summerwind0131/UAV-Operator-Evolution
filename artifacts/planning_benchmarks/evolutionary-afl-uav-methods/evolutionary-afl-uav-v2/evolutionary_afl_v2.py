"""Development-only Evolutionary AFL-UAV v2.

V2 is intentionally isolated from the hash-frozen v1 implementation.  It adds
three mechanisms motivated by v1's final evaluation: an earlier cooperative
deadline, a multi-source initial population, and explicit rooms/maze portal
topology.  Development is restricted to Train/Validation until a separately
frozen v2 and a new hidden test are preregistered.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..environment.environment import Environment2D
from ..environment.geometry import euclidean_distance
from ..environment.obstacles import RectangleObstacle
from ..path.models import Path as UAVPath, Waypoint
from .core import (
    BudgetedEvaluator,
    ObjectiveBudgetExhausted,
    PlannerResult,
    PlanningBudget,
    PlanningTimeout,
)
from .evolutionary_afl import (
    EvolutionaryAFLUAVPlanner,
    _Individual,
    _deduplicate_consecutive,
    _path_key,
)
from .planners import GridPlanner, PRMPlanner


class _DeadlineEvaluatorView:
    """Delegate accounting to one evaluator while enforcing a local deadline."""

    def __init__(
        self,
        parent: BudgetedEvaluator,
        deadline_seconds: float,
        *,
        start_guard_seconds: float = 0.0,
    ) -> None:
        self.parent = parent
        self.environment = parent.environment
        self.evaluator = parent.evaluator
        self.budget = parent.budget
        self.deadline_seconds = deadline_seconds
        self.start_guard_seconds = start_guard_seconds
        self.local_timeout_triggered = False

    @property
    def elapsed_seconds(self) -> float:
        return self.parent.elapsed_seconds

    @property
    def remaining_evaluations(self) -> int:
        return self.parent.remaining_evaluations

    def check_time(self) -> None:
        self.parent.check_time()
        if self.parent.elapsed_seconds >= (
            self.deadline_seconds - self.start_guard_seconds
        ):
            self.local_timeout_triggered = True
            raise PlanningTimeout(
                "v2 planner reached its local deadline or operation start guard"
            )

    def evaluate(self, path: UAVPath):
        self.check_time()
        return self.parent.evaluate(path)

    def point_is_free(self, point: Waypoint) -> bool:
        self.check_time()
        return self.parent.point_is_free(point)

    def segment_is_free(self, start: Waypoint, end: Waypoint) -> bool:
        self.check_time()
        return self.parent.segment_is_free(start, end)

    def expand(self, count: int = 1) -> None:
        self.check_time()
        self.parent.expand(count)

    def record_external_counts(
        self,
        *,
        objective_evaluations: int,
        collision_checks: int,
        node_expansions: int,
    ) -> None:
        self.check_time()
        self.parent.record_external_counts(
            objective_evaluations=objective_evaluations,
            collision_checks=collision_checks,
            node_expansions=node_expansions,
        )

    def result(self, *args: Any, **kwargs: Any) -> PlannerResult:
        return self.parent.result(*args, **kwargs)


def _extract_wall_portals(environment: Environment2D) -> list[Waypoint]:
    """Extract centers of door-like gaps between collinear wall rectangles."""

    rectangles = [
        obstacle
        for obstacle in environment.obstacles
        if isinstance(obstacle, RectangleObstacle)
    ]
    minimum_gap = 2.0 * environment.safety_distance + 1e-6
    maximum_gap = 0.30 * max(environment.width, environment.height)
    vertical: dict[tuple[float, float], list[RectangleObstacle]] = {}
    horizontal: dict[tuple[float, float], list[RectangleObstacle]] = {}
    for rectangle in rectangles:
        width = rectangle.max_x - rectangle.min_x
        height = rectangle.max_y - rectangle.min_y
        if height >= width:
            vertical.setdefault(
                (round(rectangle.min_x, 6), round(rectangle.max_x, 6)), []
            ).append(rectangle)
        if width >= height:
            horizontal.setdefault(
                (round(rectangle.min_y, 6), round(rectangle.max_y, 6)), []
            ).append(rectangle)

    candidates: list[Waypoint] = []
    for (min_x, max_x), walls in vertical.items():
        ordered = sorted(walls, key=lambda item: (item.min_y, item.max_y))
        for lower, upper in zip(ordered, ordered[1:]):
            gap = upper.min_y - lower.max_y
            if minimum_gap < gap <= maximum_gap:
                candidates.append(
                    (0.5 * (min_x + max_x), 0.5 * (lower.max_y + upper.min_y))
                )
    for (min_y, max_y), walls in horizontal.items():
        ordered = sorted(walls, key=lambda item: (item.min_x, item.max_x))
        for left, right in zip(ordered, ordered[1:]):
            gap = right.min_x - left.max_x
            if minimum_gap < gap <= maximum_gap:
                candidates.append(
                    (0.5 * (left.max_x + right.min_x), 0.5 * (min_y + max_y))
                )

    unique: list[Waypoint] = []
    for point in sorted(candidates):
        if not environment.in_bounds(point):
            continue
        if all(euclidean_distance(point, previous) > 1e-5 for previous in unique):
            unique.append(point)
    return unique


def _point_segment_distance(point: Waypoint, start: Waypoint, end: Waypoint) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-18:
        return euclidean_distance(point, start)
    ratio = max(
        0.0,
        min(
            1.0,
            ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy)
            / denominator,
        ),
    )
    projection = (start[0] + ratio * dx, start[1] + ratio * dy)
    return euclidean_distance(point, projection)


@dataclass
class EvolutionaryAFLUAVV2Planner(EvolutionaryAFLUAVPlanner):
    """Reliability- and topology-oriented successor to frozen v1."""

    variant: Literal["reliability_only", "multisource", "full"] = "full"
    cooperative_budget_fraction: float = 0.97
    portfolio_budget_fraction: float = 0.16
    operation_deadline_fraction: float = 0.88
    finalization_guard_fraction: float = 0.06
    fallback_deadline_fraction: float = 0.90
    source_quotas: dict[str, int] = field(
        default_factory=lambda: {
            "afl": 8,
            "astar": 6,
            "theta_star": 6,
            "prm": 6,
            "topology": 6,
        }
    )
    # Keep the registered planner family name so the existing generic runner's
    # AFL Test-split guard also applies to this development version.  ``arm_id``
    # and algorithm metadata distinguish v2 from frozen v1.
    name: str = field(default="evolutionary_afl_uav", init=False)
    research_claim_eligible: bool = field(default=False, init=False)
    _active_portals: list[Waypoint] = field(default_factory=list, init=False, repr=False)
    _active_problem: BudgetedEvaluator | None = field(default=None, init=False, repr=False)
    _active_soft_deadline_seconds: float = field(default=0.0, init=False, repr=False)
    _fast_finalization_calls: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.variant not in {"reliability_only", "multisource", "full"}:
            raise ValueError(f"unknown v2 development variant: {self.variant}")
        if not 0.90 <= self.cooperative_budget_fraction < 1.0:
            raise ValueError("v2 cooperative budget fraction must be in [0.90, 1.0)")
        if not 0.0 < self.portfolio_budget_fraction <= 0.35:
            raise ValueError("v2 portfolio budget fraction must be in (0, 0.35]")
        effective_soft_deadline = 0.88 * self.cooperative_budget_fraction
        if not effective_soft_deadline < self.operation_deadline_fraction < 1.0:
            raise ValueError(
                "v2 operation deadline must follow the evolution soft deadline "
                "and remain below 1.0"
            )
        if not 0.0 < self.finalization_guard_fraction < effective_soft_deadline:
            raise ValueError("v2 finalization guard fraction is out of range")
        if not self.operation_deadline_fraction <= self.fallback_deadline_fraction < 1.0:
            raise ValueError(
                "v2 fallback deadline must follow the operation deadline and "
                "remain below 1.0"
            )
        if sum(self.source_quotas.values()) != self.population_size:
            raise ValueError("v2 source quotas must sum to population_size")
        if set(self.source_quotas) != {
            "afl",
            "astar",
            "theta_star",
            "prm",
            "topology",
        }:
            raise ValueError("v2 source quotas must define all five seed families")
        super().__post_init__()
        # A development method cannot inherit the frozen seed artifact's claim
        # eligibility.  V2 becomes eligible only after a separate freeze.
        self.research_claim_eligible = False

    @property
    def algorithm_parameters(self) -> dict[str, Any]:
        parameters = dict(super().algorithm_parameters)
        parameters.update(
            {
                "version": "evolutionary-afl-uav-v2-development",
                "development_variant": self.variant,
                "frozen_v1_modified": False,
                "allowed_development_splits": ["train", "validation"],
                "cooperative_budget_fraction": self.cooperative_budget_fraction,
                "effective_evolution_soft_deadline_fraction": (
                    0.88 * self.cooperative_budget_fraction
                ),
                "operation_deadline_fraction": self.operation_deadline_fraction,
                "finalization_guard_fraction": self.finalization_guard_fraction,
                "effective_operation_start_deadline_fraction": (
                    self.operation_deadline_fraction
                    - self.finalization_guard_fraction
                ),
                "fallback_deadline_fraction": self.fallback_deadline_fraction,
                "population_sources": dict(self.source_quotas),
                "portfolio_planners": (
                    []
                    if self.variant == "reliability_only"
                    else ["astar", "theta_star", "prm"]
                ),
                "rooms_maze_topology": (
                    "analytic wall-gap portals + visibility graph + portal-aware archive"
                    if self.variant == "full"
                    else "disabled"
                ),
            }
        )
        return parameters

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        self._active_portals = (
            _extract_wall_portals(problem.environment)
            if self.variant == "full"
            and problem.environment.difficulty == "rooms_maze"
            else []
        )
        cooperative_budget = budget.model_copy(
            update={
                "time_limit_seconds": (
                    self.cooperative_budget_fraction * budget.time_limit_seconds
                )
            }
        )
        self._active_problem = problem
        self._active_soft_deadline_seconds = (
            0.88 * cooperative_budget.time_limit_seconds
        )
        self._fast_finalization_calls = 0
        operation_deadline = (
            self.operation_deadline_fraction * budget.time_limit_seconds
        )
        operation_start_guard = (
            self.finalization_guard_fraction * budget.time_limit_seconds
        )
        guarded_problem = _DeadlineEvaluatorView(
            problem,
            operation_deadline,
            start_guard_seconds=operation_start_guard,
        )
        try:
            result = super().plan(guarded_problem, cooperative_budget, rng)
        finally:
            self._active_problem = None
        result.diagnostics["v2_advertised_time_limit_seconds"] = budget.time_limit_seconds
        result.diagnostics["v2_cooperative_time_limit_seconds"] = (
            cooperative_budget.time_limit_seconds
        )
        result.diagnostics["v2_finalization_reserve_seconds"] = max(
            0.0,
            budget.time_limit_seconds - problem.elapsed_seconds,
        )
        result.diagnostics["portal_centers_detected"] = len(self._active_portals)
        result.diagnostics["v2_operation_deadline_seconds"] = operation_deadline
        result.diagnostics["v2_operation_start_guard_seconds"] = (
            operation_start_guard
        )
        result.diagnostics["v2_effective_operation_start_deadline_seconds"] = (
            operation_deadline - operation_start_guard
        )
        result.diagnostics["v2_local_deadline_triggered"] = (
            guarded_problem.local_timeout_triggered
        )
        result.diagnostics["v2_fast_finalization_calls"] = (
            self._fast_finalization_calls
        )
        result.diagnostics["research_claim_eligible"] = False
        if result.path is not None:
            if result.status == "timeout" and guarded_problem.local_timeout_triggered:
                result = problem.result(
                    self.name,
                    "success",
                    path=result.path,
                    message=(
                        "v2 reached its cooperative operation deadline and "
                        "returned the trusted best-so-far path"
                    ),
                    diagnostics=result.diagnostics,
                )
            return result

        fallback = self._fallback_path(problem, budget, rng)
        if fallback is None:
            return result
        try:
            evaluation = problem.evaluate(fallback)
        except (PlanningTimeout, ObjectiveBudgetExhausted):
            return result
        if not evaluation.feasible:
            return result
        result = problem.result(
            self.name,
            "success",
            path=fallback,
            message="v2 returned a trusted local-search fallback",
            diagnostics=result.diagnostics,
        )
        result.diagnostics["fallback_used"] = True
        return result

    def _near_finalization_guard(self) -> bool:
        problem = self._active_problem
        if problem is None:
            return False
        guard = self.finalization_guard_fraction * problem.budget.time_limit_seconds
        return problem.elapsed_seconds >= self._active_soft_deadline_seconds - guard

    def _update_archive(
        self,
        candidates: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        if not self._near_finalization_guard():
            return super()._update_archive(candidates, diagonal)
        self._fast_finalization_calls += 1
        unique = self._unique_feasible(candidates)
        return sorted(
            unique,
            key=lambda item: item.evaluation.total_cost,
        )[: self.archive_size]

    def _select_survivors(
        self,
        candidates: list[_Individual],
        archive: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        if not self._near_finalization_guard():
            return super()._select_survivors(candidates, archive, diagonal)
        self._fast_finalization_calls += 1
        unique = self._unique_feasible(candidates)
        return sorted(
            unique,
            key=lambda item: item.evaluation.total_cost,
        )[: self.population_size]

    def _fallback_path(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> UAVPath | None:
        final_deadline = self.fallback_deadline_fraction * budget.time_limit_seconds
        if problem.elapsed_seconds >= final_deadline:
            return None
        planners = (
            GridPlanner("theta_star", resolution=2.0),
            GridPlanner("astar", resolution=2.0),
        )
        for index, planner in enumerate(planners):
            remaining = final_deadline - problem.elapsed_seconds
            if remaining <= 0.01:
                break
            local_deadline = problem.elapsed_seconds + remaining / (len(planners) - index)
            try:
                candidate = planner.plan(
                    _DeadlineEvaluatorView(problem, local_deadline),
                    budget,
                    rng,
                )
            except (PlanningTimeout, ObjectiveBudgetExhausted):
                continue
            if candidate.path is not None:
                return _deduplicate_consecutive(list(candidate.path))
        return None

    def _generation_cap(self, problem: BudgetedEvaluator) -> int:
        if self.variant == "full" and problem.environment.difficulty == "rooms_maze":
            return min(self.max_generations, 10)
        return super()._generation_cap(problem)

    def _collect_portfolio_sources(
        self,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
        soft_deadline: float,
        diagnostics: dict[str, Any],
    ) -> dict[str, list[UAVPath]]:
        sources: dict[str, list[UAVPath]] = {
            "astar": [],
            "theta_star": [],
            "prm": [],
            "topology": [],
        }
        statuses: dict[str, str] = {}
        outer_time = problem.budget.time_limit_seconds
        portfolio_end = min(
            soft_deadline - 0.30 * outer_time,
            problem.elapsed_seconds + self.portfolio_budget_fraction * outer_time,
        )
        planner_definitions = (
            ("astar", GridPlanner("astar", resolution=3.0)),
            ("theta_star", GridPlanner("theta_star", resolution=3.0)),
            ("prm", PRMPlanner(sample_count=60, neighbor_count=8)),
        )
        for index, (source_name, planner) in enumerate(planner_definitions):
            remaining = portfolio_end - problem.elapsed_seconds
            if remaining <= 0.01:
                statuses[source_name] = "skipped_no_portfolio_time"
                continue
            local_deadline = problem.elapsed_seconds + remaining / (
                len(planner_definitions) - index + 1
            )
            try:
                result = planner.plan(
                    _DeadlineEvaluatorView(problem, local_deadline),
                    problem.budget,
                    rng,
                )
            except PlanningTimeout:
                statuses[source_name] = "local_timeout"
                continue
            except ObjectiveBudgetExhausted:
                statuses[source_name] = "evaluation_limit"
                continue
            statuses[source_name] = result.status
            if result.path is not None:
                sources[source_name].append(
                    _deduplicate_consecutive(list(result.path))
                )

        if (
            self._active_portals
            and problem.elapsed_seconds < portfolio_end
            and problem.environment.difficulty == "rooms_maze"
        ):
            sources["topology"] = self._portal_graph_paths(
                problem,
                rng,
                portfolio_end,
                limit=max(1, self.source_quotas["topology"]),
            )
            statuses["topology"] = (
                "success" if sources["topology"] else "no_path"
            )
        else:
            statuses["topology"] = "not_applicable"
        diagnostics["portfolio_source_status"] = statuses
        diagnostics["portfolio_raw_paths"] = {
            name: len(paths) for name, paths in sources.items()
        }
        return sources

    def _portal_graph_paths(
        self,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
        deadline: float,
        *,
        limit: int,
    ) -> list[UAVPath]:
        nodes = [
            problem.environment.start,
            problem.environment.goal,
            *self._active_portals,
            *self._room_center_nodes(problem.environment),
        ]
        nodes = list(dict.fromkeys(nodes))
        graph: list[list[tuple[int, float]]] = [[] for _ in nodes]
        checked_edges: set[tuple[int, int]] = set()
        for first in range(len(nodes)):
            nearest = sorted(
                (
                    (euclidean_distance(nodes[first], nodes[second]), second)
                    for second in range(len(nodes))
                    if second != first
                ),
                key=lambda item: (item[0], item[1]),
            )[:16]
            for _, second in nearest:
                edge = (min(first, second), max(first, second))
                if edge in checked_edges:
                    continue
                checked_edges.add(edge)
                if problem.elapsed_seconds >= deadline:
                    break
                try:
                    visible = problem.segment_is_free(nodes[first], nodes[second])
                except PlanningTimeout:
                    return []
                if not visible:
                    continue
                distance = euclidean_distance(nodes[first], nodes[second])
                graph[first].append((second, distance))
                graph[second].append((first, distance))

        paths: list[UAVPath] = []
        signatures: set[tuple[tuple[float, float], ...]] = set()
        for variant in range(max(limit * 3, 6)):
            if problem.elapsed_seconds >= deadline:
                break
            frontier = [(0.0, 0)]
            costs = {0: 0.0}
            parent: dict[int, int] = {}
            while frontier:
                current_cost, current = heapq.heappop(frontier)
                if current_cost > costs.get(current, math.inf) + 1e-12:
                    continue
                if current == 1:
                    indices = [1]
                    while indices[-1] in parent:
                        indices.append(parent[indices[-1]])
                    indices.reverse()
                    path = _deduplicate_consecutive([nodes[item] for item in indices])
                    key = _path_key(path)
                    if key not in signatures:
                        signatures.add(key)
                        paths.append(path)
                    break
                for neighbor, distance in graph[current]:
                    noise = 1.0 if variant == 0 else 1.0 + 0.35 * float(rng.random())
                    candidate = current_cost + distance * noise
                    if candidate + 1e-12 < costs.get(neighbor, math.inf):
                        costs[neighbor] = candidate
                        parent[neighbor] = current
                        heapq.heappush(frontier, (candidate, neighbor))
            if len(paths) >= limit:
                break
        return paths

    @staticmethod
    def _room_center_nodes(environment: Environment2D) -> list[Waypoint]:
        """Create free representative nodes for cells between partition walls."""

        vertical_lines: list[float] = []
        horizontal_lines: list[float] = []
        for obstacle in environment.obstacles:
            if not isinstance(obstacle, RectangleObstacle):
                continue
            width = obstacle.max_x - obstacle.min_x
            height = obstacle.max_y - obstacle.min_y
            if height > 2.0 * width:
                vertical_lines.append(0.5 * (obstacle.min_x + obstacle.max_x))
            if width > 2.0 * height:
                horizontal_lines.append(0.5 * (obstacle.min_y + obstacle.max_y))
        xs = [0.0, *sorted(set(round(value, 6) for value in vertical_lines)), environment.width]
        ys = [0.0, *sorted(set(round(value, 6) for value in horizontal_lines)), environment.height]
        x_centers = [0.5 * (left + right) for left, right in zip(xs, xs[1:]) if right - left > 1e-6]
        y_centers = [0.5 * (lower + upper) for lower, upper in zip(ys, ys[1:]) if upper - lower > 1e-6]
        centers = [
            (x, y)
            for x in x_centers
            for y in y_centers
            if environment.point_is_collision_free((x, y))
        ]
        # Recursive mazes can induce many cells.  Keep a deterministic spread
        # so visibility construction remains inside the portfolio time slice.
        if len(centers) > 64:
            indices = np.linspace(0, len(centers) - 1, num=64, dtype=int)
            centers = [centers[int(index)] for index in indices]
        return centers

    def _initialize_population(
        self,
        seed: _Individual,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
        cache: dict[tuple[tuple[float, float], ...], _Individual],
        diagnostics: dict[str, Any],
        soft_deadline: float,
    ) -> tuple[list[_Individual], int, int]:
        if self.variant == "reliability_only":
            diagnostics["portfolio_source_status"] = {
                "astar": "disabled",
                "theta_star": "disabled",
                "prm": "disabled",
                "topology": "disabled",
            }
            diagnostics["portfolio_raw_paths"] = {
                "astar": 0,
                "theta_star": 0,
                "prm": 0,
                "topology": 0,
            }
            population, accepted, duplicates = super()._initialize_population(
                seed,
                problem,
                rng,
                cache,
                diagnostics,
                soft_deadline,
            )
            diagnostics["population_source_counts"] = {
                "afl": len(population),
                "astar": 0,
                "theta_star": 0,
                "prm": 0,
                "topology": 0,
                "redistributed": 0,
            }
            diagnostics["population_sources_available"] = {
                "afl": True,
                "astar": False,
                "theta_star": False,
                "prm": False,
                "topology": False,
            }
            return population, accepted, duplicates
        source_paths = self._collect_portfolio_sources(
            problem, rng, soft_deadline, diagnostics
        )
        source_paths["afl"] = [list(seed.path)]
        population: list[_Individual] = [seed]
        source_individuals: dict[str, list[_Individual]] = {
            name: ([seed] if name == "afl" else []) for name in self.source_quotas
        }
        source_individuals["redistributed"] = []
        source_counts = {name: len(items) for name, items in source_individuals.items()}
        accepted = 0
        duplicates = 0
        quality_limits = {
            "afl": math.inf,
            "astar": 1.20,
            "theta_star": 1.20,
            "prm": 1.25,
            "topology": 1.35,
            "redistributed": math.inf,
        }

        def register(path: UAVPath, source: str) -> bool:
            nonlocal accepted, duplicates
            candidate = _deduplicate_consecutive(path)
            if not self._valid_shape(candidate, problem):
                return False
            key = _path_key(candidate)
            if key in cache:
                duplicates += 1
                return False
            evaluation = problem.evaluate(candidate)
            individual = _Individual(candidate, evaluation)
            cache[key] = individual
            problem.expand()
            if not evaluation.feasible:
                return False
            if evaluation.total_cost > quality_limits[source] * seed.evaluation.total_cost:
                return False
            population.append(individual)
            source_individuals[source].append(individual)
            source_counts[source] += 1
            accepted += 1
            return True

        for source in ("topology", "astar", "theta_star", "prm"):
            for path in source_paths[source]:
                if len(population) >= self.population_size:
                    break
                if problem.elapsed_seconds >= soft_deadline:
                    break
                register(path, source)

        operators = ["insert", "delete", "move", "swap"]
        for source in ("afl", "astar", "theta_star", "prm", "topology"):
            target = self.source_quotas[source]
            if not source_individuals[source]:
                continue
            attempts = 0
            while (
                source_counts[source] < target
                and len(population) < self.population_size
                and attempts < max(20, target * 24)
                and problem.elapsed_seconds < soft_deadline
            ):
                attempts += 1
                pool = source_individuals[source] or population
                parent = pool[int(rng.integers(0, len(pool)))]
                operator = operators[(attempts - 1) % len(operators)]
                diagnostics["operator_attempts"][operator] += 1
                candidate, changed = self._apply_operator(
                    operator, parent.path, problem, rng
                )
                if changed:
                    diagnostics["operator_successes"][operator] += 1
                if not changed:
                    continue
                register(candidate, source)

        # Missing portfolio sources redistribute their unused slots across all
        # feasible families instead of shrinking the population.
        attempts = 0
        while (
            len(population) < self.population_size
            and attempts < self.population_size * 30
            and problem.elapsed_seconds < soft_deadline
        ):
            attempts += 1
            ordered_parents = sorted(
                population, key=lambda item: item.evaluation.total_cost
            )[: max(1, min(8, len(population)))]
            parent = ordered_parents[int(rng.integers(0, len(ordered_parents)))]
            operator = operators[(attempts - 1) % len(operators)]
            diagnostics["operator_attempts"][operator] += 1
            candidate, changed = self._apply_operator(
                operator, parent.path, problem, rng
            )
            if changed:
                diagnostics["operator_successes"][operator] += 1
            if not changed:
                continue
            register(candidate, "redistributed")

        diagnostics["population_source_counts"] = source_counts
        diagnostics["population_sources_available"] = {
            name: bool(source_individuals[name]) for name in self.source_quotas
        }
        return population, accepted, duplicates

    def _mutate_insert(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if (
            problem.environment.difficulty == "rooms_maze"
            and self._active_portals
            and len(path) < self.max_waypoints
            and rng.random() < 0.70
        ):
            options: list[tuple[float, int, Waypoint]] = []
            for portal in self._active_portals:
                if any(euclidean_distance(portal, point) <= 1e-5 for point in path):
                    continue
                for index, (start, end) in enumerate(zip(path, path[1:])):
                    detour = (
                        euclidean_distance(start, portal)
                        + euclidean_distance(portal, end)
                        - euclidean_distance(start, end)
                    )
                    options.append((detour, index, portal))
            for _, index, portal in sorted(options)[:12]:
                if problem.segment_is_free(path[index], portal) and problem.segment_is_free(
                    portal, path[index + 1]
                ):
                    return [*path[: index + 1], portal, *path[index + 1 :]], True
        return super()._mutate_insert(path, problem, rng)

    def _mutate_move(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if (
            problem.environment.difficulty == "rooms_maze"
            and self._active_portals
            and len(path) > 2
            and rng.random() < 0.55
        ):
            proposals = sorted(
                (
                    (euclidean_distance(path[index], portal), index, portal)
                    for index in range(1, len(path) - 1)
                    for portal in self._active_portals
                ),
                key=lambda item: item[0],
            )
            maximum_distance = 0.18 * problem.environment.diagonal
            for distance, index, portal in proposals[:12]:
                if distance > maximum_distance:
                    break
                if problem.segment_is_free(path[index - 1], portal) and problem.segment_is_free(
                    portal, path[index + 1]
                ):
                    candidate = list(path)
                    candidate[index] = portal
                    return candidate, True
        return super()._mutate_move(path, problem, rng)

    def _portal_signature(self, path: UAVPath) -> tuple[int, ...]:
        if not self._active_portals:
            return ()
        threshold = 0.035 * 141.4213562373095
        signature: list[int] = []
        for index, portal in enumerate(self._active_portals):
            if any(
                _point_segment_distance(portal, start, end) <= threshold
                for start, end in zip(path, path[1:])
            ):
                signature.append(index)
        return tuple(signature)

    def _update_archive(
        self,
        candidates: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        base = super()._update_archive(candidates, diagonal)
        if not self._active_portals:
            return base
        unique = self._unique_feasible(candidates)
        if not unique:
            return base
        best_by_topology: dict[tuple[int, ...], _Individual] = {}
        for candidate in unique:
            signature = self._portal_signature(candidate.path)
            previous = best_by_topology.get(signature)
            if previous is None or candidate.evaluation.total_cost < previous.evaluation.total_cost:
                best_by_topology[signature] = candidate
        global_best = min(unique, key=lambda item: item.evaluation.total_cost)
        selected = [global_best]
        selected_keys = {global_best.key}
        for candidate in sorted(
            best_by_topology.values(), key=lambda item: item.evaluation.total_cost
        ):
            if candidate.key not in selected_keys and len(selected) < self.archive_size:
                selected.append(candidate)
                selected_keys.add(candidate.key)
        for candidate in base:
            if candidate.key not in selected_keys and len(selected) < self.archive_size:
                selected.append(candidate)
                selected_keys.add(candidate.key)
        return selected


__all__ = ["EvolutionaryAFLUAVV2Planner", "_extract_wall_portals"]
