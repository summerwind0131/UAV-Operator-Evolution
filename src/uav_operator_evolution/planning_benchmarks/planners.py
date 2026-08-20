"""Dependency-light reference planners for the UAV2D benchmark."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..environment.geometry import euclidean_distance
from ..path.models import EvaluationResult, Path, Waypoint
from .core import (
    BudgetedEvaluator,
    ObjectiveBudgetExhausted,
    PlannerResult,
    PlanningBudget,
    PlanningTimeout,
    simplify_path,
)

DIRECTIONS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _reconstruct(
    parent: dict[tuple[int, int], tuple[int, int]],
    current: tuple[int, int],
    point: callable,
) -> Path:
    nodes = [current]
    while current in parent:
        current = parent[current]
        nodes.append(current)
    nodes.reverse()
    return [point(node) for node in nodes]


def _grid_search(
    problem: BudgetedEvaluator,
    mode: str,
    resolution: float,
) -> Path | None:
    environment = problem.environment
    if problem.segment_is_free(environment.start, environment.goal):
        return [environment.start, environment.goal]
    x_count = math.ceil(environment.width / resolution) + 1
    y_count = math.ceil(environment.height / resolution) + 1
    xs = [min(index * resolution, environment.width) for index in range(x_count)]
    ys = [min(index * resolution, environment.height) for index in range(y_count)]

    def point(node: tuple[int, int]) -> Waypoint:
        return (xs[node[0]], ys[node[1]])

    traversable: dict[tuple[int, int], bool] = {}

    def is_free(node: tuple[int, int]) -> bool:
        if node not in traversable:
            traversable[node] = problem.point_is_free(point(node))
        return traversable[node]

    def visible_endpoint(endpoint: Waypoint) -> tuple[int, int] | None:
        ex, ey = endpoint
        ix = int(round(ex / resolution))
        iy = int(round(ey / resolution))
        candidates: list[tuple[float, tuple[int, int]]] = []
        radius = 0
        while radius <= max(x_count, y_count):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if max(abs(dx), abs(dy)) != radius:
                        continue
                    node = (
                        min(max(ix + dx, 0), x_count - 1),
                        min(max(iy + dy, 0), y_count - 1),
                    )
                    candidates.append((euclidean_distance(endpoint, point(node)), node))
            for _, node in sorted(set(candidates)):
                if is_free(node) and problem.segment_is_free(endpoint, point(node)):
                    return node
            radius += 1
        return None

    start_node = visible_endpoint(environment.start)
    goal_node = visible_endpoint(environment.goal)
    if start_node is None or goal_node is None:
        return None
    frontier: list[tuple[float, float, int, int]] = []
    heuristic = 0.0 if mode == "dijkstra" else euclidean_distance(
        point(start_node), point(goal_node)
    )
    heapq.heappush(frontier, (heuristic, 0.0, *start_node))
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    cost = {start_node: 0.0}
    closed: set[tuple[int, int]] = set()

    while frontier:
        _, current_cost, x_index, y_index = heapq.heappop(frontier)
        current = (x_index, y_index)
        if current in closed:
            continue
        closed.add(current)
        problem.expand()
        if current == goal_node:
            grid_path = _reconstruct(parent, current, point)
            full: Path = [environment.start]
            full.extend(
                item
                for item in grid_path
                if euclidean_distance(full[-1], item) > 1e-9
            )
            if euclidean_distance(full[-1], environment.goal) > 1e-9:
                full.append(environment.goal)
            return simplify_path(full, problem)
        for dx, dy in DIRECTIONS:
            neighbor = (x_index + dx, y_index + dy)
            if not (0 <= neighbor[0] < x_count and 0 <= neighbor[1] < y_count):
                continue
            if not is_free(neighbor):
                continue
            current_parent = current
            tentative = current_cost + euclidean_distance(point(current), point(neighbor))
            if mode == "theta_star" and current in parent:
                candidate_parent = parent[current]
                if problem.segment_is_free(point(candidate_parent), point(neighbor)):
                    candidate_cost = cost[candidate_parent] + euclidean_distance(
                        point(candidate_parent), point(neighbor)
                    )
                    if candidate_cost < tentative:
                        current_parent = candidate_parent
                        tentative = candidate_cost
            if current_parent == current and not problem.segment_is_free(
                point(current), point(neighbor)
            ):
                continue
            if tentative + 1e-12 >= cost.get(neighbor, math.inf):
                continue
            cost[neighbor] = tentative
            parent[neighbor] = current_parent
            estimate = tentative
            if mode != "dijkstra":
                estimate += euclidean_distance(point(neighbor), point(goal_node))
            heapq.heappush(frontier, (estimate, tentative, *neighbor))
    return None


@dataclass
class GridPlanner:
    """Dijkstra, A*, or Theta* on one shared 8-connected grid."""

    name: str
    resolution: float = 2.0
    stochastic: bool = False
    research_claim_eligible: bool = True

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        del budget, rng
        path = _grid_search(problem, self.name, self.resolution)
        if path is None:
            return problem.result(self.name, "no_path", message="grid search found no path")
        return problem.result(self.name, "success", path=path)


def _steer(start: Waypoint, target: Waypoint, step: float) -> Waypoint:
    distance = euclidean_distance(start, target)
    if distance <= step:
        return target
    ratio = step / distance
    return (
        start[0] + ratio * (target[0] - start[0]),
        start[1] + ratio * (target[1] - start[1]),
    )


def _tree_path(nodes: list[Waypoint], parents: list[int], index: int, goal: Waypoint) -> Path:
    path = [nodes[index]]
    while parents[index] >= 0:
        index = parents[index]
        path.append(nodes[index])
    path.reverse()
    if euclidean_distance(path[-1], goal) > 1e-9:
        path.append(goal)
    return path


@dataclass
class RRTPlanner:
    name: str
    step_size: float = 4.0
    goal_bias: float = 0.1
    rewire: bool = False
    max_samples: int = 1_000
    stochastic: bool = True
    research_claim_eligible: bool = True

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        del budget
        env = problem.environment
        if problem.segment_is_free(env.start, env.goal):
            return problem.result(self.name, "success", path=[env.start, env.goal])
        nodes: list[Waypoint] = [env.start]
        parents = [-1]
        costs = [0.0]
        best_path: Path | None = None
        best_cost = math.inf
        try:
            for _ in range(self.max_samples):
                problem.check_time()
                target = env.goal if rng.random() < self.goal_bias else (
                    float(rng.uniform(0.0, env.width)),
                    float(rng.uniform(0.0, env.height)),
                )
                nearest = min(
                    range(len(nodes)),
                    key=lambda index: euclidean_distance(nodes[index], target),
                )
                candidate = _steer(nodes[nearest], target, self.step_size)
                if not problem.point_is_free(candidate):
                    continue
                if not problem.segment_is_free(nodes[nearest], candidate):
                    continue
                parent_index = nearest
                candidate_cost = costs[nearest] + euclidean_distance(nodes[nearest], candidate)
                radius = max(self.step_size * 2.5, 12.0)
                near: list[int] = []
                if self.rewire:
                    near = [
                        index
                        for index, node in enumerate(nodes)
                        if euclidean_distance(node, candidate) <= radius
                    ]
                    for index in near:
                        trial = costs[index] + euclidean_distance(nodes[index], candidate)
                        if trial < candidate_cost and problem.segment_is_free(
                            nodes[index], candidate
                        ):
                            parent_index = index
                            candidate_cost = trial
                nodes.append(candidate)
                parents.append(parent_index)
                costs.append(candidate_cost)
                new_index = len(nodes) - 1
                problem.expand()
                if self.rewire:
                    for index in near:
                        trial = candidate_cost + euclidean_distance(candidate, nodes[index])
                        if trial + 1e-12 < costs[index] and problem.segment_is_free(
                            candidate, nodes[index]
                        ):
                            parents[index] = new_index
                            costs[index] = trial
                if euclidean_distance(candidate, env.goal) <= self.step_size * 1.5:
                    if not problem.segment_is_free(candidate, env.goal):
                        continue
                    path = simplify_path(
                        _tree_path(nodes, parents, new_index, env.goal),
                        problem,
                    )
                    evaluation = problem.evaluate(path)
                    if evaluation.feasible and evaluation.total_cost < best_cost:
                        best_path = path
                        best_cost = evaluation.total_cost
                    if not self.rewire:
                        return problem.result(
                            self.name,
                            "success",
                            path=best_path,
                            diagnostics={"tree_nodes": len(nodes)},
                        )
            return problem.result(
                self.name,
                "success" if best_path is not None else "no_path",
                path=best_path,
                message="" if best_path is not None else "sampling limit reached",
                diagnostics={"tree_nodes": len(nodes)},
            )
        except PlanningTimeout:
            return problem.result(
                self.name,
                "timeout",
                path=best_path,
                message="wall-clock planning budget exhausted",
                diagnostics={"tree_nodes": len(nodes)},
            )
        except ObjectiveBudgetExhausted:
            return problem.result(
                self.name,
                "budget_exhausted",
                path=best_path,
                message="objective-evaluation budget exhausted",
                diagnostics={"tree_nodes": len(nodes)},
            )


@dataclass
class PRMPlanner:
    name: str = "prm"
    sample_count: int = 180
    neighbor_count: int = 10
    stochastic: bool = True
    research_claim_eligible: bool = True

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        del budget
        env = problem.environment
        if problem.segment_is_free(env.start, env.goal):
            return problem.result(self.name, "success", path=[env.start, env.goal])
        points: list[Waypoint] = [env.start, env.goal]
        attempts = 0
        while len(points) < self.sample_count + 2 and attempts < self.sample_count * 30:
            attempts += 1
            candidate = (
                float(rng.uniform(0.0, env.width)),
                float(rng.uniform(0.0, env.height)),
            )
            if problem.point_is_free(candidate):
                points.append(candidate)
        graph: list[list[tuple[int, float]]] = [[] for _ in points]
        for index, point in enumerate(points):
            neighbors = sorted(
                (
                    (euclidean_distance(point, other), other_index)
                    for other_index, other in enumerate(points)
                    if other_index != index
                ),
                key=lambda item: (item[0], item[1]),
            )[: self.neighbor_count]
            for distance, other_index in neighbors:
                if any(vertex == other_index for vertex, _ in graph[index]):
                    continue
                if problem.segment_is_free(point, points[other_index]):
                    graph[index].append((other_index, distance))
                    graph[other_index].append((index, distance))
        frontier = [(0.0, 0)]
        costs = {0: 0.0}
        parent: dict[int, int] = {}
        closed: set[int] = set()
        while frontier:
            current_cost, current = heapq.heappop(frontier)
            if current in closed:
                continue
            closed.add(current)
            problem.expand()
            if current == 1:
                indices = [current]
                while current in parent:
                    current = parent[current]
                    indices.append(current)
                indices.reverse()
                path = simplify_path([points[index] for index in indices], problem)
                return problem.result(
                    self.name,
                    "success",
                    path=path,
                    diagnostics={"roadmap_nodes": len(points)},
                )
            for neighbor, edge_cost in graph[current]:
                tentative = current_cost + edge_cost
                if tentative + 1e-12 >= costs.get(neighbor, math.inf):
                    continue
                costs[neighbor] = tentative
                parent[neighbor] = current
                heapq.heappush(frontier, (tentative, neighbor))
        return problem.result(
            self.name,
            "no_path",
            message="roadmap did not connect start to goal",
            diagnostics={"roadmap_nodes": len(points)},
        )


def _resample_intermediate(path: Path, count: int) -> np.ndarray:
    cumulative = [0.0]
    for start, end in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + euclidean_distance(start, end))
    total = cumulative[-1]
    points: list[Waypoint] = []
    for target in np.linspace(0.0, total, count + 2)[1:-1]:
        segment = next(
            index
            for index in range(len(cumulative) - 1)
            if cumulative[index + 1] + 1e-12 >= target
        )
        start, end = path[segment], path[segment + 1]
        span = cumulative[segment + 1] - cumulative[segment]
        ratio = 0.0 if span <= 1e-12 else (target - cumulative[segment]) / span
        points.append(
            (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        )
    return np.asarray(points, dtype=float)


def _genome_path(genome: np.ndarray, problem: BudgetedEvaluator) -> Path:
    env = problem.environment
    clipped = genome.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0.0, env.width)
    clipped[:, 1] = np.clip(clipped[:, 1], 0.0, env.height)
    return [env.start, *[(float(x), float(y)) for x, y in clipped], env.goal]


def _initial_population(
    problem: BudgetedEvaluator,
    rng: np.random.Generator,
    population_size: int,
    waypoint_count: int,
) -> np.ndarray:
    seed_path = _grid_search(problem, "astar", 2.0)
    if seed_path is None:
        seed_path = [problem.environment.start, problem.environment.goal]
    seed = _resample_intermediate(seed_path, waypoint_count)
    population = np.repeat(seed[None, :, :], population_size, axis=0)
    scale = np.asarray(
        [0.06 * problem.environment.width, 0.06 * problem.environment.height]
    )
    if population_size > 1:
        population[1:] += rng.normal(
            0.0,
            scale,
            size=(population_size - 1, waypoint_count, 2),
        )
    population[:, :, 0] = np.clip(population[:, :, 0], 0.0, problem.environment.width)
    population[:, :, 1] = np.clip(population[:, :, 1], 0.0, problem.environment.height)
    return population


def _evaluate_genome(
    genome: np.ndarray,
    problem: BudgetedEvaluator,
) -> tuple[float, EvaluationResult, Path]:
    path = _genome_path(genome, problem)
    evaluation = problem.evaluate(path)
    return evaluation.total_cost, evaluation, path


def _evaluate_population(
    population: np.ndarray,
    problem: BudgetedEvaluator,
) -> tuple[np.ndarray, list[EvaluationResult], list[Path]]:
    costs: list[float] = []
    evaluations: list[EvaluationResult] = []
    paths: list[Path] = []
    for genome in population:
        cost, evaluation, path = _evaluate_genome(genome, problem)
        costs.append(cost)
        evaluations.append(evaluation)
        paths.append(path)
    return np.asarray(costs), evaluations, paths


@dataclass
class PopulationPlanner:
    """Shared continuous waypoint representation for GA, PSO, DE, and ACOR."""

    name: str
    population_size: int = 32
    waypoint_count: int = 10
    max_generations: int = 20
    stochastic: bool = True
    research_claim_eligible: bool = True

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        population_size = min(self.population_size, budget.max_objective_evaluations)
        population = _initial_population(
            problem,
            rng,
            population_size,
            self.waypoint_count,
        )
        best_path: Path | None = None
        best_cost = math.inf
        best_genome = population[0].copy()
        state: dict[str, np.ndarray] = {}
        try:
            costs, evaluations, paths = _evaluate_population(population, problem)
            for cost, evaluation, path, genome in zip(
                costs, evaluations, paths, population
            ):
                if evaluation.feasible and cost < best_cost:
                    best_cost, best_path, best_genome = float(cost), path, genome.copy()
            if self.name == "pso":
                state["velocity"] = np.zeros_like(population)
                state["personal"] = population.copy()
                state["personal_cost"] = costs.copy()
            generation = 0
            while generation < self.max_generations:
                problem.check_time()
                generation += 1
                if self.name == "ga":
                    population = self._ga_generation(population, costs, rng, problem)
                elif self.name == "pso":
                    population = self._pso_generation(
                        population,
                        costs,
                        best_genome,
                        state,
                        rng,
                        problem,
                    )
                elif self.name == "de":
                    population, costs, evaluations, paths = self._de_generation(
                        population,
                        costs,
                        evaluations,
                        paths,
                        rng,
                        problem,
                    )
                elif self.name == "aco_acor":
                    population = self._acor_generation(population, costs, rng)
                else:
                    raise ValueError(f"unknown population planner: {self.name}")
                if self.name != "de":
                    costs, evaluations, paths = _evaluate_population(population, problem)
                for cost, evaluation, path, genome in zip(
                    costs, evaluations, paths, population
                ):
                    if evaluation.feasible and cost < best_cost:
                        best_cost = float(cost)
                        best_path = path
                        best_genome = genome.copy()
                problem.expand(population_size)
            status = "success"
            message = ""
        except PlanningTimeout:
            status = "timeout"
            message = "wall-clock planning budget exhausted"
        except ObjectiveBudgetExhausted:
            status = "budget_exhausted"
            message = "objective-evaluation budget exhausted"
        if best_path is not None:
            try:
                best_path = simplify_path(best_path, problem)
            except PlanningTimeout:
                pass
        return problem.result(
            self.name,
            status,
            path=best_path,
            message=message,
            diagnostics={"best_internal_cost": best_cost},
        )

    @staticmethod
    def _ga_generation(
        population: np.ndarray,
        costs: np.ndarray,
        rng: np.random.Generator,
        problem: BudgetedEvaluator,
    ) -> np.ndarray:
        offspring = [population[int(np.argmin(costs))].copy()]
        scale = np.asarray(
            [0.03 * problem.environment.width, 0.03 * problem.environment.height]
        )
        while len(offspring) < len(population):
            candidates = rng.integers(0, len(population), size=(2, 3))
            first = population[candidates[0, np.argmin(costs[candidates[0]])]]
            second = population[candidates[1, np.argmin(costs[candidates[1]])]]
            alpha = rng.uniform(0.0, 1.0, size=first.shape)
            child = alpha * first + (1.0 - alpha) * second
            mask = rng.random(size=child.shape) < 0.15
            child += mask * rng.normal(0.0, scale, size=child.shape)
            offspring.append(child)
        return np.asarray(offspring)

    @staticmethod
    def _pso_generation(
        population: np.ndarray,
        costs: np.ndarray,
        global_best: np.ndarray,
        state: dict[str, np.ndarray],
        rng: np.random.Generator,
        problem: BudgetedEvaluator,
    ) -> np.ndarray:
        personal = state["personal"]
        personal_cost = state["personal_cost"]
        improved = costs < personal_cost
        personal[improved] = population[improved]
        personal_cost[improved] = costs[improved]
        velocity = state["velocity"]
        constriction = 0.72984
        coefficient = 2.05
        velocity[:] = constriction * (
            velocity
            + coefficient * rng.random(size=population.shape) * (personal - population)
            + coefficient * rng.random(size=population.shape) * (global_best - population)
        )
        population = population + velocity
        population[:, :, 0] = np.clip(
            population[:, :, 0], 0.0, problem.environment.width
        )
        population[:, :, 1] = np.clip(
            population[:, :, 1], 0.0, problem.environment.height
        )
        return population

    @staticmethod
    def _de_generation(
        population: np.ndarray,
        costs: np.ndarray,
        evaluations: list[EvaluationResult],
        paths: list[Path],
        rng: np.random.Generator,
        problem: BudgetedEvaluator,
    ) -> tuple[np.ndarray, np.ndarray, list[EvaluationResult], list[Path]]:
        next_population = population.copy()
        next_costs = costs.copy()
        next_evaluations = list(evaluations)
        next_paths = list(paths)
        indices = np.arange(len(population))
        for target_index in indices:
            choices = indices[indices != target_index]
            a, b, c = rng.choice(choices, size=3, replace=False)
            mutant = population[a] + 0.7 * (population[b] - population[c])
            mask = rng.random(size=mutant.shape) < 0.8
            forced = tuple(rng.integers(0, size) for size in mutant.shape)
            mask[forced] = True
            trial = np.where(mask, mutant, population[target_index])
            trial[:, 0] = np.clip(trial[:, 0], 0.0, problem.environment.width)
            trial[:, 1] = np.clip(trial[:, 1], 0.0, problem.environment.height)
            trial_cost, trial_evaluation, trial_path = _evaluate_genome(trial, problem)
            if trial_cost < costs[target_index]:
                next_population[target_index] = trial
                next_costs[target_index] = trial_cost
                next_evaluations[target_index] = trial_evaluation
                next_paths[target_index] = trial_path
        return next_population, next_costs, next_evaluations, next_paths

    @staticmethod
    def _acor_generation(
        population: np.ndarray,
        costs: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        order = np.argsort(costs)
        archive = population[order]
        ranks = np.arange(len(archive), dtype=float)
        q = 0.35
        weights = np.exp(-(ranks**2) / (2.0 * (q * len(archive)) ** 2))
        weights /= weights.sum()
        offspring = [archive[0].copy()]
        while len(offspring) < len(archive):
            selected = int(rng.choice(len(archive), p=weights))
            center = archive[selected]
            deviations = np.mean(np.abs(archive - center), axis=0) + 1e-6
            offspring.append(center + rng.normal(0.0, 0.85 * deviations))
        return np.asarray(offspring)


@dataclass
class AFLUAVMockPlanner(PopulationPlanner):
    """Deterministic offline proxy used only to validate the AFL execution arm."""

    name: str = "afl_uav_mock"
    stochastic: bool = False
    research_claim_eligible: bool = False

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        # The adapter follows the generated mock solver's A* seed + bounded
        # Gaussian local-search shape, while remaining inside the trusted
        # evaluator and budget contract.
        seed_path = _grid_search(problem, "astar", 2.0)
        if seed_path is None:
            return problem.result(self.name, "no_path", message="mock A* seed failed")
        genome = _resample_intermediate(seed_path, self.waypoint_count)
        best_cost, best_eval, best_path = _evaluate_genome(genome, problem)
        scale = np.asarray(
            [0.025 * problem.environment.width, 0.025 * problem.environment.height]
        )
        try:
            for _ in range(min(256, budget.max_objective_evaluations - 1)):
                candidate = genome + rng.normal(0.0, scale, size=genome.shape)
                cost, evaluation, path = _evaluate_genome(candidate, problem)
                if evaluation.feasible and cost < best_cost:
                    genome, best_cost, best_eval, best_path = (
                        candidate,
                        cost,
                        evaluation,
                        path,
                    )
                problem.expand()
        except PlanningTimeout:
            status, message = "timeout", "wall-clock planning budget exhausted"
        except ObjectiveBudgetExhausted:
            status, message = (
                "budget_exhausted",
                "objective-evaluation budget exhausted",
            )
        else:
            status, message = "success", ""
        if not best_eval.feasible:
            return problem.result(self.name, status, message=message)
        try:
            best_path = simplify_path(best_path, problem)
        except PlanningTimeout:
            pass
        return problem.result(
            self.name,
            status,
            path=best_path,
            message=message,
            diagnostics={"research_claim_eligible": False},
        )


def build_planners(
    *,
    grid_resolution: float = 2.0,
    population_size: int = 32,
    waypoint_count: int = 10,
    afl_artifact_path: str | None = None,
    afl_artifacts: dict[str, str] | None = None,
    evolutionary_afl_artifacts: dict[str, str] | None = None,
) -> dict[str, object]:
    """Create fresh planner instances keyed by stable machine identifiers."""

    planners: dict[str, object] = {
        "dijkstra": GridPlanner("dijkstra", grid_resolution),
        "astar": GridPlanner("astar", grid_resolution),
        "theta_star": GridPlanner("theta_star", grid_resolution),
        "rrt": RRTPlanner("rrt"),
        "rrt_star": RRTPlanner("rrt_star", rewire=True),
        "prm": PRMPlanner(),
        "ga": PopulationPlanner("ga", population_size, waypoint_count),
        "pso": PopulationPlanner("pso", population_size, waypoint_count),
        "de": PopulationPlanner("de", population_size, waypoint_count),
        "aco_acor": PopulationPlanner(
            "aco_acor", population_size, waypoint_count
        ),
        "afl_uav_mock": AFLUAVMockPlanner(
            population_size=population_size,
            waypoint_count=waypoint_count,
        ),
    }
    def validate_arm_id(arm_id: str) -> None:
        if (
            not arm_id
            or len(arm_id) > 64
            or arm_id[0] not in "abcdefghijklmnopqrstuvwxyz0123456789"
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_.-"
                for character in arm_id
            )
        ):
            raise ValueError(
                "AFL arm_id must start with a lowercase letter or digit, use only "
                "lowercase letters, digits, '_', '.', or '-', and be at most 64 characters"
            )

    artifact_arms = dict(afl_artifacts or {})
    if afl_artifact_path is not None:
        if "afl_uav" in artifact_arms:
            raise ValueError("legacy AFL artifact conflicts with arm_id 'afl_uav'")
        artifact_arms["afl_uav"] = afl_artifact_path
    if artifact_arms:
        from .afl_planner import FrozenAFLUAVPlanner

        for arm_id, artifact_path in sorted(artifact_arms.items()):
            validate_arm_id(arm_id)
            registry_key = (
                "afl_uav" if arm_id == "afl_uav" else f"afl_uav:{arm_id}"
            )
            planners[registry_key] = FrozenAFLUAVPlanner(
                artifact_path,
                arm_id=arm_id,
            )
    evolutionary_arms = dict(evolutionary_afl_artifacts or {})
    if evolutionary_arms:
        from .evolutionary_afl import EvolutionaryAFLUAVPlanner

        for arm_id, artifact_path in sorted(evolutionary_arms.items()):
            validate_arm_id(arm_id)
            registry_key = (
                "evolutionary_afl_uav"
                if arm_id == "evolutionary_afl_uav"
                else f"evolutionary_afl_uav:{arm_id}"
            )
            planners[registry_key] = EvolutionaryAFLUAVPlanner(
                artifact_path,
                arm_id=arm_id,
                population_size=population_size,
                archive_size=min(8, population_size),
            )
    return planners


__all__ = [
    "AFLUAVMockPlanner",
    "GridPlanner",
    "PRMPlanner",
    "PopulationPlanner",
    "RRTPlanner",
    "build_planners",
]
