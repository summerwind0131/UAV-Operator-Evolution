"""Offline quality-diversity evolution around a frozen AFL-UAV solver.

The frozen, agent-generated solver supplies one qualified feasible seed.  This
module never calls an LLM: it spends only the shared planning-time and trusted
objective-evaluation budgets on variable-length path evolution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..path.models import EvaluationResult, Path as UAVPath, Waypoint
from .afl_planner import FrozenAFLUAVPlanner
from .core import (
    BudgetedEvaluator,
    ObjectiveBudgetExhausted,
    PlannerResult,
    PlanningBudget,
    PlanningTimeout,
    path_hash,
)


@dataclass(frozen=True)
class _Individual:
    path: UAVPath
    evaluation: EvaluationResult

    @property
    def key(self) -> tuple[tuple[float, float], ...]:
        return _path_key(self.path)

    @property
    def objectives(self) -> tuple[float, float, float, float]:
        """Unweighted objectives used by the Pareto archive."""

        return (
            self.evaluation.path_length,
            self.evaluation.risk_penalty,
            self.evaluation.smoothness_penalty,
            self.evaluation.waypoint_penalty,
        )


def _path_key(path: UAVPath) -> tuple[tuple[float, float], ...]:
    return tuple((round(float(x), 9), round(float(y), 9)) for x, y in path)


def _deduplicate_consecutive(path: UAVPath) -> UAVPath:
    result: UAVPath = []
    for point in path:
        canonical = (float(point[0]), float(point[1]))
        if not result or math.dist(result[-1], canonical) > 1e-9:
            result.append(canonical)
    return result


def _resample_path(path: UAVPath, count: int = 12) -> np.ndarray:
    """Return a fixed-size geometric signature for variable-length paths."""

    if len(path) < 2:
        raise ValueError("path signature requires at least two points")
    cumulative = [0.0]
    for start, end in zip(path, path[1:]):
        cumulative.append(cumulative[-1] + math.dist(start, end))
    total = cumulative[-1]
    if total <= 1e-12:
        return np.repeat(np.asarray(path[:1], dtype=float), count, axis=0)
    signature: list[Waypoint] = []
    segment = 0
    for target in np.linspace(0.0, total, count):
        while (
            segment < len(cumulative) - 2
            and cumulative[segment + 1] + 1e-12 < target
        ):
            segment += 1
        span = cumulative[segment + 1] - cumulative[segment]
        ratio = 0.0 if span <= 1e-12 else (target - cumulative[segment]) / span
        start, end = path[segment], path[segment + 1]
        signature.append(
            (
                start[0] + ratio * (end[0] - start[0]),
                start[1] + ratio * (end[1] - start[1]),
            )
        )
    return np.asarray(signature, dtype=float)


def _dominates(first: _Individual, second: _Individual) -> bool:
    first_values = first.objectives
    second_values = second.objectives
    return all(a <= b + 1e-12 for a, b in zip(first_values, second_values)) and any(
        a < b - 1e-12 for a, b in zip(first_values, second_values)
    )


@dataclass
class EvolutionaryAFLUAVPlanner:
    """Variable-length, multiobjective evolution seeded by frozen AFL-UAV.

    The archive is deliberately quality-diverse: it always retains the best
    trusted weighted-cost path, then Pareto-efficient and geometrically novel
    alternatives.  The final result remains the minimum trusted weighted cost
    so comparison with every baseline uses the exact same ranking objective.
    """

    artifact_path: str | Path
    arm_id: str = "evolutionary_afl_uav"
    population_size: int = 32
    archive_size: int = 8
    max_generations: int = 20
    max_waypoints: int = 64
    base_iteration_limit: int = 64
    crossover_probability: float = 0.40
    extra_mutation_probability: float = 0.30
    name: str = field(default="evolutionary_afl_uav", init=False)
    stochastic: bool = field(default=True, init=False)
    research_claim_eligible: bool = field(default=False, init=False)
    base_planner: FrozenAFLUAVPlanner = field(init=False, repr=False)
    artifact: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.population_size < 4:
            raise ValueError("evolutionary AFL-UAV population_size must be at least 4")
        if not 2 <= self.archive_size <= self.population_size:
            raise ValueError("archive_size must be between 2 and population_size")
        if self.max_generations < 1:
            raise ValueError("max_generations must be positive")
        self.base_planner = FrozenAFLUAVPlanner(
            self.artifact_path,
            arm_id=f"{self.arm_id}.seed",
            iteration_limit=self.base_iteration_limit,
        )
        self.artifact = self.base_planner.artifact
        self.research_claim_eligible = self.base_planner.research_claim_eligible
        self.max_waypoints = min(self.max_waypoints, self.artifact.max_waypoints)

    @property
    def algorithm_parameters(self) -> dict[str, Any]:
        return {
            "population_size": self.population_size,
            "archive_size": self.archive_size,
            "max_generations": self.max_generations,
            "max_waypoints": self.max_waypoints,
            "base_iteration_limit": self.base_iteration_limit,
            "crossover_probability": self.crossover_probability,
            "extra_mutation_probability": self.extra_mutation_probability,
            "representation": "variable_length_waypoints",
            "archive": "pareto_quality_diversity",
            "operators": ["insert", "delete", "move", "swap", "crossover"],
            "llm_calls_during_planning": 0,
        }

    def plan(
        self,
        problem: BudgetedEvaluator,
        budget: PlanningBudget,
        rng: np.random.Generator,
    ) -> PlannerResult:
        base_result = self.base_planner.plan(problem, budget, rng)
        diagnostics: dict[str, Any] = {
            "arm_id": self.arm_id,
            "artifact_id": self.artifact.artifact_id,
            "solver_hash": self.artifact.solver_hash,
            "artifact_provider": self.artifact.provider,
            "artifact_model": self.artifact.model,
            "base_status": base_result.status,
            "base_diagnostics": base_result.diagnostics,
            "algorithm": self.algorithm_parameters,
            "rooms_maze_mode": problem.environment.difficulty == "rooms_maze",
            "operator_attempts": {
                "insert": 0,
                "delete": 0,
                "move": 0,
                "swap": 0,
                "crossover": 0,
            },
            "operator_successes": {
                "insert": 0,
                "delete": 0,
                "move": 0,
                "swap": 0,
                "crossover": 0,
            },
        }
        if base_result.path is None:
            return problem.result(
                self.name,
                base_result.status,
                message="frozen AFL-UAV did not provide a seed path: " + base_result.message,
                diagnostics=diagnostics,
            )

        seed_path = _deduplicate_consecutive(list(base_result.path))
        try:
            seed_evaluation = problem.evaluate(seed_path)
        except PlanningTimeout:
            return problem.result(
                self.name,
                "timeout",
                path=seed_path,
                message="time budget ended after frozen AFL-UAV seed generation",
                diagnostics=diagnostics,
            )
        except ObjectiveBudgetExhausted:
            return problem.result(
                self.name,
                "budget_exhausted",
                path=seed_path,
                message="evaluation budget ended after frozen AFL-UAV seed generation",
                diagnostics=diagnostics,
            )
        if not seed_evaluation.feasible:
            return problem.result(
                self.name,
                "invalid_path",
                path=seed_path,
                message="frozen AFL-UAV seed failed the shared evaluator",
                diagnostics=diagnostics,
            )

        seed = _Individual(seed_path, seed_evaluation)
        cache: dict[tuple[tuple[float, float], ...], _Individual] = {seed.key: seed}
        population = [seed]
        archive = [seed]
        best = seed
        generations = 0
        accepted_offspring = 0
        duplicate_proposals = 0
        status = "success"
        message = ""
        stop_reason = "max_generations"
        # Cooperative planners need a small margin for archive selection and
        # result serialization after their last geometry/objective call.
        soft_deadline = max(0.01, 0.88 * budget.time_limit_seconds)
        generation_cap = self._generation_cap(problem)
        diagnostics["generation_cap"] = generation_cap

        try:
            population, accepted, duplicates = self._initialize_population(
                seed,
                problem,
                rng,
                cache,
                diagnostics,
                soft_deadline,
            )
            accepted_offspring += accepted
            duplicate_proposals += duplicates
            archive = self._update_archive(population, problem.environment.diagonal)
            best = min(population, key=lambda item: item.evaluation.total_cost)

            stagnant_generations = 0
            for generation in range(generation_cap):
                problem.check_time()
                if problem.elapsed_seconds >= soft_deadline:
                    stop_reason = "soft_time_guard"
                    break
                generations = generation + 1
                children: list[_Individual] = []
                new_unique = 0
                operator_order = list(rng.permutation(["insert", "delete", "move", "swap"]))
                attempts = 0
                max_attempts = self.population_size * 8
                soft_stop = False
                while len(children) < self.population_size and attempts < max_attempts:
                    problem.check_time()
                    if problem.elapsed_seconds >= soft_deadline:
                        soft_stop = True
                        break
                    attempts += 1
                    first = self._select_parent(population, archive, rng)
                    candidate = list(first.path)
                    changed = False
                    if rng.random() < self.crossover_probability and len(population) > 1:
                        second = self._select_parent(population, archive, rng)
                        diagnostics["operator_attempts"]["crossover"] += 1
                        candidate, crossed = self._crossover(
                            first.path,
                            second.path,
                            problem,
                            rng,
                        )
                        if crossed:
                            diagnostics["operator_successes"]["crossover"] += 1
                            changed = True

                    operator_name = operator_order[(attempts - 1) % len(operator_order)]
                    diagnostics["operator_attempts"][operator_name] += 1
                    candidate, mutated = self._apply_operator(
                        operator_name,
                        candidate,
                        problem,
                        rng,
                    )
                    if mutated:
                        diagnostics["operator_successes"][operator_name] += 1
                        changed = True

                    if changed and rng.random() < self.extra_mutation_probability:
                        extra_name = str(rng.choice(operator_order))
                        diagnostics["operator_attempts"][extra_name] += 1
                        candidate, extra_changed = self._apply_operator(
                            extra_name,
                            candidate,
                            problem,
                            rng,
                        )
                        if extra_changed:
                            diagnostics["operator_successes"][extra_name] += 1

                    candidate = _deduplicate_consecutive(candidate)
                    if not changed or not self._valid_shape(candidate, problem):
                        continue
                    key = _path_key(candidate)
                    if key in cache:
                        duplicate_proposals += 1
                        continue
                    evaluation = problem.evaluate(candidate)
                    individual = _Individual(candidate, evaluation)
                    cache[key] = individual
                    problem.expand()
                    new_unique += 1
                    if evaluation.feasible:
                        children.append(individual)
                        accepted_offspring += 1
                        if evaluation.total_cost < best.evaluation.total_cost:
                            best = individual

                combined = [*population, *children, *archive]
                archive = self._update_archive(combined, problem.environment.diagonal)
                population = self._select_survivors(
                    combined,
                    archive,
                    problem.environment.diagonal,
                )
                if new_unique == 0:
                    stagnant_generations += 1
                else:
                    stagnant_generations = 0
                if stagnant_generations >= 3:
                    stop_reason = "no_new_unique_paths"
                    break
                if soft_stop:
                    stop_reason = "soft_time_guard"
                    break
        except PlanningTimeout:
            status = "timeout"
            message = "wall-clock planning budget exhausted; returning best-so-far"
            stop_reason = "time_limit"
        except ObjectiveBudgetExhausted:
            status = "budget_exhausted"
            message = "objective-evaluation budget exhausted; returning best-so-far"
            stop_reason = "evaluation_limit"

        all_final = [best, *archive, *population]
        best = min(all_final, key=lambda item: item.evaluation.total_cost)
        archive_hashes = [path_hash(item.path) for item in archive]
        diagnostics.update(
            {
                "generations": generations,
                "stop_reason": stop_reason,
                "unique_candidates_evaluated": len(cache),
                "accepted_offspring": accepted_offspring,
                "duplicate_proposals": duplicate_proposals,
                "archive_size": len(archive),
                "archive_unique_paths": len(set(archive_hashes)),
                "archive_path_hashes": archive_hashes,
                "seed_cost": seed.evaluation.total_cost,
                "best_cost": best.evaluation.total_cost,
                "absolute_improvement": (
                    seed.evaluation.total_cost - best.evaluation.total_cost
                ),
                "relative_improvement": (
                    0.0
                    if seed.evaluation.total_cost <= 1e-12
                    else (seed.evaluation.total_cost - best.evaluation.total_cost)
                    / seed.evaluation.total_cost
                ),
                "best_components": {
                    "length": best.evaluation.path_length,
                    "risk": best.evaluation.risk_penalty,
                    "smoothness": best.evaluation.smoothness_penalty,
                    "waypoints": best.evaluation.waypoint_penalty,
                },
                "llm_calls_during_planning": 0,
            }
        )
        return problem.result(
            self.name,
            status,
            path=best.path,
            message=message,
            diagnostics=diagnostics,
        )

    def _generation_cap(self, problem: BudgetedEvaluator) -> int:
        """Deterministic workload cap with extra margin for geometry-heavy maps."""

        difficulty = problem.environment.difficulty
        if difficulty == "rooms_maze":
            return min(self.max_generations, 6)
        if difficulty == "mixed":
            return min(self.max_generations, 12)
        if difficulty == "corridor":
            return min(self.max_generations, 14)
        return self.max_generations

    def _initialize_population(
        self,
        seed: _Individual,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
        cache: dict[tuple[tuple[float, float], ...], _Individual],
        diagnostics: dict[str, Any],
        soft_deadline: float,
    ) -> tuple[list[_Individual], int, int]:
        population = [seed]
        accepted = 0
        duplicates = 0
        operators = list(rng.permutation(["insert", "delete", "move", "swap"]))
        attempts = 0
        while len(population) < self.population_size and attempts < self.population_size * 10:
            problem.check_time()
            if problem.elapsed_seconds >= soft_deadline:
                break
            attempts += 1
            parent = population[int(rng.integers(0, len(population)))]
            operator_name = operators[(attempts - 1) % len(operators)]
            diagnostics["operator_attempts"][operator_name] += 1
            candidate, changed = self._apply_operator(
                operator_name,
                parent.path,
                problem,
                rng,
            )
            if changed:
                diagnostics["operator_successes"][operator_name] += 1
            candidate = _deduplicate_consecutive(candidate)
            if not changed or not self._valid_shape(candidate, problem):
                continue
            key = _path_key(candidate)
            if key in cache:
                duplicates += 1
                continue
            evaluation = problem.evaluate(candidate)
            individual = _Individual(candidate, evaluation)
            cache[key] = individual
            problem.expand()
            if evaluation.feasible:
                population.append(individual)
                accepted += 1
        return population, accepted, duplicates

    def _apply_operator(
        self,
        name: str,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if name == "insert":
            return self._mutate_insert(path, problem, rng)
        if name == "delete":
            return self._mutate_delete(path, problem, rng)
        if name == "move":
            return self._mutate_move(path, problem, rng)
        if name == "swap":
            return self._mutate_swap(path, problem, rng)
        raise ValueError(f"unknown evolutionary AFL-UAV operator: {name}")

    def _valid_shape(self, path: UAVPath, problem: BudgetedEvaluator) -> bool:
        environment = problem.environment
        return (
            2 <= len(path) <= self.max_waypoints
            and math.dist(path[0], environment.start) <= 1e-7
            and math.dist(path[-1], environment.goal) <= 1e-7
            and all(environment.in_bounds(point) for point in path)
        )

    def _mutate_insert(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if len(path) >= self.max_waypoints:
            return list(path), False
        indices = list(rng.permutation(len(path) - 1))
        diagonal = problem.environment.diagonal
        rooms_mode = problem.environment.difficulty == "rooms_maze"
        for segment_index in indices[: min(8, len(indices))]:
            start, end = path[segment_index], path[segment_index + 1]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-9:
                continue
            perpendicular = (-dy / length, dx / length)
            base_scale = min(0.12 * length, (0.012 if rooms_mode else 0.025) * diagonal)
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
                if problem.segment_is_free(start, point) and problem.segment_is_free(point, end):
                    return [*path[: segment_index + 1], point, *path[segment_index + 1 :]], True
        return list(path), False

    def _mutate_delete(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if len(path) <= 2:
            return list(path), False
        internal_count = len(path) - 2
        rooms_mode = problem.environment.difficulty == "rooms_maze"
        attempts: list[tuple[int, int]] = []
        if rooms_mode and internal_count >= 2:
            for _ in range(min(6, internal_count * 2)):
                first = int(rng.integers(1, len(path) - 1))
                maximum = min(len(path) - 2, first + max(1, internal_count // 3))
                last = int(rng.integers(first, maximum + 1))
                attempts.append((first, last))
        attempts.extend((index, index) for index in rng.permutation(np.arange(1, len(path) - 1)))
        for first, last in attempts:
            if problem.segment_is_free(path[first - 1], path[last + 1]):
                return [*path[:first], *path[last + 1 :]], True
        return list(path), False

    def _mutate_move(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if len(path) <= 2:
            return list(path), False
        environment = problem.environment
        rooms_mode = environment.difficulty == "rooms_maze"
        indices = list(rng.permutation(np.arange(1, len(path) - 1)))
        for index in indices[: min(10, len(indices))]:
            previous, current, following = path[index - 1], path[index], path[index + 1]
            midpoint = (
                0.5 * (previous[0] + following[0]),
                0.5 * (previous[1] + following[1]),
            )
            proposals: list[Waypoint] = []
            # Elastic corner relaxation is particularly effective in rooms/maze:
            # it shortens grid corners without attempting to change corridor topology.
            for alpha in (0.15, 0.30, 0.50, 0.70):
                proposals.append(
                    (
                        current[0] + alpha * (midpoint[0] - current[0]),
                        current[1] + alpha * (midpoint[1] - current[1]),
                    )
                )
            scale = (0.012 if rooms_mode else 0.035) * environment.diagonal
            for factor in (1.0, 0.5, 0.25, 0.125):
                delta = rng.normal(0.0, max(0.08, scale * factor), size=2)
                proposals.append((current[0] + float(delta[0]), current[1] + float(delta[1])))
            for point in proposals:
                if math.dist(point, current) <= 1e-9 or not environment.in_bounds(point):
                    continue
                if problem.segment_is_free(previous, point) and problem.segment_is_free(point, following):
                    candidate = list(path)
                    candidate[index] = point
                    return candidate, True
        return list(path), False

    def _mutate_swap(
        self,
        path: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if len(path) < 4:
            return list(path), False
        internal = np.arange(1, len(path) - 1)
        for _ in range(min(12, len(internal) * 3)):
            first, second = sorted(rng.choice(internal, size=2, replace=False).tolist())
            candidate = list(path)
            candidate[first], candidate[second] = candidate[second], candidate[first]
            affected = {
                segment
                for segment in (first - 1, first, second - 1, second)
                if 0 <= segment < len(candidate) - 1
            }
            if all(
                problem.segment_is_free(candidate[index], candidate[index + 1])
                for index in sorted(affected)
            ):
                return candidate, True
        return list(path), False

    def _crossover(
        self,
        first: UAVPath,
        second: UAVPath,
        problem: BudgetedEvaluator,
        rng: np.random.Generator,
    ) -> tuple[UAVPath, bool]:
        if len(first) < 3 or len(second) < 3:
            return list(first), False
        for _ in range(10):
            first_cut = int(rng.integers(1, len(first) - 1))
            second_cut = int(rng.integers(1, len(second) - 1))
            prefix = list(first[: first_cut + 1])
            suffix = list(second[second_cut:])
            if len(prefix) + len(suffix) > self.max_waypoints:
                continue
            if not problem.segment_is_free(prefix[-1], suffix[0]):
                continue
            candidate = _deduplicate_consecutive([*prefix, *suffix])
            if _path_key(candidate) != _path_key(first):
                return candidate, True
        return list(first), False

    @staticmethod
    def _select_parent(
        population: list[_Individual],
        archive: list[_Individual],
        rng: np.random.Generator,
    ) -> _Individual:
        if archive and rng.random() < 0.45:
            return archive[int(rng.integers(0, len(archive)))]
        pool = population or archive
        sample_size = min(3, len(pool))
        indices = rng.choice(len(pool), size=sample_size, replace=False)
        return min(
            (pool[int(index)] for index in indices),
            key=lambda item: item.evaluation.total_cost,
        )

    def _update_archive(
        self,
        candidates: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        unique = self._unique_feasible(candidates)
        if not unique:
            return []
        ordered = sorted(unique, key=lambda item: item.evaluation.total_cost)
        pareto = [
            candidate
            for candidate in ordered
            if not any(
                other.key != candidate.key and _dominates(other, candidate)
                for other in ordered
            )
        ]
        selected = [ordered[0]]
        pareto_keys = {item.key for item in pareto}
        preferred = [item for item in pareto if item.key != ordered[0].key]
        remaining = [
            item
            for item in ordered
            if item.key != ordered[0].key and item.key not in pareto_keys
        ]
        self._fill_diverse(selected, preferred, self.archive_size, diagonal, ordered[0])
        self._fill_diverse(selected, remaining, self.archive_size, diagonal, ordered[0])
        return selected[: self.archive_size]

    def _select_survivors(
        self,
        candidates: list[_Individual],
        archive: list[_Individual],
        diagonal: float,
    ) -> list[_Individual]:
        unique = self._unique_feasible(candidates)
        if len(unique) <= self.population_size:
            return unique
        best = min(unique, key=lambda item: item.evaluation.total_cost)
        selected = list(archive)
        selected_keys = {item.key for item in selected}
        remaining = [item for item in unique if item.key not in selected_keys]
        self._fill_diverse(selected, remaining, self.population_size, diagonal, best)
        return selected[: self.population_size]

    @staticmethod
    def _unique_feasible(candidates: list[_Individual]) -> list[_Individual]:
        unique: dict[tuple[tuple[float, float], ...], _Individual] = {}
        for candidate in candidates:
            if not candidate.evaluation.feasible:
                continue
            previous = unique.get(candidate.key)
            if previous is None or candidate.evaluation.total_cost < previous.evaluation.total_cost:
                unique[candidate.key] = candidate
        return list(unique.values())

    @staticmethod
    def _fill_diverse(
        selected: list[_Individual],
        candidates: list[_Individual],
        limit: int,
        diagonal: float,
        best: _Individual,
    ) -> None:
        available = list(candidates)
        if not available or len(selected) >= limit:
            return
        signatures = {
            item.key: _resample_path(item.path) for item in [*selected, *available]
        }
        cost_scale = max(1e-9, best.evaluation.total_cost)
        candidate_signatures = np.stack([signatures[item.key] for item in available])
        selected_signatures = np.stack([signatures[item.key] for item in selected])
        distances = np.sqrt(
            np.mean(
                np.sum(
                    (
                        candidate_signatures[:, None, :, :]
                        - selected_signatures[None, :, :, :]
                    )
                    ** 2,
                    axis=3,
                ),
                axis=2,
            )
        )
        minimum_distances = np.min(distances, axis=1) / max(diagonal, 1e-9)
        while len(selected) < limit and available:
            scores: list[tuple[float, float]] = []
            for index, candidate in enumerate(available):
                quality = 1.0 / (
                    1.0
                    + max(0.0, candidate.evaluation.total_cost - best.evaluation.total_cost)
                    / cost_scale
                )
                scores.append(
                    (
                        0.55 * quality + 0.45 * float(minimum_distances[index]),
                        -candidate.evaluation.total_cost,
                    )
                )
            chosen_index = max(range(len(available)), key=lambda index: scores[index])
            chosen = available[chosen_index]
            chosen_signature = candidate_signatures[chosen_index]
            selected.append(chosen)
            available.pop(chosen_index)
            candidate_signatures = np.delete(candidate_signatures, chosen_index, axis=0)
            minimum_distances = np.delete(minimum_distances, chosen_index)
            if available:
                distance_to_chosen = np.sqrt(
                    np.mean(
                        np.sum(
                            (candidate_signatures - chosen_signature[None, :, :]) ** 2,
                            axis=2,
                        ),
                        axis=1,
                    )
                ) / max(diagonal, 1e-9)
                minimum_distances = np.minimum(
                    minimum_distances,
                    distance_to_chosen,
                )


__all__ = ["EvolutionaryAFLUAVPlanner"]
