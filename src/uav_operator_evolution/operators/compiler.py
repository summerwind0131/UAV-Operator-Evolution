"""Trusted interpreter that compiles bounded OperatorSpec data to PathOperator."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import DSLConfig
from ..environment.environment import Environment2D, extract_environment_features
from ..path.models import Path
from ..search.context import SearchContext
from .base import OperatorResult, copied_path, unchanged_result
from .primitives import (
    delete_waypoint,
    generate_obstacle_detour,
    insert_waypoint,
    perturb_waypoint,
    reconstruct_segment,
    select_collision_segment,
    select_continuous_collision_region,
    select_high_curvature_waypoint,
    select_long_segment,
    select_low_clearance_segment,
    select_random_waypoint,
    shift_segment,
    shortcut_segment,
    smooth_segment,
    try_alternative_side,
)
from .specs import ConditionSpec, OperatorSpec, TransformationSpec


class OperatorCompilationError(ValueError):
    """Raised when a valid schema still violates configured compiler limits."""


def _changed_indices(before: Path, after: Path) -> tuple[int, ...]:
    shared = min(len(before), len(after))
    changed = [index for index in range(shared) if before[index] != after[index]]
    changed.extend(range(shared, max(len(before), len(after))))
    return tuple(changed)


def _path_metrics(path: Path, environment: Environment2D, context: SearchContext) -> dict[str, Any]:
    segment_lengths = [math.dist(a, b) for a, b in zip(path, path[1:])]
    turns: list[float] = []
    for previous, current, following in zip(path, path[1:], path[2:]):
        left = np.asarray(previous, dtype=float) - np.asarray(current, dtype=float)
        right = np.asarray(following, dtype=float) - np.asarray(current, dtype=float)
        norm = float(np.linalg.norm(left) * np.linalg.norm(right))
        turns.append(0.0 if norm <= 1e-12 else float(np.arccos(np.clip(np.dot(left, right) / norm, -1, 1))))
    env_features = extract_environment_features(environment)
    collisions = environment.colliding_segment_indices(path)
    search_values = {
        "iteration_ratio": getattr(context, "iteration_ratio", 0.0),
        "stagnation_count": getattr(context, "stagnation_count", 0),
    }
    return {
        "collision_count": len(collisions),
        "minimum_clearance": min(
            [environment.segment_clearance(a, b) for a, b in zip(path, path[1:])]
            or [environment.diagonal]
        ),
        "waypoint_count": len(path),
        "path_length": sum(segment_lengths),
        "maximum_turn_angle": max(turns, default=0.0),
        "smoothness": sum(angle * angle for angle in turns),
        "feasible": environment.path_is_collision_free(path),
        "obstacle_density": env_features.obstacle_density,
        "map_difficulty": environment.difficulty,
        **search_values,
    }


def _condition_matches(condition: ConditionSpec, values: dict[str, Any]) -> bool:
    actual = values.get(condition.feature)
    expected = condition.value
    operations = {
        "lt": lambda: actual < expected,
        "le": lambda: actual <= expected,
        "gt": lambda: actual > expected,
        "ge": lambda: actual >= expected,
        "eq": lambda: actual == expected,
        "ne": lambda: actual != expected,
    }
    try:
        return bool(operations[condition.operator]())
    except (TypeError, ValueError):
        return False


@dataclass(slots=True)
class CompiledOperator:
    """Executable interpreter over a validated, bounded DSL specification."""

    spec: OperatorSpec
    limits: DSLConfig

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def parent_operator_ids(self) -> tuple[str, ...]:
        return tuple(self.spec.parent_operators)

    @property
    def generation(self) -> int:
        value = self.spec.parameters.get("generation", 0)
        return int(value) if isinstance(value, (int, float)) else 0

    def _select(
        self, path: Path, environment: Environment2D, rng: np.random.Generator
    ) -> int | tuple[int, int] | None:
        kind = self.spec.selection_strategy.kind
        if kind == "select_random_waypoint":
            return select_random_waypoint(path, rng)
        if kind == "select_high_curvature_waypoint":
            return select_high_curvature_waypoint(path, rng)
        if kind == "select_collision_segment":
            return select_collision_segment(path, environment, rng)
        if kind == "select_long_segment":
            return select_long_segment(path, rng)
        if kind == "select_low_clearance_segment":
            return select_low_clearance_segment(path, environment, rng)
        if kind == "select_continuous_collision_region":
            return select_continuous_collision_region(path, environment, rng)
        raise OperatorCompilationError(f"unsupported selection primitive: {kind}")

    @staticmethod
    def _range(selection: int | tuple[int, int] | None, path: Path) -> tuple[int, int]:
        if isinstance(selection, tuple):
            return max(0, selection[0]), min(len(path) - 1, selection[1])
        if isinstance(selection, int):
            return max(0, selection - 1), min(len(path) - 1, selection + 1)
        return 0, max(0, len(path) - 1)

    def _transform(
        self,
        step: TransformationSpec,
        path: Path,
        selection: int | tuple[int, int] | None,
        environment: Environment2D,
        rng: np.random.Generator,
    ) -> Path:
        kind = step.kind
        if kind == "perturb_waypoint":
            index = selection if isinstance(selection, int) and 0 < selection < len(path) - 1 else select_random_waypoint(path, rng)
            if index is None:
                return copied_path(path)
            displacement = tuple(float(value) for value in rng.normal(0.0, step.scale / 2.0, size=2))
            return perturb_waypoint(path, index, displacement, environment)
        if kind == "shift_segment":
            start, end = self._range(selection, path)
            start, end = max(1, start), min(len(path) - 2, end)
            if start > end:
                return copied_path(path)
            if end - start + 1 > step.max_segment_points:
                end = start + step.max_segment_points - 1
            displacement = tuple(float(value) for value in rng.normal(0.0, step.scale / 2.0, size=2))
            return shift_segment(path, start, end, displacement, environment)
        if kind == "insert_waypoint":
            segment = selection[0] if isinstance(selection, tuple) else selection
            if not isinstance(segment, int) or not 0 <= segment < len(path) - 1:
                segment = select_long_segment(path, rng)
            if segment is None:
                return copied_path(path)
            a, b = np.asarray(path[segment]), np.asarray(path[segment + 1])
            midpoint = (a + b) / 2.0
            direction = b - a
            norm = float(np.linalg.norm(direction))
            if norm > 1e-12 and step.offset_scale > 0:
                normal = np.asarray((-direction[1], direction[0])) / norm
                midpoint += normal * float(rng.uniform(-step.offset_scale, step.offset_scale))
            point = (
                float(np.clip(midpoint[0], 0, environment.width)),
                float(np.clip(midpoint[1], 0, environment.height)),
            )
            return insert_waypoint(path, segment, point)
        if kind == "delete_waypoint":
            index = selection if isinstance(selection, int) else None
            if index is None or not 0 < index < len(path) - 1:
                index = select_high_curvature_waypoint(path, rng)
            return copied_path(path) if index is None else delete_waypoint(path, index)
        if kind == "shortcut_segment":
            start, end = self._range(selection, path)
            if end - start < 2:
                start, end = 0, len(path) - 1
            if environment.segment_is_collision_free(path[start], path[end]):
                return shortcut_segment(path, start, end)
            return copied_path(path)
        if kind == "smooth_segment":
            start, end = self._range(selection, path)
            return smooth_segment(path, max(1, start), min(len(path) - 2, end), step.strength, environment)
        if kind == "generate_obstacle_detour":
            segment = selection[0] if isinstance(selection, tuple) else selection
            if not isinstance(segment, int):
                segment = select_collision_segment(path, environment, rng)
            return copied_path(path) if segment is None else generate_obstacle_detour(
                path, segment, environment, rng, clearance_scale=step.clearance_factor
            )
        if kind == "reconstruct_segment":
            start, end = self._range(selection, path)
            if end - start < 2:
                start, end = 0, len(path) - 1
            return reconstruct_segment(path, start, end, environment, rng)
        if kind == "try_alternative_side":
            segment = selection[0] if isinstance(selection, tuple) else selection
            if not isinstance(segment, int):
                segment = select_collision_segment(path, environment, rng)
            return copied_path(path) if segment is None else try_alternative_side(
                path, segment, environment, rng, clearance_scale=step.clearance_factor
            )
        raise OperatorCompilationError(f"unsupported transformation primitive: {kind}")

    def _run_steps(
        self,
        steps: list[TransformationSpec],
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
        deadline: float,
    ) -> Path:
        current = copied_path(path)
        for step in steps:
            if time.monotonic() > deadline:
                raise TimeoutError("operator deadline exceeded")
            values = _path_metrics(current, environment, context)
            if step.when is not None and not _condition_matches(step.when, values):
                continue
            for _ in range(step.repeat):
                selection = self._select(current, environment, rng)
                candidate = self._transform(step, current, selection, environment, rng)
                if len(candidate) > self.limits.max_waypoints:
                    raise ValueError("maximum waypoint count exceeded")
                if len(candidate) - len(path) > self.limits.max_added_waypoints:
                    raise ValueError("maximum added waypoint count exceeded")
                if candidate[0] != path[0] or candidate[-1] != path[-1]:
                    raise ValueError("operator changed a fixed endpoint")
                if any(not math.isfinite(float(value)) for point in candidate for value in point):
                    raise ValueError("operator produced non-finite coordinates")
                current = candidate
                if time.monotonic() > deadline:
                    raise TimeoutError("operator deadline exceeded")
        return current

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorResult:
        original = copied_path(path)
        if len(original) < 2 or len(original) > self.limits.max_waypoints:
            return unchanged_result(original, "invalid input path")
        deadline = time.monotonic() + self.limits.deadline_ms / 1000.0
        try:
            values = _path_metrics(original, environment, context)
            if not all(_condition_matches(condition, values) for condition in self.spec.applicability_conditions):
                return unchanged_result(original, "applicability conditions not met")
            candidate = self._run_steps(
                self.spec.transformations, original, environment, rng, context, deadline
            )
            if self.spec.repair_strategy is not None and not environment.path_is_collision_free(candidate):
                for _ in range(self.spec.repair_strategy.max_attempts):
                    repaired = self._run_steps(
                        self.spec.repair_strategy.transformations,
                        candidate,
                        environment,
                        rng,
                        context,
                        deadline,
                    )
                    candidate = repaired
                    if environment.path_is_collision_free(candidate):
                        break
            if self.spec.fallback_strategy is not None and not environment.path_is_collision_free(candidate):
                candidate = original
            changed = _changed_indices(original, candidate)
            if not changed:
                return unchanged_result(original, "compiled steps produced no structural change")
            return OperatorResult(
                path=candidate,
                modified_indices=changed,
                success=True,
                info={
                    "operator_spec": self.spec.name,
                    "transformation_count": len(self.spec.transformations),
                    "fallback_used": candidate == original,
                },
            )
        except (TimeoutError, ValueError, IndexError, FloatingPointError) as exc:
            return unchanged_result(original, str(exc), exception_type=type(exc).__name__)


class OperatorCompiler:
    """Validate configured bounds and produce a trusted CompiledOperator."""

    def __init__(self, limits: DSLConfig | None = None) -> None:
        self.limits = limits or DSLConfig()

    def validate(self, spec: OperatorSpec) -> None:
        if len(spec.applicability_conditions) > self.limits.max_conditions:
            raise OperatorCompilationError("too many applicability conditions")
        if len(spec.transformations) > self.limits.max_transformations:
            raise OperatorCompilationError("too many transformations")
        if len(spec.parent_operators) > self.limits.max_parents:
            raise OperatorCompilationError("too many parents")
        all_steps = list(spec.transformations)
        if spec.repair_strategy is not None:
            all_steps.extend(spec.repair_strategy.transformations)
            if spec.repair_strategy.max_attempts > self.limits.max_repeat:
                raise OperatorCompilationError("repair attempts exceed configured maximum")
        if any(step.repeat > self.limits.max_repeat for step in all_steps):
            raise OperatorCompilationError("step repeat exceeds configured maximum")

    def compile(self, spec: OperatorSpec | dict[str, Any]) -> CompiledOperator:
        validated = spec if isinstance(spec, OperatorSpec) else OperatorSpec.model_validate(spec)
        self.validate(validated)
        return CompiledOperator(validated, self.limits)


__all__ = ["CompiledOperator", "OperatorCompilationError", "OperatorCompiler"]
