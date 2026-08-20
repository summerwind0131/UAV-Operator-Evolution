"""Complete UAV implementation of the experimental core domain boundary."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from pydantic import JsonValue

from operator_evolution_core.contracts import (
    DomainAdapter,
    ObjectiveEvaluation,
    SearchContextView,
)

from ..environment.environment import Environment2D
from ..operators.base import copied_path
from ..path.evaluator import PathEvaluator
from ..path.features import extract_path_features
from ..path.initializer import initialize_path
from ..path.models import Path, copy_and_validate_path
from ..reproducibility import stable_hash
from .adapters import (
    UAV_DOMAIN_ID,
    evaluation_result_to_objective,
    objective_to_evaluation_result,
)


class UAVPathInitializer:
    """Adapter over the existing deterministic grid A* initializer."""

    def __init__(self, grid_resolution: float = 4.0) -> None:
        if not math.isfinite(grid_resolution) or grid_resolution <= 0:
            raise ValueError("grid_resolution must be finite and positive")
        self.grid_resolution = float(grid_resolution)

    def initialize(
        self,
        instance: Environment2D,
        rng: np.random.Generator,
    ) -> Path:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        return initialize_path(
            instance,
            rng=rng,
            grid_resolution=self.grid_resolution,
        )


class UAVObjectiveEvaluator:
    """Adapter over ``PathEvaluator`` preserving its full decomposition."""

    def __init__(self, evaluator: PathEvaluator) -> None:
        self.native_evaluator = evaluator

    def evaluate(
        self,
        solution: Path,
        instance: Environment2D,
    ) -> ObjectiveEvaluation:
        return evaluation_result_to_objective(
            self.native_evaluator.evaluate(solution, instance)
        )


class UAVPathFeatureExtractor:
    """Adapter over the existing deterministic UAV path features."""

    def __init__(self, evaluator: PathEvaluator) -> None:
        self.native_evaluator = evaluator

    def extract(
        self,
        solution: Path,
        instance: Environment2D,
        evaluation: ObjectiveEvaluation,
    ) -> dict[str, JsonValue]:
        # The legacy feature function recomputes evaluation internally. Validate
        # the supplied boundary payload first, then preserve that exact behavior.
        objective_to_evaluation_result(evaluation)
        features = extract_path_features(
            solution,
            instance,
            evaluator=self.native_evaluator,
        )
        return cast(dict[str, JsonValue], features.model_dump(mode="json"))


class UAVPathCodec:
    """Canonical copy and JSON codec for two-dimensional waypoint paths."""

    def clone(self, solution: Path) -> Path:
        return copied_path(solution)

    def canonicalize(self, solution: object) -> Path:
        try:
            return copy_and_validate_path(cast(Any, solution))
        except (IndexError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid UAV path: {exc}") from exc

    def to_json(self, solution: Path) -> JsonValue:
        canonical = self.canonicalize(solution)
        return [[float(x), float(y)] for x, y in canonical]

    def stable_hash(self, solution: Path) -> str:
        return stable_hash(self.to_json(solution))


class UAVPathGuard:
    """Validate the structural rules historically enforced by SearchExecutor."""

    def __init__(self, codec: UAVPathCodec) -> None:
        self.codec = codec

    def validate_structure(
        self,
        solution: Path,
        instance: Environment2D,
    ) -> list[str]:
        try:
            canonical = self.codec.canonicalize(solution)
        except ValueError as exc:
            return [str(exc)]

        violations: list[str] = []
        if canonical[0] != instance.start:
            violations.append("path start must equal instance start")
        if canonical[-1] != instance.goal:
            violations.append("path goal must equal instance goal")
        for index, waypoint in enumerate(canonical):
            if not instance.in_bounds(waypoint):
                violations.append(f"waypoint {index} must be inside instance bounds")
        return violations


class UAVTraceEncoder:
    """Encode the exact UAV state payload currently persisted by trajectories."""

    def __init__(
        self,
        features: UAVPathFeatureExtractor,
        codec: UAVPathCodec,
    ) -> None:
        self.features = features
        self.codec = codec

    def snapshot(
        self,
        solution: Path,
        instance: Environment2D,
        evaluation: ObjectiveEvaluation,
        context: SearchContextView,
    ) -> dict[str, JsonValue]:
        native = objective_to_evaluation_result(evaluation)
        path_features = self.features.extract(solution, instance, evaluation)
        search_features = _json_mapping(context.as_features())
        return {
            "path": self.codec.to_json(solution),
            "path_features": path_features,
            "search_features": search_features,
            "objective": float(native.total_cost),
            "objective_components": {
                "path_length": float(native.path_length),
                "collision_penalty": float(native.collision_penalty),
                "smoothness_penalty": float(native.smoothness_penalty),
                "risk_penalty": float(native.risk_penalty),
                "waypoint_penalty": float(native.waypoint_penalty),
            },
            "feasible": bool(native.feasible),
            "collision_count": int(native.collision_count),
            "minimum_clearance": float(native.minimum_clearance),
        }


class UAVDomainAdapter(DomainAdapter[Environment2D, Path]):
    """Composition root for the current UAV path-planning domain."""

    def __init__(
        self,
        evaluator: PathEvaluator | None = None,
        *,
        initializer_grid_resolution: float = 4.0,
    ) -> None:
        native_evaluator = evaluator or PathEvaluator()
        initializer = UAVPathInitializer(initializer_grid_resolution)
        objective_evaluator = UAVObjectiveEvaluator(native_evaluator)
        features = UAVPathFeatureExtractor(native_evaluator)
        codec = UAVPathCodec()
        guard = UAVPathGuard(codec)
        trace_encoder = UAVTraceEncoder(features, codec)
        super().__init__(
            domain_id=UAV_DOMAIN_ID,
            initializer=initializer,
            evaluator=objective_evaluator,
            features=features,
            codec=codec,
            guard=guard,
            trace_encoder=trace_encoder,
        )


def _json_mapping(values: Mapping[str, Any]) -> dict[str, JsonValue]:
    return {str(key): _json_value(value) for key, value in values.items()}


def _json_value(value: Any) -> JsonValue:
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("trace context contains a non-finite number")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"trace context contains non-JSON value: {type(value).__name__}")


__all__ = [
    "UAVDomainAdapter",
    "UAVObjectiveEvaluator",
    "UAVPathCodec",
    "UAVPathFeatureExtractor",
    "UAVPathGuard",
    "UAVPathInitializer",
    "UAVTraceEncoder",
]
