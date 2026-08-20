"""Pure adapters between existing UAV models and experimental core contracts."""

from __future__ import annotations

import math

from operator_evolution_core.contracts import (
    DatasetSplit,
    InstanceRef,
    ObjectiveEvaluation,
)

from ..environment.environment import Environment2D
from ..path.models import EvaluationResult

UAV_DOMAIN_ID = "uav-path-planning-2d"
UAV_OBJECTIVE_ADAPTER_VERSION = "uav-objective-v1"

_UAV_COMPONENTS = (
    "path_length",
    "collision_penalty",
    "smoothness_penalty",
    "risk_penalty",
    "waypoint_penalty",
)


def environment_to_instance_ref(
    environment: Environment2D,
    split: DatasetSplit,
) -> InstanceRef:
    """Project a full UAV map into a content-addressed core identity."""

    return InstanceRef(
        domain_id=UAV_DOMAIN_ID,
        instance_id=environment.map_id,
        split=split,
        difficulty=environment.difficulty,
        content_hash=environment.content_hash,
        metadata={
            "seed": int(environment.seed),
            "width": float(environment.width),
            "height": float(environment.height),
            "safety_distance": float(environment.safety_distance),
            "layout_subtype": environment.layout_subtype,
            "terminal_hash": environment.terminal_hash,
            "obstacle_layout_hash": environment.obstacle_layout_hash,
            "geometry_hash": environment.geometry_hash,
        },
    )


def environment_matches_instance_ref(
    environment: Environment2D,
    reference: InstanceRef,
) -> bool:
    """Return whether a UAV map exactly matches an existing core reference."""

    if reference.domain_id != UAV_DOMAIN_ID:
        return False
    return environment_to_instance_ref(environment, reference.split) == reference


def evaluation_result_to_objective(
    evaluation: EvaluationResult,
) -> ObjectiveEvaluation:
    """Losslessly project a UAV evaluation into the core objective model."""

    return ObjectiveEvaluation(
        scalar_cost=float(evaluation.total_cost),
        components={
            "path_length": float(evaluation.path_length),
            "collision_penalty": float(evaluation.collision_penalty),
            "smoothness_penalty": float(evaluation.smoothness_penalty),
            "risk_penalty": float(evaluation.risk_penalty),
            "waypoint_penalty": float(evaluation.waypoint_penalty),
        },
        feasible=bool(evaluation.feasible),
        violations={
            "collision_count": float(evaluation.collision_count),
            "collision_penalty": float(evaluation.collision_penalty),
        },
        metadata={
            "domain_id": UAV_DOMAIN_ID,
            "adapter_version": UAV_OBJECTIVE_ADAPTER_VERSION,
            "minimum_clearance": float(evaluation.minimum_clearance),
        },
    )


def objective_to_evaluation_result(
    evaluation: ObjectiveEvaluation,
) -> EvaluationResult:
    """Rehydrate a core objective produced by the UAV v1 adapter."""

    if evaluation.metadata.get("domain_id") != UAV_DOMAIN_ID:
        raise ValueError("objective evaluation does not belong to the UAV domain")
    if evaluation.metadata.get("adapter_version") != UAV_OBJECTIVE_ADAPTER_VERSION:
        raise ValueError("unsupported UAV objective adapter version")

    missing = sorted(set(_UAV_COMPONENTS) - set(evaluation.components))
    if missing:
        raise ValueError(f"UAV objective is missing components: {missing}")

    raw_count = evaluation.violations.get("collision_count")
    if raw_count is None or not math.isfinite(raw_count) or int(raw_count) != raw_count:
        raise ValueError("UAV objective collision_count must be a finite integer")
    minimum_clearance = evaluation.metadata.get("minimum_clearance")
    if (
        isinstance(minimum_clearance, bool)
        or not isinstance(minimum_clearance, (int, float))
        or not math.isfinite(float(minimum_clearance))
    ):
        raise ValueError("UAV objective minimum_clearance must be finite")

    return EvaluationResult(
        total_cost=evaluation.scalar_cost,
        path_length=evaluation.components["path_length"],
        collision_penalty=evaluation.components["collision_penalty"],
        smoothness_penalty=evaluation.components["smoothness_penalty"],
        risk_penalty=evaluation.components["risk_penalty"],
        waypoint_penalty=evaluation.components["waypoint_penalty"],
        feasible=evaluation.feasible,
        collision_count=int(raw_count),
        minimum_clearance=float(minimum_clearance),
    )


__all__ = [
    "UAV_DOMAIN_ID",
    "UAV_OBJECTIVE_ADAPTER_VERSION",
    "environment_matches_instance_ref",
    "environment_to_instance_ref",
    "evaluation_result_to_objective",
    "objective_to_evaluation_result",
]
