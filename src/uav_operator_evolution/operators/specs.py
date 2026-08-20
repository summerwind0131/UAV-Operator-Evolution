"""Strict, non-executable operator DSL models."""

from __future__ import annotations

import math
from types import MappingProxyType
from collections.abc import Mapping
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DSLModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


FeatureName: TypeAlias = Literal[
    "collision_count",
    "minimum_clearance",
    "waypoint_count",
    "path_length",
    "maximum_turn_angle",
    "smoothness",
    "iteration_ratio",
    "stagnation_count",
    "feasible",
    "obstacle_density",
    "map_difficulty",
]


class ConditionSpec(DSLModel):
    feature: FeatureName
    operator: Literal["lt", "le", "gt", "ge", "eq", "ne"]
    value: float | int | bool | str

    @field_validator("value")
    @classmethod
    def finite_value(cls, value: float | int | bool | str) -> float | int | bool | str:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("condition values must be finite")
        return value


class RandomWaypointSelection(DSLModel):
    kind: Literal["select_random_waypoint"] = "select_random_waypoint"


class HighCurvatureSelection(DSLModel):
    kind: Literal["select_high_curvature_waypoint"] = "select_high_curvature_waypoint"


class CollisionSegmentSelection(DSLModel):
    kind: Literal["select_collision_segment"] = "select_collision_segment"


class LongSegmentSelection(DSLModel):
    kind: Literal["select_long_segment"] = "select_long_segment"


class LowClearanceSelection(DSLModel):
    kind: Literal["select_low_clearance_segment"] = "select_low_clearance_segment"


class ContinuousCollisionSelection(DSLModel):
    kind: Literal["select_continuous_collision_region"] = "select_continuous_collision_region"


SelectionSpec: TypeAlias = Annotated[
    RandomWaypointSelection
    | HighCurvatureSelection
    | CollisionSegmentSelection
    | LongSegmentSelection
    | LowClearanceSelection
    | ContinuousCollisionSelection,
    Field(discriminator="kind"),
]


class TransformationBase(DSLModel):
    when: ConditionSpec | None = None
    repeat: int = Field(1, ge=1, le=3)


class PerturbWaypointSpec(TransformationBase):
    kind: Literal["perturb_waypoint"] = "perturb_waypoint"
    scale: float = Field(4.0, gt=0, le=100)


class ShiftSegmentSpec(TransformationBase):
    kind: Literal["shift_segment"] = "shift_segment"
    scale: float = Field(4.0, gt=0, le=100)
    max_segment_points: int = Field(4, ge=1, le=32)


class InsertWaypointSpec(TransformationBase):
    kind: Literal["insert_waypoint"] = "insert_waypoint"
    offset_scale: float = Field(2.0, ge=0, le=100)


class DeleteWaypointSpec(TransformationBase):
    kind: Literal["delete_waypoint"] = "delete_waypoint"


class ShortcutSegmentSpec(TransformationBase):
    kind: Literal["shortcut_segment"] = "shortcut_segment"


class SmoothSegmentSpec(TransformationBase):
    kind: Literal["smooth_segment"] = "smooth_segment"
    strength: float = Field(0.5, ge=0, le=1)


class ObstacleDetourSpec(TransformationBase):
    kind: Literal["generate_obstacle_detour"] = "generate_obstacle_detour"
    clearance_factor: float = Field(1.5, gt=0, le=10)


class ReconstructSegmentSpec(TransformationBase):
    kind: Literal["reconstruct_segment"] = "reconstruct_segment"
    max_points: int = Field(8, ge=1, le=32)


class AlternativeSideSpec(TransformationBase):
    kind: Literal["try_alternative_side"] = "try_alternative_side"
    clearance_factor: float = Field(1.5, gt=0, le=10)


TransformationSpec: TypeAlias = Annotated[
    PerturbWaypointSpec
    | ShiftSegmentSpec
    | InsertWaypointSpec
    | DeleteWaypointSpec
    | ShortcutSegmentSpec
    | SmoothSegmentSpec
    | ObstacleDetourSpec
    | ReconstructSegmentSpec
    | AlternativeSideSpec,
    Field(discriminator="kind"),
]


class RepairSpec(DSLModel):
    kind: Literal["repeat_until_feasible"] = "repeat_until_feasible"
    transformations: list[TransformationSpec] = Field(min_length=1, max_length=4)
    max_attempts: int = Field(2, ge=1, le=3)


class FallbackSpec(DSLModel):
    kind: Literal["rollback_on_failure"] = "rollback_on_failure"


PrimitiveScalar: TypeAlias = float | int | bool | str


class OperatorSpec(DSLModel):
    """A bounded data-only description compiled by a trusted interpreter."""

    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z][A-Za-z0-9_]*$")
    description: str = Field(min_length=1, max_length=2000)
    parent_operators: list[str] = Field(default_factory=list, max_length=4)
    applicability_conditions: list[ConditionSpec] = Field(default_factory=list, max_length=8)
    selection_strategy: SelectionSpec
    transformations: list[TransformationSpec] = Field(min_length=1, max_length=8)
    repair_strategy: RepairSpec | None = None
    fallback_strategy: FallbackSpec | None = None
    parameters: dict[str, PrimitiveScalar] = Field(default_factory=dict, max_length=16)
    expected_mechanism: str = Field(min_length=1, max_length=2000)
    target_failure_modes: list[str] = Field(default_factory=list, max_length=16)

    @model_validator(mode="after")
    def validate_scalar_parameters(self) -> "OperatorSpec":
        for key, value in self.parameters.items():
            if not key or len(key) > 64:
                raise ValueError("parameter names must contain 1-64 characters")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("parameters must be finite")
        return self


_PRIMITIVE_CATALOG: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "selection": (
            "select_random_waypoint",
            "select_high_curvature_waypoint",
            "select_collision_segment",
            "select_long_segment",
            "select_low_clearance_segment",
            "select_continuous_collision_region",
        ),
        "transformation": (
            "perturb_waypoint",
            "shift_segment",
            "insert_waypoint",
            "delete_waypoint",
            "shortcut_segment",
            "smooth_segment",
            "generate_obstacle_detour",
            "reconstruct_segment",
            "try_alternative_side",
        ),
        "repair": ("repeat_until_feasible",),
        "fallback": ("rollback_on_failure",),
    }
)


def primitive_catalog() -> Mapping[str, tuple[str, ...]]:
    """Return the immutable public catalog understood by the DSL compiler.

    Keeping this catalog next to the discriminated unions prevents evidence
    builders and agent tools from maintaining a second, drifting whitelist.
    """

    return _PRIMITIVE_CATALOG


def allowed_primitive_names() -> tuple[str, ...]:
    """Return every permitted primitive name in stable category order."""

    return tuple(name for names in _PRIMITIVE_CATALOG.values() for name in names)


__all__ = [
    "ConditionSpec",
    "FallbackSpec",
    "OperatorSpec",
    "RepairSpec",
    "SelectionSpec",
    "TransformationSpec",
    "allowed_primitive_names",
    "primitive_catalog",
]
