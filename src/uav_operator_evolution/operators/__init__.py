"""Path operator contracts, primitives and baseline implementations."""

from .base import OperatorResult, PathOperator, copied_path, unchanged_result
from .manual import (
    DeleteWaypointOperator,
    InsertWaypointOperator,
    ManualOperator,
    ObstacleDetourOperator,
    PartialReconstructionOperator,
    SegmentShiftOperator,
    ShortcutOperator,
    SmoothSegmentOperator,
    WaypointPerturbOperator,
)
from .registry import (
    OperatorRegistry,
    build_manual_operator_registry,
    create_default_operator_registry,
    default_manual_operators,
)
from .compiler import CompiledOperator, OperatorCompilationError, OperatorCompiler
from .specs import ConditionSpec, OperatorSpec, allowed_primitive_names, primitive_catalog

__all__ = [
    "DeleteWaypointOperator",
    "InsertWaypointOperator",
    "ManualOperator",
    "ObstacleDetourOperator",
    "OperatorRegistry",
    "OperatorResult",
    "OperatorSpec",
    "OperatorCompiler",
    "OperatorCompilationError",
    "CompiledOperator",
    "ConditionSpec",
    "PartialReconstructionOperator",
    "PathOperator",
    "SegmentShiftOperator",
    "ShortcutOperator",
    "SmoothSegmentOperator",
    "WaypointPerturbOperator",
    "build_manual_operator_registry",
    "allowed_primitive_names",
    "copied_path",
    "create_default_operator_registry",
    "default_manual_operators",
    "primitive_catalog",
    "unchanged_result",
]
