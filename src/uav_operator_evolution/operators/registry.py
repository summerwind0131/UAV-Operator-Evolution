"""Small explicit registry for manual and compiled operators."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from .base import PathOperator
from .manual import (
    DeleteWaypointOperator,
    InsertWaypointOperator,
    ObstacleDetourOperator,
    PartialReconstructionOperator,
    SegmentShiftOperator,
    ShortcutOperator,
    SmoothSegmentOperator,
    WaypointPerturbOperator,
)


def default_manual_operators() -> list[PathOperator]:
    """Create fresh instances of all eight baseline operators."""

    return [
        WaypointPerturbOperator(),
        SegmentShiftOperator(),
        InsertWaypointOperator(),
        DeleteWaypointOperator(),
        ShortcutOperator(),
        SmoothSegmentOperator(),
        ObstacleDetourOperator(),
        PartialReconstructionOperator(),
    ]


class OperatorRegistry:
    """Name-indexed collection with duplicate protection and stable order."""

    def __init__(self, operators: Iterable[PathOperator] = ()) -> None:
        self._operators: dict[str, PathOperator] = {}
        for operator in operators:
            self.register(operator)

    def register(self, operator: PathOperator, *, replace: bool = False) -> None:
        name = str(operator.name).strip()
        if not name:
            raise ValueError("operator name must not be empty")
        if name in self._operators and not replace:
            raise ValueError(f"operator already registered: {name}")
        self._operators[name] = operator

    def get(self, name: str) -> PathOperator:
        try:
            return self._operators[name]
        except KeyError as exc:
            raise KeyError(f"unknown operator: {name}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(self._operators)

    def values(self) -> tuple[PathOperator, ...]:
        return tuple(self._operators.values())

    def __contains__(self, name: object) -> bool:
        return name in self._operators

    def __len__(self) -> int:
        return len(self._operators)

    def __iter__(self) -> Iterator[PathOperator]:
        return iter(self._operators.values())


def build_manual_operator_registry() -> OperatorRegistry:
    """Return a registry populated with fresh baseline instances."""

    return OperatorRegistry(default_manual_operators())


create_default_operator_registry = build_manual_operator_registry
