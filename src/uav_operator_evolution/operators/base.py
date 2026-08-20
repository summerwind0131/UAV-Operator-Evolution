"""Common contracts for path-search operators."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from ..environment.environment import Environment2D
from ..path.models import Path

if TYPE_CHECKING:
    from ..search.context import SearchContext


@dataclass(slots=True)
class OperatorResult:
    """Outcome of one operator call.

    ``path`` is always a newly allocated list, including for a safe no-op.  The
    tuple in ``modified_indices`` refers to indices in the input path whenever
    possible; insertion operators document new indices in ``info`` as well.
    """

    path: Path
    modified_indices: tuple[int, ...] = ()
    success: bool = True
    info: dict[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    @property
    def metadata(self) -> dict[str, Any]:
        """Compatibility alias for callers that call internal details metadata."""

        return self.info


@runtime_checkable
class PathOperator(Protocol):
    """Protocol implemented by manual and compiled path operators."""

    name: str

    def apply(
        self,
        path: Path,
        environment: Environment2D,
        rng: np.random.Generator,
        context: "SearchContext",
    ) -> OperatorResult:
        """Return a candidate without mutating ``path``."""


def copied_path(path: Path) -> Path:
    """Copy a path while normalizing every waypoint to a float tuple."""

    return [(float(point[0]), float(point[1])) for point in path]


def unchanged_result(path: Path, reason: str, **info: Any) -> OperatorResult:
    """Create an explicit, safe failure result that owns its path list."""

    details = {"status": "no_change", "reason": reason, **info}
    return OperatorResult(
        path=copied_path(path),
        success=False,
        info=details,
        failure_reason=reason,
    )
