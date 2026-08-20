"""Obstacle and risk-zone models for continuous two-dimensional maps."""

from __future__ import annotations

import math
from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

Point: TypeAlias = tuple[float, float]


def _is_finite_point(point: Point) -> bool:
    return len(point) == 2 and all(math.isfinite(float(value)) for value in point)


class _GeometryModel(BaseModel):
    """Strict immutable base class for serializable geometry."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class CircleObstacle(_GeometryModel):
    """A circular obstacle in map coordinates."""

    kind: Literal["circle"] = "circle"
    center: Point
    radius: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_geometry(self) -> "CircleObstacle":
        if not _is_finite_point(self.center) or not math.isfinite(self.radius):
            raise ValueError("circle coordinates and radius must be finite")
        return self

    @property
    def area(self) -> float:
        return math.pi * self.radius * self.radius


class RectangleObstacle(_GeometryModel):
    """An axis-aligned rectangular obstacle."""

    kind: Literal["rectangle"] = "rectangle"
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    @model_validator(mode="after")
    def validate_geometry(self) -> "RectangleObstacle":
        values = (self.min_x, self.min_y, self.max_x, self.max_y)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("rectangle coordinates must be finite")
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("rectangle minimum coordinates must be below maximum coordinates")
        return self

    @property
    def center(self) -> Point:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height


Obstacle: TypeAlias = Annotated[
    CircleObstacle | RectangleObstacle,
    Field(discriminator="kind"),
]


class RiskZone(_GeometryModel):
    """An axis-aligned rectangular region with an exposure multiplier."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float
    weight: float = Field(default=1.0, ge=0)
    name: str = "risk"

    @model_validator(mode="after")
    def validate_geometry(self) -> "RiskZone":
        values = (self.min_x, self.min_y, self.max_x, self.max_y, self.weight)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("risk-zone values must be finite")
        if self.min_x >= self.max_x or self.min_y >= self.max_y:
            raise ValueError("risk-zone minimum coordinates must be below maximum coordinates")
        return self

    @property
    def center(self) -> Point:
        return ((self.min_x + self.max_x) / 2.0, (self.min_y + self.max_y) / 2.0)

    @property
    def area(self) -> float:
        return (self.max_x - self.min_x) * (self.max_y - self.min_y)


__all__ = [
    "CircleObstacle",
    "Obstacle",
    "Point",
    "RectangleObstacle",
    "RiskZone",
]
