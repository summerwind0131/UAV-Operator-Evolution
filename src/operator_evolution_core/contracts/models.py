"""Strict JSON-native models at the domain/core boundary."""

from __future__ import annotations

import math
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

DatasetSplit: TypeAlias = Literal["train", "validation", "test"]


class _ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


def _validate_json_finite(value: JsonValue, location: str = "metadata") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{location} contains a non-finite number")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_finite(item, f"{location}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            if not str(key).strip():
                raise ValueError(f"{location} contains an empty key")
            _validate_json_finite(item, f"{location}.{key}")


def _validate_numeric_mapping(
    values: dict[str, float],
    *,
    field_name: str,
    non_negative: bool,
) -> dict[str, float]:
    for key, value in values.items():
        if not key.strip():
            raise ValueError(f"{field_name} names must not be empty")
        if not math.isfinite(float(value)):
            raise ValueError(f"{field_name}.{key} must be finite")
        if non_negative and value < 0:
            raise ValueError(f"{field_name}.{key} must be non-negative")
    return values


class InstanceRef(_ContractModel):
    """Content-addressed identity of one domain instance.

    The reference deliberately does not contain the full problem payload.  A
    domain adapter owns loading and validating that payload against
    ``content_hash``.
    """

    domain_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-z][a-z0-9._-]*$")
    instance_id: str = Field(min_length=1, max_length=255)
    split: DatasetSplit
    difficulty: str | None = Field(default=None, min_length=1, max_length=100)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("metadata")
    @classmethod
    def metadata_is_canonical_json(
        cls, values: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _validate_json_finite(values)
        return values


class ObjectiveEvaluation(_ContractModel):
    """Domain-independent scalar minimization result.

    ``scalar_cost`` is always finite and lower is always better.  Adapters may
    preserve domain-native objective terms in ``components`` and hard
    constraint magnitudes in ``violations``.
    """

    scalar_cost: float
    components: dict[str, float] = Field(min_length=1)
    feasible: bool
    violations: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("scalar_cost")
    @classmethod
    def scalar_cost_is_finite(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("scalar_cost must be finite")
        return value

    @field_validator("components")
    @classmethod
    def components_are_finite(cls, values: dict[str, float]) -> dict[str, float]:
        return _validate_numeric_mapping(
            values, field_name="components", non_negative=False
        )

    @field_validator("violations")
    @classmethod
    def violations_are_finite_and_non_negative(
        cls, values: dict[str, float]
    ) -> dict[str, float]:
        return _validate_numeric_mapping(
            values, field_name="violations", non_negative=True
        )

    @field_validator("metadata")
    @classmethod
    def metadata_is_canonical_json(
        cls, values: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _validate_json_finite(values)
        return values


__all__ = ["DatasetSplit", "InstanceRef", "ObjectiveEvaluation"]
