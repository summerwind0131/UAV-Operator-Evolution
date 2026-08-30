"""Strict typed ``jssp-v1`` IR with statically bounded capabilities."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

JSSP_IR_VERSION = "jssp-v1"
SelectorKind = Literal[
    "random_adjacent",
    "random_pair",
    "bounded_pair",
    "critical_block_adjacent",
    "critical_block_endpoints",
    "bottleneck_block",
    "high_idle_gap",
]
TransformKind = Literal["swap", "insert", "reverse"]


class _IRModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class SelectorSpec(_IRModel):
    kind: SelectorKind
    max_distance: int = Field(default=16, ge=1, le=128)
    max_attempts: int = Field(default=8, ge=1, le=32)


class TransformSpec(_IRModel):
    kind: TransformKind
    max_segment_length: int = Field(default=32, ge=2, le=128)


class RepairSpec(_IRModel):
    kind: Literal["multiplicity_guard"] = "multiplicity_guard"


class JSSPOperatorSpec(_IRModel):
    ir_version: Literal["jssp-v1"] = JSSP_IR_VERSION
    operator_id: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z][a-z0-9._-]*$",
    )
    name: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=2_000)
    parent_ids: list[str] = Field(default_factory=list, max_length=16)
    selector: SelectorSpec
    transform: TransformSpec
    repair: RepairSpec = Field(default_factory=RepairSpec)

    @model_validator(mode="after")
    def capability_combination_is_bounded(self) -> "JSSPOperatorSpec":
        allowed: dict[SelectorKind, frozenset[TransformKind]] = {
            "random_adjacent": frozenset({"swap"}),
            "random_pair": frozenset({"swap"}),
            "bounded_pair": frozenset({"insert", "reverse"}),
            "critical_block_adjacent": frozenset({"swap"}),
            "critical_block_endpoints": frozenset({"swap"}),
            "bottleneck_block": frozenset({"insert"}),
            "high_idle_gap": frozenset({"insert"}),
        }
        if self.transform.kind not in allowed[self.selector.kind]:
            raise ValueError(
                f"transform {self.transform.kind!r} is not allowed for "
                f"selector {self.selector.kind!r}"
            )
        if len(set(self.parent_ids)) != len(self.parent_ids):
            raise ValueError("parent_ids must be unique")
        return self


def capability_catalog() -> dict[str, tuple[str, ...]]:
    return {
        "selectors": (
            "random_adjacent",
            "random_pair",
            "bounded_pair",
            "critical_block_adjacent",
            "critical_block_endpoints",
            "bottleneck_block",
            "high_idle_gap",
        ),
        "transforms": ("swap", "insert", "reverse"),
        "repairs": ("multiplicity_guard",),
    }


__all__ = [
    "JSSP_IR_VERSION",
    "JSSPOperatorSpec",
    "RepairSpec",
    "SelectorKind",
    "SelectorSpec",
    "TransformKind",
    "TransformSpec",
    "capability_catalog",
]
