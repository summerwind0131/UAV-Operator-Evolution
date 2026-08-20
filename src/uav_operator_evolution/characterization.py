"""Deterministic identity projections used while extracting generic protocols.

The projection intentionally removes measurements that may change without a
semantic behavior change, such as wall-clock timings and timestamps.  It keeps
solutions, evaluations, operator choices, random-dependent outcomes, evidence,
lineage, and retention decisions so compatibility work has a hard regression
gate.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from .reproducibility import stable_hash


VOLATILE_IDENTITY_FIELDS = frozenset(
    {
        "average_runtime_ms",
        "created_at",
        "elapsed_ms",
        "elapsed_seconds",
        "mean_runtime_ms",
        "median_candidate_operator_runtime_ms",
        "median_operator_runtime_reduction",
        "median_parent_operator_runtime_ms",
        "median_runtime_reduction",
        "runtime",
        "runtime_ms",
        "timestamp",
        "total_runtime_ms",
        "updated_at",
    }
)


def _volatile_field(name: str) -> bool:
    normalized = str(name).lower()
    return bool(
        normalized in VOLATILE_IDENTITY_FIELDS
        or normalized.endswith("_runtime_ms")
        or normalized.endswith("_runtime_samples_ms")
        or normalized.endswith("_elapsed_seconds")
        or normalized.endswith("_latency_ms")
    )


def identity_projection(value: Any) -> Any:
    """Return a JSON-native semantic projection of a result or artifact.

    Unknown object types fail closed instead of being converted with ``repr``;
    memory addresses and implementation-specific strings must never enter a
    characterization hash unnoticed.
    """

    if isinstance(value, BaseModel):
        return identity_projection(value.model_dump(mode="json", by_alias=False))
    if is_dataclass(value) and not isinstance(value, type):
        return identity_projection(
            {field.name: getattr(value, field.name) for field in fields(value)}
        )
    if isinstance(value, Mapping):
        return {
            str(key): identity_projection(item)
            for key, item in value.items()
            if not _volatile_field(str(key))
        }
    if isinstance(value, np.ndarray):
        return identity_projection(value.tolist())
    if isinstance(value, np.generic):
        return identity_projection(value.item())
    if isinstance(value, (list, tuple)):
        return [identity_projection(item) for item in value]
    if isinstance(value, (set, frozenset)):
        projected = [identity_projection(item) for item in value]
        return sorted(projected, key=lambda item: stable_hash(item))
    if isinstance(value, Enum):
        return identity_projection(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("identity projections require finite floating-point values")
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [identity_projection(item) for item in value]
    raise TypeError(f"unsupported identity value: {type(value).__name__}")


def identity_hash(value: Any) -> str:
    """Hash the deterministic semantic projection of *value*."""

    return stable_hash(identity_projection(value))


__all__ = [
    "VOLATILE_IDENTITY_FIELDS",
    "identity_hash",
    "identity_projection",
]
