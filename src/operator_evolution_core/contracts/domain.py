"""Composable domain-boundary protocols for generic search infrastructure."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np
from pydantic import JsonValue

from .models import ObjectiveEvaluation

InstanceT = TypeVar("InstanceT")
SolutionT = TypeVar("SolutionT")


@runtime_checkable
class Initializer(Protocol[InstanceT, SolutionT]):
    """Create one solution using only the supplied instance and RNG stream."""

    def initialize(
        self,
        instance: InstanceT,
        rng: np.random.Generator,
    ) -> SolutionT: ...


@runtime_checkable
class Evaluator(Protocol[InstanceT, SolutionT]):
    """Project a domain solution into the common minimization objective."""

    def evaluate(
        self,
        solution: SolutionT,
        instance: InstanceT,
    ) -> ObjectiveEvaluation: ...


@runtime_checkable
class FeatureExtractor(Protocol[InstanceT, SolutionT]):
    """Encode deterministic, JSON-native solution features."""

    def extract(
        self,
        solution: SolutionT,
        instance: InstanceT,
        evaluation: ObjectiveEvaluation,
    ) -> dict[str, JsonValue]: ...


@runtime_checkable
class SolutionCodec(Protocol[SolutionT]):
    """Own copying, canonicalization and stable persistence of solutions."""

    def clone(self, solution: SolutionT) -> SolutionT: ...

    def canonicalize(self, solution: object) -> SolutionT: ...

    def to_json(self, solution: SolutionT) -> JsonValue: ...

    def stable_hash(self, solution: SolutionT) -> str: ...


@runtime_checkable
class SolutionGuard(Protocol[InstanceT, SolutionT]):
    """Report structural violations without imposing objective feasibility."""

    def validate_structure(
        self,
        solution: SolutionT,
        instance: InstanceT,
    ) -> list[str]: ...


@runtime_checkable
class SearchContextView(Protocol):
    """Small context surface required by a domain trace encoder."""

    def as_features(self) -> Mapping[str, JsonValue]: ...


@runtime_checkable
class TraceEncoder(Protocol[InstanceT, SolutionT]):
    """Encode one solution state using the existing trace-compatible schema."""

    def snapshot(
        self,
        solution: SolutionT,
        instance: InstanceT,
        evaluation: ObjectiveEvaluation,
        context: SearchContextView,
    ) -> dict[str, JsonValue]: ...


@dataclass(frozen=True, slots=True)
class DomainAdapter(Generic[InstanceT, SolutionT]):
    """Composition root for one domain implementation.

    Search algorithms consume this bundle but keep each concern behind its
    focused protocol. Domain-specific operators and instances remain outside
    the core package.
    """

    domain_id: str
    initializer: Initializer[InstanceT, SolutionT]
    evaluator: Evaluator[InstanceT, SolutionT]
    features: FeatureExtractor[InstanceT, SolutionT]
    codec: SolutionCodec[SolutionT]
    guard: SolutionGuard[InstanceT, SolutionT]
    trace_encoder: TraceEncoder[InstanceT, SolutionT]

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9._-]{0,99}", self.domain_id):
            raise ValueError("domain_id must be a lowercase stable identifier")


__all__ = [
    "DomainAdapter",
    "Evaluator",
    "FeatureExtractor",
    "Initializer",
    "InstanceT",
    "SearchContextView",
    "SolutionCodec",
    "SolutionGuard",
    "SolutionT",
    "TraceEncoder",
]
