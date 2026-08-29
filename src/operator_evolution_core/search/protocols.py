"""Replaceable policies consumed by the domain-independent search kernel."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Generic, Protocol, TypeVar, runtime_checkable

import numpy as np

from .models import OperatorOutcome, SearchContext

InstanceT = TypeVar("InstanceT")
SolutionT = TypeVar("SolutionT")
OperatorT = TypeVar("OperatorT")


@runtime_checkable
class SearchOperator(Protocol[InstanceT, SolutionT]):
    name: str
    operator_id: str

    def apply(
        self,
        solution: SolutionT,
        instance: InstanceT,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorOutcome[SolutionT]: ...


class OperatorScheduler(Protocol, Generic[OperatorT]):
    def select(
        self,
        operators: Sequence[OperatorT],
        iteration: int,
        rng: np.random.Generator,
    ) -> OperatorT: ...


class AcceptancePolicy(Protocol):
    def temperature(
        self,
        iteration: int,
        max_iterations: int,
        cost_scale: float,
    ) -> float: ...

    def accept(
        self,
        current_cost: float,
        candidate_cost: float,
        temperature: float,
        rng: np.random.Generator,
    ) -> bool: ...


__all__ = [
    "AcceptancePolicy",
    "InstanceT",
    "OperatorScheduler",
    "OperatorT",
    "SearchOperator",
    "SolutionT",
]

