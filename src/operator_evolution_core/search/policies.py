"""Deterministic baseline scheduling and acceptance policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import exp
from typing import Generic, TypeVar

import numpy as np

OperatorT = TypeVar("OperatorT")


class BlockRandomRoundRobinScheduler(Generic[OperatorT]):
    """Shuffle once per block and select every operator exactly once."""

    def __init__(self) -> None:
        self._block_index: int | None = None
        self._order: tuple[int, ...] = ()
        self._signature: tuple[int, ...] = ()

    def reset(self) -> None:
        self._block_index = None
        self._order = ()
        self._signature = ()

    def select(
        self,
        operators: Sequence[OperatorT],
        iteration: int,
        rng: np.random.Generator,
    ) -> OperatorT:
        if not operators:
            raise ValueError("at least one operator is required")
        signature = tuple(id(operator) for operator in operators)
        block_index = max(0, int(iteration)) // len(operators)
        if block_index != self._block_index or signature != self._signature:
            self._order = tuple(int(index) for index in rng.permutation(len(operators)))
            self._block_index = block_index
            self._signature = signature
        offset = max(0, int(iteration)) % len(operators)
        return operators[self._order[offset]]


@dataclass(frozen=True, slots=True)
class SimulatedAnnealingAcceptance:
    start_temperature_ratio: float = 0.05
    end_temperature_ratio: float = 0.001
    minimum_temperature: float = 1e-12

    def __post_init__(self) -> None:
        if self.start_temperature_ratio <= 0 or self.end_temperature_ratio <= 0:
            raise ValueError("temperature ratios must be positive")
        if self.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")

    def temperature(
        self,
        iteration: int,
        max_iterations: int,
        cost_scale: float,
    ) -> float:
        progress = min(
            max(float(iteration) / max(1, int(max_iterations) - 1), 0.0),
            1.0,
        )
        start = max(
            abs(float(cost_scale)) * self.start_temperature_ratio,
            self.minimum_temperature,
        )
        end = max(
            abs(float(cost_scale)) * self.end_temperature_ratio,
            self.minimum_temperature,
        )
        return float(start * ((end / start) ** progress))

    def accept(
        self,
        current_cost: float,
        candidate_cost: float,
        temperature: float,
        rng: np.random.Generator,
    ) -> bool:
        delta = float(candidate_cost) - float(current_cost)
        if delta <= 0.0:
            return True
        if not np.isfinite(delta) or temperature <= 0.0:
            return False
        probability = exp(-delta / max(float(temperature), self.minimum_temperature))
        return bool(rng.random() < probability)


__all__ = ["BlockRandomRoundRobinScheduler", "SimulatedAnnealingAcceptance"]

