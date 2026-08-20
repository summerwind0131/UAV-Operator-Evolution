"""Unbiased baseline scheduling for path operators."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from ..operators.base import PathOperator


class OperatorScheduler(Protocol):
    """Replaceable operator-selection interface."""

    def select(
        self,
        operators: Sequence[PathOperator],
        iteration: int,
        rng: np.random.Generator,
    ) -> PathOperator:
        """Select one operator for the current iteration."""


class BlockRandomRoundRobinScheduler:
    """Shuffle once per block and call every operator exactly once per block."""

    def __init__(self) -> None:
        self._block_index: int | None = None
        self._order: tuple[int, ...] = ()
        self._signature: tuple[int, ...] = ()

    def reset(self) -> None:
        """Forget the previous run's block state."""

        self._block_index = None
        self._order = ()
        self._signature = ()

    def select(
        self,
        operators: Sequence[PathOperator],
        iteration: int,
        rng: np.random.Generator,
    ) -> PathOperator:
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


RandomRoundRobinScheduler = BlockRandomRoundRobinScheduler
