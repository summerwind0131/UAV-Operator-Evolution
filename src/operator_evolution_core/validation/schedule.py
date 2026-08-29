"""Deterministic CRN, ABBA timing, and population-slot utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

OperatorT = TypeVar("OperatorT")
TimingOrder = Literal["parent_first", "candidate_first"]


@dataclass(frozen=True, slots=True)
class CRNSeed:
    instance_id: str
    instance_index: int
    seed: int


def build_crn_seed_schedule(
    instance_ids: Iterable[str],
    seed_factory: Callable[[int, str], int],
) -> tuple[CRNSeed, ...]:
    """Materialize a stable common-random-number schedule once."""

    return tuple(
        CRNSeed(
            instance_id=str(instance_id),
            instance_index=index,
            seed=int(seed_factory(index, str(instance_id))),
        )
        for index, instance_id in enumerate(instance_ids)
    )


def abba_timing_order(repetitions: int) -> tuple[TimingOrder, ...]:
    if repetitions < 1:
        raise ValueError("runtime repetitions must be positive")
    return tuple(
        "parent_first" if repetition % 4 in {0, 3} else "candidate_first"
        for repetition in range(repetitions)
    )


@dataclass(frozen=True, slots=True)
class SlotReplacement(Generic[OperatorT]):
    population: tuple[OperatorT, ...]
    slot_index: int


def replace_population_slot(
    population: Sequence[OperatorT],
    parent_id: str,
    candidate: OperatorT,
    *,
    operator_id: Callable[[OperatorT], str],
) -> SlotReplacement[OperatorT] | None:
    for index, operator in enumerate(population):
        if operator_id(operator) == parent_id:
            updated = list(population)
            updated[index] = candidate
            return SlotReplacement(tuple(updated), index)
    return None


__all__ = [
    "CRNSeed",
    "SlotReplacement",
    "TimingOrder",
    "abba_timing_order",
    "build_crn_seed_schedule",
    "replace_population_slot",
]
