"""Trusted compiler for the finite ``jssp-v1`` selector/transform catalog."""

from __future__ import annotations

from collections import Counter

import numpy as np

from operator_evolution_core.search import OperatorOutcome, SearchContext

from ..models import JobShopInstance, JobShopSolution
from ..schedule import JobShopSchedule, ScheduledOperation, decode_schedule
from .specs import JSSPOperatorSpec


def _pick_distinct_pair(
    length: int,
    rng: np.random.Generator,
) -> tuple[int, int] | None:
    if length < 2:
        return None
    values = rng.choice(length, size=2, replace=False)
    return int(values[0]), int(values[1])


def _operation_lookup(schedule: JobShopSchedule) -> dict[tuple[int, int], ScheduledOperation]:
    return {operation.key: operation for operation in schedule.operations}


class CompiledJSSPOperator:
    """Executable bounded operator; all failures return an unchanged solution."""

    def __init__(self, spec: JSSPOperatorSpec) -> None:
        self.spec = spec
        self.name = spec.name
        self.operator_id = spec.operator_id

    def apply(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorOutcome[JobShopSolution]:
        del context
        parent = JobShopSolution(sequence=tuple(solution.sequence))
        try:
            positions = self._select(parent, instance, rng)
            if positions is None:
                return self._noop(parent, "selector found no bounded move")
            candidate_sequence = self._transform(parent.sequence, positions)
            candidate = JobShopSolution(sequence=candidate_sequence)
            if Counter(candidate.sequence) != Counter(parent.sequence):
                return self._noop(parent, "multiplicity repair rejected candidate")
            if candidate == parent:
                return self._noop(parent, "transform produced no change")
            return OperatorOutcome(
                solution=candidate,
                changed_items=tuple(sorted(set(positions))),
                success=True,
                metadata={
                    "ir_version": self.spec.ir_version,
                    "selector": self.spec.selector.kind,
                    "transform": self.spec.transform.kind,
                    "repair": self.spec.repair.kind,
                },
            )
        except Exception as exc:
            return self._noop(
                parent,
                "operator execution failed",
                exception_type=type(exc).__name__,
            )

    @staticmethod
    def _noop(
        parent: JobShopSolution,
        reason: str,
        **metadata: str,
    ) -> OperatorOutcome[JobShopSolution]:
        return OperatorOutcome(
            solution=JobShopSolution(sequence=tuple(parent.sequence)),
            success=False,
            metadata={"status": "no_change", "reason": reason, **metadata},
            failure_reason=reason,
        )

    def _select(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
        rng: np.random.Generator,
    ) -> tuple[int, int] | None:
        kind = self.spec.selector.kind
        length = len(solution.sequence)
        if length < 2:
            return None
        if kind == "random_adjacent":
            for _ in range(self.spec.selector.max_attempts):
                left = int(rng.integers(0, length - 1))
                if solution.sequence[left] != solution.sequence[left + 1]:
                    return left, left + 1
            return None
        if kind == "random_pair":
            for _ in range(self.spec.selector.max_attempts):
                pair = _pick_distinct_pair(length, rng)
                if pair is not None and solution.sequence[pair[0]] != solution.sequence[pair[1]]:
                    return pair
            return None
        if kind == "bounded_pair":
            distance_limit = min(self.spec.selector.max_distance, length - 1)
            for _ in range(self.spec.selector.max_attempts):
                left = int(rng.integers(0, length - 1))
                distance = int(rng.integers(1, distance_limit + 1))
                right = min(length - 1, left + distance)
                if left != right:
                    return left, right
            return None

        schedule = decode_schedule(solution, instance)
        lookup = _operation_lookup(schedule)
        if kind in {"critical_block_adjacent", "critical_block_endpoints"}:
            eligible = [block for block in schedule.critical_blocks if len(block) >= 2]
            if not eligible:
                return None
            block = eligible[int(rng.integers(0, len(eligible)))]
            if kind == "critical_block_adjacent":
                offset = int(rng.integers(0, len(block) - 1))
                keys = block[offset : offset + 2]
            else:
                keys = (block[0], block[-1])
            return lookup[keys[0]].sequence_index, lookup[keys[1]].sequence_index

        if kind == "bottleneck_block":
            bottleneck = max(
                range(instance.machines),
                key=lambda machine: (schedule.machine_busy_time[machine], -machine),
            )
            keys = schedule.machine_operations[bottleneck]
            if len(keys) < 2:
                return None
            pair = _pick_distinct_pair(len(keys), rng)
            assert pair is not None
            return lookup[keys[pair[0]]].sequence_index, lookup[keys[pair[1]]].sequence_index

        if kind == "high_idle_gap":
            largest: tuple[int, int, int, int] | None = None
            for machine, keys in enumerate(schedule.machine_operations):
                previous_finish = 0
                for key in keys:
                    operation = lookup[key]
                    gap = operation.start - previous_finish
                    candidate = (gap, -machine, -operation.sequence_index, operation.sequence_index)
                    if largest is None or candidate > largest:
                        largest = candidate
                    previous_finish = operation.finish
            if largest is None or largest[0] <= 0:
                return None
            source = largest[3]
            distance = min(self.spec.selector.max_distance, source)
            if distance <= 0:
                return None
            target = source - int(rng.integers(1, distance + 1))
            return source, target
        return None

    def _transform(
        self,
        sequence: tuple[int, ...],
        positions: tuple[int, int],
    ) -> tuple[int, ...]:
        first, second = positions
        values = list(sequence)
        kind = self.spec.transform.kind
        if kind == "swap":
            values[first], values[second] = values[second], values[first]
        elif kind == "insert":
            value = values.pop(first)
            target = second - 1 if first < second else second
            values.insert(max(0, min(target, len(values))), value)
        elif kind == "reverse":
            left, right = sorted((first, second))
            if right - left + 1 > self.spec.transform.max_segment_length:
                right = left + self.spec.transform.max_segment_length - 1
            values[left : right + 1] = reversed(values[left : right + 1])
        return tuple(values)


class JSSPOperatorCompiler:
    """Parse and compile only the closed typed IR; no generated code is executed."""

    def compile(self, ir: JSSPOperatorSpec | object) -> CompiledJSSPOperator:
        spec = ir if isinstance(ir, JSSPOperatorSpec) else JSSPOperatorSpec.model_validate(ir)
        return CompiledJSSPOperator(spec)


__all__ = ["CompiledJSSPOperator", "JSSPOperatorCompiler"]
