"""Deterministic serial schedule builder for operation-based JSSP solutions."""

from __future__ import annotations

from dataclasses import dataclass

from .models import JobShopInstance, JobShopSolution

OperationKey = tuple[int, int]


@dataclass(frozen=True, slots=True)
class ScheduledOperation:
    job: int
    operation: int
    machine: int
    duration: int
    start: int
    finish: int
    sequence_index: int
    job_predecessor: OperationKey | None
    machine_predecessor: OperationKey | None

    @property
    def key(self) -> OperationKey:
        return (self.job, self.operation)


@dataclass(frozen=True, slots=True)
class JobShopSchedule:
    operations: tuple[ScheduledOperation, ...]
    makespan: int
    total_machine_idle: int
    critical_path_length: int
    critical_operations: tuple[OperationKey, ...]
    critical_blocks: tuple[tuple[OperationKey, ...], ...]
    machine_busy_time: tuple[int, ...]
    machine_operations: tuple[tuple[OperationKey, ...], ...]
    invalid_job_ids: int
    excess_operations: int
    unscheduled_operations: int

    @property
    def feasible(self) -> bool:
        return not (
            self.invalid_job_ids
            or self.excess_operations
            or self.unscheduled_operations
        )

    @property
    def multiplicity_violation(self) -> int:
        return self.invalid_job_ids + self.excess_operations + self.unscheduled_operations


def decode_schedule(
    solution: JobShopSolution,
    instance: JobShopInstance,
) -> JobShopSchedule:
    """Decode without a solver, using stable job and machine earliest-start rules."""

    next_operation = [0] * instance.job_count
    job_ready = [0] * instance.job_count
    machine_ready = [0] * instance.machines
    last_job_operation: list[OperationKey | None] = [None] * instance.job_count
    last_machine_operation: list[OperationKey | None] = [None] * instance.machines
    machine_keys: list[list[OperationKey]] = [[] for _ in range(instance.machines)]
    machine_busy = [0] * instance.machines
    scheduled: list[ScheduledOperation] = []
    by_key: dict[OperationKey, ScheduledOperation] = {}
    invalid_job_ids = 0
    excess_operations = 0

    for sequence_index, raw_job_id in enumerate(solution.sequence):
        job_id = int(raw_job_id)
        if job_id < 0 or job_id >= instance.job_count:
            invalid_job_ids += 1
            continue
        operation_index = next_operation[job_id]
        if operation_index >= len(instance.jobs[job_id]):
            excess_operations += 1
            continue
        operation = instance.jobs[job_id][operation_index]
        start = max(job_ready[job_id], machine_ready[operation.machine])
        finish = start + operation.duration
        item = ScheduledOperation(
            job=job_id,
            operation=operation_index,
            machine=operation.machine,
            duration=operation.duration,
            start=start,
            finish=finish,
            sequence_index=sequence_index,
            job_predecessor=last_job_operation[job_id],
            machine_predecessor=last_machine_operation[operation.machine],
        )
        scheduled.append(item)
        by_key[item.key] = item
        machine_keys[operation.machine].append(item.key)
        machine_busy[operation.machine] += operation.duration
        next_operation[job_id] += 1
        job_ready[job_id] = finish
        machine_ready[operation.machine] = finish
        last_job_operation[job_id] = item.key
        last_machine_operation[operation.machine] = item.key

    makespan = max((item.finish for item in scheduled), default=0)
    unscheduled = instance.operation_count - len(scheduled)
    total_machine_idle = instance.machines * makespan - sum(machine_busy)

    critical_reverse: list[OperationKey] = []
    if scheduled:
        current = min(
            (item for item in scheduled if item.finish == makespan),
            key=lambda item: item.key,
        )
        while True:
            critical_reverse.append(current.key)
            candidates = [
                by_key[key]
                for key in (current.job_predecessor, current.machine_predecessor)
                if key is not None and by_key[key].finish == current.start
            ]
            if not candidates:
                break
            current = min(candidates, key=lambda item: (item.finish, item.key))
    critical_operations = tuple(reversed(critical_reverse))
    critical_path_length = sum(by_key[key].duration for key in critical_operations)

    blocks: list[tuple[OperationKey, ...]] = []
    current_block: list[OperationKey] = []
    current_machine: int | None = None
    for key in critical_operations:
        machine = by_key[key].machine
        if machine != current_machine:
            if len(current_block) >= 2:
                blocks.append(tuple(current_block))
            current_block = [key]
            current_machine = machine
        else:
            current_block.append(key)
    if len(current_block) >= 2:
        blocks.append(tuple(current_block))

    return JobShopSchedule(
        operations=tuple(scheduled),
        makespan=makespan,
        total_machine_idle=total_machine_idle,
        critical_path_length=critical_path_length,
        critical_operations=critical_operations,
        critical_blocks=tuple(blocks),
        machine_busy_time=tuple(machine_busy),
        machine_operations=tuple(tuple(keys) for keys in machine_keys),
        invalid_job_ids=invalid_job_ids,
        excess_operations=excess_operations,
        unscheduled_operations=max(0, unscheduled),
    )


__all__ = [
    "JobShopSchedule",
    "OperationKey",
    "ScheduledOperation",
    "decode_schedule",
]
