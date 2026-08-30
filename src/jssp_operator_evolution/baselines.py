"""Deterministic JSSP sanity baselines used before evolution experiments."""

from __future__ import annotations

import numpy as np

from .adapter import JobShopEvaluator
from .models import JobShopInstance, JobShopSolution


def random_sequence(
    instance: JobShopInstance,
    seed: int,
) -> JobShopSolution:
    values = np.repeat(np.arange(instance.job_count), instance.machines)
    rng = np.random.default_rng(seed)
    rng.shuffle(values)
    return JobShopSolution(sequence=tuple(int(value) for value in values))


def spt_dispatch(instance: JobShopInstance) -> JobShopSolution:
    """Select the job whose next precedence-feasible operation is shortest."""

    next_operation = [0] * instance.job_count
    sequence: list[int] = []
    for _ in range(instance.operation_count):
        available = [
            job_id
            for job_id in range(instance.job_count)
            if next_operation[job_id] < instance.machines
        ]
        selected = min(
            available,
            key=lambda job_id: (
                instance.jobs[job_id][next_operation[job_id]].duration,
                job_id,
            ),
        )
        sequence.append(selected)
        next_operation[selected] += 1
    return JobShopSolution(sequence=tuple(sequence))


def adjacent_swap_hill_climb(
    instance: JobShopInstance,
    initial: JobShopSolution,
    *,
    max_iterations: int,
) -> JobShopSolution:
    """Best-improvement adjacent-swap descent with an evaluation-call budget."""

    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    evaluator = JobShopEvaluator()
    current = JobShopSolution(sequence=tuple(initial.sequence))
    current_cost = evaluator.evaluate(current, instance).scalar_cost
    calls = 0
    while calls < max_iterations:
        best = current
        best_cost = current_cost
        for left in range(len(current.sequence) - 1):
            if calls >= max_iterations:
                break
            if current.sequence[left] == current.sequence[left + 1]:
                continue
            values = list(current.sequence)
            values[left], values[left + 1] = values[left + 1], values[left]
            candidate = JobShopSolution(sequence=tuple(values))
            cost = evaluator.evaluate(candidate, instance).scalar_cost
            calls += 1
            if cost < best_cost:
                best = candidate
                best_cost = cost
        if best_cost >= current_cost:
            break
        current, current_cost = best, best_cost
    return current


__all__ = ["adjacent_swap_hill_climb", "random_sequence", "spt_dispatch"]
