"""JSSP implementation of the domain-independent adapter contracts."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter
from typing import Any, cast

import numpy as np
from pydantic import JsonValue, ValidationError

from operator_evolution_core.contracts import (
    DomainAdapter,
    ObjectiveEvaluation,
    SearchContextView,
)

from .models import JobShopInstance, JobShopSolution
from .schedule import JobShopSchedule, decode_schedule

JSSP_DOMAIN_ID = "jssp"


def _round_robin_solution(instance: JobShopInstance) -> JobShopSolution:
    return JobShopSolution(
        sequence=tuple(
            job_id
            for _ in range(instance.machines)
            for job_id in range(instance.job_count)
        )
    )


class JobShopInitializer:
    def initialize(
        self,
        instance: JobShopInstance,
        rng: np.random.Generator,
    ) -> JobShopSolution:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        sequence = np.repeat(np.arange(instance.job_count), instance.machines)
        rng.shuffle(sequence)
        return JobShopSolution(sequence=tuple(int(value) for value in sequence))


class JobShopEvaluator:
    def evaluate(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
    ) -> ObjectiveEvaluation:
        schedule = decode_schedule(solution, instance)
        return ObjectiveEvaluation(
            scalar_cost=float(schedule.makespan),
            components={
                "makespan": float(schedule.makespan),
                "machine_total_idle": float(schedule.total_machine_idle),
                "critical_path_length": float(schedule.critical_path_length),
            },
            feasible=schedule.feasible,
            violations={
                "multiplicity": float(schedule.multiplicity_violation),
                "unscheduled_operations": float(schedule.unscheduled_operations),
            },
            metadata={
                "scheduled_operations": len(schedule.operations),
                "critical_operation_count": len(schedule.critical_operations),
            },
        )


class JobShopCodec:
    def clone(self, solution: JobShopSolution) -> JobShopSolution:
        return JobShopSolution(sequence=tuple(solution.sequence))

    def canonicalize(self, solution: object) -> JobShopSolution:
        try:
            if isinstance(solution, JobShopSolution):
                sequence: object = solution.sequence
            elif isinstance(solution, dict):
                sequence = solution.get("sequence")
            else:
                sequence = solution
            if isinstance(sequence, (str, bytes)) or not isinstance(
                sequence, (tuple, list, np.ndarray)
            ):
                raise TypeError("solution must contain an integer sequence")
            values: list[int] = []
            for value in sequence:
                if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                    raise TypeError("job IDs must be integers")
                values.append(int(value))
            return JobShopSolution(sequence=tuple(values))
        except (TypeError, ValueError, ValidationError) as exc:
            raise ValueError(f"invalid JSSP solution: {exc}") from exc

    def to_json(self, solution: JobShopSolution) -> JsonValue:
        return self.canonicalize(solution).canonical_payload()

    def stable_hash(self, solution: JobShopSolution) -> str:
        payload = json.dumps(
            self.to_json(solution),
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JobShopGuard:
    def __init__(self, codec: JobShopCodec) -> None:
        self.codec = codec

    def validate_structure(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
    ) -> list[str]:
        try:
            canonical = self.codec.canonicalize(solution)
        except ValueError as exc:
            return [str(exc)]
        violations: list[str] = []
        if len(canonical.sequence) != instance.operation_count:
            violations.append(
                f"sequence length must be {instance.operation_count}, "
                f"got {len(canonical.sequence)}"
            )
        invalid = [
            job_id
            for job_id in canonical.sequence
            if job_id < 0 or job_id >= instance.job_count
        ]
        if invalid:
            violations.append("job IDs must be within the instance job range")
        counts = Counter(canonical.sequence)
        for job_id in range(instance.job_count):
            if counts[job_id] != instance.machines:
                violations.append(
                    f"job {job_id} multiplicity must be {instance.machines}, "
                    f"got {counts[job_id]}"
                )
        return violations


class JobShopFeatureExtractor:
    def extract(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
        evaluation: ObjectiveEvaluation,
    ) -> dict[str, JsonValue]:
        schedule = decode_schedule(solution, instance)
        makespan = max(schedule.makespan, 1)
        busy = schedule.machine_busy_time
        mean_busy = statistics.fmean(busy) if busy else 0.0
        reference = decode_schedule(_round_robin_solution(instance), instance).makespan
        expected_positions: dict[tuple[int, int], int] = {}
        for operation_index in range(instance.machines):
            for job_id in range(instance.job_count):
                expected_positions[(job_id, operation_index)] = (
                    operation_index * instance.job_count + job_id
                )
        displacement = sum(
            abs(item.sequence_index - expected_positions[item.key])
            for item in schedule.operations
        )
        displacement_scale = max(1, instance.operation_count**2)
        longest_block = max((len(block) for block in schedule.critical_blocks), default=0)
        return {
            "critical_path_ratio": len(schedule.critical_operations)
            / max(1, instance.operation_count),
            "bottleneck_machine_utilization": max(busy, default=0) / makespan,
            "machine_load_imbalance": (
                statistics.pstdev(busy) / mean_busy if mean_busy > 0 else 0.0
            ),
            "critical_block_count": len(schedule.critical_blocks),
            "critical_block_max_ratio": longest_block / max(1, instance.machines),
            "operation_displacement": displacement / displacement_scale,
            "relative_initial_improvement": (reference - schedule.makespan)
            / max(1, reference),
            "makespan": float(evaluation.scalar_cost),
        }


class JobShopTraceEncoder:
    def __init__(self, features: JobShopFeatureExtractor, codec: JobShopCodec) -> None:
        self.features = features
        self.codec = codec

    def snapshot(
        self,
        solution: JobShopSolution,
        instance: JobShopInstance,
        evaluation: ObjectiveEvaluation,
        context: SearchContextView,
    ) -> dict[str, JsonValue]:
        return cast(
            dict[str, JsonValue],
            {
                "solution": self.codec.to_json(solution),
                "solution_hash": self.codec.stable_hash(solution),
                "instance": {
                    "instance_id": instance.instance_id,
                    "jobs": instance.job_count,
                    "machines": instance.machines,
                    "content_hash": instance.content_hash,
                },
                "objective": float(evaluation.scalar_cost),
                "objective_components": dict(evaluation.components),
                "violations": dict(evaluation.violations),
                "feasible": bool(evaluation.feasible),
                "features": self.features.extract(
                    solution, instance, evaluation
                ),
                "context": dict(context.as_features()),
            },
        )


def create_jssp_domain_adapter() -> DomainAdapter[JobShopInstance, JobShopSolution]:
    codec = JobShopCodec()
    features = JobShopFeatureExtractor()
    return DomainAdapter(
        domain_id=JSSP_DOMAIN_ID,
        initializer=JobShopInitializer(),
        evaluator=JobShopEvaluator(),
        features=features,
        codec=codec,
        guard=JobShopGuard(codec),
        trace_encoder=JobShopTraceEncoder(features, codec),
    )


__all__ = [
    "JSSP_DOMAIN_ID",
    "JobShopCodec",
    "JobShopEvaluator",
    "JobShopFeatureExtractor",
    "JobShopGuard",
    "JobShopInitializer",
    "JobShopTraceEncoder",
    "create_jssp_domain_adapter",
]
