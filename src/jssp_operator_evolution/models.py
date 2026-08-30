"""Canonical immutable models for deterministic job-shop scheduling."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Operation(_FrozenModel):
    """One non-preemptive operation in job precedence order."""

    machine: int = Field(ge=0)
    # OR-Library's intentionally "doomed" orb07 instance contains one
    # documented zero-duration operation, so the classic corpus requires >= 0.
    duration: int = Field(ge=0)


class JobShopInstance(_FrozenModel):
    """Content-addressed classical job-shop instance."""

    instance_id: str = Field(min_length=1, max_length=255)
    jobs: tuple[tuple[Operation, ...], ...] = Field(min_length=1)
    machines: int = Field(ge=1)
    source: str = Field(min_length=1, max_length=255)
    source_family: str = Field(min_length=1, max_length=100)
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    description: str | None = None
    best_known_makespan: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_job_structure(self) -> Self:
        expected_machines = set(range(self.machines))
        for job_index, job in enumerate(self.jobs):
            if len(job) != self.machines:
                raise ValueError(
                    f"job {job_index} has {len(job)} operations; "
                    f"expected {self.machines}"
                )
            machine_ids = {operation.machine for operation in job}
            if machine_ids != expected_machines:
                raise ValueError(
                    f"job {job_index} machine order must be a permutation of "
                    f"0..{self.machines - 1}"
                )
        if self.content_hash != self.compute_content_hash(self.jobs, self.machines):
            raise ValueError("content_hash does not match normalized instance content")
        return self

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def operation_count(self) -> int:
        return self.job_count * self.machines

    @property
    def total_processing_time(self) -> int:
        return sum(operation.duration for job in self.jobs for operation in job)

    @staticmethod
    def normalized_content(
        jobs: tuple[tuple[Operation, ...], ...], machines: int
    ) -> dict[str, Any]:
        return {
            "jobs": [
                [[operation.machine, operation.duration] for operation in job]
                for job in jobs
            ],
            "machines": int(machines),
        }

    @classmethod
    def compute_content_hash(
        cls, jobs: tuple[tuple[Operation, ...], ...], machines: int
    ) -> str:
        return _sha256(cls.normalized_content(jobs, machines))

    @classmethod
    def create(
        cls,
        *,
        instance_id: str,
        jobs: tuple[tuple[Operation, ...], ...],
        machines: int,
        source: str,
        source_family: str,
        description: str | None = None,
        best_known_makespan: int | None = None,
    ) -> Self:
        return cls(
            instance_id=instance_id,
            jobs=jobs,
            machines=machines,
            source=source,
            source_family=source_family,
            content_hash=cls.compute_content_hash(jobs, machines),
            description=description,
            best_known_makespan=best_known_makespan,
        )

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            **self.normalized_content(self.jobs, self.machines),
            "source": self.source,
            "source_family": self.source_family,
            "content_hash": self.content_hash,
            "description": self.description,
            "best_known_makespan": self.best_known_makespan,
        }


class JobShopSolution(_FrozenModel):
    """Operation-based representation: each entry selects a job's next operation."""

    sequence: tuple[int, ...]

    def canonical_payload(self) -> list[int]:
        return [int(job_id) for job_id in self.sequence]


__all__ = ["JobShopInstance", "JobShopSolution", "Operation"]
