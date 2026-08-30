"""Pre-registered deterministic 60-instance synthetic training corpus."""

from __future__ import annotations

import numpy as np

from ..models import JobShopInstance, Operation

SYNTHETIC_MASTER_SEED = 20260823
SYNTHETIC_SHAPES = ((6, 6), (10, 10), (20, 15))
INSTANCES_PER_SHAPE = 20


def generate_training_instances(
    master_seed: int = SYNTHETIC_MASTER_SEED,
) -> tuple[JobShopInstance, ...]:
    """Generate the exact registered corpus using per-instance child seeds."""

    master_rng = np.random.default_rng(master_seed)
    instances: list[JobShopInstance] = []
    for job_count, machine_count in SYNTHETIC_SHAPES:
        for index in range(INSTANCES_PER_SHAPE):
            child_seed = int(
                master_rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
            )
            rng = np.random.default_rng(child_seed)
            jobs = tuple(
                tuple(
                    Operation(machine=int(machine), duration=int(duration))
                    for machine, duration in zip(
                        rng.permutation(machine_count),
                        rng.integers(1, 100, size=machine_count),
                        strict=True,
                    )
                )
                for _ in range(job_count)
            )
            instances.append(
                JobShopInstance.create(
                    instance_id=(
                        f"synthetic-{job_count}x{machine_count}-{index:02d}"
                    ),
                    jobs=jobs,
                    machines=machine_count,
                    source=f"synthetic-seed-{master_seed}",
                    source_family=f"synthetic-{job_count}x{machine_count}",
                    description=(
                        f"Deterministic synthetic instance; child_seed={child_seed}"
                    ),
                )
            )
    hashes = [instance.content_hash for instance in instances]
    if len(hashes) != len(set(hashes)):
        raise RuntimeError("synthetic training generation produced duplicate content")
    return tuple(instances)


__all__ = [
    "INSTANCES_PER_SHAPE",
    "SYNTHETIC_MASTER_SEED",
    "SYNTHETIC_SHAPES",
    "generate_training_instances",
]
