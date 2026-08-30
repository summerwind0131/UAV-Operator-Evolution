"""Deterministic split construction and population-freeze access guard."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from operator_evolution_core.contracts import InstanceRef
from operator_evolution_core.evolution import (
    EvolutionSplitCapabilities,
    PopulationFreezeReceipt,
)

from ..models import JobShopInstance
from .generator import generate_training_instances
from .orlib import parse_jobshop1


def _reference(instance: JobShopInstance, split: str) -> InstanceRef:
    return InstanceRef(
        domain_id="jssp",
        instance_id=instance.instance_id,
        split=split,
        difficulty=f"{instance.job_count}x{instance.machines}",
        content_hash=instance.content_hash,
        metadata={
            "source": instance.source,
            "source_family": instance.source_family,
            "jobs": instance.job_count,
            "machines": instance.machines,
        },
    )


class JSSPDatasetSplits(EvolutionSplitCapabilities[JobShopInstance]):
    """Expose held-out instances and their manifest only after population freeze."""

    def manifest(
        self,
        split: str,
        receipt: PopulationFreezeReceipt | None = None,
    ) -> tuple[InstanceRef, ...]:
        if split == "train":
            instances = self.open_train()
        elif split == "validation":
            instances = self.open_validation()
        elif split == "test":
            instances = self.open_test(receipt)
        else:
            raise ValueError("split must be train, validation, or test")
        return tuple(_reference(instance, split) for instance in instances)  # type: ignore[arg-type]


def build_jssp_splits(jobshop1_path: str | Path) -> JSSPDatasetSplits:
    train = generate_training_instances()
    classic = sorted(
        parse_jobshop1(jobshop1_path),
        key=lambda instance: (
            instance.source_family,
            instance.job_count,
            instance.machines,
            instance.content_hash,
        ),
    )
    validation = tuple(instance for index, instance in enumerate(classic) if index % 2 == 0)
    test = tuple(instance for index, instance in enumerate(classic) if index % 2 == 1)
    all_instances: Sequence[JobShopInstance] = (*train, *validation, *test)
    hashes = [instance.content_hash for instance in all_instances]
    if len(hashes) != len(set(hashes)):
        raise ValueError("JSSP train/validation/test splits are not content-disjoint")
    if len(train) != 60 or len(validation) != 41 or len(test) != 41:
        raise RuntimeError("JSSP split cardinalities differ from the registered plan")
    return JSSPDatasetSplits(train=train, validation=validation, test=test)


__all__ = ["JSSPDatasetSplits", "build_jssp_splits"]
