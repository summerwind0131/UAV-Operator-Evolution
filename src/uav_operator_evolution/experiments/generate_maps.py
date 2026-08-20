"""Map-generation workflow."""

from __future__ import annotations

from ..config import ExperimentConfig
from ..environment.generator import DatasetManifest, generate_dataset


def run_generate_maps(config: ExperimentConfig, *, overwrite: bool = False) -> DatasetManifest:
    return generate_dataset(config, overwrite=overwrite)

