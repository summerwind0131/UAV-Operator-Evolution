"""Shared dataset, JSON, and latest-run helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import ExperimentConfig
from ..environment.generator import (
    DatasetManifest,
    SplitName,
    generate_dataset,
    load_dataset,
    load_dataset_split,
)


def ensure_dataset(config: ExperimentConfig, *, overwrite: bool = False):
    manifest_path = config.output.data_dir / "manifest.json"
    if not manifest_path.exists() or overwrite:
        generate_dataset(config, overwrite=overwrite)
    return load_dataset(manifest_path)


def ensure_dataset_split(
    config: ExperimentConfig,
    split: SplitName,
    *,
    overwrite: bool = False,
) -> list:
    """Generate if needed, then open only the requested experimental split."""

    manifest_path = config.output.data_dir / "manifest.json"
    if not manifest_path.exists() or overwrite:
        generate_dataset(config, overwrite=overwrite)
    return load_dataset_split(manifest_path, split)


def write_json(path: str | Path, value: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    destination.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return destination


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination, index=False)
    return destination


def update_latest(config: ExperimentConfig, run_id: str, result_dir: Path) -> Path:
    return write_json(
        config.output.results_dir / "latest.json",
        {"run_id": run_id, "result_dir": str(result_dir.resolve())},
    )


def resolve_run_dir(config: ExperimentConfig, run_dir: str | Path | None) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    latest = config.output.results_dir / "latest.json"
    if not latest.exists():
        raise FileNotFoundError("no prior run found; pass --run-dir or run run-search/evolve/demo first")
    payload = json.loads(latest.read_text(encoding="utf-8"))
    return Path(payload["result_dir"])
