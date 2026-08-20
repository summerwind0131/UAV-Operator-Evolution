"""Create a hash-bound, reproducible Evolutionary AFL-UAV v1 method artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import socket
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "evolutionary_afl_uav_v1.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "evolutionary-afl-uav-methods"
    / "evolutionary-afl-uav-v1"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if ROOT.resolve() not in path.parents and path != ROOT.resolve():
        raise ValueError(f"freeze input escapes project root: {relative}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    if suite is None:
        raise RuntimeError("pytest JUnit XML does not contain a testsuite")
    summary = {
        name: int(float(suite.attrib.get(name, "0")))
        for name in ("tests", "failures", "errors", "skipped")
    }
    if summary["failures"] or summary["errors"]:
        raise RuntimeError(f"pytest receipt is not passing: {summary}")
    if summary["tests"] <= 0:
        raise RuntimeError("pytest receipt contains no tests")
    return summary


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    core = _resolve(config["frozen_core_source"])
    seed_artifact_path = _resolve(config["seed_artifact"])
    dataset_manifest = _resolve(config["dataset_manifest"])
    validation_receipt = _resolve(config["validation_receipt"])
    experiment_matrix = _resolve(config["experiment_matrix"])
    experiments_receipt = _resolve(config["experiments_receipt"])
    pytest_junit = _resolve(config["pytest_junit"])

    verified_hashes = {
        "core_source_sha256": _require_hash(
            core, config["expected_core_sha256"], "Evolutionary AFL-UAV core"
        ),
        "dataset_manifest_sha256": _require_hash(
            dataset_manifest,
            config["expected_dataset_manifest_sha256"],
            "uav2d-v1 manifest",
        ),
        "validation_receipt_sha256": _require_hash(
            validation_receipt,
            config["expected_validation_receipt_sha256"],
            "Validation analysis receipt",
        ),
        "experiment_matrix_sha256": _require_hash(
            experiment_matrix,
            config["expected_experiment_matrix_sha256"],
            "ablation and sensitivity matrix",
        ),
        "experiments_receipt_sha256": _require_hash(
            experiments_receipt,
            config["expected_experiments_receipt_sha256"],
            "ablation and sensitivity analysis receipt",
        ),
        "pytest_junit_sha256": _sha256(pytest_junit),
        "freeze_config_sha256": _sha256(config_path),
    }
    seed_artifact = json.loads(seed_artifact_path.read_text(encoding="utf-8"))
    if seed_artifact.get("artifact_id") != config["expected_seed_artifact_id"]:
        raise RuntimeError("frozen seed artifact id changed")
    if seed_artifact.get("solver_hash") != config["expected_seed_solver_hash"]:
        raise RuntimeError("frozen seed solver hash changed")
    if not seed_artifact.get("research_claim_eligible"):
        raise RuntimeError("frozen seed artifact is not research-claim eligible")
    test_summary = _pytest_summary(pytest_junit)

    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_source = output_dir / "evolutionary_afl_v1.py"
    frozen_config = output_dir / "evolutionary_afl_uav_v1.yaml"
    frozen_experiment_matrix = output_dir / "evolutionary_afl_uav_experiments_v1.yaml"
    shutil.copyfile(core, frozen_source)
    shutil.copyfile(config_path, frozen_config)
    shutil.copyfile(experiment_matrix, frozen_experiment_matrix)
    if _sha256(frozen_source) != verified_hashes["core_source_sha256"]:
        raise RuntimeError("frozen source copy hash mismatch")

    payload = {
        "schema_version": "evolutionary-afl-method-artifact-v1",
        "method_id": config["method_id"],
        "status": config["status"],
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "source": {
            "project_path": config["frozen_core_source"],
            "frozen_filename": frozen_source.name,
            "sha256": verified_hashes["core_source_sha256"],
        },
        "parameters": config["parameters"],
        "development_policy": config["development_policy"],
        "seed_artifact": {
            "path": config["seed_artifact"],
            "artifact_id": seed_artifact["artifact_id"],
            "solver_hash": seed_artifact["solver_hash"],
            "provider": seed_artifact["provider"],
            "model": seed_artifact["model"],
            "research_claim_eligible": seed_artifact["research_claim_eligible"],
        },
        "evidence": {
            "dataset_manifest": config["dataset_manifest"],
            "validation_receipt": config["validation_receipt"],
            "experiment_matrix": config["experiment_matrix"],
            "experiments_receipt": config["experiments_receipt"],
            "pytest_junit": config["pytest_junit"],
            "pytest_summary": test_summary,
            "verified_hashes": verified_hashes,
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pydantic", "PyYAML")
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version,
        },
        "version_control": {
            "available": False,
            "note": "workspace .git directory is empty; reproducibility is hash-bound",
        },
    }
    payload["artifact_id"] = _canonical_hash(payload)
    artifact_path = output_dir / "artifact.json"
    _write_json(artifact_path, payload)
    receipt = {
        "status": "passed",
        "method_id": config["method_id"],
        "artifact_id": payload["artifact_id"],
        "files": {
            artifact_path.name: _sha256(artifact_path),
            frozen_source.name: _sha256(frozen_source),
            frozen_config.name: _sha256(frozen_config),
            frozen_experiment_matrix.name: _sha256(frozen_experiment_matrix),
            pytest_junit.name: _sha256(pytest_junit),
        },
    }
    _write_json(output_dir / "freeze_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(args.config.resolve(), args.output_dir.resolve()),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
