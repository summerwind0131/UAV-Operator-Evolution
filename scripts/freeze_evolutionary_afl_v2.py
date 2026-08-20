"""Create a hash-bound Evolutionary AFL-UAV v2 method artifact."""

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
DEFAULT_CONFIG = ROOT / "configs/evolutionary_afl_uav_v2_freeze.yaml"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/planning_benchmarks/evolutionary-afl-uav-methods/"
    / "evolutionary-afl-uav-v2"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _resolve(relative: str) -> Path:
    path = (ROOT / relative).resolve()
    if path != ROOT.resolve() and ROOT.resolve() not in path.parents:
        raise ValueError(f"freeze input escapes project root: {relative}")
    return path


def _require_hash(path: Path, expected: str, label: str) -> str:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} hash changed: expected {expected}, got {actual}")
    return actual


def _pytest_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    summary = {
        field: sum(int(float(suite.attrib.get(field, "0"))) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }
    if summary["tests"] <= 0 or summary["failures"] or summary["errors"]:
        raise RuntimeError(f"pytest receipt is not passing: {summary}")
    return summary


def freeze(config_path: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise RuntimeError("V2 frozen artifact directory already exists; never overwrite it")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    inputs = {
        "core_source": (
            _resolve(config["frozen_core_source"]),
            config["expected_core_sha256"],
        ),
        "candidate_config": (
            _resolve(config["candidate_config"]),
            config["expected_candidate_config_sha256"],
        ),
        "seed_artifact": (
            _resolve(config["seed_artifact"]),
            config["expected_seed_artifact_sha256"],
        ),
        "dataset_manifest": (
            _resolve(config["dataset_manifest"]),
            config["expected_dataset_manifest_sha256"],
        ),
        "validation_comparison": (
            _resolve(config["validation_comparison"]),
            config["expected_validation_comparison_sha256"],
        ),
        "development_receipt": (
            _resolve(config["development_receipt"]),
            config["expected_development_receipt_sha256"],
        ),
        "pytest_junit": (
            _resolve(config["pytest_junit"]),
            config["expected_pytest_junit_sha256"],
        ),
    }
    verified = {
        name + "_sha256": _require_hash(path, expected, name)
        for name, (path, expected) in inputs.items()
    }
    verified["freeze_config_sha256"] = _sha256(config_path)
    seed = json.loads(inputs["seed_artifact"][0].read_text(encoding="utf-8"))
    if seed.get("artifact_id") != config["expected_seed_artifact_id"]:
        raise RuntimeError("seed artifact ID changed")
    if seed.get("solver_hash") != config["expected_seed_solver_hash"]:
        raise RuntimeError("seed solver hash changed")
    if not seed.get("research_claim_eligible"):
        raise RuntimeError("seed artifact is not research-claim eligible")
    tests = _pytest_summary(inputs["pytest_junit"][0])

    output_dir.mkdir(parents=True)
    frozen_source = output_dir / "evolutionary_afl_v2.py"
    frozen_config = output_dir / "evolutionary_afl_uav_v2_freeze.yaml"
    frozen_candidate = output_dir / "evolutionary_afl_uav_v2_candidate.yaml"
    frozen_pytest = output_dir / "pytest.xml"
    shutil.copyfile(inputs["core_source"][0], frozen_source)
    shutil.copyfile(config_path, frozen_config)
    shutil.copyfile(inputs["candidate_config"][0], frozen_candidate)
    shutil.copyfile(inputs["pytest_junit"][0], frozen_pytest)

    artifact: dict[str, Any] = {
        "schema_version": "evolutionary-afl-method-artifact-v2",
        "method_id": config["method_id"],
        "status": "frozen",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "research_claim_eligible": True,
        "source": {
            "project_path": config["frozen_core_source"],
            "frozen_filename": frozen_source.name,
            "sha256": verified["core_source_sha256"],
        },
        "parameters": config["parameters"],
        "analysis_contract": config["analysis_contract"],
        "development_policy": config["development_policy"],
        "seed_artifact": {
            "path": config["seed_artifact"],
            "artifact_id": seed["artifact_id"],
            "solver_hash": seed["solver_hash"],
            "provider": seed["provider"],
            "model": seed["model"],
            "research_claim_eligible": seed["research_claim_eligible"],
        },
        "evidence": {
            "dataset_manifest": config["dataset_manifest"],
            "validation_comparison": config["validation_comparison"],
            "development_receipt": config["development_receipt"],
            "pytest_junit": config["pytest_junit"],
            "pytest_summary": tests,
            "verified_hashes": verified,
            "excluded_obsolete_tests": [
                "test_authorized_entry_refuses_while_sealed",
                "test_hidden_test_v2_is_balanced_unique_and_sealed_unrun",
            ],
        },
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pydantic", "PyYAML")
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
    }
    artifact["artifact_id"] = _canonical_hash(artifact)
    artifact_path = output_dir / "artifact.json"
    _write_json(artifact_path, artifact)
    receipt: dict[str, Any] = {
        "schema_version": "evolutionary-afl-method-freeze-receipt-v2",
        "status": "passed",
        "method_id": config["method_id"],
        "artifact_id": artifact["artifact_id"],
        "research_claim_eligible": True,
        "files": {
            artifact_path.name: _sha256(artifact_path),
            frozen_source.name: _sha256(frozen_source),
            frozen_config.name: _sha256(frozen_config),
            frozen_candidate.name: _sha256(frozen_candidate),
            frozen_pytest.name: _sha256(frozen_pytest),
        },
    }
    receipt["freeze_receipt_id"] = _canonical_hash(receipt)
    _write_json(output_dir / "freeze_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.config.resolve(), args.output_dir.resolve()), indent=2))


if __name__ == "__main__":
    main()
