"""Hash and receipt utilities for the sealed UAV2D Hidden Test-v3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..environment.generator import DatasetManifest
from ..reproducibility import stable_hash


ROOT = Path(__file__).resolve().parents[3]
EXPECTED_RECORDS = 6360
DEFAULT_PROTOCOL = ROOT / "configs/uav_hidden_test_v3.yaml"
DEFAULT_EXECUTION_CONFIG = ROOT / "configs/uav_hidden_test_v3_execution_v1.yaml"
DEFAULT_DATA_ROOT = ROOT / "data/benchmarks/uav2d-hidden-test-v3"
DEFAULT_PREREGISTRATION = DEFAULT_DATA_ROOT / "preregistration.json"
DEFAULT_SEAL_RECEIPT = DEFAULT_DATA_ROOT / "seal_receipt.json"
DEFAULT_SEAL_MARKER = DEFAULT_DATA_ROOT / "SEALED.json"
DEFAULT_OPENING_RECEIPT = DEFAULT_DATA_ROOT / "opening_receipt.json"
DEFAULT_ARCHIVED_SEAL = DEFAULT_DATA_ROOT / "SEALED.preopening.json"
DEFAULT_PREFLIGHT_ROOT = (
    ROOT / "artifacts/planning_benchmarks/uav2d-hidden-test-v3-preflight-v1"
)
DEFAULT_PREFLIGHT_RECEIPT = DEFAULT_PREFLIGHT_ROOT / "preflight_receipt.json"
DEFAULT_FINAL_RESULTS = (
    ROOT / "artifacts/planning_benchmarks/uav2d-hidden-test-v3-final"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(payload: dict[str, Any], id_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(id_field, None)
    return stable_hash(canonical)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def read_protocol(path: str | Path = DEFAULT_PROTOCOL) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Hidden Test-v3 protocol must be a YAML mapping")
    return value


def resolve_project_path(value: str | Path) -> Path:
    raw = Path(value)
    resolved = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
    root = ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"path escapes project root: {value}")
    return resolved


def project_relative(path: str | Path) -> str:
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def require_hash(path: str | Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected}, observed {actual}"
        )


def validate_preregistration(
    *,
    preregistration_path: str | Path = DEFAULT_PREREGISTRATION,
    protocol_path: str | Path = DEFAULT_PROTOCOL,
    requested_id: str,
) -> dict[str, Any]:
    preregistration_path = Path(preregistration_path)
    protocol_path = Path(protocol_path)
    prereg = read_json(preregistration_path)
    if prereg.get("schema_version") != "uav2d-final-evaluation-preregistration-v1":
        raise RuntimeError("unsupported final-evaluation preregistration schema")
    if prereg.get("status") != "preregistered_sealed_unrun":
        raise RuntimeError("preregistration is not sealed-unrun")
    if requested_id != prereg.get("preregistration_id"):
        raise RuntimeError("supplied preregistration ID does not match")
    if canonical_id(prereg, "preregistration_id") != requested_id:
        raise RuntimeError("preregistration self-hash is invalid")
    require_hash(protocol_path, prereg["protocol_config_sha256"], "protocol config")
    if int(prereg["expected_execution_matrix"]["total_records"]) != EXPECTED_RECORDS:
        raise RuntimeError("preregistration does not commit to 6,360 rows")
    schedule = prereg["seed_schedule"]
    if int(schedule["rows"]) != EXPECTED_RECORDS:
        raise RuntimeError("seed schedule is not exactly 6,360 rows")
    require_hash(
        resolve_project_path(schedule["path"]),
        schedule["sha256"],
        "seed schedule",
    )
    dataset = prereg["dataset"]
    manifest_path = resolve_project_path(dataset["manifest"])
    require_hash(manifest_path, dataset["manifest_sha256"], "manifest")
    manifest = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest.manifest_hash != dataset["manifest_content_hash"]:
        raise RuntimeError("manifest content hash mismatch")
    return prereg


def validate_current_implementation_hashes(preregistration: dict[str, Any]) -> None:
    for relative, expected in preregistration["frozen_input_hashes"].items():
        require_hash(
            resolve_project_path(relative),
            str(expected),
            f"frozen input {relative}",
        )


def validate_preflight_receipt(
    path: str | Path,
    *,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("schema_version") != "uav2d-final-preflight-receipt-v3":
        raise RuntimeError("unsupported V3 preflight receipt schema")
    if receipt.get("status") != "passed":
        raise RuntimeError("V3 preflight did not pass")
    if receipt.get("preregistration_id") != preregistration["preregistration_id"]:
        raise RuntimeError("preflight targets another preregistration")
    if canonical_id(receipt, "preflight_receipt_id") != receipt.get(
        "preflight_receipt_id"
    ):
        raise RuntimeError("preflight receipt self-hash is invalid")
    validate_current_implementation_hashes(preregistration)
    return receipt


def validate_opening_receipt(
    path: str | Path,
    *,
    preregistration: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("schema_version") != "uav2d-hidden-test-opening-v3":
        raise RuntimeError("unsupported V3 opening receipt schema")
    if receipt.get("status") != "authorized_open":
        raise RuntimeError("opening receipt is not authorized")
    if receipt.get("explicit_user_authorization") is not True:
        raise RuntimeError("opening receipt lacks explicit user authorization")
    expected = {
        "benchmark_id": preregistration["benchmark_id"],
        "preregistration_id": preregistration["preregistration_id"],
        "preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "preflight_receipt_id": preflight["preflight_receipt_id"],
        "preflight_receipt_sha256": sha256_file(DEFAULT_PREFLIGHT_RECEIPT),
        "manifest_sha256": preregistration["dataset"]["manifest_sha256"],
        "seed_schedule_sha256": preregistration["seed_schedule"]["sha256"],
    }
    for field, value in expected.items():
        if receipt.get(field) != value:
            raise RuntimeError(f"opening receipt field mismatch: {field}")
    if canonical_id(receipt, "opening_id") != receipt.get("opening_id"):
        raise RuntimeError("opening receipt self-hash is invalid")
    if not DEFAULT_ARCHIVED_SEAL.is_file():
        raise RuntimeError("opening receipt exists without archived seal")
    require_hash(
        DEFAULT_ARCHIVED_SEAL,
        receipt["archived_seal_sha256"],
        "archived SEALED marker",
    )
    return receipt


__all__ = [
    "DEFAULT_ARCHIVED_SEAL",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_EXECUTION_CONFIG",
    "DEFAULT_FINAL_RESULTS",
    "DEFAULT_OPENING_RECEIPT",
    "DEFAULT_PREFLIGHT_RECEIPT",
    "DEFAULT_PREFLIGHT_ROOT",
    "DEFAULT_PREREGISTRATION",
    "DEFAULT_PROTOCOL",
    "DEFAULT_SEAL_MARKER",
    "DEFAULT_SEAL_RECEIPT",
    "EXPECTED_RECORDS",
    "ROOT",
    "canonical_id",
    "project_relative",
    "read_json",
    "read_protocol",
    "require_hash",
    "resolve_project_path",
    "sha256_file",
    "validate_current_implementation_hashes",
    "validate_opening_receipt",
    "validate_preflight_receipt",
    "validate_preregistration",
    "write_json",
]
