"""Hash and receipt utilities for the sealed UAV2D final evaluation.

The helpers in this module deliberately inspect only protocol metadata.  Hidden
map JSON files are touched only by the authorized executor after the seal has
been opened.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from ..reproducibility import stable_hash
from ..environment.generator import DatasetManifest


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = ROOT / "configs/uav_hidden_test_v2.yaml"
DEFAULT_EXECUTION_CONFIG = ROOT / "configs/uav_hidden_test_v2_execution_v1.yaml"
DEFAULT_DATA_ROOT = ROOT / "data/benchmarks/uav2d-hidden-test-v2"
DEFAULT_PREREGISTRATION = DEFAULT_DATA_ROOT / "preregistration.json"
DEFAULT_SEAL_RECEIPT = DEFAULT_DATA_ROOT / "seal_receipt.json"
DEFAULT_SEAL_MARKER = DEFAULT_DATA_ROOT / "SEALED.json"
DEFAULT_OPENING_RECEIPT = DEFAULT_DATA_ROOT / "opening_receipt.json"
DEFAULT_ARCHIVED_SEAL = DEFAULT_DATA_ROOT / "SEALED.preopening.json"
DEFAULT_PREFLIGHT_ROOT = (
    ROOT / "artifacts/planning_benchmarks/uav2d-hidden-test-v2-preflight-v1"
)
DEFAULT_ADDENDUM = DEFAULT_PREFLIGHT_ROOT / "preregistration_addendum.json"
DEFAULT_PREFLIGHT_RECEIPT = DEFAULT_PREFLIGHT_ROOT / "preflight_receipt.json"
DEFAULT_FINAL_RESULTS = (
    ROOT / "artifacts/planning_benchmarks/uav2d-hidden-test-v2-final"
)


def sha256_file(path: str | Path) -> str:
    """Return a byte-exact SHA-256 digest."""

    target = Path(path)
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(payload: dict[str, Any], id_field: str) -> str:
    """Hash a receipt after removing its self-referential identifier."""

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
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("hidden-test protocol must be a YAML mapping")
    return payload


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a preregistered path without allowing project-root escapes."""

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
    """Validate the immutable base preregistration without reading map JSON."""

    preregistration_path = Path(preregistration_path)
    protocol_path = Path(protocol_path)
    prereg = read_json(preregistration_path)
    if prereg.get("schema_version") != "uav2d-final-evaluation-preregistration-v1":
        raise RuntimeError("unsupported final-evaluation preregistration schema")
    if prereg.get("status") != "preregistered_sealed_unrun":
        raise RuntimeError("preregistration is not in sealed-unrun state")
    if requested_id != prereg.get("preregistration_id"):
        raise RuntimeError("the supplied preregistration ID does not match")
    if canonical_id(prereg, "preregistration_id") != requested_id:
        raise RuntimeError("preregistration self-hash is invalid")
    require_hash(
        protocol_path,
        str(prereg["protocol_config_sha256"]),
        "protocol config",
    )
    if int(prereg["expected_execution_matrix"]["total_records"]) != 6960:
        raise RuntimeError("preregistration does not commit to exactly 6,960 rows")
    schedule = prereg["seed_schedule"]
    schedule_path = resolve_project_path(schedule["path"])
    if int(schedule["rows"]) != 6960:
        raise RuntimeError("preregistered seed schedule row count is not 6,960")
    require_hash(schedule_path, str(schedule["sha256"]), "seed schedule")
    manifest = prereg["dataset"]
    manifest_path = resolve_project_path(manifest["manifest"])
    require_hash(manifest_path, str(manifest["manifest_sha256"]), "manifest")
    manifest_payload = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if manifest_payload.manifest_hash != manifest["manifest_content_hash"]:
        raise RuntimeError("manifest content hash does not match preregistration")
    return prereg


def validate_addendum(
    path: str | Path,
    *,
    preregistration: dict[str, Any],
) -> dict[str, Any]:
    addendum = read_json(path)
    if addendum.get("schema_version") != "uav2d-preregistration-addendum-v1":
        raise RuntimeError("unsupported preregistration addendum schema")
    if addendum.get("status") != "frozen_before_final_results":
        raise RuntimeError("preregistration addendum is not frozen")
    if addendum.get("base_preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        raise RuntimeError("addendum targets a different preregistration")
    if addendum.get("base_preregistration_sha256") != sha256_file(
        DEFAULT_PREREGISTRATION
    ):
        raise RuntimeError("addendum base-preregistration hash mismatch")
    if canonical_id(addendum, "addendum_id") != addendum.get("addendum_id"):
        raise RuntimeError("addendum self-hash is invalid")
    return addendum


def validate_preflight_receipt(
    path: str | Path,
    *,
    preregistration: dict[str, Any],
    addendum: dict[str, Any],
) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("schema_version") != "uav2d-final-preflight-receipt-v1":
        raise RuntimeError("unsupported preflight receipt schema")
    if receipt.get("status") != "passed":
        raise RuntimeError("preflight receipt did not pass")
    if receipt.get("base_preregistration_id") != preregistration.get(
        "preregistration_id"
    ):
        raise RuntimeError("preflight targets a different preregistration")
    if receipt.get("addendum_id") != addendum.get("addendum_id"):
        raise RuntimeError("preflight targets a different addendum")
    if canonical_id(receipt, "preflight_receipt_id") != receipt.get(
        "preflight_receipt_id"
    ):
        raise RuntimeError("preflight receipt self-hash is invalid")
    frozen = addendum.get("frozen_tool_hashes", {})
    for label, definition in frozen.items():
        target = resolve_project_path(definition["path"])
        require_hash(target, str(definition["sha256"]), f"frozen tool {label}")
    return receipt


def validate_opening_receipt(
    path: str | Path,
    *,
    preregistration: dict[str, Any],
    addendum: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    receipt_path = Path(path)
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != "uav2d-hidden-test-opening-v1":
        raise RuntimeError("unsupported opening receipt schema")
    if receipt.get("status") != "authorized_open":
        raise RuntimeError("opening receipt is not authorized")
    if receipt.get("explicit_user_authorization") is not True:
        raise RuntimeError("opening receipt lacks explicit user authorization")
    expected = {
        "benchmark_id": preregistration["benchmark_id"],
        "preregistration_id": preregistration["preregistration_id"],
        "preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "addendum_id": addendum["addendum_id"],
        "addendum_sha256": sha256_file(DEFAULT_ADDENDUM),
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
    if DEFAULT_ARCHIVED_SEAL.is_file():
        require_hash(
            DEFAULT_ARCHIVED_SEAL,
            str(receipt["archived_seal_sha256"]),
            "archived SEALED marker",
        )
    else:
        raise RuntimeError("opening receipt exists without the archived seal marker")
    return receipt


def validate_current_implementation_hashes(
    preregistration: dict[str, Any],
    addendum: dict[str, Any],
) -> None:
    """Validate base frozen inputs, applying only the disclosed runner addendum."""

    runner_change = addendum["protocol_changes"]["benchmark_runner"]
    runner_path = str(runner_change["path"])
    for relative, expected in preregistration["frozen_input_hashes"].items():
        target = resolve_project_path(relative)
        if relative == runner_path:
            if expected != runner_change["previous_sha256"]:
                raise RuntimeError("addendum does not supersede the registered runner")
            expected = runner_change["authorized_entry_sha256"]
        require_hash(target, str(expected), f"frozen input {relative}")


__all__ = [
    "DEFAULT_ADDENDUM",
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
    "ROOT",
    "canonical_id",
    "project_relative",
    "read_json",
    "read_protocol",
    "require_hash",
    "resolve_project_path",
    "sha256_file",
    "validate_addendum",
    "validate_current_implementation_hashes",
    "validate_opening_receipt",
    "validate_preflight_receipt",
    "validate_preregistration",
    "write_json",
]
