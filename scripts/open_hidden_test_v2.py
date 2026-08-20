"""Create an opening receipt and archive the hidden-test seal after authorization.

This script is intentionally not called by preflight.  It is reserved for a
future turn in which the user explicitly authorizes the final evaluation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.planning_benchmarks.final_evaluation_common import (
    DEFAULT_ADDENDUM,
    DEFAULT_ARCHIVED_SEAL,
    DEFAULT_FINAL_RESULTS,
    DEFAULT_OPENING_RECEIPT,
    DEFAULT_PREFLIGHT_RECEIPT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    DEFAULT_SEAL_MARKER,
    DEFAULT_SEAL_RECEIPT,
    canonical_id,
    read_json,
    require_hash,
    resolve_project_path,
    sha256_file,
    validate_addendum,
    validate_current_implementation_hashes,
    validate_preflight_receipt,
    validate_preregistration,
    write_json,
)


AUTHORIZATION_PHRASE = "AUTHORIZE UAV2D-HIDDEN-TEST-V2 FINAL"


def open_hidden_test(
    *,
    preregistration_id: str,
    authorization_phrase: str,
    protocol_path: Path = DEFAULT_PROTOCOL,
    preregistration_path: Path = DEFAULT_PREREGISTRATION,
    addendum_path: Path = DEFAULT_ADDENDUM,
    preflight_receipt_path: Path = DEFAULT_PREFLIGHT_RECEIPT,
) -> dict[str, object]:
    if authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("the exact explicit-authorization phrase is required")
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("SEALED.json is absent; refusing a second or ambiguous opening")
    if DEFAULT_OPENING_RECEIPT.exists() or DEFAULT_ARCHIVED_SEAL.exists():
        raise RuntimeError("an opening receipt or archived seal already exists")
    if DEFAULT_FINAL_RESULTS.exists():
        raise RuntimeError("final-results directory exists before opening")
    prereg = validate_preregistration(
        preregistration_path=preregistration_path,
        protocol_path=protocol_path,
        requested_id=preregistration_id,
    )
    addendum = validate_addendum(addendum_path, preregistration=prereg)
    preflight = validate_preflight_receipt(
        preflight_receipt_path,
        preregistration=prereg,
        addendum=addendum,
    )
    validate_current_implementation_hashes(prereg, addendum)

    marker = read_json(DEFAULT_SEAL_MARKER)
    seal_receipt = read_json(DEFAULT_SEAL_RECEIPT)
    if marker.get("status") != "sealed_unrun" or seal_receipt.get("status") != "sealed_unrun":
        raise RuntimeError("seal metadata is not in sealed-unrun state")
    require_hash(
        DEFAULT_SEAL_RECEIPT,
        str(marker["seal_receipt_sha256"]),
        "seal receipt",
    )
    require_hash(
        preregistration_path,
        str(marker["preregistration_sha256"]),
        "preregistration from seal marker",
    )
    if canonical_id(seal_receipt, "seal_id") != seal_receipt.get("seal_id"):
        raise RuntimeError("seal receipt self-hash is invalid")
    if marker.get("preregistration_id") != preregistration_id:
        raise RuntimeError("seal marker preregistration ID mismatch")
    # This is the first operation allowed to read hidden-map bytes.  It occurs
    # only after the explicit authorization phrase and validates their seal.
    data_root = DEFAULT_SEAL_MARKER.parent
    for relative, expected in seal_receipt["map_file_sha256"].items():
        require_hash(data_root / relative, expected, f"hidden map {relative}")

    receipt: dict[str, object] = {
        "schema_version": "uav2d-hidden-test-opening-v1",
        "status": "authorized_open",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": prereg["benchmark_id"],
        "preregistration_id": preregistration_id,
        "preregistration_sha256": sha256_file(preregistration_path),
        "addendum_id": addendum["addendum_id"],
        "addendum_sha256": sha256_file(addendum_path),
        "preflight_receipt_id": preflight["preflight_receipt_id"],
        "preflight_receipt_sha256": sha256_file(preflight_receipt_path),
        "seal_id": seal_receipt["seal_id"],
        "archived_seal_sha256": sha256_file(DEFAULT_SEAL_MARKER),
        "manifest_sha256": prereg["dataset"]["manifest_sha256"],
        "seed_schedule_sha256": prereg["seed_schedule"]["sha256"],
        "explicit_user_authorization": True,
        "authorization_statement": AUTHORIZATION_PHRASE,
        "api_calls": 0,
    }
    receipt["opening_id"] = canonical_id(receipt, "opening_id")
    # Write the receipt first.  If archiving the marker fails, SEALED.json still
    # blocks execution; an incomplete opening can never weaken the guard.
    write_json(DEFAULT_OPENING_RECEIPT, receipt)
    DEFAULT_SEAL_MARKER.replace(DEFAULT_ARCHIVED_SEAL)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--authorization-phrase", required=True)
    args = parser.parse_args()
    receipt = open_hidden_test(
        preregistration_id=args.preregistration_id,
        authorization_phrase=args.authorization_phrase,
    )
    print(f"opened {receipt['benchmark_id']} with opening {receipt['opening_id']}")


if __name__ == "__main__":
    main()
