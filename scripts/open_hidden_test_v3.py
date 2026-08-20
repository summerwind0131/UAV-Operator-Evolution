"""Create the single-use opening receipt for Hidden Test-v3."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_common import (
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
    sha256_file,
    validate_current_implementation_hashes,
    validate_preflight_receipt,
    validate_preregistration,
    write_json,
)


AUTHORIZATION_PHRASE = "AUTHORIZE UAV2D-HIDDEN-TEST-V3 FINAL ONCE"


def open_hidden_test(*, preregistration_id: str, authorization_phrase: str) -> dict[str, object]:
    if authorization_phrase != AUTHORIZATION_PHRASE:
        raise PermissionError("exact V3 explicit-authorization phrase required")
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("SEALED.json absent; refusing a second or ambiguous opening")
    if DEFAULT_OPENING_RECEIPT.exists() or DEFAULT_ARCHIVED_SEAL.exists():
        raise RuntimeError("V3 has already been opened")
    if DEFAULT_FINAL_RESULTS.exists():
        raise RuntimeError("V3 final-results directory exists before opening")
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=preregistration_id,
    )
    preflight = validate_preflight_receipt(
        DEFAULT_PREFLIGHT_RECEIPT,
        preregistration=prereg,
    )
    validate_current_implementation_hashes(prereg)
    marker, seal = read_json(DEFAULT_SEAL_MARKER), read_json(DEFAULT_SEAL_RECEIPT)
    if marker.get("status") != "sealed_unrun" or seal.get("status") != "sealed_unrun":
        raise RuntimeError("V3 seal metadata is not sealed-unrun")
    require_hash(DEFAULT_SEAL_RECEIPT, marker["seal_receipt_sha256"], "seal receipt")
    require_hash(DEFAULT_PREREGISTRATION, marker["preregistration_sha256"], "preregistration")
    if canonical_id(seal, "seal_id") != seal.get("seal_id"):
        raise RuntimeError("V3 seal receipt self-hash invalid")
    data_root = DEFAULT_SEAL_MARKER.parent
    for relative, expected in seal["map_file_sha256"].items():
        require_hash(data_root / relative, expected, f"hidden map {relative}")
    receipt: dict[str, object] = {
        "schema_version": "uav2d-hidden-test-opening-v3",
        "status": "authorized_open",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": prereg["benchmark_id"],
        "preregistration_id": preregistration_id,
        "preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "preflight_receipt_id": preflight["preflight_receipt_id"],
        "preflight_receipt_sha256": sha256_file(DEFAULT_PREFLIGHT_RECEIPT),
        "seal_id": seal["seal_id"],
        "archived_seal_sha256": sha256_file(DEFAULT_SEAL_MARKER),
        "manifest_sha256": prereg["dataset"]["manifest_sha256"],
        "seed_schedule_sha256": prereg["seed_schedule"]["sha256"],
        "explicit_user_authorization": True,
        "authorization_statement": AUTHORIZATION_PHRASE,
        "single_use": True,
        "api_calls": 0,
    }
    receipt["opening_id"] = canonical_id(receipt, "opening_id")
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
    print(f"opened {receipt['benchmark_id']} once with {receipt['opening_id']}")


if __name__ == "__main__":
    main()

