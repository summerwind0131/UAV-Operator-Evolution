"""Receipt-gated single final execution for Hidden Test-v3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment.generator import DatasetManifest
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_audit import audit_results
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_common import (
    DEFAULT_ARCHIVED_SEAL,
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_FINAL_RESULTS,
    DEFAULT_OPENING_RECEIPT,
    DEFAULT_PREFLIGHT_RECEIPT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    DEFAULT_SEAL_MARKER,
    DEFAULT_SEAL_RECEIPT,
    EXPECTED_RECORDS,
    read_json,
    require_hash,
    resolve_project_path,
    sha256_file,
    validate_current_implementation_hashes,
    validate_opening_receipt,
    validate_preflight_receipt,
    validate_preregistration,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_executor import (
    MatrixRunSettings,
    execute_matrix,
)
from uav_operator_evolution.planning_benchmarks.runner import _issue_final_evaluation_authorization


def _validate_opened_dataset(preregistration: dict[str, object]) -> None:
    seal = read_json(DEFAULT_SEAL_RECEIPT)
    if not DEFAULT_ARCHIVED_SEAL.is_file():
        raise RuntimeError("original V3 seal was not archived")
    manifest = preregistration["dataset"]
    manifest_path = resolve_project_path(manifest["manifest"])
    require_hash(manifest_path, manifest["manifest_sha256"], "opened manifest")
    payload = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if payload.manifest_hash != manifest["manifest_content_hash"]:
        raise RuntimeError("opened V3 manifest content hash mismatch")
    for relative, expected in seal["map_file_sha256"].items():
        require_hash(manifest_path.parent / relative, expected, f"opened hidden map {relative}")


def run_authorized_final(
    *,
    preregistration_id: str,
    opening_receipt_path: Path = DEFAULT_OPENING_RECEIPT,
) -> dict[str, object]:
    if DEFAULT_SEAL_MARKER.exists():
        raise PermissionError("SEALED.json exists; V3 final evaluation cannot execute")
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=preregistration_id,
    )
    preflight = validate_preflight_receipt(DEFAULT_PREFLIGHT_RECEIPT, preregistration=prereg)
    opening = validate_opening_receipt(
        opening_receipt_path,
        preregistration=prereg,
        preflight=preflight,
    )
    validate_current_implementation_hashes(prereg)
    require_hash(
        DEFAULT_EXECUTION_CONFIG,
        prereg["frozen_input_hashes"]["configs/uav_hidden_test_v3_execution_v1.yaml"],
        "V3 execution config",
    )
    _validate_opened_dataset(prereg)
    config = load_config(DEFAULT_EXECUTION_CONFIG)
    if config.output.data_dir.resolve() != DEFAULT_SEAL_MARKER.parent.resolve():
        raise RuntimeError("execution config points at wrong V3 data root")
    if config.output.results_dir.resolve() != DEFAULT_FINAL_RESULTS.resolve():
        raise RuntimeError("execution config points at wrong V3 results root")
    budget = prereg["budget"]
    authorization = _issue_final_evaluation_authorization(
        benchmark_id=prereg["benchmark_id"],
        opening_id=opening["opening_id"],
        preregistration_id=preregistration_id,
    )
    execution = execute_matrix(
        config,
        protocol_path=DEFAULT_PROTOCOL,
        destination=DEFAULT_FINAL_RESULTS,
        settings=MatrixRunSettings(
            split="test",
            maps_per_class=None,
            time_limit_seconds=float(budget["time_limit_seconds"]),
            max_objective_evaluations=int(budget["max_objective_evaluations"]),
            stochastic_repetitions=int(budget["stochastic_repetitions"]),
            expected_maps=int(prereg["dataset"]["maps"]),
            expected_records=EXPECTED_RECORDS,
            mode="final",
        ),
        authorization=authorization,
        protocol_identity={
            "preregistration_id": preregistration_id,
            "preflight_receipt_id": preflight["preflight_receipt_id"],
            "opening_id": opening["opening_id"],
            "seed_schedule_sha256": prereg["seed_schedule"]["sha256"],
        },
    )
    report = audit_results(
        DEFAULT_FINAL_RESULTS,
        time_limit_seconds=float(budget["time_limit_seconds"]),
        schedule_path=resolve_project_path(prereg["seed_schedule"]["path"]),
        preflight=False,
        expected_records=EXPECTED_RECORDS,
    )
    return {
        "status": "complete",
        "opening_id": opening["opening_id"],
        "execution_receipt_id": execution["execution_receipt_id"],
        "audit_content_id": report["audit_content_id"],
        "records": report["records"],
        "selected_paper_method": report["paper_method_decision"]["selected_method"],
        "results_dir": str(DEFAULT_FINAL_RESULTS.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--opening-receipt", type=Path, default=DEFAULT_OPENING_RECEIPT)
    args = parser.parse_args()
    print(json.dumps(run_authorized_final(
        preregistration_id=args.preregistration_id,
        opening_receipt_path=args.opening_receipt,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
