"""Dedicated, receipt-gated entry for the sealed UAV2D Hidden Test-v2."""

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
from uav_operator_evolution.planning_benchmarks.final_evaluation_audit import (
    audit_results,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_common import (
    DEFAULT_ADDENDUM,
    DEFAULT_ARCHIVED_SEAL,
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_FINAL_RESULTS,
    DEFAULT_OPENING_RECEIPT,
    DEFAULT_PREFLIGHT_RECEIPT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    DEFAULT_SEAL_MARKER,
    DEFAULT_SEAL_RECEIPT,
    read_json,
    require_hash,
    resolve_project_path,
    sha256_file,
    validate_addendum,
    validate_current_implementation_hashes,
    validate_opening_receipt,
    validate_preflight_receipt,
    validate_preregistration,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_executor import (
    MatrixRunSettings,
    execute_matrix,
)
from uav_operator_evolution.planning_benchmarks.runner import (
    _issue_final_evaluation_authorization,
)


def _validate_opened_dataset(preregistration: dict[str, object]) -> None:
    seal = read_json(DEFAULT_SEAL_RECEIPT)
    if not DEFAULT_ARCHIVED_SEAL.is_file():
        raise RuntimeError("the original seal was not archived by the opening step")
    manifest = preregistration["dataset"]
    manifest_path = resolve_project_path(manifest["manifest"])
    require_hash(manifest_path, manifest["manifest_sha256"], "opened manifest")
    payload = DatasetManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    if payload.manifest_hash != manifest["manifest_content_hash"]:
        raise RuntimeError("opened manifest content hash mismatch")
    data_root = manifest_path.parent
    for relative, expected in seal["map_file_sha256"].items():
        require_hash(data_root / relative, expected, f"opened hidden map {relative}")


def run_authorized_final(
    *,
    preregistration_id: str,
    opening_receipt_path: Path = DEFAULT_OPENING_RECEIPT,
) -> dict[str, object]:
    # The seal check is deliberately first: even a perfectly valid-looking
    # opening receipt cannot authorize execution while SEALED.json exists.
    if DEFAULT_SEAL_MARKER.exists():
        raise PermissionError("SEALED.json exists; final evaluation will not execute")
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=preregistration_id,
    )
    addendum = validate_addendum(DEFAULT_ADDENDUM, preregistration=prereg)
    preflight = validate_preflight_receipt(
        DEFAULT_PREFLIGHT_RECEIPT,
        preregistration=prereg,
        addendum=addendum,
    )
    opening = validate_opening_receipt(
        opening_receipt_path,
        preregistration=prereg,
        addendum=addendum,
        preflight=preflight,
    )
    validate_current_implementation_hashes(prereg, addendum)
    require_hash(
        DEFAULT_EXECUTION_CONFIG,
        addendum["execution_contract"]["config_sha256"],
        "final execution config",
    )
    _validate_opened_dataset(prereg)
    config = load_config(DEFAULT_EXECUTION_CONFIG)
    if config.output.data_dir.resolve() != DEFAULT_SEAL_MARKER.parent.resolve():
        raise RuntimeError("execution config points at the wrong data root")
    if config.output.results_dir.resolve() != DEFAULT_FINAL_RESULTS.resolve():
        raise RuntimeError("execution config points at the wrong results root")
    budget = prereg["budget"]
    if (
        config.planning_benchmark.time_limit_seconds != budget["time_limit_seconds"]
        or config.planning_benchmark.max_objective_evaluations
        != budget["max_objective_evaluations"]
        or config.planning_benchmark.stochastic_repetitions
        != budget["stochastic_repetitions"]
    ):
        raise RuntimeError("execution config budget differs from preregistration")
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
            expected_records=6960,
            mode="final",
        ),
        authorization=authorization,
        protocol_identity={
            "preregistration_id": preregistration_id,
            "addendum_id": addendum["addendum_id"],
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
        expected_records=6960,
    )
    return {
        "status": "complete",
        "opening_id": opening["opening_id"],
        "execution_receipt_id": execution["execution_receipt_id"],
        "audit_content_id": report["audit_content_id"],
        "records": report["records"],
        "results_dir": str(DEFAULT_FINAL_RESULTS.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument(
        "--opening-receipt", type=Path, default=DEFAULT_OPENING_RECEIPT
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_authorized_final(
                preregistration_id=args.preregistration_id,
                opening_receipt_path=args.opening_receipt,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
