"""Run and freeze the 13-arm Validation surrogate for Hidden Test-v3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.config import load_config
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_audit import (
    audit_results,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_common import (
    DEFAULT_FINAL_RESULTS,
    DEFAULT_PREFLIGHT_RECEIPT,
    DEFAULT_PREFLIGHT_ROOT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    DEFAULT_SEAL_MARKER,
    ROOT,
    canonical_id,
    project_relative,
    sha256_file,
    validate_current_implementation_hashes,
    validate_preregistration,
    write_json,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_executor import (
    FINAL_V3_ARM_IDS,
    MatrixRunSettings,
    execute_matrix,
)


PARENT_CONFIG = ROOT / "configs/uav_benchmark_v1.yaml"
PREFLIGHT_RUN = DEFAULT_PREFLIGHT_ROOT / "surrogate_run"


def _tool_hashes() -> dict[str, dict[str, str]]:
    definitions = {
        "protocol_common": ROOT / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_v3_common.py",
        "executor": ROOT / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_v3_executor.py",
        "analyzer": ROOT / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_v3_audit.py",
        "benchmark_runner": ROOT / "src/uav_operator_evolution/planning_benchmarks/runner.py",
        "opening_entry": ROOT / "scripts/open_hidden_test_v3.py",
        "execution_entry": ROOT / "scripts/run_authorized_hidden_test_v3.py",
        "audit_entry": ROOT / "scripts/audit_hidden_test_v3_results.py",
        "preflight_entry": Path(__file__).resolve(),
    }
    return {
        name: {"path": project_relative(path), "sha256": sha256_file(path)}
        for name, path in definitions.items()
    }


def run_preflight(*, preregistration_id: str) -> dict[str, Any]:
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("Hidden Test-v3 must remain sealed during preflight")
    if DEFAULT_FINAL_RESULTS.exists():
        raise RuntimeError("V3 final-results directory exists during preflight")
    if DEFAULT_PREFLIGHT_ROOT.exists():
        raise RuntimeError("V3 preflight artifacts already exist; never overwrite them")
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=preregistration_id,
    )
    validate_current_implementation_hashes(prereg)
    settings = MatrixRunSettings(
        split="validation",
        maps_per_class=1,
        time_limit_seconds=0.10,
        max_objective_evaluations=50,
        stochastic_repetitions=2,
        expected_maps=6,
        expected_records=138,
        mode="preflight",
    )
    first = execute_matrix(
        load_config(PARENT_CONFIG),
        protocol_path=DEFAULT_PROTOCOL,
        destination=PREFLIGHT_RUN,
        settings=settings,
        stop_after_arms=4,
        protocol_identity={
            "preregistration_id": preregistration_id,
            "surrogate_dataset": "uav2d-v1/validation",
        },
    )
    if first.get("status") != "checkpointed" or first.get("completed_arms") != 4:
        raise RuntimeError("V3 preflight interruption did not stop after four arms")
    second = execute_matrix(
        load_config(PARENT_CONFIG),
        protocol_path=DEFAULT_PROTOCOL,
        destination=PREFLIGHT_RUN,
        settings=settings,
        protocol_identity={
            "preregistration_id": preregistration_id,
            "surrogate_dataset": "uav2d-v1/validation",
        },
    )
    if second.get("records") != 138 or len(second.get("resumed_arms", [])) != 4:
        raise RuntimeError("V3 preflight resume did not produce the exact matrix")
    report = audit_results(
        PREFLIGHT_RUN,
        time_limit_seconds=settings.time_limit_seconds,
        schedule_path=None,
        preflight=True,
        expected_records=138,
    )
    if report.get("status") != "passed" or report.get("arms") != 13:
        raise RuntimeError("V3 preflight audit failed")
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("V3 preflight altered the seal")
    tools = _tool_hashes()
    registered = prereg["frozen_input_hashes"]
    for definition in tools.values():
        if registered.get(definition["path"]) != definition["sha256"]:
            raise RuntimeError(f"preflight tool differs from preregistration: {definition['path']}")
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-preflight-receipt-v3",
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": prereg["benchmark_id"],
        "preregistration_id": preregistration_id,
        "preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "surrogate_dataset": "uav2d-v1",
        "split": "validation",
        "maps": 6,
        "arms": list(FINAL_V3_ARM_IDS),
        "expected_records": 138,
        "actual_records": report["records"],
        "unique_records": report["unique_records"],
        "planned_interruption": first,
        "resume": {
            "resumed_arms": second["resumed_arms"],
            "newly_completed_arms": second["newly_completed_arms"],
        },
        "execution_receipt_sha256": sha256_file(PREFLIGHT_RUN / "execution_receipt.json"),
        "audit_receipt_sha256": sha256_file(PREFLIGHT_RUN / "audit_receipt.json"),
        "frozen_tool_hashes": tools,
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ("numpy", "pydantic", "PyYAML")
        },
        "host": {"hostname": socket.gethostname(), "platform": platform.platform()},
        "hidden_test_map_json_read": False,
        "hidden_test_planner_executions": 0,
        "hidden_test_results_seen": False,
        "sealed_marker_preserved": True,
        "api_calls": 0,
    }
    receipt["preflight_receipt_id"] = canonical_id(receipt, "preflight_receipt_id")
    write_json(DEFAULT_PREFLIGHT_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    args = parser.parse_args()
    print(json.dumps(run_preflight(preregistration_id=args.preregistration_id), indent=2))


if __name__ == "__main__":
    main()

