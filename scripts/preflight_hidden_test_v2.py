"""Run and freeze the 14-arm Validation surrogate for Hidden Test-v2."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import socket
import sys
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.config import load_config
from uav_operator_evolution.planning_benchmarks.final_evaluation_audit import (
    FINAL_BOOTSTRAP_REPLICATES,
    FINAL_BOOTSTRAP_SEED,
    audit_results,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_common import (
    DEFAULT_ADDENDUM,
    DEFAULT_EXECUTION_CONFIG,
    DEFAULT_FINAL_RESULTS,
    DEFAULT_PREFLIGHT_RECEIPT,
    DEFAULT_PREFLIGHT_ROOT,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    DEFAULT_SEAL_MARKER,
    ROOT,
    canonical_id,
    project_relative,
    read_json,
    sha256_file,
    validate_preregistration,
    write_json,
)
from uav_operator_evolution.planning_benchmarks.final_evaluation_executor import (
    FINAL_ARM_IDS,
    MatrixRunSettings,
    execute_matrix,
)


PARENT_CONFIG = ROOT / "configs/uav_benchmark_v1.yaml"
PREFLIGHT_RUN = DEFAULT_PREFLIGHT_ROOT / "surrogate_run"
DEFAULT_PYTEST_JUNIT = DEFAULT_PREFLIGHT_ROOT / "pytest.xml"
BASE_RUNNER_SHA256 = "7f28c8142dae4d9d4a37054c24de606b2daf82918f10d0b2290730add5970f6f"


def _pytest_summary(path: Path) -> dict[str, int] | None:
    if not path.is_file():
        return None
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    return {
        field: sum(int(suite.attrib.get(field, 0)) for suite in suites)
        for field in ("tests", "failures", "errors", "skipped")
    }


def _frozen_tool_hashes() -> dict[str, dict[str, str]]:
    definitions = {
        "benchmark_runner": ROOT
        / "src/uav_operator_evolution/planning_benchmarks/runner.py",
        "protocol_common": ROOT
        / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_common.py",
        "executor": ROOT
        / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_executor.py",
        "analyzer": ROOT
        / "src/uav_operator_evolution/planning_benchmarks/final_evaluation_audit.py",
        "opening_entry": ROOT / "scripts/open_hidden_test_v2.py",
        "execution_entry": ROOT / "scripts/run_authorized_hidden_test_v2.py",
        "audit_entry": ROOT / "scripts/audit_hidden_test_v2_results.py",
        "preflight_entry": Path(__file__).resolve(),
    }
    return {
        name: {"path": project_relative(path), "sha256": sha256_file(path)}
        for name, path in definitions.items()
    }


def run_and_freeze_preflight(
    *,
    preregistration_id: str,
    pytest_junit: Path = DEFAULT_PYTEST_JUNIT,
) -> dict[str, Any]:
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("Hidden Test-v2 must remain sealed throughout preflight")
    if DEFAULT_FINAL_RESULTS.exists():
        raise RuntimeError("final-results directory exists during preflight")
    if DEFAULT_ADDENDUM.exists() or DEFAULT_PREFLIGHT_RECEIPT.exists():
        raise RuntimeError(
            "frozen preflight artifacts already exist; never overwrite them—create a disclosed addendum version"
        )
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=preregistration_id,
    )
    settings = MatrixRunSettings(
        split="validation",
        maps_per_class=1,
        time_limit_seconds=0.10,
        max_objective_evaluations=50,
        stochastic_repetitions=2,
        expected_maps=6,
        expected_records=150,
        mode="preflight",
    )
    first = execute_matrix(
        load_config(PARENT_CONFIG),
        protocol_path=DEFAULT_PROTOCOL,
        destination=PREFLIGHT_RUN,
        settings=settings,
        stop_after_arms=5,
        protocol_identity={
            "base_preregistration_id": preregistration_id,
            "surrogate_dataset": "uav2d-v1/validation",
        },
    )
    if first.get("status") != "checkpointed" or first.get("completed_arms") != 5:
        raise RuntimeError("preflight did not create the planned five-arm interruption")
    second = execute_matrix(
        load_config(PARENT_CONFIG),
        protocol_path=DEFAULT_PROTOCOL,
        destination=PREFLIGHT_RUN,
        settings=settings,
        protocol_identity={
            "base_preregistration_id": preregistration_id,
            "surrogate_dataset": "uav2d-v1/validation",
        },
    )
    if second.get("records") != 150 or len(second.get("resumed_arms", [])) != 5:
        raise RuntimeError("preflight resume did not produce the exact 150-row matrix")
    report = audit_results(
        PREFLIGHT_RUN,
        time_limit_seconds=settings.time_limit_seconds,
        schedule_path=None,
        preflight=True,
        expected_records=150,
    )
    if report.get("status") != "passed" or report.get("arms") != 14:
        raise RuntimeError("preflight statistical report did not pass all 14 arms")
    if not DEFAULT_SEAL_MARKER.is_file():
        raise RuntimeError("preflight unexpectedly altered Hidden Test-v2 seal")

    tools = _frozen_tool_hashes()
    dependencies = {
        package: importlib.metadata.version(package)
        for package in ("numpy", "pydantic", "PyYAML")
    }
    junit_summary = _pytest_summary(pytest_junit)
    addendum: dict[str, Any] = {
        "schema_version": "uav2d-preregistration-addendum-v1",
        "status": "frozen_before_final_results",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": prereg["benchmark_id"],
        "base_preregistration_id": preregistration_id,
        "base_preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "reason": (
            "Disclose the dedicated authorization capability, explicit objective/execution "
            "configuration, resumable arm checkpoints, and frozen statistical auditor. "
            "No hidden-test result existed or was read when this addendum was made."
        ),
        "protocol_changes": {
            "benchmark_runner": {
                "path": "src/uav_operator_evolution/planning_benchmarks/runner.py",
                "previous_sha256": BASE_RUNNER_SHA256,
                "authorized_entry_sha256": tools["benchmark_runner"]["sha256"],
                "change": (
                    "Retain the default Test restriction and SEALED guard; accept only a "
                    "private capability issued by the receipt-gated final entry."
                ),
            },
            "method_or_parameters_changed": False,
            "dataset_changed": False,
            "hypotheses_or_statistics_changed": False,
        },
        "execution_contract": {
            "config_path": project_relative(DEFAULT_EXECUTION_CONFIG),
            "config_sha256": sha256_file(DEFAULT_EXECUTION_CONFIG),
            "objective_weights": {
                "length": 1.0,
                "collision": 1000.0,
                "smoothness": 5.0,
                "risk": 10.0,
                "waypoint": 0.5,
            },
            "time_limit_seconds": 1.0,
            "max_objective_evaluations": 2000,
            "stochastic_repetitions": 5,
            "expected_records": 6960,
            "seed_artifact": (
                "artifacts/planning_benchmarks/afl_uav_artifacts/"
                "deepseek-v4pro-frozen-strict-v2/artifact.json"
            ),
            "arm_checkpointing": "complete-arm atomic receipt; partial arms fail closed",
        },
        "analysis_contract": {
            "analyzer_sha256": tools["analyzer"]["sha256"],
            "records": 6960,
            "timeouts": "status timeout OR elapsed >= 1.0 seconds; always infeasible",
            "ranking": "trusted feasible rate, then median trusted feasible cost",
            "paired": "per-map lexicographic feasibility then median shared-seed cost",
            "rooms_maze": "same frozen metrics on rooms_maze subset",
            "diversity": "unique feasible path hashes and within-map cost IQR",
            "bootstrap_replicates": FINAL_BOOTSTRAP_REPLICATES,
            "bootstrap_seed": FINAL_BOOTSTRAP_SEED,
            "holm_family": ["H3", "H4"],
            "post_result_changes_allowed": False,
        },
        "frozen_tool_hashes": tools,
        "preflight_evidence": {
            "dataset": "uav2d-v1 validation; one map per each of six classes",
            "hidden_test_map_json_read": False,
            "arms": list(FINAL_ARM_IDS),
            "records": 150,
            "planned_interruption_after_arms": 5,
            "resumed_completed_arms": second["resumed_arms"],
            "execution_receipt_sha256": sha256_file(
                PREFLIGHT_RUN / "execution_receipt.json"
            ),
            "audit_receipt_sha256": sha256_file(
                PREFLIGHT_RUN / "audit_receipt.json"
            ),
            "pytest_junit_sha256": (
                sha256_file(pytest_junit) if pytest_junit.is_file() else None
            ),
            "pytest_summary": junit_summary,
        },
        "dependencies": dependencies,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
        },
        "commitments": {
            "base_receipt_overwritten": False,
            "hidden_test_results_seen": False,
            "hidden_test_planner_executions": 0,
            "future_protocol_changes_require_new_addendum": True,
            "api_calls": 0,
        },
    }
    addendum["addendum_id"] = canonical_id(addendum, "addendum_id")
    write_json(DEFAULT_ADDENDUM, addendum)
    receipt: dict[str, Any] = {
        "schema_version": "uav2d-final-preflight-receipt-v1",
        "status": "passed",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "benchmark_id": prereg["benchmark_id"],
        "base_preregistration_id": preregistration_id,
        "base_preregistration_sha256": sha256_file(DEFAULT_PREREGISTRATION),
        "addendum_id": addendum["addendum_id"],
        "addendum_sha256": sha256_file(DEFAULT_ADDENDUM),
        "surrogate_dataset": "uav2d-v1",
        "split": "validation",
        "maps": 6,
        "arms": list(FINAL_ARM_IDS),
        "expected_records": 150,
        "actual_records": report["records"],
        "unique_records": report["unique_records"],
        "planned_interruption": first,
        "resume": {
            "resumed_arms": second["resumed_arms"],
            "newly_completed_arms": second["newly_completed_arms"],
        },
        "execution_receipt_sha256": sha256_file(
            PREFLIGHT_RUN / "execution_receipt.json"
        ),
        "audit_receipt_sha256": sha256_file(PREFLIGHT_RUN / "audit_receipt.json"),
        "frozen_tool_hashes": tools,
        "dependencies": dependencies,
        "pytest_junit_sha256": (
            sha256_file(pytest_junit) if pytest_junit.is_file() else None
        ),
        "pytest_summary": junit_summary,
        "hidden_test_map_json_read": False,
        "hidden_test_planner_executions": 0,
        "hidden_test_results_seen": False,
        "sealed_marker_preserved": True,
        "api_calls": 0,
    }
    receipt["preflight_receipt_id"] = canonical_id(
        receipt, "preflight_receipt_id"
    )
    write_json(DEFAULT_PREFLIGHT_RECEIPT, receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    parser.add_argument("--pytest-junit", type=Path, default=DEFAULT_PYTEST_JUNIT)
    args = parser.parse_args()
    print(
        json.dumps(
            run_and_freeze_preflight(
                preregistration_id=args.preregistration_id,
                pytest_junit=args.pytest_junit,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
