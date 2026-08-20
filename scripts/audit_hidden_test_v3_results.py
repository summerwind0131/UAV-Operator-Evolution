"""Run the frozen 6,360-row Hidden Test-v3 statistical audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_audit import audit_results
from uav_operator_evolution.planning_benchmarks.final_evaluation_v3_common import (
    DEFAULT_FINAL_RESULTS,
    DEFAULT_PREREGISTRATION,
    DEFAULT_PROTOCOL,
    EXPECTED_RECORDS,
    resolve_project_path,
    validate_preregistration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration-id", required=True)
    args = parser.parse_args()
    prereg = validate_preregistration(
        preregistration_path=DEFAULT_PREREGISTRATION,
        protocol_path=DEFAULT_PROTOCOL,
        requested_id=args.preregistration_id,
    )
    report = audit_results(
        DEFAULT_FINAL_RESULTS,
        time_limit_seconds=float(prereg["budget"]["time_limit_seconds"]),
        schedule_path=resolve_project_path(prereg["seed_schedule"]["path"]),
        preflight=False,
        expected_records=EXPECTED_RECORDS,
    )
    print(json.dumps({
        "status": report["status"],
        "records": report["records"],
        "audit_content_id": report["audit_content_id"],
        "selected_paper_method": report["paper_method_decision"]["selected_method"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
