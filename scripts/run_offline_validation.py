"""Run the API-free traditional planner matrix on the fixed Validation split."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from uav_operator_evolution.config import load_config  # noqa: E402
from uav_operator_evolution.planning_benchmarks.runner import (  # noqa: E402
    run_planner_benchmark,
)


TRADITIONAL_PLANNERS = [
    "dijkstra",
    "astar",
    "theta_star",
    "rrt",
    "rrt_star",
    "prm",
    "ga",
    "pso",
    "de",
    "aco_acor",
]


def main() -> int:
    log_path = (
        REPOSITORY_ROOT
        / "artifacts"
        / "planning_benchmarks"
        / "offline-traditional-validation-v1.log"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    try:
        config = load_config(
            REPOSITORY_ROOT / "configs" / "uav_benchmark_v1.yaml"
        )
        report = run_planner_benchmark(
            config,
            split="validation",
            planners=TRADITIONAL_PLANNERS,
            time_limit_seconds=1.0,
            max_objective_evaluations=2_000,
            repetitions=5,
            run_id="offline-traditional-validation-v1",
        )
    except Exception:
        logging.exception("offline traditional Validation failed")
        raise
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
