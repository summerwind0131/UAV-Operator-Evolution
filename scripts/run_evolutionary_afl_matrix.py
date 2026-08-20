"""Run pre-registered Evolutionary AFL-UAV ablation/sensitivity matrices offline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from uav_operator_evolution.config import load_config
from uav_operator_evolution.planning_benchmarks.evolutionary_afl_experiments import (
    EvolutionaryAFLExperimentPlanner,
)
from uav_operator_evolution.planning_benchmarks.runner import run_planner_benchmark


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "configs" / "evolutionary_afl_uav_experiments_v1.yaml"


def _resolve(path: str) -> Path:
    resolved = (ROOT / path).resolve()
    if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
        raise ValueError(f"matrix path escapes project root: {path}")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_matrix(
    matrix_path: Path,
    *,
    sections: list[str] | None = None,
    maps_per_class: int | None = None,
    repetitions: int | None = None,
    run_id_suffix: str = "",
) -> dict[str, Any]:
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("split") not in {"train", "validation"}:
        raise ValueError("Evolutionary experiment matrices are Train/Validation-only")
    frozen_method_path = _resolve(matrix["frozen_method_artifact"])
    frozen_method = json.loads(frozen_method_path.read_text(encoding="utf-8"))
    if frozen_method.get("method_id") != "evolutionary-afl-uav-v1":
        raise RuntimeError("matrix requires the frozen Evolutionary AFL-UAV v1 method")
    frozen_source = frozen_method_path.parent / frozen_method["source"]["frozen_filename"]
    expected_core_hash = matrix["expected_frozen_core_sha256"]
    if frozen_method["source"]["sha256"] != expected_core_hash:
        raise RuntimeError("frozen method artifact records an unexpected core hash")
    if _sha256(frozen_source) != expected_core_hash:
        raise RuntimeError("frozen Evolutionary AFL-UAV v1 source copy was modified")
    current_core = ROOT / "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py"
    if _sha256(current_core) != expected_core_hash:
        raise RuntimeError("current Evolutionary AFL-UAV v1 core changed after freeze")

    config = load_config(_resolve(matrix["benchmark_config"]))
    seed_artifact = _resolve(matrix["seed_artifact"])
    selected_sections = sections or list(matrix["runs"])
    unknown = sorted(set(selected_sections) - set(matrix["runs"]))
    if unknown:
        raise ValueError("unknown matrix sections: " + ", ".join(unknown))

    reports: dict[str, Any] = {}
    for section in selected_sections:
        definition = matrix["runs"][section]
        overrides: dict[str, object] = {}
        planner_keys: list[str] = []
        for arm_id, specification in definition["arms"].items():
            planner_key = f"evolutionary_afl_uav:{arm_id}"
            overrides[planner_key] = EvolutionaryAFLExperimentPlanner(
                seed_artifact,
                arm_id=arm_id,
                variant=specification["variant"],
                population_size=int(specification["population_size"]),
                archive_size=int(specification["archive_size"]),
                max_generations=int(specification["max_generations"]),
                max_waypoints=64,
                base_iteration_limit=64,
                crossover_probability=0.40,
                extra_mutation_probability=0.30,
            )
            planner_keys.append(planner_key)
        report = run_planner_benchmark(
            config,
            split=matrix["split"],
            planners=planner_keys,
            maps_per_class=maps_per_class,
            time_limit_seconds=float(definition["time_limit_seconds"]),
            max_objective_evaluations=int(matrix["max_objective_evaluations"]),
            repetitions=(
                int(repetitions)
                if repetitions is not None
                else int(matrix["repetitions"])
            ),
            planner_overrides=overrides,
            run_id=definition["run_id"] + run_id_suffix,
        )
        reports[section] = report
    return {
        "matrix_id": matrix["matrix_id"],
        "matrix_sha256": _sha256(matrix_path),
        "frozen_method_artifact_id": frozen_method["artifact_id"],
        "api_calls": 0,
        "split": matrix["split"],
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--section", action="append")
    parser.add_argument("--maps-per-class", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--run-id-suffix", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            run_matrix(
                args.matrix.resolve(),
                sections=args.section,
                maps_per_class=args.maps_per_class,
                repetitions=args.repetitions,
                run_id_suffix=args.run_id_suffix,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
