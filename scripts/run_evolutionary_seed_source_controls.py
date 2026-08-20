"""Run the Train/Validation-only seed-source controls around frozen v1."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from uav_operator_evolution.config import load_config
from uav_operator_evolution.planning_benchmarks.evolutionary_afl import (
    EvolutionaryAFLUAVPlanner,
)
from uav_operator_evolution.planning_benchmarks.evolutionary_seed_controls import (
    FROZEN_V1_CORE_SHA256,
    SeedSourceEvolutionaryControlPlanner,
)
from uav_operator_evolution.planning_benchmarks.runner import run_planner_benchmark
from uav_operator_evolution.reproducibility import stable_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/evolutionary_seed_source_controls_v1.yaml"
SHARED_METHODS = (
    "plan",
    "_generation_cap",
    "_initialize_population",
    "_apply_operator",
    "_mutate_insert",
    "_mutate_delete",
    "_mutate_move",
    "_mutate_swap",
    "_crossover",
    "_update_archive",
    "_select_survivors",
    "_select_parent",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
        raise ValueError(f"control experiment path escapes project root: {path}")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dump_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_frozen_inputs(specification: dict[str, Any]) -> dict[str, str]:
    if specification.get("split") not in {"train", "validation"}:
        raise ValueError("seed-source controls are strictly Train/Validation-only")
    if specification["frozen_v1_core_sha256"] != FROZEN_V1_CORE_SHA256:
        raise RuntimeError("control config does not reference the frozen v1 core hash")

    current_core = ROOT / "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py"
    if _sha256(current_core) != FROZEN_V1_CORE_SHA256:
        raise RuntimeError("current Evolutionary AFL-UAV v1 core was modified")
    method_path = _resolve(specification["frozen_v1_method_artifact"])
    method = json.loads(method_path.read_text(encoding="utf-8"))
    if method.get("method_id") != "evolutionary-afl-uav-v1":
        raise RuntimeError("unexpected frozen method artifact")
    frozen_copy = method_path.parent / method["source"]["frozen_filename"]
    if _sha256(frozen_copy) != FROZEN_V1_CORE_SHA256:
        raise RuntimeError("frozen v1 source copy hash mismatch")

    seed_dir = _resolve(specification["afl_seed_artifact"])
    seed_artifact = json.loads((seed_dir / "artifact.json").read_text(encoding="utf-8"))
    if seed_artifact.get("artifact_id") != specification["afl_seed_artifact_id"]:
        raise RuntimeError("AFL seed artifact ID mismatch")

    inherited = {
        method_name: (
            getattr(SeedSourceEvolutionaryControlPlanner, method_name)
            is getattr(EvolutionaryAFLUAVPlanner, method_name)
        )
        for method_name in SHARED_METHODS
    }
    if not all(inherited.values()):
        changed = sorted(name for name, same in inherited.items() if not same)
        raise RuntimeError("control changed frozen evolution methods: " + ", ".join(changed))
    return {
        "current_v1_core_sha256": _sha256(current_core),
        "frozen_v1_copy_sha256": _sha256(frozen_copy),
        "frozen_method_artifact_sha256": _sha256(method_path),
        "afl_seed_artifact_sha256": _sha256(seed_dir / "artifact.json"),
        "shared_method_identity_hash": stable_hash(inherited),
    }


def run_controls(
    experiment_config: Path,
    *,
    maps_per_class: int | None = None,
    repetitions: int | None = None,
    run_id_suffix: str = "",
) -> dict[str, Any]:
    specification = yaml.safe_load(experiment_config.read_text(encoding="utf-8"))
    frozen_hashes = _validate_frozen_inputs(specification)
    benchmark_config = load_config(_resolve(specification["benchmark_config"]))
    shared = specification["shared_evolution_layer"]
    budget = specification["budget"]

    expected_parameters = {
        "population_size": 32,
        "archive_size": 8,
        "max_generations": 20,
        "max_waypoints": 64,
        "base_iteration_limit": 64,
        "crossover_probability": 0.40,
        "extra_mutation_probability": 0.30,
    }
    for name, expected in expected_parameters.items():
        if shared[name] != expected:
            raise RuntimeError(f"shared evolution parameter {name} differs from frozen v1")

    overrides: dict[str, object] = {}
    selected_planners: list[str] = []
    afl_arm_id: str | None = None
    for arm_id, arm in specification["arms"].items():
        source = arm["seed_source"]
        if source == "afl_uav_deepseek_v4_pro":
            if afl_arm_id is not None:
                raise ValueError("exactly one AFL-seeded reference arm is allowed")
            afl_arm_id = arm_id
            selected_planners.append(f"evolutionary_afl_uav:{arm_id}")
            continue
        key = f"evolutionary_seed_control:{arm_id}"
        overrides[key] = SeedSourceEvolutionaryControlPlanner(
            arm_id=arm_id,
            seed_source=source,
            grid_resolution=float(shared["grid_resolution"]),
            manual_seed_time_fraction=float(shared["manual_seed_time_fraction"]),
            population_size=int(shared["population_size"]),
            archive_size=int(shared["archive_size"]),
            max_generations=int(shared["max_generations"]),
            max_waypoints=int(shared["max_waypoints"]),
            base_iteration_limit=int(shared["base_iteration_limit"]),
            crossover_probability=float(shared["crossover_probability"]),
            extra_mutation_probability=float(shared["extra_mutation_probability"]),
        )
        selected_planners.append(key)
    if afl_arm_id is None:
        raise ValueError("missing AFL-seeded reference arm")

    report = run_planner_benchmark(
        benchmark_config,
        split=specification["split"],
        planners=selected_planners,
        maps_per_class=maps_per_class,
        time_limit_seconds=float(budget["time_limit_seconds"]),
        max_objective_evaluations=int(budget["max_objective_evaluations"]),
        repetitions=(
            int(repetitions) if repetitions is not None else int(budget["repetitions"])
        ),
        evolutionary_afl_artifacts={
            afl_arm_id: _resolve(specification["afl_seed_artifact"])
        },
        planner_overrides=overrides,
        run_id=specification["run_id"] + run_id_suffix,
    )
    run_dir = Path(report["run_dir"])
    result_hashes = {
        name: _sha256(run_dir / name)
        for name in (
            "benchmark_runs.csv",
            "benchmark_paths.jsonl",
            "benchmark_summary.json",
            "benchmark_metadata.json",
        )
    }
    receipt_body: dict[str, Any] = {
        "schema_version": "evolutionary-seed-source-controls-receipt-v1",
        "experiment_id": specification["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": specification["split"],
        "api_calls": 0,
        "hidden_test_access": False,
        "records": report["records"],
        "expected_records": report["expected_records"],
        "arms": specification["arms"],
        "shared_evolution_layer": shared,
        "shared_methods_inherited_verbatim": list(SHARED_METHODS),
        "experiment_config_sha256": _sha256(experiment_config),
        "control_source_sha256": _sha256(
            ROOT
            / "src/uav_operator_evolution/planning_benchmarks/evolutionary_seed_controls.py"
        ),
        "frozen_hashes": frozen_hashes,
        "result_hashes": result_hashes,
    }
    receipt = {
        **receipt_body,
        "receipt_id": stable_hash(receipt_body),
    }
    receipt_path = run_dir / "seed_source_control_receipt.json"
    _dump_json(receipt_path, receipt)
    return {
        **report,
        "experiment_id": specification["experiment_id"],
        "api_calls": 0,
        "control_receipt": str(receipt_path.resolve()),
        "control_receipt_id": receipt["receipt_id"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maps-per-class", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--run-id-suffix", default="")
    args = parser.parse_args()
    print(
        json.dumps(
            run_controls(
                args.config.resolve(),
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
