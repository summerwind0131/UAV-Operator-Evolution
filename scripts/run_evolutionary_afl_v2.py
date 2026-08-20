"""Run development-only Evolutionary AFL-UAV v2 on Train/Validation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from uav_operator_evolution.config import load_config
from uav_operator_evolution.planning_benchmarks.evolutionary_afl_v2 import (
    EvolutionaryAFLUAVV2Planner,
)
from uav_operator_evolution.planning_benchmarks.runner import run_planner_benchmark


DEFAULT_CONFIG = PROJECT_ROOT / "configs/evolutionary_afl_uav_v2_development.yaml"
FROZEN_V1_HASH = "79f0a085a0f26b246d2f1e0d0bc1ac7e8a6288b34dd0fbc8f29bc9387e1a7d4f"


def _resolve(value: str | Path) -> Path:
    raw = Path(value)
    result = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    root = PROJECT_ROOT.resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"path escapes project root: {value}")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _effective_feasible(row: dict[str, str], time_limit: float) -> bool:
    return (
        row["feasible"].lower() == "true"
        and row["status"] != "timeout"
        and float(row["elapsed_seconds"]) < time_limit
    )


def _metrics(rows: list[dict[str, str]], time_limit: float) -> dict[str, Any]:
    feasible = [row for row in rows if _effective_feasible(row, time_limit)]
    costs = [float(row["total_cost"]) for row in feasible]
    maps = {row["map_id"] for row in rows}
    successful_maps = {row["map_id"] for row in feasible}
    return {
        "runs": len(rows),
        "trusted_feasible_runs": len(feasible),
        "trusted_feasible_rate": len(feasible) / len(rows),
        "maps": len(maps),
        "map_success_rate": len(successful_maps) / len(maps),
        "timeouts": sum(
            row["status"] == "timeout"
            or float(row["elapsed_seconds"]) >= time_limit
            for row in rows
        ),
        "median_trusted_cost": float(np.median(costs)) if costs else None,
        "iqr_trusted_cost": (
            float(np.quantile(costs, 0.75) - np.quantile(costs, 0.25))
            if costs
            else None
        ),
    }


def _paired(
    v2_rows: list[dict[str, str]],
    v1_rows: list[dict[str, str]],
    time_limit: float,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, str]]]] = {
        "v1": defaultdict(list),
        "v2": defaultdict(list),
    }
    for label, rows in (("v1", v1_rows), ("v2", v2_rows)):
        for row in rows:
            grouped[label][row["map_id"]].append(row)
    wins = ties = losses = 0
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for map_id in sorted(set(grouped["v1"]) & set(grouped["v2"])):
        costs: dict[str, list[float]] = {}
        for label in ("v1", "v2"):
            costs[label] = [
                float(row["total_cost"])
                for row in grouped[label][map_id]
                if _effective_feasible(row, time_limit)
            ]
        if costs["v2"] and not costs["v1"]:
            outcome = "win"
        elif costs["v1"] and not costs["v2"]:
            outcome = "loss"
        elif not costs["v1"] and not costs["v2"]:
            outcome = "tie"
        else:
            first = float(np.median(costs["v2"]))
            second = float(np.median(costs["v1"]))
            tolerance = 1e-9 * max(1.0, abs(first), abs(second))
            outcome = "win" if first < second - tolerance else (
                "loss" if first > second + tolerance else "tie"
            )
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            ties += 1
        difficulty = grouped["v2"][map_id][0]["difficulty"]
        by_class[difficulty][outcome] += 1
    return {
        "maps": wins + ties + losses,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "half_tie_win_rate": (
            (wins + 0.5 * ties) / (wins + ties + losses)
            if wins + ties + losses
            else None
        ),
        "by_class": {
            difficulty: {
                "wins": counts["win"],
                "ties": counts["tie"],
                "losses": counts["loss"],
            }
            for difficulty, counts in sorted(by_class.items())
        },
    }


def run_v2(
    development_config_path: Path,
    *,
    maps_per_class: int | None = None,
    repetitions: int | None = None,
    run_id_suffix: str = "",
    variant: str | None = None,
) -> dict[str, Any]:
    raw = yaml.safe_load(development_config_path.read_text(encoding="utf-8"))
    split = raw["split"]
    if split not in raw["development_policy"]["allowed_splits"]:
        raise PermissionError("v2 development is restricted to Train/Validation")
    if split == "test":
        raise PermissionError("v2 must never access a Test split during development")
    v1_source = _resolve(
        "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl.py"
    )
    if _sha256(v1_source) != FROZEN_V1_HASH:
        raise RuntimeError("frozen Evolutionary AFL-UAV v1 source changed")
    benchmark_config = load_config(_resolve(raw["benchmark_config"]))
    if "hidden" in str(benchmark_config.output.data_dir).lower():
        raise PermissionError("v2 development config points at hidden data")
    parameters = dict(raw["parameters"])
    if variant is not None:
        parameters["variant"] = variant
    source_quotas = parameters.pop("source_quotas")
    selected_variant = str(parameters["variant"])
    planner = EvolutionaryAFLUAVV2Planner(
        _resolve(raw["seed_artifact"]),
        arm_id=f"evolutionary_afl_uav_v2_{selected_variant}",
        source_quotas=source_quotas,
        **parameters,
    )
    planner_key = (
        "evolutionary_afl_uav:"
        f"evolutionary_afl_uav_v2_{selected_variant}"
    )
    run_id = raw["run_id"] + run_id_suffix
    report = run_planner_benchmark(
        benchmark_config,
        split=split,
        planners=[planner_key],
        maps_per_class=maps_per_class,
        time_limit_seconds=float(raw["budget"]["time_limit_seconds"]),
        max_objective_evaluations=int(raw["budget"]["max_objective_evaluations"]),
        repetitions=(
            int(repetitions)
            if repetitions is not None
            else int(raw["budget"]["repetitions"])
        ),
        planner_overrides={planner_key: planner},
        run_id=run_id,
    )
    run_dir = Path(report["run_dir"])
    v2_rows = _read_rows(run_dir / "benchmark_runs.csv")
    selected_keys = {
        (row["map_id"], int(row["repetition"]), int(row["seed"]))
        for row in v2_rows
    }
    v1_rows = [
        row
        for row in _read_rows(_resolve(raw["v1_validation_results"]))
        if (row["map_id"], int(row["repetition"]), int(row["seed"]))
        in selected_keys
    ]
    if len(v1_rows) != len(v2_rows):
        raise RuntimeError("v1/v2 Validation records are not exactly paired")
    time_limit = float(raw["budget"]["time_limit_seconds"])
    paths = _read_jsonl(run_dir / "benchmark_paths.jsonl")
    source_status = Counter()
    source_population = Counter()
    reserve_values: list[float] = []
    portal_counts: list[int] = []
    for row in paths:
        diagnostics = row.get("diagnostics", {})
        for source, status in diagnostics.get("portfolio_source_status", {}).items():
            source_status[f"{source}:{status}"] += 1
        for source, count in diagnostics.get("population_source_counts", {}).items():
            source_population[source] += int(count)
        if "v2_finalization_reserve_seconds" in diagnostics:
            reserve_values.append(float(diagnostics["v2_finalization_reserve_seconds"]))
        portal_counts.append(int(diagnostics.get("portal_centers_detected", 0)))
    comparison = {
        "schema_version": "evolutionary-afl-uav-v2-validation-comparison-v1",
        "status": "development_only",
        "split": split,
        "maps_per_class": maps_per_class,
        "repetitions": repetitions or int(raw["budget"]["repetitions"]),
        "api_calls": 0,
        "hidden_test_v2_accessed": False,
        "research_claim_eligible": False,
        "variant": selected_variant,
        "v1_source_sha256": _sha256(v1_source),
        "v2_source_sha256": _sha256(
            _resolve(
                "src/uav_operator_evolution/planning_benchmarks/evolutionary_afl_v2.py"
            )
        ),
        "v1": _metrics(v1_rows, time_limit),
        "v2": _metrics(v2_rows, time_limit),
        "v2_vs_v1_paired": _paired(v2_rows, v1_rows, time_limit),
        "rooms_maze": {
            "v1": _metrics(
                [row for row in v1_rows if row["difficulty"] == "rooms_maze"],
                time_limit,
            ),
            "v2": _metrics(
                [row for row in v2_rows if row["difficulty"] == "rooms_maze"],
                time_limit,
            ),
        },
        "v2_diagnostics": {
            "portfolio_status_counts": dict(sorted(source_status.items())),
            "population_source_totals": dict(sorted(source_population.items())),
            "minimum_finalization_reserve_seconds": (
                min(reserve_values) if reserve_values else None
            ),
            "median_finalization_reserve_seconds": (
                float(np.median(reserve_values)) if reserve_values else None
            ),
            "maximum_portals_detected": max(portal_counts, default=0),
        },
        "next_step": (
            "iterate only on Train/Validation; freeze v2 before creating a new hidden Test-v3"
        ),
    }
    (run_dir / "v2_validation_comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"benchmark": report, "comparison": comparison}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--maps-per-class", type=int)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--run-id-suffix", default="")
    parser.add_argument(
        "--variant",
        choices=("reliability_only", "multisource", "full"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run_v2(
                args.config.resolve(),
                maps_per_class=args.maps_per_class,
                repetitions=args.repetitions,
                run_id_suffix=args.run_id_suffix,
                variant=args.variant,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
