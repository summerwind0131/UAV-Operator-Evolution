"""Analyze the predeclared Evolutionary AFL-UAV seed-source controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from uav_operator_evolution.reproducibility import stable_hash


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/evolutionary_seed_source_controls_v1.yaml"


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else ROOT / candidate).resolve()
    if resolved != ROOT.resolve() and ROOT.resolve() not in resolved.parents:
        raise ValueError(f"analysis path escapes project root: {path}")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_runs(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for raw in csv.DictReader(stream):
            row = dict(raw)
            row["feasible"] = row["feasible"].strip().lower() == "true"
            row["protocol_feasible"] = bool(
                row["feasible"] and row["status"] == "success"
            )
            for name in (
                "total_cost",
                "elapsed_seconds",
                "objective_evaluations",
                "collision_checks",
                "node_expansions",
            ):
                row[name] = None if row[name] == "" else float(row[name])
            rows.append(row)
    return rows


def _read_diagnostics(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    diagnostics: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = (str(item["arm_id"]), str(item["map_id"]), str(item["seed"]))
        diagnostics[key] = item.get("diagnostics", {})
    return diagnostics


def _quantile(values: list[float], probability: float) -> float | None:
    return None if not values else float(np.quantile(np.asarray(values), probability))


def _arm_summary(
    rows: list[dict[str, Any]],
    diagnostics: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    feasible = [row for row in rows if row["protocol_feasible"]]
    costs = [float(row["total_cost"]) for row in feasible]
    times = [float(row["elapsed_seconds"]) for row in rows]
    evaluations = [float(row["objective_evaluations"]) for row in rows]
    maps: dict[str, list[bool]] = defaultdict(list)
    seed_costs: list[float] = []
    improvements: list[float] = []
    archive_diversities: list[float] = []
    for row in rows:
        maps[str(row["map_id"])].append(bool(row["protocol_feasible"]))
        key = (str(row["arm_id"]), str(row["map_id"]), str(row["seed"]))
        detail = diagnostics.get(key, {})
        if isinstance(detail.get("seed_cost"), (int, float)):
            seed_costs.append(float(detail["seed_cost"]))
        if isinstance(detail.get("absolute_improvement"), (int, float)):
            improvements.append(float(detail["absolute_improvement"]))
        if isinstance(detail.get("archive_unique_paths"), (int, float)):
            archive_diversities.append(float(detail["archive_unique_paths"]))
    status_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        status_counts[str(row["status"])] += 1
    return {
        "records": len(rows),
        "feasible": len(feasible),
        "feasible_rate": len(feasible) / len(rows),
        "trusted_geometry_feasible": sum(bool(row["feasible"]) for row in rows),
        "timeouts_with_trusted_path": sum(
            row["status"] == "timeout" and bool(row["feasible"])
            for row in rows
        ),
        "maps": len(maps),
        "maps_with_any_feasible": sum(any(values) for values in maps.values()),
        "maps_with_all_repetitions_feasible": sum(all(values) for values in maps.values()),
        "status_counts": dict(sorted(status_counts.items())),
        "median_cost_feasible": statistics.median(costs) if costs else None,
        "cost_q25": _quantile(costs, 0.25),
        "cost_q75": _quantile(costs, 0.75),
        "median_elapsed_seconds": statistics.median(times),
        "median_objective_evaluations": statistics.median(evaluations),
        "median_seed_cost": statistics.median(seed_costs) if seed_costs else None,
        "median_outer_improvement": (
            statistics.median(improvements) if improvements else None
        ),
        "mean_archive_unique_paths": (
            statistics.fmean(archive_diversities) if archive_diversities else None
        ),
    }


def _exact_two_sided_sign_p(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, index) for index in range(min(wins, losses) + 1))
    return min(1.0, 2.0 * tail / (2**trials))


def _holm_adjust(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        running = max(running, (total - index) * raw[name])
        adjusted[name] = min(1.0, running)
    return adjusted


def _cluster_bootstrap_median_difference(
    differences: dict[str, list[float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, float] | None:
    maps = sorted(differences)
    if not maps:
        return None
    rng = np.random.default_rng(seed)
    estimates = np.empty(resamples, dtype=float)
    for index in range(resamples):
        sample = rng.choice(maps, size=len(maps), replace=True)
        values = [value for map_id in sample for value in differences[str(map_id)]]
        estimates[index] = float(np.median(values))
    observed = float(np.median([value for values in differences.values() for value in values]))
    return {
        "observed_median_control_minus_afl": observed,
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
    }


def _pairwise(
    primary_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    *,
    tolerance: float,
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    primary = {(row["map_id"], row["seed"]): row for row in primary_rows}
    control = {(row["map_id"], row["seed"]): row for row in control_rows}
    if set(primary) != set(control):
        raise RuntimeError("paired control keys do not match the AFL reference keys")
    wins = losses = ties = 0
    feasibility_wins = feasibility_losses = 0
    cost_wins = cost_losses = cost_ties = 0
    both_feasible = 0
    differences: dict[str, list[float]] = defaultdict(list)
    for key in sorted(primary):
        afl = primary[key]
        comparator = control[key]
        if afl["protocol_feasible"] and not comparator["protocol_feasible"]:
            wins += 1
            feasibility_wins += 1
        elif comparator["protocol_feasible"] and not afl["protocol_feasible"]:
            losses += 1
            feasibility_losses += 1
        elif not afl["protocol_feasible"] and not comparator["protocol_feasible"]:
            ties += 1
        else:
            both_feasible += 1
            difference = float(comparator["total_cost"]) - float(afl["total_cost"])
            differences[str(key[0])].append(difference)
            if difference > tolerance:
                wins += 1
                cost_wins += 1
            elif difference < -tolerance:
                losses += 1
                cost_losses += 1
            else:
                ties += 1
                cost_ties += 1
    return {
        "pairs": len(primary),
        "afl_wins": wins,
        "ties": ties,
        "afl_losses": losses,
        "both_feasible": both_feasible,
        "win_loss_decomposition": {
            "feasibility_wins": feasibility_wins,
            "feasibility_losses": feasibility_losses,
            "cost_wins_on_both_feasible": cost_wins,
            "cost_ties_on_both_feasible": cost_ties,
            "cost_losses_on_both_feasible": cost_losses,
        },
        "raw_two_sided_sign_p": _exact_two_sided_sign_p(wins, losses),
        "paired_cost_difference": _cluster_bootstrap_median_difference(
            differences,
            resamples=resamples,
            seed=bootstrap_seed,
        ),
    }


def analyze(config_path: Path, run_dir: Path | None = None) -> dict[str, Any]:
    specification = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if specification.get("split") not in {"train", "validation"}:
        raise ValueError("seed-source analysis is Train/Validation-only")
    destination = run_dir or (
        ROOT / "artifacts/planning_benchmarks" / specification["run_id"]
    )
    destination = _resolve(destination)
    runs_path = destination / "benchmark_runs.csv"
    paths_path = destination / "benchmark_paths.jsonl"
    rows = _read_runs(runs_path)
    diagnostics = _read_diagnostics(paths_path)
    expected_arms = list(specification["arms"])
    grouped = {arm: [row for row in rows if row["arm_id"] == arm] for arm in expected_arms}
    if any(not grouped[arm] for arm in expected_arms):
        missing = [arm for arm in expected_arms if not grouped[arm]]
        raise RuntimeError("missing seed-source control arms: " + ", ".join(missing))
    record_keys = {
        (row["arm_id"], row["map_id"], row["seed"])
        for row in rows
    }
    if len(record_keys) != len(rows):
        raise RuntimeError("duplicate arm/map/seed records in control experiment")

    analysis_spec = specification["analysis"]
    primary_arm = analysis_spec["primary_arm"]
    summaries = {arm: _arm_summary(grouped[arm], diagnostics) for arm in expected_arms}
    pairwise = {
        arm: _pairwise(
            grouped[primary_arm],
            grouped[arm],
            tolerance=float(analysis_spec["cost_tolerance"]),
            resamples=int(analysis_spec["bootstrap_resamples"]),
            bootstrap_seed=int(analysis_spec["bootstrap_seed"]),
        )
        for arm in analysis_spec["comparison_arms"]
    }
    adjusted = _holm_adjust(
        {arm: result["raw_two_sided_sign_p"] for arm, result in pairwise.items()}
    )
    for arm, value in adjusted.items():
        pairwise[arm]["holm_adjusted_sign_p"] = value

    primary_feasibility = summaries[primary_arm]["feasible_rate"]
    alpha = float(
        analysis_spec["superiority_rule"]["holm_corrected_sign_test_alpha"]
    )
    criteria = {
        arm: {
            "feasible_rate_not_lower": (
                primary_feasibility + 1e-15 >= summaries[arm]["feasible_rate"]
            ),
            "paired_wins_exceed_losses": (
                pairwise[arm]["afl_wins"] > pairwise[arm]["afl_losses"]
            ),
            "holm_significant": pairwise[arm]["holm_adjusted_sign_p"] < alpha,
        }
        for arm in analysis_spec["comparison_arms"]
    }
    clearly_superior = all(all(values.values()) for values in criteria.values())
    conclusion = (
        "afl_seed_clearly_superior_under_preregistered_validation_rule"
        if clearly_superior
        else "afl_seed_superiority_not_established"
    )
    body: dict[str, Any] = {
        "schema_version": "evolutionary-seed-source-controls-analysis-v1",
        "experiment_id": specification["experiment_id"],
        "created_at_utc": datetime.now(UTC).isoformat(),
        "split": specification["split"],
        "records": len(rows),
        "protocol_correction": {
            "rule": "status must be success and trusted path must be feasible",
            "reason": "registered protocol counts every timeout as failure",
            "raw_feasible_column_semantics": "trusted geometry feasibility",
            "raw_runner_summary_authoritative": False,
        },
        "arms": summaries,
        "pairwise_against_afl_seed": pairwise,
        "superiority_criteria": criteria,
        "conclusion": conclusion,
        "api_calls": 0,
        "hidden_test_access": False,
        "input_hashes": {
            "config_sha256": _sha256(config_path),
            "runs_sha256": _sha256(runs_path),
            "paths_sha256": _sha256(paths_path),
            "analysis_source_sha256": _sha256(Path(__file__)),
        },
    }
    report = {**body, "analysis_id": stable_hash(body)}
    json_path = destination / "seed_source_control_analysis.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Evolutionary AFL-UAV seed-source control analysis",
        "",
        f"Conclusion: `{conclusion}`",
        "",
        "Protocol feasible means `status == success` and trusted geometry feasible; every timeout is a failure.",
        "",
        "| Arm | Protocol feasible | Timeouts | Median cost | Median seed cost | Median time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm in expected_arms:
        item = summaries[arm]
        lines.append(
            f"| {arm} | {item['feasible']}/{item['records']} "
            f"({item['feasible_rate']:.3%}) | {item['status_counts'].get('timeout', 0)} | "
            f"{item['median_cost_feasible']:.6f} | "
            f"{item['median_seed_cost']:.6f} | "
            f"{item['median_elapsed_seconds']:.6f} |"
        )
    lines.extend(
        [
            "",
            "| Comparator | AFL wins | Ties | AFL losses | Reliability wins | Cost W/T/L on both feasible | Holm p | Median cost difference (control - AFL) | 95% cluster bootstrap CI |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in analysis_spec["comparison_arms"]:
        item = pairwise[arm]
        difference = item["paired_cost_difference"]
        decomposition = item["win_loss_decomposition"]
        lines.append(
            f"| {arm} | {item['afl_wins']} | {item['ties']} | "
            f"{item['afl_losses']} | {decomposition['feasibility_wins']} | "
            f"{decomposition['cost_wins_on_both_feasible']}/"
            f"{decomposition['cost_ties_on_both_feasible']}/"
            f"{decomposition['cost_losses_on_both_feasible']} | "
            f"{item['holm_adjusted_sign_p']:.6g} | "
            f"{difference['observed_median_control_minus_afl']:.6f} | "
            f"[{difference['ci95_low']:.6f}, {difference['ci95_high']:.6f}] |"
        )
    markdown_path = destination / "seed_source_control_analysis.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "analysis_id": report["analysis_id"],
        "conclusion": conclusion,
        "records": len(rows),
        "json": str(json_path.resolve()),
        "markdown": str(markdown_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(
                args.config.resolve(),
                None if args.run_dir is None else args.run_dir.resolve(),
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
