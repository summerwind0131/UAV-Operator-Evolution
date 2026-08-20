"""Audit and analyse the real DeepSeek V4 Pro AFL-UAV Validation run.

The analysis deliberately combines only fixed UAV2D-v1 Validation artifacts.
It independently checks the 300 DeepSeek path records, merges them with the
traditional baselines and the offline AFL control, and writes reproducible CSV
and JSON evidence.  It never reads the Test split.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_offline_validation import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_DIFFICULTIES,
    _by_difficulty_statistics,
    _failure_status,
    _normalise_run,
    _overall_statistics,
    _paired_comparisons,
    _per_map_results,
    _round_for_csv,
    _seed_stability,
    _sha256,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADITIONAL = (
    ROOT / "artifacts" / "planning_benchmarks" / "offline-traditional-validation-v1"
)
DEFAULT_OFFLINE_AFL = (
    ROOT / "artifacts" / "planning_benchmarks" / "afl-uav-offline-v3-validation-v1"
)
DEFAULT_DEEPSEEK = (
    ROOT / "artifacts" / "planning_benchmarks" / "deepseek-v4pro-strict-validation-v2"
)
DEFAULT_CANDIDATES = (
    ROOT / "artifacts" / "planning_benchmarks" / "afl_uav_candidates"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "deepseek-v4pro-strict-validation-analysis-v2"
)

REAL_KEY = "afl_uav:deepseek_v4pro_strict"
OFFLINE_KEY = "afl_uav:offline_v3"
LABELS = {
    "afl_uav:deepseek_v4pro_strict": "AFL-UAV (DeepSeek V4 Pro, strict)",
    "afl_uav:offline_v3": "AFL-UAV (offline v3 control)",
    "dijkstra": "Dijkstra",
    "astar": "A*",
    "theta_star": "Theta*",
    "rrt": "RRT",
    "rrt_star": "RRT*",
    "prm": "PRM",
    "ga": "GA",
    "pso": "PSO",
    "de": "DE",
    "aco_acor": "ACO/ACOR",
}
EXPECTED_COUNTS = {
    "dijkstra": 60,
    "astar": 60,
    "theta_star": 60,
    "rrt": 300,
    "rrt_star": 300,
    "prm": 300,
    "ga": 300,
    "pso": 300,
    "de": 300,
    "aco_acor": 300,
    OFFLINE_KEY: 300,
    REAL_KEY: 300,
}
STOCHASTIC_KEYS = {
    "rrt",
    "rrt_star",
    "prm",
    "ga",
    "pso",
    "de",
    "aco_acor",
    OFFLINE_KEY,
    REAL_KEY,
}

# The V4 Pro minimal probe was intentionally not persisted with a response body.
# These are the non-secret counters printed by that probe and recorded in the
# task audit: 90 input, 5 output, 95 total, one attempt, 3.157 seconds.
PROBE_USAGE = {
    "input_tokens": 90,
    "output_tokens": 5,
    "total_tokens": 95,
    "logical_calls": 1,
    "http_attempts": 1,
    "retries": 0,
    "latency_ms": 3157.0,
}


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def _load_runs(path: Path, metadata: dict[str, Any]) -> pd.DataFrame:
    frame = _normalise_run(path / "benchmark_runs.csv", metadata["run_id"])
    frame["planner_label"] = frame["planner_key"].map(LABELS)
    if frame["planner_label"].isna().any():
        missing = sorted(frame.loc[frame["planner_label"].isna(), "planner_key"].unique())
        raise ValueError(f"missing planner labels: {missing}")
    return frame


def _path_record_audit(deepseek_dir: Path, runs: pd.DataFrame) -> dict[str, Any]:
    # Imported here so this script can still show --help without project imports.
    from uav_operator_evolution.planning_benchmarks import path_hash

    paths = _read_jsonl(deepseek_dir / "benchmark_paths.jsonl")
    key_columns = ["planner", "arm_id", "map_id", "seed"]
    path_frame = pd.DataFrame(paths)
    errors: list[str] = []
    if len(paths) != 300:
        errors.append(f"expected 300 path rows, found {len(paths)}")
    if path_frame.duplicated(key_columns).any():
        errors.append("duplicate path-record keys found")
    real_runs = runs[runs["planner_key"].eq(REAL_KEY)].copy()
    run_keys = {
        tuple(row)
        for row in real_runs[key_columns].itertuples(index=False, name=None)
    }
    path_keys = {
        tuple(row)
        for row in path_frame[key_columns].itertuples(index=False, name=None)
    }
    if run_keys != path_keys:
        errors.append("CSV and JSONL path-record keys do not match")

    expected_hash = {
        tuple(row[column] for column in key_columns): row["path_hash"]
        for _, row in real_runs.iterrows()
    }
    hash_mismatches = 0
    missing_feasible_paths = 0
    for row in paths:
        key = tuple(row[column] for column in key_columns)
        actual_hash = path_hash(row.get("path"))
        if actual_hash != expected_hash.get(key):
            hash_mismatches += 1
        if bool(row.get("feasible")) and not row.get("path"):
            missing_feasible_paths += 1
    if hash_mismatches:
        errors.append(f"{hash_mismatches} path hashes do not match the CSV")
    if missing_feasible_paths:
        errors.append(f"{missing_feasible_paths} feasible records lack a path")
    return {
        "status": "passed" if not errors else "failed",
        "path_records": len(paths),
        "unique_path_record_keys": len(path_keys),
        "hash_mismatches": hash_mismatches,
        "missing_feasible_paths": missing_feasible_paths,
        "errors": errors,
    }


def _contract_audit(
    runs: pd.DataFrame,
    metadata: list[dict[str, Any]],
    deepseek_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    expected_total = sum(EXPECTED_COUNTS.values())
    key_columns = ["planner", "arm_id", "map_id", "seed"]
    if len(runs) != expected_total:
        errors.append(f"expected {expected_total} rows, found {len(runs)}")
    if runs.duplicated(key_columns).any():
        errors.append("duplicate benchmark record keys found")

    actual_counts = runs.groupby("planner_key").size().to_dict()
    if actual_counts != EXPECTED_COUNTS:
        errors.append(f"planner counts differ: {actual_counts}")
    if set(runs["split"].unique()) != {"validation"}:
        errors.append("records outside Validation were found")
    if set(runs["difficulty"].unique()) != EXPECTED_DIFFICULTIES:
        errors.append("the six expected map classes are not present")
    if runs["map_id"].nunique() != 60:
        errors.append(f"expected 60 maps, found {runs['map_id'].nunique()}")
    maps_per_class = (
        runs[["difficulty", "map_id"]]
        .drop_duplicates()
        .groupby("difficulty")
        .size()
        .to_dict()
    )
    if set(maps_per_class.values()) != {10}:
        errors.append(f"map classes are not balanced: {maps_per_class}")

    for field in ["manifest_hash", "config_hash", "benchmark_id", "budget", "objective"]:
        values = {json.dumps(item.get(field), sort_keys=True) for item in metadata}
        if len(values) != 1:
            errors.append(f"metadata field differs across runs: {field}")

    max_evaluations = int(metadata[0]["budget"]["max_objective_evaluations"])
    time_limit = float(metadata[0]["budget"]["time_limit_seconds"])
    if (runs["objective_evaluations"] < 0).any() or (
        runs["objective_evaluations"] > max_evaluations
    ).any():
        errors.append("objective-evaluation budget violation found")
    if (runs["elapsed_seconds"] < 0).any() or (
        runs["elapsed_seconds"] > time_limit + 0.05
    ).any():
        errors.append("planner time-budget violation found")
    if (runs[["collision_checks", "node_expansions"]] < 0).any().any():
        errors.append("negative diagnostic counter found")
    if (runs["feasible"] & runs["total_cost"].isna()).any():
        errors.append("a feasible row lacks trusted cost")
    if ((~runs["feasible"]) & runs["total_cost"].notna()).any():
        errors.append("an infeasible row contains trusted cost")

    for key in ["dijkstra", "astar", "theta_star"]:
        group = runs[runs["planner_key"].eq(key)]
        if len(group) != 60 or group["map_id"].nunique() != 60:
            errors.append(f"{key} is not one run per map")
    seed_sets: dict[str, dict[str, tuple[int, ...]]] = {}
    for key in sorted(STOCHASTIC_KEYS):
        group = runs[runs["planner_key"].eq(key)]
        counts = group.groupby("map_id").size()
        if len(counts) != 60 or not counts.eq(5).all():
            errors.append(f"{key} is not five runs per map")
        seed_sets[key] = {
            map_id: tuple(sorted(map_group["seed"].astype("int64").tolist()))
            for map_id, map_group in group.groupby("map_id")
        }
    reference = seed_sets["rrt"]
    for key, value in seed_sets.items():
        if value != reference:
            errors.append(f"{key} does not use the shared seed set")

    real = runs[runs["planner_key"].eq(REAL_KEY)]
    offline = runs[runs["planner_key"].eq(OFFLINE_KEY)]
    if not real["research_claim_eligible"].all():
        errors.append("DeepSeek arm is not consistently research eligible")
    if offline["research_claim_eligible"].any():
        errors.append("offline control is incorrectly research eligible")
    artifact = metadata[-1].get("afl_uav_artifact") or {}
    if artifact.get("provider") != "deepseek" or artifact.get("model") != "deepseek-v4-pro":
        errors.append("DeepSeek artifact provider/model mismatch")
    if not artifact.get("qualification_passed"):
        errors.append("DeepSeek artifact lacks a passed qualification")
    if artifact.get("approved_source_hash") != artifact.get("candidate_source_hash"):
        errors.append("approved and candidate source hashes differ")
    artifact_manifest_path = Path(str(artifact.get("artifact_path", ""))) / "artifact.json"
    artifact_manifest = (
        _read_json(artifact_manifest_path) if artifact_manifest_path.is_file() else {}
    )
    human_review = artifact_manifest.get("human_review") or artifact.get("human_review") or {}
    if human_review.get("decision") != "approved_for_restricted_qualification":
        errors.append("strict DeepSeek artifact lacks the delegated human source decision")
    if human_review.get("approved_source_hash") != artifact.get("solver_hash"):
        errors.append("human source decision is not bound to the frozen solver hash")

    path_audit = _path_record_audit(deepseek_dir, runs)
    errors.extend(path_audit["errors"])
    return {
        "status": "passed" if not errors else "failed",
        "expected_records": expected_total,
        "actual_records": int(len(runs)),
        "unique_record_keys": int(runs[key_columns].drop_duplicates().shape[0]),
        "unique_maps": int(runs["map_id"].nunique()),
        "maps_per_class": maps_per_class,
        "planner_record_counts": actual_counts,
        "all_sources_validation_only": set(runs["split"].unique()) == {"validation"},
        "shared_stochastic_seed_sets": not any("shared seed" in item for item in errors),
        "budget": metadata[0]["budget"],
        "path_records": path_audit,
        "errors": errors,
    }


def _exact_seed_comparison(runs: pd.DataFrame) -> pd.DataFrame:
    candidate = runs[runs["planner_key"].eq(REAL_KEY)][
        ["map_id", "seed", "feasible", "total_cost"]
    ].copy()
    rows: list[dict[str, Any]] = []
    for baseline_key in [OFFLINE_KEY, "rrt_star", "rrt", "ga", "pso", "de", "prm", "aco_acor"]:
        baseline = runs[runs["planner_key"].eq(baseline_key)][
            ["map_id", "seed", "feasible", "total_cost"]
        ].copy()
        joined = candidate.merge(
            baseline,
            on=["map_id", "seed"],
            suffixes=("_candidate", "_baseline"),
            validate="one_to_one",
        )
        both = joined[joined["feasible_candidate"] & joined["feasible_baseline"]].copy()
        delta = both["total_cost_candidate"] - both["total_cost_baseline"]
        rows.append(
            {
                "candidate": REAL_KEY,
                "baseline": baseline_key,
                "paired_runs": int(len(joined)),
                "both_feasible_runs": int(len(both)),
                "candidate_only_feasible_runs": int(
                    (joined["feasible_candidate"] & ~joined["feasible_baseline"]).sum()
                ),
                "baseline_only_feasible_runs": int(
                    (~joined["feasible_candidate"] & joined["feasible_baseline"]).sum()
                ),
                "wins": int((delta < -1e-9).sum()),
                "ties": int((delta.abs() <= 1e-9).sum()),
                "losses": int((delta > 1e-9).sum()),
                "strict_win_rate_on_both_feasible": (
                    float((delta < -1e-9).mean()) if len(delta) else np.nan
                ),
                "median_cost_delta": float(delta.median()) if len(delta) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _paired_by_difficulty(per_map: pd.DataFrame) -> pd.DataFrame:
    """Compare per-map median DeepSeek cost with deterministic baselines."""

    rows: list[dict[str, Any]] = []
    candidate = per_map[per_map["planner_key"].eq(REAL_KEY)]
    for difficulty in sorted(EXPECTED_DIFFICULTIES):
        candidate_class = candidate[candidate["difficulty"].eq(difficulty)][
            ["map_id", "map_feasible", "median_feasible_cost"]
        ]
        for baseline_key in ["astar", "theta_star"]:
            baseline = per_map[
                per_map["planner_key"].eq(baseline_key)
                & per_map["difficulty"].eq(difficulty)
            ][["map_id", "map_feasible", "median_feasible_cost"]]
            joined = candidate_class.merge(
                baseline,
                on="map_id",
                suffixes=("_candidate", "_baseline"),
                validate="one_to_one",
            )
            both = joined[
                joined["map_feasible_candidate"] & joined["map_feasible_baseline"]
            ]
            delta = both["median_feasible_cost_candidate"] - both["median_feasible_cost_baseline"]
            rows.append(
                {
                    "difficulty": difficulty,
                    "baseline": baseline_key,
                    "maps": int(len(joined)),
                    "both_feasible_maps": int(len(both)),
                    "candidate_only_feasible_maps": int(
                        (joined["map_feasible_candidate"] & ~joined["map_feasible_baseline"]).sum()
                    ),
                    "baseline_only_feasible_maps": int(
                        (~joined["map_feasible_candidate"] & joined["map_feasible_baseline"]).sum()
                    ),
                    "wins": int((delta < -1e-9).sum()),
                    "ties": int((delta.abs() <= 1e-9).sum()),
                    "losses": int((delta > 1e-9).sum()),
                    "median_cost_delta": float(delta.median()) if len(delta) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _generation_usage(candidate_root: Path) -> dict[str, Any]:
    calls: dict[str, dict[str, Any]] = {}
    files: list[Path] = []
    for directory in sorted(candidate_root.glob("deepseek-v4pro-*")):
        for name in ["candidate.json", "candidate_failure.json"]:
            path = directory / name
            if path.is_file():
                files.append(path)
                payload = _read_json(path)
                for index, call in enumerate(payload.get("provider_calls", [])):
                    # Audit revisions preserve upstream calls verbatim.  Response ID
                    # deduplication counts each actual API response only once.
                    identity = call.get("response_id") or (
                        f"{path}:{index}:{call.get('request_hash')}:{call.get('latency_ms')}"
                    )
                    calls.setdefault(str(identity), call)
    input_tokens = sum(int(call.get("usage", {}).get("input_tokens", 0)) for call in calls.values())
    output_tokens = sum(int(call.get("usage", {}).get("output_tokens", 0)) for call in calls.values())
    retries = sum(int(call.get("retry_count", 0)) for call in calls.values())
    http_attempts = sum(int(call.get("attempts", 1)) for call in calls.values())
    latency_ms = sum(float(call.get("latency_ms", 0.0)) for call in calls.values())
    # Current experiment accounting rate.  The source URL is saved so a later
    # rerun can replace these rates without changing token counts.
    input_rate_per_million = 0.435
    output_rate_per_million = 0.87
    generation_cost = (
        input_tokens * input_rate_per_million
        + output_tokens * output_rate_per_million
    ) / 1_000_000
    probe_cost = (
        PROBE_USAGE["input_tokens"] * input_rate_per_million
        + PROBE_USAGE["output_tokens"] * output_rate_per_million
    ) / 1_000_000
    return {
        "deduplication_key": "provider response_id",
        "candidate_audit_files": len(files),
        "generation_logical_calls": len(calls),
        "generation_http_attempts": http_attempts,
        "generation_retries": retries,
        "generation_input_tokens": input_tokens,
        "generation_output_tokens": output_tokens,
        "generation_total_tokens": input_tokens + output_tokens,
        "generation_provider_latency_ms": latency_ms,
        "minimal_probe": PROBE_USAGE,
        "all_live_calls_including_probe": len(calls) + 1,
        "all_input_tokens_including_probe": input_tokens + PROBE_USAGE["input_tokens"],
        "all_output_tokens_including_probe": output_tokens + PROBE_USAGE["output_tokens"],
        "all_total_tokens_including_probe": input_tokens + output_tokens + PROBE_USAGE["total_tokens"],
        "estimated_generation_cost_usd": generation_cost,
        "estimated_probe_cost_usd": probe_cost,
        "estimated_all_live_cost_usd": generation_cost + probe_cost,
        "price_assumption_usd_per_million_tokens": {
            "input": input_rate_per_million,
            "output": output_rate_per_million,
            "source": "https://api-docs.deepseek.com/quick_start/pricing/",
        },
    }


def analyse(
    traditional_dir: Path,
    offline_dir: Path,
    deepseek_dir: Path,
    candidate_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_dirs = [traditional_dir, offline_dir, deepseek_dir]
    metadata = [_read_json(path / "benchmark_metadata.json") for path in source_dirs]
    runs = pd.concat(
        [_load_runs(path, item) for path, item in zip(source_dirs, metadata, strict=True)],
        ignore_index=True,
        sort=False,
    ).sort_values(["planner_key", "map_id", "seed"]).reset_index(drop=True)

    audit = _contract_audit(runs, metadata, deepseek_dir)
    if audit["errors"]:
        raise ValueError("DeepSeek Validation contract failed:\n- " + "\n- ".join(audit["errors"]))

    overall = _overall_statistics(runs)
    by_difficulty = _by_difficulty_statistics(runs)
    per_map = _per_map_results(runs)
    paired = _paired_comparisons(per_map)
    stability, stability_per_map = _seed_stability(runs)
    status = _failure_status(runs)
    exact_seed = _exact_seed_comparison(runs)
    paired_difficulty = _paired_by_difficulty(per_map)
    usage = _generation_usage(candidate_root)
    frozen_artifact_metadata = metadata[-1].get("afl_uav_artifact") or {}
    frozen_artifact_path = Path(str(frozen_artifact_metadata.get("artifact_path", ""))) / "artifact.json"
    frozen_artifact_manifest = (
        _read_json(frozen_artifact_path) if frozen_artifact_path.is_file() else {}
    )
    frozen_usage = frozen_artifact_metadata.get("llm_usage") or {}
    frozen_cost = (
        int(frozen_usage.get("input_tokens", 0)) * 0.435
        + int(frozen_usage.get("output_tokens", 0)) * 0.87
    ) / 1_000_000

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_outputs = {
        "all_runs": (runs, output_dir / "validation_runs.csv"),
        "overall": (overall, output_dir / "overall_statistics.csv"),
        "by_difficulty": (by_difficulty, output_dir / "by_difficulty_statistics.csv"),
        "per_map": (per_map, output_dir / "per_map_statistics.csv"),
        "paired_map_medians": (paired, output_dir / "paired_map_comparisons.csv"),
        "paired_exact_seed": (exact_seed, output_dir / "paired_seed_comparisons.csv"),
        "paired_by_difficulty": (
            paired_difficulty,
            output_dir / "paired_difficulty_comparisons.csv",
        ),
        "seed_stability": (stability, output_dir / "seed_stability.csv"),
        "seed_stability_per_map": (stability_per_map, output_dir / "seed_stability_per_map.csv"),
        "failure_status": (status, output_dir / "failure_status.csv"),
    }
    for frame, path in csv_outputs.values():
        _round_for_csv(frame).to_csv(path, index=False)
    _write_json(output_dir / "validation_audit.json", audit)
    _write_json(output_dir / "generation_usage.json", usage)

    real = overall[overall["planner_key"].eq(REAL_KEY)].iloc[0]
    real_pairs = paired[paired["candidate_planner_key"].eq(REAL_KEY)].set_index("baseline")
    real_stability = stability[stability["planner_key"].eq(REAL_KEY)].iloc[0]
    summary = {
        "analysis_id": output_dir.name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": "uav2d-v1 Validation only; Test was not read",
        "confidence": "ready_to_share_with_experimental_caveats",
        "contract": audit,
        "deepseek_v4pro_strict": {
            "runs": int(real["runs"]),
            "maps": int(real["maps"]),
            "feasible_runs": int(real["feasible_runs"]),
            "feasible_rate": float(real["feasible_rate"]),
            "maps_solved": int(real["maps_solved"]),
            "map_solved_rate": float(real["map_solved_rate"]),
            "maps_all_five_seeds_feasible": int(real["maps_all_repetitions_feasible"]),
            "median_feasible_cost": float(real["median_feasible_cost"]),
            "cost_q1": float(real["cost_q1"]),
            "cost_q3": float(real["cost_q3"]),
            "median_elapsed_seconds": float(real["median_elapsed_seconds"]),
            "median_objective_evaluations": float(real["median_objective_evaluations"]),
            "max_objective_evaluations": int(real["max_objective_evaluations"]),
            "median_unique_paths_per_map": float(real_stability["median_unique_feasible_paths"]),
            "median_within_map_cost_iqr": float(real_stability["median_within_map_cost_iqr"]),
            "p90_within_map_cost_iqr": float(real_stability["p90_within_map_cost_iqr"]),
            "research_claim_eligible": bool(real["research_claim_eligible"]),
        },
        "paired_map_median_vs_astar": real_pairs.loc["astar"].to_dict(),
        "paired_map_median_vs_theta_star": real_pairs.loc["theta_star"].to_dict(),
        "generation_usage_all_attempts": usage,
        "frozen_artifact_generation": {
            **frozen_usage,
            "estimated_cost_usd": frozen_cost,
            "human_review": frozen_artifact_manifest.get("human_review")
            or frozen_artifact_metadata.get("human_review"),
        },
        "methodology": {
            "ranking": "feasible rate first, then median trusted feasible cost",
            "paired_deterministic_baselines": "one median DeepSeek cost per map versus the deterministic baseline result on that map",
            "paired_stochastic_baselines": "exact map and shared seed",
            "uncertainty": f"cluster bootstrap over map_id, {BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}",
        },
        "inputs": {
            str((path / "benchmark_runs.csv").relative_to(ROOT)).replace("\\", "/"): _sha256(path / "benchmark_runs.csv")
            for path in source_dirs
        },
    }
    _write_json(output_dir / "analysis_summary.json", summary)
    receipt = {
        "status": "passed",
        "output_dir": str(output_dir),
        "records": int(len(runs)),
        "maps": int(runs["map_id"].nunique()),
        "planners_or_arms": int(runs["planner_key"].nunique()),
        "files": {
            path.name: _sha256(path)
            for path in output_dir.iterdir()
            if path.is_file()
        },
    }
    _write_json(output_dir / "analysis_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traditional-dir", type=Path, default=DEFAULT_TRADITIONAL)
    parser.add_argument("--offline-dir", type=Path, default=DEFAULT_OFFLINE_AFL)
    parser.add_argument("--deepseek-dir", type=Path, default=DEFAULT_DEEPSEEK)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = analyse(
        args.traditional_dir.resolve(),
        args.offline_dir.resolve(),
        args.deepseek_dir.resolve(),
        args.candidate_root.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
