"""Validate and analyse the API-free UAV2D-v1 Validation benchmark.

This script merges the traditional-planner run with the previously frozen
offline AFL-UAV v3 run, checks the 2,580-record experiment contract, computes
cluster-aware uncertainty and paired comparisons, and writes the bounded data
artifact used by the Data Analytics report renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRADITIONAL = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "offline-traditional-validation-v1"
)
DEFAULT_AFL = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "afl-uav-offline-v3-validation-v1"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "planning_benchmarks"
    / "offline-validation-analysis-v1"
)

PLANNER_LABELS = {
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
    "afl_uav": "AFL-UAV (offline v3)",
}
DETERMINISTIC = {"dijkstra", "astar", "theta_star"}
STOCHASTIC = {
    "rrt",
    "rrt_star",
    "prm",
    "ga",
    "pso",
    "de",
    "aco_acor",
    "afl_uav",
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
    "afl_uav": 300,
}
EXPECTED_DIFFICULTIES = {
    "sparse",
    "dense",
    "corridor",
    "clustered",
    "rooms_maze",
    "mixed",
}
BOOTSTRAP_SEED = 20260731
BOOTSTRAP_REPLICATES = 2_000


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .astype(bool)
    )


def _normalise_run(path: Path, source_run: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "arm_id" not in frame.columns:
        frame["arm_id"] = np.where(
            frame["planner"].eq("afl_uav"), "offline_v3", frame["planner"]
        )
    if "execution_arm" not in frame.columns:
        frame["execution_arm"] = np.where(
            frame["planner"].eq("afl_uav"),
            "afl_uav:offline_v3",
            frame["planner"],
        )
    frame["source_run"] = source_run
    frame["feasible"] = _as_bool(frame["feasible"])
    frame["research_claim_eligible"] = _as_bool(
        frame["research_claim_eligible"]
    )
    numeric_columns = [
        "repetition",
        "seed",
        "total_cost",
        "path_length",
        "collision_penalty",
        "smoothness_penalty",
        "risk_penalty",
        "waypoint_penalty",
        "minimum_clearance",
        "elapsed_seconds",
        "objective_evaluations",
        "collision_checks",
        "node_expansions",
        "waypoint_count",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["planner_label"] = frame["planner"].map(PLANNER_LABELS)
    frame["planner_key"] = np.where(
        frame["planner"].eq("afl_uav"),
        frame["planner"] + ":" + frame["arm_id"].astype(str),
        frame["planner"],
    )
    return frame


def _assert_contract(
    runs: pd.DataFrame,
    traditional_metadata: dict[str, Any],
    afl_metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    expected_total = sum(EXPECTED_COUNTS.values())
    if len(runs) != expected_total:
        errors.append(f"expected {expected_total} records, found {len(runs)}")

    actual_counts = runs.groupby("planner", sort=False).size().to_dict()
    if actual_counts != EXPECTED_COUNTS:
        errors.append(
            f"planner record counts differ: expected={EXPECTED_COUNTS}, "
            f"actual={actual_counts}"
        )

    key_columns = ["planner", "arm_id", "map_id", "seed"]
    duplicate_count = int(runs.duplicated(key_columns).sum())
    if duplicate_count:
        errors.append(f"{duplicate_count} duplicate record keys")

    if set(runs["split"].dropna().unique()) != {"validation"}:
        errors.append("records outside the Validation split were found")
    if set(runs["difficulty"].dropna().unique()) != EXPECTED_DIFFICULTIES:
        errors.append("the six expected map classes are not all present")
    if runs["map_id"].nunique() != 60:
        errors.append(f"expected 60 unique maps, found {runs['map_id'].nunique()}")
    per_class_maps = (
        runs[["difficulty", "map_id"]]
        .drop_duplicates()
        .groupby("difficulty")
        .size()
        .to_dict()
    )
    if any(count != 10 for count in per_class_maps.values()):
        errors.append(f"map classes are not balanced at 10 maps: {per_class_maps}")

    manifest_hashes = {
        traditional_metadata.get("manifest_hash"),
        afl_metadata.get("manifest_hash"),
    }
    config_hashes = {
        traditional_metadata.get("config_hash"),
        afl_metadata.get("config_hash"),
    }
    budgets = {
        json.dumps(
            traditional_metadata.get("budget", {}), sort_keys=True
        ),
        json.dumps(afl_metadata.get("budget", {}), sort_keys=True),
    }
    if len(manifest_hashes) != 1:
        errors.append("input runs use different dataset manifest hashes")
    if len(config_hashes) != 1:
        errors.append("input runs use different benchmark config hashes")
    if len(budgets) != 1:
        errors.append("input runs use different planning budgets")

    if (runs["objective_evaluations"] < 0).any():
        errors.append("negative objective-evaluation count found")
    max_evaluations = int(
        traditional_metadata.get("budget", {}).get(
            "max_objective_evaluations", 2_000
        )
    )
    if (runs["objective_evaluations"] > max_evaluations).any():
        errors.append("objective-evaluation budget was exceeded")
    for column in ["collision_checks", "node_expansions", "elapsed_seconds"]:
        if (runs[column] < 0).any():
            errors.append(f"negative {column} found")

    feasible_without_cost = int(
        (runs["feasible"] & runs["total_cost"].isna()).sum()
    )
    infeasible_with_cost = int(
        ((~runs["feasible"]) & runs["total_cost"].notna()).sum()
    )
    if feasible_without_cost:
        errors.append(f"{feasible_without_cost} feasible records lack cost")
    if infeasible_with_cost:
        errors.append(
            f"{infeasible_with_cost} infeasible records contain trusted cost"
        )

    afl = runs[runs["planner"].eq("afl_uav")]
    if afl["research_claim_eligible"].any():
        errors.append("offline AFL-UAV was incorrectly marked research eligible")

    for planner in DETERMINISTIC:
        subset = runs[runs["planner"].eq(planner)]
        if subset["map_id"].nunique() != 60 or len(subset) != 60:
            errors.append(f"{planner} is not one-run-per-map")
    for planner in STOCHASTIC:
        counts = (
            runs[runs["planner"].eq(planner)]
            .groupby("map_id")
            .size()
        )
        if len(counts) != 60 or not counts.eq(5).all():
            errors.append(f"{planner} is not five-runs-per-map")

    stochastic_seed_sets: dict[str, dict[str, tuple[int, ...]]] = {}
    for planner in sorted(STOCHASTIC):
        stochastic_seed_sets[planner] = {
            map_id: tuple(sorted(group["seed"].astype("int64").tolist()))
            for map_id, group in runs[runs["planner"].eq(planner)].groupby(
                "map_id"
            )
        }
    reference = stochastic_seed_sets["rrt"]
    for planner, seed_sets in stochastic_seed_sets.items():
        if seed_sets != reference:
            errors.append(f"{planner} does not use the shared stochastic seeds")

    audit = {
        "status": "passed" if not errors else "failed",
        "expected_records": expected_total,
        "actual_records": int(len(runs)),
        "unique_record_keys": int(
            runs[key_columns].drop_duplicates().shape[0]
        ),
        "unique_maps": int(runs["map_id"].nunique()),
        "planner_record_counts": actual_counts,
        "maps_per_class": per_class_maps,
        "manifest_hash": next(iter(manifest_hashes)),
        "config_hash": next(iter(config_hashes)),
        "budget": traditional_metadata.get("budget"),
        "shared_stochastic_seed_sets": not any(
            "shared stochastic seeds" in error for error in errors
        ),
        "research_claim_eligible": False,
        "errors": errors,
    }
    if errors:
        raise ValueError("Validation contract failed:\n- " + "\n- ".join(errors))
    return audit


def _iqr(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return (np.nan, np.nan, np.nan)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return float(q1), float(median), float(q3)


def _cluster_bootstrap(
    group: pd.DataFrame,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    per_map = [
        (
            map_group["feasible"].to_numpy(dtype=float),
            map_group.loc[map_group["feasible"], "total_cost"]
            .dropna()
            .to_numpy(dtype=float),
        )
        for _, map_group in group.groupby("map_id", sort=True)
    ]
    map_count = len(per_map)
    rng = np.random.default_rng(seed)
    feasible_rates = np.empty(replicates, dtype=float)
    median_costs = np.full(replicates, np.nan, dtype=float)
    for index in range(replicates):
        sampled = rng.integers(0, map_count, size=map_count)
        feasible = np.concatenate([per_map[item][0] for item in sampled])
        feasible_rates[index] = feasible.mean()
        costs = [
            per_map[item][1]
            for item in sampled
            if len(per_map[item][1])
        ]
        if costs:
            median_costs[index] = np.median(np.concatenate(costs))
    rate_low, rate_high = np.quantile(feasible_rates, [0.025, 0.975])
    finite_costs = median_costs[np.isfinite(median_costs)]
    if len(finite_costs):
        cost_low, cost_high = np.quantile(
            finite_costs, [0.025, 0.975]
        )
    else:
        cost_low = cost_high = np.nan
    return {
        "feasible_rate_ci_low": float(rate_low),
        "feasible_rate_ci_high": float(rate_high),
        "median_cost_ci_low": float(cost_low),
        "median_cost_ci_high": float(cost_high),
    }


def _summary_row(
    group: pd.DataFrame,
    include_bootstrap: bool,
    bootstrap_seed: int,
) -> dict[str, Any]:
    feasible = group[group["feasible"]]
    cost_q1, cost_median, cost_q3 = _iqr(feasible["total_cost"])
    time_q1, time_median, time_q3 = _iqr(group["elapsed_seconds"])
    per_map_solved = group.groupby("map_id")["feasible"].any()
    per_map_robust = group.groupby("map_id")["feasible"].all()
    row: dict[str, Any] = {
        "planner": group["planner"].iloc[0],
        "planner_key": group["planner_key"].iloc[0],
        "planner_label": group["planner_label"].iloc[0],
        "arm_id": group["arm_id"].iloc[0],
        "runs": int(len(group)),
        "maps": int(group["map_id"].nunique()),
        "feasible_runs": int(group["feasible"].sum()),
        "feasible_rate": float(group["feasible"].mean()),
        "maps_solved": int(per_map_solved.sum()),
        "map_solved_rate": float(per_map_solved.mean()),
        "maps_all_repetitions_feasible": int(per_map_robust.sum()),
        "map_all_repetitions_feasible_rate": float(per_map_robust.mean()),
        "median_feasible_cost": cost_median,
        "cost_q1": cost_q1,
        "cost_q3": cost_q3,
        "cost_iqr": cost_q3 - cost_q1,
        "median_elapsed_seconds": time_median,
        "elapsed_q1": time_q1,
        "elapsed_q3": time_q3,
        "median_objective_evaluations": float(
            group["objective_evaluations"].median()
        ),
        "max_objective_evaluations": int(
            group["objective_evaluations"].max()
        ),
        "median_collision_checks": float(group["collision_checks"].median()),
        "median_node_expansions": float(group["node_expansions"].median()),
        "timeout_rate": float(group["status"].eq("timeout").mean()),
        "budget_exhausted_rate": float(
            group["status"].eq("budget_exhausted").mean()
        ),
        "research_claim_eligible": bool(
            group["research_claim_eligible"].all()
        ),
    }
    if include_bootstrap:
        row.update(
            _cluster_bootstrap(
                group, seed=bootstrap_seed, replicates=BOOTSTRAP_REPLICATES
            )
        )
    return row


def _overall_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for offset, (_, group) in enumerate(
        runs.groupby("planner_key", sort=True)
    ):
        rows.append(
            _summary_row(
                group,
                include_bootstrap=True,
                bootstrap_seed=BOOTSTRAP_SEED + offset * 997,
            )
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        ["feasible_rate", "median_feasible_cost", "planner_label"],
        ascending=[False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    frame.insert(0, "rank", np.arange(1, len(frame) + 1))
    return frame


def _by_difficulty_statistics(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (difficulty, _), group in runs.groupby(
        ["difficulty", "planner_key"], sort=True
    ):
        row = _summary_row(
            group, include_bootstrap=False, bootstrap_seed=BOOTSTRAP_SEED
        )
        row["difficulty"] = difficulty
        rows.append(row)
    frame = pd.DataFrame(rows)
    return frame.sort_values(
        ["difficulty", "feasible_rate", "median_feasible_cost"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _per_map_results(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (planner_key, map_id), group in runs.groupby(
        ["planner_key", "map_id"], sort=True
    ):
        feasible_costs = group.loc[group["feasible"], "total_cost"].dropna()
        rows.append(
            {
                "planner": group["planner"].iloc[0],
                "planner_key": planner_key,
                "planner_label": group["planner_label"].iloc[0],
                "map_id": map_id,
                "difficulty": group["difficulty"].iloc[0],
                "map_feasible": bool(group["feasible"].any()),
                "feasible_repetitions": int(group["feasible"].sum()),
                "repetitions": int(len(group)),
                "median_feasible_cost": (
                    float(feasible_costs.median())
                    if len(feasible_costs)
                    else np.nan
                ),
                "median_elapsed_seconds": float(
                    group["elapsed_seconds"].median()
                ),
            }
        )
    return pd.DataFrame(rows)


def _paired_comparisons(
    per_map: pd.DataFrame, baseline_planners: tuple[str, ...] = ("astar", "theta_star")
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    tolerance = 1e-9
    planner_keys = sorted(per_map["planner_key"].unique())
    rng = np.random.default_rng(BOOTSTRAP_SEED + 31_337)
    for baseline in baseline_planners:
        base = (
            per_map[per_map["planner_key"].eq(baseline)]
            .set_index("map_id")
            .sort_index()
        )
        for candidate_key in planner_keys:
            if candidate_key == baseline:
                continue
            candidate = (
                per_map[per_map["planner_key"].eq(candidate_key)]
                .set_index("map_id")
                .sort_index()
            )
            joined = candidate.join(
                base[
                    [
                        "map_feasible",
                        "median_feasible_cost",
                    ]
                ],
                how="inner",
                lsuffix="_candidate",
                rsuffix="_baseline",
            )
            both = joined[
                joined["map_feasible_candidate"]
                & joined["map_feasible_baseline"]
            ].copy()
            deltas = (
                both["median_feasible_cost_candidate"]
                - both["median_feasible_cost_baseline"]
            )
            outcomes = np.where(
                deltas < -tolerance,
                1.0,
                np.where(np.abs(deltas) <= tolerance, 0.5, 0.0),
            )
            strict_wins = int((deltas < -tolerance).sum())
            ties = int((np.abs(deltas) <= tolerance).sum())
            losses = int((deltas > tolerance).sum())
            strict_win_rate = (
                strict_wins / len(both) if len(both) else np.nan
            )
            half_tie_win_rate = (
                float(outcomes.mean()) if len(outcomes) else np.nan
            )
            if len(outcomes):
                boot = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
                for index in range(BOOTSTRAP_REPLICATES):
                    sampled = rng.integers(
                        0, len(outcomes), size=len(outcomes)
                    )
                    boot[index] = outcomes[sampled].mean()
                ci_low, ci_high = np.quantile(boot, [0.025, 0.975])
            else:
                ci_low = ci_high = np.nan
            candidate_only = int(
                (
                    joined["map_feasible_candidate"]
                    & ~joined["map_feasible_baseline"]
                ).sum()
            )
            baseline_only = int(
                (
                    ~joined["map_feasible_candidate"]
                    & joined["map_feasible_baseline"]
                ).sum()
            )
            rows.append(
                {
                    "candidate_planner_key": candidate_key,
                    "planner": candidate["planner"].iloc[0],
                    "planner_label": candidate["planner_label"].iloc[0],
                    "baseline": baseline,
                    "baseline_label": PLANNER_LABELS[baseline],
                    "maps_compared": int(len(joined)),
                    "both_feasible_maps": int(len(both)),
                    "candidate_only_feasible_maps": candidate_only,
                    "baseline_only_feasible_maps": baseline_only,
                    "wins": strict_wins,
                    "ties": ties,
                    "losses": losses,
                    "strict_win_rate": strict_win_rate,
                    "half_tie_win_rate": half_tie_win_rate,
                    "half_tie_win_rate_ci_low": float(ci_low),
                    "half_tie_win_rate_ci_high": float(ci_high),
                    "median_cost_delta": (
                        float(deltas.median()) if len(deltas) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["baseline", "half_tie_win_rate", "planner_label"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _seed_stability(runs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    stochastic = runs[runs["planner"].isin(STOCHASTIC)]
    per_map_rows: list[dict[str, Any]] = []
    for (planner_key, map_id), group in stochastic.groupby(
        ["planner_key", "map_id"], sort=True
    ):
        costs = group.loc[group["feasible"], "total_cost"].dropna()
        q1, median, q3 = _iqr(costs)
        hashes = group.loc[
            group["feasible"] & group["path_hash"].notna(), "path_hash"
        ]
        per_map_rows.append(
            {
                "planner": group["planner"].iloc[0],
                "planner_key": planner_key,
                "planner_label": group["planner_label"].iloc[0],
                "map_id": map_id,
                "difficulty": group["difficulty"].iloc[0],
                "repetitions": int(len(group)),
                "feasible_repetitions": int(group["feasible"].sum()),
                "within_map_feasible_rate": float(group["feasible"].mean()),
                "unique_feasible_paths": int(hashes.nunique()),
                "median_feasible_cost": median,
                "within_map_cost_q1": q1,
                "within_map_cost_q3": q3,
                "within_map_cost_iqr": q3 - q1,
            }
        )
    per_map = pd.DataFrame(per_map_rows)
    summary_rows: list[dict[str, Any]] = []
    for _, group in per_map.groupby("planner_key", sort=True):
        summary_rows.append(
            {
                "planner": group["planner"].iloc[0],
                "planner_key": group["planner_key"].iloc[0],
                "planner_label": group["planner_label"].iloc[0],
                "maps": int(len(group)),
                "mean_within_map_feasible_rate": float(
                    group["within_map_feasible_rate"].mean()
                ),
                "fully_feasible_maps": int(
                    group["within_map_feasible_rate"].eq(1.0).sum()
                ),
                "never_feasible_maps": int(
                    group["within_map_feasible_rate"].eq(0.0).sum()
                ),
                "median_unique_feasible_paths": float(
                    group["unique_feasible_paths"].median()
                ),
                "median_within_map_cost_iqr": float(
                    group["within_map_cost_iqr"].median()
                ),
                "p90_within_map_cost_iqr": float(
                    group["within_map_cost_iqr"].quantile(0.9)
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(
        ["mean_within_map_feasible_rate", "median_within_map_cost_iqr"],
        ascending=[False, True],
    )
    return summary.reset_index(drop=True), per_map


def _failure_status(runs: pd.DataFrame) -> pd.DataFrame:
    frame = (
        runs.groupby(["planner_key", "planner_label", "status"], dropna=False)
        .size()
        .rename("records")
        .reset_index()
    )
    totals = frame.groupby("planner_key")["records"].transform("sum")
    frame["record_share"] = frame["records"] / totals
    return frame.sort_values(
        ["planner_label", "records"], ascending=[True, False]
    ).reset_index(drop=True)


def _make_artifact(
    output_dir: Path,
    audit: dict[str, Any],
    overall: pd.DataFrame,
    by_difficulty: pd.DataFrame,
    paired: pd.DataFrame,
    stability: pd.DataFrame,
    status: pd.DataFrame,
) -> dict[str, Any]:
    generated_at = datetime.now(UTC).isoformat()
    best = overall.iloc[0]
    lowest_cost = overall.dropna(subset=["median_feasible_cost"]).sort_values(
        "median_feasible_cost"
    ).iloc[0]
    afl = overall[overall["planner"].eq("afl_uav")].iloc[0]
    afl_pairs = paired[paired["planner"].eq("afl_uav")].set_index("baseline")
    astar_pair = afl_pairs.loc["astar"]
    theta_pair = afl_pairs.loc["theta_star"]
    rooms = by_difficulty[by_difficulty["difficulty"].eq("rooms_maze")]
    rooms_afl = rooms[rooms["planner"].eq("afl_uav")].iloc[0]
    rooms_traditional = rooms[~rooms["planner"].eq("afl_uav")]
    rooms_best_traditional = rooms_traditional.sort_values(
        ["feasible_rate", "median_feasible_cost"],
        ascending=[False, True],
    ).iloc[0]

    overall_rows = overall[
        [
            "rank",
            "planner",
            "planner_label",
            "runs",
            "maps",
            "feasible_runs",
            "feasible_rate",
            "feasible_rate_ci_low",
            "feasible_rate_ci_high",
            "maps_solved",
            "map_solved_rate",
            "maps_all_repetitions_feasible",
            "map_all_repetitions_feasible_rate",
            "median_feasible_cost",
            "cost_q1",
            "cost_q3",
            "median_cost_ci_low",
            "median_cost_ci_high",
            "median_elapsed_seconds",
            "elapsed_q1",
            "elapsed_q3",
            "median_objective_evaluations",
            "median_collision_checks",
            "median_node_expansions",
            "timeout_rate",
            "budget_exhausted_rate",
            "research_claim_eligible",
        ]
    ].to_dict(orient="records")
    paired_rows = paired.to_dict(orient="records")
    difficulty_rows = by_difficulty[
        [
            "difficulty",
            "planner",
            "planner_label",
            "runs",
            "maps",
            "feasible_runs",
            "feasible_rate",
            "map_solved_rate",
            "median_feasible_cost",
            "cost_q1",
            "cost_q3",
            "median_elapsed_seconds",
        ]
    ].to_dict(orient="records")
    stability_rows = stability.to_dict(orient="records")
    status_rows = status.to_dict(orient="records")

    source_filters = [
        "benchmark_id = uav2d-v1",
        "split = validation",
        "60 fixed maps; 10 maps in each of six classes",
        "time limit = 1 second; objective-evaluation limit = 2,000",
        "deterministic planners run once per map",
        "stochastic planners and offline AFL-UAV run five shared seeds per map",
    ]
    metric_definitions = [
        (
            "Feasible rate = trusted hard-validator feasible runs / "
            "all runs for the planner."
        ),
        (
            "Map solved rate = maps with at least one feasible run / "
            "60 Validation maps."
        ),
        (
            "Median feasible cost uses only paths accepted by the "
            "shared PathEvaluator and hard validator."
        ),
        (
            "Uncertainty intervals are 95% percentile intervals from "
            "2,000 map-cluster bootstrap replicates."
        ),
        (
            "Paired win rate uses per-map median feasible cost; ties "
            "contribute one half to the displayed half-tie win rate."
        ),
    ]

    def file_source(
        source_id: str,
        label: str,
        filename: str,
        description: str,
    ) -> dict[str, Any]:
        relative_path = (
            "artifacts/planning_benchmarks/offline-validation-analysis-v1/"
            f"{filename}"
        )
        return {
            "id": source_id,
            "label": label,
            "path": relative_path,
            "query": {
                "engine": "DuckDB",
                "language": "sql",
                "sql": (
                    f"SELECT * FROM read_csv_auto('{relative_path}', "
                    "header = true);"
                ),
                "description": description,
                "executed_at": generated_at,
                "tables_used": [relative_path],
                "filters": source_filters,
                "metric_definitions": metric_definitions,
            },
        }

    sources = [
        file_source(
            "raw_runs",
            "Merged offline Validation run records",
            "offline_validation_runs.csv",
            (
                "Read the audited 2,580-record merged Validation result table "
                "used as the analysis population."
            ),
        ),
        file_source(
            "overall_stats",
            "Overall planner statistics",
            "overall_statistics.csv",
            (
                "Read the reviewed planner-level feasibility, cost, runtime, "
                "and map-cluster bootstrap statistics."
            ),
        ),
        file_source(
            "paired_stats",
            "Paired planner comparisons",
            "paired_comparisons.csv",
            (
                "Read the reviewed per-map paired comparisons against A* and "
                "Theta*."
            ),
        ),
        file_source(
            "difficulty_stats",
            "Planner statistics by map class",
            "by_difficulty_statistics.csv",
            "Read the reviewed planner statistics for each of six map classes.",
        ),
        file_source(
            "stability_stats",
            "Shared-seed stability statistics",
            "seed_stability.csv",
            "Read the reviewed five-seed stability statistics by planner.",
        ),
        file_source(
            "failure_stats",
            "Planner termination-status counts",
            "failure_status.csv",
            "Read the reviewed termination-status counts by planner.",
        ),
    ]

    summary_body = (
        "## 技术摘要\n\n"
        f"本次离线 Validation 已完整生成 **{audit['actual_records']:,}/"
        f"{audit['expected_records']:,}** 条唯一记录，覆盖 60 张固定地图、"
        "6 类场景和 11 个执行臂。按“先可行率、后可行成本”的预注册规则，"
        f"当前第一名是 **{best['planner_label']}**（可行率 "
        f"{best['feasible_rate']:.1%}，可行路径成本中位数 "
        f"{best['median_feasible_cost']:.3f}）。\n\n"
        f"离线 AFL-UAV v3 的运行可行率为 **{afl['feasible_rate']:.1%}**，"
        f"至少解出一条路径的地图占 **{afl['map_solved_rate']:.1%}**；"
        f"与 A*、Theta* 在共同可行地图上的半计平局成本胜率分别为 "
        f"**{astar_pair['half_tie_win_rate']:.1%}** 和 "
        f"**{theta_pair['half_tie_win_rate']:.1%}**。这些结果只验证离线 "
        "Agent→求解器→统一评测链路，不构成真实 LLM 方法效果证据。"
    )
    feasibility_body = (
        "## 关键发现：可行性\n\n"
        "图中同时给出逐次运行可行率和地图解出率。随机算法的两者含义不同："
        "前者衡量跨种子稳健性，后者只要求五个共享种子中至少一次成功。"
        "误差区间以地图为重采样单位，因此不会把同一地图的五次重复误当成"
        "独立样本。"
    )
    cost_body = (
        "## 关键发现：统一成本\n\n"
        f"在所有可信可行路径中，成本中位数最低的是 "
        f"**{lowest_cost['planner_label']}**（"
        f"{lowest_cost['median_feasible_cost']:.3f}）。该比较是条件性的："
        "失败与超时没有被替换为惩罚成本，所以必须与上面的可行率一起阅读。"
    )
    paired_body = (
        "## 关键发现：成对比较\n\n"
        "为控制地图难度，先在每张地图内对随机算法的可行成本取中位数，再与"
        "同图 A* 或 Theta* 比较。显示值为“胜=1、平=0.5、负=0”的平均值；"
        "只有双方都可行的地图进入成本胜率，单方可行数量保留在明细表中。"
    )
    class_body = (
        "## 六类地图表现\n\n"
        "下表保留每类地图的运行数、可行率、地图解出率、条件成本和运行时间，"
        "用于定位 corridor、rooms/maze 等结构性失败。每类固定 10 张地图；"
        "随机方法每类 50 次运行，确定性方法每类 10 次。"
        f"当前最有区分度的是 **rooms/maze**：离线 AFL-UAV 的运行可行率为 "
        f"**{rooms_afl['feasible_rate']:.1%}**，传统方法中最高的是 "
        f"**{rooms_best_traditional['planner_label']}**（"
        f"**{rooms_best_traditional['feasible_rate']:.1%}**）。"
    )
    stability_body = (
        "## 随机种子稳定性\n\n"
        "五个规划种子在所有随机执行臂间共享。完全可行地图数越高，说明结果"
        "越不依赖碰巧抽到的种子；单图成本 IQR 越低，说明成功路径质量越稳定。"
    )
    failure_body = (
        "## 终止状态与失败模式\n\n"
        "终止状态与可信可行性是两个维度：达到 1 秒后返回的 best-so-far "
        "仍可能通过硬约束验证，因此 `timeout` 不会自动计为不可行；反之，"
        "任何未通过统一验证的返回都不进入成本统计。下表用于区分搜索超时与"
        "明确的 `no_path`。"
    )
    methods_body = (
        "## 范围、数据与方法\n\n"
        "范围限定为二维、静态、已知障碍物的 UAV2D-v1 Validation；未读取 "
        "Test。传统确定性算法每图一次，传统随机算法与离线 AFL-UAV 每图五次，"
        "共用地图、起终点、随机种子、连续碰撞检测、硬约束验证器、目标函数、"
        "1 秒时间预算和 2,000 次可信目标评价预算。最终排名先比较运行可行率，"
        "再比较可信可行路径的统一总成本中位数。95% 区间使用 2,000 次地图"
        "聚类 bootstrap，保留同一地图内重复运行的相关性。"
    )
    limitations_body = (
        "## 局限性与不确定性\n\n"
        "Validation 已参与方法开发，不能替代隐藏终测集；此前工程 Pilot "
        "打开过原 Test，因此最终论文应另建未触碰的隐藏测试集。离线 AFL-UAV "
        "由 mock provider 生成并标记 `research_claim_eligible=false`，只能"
        "验证架构与实验协议。1 秒墙钟预算会受到本机调度影响；bootstrap "
        "区间反映 60 张地图的抽样不确定性，不包含地图生成分布或硬件变化。"
    )
    recommendations_body = (
        "## 建议的下一步\n\n"
        "先冻结本次离线结果和分析脚本，基于失败状态与六类地图分层定位可行性"
        "短板；随后在不访问 Test 的前提下完善真实 Provider 候选生成、人工"
        "哈希批准和六图资格门禁。真实模型接入后沿用相同 Validation 协议，"
        "最后为论文重新生成独立隐藏终测集，再一次性比较 AFL-UAV 与 "
        "Evolutionary AFL-UAV。"
    )
    questions_body = (
        "## 进一步研究问题\n\n"
        "离线 Agent 生成的求解器究竟在哪些地图结构上改变了成功概率？"
        "在相同 LLM 调用与 token 预算下，进化修订带来的收益来自更高可行率，"
        "还是来自共同可行地图上的成本下降？这种收益能否跨模型、跨隐藏地图"
        "种子复现？"
    )

    manifest = {
        "version": 1,
        "surface": "report",
        "title": "UAV2D-v1 离线路径规划 Validation 报告",
        "description": (
            "API-free traditional planners and offline AFL-UAV v3 on the "
            "fixed UAV2D-v1 Validation split."
        ),
        "generatedAt": generated_at,
        "sources": sources,
        "charts": [
            {
                "id": "feasibility_chart",
                "title": "各规划器的 Validation 运行可行率",
                "subtitle": "误差范围为按地图聚类 bootstrap 的 95% 区间；明细表保留地图解出率。",
                "type": "bar",
                "dataset": "overall",
                "sourceId": "overall_stats",
                "encodings": {
                    "x": {
                        "field": "planner_label",
                        "type": "nominal",
                        "label": "规划器",
                    },
                    "y": {
                        "field": "feasible_rate",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "运行可行率",
                    },
                    "tooltip": [
                        {"field": "runs", "type": "quantitative", "label": "运行数"},
                        {
                            "field": "maps_solved",
                            "type": "quantitative",
                            "label": "解出地图数",
                        },
                        {
                            "field": "map_solved_rate",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "地图解出率",
                        },
                        {
                            "field": "feasible_rate_ci_low",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "95% CI 下界",
                        },
                        {
                            "field": "feasible_rate_ci_high",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "95% CI 上界",
                        },
                    ],
                },
                "settings": {"orientation": "vertical", "showValues": True},
                "layout": "full",
            },
            {
                "id": "cost_chart",
                "title": "可信可行路径的统一总成本中位数",
                "subtitle": "仅统计通过共享硬约束验证器的路径；失败和超时不转为惩罚成本。",
                "type": "bar",
                "dataset": "overall",
                "sourceId": "overall_stats",
                "encodings": {
                    "x": {
                        "field": "planner_label",
                        "type": "nominal",
                        "label": "规划器",
                    },
                    "y": {
                        "field": "median_feasible_cost",
                        "type": "quantitative",
                        "format": "number",
                        "label": "总成本中位数",
                    },
                    "tooltip": [
                        {
                            "field": "feasible_runs",
                            "type": "quantitative",
                            "label": "可行运行数",
                        },
                        {
                            "field": "cost_q1",
                            "type": "quantitative",
                            "format": "number",
                            "label": "成本 Q1",
                        },
                        {
                            "field": "cost_q3",
                            "type": "quantitative",
                            "format": "number",
                            "label": "成本 Q3",
                        },
                    ],
                },
                "settings": {"orientation": "vertical", "showValues": True},
                "layout": "full",
            },
            {
                "id": "paired_chart",
                "title": "相对 A* 与 Theta* 的成对成本胜率",
                "subtitle": "同图双方均可行时比较；胜计 1、平计 0.5、负计 0。",
                "type": "bar",
                "dataset": "paired",
                "sourceId": "paired_stats",
                "encodings": {
                    "x": {
                        "field": "planner_label",
                        "type": "nominal",
                        "label": "候选规划器",
                    },
                    "y": {
                        "field": "half_tie_win_rate",
                        "type": "quantitative",
                        "format": "percent",
                        "label": "半计平局胜率",
                    },
                    "color": {
                        "field": "baseline_label",
                        "type": "nominal",
                        "label": "基线",
                    },
                    "tooltip": [
                        {
                            "field": "both_feasible_maps",
                            "type": "quantitative",
                            "label": "双方可行地图",
                        },
                        {"field": "wins", "type": "quantitative", "label": "胜"},
                        {"field": "ties", "type": "quantitative", "label": "平"},
                        {"field": "losses", "type": "quantitative", "label": "负"},
                    ],
                },
                "settings": {
                    "orientation": "vertical",
                    "groupMode": "grouped",
                    "showValues": False,
                },
                "legend": {
                    "position": "bottom",
                    "sort": "spec",
                    "title": "比较基线",
                },
                "layout": "full",
            },
            {
                "id": "runtime_quality_chart",
                "title": "运行时间与可信可行路径成本",
                "subtitle": "每个点为一个规划器；成本仅对可行路径计算。",
                "type": "scatter",
                "dataset": "overall",
                "sourceId": "overall_stats",
                "encodings": {
                    "x": {
                        "field": "median_elapsed_seconds",
                        "type": "quantitative",
                        "format": "number",
                        "label": "运行时间中位数（秒）",
                    },
                    "y": {
                        "field": "median_feasible_cost",
                        "type": "quantitative",
                        "format": "number",
                        "label": "总成本中位数",
                    },
                    "label": {
                        "field": "planner_label",
                        "type": "text",
                        "label": "规划器",
                    },
                    "tooltip": [
                        {
                            "field": "feasible_rate",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "运行可行率",
                        },
                        {
                            "field": "runs",
                            "type": "quantitative",
                            "label": "运行数",
                        },
                    ],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "overall_table",
                "title": "总体规划器统计",
                "subtitle": "排名先按可行率降序，再按条件成本中位数升序。",
                "dataset": "overall",
                "defaultSort": {"field": "rank", "direction": "asc"},
                "density": "dense",
                "sourceId": "overall_stats",
                "layout": "full",
                "columns": [
                    {"field": "rank", "label": "排名", "format": "number"},
                    {"field": "planner_label", "label": "规划器", "type": "text"},
                    {"field": "runs", "label": "运行数", "format": "number"},
                    {
                        "field": "feasible_rate",
                        "label": "运行可行率",
                        "format": "percent",
                    },
                    {
                        "field": "map_solved_rate",
                        "label": "地图解出率",
                        "format": "percent",
                    },
                    {
                        "field": "median_feasible_cost",
                        "label": "成本中位数",
                        "format": "number",
                    },
                    {
                        "field": "cost_q1",
                        "label": "成本 Q1",
                        "format": "number",
                    },
                    {
                        "field": "cost_q3",
                        "label": "成本 Q3",
                        "format": "number",
                    },
                    {
                        "field": "median_elapsed_seconds",
                        "label": "时间中位数（秒）",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "difficulty_table",
                "title": "按地图类别分层的规划器统计",
                "subtitle": "六类地图各 10 张；随机执行臂每类 50 次运行。",
                "dataset": "by_difficulty",
                "defaultSort": {
                    "field": "feasible_rate",
                    "direction": "desc",
                },
                "density": "dense",
                "sourceId": "difficulty_stats",
                "layout": "full",
                "columns": [
                    {"field": "difficulty", "label": "地图类别", "type": "text"},
                    {"field": "planner_label", "label": "规划器", "type": "text"},
                    {"field": "runs", "label": "运行数", "format": "number"},
                    {
                        "field": "feasible_rate",
                        "label": "运行可行率",
                        "format": "percent",
                    },
                    {
                        "field": "map_solved_rate",
                        "label": "地图解出率",
                        "format": "percent",
                    },
                    {
                        "field": "median_feasible_cost",
                        "label": "成本中位数",
                        "format": "number",
                    },
                    {
                        "field": "median_elapsed_seconds",
                        "label": "时间中位数（秒）",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "stability_table",
                "title": "随机执行臂的跨种子稳定性",
                "subtitle": "每图五个共享种子；成本 IQR 仅在可行重复中计算。",
                "dataset": "stability",
                "defaultSort": {
                    "field": "mean_within_map_feasible_rate",
                    "direction": "desc",
                },
                "density": "dense",
                "sourceId": "stability_stats",
                "layout": "full",
                "columns": [
                    {"field": "planner_label", "label": "规划器", "type": "text"},
                    {
                        "field": "mean_within_map_feasible_rate",
                        "label": "平均单图可行率",
                        "format": "percent",
                    },
                    {
                        "field": "fully_feasible_maps",
                        "label": "五次全可行地图",
                        "format": "number",
                    },
                    {
                        "field": "never_feasible_maps",
                        "label": "五次全失败地图",
                        "format": "number",
                    },
                    {
                        "field": "median_unique_feasible_paths",
                        "label": "单图不同可行路径中位数",
                        "format": "number",
                    },
                    {
                        "field": "median_within_map_cost_iqr",
                        "label": "单图成本 IQR 中位数",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "failure_table",
                "title": "各规划器的终止状态",
                "subtitle": "终止状态独立于统一硬约束验证结果。",
                "dataset": "status",
                "defaultSort": {"field": "records", "direction": "desc"},
                "density": "dense",
                "sourceId": "failure_stats",
                "layout": "full",
                "columns": [
                    {"field": "planner_label", "label": "规划器", "type": "text"},
                    {"field": "status", "label": "终止状态", "type": "text"},
                    {"field": "records", "label": "记录数", "format": "number"},
                    {
                        "field": "record_share",
                        "label": "规划器内占比",
                        "format": "percent",
                    },
                ],
            },
        ],
        "blocks": [
            {
                "id": "title",
                "type": "markdown",
                "body": "# UAV2D-v1 离线路径规划 Validation 报告",
                "layout": "full",
            },
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": summary_body,
                "layout": "full",
            },
            {
                "id": "feasibility_narrative",
                "type": "markdown",
                "body": feasibility_body,
                "sourceId": "overall_stats",
                "layout": "full",
            },
            {
                "id": "feasibility_visual",
                "type": "chart",
                "chartId": "feasibility_chart",
                "layout": "full",
            },
            {
                "id": "overall_exact",
                "type": "table",
                "tableId": "overall_table",
                "layout": "full",
            },
            {
                "id": "cost_narrative",
                "type": "markdown",
                "body": cost_body,
                "sourceId": "overall_stats",
                "layout": "full",
            },
            {
                "id": "cost_visual",
                "type": "chart",
                "chartId": "cost_chart",
                "layout": "full",
            },
            {
                "id": "paired_narrative",
                "type": "markdown",
                "body": paired_body,
                "sourceId": "paired_stats",
                "layout": "full",
            },
            {
                "id": "paired_visual",
                "type": "chart",
                "chartId": "paired_chart",
                "layout": "full",
            },
            {
                "id": "runtime_quality_visual",
                "type": "chart",
                "chartId": "runtime_quality_chart",
                "layout": "full",
            },
            {
                "id": "class_narrative",
                "type": "markdown",
                "body": class_body,
                "sourceId": "difficulty_stats",
                "layout": "full",
            },
            {
                "id": "class_table",
                "type": "table",
                "tableId": "difficulty_table",
                "layout": "full",
            },
            {
                "id": "stability_narrative",
                "type": "markdown",
                "body": stability_body,
                "sourceId": "stability_stats",
                "layout": "full",
            },
            {
                "id": "stability_detail",
                "type": "table",
                "tableId": "stability_table",
                "layout": "full",
            },
            {
                "id": "failure_narrative",
                "type": "markdown",
                "body": failure_body,
                "sourceId": "failure_stats",
                "layout": "full",
            },
            {
                "id": "failure_detail",
                "type": "table",
                "tableId": "failure_table",
                "layout": "full",
            },
            {
                "id": "scope_methods",
                "type": "markdown",
                "body": methods_body,
                "sourceId": "raw_runs",
                "layout": "full",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": limitations_body,
                "layout": "full",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "body": recommendations_body,
                "layout": "full",
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": questions_body,
                "layout": "full",
            },
        ],
    }
    snapshot = {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "overall": overall_rows,
            "paired": paired_rows,
            "by_difficulty": difficulty_rows,
            "stability": stability_rows,
            "status": status_rows,
        },
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": snapshot,
        "sources": sources,
    }


def _round_for_csv(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.select_dtypes(include=["float"]).columns:
        output[column] = output[column].round(9)
    return output


def analyse(
    traditional_dir: Path,
    afl_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    traditional_csv = traditional_dir / "benchmark_runs.csv"
    afl_csv = afl_dir / "benchmark_runs.csv"
    traditional_metadata_path = traditional_dir / "benchmark_metadata.json"
    afl_metadata_path = afl_dir / "benchmark_metadata.json"
    for path in [
        traditional_csv,
        afl_csv,
        traditional_metadata_path,
        afl_metadata_path,
    ]:
        if not path.is_file():
            raise FileNotFoundError(path)

    traditional_metadata = _read_json(traditional_metadata_path)
    afl_metadata = _read_json(afl_metadata_path)
    traditional = _normalise_run(
        traditional_csv, traditional_metadata.get("run_id", traditional_dir.name)
    )
    afl = _normalise_run(afl_csv, afl_metadata.get("run_id", afl_dir.name))
    runs = pd.concat([traditional, afl], ignore_index=True, sort=False)
    runs = runs.sort_values(
        ["planner", "arm_id", "map_id", "seed"]
    ).reset_index(drop=True)

    audit = _assert_contract(runs, traditional_metadata, afl_metadata)
    overall = _overall_statistics(runs)
    by_difficulty = _by_difficulty_statistics(runs)
    per_map = _per_map_results(runs)
    paired = _paired_comparisons(per_map)
    stability, stability_per_map = _seed_stability(runs)
    failure_status = _failure_status(runs)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "runs": output_dir / "offline_validation_runs.csv",
        "overall": output_dir / "overall_statistics.csv",
        "by_difficulty": output_dir / "by_difficulty_statistics.csv",
        "per_map": output_dir / "per_map_statistics.csv",
        "paired": output_dir / "paired_comparisons.csv",
        "stability": output_dir / "seed_stability.csv",
        "stability_per_map": output_dir / "seed_stability_per_map.csv",
        "failure_status": output_dir / "failure_status.csv",
        "audit": output_dir / "validation_audit.json",
        "summary": output_dir / "analysis_summary.json",
        "artifact": output_dir / "report_artifact.json",
    }
    _round_for_csv(runs).to_csv(paths["runs"], index=False)
    _round_for_csv(overall).to_csv(paths["overall"], index=False)
    _round_for_csv(by_difficulty).to_csv(
        paths["by_difficulty"], index=False
    )
    _round_for_csv(per_map).to_csv(paths["per_map"], index=False)
    _round_for_csv(paired).to_csv(paths["paired"], index=False)
    _round_for_csv(stability).to_csv(paths["stability"], index=False)
    _round_for_csv(stability_per_map).to_csv(
        paths["stability_per_map"], index=False
    )
    _round_for_csv(failure_status).to_csv(
        paths["failure_status"], index=False
    )
    _write_json(paths["audit"], audit)

    best = overall.iloc[0]
    afl_overall = overall[overall["planner"].eq("afl_uav")].iloc[0]
    summary = {
        "analysis_id": output_dir.name,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "contract": audit,
        "bootstrap": {
            "unit": "map_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "interval": "percentile_95",
        },
        "ranking_rule": (
            "feasible_rate descending, then median trusted feasible cost "
            "ascending"
        ),
        "top_ranked_planner": {
            "planner": best["planner"],
            "planner_label": best["planner_label"],
            "feasible_rate": best["feasible_rate"],
            "median_feasible_cost": best["median_feasible_cost"],
        },
        "offline_afl_uav": {
            "arm_id": "offline_v3",
            "feasible_rate": afl_overall["feasible_rate"],
            "map_solved_rate": afl_overall["map_solved_rate"],
            "median_feasible_cost": afl_overall["median_feasible_cost"],
            "research_claim_eligible": False,
        },
        "input_files": {
            str(traditional_csv.relative_to(ROOT)).replace("\\", "/"): _sha256(
                traditional_csv
            ),
            str(afl_csv.relative_to(ROOT)).replace("\\", "/"): _sha256(afl_csv),
        },
    }
    _write_json(paths["summary"], summary)
    artifact = _make_artifact(
        output_dir,
        audit,
        overall,
        by_difficulty,
        paired,
        stability,
        failure_status,
    )
    _write_json(paths["artifact"], artifact)

    receipt = {
        "status": "passed",
        "output_dir": str(output_dir),
        "record_count": int(len(runs)),
        "planner_count": int(runs["planner_key"].nunique()),
        "map_count": int(runs["map_id"].nunique()),
        "files": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if path.is_file()
        },
    }
    _write_json(output_dir / "analysis_receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--traditional-dir",
        type=Path,
        default=DEFAULT_TRADITIONAL,
    )
    parser.add_argument("--afl-dir", type=Path, default=DEFAULT_AFL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    receipt = analyse(
        args.traditional_dir.resolve(),
        args.afl_dir.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
