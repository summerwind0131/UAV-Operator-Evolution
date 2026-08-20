"""Data-backed diagnostic figures with explicit insufficient-evidence states."""

from __future__ import annotations

from pathlib import Path
import textwrap
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _records(values: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for value in values:
        if hasattr(value, "model_dump"):
            rows.append(value.model_dump(mode="python"))
        elif isinstance(value, dict):
            rows.append(dict(value))
    return rows


def _feature(frame: pd.DataFrame, column: str, key: str, default: Any = None) -> pd.Series:
    if column not in frame:
        return pd.Series([default] * len(frame), index=frame.index)
    return frame[column].map(lambda value: value.get(key, default) if isinstance(value, dict) else default)


def _save_empty(path: Path, title: str, message: str = "insufficient evidence") -> Path:
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def plot_immediate_improvement(profiles: Iterable[Any], output: Path) -> Path:
    frame = pd.DataFrame(_records(profiles))
    if frame.empty or "operator_name" not in frame or "immediate_improvement_rate" not in frame:
        return _save_empty(output, "Immediate improvement rate")
    frame = frame.sort_values("immediate_improvement_rate", ascending=False)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(frame["operator_name"], frame["immediate_improvement_rate"], color="#4472C4")
    axis.set_ylabel("rate")
    axis.set_ylim(0, max(1.0, float(frame["immediate_improvement_rate"].max()) * 1.1))
    axis.tick_params(axis="x", rotation=35)
    axis.set_title("Immediate improvement by operator")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def plot_delayed_reward(profiles: Iterable[Any], output: Path) -> Path:
    frame = pd.DataFrame(_records(profiles))
    if frame.empty or "operator_name" not in frame or "average_delayed_reward" not in frame:
        return _save_empty(output, "Delayed reward")
    frame = frame.sort_values("average_delayed_reward", ascending=False)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    colors = np.where(frame["average_delayed_reward"] >= 0, "#70AD47", "#C0504D")
    axis.bar(frame["operator_name"], frame["average_delayed_reward"], color=colors)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("mean delayed reward")
    axis.tick_params(axis="x", rotation=35)
    axis.set_title("Delayed contribution by operator")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def _heatmap(
    pivot: pd.DataFrame,
    output: Path,
    title: str,
    colorbar_label: str = "mean immediate reward",
) -> Path:
    if pivot.empty:
        return _save_empty(output, title)
    figure, axis = plt.subplots(
        figsize=(max(8.0, 0.68 * len(pivot.columns)), max(5.0, 0.48 * len(pivot.index))),
        constrained_layout=True,
    )
    values = pivot.to_numpy(dtype=float)
    image = axis.imshow(values, aspect="auto", cmap="RdYlGn")
    x_labels = [textwrap.fill(str(value), width=18, break_long_words=True) for value in pivot.columns]
    y_labels = [textwrap.fill(str(value), width=24, break_long_words=True) for value in pivot.index]
    axis.set_xticks(range(len(pivot.columns)), labels=x_labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(pivot.index)), labels=y_labels, fontsize=8)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(output, dpi=140, bbox_inches="tight")
    plt.close(figure)
    return output


def plot_context_heatmap(traces: Iterable[Any], context: str, output: Path) -> Path:
    frame = pd.DataFrame(_records(traces))
    if frame.empty or "operator_name" not in frame or "immediate_reward" not in frame:
        return _save_empty(output, f"Operator performance by {context}")
    if context == "map_type":
        frame[context] = _feature(frame, "environment_features", "difficulty", "unknown")
    elif context == "search_phase":
        ratio = pd.to_numeric(_feature(frame, "search_features_before", "iteration_ratio", 0.0))
        frame[context] = pd.cut(ratio, [-0.01, 1 / 3, 2 / 3, 1.01], labels=["early", "middle", "late"])
    if context not in frame:
        return _save_empty(output, f"Operator performance by {context}")
    pivot = frame.pivot_table(
        index="operator_name", columns=context, values="immediate_reward", aggfunc="mean", observed=True
    )
    return _heatmap(pivot, output, f"Operator performance by {context.replace('_', ' ')}")


def plot_synergy_matrix(relations: Iterable[Any], output: Path) -> Path:
    frame = pd.DataFrame(_records(relations))
    aliases = {
        "first_operator": ["first_operator", "operator_i"],
        "second_operator": ["second_operator", "operator_j"],
        "reward_delta": ["reward_delta", "delta", "synergy_score"],
    }
    chosen: dict[str, str] = {}
    for target, candidates in aliases.items():
        selected = next((name for name in candidates if name in frame), "")
        if not selected:
            return _save_empty(output, "Operator synergy matrix")
        chosen[target] = selected
    pivot = frame.pivot_table(
        index=chosen["first_operator"],
        columns=chosen["second_operator"],
        values=chosen["reward_delta"],
        aggfunc="mean",
    ).fillna(0.0)
    return _heatmap(pivot, output, "Operator synergy matrix", "reward delta vs follow-up baseline")


def plot_generation_performance(rows: Iterable[Any], output: Path) -> Path:
    frame = pd.DataFrame(_records(rows))
    if frame.empty or not {"generation", "best_cost"}.issubset(frame.columns):
        return _save_empty(output, "Best performance by generation")
    if "phase" in frame.columns and (frame["phase"] == "train").any():
        # Test maps intentionally use a harder distribution and must not be
        # spliced into the training-generation trend.
        frame = frame[frame["phase"] == "train"]
    summary = frame.groupby("generation", as_index=False)["best_cost"].min()
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(summary["generation"], summary["best_cost"], marker="o", color="#4472C4")
    axis.set_xlabel("generation")
    axis.set_ylabel("best cost (lower is better)")
    axis.set_title("Best performance by generation")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def plot_paired_comparison(outcomes: Iterable[Any], output: Path) -> Path:
    frame = pd.DataFrame(_records(outcomes))
    required = {"parent_best_cost", "candidate_best_cost"}
    if frame.empty or not required.issubset(frame.columns):
        return _save_empty(output, "Parent/candidate paired comparison")
    figure, axis = plt.subplots(figsize=(5.5, 5.5))
    axis.scatter(frame["parent_best_cost"], frame["candidate_best_cost"], alpha=0.8)
    lower = float(min(frame["parent_best_cost"].min(), frame["candidate_best_cost"].min()))
    upper = float(max(frame["parent_best_cost"].max(), frame["candidate_best_cost"].max()))
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="black", linewidth=1)
    axis.set_xlabel("parent best cost")
    axis.set_ylabel("candidate best cost")
    axis.set_title("Paired validation (below line is better)")
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def plot_representative_case(case: Any, output: Path, title: str) -> Path:
    row = _records([case])
    if not row:
        return _save_empty(output, title)
    value = row[0]
    before = value.get("path_before") or []
    after = value.get("candidate_path") or value.get("state_path_after") or []
    if len(before) < 2 or len(after) < 2:
        return _save_empty(output, title)
    figure, axis = plt.subplots(figsize=(6, 6))
    before_array = np.asarray(before, dtype=float)
    after_array = np.asarray(after, dtype=float)
    axis.plot(before_array[:, 0], before_array[:, 1], "--o", label="before", alpha=0.65)
    axis.plot(after_array[:, 0], after_array[:, 1], "-o", label="candidate", alpha=0.85)
    axis.legend()
    axis.set_aspect("equal", adjustable="box")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(output, dpi=140)
    plt.close(figure)
    return output


def generate_diagnostic_figures(
    traces: Iterable[Any],
    profiles: Iterable[Any],
    synergies: Iterable[Any],
    generation_rows: Iterable[Any],
    paired_outcomes: Iterable[Any],
    output_dir: str | Path,
) -> list[Path]:
    """Generate the standard diagnostic figure set and return all paths."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    trace_rows = _records(traces)
    paths = [
        plot_immediate_improvement(profiles, directory / "02_immediate_improvement.png"),
        plot_delayed_reward(profiles, directory / "03_delayed_reward.png"),
        plot_context_heatmap(trace_rows, "map_type", directory / "04_map_type_performance.png"),
        plot_context_heatmap(trace_rows, "search_phase", directory / "05_search_phase_performance.png"),
        plot_synergy_matrix(synergies, directory / "06_synergy_matrix.png"),
    ]
    generation_path = directory / "08_generation_performance.png"
    generation_records = _records(generation_rows)
    if generation_records or not generation_path.exists():
        plot_generation_performance(generation_records, generation_path)
    paths.append(generation_path)
    paired_path = directory / "09_paired_comparison.png"
    paired_records = _records(paired_outcomes)
    if paired_records or not paired_path.exists():
        plot_paired_comparison(paired_records, paired_path)
    paths.append(paired_path)
    rewards = sorted(trace_rows, key=lambda row: float(row.get("immediate_reward", 0.0)))
    success = rewards[-1] if rewards else {}
    failure = rewards[0] if rewards else {}
    paths.append(plot_representative_case(success, directory / "10a_success_case.png", "Representative success"))
    paths.append(plot_representative_case(failure, directory / "10b_failure_case.png", "Representative failure"))
    return paths
