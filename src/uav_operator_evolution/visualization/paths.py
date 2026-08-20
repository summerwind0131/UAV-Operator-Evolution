"""Headless-safe path and environment figures."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle

from ..environment.environment import Environment2D
from ..environment.obstacles import CircleObstacle, RectangleObstacle
from ..path.models import EvaluationResult


def _draw_environment(environment: Environment2D, axes: Axes) -> None:
    for zone in environment.risk_zones:
        axes.add_patch(
            Rectangle(
                (zone.min_x, zone.min_y),
                zone.max_x - zone.min_x,
                zone.max_y - zone.min_y,
                facecolor="tab:orange",
                edgecolor="tab:orange",
                alpha=0.16,
                hatch="//",
                label="Risk zone" if not any(patch.get_label() == "Risk zone" for patch in axes.patches) else None,
            )
        )
    obstacle_label_used = False
    clearance_label_used = False
    for obstacle in environment.obstacles:
        obstacle_label = "Obstacle" if not obstacle_label_used else None
        clearance_label = "Safety margin" if not clearance_label_used and environment.safety_distance > 0 else None
        if isinstance(obstacle, CircleObstacle):
            axes.add_patch(
                Circle(obstacle.center, obstacle.radius, color="0.25", alpha=0.85, label=obstacle_label)
            )
            if environment.safety_distance > 0:
                axes.add_patch(
                    Circle(
                        obstacle.center,
                        obstacle.radius + environment.safety_distance,
                        fill=False,
                        edgecolor="tab:red",
                        linestyle=":",
                        linewidth=0.8,
                        label=clearance_label,
                    )
                )
                clearance_label_used = True
        elif isinstance(obstacle, RectangleObstacle):
            axes.add_patch(
                Rectangle(
                    (obstacle.min_x, obstacle.min_y),
                    obstacle.width,
                    obstacle.height,
                    color="0.25",
                    alpha=0.85,
                    label=obstacle_label,
                )
            )
            if environment.safety_distance > 0:
                axes.add_patch(
                    Rectangle(
                        (
                            obstacle.min_x - environment.safety_distance,
                            obstacle.min_y - environment.safety_distance,
                        ),
                        obstacle.width + 2 * environment.safety_distance,
                        obstacle.height + 2 * environment.safety_distance,
                        fill=False,
                        edgecolor="tab:red",
                        linestyle=":",
                        linewidth=0.8,
                        label=clearance_label,
                    )
                )
                clearance_label_used = True
        obstacle_label_used = True
    axes.scatter(*environment.start, marker="o", s=55, color="tab:green", zorder=5, label="Start")
    axes.scatter(*environment.goal, marker="*", s=90, color="tab:red", zorder=5, label="Goal")
    axes.set_xlim(0, environment.width)
    axes.set_ylim(0, environment.height)
    axes.set_aspect("equal", adjustable="box")
    axes.set_xlabel("x")
    axes.set_ylabel("y")
    axes.grid(alpha=0.18)


def _draw_path(path: Sequence[Sequence[float]], axes: Axes, *, label: str, color: str) -> None:
    x_values = [point[0] for point in path]
    y_values = [point[1] for point in path]
    axes.plot(x_values, y_values, color=color, linewidth=2.0, marker="o", markersize=3, label=label)


def plot_path(
    environment: Environment2D,
    path: Sequence[Sequence[float]],
    *,
    initial_path: Sequence[Sequence[float]] | None = None,
    evaluation: EvaluationResult | None = None,
    title: str | None = None,
    output_path: str | FilePath | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot one planned path, optionally overlaid with its initial path."""

    if ax is None:
        figure, axes = plt.subplots(figsize=(7.2, 6.2), constrained_layout=True)
    else:
        axes = ax
        figure = axes.figure
    _draw_environment(environment, axes)
    if initial_path is not None:
        _draw_path(initial_path, axes, label="Initial path", color="0.55")
    _draw_path(path, axes, label="Planned path", color="tab:blue")
    resolved_title = title or f"{environment.map_id} ({environment.difficulty})"
    if evaluation is not None:
        resolved_title += f"\ncost={evaluation.total_cost:.2f}, feasible={evaluation.feasible}"
    axes.set_title(resolved_title)
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes.legend(unique.values(), unique.keys(), loc="best", fontsize="small")
    if output_path is not None:
        destination = FilePath(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    return figure


def plot_path_comparison(
    environment: Environment2D,
    before: Sequence[Sequence[float]],
    after: Sequence[Sequence[float]],
    *,
    before_evaluation: EvaluationResult | None = None,
    after_evaluation: EvaluationResult | None = None,
    output_path: str | FilePath | None = None,
) -> Figure:
    """Create a side-by-side before/after planning comparison."""

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.8), constrained_layout=True)
    before_title = "Before"
    if before_evaluation is not None:
        before_title += f"\ncost={before_evaluation.total_cost:.2f}, feasible={before_evaluation.feasible}"
    after_title = "After"
    if after_evaluation is not None:
        after_title += f"\ncost={after_evaluation.total_cost:.2f}, feasible={after_evaluation.feasible}"
    for axis, candidate, title, color in (
        (axes[0], before, before_title, "0.45"),
        (axes[1], after, after_title, "tab:blue"),
    ):
        _draw_environment(environment, axis)
        _draw_path(candidate, axis, label=title.splitlines()[0], color=color)
        axis.set_title(title)
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), loc="best", fontsize="small")
    if output_path is not None:
        destination = FilePath(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=150, bbox_inches="tight")
    return figure


__all__ = ["plot_path", "plot_path_comparison"]
