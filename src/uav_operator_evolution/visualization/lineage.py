"""Dependency-free matplotlib rendering for operator lineage."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_lineage(records: Iterable[Any], output: str | Path) -> Path:
    output_path = Path(output)
    rows = [
        record.model_dump(mode="python") if hasattr(record, "model_dump") else dict(record)
        for record in records
    ]
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.set_axis_off()
    axis.set_title("Operator lineage")
    if not rows:
        axis.text(0.5, 0.5, "insufficient evidence", ha="center", va="center")
    else:
        generations = sorted({int(row.get("generation", 0)) for row in rows})
        grouped = {generation: [row for row in rows if int(row.get("generation", 0)) == generation] for generation in generations}
        positions: dict[str, tuple[float, float]] = {}
        for x_index, generation in enumerate(generations):
            items = grouped[generation]
            for y_index, row in enumerate(items):
                name = str(row.get("child_operator", row.get("operator_name", row.get("name", "unknown"))))
                x = x_index / max(len(generations) - 1, 1)
                y = 1.0 - (y_index + 1) / (len(items) + 1)
                positions[name] = (x, y)
                retained = bool(row.get("retained", row.get("active_status", True)))
                axis.text(
                    x,
                    y,
                    name,
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round", "facecolor": "#C6E0B4" if retained else "#F4B084"},
                )
        for row in rows:
            child = str(row.get("child_operator", row.get("operator_name", row.get("name", ""))))
            parents = row.get("parent_operators", row.get("parent_ids", [])) or []
            if isinstance(parents, str):
                parents = [parents]
            for parent in parents:
                if str(parent) in positions and child in positions:
                    start = positions[str(parent)]
                    end = positions[child]
                    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "color": "#666666"})
        axis.set_xlim(-0.12, 1.12)
        axis.set_ylim(-0.05, 1.05)
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)
    return output_path

