"""Human-readable summary of an existing evolution run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    directory = Path(run_dir)
    summary_path = directory / "evolution_summary.json"
    if not summary_path.exists():
        summary_path = directory / "search_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"no summary JSON found in {directory}")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if "generations" in payload:
        return {
            "run_id": payload["run_id"],
            "generations": len(payload["generations"]),
            "trace_count": payload["trace_count"],
            "initial_population": payload["initial_population"],
            "final_population": payload["final_population"],
            "retained_candidates": payload["retained_candidates"],
            "test_pairs": len(payload.get("test_outcomes", [])),
        }
    return payload

