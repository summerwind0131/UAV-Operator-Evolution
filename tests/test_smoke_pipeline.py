from __future__ import annotations

from pathlib import Path
import json
import sqlite3

import yaml

from uav_operator_evolution.cli import main
from uav_operator_evolution.config import load_config


def test_demo_cli_runs_complete_offline_loop(tmp_path: Path) -> None:
    config = load_config("configs/smoke.yaml").model_copy(deep=True)
    for split in (config.maps.train, config.maps.validation, config.maps.test):
        split.count = 1
        split.width = 45
        split.height = 45
        split.safety_distance = 1
    config.maps.train.difficulties = ["sparse"]
    config.maps.validation.difficulties = ["medium"]
    config.maps.test.difficulties = ["dense"]
    config.maps.grid_resolution = 5
    config.search.train_iterations = 8
    config.search.validation_iterations = 6
    config.search.test_iterations = 6
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    config.diagnostics.minimum_context_samples = 1
    config.output.data_dir = tmp_path / "data"
    config.output.results_dir = tmp_path / "results"
    config.output.figures_dir = tmp_path / "figures"
    config_path = tmp_path / "tiny.yaml"
    config_path.write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    exit_code = main(
        ["demo", "--config", str(config_path), "--run-id", "tiny", "--run-dir", str(run_dir)]
    )
    assert exit_code == 0
    assert (run_dir / "experiment.sqlite").exists()
    assert (run_dir / "evolution_summary.json").exists()
    assert (run_dir / "diagnosis.json").exists()
    assert (tmp_path / "figures" / "tiny" / "01_path_comparison.png").exists()
    assert len(list((tmp_path / "figures" / "tiny").glob("*.png"))) >= 10
    summary = json.loads((run_dir / "evolution_summary.json").read_text(encoding="utf-8"))
    assert summary["trace_count"] == 8 + 2 * 6 + 2 * 6
    report = summary["generations"][0]["validations"][0]
    with sqlite3.connect(run_dir / "experiment.sqlite") as connection:
        status = connection.execute(
            "SELECT status FROM mechanisms WHERE mechanism_id = ?",
            (report["candidate_operator"],),
        ).fetchone()[0]
    assert status == ("active" if report["retained"] else "rejected")
