from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from pydantic import ValidationError

from uav_operator_evolution.characterization import (
    identity_hash,
    identity_projection,
)
from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment.generator import (
    DatasetManifest,
    MapGenerator,
    load_dataset_split,
)
from uav_operator_evolution.evolution.manager import OperatorEvolutionManager
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.registry import default_manual_operators
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.reproducibility import stable_hash
from uav_operator_evolution.search.executor import SearchExecutor
from uav_operator_evolution.trajectory import OperatorTrace, TrajectoryRecorder


ROOT = Path(__file__).resolve().parents[1]
BASELINE_PATH = ROOT / "tests" / "baselines" / "uav_phase1_identity_v1.json"
SEARCH_SEED = 20_260_820


class _DeterministicClock:
    """Monotonic test clock that makes runtime-weighted ranking repeatable."""

    def __init__(self, tick_seconds: float = 0.0001) -> None:
        self.value = 0.0
        self.tick_seconds = tick_seconds

    def __call__(self) -> float:
        self.value += self.tick_seconds
        return self.value


def _baseline() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text(encoding="utf-8"))


def _search_summary() -> dict[str, Any]:
    environment = load_dataset_split(ROOT / "data" / "multi-agent-smoke", "train")[0]
    operators = default_manual_operators()
    with TrajectoryRecorder(":memory:") as recorder:
        result = SearchExecutor(
            operators,
            evaluator=PathEvaluator(),
            max_iterations=16,
            initializer_grid_resolution=4.0,
            recorder=recorder,
        ).run(
            environment,
            np.random.default_rng(SEARCH_SEED),
            run_id="uav-phase1-search",
        )
        traces = recorder.list_traces("uav-phase1-search")

    return {
        "environment_content_hash": environment.content_hash,
        "operator_catalog_hash": stable_hash(
            {
                name: spec.model_dump(mode="json")
                for name, spec in sorted(manual_operator_specs().items())
            }
        ),
        "operator_sequence": [step.operator_id for step in result.steps],
        "accepted_sequence": [step.accepted for step in result.steps],
        "created_new_best_sequence": [step.created_new_best for step in result.steps],
        "initial_path_hash": stable_hash(result.initial_path),
        "final_path_hash": stable_hash(result.final_path),
        "best_path_hash": stable_hash(result.best_path),
        "result_identity_hash": identity_hash(result),
        "step_identity_hashes": [identity_hash(step) for step in result.steps],
        "trace_identity_hash": identity_hash(traces),
        "trace_identity_hashes": [identity_hash(trace) for trace in traces],
        "combined_identity_hash": identity_hash({"result": result, "traces": traces}),
    }


def _tiny_evolution_config():
    config = load_config(ROOT / "configs" / "smoke.yaml").model_copy(deep=True)
    config.search.train_iterations = 8
    config.search.validation_iterations = 6
    config.search.test_iterations = 6
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    config.evolution.runtime_validation_repetitions = 2
    # Runtime measurements are intentionally outside the identity projection.
    # Disable the runtime-only retention path so wall-clock noise cannot change
    # the semantic retained/rejected decision captured by this baseline.
    config.evolution.min_runtime_reduction = 1.0
    config.evolution.min_runtime_effective_call_rate = 1.0
    config.diagnostics.minimum_context_samples = 1
    config.diagnostics.delayed_horizons = [2, 4]
    return config


def _evolution_summary(tmp_path: Path) -> dict[str, Any]:
    config = _tiny_evolution_config()
    generator = MapGenerator(config.seed, grid_resolution=5.0, generation_attempts=20)
    datasets = {
        "train": [
            generator.generate_map(
                "train", 0, "sparse", width=50, height=50, safety_distance=1
            )
        ],
        "validation": [
            generator.generate_map(
                "validation", 0, "medium", width=50, height=50, safety_distance=1
            )
        ],
        "test": [
            generator.generate_map(
                "test", 0, "dense", width=55, height=55, safety_distance=1
            )
        ],
    }
    database = tmp_path / "uav-phase1-characterization.sqlite"
    clock = _DeterministicClock()
    # Runtime currently contributes 5% of parent fitness.  A deterministic
    # characterization clock freezes that existing rule without pretending
    # real wall-clock measurements are reproducible semantic identity.
    with patch("time.perf_counter", new=clock), patch(
        "uav_operator_evolution.search.executor.perf_counter", new=clock
    ):
        result = OperatorEvolutionManager(config, database).run(
            datasets, "uav-phase1-evolution"
        )
    generation = result.generations[0]
    with closing(sqlite3.connect(database)) as connection:
        candidate_statuses = {}
        for proposal in generation.proposals:
            name = proposal.spec.name
            row = connection.execute(
                "SELECT status FROM mechanisms WHERE mechanism_id = ?",
                (name,),
            ).fetchone()
            candidate_statuses[name] = None if row is None else row[0]

    return {
        "dataset_content_hashes": {
            split: environments[0].content_hash
            for split, environments in datasets.items()
        },
        "initial_population": result.initial_population,
        "final_population": result.final_population,
        "retained_candidates": result.retained_candidates,
        "trace_count": result.trace_count,
        "candidate_statuses": candidate_statuses,
        "proposal_identity_hashes": [
            identity_hash(proposal) for proposal in generation.proposals
        ],
        "validation_identity_hashes": [
            identity_hash(report) for report in generation.validations
        ],
        "test_outcomes_identity_hash": identity_hash(result.test_outcomes),
        "result_identity_hash": identity_hash(result),
    }


def _inventory_summary() -> dict[str, Any]:
    config_hashes = {}
    for path in sorted((ROOT / "configs").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        entry = {"source_hash": stable_hash(payload)}
        try:
            entry["experiment_config_hash"] = load_config(path).config_hash
        except ValidationError:
            # Matrix/protocol YAML files are intentionally not ExperimentConfig.
            pass
        config_hashes[path.name] = entry
    manifests = {}
    for name, path in {
        "multi-agent-smoke": ROOT / "data" / "multi-agent-smoke" / "manifest.json",
        "uav2d-v1": ROOT / "data" / "benchmarks" / "uav2d-v1" / "manifest.json",
    }.items():
        manifest = DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))
        manifests[name] = {
            "config_hash": manifest.config_hash,
            "manifest_hash": manifest.manifest_hash,
            "map_count": len(manifest.maps),
        }
    return {"config_hashes": config_hashes, "dataset_manifests": manifests}


def test_identity_projection_excludes_only_volatile_measurements() -> None:
    first = OperatorTrace(
        trace_id=1,
        run_id="identity",
        operator_id="shortcut",
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        runtime_ms=0.1,
        before_state={"path": [[0, 0], [1, 1]], "objective": 2.0},
        candidate_state={"path": [[0, 0], [1, 1]], "objective": 1.0},
        accepted=True,
    )
    second = first.model_copy(
        update={
            "timestamp": datetime(2026, 8, 20, tzinfo=timezone.utc),
            "runtime_ms": 999.0,
        }
    )
    assert identity_projection(first) == identity_projection(second)
    assert identity_hash(first) == identity_hash(second)
    assert identity_hash(first) != identity_hash(
        second.model_copy(update={"accepted": False})
    )


def test_config_and_dataset_inventory_matches_phase1_baseline() -> None:
    assert _inventory_summary() == _baseline()["inventory"]


def test_search_and_three_state_trace_match_phase1_baseline() -> None:
    first = _search_summary()
    second = _search_summary()
    assert first == second
    assert first == _baseline()["search"]


def test_tiny_evolution_loop_matches_phase1_baseline(tmp_path: Path) -> None:
    assert _evolution_summary(tmp_path) == _baseline()["evolution"]
