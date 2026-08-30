from __future__ import annotations

from pathlib import Path

from operator_evolution_core.trajectory import TrajectoryRecorder

from jssp_operator_evolution.data import build_jssp_splits
from jssp_operator_evolution.evolution import (
    JSSPEvolutionSmokeConfig,
    run_offline_evolution_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"


def test_two_generation_two_candidate_offline_lifecycle_and_freeze_gate() -> None:
    splits = build_jssp_splits(RAW)
    with TrajectoryRecorder(":memory:") as recorder:
        outcome = run_offline_evolution_smoke(
            splits,
            recorder=recorder,
            config=JSSPEvolutionSmokeConfig(
                search_calls=8,
                generations=2,
                candidates_per_generation=2,
                validation_instances=1,
                runtime_repetitions=1,
            ),
        )
        assert outcome.report.trace_count == len(recorder)

    assert len(outcome.report.initial_population_ids) == 8
    assert len(outcome.report.final_population_ids) == 8
    assert len(outcome.report.candidate_records) == 4
    assert all(record.smoke_passed for record in outcome.report.candidate_records)
    assert all(record.validation_outcomes == 1 for record in outcome.report.candidate_records)
    assert all(len(record.envelope_hash) == 64 for record in outcome.report.candidate_records)
    assert len(outcome.report.final_population_fingerprint) == 64
    assert outcome.report.freeze_receipt_id == outcome.freeze_receipt.receipt_id
    assert len(splits.open_test(outcome.freeze_receipt)) == 41
