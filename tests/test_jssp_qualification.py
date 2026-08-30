from __future__ import annotations

from pathlib import Path

from operator_evolution_core.memory import MechanismMemory
from operator_evolution_core.trajectory import TrajectoryRecorder

from jssp_operator_evolution.data import build_jssp_splits
from jssp_operator_evolution.qualification import (
    JSSPFormalQualificationConfig,
    run_formal_qualification,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"


def test_registered_formal_configuration_matches_the_project_plan() -> None:
    config = JSSPFormalQualificationConfig()
    assert config.population_slots == 8
    assert (config.train_calls, config.validation_calls, config.test_calls) == (
        400,
        240,
        400,
    )
    assert (config.generations, config.candidates_per_generation) == (3, 3)
    assert (
        config.train_instances,
        config.validation_instances,
        config.test_instances,
    ) == (60, 41, 41)
    assert config.runtime_repetitions == 4


def test_end_to_end_qualification_uses_shared_memory_and_frozen_test_gate() -> None:
    splits = build_jssp_splits(RAW)
    config = JSSPFormalQualificationConfig(
        train_calls=8,
        validation_calls=8,
        test_calls=8,
        generations=1,
        candidates_per_generation=1,
        train_instances=2,
        validation_instances=1,
        test_instances=2,
        runtime_repetitions=1,
    )
    with (
        TrajectoryRecorder(":memory:") as recorder,
        MechanismMemory(":memory:") as memory,
    ):
        report, outcome = run_formal_qualification(
            splits,
            recorder,
            memory,
            config=config,
        )
        histories = memory.get_operator_history(limit=None)
        profiles = memory.get_operator_profiles(limit=None)

    assert report.training.total_search_calls == 16
    assert report.training.trace_count == 16
    assert report.training.profile_count > 0
    assert histories and profiles
    assert report.freeze_receipt_id == outcome.freeze_receipt.receipt_id
    assert report.frozen_test.test_instances == 2
    assert report.frozen_test.search_calls_per_arm == 8
    assert len(report.frozen_test.outcomes) == 2
    assert report.frozen_test.p0.feasibility_rate == 1.0
    assert report.frozen_test.pn.feasibility_rate == 1.0
