from __future__ import annotations

from uav_operator_evolution.config import EvolutionConfig
from uav_operator_evolution.evolution.validation import PairedOutcome, decide_retention


def test_retention_uses_paired_effect_threshold() -> None:
    outcomes = [
        PairedOutcome(
            map_id=f"v{index}",
            difficulty="dense",
            parent_best_cost=100.0,
            candidate_best_cost=95.0,
            parent_feasible=True,
            candidate_feasible=True,
            parent_runtime_ms=2.0,
            candidate_runtime_ms=2.1,
        )
        for index in range(4)
    ]
    report = decide_retention("parent", "candidate", outcomes, EvolutionConfig())
    assert report.retained
    assert report.mean_gain == 0.05


def test_safety_failure_cannot_be_overridden_by_performance() -> None:
    outcome = PairedOutcome(
        map_id="v0",
        difficulty="dense",
        parent_best_cost=100.0,
        candidate_best_cost=1.0,
        parent_feasible=False,
        candidate_feasible=True,
        parent_runtime_ms=10.0,
        candidate_runtime_ms=1.0,
    )
    report = decide_retention(
        "parent", "candidate", [outcome], EvolutionConfig(), safety_passed=False, safety_failures=["mutated input"]
    )
    assert not report.retained
    assert "failed safety gate" in report.retention_reasons


def test_runtime_only_noop_candidate_is_not_retained() -> None:
    outcome = PairedOutcome(
        map_id="v0",
        difficulty="dense",
        parent_best_cost=100.0,
        candidate_best_cost=100.0,
        parent_feasible=True,
        candidate_feasible=True,
        parent_runtime_ms=10.0,
        candidate_runtime_ms=5.0,
        parent_operator_runtime_ms=2.0,
        candidate_operator_runtime_ms=1.0,
        candidate_operator_call_count=20,
        candidate_operator_changed_call_count=0,
        candidate_operator_accepted_call_count=0,
    )
    report = decide_retention("parent", "candidate", [outcome], EvolutionConfig())
    assert not report.retained
    assert report.median_runtime_reduction == 0.5
    assert not report.runtime_evidence_eligible
    assert report.candidate_effective_call_rate == 0.0
    assert report.retention_reasons == [
        "runtime evidence ineligible: candidate operator made no effective path changes "
        "(0/20 calls)"
    ]


def test_runtime_only_candidate_requires_pre_registered_effective_call_rate() -> None:
    outcome = PairedOutcome(
        map_id="v0",
        difficulty="dense",
        parent_best_cost=100.0,
        candidate_best_cost=100.0,
        parent_feasible=True,
        candidate_feasible=True,
        parent_runtime_ms=10.0,
        candidate_runtime_ms=5.0,
        parent_operator_runtime_ms=2.0,
        candidate_operator_runtime_ms=1.0,
        candidate_operator_call_count=20,
        candidate_operator_changed_call_count=4,
        candidate_operator_accepted_call_count=2,
    )
    report = decide_retention("parent", "candidate", [outcome], EvolutionConfig())
    assert report.retained
    assert report.retention_reasons == ["runtime reduction"]
    assert report.runtime_evidence_eligible
    assert report.candidate_effective_call_rate == 0.2
    assert report.candidate_operator_acceptance_rate == 0.1
    assert report.median_operator_runtime_reduction == 0.5
