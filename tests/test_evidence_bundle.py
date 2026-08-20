from __future__ import annotations

import json

import pytest

from uav_operator_evolution.agents.evidence import (
    DesignBudget,
    EvidenceBundleBuilder,
    OperatorEvidenceBundle,
)
from uav_operator_evolution.diagnosis.counterfactual import CounterfactualResult
from uav_operator_evolution.memory import MechanismMemory
from uav_operator_evolution.operators.registry import build_manual_operator_registry
from uav_operator_evolution.trajectory import OperatorTrace, TrajectoryRecorder


def _trace(operator_id: str, accepted: bool, before: float, candidate: float) -> OperatorTrace:
    carried = candidate if accepted else before
    return OperatorTrace(
        run_id="evidence-run",
        episode_id="episode",
        map_id="map-1",
        map_difficulty="dense",
        iteration=0 if accepted else 1,
        operator_id=operator_id,
        context={"phase": "train", "search_features": {"stagnation_count": 4}},
        before_state={"path": [[0, 0], [1, 1]], "objective": before, "feasible": True},
        candidate_state={
            "path": [[0, 0], [0.5, 0.4], [1, 1]],
            "objective": candidate,
            "feasible": candidate < 20,
        },
        accepted_state={"path": [[0, 0], [1, 1]], "objective": carried, "feasible": True},
        accepted=accepted,
        acceptance_reason="improved" if accepted else "rejected",
        runtime_ms=0.25,
    )


def _build_fixture(tmp_path):
    database = tmp_path / "experiment.sqlite"
    memory = MechanismMemory(database)
    recorder = TrajectoryRecorder(database)
    success = _trace("segment_shift", True, 10.0, 8.0)
    failure = _trace("segment_shift", False, 10.0, 30.0)
    recorder.record_many([success, failure])
    memory.add_operator_profile(
        {
            "operator_id": "segment_shift",
            "attempts": 12,
            "acceptance_rate": 0.5,
            "mean_immediate_reward": 1.5,
            "mean_delayed_rewards": {"5": 2.5},
            "feasibility_rate": 0.9,
            "effective_context_groups": [
                {"context": {"map_difficulty": "dense"}, "calls": 5, "average_reward": 3.0}
            ],
            "failure_context_groups": [
                {"context": {"stagnation": "high"}, "calls": 2, "average_reward": -2.0}
            ],
            "representative_success_ids": [success.trace_id],
            "representative_failure_ids": [failure.trace_id],
        },
        operator_id="segment_shift",
        run_id="evidence-run",
    )
    memory.add_failure_mode(
        "large_cost_increase", operator_id="segment_shift", count=2, severity=3.0
    )
    memory.add_failure_mode(
        "large_cost_increase", operator_id="segment_shift", count=2, severity=3.0
    )
    memory.add_synergy("segment_shift", "smooth_segment", 1.25, sample_count=5)
    return memory, recorder


def test_bundle_is_compact_content_addressed_and_deterministic(tmp_path, monkeypatch) -> None:
    memory, recorder = _build_fixture(tmp_path)
    monkeypatch.setattr(
        recorder,
        "list_traces",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("bulk trace load forbidden")),
    )
    builder = EvidenceBundleBuilder(
        memory, build_manual_operator_registry(), recorder=recorder, minimum_reliable_samples=3
    )
    counterfactual = [
        CounterfactualResult(
            state_index=0,
            source_trace_id=1,
            operator_id="segment_shift",
            before_objective=10,
            candidate_objective=8,
            reward=2,
            advantage=1,
            feasible=True,
            runtime_ms=0.2,
        )
    ]
    first = builder.build(
        "Improve dense-map search.",
        ["segment_shift"],
        counterfactual_results=counterfactual,
        counterfactual_seed=7,
    )
    second = builder.build(
        "Improve dense-map search.",
        ["segment_shift"],
        counterfactual_results=counterfactual,
        counterfactual_seed=7,
    )
    runtime_variant = builder.build(
        "Improve dense-map search.",
        ["segment_shift"],
        counterfactual_results=[
            counterfactual[0].model_copy(update={"runtime_ms": 999.0})
        ],
        counterfactual_seed=7,
    )
    assert first.bundle_hash == second.bundle_hash
    assert first.bundle_hash == runtime_variant.bundle_hash
    assert first.evidence_ids() == second.evidence_ids()
    assert len(first.failure_modes) == 1
    assert len(first.failure_modes[0].source_refs) == 2
    assert first.failure_contexts[0].low_confidence is True
    assert first.effective_contexts[0].low_confidence is False
    assert first.counterfactual_evidence[0].seed == 7
    payload = first.model_dump(mode="json")
    assert len(json.dumps(payload, ensure_ascii=False)) < first.design_budget.max_bundle_chars
    assert "path" not in payload["representative_success_cases"][0]
    assert payload["representative_success_cases"][0]["before_objective"] == 10
    recorder.close()
    memory.close()


def test_bundle_hash_rejects_tampering(tmp_path) -> None:
    memory, recorder = _build_fixture(tmp_path)
    bundle = EvidenceBundleBuilder(
        memory, build_manual_operator_registry(), recorder=recorder
    ).build("Improve search.", ["segment_shift"], DesignBudget(max_bundle_chars=20_000))
    payload = bundle.model_dump(mode="json")
    payload["problem_summary"] = "Changed after hashing."
    with pytest.raises(ValueError, match="bundle_hash"):
        OperatorEvidenceBundle.model_validate(payload)
    recorder.close()
    memory.close()


def test_parent_and_bundle_limits_are_enforced(tmp_path) -> None:
    memory, recorder = _build_fixture(tmp_path)
    builder = EvidenceBundleBuilder(memory, build_manual_operator_registry(), recorder=recorder)
    with pytest.raises(ValueError, match="parent operator count"):
        builder.build(
            "Too many parents.",
            ["segment_shift", "shortcut"],
            DesignBudget(max_parent_specs=1),
        )
    recorder.close()
    memory.close()
