from __future__ import annotations

from uav_operator_evolution.config import load_config
from uav_operator_evolution.experiments.agent_evidence import (
    build_evidence_for_run,
    select_evidence_parents,
)
from uav_operator_evolution.memory import MechanismMemory
from uav_operator_evolution.operators.registry import build_manual_operator_registry
from uav_operator_evolution.trajectory import OperatorTrace, TrajectoryRecorder


def test_parent_selection_prefers_profiled_operator_with_failure_evidence(tmp_path) -> None:
    database = tmp_path / "experiment.sqlite"
    with MechanismMemory(database) as memory:
        memory.add_operator_profile(
            {"operator_id": "segment_shift", "attempts": 8, "mean_immediate_reward": 1.0},
            operator_id="segment_shift",
        )
        memory.add_failure_mode(
            "large_cost_increase", operator_id="segment_shift", count=3
        )
        selected = select_evidence_parents(
            memory, build_manual_operator_registry(), limit=1
        )
    assert selected == ["segment_shift"]


def test_workflow_builds_bundle_from_typed_point_queries(tmp_path, monkeypatch) -> None:
    database = tmp_path / "experiment.sqlite"
    with TrajectoryRecorder(database) as recorder:
        trace = OperatorTrace(
            run_id="run",
            map_id="map",
            map_difficulty="medium",
            iteration=0,
            operator_id="segment_shift",
            before_state={"path": [[0, 0], [1, 1]], "objective": 10, "feasible": True},
            candidate_state={"path": [[0, 0], [1, 1]], "objective": 12, "feasible": True},
            accepted_state={"path": [[0, 0], [1, 1]], "objective": 10, "feasible": True},
            accepted=False,
            immediate_reward=-2,
        )
        recorder.record(trace)
    with MechanismMemory(database) as memory:
        memory.add_operator_profile(
            {
                "operator_id": "segment_shift",
                "attempts": 1,
                "mean_immediate_reward": -2,
                "representative_failure_ids": [trace.trace_id],
                "failure_contexts": [
                    {"context": {"map_type": "medium"}, "calls": 1, "average_reward": -2}
                ],
            },
            operator_id="segment_shift",
        )
        memory.add_failure_mode(
            "large_cost_increase", operator_id="segment_shift", count=1
        )

    monkeypatch.setattr(
        TrajectoryRecorder,
        "list_traces",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("workflow must not bulk-load traces")
        ),
    )
    config = load_config("configs/agent_smoke.yaml")
    bundle = build_evidence_for_run(
        config,
        database,
        parent_operator_ids=["segment_shift"],
    )
    assert bundle.parent_specs[0].name == "segment_shift"
    assert bundle.failure_modes[0].failure_mode == "large_cost_increase"
    assert bundle.representative_failure_cases[0].trace_id == trace.trace_id
    assert "counterfactual evidence was not supplied" in bundle.limitations
