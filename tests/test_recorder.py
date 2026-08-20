from __future__ import annotations

import json

from uav_operator_evolution.trajectory import (
    OperatorTrace,
    TrajectoryRecorder,
    compute_delayed_rewards,
)


def _trace(run: str, map_id: str, iteration: int, before: float, after: float) -> OperatorTrace:
    return OperatorTrace(
        run_id=run,
        episode_id="episode",
        map_id=map_id,
        map_difficulty="dense",
        iteration=iteration,
        operator_id="two_opt",
        operator_params={"radius": 2},
        context={"stagnation": True},
        before_state={"path": [[0, 0], [1, 1]], "objective": before, "feasible": True},
        candidate_state={"path": [[0, 0], [1, 0]], "objective": after, "feasible": True},
        accepted_state={"path": [[0, 0], [1, 0]], "objective": after, "feasible": True},
        accepted=True,
        runtime_ms=1.5,
    )


def test_operator_trace_completes_summaries() -> None:
    trace = OperatorTrace(
        operator="repair",
        before={"cost": 10, "feasible": False, "components": {"risk": 4}},
        candidate={"cost": 7, "feasible": True, "components": {"risk": 1}},
        accepted=True,
    )
    assert trace.operator_id == "repair"
    assert trace.accepted_state == trace.candidate_state
    assert trace.before_objective == 10
    assert trace.accepted_objective == 7
    assert trace.immediate_reward == 3
    assert trace.before_components == {"risk": 4.0}


def test_recorder_round_trip_jsonl_and_delayed_rewards(tmp_path) -> None:
    jsonl = tmp_path / "traces.jsonl"
    with TrajectoryRecorder(tmp_path / "traces.sqlite", jsonl) as recorder:
        first = _trace("run", "a", 0, 10, 9)
        second = _trace("run", "a", 1, 9, 7)
        other_map = _trace("run", "b", 0, 100, 50)
        identifiers = recorder.record_many([first, second, other_map])
        assert identifiers == [1, 2, 3]
        assert first.trace_id == 1

        restored = recorder.get_trace(1)
        assert restored is not None
        assert restored.before_state["path"] == [[0, 0], [1, 1]]
        assert restored.operator_params == {"radius": 2}

        updated = recorder.update_delayed_rewards([1, 2], run_id="run")
        by_id = {trace.trace_id: trace for trace in updated}
        assert by_id[1].delayed_rewards == {1: 1.0, 2: 3.0}
        assert by_id[2].delayed_rewards == {1: 2.0, 2: None}
        # A delayed reward is never computed using a different map's state.
        assert by_id[3].delayed_rewards == {1: 50.0, 2: None}

        assert len(recorder.list_traces("run")) == 3

    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines()]
    assert [row["trace_id"] for row in rows] == [1, 2, 3]
    assert rows[0]["delayed_rewards"] == {"1": 1.0, "2": 3.0}


def test_compute_delayed_rewards_preserves_input_and_marks_censoring() -> None:
    traces = [_trace("r", "m", 0, 10, 8), _trace("r", "m", 1, 8, 7)]
    computed = compute_delayed_rewards(traces, [2])
    assert traces[0].delayed_rewards == {}
    assert computed[0].delayed_rewards[2] == 3
    assert computed[1].delayed_rewards[2] is None
