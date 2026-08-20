from __future__ import annotations

from uav_operator_evolution.diagnosis import (
    CounterfactualEvaluator,
    OperatorDiagnoser,
    compute_sequential_synergies,
)
from uav_operator_evolution.trajectory import OperatorTrace, TrajectoryRecorder


def _trace(iteration: int, operator: str, reward: float, difficulty: str) -> OperatorTrace:
    return OperatorTrace(
        run_id="run",
        episode_id="episode",
        map_id="map",
        iteration=iteration,
        operator_id=operator,
        map_difficulty=difficulty,
        before_objective=10,
        accepted_objective=10 - reward,
        accepted=reward > 0,
        immediate_reward=reward,
        delayed_rewards={5: reward * 2},
        acceptance_reason=None if reward > 0 else "no gain",
        runtime_ms=iteration + 1,
    )


def test_grouped_profiles_can_read_recorder(tmp_path) -> None:
    with TrajectoryRecorder(tmp_path / "traces.sqlite") as recorder:
        for trace in (
            _trace(0, "a", 2, "dense"),
            _trace(1, "a", -1, "dense"),
            _trace(2, "a", 4, "sparse"),
            _trace(3, "b", 1, "dense"),
        ):
            recorder.record(trace)
        profiles = OperatorDiagnoser(
            recorder, minimum_context_samples=2, group_by="map_difficulty"
        ).diagnose()

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.operator_id == "a"
    assert profile.context == {"map_difficulty": "dense"}
    assert profile.attempts == 2
    assert profile.acceptance_rate == 0.5
    assert profile.mean_immediate_reward == 0.5
    assert profile.mean_delayed_rewards == {5: 1.0}
    assert profile.failure_modes == {"no gain": 1}
    serialized = profile.model_dump()
    assert serialized["operator_name"] == "a"
    assert serialized["total_calls"] == 2
    assert serialized["immediate_improvement_rate"] == 0.5
    assert serialized["delayed_improvement_rate"] == 0.5
    assert serialized["average_immediate_reward"] == 0.5
    assert serialized["average_delayed_reward"] == 1.0
    assert serialized["average_runtime_ms"] == 1.5


def test_sequential_synergy_compares_followup_to_baseline() -> None:
    traces = [
        _trace(0, "a", 1, "dense"),
        _trace(1, "b", 5, "dense"),
        _trace(2, "c", 1, "dense"),
        _trace(3, "b", -1, "dense"),
    ]
    synergies = compute_sequential_synergies(traces)
    a_then_b = next(
        item
        for item in synergies
        if item.first_operator == "a" and item.second_operator == "b"
    )
    assert a_then_b.baseline_followup_reward == 2
    assert a_then_b.synergy == 3
    assert a_then_b.model_dump()["reward_delta"] == 3
    assert a_then_b.model_dump()["relation"] == "a->b"


def test_counterfactual_evaluator_uses_identical_copies() -> None:
    states = [{"objective": 10, "path": [1, 2]}]

    def improve(state):
        state["objective"] -= 3
        state["path"].append(3)
        return state

    def fail(state):
        raise RuntimeError("unsupported")

    results = CounterfactualEvaluator(max_states=1).evaluate(
        states, {"improve": improve, "fail": fail}
    )
    assert states == [{"objective": 10, "path": [1, 2]}]
    assert results[0].reward == 3
    assert results[1].error is not None


def test_counterfactual_advantage_is_reward_minus_other_operator_mean() -> None:
    states = [{"objective": 10}]

    def improve_by(amount):
        def apply(state):
            state["objective"] -= amount
            return state

        return apply

    results = CounterfactualEvaluator(max_states=1).evaluate(
        states, {"strong": improve_by(4), "weak": improve_by(1)}
    )
    by_name = {result.operator_id: result for result in results}
    assert by_name["strong"].advantage == 3
    assert by_name["weak"].advantage == -3
