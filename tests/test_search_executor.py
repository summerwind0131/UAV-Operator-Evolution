from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.operators import OperatorResult, ShortcutOperator, unchanged_result
from uav_operator_evolution.path import PathEvaluator
from uav_operator_evolution.search import SearchContext, SearchExecutor
from uav_operator_evolution.trajectory import TrajectoryRecorder


def environment() -> Environment2D:
    return Environment2D(
        map_id="search",
        width=100.0,
        height=100.0,
        start=(10.0, 10.0),
        goal=(90.0, 90.0),
        obstacles=[],
        safety_distance=1.0,
        difficulty="sparse",
    )


def test_executor_keeps_input_immutable_and_returns_best_path() -> None:
    env = environment()
    initial = [env.start, (10.0, 90.0), env.goal]
    snapshot = list(initial)
    executor = SearchExecutor(
        [ShortcutOperator()],
        evaluator=PathEvaluator(),
        max_iterations=3,
    )

    result = executor.run(env, np.random.default_rng(19), initial_path=initial)

    assert initial == snapshot
    assert result.iterations == 3
    assert result.best_path == [env.start, env.goal]
    assert result.best_evaluation.total_cost < result.initial_evaluation.total_cost
    assert result.best_path[0] == env.start
    assert result.best_path[-1] == env.goal
    assert result.best_cost_history[-1] <= result.best_cost_history[0]


@dataclass
class RecordingNoOp:
    name: str
    calls: list[str]

    def apply(self, path, environment, rng, context: SearchContext) -> OperatorResult:
        del environment, rng, context
        self.calls.append(self.name)
        return unchanged_result(path, "intentional test no-op")


def test_scheduler_calls_each_operator_once_per_randomized_block() -> None:
    env = environment()
    calls: list[str] = []
    operators = [RecordingNoOp(name, calls) for name in ("a", "b", "c")]
    executor = SearchExecutor(operators, max_iterations=6)

    result = executor.run(env, np.random.default_rng(7), initial_path=[env.start, env.goal])

    assert result.iterations == 6
    assert set(calls[:3]) == {"a", "b", "c"}
    assert set(calls[3:]) == {"a", "b", "c"}
    assert result.accepted_count == 0


class EndpointBreakingOperator:
    name: ClassVar[str] = "endpoint_breaker"

    def apply(self, path, environment, rng, context) -> OperatorResult:
        del environment, rng, context
        candidate = list(path)
        candidate[0] = (0.0, 0.0)
        return OperatorResult(candidate, (0,))


def test_executor_rejects_invalid_operator_output_safely() -> None:
    env = environment()
    initial = [env.start, env.goal]
    executor = SearchExecutor([EndpointBreakingOperator()], max_iterations=1)

    result = executor.run(env, np.random.default_rng(2), initial_path=initial)

    assert result.final_path == initial
    assert not result.steps[0].accepted
    assert not result.steps[0].operator_result.success
    assert result.steps[0].operator_result.failure_reason == "operator returned an invalid path"


def test_search_is_reproducible_with_equal_seed() -> None:
    env = environment()
    initial = [env.start, (20.0, 70.0), (60.0, 40.0), env.goal]

    def run_once():
        executor = SearchExecutor([ShortcutOperator()], max_iterations=4)
        return executor.run(env, np.random.default_rng(101), initial_path=initial)

    first = run_once()
    second = run_once()
    assert first.final_path == second.final_path
    assert first.cost_history == second.cost_history
    assert [step.accepted for step in first.steps] == [step.accepted for step in second.steps]


def test_executor_records_complete_before_candidate_and_accepted_states() -> None:
    env = environment()
    initial = [env.start, (10.0, 90.0), env.goal]
    with TrajectoryRecorder(":memory:") as recorder:
        executor = SearchExecutor([ShortcutOperator()], max_iterations=1, recorder=recorder)
        result = executor.run(env, np.random.default_rng(3), initial_path=initial, run_id="search-test")
        traces = recorder.list_traces("search-test")

    assert result.steps[0].accepted
    assert len(traces) == 1
    trace = traces[0]
    assert trace.before_state["path"] == [list(point) for point in initial]
    assert trace.candidate_state["path"] == [list(env.start), list(env.goal)]
    assert trace.accepted_state == trace.candidate_state
    assert trace.before_objective == result.initial_evaluation.total_cost
    assert trace.candidate_objective == result.best_evaluation.total_cost
    assert trace.context["environment_features"]["difficulty"] == "sparse"
    assert trace.metadata["created_new_best"] is True
