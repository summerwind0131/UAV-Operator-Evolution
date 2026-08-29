from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np

from operator_evolution_core import DomainAdapter, ObjectiveEvaluation
from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    OperatorOutcome,
    SearchBudget,
    SearchContext,
    SimulatedAnnealingAcceptance,
)


@dataclass(frozen=True)
class NumberInstance:
    length: int = 3


class NumberInitializer:
    def initialize(self, instance: NumberInstance, rng: np.random.Generator) -> list[int]:
        return [int(rng.integers(2, 8)) for _ in range(instance.length)]


class NumberEvaluator:
    def evaluate(
        self, solution: list[int], instance: NumberInstance
    ) -> ObjectiveEvaluation:
        del instance
        cost = float(sum(value * value for value in solution))
        return ObjectiveEvaluation(
            scalar_cost=cost,
            components={"squared_sum": cost},
            feasible=True,
        )


class NumberFeatures:
    def extract(self, solution, instance, evaluation):
        del instance, evaluation
        return {"nonzero": sum(value != 0 for value in solution)}


class NumberCodec:
    def clone(self, solution: list[int]) -> list[int]:
        return [int(value) for value in solution]

    def canonicalize(self, solution: object) -> list[int]:
        if not isinstance(solution, list):
            raise ValueError("solution must be a list")
        return [int(value) for value in solution]

    def to_json(self, solution: list[int]):
        return self.clone(solution)

    def stable_hash(self, solution: list[int]) -> str:
        payload = json.dumps(self.to_json(solution), separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class NumberGuard:
    def validate_structure(self, solution: list[int], instance: NumberInstance):
        return [] if len(solution) == instance.length else ["invalid length"]


class NumberTraceEncoder:
    def snapshot(self, solution, instance, evaluation, context):
        del instance
        return {
            "solution": list(solution),
            "objective": evaluation.scalar_cost,
            "search": context.as_features(),
        }


@dataclass
class ZeroOneItem:
    name: str
    operator_id: str

    def apply(
        self,
        solution: list[int],
        instance: NumberInstance,
        rng: np.random.Generator,
        context: SearchContext,
    ) -> OperatorOutcome[list[int]]:
        del instance, context
        index = int(rng.integers(0, len(solution)))
        candidate = list(solution)
        candidate[index] = 0
        return OperatorOutcome(candidate, changed_items=(index,))


def adapter() -> DomainAdapter[NumberInstance, list[int]]:
    codec = NumberCodec()
    return DomainAdapter(
        domain_id="number-vector",
        initializer=NumberInitializer(),
        evaluator=NumberEvaluator(),
        features=NumberFeatures(),
        codec=codec,
        guard=NumberGuard(),
        trace_encoder=NumberTraceEncoder(),
    )


def run_once(seed: int, initial: list[int] | None = None):
    kernel = GenericSearchKernel(
        adapter=adapter(),
        operators=[ZeroOneItem("zero-a", "zero-a"), ZeroOneItem("zero-b", "zero-b")],
        scheduler=BlockRandomRoundRobinScheduler(),
        acceptance=SimulatedAnnealingAcceptance(),
        budget=SearchBudget(max_iterations=6, recent_window=3),
        clock=lambda: 0.0,
    )
    rng = np.random.default_rng(seed)
    return kernel.run(NumberInstance(), rng, initial), rng


def test_generic_kernel_is_deterministic_and_does_not_mutate_input() -> None:
    initial = [5, 4, 3]
    first, _ = run_once(41, initial)
    second, _ = run_once(41, initial)

    assert initial == [5, 4, 3]
    assert first == second
    assert first.iterations == 6
    assert first.best_evaluation.scalar_cost <= first.initial_evaluation.scalar_cost
    assert {step.operator_id for step in first.steps[:2]} == {"zero-a", "zero-b"}
    assert {step.operator_id for step in first.steps[2:4]} == {"zero-a", "zero-b"}


def test_caller_rng_consumes_only_the_four_documented_substream_seeds() -> None:
    _, actual_rng = run_once(20260829)
    expected_rng = np.random.default_rng(20260829)
    expected_rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=4,
        dtype=np.uint64,
    )

    assert actual_rng.integers(0, 2**32) == expected_rng.integers(0, 2**32)


def test_invalid_outcome_is_a_safe_noop() -> None:
    class InvalidOperator:
        name = "invalid"
        operator_id = "invalid"

        def apply(self, solution, instance, rng, context):
            del solution, instance, rng, context
            return OperatorOutcome([1], changed_items=(0,))

    kernel = GenericSearchKernel(
        adapter=adapter(),
        operators=[InvalidOperator()],
        scheduler=BlockRandomRoundRobinScheduler(),
        acceptance=SimulatedAnnealingAcceptance(),
        budget=SearchBudget(max_iterations=1),
        clock=lambda: 0.0,
    )
    result = kernel.run(
        NumberInstance(), np.random.default_rng(3), initial_solution=[2, 2, 2]
    )

    assert result.final_solution == [2, 2, 2]
    assert not result.steps[0].operator_outcome.success
    assert result.steps[0].operator_outcome.failure_reason == (
        "operator returned an invalid solution"
    )

