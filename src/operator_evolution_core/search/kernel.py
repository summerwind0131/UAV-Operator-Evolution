"""Domain-independent fixed-budget search loop."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Generic, TypeVar, cast

import numpy as np
from pydantic import JsonValue

from ..contracts import DomainAdapter, ObjectiveEvaluation
from .models import (
    OperatorOutcome,
    SearchBudget,
    SearchContext,
    SearchResult,
    SearchStep,
)
from .protocols import AcceptancePolicy, OperatorScheduler, SearchOperator

InstanceT = TypeVar("InstanceT")
SolutionT = TypeVar("SolutionT")
CoreOperator = SearchOperator[InstanceT, SolutionT]
StepCallback = Callable[[SearchStep[SolutionT], CoreOperator], None]


class GenericSearchKernel(Generic[InstanceT, SolutionT]):
    """Run a search using only adapter and policy contracts."""

    def __init__(
        self,
        *,
        adapter: DomainAdapter[InstanceT, SolutionT],
        operators: Sequence[CoreOperator],
        scheduler: OperatorScheduler[CoreOperator],
        acceptance: AcceptancePolicy,
        budget: SearchBudget,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not operators:
            raise ValueError("GenericSearchKernel requires at least one operator")
        self.adapter = adapter
        self.operators = tuple(operators)
        self.scheduler = scheduler
        self.acceptance = acceptance
        self.budget = budget
        self.clock = clock

    def run(
        self,
        instance: InstanceT,
        rng: np.random.Generator,
        initial_solution: SolutionT | None = None,
        *,
        on_step: StepCallback[SolutionT] | None = None,
    ) -> SearchResult[SolutionT]:
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")

        # This exact derivation is a compatibility boundary. Domain work may
        # consume child streams but must never consume the caller's stream.
        stream_seeds = rng.integers(
            0,
            np.iinfo(np.uint64).max,
            size=4,
            dtype=np.uint64,
        )
        initializer_rng = np.random.default_rng(stream_seeds[0])
        scheduler_rng = np.random.default_rng(stream_seeds[1])
        operator_seed_rng = np.random.default_rng(stream_seeds[2])
        acceptance_rng = np.random.default_rng(stream_seeds[3])

        source = (
            self.adapter.initializer.initialize(instance, initializer_rng)
            if initial_solution is None
            else initial_solution
        )
        initial = self.adapter.codec.clone(source)
        violations = self.adapter.guard.validate_structure(initial, instance)
        if violations:
            raise ValueError(
                "initial solution violates structure: " + "; ".join(violations)
            )
        initial_evaluation = self.adapter.evaluator.evaluate(initial, instance)
        current_solution = self.adapter.codec.clone(initial)
        current_evaluation = initial_evaluation
        best_solution = self.adapter.codec.clone(initial)
        best_evaluation = initial_evaluation
        cost_scale = max(abs(float(initial_evaluation.scalar_cost)), 1.0)
        recent_improvements: list[float] = []
        recent_acceptances: list[bool] = []
        stagnation_count = 0
        last_created_new_best = False
        steps: list[SearchStep[SolutionT]] = []
        reset = getattr(self.scheduler, "reset", None)
        if callable(reset):
            reset()

        for iteration in range(self.budget.max_iterations):
            context_before = SearchContext(
                iteration=iteration,
                max_iterations=self.budget.max_iterations,
                current_evaluation=current_evaluation,
                best_evaluation=best_evaluation,
                stagnation_count=stagnation_count,
                recent_improvements=tuple(recent_improvements),
                recent_acceptances=tuple(recent_acceptances),
                last_created_new_best=last_created_new_best,
            )
            operator = self.scheduler.select(
                self.operators, iteration, scheduler_rng
            )
            operator_name = str(operator.name)
            operator_id = str(getattr(operator, "operator_id", operator_name))
            solution_before = self.adapter.codec.clone(current_solution)
            operator_input = self.adapter.codec.clone(current_solution)
            operator_rng = np.random.default_rng(
                operator_seed_rng.integers(
                    0,
                    np.iinfo(np.uint64).max,
                    dtype=np.uint64,
                )
            )
            start_time = self.clock()
            try:
                raw_outcome = operator.apply(
                    operator_input,
                    instance,
                    operator_rng,
                    context_before,
                )
            except Exception as exc:
                raw_outcome = self._unchanged_outcome(
                    solution_before,
                    "operator raised an exception",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
            runtime_ms = (self.clock() - start_time) * 1000.0
            outcome = self._sanitize_outcome(
                raw_outcome,
                solution_before,
                instance,
            )
            try:
                candidate_evaluation = self.adapter.evaluator.evaluate(
                    outcome.solution,
                    instance,
                )
            except Exception as exc:
                outcome = self._unchanged_outcome(
                    solution_before,
                    "candidate evaluation failed",
                    exception_type=type(exc).__name__,
                    exception_message=str(exc),
                )
                candidate_evaluation = current_evaluation

            temperature = self.acceptance.temperature(
                iteration,
                self.budget.max_iterations,
                cost_scale,
            )
            accepted = bool(
                outcome.success
                and self.acceptance.accept(
                    float(current_evaluation.scalar_cost),
                    float(candidate_evaluation.scalar_cost),
                    temperature,
                    acceptance_rng,
                )
            )
            best_before = best_evaluation
            if accepted:
                current_solution = self.adapter.codec.clone(outcome.solution)
                current_evaluation = candidate_evaluation
            created_new_best = bool(
                accepted
                and float(current_evaluation.scalar_cost)
                < float(best_evaluation.scalar_cost) - 1e-12
            )
            if created_new_best:
                best_solution = self.adapter.codec.clone(current_solution)
                best_evaluation = current_evaluation
                stagnation_count = 0
            else:
                stagnation_count += 1

            immediate_reward = float(
                context_before.current_cost - candidate_evaluation.scalar_cost
            )
            recent_improvements.append(immediate_reward)
            recent_acceptances.append(accepted)
            del recent_improvements[
                : max(0, len(recent_improvements) - self.budget.recent_window)
            ]
            del recent_acceptances[
                : max(0, len(recent_acceptances) - self.budget.recent_window)
            ]
            context_after = SearchContext(
                iteration=iteration + 1,
                max_iterations=self.budget.max_iterations,
                current_evaluation=current_evaluation,
                best_evaluation=best_evaluation,
                stagnation_count=stagnation_count,
                recent_improvements=tuple(recent_improvements),
                recent_acceptances=tuple(recent_acceptances),
                last_created_new_best=created_new_best,
            )
            step = SearchStep(
                iteration=iteration,
                operator_id=operator_id,
                operator_name=operator_name,
                solution_before=solution_before,
                candidate_solution=self.adapter.codec.clone(outcome.solution),
                current_solution_after=self.adapter.codec.clone(current_solution),
                evaluation_before=cast(
                    ObjectiveEvaluation, context_before.current_evaluation
                ),
                candidate_evaluation=candidate_evaluation,
                current_evaluation_after=current_evaluation,
                best_evaluation_before=best_before,
                best_evaluation_after=best_evaluation,
                context_before=context_before,
                context_after=context_after,
                operator_outcome=outcome,
                accepted=accepted,
                created_new_best=created_new_best,
                temperature=temperature,
                runtime_ms=runtime_ms,
            )
            steps.append(step)
            if on_step is not None:
                on_step(step, operator)
            last_created_new_best = created_new_best

        return SearchResult(
            initial_solution=self.adapter.codec.clone(initial),
            final_solution=self.adapter.codec.clone(current_solution),
            best_solution=self.adapter.codec.clone(best_solution),
            initial_evaluation=initial_evaluation,
            final_evaluation=current_evaluation,
            best_evaluation=best_evaluation,
            steps=tuple(steps),
            accepted_count=sum(step.accepted for step in steps),
        )

    def _sanitize_outcome(
        self,
        raw_outcome: object,
        parent: SolutionT,
        instance: InstanceT,
    ) -> OperatorOutcome[SolutionT]:
        if not isinstance(raw_outcome, OperatorOutcome):
            return self._unchanged_outcome(
                parent, "operator returned an invalid result type"
            )
        try:
            candidate = self.adapter.codec.canonicalize(raw_outcome.solution)
            violations = self.adapter.guard.validate_structure(candidate, instance)
        except Exception:
            return self._unchanged_outcome(
                parent, "operator returned an invalid solution"
            )
        if violations:
            return self._unchanged_outcome(
                parent, "operator returned an invalid solution"
            )
        return OperatorOutcome(
            solution=self.adapter.codec.clone(candidate),
            changed_items=tuple(raw_outcome.changed_items),
            success=bool(raw_outcome.success),
            metadata=dict(raw_outcome.metadata),
            failure_reason=raw_outcome.failure_reason,
        )

    def _unchanged_outcome(
        self,
        solution: SolutionT,
        reason: str,
        **metadata: JsonValue,
    ) -> OperatorOutcome[SolutionT]:
        return OperatorOutcome(
            solution=self.adapter.codec.clone(solution),
            success=False,
            metadata={"status": "no_change", "reason": reason, **metadata},
            failure_reason=reason,
        )


__all__ = ["GenericSearchKernel", "StepCallback"]

