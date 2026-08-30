from __future__ import annotations

from pathlib import Path

import numpy as np

from operator_evolution_core.diagnosis import OperatorDiagnoser
from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    SearchBudget,
    SimulatedAnnealingAcceptance,
)
from operator_evolution_core.trajectory import TrajectoryRecorder
from operator_evolution_core.validation import FitnessPolicy

from jssp_operator_evolution.adapter import create_jssp_domain_adapter
from jssp_operator_evolution.baselines import (
    adjacent_swap_hill_climb,
    random_sequence,
    spt_dispatch,
)
from jssp_operator_evolution.data import build_jssp_splits
from jssp_operator_evolution.operators import (
    JSSPOperatorCompiler,
    initial_operator_population,
    initial_operator_specs,
)
from jssp_operator_evolution.trajectory import JSSPTrajectorySink
from jssp_operator_evolution.validation import JSSPCandidateValidator

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"


def test_sanity_baselines_are_valid_deterministic_and_hill_climb_nonworsening() -> None:
    instance = build_jssp_splits(RAW).open_train()[0]
    adapter = create_jssp_domain_adapter()
    random_a = random_sequence(instance, 7)
    random_b = random_sequence(instance, 7)
    spt_a = spt_dispatch(instance)
    spt_b = spt_dispatch(instance)
    improved = adjacent_swap_hill_climb(instance, random_a, max_iterations=64)

    assert random_a == random_b
    assert spt_a == spt_b
    for solution in (random_a, spt_a, improved):
        assert adapter.guard.validate_structure(solution, instance) == []
    assert adapter.evaluator.evaluate(improved, instance).scalar_cost <= adapter.evaluator.evaluate(random_a, instance).scalar_cost


def test_shared_trace_recorder_and_diagnoser_accept_jssp_three_state_traces() -> None:
    instance = build_jssp_splits(RAW).open_train()[0]
    adapter = create_jssp_domain_adapter()
    population = initial_operator_population()
    with TrajectoryRecorder(":memory:") as recorder:
        sink = JSSPTrajectorySink(
            recorder,
            run_id="jssp-trace-test",
            instance=instance,
            seed=31,
        )
        result = GenericSearchKernel(
            adapter=adapter,
            operators=population,
            scheduler=BlockRandomRoundRobinScheduler(),
            acceptance=SimulatedAnnealingAcceptance(),
            budget=SearchBudget(max_iterations=16),
            clock=lambda: 0.0,
        ).run(instance, np.random.default_rng(31), on_step=sink)

        traces = recorder.update_delayed_rewards((2, 4))
        profiles = OperatorDiagnoser(recorder).diagnose()

    assert len(traces) == result.iterations == 16
    assert traces[0].before_state["instance"]["instance_id"] == instance.instance_id
    assert traces[0].candidate_state["objective_components"]["makespan"] >= 0
    assert traces[0].accepted_state["solution_hash"]
    assert profiles


def test_fixed_budget_paired_candidate_validation_uses_deterministic_v2() -> None:
    validation = build_jssp_splits(RAW).open_validation()[:2]
    population = initial_operator_population()
    parent = initial_operator_specs()[0]
    payload = parent.model_dump(mode="json")
    payload.update(
        {
            "operator_id": "candidate-random-adjacent-swap",
            "name": "Candidate random adjacent swap",
            "parent_ids": [parent.operator_id],
        }
    )
    candidate = JSSPOperatorCompiler().compile(payload)
    validator = JSSPCandidateValidator(
        search_calls=16,
        runtime_repetitions=2,
        clock=lambda: 0.0,
    )

    report = validator.validate(
        validation,
        population,
        parent.operator_id,
        candidate,
        generation=1,
        candidate_index=0,
        root_run_id="jssp-paired-test",
    )

    assert validator.generic.fitness_policy is FitnessPolicy.DETERMINISTIC_V2
    assert len(report.outcomes) == 2
    assert all(outcome.parent_best_cost == outcome.candidate_best_cost for outcome in report.outcomes)
    assert all(outcome.runtime_repetitions == 2 for outcome in report.outcomes)
    assert not report.retained
