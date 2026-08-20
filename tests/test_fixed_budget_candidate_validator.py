from __future__ import annotations

import inspect

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.evolution.candidate_validator import FixedBudgetCandidateValidator
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.compiler import OperatorCompiler
from uav_operator_evolution.operators.registry import default_manual_operators
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.path.models import ObjectiveWeights


def _environment() -> Environment2D:
    return Environment2D(
        map_id="validation-only",
        width=30,
        height=30,
        start=(1, 1),
        goal=(29, 29),
        obstacles=[],
        difficulty="dense",
        seed=5,
    )


def test_validator_api_cannot_receive_a_dataset_or_test_split(tmp_path) -> None:
    signature = inspect.signature(FixedBudgetCandidateValidator.validate)
    assert "datasets" not in signature.parameters
    assert "test_environments" not in signature.parameters

    config = load_config("configs/agent_smoke.yaml")
    evaluator = PathEvaluator(ObjectiveWeights.model_validate(config.objective.model_dump()))
    validator = FixedBudgetCandidateValidator(config, evaluator)
    population = default_manual_operators()
    spec = manual_operator_specs()["segment_shift"].model_copy(
        update={"name": "candidate_slot_replacement"}
    )
    candidate = OperatorCompiler(config.dsl).compile(spec)
    report = validator.validate(
        population,
        "segment_shift",
        candidate,
        [_environment()],
        generation=0,
        candidate_index=0,
        recorder=None,
        root_run_id="fixed-budget",
    )
    assert len(report.outcomes) == 1
    outcome = report.outcomes[0]
    assert outcome.map_id == "validation-only"
    assert outcome.runtime_repetitions == 4
    assert outcome.timing_order == [
        "parent_first",
        "candidate_first",
        "candidate_first",
        "parent_first",
    ]
    assert len(outcome.parent_runtime_samples_ms) == 4
    assert len(outcome.candidate_runtime_samples_ms) == 4
    assert len(outcome.parent_operator_runtime_samples_ms) == 4
    assert len(outcome.candidate_operator_runtime_samples_ms) == 4
    assert outcome.candidate_operator_call_count > 0
    assert 0.0 <= outcome.candidate_effective_call_rate <= 1.0
    assert 0.0 <= outcome.candidate_operator_acceptance_rate <= 1.0
    assert report.candidate_operator_call_count == outcome.candidate_operator_call_count


def test_common_random_numbers_make_cost_outcomes_repeatable() -> None:
    config = load_config("configs/agent_smoke.yaml")
    evaluator = PathEvaluator(ObjectiveWeights.model_validate(config.objective.model_dump()))
    validator = FixedBudgetCandidateValidator(config, evaluator)
    population = default_manual_operators()
    spec = manual_operator_specs()["segment_shift"].model_copy(
        update={"name": "candidate_repeatable"}
    )
    candidate = OperatorCompiler(config.dsl).compile(spec)
    arguments = dict(
        population=population,
        parent_name="segment_shift",
        candidate=candidate,
        validation_environments=[_environment()],
        generation=0,
        candidate_index=0,
        recorder=None,
        root_run_id="repeatable",
    )
    first = validator.validate(**arguments)
    second = validator.validate(**arguments)
    assert first.outcomes[0].parent_best_cost == second.outcomes[0].parent_best_cost
    assert first.outcomes[0].candidate_best_cost == second.outcomes[0].candidate_best_cost
    assert first.outcomes[0].parent_feasible == second.outcomes[0].parent_feasible
    assert first.outcomes[0].candidate_feasible == second.outcomes[0].candidate_feasible
