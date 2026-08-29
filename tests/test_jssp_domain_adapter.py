from __future__ import annotations

import numpy as np

from operator_evolution_core.search import SearchContext

from jssp_operator_evolution.adapter import create_jssp_domain_adapter
from jssp_operator_evolution.models import JobShopInstance, JobShopSolution, Operation
from jssp_operator_evolution.schedule import decode_schedule


def tiny_instance() -> JobShopInstance:
    jobs = (
        (Operation(machine=0, duration=3), Operation(machine=1, duration=2)),
        (Operation(machine=1, duration=2), Operation(machine=0, duration=1)),
    )
    return JobShopInstance.create(
        instance_id="tiny-2x2",
        jobs=jobs,
        machines=2,
        source="test",
        source_family="unit",
    )


def test_deterministic_schedule_decoder_and_objective() -> None:
    instance = tiny_instance()
    solution = JobShopSolution(sequence=(0, 1, 0, 1))

    first = decode_schedule(solution, instance)
    second = decode_schedule(solution, instance)

    assert first == second
    assert first.makespan == 5
    assert first.total_machine_idle == 2
    assert first.critical_path_length == 5
    assert first.feasible

    evaluation = create_jssp_domain_adapter().evaluator.evaluate(solution, instance)
    assert evaluation.scalar_cost == 5.0
    assert evaluation.components == {
        "makespan": 5.0,
        "machine_total_idle": 2.0,
        "critical_path_length": 5.0,
    }
    assert evaluation.feasible


def test_codec_guard_hash_and_clone_contracts() -> None:
    adapter = create_jssp_domain_adapter()
    instance = tiny_instance()
    solution = JobShopSolution(sequence=(0, 1, 0, 1))

    clone = adapter.codec.clone(solution)
    assert clone == solution
    assert clone is not solution
    assert adapter.codec.to_json(solution) == [0, 1, 0, 1]
    assert adapter.codec.stable_hash(solution) == adapter.codec.stable_hash(clone)
    assert adapter.codec.canonicalize([0, 1, 0, 1]) == solution
    assert adapter.guard.validate_structure(solution, instance) == []

    violations = adapter.guard.validate_structure(
        JobShopSolution(sequence=(0, 0, 0, 2)), instance
    )
    assert any("job IDs" in violation for violation in violations)
    assert any("multiplicity" in violation for violation in violations)


def test_initializer_features_and_trace_are_deterministic_and_finite() -> None:
    adapter = create_jssp_domain_adapter()
    instance = tiny_instance()
    first = adapter.initializer.initialize(instance, np.random.default_rng(8))
    second = adapter.initializer.initialize(instance, np.random.default_rng(8))
    assert first == second
    assert adapter.guard.validate_structure(first, instance) == []

    evaluation = adapter.evaluator.evaluate(first, instance)
    features = adapter.features.extract(first, instance, evaluation)
    assert set(features) >= {
        "critical_path_ratio",
        "bottleneck_machine_utilization",
        "machine_load_imbalance",
        "critical_block_count",
        "operation_displacement",
        "relative_initial_improvement",
    }
    assert all(np.isfinite(value) for value in features.values())

    snapshot = adapter.trace_encoder.snapshot(
        first, instance, evaluation, SearchContext()
    )
    assert snapshot["instance"]["instance_id"] == "tiny-2x2"
    assert snapshot["objective"] == evaluation.scalar_cost
    assert snapshot["solution_hash"] == adapter.codec.stable_hash(first)


def test_invalid_direct_evaluation_reports_multiplicity_and_unscheduled() -> None:
    adapter = create_jssp_domain_adapter()
    instance = tiny_instance()
    evaluation = adapter.evaluator.evaluate(
        JobShopSolution(sequence=(0, 0, 0, 9)), instance
    )

    assert not evaluation.feasible
    assert evaluation.violations["multiplicity"] > 0
    assert evaluation.violations["unscheduled_operations"] > 0
