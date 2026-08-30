from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from pydantic import ValidationError

from operator_evolution_core.proposal import ensure_domain_compatibility
from operator_evolution_core.search import (
    BlockRandomRoundRobinScheduler,
    GenericSearchKernel,
    SearchBudget,
    SearchContext,
    SimulatedAnnealingAcceptance,
)

from jssp_operator_evolution.adapter import create_jssp_domain_adapter
from jssp_operator_evolution.models import JobShopInstance, JobShopSolution, Operation
from jssp_operator_evolution.operators import (
    JSSPDomainKit,
    JSSPOperatorSpec,
    JSSPSmokeFixture,
    initial_operator_population,
    initial_operator_specs,
)


def fixture_instance() -> JobShopInstance:
    jobs = (
        (Operation(machine=0, duration=3), Operation(machine=1, duration=2), Operation(machine=2, duration=4)),
        (Operation(machine=1, duration=2), Operation(machine=2, duration=5), Operation(machine=0, duration=2)),
        (Operation(machine=2, duration=4), Operation(machine=0, duration=1), Operation(machine=1, duration=3)),
    )
    return JobShopInstance.create(
        instance_id="operators-3x3",
        jobs=jobs,
        machines=3,
        source="test",
        source_family="unit",
    )


def fixture_solution() -> JobShopSolution:
    return JobShopSolution(sequence=(0, 1, 2, 0, 1, 2, 0, 1, 2))


def test_fixed_p0_contains_eight_distinct_bounded_operators() -> None:
    specs = initial_operator_specs()
    population = initial_operator_population()

    assert len(specs) == len(population) == 8
    assert len({spec.operator_id for spec in specs}) == 8
    assert {operator.operator_id for operator in population} == {
        spec.operator_id for spec in specs
    }


@pytest.mark.parametrize("operator", initial_operator_population())
def test_all_p0_operators_preserve_input_multiplicity_and_are_seed_deterministic(operator) -> None:
    instance = fixture_instance()
    solution = fixture_solution()
    original = tuple(solution.sequence)

    first = operator.apply(solution, instance, np.random.default_rng(19), SearchContext())
    second = operator.apply(solution, instance, np.random.default_rng(19), SearchContext())

    assert first == second
    assert solution.sequence == original
    assert Counter(first.solution.sequence) == Counter(original)
    assert len(first.solution.sequence) == instance.operation_count


def test_jssp_v1_rejects_unknown_code_and_invalid_capability_combinations() -> None:
    payload = initial_operator_specs()[0].model_dump(mode="json")
    payload["selector"] = {"kind": "execute_python", "source": "pass"}
    with pytest.raises(ValidationError):
        JSSPOperatorSpec.model_validate(payload)

    payload = initial_operator_specs()[0].model_dump(mode="json")
    payload["transform"] = {"kind": "reverse", "max_segment_length": 8}
    with pytest.raises(ValidationError, match="not allowed"):
        JSSPOperatorSpec.model_validate(payload)


def test_domain_kit_compile_smoke_catalog_and_fingerprints() -> None:
    kit = JSSPDomainKit()
    spec = initial_operator_specs()[0]
    parsed = kit.parse_ir(spec.model_dump(mode="json"))
    report = kit.smoke(
        parsed,
        JSSPSmokeFixture(fixture_instance(), fixture_solution(), tuple(range(8))),
    )

    assert report.smoke_passed and report.seeds_tested == 8
    assert kit.compile(parsed).operator_id == spec.operator_id
    assert set(kit.capability_usage(parsed)).issubset(
        {item for group in kit.capability_catalog().values() for item in group}
    )
    assert len(kit.topology_fingerprint(parsed)) == 64
    assert len(kit.behavior_fingerprint(parsed)) == 64
    assert kit.static_safety_score(parsed) == 1.0
    assert kit.builtin_ir(spec.operator_id) == spec
    ensure_domain_compatibility(
        kit, {"domain_id": "jssp", "ir_version": "jssp-v1"}
    )


def test_same_generic_search_kernel_runs_jssp_for_64_calls() -> None:
    kernel = GenericSearchKernel(
        adapter=create_jssp_domain_adapter(),
        operators=initial_operator_population(),
        scheduler=BlockRandomRoundRobinScheduler(),
        acceptance=SimulatedAnnealingAcceptance(),
        budget=SearchBudget(max_iterations=64),
        clock=lambda: 0.0,
    )
    result = kernel.run(
        fixture_instance(),
        np.random.default_rng(20260830),
        initial_solution=fixture_solution(),
    )

    assert result.iterations == 64
    assert result.best_evaluation.scalar_cost <= result.initial_evaluation.scalar_cost
    assert all(step.candidate_evaluation.feasible for step in result.steps)
    assert {step.operator_id for step in result.steps[:8]} == {
        spec.operator_id for spec in initial_operator_specs()
    }
