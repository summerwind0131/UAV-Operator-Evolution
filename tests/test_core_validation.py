from __future__ import annotations

import inspect
from dataclasses import dataclass

import numpy as np

from operator_evolution_core.contracts import DomainAdapter
from operator_evolution_core.search import SearchBudget
from operator_evolution_core.validation import (
    ArmMeasurement,
    FitnessPolicy,
    GenericPairedCandidateValidator,
    PairedOutcome,
    abba_timing_order,
    compute_fitness,
    decide_retention,
)


@dataclass(frozen=True)
class _RetentionSettings:
    min_global_gain: float = 0.02
    min_specialist_gain: float = 0.05
    min_feasibility_gain: float = 0.10
    min_runtime_reduction: float = 0.25
    min_runtime_effective_call_rate: float = 0.10
    require_bootstrap_ci: bool = False


def _runtime_only_outcome() -> PairedOutcome:
    return PairedOutcome(
        instance_id="validation-01",
        difficulty="small",
        parent_best_cost=100.0,
        candidate_best_cost=100.0,
        parent_feasible=True,
        candidate_feasible=True,
        parent_runtime_ms=10.0,
        candidate_runtime_ms=5.0,
        candidate_operator_call_count=20,
        candidate_operator_changed_call_count=4,
        candidate_operator_accepted_call_count=2,
    )


def test_generic_validator_api_exposes_validation_instances_only() -> None:
    signature = inspect.signature(GenericPairedCandidateValidator.validate)
    assert "validation_instances" in signature.parameters
    assert "operator_population" in signature.parameters
    assert "datasets" not in signature.parameters
    assert "test_instances" not in signature.parameters


def test_instance_id_keeps_the_uav_v1_map_id_projection() -> None:
    outcome = _runtime_only_outcome()
    assert outcome.instance_id == "validation-01"
    assert outcome.map_id == "validation-01"
    assert outcome.model_dump()["map_id"] == "validation-01"
    assert "instance_id" not in outcome.model_dump()
    assert outcome.context_label == "small"
    assert outcome.model_dump()["difficulty"] == "small"
    assert "context_label" not in outcome.model_dump()


def test_abba_timing_schedule_is_balanced_and_rejects_zero() -> None:
    assert abba_timing_order(4) == (
        "parent_first",
        "candidate_first",
        "candidate_first",
        "parent_first",
    )
    try:
        abba_timing_order(0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:  # pragma: no cover - documents the fail-closed contract
        raise AssertionError("zero repetitions must be rejected")


def test_deterministic_fitness_and_retention_ignore_wall_clock_advantage() -> None:
    rows = [
        {
            "operator_name": "parent",
            "cost": 1.0,
            "feasible": 1.0,
            "delayed": 1.0,
            "worst_context": 1.0,
            "runtime": 10.0,
        },
        {
            "operator_name": "candidate",
            "cost": 1.0,
            "feasible": 1.0,
            "delayed": 1.0,
            "worst_context": 1.0,
            "runtime": 1.0,
        },
    ]
    deterministic = compute_fitness(rows, policy=FitnessPolicy.DETERMINISTIC_V2)
    legacy = compute_fitness(rows, policy=FitnessPolicy.UAV_LEGACY_V1)
    assert deterministic["parent"] == deterministic["candidate"]
    assert legacy["parent"] != legacy["candidate"]

    legacy_report = decide_retention(
        "parent",
        "candidate",
        [_runtime_only_outcome()],
        _RetentionSettings(),
        fitness_policy=FitnessPolicy.UAV_LEGACY_V1,
    )
    deterministic_report = decide_retention(
        "parent",
        "candidate",
        [_runtime_only_outcome()],
        _RetentionSettings(),
        fitness_policy=FitnessPolicy.DETERMINISTIC_V2,
    )
    assert legacy_report.retained
    assert legacy_report.retention_reasons == ["runtime reduction"]
    assert not deterministic_report.retained
    assert deterministic_report.retention_reasons == [
        "no pre-registered effect threshold met"
    ]
    assert deterministic_report.median_runtime_reduction == 0.5


@dataclass(frozen=True)
class _Instance:
    instance_id: str
    difficulty: str


class _Initializer:
    def __init__(self) -> None:
        self.initialized: list[tuple[str, int]] = []

    def initialize(
        self, instance: _Instance, rng: np.random.Generator
    ) -> tuple[int, ...]:
        solution = (int(rng.integers(0, 1_000_000)),)
        self.initialized.append((instance.instance_id, solution[0]))
        return solution


def _derive_seed(*parts: object) -> int:
    return sum(
        (index + 1) * sum(str(part).encode("utf-8"))
        for index, part in enumerate(parts)
    ) % (2**32)


def test_generic_validator_uses_crn_abba_and_separate_evidence_runs() -> None:
    initializer = _Initializer()
    calls: list[tuple[str, int, tuple[int, ...], bool]] = []

    def run_arm(
        population,
        instance,
        initial,
        seed,
        target_id,
        generation,
        run_id,
        persist_evidence,
    ) -> ArmMeasurement:
        del population, instance, generation, run_id
        calls.append((target_id, seed, initial, persist_evidence))
        candidate = target_id == "candidate"
        return ArmMeasurement(
            best_cost=90.0 if candidate else 100.0,
            feasible=True,
            total_runtime_ms=2.0 if candidate else 3.0,
            operator_runtime_ms=0.2 if candidate else 0.3,
            operator_call_count=2,
            operator_changed_call_count=1,
            operator_accepted_call_count=1,
        )

    adapter = DomainAdapter(
        domain_id="validation-test",
        initializer=initializer,
        evaluator=object(),
        features=object(),
        codec=object(),
        guard=object(),
        trace_encoder=object(),
    )
    validator = GenericPairedCandidateValidator(
        adapter=adapter,
        budget=SearchBudget(max_iterations=8, recent_window=2),
        retention_config=_RetentionSettings(),
        master_seed=17,
        seed_deriver=_derive_seed,
        arm_runner=run_arm,
        operator_id=str,
        instance_id=lambda instance: instance.instance_id,
        context_label=lambda instance: instance.difficulty,
        runtime_repetitions=4,
        fitness_policy=FitnessPolicy.DETERMINISTIC_V2,
    )

    report = validator.validate(
        [_Instance("validation-01", "small")],
        ["parent", "other"],
        "parent",
        "candidate",
        generation=2,
        candidate_index=3,
        root_run_id="core-validation",
        persist_evidence=True,
    )

    assert report.retained
    assert initializer.initialized and len(initializer.initialized) == 1
    assert [call[0] for call in calls[:8]] == [
        "parent",
        "candidate",
        "candidate",
        "parent",
        "candidate",
        "parent",
        "parent",
        "candidate",
    ]
    assert len({call[1] for call in calls}) == 1
    assert len({call[2] for call in calls}) == 1
    assert all(not call[3] for call in calls[:8])
    assert [call[3] for call in calls[8:]] == [True, True]
    assert report.outcomes[0].timing_order == list(abba_timing_order(4))
