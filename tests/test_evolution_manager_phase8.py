from __future__ import annotations

import sqlite3

from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.evolution.manager import OperatorEvolutionManager
from uav_operator_evolution.evolution.validation import PairedOutcome, ValidationReport


def _environment(map_id: str, goal: tuple[float, float], difficulty: str) -> Environment2D:
    return Environment2D(
        map_id=map_id,
        width=30,
        height=30,
        start=(1, 1),
        goal=goal,
        obstacles=[],
        difficulty=difficulty,
        seed=7,
    )


def test_mock_phase8_manager_uses_validation_only_and_replaces_one_slot(
    tmp_path, monkeypatch
) -> None:
    config = load_config("configs/agent_smoke.yaml").model_copy(deep=True)
    config.agent.designer_mode = "llm_single_call"
    config.agent.provider = "mock"
    config.agent.review_mode = "none"
    config.diagnostics.minimum_context_samples = 1
    config.search.train_iterations = 8
    config.search.validation_iterations = 1
    config.search.test_iterations = 1
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    train = _environment("train-smoke", (28, 28), "sparse")
    validation = _environment("validation-only", (27, 28), "medium")
    test = _environment("held-out-test", (28, 27), "dense")
    seen_validation_ids: list[list[str]] = []

    def accept_candidate(
        validator,
        population,
        parent_name,
        candidate,
        validation_environments,
        **kwargs,
    ) -> ValidationReport:
        del validator, population, kwargs
        map_ids = [environment.map_id for environment in validation_environments]
        seen_validation_ids.append(map_ids)
        return ValidationReport(
            parent_operator=parent_name,
            candidate_operator=str(candidate.name),
            safety_passed=True,
            outcomes=[
                PairedOutcome(
                    map_id=map_ids[0],
                    difficulty=validation_environments[0].difficulty,
                    parent_best_cost=100.0,
                    candidate_best_cost=90.0,
                    parent_feasible=True,
                    candidate_feasible=True,
                    parent_runtime_ms=10.0,
                    candidate_runtime_ms=9.0,
                )
            ],
            mean_gain=0.1,
            win_rate=1.0,
            parent_feasibility_rate=1.0,
            candidate_feasibility_rate=1.0,
            retained=True,
            retention_reasons=["global paired gain"],
        )

    monkeypatch.setattr(
        "uav_operator_evolution.evolution.candidate_validator.FixedBudgetCandidateValidator.validate",
        accept_candidate,
    )
    database = tmp_path / "phase8.sqlite"
    manager = OperatorEvolutionManager(config, database)
    result = manager.run(
        {"train": [train], "validation": [validation], "test": [test]},
        "manager-phase8",
    )

    assert seen_validation_ids == [["validation-only"]]
    assert all("held-out-test" not in map_ids for map_ids in seen_validation_ids)
    assert len(result.initial_population) == len(result.final_population) == 8
    assert len(set(result.initial_population) - set(result.final_population)) == 1
    assert len(set(result.final_population) - set(result.initial_population)) == 1
    assert len(result.retained_candidates) == 1
    assert result.generations[0].validations[0].retained is True
    assert result.generations[0].proposals[0].spec.name == result.retained_candidates[0]

    with sqlite3.connect(database) as connection:
        statuses = [
            row[0]
            for row in connection.execute(
                "SELECT status FROM candidate_events ORDER BY rowid"
            ).fetchall()
        ]
    assert statuses == [
        "PROPOSED",
        "SCHEMA_VALID",
        "REVIEWED",
        "COMPILED",
        "SMOKE_PASSED",
        "VALIDATED",
        "ACCEPTED",
    ]
