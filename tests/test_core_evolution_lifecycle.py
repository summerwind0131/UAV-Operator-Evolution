from __future__ import annotations

from dataclasses import replace

import pytest

from operator_evolution_core.evolution import (
    EvolutionManagerDependencies,
    EvolutionSplitCapabilities,
    PopulationSeed,
)
from uav_operator_evolution.agents.heuristic_designer import HeuristicDesigner
from uav_operator_evolution.agents.orchestrator import OperatorDesignOrchestrator
from uav_operator_evolution.config import load_config
from uav_operator_evolution.domain import UAVDomainAdapter, UAVDomainKit
from uav_operator_evolution.environment.generator import MapGenerator
from uav_operator_evolution.evolution.candidate_validator import (
    FixedBudgetCandidateValidator,
)
from uav_operator_evolution.evolution.manager import OperatorEvolutionManager
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.operators.compiler import OperatorCompiler
from uav_operator_evolution.operators.registry import default_manual_operators
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.path.models import ObjectiveWeights


class _ArtifactSink:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def emit(self, event: str, payload) -> None:
        self.events.append((event, dict(payload)))


def _tiny_config():
    config = load_config("configs/smoke.yaml").model_copy(deep=True)
    config.search.train_iterations = 4
    config.search.validation_iterations = 2
    config.search.test_iterations = 2
    config.evolution.generations = 1
    config.evolution.candidates_per_generation = 1
    return config


def _datasets(config):
    generator = MapGenerator(
        config.seed,
        grid_resolution=5.0,
        generation_attempts=20,
    )
    return {
        "train": [
            generator.generate_map(
                "train", 0, "sparse", width=50, height=50, safety_distance=1
            )
        ],
        "validation": [
            generator.generate_map(
                "validation",
                0,
                "medium",
                width=50,
                height=50,
                safety_distance=1,
            )
        ],
        "test": [
            generator.generate_map(
                "test", 0, "dense", width=55, height=55, safety_distance=1
            )
        ],
    }


def test_test_split_requires_the_active_population_freeze_receipt() -> None:
    splits = EvolutionSplitCapabilities(
        train=["train"],
        validation=["validation"],
        test=["test"],
    )
    assert splits.open_train() == ("train",)
    assert splits.open_validation() == ("validation",)
    with pytest.raises(PermissionError, match="population freeze receipt"):
        splits.open_test()

    receipt = splits.freeze_population(["operator-a"], "a" * 64)
    with pytest.raises(PermissionError, match="population freeze receipt"):
        splits.open_test(replace(receipt))
    assert splits.open_test(receipt) == ("test",)


def test_default_manager_constructor_still_auto_assembles_uav(tmp_path) -> None:
    config = _tiny_config()
    config.output.results_dir = tmp_path / "results"
    manager = OperatorEvolutionManager(config)
    assert manager.database_path == (
        tmp_path / "results" / "operator_evolution.sqlite"
    )
    assert manager.domain_adapter.domain_id == manager.domain_kit.domain_id
    assert isinstance(manager.domain_kit, UAVDomainKit)


def test_injected_dependencies_drive_split_freeze_and_artifact_lifecycle(
    tmp_path,
) -> None:
    config = _tiny_config()
    evaluator = PathEvaluator(
        ObjectiveWeights.model_validate(config.objective.model_dump())
    )
    compiler = OperatorCompiler(config.dsl)
    adapter = UAVDomainAdapter(
        evaluator,
        initializer_grid_resolution=config.maps.grid_resolution,
    )
    kit = UAVDomainKit(compiler)
    sink = _ArtifactSink()
    dependencies = EvolutionManagerDependencies(
        domain_adapter=adapter,
        domain_kit=kit,
        population_factory=lambda: PopulationSeed(
            operators=tuple(default_manual_operators()),
            ir_by_id=manual_operator_specs(),
        ),
        candidate_validator=FixedBudgetCandidateValidator(config, evaluator),
        designer=HeuristicDesigner(),
        orchestrator_factory=OperatorDesignOrchestrator,
        artifact_sink=sink,
    )
    manager = OperatorEvolutionManager(
        config,
        tmp_path / "injected.sqlite",
        dependencies=dependencies,
    )
    splits = EvolutionSplitCapabilities.from_mapping(_datasets(config))

    result = manager.run(splits, "injected-step7")

    assert manager.dependencies is dependencies
    assert len(result.final_population) == 8
    assert [event for event, _ in sink.events] == [
        "run_started",
        "generation_completed",
        "population_frozen",
        "run_completed",
    ]
    assert all(
        payload["test_opened"] is False
        for _, payload in sink.events[:3]
    )
    assert sink.events[-1][1]["test_opened"] is True
    assert sink.events[-1][1]["test_instances"] == 1
