"""Experiment workflow for the AFL paper reproduction on UAV maps."""

from __future__ import annotations

from pathlib import Path

from ..afl_uav.coordinator import AFLUAVCoordinator
from ..afl_uav.buffer import SolverBuffer
from ..afl_uav.mock_solver import afl_uav_mock_factory
from ..afl_uav.models import AFLUAVLimits, AFLUAVRunResult
from ..agents.providers import LLMCallConfig, MockLLMProvider
from ..config import ExperimentConfig
from ..environment.environment import Environment2D
from ..path.evaluator import PathEvaluator
from ..path.models import ObjectiveWeights
from ..runtime import RunPaths
from .common import update_latest, write_json


def run_afl_uav_workflow(
    config: ExperimentConfig,
    paths: RunPaths,
    environment: Environment2D,
    *,
    provider: str = "mock",
    model: str | None = None,
    iterations: int = 100,
    execute_untrusted_code: bool = False,
) -> AFLUAVRunResult:
    """Generate, optionally execute, and independently verify one UAV solver."""

    if provider != "mock":
        raise ValueError(
            "real AFL-UAV providers must use generate-afl-uav-candidate followed "
            "by hash-approved freeze-afl-uav"
        )
    llm_provider = MockLLMProvider(factory=afl_uav_mock_factory)
    execute_generated = True

    configured = config.agent.llm_call
    call_config = LLMCallConfig(
        model=model,
        timeout_seconds=configured.timeout_seconds,
        max_retries=configured.max_retries,
        max_output_tokens=max(16_384, configured.max_output_tokens),
        max_total_tokens=max(250_000, configured.max_total_tokens),
    )
    coordinator = AFLUAVCoordinator(
        provider=llm_provider,
        call_config=call_config,
        evaluator=PathEvaluator(config.objective),
        limits=AFLUAVLimits(
            execution_timeout_seconds=max(20.0, configured.timeout_seconds),
            max_source_chars=500_000,
        ),
        solver_buffer=SolverBuffer(config.output.results_dir / "afl_uav_solver_buffer"),
    )
    result = coordinator.run(
        run_id=paths.run_id,
        environment=environment,
        objective_weights=ObjectiveWeights.model_validate(config.objective.model_dump()),
        output_dir=paths.result_dir,
        iterations=iterations,
        grid_resolution=config.maps.grid_resolution,
        max_waypoints=config.dsl.max_waypoints,
        execute_generated=execute_generated,
    )
    write_json(paths.result_dir / "afl_uav_result.json", result)
    update_latest(config, paths.run_id, paths.result_dir)
    return result


def load_afl_uav_environment(
    map_path: str | Path | None,
    default_environment: Environment2D,
) -> Environment2D:
    if map_path is None:
        return default_environment
    return Environment2D.load_json(map_path)


__all__ = ["load_afl_uav_environment", "run_afl_uav_workflow"]
