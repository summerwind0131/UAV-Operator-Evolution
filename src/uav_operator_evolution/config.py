"""Validated YAML configuration for experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .reproducibility import stable_hash


class StrictModel(BaseModel):
    """Base model that rejects misspelled and unknown configuration fields."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ObjectiveConfig(StrictModel):
    length: float = Field(1.0, ge=0)
    collision: float = Field(1000.0, ge=0)
    smoothness: float = Field(5.0, ge=0)
    risk: float = Field(10.0, ge=0)
    waypoint: float = Field(0.5, ge=0)


class MapSplitConfig(StrictModel):
    count: int = Field(6, ge=1)
    difficulties: list[
        Literal[
            "sparse",
            "medium",
            "dense",
            "corridor",
            "clustered",
            "rooms_maze",
            "mixed",
        ]
    ]
    width: float = Field(100.0, gt=10)
    height: float = Field(100.0, gt=10)
    safety_distance: float = Field(2.0, ge=0)


class MapsConfig(StrictModel):
    train: MapSplitConfig
    validation: MapSplitConfig
    test: MapSplitConfig
    grid_resolution: float = Field(4.0, gt=0)
    generation_attempts: int = Field(30, ge=1, le=200)


class SearchConfig(StrictModel):
    train_iterations: int = Field(96, ge=1)
    validation_iterations: int = Field(64, ge=1)
    test_iterations: int = Field(96, ge=1)
    temperature_start_ratio: float = Field(0.05, gt=0)
    temperature_end_ratio: float = Field(0.001, gt=0)
    recent_window: int = Field(10, ge=1)


class DSLConfig(StrictModel):
    max_conditions: int = Field(8, ge=1, le=32)
    max_transformations: int = Field(8, ge=1, le=32)
    max_repeat: int = Field(3, ge=1, le=10)
    max_parents: int = Field(4, ge=1, le=8)
    max_waypoints: int = Field(128, ge=3, le=2048)
    max_added_waypoints: int = Field(16, ge=1, le=256)
    deadline_ms: float = Field(100.0, gt=0, le=10_000)


class EvolutionConfig(StrictModel):
    generations: int = Field(2, ge=1, le=20)
    candidates_per_generation: int = Field(2, ge=1, le=8)
    population_size: int = Field(8, ge=8, le=32)
    min_global_gain: float = Field(0.02, ge=0)
    min_specialist_gain: float = Field(0.05, ge=0)
    min_feasibility_gain: float = Field(0.10, ge=0, le=1)
    min_runtime_reduction: float = Field(0.25, ge=0, le=1)
    runtime_validation_repetitions: int = Field(4, ge=2, le=20)
    min_runtime_effective_call_rate: float = Field(0.10, gt=0, le=1)
    require_bootstrap_ci: bool = False

    @field_validator("runtime_validation_repetitions")
    @classmethod
    def runtime_repetitions_are_balanced(cls, repetitions: int) -> int:
        if repetitions % 2 != 0:
            raise ValueError(
                "runtime_validation_repetitions must be even for balanced ABBA timing"
            )
        return repetitions


class DiagnosticsConfig(StrictModel):
    delayed_horizons: list[int] = Field(default_factory=lambda: [5, 10, 20])
    minimum_context_samples: int = Field(3, ge=1)
    representative_cases: int = Field(3, ge=1, le=20)
    counterfactual_states: int = Field(4, ge=0, le=100)
    counterfactual_operators: int = Field(4, ge=2, le=16)

    @field_validator("delayed_horizons")
    @classmethod
    def horizons_are_unique(cls, horizons: list[int]) -> list[int]:
        if any(h <= 0 for h in horizons):
            raise ValueError("delayed horizons must be positive")
        return sorted(set(horizons))


class AgentDesignBudgetConfig(StrictModel):
    """Bound the amount of evidence exposed to one design attempt."""

    max_parent_specs: int = Field(4, ge=1, le=8)
    max_context_evidence: int = Field(8, ge=0, le=64)
    max_failure_evidence: int = Field(8, ge=0, le=64)
    max_synergy_evidence: int = Field(8, ge=0, le=64)
    max_counterfactual_evidence: int = Field(8, ge=0, le=64)
    max_success_cases: int = Field(3, ge=0, le=20)
    max_failure_cases: int = Field(3, ge=0, le=20)
    max_bundle_chars: int = Field(60_000, ge=2_000, le=1_000_000)
    max_candidate_specs: int = Field(1, ge=1, le=8)


class LLMCallSettings(StrictModel):
    """Provider-independent limits for one structured model stage."""

    timeout_seconds: float = Field(60.0, gt=0, le=600)
    max_retries: int = Field(2, ge=0, le=10)
    max_output_tokens: int = Field(4_096, ge=1, le=100_000)
    max_total_tokens: int = Field(20_000, ge=1, le=1_000_000)


class AgentBudgetConfig(StrictModel):
    """Hard local limits applied in addition to any SDK limits."""

    max_turns: int = Field(6, ge=1, le=32)
    max_tool_calls: int = Field(12, ge=0, le=128)
    max_candidate_specs: int = Field(2, ge=1, le=8)
    max_revisions: int = Field(1, ge=0, le=4)
    max_smoke_tests: int = Field(2, ge=0, le=16)


class AgentConfig(StrictModel):
    """Optional Phase-8 design layer; deterministic heuristic remains the default."""

    designer_mode: Literal[
        "heuristic", "llm_single_call", "llm_staged", "single_agent", "multi_agent"
    ] = "heuristic"
    provider: Literal["mock", "openai"] = "mock"
    memory_mode: Literal[
        "none", "score_only", "mechanism", "mechanism_and_lineage"
    ] = "mechanism_and_lineage"
    feedback_mode: Literal["score_only", "diagnosis"] = "diagnosis"
    review_mode: Literal["none", "rule_based", "llm"] = "rule_based"
    remote_tracing: bool = False
    trace_include_sensitive_data: bool = False
    design_budget: AgentDesignBudgetConfig = Field(default_factory=AgentDesignBudgetConfig)
    llm_call: LLMCallSettings = Field(default_factory=LLMCallSettings)
    agent_budget: AgentBudgetConfig = Field(default_factory=AgentBudgetConfig)

    @model_validator(mode="after")
    def validate_offline_multi_agent_contract(self) -> "AgentConfig":
        """Require enough shared budget to execute the fixed four-role topology."""

        if self.designer_mode != "multi_agent":
            return self

        violations: list[str] = []
        if self.provider != "mock":
            violations.append("provider='mock'")
        if self.design_budget.max_candidate_specs < 2:
            violations.append("design_budget.max_candidate_specs>=2")
        if self.agent_budget.max_turns < 4:
            violations.append("agent_budget.max_turns>=4")
        if self.agent_budget.max_tool_calls < 12:
            violations.append("agent_budget.max_tool_calls>=12")
        if self.agent_budget.max_candidate_specs < 2:
            violations.append("agent_budget.max_candidate_specs>=2")
        if self.agent_budget.max_smoke_tests < 2:
            violations.append("agent_budget.max_smoke_tests>=2")
        if self.agent_budget.max_revisions != 0:
            violations.append("agent_budget.max_revisions=0")
        if violations:
            requirements = ", ".join(violations)
            raise ValueError(
                "designer_mode='multi_agent' is offline-only and requires "
                f"{requirements}"
            )
        return self


class PlanningBenchmarkConfig(StrictModel):
    """Fair-budget settings for the standalone planner benchmark."""

    benchmark_id: str = "uav2d-v1"
    time_limit_seconds: float = Field(1.0, gt=0, le=600)
    max_objective_evaluations: int = Field(2_000, ge=1)
    stochastic_repetitions: int = Field(5, ge=1, le=100)
    waypoint_count: int = Field(10, ge=1, le=128)
    population_size: int = Field(32, ge=4, le=512)
    planners: list[str] = Field(
        default_factory=lambda: [
            "dijkstra",
            "astar",
            "theta_star",
            "rrt",
            "rrt_star",
            "prm",
            "ga",
            "pso",
            "de",
            "aco_acor",
            "afl_uav_mock",
        ]
    )


class OutputConfig(StrictModel):
    data_dir: Path = Path("data/generated")
    results_dir: Path = Path("artifacts/results")
    figures_dir: Path = Path("artifacts/figures")
    export_jsonl: bool = True


class ExperimentConfig(StrictModel):
    name: str = "smoke"
    seed: int = Field(42, ge=0)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    maps: MapsConfig
    search: SearchConfig = Field(default_factory=SearchConfig)
    diagnostics: DiagnosticsConfig = Field(default_factory=DiagnosticsConfig)
    dsl: DSLConfig = Field(default_factory=DSLConfig)
    evolution: EvolutionConfig = Field(default_factory=EvolutionConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    planning_benchmark: PlanningBenchmarkConfig = Field(
        default_factory=PlanningBenchmarkConfig
    )
    output: OutputConfig = Field(default_factory=OutputConfig)

    @property
    def config_hash(self) -> str:
        return stable_hash(self.model_dump(mode="json"))


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment YAML file."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ExperimentConfig.model_validate(payload)
