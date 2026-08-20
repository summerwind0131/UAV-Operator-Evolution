from __future__ import annotations

import pytest
from pydantic import ValidationError

from uav_operator_evolution.config import ExperimentConfig, load_config


def test_legacy_yaml_defaults_to_heuristic_without_sdk_requirements() -> None:
    config = load_config("configs/smoke.yaml")
    assert config.agent.designer_mode == "heuristic"
    assert config.agent.provider == "mock"
    assert config.agent.remote_tracing is False
    assert config.agent.trace_include_sensitive_data is False
    assert config.agent.llm_call.max_output_tokens == 4096
    assert config.agent.agent_budget.max_tool_calls == 12


def test_agent_smoke_configuration_is_bounded() -> None:
    config = load_config("configs/agent_smoke.yaml")
    assert config.agent.designer_mode == "single_agent"
    assert config.maps.train.count == 2
    assert config.maps.validation.count == 2
    assert config.maps.test.count == 1
    assert config.search.train_iterations == 16
    assert config.search.validation_iterations == 12
    assert config.search.test_iterations == 12
    assert config.evolution.candidates_per_generation == 1
    assert config.evolution.runtime_validation_repetitions == 4
    assert config.evolution.min_runtime_effective_call_rate == 0.10


def test_agent_configuration_rejects_unknown_fields() -> None:
    payload = load_config("configs/smoke.yaml").model_dump(mode="python")
    payload["agent"]["allow_shell"] = True
    with pytest.raises(ValidationError, match="allow_shell"):
        ExperimentConfig.model_validate(payload)


def test_runtime_validation_repetitions_must_support_balanced_abba() -> None:
    payload = load_config("configs/smoke.yaml").model_dump(mode="python")
    payload["evolution"]["runtime_validation_repetitions"] = 3
    with pytest.raises(ValidationError, match="must be even"):
        ExperimentConfig.model_validate(payload)
