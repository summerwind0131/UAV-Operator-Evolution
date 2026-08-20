from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from uav_operator_evolution.config import ExperimentConfig, load_config


def test_multi_agent_smoke_uses_the_offline_portfolio_budget() -> None:
    config = load_config("configs/multi_agent_smoke.yaml")

    assert config.agent.designer_mode == "multi_agent"
    assert config.agent.provider == "mock"
    assert config.agent.design_budget.max_candidate_specs == 2
    assert config.agent.agent_budget.max_turns == 4
    assert config.agent.agent_budget.max_tool_calls == 12
    assert config.agent.agent_budget.max_candidate_specs == 2
    assert config.agent.agent_budget.max_revisions == 0
    assert config.agent.agent_budget.max_smoke_tests == 2
    assert config.maps.train.count == 2
    assert config.maps.validation.count == 2
    assert config.maps.test.count == 1
    assert config.search.train_iterations == 16
    assert config.search.validation_iterations == 12
    assert config.search.test_iterations == 12


def test_multi_agent_mode_rejects_non_mock_provider_with_clear_error() -> None:
    payload = load_config("configs/multi_agent_smoke.yaml").model_dump(mode="python")
    payload["agent"]["provider"] = "openai"

    with pytest.raises(
        ValidationError,
        match="designer_mode='multi_agent' is offline-only and requires provider='mock'",
    ):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    ("section", "field", "value", "requirement"),
    [
        ("design_budget", "max_candidate_specs", 1, "design_budget.max_candidate_specs>=2"),
        ("agent_budget", "max_turns", 3, "agent_budget.max_turns>=4"),
        ("agent_budget", "max_tool_calls", 11, "agent_budget.max_tool_calls>=12"),
        (
            "agent_budget",
            "max_candidate_specs",
            1,
            "agent_budget.max_candidate_specs>=2",
        ),
        ("agent_budget", "max_smoke_tests", 1, "agent_budget.max_smoke_tests>=2"),
        ("agent_budget", "max_revisions", 1, "agent_budget.max_revisions=0"),
    ],
)
def test_multi_agent_mode_rejects_an_undersized_shared_budget(
    section: str,
    field: str,
    value: int,
    requirement: str,
) -> None:
    payload = load_config("configs/multi_agent_smoke.yaml").model_dump(mode="python")
    payload["agent"][section][field] = value

    with pytest.raises(ValidationError, match=re.escape(requirement)):
        ExperimentConfig.model_validate(payload)


def test_existing_modes_and_defaults_remain_compatible() -> None:
    legacy = load_config("configs/smoke.yaml")
    assert legacy.agent.designer_mode == "heuristic"
    assert legacy.agent.provider == "mock"

    payload = legacy.model_dump(mode="python")
    payload["agent"]["designer_mode"] = "llm_staged"
    payload["agent"]["provider"] = "openai"
    payload["agent"]["design_budget"]["max_candidate_specs"] = 1
    payload["agent"]["agent_budget"].update(
        {
            "max_turns": 1,
            "max_tool_calls": 0,
            "max_candidate_specs": 1,
            "max_revisions": 4,
            "max_smoke_tests": 0,
        }
    )
    configured = ExperimentConfig.model_validate(payload)
    assert configured.agent.designer_mode == "llm_staged"
    assert configured.agent.provider == "openai"
