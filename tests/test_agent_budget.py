from __future__ import annotations

import pytest

from uav_operator_evolution.agents.providers import MockLLMProvider
from uav_operator_evolution.agents.research_agent import (
    DeterministicMockResearchAgent,
    OpenAIAgentsResearchAgent,
    OpenAIAgentsSDKUnavailableError,
)
from uav_operator_evolution.agents.tools import (
    AgentBudget,
    AgentBudgetController,
    AgentBudgetExceeded,
)

from test_mock_research_agent import _context


def test_agent_budget_controller_enforces_each_local_counter() -> None:
    controller = AgentBudgetController(
        AgentBudget(
            max_turns=1,
            max_tool_calls=1,
            max_candidate_specs=2,
            max_revisions=1,
            max_smoke_tests=1,
        )
    )
    controller.start_turn()
    with pytest.raises(AgentBudgetExceeded, match="turns budget exceeded"):
        controller.start_turn()

    controller.register_tool(smoke=True)
    with pytest.raises(AgentBudgetExceeded, match="tool_calls budget exceeded"):
        controller.register_tool()

    controller.register_candidate()
    controller.register_candidate(revision=True)
    with pytest.raises(AgentBudgetExceeded, match="candidate_specs budget exceeded"):
        controller.register_candidate(revision=True)

    controller.add_tokens(input_tokens=11, output_tokens=7)
    assert controller.usage.total_tokens == 18
    assert controller.usage.smoke_tests == 1
    assert controller.usage.revisions == 1


def test_research_agent_never_runs_model_after_evidence_tool_budget_exhaustion() -> None:
    provider = MockLLMProvider()
    backend = DeterministicMockResearchAgent(provider)
    with pytest.raises(AgentBudgetExceeded, match="tool_calls budget exceeded"):
        backend.run(_context(), budget=AgentBudget(max_tool_calls=7))
    assert provider.call_records == []


def test_research_agent_reserves_two_structured_turns_before_calling_provider() -> None:
    provider = MockLLMProvider()
    backend = DeterministicMockResearchAgent(provider)
    with pytest.raises(AgentBudgetExceeded, match="turns budget exceeded"):
        backend.run(_context(), budget=AgentBudget(max_turns=2))
    assert provider.call_records == []


def test_optional_agents_sdk_is_lazy_and_has_a_clear_missing_dependency_error() -> None:
    if OpenAIAgentsResearchAgent.available():
        pytest.skip("optional OpenAI Agents SDK is installed in this environment")
    with pytest.raises(OpenAIAgentsSDKUnavailableError, match="agent.*optional dependency"):
        OpenAIAgentsResearchAgent._load_sdk()

