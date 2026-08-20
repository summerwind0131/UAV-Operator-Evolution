from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from uav_operator_evolution.agents.designer_base import OperatorProposal
from uav_operator_evolution.agents.providers import (
    LLMCallConfig,
    LLMConfigurationError,
    LLMRefusalError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTokenBudgetError,
    LLMUsage,
    MockLLMProvider,
    OpenAIProvider,
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


def _generate(provider, config: LLMCallConfig | None = None) -> Answer:
    return provider.generate_structured(
        system_prompt="Return a validated answer.",
        user_payload={"question": "one plus one"},
        output_model=Answer,
        config=config or LLMCallConfig(),
        prompt_version="test_v1",
        prompt_hash="a" * 64,
    )


def test_call_config_is_strict_and_has_bounded_defaults() -> None:
    config = LLMCallConfig()
    assert config.timeout_seconds == 60.0
    assert config.max_retries == 2
    assert config.max_output_tokens == 4_096
    assert config.max_total_tokens == 20_000
    with pytest.raises(ValidationError):
        LLMCallConfig.model_validate({"unknown": True})


def test_mock_fixture_is_deterministic_and_records_compact_metadata() -> None:
    provider = MockLLMProvider(fixtures={Answer: {"value": 2}})
    assert _generate(provider) == Answer(value=2)
    assert _generate(provider) == Answer(value=2)
    assert len(provider.call_records) == 2
    first = provider.call_records[0]
    assert first.status == "success"
    assert first.provider == "mock"
    assert first.model == "mock-structured-v1"
    assert first.response_id == "mock_response_000001"
    assert first.prompt_version == "test_v1"
    assert first.prompt_hash == "a" * 64
    assert first.usage.total_tokens > 0
    assert "question" not in first.model_dump_json()


def test_mock_factory_supports_retry_and_usage_accounting() -> None:
    provider = MockLLMProvider(
        factory=lambda output_model, user_payload: {"value": len(user_payload["question"])},
        failure_sequence=["timeout", "success"],
        usage=LLMUsage(input_tokens=3, output_tokens=2, total_tokens=5),
    )
    assert _generate(provider).value == 12
    record = provider.last_record
    assert record is not None
    assert record.attempts == 2
    assert record.retry_count == 1
    assert record.usage.total_tokens == 5
    assert record.cumulative_total_tokens == 5


@pytest.mark.parametrize(
    ("mode", "error_type", "status"),
    [
        ("schema_error", LLMStructuredOutputError, "schema_error"),
        ("refusal", LLMRefusalError, "refusal"),
    ],
)
def test_mock_terminal_failures_are_typed(mode, error_type, status) -> None:
    provider = MockLLMProvider(fixtures={Answer: {"value": 2}}, mode=mode)
    with pytest.raises(error_type):
        _generate(provider)
    assert provider.last_record is not None
    assert provider.last_record.status == status
    assert provider.last_record.attempts == 1


def test_mock_timeout_uses_bounded_retry_count() -> None:
    provider = MockLLMProvider(fixtures={Answer: {"value": 2}}, mode="timeout")
    with pytest.raises(LLMTimeoutError):
        _generate(provider, LLMCallConfig(max_retries=2))
    assert provider.last_record is not None
    assert provider.last_record.attempts == 3
    assert provider.last_record.retry_count == 2


def test_invalid_fixture_and_cumulative_budget_are_rejected() -> None:
    invalid = MockLLMProvider(fixtures={Answer: {"value": "two"}})
    with pytest.raises(LLMStructuredOutputError):
        _generate(invalid)

    budgeted = MockLLMProvider(
        fixtures={Answer: {"value": 2}},
        usage=LLMUsage(input_tokens=4, output_tokens=2, total_tokens=6),
    )
    config = LLMCallConfig(max_total_tokens=10)
    assert _generate(budgeted, config).value == 2
    with pytest.raises(LLMTokenBudgetError):
        _generate(budgeted, config)
    assert budgeted.last_record is not None
    assert budgeted.last_record.status == "budget_exceeded"
    assert budgeted.last_record.cumulative_total_tokens == 12
    budgeted.reset_token_budget()
    assert _generate(budgeted, config).value == 2


def test_bare_mock_builds_evidence_grounded_operator_proposal() -> None:
    bundle = {
        "parent_specs": [
            {
                "name": "Parent",
                "description": "parent operator",
                "selection_strategy": {"kind": "select_collision_segment"},
                "transformations": [{"kind": "generate_obstacle_detour"}],
                "expected_mechanism": "detour",
            }
        ],
        "failure_modes": [
            {
                "evidence_id": "fail_0123456789abcdef01234567",
                "failure_mode": "excess curvature",
                "confidence": 0.75,
            }
        ],
        "effective_contexts": [],
        "synergy_evidence": [],
        "limitations": ["no counterfactual evidence"],
    }
    proposal = MockLLMProvider().generate_structured(
        system_prompt="Design safely.",
        user_payload=bundle,
        output_model=OperatorProposal,
        config=LLMCallConfig(),
    )
    assert proposal.diagnosis is not None
    assert proposal.diagnosis.failure_modes[0].claim == "excess curvature"
    assert proposal.hypothesis is not None
    assert proposal.hypothesis.target_failure_mode == "excess curvature"
    assert proposal.used_evidence_ids == ["fail_0123456789abcdef01234567"]
    assert proposal.spec.parent_operators == ["Parent"]
    assert proposal.spec.fallback_strategy is not None


def test_openai_provider_missing_credentials_fails_without_sdk_import(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("UOE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("UOE_LLM_MODEL", raising=False)
    provider = OpenAIProvider()
    with pytest.raises(LLMConfigurationError, match="UOE_LLM_MODEL"):
        _generate(provider)
    assert provider.last_record is not None
    assert provider.last_record.status == "configuration_error"


def test_openai_provider_uses_responses_parse_and_records_usage() -> None:
    class Responses:
        def __init__(self) -> None:
            self.kwargs = None

        def parse(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                id="resp_123",
                model="test-model",
                output_parsed=Answer(value=2),
                output=[],
                usage=SimpleNamespace(input_tokens=7, output_tokens=3, total_tokens=10),
            )

    responses = Responses()
    provider = OpenAIProvider(client=SimpleNamespace(responses=responses), model="test-model")
    assert _generate(provider).value == 2
    assert responses.kwargs["text_format"] is Answer
    assert responses.kwargs["max_output_tokens"] == 4_096
    assert provider.last_record is not None
    assert provider.last_record.response_id == "resp_123"
    assert provider.last_record.usage.total_tokens == 10


def test_openai_refusal_and_empty_parsed_output_are_rejected() -> None:
    refusal_response = SimpleNamespace(
        id="resp_refusal",
        model="test-model",
        output_parsed=None,
        output=[SimpleNamespace(content=[SimpleNamespace(type="refusal", refusal="no")])],
        usage=None,
    )
    refusal_client = SimpleNamespace(
        responses=SimpleNamespace(parse=lambda **kwargs: refusal_response)
    )
    with pytest.raises(LLMRefusalError):
        _generate(OpenAIProvider(client=refusal_client, model="test-model"))

    empty_response = SimpleNamespace(
        id="resp_empty", model="test-model", output_parsed=None, output=[], usage=None
    )
    empty_client = SimpleNamespace(responses=SimpleNamespace(parse=lambda **kwargs: empty_response))
    with pytest.raises(LLMStructuredOutputError):
        _generate(OpenAIProvider(client=empty_client, model="test-model"))

