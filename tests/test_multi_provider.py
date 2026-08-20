from __future__ import annotations

from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict

import uav_operator_evolution.agents.providers as providers_module

from uav_operator_evolution.agents.providers import (
    DeepSeekProvider,
    GeminiProvider,
    LLMCallConfig,
    LLMConfigurationError,
    LLMRefusalError,
    LLMStructuredOutputError,
    LLMTimeoutError,
    LLMTokenBudgetError,
    MockLLMProvider,
    OpenAIProvider,
)


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    value: int


def _generate(provider, config: LLMCallConfig | None = None) -> Answer:
    return provider.generate_structured(
        system_prompt="Return one typed answer.",
        user_payload={"question": "one plus one"},
        output_model=Answer,
        config=config or LLMCallConfig(model="provider-model"),
        prompt_version="multi_provider_test_v1",
    )


class _DeepSeekCompletions:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _deepseek_response(
    content: str = '{"value":2}',
    *,
    finish_reason: str = "stop",
    refusal: str | None = None,
):
    return SimpleNamespace(
        id="chatcmpl_ds_1",
        model="deepseek-v4-pro",
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4, total_tokens=15),
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, refusal=refusal),
            )
        ],
    )


def test_deepseek_json_mode_schema_usage_and_retry() -> None:
    completions = _DeepSeekCompletions([TimeoutError("temporary"), _deepseek_response()])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekProvider(client=client, model="deepseek-v4-pro")

    assert _generate(provider).value == 2
    assert len(completions.calls) == 2
    assert completions.calls[0]["response_format"] == {"type": "json_object"}
    assert completions.calls[0]["extra_body"] == {
        "thinking": {"type": "disabled"}
    }
    assert "JSON must validate against this schema" in completions.calls[0]["messages"][0]["content"]
    assert provider.last_record is not None
    assert provider.last_record.response_id == "chatcmpl_ds_1"
    assert provider.last_record.attempts == 2
    assert provider.last_record.usage.total_tokens == 15


def test_deepseek_retries_connection_errors() -> None:
    class APIConnectionError(Exception):
        pass

    completions = _DeepSeekCompletions(
        [APIConnectionError("temporary connection failure"), _deepseek_response()]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekProvider(client=client, model="deepseek-v4-pro")

    assert _generate(provider).value == 2
    assert len(completions.calls) == 2
    assert provider.last_record is not None
    assert provider.last_record.status == "success"
    assert provider.last_record.attempts == 2


def test_deepseek_retries_schema_errors_with_validation_feedback() -> None:
    completions = _DeepSeekCompletions(
        [_deepseek_response('{"value":"two"}'), _deepseek_response()]
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekProvider(client=client, model="deepseek-v4-pro")

    assert _generate(provider).value == 2
    assert len(completions.calls) == 2
    retry_system_prompt = completions.calls[1]["messages"][0]["content"]
    assert "failed local schema validation" in retry_system_prompt
    assert "string length" in retry_system_prompt
    assert provider.last_record is not None
    assert provider.last_record.attempts == 2
    assert provider.last_record.retry_count == 1
    assert provider.last_record.usage.total_tokens == 30


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (_deepseek_response("", finish_reason="length"), LLMStructuredOutputError),
        (_deepseek_response('{"value":"two"}'), LLMStructuredOutputError),
        (_deepseek_response(refusal="no"), LLMRefusalError),
    ],
)
def test_deepseek_rejects_truncation_invalid_schema_and_refusal(response, error_type) -> None:
    completions = _DeepSeekCompletions([response])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    with pytest.raises(error_type):
        _generate(
            DeepSeekProvider(client=client, model="deepseek-v4-pro"),
            LLMCallConfig(model="deepseek-v4-pro", max_retries=0),
        )


class _GeminiModels:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls: list[dict] = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _gemini_response(
    text: str = '{"value":2}',
    *,
    finish_reason: str = "STOP",
):
    return SimpleNamespace(
        response_id="gemini_response_1",
        model_version="gemini-2.5-pro",
        text=text,
        candidates=[SimpleNamespace(finish_reason=finish_reason)],
        usage_metadata=SimpleNamespace(
            prompt_token_count=13,
            candidates_token_count=3,
            thoughts_token_count=2,
            total_token_count=18,
        ),
    )


def test_gemini_json_schema_usage_and_retry() -> None:
    models = _GeminiModels([TimeoutError("temporary"), _gemini_response()])
    provider = GeminiProvider(
        client=SimpleNamespace(models=models),
        model="gemini-2.5-pro",
    )

    assert _generate(provider).value == 2
    assert len(models.calls) == 2
    config = models.calls[0]["config"]
    assert config["response_mime_type"] == "application/json"
    assert config["response_json_schema"]["properties"]["value"]["type"] == "integer"
    assert provider.last_record is not None
    assert provider.last_record.response_id == "gemini_response_1"
    assert provider.last_record.attempts == 2
    assert provider.last_record.usage.output_tokens == 5


@pytest.mark.parametrize(
    ("response", "error_type"),
    [
        (_gemini_response("", finish_reason="MAX_TOKENS"), LLMStructuredOutputError),
        (_gemini_response('{"value":"two"}'), LLMStructuredOutputError),
        (_gemini_response("", finish_reason="SAFETY"), LLMRefusalError),
    ],
)
def test_gemini_rejects_truncation_invalid_schema_and_safety(response, error_type) -> None:
    models = _GeminiModels([response])
    client = SimpleNamespace(models=models)
    with pytest.raises(error_type):
        _generate(GeminiProvider(client=client, model="gemini-2.5-pro"))


@pytest.mark.parametrize(
    ("provider", "environment_name"),
    [
        (DeepSeekProvider(model="deepseek-v4-pro"), "DEEPSEEK_API_KEY"),
        (GeminiProvider(model="gemini-2.5-pro"), "GEMINI_API_KEY"),
    ],
)
def test_real_providers_fail_closed_without_credentials(monkeypatch, provider, environment_name) -> None:
    monkeypatch.delenv(environment_name, raising=False)
    with pytest.raises(LLMConfigurationError):
        _generate(provider)
    assert provider.last_record is not None
    assert provider.last_record.status == "configuration_error"


def test_afl_openai_mode_ignores_legacy_environment_keys(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UOE_LLM_API_KEY", "legacy-key-must-not-be-used")
    provider = OpenAIProvider(
        model="gpt-4.1-2025-04-14",
        allow_legacy_environment=False,
    )
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        _generate(
            provider,
            LLMCallConfig(model="gpt-4.1-2025-04-14"),
        )


@pytest.mark.parametrize(
    "provider",
    [
        DeepSeekProvider(api_key="secret-key", model="deepseek-v4-pro"),
        GeminiProvider(api_key="secret-key", model="gemini-2.5-pro"),
    ],
)
def test_real_providers_fail_closed_when_sdk_is_missing(monkeypatch, provider) -> None:
    def missing_sdk(*_args, **_kwargs):
        raise ModuleNotFoundError("optional SDK is unavailable")

    monkeypatch.setattr(provider, "_get_client", missing_sdk)
    with pytest.raises(LLMConfigurationError, match="SDK"):
        _generate(provider)
    assert provider.last_record is not None
    assert provider.last_record.status == "configuration_error"


def test_provider_error_redacts_api_key_and_timeout_is_typed() -> None:
    secret = "sk-test-secret-123456789"
    completions = _DeepSeekCompletions([TimeoutError(f"request failed with {secret}")])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    provider = DeepSeekProvider(
        client=client,
        api_key=secret,
        model="deepseek-v4-pro",
    )
    with pytest.raises(LLMTimeoutError) as error:
        _generate(provider, LLMCallConfig(model="deepseek-v4-pro", max_retries=0))
    assert secret not in str(error.value)
    assert provider.last_record is not None
    assert secret not in (provider.last_record.error or "")
    assert "[REDACTED]" in (provider.last_record.error or "")


def test_deepseek_rejects_response_past_end_to_end_attempt_deadline(
    monkeypatch,
) -> None:
    clock = iter([0.0, 0.0, 61.0, 61.0])
    monkeypatch.setattr(providers_module, "perf_counter", lambda: next(clock))
    completions = _DeepSeekCompletions([_deepseek_response()])
    provider = DeepSeekProvider(
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
        model="deepseek-v4-pro",
    )

    with pytest.raises(LLMTimeoutError, match="end-to-end attempt deadline"):
        _generate(
            provider,
            LLMCallConfig(
                model="deepseek-v4-pro",
                timeout_seconds=60.0,
                max_retries=0,
            ),
        )
    assert provider.last_record is not None
    assert provider.last_record.status == "timeout"
    assert provider.last_record.usage.total_tokens == 15


def test_logical_call_and_token_budgets_fail_closed() -> None:
    provider = MockLLMProvider(fixtures={Answer: {"value": 2}})
    config = LLMCallConfig(model="mock", max_logical_calls=1, max_total_tokens=100_000)
    assert _generate(provider, config).value == 2
    with pytest.raises(LLMTokenBudgetError):
        _generate(provider, config)
    assert provider.last_record is not None
    assert provider.last_record.status == "budget_exceeded"
