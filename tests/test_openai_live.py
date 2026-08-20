"""Optional one-call probes; skipped unless each provider is configured."""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, ConfigDict

from uav_operator_evolution.agents.providers import (
    DeepSeekProvider,
    GeminiProvider,
    LLMCallConfig,
    OpenAIProvider,
)


class _LiveProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool


@pytest.mark.live
def test_openai_responses_structured_output_live() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not configured")
    pytest.importorskip("openai")
    model = "gpt-4.1-2025-04-14"
    result = OpenAIProvider(model=model).generate_structured(
        system_prompt="Return the requested typed boolean and no additional fields.",
        user_payload={"set_ok_to": True},
        output_model=_LiveProbe,
        config=LLMCallConfig(
            model=model,
            timeout_seconds=60.0,
            max_retries=1,
            max_output_tokens=64,
            max_total_tokens=2_000,
            max_logical_calls=1,
        ),
        prompt_version="live_probe_v1",
    )
    assert result.ok is True


@pytest.mark.live
def test_deepseek_json_output_live() -> None:
    if not os.getenv("DEEPSEEK_API_KEY"):
        pytest.skip("DEEPSEEK_API_KEY is not configured")
    pytest.importorskip("openai")
    model = "deepseek-v4-pro"
    result = DeepSeekProvider(model=model).generate_structured(
        system_prompt="Return the requested typed boolean and no additional fields.",
        user_payload={"set_ok_to": True},
        output_model=_LiveProbe,
        config=LLMCallConfig(
            model=model,
            timeout_seconds=60.0,
            max_retries=1,
            max_output_tokens=64,
            max_total_tokens=2_000,
            max_logical_calls=1,
        ),
        prompt_version="deepseek_live_probe_v1",
    )
    assert result.ok is True


@pytest.mark.live
def test_gemini_structured_output_live() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY is not configured")
    pytest.importorskip("google.genai")
    model = "gemini-2.5-pro"
    result = GeminiProvider(model=model).generate_structured(
        system_prompt="Return the requested typed boolean and no additional fields.",
        user_payload={"set_ok_to": True},
        output_model=_LiveProbe,
        config=LLMCallConfig(
            model=model,
            timeout_seconds=60.0,
            max_retries=1,
            max_output_tokens=64,
            max_total_tokens=2_000,
            max_logical_calls=1,
        ),
        prompt_version="gemini_live_probe_v1",
    )
    assert result.ok is True
