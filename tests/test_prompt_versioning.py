from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from uav_operator_evolution.agents.prompts import (
    DESIGNER_V1,
    DIAGNOSER_V1,
    PROMPT_TEMPLATES,
    RESEARCH_AGENT_V1,
    REVIEWER_V1,
    SYSTEM_POLICY,
    PromptTemplate,
    get_prompt_template,
)


def test_prompt_templates_have_stable_unique_content_hashes() -> None:
    templates = [DIAGNOSER_V1, DESIGNER_V1, REVIEWER_V1, RESEARCH_AGENT_V1]
    assert [item.version for item in templates] == [
        "diagnoser_v1",
        "designer_v1",
        "reviewer_v1",
        "research_agent_v1",
    ]
    assert all(len(item.prompt_hash) == 64 for item in templates)
    assert len({item.prompt_hash for item in templates}) == len(templates)
    assert all(SYSTEM_POLICY in item.system_text for item in templates)
    assert get_prompt_template("designer_v1") is DESIGNER_V1


def test_prompt_hash_tracks_content_independently_of_version() -> None:
    baseline = PromptTemplate(name="test", version="test_v1", system_text="bounded")
    identical = PromptTemplate(name="test", version="test_v1", system_text="bounded")
    changed_version = PromptTemplate(name="test", version="test_v2", system_text="bounded")
    changed_text = PromptTemplate(name="test", version="test_v1", system_text="bounded safely")
    assert baseline.prompt_hash == identical.prompt_hash == identical.content_hash
    assert baseline.prompt_hash == changed_version.prompt_hash
    assert baseline.prompt_hash != changed_text.prompt_hash


def test_prompt_objects_and_registry_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        DESIGNER_V1.system_text = "replace"  # type: ignore[misc]
    with pytest.raises(TypeError):
        PROMPT_TEMPLATES["other"] = DESIGNER_V1  # type: ignore[index]
    with pytest.raises(KeyError, match="unknown prompt template"):
        get_prompt_template("missing")
