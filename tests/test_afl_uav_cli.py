from __future__ import annotations

import pytest

from uav_operator_evolution.cli import _parse_afl_artifacts, _parser


def test_multi_artifact_cli_is_repeatable_and_preserves_windows_paths() -> None:
    args = _parser().parse_args(
        [
            "benchmark-planners",
            "--afl-artifact",
            r"openai_gpt41=C:\artifacts\openai",
            "--afl-artifact",
            "gemini_25pro=artifacts/gemini",
        ]
    )
    assert _parse_afl_artifacts(args.afl_artifact) == {
        "openai_gpt41": r"C:\artifacts\openai",
        "gemini_25pro": "artifacts/gemini",
    }


@pytest.mark.parametrize(
    "value",
    ["artifact-without-arm", "=missing-arm", "missing-path="],
)
def test_multi_artifact_cli_rejects_invalid_mapping(value: str) -> None:
    with pytest.raises(ValueError, match="ARM_ID=PATH"):
        _parse_afl_artifacts([value])


def test_candidate_and_freeze_commands_require_explicit_safety_inputs() -> None:
    candidate = _parser().parse_args(
        [
            "generate-afl-uav-candidate",
            "--provider",
            "openai",
            "--model",
            "gpt-4.1-2025-04-14",
        ]
    )
    assert candidate.provider == "openai"
    assert candidate.model == "gpt-4.1-2025-04-14"

    frozen = _parser().parse_args(
        [
            "freeze-afl-uav",
            "--candidate",
            "candidate.json",
            "--approve-source-hash",
            "a" * 64,
        ]
    )
    assert frozen.approve_source_hash == "a" * 64


@pytest.mark.parametrize("command", ["afl-uav-demo", "build-afl-uav"])
def test_legacy_one_step_commands_reject_real_providers(command: str) -> None:
    with pytest.raises(SystemExit):
        _parser().parse_args([command, "--provider", "openai"])
