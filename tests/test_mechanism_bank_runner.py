from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_bank_runner_has_no_test_loader_or_remote_provider_path() -> None:
    source = (ROOT / "scripts" / "build_mechanism_banks.py").read_text(
        encoding="utf-8"
    )

    assert 'load_dataset_split(uav_manifest, "test")' not in source
    assert ".open_test(" not in source
    assert "remote_provider_calls\": 0" in source
    assert 'choices=("smoke", "formal")' in source
