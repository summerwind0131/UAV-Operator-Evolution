from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "artifacts" / "releases" / "trajectory-core-v0.1.0.receipt.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_core_v010_release_receipt_matches_source_docs_and_frozen_commit() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assets = {asset["name"]: asset for asset in receipt["assets"]}

    assert receipt["tag"] == "trajectory-core-v0.1.0"
    assert receipt["source_commit"] == "4de3d6d5a39105d52f365865a952168c41b7284c"
    assert receipt["package_name"] == "trajectory-operator-evolution"
    assert receipt["package_version"] == "0.1.0"
    assert receipt["prerelease"] is True
    assert receipt["pypi_published"] is False
    assert receipt["qualification"]["full_regression"] == "343 passed, 3 skipped, 1 deselected"

    for name, local_path in {
        "core_api.md": ROOT / "docs" / "core_api.md",
        "core_migration_v0.1.0.md": ROOT / "docs" / "core_migration_v0.1.0.md",
    }.items():
        assert assets[name]["size"] == local_path.stat().st_size
        assert assets[name]["sha256"] == _sha256(local_path)

    for name in (
        "trajectory_operator_evolution-0.1.0-py3-none-any.whl",
        "trajectory_operator_evolution-0.1.0.tar.gz",
    ):
        assert len(assets[name]["sha256"]) == 64
        assert assets[name]["size"] > 0
