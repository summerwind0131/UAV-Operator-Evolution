from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    ROOT
    / "artifacts"
    / "releases"
    / "cross-domain-core-qualification-v1.release.json"
)


def test_jssp_github_release_receipt_lists_the_frozen_assets() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert payload["tag"] == "cross-domain-core-qualification-v1"
    assert payload["release_url"].endswith(
        "/releases/tag/cross-domain-core-qualification-v1"
    )
    assets = {asset["name"]: asset for asset in payload["assets"]}
    assert set(assets) == {
        "cross-domain-core-qualification-v1-artifacts.tar.gz",
        "cross-domain-core-qualification-v1-artifacts.sha256",
        "cross-domain-core-qualification-v1.formal.json",
        "cross-domain-core-qualification-v1.smoke.json",
    }
    archive = assets["cross-domain-core-qualification-v1-artifacts.tar.gz"]
    assert archive["size"] == 18_758_307
    assert archive["sha256"] == (
        "7c15634b708fd30f2505ceec5fedf2b1d2870ebcd3a71fcac486d9f35139851b"
    )
    assert all(len(asset["sha256"]) == 64 for asset in assets.values())
    assert all(asset["url"].startswith("https://github.com/") for asset in assets.values())
