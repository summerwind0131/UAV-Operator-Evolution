"""Regenerate the public JSSP train/validation manifests deterministically."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "jssp"
sys.path.insert(0, str(ROOT / "src"))

from jssp_operator_evolution.data import build_jssp_splits  # noqa: E402


def _write(name: str, references: object) -> None:
    payload = {
        "schema_version": "jssp-instance-manifest-v1",
        "split": name,
        "instances": [item.model_dump(mode="json") for item in references],
    }
    target = DATA_ROOT / f"{name}_manifest.json"
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main() -> None:
    splits = build_jssp_splits(DATA_ROOT / "orlib" / "jobshop1.txt")
    _write("train", splits.manifest("train"))
    _write("validation", splits.manifest("validation"))


if __name__ == "__main__":
    main()
