"""Execute and receipt the registered full-budget JSSP qualification."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pydantic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from operator_evolution_core.memory import MechanismMemory  # noqa: E402
from operator_evolution_core.trajectory import TrajectoryRecorder  # noqa: E402

from jssp_operator_evolution.data import build_jssp_splits  # noqa: E402
from jssp_operator_evolution.data.orlib import sha256_file  # noqa: E402
from jssp_operator_evolution.qualification import (  # noqa: E402
    JSSPFormalQualificationConfig,
    run_formal_qualification,
)


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _progress(message: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{stamp}] {message}", flush=True)


def main() -> None:
    results = ROOT / "artifacts" / "results"
    releases = ROOT / "artifacts" / "releases"
    results.mkdir(parents=True, exist_ok=True)
    releases.mkdir(parents=True, exist_ok=True)
    database = results / "jssp-cross-domain-qualification-v1.sqlite"
    memory_database = results / "jssp-cross-domain-qualification-v1.memory.sqlite"
    jsonl = results / "jssp-cross-domain-qualification-v1.jsonl"
    receipt_path = releases / "cross-domain-core-qualification-v1.formal.json"
    for target in (database, memory_database, jsonl, receipt_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    raw = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
    splits = build_jssp_splits(raw)
    config = JSSPFormalQualificationConfig()
    started = datetime.now(timezone.utc)
    _progress("formal qualification started; test remains sealed")
    with (
        TrajectoryRecorder(database) as recorder,
        MechanismMemory(memory_database) as memory,
    ):
        report, outcome = run_formal_qualification(
            splits,
            recorder,
            memory,
            config=config,
            progress_callback=_progress,
        )
        recorder.export_jsonl(jsonl)
    finished = datetime.now(timezone.utc)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    payload = {
        "schema_version": "cross-domain-core-qualification-formal-receipt-v1",
        "code_commit": commit,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "configuration": config.as_dict(),
        "dataset": {
            "train_instances": 60,
            "validation_instances": 41,
            "test_instances": 41,
            "orlib_jobshop1_sha256": sha256_file(raw),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pydantic": pydantic.__version__,
        },
        "report": report.model_dump(mode="json"),
        "artifacts": {
            "trajectory_jsonl": str(jsonl.relative_to(ROOT)).replace("\\", "/"),
            "trajectory_jsonl_sha256": sha256_file(jsonl),
            "trajectory_sqlite": str(database.relative_to(ROOT)).replace("\\", "/"),
            "trajectory_sqlite_sha256": sha256_file(database),
            "memory_sqlite": str(memory_database.relative_to(ROOT)).replace("\\", "/"),
            "memory_sqlite_sha256": sha256_file(memory_database),
        },
        "test_access": {
            "opened": True,
            "opened_after_population_freeze": True,
            "freeze_receipt_id": outcome.freeze_receipt.receipt_id,
            "used_for_retention": False,
        },
    }
    body = _canonical_bytes(payload)
    payload["receipt_payload_sha256"] = hashlib.sha256(body).hexdigest()
    receipt_path.write_bytes(_canonical_bytes(payload))
    _progress(f"formal receipt written: {receipt_path}")
    print(payload["receipt_payload_sha256"], flush=True)


if __name__ == "__main__":
    main()
