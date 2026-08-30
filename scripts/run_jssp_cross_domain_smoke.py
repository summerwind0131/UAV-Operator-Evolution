"""Run and receipt the registered 64-call, 2x2 JSSP offline smoke."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pydantic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from operator_evolution_core.diagnosis import OperatorDiagnoser  # noqa: E402
from operator_evolution_core.trajectory import TrajectoryRecorder  # noqa: E402

from jssp_operator_evolution.data import build_jssp_splits  # noqa: E402
from jssp_operator_evolution.data.orlib import sha256_file  # noqa: E402
from jssp_operator_evolution.evolution import (  # noqa: E402
    JSSPEvolutionSmokeConfig,
    run_offline_evolution_smoke,
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


def main() -> None:
    results = ROOT / "artifacts" / "results"
    releases = ROOT / "artifacts" / "releases"
    results.mkdir(parents=True, exist_ok=True)
    releases.mkdir(parents=True, exist_ok=True)
    database = results / "jssp-cross-domain-smoke-v1.sqlite"
    jsonl = results / "jssp-cross-domain-smoke-v1.jsonl"
    receipt_path = releases / "cross-domain-core-qualification-v1.smoke.json"
    for target in (database, jsonl, receipt_path):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact: {target}")

    raw = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
    splits = build_jssp_splits(raw)
    config = JSSPEvolutionSmokeConfig()
    with TrajectoryRecorder(database, jsonl) as recorder:
        outcome = run_offline_evolution_smoke(
            splits,
            recorder=recorder,
            config=config,
        )
        recorder.update_delayed_rewards((5, 10, 20))
        profiles = OperatorDiagnoser(recorder).diagnose()
        recorder.export_jsonl(jsonl)

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
    ).strip()
    payload = {
        "schema_version": "cross-domain-core-qualification-smoke-receipt-v1",
        "code_commit": commit,
        "configuration": {
            "master_seed": config.master_seed,
            "search_calls": config.search_calls,
            "generations": config.generations,
            "candidates_per_generation": config.candidates_per_generation,
            "validation_instances": config.validation_instances,
            "runtime_repetitions": config.runtime_repetitions,
            "fitness_policy": "deterministic-v2",
        },
        "dataset": {
            "train_instances": len(splits.open_train()),
            "validation_instances": len(splits.open_validation()),
            "sealed_test_instances": 41,
            "orlib_jobshop1_sha256": sha256_file(raw),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pydantic": pydantic.__version__,
        },
        "report": outcome.report.model_dump(mode="json"),
        "diagnosis": {
            "operator_profiles": len(profiles),
            "attempts": sum(profile.attempts for profile in profiles),
            "accepted": sum(profile.acceptances for profile in profiles),
        },
        "artifacts": {
            "trajectory_jsonl": str(jsonl.relative_to(ROOT)).replace("\\", "/"),
            "trajectory_jsonl_sha256": sha256_file(jsonl),
            "trajectory_sqlite": str(database.relative_to(ROOT)).replace("\\", "/"),
            "trajectory_sqlite_sha256": sha256_file(database),
        },
        "test_accessed": False,
    }
    body = _canonical_bytes(payload)
    payload["receipt_payload_sha256"] = hashlib.sha256(body).hexdigest()
    receipt_path.write_bytes(_canonical_bytes(payload))
    print(receipt_path)
    print(payload["receipt_payload_sha256"])


if __name__ == "__main__":
    main()
