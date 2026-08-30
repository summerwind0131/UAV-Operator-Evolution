"""Run validation-only scratch/same/cross mechanism-transfer arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jssp_operator_evolution.data import build_jssp_splits  # noqa: E402
from jssp_operator_evolution.transfer_experiment import (  # noqa: E402
    JSSPTransferArmConfig,
    run_jssp_transfer_arm,
)
from operator_evolution_core.evolution import (  # noqa: E402
    MechanismTransferPreregistrationV1,
)
from operator_evolution_core.memory import MechanismBankV1  # noqa: E402
from uav_operator_evolution.config import load_config  # noqa: E402
from uav_operator_evolution.environment import load_dataset_split  # noqa: E402
from uav_operator_evolution.transfer_experiment import (  # noqa: E402
    UAVTransferArmConfig,
    run_uav_transfer_arm,
)


def _canonical_hash(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _source_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _load_banks(path: Path) -> tuple[dict[str, object], MechanismBankV1, MechanismBankV1]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = document["payload"]
    if _canonical_hash(payload) != document["payload_sha256"]:
        raise ValueError("formal mechanism-bank receipt hash mismatch")
    if payload["mode"] != "formal" or payload["test_instances_opened"]:
        raise ValueError("transfer arms require the sealed formal mechanism banks")
    return (
        document,
        MechanismBankV1.model_validate(payload["uav_bank"]),
        MechanismBankV1.model_validate(payload["jssp_bank"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run equal-budget validation-only transfer design arms."
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registration = MechanismTransferPreregistrationV1.model_validate(
        yaml.safe_load(
            (ROOT / "configs" / "mechanism_transfer_v1.yaml").read_text(
                encoding="utf-8"
            )
        )
    )
    bank_path = (
        ROOT
        / "artifacts"
        / "releases"
        / "mechanism-transfer-v1.bank-formal.json"
    )
    bank_document, uav_bank, jssp_bank = _load_banks(bank_path)
    if args.mode == "smoke":
        seeds = registration.master_seeds[:1]
        uav_config = UAVTransferArmConfig(
            search_calls=64,
            generations=1,
            candidates_per_generation=1,
            validation_instances=1,
            runtime_repetitions=2,
        )
        jssp_config = JSSPTransferArmConfig(
            search_calls=64,
            generations=1,
            candidates_per_generation=1,
            validation_instances=1,
            runtime_repetitions=2,
        )
    else:
        seeds = registration.master_seeds
        uav_config = UAVTransferArmConfig(
            search_calls=registration.budget.validation_search_calls,
            generations=registration.budget.generations,
            candidates_per_generation=(
                registration.budget.candidates_per_generation
            ),
            validation_instances=40,
            runtime_repetitions=2,
        )
        jssp_config = JSSPTransferArmConfig(
            search_calls=registration.budget.validation_search_calls,
            generations=registration.budget.generations,
            candidates_per_generation=(
                registration.budget.candidates_per_generation
            ),
            validation_instances=41,
            runtime_repetitions=2,
        )

    uav_manifest = ROOT / "data" / "benchmarks" / "uav2d-v1" / "manifest.json"
    uav_train = load_dataset_split(uav_manifest, "train")
    uav_validation = load_dataset_split(uav_manifest, "validation")
    uav_base_config = load_config(ROOT / "configs" / "smoke.yaml")
    jssp_splits = build_jssp_splits(
        ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
    )

    outcomes = []
    for seed in seeds:
        for arm in registration.arms:
            outcomes.append(
                run_uav_transfer_arm(
                    uav_train,
                    uav_validation,
                    arm=arm,
                    master_seed=seed,
                    same_domain_bank=uav_bank,
                    cross_domain_bank=jssp_bank,
                    base_config=uav_base_config,
                    config=uav_config,
                ).model_dump(mode="json")
            )
            outcomes.append(
                run_jssp_transfer_arm(
                    jssp_splits,
                    arm=arm,
                    master_seed=seed,
                    same_domain_bank=jssp_bank,
                    cross_domain_bank=uav_bank,
                    config=jssp_config,
                ).model_dump(mode="json")
            )

    payload = {
        "schema_version": "mechanism-transfer-arm-receipt-v1",
        "experiment_id": registration.experiment_id,
        "mode": args.mode,
        "source_commit": _source_commit(),
        "preregistration_hash": registration.content_hash,
        "formal_bank_receipt_sha256": bank_document["payload_sha256"],
        "master_seeds": list(seeds),
        "arms": list(registration.arms),
        "retrieval_limit": registration.retrieval_limit,
        "designer": registration.designer,
        "test_instances_opened": False,
        "remote_provider_calls": 0,
        "outcomes": outcomes,
    }
    receipt = {"payload": payload, "payload_sha256": _canonical_hash(payload)}
    output = args.output
    if output is None:
        output = (
            ROOT
            / "artifacts"
            / "releases"
            / f"mechanism-transfer-v1.arms-{args.mode}.json"
        )
    elif not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "payload_sha256": receipt["payload_sha256"],
                "outcomes": len(outcomes),
                "test_instances_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
