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
from jssp_operator_evolution.transfer import (  # noqa: E402
    JSSPMechanismBankConfig,
    build_jssp_mechanism_bank,
)
from operator_evolution_core.evolution import (  # noqa: E402
    MechanismTransferPreregistrationV1,
)
from uav_operator_evolution.environment import load_dataset_split  # noqa: E402
from uav_operator_evolution.transfer import (  # noqa: E402
    UAVMechanismBankConfig,
    build_uav_mechanism_bank,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build sealed train/validation mechanism banks for both domains."
    )
    parser.add_argument("--mode", choices=("smoke", "formal"), default="smoke")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    registration_payload = yaml.safe_load(
        (ROOT / "configs" / "mechanism_transfer_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    registration = MechanismTransferPreregistrationV1.model_validate(
        registration_payload
    )
    commit = _source_commit()
    if args.mode == "smoke":
        uav_seeds = (registration.uav_bank_seeds[0],)
        jssp_seeds = (registration.jssp_bank_seeds[0],)
        uav_config = UAVMechanismBankConfig(
            train_calls=64,
            validation_calls=64,
            train_instances=2,
            validation_instances=2,
        )
        jssp_config = JSSPMechanismBankConfig(
            train_calls=64,
            validation_calls=64,
            train_instances=2,
            validation_instances=2,
        )
    else:
        uav_seeds = registration.uav_bank_seeds
        jssp_seeds = registration.jssp_bank_seeds
        uav_config = UAVMechanismBankConfig(
            train_calls=registration.budget.train_search_calls,
            validation_calls=registration.budget.validation_search_calls,
            train_instances=60,
            validation_instances=40,
        )
        jssp_config = JSSPMechanismBankConfig(
            train_calls=registration.budget.train_search_calls,
            validation_calls=registration.budget.validation_search_calls,
            train_instances=60,
            validation_instances=41,
        )

    uav_manifest = ROOT / "data" / "benchmarks" / "uav2d-v1" / "manifest.json"
    uav_train = load_dataset_split(uav_manifest, "train")
    uav_validation = load_dataset_split(uav_manifest, "validation")
    jssp_splits = build_jssp_splits(
        ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
    )
    uav_bank = build_uav_mechanism_bank(
        uav_train,
        uav_validation,
        bank_master_seeds=uav_seeds,
        source_code_commit=commit,
        config=uav_config,
    )
    jssp_bank = build_jssp_mechanism_bank(
        jssp_splits,
        bank_master_seeds=jssp_seeds,
        source_code_commit=commit,
        config=jssp_config,
    )
    payload = {
        "schema_version": "mechanism-bank-build-receipt-v1",
        "experiment_id": registration.experiment_id,
        "mode": args.mode,
        "source_commit": commit,
        "preregistration_hash": registration.content_hash,
        "test_instances_opened": False,
        "remote_provider_calls": 0,
        "uav_config": {
            "train_calls": uav_config.train_calls,
            "validation_calls": uav_config.validation_calls,
            "train_instances": uav_config.train_instances,
            "validation_instances": uav_config.validation_instances,
        },
        "jssp_config": {
            "train_calls": jssp_config.train_calls,
            "validation_calls": jssp_config.validation_calls,
            "train_instances": jssp_config.train_instances,
            "validation_instances": jssp_config.validation_instances,
        },
        "uav_bank": uav_bank.model_dump(mode="json"),
        "jssp_bank": jssp_bank.model_dump(mode="json"),
    }
    receipt = {
        "payload": payload,
        "payload_sha256": _canonical_hash(payload),
    }
    output = args.output
    if output is None:
        output = (
            ROOT
            / "artifacts"
            / "releases"
            / f"mechanism-transfer-v1.bank-{args.mode}.json"
        )
    elif not output.is_absolute():
        output = ROOT / output
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
                "uav_records": len(uav_bank.records),
                "jssp_records": len(jssp_bank.records),
                "test_instances_opened": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
