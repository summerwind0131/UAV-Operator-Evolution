from __future__ import annotations

import json
from pathlib import Path

from jssp_operator_evolution.data import build_jssp_splits
from jssp_operator_evolution.transfer_experiment import (
    JSSPTransferArmConfig,
    run_jssp_transfer_arm,
)
from operator_evolution_core.evolution import (
    select_transfer_evidence_v1,
    transfer_candidate_context_v1,
)
from operator_evolution_core.memory import MechanismBankV1
from uav_operator_evolution.config import load_config
from uav_operator_evolution.domain.adapters import UAV_DOMAIN_ID
from uav_operator_evolution.environment import load_dataset_split
from uav_operator_evolution.transfer_experiment import (
    UAVTransferArmConfig,
    run_uav_transfer_arm,
)


ROOT = Path(__file__).resolve().parents[1]


def _banks() -> tuple[MechanismBankV1, MechanismBankV1]:
    document = json.loads(
        (
            ROOT
            / "artifacts"
            / "releases"
            / "mechanism-transfer-v1.bank-formal.json"
        ).read_text(encoding="utf-8")
    )
    payload = document["payload"]
    return (
        MechanismBankV1.model_validate(payload["uav_bank"]),
        MechanismBankV1.model_validate(payload["jssp_bank"]),
    )


def test_three_arm_evidence_selection_is_directional_and_fixed_top4() -> None:
    uav_bank, jssp_bank = _banks()
    context = transfer_candidate_context_v1(0, 0, generations=1)

    scratch = select_transfer_evidence_v1(
        arm="scratch",
        target_domain_id="jssp",
        same_domain_bank=jssp_bank,
        cross_domain_bank=uav_bank,
        context=context,
    )
    same = select_transfer_evidence_v1(
        arm="same-domain",
        target_domain_id="jssp",
        same_domain_bank=jssp_bank,
        cross_domain_bank=uav_bank,
        context=context,
    )
    cross = select_transfer_evidence_v1(
        arm="cross-domain",
        target_domain_id="jssp",
        same_domain_bank=jssp_bank,
        cross_domain_bank=uav_bank,
        context=context,
    )

    assert scratch.mechanism_ids == ()
    assert scratch.source_domain_id is None
    assert len(same.mechanism_ids) == len(cross.mechanism_ids) == 4
    assert same.source_domain_id == "jssp"
    assert cross.source_domain_id == UAV_DOMAIN_ID
    assert same.source_bank_hash == jssp_bank.bank_hash
    assert cross.source_bank_hash == uav_bank.bank_hash


def test_both_target_domains_run_all_three_validation_only_arms() -> None:
    uav_bank, jssp_bank = _banks()
    uav_manifest = ROOT / "data" / "benchmarks" / "uav2d-v1" / "manifest.json"
    uav_train = load_dataset_split(uav_manifest, "train")
    uav_validation = load_dataset_split(uav_manifest, "validation")
    jssp_splits = build_jssp_splits(
        ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
    )
    seed = 2026090101
    outcomes = []
    for arm in ("scratch", "same-domain", "cross-domain"):
        outcomes.append(
            run_uav_transfer_arm(
                uav_train,
                uav_validation,
                arm=arm,
                master_seed=seed,
                same_domain_bank=uav_bank,
                cross_domain_bank=jssp_bank,
                base_config=load_config(ROOT / "configs" / "smoke.yaml"),
                config=UAVTransferArmConfig(
                    search_calls=2,
                    generations=1,
                    candidates_per_generation=1,
                    validation_instances=1,
                    runtime_repetitions=2,
                ),
            )
        )
        outcomes.append(
            run_jssp_transfer_arm(
                jssp_splits,
                arm=arm,
                master_seed=seed,
                same_domain_bank=jssp_bank,
                cross_domain_bank=uav_bank,
                config=JSSPTransferArmConfig(
                    search_calls=2,
                    generations=1,
                    candidates_per_generation=1,
                    validation_instances=1,
                    runtime_repetitions=2,
                ),
            )
        )

    assert len(outcomes) == 6
    assert {outcome.arm for outcome in outcomes} == {
        "scratch",
        "same-domain",
        "cross-domain",
    }
    assert all(not outcome.test_instances_opened for outcome in outcomes)
    assert all(outcome.remote_provider_calls == 0 for outcome in outcomes)
    assert all(len(outcome.candidates) == 1 for outcome in outcomes)
    assert all(outcome.candidates[0].smoke_passed for outcome in outcomes)
    assert all(outcome.candidates[0].validation_outcomes == 1 for outcome in outcomes)
    for outcome in outcomes:
        evidence = outcome.candidates[0].evidence
        expected = 0 if outcome.arm == "scratch" else 4
        assert len(evidence.mechanism_ids) == expected


def test_arm_runner_has_no_test_loader_or_remote_provider_path() -> None:
    source = (ROOT / "scripts" / "run_mechanism_transfer_arms.py").read_text(
        encoding="utf-8"
    )

    assert 'load_dataset_split(uav_manifest, "test")' not in source
    assert ".open_test(" not in source
    assert 'choices=("smoke", "formal")' in source
    assert '"remote_provider_calls": 0' in source
