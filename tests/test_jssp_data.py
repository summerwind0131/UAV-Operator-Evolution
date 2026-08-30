from __future__ import annotations

import json
from pathlib import Path

import pytest

from operator_evolution_core.evolution import population_fingerprint

from jssp_operator_evolution.data import (
    ORLIB_JOBSHOP1_SHA256,
    build_jssp_splits,
    generate_training_instances,
    parse_jobshop1,
)
from jssp_operator_evolution.data.orlib import sha256_file
from jssp_operator_evolution.operators import JSSPDomainKit, initial_operator_specs

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "jssp" / "orlib" / "jobshop1.txt"
SOURCE = ROOT / "data" / "jssp" / "orlib" / "SOURCE.json"


def test_official_archive_and_source_receipt_hashes_are_exact() -> None:
    receipt = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert sha256_file(RAW) == ORLIB_JOBSHOP1_SHA256
    assert receipt["dataset"]["sha256"] == ORLIB_JOBSHOP1_SHA256
    assert receipt["dataset"]["instance_count"] == 82
    assert receipt["license"]["name"] == "MIT License"
    assert sha256_file(ROOT / "data" / "jssp" / "orlib" / "legal.html") == receipt["license"]["sha256"]


def test_orlib_parser_recovers_82_unique_normalized_instances() -> None:
    instances = parse_jobshop1(RAW)
    assert len(instances) == 82
    assert len({instance.instance_id for instance in instances}) == 82
    assert len({instance.content_hash for instance in instances}) == 82
    ft06 = next(instance for instance in instances if instance.instance_id == "ft06")
    assert (ft06.job_count, ft06.machines) == (6, 6)
    assert ft06.jobs[0][0].machine == 2
    assert ft06.jobs[0][0].duration == 1


def test_registered_synthetic_training_corpus_is_exact_and_deterministic() -> None:
    first = generate_training_instances()
    second = generate_training_instances()
    assert first == second
    assert len(first) == 60
    assert [(item.job_count, item.machines) for item in first].count((6, 6)) == 20
    assert [(item.job_count, item.machines) for item in first].count((10, 10)) == 20
    assert [(item.job_count, item.machines) for item in first].count((20, 15)) == 20
    assert all(
        1 <= operation.duration <= 99
        for instance in first
        for job in instance.jobs
        for operation in job
    )


def test_split_is_60_41_41_disjoint_and_test_manifest_is_sealed() -> None:
    splits = build_jssp_splits(RAW)
    assert len(splits.open_train()) == 60
    assert len(splits.open_validation()) == 41
    with pytest.raises(PermissionError, match="freeze receipt"):
        splits.open_test()
    with pytest.raises(PermissionError, match="freeze receipt"):
        splits.manifest("test")

    kit = JSSPDomainKit()
    specs = initial_operator_specs()
    identifiers = [spec.operator_id for spec in specs]
    ir_by_id = {spec.operator_id: spec for spec in specs}
    fingerprint = population_fingerprint(identifiers, ir_by_id, kit)
    receipt = splits.freeze_population(identifiers, fingerprint)
    test = splits.open_test(receipt)
    assert len(test) == 41
    assert len(splits.manifest("test", receipt)) == 41

    all_hashes = {
        instance.content_hash
        for instance in (
            *splits.open_train(),
            *splits.open_validation(),
            *test,
        )
    }
    assert len(all_hashes) == 142


def test_committed_public_manifests_match_the_split_builder() -> None:
    splits = build_jssp_splits(RAW)
    for split in ("train", "validation"):
        committed = json.loads(
            (ROOT / "data" / "jssp" / f"{split}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        expected = [
            reference.model_dump(mode="json")
            for reference in splits.manifest(split)
        ]
        assert committed == {
            "schema_version": "jssp-instance-manifest-v1",
            "split": split,
            "instances": expected,
        }
    assert not (ROOT / "data" / "jssp" / "test_manifest.json").exists()
