from __future__ import annotations

import shutil
import json
from pathlib import Path

import numpy as np
import pytest

from uav_operator_evolution.afl_uav.artifact import (
    QualificationResult,
    build_solver_artifact,
    extract_solver_counters,
    freeze_solver_candidate,
    generate_solver_candidate,
    load_solver_artifact,
    load_solver_candidate,
)
from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment.environment import Environment2D
from uav_operator_evolution.environment.obstacles import CircleObstacle
from uav_operator_evolution.path.evaluator import PathEvaluator
from uav_operator_evolution.planning_benchmarks import (
    FrozenAFLUAVPlanner,
    PlanningBudget,
    run_with_trusted_validation,
)


def _environment() -> Environment2D:
    return Environment2D(
        map_id="afl-artifact-train-map",
        width=20.0,
        height=20.0,
        start=(2.0, 2.0),
        goal=(18.0, 18.0),
        obstacles=[CircleObstacle(center=(10.0, 10.0), radius=2.0)],
        safety_distance=0.75,
        difficulty="medium",
        seed=17,
    )


@pytest.fixture(scope="module")
def frozen_artifact(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    root = tmp_path_factory.mktemp("afl_frozen_artifact")
    artifact, manifest = build_solver_artifact(
        load_config("configs/smoke.yaml"),
        _environment(),
        root / "artifact",
        provider="mock",
    )
    return manifest.parent, artifact


def test_build_freezes_audited_solver_with_cli_v2_contract(
    frozen_artifact: tuple[Path, object],
) -> None:
    artifact_dir, built = frozen_artifact
    artifact, source, solver_path = load_solver_artifact(artifact_dir)
    assert artifact == built
    assert solver_path.is_file()
    assert "--max-evaluations" in source
    assert "--seed" in source
    assert artifact.provider == "mock"
    assert artifact.research_claim_eligible is False
    assert artifact.generated_from_split == "train"
    assert artifact.contract_smoke_counters["objective_evaluations"] <= 16
    assert artifact.candidate_source_hash == artifact.approved_source_hash
    assert artifact.approved_source_hash == artifact.solver_hash


def test_artifact_loader_rejects_source_tampering(
    frozen_artifact: tuple[Path, object],
    tmp_path: Path,
) -> None:
    artifact_dir, _ = frozen_artifact
    copied = tmp_path / "tampered"
    shutil.copytree(artifact_dir, copied)
    solver_path = copied / "frozen_solver.py"
    solver_path.write_text(
        solver_path.read_text(encoding="utf-8") + "\n# tampered\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="solver hash"):
        load_solver_artifact(copied)


def test_frozen_solver_runs_through_common_budget_and_trusted_validator(
    frozen_artifact: tuple[Path, object],
) -> None:
    artifact_dir, _ = frozen_artifact
    planner = FrozenAFLUAVPlanner(artifact_dir)
    result = run_with_trusted_validation(
        planner,
        _environment(),
        PathEvaluator(),
        PlanningBudget(
            time_limit_seconds=1.0,
            max_objective_evaluations=20,
        ),
        np.random.default_rng(123),
    )
    assert result.path is not None
    assert result.trusted_evaluation is not None
    assert result.trusted_evaluation.feasible
    assert result.objective_evaluations == 20
    assert result.collision_checks > 0
    assert result.node_expansions > 0
    assert result.diagnostics["artifact_id"] == planner.artifact.artifact_id
    assert planner.stochastic is True
    assert planner.research_claim_eligible is False


def test_counter_contract_rejects_missing_or_excessive_evaluations() -> None:
    with pytest.raises(ValueError, match="collision_checks"):
        extract_solver_counters(
            {"objective_evaluations": 1, "node_expansions": 0},
            max_evaluations=10,
        )


def test_candidate_generation_never_executes_and_wrong_hash_cannot_freeze(
    tmp_path: Path,
) -> None:
    config = load_config("configs/smoke.yaml")
    candidate, candidate_path = generate_solver_candidate(
        config,
        _environment(),
        tmp_path / "candidate",
        provider="mock",
        model=None,
    )
    assert candidate_path.is_file()
    assert (candidate_path.parent / candidate.solver_filename).is_file()
    assert not (candidate_path.parent / "qualification").exists()
    assert candidate.usage.logical_calls == len(candidate.provider_calls)
    with pytest.raises(ValueError, match="approved source hash"):
        freeze_solver_candidate(
            config,
            candidate_path,
            [_environment()],
            tmp_path / "artifact",
            approved_source_hash="0" * 64,
            require_train_map_ids=False,
        )
    assert not (tmp_path / "artifact" / "frozen_solver.py").exists()


def test_real_candidate_generation_rejects_non_fixed_model_before_api_call(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="requires fixed model"):
        generate_solver_candidate(
            load_config("configs/smoke.yaml"),
            _environment(),
            tmp_path / "candidate",
            provider="openai",
            model="gpt-4.1",
        )


def test_failed_real_generation_persists_sanitized_call_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("UOE_LLM_API_KEY", "legacy-key-must-not-be-used")
    destination = tmp_path / "candidate"
    with pytest.raises(RuntimeError, match="failure audit"):
        generate_solver_candidate(
            load_config("configs/smoke.yaml"),
            _environment(),
            destination,
            provider="openai",
            model="gpt-4.1-2025-04-14",
        )
    payload = json.loads(
        (destination / "candidate_failure.json").read_text(encoding="utf-8")
    )
    assert payload["provider"] == "openai"
    assert payload["provider_calls"][0]["status"] == "configuration_error"
    assert "legacy-key-must-not-be-used" not in json.dumps(payload)


def test_candidate_loader_rejects_source_tampering(tmp_path: Path) -> None:
    candidate, candidate_path = generate_solver_candidate(
        load_config("configs/smoke.yaml"),
        _environment(),
        tmp_path / "candidate",
        provider="mock",
        model=None,
    )
    source_path = candidate_path.parent / candidate.solver_filename
    source_path.write_text(
        source_path.read_text(encoding="utf-8") + "\n# changed after approval\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate solver hash"):
        load_solver_candidate(candidate_path)


def test_failed_qualification_cannot_create_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import uav_operator_evolution.afl_uav.artifact as artifact_module

    config = load_config("configs/smoke.yaml")
    candidate, candidate_path = generate_solver_candidate(
        config,
        _environment(),
        tmp_path / "candidate",
        provider="mock",
        model=None,
    )

    def reject(*_args, **_kwargs):
        return (
            QualificationResult(
                map_id=_environment().map_id,
                difficulty=_environment().difficulty,
                status="success",
                passed=False,
                duration_ms=1.0,
                objective_evaluations=1,
                collision_checks=1,
                node_expansions=1,
                failures=["trusted hard constraint failed"],
            ),
            None,
        )

    monkeypatch.setattr(artifact_module, "_execute_qualification", reject)
    with pytest.raises(RuntimeError, match="contract smoke"):
        freeze_solver_candidate(
            config,
            candidate_path,
            [_environment()],
            tmp_path / "artifact",
            approved_source_hash=candidate.solver_hash,
            require_train_map_ids=False,
        )
    assert not (tmp_path / "artifact" / "artifact.json").exists()


def test_checked_in_v1_artifact_remains_loadable() -> None:
    artifact, _, _ = load_solver_artifact(
        Path("artifacts/planning_benchmarks/afl_uav_artifacts/afl-uav-offline-v3")
    )
    assert artifact.schema_version == "afl-uav-solver-artifact-v1"
    assert artifact.candidate_id is None
    assert artifact.research_claim_eligible is False


def test_v2_artifact_content_hash_rejects_manifest_tampering(
    frozen_artifact: tuple[Path, object],
    tmp_path: Path,
) -> None:
    artifact_dir, _ = frozen_artifact
    copied = tmp_path / "tampered_manifest"
    shutil.copytree(artifact_dir, copied)
    manifest_path = copied / "artifact.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["description_revisions"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity hash"):
        load_solver_artifact(copied)
    with pytest.raises(ValueError, match="exceeded"):
        extract_solver_counters(
            {
                "objective_evaluations": 11,
                "collision_checks": 1,
                "node_expansions": 1,
            },
            max_evaluations=10,
        )
