from __future__ import annotations

from pathlib import Path

import pytest

from uav_operator_evolution.agents.audit import AgentAuditStore
from uav_operator_evolution.agents.design_models import CandidateStatus
from uav_operator_evolution.config import load_config
from uav_operator_evolution.environment import Environment2D
from uav_operator_evolution.experiments.agent_workflows import (
    agent_demo_workflow,
    build_evidence_workflow,
    create_llm_provider,
    propose_operator_workflow,
    run_agent_ablations_workflow,
    run_agent_workflow,
    validate_candidate_workflow,
)
from uav_operator_evolution.memory import MechanismMemory
from uav_operator_evolution.operators.catalog import manual_operator_specs
from uav_operator_evolution.reproducibility import stable_hash
from uav_operator_evolution.runtime import RunPaths


PARENT = "waypoint_perturb"


def _config(tmp_path: Path):
    base = load_config("configs/agent_smoke.yaml")
    return base.model_copy(
        update={
            "output": base.output.model_copy(
                update={
                    "data_dir": tmp_path / "data",
                    "results_dir": tmp_path / "results",
                    "figures_dir": tmp_path / "figures",
                    "export_jsonl": False,
                }
            ),
            "search": base.search.model_copy(
                update={
                    "train_iterations": 4,
                    "validation_iterations": 3,
                    "test_iterations": 3,
                }
            ),
        }
    )


def _seed_run(tmp_path: Path, config) -> Path:
    directory = tmp_path / "existing-run"
    directory.mkdir()
    database = directory / "experiment.sqlite"
    spec = manual_operator_specs()[PARENT]
    with MechanismMemory(database) as memory:
        memory.add_mechanism(
            PARENT,
            spec.model_dump(mode="json"),
            name=PARENT,
            description=spec.description,
            score=1.0,
            evidence_count=20,
            tags=["parent", "profiled"],
        )
        memory.add_operator_profile(
            {
                "operator_name": PARENT,
                "operator_id": PARENT,
                "total_calls": 20,
                "attempts": 20,
                "average_immediate_reward": -0.5,
                "average_delayed_reward": -0.25,
                "feasibility_rate": 0.8,
                "failure_contexts": [{"map_type": "dense", "calls": 12}],
            },
            operator_id=PARENT,
            run_id="seed",
        )
        memory.add_failure_mode(
            "dense_stagnation",
            operator_id=PARENT,
            count=20,
            severity=1.5,
            context={"map_type": "dense"},
            evidence=[{"profile": "seed"}],
        )
    return directory


def _map(map_id: str, size: float, seed: int) -> Environment2D:
    return Environment2D(
        map_id=map_id,
        width=size,
        height=size,
        start=(1, 1),
        goal=(size - 1, size - 1),
        obstacles=[],
        seed=seed,
    )


def test_evidence_proposal_agent_and_validation_workflows_preserve_boundaries(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    run_dir = _seed_run(tmp_path, config)
    train = _map("train-smoke", 20, 1)
    validation = _map("validation-only", 22, 2)
    heldout = _map("heldout-test", 24, 3)

    evidence = build_evidence_workflow(
        config,
        run_dir,
        parent_operator_ids=[PARENT],
        train_maps=[train],
    )
    assert evidence["counts"]["failure_modes"] == 1
    assert Path(evidence["canonical_path"]).exists()

    proposed = propose_operator_workflow(
        config,
        run_dir,
        provider="mock",
        mode="staged",
        parent_operator_ids=[PARENT],
        train_maps=[train],
    )
    assert proposed["status"] == CandidateStatus.REVIEWED.value
    assert proposed["compile_executed"] is False
    assert proposed["formal_validation_executed"] is False

    agent = run_agent_workflow(
        config,
        run_dir,
        provider="mock",
        parent_operator_ids=[PARENT],
        train_maps=[train],
        smoke_environment=train,
    )
    assert agent["status"] == CandidateStatus.SMOKE_PASSED.value
    assert agent["formal_validation_executed"] is False

    with AgentAuditStore(run_dir / "experiment.sqlite") as audit:
        proposal_events = audit.list_candidate_events(proposed["candidate_id"])
        assert [item.status for item in proposal_events] == [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
        ]
        tool_calls = audit.list_tool_calls(agent["agent_run_id"])
        assert tool_calls[-2].tool_name == "compile_operator_spec"
        assert tool_calls[-1].tool_name == "run_operator_smoke_test"
        assert all("validat" not in item.tool_name for item in tool_calls)

    with pytest.raises(ValueError, match="held-out test maps"):
        validate_candidate_workflow(
            config,
            run_dir,
            proposed["candidate_id"],
            [heldout],
            forbidden_map_hashes=[heldout.content_hash],
        )

    validated = validate_candidate_workflow(
        config,
        run_dir,
        proposed["candidate_id"],
        [validation],
        forbidden_map_hashes=[heldout.content_hash],
    )
    assert validated["validation_map_ids"] == ["validation-only"]
    assert validated["test_split_accessed"] is False
    assert all(
        item["map_id"] == "validation-only"
        for item in validated["validation_report"]["outcomes"]
    )
    with AgentAuditStore(run_dir / "experiment.sqlite") as audit:
        assert audit.get_candidate_status(proposed["candidate_id"]) in {
            CandidateStatus.ACCEPTED,
            CandidateStatus.REJECTED,
        }
    with MechanismMemory(run_dir / "experiment.sqlite") as memory:
        if validated["retained"]:
            assert memory.get_mechanism(validated["operator_name"]) is not None
            assert memory.get_lineage(validated["operator_name"])
        else:
            assert memory.get_failure_modes(validated["operator_name"])


def test_run_agent_multi_mode_persists_portfolio_and_role_audit(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = _seed_run(tmp_path, config)
    train = _map("multi-train", 20, 41)

    result = run_agent_workflow(
        config,
        run_dir,
        provider="mock",
        agent_mode="multi_agent",
        parent_operator_ids=[PARENT],
        train_maps=[train],
        smoke_environment=train,
    )

    assert result["agent_mode"] == "multi_agent"
    assert result["portfolio"]["selected_candidate_id"] == result["selected_candidate_id"]
    assert len(result["portfolio"]["candidates"]) == 2
    assert result["usage"]["turns"] == 4
    assert result["usage"]["tool_calls"] == 12
    with AgentAuditStore(run_dir / "experiment.sqlite") as audit:
        runs = audit.list_multi_agent_runs(result["agent_run_id"])
        assert len(runs) == 1
        portfolio = audit.get_candidate_portfolio(runs[0].portfolio_id or "")
        assert portfolio is not None
        assert portfolio.portfolio_hash == result["portfolio"]["portfolio_hash"]
        events = audit.list_multi_agent_role_events(runs[0].multi_agent_run_id)
        assert [event.action for event in events] == [
            "diagnose",
            "design",
            "design",
            "review",
            "select",
        ]
        assert len(result["role_traces"]) == 4
        llm_calls = audit.list_llm_calls(result["agent_run_id"])
        assert len(llm_calls) == 4
        llm_by_id = {call.call_id: call for call in llm_calls}
        for event, trace in zip(events[:-1], result["role_traces"]):
            assert event.input_hash == trace["input_hash"]
            assert event.output_hash == trace["output_hash"]
            assert event.prompt_hash == trace["prompt_hash"]
            assert event.summary_input_hash == stable_hash(event.input_summary)
            assert event.summary_output_hash == stable_hash(event.output_summary)
            assert event.provider_call_id in llm_by_id
            assert (
                llm_by_id[event.provider_call_id].prompt["provider_prompt_hash"]
                == trace["prompt_hash"]
            )
        assert all(
            "validat" not in call.tool_name.lower()
            for call in audit.list_tool_calls(result["agent_run_id"])
        )


def test_agent_demo_audits_state_memory_and_keeps_test_after_retention(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config = config.model_copy(
        update={
            "maps": config.maps.model_copy(
                update={
                    "train": config.maps.train.model_copy(
                        update={"count": 1, "difficulties": ["sparse"], "width": 40.0, "height": 40.0}
                    ),
                    "validation": config.maps.validation.model_copy(
                        update={"count": 1, "difficulties": ["medium"], "width": 42.0, "height": 42.0}
                    ),
                    "test": config.maps.test.model_copy(
                        update={"count": 1, "difficulties": ["mixed"], "width": 46.0, "height": 46.0}
                    ),
                }
            )
        }
    )
    report = agent_demo_workflow(
        config,
        provider="mock",
        run_id="workflow-agent-demo",
        run_dir=tmp_path / "demo-run",
    )

    assert Path(report["run_dir"], "agent_demo.json").exists()
    evidence_path = Path(report["evidence_bundle"]["canonical_path"])
    assert evidence_path.exists()
    assert report["evidence_bundle"]["bundle_hash"] in evidence_path.read_text(
        encoding="utf-8"
    )
    assert report["search"]["trace_count"] == config.search.train_iterations
    assert report["split_guard"]["test_split_used_for_retention"] is False
    assert set(report["split_guard"]["retention_map_hashes"]).isdisjoint(
        report["split_guard"]["heldout_test_hashes"]
    )
    orchestration = report["orchestration"]
    statuses = [item["status"] for item in report["audit"]["candidate_events"]]
    assert statuses[0] == CandidateStatus.PROPOSED.value
    assert statuses[-1] in {
        CandidateStatus.ACCEPTED.value,
        CandidateStatus.REJECTED.value,
    }
    assert len(report["audit"]["llm_calls"]) >= 2
    assert len(report["audit"]["tool_calls"]) >= 10
    assert report["memory"]["mechanism_count"] >= 1
    if orchestration["retained"]:
        assert report["test_comparison"]["executed"] is True
        assert report["memory"]["mechanism"] is not None
        assert report["memory"]["lineage"]
    else:
        assert report["test_comparison"]["executed"] is False


def test_provider_factory_never_silently_falls_back_to_mock(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("UOE_LLM_API_KEY", raising=False)
    monkeypatch.delenv("UOE_LLM_MODEL", raising=False)
    assert create_llm_provider("mock").__class__.__name__ == "MockLLMProvider"
    with pytest.raises(Exception, match="UOE_LLM_MODEL"):
        create_llm_provider("openai")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        create_llm_provider("silent-fallback")


def test_agent_ablations_share_parent_maps_seed_budget_and_report_tokens(tmp_path: Path) -> None:
    config = _config(tmp_path)
    run_dir = _seed_run(tmp_path, config)
    paths = RunPaths(
        run_id="ablation-run",
        result_dir=run_dir,
        figure_dir=tmp_path / "figures" / "ablation-run",
        database=run_dir / "experiment.sqlite",
        log_file=run_dir / "run.log",
    )
    report = run_agent_ablations_workflow(
        config,
        provider="mock",
        paths=paths,
        train_maps=[_map("ablation-train", 20, 10)],
        validation_maps=[_map("ablation-validation", 22, 11)],
        parent_operator_ids=[PARENT],
    )
    assert [row["arm"] for row in report["arms"]] == [
        "heuristic",
        "score_only_llm",
        "diagnostic_llm",
        "diagnosis_memory_llm",
        "single_agent",
        "multi_agent",
    ]
    assert report["shared_validation_map_ids"] == ["ablation-validation"]
    assert report["shared_candidate_index"] == 0
    assert report["test_split_accessed"] is False
    assert report["token_summary"]["heuristic"] == 0
    assert report["token_summary"]["single_agent"] > 0
    assert report["token_summary"]["multi_agent"] > 0
    single_report = report["validation_reports"]["single_agent"]
    assert single_report["outcomes"][0]["runtime_repetitions"] == 4
    assert len(single_report["outcomes"][0]["parent_runtime_samples_ms"]) == 4
    assert len(single_report["outcomes"][0]["candidate_runtime_samples_ms"]) == 4
    assert (run_dir / "agent_ablations.json").exists()
