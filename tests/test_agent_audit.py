"""Focused tests for the append-only agent audit persistence layer."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from uav_operator_evolution.agents.audit import (
    AUDIT_SCHEMA_VERSION,
    AgentAuditStore,
    AgentBudget,
    AgentRunRecord,
    AgentToolCallRecord,
    AgentUsage,
    AuthorizationDecision,
    CandidateEventRecord,
    CandidateTransitionError,
    EvidenceBundleRecord,
    LLMCallRecord,
    ModelUsage,
    bounded_json_summary,
)
from uav_operator_evolution.agents.design_models import CandidateStatus
from uav_operator_evolution.reproducibility import stable_hash


def _record_bundle(store: AgentAuditStore) -> EvidenceBundleRecord:
    return store.record_evidence_bundle(
        EvidenceBundleRecord(
            bundle_id="bundle-1",
            experiment_id="experiment-1",
            run_id="search-1",
            candidate_id="candidate-1",
            bundle={
                "z": 2,
                "a": {"api_key": "sk-do-not-store", "sample_count": 8},
                "prompt_note": "Authorization: Bearer hidden-token",
            },
            metadata={"access_token": "also-secret", "source": "diagnosis"},
        )
    )


def _record_agent_run(store: AgentAuditStore) -> AgentRunRecord:
    return store.record_agent_run(
        AgentRunRecord(
            agent_run_id="agent-run-1",
            experiment_id="experiment-1",
            provider="local",
            mode="deterministic-review",
            budget=AgentBudget(
                max_steps=20,
                max_tool_calls=10,
                max_llm_calls=2,
                max_tokens=4_000,
                max_wall_time_ms=30_000.0,
            ),
            usage=AgentUsage(
                steps=7,
                tool_calls=2,
                llm_calls=1,
                tokens=1_250,
                wall_time_ms=321.5,
            ),
            local_trace_id="local-trace-001",
            sdk_trace_id="sdk-trace-xyz",
            status="completed",
            metadata={"authorization": "must-not-leak", "phase": "design"},
        )
    )


def test_schema_bootstrap_shares_database_and_enforces_append_only(tmp_path: Path) -> None:
    database = tmp_path / "experiment.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE existing_results(id INTEGER PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO existing_results(value) VALUES ('preserved')")
    connection.commit()
    connection.close()

    with AgentAuditStore(database) as store:
        _record_bundle(store)
        table_names = {
            row["name"]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "existing_results",
            "evidence_bundles",
            "llm_calls",
            "agent_runs",
            "agent_tool_calls",
            "candidate_events",
            "agent_schema_meta",
        }.issubset(table_names)
        assert store.connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        meta = store.connection.execute(
            "SELECT version FROM agent_schema_meta WHERE schema_name = 'agent_audit'"
        ).fetchall()
        assert [row["version"] for row in meta] == [AUDIT_SCHEMA_VERSION]
        assert store.connection.execute("SELECT value FROM existing_results").fetchone()[0] == "preserved"
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute(
                "UPDATE evidence_bundles SET experiment_id = 'changed' WHERE bundle_id = 'bundle-1'"
            )
        store.connection.rollback()
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            store.connection.execute("DELETE FROM evidence_bundles WHERE bundle_id = 'bundle-1'")
        store.connection.rollback()

    with AgentAuditStore(database) as reopened:
        count = reopened.connection.execute(
            "SELECT COUNT(*) FROM agent_schema_meta WHERE schema_name = 'agent_audit'"
        ).fetchone()[0]
        assert count == 1


def test_evidence_bundle_is_canonical_hashed_and_recursively_redacted(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        persisted = _record_bundle(store)
        row = store.connection.execute(
            "SELECT * FROM evidence_bundles WHERE bundle_id = ?", (persisted.bundle_id,)
        ).fetchone()
        assert row is not None
        assert "sk-do-not-store" not in row["bundle_json"]
        assert "hidden-token" not in row["bundle_json"]
        assert "also-secret" not in row["metadata_json"]
        assert row["bundle_json"] == json.dumps(
            json.loads(row["bundle_json"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        assert row["bundle_hash"] == persisted.bundle_hash
        restored = store.get_evidence_bundle(persisted.bundle_id)
        assert restored == persisted
        assert restored is not None
        assert restored.bundle["a"]["api_key"] == "[REDACTED]"


def test_self_addressed_evidence_bundle_keeps_its_experiment_hash(tmp_path: Path) -> None:
    payload = {"bundle_version": "1", "problem_summary": "bounded evidence"}
    payload["bundle_hash"] = stable_hash(payload)
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        record = store.record_evidence_bundle(
            EvidenceBundleRecord(
                bundle_id="self-addressed",
                experiment_id="experiment-1",
                bundle=payload,
                bundle_hash=payload["bundle_hash"],
            )
        )
        assert record.bundle_hash == payload["bundle_hash"]
        assert record.bundle["bundle_hash"] == payload["bundle_hash"]


def test_llm_agent_and_tool_audit_records_capture_required_fields(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        bundle = _record_bundle(store)
        agent_run = _record_agent_run(store)
        llm = store.record_llm_call(
            LLMCallRecord(
                call_id="llm-call-1",
                experiment_id="experiment-1",
                agent_run_id=agent_run.agent_run_id,
                candidate_id="candidate-1",
                bundle_id=bundle.bundle_id,
                provider="test-provider",
                model="schema-model-v1",
                response_id="resp-123",
                prompt_version="operator-design-v3",
                prompt={
                    "instruction": "return JSON",
                    "headers": {"Authorization": "Bearer llm-secret"},
                },
                response={"name": "candidate", "valid": True},
                usage=ModelUsage(
                    input_tokens=100,
                    output_tokens=25,
                    cached_tokens=10,
                    total_tokens=125,
                    cost_usd=0.01,
                    extra={"provider_request_id": "req-1"},
                ),
                retries=2,
                latency_ms=42.5,
                status="succeeded",
                error=None,
            )
        )
        assert llm.prompt_hash is not None
        assert "llm-secret" not in json.dumps(llm.prompt)
        restored_calls = store.list_llm_calls(agent_run.agent_run_id)
        assert restored_calls == [llm]
        assert restored_calls[0].usage.total_tokens == 125
        assert restored_calls[0].response_id == "resp-123"
        restored_run = store.get_agent_run(agent_run.agent_run_id)
        assert restored_run == agent_run
        assert restored_run is not None
        assert restored_run.local_trace_id == "local-trace-001"
        assert restored_run.sdk_trace_id == "sdk-trace-xyz"

        huge_value = "x" * 30_000
        first = store.record_tool_call(
            AgentToolCallRecord(
                tool_call_id="tool-call-1",
                agent_run_id=agent_run.agent_run_id,
                sequence=0,
                tool_name="read_evidence",
                authorization=AuthorizationDecision.READ_ONLY,
                arguments={"api_key": "tool-secret", "payload": huge_value},
                result={"rows": list(range(250)), "access_token": "result-secret"},
                latency_ms=3.5,
                status="succeeded",
            )
        )
        second = store.record_tool_call(
            AgentToolCallRecord(
                tool_call_id="tool-call-2",
                agent_run_id=agent_run.agent_run_id,
                sequence=1,
                tool_name="compile_candidate",
                authorization=AuthorizationDecision.WORKSPACE_WRITE,
                arguments={"candidate": "candidate-1"},
                result={"compiled": True},
                latency_ms=8.0,
                status="succeeded",
            )
        )
        calls = store.list_tool_calls(agent_run.agent_run_id)
        assert calls == [first, second]
        raw = store.connection.execute(
            "SELECT arguments_summary_json, result_summary_json FROM agent_tool_calls "
            "WHERE tool_call_id = 'tool-call-1'"
        ).fetchone()
        assert len(raw["arguments_summary_json"]) <= 8_192
        assert len(raw["result_summary_json"]) <= 8_192
        assert "tool-secret" not in raw["arguments_summary_json"]
        assert "result-secret" not in raw["result_summary_json"]


def test_candidate_progression_rejection_and_terminal_guards(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        statuses = [
            CandidateStatus.PROPOSED,
            CandidateStatus.SCHEMA_VALID,
            CandidateStatus.REVIEWED,
            CandidateStatus.COMPILED,
            CandidateStatus.SMOKE_PASSED,
            CandidateStatus.VALIDATED,
            CandidateStatus.ACCEPTED,
        ]
        events = [
            store.record_candidate_event(
                CandidateEventRecord(
                    candidate_id="accepted-candidate",
                    status=status,
                    reason=f"passed {status.value}",
                )
            )
            for status in statuses
        ]
        assert [event.sequence for event in events] == list(range(len(statuses)))
        assert events[0].previous_status is None
        assert events[-1].previous_status == CandidateStatus.VALIDATED
        assert store.get_candidate_status("accepted-candidate") == CandidateStatus.ACCEPTED
        assert store.list_candidate_events("accepted-candidate") == events
        with pytest.raises(CandidateTransitionError, match="terminal"):
            store.record_candidate_event(
                CandidateEventRecord(
                    candidate_id="accepted-candidate",
                    status=CandidateStatus.REJECTED,
                    reason="too late",
                )
            )

        proposed = store.record_candidate_event(
            CandidateEventRecord(
                candidate_id="rejected-candidate",
                status=CandidateStatus.PROPOSED,
                reason="generated from evidence",
            )
        )
        rejected = store.record_candidate_event(
            CandidateEventRecord(
                candidate_id="rejected-candidate",
                status=CandidateStatus.REJECTED,
                reason="schema review failed",
            )
        )
        assert proposed.sequence == 0
        assert rejected.previous_status == CandidateStatus.PROPOSED
        with pytest.raises(CandidateTransitionError, match="terminal"):
            store.record_candidate_event(
                CandidateEventRecord(
                    candidate_id="rejected-candidate",
                    status=CandidateStatus.SCHEMA_VALID,
                    reason="cannot resume",
                )
            )

        with pytest.raises(CandidateTransitionError, match="begin"):
            store.record_candidate_event(
                CandidateEventRecord(
                    candidate_id="bad-first-state",
                    status=CandidateStatus.SCHEMA_VALID,
                    reason="illegal jump",
                )
            )
        store.record_candidate_event(
            CandidateEventRecord(
                candidate_id="jumping-candidate",
                status=CandidateStatus.PROPOSED,
                reason="created",
            )
        )
        with pytest.raises(CandidateTransitionError, match="invalid candidate transition"):
            store.record_candidate_event(
                CandidateEventRecord(
                    candidate_id="jumping-candidate",
                    status=CandidateStatus.COMPILED,
                    reason="skipped review",
                )
            )


def test_database_trigger_rejects_direct_invalid_transition(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        with pytest.raises(sqlite3.IntegrityError, match="begin in PROPOSED"):
            store.connection.execute(
                """
                INSERT INTO candidate_events(
                    event_id, candidate_id, sequence, previous_status, status, reason,
                    agent_run_id, evidence_bundle_id, details_json, created_at
                ) VALUES ('event-x', 'candidate-x', 0, NULL, 'COMPILED', 'invalid',
                          NULL, NULL, '{}', '2026-01-01T00:00:00+00:00')
                """
            )
        store.connection.rollback()


def test_models_are_strict_and_summary_bound_is_hard() -> None:
    with pytest.raises(ValidationError):
        AgentBudget(max_steps="10")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="timezone-aware"):
        AgentRunRecord(
            experiment_id="experiment",
            provider="local",
            mode="test",
            local_trace_id="trace",
            status="completed",
            started_at=datetime(2026, 1, 1),
        )
    with pytest.raises(ValidationError):
        EvidenceBundleRecord(
            experiment_id="experiment",
            bundle={},
            unexpected="forbidden",  # type: ignore[call-arg]
        )
    summary = bounded_json_summary(
        {"secret": "do-not-store", "payload": "quoted \" text " * 5_000},
        max_chars=512,
    )
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    assert len(encoded) <= 512
    assert "do-not-store" not in encoded
