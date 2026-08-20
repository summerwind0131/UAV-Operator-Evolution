"""Audit-schema v3 coverage for deterministic multi-agent runs."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from uav_operator_evolution.agents.audit import (
    AUDIT_SCHEMA_VERSION,
    AgentAuditStore,
    AgentBudget,
    AgentRunRecord,
    AgentUsage,
    CandidatePortfolioRecord,
    EvidenceBundleRecord,
    LLMCallRecord,
    ModelUsage,
    MultiAgentRoleEventRecord,
    MultiAgentRunRecord,
)
from uav_operator_evolution.reproducibility import stable_hash


def _parents(store: AgentAuditStore) -> tuple[EvidenceBundleRecord, AgentRunRecord]:
    bundle = store.record_evidence_bundle(
        EvidenceBundleRecord(
            bundle_id="bundle-v3",
            experiment_id="experiment-v3",
            bundle={"evidence_ids": ["failure:1"], "diagnosis": "oscillation"},
        )
    )
    assert bundle.bundle_hash is not None
    run = store.record_agent_run(
        AgentRunRecord(
            agent_run_id="agent-v3",
            experiment_id="experiment-v3",
            provider="mock",
            mode="multi_agent",
            budget=AgentBudget(max_steps=4, max_tool_calls=12, max_llm_calls=4),
            usage=AgentUsage(steps=4, tool_calls=12, llm_calls=4, tokens=80),
            local_trace_id="trace-v3",
            status="completed",
        )
    )
    return bundle, run


def _portfolio() -> dict[str, object]:
    return {
        "bundle_hash": "a" * 64,
        "diagnosis_hash": "b" * 64,
        "candidates": [
            {"candidate_id": "exploit", "score": 0.8},
            {"candidate_id": "explore", "score": 0.7},
        ],
        "selected_candidate_id": "exploit",
        "selection_reason": "stable deterministic winner",
        "credentials": "must-not-be-stored",
    }


def _multi_run(
    bundle: EvidenceBundleRecord,
    run: AgentRunRecord,
    *, portfolio: dict[str, object] | None = None,
) -> MultiAgentRunRecord:
    assert bundle.bundle_hash is not None
    return MultiAgentRunRecord(
        multi_agent_run_id="multi-v3",
        agent_run_id=run.agent_run_id,
        coordinator_version="multi_agent_coordinator_v1",
        bundle_id=bundle.bundle_id,
        bundle_hash=bundle.bundle_hash,
        budget=AgentBudget(max_steps=4, max_tool_calls=12, max_llm_calls=4),
        usage=AgentUsage(steps=4, tool_calls=12, llm_calls=4, tokens=80),
        portfolio_id="portfolio-v3",
        portfolio=portfolio,
        selected_candidate_id="exploit",
        selection_reason="stable deterministic winner",
        status="completed",
    )


def test_v2_database_migrates_to_v3_without_rewriting_existing_rows(tmp_path: Path) -> None:
    database = tmp_path / "v2.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE agent_schema_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL,
            UNIQUE(schema_name, version)
        );
        INSERT INTO agent_schema_meta(schema_name, version, applied_at)
        VALUES ('agent_audit', 2, '2026-01-01T00:00:00+00:00');
        CREATE TABLE legacy_marker(value TEXT NOT NULL);
        INSERT INTO legacy_marker(value) VALUES ('preserved');
        """
    )
    connection.commit()
    connection.close()

    with AgentAuditStore(database) as store:
        assert AUDIT_SCHEMA_VERSION == 3
        versions = store.connection.execute(
            "SELECT version FROM agent_schema_meta WHERE schema_name = ? ORDER BY version",
            ("agent_audit",),
        ).fetchall()
        assert [int(row["version"]) for row in versions] == [2, 3]
        tables = {
            str(row["name"])
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert {
            "multi_agent_runs",
            "multi_agent_role_events",
            "candidate_portfolios",
        }.issubset(tables)
        assert store.connection.execute("SELECT value FROM legacy_marker").fetchone()[0] == "preserved"


def test_multi_agent_records_round_trip_canonically_and_redacted(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        bundle, agent_run = _parents(store)
        template_hash = stable_hash({"template": "designer_exploitation_v1"})
        provider_input_hash = stable_hash({"actual_provider_payload": [1, 2, 3]})
        provider_output_hash = stable_hash({"actual_provider_output": "proposal"})
        llm = store.record_llm_call(
            LLMCallRecord(
                call_id="provider-call-v3",
                experiment_id="experiment-v3",
                agent_run_id=agent_run.agent_run_id,
                bundle_id=bundle.bundle_id,
                provider="mock",
                model="deterministic-mock-v1",
                prompt_version="designer_exploitation_v1",
                prompt={
                    "task": "design",
                    "api_key": "prompt-secret",
                    "provider_prompt_hash": template_hash,
                },
                response={"candidate_id": "exploit"},
                usage=ModelUsage(input_tokens=10, output_tokens=5, total_tokens=15),
                status="succeeded",
            )
        )
        portfolio_payload = _portfolio()
        multi_run = store.record_multi_agent_run(
            _multi_run(bundle, agent_run, portfolio=portfolio_payload)
        )
        assert multi_run.portfolio_hash is not None
        portfolio = store.record_candidate_portfolio(
            CandidatePortfolioRecord(
                portfolio_id="portfolio-v3",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                bundle_hash=bundle.bundle_hash,
                portfolio=portfolio_payload,
                portfolio_hash=multi_run.portfolio_hash,
                selected_candidate_id="exploit",
                selection_reason="Authorization: Bearer selection-secret",
            )
        )
        event = store.record_role_event(
            MultiAgentRoleEventRecord(
                role_event_id="role-v3",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                sequence=0,
                agent_role="exploitation_designer",
                action="design",
                candidate_id="exploit",
                prompt_hash=template_hash,
                provider_call_id=llm.call_id,
                input_hash=provider_input_hash,
                output_hash=provider_output_hash,
                input_summary={"api_token": "input-secret", "bundle_hash": bundle.bundle_hash},
                output_summary={"proposal": "x" * 20_000, "password": "output-secret"},
                tokens=15,
                latency_ms=2.5,
                status="succeeded",
            )
        )

        assert event.prompt_version == llm.prompt_version
        assert event.prompt_hash == template_hash
        assert event.prompt_hash != llm.prompt_hash
        assert event.input_hash == provider_input_hash
        assert event.output_hash == provider_output_hash
        assert event.summary_input_hash == stable_hash(event.input_summary)
        assert event.summary_output_hash == stable_hash(event.output_summary)
        assert store.get_multi_agent_run(multi_run.multi_agent_run_id) == multi_run
        assert store.list_multi_agent_runs(agent_run.agent_run_id) == [multi_run]
        assert store.get_candidate_portfolio(portfolio.portfolio_id) == portfolio
        assert store.get_candidate_portfolio_by_hash(portfolio.portfolio_hash or "") == portfolio
        assert store.list_candidate_portfolios(multi_run.multi_agent_run_id) == [portfolio]
        assert store.get_role_event(event.role_event_id) == event
        assert store.list_role_events(multi_run.multi_agent_run_id) == [event]

        raw_portfolio = store.connection.execute(
            "SELECT portfolio_json FROM candidate_portfolios WHERE portfolio_id = ?",
            (portfolio.portfolio_id,),
        ).fetchone()[0]
        assert raw_portfolio == json.dumps(
            json.loads(raw_portfolio),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        raw_event = store.connection.execute(
            """
            SELECT input_hash, output_hash, summary_input_hash, summary_output_hash,
                   input_summary_json, output_summary_json
            FROM multi_agent_role_events
            """
        ).fetchone()
        assert raw_event["input_hash"] == provider_input_hash
        assert raw_event["output_hash"] == provider_output_hash
        assert raw_event["summary_input_hash"] == event.summary_input_hash
        assert raw_event["summary_output_hash"] == event.summary_output_hash
        persisted_text = raw_portfolio + raw_event["input_summary_json"] + raw_event["output_summary_json"]
        assert "must-not-be-stored" not in persisted_text
        assert "input-secret" not in persisted_text
        assert "output-secret" not in persisted_text
        assert "selection-secret" not in portfolio.selection_reason
        assert len(raw_event["output_summary_json"]) <= 8_192
        assert portfolio.portfolio_hash == stable_hash(
            {**portfolio.portfolio, "credentials": "[REDACTED]"}
        )


def test_existing_v3_role_table_adds_summary_hash_columns_in_place(
    tmp_path: Path,
) -> None:
    database = tmp_path / "early-v3.sqlite"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE agent_schema_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_name TEXT NOT NULL,
            version INTEGER NOT NULL,
            applied_at TEXT NOT NULL,
            UNIQUE(schema_name, version)
        );
        INSERT INTO agent_schema_meta(schema_name, version, applied_at)
        VALUES ('agent_audit', 3, '2026-01-01T00:00:00+00:00');
        CREATE TABLE multi_agent_role_events (
            role_event_id TEXT PRIMARY KEY,
            multi_agent_run_id TEXT NOT NULL,
            sequence INTEGER NOT NULL,
            agent_role TEXT NOT NULL,
            action TEXT NOT NULL,
            candidate_id TEXT,
            prompt_version TEXT,
            prompt_hash TEXT,
            provider_call_id TEXT,
            input_hash TEXT NOT NULL,
            output_hash TEXT,
            input_summary_json TEXT NOT NULL,
            output_summary_json TEXT,
            tokens INTEGER NOT NULL,
            latency_ms REAL NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(multi_agent_run_id, sequence)
        );
        INSERT INTO multi_agent_role_events(
            role_event_id, multi_agent_run_id, sequence, agent_role, action,
            input_hash, input_summary_json, tokens, latency_ms, status, created_at
        ) VALUES (
            'legacy-role', 'legacy-run', 0, 'diagnoser', 'diagnose',
            '1111111111111111111111111111111111111111111111111111111111111111',
            '{"legacy":true}', 0, 0.0, 'succeeded', '2026-01-01T00:00:00+00:00'
        );
        """
    )
    connection.commit()
    connection.close()

    with AgentAuditStore(database) as store:
        columns = {
            str(row["name"])
            for row in store.connection.execute(
                "PRAGMA table_info(multi_agent_role_events)"
            ).fetchall()
        }
        assert {"summary_input_hash", "summary_output_hash"}.issubset(columns)
        legacy = store.get_role_event("legacy-role")
        assert legacy is not None
        assert legacy.input_hash == "1" * 64
        assert legacy.summary_input_hash is None
        assert legacy.summary_output_hash is None


def test_v3_foreign_keys_enforce_documented_write_order(tmp_path: Path) -> None:
    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        bundle = store.record_evidence_bundle(
            EvidenceBundleRecord(
                bundle_id="bundle-v3",
                experiment_id="experiment-v3",
                bundle={"failure": "oscillation"},
            )
        )
        assert bundle.bundle_hash is not None
        absent_run = AgentRunRecord(
            agent_run_id="agent-v3",
            experiment_id="experiment-v3",
            provider="mock",
            mode="multi_agent",
            local_trace_id="trace-v3",
            status="completed",
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.record_multi_agent_run(_multi_run(bundle, absent_run))

        agent_run = store.record_agent_run(absent_run)
        multi_run = store.record_multi_agent_run(_multi_run(bundle, agent_run))
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            store.record_role_event(
                MultiAgentRoleEventRecord(
                    multi_agent_run_id=multi_run.multi_agent_run_id,
                    sequence=0,
                    agent_role="critic",
                    action="review",
                    provider_call_id="missing-provider-call",
                    status="failed",
                )
            )

        llm = store.record_llm_call(
            LLMCallRecord(
                call_id="provider-call-v3",
                experiment_id="experiment-v3",
                agent_run_id=agent_run.agent_run_id,
                bundle_id=bundle.bundle_id,
                provider="mock",
                model="mock-v1",
                prompt_version="critic_v1",
                prompt={"review": ["exploit", "explore"]},
                status="succeeded",
            )
        )
        store.record_candidate_portfolio(
            CandidatePortfolioRecord(
                portfolio_id="portfolio-v3",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                portfolio=_portfolio(),
                selected_candidate_id="exploit",
                selection_reason="winner",
            )
        )
        store.record_role_event(
            MultiAgentRoleEventRecord(
                role_event_id="role-0",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                sequence=0,
                agent_role="critic",
                action="review",
                provider_call_id=llm.call_id,
                status="succeeded",
            )
        )
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
            store.record_role_event(
                MultiAgentRoleEventRecord(
                    role_event_id="role-duplicate",
                    multi_agent_run_id=multi_run.multi_agent_run_id,
                    sequence=0,
                    agent_role="coordinator",
                    action="select",
                    status="succeeded",
                )
            )


def test_v3_tables_are_append_only_and_models_and_hashes_are_strict(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        MultiAgentRoleEventRecord(
            multi_agent_run_id="run",
            sequence=0,
            agent_role="critic",
            action="revise",  # type: ignore[arg-type]
            status="succeeded",
        )
    with pytest.raises(ValidationError):
        CandidatePortfolioRecord(
            multi_agent_run_id="run",
            portfolio={},
            selection_reason="none",
            unexpected=True,  # type: ignore[call-arg]
        )

    with AgentAuditStore(tmp_path / "audit.sqlite") as store:
        bundle, agent_run = _parents(store)
        multi_run = store.record_multi_agent_run(_multi_run(bundle, agent_run))
        with pytest.raises(ValueError, match="portfolio_hash"):
            store.record_candidate_portfolio(
                CandidatePortfolioRecord(
                    multi_agent_run_id=multi_run.multi_agent_run_id,
                    portfolio=_portfolio(),
                    portfolio_hash="0" * 64,
                    selected_candidate_id="exploit",
                    selection_reason="winner",
                )
            )
        portfolio = store.record_candidate_portfolio(
            CandidatePortfolioRecord(
                portfolio_id="portfolio-v3",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                portfolio=_portfolio(),
                selected_candidate_id="exploit",
                selection_reason="winner",
            )
        )
        event = store.record_role_event(
            MultiAgentRoleEventRecord(
                role_event_id="role-v3",
                multi_agent_run_id=multi_run.multi_agent_run_id,
                sequence=0,
                agent_role="coordinator",
                action="select",
                status="succeeded",
            )
        )
        for table, key, value in (
            ("multi_agent_runs", "multi_agent_run_id", multi_run.multi_agent_run_id),
            ("candidate_portfolios", "portfolio_id", portfolio.portfolio_id),
            ("multi_agent_role_events", "role_event_id", event.role_event_id),
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                store.connection.execute(
                    f"UPDATE {table} SET {key} = {key} WHERE {key} = ?", (value,)
                )
            store.connection.rollback()
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                store.connection.execute(f"DELETE FROM {table} WHERE {key} = ?", (value,))
            store.connection.rollback()
