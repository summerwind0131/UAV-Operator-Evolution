"""Append-only audit records for evidence-grounded designer and agent runs.

The audit store intentionally owns only ``agent_*`` and explicitly named audit
tables.  It can therefore share an experiment SQLite database with trajectory
and mechanism-memory stores without migrating or updating their schemas.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..reproducibility import canonical_json, stable_hash
from .design_models import CandidateStatus

AUDIT_SCHEMA_NAME = "agent_audit"
AUDIT_SCHEMA_VERSION = 3
REDACTED = "[REDACTED]"
MAX_BUNDLE_BYTES = 2_000_000
MAX_PORTFOLIO_BYTES = 2_000_000
MAX_LLM_PAYLOAD_CHARS = 262_144
DEFAULT_SUMMARY_CHARS = 8_192

_SENSITIVE_COMPACT_KEYS = {
    "apikey",
    "apitoken",
    "authorization",
    "authtoken",
    "bearertoken",
    "clientsecret",
    "cookie",
    "credentials",
    "idtoken",
    "password",
    "privatekey",
    "refreshtoken",
    "secret",
    "sessioncookie",
    "token",
}
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(api[_ -]?key|access[_ -]?token|refresh[_ -]?token|authorization|"
    r"client[_ -]?secret|password|private[_ -]?key|secret)\b\s*[:=]\s*"
    r"(?:bearer\s+)?([^\s,;]+)"
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _is_sensitive_key(key: object) -> bool:
    compact = re.sub(r"[^a-z0-9]", "", str(key).lower())
    if compact in _SENSITIVE_COMPACT_KEYS:
        return True
    return compact.endswith(("apikey", "apitoken", "accesstoken", "refreshtoken", "clientsecret"))


def _redact_string(value: str) -> str:
    result = _BEARER_PATTERN.sub(f"Bearer {REDACTED}", value)
    return _SENSITIVE_VALUE_PATTERN.sub(lambda match: f"{match.group(1)}={REDACTED}", result)


def redact_sensitive(value: Any) -> Any:
    """Recursively convert a value to JSON-safe data and redact secret-bearing keys."""

    if isinstance(value, BaseModel):
        return redact_sensitive(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, (set, frozenset)):
        redacted = [redact_sensitive(item) for item in value]
        return sorted(redacted, key=canonical_json)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return redact_sensitive(value.value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, str):
        return _redact_string(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _redact_string(str(value))


def _bounded_structure(
    value: Any,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string_chars: int,
) -> Any:
    if depth >= max_depth:
        return {"_truncated": "maximum depth reached"}
    value = redact_sensitive(value)
    if isinstance(value, Mapping):
        items = list(value.items())
        result = {
            str(key): _bounded_structure(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for key, item in items[:max_items]
        }
        if len(items) > max_items:
            result["_truncated_items"] = len(items) - max_items
        return result
    if isinstance(value, list):
        result = [
            _bounded_structure(
                item,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_items]
        ]
        if len(value) > max_items:
            result.append({"_truncated_items": len(value) - max_items})
        return result
    if isinstance(value, str) and len(value) > max_string_chars:
        return {
            "_truncated": True,
            "preview": value[:max_string_chars],
            "original_chars": len(value),
        }
    return value


def bounded_json_summary(
    value: Any,
    *,
    max_chars: int = DEFAULT_SUMMARY_CHARS,
    max_depth: int = 8,
    max_items: int = 100,
    max_string_chars: int = 2_048,
) -> Any:
    """Return a recursively redacted JSON summary with deterministic hard bounds."""

    if max_chars < 128:
        raise ValueError("max_chars must be at least 128")
    structured = _bounded_structure(
        value,
        depth=0,
        max_depth=max_depth,
        max_items=max_items,
        max_string_chars=max_string_chars,
    )
    encoded = canonical_json(structured)
    if len(encoded) <= max_chars:
        return structured
    low, high = 0, min(len(encoded), max_chars)
    summary: dict[str, Any] = {"_truncated": True, "preview": "", "original_chars": len(encoded)}
    while low <= high:
        middle = (low + high) // 2
        candidate = {**summary, "preview": encoded[:middle]}
        if len(canonical_json(candidate)) <= max_chars:
            summary = candidate
            low = middle + 1
        else:
            high = middle - 1
    return summary


class AuditModel(BaseModel):
    """Strict immutable base for persisted audit rows."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    @field_validator("*", mode="before")
    @classmethod
    def no_naive_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime) and value.tzinfo is None:
            raise ValueError("audit timestamps must be timezone-aware")
        return value


class AgentBudget(AuditModel):
    max_steps: int | None = Field(default=None, ge=0)
    max_tool_calls: int | None = Field(default=None, ge=0)
    max_llm_calls: int | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, ge=0)
    max_wall_time_ms: float | None = Field(default=None, ge=0)


class AgentUsage(AuditModel):
    steps: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    llm_calls: int = Field(default=0, ge=0)
    tokens: int = Field(default=0, ge=0)
    wall_time_ms: float = Field(default=0.0, ge=0)


class ModelUsage(AuditModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_total(self) -> "ModelUsage":
        minimum = self.input_tokens + self.output_tokens
        if self.total_tokens and self.total_tokens < minimum:
            raise ValueError("total_tokens cannot be below input_tokens + output_tokens")
        return self


class EvidenceBundleRecord(AuditModel):
    bundle_id: str = Field(default_factory=lambda: _new_id("bundle"), min_length=1, max_length=200)
    experiment_id: str = Field(min_length=1, max_length=200)
    run_id: str | None = Field(default=None, max_length=200)
    candidate_id: str | None = Field(default=None, max_length=200)
    bundle: dict[str, Any]
    bundle_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class LLMCallRecord(AuditModel):
    call_id: str = Field(default_factory=lambda: _new_id("llm"), min_length=1, max_length=200)
    experiment_id: str = Field(min_length=1, max_length=200)
    agent_run_id: str | None = Field(default=None, max_length=200)
    candidate_id: str | None = Field(default=None, max_length=200)
    bundle_id: str | None = Field(default=None, max_length=200)
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=200)
    response_id: str | None = Field(default=None, max_length=300)
    prompt_version: str = Field(min_length=1, max_length=100)
    prompt: Any
    prompt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    response: Any | None = None
    usage: ModelUsage = Field(default_factory=ModelUsage)
    retries: int = Field(default=0, ge=0, le=100)
    latency_ms: float = Field(default=0.0, ge=0)
    status: Literal["succeeded", "failed", "fallback"]
    error: str | None = Field(default=None, max_length=20_000)
    created_at: datetime = Field(default_factory=_utc_now)


class AgentRunRecord(AuditModel):
    agent_run_id: str = Field(default_factory=lambda: _new_id("agent"), min_length=1, max_length=200)
    experiment_id: str = Field(min_length=1, max_length=200)
    provider: str = Field(min_length=1, max_length=100)
    mode: str = Field(min_length=1, max_length=100)
    budget: AgentBudget = Field(default_factory=AgentBudget)
    usage: AgentUsage = Field(default_factory=AgentUsage)
    local_trace_id: str = Field(min_length=1, max_length=300)
    sdk_trace_id: str | None = Field(default=None, max_length=300)
    status: Literal["completed", "failed", "cancelled"]
    error: str | None = Field(default=None, max_length=20_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def completed_after_start(self) -> "AgentRunRecord":
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        return self


class AuthorizationDecision(str, Enum):
    NOT_REQUIRED = "not_required"
    READ_ONLY = "read_only"
    WORKSPACE_WRITE = "workspace_write"
    APPROVED_EXTERNAL = "approved_external"
    DENIED = "denied"


class AgentToolCallRecord(AuditModel):
    tool_call_id: str = Field(default_factory=lambda: _new_id("tool"), min_length=1, max_length=200)
    agent_run_id: str = Field(min_length=1, max_length=200)
    sequence: int = Field(ge=0)
    tool_name: str = Field(min_length=1, max_length=300)
    authorization: AuthorizationDecision
    arguments: Any = Field(default_factory=dict)
    result: Any | None = None
    latency_ms: float = Field(default=0.0, ge=0)
    status: Literal["succeeded", "failed", "denied", "timeout"]
    error: str | None = Field(default=None, max_length=20_000)
    created_at: datetime = Field(default_factory=_utc_now)


class CandidateEventRecord(AuditModel):
    event_id: str = Field(default_factory=lambda: _new_id("candidate_event"), min_length=1, max_length=200)
    candidate_id: str = Field(min_length=1, max_length=200)
    status: CandidateStatus
    reason: str = Field(min_length=1, max_length=20_000)
    previous_status: CandidateStatus | None = None
    sequence: int | None = Field(default=None, ge=0)
    agent_run_id: str | None = Field(default=None, max_length=200)
    evidence_bundle_id: str | None = Field(default=None, max_length=200)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utc_now)


class MultiAgentRunRecord(AuditModel):
    """Coordinator-level audit row for one deterministic multi-agent run."""

    multi_agent_run_id: str = Field(
        default_factory=lambda: _new_id("multi_agent_run"), min_length=1, max_length=200
    )
    agent_run_id: str = Field(min_length=1, max_length=200)
    coordinator_version: str = Field(min_length=1, max_length=100)
    bundle_id: str = Field(min_length=1, max_length=200)
    bundle_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    budget: AgentBudget = Field(default_factory=AgentBudget)
    usage: AgentUsage = Field(default_factory=AgentUsage)
    portfolio_id: str | None = Field(default=None, max_length=200)
    portfolio: dict[str, Any] | None = None
    portfolio_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str | None = Field(default=None, max_length=200)
    selection_reason: str | None = Field(default=None, max_length=20_000)
    status: Literal["completed", "failed", "cancelled"]
    error: str | None = Field(default=None, max_length=20_000)
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime = Field(default_factory=_utc_now)

    @model_validator(mode="after")
    def completed_after_start(self) -> "MultiAgentRunRecord":
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot be earlier than started_at")
        if self.portfolio is None and self.portfolio_hash is not None:
            # A caller may persist a compact coordinator row that refers to the
            # separately stored portfolio by hash.  It must provide an id too,
            # so the relationship remains replayable without a reverse lookup.
            if self.portfolio_id is None:
                raise ValueError("portfolio_id is required when only portfolio_hash is supplied")
        return self


class CandidatePortfolioRecord(AuditModel):
    """Full canonical portfolio payload selected by one coordinator run."""

    portfolio_id: str = Field(
        default_factory=lambda: _new_id("portfolio"), min_length=1, max_length=200
    )
    multi_agent_run_id: str = Field(min_length=1, max_length=200)
    bundle_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    portfolio: dict[str, Any]
    portfolio_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    selected_candidate_id: str | None = Field(default=None, max_length=200)
    selection_reason: str = Field(min_length=1, max_length=20_000)
    created_at: datetime = Field(default_factory=_utc_now)


class MultiAgentRoleEventRecord(AuditModel):
    """One ordered role action, optionally linked to its provider call."""

    role_event_id: str = Field(
        default_factory=lambda: _new_id("role_event"), min_length=1, max_length=200
    )
    multi_agent_run_id: str = Field(min_length=1, max_length=200)
    sequence: int = Field(ge=0)
    agent_role: str = Field(min_length=1, max_length=100)
    action: Literal["diagnose", "design", "review", "select"]
    candidate_id: str | None = Field(default=None, max_length=200)
    prompt_version: str | None = Field(default=None, max_length=100)
    prompt_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    provider_call_id: str | None = Field(default=None, max_length=200)
    input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    # ``input_hash``/``output_hash`` identify the exact payload passed to and
    # returned by the structured provider.  The persisted summaries are
    # deliberately compact/redacted and therefore have their own hashes.
    summary_input_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    summary_output_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    input_summary: Any = Field(default_factory=dict)
    output_summary: Any | None = None
    tokens: int = Field(default=0, ge=0)
    latency_ms: float = Field(default=0.0, ge=0)
    status: Literal["succeeded", "failed", "skipped"]
    error: str | None = Field(default=None, max_length=20_000)
    created_at: datetime = Field(default_factory=_utc_now)


_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS agent_schema_meta (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    schema_name TEXT NOT NULL,
    version INTEGER NOT NULL,
    applied_at TEXT NOT NULL,
    UNIQUE(schema_name, version)
);

CREATE TABLE IF NOT EXISTS evidence_bundles (
    bundle_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    run_id TEXT,
    candidate_id TEXT,
    bundle_hash TEXT NOT NULL,
    bundle_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_evidence_bundles_experiment
    ON evidence_bundles(experiment_id, created_at, bundle_id);
CREATE INDEX IF NOT EXISTS idx_evidence_bundles_hash
    ON evidence_bundles(bundle_hash);

CREATE TABLE IF NOT EXISTS agent_runs (
    agent_run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    mode TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    local_trace_id TEXT NOT NULL,
    sdk_trace_id TEXT,
    status TEXT NOT NULL CHECK(status IN ('completed', 'failed', 'cancelled')),
    error TEXT,
    metadata_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_runs_experiment
    ON agent_runs(experiment_id, started_at, agent_run_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    call_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL,
    agent_run_id TEXT,
    candidate_id TEXT,
    bundle_id TEXT,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    response_id TEXT,
    prompt_version TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_json TEXT NOT NULL,
    response_json TEXT,
    usage_json TEXT NOT NULL,
    retries INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed', 'fallback')),
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(agent_run_id),
    FOREIGN KEY(bundle_id) REFERENCES evidence_bundles(bundle_id)
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run
    ON llm_calls(agent_run_id, created_at, call_id);

CREATE TABLE IF NOT EXISTS agent_tool_calls (
    tool_call_id TEXT PRIMARY KEY,
    agent_run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    authorization TEXT NOT NULL,
    arguments_summary_json TEXT NOT NULL,
    result_summary_json TEXT,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed', 'denied', 'timeout')),
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(agent_run_id),
    UNIQUE(agent_run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_agent_tool_calls_run
    ON agent_tool_calls(agent_run_id, sequence);

CREATE TABLE IF NOT EXISTS candidate_events (
    event_id TEXT PRIMARY KEY,
    candidate_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    previous_status TEXT,
    status TEXT NOT NULL CHECK(status IN (
        'PROPOSED', 'SCHEMA_VALID', 'REVIEWED', 'COMPILED',
        'SMOKE_PASSED', 'VALIDATED', 'ACCEPTED', 'REJECTED'
    )),
    reason TEXT NOT NULL,
    agent_run_id TEXT,
    evidence_bundle_id TEXT,
    details_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(agent_run_id),
    FOREIGN KEY(evidence_bundle_id) REFERENCES evidence_bundles(bundle_id),
    UNIQUE(candidate_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_candidate_events_candidate
    ON candidate_events(candidate_id, sequence);

CREATE TRIGGER IF NOT EXISTS audit_candidate_transition
BEFORE INSERT ON candidate_events
BEGIN
    SELECT CASE
        WHEN NOT EXISTS (
            SELECT 1 FROM candidate_events WHERE candidate_id = NEW.candidate_id
        ) AND (
            NEW.sequence != 0 OR NEW.previous_status IS NOT NULL OR NEW.status != 'PROPOSED'
        ) THEN RAISE(ABORT, 'candidate must begin in PROPOSED at sequence 0')
        WHEN EXISTS (
            SELECT 1 FROM candidate_events WHERE candidate_id = NEW.candidate_id
        ) AND NEW.sequence != (
            SELECT MAX(sequence) + 1 FROM candidate_events WHERE candidate_id = NEW.candidate_id
        ) THEN RAISE(ABORT, 'candidate event sequence must be contiguous')
        WHEN EXISTS (
            SELECT 1 FROM candidate_events
            WHERE candidate_id = NEW.candidate_id AND status IN ('ACCEPTED', 'REJECTED')
            ORDER BY sequence DESC LIMIT 1
        ) THEN RAISE(ABORT, 'candidate terminal status cannot transition')
        WHEN EXISTS (
            SELECT 1 FROM candidate_events WHERE candidate_id = NEW.candidate_id
        ) AND NEW.previous_status IS NOT (
            SELECT status FROM candidate_events
            WHERE candidate_id = NEW.candidate_id ORDER BY sequence DESC LIMIT 1
        ) THEN RAISE(ABORT, 'candidate previous_status mismatch')
        WHEN NEW.status != 'REJECTED' AND EXISTS (
            SELECT 1 FROM candidate_events WHERE candidate_id = NEW.candidate_id
        ) AND NOT (
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'PROPOSED' AND NEW.status = 'SCHEMA_VALID') OR
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'SCHEMA_VALID' AND NEW.status = 'REVIEWED') OR
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'REVIEWED' AND NEW.status = 'COMPILED') OR
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'COMPILED' AND NEW.status = 'SMOKE_PASSED') OR
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'SMOKE_PASSED' AND NEW.status = 'VALIDATED') OR
            ((SELECT status FROM candidate_events WHERE candidate_id = NEW.candidate_id
              ORDER BY sequence DESC LIMIT 1) = 'VALIDATED' AND NEW.status = 'ACCEPTED')
        ) THEN RAISE(ABORT, 'invalid candidate status transition')
    END;
END;

CREATE TABLE IF NOT EXISTS multi_agent_runs (
    multi_agent_run_id TEXT PRIMARY KEY,
    agent_run_id TEXT NOT NULL UNIQUE,
    coordinator_version TEXT NOT NULL,
    bundle_id TEXT NOT NULL,
    bundle_hash TEXT NOT NULL,
    budget_json TEXT NOT NULL,
    usage_json TEXT NOT NULL,
    portfolio_id TEXT,
    portfolio_hash TEXT,
    portfolio_summary_json TEXT,
    selected_candidate_id TEXT,
    selection_reason TEXT,
    status TEXT NOT NULL CHECK(status IN ('completed', 'failed', 'cancelled')),
    error TEXT,
    metadata_json TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    FOREIGN KEY(agent_run_id) REFERENCES agent_runs(agent_run_id),
    FOREIGN KEY(bundle_id) REFERENCES evidence_bundles(bundle_id)
);
CREATE INDEX IF NOT EXISTS idx_multi_agent_runs_bundle
    ON multi_agent_runs(bundle_id, started_at, multi_agent_run_id);

CREATE TABLE IF NOT EXISTS candidate_portfolios (
    portfolio_id TEXT PRIMARY KEY,
    multi_agent_run_id TEXT NOT NULL UNIQUE,
    bundle_hash TEXT,
    portfolio_hash TEXT NOT NULL,
    portfolio_json TEXT NOT NULL,
    selected_candidate_id TEXT,
    selection_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(multi_agent_run_id) REFERENCES multi_agent_runs(multi_agent_run_id)
);
CREATE INDEX IF NOT EXISTS idx_candidate_portfolios_hash
    ON candidate_portfolios(portfolio_hash, created_at, portfolio_id);

CREATE TABLE IF NOT EXISTS multi_agent_role_events (
    role_event_id TEXT PRIMARY KEY,
    multi_agent_run_id TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    agent_role TEXT NOT NULL,
    action TEXT NOT NULL CHECK(action IN ('diagnose', 'design', 'review', 'select')),
    candidate_id TEXT,
    prompt_version TEXT,
    prompt_hash TEXT,
    provider_call_id TEXT,
    input_hash TEXT NOT NULL,
    output_hash TEXT,
    summary_input_hash TEXT,
    summary_output_hash TEXT,
    input_summary_json TEXT NOT NULL,
    output_summary_json TEXT,
    tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed', 'skipped')),
    error TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(multi_agent_run_id) REFERENCES multi_agent_runs(multi_agent_run_id),
    FOREIGN KEY(provider_call_id) REFERENCES llm_calls(call_id),
    UNIQUE(multi_agent_run_id, sequence)
);
CREATE INDEX IF NOT EXISTS idx_multi_agent_role_events_run
    ON multi_agent_role_events(multi_agent_run_id, sequence);
CREATE INDEX IF NOT EXISTS idx_multi_agent_role_events_provider
    ON multi_agent_role_events(provider_call_id);
"""

_APPEND_ONLY_TABLES = (
    "evidence_bundles",
    "llm_calls",
    "agent_runs",
    "agent_tool_calls",
    "candidate_events",
    "multi_agent_runs",
    "candidate_portfolios",
    "multi_agent_role_events",
    "agent_schema_meta",
)

_STATUS_ORDER = (
    CandidateStatus.PROPOSED,
    CandidateStatus.SCHEMA_VALID,
    CandidateStatus.REVIEWED,
    CandidateStatus.COMPILED,
    CandidateStatus.SMOKE_PASSED,
    CandidateStatus.VALIDATED,
    CandidateStatus.ACCEPTED,
)
_TERMINAL_STATUSES = {CandidateStatus.ACCEPTED, CandidateStatus.REJECTED}


class CandidateTransitionError(ValueError):
    """Raised when a candidate audit event violates the fixed progression."""


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _canonical_payload_hash(value: Any, *, self_hash_key: str | None = None) -> str:
    """Hash canonical redacted JSON, excluding an optional self-address field."""

    if self_hash_key is not None and isinstance(value, Mapping):
        embedded_hash = value.get(self_hash_key)
        if isinstance(embedded_hash, str) and re.fullmatch(r"[0-9a-f]{64}", embedded_hash):
            payload = dict(value)
            payload.pop(self_hash_key, None)
            payload_hash = stable_hash(payload)
            if embedded_hash != payload_hash:
                raise ValueError(
                    f"embedded {self_hash_key} does not match canonical redacted JSON"
                )
            return payload_hash
    return stable_hash(value)


class AgentAuditStore:
    """Append-only SQLite store for evidence, LLM, agent, tool, and candidate audit data."""

    def __init__(self, database: str | Path | sqlite3.Connection) -> None:
        self.db_path: Path | None = None
        self._owns_connection = not isinstance(database, sqlite3.Connection)
        if isinstance(database, sqlite3.Connection):
            self._connection = database
        else:
            if str(database) != ":memory:":
                self.db_path = Path(database)
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connection = sqlite3.connect(
                str(database), timeout=30.0, check_same_thread=False
            )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        self._bootstrap()

    @property
    def connection(self) -> sqlite3.Connection:
        self._ensure_open()
        return self._connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("agent audit store is closed")

    def _bootstrap(self) -> None:
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if self.db_path is not None:
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
            # Version 2 adds the provider response identifier.  SQLite has no
            # IF NOT EXISTS form for ADD COLUMN, so inspect before migrating an
            # experiment database created by an earlier Phase-8 build.
            llm_columns = {
                str(row["name"])
                for row in self._connection.execute("PRAGMA table_info(llm_calls)").fetchall()
            }
            if "response_id" not in llm_columns:
                self._connection.execute("ALTER TABLE llm_calls ADD COLUMN response_id TEXT")
            # Schema v3 originally stored hashes derived from the compact role
            # summaries in ``input_hash``/``output_hash``.  Keep v3 databases
            # compatible while separating those summary hashes from the exact
            # provider payload hashes on subsequent writes.
            role_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(multi_agent_role_events)"
                ).fetchall()
            }
            if "summary_input_hash" not in role_columns:
                self._connection.execute(
                    "ALTER TABLE multi_agent_role_events ADD COLUMN summary_input_hash TEXT"
                )
            if "summary_output_hash" not in role_columns:
                self._connection.execute(
                    "ALTER TABLE multi_agent_role_events ADD COLUMN summary_output_hash TEXT"
                )
            for table in _APPEND_ONLY_TABLES:
                self._connection.executescript(
                    f"""
                    CREATE TRIGGER IF NOT EXISTS audit_no_update_{table}
                    BEFORE UPDATE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END;
                    CREATE TRIGGER IF NOT EXISTS audit_no_delete_{table}
                    BEFORE DELETE ON {table}
                    BEGIN
                        SELECT RAISE(ABORT, '{table} is append-only');
                    END;
                    """
                )
            rows = self._connection.execute(
                "SELECT version FROM agent_schema_meta WHERE schema_name = ? ORDER BY version",
                (AUDIT_SCHEMA_NAME,),
            ).fetchall()
            versions = [int(row["version"]) for row in rows]
            if versions and max(versions) > AUDIT_SCHEMA_VERSION:
                raise RuntimeError(
                    f"agent audit schema {max(versions)} is newer than supported "
                    f"version {AUDIT_SCHEMA_VERSION}"
                )
            if AUDIT_SCHEMA_VERSION not in versions:
                self._connection.execute(
                    "INSERT INTO agent_schema_meta(schema_name, version, applied_at) VALUES (?, ?, ?)",
                    (AUDIT_SCHEMA_NAME, AUDIT_SCHEMA_VERSION, _utc_now().isoformat()),
                )
            self._connection.commit()

    def record_evidence_bundle(self, record: EvidenceBundleRecord) -> EvidenceBundleRecord:
        self._ensure_open()
        redacted_bundle = redact_sensitive(record.bundle)
        bundle_json = _json(redacted_bundle)
        if len(bundle_json.encode("utf-8")) > MAX_BUNDLE_BYTES:
            raise ValueError(f"evidence bundle exceeds {MAX_BUNDLE_BYTES} bytes")
        # OperatorEvidenceBundle is self-addressed: its public bundle_hash is
        # computed from the canonical payload with that field excluded.  Keep
        # that experiment hash in SQLite instead of hashing the hash again.
        embedded_hash = (
            redacted_bundle.get("bundle_hash")
            if isinstance(redacted_bundle, Mapping)
            else None
        )
        if isinstance(embedded_hash, str) and re.fullmatch(r"[0-9a-f]{64}", embedded_hash):
            hash_payload = dict(redacted_bundle)
            hash_payload.pop("bundle_hash", None)
            bundle_hash = stable_hash(hash_payload)
            if embedded_hash != bundle_hash:
                raise ValueError("embedded bundle_hash does not match canonical bundle JSON")
        else:
            bundle_hash = stable_hash(redacted_bundle)
        if record.bundle_hash is not None and record.bundle_hash != bundle_hash:
            raise ValueError("bundle_hash does not match canonical redacted bundle JSON")
        metadata = bounded_json_summary(record.metadata)
        persisted = record.model_copy(
            update={"bundle": redacted_bundle, "bundle_hash": bundle_hash, "metadata": metadata}
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO evidence_bundles(
                    bundle_id, experiment_id, run_id, candidate_id, bundle_hash,
                    bundle_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.bundle_id,
                    persisted.experiment_id,
                    persisted.run_id,
                    persisted.candidate_id,
                    bundle_hash,
                    bundle_json,
                    _json(metadata),
                    persisted.created_at.isoformat(),
                ),
            )
        return persisted

    def record_agent_run(self, record: AgentRunRecord) -> AgentRunRecord:
        self._ensure_open()
        metadata = bounded_json_summary(record.metadata)
        redacted_error = _redact_string(record.error) if record.error else None
        persisted = record.model_copy(update={"metadata": metadata, "error": redacted_error})
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agent_runs(
                    agent_run_id, experiment_id, provider, mode, budget_json, usage_json,
                    local_trace_id, sdk_trace_id, status, error, metadata_json,
                    started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.agent_run_id,
                    persisted.experiment_id,
                    persisted.provider,
                    persisted.mode,
                    _json(persisted.budget.model_dump(mode="json")),
                    _json(persisted.usage.model_dump(mode="json")),
                    persisted.local_trace_id,
                    persisted.sdk_trace_id,
                    persisted.status,
                    persisted.error,
                    _json(metadata),
                    persisted.started_at.isoformat(),
                    persisted.completed_at.isoformat(),
                ),
            )
        return persisted

    def record_multi_agent_run(self, record: MultiAgentRunRecord) -> MultiAgentRunRecord:
        """Persist the coordinator row after its parent agent run and bundle exist."""

        self._ensure_open()
        portfolio = None if record.portfolio is None else redact_sensitive(record.portfolio)
        portfolio_hash = record.portfolio_hash
        portfolio_summary = None
        if portfolio is not None:
            portfolio_json = _json(portfolio)
            if len(portfolio_json.encode("utf-8")) > MAX_PORTFOLIO_BYTES:
                raise ValueError(f"candidate portfolio exceeds {MAX_PORTFOLIO_BYTES} bytes")
            computed_hash = _canonical_payload_hash(portfolio, self_hash_key="portfolio_hash")
            if portfolio_hash is not None and portfolio_hash != computed_hash:
                raise ValueError("portfolio_hash does not match canonical redacted portfolio")
            portfolio_hash = computed_hash
            portfolio_summary = bounded_json_summary(portfolio)
        metadata = bounded_json_summary(record.metadata)
        redacted_error = _redact_string(record.error) if record.error else None
        selection_reason = (
            _redact_string(record.selection_reason) if record.selection_reason else None
        )
        persisted = record.model_copy(
            update={
                "portfolio": portfolio_summary,
                "portfolio_hash": portfolio_hash,
                "selection_reason": selection_reason,
                "error": redacted_error,
                "metadata": metadata,
            }
        )
        with self._lock, self._connection:
            bundle_row = self._connection.execute(
                "SELECT bundle_hash FROM evidence_bundles WHERE bundle_id = ?",
                (persisted.bundle_id,),
            ).fetchone()
            if bundle_row is not None and str(bundle_row["bundle_hash"]) != persisted.bundle_hash:
                raise ValueError("multi-agent bundle_hash does not match evidence bundle")
            self._connection.execute(
                """
                INSERT INTO multi_agent_runs(
                    multi_agent_run_id, agent_run_id, coordinator_version, bundle_id,
                    bundle_hash, budget_json, usage_json, portfolio_id, portfolio_hash,
                    portfolio_summary_json, selected_candidate_id, selection_reason,
                    status, error, metadata_json, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.multi_agent_run_id,
                    persisted.agent_run_id,
                    persisted.coordinator_version,
                    persisted.bundle_id,
                    persisted.bundle_hash,
                    _json(persisted.budget.model_dump(mode="json")),
                    _json(persisted.usage.model_dump(mode="json")),
                    persisted.portfolio_id,
                    persisted.portfolio_hash,
                    None if portfolio_summary is None else _json(portfolio_summary),
                    persisted.selected_candidate_id,
                    persisted.selection_reason,
                    persisted.status,
                    persisted.error,
                    _json(metadata),
                    persisted.started_at.isoformat(),
                    persisted.completed_at.isoformat(),
                ),
            )
        return persisted

    def record_candidate_portfolio(
        self, record: CandidatePortfolioRecord
    ) -> CandidatePortfolioRecord:
        """Persist one complete, canonical, replayable candidate portfolio."""

        self._ensure_open()
        portfolio = redact_sensitive(record.portfolio)
        portfolio_json = _json(portfolio)
        if len(portfolio_json.encode("utf-8")) > MAX_PORTFOLIO_BYTES:
            raise ValueError(f"candidate portfolio exceeds {MAX_PORTFOLIO_BYTES} bytes")
        portfolio_hash = _canonical_payload_hash(portfolio, self_hash_key="portfolio_hash")
        if record.portfolio_hash is not None and record.portfolio_hash != portfolio_hash:
            raise ValueError("portfolio_hash does not match canonical redacted portfolio")
        selection_reason = _redact_string(record.selection_reason)
        persisted = record.model_copy(
            update={
                "portfolio": portfolio,
                "portfolio_hash": portfolio_hash,
                "selection_reason": selection_reason,
            }
        )
        with self._lock, self._connection:
            run_row = self._connection.execute(
                """
                SELECT bundle_hash, portfolio_id, portfolio_hash, selected_candidate_id
                FROM multi_agent_runs WHERE multi_agent_run_id = ?
                """,
                (persisted.multi_agent_run_id,),
            ).fetchone()
            if run_row is not None:
                if persisted.bundle_hash is not None and persisted.bundle_hash != run_row["bundle_hash"]:
                    raise ValueError("portfolio bundle_hash does not match multi-agent run")
                if run_row["portfolio_id"] is not None and run_row["portfolio_id"] != persisted.portfolio_id:
                    raise ValueError("portfolio_id does not match multi-agent run")
                if (
                    run_row["portfolio_hash"] is not None
                    and run_row["portfolio_hash"] != portfolio_hash
                ):
                    raise ValueError("portfolio_hash does not match multi-agent run")
                if (
                    run_row["selected_candidate_id"] is not None
                    and run_row["selected_candidate_id"] != persisted.selected_candidate_id
                ):
                    raise ValueError("selected candidate does not match multi-agent run")
            self._connection.execute(
                """
                INSERT INTO candidate_portfolios(
                    portfolio_id, multi_agent_run_id, bundle_hash, portfolio_hash,
                    portfolio_json, selected_candidate_id, selection_reason, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.portfolio_id,
                    persisted.multi_agent_run_id,
                    persisted.bundle_hash,
                    portfolio_hash,
                    portfolio_json,
                    persisted.selected_candidate_id,
                    persisted.selection_reason,
                    persisted.created_at.isoformat(),
                ),
            )
        return persisted

    def record_role_event(
        self, record: MultiAgentRoleEventRecord
    ) -> MultiAgentRoleEventRecord:
        """Append one ordered role event after any linked provider call exists."""

        self._ensure_open()
        input_payload = redact_sensitive(record.input_summary)
        output_payload = (
            None if record.output_summary is None else redact_sensitive(record.output_summary)
        )
        input_summary = bounded_json_summary(input_payload)
        output_summary = (
            None if output_payload is None else bounded_json_summary(output_payload)
        )
        summary_input_hash = stable_hash(input_summary)
        summary_output_hash = (
            None if output_summary is None else stable_hash(output_summary)
        )
        if (
            record.summary_input_hash is not None
            and record.summary_input_hash != summary_input_hash
        ):
            raise ValueError(
                "summary_input_hash does not match canonical persisted role input summary"
            )
        if (
            record.summary_output_hash is not None
            and record.summary_output_hash != summary_output_hash
        ):
            raise ValueError(
                "summary_output_hash does not match canonical persisted role output summary"
            )
        # Older callers did not carry the provider hashes.  Falling back to
        # the summary hash preserves their replay behaviour without replacing
        # an explicitly supplied MultiAgentRoleTrace hash.
        input_hash = record.input_hash or summary_input_hash
        output_hash = (
            record.output_hash
            if "output_hash" in record.model_fields_set
            else summary_output_hash
        )
        redacted_error = _redact_string(record.error) if record.error else None
        prompt_version = record.prompt_version
        prompt_hash = record.prompt_hash
        with self._lock, self._connection:
            if record.provider_call_id is not None:
                provider_row = self._connection.execute(
                    """
                    SELECT llm_calls.agent_run_id, llm_calls.prompt_version,
                           llm_calls.prompt_hash, llm_calls.prompt_json,
                           multi_agent_runs.agent_run_id AS expected_run_id
                    FROM llm_calls
                    JOIN multi_agent_runs ON multi_agent_runs.multi_agent_run_id = ?
                    WHERE llm_calls.call_id = ?
                    """,
                    (record.multi_agent_run_id, record.provider_call_id),
                ).fetchone()
                if provider_row is not None:
                    if provider_row["agent_run_id"] != provider_row["expected_run_id"]:
                        raise ValueError("provider call belongs to a different agent run")
                    if prompt_version is not None and prompt_version != provider_row["prompt_version"]:
                        raise ValueError("role prompt_version does not match provider call")
                    provider_prompt = json.loads(provider_row["prompt_json"])
                    linked_prompt_hash = (
                        provider_prompt.get("provider_prompt_hash")
                        if isinstance(provider_prompt, Mapping)
                        else None
                    )
                    if not (
                        isinstance(linked_prompt_hash, str)
                        and re.fullmatch(r"[0-9a-f]{64}", linked_prompt_hash)
                    ):
                        # Legacy/manual LLM audit rows only have the audit hash
                        # of ``prompt_json``.  Use it as a compatibility value.
                        linked_prompt_hash = str(provider_row["prompt_hash"])
                    if prompt_hash is not None and prompt_hash != linked_prompt_hash:
                        raise ValueError("role prompt_hash does not match provider call")
                    prompt_version = str(provider_row["prompt_version"])
                    prompt_hash = linked_prompt_hash
            persisted = record.model_copy(
                update={
                    "prompt_version": prompt_version,
                    "prompt_hash": prompt_hash,
                    "input_hash": input_hash,
                    "output_hash": output_hash,
                    "summary_input_hash": summary_input_hash,
                    "summary_output_hash": summary_output_hash,
                    "input_summary": input_summary,
                    "output_summary": output_summary,
                    "error": redacted_error,
                }
            )
            self._connection.execute(
                """
                INSERT INTO multi_agent_role_events(
                    role_event_id, multi_agent_run_id, sequence, agent_role, action,
                    candidate_id, prompt_version, prompt_hash, provider_call_id,
                    input_hash, output_hash, summary_input_hash, summary_output_hash,
                    input_summary_json, output_summary_json, tokens, latency_ms,
                    status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.role_event_id,
                    persisted.multi_agent_run_id,
                    persisted.sequence,
                    persisted.agent_role,
                    persisted.action,
                    persisted.candidate_id,
                    persisted.prompt_version,
                    persisted.prompt_hash,
                    persisted.provider_call_id,
                    input_hash,
                    output_hash,
                    summary_input_hash,
                    summary_output_hash,
                    _json(input_summary),
                    None if output_summary is None else _json(output_summary),
                    persisted.tokens,
                    persisted.latency_ms,
                    persisted.status,
                    persisted.error,
                    persisted.created_at.isoformat(),
                ),
            )
        return persisted

    # Explicit alias retained for callers that prefer the table-qualified name.
    def record_multi_agent_role_event(
        self, record: MultiAgentRoleEventRecord
    ) -> MultiAgentRoleEventRecord:
        return self.record_role_event(record)

    def record_llm_call(self, record: LLMCallRecord) -> LLMCallRecord:
        self._ensure_open()
        prompt = bounded_json_summary(record.prompt, max_chars=MAX_LLM_PAYLOAD_CHARS)
        response = (
            None
            if record.response is None
            else bounded_json_summary(record.response, max_chars=MAX_LLM_PAYLOAD_CHARS)
        )
        prompt_hash = stable_hash(prompt)
        if record.prompt_hash is not None and record.prompt_hash != prompt_hash:
            raise ValueError("prompt_hash does not match persisted redacted prompt")
        usage_payload = record.usage.model_dump(mode="json")
        usage_payload["extra"] = bounded_json_summary(record.usage.extra)
        persisted_usage = ModelUsage.model_validate(usage_payload)
        redacted_error = _redact_string(record.error) if record.error else None
        persisted = record.model_copy(
            update={
                "prompt": prompt,
                "prompt_hash": prompt_hash,
                "response": response,
                "usage": persisted_usage,
                "error": redacted_error,
            }
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO llm_calls(
                    call_id, experiment_id, agent_run_id, candidate_id, bundle_id,
                    provider, model, response_id, prompt_version, prompt_hash, prompt_json,
                    response_json, usage_json, retries, latency_ms, status, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.call_id,
                    persisted.experiment_id,
                    persisted.agent_run_id,
                    persisted.candidate_id,
                    persisted.bundle_id,
                    persisted.provider,
                    persisted.model,
                    persisted.response_id,
                    persisted.prompt_version,
                    prompt_hash,
                    _json(prompt),
                    None if response is None else _json(response),
                    _json(persisted_usage.model_dump(mode="json")),
                    persisted.retries,
                    persisted.latency_ms,
                    persisted.status,
                    persisted.error,
                    persisted.created_at.isoformat(),
                ),
            )
        return persisted

    def record_tool_call(self, record: AgentToolCallRecord) -> AgentToolCallRecord:
        self._ensure_open()
        arguments = bounded_json_summary(record.arguments)
        result = None if record.result is None else bounded_json_summary(record.result)
        redacted_error = _redact_string(record.error) if record.error else None
        persisted = record.model_copy(
            update={"arguments": arguments, "result": result, "error": redacted_error}
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO agent_tool_calls(
                    tool_call_id, agent_run_id, sequence, tool_name, authorization,
                    arguments_summary_json, result_summary_json, latency_ms, status,
                    error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    persisted.tool_call_id,
                    persisted.agent_run_id,
                    persisted.sequence,
                    persisted.tool_name,
                    persisted.authorization.value,
                    _json(arguments),
                    None if result is None else _json(result),
                    persisted.latency_ms,
                    persisted.status,
                    persisted.error,
                    persisted.created_at.isoformat(),
                ),
            )
        return persisted

    def record_candidate_event(self, record: CandidateEventRecord) -> CandidateEventRecord:
        """Validate and atomically append one candidate state transition."""

        self._ensure_open()
        details = bounded_json_summary(record.details)
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    """
                    SELECT sequence, status FROM candidate_events
                    WHERE candidate_id = ? ORDER BY sequence DESC LIMIT 1
                    """,
                    (record.candidate_id,),
                ).fetchone()
                previous = None if row is None else CandidateStatus(row["status"])
                sequence = 0 if row is None else int(row["sequence"]) + 1
                self._validate_transition(previous, record.status)
                if record.previous_status is not None and record.previous_status != previous:
                    raise CandidateTransitionError(
                        f"declared previous status {record.previous_status.value} does not match "
                        f"persisted status {None if previous is None else previous.value}"
                    )
                if record.sequence is not None and record.sequence != sequence:
                    raise CandidateTransitionError(
                        f"declared sequence {record.sequence} does not match next sequence {sequence}"
                    )
                redacted_reason = _redact_string(record.reason)
                persisted = record.model_copy(
                    update={
                        "previous_status": previous,
                        "sequence": sequence,
                        "details": details,
                        "reason": redacted_reason,
                    }
                )
                self._connection.execute(
                    """
                    INSERT INTO candidate_events(
                        event_id, candidate_id, sequence, previous_status, status,
                        reason, agent_run_id, evidence_bundle_id, details_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        persisted.event_id,
                        persisted.candidate_id,
                        sequence,
                        None if previous is None else previous.value,
                        persisted.status.value,
                        persisted.reason,
                        persisted.agent_run_id,
                        persisted.evidence_bundle_id,
                        _json(details),
                        persisted.created_at.isoformat(),
                    ),
                )
                self._connection.commit()
                return persisted
            except Exception:
                self._connection.rollback()
                raise

    @staticmethod
    def _validate_transition(
        previous: CandidateStatus | None,
        status: CandidateStatus,
    ) -> None:
        if previous is None:
            if status != CandidateStatus.PROPOSED:
                raise CandidateTransitionError("a candidate must begin in PROPOSED")
            return
        if previous in _TERMINAL_STATUSES:
            raise CandidateTransitionError(f"no transition is allowed after terminal state {previous.value}")
        if status == CandidateStatus.REJECTED:
            return
        expected = _STATUS_ORDER[_STATUS_ORDER.index(previous) + 1]
        if status != expected:
            raise CandidateTransitionError(
                f"invalid candidate transition {previous.value} -> {status.value}; "
                f"expected {expected.value} or REJECTED"
            )

    def get_evidence_bundle(self, bundle_id: str) -> EvidenceBundleRecord | None:
        row = self._fetch_one("SELECT * FROM evidence_bundles WHERE bundle_id = ?", (bundle_id,))
        if row is None:
            return None
        return EvidenceBundleRecord(
            bundle_id=row["bundle_id"],
            experiment_id=row["experiment_id"],
            run_id=row["run_id"],
            candidate_id=row["candidate_id"],
            bundle=json.loads(row["bundle_json"]),
            bundle_hash=row["bundle_hash"],
            metadata=json.loads(row["metadata_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_agent_run(self, agent_run_id: str) -> AgentRunRecord | None:
        row = self._fetch_one("SELECT * FROM agent_runs WHERE agent_run_id = ?", (agent_run_id,))
        return None if row is None else self._agent_run_from_row(row)

    def get_multi_agent_run(self, multi_agent_run_id: str) -> MultiAgentRunRecord | None:
        row = self._fetch_one(
            "SELECT * FROM multi_agent_runs WHERE multi_agent_run_id = ?",
            (multi_agent_run_id,),
        )
        return None if row is None else self._multi_agent_run_from_row(row)

    def list_multi_agent_runs(self, agent_run_id: str | None = None) -> list[MultiAgentRunRecord]:
        if agent_run_id is None:
            rows = self._fetch_all(
                "SELECT * FROM multi_agent_runs ORDER BY started_at, multi_agent_run_id",
                (),
            )
        else:
            rows = self._fetch_all(
                """
                SELECT * FROM multi_agent_runs
                WHERE agent_run_id = ? ORDER BY started_at, multi_agent_run_id
                """,
                (agent_run_id,),
            )
        return [self._multi_agent_run_from_row(row) for row in rows]

    def get_candidate_portfolio(self, portfolio_id: str) -> CandidatePortfolioRecord | None:
        row = self._fetch_one(
            "SELECT * FROM candidate_portfolios WHERE portfolio_id = ?", (portfolio_id,)
        )
        return None if row is None else self._candidate_portfolio_from_row(row)

    def get_candidate_portfolio_by_hash(
        self, portfolio_hash: str
    ) -> CandidatePortfolioRecord | None:
        row = self._fetch_one(
            """
            SELECT * FROM candidate_portfolios WHERE portfolio_hash = ?
            ORDER BY created_at, portfolio_id LIMIT 1
            """,
            (portfolio_hash,),
        )
        return None if row is None else self._candidate_portfolio_from_row(row)

    def list_candidate_portfolios(
        self, multi_agent_run_id: str | None = None
    ) -> list[CandidatePortfolioRecord]:
        if multi_agent_run_id is None:
            rows = self._fetch_all(
                "SELECT * FROM candidate_portfolios ORDER BY created_at, portfolio_id", ()
            )
        else:
            rows = self._fetch_all(
                """
                SELECT * FROM candidate_portfolios WHERE multi_agent_run_id = ?
                ORDER BY created_at, portfolio_id
                """,
                (multi_agent_run_id,),
            )
        return [self._candidate_portfolio_from_row(row) for row in rows]

    def get_role_event(self, role_event_id: str) -> MultiAgentRoleEventRecord | None:
        row = self._fetch_one(
            "SELECT * FROM multi_agent_role_events WHERE role_event_id = ?", (role_event_id,)
        )
        return None if row is None else self._role_event_from_row(row)

    def get_multi_agent_role_event(
        self, role_event_id: str
    ) -> MultiAgentRoleEventRecord | None:
        return self.get_role_event(role_event_id)

    def list_role_events(self, multi_agent_run_id: str) -> list[MultiAgentRoleEventRecord]:
        rows = self._fetch_all(
            """
            SELECT * FROM multi_agent_role_events WHERE multi_agent_run_id = ?
            ORDER BY sequence
            """,
            (multi_agent_run_id,),
        )
        return [self._role_event_from_row(row) for row in rows]

    def list_multi_agent_role_events(
        self, multi_agent_run_id: str
    ) -> list[MultiAgentRoleEventRecord]:
        return self.list_role_events(multi_agent_run_id)

    def list_llm_calls(self, agent_run_id: str | None = None) -> list[LLMCallRecord]:
        if agent_run_id is None:
            rows = self._fetch_all("SELECT * FROM llm_calls ORDER BY created_at, call_id", ())
        else:
            rows = self._fetch_all(
                "SELECT * FROM llm_calls WHERE agent_run_id = ? ORDER BY created_at, call_id",
                (agent_run_id,),
            )
        return [self._llm_call_from_row(row) for row in rows]

    def list_tool_calls(self, agent_run_id: str) -> list[AgentToolCallRecord]:
        rows = self._fetch_all(
            "SELECT * FROM agent_tool_calls WHERE agent_run_id = ? ORDER BY sequence",
            (agent_run_id,),
        )
        return [self._tool_call_from_row(row) for row in rows]

    def list_candidate_events(self, candidate_id: str) -> list[CandidateEventRecord]:
        rows = self._fetch_all(
            "SELECT * FROM candidate_events WHERE candidate_id = ? ORDER BY sequence",
            (candidate_id,),
        )
        return [self._candidate_event_from_row(row) for row in rows]

    def get_candidate_status(self, candidate_id: str) -> CandidateStatus | None:
        row = self._fetch_one(
            "SELECT status FROM candidate_events WHERE candidate_id = ? ORDER BY sequence DESC LIMIT 1",
            (candidate_id,),
        )
        return None if row is None else CandidateStatus(row["status"])

    def _fetch_one(self, query: str, params: Sequence[Any]) -> sqlite3.Row | None:
        self._ensure_open()
        with self._lock:
            return self._connection.execute(query, params).fetchone()

    def _fetch_all(self, query: str, params: Sequence[Any]) -> list[sqlite3.Row]:
        self._ensure_open()
        with self._lock:
            return list(self._connection.execute(query, params).fetchall())

    @staticmethod
    def _agent_run_from_row(row: sqlite3.Row) -> AgentRunRecord:
        return AgentRunRecord(
            agent_run_id=row["agent_run_id"],
            experiment_id=row["experiment_id"],
            provider=row["provider"],
            mode=row["mode"],
            budget=AgentBudget.model_validate_json(row["budget_json"]),
            usage=AgentUsage.model_validate_json(row["usage_json"]),
            local_trace_id=row["local_trace_id"],
            sdk_trace_id=row["sdk_trace_id"],
            status=row["status"],
            error=row["error"],
            metadata=json.loads(row["metadata_json"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]),
        )

    @staticmethod
    def _llm_call_from_row(row: sqlite3.Row) -> LLMCallRecord:
        return LLMCallRecord(
            call_id=row["call_id"],
            experiment_id=row["experiment_id"],
            agent_run_id=row["agent_run_id"],
            candidate_id=row["candidate_id"],
            bundle_id=row["bundle_id"],
            provider=row["provider"],
            model=row["model"],
            response_id=row["response_id"],
            prompt_version=row["prompt_version"],
            prompt=json.loads(row["prompt_json"]),
            prompt_hash=row["prompt_hash"],
            response=None if row["response_json"] is None else json.loads(row["response_json"]),
            usage=ModelUsage.model_validate_json(row["usage_json"]),
            retries=int(row["retries"]),
            latency_ms=float(row["latency_ms"]),
            status=row["status"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _multi_agent_run_from_row(row: sqlite3.Row) -> MultiAgentRunRecord:
        return MultiAgentRunRecord(
            multi_agent_run_id=row["multi_agent_run_id"],
            agent_run_id=row["agent_run_id"],
            coordinator_version=row["coordinator_version"],
            bundle_id=row["bundle_id"],
            bundle_hash=row["bundle_hash"],
            budget=AgentBudget.model_validate_json(row["budget_json"]),
            usage=AgentUsage.model_validate_json(row["usage_json"]),
            portfolio_id=row["portfolio_id"],
            portfolio=None
            if row["portfolio_summary_json"] is None
            else json.loads(row["portfolio_summary_json"]),
            portfolio_hash=row["portfolio_hash"],
            selected_candidate_id=row["selected_candidate_id"],
            selection_reason=row["selection_reason"],
            status=row["status"],
            error=row["error"],
            metadata=json.loads(row["metadata_json"]),
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=datetime.fromisoformat(row["completed_at"]),
        )

    @staticmethod
    def _candidate_portfolio_from_row(row: sqlite3.Row) -> CandidatePortfolioRecord:
        return CandidatePortfolioRecord(
            portfolio_id=row["portfolio_id"],
            multi_agent_run_id=row["multi_agent_run_id"],
            bundle_hash=row["bundle_hash"],
            portfolio=json.loads(row["portfolio_json"]),
            portfolio_hash=row["portfolio_hash"],
            selected_candidate_id=row["selected_candidate_id"],
            selection_reason=row["selection_reason"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _role_event_from_row(row: sqlite3.Row) -> MultiAgentRoleEventRecord:
        return MultiAgentRoleEventRecord(
            role_event_id=row["role_event_id"],
            multi_agent_run_id=row["multi_agent_run_id"],
            sequence=int(row["sequence"]),
            agent_role=row["agent_role"],
            action=row["action"],
            candidate_id=row["candidate_id"],
            prompt_version=row["prompt_version"],
            prompt_hash=row["prompt_hash"],
            provider_call_id=row["provider_call_id"],
            input_hash=row["input_hash"],
            output_hash=row["output_hash"],
            summary_input_hash=row["summary_input_hash"],
            summary_output_hash=row["summary_output_hash"],
            input_summary=json.loads(row["input_summary_json"]),
            output_summary=None
            if row["output_summary_json"] is None
            else json.loads(row["output_summary_json"]),
            tokens=int(row["tokens"]),
            latency_ms=float(row["latency_ms"]),
            status=row["status"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _tool_call_from_row(row: sqlite3.Row) -> AgentToolCallRecord:
        return AgentToolCallRecord(
            tool_call_id=row["tool_call_id"],
            agent_run_id=row["agent_run_id"],
            sequence=int(row["sequence"]),
            tool_name=row["tool_name"],
            authorization=AuthorizationDecision(row["authorization"]),
            arguments=json.loads(row["arguments_summary_json"]),
            result=None
            if row["result_summary_json"] is None
            else json.loads(row["result_summary_json"]),
            latency_ms=float(row["latency_ms"]),
            status=row["status"],
            error=row["error"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _candidate_event_from_row(row: sqlite3.Row) -> CandidateEventRecord:
        return CandidateEventRecord(
            event_id=row["event_id"],
            candidate_id=row["candidate_id"],
            sequence=int(row["sequence"]),
            previous_status=None
            if row["previous_status"] is None
            else CandidateStatus(row["previous_status"]),
            status=CandidateStatus(row["status"]),
            reason=row["reason"],
            agent_run_id=row["agent_run_id"],
            evidence_bundle_id=row["evidence_bundle_id"],
            details=json.loads(row["details_json"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def close(self) -> None:
        if not self._closed:
            with self._lock:
                if self._owns_connection:
                    self._connection.close()
                self._closed = True

    def __enter__(self) -> "AgentAuditStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


__all__ = [
    "AUDIT_SCHEMA_NAME",
    "AUDIT_SCHEMA_VERSION",
    "AgentAuditStore",
    "AgentBudget",
    "AgentRunRecord",
    "AgentToolCallRecord",
    "AgentUsage",
    "AuthorizationDecision",
    "CandidateEventRecord",
    "CandidatePortfolioRecord",
    "CandidateTransitionError",
    "EvidenceBundleRecord",
    "LLMCallRecord",
    "ModelUsage",
    "MultiAgentRoleEventRecord",
    "MultiAgentRunRecord",
    "bounded_json_summary",
    "redact_sensitive",
]
