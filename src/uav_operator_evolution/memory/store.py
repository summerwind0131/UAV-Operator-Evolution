"""SQLite implementation of mechanism memory and evidence retrieval."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from ..trajectory import OperatorTrace
from .models import (
    CaseRecord,
    FailureModeRecord,
    LineageRecord,
    MechanismInsight,
    MechanismRecord,
    OperatorHistoryRecord,
    OperatorProfileRecord,
    SynergyRecord,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS mechanisms (
    mechanism_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    definition_json TEXT NOT NULL,
    status TEXT NOT NULL,
    score REAL NOT NULL,
    evidence_count INTEGER NOT NULL,
    success_rate REAL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mechanisms_rank
    ON mechanisms(status, score DESC, evidence_count DESC);

CREATE TABLE IF NOT EXISTS operator_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id TEXT REFERENCES mechanisms(mechanism_id) ON DELETE SET NULL,
    operator_id TEXT NOT NULL,
    run_id TEXT,
    trace_id INTEGER,
    accepted INTEGER,
    immediate_reward REAL,
    delayed_rewards_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_operator
    ON operator_history(operator_id, history_id DESC);
CREATE INDEX IF NOT EXISTS idx_history_mechanism
    ON operator_history(mechanism_id, history_id DESC);

CREATE TABLE IF NOT EXISTS failure_modes (
    failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
    mechanism_id TEXT REFERENCES mechanisms(mechanism_id) ON DELETE SET NULL,
    operator_id TEXT,
    mode TEXT NOT NULL,
    count INTEGER NOT NULL,
    severity REAL NOT NULL,
    context_json TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_failures_operator
    ON failure_modes(operator_id, count DESC, severity DESC);

CREATE TABLE IF NOT EXISTS synergies (
    synergy_id INTEGER PRIMARY KEY AUTOINCREMENT,
    first_operator TEXT NOT NULL,
    second_operator TEXT NOT NULL,
    score REAL NOT NULL,
    sample_count INTEGER NOT NULL,
    context_json TEXT NOT NULL,
    mechanism_ids_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_synergies_pair
    ON synergies(first_operator, second_operator, score DESC);

CREATE TABLE IF NOT EXISTS cases (
    case_id TEXT PRIMARY KEY,
    mechanism_id TEXT REFERENCES mechanisms(mechanism_id) ON DELETE SET NULL,
    operator_id TEXT,
    outcome TEXT NOT NULL,
    score REAL,
    context_json TEXT NOT NULL,
    state_json TEXT NOT NULL,
    action_json TEXT NOT NULL,
    result_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cases_operator
    ON cases(operator_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_cases_mechanism
    ON cases(mechanism_id, created_at DESC);

CREATE TABLE IF NOT EXISTS lineage (
    lineage_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id TEXT NOT NULL REFERENCES mechanisms(mechanism_id) ON DELETE CASCADE,
    child_id TEXT NOT NULL REFERENCES mechanisms(mechanism_id) ON DELETE CASCADE,
    relation TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(parent_id, child_id, relation),
    CHECK(parent_id <> child_id)
);
CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage(parent_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage(child_id);

CREATE TABLE IF NOT EXISTS mechanism_insights (
    insight_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id TEXT NOT NULL,
    insight_type TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    confidence REAL NOT NULL,
    applicable_context_json TEXT NOT NULL,
    failure_context_json TEXT NOT NULL,
    source_profile_id TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_insights_operator
    ON mechanism_insights(operator_id, insight_type, insight_id DESC);

CREATE TABLE IF NOT EXISTS operator_profiles (
    profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
    operator_id TEXT NOT NULL,
    run_id TEXT,
    generation INTEGER NOT NULL,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_profiles_operator
    ON operator_profiles(operator_id, generation DESC, profile_id DESC);
"""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def _load(value: str) -> Any:
    return json.loads(value)


def _context_similarity(query: Mapping[str, Any], candidate: Mapping[str, Any]) -> float:
    """Simple exact-overlap score that remains interpretable in reports."""

    if not query:
        return 0.0
    matches = sum(
        1 for key, expected in query.items() if key in candidate and candidate[key] == expected
    )
    conflicts = sum(
        1 for key, expected in query.items() if key in candidate and candidate[key] != expected
    )
    return (matches - conflicts) / len(query)


class MechanismMemory:
    """Durable mechanisms plus the evidence needed to reuse them responsibly."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(
            str(db_path), timeout=30.0, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._closed = False
        with self._lock:
            self._connection.execute("PRAGMA foreign_keys = ON")
            self._connection.execute("PRAGMA busy_timeout = 30000")
            if self.db_path is not None:
                self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    @property
    def connection(self) -> sqlite3.Connection:
        self._ensure_open()
        return self._connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("mechanism memory is closed")

    # -- mechanisms -----------------------------------------------------
    def add_mechanism(
        self,
        mechanism: MechanismRecord | Mapping[str, Any] | str | None = None,
        definition: Any = None,
        *,
        mechanism_id: str | None = None,
        name: str | None = None,
        description: str = "",
        status: str = "active",
        score: float = 0.0,
        evidence_count: int = 0,
        success_rate: float | None = None,
        tags: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
        parent_ids: Sequence[str] = (),
    ) -> str:
        """Insert or replace a mechanism and return its stable identifier."""

        if isinstance(mechanism, MechanismRecord):
            record = mechanism.model_copy(deep=True)
        elif isinstance(mechanism, Mapping):
            payload = dict(mechanism)
            if definition is not None and "definition" not in payload:
                payload["definition"] = definition
            payload.setdefault("mechanism_id", mechanism_id or payload.get("id") or uuid.uuid4().hex)
            payload.setdefault("name", name or payload["mechanism_id"])
            record = MechanismRecord.model_validate(payload)
        else:
            inferred_id = mechanism_id or (str(mechanism) if mechanism is not None else uuid.uuid4().hex)
            record = MechanismRecord(
                mechanism_id=inferred_id,
                name=name or (str(mechanism) if mechanism is not None else inferred_id),
                description=description,
                definition={} if definition is None else definition,
                status=status,
                score=score,
                evidence_count=evidence_count,
                success_rate=success_rate,
                tags=list(tags),
                metadata=dict(metadata or {}),
            )
        values = (
            record.mechanism_id,
            record.name,
            record.description,
            _json(record.definition),
            record.status,
            record.score,
            record.evidence_count,
            record.success_rate,
            _json(record.tags),
            _json(record.metadata),
            record.created_at.isoformat(),
            record.updated_at.isoformat(),
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO mechanisms (
                    mechanism_id, name, description, definition_json, status,
                    score, evidence_count, success_rate, tags_json, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(mechanism_id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    definition_json=excluded.definition_json,
                    status=excluded.status,
                    score=excluded.score,
                    evidence_count=excluded.evidence_count,
                    success_rate=excluded.success_rate,
                    tags_json=excluded.tags_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                values,
            )
            for parent_id in parent_ids:
                self._insert_lineage(parent_id, record.mechanism_id, "derived_from", {})
        return record.mechanism_id

    create_mechanism = add_mechanism
    store_mechanism = add_mechanism
    remember_mechanism = add_mechanism
    upsert_mechanism = add_mechanism

    @staticmethod
    def _mechanism(row: sqlite3.Row) -> MechanismRecord:
        return MechanismRecord(
            mechanism_id=row["mechanism_id"],
            name=row["name"],
            description=row["description"],
            definition=_load(row["definition_json"]),
            status=row["status"],
            score=row["score"],
            evidence_count=row["evidence_count"],
            success_rate=row["success_rate"],
            tags=_load(row["tags_json"]),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_mechanism(self, mechanism_id: str) -> MechanismRecord | None:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM mechanisms WHERE mechanism_id = ?", (mechanism_id,)
            ).fetchone()
        return None if row is None else self._mechanism(row)

    def list_mechanisms(self, status: str | None = None) -> list[MechanismRecord]:
        self._ensure_open()
        query = "SELECT * FROM mechanisms"
        params: tuple[Any, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY score DESC, evidence_count DESC, mechanism_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._mechanism(row) for row in rows]

    def update_mechanism(
        self,
        mechanism_id: str,
        updates: Mapping[str, Any] | None = None,
        **changes: Any,
    ) -> MechanismRecord:
        current = self.get_mechanism(mechanism_id)
        if current is None:
            raise KeyError(mechanism_id)
        payload = current.model_dump()
        payload.update(dict(updates or {}))
        payload.update(changes)
        payload["mechanism_id"] = mechanism_id
        payload["updated_at"] = _now()
        record = MechanismRecord.model_validate(payload)
        self.add_mechanism(record)
        return record

    def delete_mechanism(self, mechanism_id: str) -> bool:
        self._ensure_open()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "DELETE FROM mechanisms WHERE mechanism_id = ?", (mechanism_id,)
            )
        return cursor.rowcount > 0

    # -- operator history -----------------------------------------------
    def record_operator_history(
        self,
        record: OperatorHistoryRecord | OperatorTrace | Mapping[str, Any] | None = None,
        *,
        mechanism_id: str | None = None,
        operator_id: str | None = None,
        run_id: str | None = None,
        trace_id: int | None = None,
        accepted: bool | None = None,
        immediate_reward: float | None = None,
        delayed_rewards: Mapping[int, float | None] | None = None,
        context: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if isinstance(record, OperatorHistoryRecord):
            item = record.model_copy(deep=True)
            if mechanism_id is not None:
                item.mechanism_id = mechanism_id
        elif isinstance(record, OperatorTrace):
            item = OperatorHistoryRecord(
                mechanism_id=mechanism_id,
                operator_id=record.operator_id,
                run_id=record.run_id,
                trace_id=record.trace_id,
                accepted=record.accepted,
                immediate_reward=record.immediate_reward,
                delayed_rewards=record.delayed_rewards,
                context={
                    "map_id": record.map_id,
                    "map_difficulty": record.map_difficulty,
                    **record.context,
                },
                metadata=record.metadata,
                created_at=record.timestamp,
            )
        else:
            payload = dict(record or {})
            payload.update(
                {
                    key: value
                    for key, value in {
                        "mechanism_id": mechanism_id,
                        "operator_id": operator_id,
                        "run_id": run_id,
                        "trace_id": trace_id,
                        "accepted": accepted,
                        "immediate_reward": immediate_reward,
                        "delayed_rewards": delayed_rewards,
                        "context": context,
                        "metadata": metadata,
                    }.items()
                    if value is not None
                }
            )
            item = OperatorHistoryRecord.model_validate(payload)
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO operator_history (
                    mechanism_id, operator_id, run_id, trace_id, accepted,
                    immediate_reward, delayed_rewards_json, context_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.mechanism_id,
                    item.operator_id,
                    item.run_id,
                    item.trace_id,
                    None if item.accepted is None else int(item.accepted),
                    item.immediate_reward,
                    _json(item.delayed_rewards),
                    _json(item.context),
                    _json(item.metadata),
                    item.created_at.isoformat(),
                ),
            )
        item.history_id = int(cursor.lastrowid)
        return item.history_id

    add_operator_history = record_operator_history
    record_trace = record_operator_history

    @staticmethod
    def _history(row: sqlite3.Row) -> OperatorHistoryRecord:
        return OperatorHistoryRecord(
            history_id=row["history_id"],
            mechanism_id=row["mechanism_id"],
            operator_id=row["operator_id"],
            run_id=row["run_id"],
            trace_id=row["trace_id"],
            accepted=None if row["accepted"] is None else bool(row["accepted"]),
            immediate_reward=row["immediate_reward"],
            delayed_rewards=_load(row["delayed_rewards_json"]),
            context=_load(row["context_json"]),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def get_operator_history(
        self,
        operator_id: str | None = None,
        *,
        mechanism_id: str | None = None,
        run_id: str | None = None,
        limit: int | None = 100,
    ) -> list[OperatorHistoryRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("operator_id", operator_id),
            ("mechanism_id", mechanism_id),
            ("run_id", run_id),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        query = "SELECT * FROM operator_history"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY history_id DESC"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit cannot be negative")
            query += " LIMIT ?"
            params.append(limit)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._history(row) for row in rows]

    # -- profile snapshots --------------------------------------------
    def add_operator_profile(
        self,
        profile: OperatorProfileRecord | Mapping[str, Any],
        *,
        operator_id: str | None = None,
        run_id: str | None = None,
        generation: int = 0,
    ) -> int:
        if isinstance(profile, OperatorProfileRecord):
            item = profile
        else:
            payload = dict(profile)
            resolved_operator = operator_id or payload.get("operator_id") or payload.get("operator_name")
            if not resolved_operator:
                raise ValueError("operator profile requires an operator id")
            item = OperatorProfileRecord(
                operator_id=str(resolved_operator),
                run_id=run_id,
                generation=generation,
                profile=payload,
            )
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO operator_profiles (
                    operator_id, run_id, generation, profile_json, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    item.operator_id,
                    item.run_id,
                    item.generation,
                    _json(item.profile),
                    item.created_at.isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    @staticmethod
    def _profile_record(row: sqlite3.Row) -> OperatorProfileRecord:
        return OperatorProfileRecord(
            profile_id=row["profile_id"],
            operator_id=row["operator_id"],
            run_id=row["run_id"],
            generation=row["generation"],
            profile=_load(row["profile_json"]),
            created_at=row["created_at"],
        )

    def get_operator_profiles(
        self,
        operator_id: str | None = None,
        *,
        run_id: str | None = None,
        limit: int | None = 100,
    ) -> list[OperatorProfileRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if operator_id is not None:
            clauses.append("operator_id = ?")
            params.append(operator_id)
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        query = "SELECT * FROM operator_profiles"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY generation DESC, profile_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._profile_record(row) for row in rows]

    def get_best_mechanisms(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        limit: int = 5,
        status: str | None = "active",
        min_evidence: int = 0,
    ) -> list[MechanismRecord]:
        """Rank mechanisms by stored score, evidence, and context overlap."""

        if limit < 0:
            raise ValueError("limit cannot be negative")
        records = [
            record
            for record in self.list_mechanisms(status)
            if record.evidence_count >= min_evidence
        ]
        query_context = dict(context or {})

        def rank(record: MechanismRecord) -> tuple[float, float, int, str]:
            declared = record.metadata.get("context", {})
            if not isinstance(declared, Mapping):
                declared = {}
            similarity = _context_similarity(query_context, declared)
            return (similarity, record.score, record.evidence_count, record.mechanism_id)

        records.sort(key=rank, reverse=True)
        return records[:limit]

    # -- failure modes --------------------------------------------------
    def add_failure_mode(
        self,
        mode: str | FailureModeRecord | Mapping[str, Any],
        *,
        mechanism_id: str | None = None,
        operator_id: str | None = None,
        count: int = 1,
        severity: float = 1.0,
        context: Mapping[str, Any] | None = None,
        evidence: Sequence[Any] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if isinstance(mode, FailureModeRecord):
            item = mode
        elif isinstance(mode, Mapping):
            item = FailureModeRecord.model_validate(mode)
        else:
            item = FailureModeRecord(
                mechanism_id=mechanism_id,
                operator_id=operator_id,
                mode=mode,
                count=count,
                severity=severity,
                context=dict(context or {}),
                evidence=list(evidence),
                metadata=dict(metadata or {}),
            )
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO failure_modes (
                    mechanism_id, operator_id, mode, count, severity, context_json,
                    evidence_json, metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.mechanism_id,
                    item.operator_id,
                    item.mode,
                    item.count,
                    item.severity,
                    _json(item.context),
                    _json(item.evidence),
                    _json(item.metadata),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    record_failure_mode = add_failure_mode
    store_failure_mode = add_failure_mode

    @staticmethod
    def _failure(row: sqlite3.Row) -> FailureModeRecord:
        return FailureModeRecord(
            failure_id=row["failure_id"],
            mechanism_id=row["mechanism_id"],
            operator_id=row["operator_id"],
            mode=row["mode"],
            count=row["count"],
            severity=row["severity"],
            context=_load(row["context_json"]),
            evidence=_load(row["evidence_json"]),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_failure_modes(
        self,
        operator_id: str | None = None,
        *,
        mechanism_id: str | None = None,
        limit: int | None = 100,
    ) -> list[FailureModeRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if operator_id is not None:
            clauses.append("operator_id = ?")
            params.append(operator_id)
        if mechanism_id is not None:
            clauses.append("mechanism_id = ?")
            params.append(mechanism_id)
        query = "SELECT * FROM failure_modes"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY count DESC, severity DESC, failure_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._failure(row) for row in rows]

    # -- structured insights ------------------------------------------
    def add_insight(
        self,
        insight: MechanismInsight | Mapping[str, Any] | None = None,
        *,
        operator_id: str | None = None,
        insight_type: str | None = None,
        evidence: Any = None,
        confidence: float = 0.0,
        applicable_context: Mapping[str, Any] | None = None,
        failure_context: Mapping[str, Any] | None = None,
        source_profile_id: str | int | None = None,
    ) -> int:
        if isinstance(insight, MechanismInsight):
            item = insight
        elif isinstance(insight, Mapping):
            item = MechanismInsight.model_validate(insight)
        else:
            if operator_id is None or insight_type is None:
                raise ValueError("operator_id and insight_type are required")
            item = MechanismInsight(
                operator_id=operator_id,
                insight_type=insight_type,  # type: ignore[arg-type]
                evidence={} if evidence is None else evidence,
                confidence=confidence,
                applicable_context=dict(applicable_context or {}),
                failure_context=dict(failure_context or {}),
                source_profile_id=source_profile_id,
            )
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO mechanism_insights (
                    operator_id, insight_type, evidence_json, confidence,
                    applicable_context_json, failure_context_json,
                    source_profile_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.operator_id,
                    item.insight_type,
                    _json(item.evidence),
                    item.confidence,
                    _json(item.applicable_context),
                    _json(item.failure_context),
                    None if item.source_profile_id is None else str(item.source_profile_id),
                    item.created_at.isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    @staticmethod
    def _insight(row: sqlite3.Row) -> MechanismInsight:
        return MechanismInsight(
            insight_id=row["insight_id"],
            operator_id=row["operator_id"],
            insight_type=row["insight_type"],
            evidence=_load(row["evidence_json"]),
            confidence=row["confidence"],
            applicable_context=_load(row["applicable_context_json"]),
            failure_context=_load(row["failure_context_json"]),
            source_profile_id=row["source_profile_id"],
            created_at=row["created_at"],
        )

    def get_insights(
        self,
        operator_id: str | None = None,
        *,
        insight_type: str | None = None,
        limit: int | None = 100,
    ) -> list[MechanismInsight]:
        clauses: list[str] = []
        params: list[Any] = []
        if operator_id is not None:
            clauses.append("operator_id = ?")
            params.append(operator_id)
        if insight_type is not None:
            clauses.append("insight_type = ?")
            params.append(insight_type)
        query = "SELECT * FROM mechanism_insights"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY confidence DESC, insight_id DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._insight(row) for row in rows]

    # -- synergies ------------------------------------------------------
    def add_synergy(
        self,
        first_operator: str | SynergyRecord | Mapping[str, Any],
        second_operator: str | None = None,
        score: float | None = None,
        *,
        sample_count: int = 1,
        context: Mapping[str, Any] | None = None,
        mechanism_ids: Sequence[str] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if isinstance(first_operator, SynergyRecord):
            item = first_operator
        elif isinstance(first_operator, Mapping):
            item = SynergyRecord.model_validate(first_operator)
        else:
            if second_operator is None or score is None:
                raise ValueError("second_operator and score are required")
            item = SynergyRecord(
                first_operator=first_operator,
                second_operator=second_operator,
                score=score,
                sample_count=sample_count,
                context=dict(context or {}),
                mechanism_ids=list(mechanism_ids),
                metadata=dict(metadata or {}),
            )
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO synergies (
                    first_operator, second_operator, score, sample_count,
                    context_json, mechanism_ids_json, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.first_operator,
                    item.second_operator,
                    item.score,
                    item.sample_count,
                    _json(item.context),
                    _json(item.mechanism_ids),
                    _json(item.metadata),
                    item.created_at.isoformat(),
                    item.updated_at.isoformat(),
                ),
            )
        return int(cursor.lastrowid)

    record_synergy = add_synergy
    store_synergy = add_synergy

    @staticmethod
    def _synergy(row: sqlite3.Row) -> SynergyRecord:
        return SynergyRecord(
            synergy_id=row["synergy_id"],
            first_operator=row["first_operator"],
            second_operator=row["second_operator"],
            score=row["score"],
            sample_count=row["sample_count"],
            context=_load(row["context_json"]),
            mechanism_ids=_load(row["mechanism_ids_json"]),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_synergies(
        self,
        operator_id: str | None = None,
        *,
        first_operator: str | None = None,
        second_operator: str | None = None,
        min_score: float | None = None,
        context: Mapping[str, Any] | None = None,
        limit: int = 100,
    ) -> list[SynergyRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        if operator_id is not None:
            clauses.append("(first_operator = ? OR second_operator = ?)")
            params.extend((operator_id, operator_id))
        if first_operator is not None:
            clauses.append("first_operator = ?")
            params.append(first_operator)
        if second_operator is not None:
            clauses.append("second_operator = ?")
            params.append(second_operator)
        if min_score is not None:
            clauses.append("score >= ?")
            params.append(min_score)
        query = "SELECT * FROM synergies"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY score DESC, sample_count DESC, synergy_id DESC"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        records = [self._synergy(row) for row in rows]
        if context:
            records.sort(
                key=lambda item: (
                    _context_similarity(context, item.context),
                    item.score,
                    item.sample_count,
                ),
                reverse=True,
            )
        return records[: max(0, limit)]

    # -- representative cases -----------------------------------------
    def add_case(
        self,
        case: CaseRecord | Mapping[str, Any] | None = None,
        *,
        case_id: str | None = None,
        mechanism_id: str | None = None,
        operator_id: str | None = None,
        outcome: str = "unknown",
        score: float | None = None,
        context: Mapping[str, Any] | None = None,
        state: Any = None,
        action: Any = None,
        result: Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if isinstance(case, CaseRecord):
            item = case
        elif isinstance(case, Mapping):
            payload = dict(case)
            payload.setdefault("case_id", case_id or payload.get("id") or uuid.uuid4().hex)
            item = CaseRecord.model_validate(payload)
        else:
            item = CaseRecord(
                case_id=case_id or uuid.uuid4().hex,
                mechanism_id=mechanism_id,
                operator_id=operator_id,
                outcome=outcome,
                score=score,
                context=dict(context or {}),
                state={} if state is None else state,
                action={} if action is None else action,
                result={} if result is None else result,
                metadata=dict(metadata or {}),
            )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO cases (
                    case_id, mechanism_id, operator_id, outcome, score,
                    context_json, state_json, action_json, result_json,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    mechanism_id=excluded.mechanism_id,
                    operator_id=excluded.operator_id,
                    outcome=excluded.outcome,
                    score=excluded.score,
                    context_json=excluded.context_json,
                    state_json=excluded.state_json,
                    action_json=excluded.action_json,
                    result_json=excluded.result_json,
                    metadata_json=excluded.metadata_json
                """,
                (
                    item.case_id,
                    item.mechanism_id,
                    item.operator_id,
                    item.outcome,
                    item.score,
                    _json(item.context),
                    _json(item.state),
                    _json(item.action),
                    _json(item.result),
                    _json(item.metadata),
                    item.created_at.isoformat(),
                ),
            )
        return item.case_id

    record_case = add_case
    store_case = add_case

    @staticmethod
    def _case(row: sqlite3.Row) -> CaseRecord:
        return CaseRecord(
            case_id=row["case_id"],
            mechanism_id=row["mechanism_id"],
            operator_id=row["operator_id"],
            outcome=row["outcome"],
            score=row["score"],
            context=_load(row["context_json"]),
            state=_load(row["state_json"]),
            action=_load(row["action_json"]),
            result=_load(row["result_json"]),
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
        )

    def get_case(self, case_id: str) -> CaseRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM cases WHERE case_id = ?", (case_id,)
            ).fetchone()
        return None if row is None else self._case(row)

    def get_relevant_cases(
        self,
        context: Mapping[str, Any] | None = None,
        *,
        operator_id: str | None = None,
        mechanism_id: str | None = None,
        outcome: str | None = None,
        limit: int = 5,
    ) -> list[CaseRecord]:
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("operator_id", operator_id),
            ("mechanism_id", mechanism_id),
            ("outcome", outcome),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                params.append(value)
        query = "SELECT * FROM cases"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        query_context = dict(context or {})
        records = [self._case(row) for row in rows]
        records.sort(
            key=lambda item: (
                _context_similarity(query_context, item.context),
                item.score if item.score is not None else float("-inf"),
                item.created_at,
            ),
            reverse=True,
        )
        return records[: max(0, limit)]

    # -- lineage --------------------------------------------------------
    def _insert_lineage(
        self,
        parent_id: str,
        child_id: str,
        relation: str,
        metadata: Mapping[str, Any],
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO lineage (parent_id, child_id, relation, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(parent_id, child_id, relation) DO UPDATE SET
                metadata_json=excluded.metadata_json
            """,
            (parent_id, child_id, relation, _json(metadata), _now().isoformat()),
        )
        if cursor.lastrowid:
            return int(cursor.lastrowid)
        row = self._connection.execute(
            """SELECT lineage_id FROM lineage
               WHERE parent_id = ? AND child_id = ? AND relation = ?""",
            (parent_id, child_id, relation),
        ).fetchone()
        return int(row["lineage_id"])

    def add_lineage(
        self,
        parent_id: str,
        child_id: str,
        relation: str = "derived_from",
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        if parent_id == child_id:
            raise ValueError("a mechanism cannot be its own parent")
        with self._lock, self._connection:
            return self._insert_lineage(parent_id, child_id, relation, metadata or {})

    link_lineage = add_lineage
    record_lineage = add_lineage

    @staticmethod
    def _lineage(row: sqlite3.Row, depth: int = 1) -> LineageRecord:
        return LineageRecord(
            lineage_id=row["lineage_id"],
            parent_id=row["parent_id"],
            child_id=row["child_id"],
            relation=row["relation"],
            metadata=_load(row["metadata_json"]),
            created_at=row["created_at"],
            depth=depth,
        )

    def get_lineage(
        self,
        mechanism_id: str,
        *,
        direction: Literal["ancestors", "descendants", "both"] = "both",
        recursive: bool = True,
        max_depth: int = 32,
    ) -> list[LineageRecord]:
        """Return lineage edges nearest-first without duplicating cycles."""

        if direction not in {"ancestors", "descendants", "both"}:
            raise ValueError("direction must be ancestors, descendants, or both")
        if max_depth < 1:
            return []
        frontier: list[tuple[str, int]] = [(mechanism_id, 0)]
        seen_nodes = {mechanism_id}
        seen_edges: set[int] = set()
        output: list[LineageRecord] = []
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= max_depth:
                continue
            clauses: list[str] = []
            params: list[str] = []
            if direction in {"ancestors", "both"}:
                clauses.append("child_id = ?")
                params.append(current)
            if direction in {"descendants", "both"}:
                clauses.append("parent_id = ?")
                params.append(current)
            with self._lock:
                rows = self._connection.execute(
                    "SELECT * FROM lineage WHERE " + " OR ".join(clauses), params
                ).fetchall()
            for row in rows:
                edge_id = int(row["lineage_id"])
                if edge_id in seen_edges:
                    continue
                seen_edges.add(edge_id)
                output.append(self._lineage(row, depth + 1))
                if not recursive:
                    continue
                neighbours: list[str] = []
                if direction in {"ancestors", "both"} and row["child_id"] == current:
                    neighbours.append(row["parent_id"])
                if direction in {"descendants", "both"} and row["parent_id"] == current:
                    neighbours.append(row["child_id"])
                for neighbour in neighbours:
                    if neighbour not in seen_nodes:
                        seen_nodes.add(neighbour)
                        frontier.append((neighbour, depth + 1))
        return sorted(output, key=lambda edge: (edge.depth, edge.lineage_id or 0))

    def get_ancestors(self, mechanism_id: str, **kwargs: Any) -> list[LineageRecord]:
        return self.get_lineage(mechanism_id, direction="ancestors", **kwargs)

    def get_descendants(self, mechanism_id: str, **kwargs: Any) -> list[LineageRecord]:
        return self.get_lineage(mechanism_id, direction="descendants", **kwargs)

    # -- lifecycle ------------------------------------------------------
    def close(self) -> None:
        if not self._closed:
            with self._lock:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "MechanismMemory":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
