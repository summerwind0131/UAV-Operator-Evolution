"""SQLite-backed operator trajectory recorder with optional JSONL mirroring."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from .models import OperatorTrace
from .rewards import compute_delayed_rewards


_SCHEMA = """
CREATE TABLE IF NOT EXISTS operator_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    episode_id TEXT,
    map_id TEXT NOT NULL,
    map_difficulty TEXT,
    iteration INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    seed INTEGER,
    operator_id TEXT NOT NULL,
    operator_family TEXT,
    operator_version TEXT,
    operator_params_json TEXT NOT NULL,
    context_json TEXT NOT NULL,
    before_state_json TEXT NOT NULL,
    candidate_state_json TEXT NOT NULL,
    accepted_state_json TEXT NOT NULL,
    before_objective REAL,
    candidate_objective REAL,
    accepted_objective REAL,
    before_components_json TEXT NOT NULL,
    candidate_components_json TEXT NOT NULL,
    accepted_components_json TEXT NOT NULL,
    before_feasible INTEGER,
    candidate_feasible INTEGER,
    accepted_feasible INTEGER,
    accepted INTEGER NOT NULL,
    acceptance_reason TEXT,
    acceptance_probability REAL,
    temperature REAL,
    immediate_reward REAL,
    delayed_rewards_json TEXT NOT NULL,
    runtime_ms REAL NOT NULL,
    error TEXT,
    metadata_json TEXT NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_operator_traces_run_iteration
    ON operator_traces(run_id, episode_id, map_id, iteration, id);
CREATE INDEX IF NOT EXISTS idx_operator_traces_operator
    ON operator_traces(operator_id);
CREATE INDEX IF NOT EXISTS idx_operator_traces_map
    ON operator_traces(map_id);
"""


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _bool_sql(value: bool | None) -> int | None:
    return None if value is None else int(value)


class TrajectoryRecorder:
    """Persist :class:`OperatorTrace` records transactionally.

    Parameters
    ----------
    db_path:
        SQLite filename, or ``":memory:"`` for an ephemeral recorder.
    jsonl_path:
        Optional append-only human-readable mirror.  Calling a delayed-reward
        update rewrites this mirror from SQLite so it remains consistent.
    """

    def __init__(
        self,
        db_path: str | Path,
        jsonl_path: str | Path | None = None,
    ) -> None:
        self.db_path = Path(db_path) if str(db_path) != ":memory:" else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path is not None else None
        if self.db_path is not None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
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
        """Expose the connection for read-only research queries."""

        self._ensure_open()
        return self._connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("trajectory recorder is closed")

    @staticmethod
    def _values(trace: OperatorTrace) -> tuple[Any, ...]:
        payload = trace.model_dump(mode="json", exclude={"trace_id"})
        return (
            trace.run_id,
            trace.episode_id,
            trace.map_id,
            trace.map_difficulty,
            trace.iteration,
            trace.timestamp.isoformat(),
            trace.seed,
            trace.operator_id,
            trace.operator_family,
            trace.operator_version,
            _json(trace.operator_params),
            _json(trace.context),
            _json(trace.before_state),
            _json(trace.candidate_state),
            _json(trace.accepted_state),
            trace.before_objective,
            trace.candidate_objective,
            trace.accepted_objective,
            _json(trace.before_components),
            _json(trace.candidate_components),
            _json(trace.accepted_components),
            _bool_sql(trace.before_feasible),
            _bool_sql(trace.candidate_feasible),
            _bool_sql(trace.accepted_feasible),
            int(trace.accepted),
            trace.acceptance_reason,
            trace.acceptance_probability,
            trace.temperature,
            trace.immediate_reward,
            _json(trace.delayed_rewards),
            trace.runtime_ms,
            trace.error,
            _json(trace.metadata),
            _json(payload),
        )

    def record(self, trace: OperatorTrace | Mapping[str, Any]) -> int:
        """Record one trace and return its SQLite identifier."""

        self._ensure_open()
        item = (
            trace
            if isinstance(trace, OperatorTrace)
            else OperatorTrace.model_validate(trace)
        )
        sql = """
            INSERT INTO operator_traces (
                run_id, episode_id, map_id, map_difficulty, iteration, timestamp,
                seed, operator_id, operator_family, operator_version,
                operator_params_json, context_json, before_state_json,
                candidate_state_json, accepted_state_json, before_objective,
                candidate_objective, accepted_objective, before_components_json,
                candidate_components_json, accepted_components_json,
                before_feasible, candidate_feasible, accepted_feasible, accepted,
                acceptance_reason, acceptance_probability, temperature,
                immediate_reward, delayed_rewards_json, runtime_ms, error,
                metadata_json, payload_json
            ) VALUES ({})
        """.format(",".join("?" for _ in range(34)))
        with self._lock, self._connection:
            cursor = self._connection.execute(sql, self._values(item))
            trace_id = int(cursor.lastrowid)
            if self.jsonl_path is not None:
                self._append_jsonl(item.model_copy(update={"trace_id": trace_id}))
        item.trace_id = trace_id
        return trace_id

    def record_many(
        self, traces: Iterable[OperatorTrace | Mapping[str, Any]]
    ) -> list[int]:
        """Record several traces, returning identifiers in input order."""

        return [self.record(trace) for trace in traces]

    def _append_jsonl(self, trace: OperatorTrace) -> None:
        assert self.jsonl_path is not None
        with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_json(trace.model_dump(mode="json")))
            handle.write("\n")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> OperatorTrace:
        payload = json.loads(row["payload_json"])
        payload["trace_id"] = int(row["id"])
        # These two fields can be updated after the immutable payload is stored.
        payload["delayed_rewards"] = json.loads(row["delayed_rewards_json"])
        payload["immediate_reward"] = row["immediate_reward"]
        return OperatorTrace.model_validate(payload)

    def get_trace(self, trace_id: int) -> OperatorTrace | None:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM operator_traces WHERE id = ?", (int(trace_id),)
            ).fetchone()
        return None if row is None else self._from_row(row)

    def list_traces(self, run_id: str | None = None) -> list[OperatorTrace]:
        """Return traces in deterministic trajectory/iteration order."""

        self._ensure_open()
        if run_id is None:
            query = """
                SELECT * FROM operator_traces
                ORDER BY run_id, COALESCE(episode_id, ''), map_id, iteration, id
            """
            params: Sequence[Any] = ()
        else:
            query = """
                SELECT * FROM operator_traces WHERE run_id = ?
                ORDER BY COALESCE(episode_id, ''), map_id, iteration, id
            """
            params = (run_id,)
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return [self._from_row(row) for row in rows]

    def iter_traces(self, run_id: str | None = None) -> Iterator[OperatorTrace]:
        yield from self.list_traces(run_id)

    def update_delayed_rewards(
        self,
        horizons: Iterable[int] = (5, 10, 20),
        *,
        run_id: str | None = None,
        baseline: str = "before",
    ) -> list[OperatorTrace]:
        """Compute delayed rewards, persist them, and return updated traces."""

        traces = compute_delayed_rewards(
            self.list_traces(run_id), horizons, baseline=baseline  # type: ignore[arg-type]
        )
        with self._lock, self._connection:
            self._connection.executemany(
                "UPDATE operator_traces SET delayed_rewards_json = ? WHERE id = ?",
                [(_json(trace.delayed_rewards), trace.trace_id) for trace in traces],
            )
        if self.jsonl_path is not None:
            self.export_jsonl(self.jsonl_path)
        return traces

    compute_delayed_rewards = update_delayed_rewards

    def export_jsonl(
        self,
        path: str | Path | None = None,
        *,
        run_id: str | None = None,
    ) -> Path:
        """Write a complete, deterministic JSONL export and return its path."""

        target = Path(path) if path is not None else self.jsonl_path
        if target is None:
            raise ValueError("a JSONL path must be supplied")
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, target.open("w", encoding="utf-8", newline="\n") as handle:
            for trace in self.list_traces(run_id):
                handle.write(_json(trace.model_dump(mode="json")))
                handle.write("\n")
        return target

    def __len__(self) -> int:
        self._ensure_open()
        with self._lock:
            row = self._connection.execute(
                "SELECT COUNT(*) AS count FROM operator_traces"
            ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        if not self._closed:
            with self._lock:
                self._connection.close()
                self._closed = True

    def __enter__(self) -> "TrajectoryRecorder":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
