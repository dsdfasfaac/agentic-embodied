# Copyright (c) 2026 Zetta Contributors
"""Durable request identity, decisions, step intents, and immutable outcomes."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .contracts import GatewayError, canonical, digest


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_depth = 0
        self.db = sqlite3.connect(path, timeout=10, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
          CREATE TABLE IF NOT EXISTS operations (
            request_id TEXT PRIMARY KEY, digest TEXT NOT NULL, request TEXT NOT NULL,
            result TEXT NOT NULL, final INTEGER NOT NULL DEFAULT 0);
          CREATE TABLE IF NOT EXISTS decisions (
            ref TEXT PRIMARY KEY, digest TEXT NOT NULL, source TEXT NOT NULL,
            evidence TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS records (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL,
            public INTEGER NOT NULL, payload TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS snapshot (id INTEGER PRIMARY KEY CHECK(id=1), payload TEXT NOT NULL);
        """)

    @contextmanager
    def batch(self):
        """Durably commit one post-action observation/critic barrier together.

        Action intent and physical step commit stay outside this batch. The
        owner must exit successfully before dispatching another motor command.
        FULL synchronous WAL durability remains enabled.
        """
        if self._batch_depth:
            yield
            return
        self._batch_depth += 1
        try:
            with self.db:
                yield
        finally:
            self._batch_depth -= 1

    @contextmanager
    def _transaction(self):
        if self._batch_depth:
            yield
        else:
            with self.db:
                yield

    def record(self, kind, payload, *, public=False):
        with self._transaction():
            cursor = self.db.execute(
                "INSERT INTO records(kind,public,payload) VALUES(?,?,?)",
                (kind, int(public), canonical(payload)),
            )
        return cursor.lastrowid

    def register_decision(self, request, *, source, evidence):
        if source not in {"agent", "runner"}:
            raise ValueError("invalid decision source")
        value = digest(request.model_dump())
        with self._transaction():
            row = self.db.execute(
                "SELECT digest FROM decisions WHERE ref=?", (request.decision_ref,)
            ).fetchone()
            if row and row[0] != value:
                raise GatewayError("DECISION_CONFLICT")
            self.db.execute(
                "INSERT OR IGNORE INTO decisions VALUES(?,?,?,?)",
                (request.decision_ref, value, source, canonical(evidence)),
            )

    def check_decision(self, request):
        row = self.db.execute(
            "SELECT digest FROM decisions WHERE ref=?", (request.decision_ref,)
        ).fetchone()
        if not row or row[0] != digest(request.model_dump()):
            raise GatewayError("UNREGISTERED_DECISION")

    def duplicate(self, request):
        row = self.db.execute(
            "SELECT digest,result FROM operations WHERE request_id=?",
            (request.request_id,),
        ).fetchone()
        if row:
            if row[0] != digest(request.model_dump()):
                raise GatewayError("REQUEST_ID_CONFLICT")
            return json.loads(row[1])
        return None

    def admit(self, request, result):
        with self._transaction():
            self.db.execute(
                "INSERT INTO operations VALUES(?,?,?,?,0)",
                (
                    request.request_id,
                    digest(request.model_dump()),
                    canonical(request.model_dump()),
                    canonical(result),
                ),
            )

    def update(self, result, *, final=False):
        with self._transaction():
            self.db.execute(
                "UPDATE operations SET result=?,final=? WHERE request_id=? AND final=0",
                (canonical(result), int(final), result["request_id"]),
            )

    def status(self, request_id):
        row = self.db.execute(
            "SELECT result FROM operations WHERE request_id=?", (request_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def save_snapshot(self, snapshot):
        with self._transaction():
            self.db.execute(
                "INSERT OR REPLACE INTO snapshot VALUES(1,?)", (canonical(snapshot),)
            )

    def snapshot(self):
        row = self.db.execute("SELECT payload FROM snapshot WHERE id=1").fetchone()
        return json.loads(row[0]) if row else None

    def events(self, after):
        return [
            {"sequence": row[0], "kind": row[1], "payload": json.loads(row[2])}
            for row in self.db.execute(
                "SELECT sequence,kind,payload FROM records WHERE public=1 AND sequence>? ORDER BY sequence LIMIT 100",
                (after,),
            )
        ]

    def mark_lost(self):
        """Never resume an old physical episode, even if its last commit is known."""
        for request_id, payload in self.db.execute(
            "SELECT request_id,result FROM operations WHERE final=0"
        ).fetchall():
            result = json.loads(payload)
            result.update(
                status="unknown",
                write_certainty="unknown",
                error={
                    "code": "WORKER_LOST",
                    "phase": "execution",
                    "retry_class": "never",
                    "public_message": "Episode worker lost; start a fresh attempt",
                },
            )
            self.update(result, final=True)
        snapshot = self.snapshot()
        if snapshot:
            snapshot.update(
                state="EXECUTION_UNCERTAIN",
                recovery_context=None,
                environment_ended=True,
            )
            self.save_snapshot(snapshot)
        self.record("worker_lost", {"artifact_completeness": "incomplete"}, public=True)

    def close(self):
        self.db.close()
