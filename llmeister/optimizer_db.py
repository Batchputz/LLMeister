"""LLMeister optimizer DB — schema + queries for parameter optimization runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import db as main_db

OPTIMIZER_SCHEMA = """
CREATE TABLE IF NOT EXISTS optimization_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,
    hf_model_id     TEXT NOT NULL,
    use_case        TEXT,
    workload        TEXT,                   -- JSON: {type, concurrency, input_tokens, output_tokens, priority}
    status          TEXT DEFAULT 'interview',  -- interview|researching|running|completed|failed|stopped
    research_json   TEXT,
    baseline_step   INTEGER,
    best_step       INTEGER,
    total_steps     INTEGER DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS optimization_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    step_number     INTEGER NOT NULL,
    parameter       TEXT,
    old_value       TEXT,
    new_value       TEXT,
    reasoning       TEXT,
    config_json     TEXT NOT NULL,
    benchmark_json  TEXT,
    metrics_json    TEXT,
    score           REAL,
    is_improvement  INTEGER DEFAULT 0,
    kept            INTEGER DEFAULT 0,
    restart_time_s  REAL,
    benchmark_time_s REAL,
    timestamp       TEXT,
    FOREIGN KEY (run_id) REFERENCES optimization_runs(id)
);

CREATE INDEX IF NOT EXISTS idx_opt_steps_run ON optimization_steps(run_id);
"""


def init_optimizer_db(conn) -> None:
    conn.executescript(OPTIMIZER_SCHEMA)
    conn.commit()


def create_run(conn, model_name: str, hf_model_id: str) -> int:
    now_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    cur = conn.execute(
        "INSERT INTO optimization_runs (model_name, hf_model_id, status, started_at) "
        "VALUES (?, ?, 'interview', ?)",
        (model_name, hf_model_id, now_str),
    )
    conn.commit()
    return cur.lastrowid


def get_run(conn, run_id: int) -> dict[str, Any] | None:
    conn.row_factory = __import__("sqlite3").Row
    row = conn.execute("SELECT * FROM optimization_runs WHERE id=?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_active_run(conn, model_name: str) -> dict[str, Any] | None:
    conn.row_factory = __import__("sqlite3").Row
    row = conn.execute(
        "SELECT * FROM optimization_runs WHERE model_name=? AND status IN ('interview','researching','running') "
        "ORDER BY id DESC LIMIT 1",
        (model_name,),
    ).fetchone()
    return dict(row) if row else None


def update_run(conn, run_id: int, **fields) -> None:
    sets = ", ".join(f"{k}=?" for k in fields)
    conn.execute(f"UPDATE optimization_runs SET {sets} WHERE id=?", (*fields.values(), run_id))
    conn.commit()


def add_step(conn, run_id: int, step: dict[str, Any]) -> int:
    now_str = time.strftime("%Y-%m-%dT%H:%M:%S")
    cur = conn.execute(
        "INSERT INTO optimization_steps "
        "(run_id, step_number, parameter, old_value, new_value, reasoning, "
        "config_json, benchmark_json, metrics_json, score, is_improvement, kept, "
        "restart_time_s, benchmark_time_s, timestamp) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            run_id, step["step_number"], step.get("parameter"), step.get("old_value"),
            step.get("new_value"), step.get("reasoning"),
            json.dumps(step.get("config", {})), json.dumps(step.get("benchmark")),
            json.dumps(step.get("metrics")), step.get("score"),
            step.get("is_improvement", 0), step.get("kept", 0),
            step.get("restart_time_s"), step.get("benchmark_time_s"), now_str,
        ),
    )
    conn.commit()
    # Update run totals
    run = get_run(conn, run_id)
    if run:
        total = (run.get("total_steps") or 0) + 1
        best_step = run.get("best_step")
        if step.get("is_improvement"):
            best_step = cur.lastrowid
        update_run(conn, run_id, total_steps=total, best_step=best_step)
    return cur.lastrowid


def get_steps(conn, run_id: int) -> list[dict[str, Any]]:
    conn.row_factory = __import__("sqlite3").Row
    rows = conn.execute(
        "SELECT * FROM optimization_steps WHERE run_id=? ORDER BY step_number", (run_id,)
    ).fetchall()
    steps = []
    for row in rows:
        d = dict(row)
        d["config"] = json.loads(d.pop("config_json") or "{}")
        d["benchmark"] = json.loads(d.pop("benchmark_json") or "null")
        d["metrics"] = json.loads(d.pop("metrics_json") or "null")
        steps.append(d)
    return steps


def get_step(conn, step_id: int) -> dict[str, Any] | None:
    conn.row_factory = __import__("sqlite3").Row
    row = conn.execute("SELECT * FROM optimization_steps WHERE id=?", (step_id,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["config"] = json.loads(d.pop("config_json") or "{}")
    d["benchmark"] = json.loads(d.pop("benchmark_json") or "null")
    d["metrics"] = json.loads(d.pop("metrics_json") or "null")
    return d
