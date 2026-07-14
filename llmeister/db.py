"""SQLite model registry + migration from existing recipe YAMLs / llama-swap config.

Single source of truth for per-model launch config, routing metadata, lifecycle
state, and measured/estimated memory. System-level config (ports, keys) lives in
config.yaml; this DB holds *model* data.
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

import yaml
from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,              -- canonical short name, e.g. "qwen9b"
    hf_model_id TEXT NOT NULL,              -- actual model path vLLM loads
    config TEXT NOT NULL,                   -- JSON: {served_model_names, aliases, use_model_name, defaults, env, mods, command, container, recipe_name}
    state TEXT NOT NULL DEFAULT 'STOPPED',  -- STOPPED|STARTING|AWAKE|SLEEPING|WAKING|ERROR|PENDING
    container_name TEXT,
    port INTEGER,
    measured_max_memory_mb INTEGER,
    measured_weight_memory_mb INTEGER,  -- memory retained when SLEEPING (weights)
    estimated_memory_mb INTEGER,
    sleep_level INTEGER NOT NULL DEFAULT 1,
    last_active_at REAL,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    model_id INTEGER,
    event TEXT NOT NULL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_models_state ON models(state);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts);
"""

_UNSET = object()

from llmeister import PROJECT_ROOT
DEFAULT_DB_PATH = PROJECT_ROOT / "llmeister.db"
RECIPE_DIR = Path(config.load().get("paths", {}).get("recipe_dir", "/home/batchputz/spark-vllm-docker/recipes"))
LLAMA_SWAP_CONFIG = Path(config.load().get("paths", {}).get("llama_swap_config", "/home/batchputz/llama-swap/config.yaml"))

# Recipes to seed, mapped to a canonical name. Port comes from each recipe's defaults.
SEED_RECIPES = {
    "qwen9b": "qwen3.5-9b-nvfp4.yaml",
    "qwen35b": "qwen3.6-35b-a3b-prism-nvfp4.yaml",
    "gemma4-26b": "gemma-4-26b-a4b-it-nvfp4.yaml",
}


def connect(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    # lightweight migration for columns added after the initial schema
    cols = {r[1] for r in conn.execute("PRAGMA table_info(models)").fetchall()}
    if "measured_weight_memory_mb" not in cols:
        conn.execute("ALTER TABLE models ADD COLUMN measured_weight_memory_mb INTEGER")
    conn.commit()


def now() -> float:
    return time.time()


def audit(conn: sqlite3.Connection, model_id: int | None, event: str, detail: str = "") -> None:
    conn.execute(
        "INSERT INTO audit_log(ts, model_id, event, detail) VALUES (?, ?, ?, ?)",
        (now(), model_id, event, detail),
    )
    conn.commit()


def _parse_served_names(command: str) -> list[str]:
    """Extract --served-model-name values from a vllm serve command block."""
    names: list[str] = []
    lines = command.splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--served-model-name"):
            # values are the rest of the line after the flag, space-separated
            rest = stripped[len("--served-model-name"):].strip().rstrip("\\").strip()
            names.extend(n for n in rest.split() if n)
    return names


def _load_llama_swap_routing() -> dict[int, dict[str, Any]]:
    """Return {port: {aliases, use_model_name, gateway_name}} from llama-swap config."""
    routing: dict[int, dict[str, Any]] = {}
    if not LLAMA_SWAP_CONFIG.exists():
        return routing
    cfg = yaml.safe_load(LLAMA_SWAP_CONFIG.read_text()) or {}
    for gw_name, entry in (cfg.get("models") or {}).items():
        proxy = entry.get("proxy", "")
        port = None
        if ":" in proxy:
            try:
                port = int(proxy.rsplit(":", 1)[1])
            except ValueError:
                pass
        if port is None:
            continue
        routing[port] = {
            "gateway_name": gw_name,
            "aliases": entry.get("aliases") or [],
            "use_model_name": entry.get("useModelName"),
        }
    return routing


def seed_from_recipes(conn: sqlite3.Connection) -> int:
    """Migrate existing recipe YAMLs (+ llama-swap routing) into the models table.

    Idempotent: existing names are skipped. Returns count of newly seeded rows.
    """
    routing = _load_llama_swap_routing()
    seeded = 0
    for name, recipe_file in SEED_RECIPES.items():
        path = RECIPE_DIR / recipe_file
        if not path.exists():
            continue
        recipe = yaml.safe_load(path.read_text()) or {}
        defaults = recipe.get("defaults") or {}
        port = int(defaults.get("port", 0))
        hf_model_id = recipe.get("model", "")
        command = recipe.get("command", "")
        served_names = _parse_served_names(command)
        rt = routing.get(port, {})
        aliases = rt.get("aliases") or []
        use_model_name = rt.get("use_model_name")
        # If the recipe declares served names, the first is the canonical gateway id.
        gateway_name = rt.get("gateway_name") or (served_names[0] if served_names else hf_model_id)
        config = {
            "recipe_name": recipe.get("name"),
            "hf_model_id": hf_model_id,
            "container": recipe.get("container", "vllm-node"),
            "defaults": defaults,
            "env": recipe.get("env") or {},
            "mods": recipe.get("mods") or [],
            "command": command,
            "served_model_names": served_names,
            "aliases": aliases,
            "use_model_name": use_model_name,
            "gateway_name": gateway_name,
            "recipe_file": recipe_file,
        }
        try:
            conn.execute(
                """INSERT INTO models
                   (name, hf_model_id, config, state, port, sleep_level, created_at, updated_at)
                   VALUES (?, ?, ?, 'STOPPED', ?, 1, ?, ?)""",
                (name, hf_model_id, json.dumps(config), port, now(), now()),
            )
            seeded += 1
            audit(conn, None, "seed", f"imported recipe {recipe_file} as {name}")
        except sqlite3.IntegrityError:
            pass  # already exists
    conn.commit()
    return seeded


def list_models(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM models ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def get_model(conn: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM models WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def add_model(conn: sqlite3.Connection, name: str, hf_model_id: str, config: dict[str, Any],
              state: str = "PENDING", estimated_memory_mb: int | None = None) -> int:
    """Insert a new model row (e.g. a PENDING research candidate). Returns its id."""
    import time as _t
    cur = conn.execute(
        """INSERT INTO models (name, hf_model_id, config, state, estimated_memory_mb,
                              sleep_level, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
        (name, hf_model_id, json.dumps(config), state,
         estimated_memory_mb, _t.time(), _t.time()),
    )
    conn.commit()
    return cur.lastrowid


def set_state(
    conn: sqlite3.Connection,
    name: str,
    state: str,
    *,
    container_name: str | None = None,
    port: int | None = None,
    error: str | None = None,
    measured_max_memory_mb: int | None = None,
    measured_weight_memory_mb: int | None = None,
) -> None:
    fields = ["state = ?", "updated_at = ?"]
    vals: list[Any] = [state, now()]
    if container_name is not None:
        fields.append("container_name = ?")
        vals.append(container_name)
    if port is not None:
        fields.append("port = ?")
        vals.append(port)
    if error is not _UNSET:
        fields.append("error = ?")
        vals.append(error)  # None here clears (NULL)
    if measured_max_memory_mb is not None:
        fields.append("measured_max_memory_mb = ?")
        vals.append(measured_max_memory_mb)
    if measured_weight_memory_mb is not None:
        fields.append("measured_weight_memory_mb = ?")
        vals.append(measured_weight_memory_mb)
    vals.append(name)
    conn.execute(f"UPDATE models SET {', '.join(fields)} WHERE name = ?", vals)
    conn.commit()


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="LLMeister DB init/seed")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH))
    p.add_argument("--seed", action="store_true", help="migrate recipes into the DB")
    args = p.parse_args()
    conn = connect(Path(args.db))
    init_db(conn)
    print(f"DB initialized at {args.db}")
    if args.seed:
        n = seed_from_recipes(conn)
        print(f"Seeded {n} new model(s).")
        for m in list_models(conn):
            print(f"  - {m['name']:14s} state={m['state']:8s} port={m['port']} model={m['hf_model_id']}")


if __name__ == "__main__":
    main()
