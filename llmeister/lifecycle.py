"""Lifecycle state machine for vLLM instances.

Wraps the vLLM HTTP endpoints (/health, /is_sleeping, /sleep, /wake_up) and drives
state transitions, backed by the SQLite registry. A background poll loop reconciles
DB state with reality and detects crashed containers.

Route guard (enforced by the /v1 proxy in manager.py): only AWAKE models may serve.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx

from . import db
from . import launcher
from . import planner as planner_mod

log = logging.getLogger("lifecycle")

# States
STOPPED = "STOPPED"
STARTING = "STARTING"
AWAKE = "AWAKE"
SLEEPING = "SLEEPING"
WAKING = "WAKING"
ERROR = "ERROR"
PENDING = "PENDING"
DISCOVERED = "DISCOVERED"  # in HF cache, not yet researched

CAN_SERVE = {AWAKE}

POLL_INTERVAL = 3.0          # seconds between reconcile polls
HEALTH_TIMEOUT = 3.0
START_POLL_INTERVAL = 5.0    # during cold start
START_MAX_WAIT = 600         # 10 min cap for cold start


class VLLMClient:
    """Thin async wrapper over a vLLM backend's HTTP API."""

    def __init__(self, port: int, host: str = "127.0.0.1"):
        self.base = f"http://{host}:{port}"

    async def health(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as c:
                r = await c.get(f"{self.base}/health")
                return r.status_code == 200
        except Exception:
            return False

    async def is_sleeping(self) -> bool | None:
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as c:
                r = await c.get(f"{self.base}/is_sleeping")
                if r.status_code == 200:
                    return bool(r.json().get("is_sleeping"))
        except Exception:
            pass
        return None

    async def sleep(self, level: int = 1) -> bool:
        try:
            async with httpx.AsyncClient(timeout=30) as c:
                r = await c.post(f"{self.base}/sleep", params={"level": level})
                return r.status_code == 200
        except Exception as e:
            log.error("sleep failed on %s: %s", self.base, e)
            return False

    async def wake(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(f"{self.base}/wake_up")
                return r.status_code == 200
        except Exception as e:
            log.error("wake failed on %s: %s", self.base, e)
            return False


def _container_name(name: str) -> str:
    return f"vllm_mgr_{name}"


class LifecycleManager:
    def __init__(self, db_path: Any = None):
        self.db_path = db_path or db.DEFAULT_DB_PATH
        # per-model async locks so only one lifecycle op runs at a time
        self._locks: dict[str, asyncio.Lock] = {}
        self._poll_task: asyncio.Task | None = None
        self._inflight: dict[str, int] = {}   # model name -> in-flight request count (B3)
        self._procs: dict[str, Any] = {}
        self.planner = planner_mod.Planner(self)

    def _conn(self):
        c = db.connect(self.db_path)
        db.init_db(c)
        return c

    def _lock(self, name: str) -> asyncio.Lock:
        if name not in self._locks:
            self._locks[name] = asyncio.Lock()
        return self._locks[name]

    def _get(self, name: str) -> dict[str, Any] | None:
        with self._conn() as c:
            return db.get_model(c, name)

    def _set_state(self, name: str, state: str, **kw: Any) -> None:
        with self._conn() as c:
            db.set_state(c, name, state, **kw)
            db.audit(c, self._get_id(name), state, json.dumps(kw, default=str))

    def _get_id(self, name: str) -> int | None:
        m = self._get(name)
        return m["id"] if m else None

    def _touch(self, name: str) -> None:
        """Update last_active_at (LRU for eviction)."""
        import time as _t
        with self._conn() as c:
            c.execute("UPDATE models SET last_active_at=?, updated_at=? WHERE name=?",
                      (_t.time(), _t.time(), name))
            c.commit()

    def _measure(self, name: str, container: str, sleep: bool = False) -> None:
        """Measure actual memory and record it (B1)."""
        try:
            gpu_mb = launcher.container_gpu_mem_mb(container)
            if gpu_mb:
                if sleep:
                    self._set_state(name, SLEEPING, measured_weight_memory_mb=gpu_mb)
                else:
                    self._set_state(name, AWAKE, measured_max_memory_mb=gpu_mb)
            # weight memory from launch log (only meaningful once, after cold start)
            if not sleep:
                log_path = launcher.RECIPE_OUT_DIR / f"{name}.launch.log"
                wmb = launcher.parse_weight_memory_gb(log_path)
                if wmb:
                    self._set_state(name, AWAKE, measured_weight_memory_mb=wmb)
        except Exception as e:
            log.warning("measure failed for %s: %s", name, e)

    # ---- in-flight tracking (B3 graceful drain) ----
    def incr_inflight(self, name: str) -> None:
        self._inflight[name] = self._inflight.get(name, 0) + 1

    def decr_inflight(self, name: str) -> None:
        self._inflight[name] = max(0, self._inflight.get(name, 0) - 1)

    async def _drain_inflight(self, name: str, timeout: float = 30.0) -> None:
        import time as _t
        deadline = _t.time() + timeout
        while self._inflight.get(name, 0) > 0 and _t.time() < deadline:
            await asyncio.sleep(0.5)

    # ---- public lifecycle ops ----

    async def start(self, name: str) -> bool:
        """Cold-start a STOPPED model: launch container, wait for /health."""
        async with self._lock(name):
            m = self._get(name)
            if not m:
                return False
            if m["state"] in (AWAKE, STARTING):
                return True
            # B1/B2: make room before launching (sleep/stop LRU models to fit RAM)
            ok, reason = await self.planner.make_room(name)
            if not ok:
                log.warning("not starting %s: %s", name, reason)
                self._set_state(name, ERROR, error=f"insufficient memory: {reason}")
                return False
            container = _container_name(name)
            cfg = json.loads(m["config"])
            log.info("starting %s (container %s, port %s)", name, container, m["port"])
            self._set_state(name, STARTING, container_name=container, port=m["port"], error=None)
            try:
                proc = launcher.launch(name, cfg, container)
                self._procs[name] = proc
            except Exception as e:
                self._set_state(name, ERROR, error=str(e))
                return False
            # poll for health (cold start can take minutes)
            client = VLLMClient(m["port"])
            ok = await self._wait_health(client, name)
            if ok:
                self._set_state(name, AWAKE, error=None)
                self._touch(name)
                self._measure(name, container)
                log.info("%s AWAKE", name)
            else:
                self._set_state(name, ERROR, error="cold start health timeout")
            return ok

    async def _wait_health(self, client: VLLMClient, name: str) -> bool:
        elapsed = 0.0
        while elapsed < START_MAX_WAIT:
            if await client.health():
                return True
            await asyncio.sleep(START_POLL_INTERVAL)
            elapsed += START_POLL_INTERVAL
        return False

    async def sleep_model(self, name: str, level: int = 1) -> bool:
        """Put an AWAKE model to sleep. Drains in-flight requests first (B3)."""
        async with self._lock(name):
            m = self._get(name)
            if not m or m["state"] != AWAKE:
                return False
            await self._drain_inflight(name)
            client = VLLMClient(m["port"])
            log.info("sleeping %s level=%d", name, level)
            if not await client.sleep(level):
                self._set_state(name, ERROR, error="sleep request failed")
                return False
            sleeping = await client.is_sleeping()
            if sleeping:
                self._set_state(name, SLEEPING)
                self._measure(name, m["container_name"], sleep=True)
                log.info("%s SLEEPING", name)
                return True
            self._set_state(name, ERROR, error="sleep did not confirm")
            return False

    async def wake_model(self, name: str) -> bool:
        """Wake a SLEEPING model."""
        async with self._lock(name):
            m = self._get(name)
            if not m or m["state"] not in (SLEEPING,):
                return False
            self._set_state(name, WAKING)
            client = VLLMClient(m["port"])
            log.info("waking %s", name)
            if not await client.wake():
                self._set_state(name, ERROR, error="wake request failed")
                return False
            # wait for not-sleeping + healthy
            for _ in range(30):
                is_s = await client.is_sleeping()
                if is_s is False and await client.health():
                    self._set_state(name, AWAKE, error=None)
                    self._touch(name)
                    log.info("%s AWAKE", name)
                    return True
                await asyncio.sleep(1)
            self._set_state(name, ERROR, error="wake did not become healthy")
            return False

    async def stop(self, name: str) -> bool:
        """Stop and remove the container for a model."""
        async with self._lock(name):
            m = self._get(name)
            if not m:
                return False
            container = m["container_name"] or _container_name(name)
            log.info("stopping %s (container %s)", name, container)
            ok = launcher.stop(container)
            self._set_state(name, STOPPED, container_name=None)
            return ok

    def can_serve(self, name: str) -> bool:
        """Route guard: True only if the model is AWAKE right now."""
        m = self._get(name)
        return bool(m and m["state"] in CAN_SERVE)

    def status(self) -> list[dict[str, Any]]:
        with self._conn() as c:
            return db.list_models(c)

    # ---- background reconcile poll ----

    async def start_polling(self) -> None:
        if self._poll_task is None or self._poll_task.done():
            self._poll_task = asyncio.create_task(self._poll_loop())

    async def stop_polling(self) -> None:
        if self._poll_task:
            self._poll_task.cancel()
            self._poll_task = None

    async def _poll_loop(self) -> None:
        log.info("lifecycle poll loop started")
        while True:
            try:
                await self._reconcile()
            except Exception as e:
                log.exception("poll loop error: %s", e)
            await asyncio.sleep(POLL_INTERVAL)

    async def _reconcile(self) -> None:
        """Probe every non-STOPPED model and fix drifted state."""
        models = self.status()
        for m in models:
            if m["state"] in (PENDING, DISCOVERED, WAKING):
                continue  # don't fight no-container states (STARTING gets probed for crash detection)
            # a STOPPED model that still has a container is an orphan (survived a
            # manager restart) — probe and adopt it instead of skipping.
            if not m["container_name"]:
                continue
            client = VLLMClient(m["port"])
            healthy = await client.health()
            is_s = await client.is_sleeping()
            # detect dead container
            if not m["container_name"] or not launcher.is_running(m["container_name"]):
                if m["state"] != STOPPED:
                    log.warning("%s container gone -> STOPPED", m["name"])
                    self._set_state(m["name"], STOPPED, container_name=None)
                continue
            if not healthy and is_s is not True:
                # container alive but not healthy and not sleeping -> error
                if m["state"] != ERROR:
                    self._set_state(m["name"], ERROR, error="health check failing")
            elif healthy and is_s is True and m["state"] != SLEEPING:
                self._set_state(m["name"], SLEEPING)
            elif healthy and is_s is False and (m["state"] != AWAKE or m["error"]):
                self._set_state(m["name"], AWAKE, error=None)
