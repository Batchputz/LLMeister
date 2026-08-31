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

from . import config
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

    async def ready(self) -> bool:
        """Check if the model is fully loaded and the inference API is ready.

        /health returns 200 as soon as the HTTP server starts, but the model
        weights may still be loading. /v1/models only lists models once they
        are fully loaded and ready to serve. This is the correct readiness
        signal for benchmarking.
        """
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as c:
                r = await c.get(f"{self.base}/v1/models")
                if r.status_code != 200:
                    return False
                data = r.json()
                # vLLM only populates /v1/models when the model is fully loaded
                return bool(data.get("data"))
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
        # crash-robustness: auto-restart bookkeeping + boot restore
        self._restart_attempts: dict[str, list[float]] = {}   # name -> timestamps of auto-restarts
        self._restart_inflight: set[str] = set()              # models with an auto-restart in flight
        self._unhealthy_streak: dict[str, int] = {}           # consecutive failed health polls
        self._restore_task: asyncio.Task | None = None        # boot restore, so it isn't GC'd
        lc_cfg = config.load().get("lifecycle", {})
        self.auto_restore_on_boot = bool(lc_cfg.get("auto_restore_on_boot", True))
        self.restart_max_attempts = int(lc_cfg.get("restart_max_attempts", 3))
        self.restart_window_s = float(lc_cfg.get("restart_window_s", 1800))
        self.restart_backoff_s = float(lc_cfg.get("restart_backoff_s", 30))
        self.unhealthy_restart_after = int(lc_cfg.get("unhealthy_restart_after", 3))

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
                # AWAKE may be stale after a host reboot / container loss: force a real
                # cold start instead of returning True if the tracked container is gone.
                if (m["state"] == AWAKE and m["container_name"]
                        and not launcher.is_running(m["container_name"])):
                    log.warning("%s marked AWAKE but container %s is gone — cold-starting",
                                name, m["container_name"])
                    self._set_state(name, STOPPED, container_name=None)
                else:
                    return True
            # B1/B2: make room before launching (sleep/stop LRU models to fit RAM)
            ok, reason = await self.planner.make_room(name)
            if not ok:
                log.warning("not starting %s: %s", name, reason)
                self._set_state(name, ERROR, error=f"insufficient memory: {reason}")
                return False
            container = _container_name(name)
            cfg = json.loads(m["config"])
            # Stale-container guard: an orphaned container from a previous
            # timed-out start would make launch-cluster.sh exec a SECOND
            # instance into it (duplicate vLLM eating double memory). Remove
            # any leftover container before launching fresh.
            if launcher.is_running(container):
                log.warning("%s stale container %s still running — removing before launch", name, container)
                launcher.stop(container)
            # Also reap any leftover launch supervisor from a previous attempt.
            stale = self._procs.pop(name, None)
            if stale and stale.poll() is None:
                log.warning("%s stale launch proc %d still alive — terminating", name, stale.pid)
                try:
                    stale.terminate(); stale.wait(timeout=10)
                except Exception:
                    try: stale.kill()
                    except Exception: pass
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
                self._set_state(name, AWAKE, error=None, restore_on_boot=1)
                self._touch(name)
                self._measure(name, container)
                self._restart_attempts.pop(name, None)
                self._unhealthy_streak.pop(name, None)
                log.info("%s AWAKE", name)
            else:
                self._set_state(name, ERROR, error="cold start health timeout")
                # Cleanup: the launched vLLM may still be loading (slow cold
                # start) and would otherwise keep consuming memory + become an
                # orphan that a later start() execs a duplicate instance into.
                log.warning("%s cold start timed out — cleaning up container %s", name, container)
                proc = self._procs.pop(name, None)
                if proc and proc.poll() is None:
                    try:
                        proc.terminate(); proc.wait(timeout=10)
                    except Exception:
                        try: proc.kill()
                        except Exception: pass
                try:
                    launcher.stop(container)
                except Exception as e:
                    log.warning("cleanup stop failed for %s: %s", name, e)
            return ok

    async def _wait_health(self, client: VLLMClient, name: str) -> bool:
        """Wait until the vLLM backend is ready to serve requests.

        Single budget (START_MAX_WAIT): DFlash cold starts with CUDA graph
        capture can take 6-10 min, especially when models load in parallel.
        The API server (/health) only starts after the engine finishes
        loading, so we poll both /health and /v1/models within one budget
        instead of failing after a fixed 5-min phase-1.
        """
        elapsed = 0.0
        server_up = False
        while elapsed < START_MAX_WAIT:
            if not server_up:
                server_up = await client.health()
                if server_up:
                    log.debug("%s HTTP server up after %.0fs", name, elapsed)
            if server_up and await client.ready():
                log.debug("%s model ready after %.0fs", name, elapsed)
                return True
            await asyncio.sleep(START_POLL_INTERVAL)
            elapsed += START_POLL_INTERVAL
        if not server_up:
            log.warning("%s HTTP server never came up within %ds", name, START_MAX_WAIT)
        else:
            log.warning("%s model did not become ready within %ds", name, START_MAX_WAIT)
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
                    self._set_state(name, AWAKE, error=None, restore_on_boot=1)
                    self._touch(name)
                    self._restart_attempts.pop(name, None)
                    self._unhealthy_streak.pop(name, None)
                    log.info("%s AWAKE", name)
                    return True
                await asyncio.sleep(1)
            self._set_state(name, ERROR, error="wake did not become healthy")
            return False

    async def stop(self, name: str, manual: bool = True) -> bool:
        """Stop and remove the container for a model.

        manual=True (operator action) clears the restore_on_boot intent; eviction /
        maintenance stops (manual=False) keep it, so the model still gets restored
        after a reboot if it fits.
        """
        async with self._lock(name):
            m = self._get(name)
            if not m:
                return False
            container = m["container_name"] or _container_name(name)
            log.info("stopping %s (container %s, manual=%s)", name, container, manual)
            ok = launcher.stop(container)
            kw: dict[str, Any] = {"container_name": None}
            if manual:
                kw["restore_on_boot"] = 0
            self._set_state(name, STOPPED, **kw)
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
        """Probe every non-STOPPED model and fix drifted state.

        Self-healing: a wanted model (restore_on_boot) whose container died or is
        unhealthy gets auto-restarted through the normal start() path (planner
        admission + crash-loop budget), so vLLM crashes and host reboots are
        recovered without operator action.
        """
        models = self.status()
        for m in models:
            if m["state"] in (PENDING, DISCOVERED, WAKING, STARTING):
                continue  # no-container states; STARTING is owned by an in-flight start()
            if not m["container_name"]:
                continue
            client = VLLMClient(m["port"])
            healthy = await client.health()
            is_s = await client.is_sleeping()
            # detect dead container
            if not launcher.is_running(m["container_name"]):
                if m["state"] != STOPPED:
                    log.warning("%s container gone -> STOPPED", m["name"])
                    self._set_state(m["name"], STOPPED, container_name=None)
                if m["restore_on_boot"] and self._restart_budget_ok(m["name"]):
                    log.info("%s auto-restarting (container died)", m["name"])
                    asyncio.create_task(self._auto_restart(m["name"], reason="container died"))
                continue
            if not healthy and is_s is not True:
                # container alive but not healthy and not sleeping -> error
                streak = self._unhealthy_streak.get(m["name"], 0) + 1
                self._unhealthy_streak[m["name"]] = streak
                if m["state"] != ERROR:
                    self._set_state(m["name"], ERROR, error="health check failing")
                if (m["restore_on_boot"] and streak >= self.unhealthy_restart_after
                        and self._restart_budget_ok(m["name"])):
                    log.warning("%s unhealthy for %d polls — auto-restarting",
                                m["name"], streak)
                    self._unhealthy_streak.pop(m["name"], None)
                    asyncio.create_task(self._auto_restart(m["name"], reason="unhealthy"))
            elif healthy and is_s is True and m["state"] != SLEEPING:
                self._set_state(m["name"], SLEEPING)
                self._unhealthy_streak.pop(m["name"], None)
            elif healthy and is_s is False and (m["state"] != AWAKE or m["error"]):
                self._set_state(m["name"], AWAKE, error=None)
                self._unhealthy_streak.pop(m["name"], None)

    # ---- crash-robustness helpers ----

    def _restore_candidates(self) -> list[dict[str, Any]]:
        """Models that were online (AWAKE/SLEEPING) and wanted before a restart."""
        with self._conn() as c:
            return [m for m in db.list_models(c)
                    if m["restore_on_boot"] and m["state"] in (AWAKE, SLEEPING)]

    async def restore_on_boot(self) -> None:
        """Cold-start every model that was online before a manager/host restart.

        Skips models whose container genuinely survived (manager restart while the
        host stayed up — reconcile will adopt them). Each start() is lock-guarded
        and goes through the memory planner (admission + LRU eviction), so a full
        model set is restored without OOMing the box.
        """
        candidates: list[dict[str, Any]] = []
        for m in self._restore_candidates():
            if m["container_name"] and launcher.is_running(m["container_name"]):
                continue  # container survived — reconcile adopts it back
            candidates.append(m)
        if not candidates:
            return
        log.info("restoring %d model(s) after (re)start: %s",
                 len(candidates), [m["name"] for m in candidates])
        await asyncio.gather(*(self.start(m["name"]) for m in candidates))

    def _restart_budget_ok(self, name: str) -> bool:
        """Crash-loop guard: allow an auto-restart if attempts in the window are below max."""
        import time as _t
        now = _t.time()
        attempts = [t for t in self._restart_attempts.get(name, [])
                    if now - t < self.restart_window_s]
        self._restart_attempts[name] = attempts
        return len(attempts) < self.restart_max_attempts

    async def _auto_restart(self, name: str, reason: str = "auto") -> None:
        """Re-launch a model through the normal start path, with backoff + budget."""
        if name in self._restart_inflight:
            return
        self._restart_inflight.add(name)
        try:
            if not self._restart_budget_ok(name):
                self._set_state(name, ERROR, error="auto-restart budget exhausted")
                log.warning("%s not auto-restarting: crash-loop budget exhausted (%s)",
                            name, reason)
                return
            import time as _t
            self._restart_attempts.setdefault(name, []).append(_t.time())
            if self.restart_backoff_s:
                await asyncio.sleep(self.restart_backoff_s)
            log.info("auto-restarting %s (%s)", name, reason)
            await self.start(name)
        except Exception as e:
            log.exception("auto-restart failed for %s: %s", name, e)
        finally:
            self._restart_inflight.discard(name)
