"""Memory planner: admission control + eviction for fitting models into RAM.

Before starting a model, check it fits: vLLM reserves gpu_memory_utilization * total
of FREE memory, so we gate on LIVE available memory (psutil) — not a computed
estimate, because real OS/other-process overhead is only visible at runtime.

If it doesn't fit, make room by:
  Tier 1: sleep LRU AWAKE models (frees KV cache, keeps weights)
  Tier 2: stop LRU SLEEPING models (frees weights entirely)
If still not enough, refuse with a reason.

Memory accounting on GB10 unified memory:
  - AWAKE footprint = nvidia-smi GPU mem (accurate; docker-stats undercounts).
  - SLEEPING footprint = weight memory (retained in physical RAM after cuMemUnmap).
  - vLLM's free-memory check sees the same pool psutil reports (~2 GiB delta).
"""
from __future__ import annotations

import json
import logging
import psutil
from typing import Any, TYPE_CHECKING

import config
import db

if TYPE_CHECKING:
    import lifecycle as lc

log = logging.getLogger("planner")

# safety buffer: psutil.available overstates vLLM's cuda free by ~2 GiB (reclaimable cache)
SAFETY_BUFFER_MB = 3 * 1024


def system_mem_mb() -> tuple[int, int]:
    vm = psutil.virtual_memory()
    return int(vm.total / 1024**2), int(vm.available / 1024**2)


def _reservation_mb(m: dict[str, Any], total_mb: int) -> int:
    """What vLLM will try to reserve on startup = gpu_memory_utilization * total."""
    cfg = json.loads(m["config"])
    util = (cfg.get("defaults") or {}).get("gpu_memory_utilization") or 0.4
    return int(util * total_mb)


def _awake_mem_mb(m: dict[str, Any], total_mb: int) -> int:
    """Current AWAKE footprint. Measured if available, else the reservation estimate."""
    return m["measured_max_memory_mb"] or _reservation_mb(m, total_mb)


def _sleep_mem_mb(m: dict[str, Any], total_mb: int) -> int:
    """SLEEPING footprint (weights retained). Measured if available, else ~40% of reservation."""
    return m["measured_weight_memory_mb"] or int(_reservation_mb(m, total_mb) * 0.4)


class Planner:
    def __init__(self, mgr: "lc.LifecycleManager"):
        self.mgr = mgr

    def _all_models(self) -> list[dict[str, Any]]:
        with db.connect(self.mgr.db_path) as c:
            db.init_db(c)
            return db.list_models(c)

    def admission(self, target_name: str) -> dict[str, Any]:
        """Check if target fits in LIVE free memory. Returns admission + eviction plan."""
        total_mb, available_mb = system_mem_mb()
        models = self._all_models()
        target = next((m for m in models if m["name"] == target_name), None)
        if not target:
            return {"fits": False, "reason": f"unknown model {target_name}"}

        target_need = _reservation_mb(target, total_mb) + SAFETY_BUFFER_MB

        if available_mb >= target_need:
            return {"fits": True, "need_to_free_mb": 0, "plan": [],
                    "target_mem_mb": target_need, "available_mb": available_mb,
                    "reason": "fits"}

        need = target_need - available_mb
        plan: list[dict[str, Any]] = []
        freed = 0

        # Tier 1: sleep LRU AWAKE models (frees awake - sleep)
        awake_others = sorted(
            [m for m in models if m["state"] == "AWAKE" and m["name"] != target_name],
            key=lambda m: m.get("last_active_at") or 0,
        )
        for m in awake_others:
            if freed >= need:
                break
            gain = _awake_mem_mb(m, total_mb) - _sleep_mem_mb(m, total_mb)
            if gain <= 0:
                continue
            plan.append({"name": m["name"], "action": "sleep", "frees_mb": gain})
            freed += gain

        # Tier 2: stop LRU SLEEPING models (frees all weight memory)
        sleeping_others = sorted(
            [m for m in models if m["state"] == "SLEEPING" and m["name"] != target_name],
            key=lambda m: m.get("last_active_at") or 0,
        )
        for m in sleeping_others:
            if freed >= need:
                break
            gain = _sleep_mem_mb(m, total_mb)
            if gain <= 0:
                continue
            plan.append({"name": m["name"], "action": "stop", "frees_mb": gain})
            freed += gain

        if freed < need:
            return {"fits": False, "need_to_free_mb": need, "plan": plan,
                    "target_mem_mb": target_need, "available_mb": available_mb,
                    "reason": f"need {need} MB free, can only free {freed} MB by evicting "
                              f"{len(plan)} model(s) — not enough room"}

        return {"fits": True, "need_to_free_mb": need, "plan": plan,
                "target_mem_mb": target_need, "available_mb": available_mb,
                "reason": f"fits after evicting {len(plan)} model(s): "
                          + ", ".join(f"{s['action']} {s['name']}" for s in plan)}

    async def make_room(self, target_name: str) -> tuple[bool, str]:
        """Execute the eviction plan so target fits. Returns (ok, reason)."""
        res = self.admission(target_name)
        if res["fits"] and not res["plan"]:
            return True, res["reason"]
        log.info("make_room for %s: %s", target_name, res["reason"])
        for step in res["plan"]:
            name = step["name"]
            action = step["action"]
            log.info("evicting %s (%s, ~%d MB)", name, action, step["frees_mb"])
            if action == "sleep":
                ok = await self.mgr.sleep_model(name)
            else:  # stop
                ok = await self.mgr.stop(name)
            if not ok:
                log.warning("eviction %s %s failed", action, name)
        # re-check after evictions
        res2 = self.admission(target_name)
        if res2["fits"] and not res2["plan"]:
            return True, res2["reason"]
        return False, res2.get("reason", "")
