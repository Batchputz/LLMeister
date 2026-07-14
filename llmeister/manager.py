"""LLMeister — FastAPI app. /v1/* proxy, /api/* dashboard+research, / static SPA."""
from __future__ import annotations

import re
import asyncio
import json
import logging
import shutil
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import config
from . import db
from . import launcher
from . import benchmark
import threading
import time as _time
import subprocess
from . import lifecycle as lc
import research

log = logging.getLogger("manager")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MGR = lc.LifecycleManager()

# --- resource monitor (cpu/gpu) ---
_cpu_pct = 0.0
_gpu_pct = 0

def _monitor_resources():
    global _cpu_pct, _gpu_pct
    import psutil
    psutil.cpu_percent(interval=0.3)  # prime
    while True:
        _cpu_pct = round(psutil.cpu_percent(interval=2.0), 1)
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3)
            _gpu_pct = int(out.stdout.strip().split()[0])
        except Exception:
            _gpu_pct = -1
        _time.sleep(1)

threading.Thread(target=_monitor_resources, daemon=True).start()

# --- one-shot system info (cached at startup) ---
_sys_info = {"os": "", "gpu": "", "cuda_driver": "", "vllm_version": "", "docker_version": "", "arch": ""}

def _collect_sys_info():
    global _sys_info
    import platform
    _sys_info["arch"] = platform.machine()
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    _sys_info["os"] = line.split("=",1)[1].strip().strip('"')
                    break
    except Exception: pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        parts = out.stdout.strip().split(", ")
        if len(parts) >= 2:
            _sys_info["gpu"] = parts[0]
            _sys_info["cuda_driver"] = parts[1]
    except Exception: pass
    try:
        out = subprocess.run(["docker", "run", "--rm", "vllm-node:latest", "pip", "show", "vllm"],
                           capture_output=True, text=True, timeout=20)
        for line in out.stdout.splitlines():
            if line.startswith("Version: "):
                _sys_info["vllm_version"] = line.split(": ",1)[1].strip()
                break
    except Exception: pass
    try:
        out = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        _sys_info["docker_version"] = out.stdout.strip().split()[2].rstrip(",")
    except Exception: pass

threading.Thread(target=_collect_sys_info, daemon=True).start()

_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"


def _build_name_index() -> dict[str, str]:
    idx: dict[str, str] = {}
    for m in MGR.status():
        name = m["name"]; idx[name] = name
        cfg = json.loads(m["config"])
        for a in cfg.get("aliases") or []:
            idx[a] = name
        if cfg.get("gateway_name"):
            idx[cfg["gateway_name"]] = name
        for s in cfg.get("served_model_names") or []:
            idx[s] = name
        idx[m["hf_model_id"]] = name
    return idx


def _resolve(model: str | None) -> str | None:
    return _build_name_index().get(model) if model else None


def _upstream_id(m: dict[str, Any]) -> str:
    cfg = json.loads(m["config"])
    return cfg.get("use_model_name") or (cfg.get("served_model_names") or [m["hf_model_id"]])[0]


def _scan_cached() -> list[str]:
    ids = []
    if _HF_CACHE.exists():
        for d in sorted(_HF_CACHE.iterdir()):
            if d.is_dir() and d.name.startswith("models--"):
                parts = d.name.split("--")[1:]
                if len(parts) >= 2:
                    ids.append("/".join(parts))
    return ids


def _sname(repo_id: str) -> str:
    return repo_id.lower().replace("/", "_").replace("-", "_").replace(".", "_")


def _hf_cache_dir_for(hf_model_id: str) -> Path:
    """HF hub cache dir for a repo id: models--org--name."""
    return _HF_CACHE / ("models--" + hf_model_id.replace("/", "--"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    with db.connect() as c:
        db.init_db(c)
    await MGR.start_polling()
    log.info("manager ready on port %s", config.load()["manager"]["port"])
    yield
    await MGR.stop_polling()


app = FastAPI(title="LLMeister", lifespan=lifespan)


@app.get("/v1/models")
async def list_models() -> dict:
    data = []
    for m in MGR.status():
        cfg = json.loads(m["config"])
        names = {m["name"], cfg.get("gateway_name") or m["hf_model_id"], m["hf_model_id"]}
        names.update(cfg.get("aliases") or []); names.update(cfg.get("served_model_names") or [])
        names.discard(None)
        for n in names:
            data.append({"id": n, "object": "model", "owned_by": "dgx-spark",
                         "state": m["state"], "awake": m["state"] == lc.AWAKE})
    return {"object": "list", "data": data}


@app.api_route("/v1/{path:path}", methods=["GET", "POST"])
async def v1_proxy(path: str, request: Request) -> Response:
    if path == "models":
        return await list_models()
    body = await request.body()
    try:
        payload = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON body"}, status_code=400)
    name = _resolve(payload.get("model"))
    if not name:
        return JSONResponse({"error": {"message": f"unknown model '{payload.get('model')}'", "type": "invalid_request_error"}}, status_code=404)
    m = MGR._get(name)
    if not m:
        return JSONResponse({"error": {"message": f"model '{name}' not registered"}}, status_code=404)
    if not MGR.can_serve(name):
        state = m["state"]
        if state == lc.SLEEPING and config.load().get("wake_on_request"):
            await MGR.wake_model(name); m = MGR._get(name)
        else:
            return JSONResponse({"error": {"message": f"model '{payload.get('model')}' is {state} (not AWAKE). Wake via POST /api/{name}/wake.", "type": "service_unavailable", "state": state}}, status_code=503)
    uid = _upstream_id(m)
    if payload.get("model") != uid:
        payload["model"] = uid; body = json.dumps(payload).encode()
    url = f"http://{config.load()['vllm']['host']}:{m['port']}/v1/{path}"
    drop = {"host", "content-length", "transfer-encoding", "connection"}
    headers = {k: v for k, v in request.headers.items() if k.lower() not in drop}
    stream = payload.get("stream", False)
    MGR.incr_inflight(name)
    if stream:
        async def gen():
            try:
                async with httpx.AsyncClient(timeout=None) as c:
                    async with c.stream("POST", url, content=body, headers=headers) as r:
                        async for chunk in r.aiter_raw():
                            yield chunk
            finally:
                MGR.decr_inflight(name)
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            r = await c.request(request.method, url, content=body, headers=headers)
        return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type", "application/json"))
    finally:
        MGR.decr_inflight(name)


# ---- /api dashboard ----

@app.get("/api/status")
async def api_status() -> dict:
    models = []
    for m in MGR.status():
        cfg = json.loads(m["config"])
        models.append({"name": m["name"], "hf_model_id": m["hf_model_id"], "state": m["state"],
                       "port": m["port"], "container": m["container_name"], "aliases": cfg.get("aliases") or [],
                       "measured_max_memory_mb": m["measured_max_memory_mb"], "measured_weight_memory_mb": m["measured_weight_memory_mb"],
                       "error": m["error"], "notes": cfg.get("research_notes"), "needs_research": cfg.get("needs_research", False)})
    return {"models": models}


@app.post("/api/{name}/start")
async def api_start(name: str) -> dict:
    ok = await MGR.start(name); return {"ok": ok, "state": (MGR._get(name) or {}).get("state")}


@app.post("/api/{name}/wake")
async def api_wake(name: str) -> dict:
    ok = await MGR.wake_model(name); return {"ok": ok, "state": (MGR._get(name) or {}).get("state")}


@app.post("/api/{name}/sleep")
async def api_sleep(name: str) -> dict:
    ok = await MGR.sleep_model(name); return {"ok": ok, "state": (MGR._get(name) or {}).get("state")}


@app.post("/api/{name}/stop")
async def api_stop(name: str) -> dict:
    ok = await MGR.stop(name); return {"ok": ok, "state": (MGR._get(name) or {}).get("state")}


@app.get("/api/system")
async def api_system() -> dict:
    import psutil
    vm = psutil.virtual_memory()
    return {"mem_total_gb": round(vm.total / 1024**3, 1), "mem_used_gb": round(vm.used / 1024**3, 1),
            "mem_available_gb": round(vm.available / 1024**3, 1), "mem_percent": vm.percent,
            "cpu_percent": _cpu_pct, "gpu_percent": _gpu_pct,
            "sys_os": _sys_info.get("os",""), "sys_gpu": _sys_info.get("gpu",""),
            "sys_cuda_driver": _sys_info.get("cuda_driver",""), "sys_vllm": _sys_info.get("vllm_version",""),
            "sys_docker": _sys_info.get("docker_version",""), "sys_arch": _sys_info.get("arch",""),
            "uptime_seconds": int(_time.time() - psutil.boot_time())}


@app.get("/api/planner/{name}")
async def api_planner(name: str) -> dict:
    return MGR.planner.admission(name)


@app.post("/api/{name}/benchmark")
async def api_benchmark(name: str) -> dict:
    import json as _json
    m = MGR._get(name)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    if m["state"] != "AWAKE":
        return JSONResponse({"error": f"model not AWAKE (state={m['state']})"}, status_code=400)
    port = m["port"]
    if not port:
        return JSONResponse({"error": "no port"}, status_code=400)
    cfg = _json.loads(m["config"])
    served_name = cfg.get("served_model_name") or m["hf_model_id"]
    try:
        result = await benchmark.run_benchmark(port, served_name)
        return result
    except Exception as e:
        log.exception("benchmark failed for %s", name)
        return JSONResponse({"error": str(e)}, status_code=500)


# ---- /api new-model research ----

@app.post("/api/research")
async def api_research(req: Request) -> dict:
    payload = await req.json()
    model_id = (payload.get("model_id") or "").strip()
    if not model_id:
        return JSONResponse({"error": "model_id required"}, status_code=400)
    try:
        result = await asyncio.to_thread(research.research_model, model_id)
    except Exception as e:
        log.exception("research failed"); return JSONResponse({"error": str(e)}, status_code=500)
    cand = result["candidate"]
    short = _sname(model_id)
    with db.connect() as c:
        db.init_db(c)
        c.execute("DELETE FROM models WHERE name = ? AND state IN ('PENDING','DISCOVERED')", (short,))
        c.commit(); db.add_model(c, short, cand["hf_model_id"], cand, state="PENDING")
    return {"name": short, "candidate": cand, "citations": result["citations"], "notes": result["notes"]}


@app.get("/api/models/{name}")
async def api_get_model(name: str) -> dict:
    with db.connect() as c:
        db.init_db(c); m = db.get_model(c, name)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    m["config"] = json.loads(m["config"]); return m


@app.post("/api/models/{name}/approve")
async def api_approve(name: str) -> dict:
    with db.connect() as c:
        db.init_db(c); m = db.get_model(c, name)
        if not m:
            return JSONResponse({"error": "not found"}, status_code=404)
        if m["state"] != "PENDING":
            return JSONResponse({"error": f"not PENDING (state={m['state']})"}, status_code=400)
        used = {row["port"] for row in db.list_models(c) if row["port"] and row["name"] != name}
        lo, hi = config.load()["vllm"]["port_range"]
        port = launcher.allocate_port(used, lo, hi)
        if not port:
            return JSONResponse({"error": "no free port in range"}, status_code=500)
        cfg = json.loads(m["config"]); cfg.setdefault("defaults", {})["port"] = port
        c.execute("UPDATE models SET state='STOPPED', port=?, config=?, updated_at=? WHERE name=?", (port, json.dumps(cfg), db.now(), name))
        c.commit()
    return {"ok": True, "name": name, "port": port, "state": "STOPPED"}


@app.put("/api/models/{name}/config")
async def api_update_config(name: str, req: Request) -> dict:
    payload = await req.json()
    with db.connect() as c:
        db.init_db(c)
        if not db.get_model(c, name):
            return JSONResponse({"error": "not found"}, status_code=404)
        c.execute("UPDATE models SET config=?, error=NULL, updated_at=? WHERE name=?", (json.dumps(payload), db.now(), name))
        c.commit()
    return {"ok": True}


@app.delete("/api/models/{name}")
async def api_delete_model(name: str) -> dict:
    """Full delete: stop+remove container, delete HF cache weights, drop registry row."""
    with db.connect() as c:
        db.init_db(c); m = db.get_model(c, name)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    # 1. stop + remove the docker container if one exists
    container = m["container_name"]
    if container:
        await asyncio.to_thread(launcher.stop, container)
    # 2. delete the model weights from the HF cache (root-owned files need docker rm)
    cache_dir = _hf_cache_dir_for(m["hf_model_id"])
    removed_cache = False
    cache_err = None
    if cache_dir.exists() and cache_dir.name.startswith("models--"):
        try:
            await asyncio.to_thread(shutil.rmtree, str(cache_dir))
            removed_cache = True
        except Exception as e:
            log.warning("rmtree failed (%s); falling back to root container rm", e)
            ok = await asyncio.to_thread(launcher.rm_path_as_root, cache_dir)
            removed_cache = ok
            if not ok:
                cache_err = str(e)
    # 3. drop the registry row (only if cache deletion succeeded or there was no cache)
    if cache_err:
        with db.connect() as c:
            db.init_db(c)
            c.execute("UPDATE models SET error=?, updated_at=? WHERE name=?", ("cache delete failed: "+cache_err, db.now(), name))
            c.commit()
        return JSONResponse({"error": "cache delete failed: "+cache_err}, status_code=500)
    with db.connect() as c:
        db.init_db(c)
        c.execute("DELETE FROM models WHERE name=?", (name,)); c.commit()
    log.info("deleted model %s (container=%s, cache_removed=%s)", name, container, removed_cache)
    return {"ok": True, "container_removed": bool(container), "cache_removed": removed_cache, "cache_path": str(cache_dir)}


@app.get("/api/models/{name}/log")
async def api_get_log(name: str, lines: int = 2000) -> dict:
    """Tail the vLLM launch log for a model (docker logs is empty for these containers)."""
    import subprocess
    path = launcher.RECIPE_OUT_DIR / f"{name}.launch.log"
    if not path.exists():
        return JSONResponse({"error": "no launch log for this model"}, status_code=404)
    n = max(50, min(int(lines), 8000))
    r = subprocess.run(["tail", "-n", str(n), str(path)], capture_output=True, text=True, timeout=10)
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", r.stdout)  # strip ANSI color codes
    return {"path": str(path), "lines": n, "text": text}


# ---- /api discovered (cached HF models) ----

@app.post("/api/import-cached")
async def api_import_cached() -> dict:
    added, skipped = 0, 0
    with db.connect() as c:
        db.init_db(c)
        existing = {row["hf_model_id"] for row in db.list_models(c)}
        for repo_id in _scan_cached():
            if repo_id in existing:
                skipped += 1; continue
            name = _sname(repo_id)
            base, n = name, 2
            while c.execute("SELECT 1 FROM models WHERE name=?", (name,)).fetchone():
                name = f"{base}_{n}"; n += 1
            cfg = {"hf_model_id": repo_id, "container": "vllm-node", "defaults": {"port": 0},
                   "env": {}, "mods": [], "command": "", "served_model_names": [], "aliases": [],
                   "use_model_name": None, "gateway_name": repo_id, "needs_research": True}
            try:
                db.add_model(c, name, repo_id, cfg, state="DISCOVERED"); added += 1
            except sqlite3.IntegrityError:
                skipped += 1
    return {"added": added, "skipped": skipped}


@app.post("/api/models/{name}/research")
async def api_research_existing(name: str) -> dict:
    with db.connect() as c:
        db.init_db(c); m = db.get_model(c, name)
    if not m:
        return JSONResponse({"error": "not found"}, status_code=404)
    if m["state"] not in ("DISCOVERED", "PENDING"):
        return JSONResponse({"error": f"cannot research model in state {m['state']}"}, status_code=400)
    try:
        result = await asyncio.to_thread(research.research_model, m["hf_model_id"])
    except Exception as e:
        log.exception("research failed"); return JSONResponse({"error": str(e)}, status_code=500)
    cand = result["candidate"]
    with db.connect() as c:
        db.init_db(c)
        c.execute("UPDATE models SET config=?, state='PENDING', error=NULL, updated_at=? WHERE name=?", (json.dumps(cand), db.now(), name))
        c.commit()
    return {"ok": True, "name": name, "notes": result["notes"], "candidate": cand}


# ---- static dashboard ----
from llmeister import PROJECT_ROOT
STATIC_DIR = PROJECT_ROOT / "static"


class NoCacheStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp


if STATIC_DIR.exists():
    app.mount("/", NoCacheStatic(directory=str(STATIC_DIR), html=True), name="static")


def main() -> None:
    import uvicorn
    cfg = config.load()["manager"]
    uvicorn.run("manager:app", host=cfg["host"], port=int(cfg["port"]), log_level="info")


if __name__ == "__main__":
    main()
