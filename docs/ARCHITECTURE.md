# LLMeister Architecture

## Design Decisions

### vLLM Sleep Mode over CRIU/cuda-checkpoint

cuda-checkpoint is x86_64-only; the DGX Spark is aarch64 (GB10 Grace Blackwell). vLLM's built-in sleep mode (`/sleep?level=1` + `/wake_up` via `CuMemAllocator`) works on any CUDA platform and is zero-copy on GB10's unified memory.

- **Sleep level 1**: frees KV cache memory via `cuMemUnmap`. Weights stay mapped.
- **Wake**: re-maps the freed pages. ~2 seconds on GB10.
- **CUDA graphs preserved**: post-wake latency is identical to baseline.

### Mamba/GDN Hybrid Fix

vLLM 0.25.1 has a bug in `init_fp8_kv_scales` that breaks wake for Mamba/GDN-hybrid models with quantized KV cache (`'list' object has no attribute 'zero_'`). The `fix-sleep-wake-mamba` mod patches this idempotently at launch time via the recipe `mods:` list.

### Memory Planner: Live psutil, Not Estimates

The admission check gates on **live `psutil.available`** (not computed estimates), with a safety buffer. This was fixed after a crash where `os_reserve=12GB` was too optimistic (real overhead ~28GB on the DGX Spark).

Eviction policy:
1. **Tier 1**: sleep LRU awake models (frees KV cache, keeps weights)
2. **Tier 2**: stop LRU sleeping models (frees weights)
3. **Refuse**: if still not enough room, return error

### One systemd Service

A single `llmeister.service` manages ALL vLLM instances. The manager process launches containers via `run-recipe.py --solo` (non-blocking), tracks them by container name + port, and reconciles state on restart.

### No-Build SPA

The dashboard is a single `index.html` using Alpine.js (loaded from CDN). No Node.js, no build step, no npm. FastAPI serves it directly with no-cache headers.

### State Machine

```
STOPPED → STARTING → AWAKE ⇄ SLEEPING
                ↘ ERROR ↗
PENDING → (approve) → STOPPED
DISCOVERED → (import) → STOPPED
```

The reconcile loop (every 3 seconds) probes all models with containers and fixes drifted state:
- Container gone → STOPPED
- Healthy + not sleeping → AWAKE
- Healthy + sleeping → SLEEPING
- Not healthy + not sleeping → ERROR

### Proxy Routing

`/v1/*` requests are proxied to the correct vLLM backend by:
1. Resolving the model name (alias → model → served_model_name)
2. Rewriting the `model` field in the request body to the vLLM-served name
3. Forwarding to `localhost:{port}`
4. 503 if the model is sleeping (optional wake-on-request)

### Benchmark

Three standard prompts (short prefill, long prefill, code generation) sent via streaming with `stream_options: {"include_usage": true}`. Measures:
- **TTFT**: time to first token (reasoning or content)
- **tok/s**: completion tokens / generation time
- **prefill tok/s**: prompt tokens / TTFT

No external tools — pure httpx timing + vLLM's usage stats.

## Portability

vLLM is the only hard dependency. The DGX Spark makes sleep/wake *fast* (unified memory → zero-copy), but the manager doesn't assume that. On a discrete-GPU box, sleep/wake still works, just with a VRAM↔RAM copy (slower wake).

The launcher (`launcher.py`) is the one deployment-specific piece — it shells out to `run-recipe.py` from the `spark-vllm-docker` repo. Generalizing this into a configurable `docker run` path is the main refactor needed for other vLLM systems.
