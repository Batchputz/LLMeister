# LLMeister

Centralized vLLM lifecycle manager — sleep/wake hot-swap, memory-aware scheduling, web dashboard, and an AI research agent for adding new models.

**vLLM is the only hard dependency.** Originally built for an NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64, 128GB unified memory), but the lifecycle/proxy/planner/dashboard logic is vLLM-generic and works on any vLLM deployment.

## Features

- **Sleep/wake hot-swap** — vLLM sleep mode (`/sleep` + `/wake_up` via `CuMemAllocator`) frees GPU memory without stopping containers. Wake in ~2 seconds.
- **Memory-aware scheduling** — admission gates on live `psutil.available` RAM, LRU eviction (sleep → stop → refuse).
- **`/v1/*` proxy** — OpenAI-compatible API with model alias resolution and `useModelName` rewriting. 503 on sleeping models (optional wake-on-request).
- **Web dashboard** — Alpine.js no-build SPA with Running/Available/Pending/Discovered sections, live RAM/CPU/GPU gauges, chat modal, log viewer, config editor.
- **Benchmark** — 3 standard prompts measuring TTFT, tokens/sec, prefill throughput. One click per model.
- **Research agent** — Perplexity search + DeepSeek synthesis generates candidate vLLM configs for new models.
- **One systemd service** — manages ALL vLLM instances, not one per model.

## Quick Start

```bash
# Clone
git clone https://github.com/Batchputz/LLMeister.git
cd LLMeister

# Install dependencies
uv sync

# Configure
cp config.yaml config.yaml.local  # edit to match your system
# API keys for research agent (optional):
cat > ~/.config/llmeister.env << 'EOF'
PERPLEXITY_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
EOF

# Run
uv run python -m llmeister

# Open dashboard
open http://localhost:9001
```

## Project Structure

```
LLMeister/
├── config.yaml               # system config (paths, ports, API key env names)
├── pyproject.toml            # package metadata
├── llmeister/                # Python package
│   ├── manager.py            # FastAPI: /v1 proxy, /api endpoints, static serving
│   ├── config.py             # config.yaml loader
│   ├── db.py                 # SQLite model registry + seeding
│   ├── lifecycle.py          # state machine, reconcile, sleep/wake/start/stop
│   ├── launcher.py           # recipe YAML generation + run-recipe.py subprocess
│   ├── planner.py            # memory admission + LRU eviction
│   ├── research.py           # Perplexity + DeepSeek research agent
│   └── benchmark.py          # 3-prompt standard benchmark
├── static/                   # no-build SPA dashboard
│   └── index.html
├── deploy/                   # systemd unit + install scripts
│   └── llmeister.service
└── docs/
    └── ARCHITECTURE.md
```

## Configuration

`config.yaml` is the single source of truth for system config. Per-model config lives in the SQLite registry (`llmeister.db`).

API keys are loaded from `~/.config/llmeister.env` (referenced by the systemd unit's `EnvironmentFile`).

## Portability

The sleep/wake lifecycle, `/v1` proxy, memory planner, dashboard, and research agent are all vLLM-generic. The one deployment-specific piece is the **launcher** — it shells out to `run-recipe.py` (from the `spark-vllm-docker` repo) and assumes a specific vLLM Docker image. To run on another vLLM system, point `paths.spark_vllm_dir` in `config.yaml` to your launch infrastructure, or generalize the launcher.

## License

MIT
