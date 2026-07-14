# vLLM Parameter Optimization Agent — Design Plan

## Overview

An autonomous, iterative agent that optimizes vLLM serving parameters for a specific
model + workload. Triggered from the Edit modal in the LLMeister dashboard. Runs
overnight on development systems. Uses LLM-driven reasoning (DeepSeek) + online
research (Perplexity) + benchmark feedback to squeeze out optimal performance.

Separate from the existing research agent (which discovers new models). This agent
optimizes parameters for an already-registered model.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (Edit Modal)                     │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────────────┐ │
│  │ Interview    │→ │ Optimization │→ │ Results Timeline    │ │
│  │ (chat-like)  │  │ Progress     │  │ (steps + benchmarks) │ │
│  └─────────────┘  └──────────────┘  └─────────────────────┘ │
└──────────────────────────┬──────────────────────────────────┘
                           │
                    POST /api/{name}/optimize/*
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Optimization Agent                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐ │
│  │ Interview │→ │ Research │→ │ Baseline │→ │ Optimize    │ │
│  │ (DeepSeek)│  │(Perplexity)│ │ Benchmark│  │ Loop (RL)   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────┬──────┘ │
│                                                    │         │
│  ┌─────────────────────────────────────────────────┘         │
│  │  For each iteration:                                      │
│  │  1. LLM reasons about next parameter change              │
│  │  2. Apply change to config                               │
│  │  3. Restart vLLM container                               │
│  │  4. Wait for health check                                │
│  │  5. Run workload-aware benchmark                          │
│  │  6. Capture vLLM metrics (/metrics endpoint)             │
│  │  7. Compare to best → keep or revert                     │
│  │  8. Store step in DB                                     │
│  │  9. Repeat until converged or max iterations             │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Database Schema

### Table: `optimization_runs`

Tracks each optimization session (one per model + workload).

```sql
CREATE TABLE optimization_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name      TEXT NOT NULL,           -- e.g. 'qwen9b'
    hf_model_id     TEXT NOT NULL,           -- e.g. 'Intel/Qwen3.5-9B-int4-AutoRound'
    use_case        TEXT,                    -- user's workload description
    workload        TEXT,                    -- JSON: {concurrency, input_tokens, output_tokens, type}
    status          TEXT DEFAULT 'interview',-- interview|researching|running|completed|failed|stopped
    research_json   TEXT,                    -- Perplexity research results (cached)
    baseline_step   INTEGER,                 -- FK to optimization_steps.id (step 0)
    best_step       INTEGER,                 -- FK to optimization_steps.id (best result)
    total_steps     INTEGER DEFAULT 0,
    started_at      TEXT,
    completed_at    TEXT,
    error           TEXT
);
```

### Table: `optimization_steps`

Tracks each parameter change + benchmark result.

```sql
CREATE TABLE optimization_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL,
    step_number     INTEGER NOT NULL,
    -- what changed
    parameter       TEXT,                    -- e.g. 'max_num_seqs'
    old_value       TEXT,                    -- e.g. '128'
    new_value       TEXT,                    -- e.g. '256'
    reasoning       TEXT,                    -- LLM's reasoning for this change
    -- full config snapshot
    config_json     TEXT NOT NULL,           -- complete config at this step
    -- benchmark results
    benchmark_json  TEXT,                    -- {ttft, tok_per_sec, prefill_tok_per_sec, ...}
    metrics_json    TEXT,                    -- vLLM /metrics snapshot (KV cache, queue depth, etc.)
    -- evaluation
    score           REAL,                    -- composite score (higher = better)
    is_improvement  INTEGER DEFAULT 0,       -- 1 if better than previous best
    kept            INTEGER DEFAULT 0,       -- 1 if this config was kept (not reverted)
    -- timing
    restart_time_s  REAL,                    -- how long the restart took
    benchmark_time_s REAL,                   -- how long the benchmark took
    timestamp       TEXT,
    FOREIGN KEY (run_id) REFERENCES optimization_runs(id)
);
```

## Agent Workflow

### Phase 1: Interview (user chat)

The agent asks the user what the LLM is needed for. The user answers in natural language:

```
Agent: What is this model needed for? Describe your workload.
User: Batch processing of images, 64 parallel requests each ~8000 tokens
Agent: What's the expected output length per request?
User: Around 500 tokens
Agent: Is latency critical (interactive) or is throughput the priority?
User: Throughput — it runs as a batch job
```

The agent (DeepSeek) parses this into a structured workload:

```json
{
    "type": "batch",
    "concurrency": 64,
    "input_tokens": 8000,
    "output_tokens": 500,
    "priority": "throughput",
    "description": "Batch processing of images, 64 parallel requests each ~8000 tokens, ~500 token outputs, throughput priority"
}
```

If the user isn't specific enough, the agent asks follow-up questions until it has all fields.

### Phase 2: Research (Perplexity)

The agent searches for optimization tips for the specific model + workload:

```
Query 1: "vLLM Qwen3.5-9B optimization parameters batch throughput"
Query 2: "vLLM max_num_seqs max_num_batched_tokens tuning large context"
Query 3: "vLLM chunked prefill prefix caching performance impact"
```

Results are cached in `optimization_runs.research_json`. The DeepSeek LLM synthesizes them into a list of recommended starting parameters and optimization candidates.

### Phase 3: Baseline Benchmark

Run the workload-aware benchmark with the current (starting) parameters. Store as step 0.

The benchmark is **adapted to the workload**:
- Generates synthetic prompts matching the input token count
- Sends `concurrency` parallel requests
- Measures: TTFT, tokens/sec, prefill tokens/sec, total throughput, wall time
- Also captures vLLM `/metrics` for KV cache usage, queue depth, etc.

### Phase 4: Optimization Loop (the core)

Each iteration:

1. **LLM reasoning** — DeepSeek receives:
   - Current parameters + benchmark results
   - vLLM metrics (KV cache usage, queue depth, prefix hit rate)
   - History of all previous steps (what was tried, what worked, what didn't)
   - Research recommendations
   - The workload target

   DeepSeek outputs:
   ```json
   {
     "parameter": "max_num_batched_tokens",
     "old_value": "16384",
     "new_value": "32768",
     "reasoning": "Current KV cache usage is only 45% with 64 concurrent requests. Increasing max_num_batched_tokens allows more tokens per batch, which should improve prefill throughput for the 8000-token inputs. The queue depth shows 12 requests waiting, indicating the batch size is the bottleneck."
   }
   ```

2. **Apply change** — update the config in the DB

3. **Restart vLLM** — stop container, start with new config, wait for health check

4. **Run benchmark** — workload-aware benchmark (same as baseline)

5. **Capture metrics** — read vLLM `/metrics` endpoint

6. **Evaluate** — compute composite score:
   ```
   score = w1 * throughput + w2 * (1 / TTFT) + w3 * prefill_speed
   ```
   Weights depend on the workload priority (throughput vs latency).

7. **Keep or revert**:
   - If score > best score: keep, update best_step
   - If score <= best score: revert to previous config
   - If model fails to start: revert immediately, mark step as failed

8. **Store step** in `optimization_steps`

9. **Repeat** until:
   - LLM says "no more promising changes" (converged)
   - Max iterations reached (default: 30)
   - User stops the agent
   - Time limit reached (default: 8 hours)

### Phase 5: Report

Generate a summary:
- Starting config → best config (diff)
- Improvement: baseline score → best score (% improvement)
- Timeline of all steps (parameter changes + results)
- The LLM's final reasoning for the best config

## Parameters to Optimize

| Parameter | Type | Range | Impact |
|---|---|---|---|
| `--gpu-memory-utilization` | continuous | 0.3–0.8 | KV cache size → max concurrency |
| `--max-num-seqs` | discrete | 16–512 | Max concurrent requests |
| `--max-num-batched-tokens` | discrete | 4096–65536 | Prefill batch size |
| `--max-model-len` | discrete | 4096–131072 | Max context length |
| `--block-size` | discrete | 16, 32, 64, 128 | KV cache block size |
| `--kv-cache-dtype` | enum | auto, fp8 | KV cache memory |
| `--enable-chunked-prefill` | binary | on/off | Long-context prefill |
| `--enable-prefix-caching` | binary | on/off | Repeat prompt caching |
| `--enforce-eager` | binary | on/off | CUDA graphs vs eager |
| `--num-scheduler-steps` | discrete | 1–8 | Multi-step scheduling |
| `--swap-space` | continuous | 0–16 (GB) | CPU swap for KV cache |
| `--tensor-parallel-size` | discrete | 1–4 | GPU parallelism (hardware-limited) |

## Workload-Aware Benchmark

The current `benchmark.py` uses 3 fixed prompts. The optimization agent needs a
**workload simulator** that matches the user's actual usage:

```python
async def run_workload_benchmark(
    port: int,
    model_name: str,
    workload: dict,  # {concurrency, input_tokens, output_tokens, type}
) -> dict:
    """Simulate the user's workload and measure performance."""
    
    # Generate synthetic prompts matching input_tokens
    prompts = generate_synthetic_prompts(
        count=workload["concurrency"],
        token_count=workload["input_tokens"],
    )
    
    # Send all requests concurrently
    async with httpx.AsyncClient() as client:
        tasks = [
            _send_one(client, port, model_name, prompt, workload["output_tokens"])
            for prompt in prompts
        ]
        results = await asyncio.gather(*tasks)
    
    # Aggregate results
    return {
        "total_wall_time_s": ...,
        "total_tokens_generated": ...,
        "aggregate_throughput_tok_s": ...,  # total tokens / wall time
        "avg_ttft_ms": ...,
        "p50_ttft_ms": ...,
        "p95_ttft_ms": ...,
        "avg_tok_per_sec_per_request": ...,
        "concurrency": workload["concurrency"],
    }
```

Synthetic prompt generation: use repeated sentences to reach the target token count.
vLLM's tokenizer counts tokens, so we can verify the count via `stream_options.include_usage`.

## Composite Score

The score determines whether a parameter change is an improvement:

```python
def compute_score(benchmark: dict, workload: dict) -> float:
    """Higher = better."""
    if workload["priority"] == "throughput":
        # Maximize aggregate throughput
        return benchmark["aggregate_throughput_tok_s"]
    elif workload["priority"] == "latency":
        # Minimize TTFT (invert so higher = better)
        return 1000.0 / benchmark["p95_ttft_ms"]
    else:  # balanced
        return (
            0.5 * benchmark["aggregate_throughput_tok_s"] +
            0.5 * (1000.0 / benchmark["p95_ttft_ms"])
        )
```

## API Endpoints

```python
POST /api/{name}/optimize/start
    Body: {"message": "batch processing of images, 64 parallel requests each ~8000 tokens"}
    → Starts the interview phase. Returns run_id.

POST /api/{name}/optimize/chat
    Body: {"message": "around 500 tokens"}
    → Continue the interview. Agent may ask follow-ups or start optimization.

GET /api/{name}/optimize/status
    → {status, current_step, total_steps, best_score, baseline_score, ...}

GET /api/{name}/optimize/history
    → [{step_number, parameter, old_value, new_value, score, is_improvement, reasoning}, ...]

POST /api/{name}/optimize/stop
    → Stops the agent after current step completes.

GET /api/{name}/optimize/report
    → {baseline, best, improvement_pct, timeline, final_reasoning}
```

## Dashboard UI

### In the Edit modal, add:

```
┌─────────────────────────────────────────────────┐
│ Edit config — qwen9b                    [×]     │
│                                                  │
│ [existing edit fields...]                        │
│                                                  │
│ ─────────────────────────────────────────────── │
│                                                  │
│ ┌─────────────────────────────────────────────┐ │
│ │ 🔧 Parameter Optimization Agent             │ │
│ │                                              │ │
│ │ Status: Not started                          │ │
│ │                                              │ │
│ │ [Start Optimization]                         │ │
│ │                                              │
│ │ Or ask the agent:                            │ │
│ │ ┌─────────────────────────────────────────┐  │ │
│ │ │ What is this model needed for?          │  │ │
│ │ │ e.g. "batch processing, 64 parallel..." │  │ │
│ │ └─────────────────────────────────────────┘  │ │
│ │ [Send]                                       │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### When running, the panel shows:

```
┌─────────────────────────────────────────────────┐
│ 🔧 Parameter Optimization — qwen9b              │
│                                                  │
│ Workload: Batch, 64 concurrent, 8K in / 500 out │
│ Status: Running step 7/30                       │
│ Testing: max_num_batched_tokens 16384 → 32768   │
│                                                  │
│ Baseline: 1240 tok/s | Best: 1820 tok/s (+47%) │
│                                                  │
│ [████████░░░░░░░░░░] Step 7/30                  │
│                                                  │
│ Timeline:                                        │
│  #0 baseline              1240 tok/s             │
│  #1 gpu_mem_util 0.4→0.5  1310 tok/s ✓          │
│  #2 max_num_seqs 128→256  1190 tok/s ✗ reverted │
│  #3 chunked_prefill on    1380 tok/s ✓          │
│  #4 max_batched 8K→16K    1550 tok/s ✓          │
│  #5 prefix_cache on       1620 tok/s ✓          │
│  #6 kv_cache_dtype fp8    1820 tok/s ✓          │
│  #7 max_batched 16K→32K   testing...            │
│                                                  │
│ [Stop] [View Full Report]                       │
└─────────────────────────────────────────────────┘
```

## Safety & Constraints

1. **Parameter validation** — before applying a change, validate it's within safe bounds
2. **Memory limits** — don't exceed available RAM (check via planner before restart)
3. **Restart timeout** — if vLLM doesn't become healthy within 5 minutes, revert
4. **Max iterations** — default 30 (configurable)
5. **Time limit** — default 8 hours (overnight)
6. **Revert on failure** — if model fails to start, immediately revert to last known good config
7. **One change at a time** — never change multiple parameters simultaneously (clean attribution)
8. **No destructive changes** — don't delete weights, don't change the model itself, only serving params

## LLM Agent Prompt (DeepSeek)

The agent LLM receives a structured prompt at each iteration:

```
You are a vLLM performance optimization agent. Your goal is to find the optimal
serving parameters for the following workload:

Model: {hf_model_id}
Workload: {use_case_description}
Priority: {throughput|latency|balanced}

Current parameters:
  gpu_memory_utilization: 0.4
  max_num_seqs: 128
  max_num_batched_tokens: 16384
  ...

Current performance:
  Aggregate throughput: 1240 tok/s
  TTFT (p50): 340ms
  KV cache usage: 45%
  Queue depth: 12 waiting

Previous attempts (last 5):
  #5 prefix_cache on → 1620 tok/s ✓ (improved)
  #4 max_batched 8K→16K → 1550 tok/s ✓ (improved)
  #3 chunked_prefill on → 1380 tok/s ✓ (improved)
  #2 max_num_seqs 128→256 → 1190 tok/s ✗ (reverted, OOM risk)
  #1 gpu_mem_util 0.4→0.5 → 1310 tok/s ✓ (improved)

Research findings:
  - Qwen3.5-9B benefits from fp8 KV cache on GB10
  - max_num_batched_tokens should be >= 2x max input tokens for chunked prefill
  - block_size 32 is optimal for attention patterns

Based on the current metrics and history, suggest the next SINGLE parameter change
that is most likely to improve performance. Consider what hasn't been tried yet
and what the metrics suggest is the current bottleneck.

Respond as JSON:
{
  "parameter": "parameter_name",
  "new_value": "value",
  "reasoning": "why this change, what bottleneck it addresses"
}

If no more promising changes remain, respond:
{
  "parameter": null,
  "reasoning": "converged — no more improvements likely"
}
```

## Implementation Plan

### Phase 1: Database + Agent Core (backend)
- `llmeister/optimizer_db.py` — DB schema + queries for runs/steps
- `llmeister/optimizer.py` — agent logic (interview, research, optimization loop)
- `llmeister/workload_benchmark.py` — workload-aware benchmark

### Phase 2: API Endpoints
- `POST /api/{name}/optimize/start` — start interview
- `POST /api/{name}/optimize/chat` — continue interview
- `GET /api/{name}/optimize/status` — current status
- `GET /api/{name}/optimize/history` — all steps
- `POST /api/{name}/optimize/stop` — stop agent
- `GET /api/{name}/optimize/report` — final report

### Phase 3: Dashboard UI
- Optimization panel in the Edit modal
- Interview chat interface
- Progress display (current step, timeline)
- Report view

### Phase 4: Hardening
- Parameter validation (safe bounds)
- Restart timeout + auto-revert
- Max iterations / time limit
- Error recovery (model fails to start → revert)

## File Structure (new files)

```
llmeister/
├── optimizer_db.py          # DB schema + queries
├── optimizer.py             # agent logic (interview, research, loop)
├── workload_benchmark.py    # workload-aware benchmark
├── ...                      # (existing files unchanged)
```

## KISS Notes

- The agent is a single async task per model (not multi-threaded)
- Only one optimization run per model at a time
- The LLM (DeepSeek) drives parameter selection — no complex RL algorithm
- Hill climbing with LLM-guided exploration — simple, effective, explainable
- The benchmark is synthetic but matches the user's workload parameters
- All state is in SQLite — survives manager restarts
