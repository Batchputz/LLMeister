# Optimizer Improvement Plan — From Reactive Tweaker to Strategic Planner

## Problem

The current optimizer is **reactive**: each iteration asks DeepSeek "given the current
state, suggest one change." It lacks the context a human expert would gather before
touching a single knob:

| Human expert does | Current optimizer |
|---|---|
| 1. Checks the GPU (128 GB shared, ARM64 GB10) | ❌ No system context in prompts |
| 2. Lists all vLLM parameters + their ranges | ❌ Only sees current values |
| 3. Researches what each parameter means + cross-interactions | ❌ Raw Perplexity text, no synthesis |
| 4. Researches model-specific optimal configs | ⚠️ Generic queries, no GPU context |
| 5. **Makes a branching plan** (if X works → Y, else → Z) | ❌ No planning phase at all |
| 6. Executes the plan step by step | ⚠️ Reactive loop, no plan awareness |

Result: DeepSeek hammers `max_num_seqs` three times in a row (different values, same
dead-end) because it has no memory of the strategy, only the last attempt.

## Target Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Improved Optimization Agent                     │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ Interview │→ │ System   │→ │ Research │→ │ Parameter    │    │
│  │ (workload)│  │ Analysis │  │ (enriched)│  │ Inventory    │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────┬───────┘    │
│                                                    │            │
│                              ┌─────────────────────▼──────────┐ │
│                              │  PLANNING PHASE (NEW)           │ │
│                              │  DeepSeek synthesizes all ctx  │ │
│                              │  into a branching plan:        │ │
│                              │  1. Try X (high-value, low-risk)│ │
│                              │     if works → try Y           │ │
│                              │     if fails → try Z           │ │
│                              │  2. Try W ...                   │ │
│                              └─────────────┬──────────────────┘ │
│                                            │                    │
│  ┌──────────┐  ┌──────────┐  ┌────────────▼─────────────────┐ │
│  │ Baseline │→ │ Execute  │→ │ Report                        │ │
│  │ Benchmark│  │ Plan     │  │ (plan vs actual journey)      │ │
│  └──────────┘  │ (plan-   │  └───────────────────────────────┘ │
│                │  aware)  │                                    │
│                └──────────┘                                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Work Packages

### WP1 — System Analysis Phase (NEW)

**Goal:** Give DeepSeek the same hardware context a human would read off `nvidia-smi`.

**What exists:** `manager.py` already collects `_sys_info` (OS, GPU name, CUDA driver,
vLLM version, arch, Docker version) and live memory/CPU/GPU utilization via `/api/system`.

**What to build:**
- New method `OptimizationAgent._gather_system_context()` that:
  - Calls `/api/system` (already exists) for live specs
  - Queries `nvidia-smi --query-gpu=memory.total,memory.free,name,driver_version`
    for exact VRAM (the GB10 has 128 GB **unified** memory, not discrete VRAM)
  - Reads the model's measured memory from the DB (`measured_max_memory_mb`,
    `measured_weight_memory_mb`)
  - Computes the **memory budget**: `total - os_reserve - other_models - weights`
- Store as `self.system_context` dict and persist to `optimization_runs.system_json`
- Feed into every DeepSeek prompt and every Perplexity query

**Schema change:** Add `system_json TEXT` column to `optimization_runs`.

**Example context block in prompts:**
```
SYSTEM CONTEXT:
  GPU: NVIDIA GB10 Grace Blackwell (ARM64)
  Total memory: 121.6 GiB unified (CPU+GPU shared)
  vLLM version: 0.25.1.dev24
  Model weights: 8.14 GiB (INT4 AutoRound)
  Current KV cache: 38.4 GiB @ gpu_mem_util=0.4
  Available for KV cache: ~83 GiB headroom at gpu_mem_util=0.85
  Other running models: none
```

---

### WP2 — Parameter Knowledge Base (NEW)

**Goal:** Stop relying on DeepSeek's training-data memory of vLLM flags. Build a
curated, version-aware knowledge base the agent reasons over.

**What to build:**
- New file `llmeister/param_kb.py` — a static, curated dict of vLLM serve parameters:
  ```python
  PARAM_KB = {
      "max-num-seqs": {
          "category": "MEMORY",
          "type": "int",
          "range": [1, 1024],
          "default": 1024,
          "unit": "sequences",
          "memory_impact": "linear (each seq needs KV cache slots)",
          "interacts_with": ["gpu_memory_utilization", "max_model_len", "block_size"],
          "description": "Max concurrent sequences per iteration. ...",
          "throughput_effect": "higher = more batching = more throughput, until OOM",
          "latency_effect": "higher = more queueing = worse TTFT",
      },
      "gpu-memory-utilization": { ... },
      "max-num-batched-tokens": { ... },
      "enable-prefix-caching": { ... },
      ...
  }
  ```
- Cover the 12 parameters from `OPTIMIZER_AGENT.md` plus the key flags already in the
  command (`--kv-cache-dtype`, `--enable-chunked-prefill`, `--enable-sleep-mode`, etc.)
- Include **interaction rules** as a separate structure:
  ```python
  INTERACTIONS = [
      {"params": ["max_num_seqs", "gpu_memory_utilization"],
       "rule": "max_num_seqs is bounded by KV cache size, which scales with gpu_memory_utilization"},
      {"params": ["max_num_batched_tokens", "max_model_len"],
       "rule": "max_num_batched_tokens should be >= max_model_len for single-shot prefill"},
      ...
  ]
  ```
- **Validate against the running vLLM** by parsing `vllm serve --help` inside the
  container once (cache the result). This catches version-specific flag renames.

**Why curated, not LLM-generated:** vLLM parameter semantics are stable and few (~15).
A curated KB is more reliable than asking DeepSeek to recall flag names from training
data, and it lets us compute memory impact deterministically.

---

### WP3 — Enriched Research Phase (IMPROVE)

**Goal:** Make Perplexity queries context-aware instead of generic.

**Current queries (generic):**
```
1. "vLLM {hf_id} optimization parameters performance tuning"
2. "vLLM max_num_seqs max_num_batched_tokens tuning batch"
3. "vLLM chunked prefill prefix caching KV cache optimization"
```

**Improved queries (context-aware):**
```
1. "vLLM {hf_id} on {gpu_name} ({vram}GB) optimal serving config {vllm_version}"
2. "vLLM {vllm_version} parameter interactions max_num_seqs gpu_memory_utilization max_model_len"
3. "vLLM {hf_id} known issues OOM KV cache ARM64 Grace Blackwell"
4. "vLLM {hf_id} {workload_type} {concurrency} concurrent {input_tokens} token throughput"
```

**What to build:**
- Pass `system_context` (from WP1) into every query
- Pass the workload spec into query 4
- Add a 4th query for known issues / gotchas on this specific GPU+model combo
- Store raw results in `research_json` (already exists)
- **Synthesize** research into structured notes via a DeepSeek call (NEW):
  ```
  Given these research results, extract:
  - Recommended starting parameters for this GPU+model+workload
  - Parameters to AVOID and why
  - Known interactions to be aware of
  - Suggested experiment ordering (which to try first)
  ```

---

### WP4 — Planning Phase (NEW, the core improvement)

**Goal:** Before touching any knob, DeepSeek creates a **branching optimization plan**
that the execution loop follows.

**What to build:**
- New method `OptimizationAgent._create_plan()` that calls DeepSeek ONCE with:
  - System context (WP1)
  - Parameter knowledge base (WP2)
  - Synthesized research (WP3)
  - Current config + baseline benchmark
  - The workload spec
- Ask DeepSeek to output a structured plan:
  ```json
  {
    "goal": "Maximize throughput for 64-concurrent 8K-token batch workload",
    "experiments": [
      {
        "id": 1,
        "hypothesis": "max_num_seqs=8 is the bottleneck; workload needs 64",
        "change": {"max_num_seqs": 64},
        "expected": "throughput ~2x; memory OK at 0.4 util",
        "on_success": [2, 3],
        "on_failure": [4],
        "risk": "low",
        "rationale": "..."
      },
      {
        "id": 2,
        "hypothesis": "With 64 seqs working, larger batch tokens help prefill",
        "change": {"max_num_batched_tokens": 24576},
        "expected": "+10-20% throughput",
        "on_success": [],
        "on_failure": [],
        "risk": "medium (may OOM)",
        "rationale": "..."
      },
      ...
    ]
  }
  ```
- Store the plan in `optimization_runs.plan_json` (NEW column)
- **Execution loop follows the plan**: track current experiment, branch on success/failure
- DeepSeek can still deviate if benchmark data contradicts the plan, but it must explain why

**Schema change:** Add `plan_json TEXT` column to `optimization_runs`.

**Why this is the biggest win:** The current loop has no memory of strategy — it reacts
to the last step. A plan gives DeepSeek forward context: "we're on experiment 2 of 5,
experiment 1 succeeded so we're now exploring the success branch." This is what stops
the hammer-one-parameter behavior.

---

### WP5 — Plan-Aware Execution Loop (IMPROVE)

**Goal:** The optimization loop follows the plan instead of free-associating.

**Current loop:**
```
for i in range(1, MAX_ITERATIONS):
    suggestion = ask_deepseek("what's next?")  # no plan context
    apply, restart, benchmark, keep/revert
```

**Improved loop:**
```
plan = load_plan()
current_experiments = [plan.experiments[0]]  # start with first
for i in range(1, MAX_ITERATIONS):
    exp = current_experiments.pop(0)
    apply(exp.change)
    restart, benchmark
    if improved:
        branch_to(exp.on_success)
    else:
        branch_to(exp.on_failure)
    # DeepSeek reviews: should we continue the plan or deviate?
    if deepseek_review_says_deviate(benchmark, plan):
        new_exp = deepseek_proposes_deviation(...)
        current_experiments.insert(0, new_exp)
```

**What to build:**
- Track `current_experiment_id` in the run state
- Each step records which plan experiment it executed
- Add a `plan_deviation` field to `optimization_steps` when DeepSeek overrides the plan
- Keep the existing loop detection + blacklist as safety nets

---

### WP6 — Memory-Aware Pre-Check (NEW)

**Goal:** Stop wasting 5-minute restart cycles on changes that will obviously OOM.

**What to build:**
- Before applying a config change, estimate the memory impact:
  ```python
  def estimate_kv_cache(config, system_context) -> float:
      """Rough KV cache size in GiB."""
      seqs = config.max_num_seqs
      model_len = config.max_model_len
      block_size = config.block_size
      # bytes per token depends on kv_cache_dtype (fp8=1, fp16=2)
      bytes_per_token = 1 if config.kv_cache_dtype == "fp8" else 2
      # rough: layers * 2 (k+v) * hidden_dim * bytes_per_token
      # use model's measured_weight_memory as proxy for size
      return seqs * model_len * bytes_per_token * ESTIMATED_LAYERS / 1e9
  ```
- If `estimated_kv_cache + weights > available_memory * gpu_mem_util`:
  - Skip the restart, mark step as "rejected (would OOM)", suggest alternative
  - Tell DeepSeek "this change would require X GiB but only Y is available"
- This alone would have saved 3 wasted 5-minute cycles in the last run

---

### WP7 — Richer Reporting (IMPROVE)

**Goal:** Show the human the full journey, not just a step list.

**What to build:**
- Extend the report to show:
  - The original plan vs what actually happened
  - Which experiments succeeded/failed/were skipped
  - The memory budget analysis
  - A "what we learned" summary generated by DeepSeek
- New endpoint `GET /api/{name}/optimize/report` (already specced but not built)
- Dashboard: show the plan tree alongside the timeline

---

## Implementation Order

| Priority | WP | Effort | Impact |
|---|---|---|---|
| 1 | WP4 — Planning Phase | M | **Huge** — stops reactive hammering |
| 2 | WP1 — System Analysis | S | High — gives DeepSeek hardware context |
| 3 | WP6 — Memory Pre-Check | S | High — saves wasted 5-min cycles |
| 4 | WP2 — Parameter KB | M | Medium — reliable param knowledge |
| 5 | WP3 — Enriched Research | S | Medium — better starting point |
| 6 | WP5 — Plan-Aware Loop | M | Medium — follows the plan |
| 7 | WP7 — Richer Reporting | S | Low — nice to have |

**Recommended first iteration:** WP1 + WP4 + WP6. These three together transform the
optimizer from "reactive tweaker" to "strategic planner" with the smallest code change.

---

## Schema Changes Summary

```sql
ALTER TABLE optimization_runs ADD COLUMN system_json TEXT;   -- WP1
ALTER TABLE optimization_runs ADD COLUMN plan_json TEXT;     -- WP4
ALTER TABLE optimization_runs ADD COLUMN research_notes TEXT; -- WP3 (synthesized)

ALTER TABLE optimization_steps ADD COLUMN experiment_id INTEGER; -- WP5
ALTER TABLE optimization_steps ADD COLUMN plan_deviation INTEGER DEFAULT 0; -- WP5
ALTER TABLE optimization_steps ADD COLUMN memory_estimate TEXT;  -- WP6
```

## New Files

```
llmeister/
├── param_kb.py          # WP2 — curated vLLM parameter knowledge base
└── optimizer.py         # modified — new phases, plan-aware loop
```

## Risk & Notes

- **Planning phase adds one DeepSeek call** (~5s) before the loop starts — negligible
- **Memory pre-check is heuristic** — may occasionally reject viable configs. Mitigation:
  show the estimate in the step reasoning so DeepSeek can override with justification.
- **Parameter KB needs maintenance** as vLLM evolves. Mitigation: validate against
  `vllm serve --help` at runtime and warn on unknown params.
- **The plan is a guide, not a prison** — DeepSeek can always deviate. The plan just
  gives it forward context it currently lacks.
