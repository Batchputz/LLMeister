"""Curated vLLM parameter knowledge base for the optimization agent.

Based on vLLM 0.25.1 `vllm serve` documentation — parameters organized by
config class (CacheConfig, SchedulerConfig, ModelConfig, CompilationConfig).

Sources:
  - https://docs.vllm.ai/en/stable/cli/serve/
  - https://docs.vllm.ai/en/stable/configuration/engine_args/
  - https://docs.vllm.ai/en/stable/configuration/optimization/
"""

from __future__ import annotations

# Each entry: CLI flag (with dashes) -> metadata
PARAM_KB: dict[str, dict] = {
    # ═══════ CacheConfig ═══════
    "gpu-memory-utilization": {
        "category": "MEMORY",
        "type": "float",
        "range": [0.1, 0.95],
        "default": 0.92,
        "unit": "fraction",
        "memory_impact": "directly controls KV cache pool size; higher = more cache = more concurrency",
        "interacts_with": ["max-num-seqs", "max-model-len", "kv-cache-dtype"],
        "throughput_effect": "higher allows more concurrent seqs → more throughput",
        "latency_effect": "neutral (until OOM causes preemption)",
        "notes": "Per-instance limit. On unified-memory systems (GB10), competes with OS/CPU memory.",
    },
    "kv-cache-memory-bytes": {
        "category": "MEMORY",
        "type": "str (human-readable)",
        "range": ["0", "full GPU memory"],
        "default": None,
        "unit": "bytes (e.g. 40G, 10GiB)",
        "memory_impact": "explicit KV cache size, bypasses gpu_memory_utilization",
        "interacts_with": ["max-num-seqs", "max-model-len"],
        "throughput_effect": "fine-grained control over cache budget",
        "latency_effect": "neutral",
        "notes": "When set, ignores gpu_memory_utilization. Use for precise memory budgeting.",
    },
    "block-size": {
        "category": "BATCHING",
        "type": "int",
        "range": [8, 128],
        "default": None,  # auto-determined
        "unit": "tokens",
        "memory_impact": "low — affects KV cache block granularity",
        "interacts_with": ["enable-prefix-caching"],
        "throughput_effect": "larger = less overhead per block, but more waste on partial blocks",
        "latency_effect": "neutral",
        "notes": "None = auto. 32 or 64 often optimal for throughput.",
    },
    "kv-cache-dtype": {
        "category": "MEMORY",
        "type": "enum",
        "range": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2", "fp8_inc", "fp8_ds_mla",
                  "fp8_per_token_head", "int4_per_token_head", "int8_per_token_head",
                  "nvfp4", "turboquant_3bit_nc", "turboquant_4bit_nc", "turboquant_k3v4_nc",
                  "turboquant_k8v4", "bfloat16", "float16"],
        "default": "auto",
        "unit": "dtype",
        "memory_impact": "fp8 = 50% memory vs fp16; nvfp4 = 25%; turboquant_4bit = 25%",
        "interacts_with": ["max-num-seqs", "max-model-len", "gpu-memory-utilization"],
        "throughput_effect": "lower precision = more KV cache = more concurrency",
        "latency_effect": "fp8 may add slight compute overhead on some kernels",
        "notes": "fp8 widely supported on Blackwell. 4bit needs specialized kernels (turboquant, nvfp4).",
    },
    "enable-prefix-caching": {
        "category": "CACHING",
        "type": "bool",
        "range": [True, False],
        "default": True,
        "unit": "flag",
        "memory_impact": "low — reuses existing KV blocks (no extra allocation)",
        "interacts_with": ["block-size", "enable-chunked-prefill", "prefix-caching-hash-algo"],
        "throughput_effect": "huge for shared-prefix workloads (system prompts, repeated structure)",
        "latency_effect": "slightly better (cache hit skips prefill computation)",
        "notes": "V1 enables this by default. Critical for document processing with repeated structure.",
    },
    "prefix-caching-hash-algo": {
        "category": "CACHING",
        "type": "enum",
        "range": ["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"],
        "default": "sha256",
        "unit": "algorithm",
        "memory_impact": "none",
        "interacts_with": ["enable-prefix-caching"],
        "throughput_effect": "xxhash = faster hash computation, slightly higher collision risk",
        "latency_effect": "faster hashing = lower cache lookup overhead",
        "notes": "xxhash requires optional package. sha256 is cryptographically secure.",
    },

    # ═══════ SchedulerConfig ═══════
    "max-num-seqs": {
        "category": "MEMORY",
        "type": "int",
        "range": [1, 2048],
        "default": 1024,
        "unit": "sequences",
        "memory_impact": "linear — each seq needs KV cache = max_model_len × bytes_per_token",
        "interacts_with": ["gpu-memory-utilization", "max-model-len", "block-size", "kv-cache-dtype"],
        "throughput_effect": "higher = more batching = more throughput, until KV cache exhausts",
        "latency_effect": "higher = more queueing = worse TTFT under load",
        "notes": "Should be >= expected concurrency for batch workloads. Too high → OOM.",
    },
    "max-num-batched-tokens": {
        "category": "BATCHING",
        "type": "int (human-readable: 1k, 1K, 25.6k)",
        "range": [512, 131072],
        "default": None,  # set by EngineArgs.create_engine_config
        "unit": "tokens per iteration",
        "memory_impact": "moderate — affects activation memory per iteration",
        "interacts_with": ["max-num-seqs", "max-model-len", "enable-chunked-prefill"],
        "throughput_effect": "higher = larger prefill chunks = better long-prompt throughput",
        "latency_effect": "higher = can delay decodes (long prefills block short decodes)",
        "notes": "Should be ≥ 2× max_model_len for single-shot prefill of long prompts.",
    },
    "enable-chunked-prefill": {
        "category": "SCHEDULER",
        "type": "bool",
        "range": [True, False],
        "default": True,
        "unit": "flag",
        "memory_impact": "low — affects scheduling, not allocation",
        "interacts_with": ["max-num-batched-tokens", "enable-prefix-caching", "long-prefill-token-threshold"],
        "throughput_effect": "on = better mixed prefill+decode throughput; prevents long prompts from hogging scheduler",
        "latency_effect": "on = prevents single long prompt from blocking all decodes (better P95 TTFT)",
        "notes": "V1 enables this implicitly. With prefix caching, only first chunk benefits from cache.",
    },
    "max-num-partial-prefills": {
        "category": "SCHEDULER",
        "type": "int",
        "range": [1, 16],
        "default": 1,
        "unit": "sequences",
        "memory_impact": "low",
        "interacts_with": ["enable-chunked-prefill", "max-num-batched-tokens"],
        "throughput_effect": "higher = more concurrent partial prefills = better scheduler utilization",
        "latency_effect": "higher helps when many long prompts arrive together",
        "notes": "Only relevant with chunked prefill. Increase for batch workloads with many concurrent long prompts.",
    },
    "long-prefill-token-threshold": {
        "category": "SCHEDULER",
        "type": "int",
        "range": [0, 131072],
        "default": 0,
        "unit": "tokens",
        "memory_impact": "none",
        "interacts_with": ["enable-chunked-prefill"],
        "throughput_effect": "prompts longer than this get chunked; 0 = chunk everything",
        "latency_effect": "lower threshold = more chunking = better latency at cost of throughput",
        "notes": "0 disables the cap (all prompts can be chunked).",
    },
    "scheduling-policy": {
        "category": "SCHEDULER",
        "type": "enum",
        "range": ["fcfs", "priority"],
        "default": "fcfs",
        "unit": "policy",
        "memory_impact": "none",
        "interacts_with": [],
        "throughput_effect": "fcfs is standard; priority allows preemption-based scheduling",
        "latency_effect": "priority can improve latency for high-priority requests",
        "notes": "Not typically tuned for throughput optimization.",
    },
    "watermark": {
        "category": "SCHEDULER",
        "type": "float",
        "range": [0.0, 1.0],
        "default": 0.0,
        "unit": "fraction",
        "memory_impact": "keeps KV cache blocks free for headroom",
        "interacts_with": ["gpu-memory-utilization", "max-num-seqs"],
        "throughput_effect": "0.0 = no headroom (max utilization); higher = safer but less throughput",
        "latency_effect": "higher watermark reduces KV cache thrashing → better latency stability",
        "notes": "Consider 0.05-0.10 if seeing repeated evictions under heavy load.",
    },
    "async-scheduling": {
        "category": "SCHEDULER",
        "type": "bool",
        "range": [True, False],
        "default": True,
        "unit": "flag",
        "memory_impact": "none",
        "interacts_with": [],
        "throughput_effect": "on = reduces GPU idle gaps → better throughput and latency",
        "latency_effect": "on = lower latency under variable load",
        "notes": "Keep enabled unless debugging scheduler issues.",
    },
    "stream-interval": {
        "category": "SCHEDULER",
        "type": "int",
        "range": [1, 64],
        "default": 1,
        "unit": "tokens",
        "memory_impact": "none",
        "interacts_with": [],
        "throughput_effect": "higher = less host overhead = more throughput; lower = smoother streaming",
        "latency_effect": "higher = coarser streaming (may feel less responsive)",
        "notes": "Only affects streaming responses. 1 = per-token, 10 = batch 10 tokens.",
    },

    # ═══════ ModelConfig ═══════
    "max-model-len": {
        "category": "MEMORY",
        "type": "int (human-readable: 1k, 1K, 25.6k, auto, -1)",
        "range": [1024, "model max"],
        "default": "auto (model's native context length)",
        "unit": "tokens",
        "memory_impact": "linear — each token of context needs KV cache per active sequence",
        "interacts_with": ["max-num-seqs", "gpu-memory-utilization", "kv-cache-dtype"],
        "throughput_effect": "higher = supports longer prompts, but trades off max concurrency",
        "latency_effect": "higher = longer prefill time per single request",
        "notes": "-1 or 'auto' = auto-fit to GPU memory. Supports 1k/1K/25.6k format.",
    },
    "enforce-eager": {
        "category": "MEMORY",
        "type": "bool",
        "range": [True, False],
        "default": False,
        "unit": "flag",
        "memory_impact": "on = disables CUDA graphs, saving ~0.5-1.5 GiB",
        "interacts_with": ["gpu-memory-utilization"],
        "throughput_effect": "on = no CUDA graphs = slower decode (10-20% penalty)",
        "latency_effect": "off = CUDA graphs = faster per-token latency",
        "notes": "Turn on only if CUDA graph capture OOMs or for debugging eager-mode issues.",
    },
    "enable-sleep-mode": {
        "category": "MEMORY",
        "type": "bool",
        "range": [True, False],
        "default": False,
        "unit": "flag",
        "memory_impact": "on = enables cumem allocator; allows weight offload during idle",
        "interacts_with": ["enable-cumem-allocator"],
        "throughput_effect": "neutral (affects idle state, not active serving)",
        "latency_effect": "adds wake-up latency when model resumes from sleep",
        "notes": "Only cuda/hip platforms. Automatically enables cumem allocator.",
    },
    "enable-cumem-allocator": {
        "category": "MEMORY",
        "type": "bool",
        "range": [True, False],
        "default": False,
        "unit": "flag",
        "memory_impact": "advanced GPU memory allocation (multi-node NVLink support)",
        "interacts_with": ["enable-sleep-mode"],
        "throughput_effect": "may improve memory efficiency on supported hardware",
        "latency_effect": "neutral",
        "notes": "Auto-enabled by sleep mode. Only cuda/hip. GB10 may benefit.",
    },
    "disable-cascade-attn": {
        "category": "MEMORY",
        "type": "bool",
        "range": [True, False],
        "default": True,  # cascade attention is OFF by default (must opt in)
        "unit": "flag",
        "memory_impact": "on = disables cascade attention = uses standard attention",
        "interacts_with": [],
        "throughput_effect": "cascade attention may improve performance for certain patterns",
        "latency_effect": "potential numerical improvements with cascade attention",
        "notes": "Set to False to enable cascade attention (V1 feature).",
    },

    # ═══════ CompilationConfig / Performance ═══════
    "optimization-level": {
        "category": "SCHEDULER",
        "type": "int (-O flag)",
        "range": [0, 3],
        "default": 2,
        "unit": "level",
        "memory_impact": "higher = more CUDA graphs = more memory for graph cache",
        "interacts_with": ["enforce-eager"],
        "throughput_effect": "-O2/-O3 = best performance; -O1 = faster startup; -O0 = debug",
        "latency_effect": "higher = better latency via compiled graphs",
        "notes": "-O2 is default and best for most cases. -O3 has longest startup.",
    },
    "performance-mode": {
        "category": "SCHEDULER",
        "type": "enum",
        "range": ["balanced", "interactivity", "throughput"],
        "default": "balanced",
        "unit": "mode",
        "memory_impact": "throughput mode uses larger CUDA graphs = more memory",
        "interacts_with": ["optimization-level", "enforce-eager"],
        "throughput_effect": "'throughput' = most aggressive batching, largest CUDA graphs",
        "latency_effect": "'interactivity' = fine-grained CUDA graphs, better per-request latency",
        "notes": "NEW in v0.25+. Use 'throughput' for batch workloads like document processing.",
    },

    # ═══════ ParallelConfig ═══════
    "tensor-parallel-size": {
        "category": "MEMORY",
        "type": "int",
        "range": [1, "GPU count"],
        "default": 1,
        "unit": "GPUs",
        "memory_impact": "splits model weights + KV across N GPUs (per-GPU memory ÷ N)",
        "interacts_with": ["gpu-memory-utilization"],
        "throughput_effect": "higher = more compute parallelism, but adds NCCL communication overhead",
        "latency_effect": "higher = inter-GPU communication adds latency per layer",
        "notes": "GB10 is single-GPU — always 1 here. Only relevant for multi-GPU systems.",
    },
}

# Cross-parameter interaction rules — natural-language descriptions the LLM reasons over.
INTERACTIONS: list[dict] = [
    {
        "params": ["max-num-seqs", "gpu-memory-utilization", "max-model-len", "kv-cache-dtype"],
        "rule": (
            "max_num_seqs is bounded by KV cache: "
            "KV_cache ≈ max_num_seqs × max_model_len × bytes_per_token × 2 (K+V) × layers × hidden_dim. "
            "Increasing gpu_memory_utilization enlarges the KV pool. "
            "Switching kv-cache-dtype to fp8 halves memory, allowing ~2× max_num_seqs. "
            "Reducing max_model_len frees KV slots for more concurrent seqs."
        ),
    },
    {
        "params": ["max-num-batched-tokens", "max-model-len", "enable-chunked-prefill"],
        "rule": (
            "max_num_batched_tokens should be ≥ 2× max_model_len for full single-shot prefill of long prompts. "
            "With chunked-prefill enabled, smaller values work (prefill is chunked). "
            "Doubling max_num_batched_tokens roughly doubles activation memory per iteration."
        ),
    },
    {
        "params": ["enable-prefix-caching", "block-size", "prefix-caching-hash-algo"],
        "rule": (
            "Prefix caching works at block granularity. Smaller block-size = finer cache granularity "
            "= better hit rate for short shared prefixes. xxhash is faster than sha256 but has "
            "theoretically higher collision risk."
        ),
    },
    {
        "params": ["enforce-eager", "gpu-memory-utilization", "optimization-level"],
        "rule": (
            "CUDA graphs (enforce-eager=false) consume ~0.5-1.5 GiB depending on optimization level "
            "and max_num_seqs. If gpu_memory_utilization is tight, enabling eager frees that memory "
            "for KV cache at ~10-20% decode throughput cost. Higher optimization levels use more "
            "CUDA graph memory."
        ),
    },
    {
        "params": ["performance-mode", "optimization-level", "max-num-seqs"],
        "rule": (
            "'throughput' mode uses larger CUDA graphs and more aggressive batching — beneficial "
            "for batch workloads with high concurrency. 'interactivity' uses fine-grained graphs "
            "for lower per-request latency. 'balanced' splits the difference. "
            "Larger max_num_seqs amplifies the difference between modes."
        ),
    },
    {
        "params": ["enable-sleep-mode", "enable-cumem-allocator", "gpu-memory-utilization"],
        "rule": (
            "Sleep mode automatically enables the cumem allocator which provides advanced GPU memory "
            "allocation. During sleep, model weights can be offloaded, freeing GPU memory for other "
            "workloads. Wake-up latency is the tradeoff."
        ),
    },
    {
        "params": ["max-num-partial-prefills", "enable-chunked-prefill", "max-num-batched-tokens"],
        "rule": (
            "With chunked prefill, max_num_partial_prefills controls how many long prompts can be "
            "partially prefilled concurrently. Higher values help when many long prompts arrive "
            "simultaneously but increase contention for the max_num_batched_tokens budget."
        ),
    },
    {
        "params": ["watermark", "gpu-memory-utilization", "max-num-seqs"],
        "rule": (
            "The watermark reserves a fraction of KV cache blocks as free headroom. "
            "At 0.0, all blocks can be allocated. At 0.10, 10% remain free. "
            "A non-zero watermark stabilizes latency under memory pressure but reduces "
            "effective KV cache capacity, requiring lower max_num_seqs or higher gpu_memory_utilization."
        ),
    },
]


def kb_summary() -> str:
    """Return a compact text summary of the parameter KB for LLM prompts."""
    lines = ["vLLM 0.25.1 PARAMETER KNOWLEDGE BASE:"]
    by_cat: dict[str, list[str]] = {}
    for flag, meta in PARAM_KB.items():
        by_cat.setdefault(meta["category"], []).append(
            f"  {flag} [{meta['type']}, range {meta['range']}, default {meta['default']}]\n"
            f"    memory: {meta['memory_impact']}\n"
            f"    throughput: {meta['throughput_effect']}\n"
            f"    interacts: {', '.join(meta['interacts_with']) or 'none'}"
        )
    for cat in sorted(by_cat):
        lines.append(f"\n{cat}:")
        lines.extend(by_cat[cat])
    lines.append("\nINTERACTION RULES:")
    for i, inter in enumerate(INTERACTIONS, 1):
        lines.append(f"  {i}. {' + '.join(inter['params'])}:")
        lines.append(f"     {inter['rule']}")
    return "\n".join(lines)


def estimate_kv_cache_gib(
    max_num_seqs: int,
    max_model_len: int,
    kv_cache_dtype: str,
    num_layers: int = 36,
    hidden_dim: int = 4096,
) -> float:
    """Rough KV cache estimate in GiB.

    Args:
        max_num_seqs: concurrent sequences
        max_model_len: max context length
        kv_cache_dtype: one of 'auto'/'bfloat16'/'float16'(2 bytes), 'fp8'/etc(1 byte),
                        'nvfp4'/'turboquant_4bit_nc'(0.5 byte)
        num_layers: model layer count (Qwen3.5-9B ≈ 36, fallback estimate)
        hidden_dim: model hidden dimension (fallback)

    Returns:
        Estimated KV cache size in GiB.
    """
    BYTES = {
        "auto": 2, "bfloat16": 2, "float16": 2,
        "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1, "fp8_inc": 1, "fp8_ds_mla": 1,
        "fp8_per_token_head": 1, "int4_per_token_head": 0.5, "int8_per_token_head": 1,
        "nvfp4": 0.5, "turboquant_3bit_nc": 0.375, "turboquant_4bit_nc": 0.5,
        "turboquant_k3v4_nc": 0.4375, "turboquant_k8v4": 1.5,
    }
    bytes_per_token = BYTES.get(kv_cache_dtype, 2)
    # K and V each stored, so ×2; hidden_dim per token per layer
    bytes = max_num_seqs * max_model_len * num_layers * hidden_dim * 2 * bytes_per_token
    return bytes / (1024 ** 3)
