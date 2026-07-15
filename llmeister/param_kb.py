"""Curated vLLM parameter knowledge base for the optimization agent.

Instead of relying on DeepSeek's training-data memory of vLLM flags, this module
provides a structured, curated knowledge base the agent reasons over. It covers
parameter ranges, categories, memory impact, and cross-parameter interactions.

Validate against the running vLLM by parsing `vllm serve --help` (future work).
"""

from __future__ import annotations

# Each entry: CLI flag (with dashes) -> metadata
PARAM_KB: dict[str, dict] = {
    "gpu-memory-utilization": {
        "category": "MEMORY",
        "type": "float",
        "range": [0.1, 0.95],
        "default": 0.9,
        "unit": "fraction of GPU memory",
        "memory_impact": "controls KV cache pool size; higher = more cache = more concurrency",
        "interacts_with": ["max-num-seqs", "max-model-len", "kv-cache-dtype"],
        "throughput_effect": "higher allows more concurrent seqs → more throughput",
        "latency_effect": "neutral (until OOM causes preemption)",
        "notes": "On unified-memory systems (GB10), this competes with OS/CPU memory.",
    },
    "max-num-seqs": {
        "category": "MEMORY",
        "type": "int",
        "range": [1, 1024],
        "default": 1024,
        "unit": "sequences",
        "memory_impact": "linear — each seq needs KV cache slots = max_model_len * bytes_per_token",
        "interacts_with": ["gpu-memory-utilization", "max-model-len", "block-size", "kv-cache-dtype"],
        "throughput_effect": "higher = more batching = more throughput, until KV cache exhausts",
        "latency_effect": "higher = more queueing = worse TTFT under load",
        "notes": "Should be >= expected concurrency for batch workloads.",
    },
    "max-model-len": {
        "category": "MEMORY",
        "type": "int",
        "range": [1024, 131072],
        "default": "model dependent",
        "unit": "tokens",
        "memory_impact": "linear — defines max KV cache slots per sequence",
        "interacts_with": ["max-num-seqs", "gpu-memory-utilization", "kv-cache-dtype"],
        "throughput_effect": "higher = supports longer prompts, but reduces max concurrency",
        "latency_effect": "neutral",
        "notes": "Set to the actual max prompt+output length needed. Oversizing wastes KV cache.",
    },
    "max-num-batched-tokens": {
        "category": "BATCHING",
        "type": "int",
        "range": [512, 65536],
        "default": 8192,
        "unit": "tokens per iteration",
        "memory_impact": "moderate — affects activation memory per iteration",
        "interacts_with": ["max-num-seqs", "max-model-len", "enable-chunked-prefill"],
        "throughput_effect": "higher = larger prefill chunks = better long-prompt throughput",
        "latency_effect": "higher = can delay decode steps = worse ITL for short prompts",
        "notes": "Should be >= max_model_len for single-shot prefill. With chunked prefill, smaller is OK.",
    },
    "block-size": {
        "category": "BATCHING",
        "type": "int",
        "range": [16, 128],
        "default": 16,
        "unit": "tokens",
        "memory_impact": "low — affects KV cache block granularity",
        "interacts_with": ["max-num-seqs", "enable-prefix-caching"],
        "throughput_effect": "larger = less overhead per block, but more waste on partial blocks",
        "latency_effect": "neutral",
        "notes": "32 or 64 often optimal. Must match paged-attention block size expectations.",
    },
    "kv-cache-dtype": {
        "category": "MEMORY",
        "type": "enum",
        "range": ["auto", "fp8", "turboquant_4bit_nc"],
        "default": "auto",
        "unit": "dtype",
        "memory_impact": "fp8 = 50% KV memory vs fp16; 4bit = 25%",
        "interacts_with": ["max-num-seqs", "max-model-len", "gpu-memory-utilization"],
        "throughput_effect": "lower precision = more cache = more concurrency",
        "latency_effect": "fp8 may add slight compute overhead on some kernels",
        "notes": "fp8 is widely supported on Blackwell. 4bit needs specialized kernels.",
    },
    "enable-chunked-prefill": {
        "category": "CACHING",
        "type": "bool",
        "range": [True, False],
        "default": True,
        "unit": "flag",
        "memory_impact": "low — affects scheduling, not allocation",
        "interacts_with": ["max-num-batched-tokens", "enable-prefix-caching"],
        "throughput_effect": "on = better mixed prefill+decode throughput",
        "latency_effect": "on = prevents long prompts from blocking decodes",
        "notes": "V1 enables this implicitly. With prefix caching, only first chunk benefits.",
    },
    "enable-prefix-caching": {
        "category": "CACHING",
        "type": "bool",
        "range": [True, False],
        "default": True,
        "unit": "flag",
        "memory_impact": "low — reuses existing KV blocks",
        "interacts_with": ["block-size", "enable-chunked-prefill"],
        "throughput_effect": "huge for shared-prefix workloads (system prompts, few-shot)",
        "latency_effect": "neutral or slightly better (cache hit skips prefill)",
        "notes": "Critical for document processing with repeated structure.",
    },
    "enforce-eager": {
        "category": "SCHEDULER",
        "type": "bool",
        "range": [True, False],
        "default": False,
        "unit": "flag",
        "memory_impact": "on = saves CUDA graph memory (~0.5-1 GiB)",
        "interacts_with": [],
        "throughput_effect": "off (default) = CUDA graphs = faster decode",
        "latency_effect": "off = slightly better latency due to graph reuse",
        "notes": "Turn on only if CUDA graph capture OOMs or for debugging.",
    },
    "num-scheduler-steps": {
        "category": "SCHEDULER",
        "type": "int",
        "range": [1, 8],
        "default": 1,
        "unit": "steps",
        "memory_impact": "low",
        "interacts_with": ["max-num-seqs"],
        "throughput_effect": "higher = less scheduler overhead = more decode throughput",
        "latency_effect": "higher = slightly worse responsiveness",
        "notes": "Multi-step scheduling trades latency for throughput.",
    },
    "tensor-parallel-size": {
        "category": "MEMORY",
        "type": "int",
        "range": [1, 4],
        "default": 1,
        "unit": "GPUs",
        "memory_impact": "splits model + KV across N GPUs",
        "interacts_with": ["gpu-memory-utilization"],
        "throughput_effect": "higher = more compute, but adds communication overhead",
        "latency_effect": "higher = communication overhead per step",
        "notes": "Hardware-limited. GB10 is single-GPU, so always 1 here.",
    },
}

# Cross-parameter interaction rules — natural-language descriptions the LLM reasons over.
INTERACTIONS: list[dict] = [
    {
        "params": ["max-num-seqs", "gpu-memory-utilization", "max-model-len"],
        "rule": (
            "max_num_seqs is bounded by available KV cache: "
            "KV_cache ≈ max_num_seqs × max_model_len × bytes_per_token. "
            "Increasing gpu_memory_utilization enlarges the KV pool. "
            "Increasing max_model_len shrinks how many seqs fit."
        ),
    },
    {
        "params": ["max-num-batched-tokens", "max-model-len", "enable-chunked-prefill"],
        "rule": (
            "max_num_batched_tokens should be ≥ max_model_len for single-shot prefill. "
            "With chunked-prefill enabled, smaller values are acceptable (prefill is chunked). "
            "Doubling max_num_batched_tokens roughly doubles activation memory per iteration."
        ),
    },
    {
        "params": ["kv-cache-dtype", "max-num-seqs"],
        "rule": (
            "Switching kv-cache-dtype from auto (fp16) to fp8 halves KV memory, "
            "effectively doubling the max_num_seqs that fit in the same gpu_memory_utilization budget."
        ),
    },
    {
        "params": ["enable-prefix-caching", "block-size"],
        "rule": (
            "Prefix caching works at block granularity. Smaller block-size = finer cache granularity "
            "= better hit rate, but more block metadata overhead."
        ),
    },
    {
        "params": ["enforce-eager", "gpu-memory-utilization"],
        "rule": (
            "CUDA graphs (enforce-eager=false) consume ~0.5-1 GiB. If gpu_memory_utilization is tight, "
            "enabling eager mode frees that memory for KV cache."
        ),
    },
]


def kb_summary() -> str:
    """Return a compact text summary of the parameter KB for LLM prompts."""
    lines = ["PARAMETER KNOWLEDGE BASE:"]
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
        kv_cache_dtype: 'auto' (fp16=2 bytes), 'fp8' (1 byte), 'turboquant_4bit_nc' (0.5 byte)
        num_layers: model layer count (Qwen3.5-9B ≈ 36, fallback estimate)
        hidden_dim: model hidden dimension (fallback)

    Returns:
        Estimated KV cache size in GiB.
    """
    bytes_per_token = {"auto": 2, "fp8": 1, "turboquant_4bit_nc": 0.5}.get(kv_cache_dtype, 2)
    # K and V each stored, so ×2; hidden_dim per token per layer
    bytes = max_num_seqs * max_model_len * num_layers * hidden_dim * 2 * bytes_per_token
    return bytes / (1024 ** 3)
