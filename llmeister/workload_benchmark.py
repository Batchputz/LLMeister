"""LLMeister workload benchmark — simulate the user's actual workload.

Unlike the fixed 3-prompt benchmark, this generates synthetic prompts matching
the user's described usage pattern (concurrency, input/output tokens) and
measures aggregate throughput, TTFT percentiles, and per-request stats.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

# A pool of varied sentences for synthetic prompt generation.
_SENTENCES = [
    "The rapid advancement of machine learning has transformed how we process data.",
    "Quantum computing promises exponential speedups for certain classes of problems.",
    "Distributed systems require careful coordination to maintain consistency.",
    "Neural networks learn hierarchical representations from raw input data.",
    "The transformer architecture revolutionized natural language processing.",
    "Optimization algorithms seek minima in high-dimensional parameter spaces.",
    "Cloud infrastructure enables elastic scaling of computational resources.",
    "Information theory provides the mathematical foundation for data compression.",
    "Graph algorithms solve problems involving networks and relationships.",
    "Statistical inference draws conclusions from sampled data with uncertainty.",
    "Reinforcement learning agents improve through trial and error interaction.",
    "Computer vision systems extract meaning from pixels and video frames.",
    "Natural language understanding bridges the gap between text and meaning.",
    "Memory hierarchies trade speed for capacity across multiple levels.",
    "Parallel computing harnesses multiple processors to reduce wall time.",
    "Encryption ensures confidentiality of data in transit and at rest.",
]


def _generate_prompt(token_target: int) -> str:
    """Generate a synthetic prompt of approximately token_target tokens."""
    words = []
    target_words = int(token_target * 0.75)  # ~0.75 words per token
    i = 0
    while len(words) < target_words:
        words.append(_SENTENCES[i % len(_SENTENCES)])
        i += 1
    return " ".join(words)


async def run_workload_benchmark(
    port: int,
    model_name: str,
    workload: dict[str, Any],
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Run a workload-aware benchmark.

    Args:
        port: vLLM server port.
        model_name: served model name for the request.
        workload: {concurrency, input_tokens, output_tokens, type, priority}
        timeout: total timeout for all requests.

    Returns:
        {total_wall_time_s, total_tokens, aggregate_tok_s, avg_ttft_ms,
         p50_ttft_ms, p95_ttft_ms, avg_tok_per_sec, concurrency, errors}
    """
    concurrency = workload.get("concurrency", 1)
    input_tokens = workload.get("input_tokens", 500)
    output_tokens = workload.get("output_tokens", 128)

    prompts = [_generate_prompt(input_tokens) for _ in range(concurrency)]
    base_url = f"http://127.0.0.1:{port}/v1/chat/completions"

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        tasks = [
            _send_one(client, base_url, model_name, prompt, output_tokens)
            for prompt in prompts
        ]
        t_start = time.monotonic()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_time = time.monotonic() - t_start

    # Aggregate
    ttfts = []
    tok_per_sec_list = []
    total_tokens = 0
    errors = 0

    for r in results:
        if isinstance(r, Exception):
            errors += 1
            continue
        if r.get("error"):
            errors += 1
            continue
        ttfts.append(r["ttft_ms"])
        tok_per_sec_list.append(r["tok_per_sec"])
        total_tokens += r.get("completion_tokens", 0)

    ttfts.sort()
    n = len(ttfts)
    p50 = ttfts[n // 2] if n > 0 else 0
    p95 = ttfts[int(n * 0.95)] if n > 0 else 0
    avg_ttft = sum(ttfts) / n if n > 0 else 0
    avg_tok_s = sum(tok_per_sec_list) / n if n > 0 else 0
    aggregate = total_tokens / wall_time if wall_time > 0 else 0

    return {
        "total_wall_time_s": round(wall_time, 2),
        "total_tokens": total_tokens,
        "aggregate_tok_s": round(aggregate, 1),
        "avg_ttft_ms": round(avg_ttft, 1),
        "p50_ttft_ms": round(p50, 1),
        "p95_ttft_ms": round(p95, 1),
        "avg_tok_per_sec": round(avg_tok_s, 1),
        "concurrency": concurrency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "errors": errors,
    }


async def _send_one(
    client: httpx.AsyncClient, base_url: str, model_name: str, prompt: str, max_tokens: int
) -> dict[str, Any]:
    """Send a single request and measure timing."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.monotonic()
    ttft_ms = 0.0
    first_token = True
    completion_tokens = 0

    async with client.stream("POST", base_url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if "usage" in chunk and chunk["usage"]:
                completion_tokens = chunk["usage"].get("completion_tokens", 0)

            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                token = delta.get("content") or delta.get("reasoning") or ""
                if token and first_token:
                    ttft_ms = (time.monotonic() - t_start) * 1000
                    first_token = False

    e2e_ms = (time.monotonic() - t_start) * 1000
    gen_ms = e2e_ms - ttft_ms
    tok_per_sec = completion_tokens / (gen_ms / 1000) if gen_ms > 0 and completion_tokens > 0 else 0

    return {
        "ttft_ms": round(ttft_ms, 1),
        "e2e_ms": round(e2e_ms, 1),
        "completion_tokens": completion_tokens,
        "tok_per_sec": round(tok_per_sec, 1),
    }


def compute_score(benchmark: dict[str, Any], workload: dict[str, Any]) -> float:
    """Compute a composite score. Higher = better."""
    priority = workload.get("priority", "balanced")
    if priority == "throughput":
        return benchmark.get("aggregate_tok_s", 0)
    elif priority == "latency":
        p95 = benchmark.get("p95_ttft_ms", 1)
        return 1000.0 / p95 if p95 > 0 else 0
    else:  # balanced
        agg = benchmark.get("aggregate_tok_s", 0)
        p95 = benchmark.get("p95_ttft_ms", 1)
        return 0.5 * agg + 0.5 * (1000.0 / p95 if p95 > 0 else 0)
