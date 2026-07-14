"""LLMeister benchmark — standard prompts + streaming timing against vLLM.

No external tools. Sends 3 standard prompts via OpenAI-compatible streaming,
measures TTFT / E2E / tokens/sec client-side using httpx.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

# ── standard prompts ──────────────────────────────────────────────────────

LONG_PREFILL = (
    "The history of computing spans thousands of years, beginning with simple "
    "counting devices like the abacus, invented in ancient Mesopotamia around "
    "2400 BCE. The abacus allowed merchants and traders to perform arithmetic "
    "calculations rapidly using beads sliding on rods, and variants of it spread "
    "across China, Japan, Russia, and the Mediterranean world over the following "
    "millennia. In the 17th century, Blaise Pascal invented the Pascaline, a "
    "mechanical calculator capable of addition and subtraction, while Gottfried "
    "Wilhelm Leibniz later built the stepped reckoner that could multiply and "
    "divide. The 19th century saw Charles Babbage conceptualize the Difference "
    "Engine and the Analytical Engine — mechanical computers designed to compute "
    "polynomial functions and execute arbitrary sequences of punched-card "
    "instructions. Ada Lovelace, working with Babbage, wrote what is considered "
    "the first computer algorithm, envisioning machines that could manipulate "
    "symbols beyond mere numbers. The early 20th century brought electromechanical "
    "computing with Konrad Zuse's Z3 in Germany and Howard Aiken's Harvard Mark I "
    "in the United States. World War II accelerated development dramatically: "
    "Alan Turing's bombe helped crack the Enigma code at Bletchley Park, while "
    "the ENIAC at the University of Pennsylvania performed ballistic trajectory "
    "calculations using 17,468 vacuum tubes. The invention of the transistor at "
    "Bell Labs in 1947 and the integrated circuit by Jack Kilby and Robert Noyce "
    "in 1958-59 miniaturized computing from room-sized machines to desktop "
    "devices. The microprocessor revolution of the 1970s — led by Intel's 4004 "
    "and 8080, the MOS Technology 6502, and the Motorola 68000 — put computing "
    "power into homes with the Apple II, Commodore 64, and IBM PC. The 1990s "
    "connected the world through the World Wide Web, invented by Tim Berners-Lee "
    "at CERN, transforming computers from isolated productivity tools into "
    "globally networked communication platforms. The 21st century brought "
    "smartphones, cloud computing, and the rise of artificial intelligence "
    "powered by deep neural networks running on specialized GPU hardware."
    "\n\nSummarize this history in exactly three bullet points."
)

BENCH_PROMPTS: list[dict[str, Any]] = [
    {
        "label": "short",
        "desc": "Short prefill → measures TTFT + base decode speed",
        "messages": [{"role": "user", "content": "Explain what a large language model is in exactly one sentence."}],
        "max_tokens": 64,
    },
    {
        "label": "long",
        "desc": "Long prefill (~500 words) → measures prompt processing throughput",
        "messages": [{"role": "user", "content": LONG_PREFILL}],
        "max_tokens": 128,
    },
    {
        "label": "code",
        "desc": "Code generation → realistic mixed workload",
        "messages": [{"role": "user", "content": (
            "Write a Python function that finds the longest palindromic substring "
            "in a given string. Include a brief explanation as comments."
        )}],
        "max_tokens": 256,
    },
]


# ── runner ─────────────────────────────────────────────────────────────────

async def run_benchmark(port: int, model_name: str, timeout: float = 120.0) -> dict[str, Any]:
    """Run all three benchmark prompts against a vLLM instance.

    Args:
        port: vLLM server port (e.g. 8001).
        model_name: served model name to pass in the request.
        timeout: per-request HTTP timeout in seconds.

    Returns:
        {"model": str, "results": [...], "summary": {...}}
    """
    base_url = f"http://127.0.0.1:{port}/v1/chat/completions"
    results: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        for bp in BENCH_PROMPTS:
            result = await _bench_one(client, base_url, model_name, bp)
            results.append(result)

    # summary: compute weighted averages
    total_prompt = sum(r["prompt_tokens"] for r in results)
    total_completion = sum(r["completion_tokens"] for r in results)
    total_gen_ms = sum(r["e2e_ms"] - r["ttft_ms"] for r in results)
    avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
    avg_tok_s = total_completion / (total_gen_ms / 1000) if total_gen_ms > 0 else 0

    return {
        "model": model_name,
        "results": results,
        "summary": {
            "avg_ttft_ms": round(avg_ttft, 1),
            "avg_tok_per_sec": round(avg_tok_s, 1),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
        },
    }


async def _bench_one(
    client: httpx.AsyncClient, base_url: str, model_name: str, bp: dict[str, Any]
) -> dict[str, Any]:
    """Run a single benchmark prompt and measure timing."""
    payload = {
        "model": model_name,
        "messages": bp["messages"],
        "max_tokens": bp["max_tokens"],
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    t_start = time.monotonic()
    ttft_ms: float = 0.0
    completion_tokens = 0
    prompt_tokens = 0
    first_token = True

    async with client.stream("POST", base_url, json=payload) as resp:
        resp.raise_for_status()
        async for line in resp.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            # extract usage from the final chunk (stream_options include_usage)
            if "usage" in chunk and chunk["usage"]:
                prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
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
    tok_per_sec = completion_tokens / (gen_ms / 1000) if gen_ms > 0 and completion_tokens > 0 else 0.0
    prefill_tok_s = prompt_tokens / (ttft_ms / 1000) if ttft_ms > 0 and prompt_tokens > 0 else 0.0

    return {
        "label": bp["label"],
        "desc": bp["desc"],
        "ttft_ms": round(ttft_ms, 1),
        "e2e_ms": round(e2e_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "tok_per_sec": round(tok_per_sec, 1),
        "prefill_tok_per_sec": round(prefill_tok_s, 1),
    }
