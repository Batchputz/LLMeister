"""New-model research agent: two-stage discovery of optimal vLLM config.

Stage 1 — Perplexity (`sonar-pro`): online search for the HuggingFace model
    (architecture, params, quantization options, vLLM flags, max context, GB10/
    aarch64 caveats, NVFP4 availability, known issues).
Stage 2 — DeepSeek (`deepseek-chat`): synthesizes the search results into a
    candidate vLLM config (JSON) tuned for the DGX Spark GB10.

Keys come from env vars named in config.yaml (perplexity_key_env / deepseek_key_env).
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import yaml

import config

log = logging.getLogger("research")

PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

SEARCH_PROMPT = """\
Research the HuggingFace model "{model_id}" for serving with vLLM on an NVIDIA DGX Spark
(GB10 Grace Blackwell, aarch64/ARM64, single GPU, 128 GB unified CPU/GPU memory, no separate VRAM).
Find:
1. Architecture: is it a standard transformer, Mamba/hybrid (Mamba2+attention), or MoE? Param count.
2. Available quantizations on HuggingFace (especially NVFP4, INT4/AutoRound, FP8, GGUF) — list repo ids.
3. Recommended vLLM serve flags: --quantization, --max-model-len, --kv-cache-dtype, --chat-template,
   --reasoning-parser, --tool-call-parser, --enable-chunked-prefill, --enable-prefix-caching.
4. Known vLLM issues for this model on aarch64/GB10 or with Mamba/hybrid architectures.
5. The model's context window and whether it needs a specific chat template.
6. Whether it is gated (requires license acceptance) on HuggingFace.
Be specific and cite repos. If the exact model isn't found, suggest the closest NVFP4 or INT4 variant
suitable for the GB10.
"""

SYNTH_PROMPT = """\
You are configuring vLLM to serve a model on an NVIDIA DGX Spark (GB10 Grace Blackwell, aarch64,
single GPU, 128 GB unified memory). Using the research below, produce a candidate vLLM config as JSON.

Hard constraints for this hardware/stack:
- The image is vLLM 0.25.1.dev24 on aarch64. Always include `--enable-sleep-mode` (the stack uses sleep/wake hot-swap).
- For Mamba/hybrid models, include the mod "mods/fix-sleep-wake-mamba" (a wake bug fix).
- gpu_memory_utilization: pick a value that fits alongside one other model in 128 GB (typically 0.25-0.45).
- max-model-len: the model's native context or a safe subset (8192-65536) that fits the memory budget.
- Use {{port}} and {{host}} and {{gpu_memory_utilization}} and {{max_model_len}} and {{tensor_parallel}} as placeholders
  in the command (they are filled at launch). Keep {{port}} etc. as literal braced placeholders.
- Prefer NVFP4 or INT4 quantization for GB10 if available; otherwise the model's native quantization.
- served_model_names: include the short canonical id and a few aliases.

Return ONLY a JSON object with this exact shape:
{{
  "hf_model_id": "<repo id to load>",
  "display_name": "<short canonical name>",
  "served_model_names": ["<name1>", "<alias1>"],
  "aliases": ["<alias2>"],
  "quantization": "<vllm --quantization value or null>",
  "max_model_len": <int>,
  "gpu_memory_utilization": <float>,
  "kv_cache_dtype": "<value or null>",
  "chat_template": "<name or null>",
  "reasoning_parser": "<name or null>",
  "tool_call_parser": "<name or null>",
  "is_mamba_hybrid": <bool>,
  "mods": ["mods/fix-sleep-wake-mamba"],
  "estimated_memory_gb": <float>,
  "gated": <bool>,
  "notes": "<one-line caveat summary>",
  "command": "vllm serve <hf_model_id> \\\\n    --host {{host}} \\\\n    --port {{port}} \\\\n    ... --enable-sleep-mode \\\\n    -tp {{tensor_parallel}}"
}}

The "command" must be a complete `vllm serve` invocation with {{placeholder}} tokens for
host, port, gpu_memory_utilization, max_model_len, tensor_parallel (braced exactly like that),
including --enable-sleep-mode and any model-specific flags, but NOT --gpu-memory-utilization
or --max-model-len as literals (use the placeholders). End with `-tp {{tensor_parallel}}`.

RESEARCH:
{search}
"""


def _key(env_name: str) -> str | None:
    return os.environ.get(env_name)


def _perplexity_search(model_id: str) -> tuple[str, list[str]]:
    cfg = config.load().get("research", {})
    key = _key(cfg.get("perplexity_key_env", "PERPLEXITY_API_KEY"))
    if not key:
        raise RuntimeError("PERPLEXITY_API_KEY not set in env")
    model = cfg.get("perplexity_model", "sonar-pro")
    body = {
        "model": model,
        "messages": [{"role": "user", "content": SEARCH_PROMPT.format(model_id=model_id)}],
        "max_tokens": 2500,
    }
    r = httpx.post(PERPLEXITY_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=60)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    citations = data.get("citations") or []
    return content, citations


def _deepseek_synthesize(model_id: str, search: str) -> dict[str, Any]:
    cfg = config.load().get("research", {})
    key = _key(cfg.get("deepseek_key_env", "DEEPSEEK_API_KEY"))
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY not set in env")
    model = cfg.get("deepseek_model", "deepseek-chat")
    prompt = SYNTH_PROMPT.format(model_id=model_id, search=search[:6000])
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": 2500,
    }
    r = httpx.post(DEEPSEEK_URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    # strip code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content
        if content.endswith("```"):
            content = content.rsplit("```", 1)[0]
    return json.loads(content)


def _to_db_config(cand: dict[str, Any]) -> dict[str, Any]:
    """Convert the candidate JSON into the DB config shape (recipe-like)."""
    util = cand.get("gpu_memory_utilization") or 0.35
    mml = cand.get("max_model_len") or 8192
    return {
        "recipe_name": cand.get("display_name") or cand.get("hf_model_id"),
        "hf_model_id": cand["hf_model_id"],
        "container": "vllm-node",
        "defaults": {
            "port": 0,  # allocated at start
            "host": "0.0.0.0",
            "tensor_parallel": 1,
            "gpu_memory_utilization": util,
            "max_model_len": mml,
        },
        "env": {"VLLM_SERVER_DEV_MODE": "1"},
        "mods": cand.get("mods") or (["mods/fix-sleep-wake-mamba"] if cand.get("is_mamba_hybrid") else []),
        "command": cand["command"],
        "served_model_names": cand.get("served_model_names") or [cand["hf_model_id"]],
        "aliases": cand.get("aliases") or [],
        "use_model_name": None,
        "gateway_name": (cand.get("served_model_names") or [cand["hf_model_id"]])[0],
        "estimated_memory_gb": cand.get("estimated_memory_gb"),
        "research_notes": cand.get("notes"),
        "gated": cand.get("gated", False),
    }


def research_model(model_id: str) -> dict[str, Any]:
    """Run the two-stage agent. Returns {candidate: <db config>, citations, notes}."""
    log.info("researching %s", model_id)
    search, citations = _perplexity_search(model_id)
    log.info("perplexity returned %d chars, %d citations", len(search), len(citations))
    cand = _deepseek_synthesize(model_id, search)
    log.info("deepseek candidate: %s", cand.get("hf_model_id"))
    return {
        "candidate": _to_db_config(cand),
        "citations": citations,
        "notes": cand.get("notes", ""),
        "raw": cand,
    }
