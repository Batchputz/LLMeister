"""LLMeister parameter optimization agent.

Autonomous, iterative vLLM parameter optimizer. Uses DeepSeek (LLM) to reason
about parameter changes, Perplexity for research, and workload-aware benchmarks
for evaluation. Runs as a background async task.

Phases: interview → research → baseline → optimize loop → report
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import httpx

from . import config as cfg_mod
from . import db
from . import optimizer_db as optdb
from . import workload_benchmark as wb
from . import metrics as metrics_mod
from . import launcher

log = logging.getLogger("optimizer")

# API endpoints (reuse from research.py)
PERPLEXITY_URL = "https://api.perplexity.ai/chat/completions"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

MAX_ITERATIONS = 30
RESTART_TIMEOUT_S = 300  # 5 minutes for cold start


class OptimizationAgent:
    """Runs a single optimization session for one model."""

    def __init__(self, mgr, model_name: str):
        self.mgr = mgr  # LifecycleManager
        self.model_name = model_name
        self.run_id: int | None = None
        self.workload: dict[str, Any] = {}
        self.research: str = ""
        self.best_score: float = 0
        self.best_config: dict[str, Any] = {}
        self.baseline_score: float = 0
        self.current_step: int = 0
        self.stop_requested: bool = False
        self._task: asyncio.Task | None = None
        self._interview_messages: list[dict[str, str]] = []

    # ── Phase 1: Interview ────────────────────────────────────────────

    async def interview(self, user_message: str) -> str:
        """Process user's workload description. Returns agent's response."""
        # Get or create run
        conn = db.connect()
        db.init_db(conn)
        optdb.init_optimizer_db(conn)

        if not self.run_id:
            m = db.get_model(conn, self.model_name)
            if not m:
                return "Model not found."
            self.run_id = optdb.create_run(conn, self.model_name, m["hf_model_id"])
            self._interview_messages = []

        self._interview_messages.append({"role": "user", "content": user_message})

        # Use DeepSeek to parse the workload
        system_prompt = (
            "You are a vLLM workload analyst. The user describes how they use an LLM. "
            "Extract a structured workload specification. Ask follow-up questions if "
            "the user is not specific enough about: concurrency (how many parallel "
            "requests), input_tokens (approximate prompt length), output_tokens "
            "(expected response length), and priority (throughput vs latency).\n\n"
            "If you have enough information, respond with ONLY a JSON object:\n"
            '{"type": "batch|interactive|mixed", "concurrency": N, '
            '"input_tokens": N, "output_tokens": N, "priority": "throughput|latency|balanced", '
            '"description": "one-line summary"}\n\n'
            "If you need more info, ask a concise follow-up question."
        )

        self._interview_messages.insert(0, {"role": "system", "content": system_prompt})
        response = await self._call_deepseek(self._interview_messages)
        self._interview_messages.pop(0)  # remove system prompt from history

        # Check if DeepSeek returned JSON (workload complete)
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                self.workload = json.loads(json_match.group())
                optdb.update_run(conn, self.run_id,
                                 workload=json.dumps(self.workload),
                                 use_case=user_message)
                self._interview_messages.append({"role": "assistant", "content": response})

                # Start optimization in background
                conn.close()
                self._task = asyncio.create_task(self._run_optimization())
                return f"✅ Workload understood. Starting optimization:\n{json.dumps(self.workload, indent=2)}"
        except (json.JSONDecodeError, AttributeError):
            pass

        self._interview_messages.append({"role": "assistant", "content": response})
        conn.close()
        return response

    # ── Phase 2-5: Optimization ──────────────────────────────────────

    async def _run_optimization(self):
        """Main optimization loop. Runs as background task."""
        conn = db.connect()
        db.init_db(conn)
        optdb.init_optimizer_db(conn)

        try:
            # Phase 2: Research
            optdb.update_run(conn, self.run_id, status="researching")
            await self._do_research(conn)

            # Phase 3: Baseline
            optdb.update_run(conn, self.run_id, status="running")
            m = db.get_model(conn, self.model_name)
            if not m:
                raise ValueError("model not found")
            baseline_config = json.loads(m["config"])
            baseline_bench = await self._benchmark_model(conn)
            baseline_score = wb.compute_score(baseline_bench, self.workload)

            self.baseline_score = baseline_score
            self.best_score = baseline_score
            self.best_config = baseline_config

            baseline_step = optdb.add_step(conn, self.run_id, {
                "step_number": 0,
                "parameter": None,
                "old_value": None,
                "new_value": None,
                "reasoning": "Baseline measurement with current parameters",
                "config": baseline_config,
                "benchmark": baseline_bench,
                "score": baseline_score,
                "is_improvement": 0,
                "kept": 1,
            })
            optdb.update_run(conn, self.run_id, baseline_step=baseline_step, best_step=baseline_step)
            self.current_step = 0
            log.info("baseline score: %.1f", baseline_score)

            # Phase 4: Optimization loop
            for i in range(1, MAX_ITERATIONS + 1):
                if self.stop_requested:
                    log.info("optimization stopped by user at step %d", i)
                    break

                self.current_step = i
                success = await self._optimization_iteration(conn, i)
                if not success:
                    log.info("optimization converged at step %d", i)
                    break

            # Phase 5: Final restart with best config
            if self.best_config:
                db.set_config(conn, self.model_name, json.dumps(self.best_config))
                await self.mgr.stop(self.model_name)
                await asyncio.sleep(2)
                await self.mgr.start(self.model_name)

            # Mark complete
            optdb.update_run(conn, self.run_id,
                             status="completed",
                             completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
            log.info("optimization complete. best score: %.1f (baseline: %.1f, +%.1f%%)",
                     self.best_score, self.baseline_score,
                     ((self.best_score - self.baseline_score) / self.baseline_score * 100
                      if self.baseline_score > 0 else 0))

        except Exception as e:
            log.exception("optimization failed")
            optdb.update_run(conn, self.run_id,
                             status="failed",
                             error=str(e),
                             completed_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        finally:
            conn.close()

    async def _do_research(self, conn):
        """Search Perplexity for model-specific optimization tips."""
        m = db.get_model(conn, self.model_name)
        hf_id = m["hf_model_id"] if m else self.model_name

        queries = [
            f"vLLM {hf_id} optimization parameters performance tuning",
            f"vLLM max_num_seqs max_num_batched_tokens tuning {self.workload.get('type', 'batch')}",
            "vLLM chunked prefill prefix caching KV cache optimization",
        ]

        results = []
        for query in queries:
            try:
                result = await self._call_perplexity(query)
                results.append(f"Q: {query}\n{result}")
            except Exception as e:
                log.warning("research query failed: %s", e)

        self.research = "\n\n---\n\n".join(results)
        optdb.update_run(conn, self.run_id, research_json=self.research)

    async def _optimization_iteration(self, conn, step_num: int) -> bool:
        """One optimization step. Returns False if converged."""
        # Get current metrics
        m = db.get_model(conn, self.model_name)
        if not m:
            return False

        current_config = json.loads(m["config"])
        steps = optdb.get_steps(conn, self.run_id)

        # Ask DeepSeek for next parameter change
        suggestion = await self._ask_llm_for_change(current_config, steps)

        if suggestion.get("parameter") is None:
            return False  # converged

        param = suggestion["parameter"]
        new_value = suggestion["new_value"]
        reasoning = suggestion.get("reasoning", "")

        # Apply the change
        old_config = json.loads(json.dumps(current_config))  # deep copy
        old_value = _extract_param(current_config, param)
        new_config = _apply_param_change(current_config, param, new_value)

        if new_config == old_config:
            log.info("step %d: no change applied (param not found or same value)", step_num)
            return True  # try again

        # Save new config and restart
        db.set_config(conn, self.model_name, json.dumps(new_config))
        t_restart = time.monotonic()
        restart_ok = await self._restart_model()

        if not restart_ok:
            # Revert
            db.set_config(conn, self.model_name, json.dumps(old_config))
            optdb.add_step(conn, self.run_id, {
                "step_number": step_num,
                "parameter": param,
                "old_value": old_value,
                "new_value": new_value,
                "reasoning": reasoning + " [FAILED TO START — reverted]",
                "config": new_config,
                "benchmark": None,
                "score": 0,
                "is_improvement": 0,
                "kept": 0,
                "restart_time_s": time.monotonic() - t_restart,
            })
            return True  # continue trying

        restart_time = time.monotonic() - t_restart

        # Benchmark
        t_bench = time.monotonic()
        bench_result = await self._benchmark_model(conn)
        bench_time = time.monotonic() - t_bench

        score = wb.compute_score(bench_result, self.workload)
        is_improvement = score > self.best_score

        if is_improvement:
            self.best_score = score
            self.best_config = new_config
            kept = 1
            log.info("step %d: %s=%s → score %.1f (IMPROVED, kept)", step_num, param, new_value, score)
        else:
            # Revert config in DB (running model keeps tested config until next restart)
            db.set_config(conn, self.model_name, json.dumps(old_config))
            kept = 0
            log.info("step %d: %s=%s → score %.1f (not better than %.1f, reverted)",
                     step_num, param, new_value, score, self.best_score)

        optdb.add_step(conn, self.run_id, {
            "step_number": step_num,
            "parameter": param,
            "old_value": old_value,
            "new_value": new_value,
            "reasoning": reasoning,
            "config": new_config,
            "benchmark": bench_result,
            "metrics": await self._get_metrics(),
            "score": score,
            "is_improvement": is_improvement,
            "kept": kept,
            "restart_time_s": restart_time,
            "benchmark_time_s": bench_time,
        })

        return True

    # ── Helpers ───────────────────────────────────────────────────────

    async def _restart_model(self) -> bool:
        """Stop and start the model. Returns True if healthy."""
        try:
            await self.mgr.stop(self.model_name)
            await asyncio.sleep(2)
            ok = await self.mgr.start(self.model_name)
            return ok
        except Exception as e:
            log.warning("restart failed: %s", e)
            return False

    async def _benchmark_model(self, conn) -> dict[str, Any]:
        """Run workload benchmark on the current model."""
        m = db.get_model(conn, self.model_name)
        if not m or not m["port"]:
            return {"error": "model not running"}
        port = m["port"]
        cfg = json.loads(m["config"])
        served_name = (cfg.get("served_model_names") or [m["hf_model_id"]])[0]
        return await wb.run_workload_benchmark(port, served_name, self.workload)

    async def _get_metrics(self) -> dict[str, Any] | None:
        """Get vLLM metrics snapshot."""
        conn = db.connect()
        try:
            m = db.get_model(conn, self.model_name)
            if not m or not m["port"]:
                return None
            log_path = launcher.RECIPE_OUT_DIR / f"{self.model_name}.launch.log"
            if not log_path.exists():
                return None
            return metrics_mod.parse_engine_stats(log_path, max_points=5)
        finally:
            conn.close()

    async def _ask_llm_for_change(self, current_config: dict, steps: list[dict]) -> dict:
        """Ask DeepSeek for the next parameter change."""
        # Build current params summary
        cmd = current_config.get("command", "")
        defaults = current_config.get("defaults", {})
        params = _extract_all_params(cmd, defaults)

        # Build history summary (last 10 steps)
        history = ""
        for s in steps[-10:]:
            status = "✓" if s.get("kept") else "✗"
            history += f"  #{s['step_number']} {s.get('parameter','')}={s.get('new_value','')} → score {s.get('score',0):.1f} {status}\n"

        # Get latest benchmark
        last_bench = steps[-1].get("benchmark") if steps else None
        bench_summary = ""
        if last_bench:
            bench_summary = (
                f"Aggregate throughput: {last_bench.get('aggregate_tok_s', 0)} tok/s\n"
                f"TTFT p50: {last_bench.get('p50_ttft_ms', 0)}ms, p95: {last_bench.get('p95_ttft_ms', 0)}ms\n"
                f"Concurrency: {last_bench.get('concurrency', 0)}, errors: {last_bench.get('errors', 0)}"
            )

        prompt = f"""You are a vLLM performance optimization agent. Suggest the next SINGLE parameter change.

Model: {current_config.get('hf_model_id', self.model_name)}
Workload: {json.dumps(self.workload)}
Priority: {self.workload.get('priority', 'balanced')}

Current parameters:
{json.dumps(params, indent=2)}

Current performance:
{bench_summary or 'No baseline yet'}

Previous attempts:
{history or 'None yet'}

Research findings:
{self.research[:2000]}

Rules:
- Change ONLY ONE parameter at a time
- Consider what hasn't been tried and what the metrics suggest is the bottleneck
- Don't repeat changes that were already tried and reverted
- Stay within safe bounds (gpu_mem_util max 0.85, max_num_seqs max 512)

Respond as JSON:
{{"parameter": "param_name", "new_value": "value", "reasoning": "why"}}

If no more promising changes remain, respond:
{{"parameter": null, "reasoning": "converged"}}"""

        messages = [{"role": "user", "content": prompt}]
        response = await self._call_deepseek(messages)

        try:
            json_match = re.search(r'\{[^{}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"parameter": None, "reasoning": "LLM response not parseable: " + response[:200]}

    async def _call_deepseek(self, messages: list[dict]) -> str:
        """Call DeepSeek API."""
        c = cfg_mod.load()
        key = _get_env(c.get("research", {}).get("deepseek_key_env", "DEEPSEEK_API_KEY"))
        model = c.get("research", {}).get("deepseek_model", "deepseek-chat")
        r = await asyncio.to_thread(
            lambda: httpx.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": 1000},
                timeout=90,
            )
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    async def _call_perplexity(self, query: str) -> str:
        """Call Perplexity API."""
        c = cfg_mod.load()
        key = _get_env(c.get("research", {}).get("perplexity_key_env", "PERPLEXITY_API_KEY"))
        model = c.get("research", {}).get("perplexity_model", "sonar-pro")
        r = await asyncio.to_thread(
            lambda: httpx.post(
                PERPLEXITY_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": query}],
                    "temperature": 0.2,
                    "max_tokens": 2000,
                },
                timeout=60,
            )
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def stop(self):
        """Request the agent to stop after the current step."""
        self.stop_requested = True


# ── Parameter manipulation helpers ─────────────────────────────────────

def _extract_param(config: dict, param: str) -> str | None:
    """Extract current value of a vLLM parameter from the command or defaults."""
    cmd = config.get("command", "")
    # Try command line first (--param-name value)
    m = re.search(rf'--{re.escape(param)}\s+(\S+)', cmd)
    if m:
        return m.group(1)
    # Try defaults dict (param_name with underscores)
    defaults = config.get("defaults", {})
    key = param.replace("-", "_")
    if key in defaults:
        return str(defaults[key])
    # Check if it's a flag (--enable-xxx present or not)
    if f"--{param}" in cmd:
        return "true"
    return None


def _apply_param_change(config: dict, param: str, new_value: str) -> dict:
    """Apply a parameter change to the config. Returns a new config dict."""
    import copy
    new_config = copy.deepcopy(config)
    cmd = new_config.get("command", "")

    # Check if it's a boolean flag (enable/disable)
    if new_value.lower() in ("true", "false", "on", "off"):
        flag = f"--{param}"
        if new_value.lower() in ("true", "on"):
            # Add flag if not present
            if flag not in cmd:
                # Insert before the last backslash-continuation or at end
                cmd = cmd.rstrip() + f" \\\n    {flag}"
        else:
            # Remove flag
            cmd = re.sub(rf'\s*{re.escape(flag)}\s*\\?\n\s*', '\n    ', cmd)
            cmd = re.sub(rf'\s+{re.escape(flag)}', '', cmd)
        new_config["command"] = cmd
        return new_config

    # Numeric/string parameter
    pattern = rf'(--{re.escape(param)}\s+)\S+'
    if re.search(pattern, cmd):
        new_config["command"] = re.sub(pattern, rf'\g<1>{new_value}', cmd)
    else:
        # Parameter not in command, add it
        cmd = cmd.rstrip() + f" \\\n    --{param} {new_value}"
        new_config["command"] = cmd

    # Also update defaults dict if the key exists there
    defaults = new_config.get("defaults", {})
    key = param.replace("-", "_")
    if key in defaults:
        try:
            defaults[key] = type(defaults[key])(new_value)
        except (ValueError, TypeError):
            defaults[key] = new_value

    return new_config


def _extract_all_params(cmd: str, defaults: dict) -> dict:
    """Extract all vLLM parameters from the command line and defaults."""
    params = {}
    # From defaults
    for k, v in defaults.items():
        params[k] = v
    # From command line
    for m in re.finditer(r'--([\w-]+)\s+(\S+)', cmd):
        params[m.group(1)] = m.group(2)
    # Flags (boolean params)
    for m in re.finditer(r'--(enable-[\w-]+)', cmd):
        if m.group(1) not in params:
            params[m.group(1)] = True
    return params


def _get_env(name: str) -> str:
    import os
    return os.environ.get(name, "")
