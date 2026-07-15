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
from . import param_kb

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
        self.system_context: dict[str, Any] = {}   # WP1
        self.plan: dict[str, Any] = {}             # WP4
        self._plan_queue: list[dict] = []          # WP5 — pending experiments

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
            workload_json = _extract_json(response)
            if workload_json:
                self.workload = workload_json
                optdb.update_run(conn, self.run_id,
                                 workload=json.dumps(self.workload),
                                 use_case=user_message)
                self._interview_messages.append({"role": "assistant", "content": response})

                # Start optimization in background
                optdb.update_run(conn, self.run_id, chat_json=json.dumps(self._interview_messages))
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
            # Phase 1.5: System analysis (WP1) — gather GPU/memory context
            self.system_context = await self._gather_system_context(conn)
            optdb.update_run(conn, self.run_id, system_json=json.dumps(self.system_context))
            log.info("system context: %s", self.system_context.get("gpu", "unknown"))

            # Phase 2: Research (WP3 — enriched with system context)
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

            # Phase 3.5: Planning (WP4) — create a branching optimization plan
            self.plan = await self._create_plan(conn, baseline_config, baseline_bench)
            optdb.update_run(conn, self.run_id, plan_json=json.dumps(self.plan))
            log.info("plan created: %d experiments", len(self.plan.get("experiments", [])))

            # Phase 4: Optimization loop (WP5 — plan-aware)
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
        gpu = self.system_context.get("gpu", "unknown GPU")
        vram = self.system_context.get("total_memory_gib", "unknown")
        vllm_ver = self.system_context.get("vllm_version", "")
        wtype = self.workload.get("type", "batch")
        concurrency = self.workload.get("concurrency", 1)
        in_tok = self.workload.get("input_tokens", 0)

        # WP3 — context-aware queries (GPU, vLLM version, workload specifics)
        queries = [
            f"vLLM {hf_id} on {gpu} ({vram}GB) optimal serving config" + (f" {vllm_ver}" if vllm_ver else ""),
            f"vLLM {vllm_ver or ''} parameter interactions max_num_seqs gpu_memory_utilization max_model_len".strip(),
            f"vLLM {hf_id} known issues OOM KV cache",
            f"vLLM {hf_id} {wtype} {concurrency} concurrent {in_tok} token throughput tuning",
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

        # WP3 — synthesize research into structured notes via DeepSeek
        await self._synthesize_research(conn)

    async def _synthesize_research(self, conn):
        """Ask DeepSeek to distill raw research into structured notes."""
        prompt = f"""Synthesize the following vLLM optimization research into structured notes.

Model: {self.model_name}
Workload: {json.dumps(self.workload)}
System: {json.dumps(self.system_context, indent=2)}

Research results:
{self.research[:6000]}

Extract and respond as JSON:
{{
  "recommended_params": {{"param_name": "value with reason"}},
  "params_to_avoid": [{{"param": "name", "reason": "why"}}],
  "interactions": ["rule 1", "rule 2"],
  "experiment_order": ["try X first because...", "then Y..."],
  "gotchas": ["known issue 1", "known issue 2"]
}}

If research is thin, fill what you can and leave arrays empty."""
        response = await self._call_deepseek([{"role": "user", "content": prompt}], max_tokens=2000)
        notes = _extract_json(response) or {}
        optdb.update_run(conn, self.run_id, research_notes=json.dumps(notes))
        self.research_notes = notes
        log.info("research synthesized: %d recommendations, %d gotchas",
                 len(notes.get("recommended_params", {})), len(notes.get("gotchas", [])))

    async def _gather_system_context(self, conn) -> dict[str, Any]:
        """WP1 — Gather GPU specs, memory budget, vLLM version for the prompts."""
        import subprocess
        ctx: dict[str, Any] = {}

        # GPU info via nvidia-smi
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            parts = [p.strip() for p in out.stdout.strip().split(",")]
            if len(parts) >= 4:
                ctx["gpu"] = parts[0]
                ctx["total_memory_gib"] = round(float(parts[1]) / 1024, 1)
                ctx["used_memory_gib"] = round(float(parts[2]) / 1024, 1)
                ctx["driver_version"] = parts[3]
        except Exception:
            pass

        # OS + arch
        import platform
        ctx["arch"] = platform.machine()
        try:
            with open("/etc/os-release") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        ctx["os"] = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass

        # vLLM version (from manager's cached sys_info via API would be ideal;
        # here we read it from the container)
        m = db.get_model(conn, self.model_name)
        if m:
            cfg = json.loads(m["config"])
            container = cfg.get("container", "vllm-node")
            try:
                out = subprocess.run(
                    ["docker", "run", "--rm", container, "pip", "show", "vllm"],
                    capture_output=True, text=True, timeout=20,
                )
                for line in out.stdout.splitlines():
                    if line.startswith("Version: "):
                        ctx["vllm_version"] = line.split(": ", 1)[1].strip()
                        break
            except Exception:
                pass

            # Model memory footprint
            if m.get("measured_weight_memory_mb"):
                ctx["weight_memory_gib"] = round(m["measured_weight_memory_mb"] / 1024, 1)
            if m.get("measured_max_memory_mb"):
                ctx["model_max_memory_gib"] = round(m["measured_max_memory_mb"] / 1024, 1)

        # Memory budget calculation
        total = ctx.get("total_memory_gib", 0)
        if total:
            ctx["os_reserve_gib"] = 12  # config default
            ctx["available_for_kv_gib"] = round(
                total - ctx.get("os_reserve_gib", 12) - ctx.get("weight_memory_gib", 0), 1
            )

        return ctx

    async def _create_plan(self, conn, baseline_config: dict, baseline_bench: dict) -> dict:
        """WP4 — Ask DeepSeek to create a branching optimization plan.

        The plan is a list of experiments with success/failure branches.
        The execution loop follows this plan instead of free-associating.
        """
        # Extract current params for context
        cmd = baseline_config.get("command", "")
        defaults = baseline_config.get("defaults", {})
        current_params = _extract_all_params(cmd, defaults)

        prompt = f"""You are a vLLM performance optimization strategist. Create a branching optimization plan.

DO NOT just suggest one change. Think like an expert: plan a SEQUENCE of experiments
with branches (if X works → try Y; if X fails → try Z).

SYSTEM CONTEXT:
{json.dumps(self.system_context, indent=2)}

MODEL: {baseline_config.get('hf_model_id', self.model_name)}
WORKLOAD: {json.dumps(self.workload)}
PRIORITY: {self.workload.get('priority', 'balanced')}

CURRENT PARAMETERS:
{json.dumps(current_params, indent=2)}

BASELINE PERFORMANCE:
  Throughput: {baseline_bench.get('aggregate_tok_s', 0)} tok/s
  TTFT p50: {baseline_bench.get('p50_ttft_ms', 0)}ms
  Errors: {baseline_bench.get('errors', 0)}/{baseline_bench.get('concurrency', 0)}

{param_kb.kb_summary()}

SYNTHESIZED RESEARCH NOTES:
{json.dumps(getattr(self, 'research_notes', {}), indent=2)[:3000]}

Create a plan with 4-8 experiments. Order by expected value (high-impact, low-risk first).
Each experiment changes 1-2 related parameters. Include branches.

Respond as JSON:
{{
  "goal": "one sentence",
  "rationale": "why this plan",
  "experiments": [
    {{
      "id": 1,
      "hypothesis": "what we expect",
      "change": {{"param_name": "value"}},
      "expected": "expected outcome",
      "on_success": [2, 3],
      "on_failure": [4],
      "risk": "low|medium|high",
      "rationale": "why this experiment"
    }}
  ]
}}"""
        response = await self._call_deepseek([{"role": "user", "content": prompt}], max_tokens=3000)
        plan = _extract_json(response)
        if not plan or "experiments" not in plan:
            log.warning("plan parsing failed, using fallback single-experiment plan")
            plan = {"goal": "optimize throughput", "experiments": [], "rationale": "fallback"}
        # Initialize the plan queue with the first experiment
        if plan.get("experiments"):
            self._plan_queue = [plan["experiments"][0]]
        return plan

    def _branch_plan(self, experiment: dict | None, success: bool):
        """WP5 — Follow the plan's branch after an experiment succeeds or fails.

        Queues the next experiments based on on_success/on_failure.
        """
        if not experiment or not self.plan:
            return
        exp_id = experiment.get("id")
        branch_key = "on_success" if success else "on_failure"
        next_ids = experiment.get(branch_key, [])
        if not next_ids:
            return
        experiments_by_id = {e.get("id"): e for e in self.plan.get("experiments", [])}
        for nid in next_ids:
            next_exp = experiments_by_id.get(nid)
            if next_exp and next_exp not in self._plan_queue:
                self._plan_queue.append(next_exp)
                log.info("plan branch: experiment #%s %s → queued #%s",
                         exp_id, "succeeded" if success else "failed", nid)

    def _memory_precheck(self, config: dict) -> tuple[bool, str]:
        """WP6 — Estimate if a config change would OOM before wasting a 5-min restart.

        Returns (ok, message).
        """
        defaults = config.get("defaults", {})
        cmd = config.get("command", "")
        max_seqs = _extract_param(config, "max-num-seqs")
        max_len = _extract_param(config, "max-model-len")
        gpu_util = _extract_param(config, "gpu-memory-utilization")
        kv_dtype = _extract_param(config, "kv-cache-dtype") or "auto"

        try:
            max_seqs = int(max_seqs) if max_seqs else 64
            max_len = int(max_len) if max_len else 8192
            gpu_util = float(gpu_util) if gpu_util else 0.4
        except (ValueError, TypeError):
            return True, "precheck skipped (unparseable params)"

        # Estimate KV cache size
        kv_gib = param_kb.estimate_kv_cache_gib(max_seqs, max_len, kv_dtype)
        weight_gib = self.system_context.get("weight_memory_gib", 8)
        total_gib = self.system_context.get("total_memory_gib", 121)
        available = total_gib * gpu_util

        needed = kv_gib + weight_gib + 2.5  # +2.5 for activations + CUDA graphs
        if needed > available:
            msg = (f"would need ~{needed:.1f} GiB (KV {kv_gib:.1f} + weights {weight_gib} + overhead) "
                   f"but only {available:.1f} GiB available at gpu_util={gpu_util}")
            return False, msg
        return True, f"est. {needed:.1f} GiB needed, {available:.1f} available — OK"

    async def _optimization_iteration(self, conn, step_num: int) -> bool:
        """One optimization step. Returns False if converged.

        WP5 — plan-aware: prefers experiments from the branching plan.
        Falls back to asking DeepSeek if the plan queue is empty.
        """
        # Get current metrics
        m = db.get_model(conn, self.model_name)
        if not m:
            return False

        current_config = json.loads(m["config"])
        steps = optdb.get_steps(conn, self.run_id)

        # WP5 — Try the next planned experiment first
        experiment = None
        if self._plan_queue:
            experiment = self._plan_queue.pop(0)
            log.info("step %d: executing planned experiment #%s: %s",
                     step_num, experiment.get("id", "?"), experiment.get("hypothesis", "")[:80])
            change = experiment.get("change", {})
            if change:
                suggestion = {"parameters": change, "reasoning": experiment.get("rationale", "")}
            else:
                suggestion = await self._ask_llm_for_change(current_config, steps)
        else:
            # Plan exhausted — ask DeepSeek for ad-hoc suggestion
            suggestion = await self._ask_llm_for_change(current_config, steps)

        if suggestion.get("parameter") is None and not suggestion.get("parameters"):
            return False  # converged

        param = suggestion.get("parameter")
        multi_params = suggestion.get("parameters")  # multi-param change
        new_value = str(suggestion["new_value"]) if suggestion.get("new_value") is not None else None
        reasoning = suggestion.get("reasoning", "")

        # Multi-parameter changes: apply all at once
        if multi_params and isinstance(multi_params, dict):
            new_config = json.loads(json.dumps(current_config))
            for p, v in multi_params.items():
                new_config = _apply_param_change(new_config, p, v)
            param_desc = ", ".join(f"{p}={v}" for p, v in multi_params.items())
            param = param_desc
            new_value = param_desc

        # Loop detection for single-param changes only
        if not multi_params:
            param_fails = sum(
                1 for s in steps[-6:]
                if s.get("parameter") == param and not s.get("kept")
            )
            if param_fails >= 2:
                log.info("step %d: rejecting %s — already failed %d times with different values — converging",
                         step_num, param, param_fails)
                return False

            exact_retry = [
                s for s in steps[-6:]
                if s.get("parameter") == param
                and str(s.get("new_value")) == str(new_value)
                and not s.get("kept")
            ]
            if len(exact_retry) >= 1:
                log.info("step %d: %s=%s already tried and rejected — re-prompting DeepSeek",
                         step_num, param, new_value)
                retry_suggestion = await self._ask_llm_for_change(
                    current_config, steps, blacklist=set(f"{s.get('parameter')}={s.get('new_value')}"
                                                         for s in steps if not s.get('kept'))
                )
                if retry_suggestion.get("parameter") or retry_suggestion.get("parameters"):
                    suggestion = retry_suggestion
                    param = suggestion.get("parameter")
                    multi_params = suggestion.get("parameters")
                    new_value = str(suggestion["new_value"]) if suggestion.get("new_value") is not None else None
                    reasoning = suggestion.get("reasoning", "")
                    if multi_params:
                        new_config = json.loads(json.dumps(current_config))
                        for p, v in multi_params.items():
                            new_config = _apply_param_change(new_config, p, v)
                        param_desc = ", ".join(f"{p}={v}" for p, v in multi_params.items())
                        param = param_desc
                        new_value = param_desc
                else:
                    return False

            if new_value is None and not multi_params:
                log.info("step %d: DeepSeek returned null new_value — converging", step_num)
                return False

        # Apply the change
        old_config = json.loads(json.dumps(current_config))  # deep copy
        old_value = ""  # default for multi-param changes
        if not multi_params:
            old_value = _extract_param(current_config, param)
            new_config = _apply_param_change(current_config, param, new_value)

        if new_config == old_config:
            log.info("step %d: no change applied (param not found or same value)", step_num)
            return True  # try again

        # WP6 — Memory pre-check: skip restart if the change would obviously OOM
        mem_ok, mem_msg = self._memory_precheck(new_config)
        if not mem_ok:
            log.info("step %d: %s=%s SKIPPED (memory pre-check: %s)",
                     step_num, param, new_value, mem_msg)
            optdb.add_step(conn, self.run_id, {
                "step_number": step_num,
                "parameter": param,
                "old_value": old_value if not multi_params else "",
                "new_value": new_value,
                "reasoning": f"{reasoning} [SKIPPED — {mem_msg}]",
                "config": new_config,
                "benchmark": None,
                "score": 0,
                "is_improvement": 0,
                "kept": 0,
                "memory_estimate": mem_msg,
            })
            # Branch to failure path of this experiment
            self._branch_plan(experiment, success=False)
            return True

        # Save new config and restart
        db.set_config(conn, self.model_name, json.dumps(new_config))
        t_restart = time.monotonic()
        restart_ok = await self._restart_model()

        if not restart_ok:
            # Revert
            db.set_config(conn, self.model_name, json.dumps(old_config))
            restart_time_s = time.monotonic() - t_restart
            log.info("step %d: %s=%s FAILED TO START after %.0fs — reverted",
                     step_num, param, new_value, restart_time_s)
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
                "restart_time_s": restart_time_s,
                "experiment_id": experiment.get("id") if experiment else None,
            })
            # Branch to failure path of this experiment
            self._branch_plan(experiment, success=False)
            # Consecutive restart failure detection — don't burn all 30 steps
            recent_fails = sum(
                1 for s in steps
                if s.get("benchmark") is None and not s.get("kept")
            )
            if recent_fails >= 2:
                log.info("step %d: %d consecutive restart failures — converging to avoid endless crashes",
                         step_num, recent_fails + 1)
                return False
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
            self._branch_plan(experiment, success=True)
        else:
            # Revert config in DB (running model keeps tested config until next restart)
            db.set_config(conn, self.model_name, json.dumps(old_config))
            kept = 0
            log.info("step %d: %s=%s → score %.1f (not better than %.1f, reverted)",
                     step_num, param, new_value, score, self.best_score)
            self._branch_plan(experiment, success=False)

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
            "experiment_id": experiment.get("id") if experiment else None,
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
        # Inject current max_model_len so the benchmark can respect it
        workload = dict(self.workload)
        ml = _extract_param(cfg, "max-model-len")
        if ml:
            try:
                workload["max_model_len"] = int(ml)
            except (ValueError, TypeError):
                workload["max_model_len"] = int(cfg.get("defaults", {}).get("max_model_len", 12288))
        return await wb.run_workload_benchmark(port, served_name, workload)

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

    async def _ask_llm_for_change(self, current_config: dict, steps: list[dict],
                                   blacklist: set[str] | None = None) -> dict:
        """Ask DeepSeek for the next parameter change.

        Args:
            blacklist: set of "param=value" strings that must NOT be suggested.
        """
        # Build current params summary
        cmd = current_config.get("command", "")
        defaults = current_config.get("defaults", {})
        params = _extract_all_params(cmd, defaults)

        # Build history summary (last 12 steps)
        history = ""
        for s in steps[-12:]:
            status = "✓" if s.get("kept") else "✗"
            err_info = ""
            bench = s.get("benchmark")
            if bench and isinstance(bench, dict):
                tok_s = bench.get("aggregate_tok_s", 0)
                errs = bench.get("errors", 0)
                if errs > 0:
                    err_detail = bench.get("error_summary", "")
                    err_info = f"  {errs} errors" + (f": {err_detail}" if err_detail else "")
            history += f"  #{s['step_number']} {s.get('parameter','')}={s.get('new_value','')} → score {s.get('score',0):.1f} {status}{err_info}\n"

        # Get latest benchmark with rich diagnostic info
        last_bench = steps[-1].get("benchmark") if steps else None
        bench_summary = ""
        if last_bench:
            bench_summary = (
                f"Aggregate throughput: {last_bench.get('aggregate_tok_s', 0)} tok/s\n"
                f"TTFT p50: {last_bench.get('p50_ttft_ms', 0)}ms, p95: {last_bench.get('p95_ttft_ms', 0)}ms\n"
                f"Total tokens generated: {last_bench.get('total_tokens', 0)}\n"
                f"Concurrency: {last_bench.get('concurrency', 0)}, errors: {last_bench.get('errors', 0)}"
            )
            if last_bench.get("errors", 0) > 0:
                error_detail = last_bench.get("error_summary", "all requests failed")
                bench_summary += f"\nCRITICAL: {last_bench.get('errors', 0)}/{last_bench.get('concurrency', 0)} requests errored — {error_detail}"

        # Blacklist warning
        blacklist_str = ""
        if blacklist:
            blacklist_str = "\nALREADY TRIED AND REVERTED (do NOT suggest these again):\n"
            blacklist_str += "\n".join(f"  - {b}" for b in sorted(blacklist))

        prompt = f"""You are a vLLM performance optimization agent. Suggest the next parameter change(s).

Model: {current_config.get('hf_model_id', self.model_name)}
Workload: {json.dumps(self.workload)}
Priority: {self.workload.get('priority', 'balanced')}

SYSTEM CONTEXT:
{json.dumps(self.system_context, indent=2)}

Current parameters:
{json.dumps(params, indent=2)}

Current performance:
{bench_summary or 'No baseline yet'}

Previous attempts:
{history or 'None yet'}
{blacklist_str}

{param_kb.kb_summary()}

Research findings:
{self.research[:2000]}

Strategy:
- If a parameter keeps OOMing or failing, STOP changing it and switch categories entirely.
- If no single parameter improves things, try 1-2 RELATED parameters together
  (e.g. increase gpu_memory_utilization AND increase max_num_seqs).
- If errors=0 and throughput>0, try INCREASING concurrency (max_num_seqs) or batch size.
- If errors>0 or OOM, try REDUCING max_num_seqs or max_model_len.
- Stay within: gpu_mem_util max 0.85, max_num_seqs max 64, max_model_len max 131072.

Respond as JSON:
{{"parameter": "param_name", "new_value": "value", "reasoning": "why"}}

Or for a multi-parameter change:
{{"parameters": {{"gpu_memory_utilization": 0.6, "max_num_seqs": 16}}, "reasoning": "why"}}

If no more promising changes remain, respond:
{{"parameter": null, "reasoning": "converged"}}"""

        messages = [{"role": "user", "content": prompt}]
        response = await self._call_deepseek(messages)

        try:
            suggestion = _extract_json(response)
            if suggestion:
                return suggestion
        except (json.JSONDecodeError, AttributeError):
            pass
        return {"parameter": None, "reasoning": "LLM response not parseable: " + response[:200]}

    async def _call_deepseek(self, messages: list[dict], max_tokens: int = 1000) -> str:
        """Call DeepSeek API."""
        c = cfg_mod.load()
        key = _get_env(c.get("research", {}).get("deepseek_key_env", "DEEPSEEK_API_KEY"))
        model = c.get("research", {}).get("deepseek_model", "deepseek-chat")
        r = await asyncio.to_thread(
            lambda: httpx.post(
                DEEPSEEK_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "temperature": 0.3, "max_tokens": max_tokens},
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
    # JSON booleans/numbers from LLM responses may not be strings — coerce.
    new_value = str(new_value) if new_value is not None else ""
    new_config = copy.deepcopy(config)
    cmd = new_config.get("command", "")
    
    # Normalize line endings - ensure consistent \n format
    cmd = cmd.replace("\\n", "\n")

    # Check if it's a boolean flag (enable/disable)
    if new_value.lower() in ("true", "false", "on", "off"):
        flag = f"--{param}"
        if new_value.lower() in ("true", "on"):
            # Add flag if not present
            if flag not in cmd:
                # Insert before the last line continuation
                lines = cmd.split("\n")
                # Find last non-empty line
                for i in range(len(lines)-1, -1, -1):
                    if lines[i].strip():
                        lines.insert(i+1, f"    {flag}")
                        break
                cmd = "\n".join(lines)
        else:
            # Remove flag - handle both with and without line continuation
            cmd = re.sub(rf'\s*{re.escape(flag)}(\s*\\\s*\n|\s*$)', '', cmd, flags=re.MULTILINE)
            cmd = re.sub(rf'\s+{re.escape(flag)}', '', cmd)
        new_config["command"] = cmd
        return new_config

    # Numeric/string parameter - find and replace.
    # Try the literal value first, then try a template placeholder like {max_model_len}.
    pattern = rf'(--{re.escape(param)}\s+)\S+'
    if re.search(pattern, cmd):
        new_config["command"] = re.sub(pattern, rf'\g<1>{new_value}', cmd)
    else:
        # Maybe the command uses a template placeholder `{param_name}`?
        template_key = param.replace("-", "_")
        template_pattern = rf'(--{re.escape(param)}\s+)' + r'\{([^}]*)\}'
        if re.search(template_pattern, cmd):
            # Replace template placeholder with the actual value
            new_config["command"] = re.sub(template_pattern, rf'\g<1>{new_value}', cmd)
        else:
            # Parameter not in command at all, add it before the last line
            lines = cmd.split("\n")
            for i in range(len(lines)-1, -1, -1):
                if lines[i].strip():
                    lines.insert(i+1, f"    --{param} {new_value}")
                    break
            cmd = "\n".join(lines)
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


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from LLM output, handling nested braces and code fences."""
    # Strip markdown code fences
    text = text.replace("```json", "").replace("```", "")
    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except (json.JSONDecodeError, ValueError):
                    return None
    return None
