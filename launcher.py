"""Container launcher: generate a recipe YAML from a DB config row, shell out to
run-recipe.py --solo (non-blocking), and stop/kill the resulting container.

Reuses the proven launch path (run-recipe.py + launch-cluster.sh) so the mod
system (fix-sleep-wake-mamba), the vllm-node image, and all mounts are applied
identically to the existing hand-launched backends.
"""
from __future__ import annotations

import os
import logging
import re

log = logging.getLogger("launcher")
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml

SPARK_VLLM_DIR = Path("/home/batchputz/spark-vllm-docker")
RUN_RECIPE = SPARK_VLLM_DIR / "run-recipe.py"
RECIPE_OUT_DIR = Path(__file__).resolve().parent / "generated-recipes"
DOCKER = "/usr/bin/docker"


def _ensure_dirs() -> None:
    RECIPE_OUT_DIR.mkdir(parents=True, exist_ok=True)


def _normalize_command(cmd: str) -> str:
    """Fix common vLLM CLI mistakes in generated commands."""
    # --served-model-names (plural, space-separated) -> repeated --served-model-name
    def _fix_plural(m: re.Match) -> str:
        names_str = m.group(1)
        names = names_str.strip().split()
        trailing = names_str[len(names_str.rstrip()):]
        return "--served-model-name " + " --served-model-name ".join(names) + trailing
    cmd = re.sub(r'--served-model-names\s+((?:[^\s\\]+\s*)+)', _fix_plural, cmd)
    return cmd


def generate_recipe_yaml(config: dict[str, Any], name: str) -> Path:
    """Reconstruct a recipe YAML file from a stored config dict, written to disk.

    The config dict mirrors the recipe schema (recipe_version, name, model,
    container, mods, defaults, env, command).
    """
    _ensure_dirs()
    recipe = {
        "recipe_version": "2",
        "name": config.get("recipe_name") or name,
        "model": config["defaults"].get("model") if "model" in config.get("defaults", {}) else None,
        "container": config.get("container", "vllm-node"),
        "mods": config.get("mods") or [],
        "defaults": config.get("defaults") or {},
        "env": config.get("env") or {},
        "command": config.get("command") or "",
    }
    # model lives at top-level of the original recipe; fall back to hf id if absent
    if not recipe["model"]:
        recipe["model"] = config.get("hf_model_id")
    command = _normalize_command(recipe.get("command") or "")
    recipe["command"] = command
    path = RECIPE_OUT_DIR / f"{name}.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False))
    return path


def launch(name: str, config: dict[str, Any], container_name: str) -> subprocess.Popen[bytes]:
    """Launch a vLLM container for the model via run-recipe.py --solo (non-blocking).

    Returns the Popen handle for the run-recipe.py process. The container itself
    runs detached (-d) inside; run-recipe.py stays alive as the supervisor.
    """
    recipe_path = generate_recipe_yaml(config, name)
    cmd = [
        "python3", str(RUN_RECIPE),
        str(recipe_path),
        "--solo",
        "--name", container_name,
    ]
    env = os.environ.copy()
    env["HOME"] = str(Path.home())
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    # run-recipe.py is a foreground supervisor; run it detached from the manager
    # so the manager stays responsive. Its stdout/stderr go to a log file.
    log_path = RECIPE_OUT_DIR / f"{name}.launch.log"
    log_f = open(log_path, "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(SPARK_VLLM_DIR),
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return proc


def stop(container_name: str, timeout: int = 30) -> bool:
    """Stop and remove a container. Returns True if it was running and is now gone."""
    try:
        subprocess.run(
            [DOCKER, "stop", "-t", str(timeout), container_name],
            check=False, capture_output=True, timeout=timeout + 10,
        )
    except subprocess.TimeoutExpired:
        pass
    try:
        subprocess.run(
            [DOCKER, "rm", "-f", container_name],
            check=False, capture_output=True, timeout=15,
        )
    except subprocess.TimeoutExpired:
        pass
    return not is_running(container_name)


def is_running(container_name: str) -> bool:
    """True if a container with this name exists and is running."""
    r = subprocess.run(
        [DOCKER, "inspect", "--format", "{{.State.Running}}", container_name],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


def container_pid(container_name: str) -> int | None:
    """Main PID of the container, or None."""
    r = subprocess.run(
        [DOCKER, "inspect", "--format", "{{.State.Pid}}", container_name],
        capture_output=True, text=True,
    )
    if r.returncode == 0 and r.stdout.strip().isdigit():
        return int(r.stdout.strip())
    return None


def is_port_listening(port: int, host: str = "127.0.0.1") -> bool:
    """True if something is already listening on the port."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(0.3)
    try:
        return s.connect_ex((host, port)) == 0
    finally:
        s.close()


def allocate_port(used_ports: set[int], lo: int = 8001, hi: int = 8099) -> int | None:
    """Find the lowest free port in [lo, hi] not in used_ports and not listening."""
    for p in range(lo, hi + 1):
        if p in used_ports:
            continue
        if not is_port_listening(p):
            return p
    return None


def container_gpu_mem_mb(container_name: str) -> int:
    """GPU/unified memory used by a container's processes, via nvidia-smi.

    On GB10 unified memory this is the accurate full footprint when AWAKE
    (cgroup/docker-stats undercounts CUDA VMM allocations)."""
    # PIDs inside the container (host-side)
    r = subprocess.run([DOCKER, "top", container_name, "-o", "pid"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return 0
    container_pids = set()
    for line in r.stdout.splitlines()[1:]:
        parts = line.split()
        if parts and parts[0].isdigit():
            container_pids.add(int(parts[0]))
    # nvidia-smi compute-apps: pid,used_memory
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True,
    )
    total = 0
    for line in r.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) in container_pids:
            m = parts[1].replace("MiB", "").strip()
            try:
                total += int(m)
            except ValueError:
                pass
    return total


def parse_weight_memory_gb(launch_log_path: Path) -> int | None:
    """Parse 'Model loading took X GiB' from a launch log -> MiB (weight memory retained when sleeping)."""
    import re
    try:
        text = Path(launch_log_path).read_text(errors="ignore")
    except Exception:
        return None
    m = re.search(r"Model loading took ([0-9.]+) GiB", text)
    return int(float(m.group(1)) * 1024) if m else None


def rm_path_as_root(path) -> bool:
    """Remove a path using a root container (for root-owned HF cache files).

    The manager runs as a normal user and can't delete files written by the
    vLLM container (root). Spawning a throwaway root container lets us rm them.
    """
    p = Path(path)
    parent = str(p.parent)
    name = p.name
    r = subprocess.run(
        [DOCKER, "run", "--rm", "-v", f"{parent}:/mnt", "vllm-node:latest",
         "rm", "-rf", f"/mnt/{name}"],
        capture_output=True, text=True, timeout=180,
    )
    if r.returncode != 0:
        log.warning("rm_path_as_root failed for %s: %s", path, r.stderr[:200])
    return r.returncode == 0
