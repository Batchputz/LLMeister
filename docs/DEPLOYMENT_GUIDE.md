# DGX Spark — Deployment Guide for Fresh Machines

This guide enables an agent or operator to set up a new NVIDIA DGX Spark (GB10)
as an LLM inference appliance, from bare OS to a fully managed multi-model
gateway with automated parameter optimization.

**Target audience:** AI coding agents performing hands-on deployment on a
client's DGX Spark.

**Expected outcome:** A machine with:
- Dockerized vLLM serving multiple models
- llama-swap OpenAI-compatible gateway on port 8080
- LLMeister dashboard + lifecycle manager on port 9001
- Automated parameter optimization agent
- All services running under systemd user scope

---

## 0. Prerequisites & Assumptions

- DGX Spark (NVIDIA GB10 Grace Blackwell, ARM64, 121 GiB unified memory)
- Ubuntu / DGX OS with `systemctl --user` linger enabled for the deploy user
- Docker installed with `--gpus all` support
- The machine is reachable via SSH (WireGuard or local network)
- Deploy user has passwordless sudo for Docker operations
- Internet access for HuggingFace model downloads and API calls
- Working WireGuard overlay (optional but recommended for production)

---

## 1. Machine Verification

SSH into the machine as the deploy user and verify hardware:

```bash
# Architecture must be aarch64
uname -m

# GPU must be detected
nvidia-smi -L
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

# Memory
free -h

# Docker
docker run --rm hello-world
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu24.04 nvidia-smi

# User linger (services survive logout)
loginctl enable-linger $USER
loginctl show-user $USER | grep Linger
```

**Critical checks:**
- `nvidia-smi` shows a GB10 GPU with ~121 GiB memory
- Docker works with `--gpus all`
- Architecture is `aarch64` (not x86_64)
- User linger is enabled

---

## 2. Install Core Dependencies (on the Spark)

```bash
# Python toolchain
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Verify
uv --version  # should be 0.5+
python3 --version  # should be 3.12+

# Essentials
sudo apt-get update && sudo apt-get install -y \
  jq curl wget git build-essential python3-dev
```

---

## 3. Clone & Configure the vLLM Stack

### 3.1 Clone the orchestrator repo

```bash
mkdir -p ~/Software
cd ~/Software
git clone https://github.com/<your-org>/dgx-spark.git
cd dgx-spark/llmeister
```

If using a private repo, set up SSH keys or a GitHub token first.

### 3.2 Install LLMeister

```bash
cd ~/Software/dgx-spark/llmeister
uv sync
```

This creates a `.venv` with FastAPI, uvicorn, httpx, docker, etc.

### 3.3 Configure LLMeister

Edit `config.yaml` — the single source of truth:

```yaml
paths:
  spark_vllm_dir: "/home/<user>/spark-vllm-docker"
  recipe_dir: "/home/<user>/spark-vllm-docker/recipes"
  llama_swap_config: "/home/<user>/llama-swap/config.yaml"

manager:
  host: "0.0.0.0"
  port: 9001

vllm:
  port_range: [8001, 8099]
  image: "vllm-node:latest"
  host: "127.0.0.1"

memory:
  os_reserve_gb: 12
  poll_interval_s: 5

research:
  perplexity_key_env: "PERPLEXITY_API_KEY"
  perplexity_model: "sonar-pro"
  deepseek_key_env: "DEEPSEEK_API_KEY"
  deepseek_model: "deepseek-chat"

wake_on_request: false
```

Adjust `paths` to the deploy user's home directory. The research agent needs
`PERPLEXITY_API_KEY` and `DEEPSEEK_API_KEY` in the environment.

### 3.4 Set up API keys

Create `~/.config/llmeister.env`:

```bash
cat > ~/.config/llmeister.env << 'EOF'
PERPLEXITY_API_KEY=pplx-xxxxxxxxxxxxxxxxxxxx
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
EOF
chmod 600 ~/.config/llmeister.env
```

### 3.5 Set up the vLLM recipe infrastructure

```bash
mkdir -p ~/spark-vllm-docker/recipes
mkdir -p ~/spark-vllm-docker/mods
mkdir -p ~/spark-vllm-docker/llmeister/generated-recipes

# Copy the run-recipe.py launcher (if bundled in the repo) or fetch it:
cp ~/Software/dgx-spark/scripts/run-recipe.py ~/spark-vllm-docker/
cp -r ~/Software/dgx-spark/scripts/mods/* ~/spark-vllm-docker/mods/
```

### 3.6 Build or pull the vLLM Docker image

The vLLM container must run on ARM64. Build from the Dockerfile or pull a
pre-built image:

```bash
# Option A: Pull pre-built (if available)
docker pull ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest
docker tag ghcr.io/spark-arena/dgx-vllm-eugr-nightly:latest vllm-node:latest

# Option B: Build locally
cd ~/spark-vllm-docker
# Write a minimal Dockerfile or use the one from the repo
docker build -t vllm-node:latest .
```

**Verify the image works:**

```bash
docker run --rm --gpus all vllm-node:latest nvidia-smi
docker run --rm --gpus all vllm-node:latest python3 -c "import vllm; print(vllm.__version__)"
```

---

## 4. Install LLMeister Systemd Service

```bash
mkdir -p ~/.config/systemd/user

cat > ~/.config/systemd/user/llmeister.service << 'EOF'
[Unit]
Description=LLMeister — centralized vLLM lifecycle/routing/dashboard
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/<user>/Software/dgx-spark/llmeister
Environment=HOME=/home/<user>
Environment=PATH=/home/<user>/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
EnvironmentFile=-%h/.config/llmeister.env
ExecStart=/home/<user>/.local/bin/uv run python -m llmeister
ExecStop=/bin/kill -TERM $MAINPID
Restart=on-failure
RestartSec=10
TimeoutStopSec=30

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now llmeister
systemctl --user status llmeister
```

Replace `<user>` with the actual deploy username.

**Verify:**

```bash
curl http://localhost:9001/api/status
curl http://localhost:9001/api/system
```

---

## 5. Install llama-swap Gateway

```bash
mkdir -p ~/Software/llama-swap
cd ~/Software/llama-swap

# Download latest ARM64 release
curl -L -o llama-swap https://github.com/<llama-swap-repo>/releases/latest/download/llama-swap-linux-arm64
chmod +x llama-swap

# Create config
cat > ~/llama-swap/config.yaml << 'EOF'
models: []
port: 8080
host: "0.0.0.0"
EOF
```

Models are added through the LLMeister dashboard (Edit → configure gateway_name and
aliases). LLMeister manages the llama-swap config via the recipe system.

Set up the systemd service:

```bash
cat > ~/.config/systemd/user/llama-swap.service << 'EOF'
[Unit]
Description=llama-swap OpenAI-compatible gateway
After=network.target

[Service]
Type=simple
ExecStart=/home/<user>/Software/llama-swap/llama-swap
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now llama-swap
```

---

## 6. Register Your First Model

### 6.1 Via the dashboard (recommended)

Open `http://<spark-ip>:9001` in a browser.

1. *Scan cache* — if HF models are pre-downloaded to `~/.cache/huggingface/hub/`,
   they'll appear under "Discovered"
2. Click **Research** on a model to auto-configure vLLM parameters via Perplexity+DeepSeek
3. After research completes, click **Approve** — the model moves to "Available"
4. Click **Start** to launch the container
5. Click **Edit** → set gateway_name and aliases for llama-swap routing

### 6.2 Via manual config

```bash
# Direct SQLite insert (LLMeister must be stopped or use the API)
sqlite3 ~/spark-vllm-docker/llmeister/llmeister.db "
INSERT INTO models (name, hf_model_id, config, state, port, updated_at)
VALUES (
  'qwen9b',
  'Intel/Qwen3.5-9B-int4-AutoRound',
  '{"hf_model_id":"Intel/Qwen3.5-9B-int4-AutoRound","container":"vllm-node","defaults":{"port":8001,"host":"0.0.0.0","tensor_parallel":1,"gpu_memory_utilization":0.4,"max_model_len":12288,"max_num_batched_tokens":16384,"max_num_seqs":8,"block_size":32},"env":{"VLLM_SERVER_DEV_MODE":"1"},"mods":["mods/fix-qwen3.5-autoround","mods/fix-sleep-wake-mamba"],"command":"vllm serve Intel/Qwen3.5-9B-int4-AutoRound --host {host} --port {port} --max-model-len {max_model_len} --max-num-batched-tokens {max_num_batched_tokens} --max-num-seqs {max_num_seqs} --block-size {block_size} --gpu-memory-utilization {gpu_memory_utilization} --kv-cache-dtype fp8 --load-format fastsafetensors --enable-chunked-prefill --enable-prefix-caching --enable-sleep-mode --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 -tp {tensor_parallel}","served_model_names":[],"aliases":[],"use_model_name":"Intel/Qwen3.5-9B-int4-AutoRound","gateway_name":"Qwen/Qwen3.5-9B"}',
  'STOPPED',
  8001,
  datetime('now')
);
"
```

Then start from the dashboard or API:
```bash
curl -X POST http://localhost:9001/api/qwen9b/start
```

---

## 7. Verify the Full Stack

```bash
# LLMeister
curl http://localhost:9001/api/status | jq

# vLLM backend (after starting a model)
curl http://localhost:8001/v1/models | jq

# Gateway
curl http://localhost:8080/v1/models | jq

# Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen9b","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'
```

---

## 8. Run Parameter Optimization

Once a model is AWAKE, trigger optimization:

```bash
curl -X POST http://localhost:9001/api/qwen9b/optimize/start \
  -H "Content-Type: application/json" \
  -d '{"message":"64 parallel doc processing, 8K in, 500-1K out, WebP, throughput priority"}'
```

Monitor:
```bash
# Status
curl http://localhost:9001/api/qwen9b/optimize/status

# History + plan
curl http://localhost:9001/api/qwen9b/optimize/history | jq
```

Or use the dashboard — click **Edit** → **🔧 Optimize**.

The optimizer will:
1. Gather system specs (GPU, memory, vLLM version)
2. Research optimal configs via Perplexity
3. Create a branching optimization plan
4. Execute experiments, restarting the model between each
5. Leave the best config running when done

Each iteration takes ~5-7 minutes (model restart + benchmark). Plan for 30-60
minutes for a full run.

---

## 9. Firewall & Network

The Spark should NOT be exposed to the public internet. Access is through:

- **WireGuard** — the Spark gets a static IP (e.g. `10.66.0.3`) on the mesh
- **SSH tunnel** — forward the dashboard port: `ssh -L 9001:localhost:9001 user@10.66.0.3`
- **Direct LAN** — if on the same network

Key ports:
| Port | Service | Exposure |
|---|---|---|
| 8080 | llama-swap gateway | WireGuard only |
| 9001 | LLMeister dashboard | localhost or tunnel |
| 8001-8099 | vLLM backends | localhost |

---

## 10. Maintenance

```bash
# Check all services
systemctl --user status llmeister llama-swap

# View logs
journalctl --user -u llmeister -f
journalctl --user -u llama-swap -f

# Restart after code updates
cd ~/Software/dgx-spark/llmeister && git pull && uv sync
systemctl --user restart llmeister

# Check disk (HF cache can grow large)
du -sh ~/.cache/huggingface/hub/
```

---

## Quick-Start Checklist

- [ ] Machine verified (aarch64, GB10 GPU, 121 GiB, Docker with GPU)
- [ ] User linger enabled (`loginctl enable-linger`)
- [ ] `uv` installed, `python3.12+` available
- [ ] Repo cloned to `~/Software/dgx-spark/llmeister`
- [ ] `config.yaml` edited with correct paths
- [ ] `~/.config/llmeister.env` with API keys
- [ ] `vllm-node:latest` Docker image pulled/built
- [ ] `spark-vllm-docker/` directory structure created
- [ ] LLMeister systemd service enabled and running
- [ ] llama-swap installed and running
- [ ] Dashboard accessible at `http://<ip>:9001`
- [ ] First model registered and serving
- [ ] Parameter optimization tested
