# Deploying Sarvam-30B on an Azure ML Compute Instance (CPU-only)

Dev/test deployment of [Sarvam-30B](https://www.sarvam.ai/blogs/sarvam-30b-105b) (Apache 2.0, MoE — 30B total / 2.4B active params) served via [llama.cpp](https://github.com/ggml-org/llama.cpp) on a single Azure ML **compute instance**, with no GPU.

## Why this works

| Component | Status |
|---|---|
| Model weights | Official GGUF at [`sarvamai/sarvam-30b-gguf`](https://huggingface.co/sarvamai/sarvam-30b-gguf) — `sarvam-30b-Q4_K_M.gguf` (~19.6 GB) |
| llama.cpp support | `sarvam_moe` architecture merged mainline via [PR #20275](https://github.com/ggml-org/llama.cpp/pull/20275), shipped in release `b9093` (~May 2026). Any current build works — no fork/patch needed. |
| Compute target | An [Azure ML compute instance](https://learn.microsoft.com/en-us/azure/machine-learning/concept-compute-instance?view=azureml-api-2) is a fully managed, single-node Ubuntu VM with root/terminal access and broad Azure VM SKU support — you can build and run arbitrary binaries on it, including llama.cpp. |

Because it's a Mixture-of-Experts model, all 128 experts must be resident in memory even though only 2.4B params activate per token — so this is a memory-capacity/bandwidth problem more than a raw-FLOPs one, which is why a CPU-only VM is viable.

**Scope**: this is a dev/test setup, matching Microsoft's own guidance that compute instances are for dev/test workloads, not production serving. For production you'd containerize the same llama.cpp binary behind an Azure ML managed online endpoint (custom container) instead — not covered here.

## Prerequisites

- An existing Azure ML workspace.
- Quota for the `standardDDSv5` (or `standardEDSv5`) VM family in your target region.
- Outbound internet access from the compute instance (to reach GitHub and Hugging Face during setup) — if your workspace is VNet-isolated with restricted egress, allow those endpoints or pre-stage the files.

## VM sizing

| SKU | vCPU | RAM | Use case |
|---|---|---|---|
| `Standard_D16ds_v5` (budget) | 16 | 64 GiB | Fits Q4_K_M with headroom; lower throughput (fewer threads for `llama-server -t`) — fine if latency isn't critical |
| **`Standard_D32ds_v5`** (default) | 32 | 128 GiB | Comfortable headroom for Q4_K_M weights (19.6 GB) + KV cache + OS, better throughput |
| `Standard_E32ds_v5` | 32 | 256 GiB | If you want to run Q6_K/Q8_0 for higher quality, or a much larger context window |

Avoid anything below ~64 GiB RAM — too tight once you add KV cache and OS overhead on top of the ~20 GB weight file. No script changes are needed to switch SKUs: the systemd unit uses `-t $(nproc)`, which picks up whatever core count the chosen VM has.

## Storage layout

Azure VM **temp/local disk (`/mnt`) is not guaranteed to survive stop/deallocate**, but the **OS disk (120 GB, persistent)** does. Compute instances routinely stop/start (idle shutdown, schedules), so the build and model weights go on the **OS disk**, under the `azureuser` home directory — not `/mnt` — to avoid a ~20 GB re-download every restart.

```
/home/azureuser/sarvam/
├── llama.cpp/build/bin/llama-server   # built once
└── models/sarvam-30b-Q4_K_M.gguf      # downloaded once
```

## Setup mechanism

Azure ML compute instances support two kinds of [setup scripts](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-customize-compute-instance?view=azureml-api-2), both run as root:

- **Creation script** — runs once, when the instance is first provisioned. Used here for the one-time build + download.
- **Startup script** — runs every time the instance starts (including creation). Used here as a safety net to confirm the systemd service is active.

### Steps (Studio)

1. In [ml.azure.com](https://ml.azure.com) → your workspace → **Notebooks**, upload `creation-script.sh` and `startup-script.sh` (below) — filenames must end in `.sh`.
2. **Compute** → **+New** → choose the VM size (`Standard_D32ds_v5`).
3. On the **Applications** step of the creation form, toggle on both **creation script** and **startup script**, and browse to the two uploaded files.
4. The default setup-script timeout is 15 minutes, and Azure ML caps the override at **25 minutes max** — not enough to reliably fit the llama.cpp build plus a ~20 GB weight download. Set `DURATION=25m` (the ceiling; available as an ARM template / advanced-settings parameter on the creation script), and rely on the creation script backgrounding the actual work (below) rather than the timeout to cover it.
5. Create the instance. Watch progress under the instance's **Logs** tab (setup script output is mirrored to the Notebooks file share under `Logs/<compute-instance-name>`).

> CLI/YAML note: `az ml compute create` also supports a `setup_scripts` block in the compute-instance YAML, but the exact field syntax varies by CLI version — check `az ml compute create -h` and the `computeInstance.schema.json` for your installed extension before scripting this end-to-end. The Studio flow above is the verified path.

## Scripts

### `creation-script.sh`
Runs once. Installs build deps, then hands the build + download off to a **detached systemd unit** so the real work keeps running after this script's own process — and its 25-minute cap — exits.

```bash
#!/bin/bash
set -euo pipefail

# Runs once, as root, when the compute instance is created.
# NOTE: this script itself must return well within Azure ML's 25-minute
# creation-script cap. The build + ~20 GB download won't reliably fit in
# that window, so the actual work is launched as a transient systemd unit
# (systemd-run --collect) that survives past this script's exit.

apt-get update
apt-get install -y --no-install-recommends build-essential cmake git python3-pip

cat > /usr/local/bin/sarvam-setup.sh <<'SETUP'
#!/bin/bash
set -euo pipefail

# Everything under /home/azureuser must be owned by azureuser, and pip/hf
# downloads should land in that user's environment.
sudo -u azureuser -i <<'EOF'
set -euo pipefail

SARVAM_HOME="$HOME/sarvam"
mkdir -p "$SARVAM_HOME"
cd "$SARVAM_HOME"

pip install --user -U "huggingface_hub[cli]"

# Requires a llama.cpp build at/after release b9093 (sarvam_moe architecture
# support). A fresh clone of main is fine as of Aug 2026.
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_OPENMP=ON
cmake --build build --config Release -j"$(nproc)"

mkdir -p "$SARVAM_HOME/models"
"$HOME/.local/bin/huggingface-cli" download sarvamai/sarvam-30b-gguf \
  sarvam-30b-Q4_K_M.gguf \
  --local-dir "$SARVAM_HOME/models"
EOF

# Install the systemd unit (keep in sync with sarvam-llama.service below).
cat > /etc/systemd/system/sarvam-llama.service <<'UNIT'
[Unit]
Description=Sarvam-30B llama.cpp server (CPU-only)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/sarvam
ExecStart=/bin/bash -c '/home/azureuser/sarvam/llama.cpp/build/bin/llama-server \
  -m /home/azureuser/sarvam/models/sarvam-30b-Q4_K_M.gguf \
  -t $(nproc) \
  -c 8192 \
  --mlock \
  --host 127.0.0.1 --port 8080'
Restart=on-failure
RestartSec=5
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now sarvam-llama.service
SETUP
chmod +x /usr/local/bin/sarvam-setup.sh

# --collect cleans up the transient unit once it exits; the unit's own log
# stays queryable via `journalctl -u sarvam-setup` (see Troubleshooting).
systemd-run --unit=sarvam-setup --collect /usr/local/bin/sarvam-setup.sh
```

### `sarvam-llama.service`
Reference copy of the unit installed above — kept here for visibility/diffing; not uploaded separately.

```ini
[Unit]
Description=Sarvam-30B llama.cpp server (CPU-only)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/sarvam
ExecStart=/bin/bash -c '/home/azureuser/sarvam/llama.cpp/build/bin/llama-server \
  -m /home/azureuser/sarvam/models/sarvam-30b-Q4_K_M.gguf \
  -t $(nproc) \
  -c 8192 \
  --mlock \
  --host 127.0.0.1 --port 8080'
Restart=on-failure
RestartSec=5
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
```

`--host 127.0.0.1` keeps the server reachable only from inside the instance (via terminal, SSH tunnel, or Jupyter proxy) since compute instances aren't meant to be public-internet-facing. Change to `--host 0.0.0.0` only if you need other hosts in the same VNet to reach it, and lock that down with an NSG rule for port 8080.

### `startup-script.sh`
Runs on every start, including once right after the creation script during initial provisioning — at that point the creation script's background build/download (`sarvam-setup`, a detached `systemd-run` unit, see above) is normally still in progress, so the model/binary won't exist yet. The script must tell that apart from a genuine failure: it checks whether `sarvam-setup` is still running before treating missing artifacts as an error, otherwise a non-zero exit here gets reported as a **compute-instance creation failure** even though the build is proceeding normally in the background.

```bash
#!/bin/bash
set -euo pipefail

MODEL=/home/azureuser/sarvam/models/sarvam-30b-Q4_K_M.gguf
BIN=/home/azureuser/sarvam/llama.cpp/build/bin/llama-server

if [[ ! -f "$MODEL" || ! -x "$BIN" ]]; then
  if systemctl is-active --quiet sarvam-setup; then
    echo "sarvam-setup is still building/downloading — will be picked up on next start." >&2
    exit 0
  fi
  echo "ERROR: sarvam-llama prerequisites missing ($MODEL or $BIN not found)," \
       "and sarvam-setup is not running. Re-run the creation script." >&2
  exit 1
fi

systemctl daemon-reload
systemctl enable --now sarvam-llama.service
systemctl --no-pager status sarvam-llama.service
```

## Verification

From the compute instance's own terminal (Studio → Compute → your instance → **Terminal**):

```bash
# 1. Liveness
curl -s http://127.0.0.1:8080/health

# 2. Indic-language smoke test — this is the meaningful check, since the
#    sarvam_moe llama.cpp support went through a documented tokenizer fix
#    for Indic-script parity (SPM-style BPE vs GPT-2 byte-level).
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"नमस्ते, आप कैसे हैं?"}]}'

# 3. Service status / logs
systemctl status sarvam-llama.service
journalctl -u sarvam-llama -f
```

Confirm the response is coherent Devanagari text, not garbled tokens — that's the signal the architecture/tokenizer support is actually working, not just that a process is listening.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| Creation script times out | Model download exceeded 15-min default | Set `DURATION=60m` (or higher) on the creation script |
| Compute instance shows **"Create failed"**, but the setup-script logs themselves show no error | Startup script ran right after the creation script (as it does on initial provisioning) and exited 1 because `sarvam-setup`'s build/download was still in progress — the file checks in `startup-script.sh` predate that guard | Confirmed fixed by the `systemctl is-active --quiet sarvam-setup` check now in `startup-script.sh` above; if you're on an older copy of the script, update it and recreate the instance |
| `unknown model architecture 'sarvam_moe'` | llama.cpp built before release `b9093` | `cd ~/sarvam/llama.cpp && git pull && cmake --build build -j$(nproc)` |
| Service fails with OOM / gets killed | RAM too tight for chosen SKU + context size | Move to `Standard_E32ds_v5`, or lower `-c` (context size) |
| `/health` never returns 200 | Model still loading (large mmap) | Wait — a 20 GB weight file can take a minute-plus to page in on first request; check `journalctl -u sarvam-llama` for progress |
| Root disk filling up | 120 GB OS disk cap; build artifacts + weights + Ubuntu base | `df -h`; clear at least 5 GB before rebooting per Microsoft's compute-instance guidance, or move to a SKU that supports an attached data disk |

## Production note

This setup is intentionally scoped to a single dev/test compute instance. If you later need durable, autoscaled, production-grade serving, the same llama.cpp binary and GGUF weights can be containerized and deployed behind an **Azure ML managed online endpoint** (custom container / BYOC) instead — that path was scoped out of this doc but reuses everything built here.
