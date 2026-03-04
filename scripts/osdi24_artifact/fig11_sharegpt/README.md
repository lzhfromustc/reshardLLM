# Figure 11 ShareGPT — How to run

## 1. Start Llumnix on both nodes

The benchmark assumes the server is already running. Start it on each node from `reshardLLM/scripts/`:

- **Head node:**
  ```bash
  cd /path/to/reshardLLM/reshardLLM/scripts
  bash simple_bench.sh 0 8000 0.95
  ```
- **Worker node:**
  ```bash
  cd /path/to/reshardLLM/reshardLLM/scripts
  bash simple_bench.sh 1 8000 0.95
  ```

Set `HEAD_NODE_IP` in `simple_bench.sh` if needed. Wait until the API is ready (e.g. `curl http://<HEAD_NODE_IP>:8000/is_ready`).

## 2. Optional: set env and dataset

In **config/serving_exp_dataset** the defaults are: `HEAD_NODE_IP=172.31.9.153`, `PORT=8000`, `IP_PORTS=${HEAD_NODE_IP}:${PORT}`, and a default `MODEL_PATH`. Override if your setup differs:

```bash
export HEAD_NODE_IP='172.31.9.153'
export DATASET_PATH='/path/to/sharegpt_gpt4.jsonl'   # optional
```

Put the ShareGPT file in `fig11_sharegpt/` (e.g. `./sharegpt_gpt4_large.jsonl`) or set `DATASET_PATH` before running.

## 3. Run the experiment (single run)

From **this directory** (`fig11_sharegpt/`):

```bash
cd /path/to/reshardLLM/reshardLLM/scripts/osdi24_artifact/fig11_sharegpt
bash run_part_sharegpt.sh
```

This runs **one** benchmark: conversation_mode `all`, QPS 3.5, ShareGPT dataset. Logs go under `./log/...` and a `.data` file is produced for plotting.

## 4. Plot (after at least one run)

```bash
cd plot && python plot_part_sharegpt.py --log-path ../log
```

By default the script uses the **latest** `.data` file (by file modification time) **for each unique (QPS, method)**. So with multiple QPS runs (e.g. 3.5, 3.75, 4.0, 4.25, 4.5), you get one point per QPS per method from its most recent run, and one figure with all those points. To **average** all runs instead:

```bash
cd plot && python plot_part_sharegpt.py --log-path ../log --average
```

Output: `figure11_sharegpt_only.png` in `plot/`.

---

## Output files (per run under `./log/...`)

| File | Source | Description |
|------|--------|-------------|
| **`.log`** | Benchmark stdout (tee) | Console output: throughput, CDF, and any warnings. |
| **`.data`** | `process_log_part_sharegpt.py` | Summary for plotting: Request/Prefill/Decode Mean & P99, Preemption Loss (or N/A). |
| **`_latency_info.json`** | Benchmark | Per-request latencies: `request_latencies`, `prefill_latencies`, `decode_latencies`, `inference_latencies`. |
| **`.log.npy`** | Benchmark | NumPy array of (timestamp, latency) pairs for **decode** per-token latencies; used for CDF/analysis. |
| **`_instance.csv`** | Server (Llumnix) | Instance-level metrics (GPU usage, instance count). **Not written** by current Llumnix → benchmark uses `avg_instance_num=0` and logs a warning. |
| **`_req.csv`** | Server (Llumnix) | Request events (prefill, killed, migrate_in, etc.) for preemption loss. **Not written** by current Llumnix → `.data` shows "Preemption Loss: N/A". |

**Request latency** in `.data` is **end-to-end** (queuing + inference). If the server does not return `inference_time` in the API response, the benchmark sets `inference_latencies` to 0, so **queuing is not broken out** and Request ≈ total time in queue + compute.
