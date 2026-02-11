# Figure 11 — ShareGPT (Serving Performance)

Scripts to reproduce the ShareGPT part of Figure 11 from the OSDI'24 Llumnix paper. Copied from the artifact **unchanged**; adapt config and paths for your setup (e.g. 2 machines, 1 A10G each).

## Directory layout

- **config/** — Sourced configs: `serving_exp_dataset` (ShareGPT dataset runs), `serving_exp`, `serving_exp_test`.
- **llumnix_exp_dataset** — Main experiment script for dataset-based runs (ShareGPT). Sources config, starts vLLM server, runs `../../benchmarks-repo/benchmark_throughput.py`, then `./plot/process_log_part_sharegpt.py`.
- **llumnix_exp** — Used by `run.sh` for non-dataset traces (128-128, 256-256, etc.).
- **run_part_sharegpt.sh** — Runs ShareGPT-only points (conversation_mode `all` at 5 QPS values).
- **run.sh** — Full Figure 11 (ShareGPT, BurstGPT, 128-128, 256-256, 512-512, 128-512, 512-128).
- **plot/** — `plot_part_sharegpt.py` (ShareGPT-only figure → `figure11_sharegpt_only.png`), `process_log_part_sharegpt.py`, plus generic `plot.py` / `process_log.py`.

## How to run (ShareGPT only)

1. **Run from this directory** (`fig11_sharegpt/`) so `../../benchmarks-repo` points to `scripts/osdi24_artifact/benchmarks-repo/`.

2. **Dataset:** Put ShareGPT data in this directory (or set paths in the commands below), e.g.:
   - `sharegpt_gpt4_large.jsonl` or `sharegpt_gpt4.jsonl`
   - Original artifact: e.g. `wget` from HuggingFace `shibing624/sharegpt_gpt4`.

3. **Config:** The scripts use hardcoded paths (e.g. model at `/mnt/data/models/...`, `CUDA_VISIBLE_DEVICES=0,1,2,3`, `instance_parallel_size=8`). For 2× A10G you will need to:
   - Adjust `config/serving_exp_dataset` (and/or env) for your model path, GPU visibility, and `instance_parallel_size`.
   - Or set env vars / wrapper scripts that override without editing the copied scripts if you want to keep them pristine.

4. **Run ShareGPT subset:**
   ```bash
   bash run_part_sharegpt.sh
   ```
   Or a single point, e.g.:
   ```bash
   bash ./llumnix_exp_dataset 'all' ./config/serving_exp_dataset 3.5 poisson 1.0 sharegpt './sharegpt_gpt4_large.jsonl' load 1 1 3 'sharegpt/Llumnix-all'
   ```

5. **Plot (after logs exist under `./log/`):**
   ```bash
   cd plot && python plot_part_sharegpt.py --log-path ../log
   ```
   (Adjust `--log-path` if your log dir is elsewhere; default is `../log`.)

## Dependencies

- Python deps used by the artifact (e.g. `vllm`, `pandas`, `matplotlib`). The plot script `process_log_part_sharegpt.py` imports `vllm.simulator.profiling`; for reshardLLM you may need the artifact’s vLLM tree or a compatible env.
- vLLM server and benchmark client as in the original artifact.

## Reference

- Original: `artifact-llumnix/llumnix/artifact/63_serving_performance/`
- Paper: Figure 11 — Serving Performance (ShareGPT trace).
