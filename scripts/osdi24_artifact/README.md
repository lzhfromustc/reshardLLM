# OSDI'24 Llumnix artifact scripts

Scripts adapted from the Llumnix OSDI'24 artifact for Figure 11 (Serving Performance) ShareGPT. They assume a **2-node** Llumnix setup (head + worker) and do **not** start the server; start Llumnix separately (e.g. with `simple_bench.sh`).

## Layout

- **fig11_sharegpt/** — Figure 11 ShareGPT: run scripts, configs, and plotting. Run commands from this directory.
- **benchmarks-repo/** — Benchmark driver (`benchmark_throughput.py`) used by the experiments; supports `--ip_ports` and Llumnix `/generate_benchmark` API.

## Running Figure 11 ShareGPT

1. **Start Llumnix** on both nodes (see **fig11_sharegpt/README.md**).
2. **Run the experiment** from `fig11_sharegpt/`:
   ```bash
   cd reshardLLM/scripts/osdi24_artifact/fig11_sharegpt
   bash run_part_sharegpt.sh
   ```
3. **Plot** (after at least one run): `cd plot && python plot_part_sharegpt.py --log-path ../log`

Full steps and optional env vars are in **fig11_sharegpt/README.md**.

## Dataset

Place the ShareGPT dataset (e.g. `sharegpt_gpt4_large.jsonl` or `sharegpt_gpt4.jsonl`) in `fig11_sharegpt/` or set `DATASET_PATH`; the config uses `./sharegpt_gpt4_large.jsonl` by default.

## Original artifact

- Source: `artifact-llumnix/llumnix/artifact/63_serving_performance/`
- Benchmarks: `artifact-llumnix/llumnix/benchmarks-repo/`
