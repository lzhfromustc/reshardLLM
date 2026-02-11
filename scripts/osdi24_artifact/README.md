# OSDI'24 Llumnix artifact scripts

Scripts copied from the Llumnix OSDI'24 artifact (`artifact-llumnix/llumnix/artifact/`) for reproduction. They are **unchanged** from the artifact; adapt paths and config for your environment (e.g. 2 machines × 1 A10G).

## Layout

- **fig11_sharegpt/** — Figure 11 (Serving Performance) ShareGPT part: run scripts, configs, and plotting.
- **benchmarks-repo/** — Benchmark driver used by the experiments (e.g. `benchmark_throughput.py`).

## Running Figure 11 ShareGPT

See **fig11_sharegpt/README.md**. Run commands from **fig11_sharegpt/** so that `../../benchmarks-repo` resolves to this `benchmarks-repo/` directory.

## Dataset

Obtain the ShareGPT dataset (e.g. `sharegpt_gpt4_large.jsonl` or `sharegpt_gpt4.jsonl`) as in the original artifact (e.g. HuggingFace `shibing624/sharegpt_gpt4`). Place it where your config’s `dataset_path` points (scripts use `./sharegpt_gpt4_large.jsonl` or `./sharegpt_gpt4.jsonl` by default).

## Original artifact

- Source: `artifact-llumnix/llumnix/artifact/63_serving_performance/`
- Benchmarks: `artifact-llumnix/llumnix/benchmarks-repo/`
