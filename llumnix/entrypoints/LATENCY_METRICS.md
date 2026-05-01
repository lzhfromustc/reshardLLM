# Latency metrics: artifact-llumnix vs vLLM v1

## 1. RequestOutput and `total_inference_time`

The **installed** vLLM package (`site-packages/vllm/outputs.py`) defines `RequestOutput` **without** `total_inference_time`. That is the standard vLLM codebase.

**Artifact-llumnix** uses a **forked** vLLM living under `artifact-llumnix/llumnix/vllm/`. In that fork:

- `RequestOutput` is defined in `artifact-llumnix/llumnix/vllm/outputs.py` with constructor args including `total_inference_time: float`.
- It is built via `RequestOutput.from_seq_group(seq_group)`, which does `total_inference_time = seq_group.total_inference_time`.
- `SequenceGroup.total_inference_time` is accumulated in the **engine** in `_update_sequences_time()`: for each step where the request is scheduled, the engine does `seq_group.total_inference_time += t1_inference_end - t0_inference_begin` (see `artifact-llumnix/llumnix/vllm/engine/llm_engine.py`).

So `final_output.total_inference_time` exists only when the **artifact-llumnix** vLLM fork is used. ReshardLLM’s Llumnix uses the **vLLM v1** engine, which attaches `RequestOutput.metrics` (RequestStateStats) instead and does **not** set `total_inference_time` on RequestOutput.

## 2. Does artifact-llumnix `total_inference_time` include prefill?

**Yes.** In the engine’s `step()`:

- `t0_inference_begin = time.time()` before `_run_workers()` (and decode/stop).
- `t1_inference_end = time.time()` after `_stop_sequences()`.
- `_update_sequences_time(seq_groups, t0_inference_begin, t1_inference_end)` adds this span to **every** `seq_group` in that step.

A request appears in `seq_groups` in both its **prefill** step and in every **decode** step. So each of those steps adds that step’s wall time to that request’s `total_inference_time`. So **prefill and decode** are both included; it is the **sum of step run times** (no queue, no preemption gaps).

## 3. Does Llumnix (reshardLLM) use vLLM stats formulas?

**Partially.** ReshardLLM uses the **vLLM v1** engine. We currently:

- Expose **inference_time** from v1 `RequestStateStats`: `inference_time = last_token_ts - scheduled_ts` (same formula as in `vllm/v1/metrics/stats.py`).
- We do **not** yet expose **queued_time**, **prefill_time**, or **decode_time** from the same stats. The v1 engine computes them in `IterationStats.update_from_finished_request()` but they are not passed through to the API response. With the new changes we add those to the generate_benchmark response so the benchmark can log and compare them.

## 4. v1 formulas (from `vllm/v1/metrics/stats.py`)

- **queued_time** = `scheduled_ts - queued_ts`
- **prefill_time** = `first_token_ts - scheduled_ts`
- **decode_time** = `last_token_ts - first_token_ts`
- **inference_time** = `last_token_ts - scheduled_ts`

(all in seconds; timestamps from engine core events).
