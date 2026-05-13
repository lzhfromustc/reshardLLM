# Master notes file — PP-reshard research session

> The user pinned the entire research session to **only this file**. All sections below
> are appended over time. No other files are touched.

---
## TL;DR (60-second read on your return)

This file is the master notes for the PP-reshard project. Two work sessions
so far:

**Session 1 (structural mapping)**: 9 parallel Explore agents over the four
codebases. Findings in §3. Refined PP-reshard research idea in §2.

**Session 2 (deep code reads + implementation plan)**: read 12 critical
files end to end myself for line-level accuracy + 5 deep-dive agents on
NIXL, P2P-NCCL, ubatching/connector hooks, disagg-PD lifecycle, and example
connectors. Wrote a concrete step-by-step implementation plan in §4.

**The plan in one paragraph**: Build PP-reshard on the **vLLM v0 backend**
(`llumnix/backends/vllm/`) first, since the v1 backend is currently a
stub. Each instance runs `pipeline_parallel_size = 1` (full model loaded);
"PP=2 across instances" is a Llumnix-level concept implemented by per-request
layer masking. Add a small `layer_range` parameter to Llumnix's existing
`RayColMigrationBackend.do_send`/`do_recv` (which already iterates layer-by-layer
on `migration_backend.py:384`) — this is the biggest functional change for
the smallest code edit. Reshard protocol mirrors Llumnix's existing
multi-stage migration but commits as HEAD/TAIL split instead of full move.
Hidden-state crossing is Ray-RPC in Phase 1 (correctness), NCCL p2p in
Phase 3 (performance). Manual API in Phase 1, scheduler-driven policy
in Phase 4. Port to v1 / `KVConnectorBase_V1` in Phase 6.

**Key insight from the deep reads** (in §4.4 Step 3.5): vLLM blocks are
*per-position*, not per-layer — a block holds KV for all L layers' worth
of one 16-token chunk. So PP-reshard cannot literally "free upper-layer
memory" on SRC without changing vLLM's block layout. That changes the
demo emphasis from "OOM avoidance via memory reclaim" to "preemption
avoidance via load offloading" — both still work, but worth re-examining
the paper claim.

**Risks worth your attention before we start coding** (§4.9.4):
1. R1 — vLLM's `LlamaModel.forward` doesn't natively accept a layer mask.
   Phase 1 Step 1.8 needs to either patch it or subclass it.
2. R3 — Two independent asyncio loops + Ray RPC may have a latency floor
   that limits per-iteration hybrid PP throughput. Profile early.
3. The Step 3.5 insight above.

**What's still queued for you** (§99): 15 design questions (Q2),
~44 files I want to read in Round 2 (Q4), 5 follow-up agents (Q5),
Q1 permissions ask. Now there's also Q8 — questions specific to the
implementation plan.

## Table of contents

- [§0. Session ground rules](#0-session-ground-rules)
- [§1. Permission preflight](#1-what-i-need-to-run-and-why-permission-preflight)
- [§2. The PP-reshard idea (refined)](#2-the-pp-reshard-research-idea-my-current-understanding)
  - [§2.1 First-cut implementation strategy](#21-first-cut-implementation-strategy-for-discussion-with-user)
- [§3. Findings](#3-findings-filled-in-as-agents-return)
  - [§3.1 reshardLLM (main fork)](#31-reshardllm-main-fork) → §3.1.1 layout, §3.1.2 migration, §3.1.3 scheduler/policy, §3.1.4 verify list
  - [§3.2 vllm 0.6.3](#32-vllm-063-the-version-llumnix-currently-targets) → §3.2.1–§3.2.6
  - [§3.3 vllm-latest](#33-vllm-latest) → §3.3.1 layout, §3.3.2 v1 engine, §3.3.3 PP, §3.3.4 KV connector, §3.3.5 disagg PD, §3.3.6 verify list
  - [§3.4 artifact-llumnix (light)](#34-artifact-llumnix-done--light-pass)
- **[§4. Concrete implementation plan](#4-concrete-implementation-plan)**
  - §4.0 Decisions, risks, phase overview
  - §4.1 Phase 0 — environment & baseline (5 steps)
  - §4.2 Phase 1 — MVP, 4–6 weeks (10 steps, detailed)
  - §4.3 Phase 2 — correctness (6 steps)
  - §4.4 Phase 3 — performance (5 steps, with caveats)
  - §4.5 Phase 4 — auto-trigger policy (sketch)
  - §4.6 Phase 5 — merge back to PP=1 (sketch)
  - §4.7 Phase 6 — port to v1 (sketch)
  - §4.8 Phase 7 — evaluation (sketch)
  - §4.9 Cross-cutting (data structures, tests, knobs, risks, non-goals)
- **[§5. Investigation: Llumnix goodput collapse on H100 multi-node TP](#5-investigation-plan-llumnix-goodput-collapse-on-high-end-multi-node-tp-clusters)** ← NEW
  - §5.1 What the symptom tells us up front
  - §5.2 Info to gather from the researchers (15 questions, free)
  - §5.3 10 hypotheses ranked + signatures
  - §5.4 Free-of-charge work to do before booking cluster
  - §5.5 Single-node experiments (10 sub-steps)
  - §5.6 Multi-node experiments (4 sub-steps)
  - §5.7 Instrumentation to add up-front
  - §5.8 Decision tree
  - §5.9 Cluster-hours budget (best 3 / median 6 / worst 12 hours)
  - §5.10 Pre-flight checklist
  - §5.11 Prior probability table
  - §5.12 Things to ask the researchers
- [§99. Queued for user return](#99-queued-for-user-return)
  - Q1 permissions; Q2 general design questions; Q3 commands skipped; Q4 file-read queue;
    Q5 Round-3 agents; Q6 unfinished; Q7 status; Q8 plan-specific questions (NEW)

---

# §0. Session ground rules

- **Write target**: only `/home/ubuntu/reshardLLM/notes/00_permission_test.md` (this file).
- **No bash**, no edits to any other file, no commits, no network I/O.
- While the user is away: I do not invoke tools that require permission. I write
  findings here and queue any commands I want to run in §99 ("Queued for user return").
- Llumnix paper: already read end-to-end (PDF was attached to the first user message).
  I won't re-read the local copy unless I find a specific gap.

---

# §1. What I need to run, and why (permission preflight)

The user offered to approve a batch of tool calls in one go before stepping away. I'm
going to issue ~10 tool calls now, all read-only. I will not issue more after this batch
until the user is back.

## 1a. Tool classes I will use

1. **Agent tool, subagent_type=Explore** (read-only, all tools except edits).
   I use Explore to map directories I haven't read and to gather the names of files
   I should read myself. Each Explore agent runs in its own context, so it does not
   bloat my main context with raw file contents.
2. **Read tool** on specific files I know I need.
3. **Edit tool** on this file only (already auto-allowed).

I will NOT use: Bash, Write (other files), git, network, anything destructive. Explore
agents may internally use grep/find/ls — that's fine because those are sandbox-local.

## 1b. The Round-1 batch I want to fire now (10 Explore agents)

These can all run in parallel; none depends on another.

**reshardLLM (the main Llumnix fork)** — 3 agents:
- A1 — Top-level map of `reshardLLM/reshardLLM/`: directory tree, entry points,
  benchmarks, configs, fork-vs-upstream diff signal.
- A2 — Migration mechanism deep-dive in `reshardLLM/reshardLLM/llumnix/`: handshake
  (PRE-ALLOC/ACK/ABORT/COMMIT), sender/receiver, KV transfer (Gloo / NCCL / other),
  multi-stage pipeline, downtime path, integration with the inference engine.
- A3 — Scheduler/llumlet/policy deep-dive: global scheduler, llumlet, virtual usage
  + freeness (Algorithm 1), dispatching, migration triggering, auto-scaling, priority
  headroom, configs.

**vllm 0.6.3 (the version Llumnix currently targets)** — 2 agents:
- A4 — Engine + scheduler + worker + model_runner: where is the per-step loop, where
  does continuous batching happen, where is the PP send/recv, how does a worker pick
  its layer slice when PP>1.
- A5 — KV cache & block manager & attention metadata: block layout (per-layer or
  unified?), allocation/free, prefix caching, swap-in/swap-out, PagedAttention metadata,
  attention-backend kernels.

**vllm-latest (the eventual base; the user said: "read migration & disagg PD carefully — we will borrow")** — 4 agents:
- A6 — Top-level map of `vllm/vllm-latest/vllm/`, distinguish v0 vs v1 engine, list
  config dataclasses, list executors and parallel layouts.
- A7 — v1 engine architecture: engine core, scheduler, model_runner, executor, IPC,
  PP forward path in v1.
- A8 — **KV-connector layer (CRITICAL — borrow target)**: connector base class, every
  concrete impl (NIXL, LMCache, Mooncake, P2P NCCL, shared memory, etc.), worker hooks,
  block-transfer lifecycle, handshake/metadata protocol.
- A9 — **Disaggregated prefill-decode (CRITICAL — borrow target)**: how a request is
  split between prefill instance and decode instance, how its KV cache moves, scheduler
  changes, role config, anything labeled "disagg" / "remote prefill" / "PD".

(`artifact-llumnix` is already mapped — see §3.4.)

## 1c. After Round 1 — what I'd like to do but will queue

I will queue (NOT run) these in §99:

- **Round 2 (deep file reads with my own Read calls)** on the specific files identified
  by Round 1 agents. ~30-60 file reads, no agents.
- **Round 3 (cross-cutting design questions)**: 2-3 more Explore agents to answer
  things like "what's the cleanest hook point in vllm-latest's v1 model runner to
  inject PP-reshard?" and "how does the PP rank layout differ between 0.6.3 and latest?"

If the user approves Agent and Read globally for this session, Round 2 reads I can do
myself without re-prompting; Round 3 agents I will queue and ask permission for on
the user's return.

---

# §2. The PP-reshard research idea (my current understanding)

**Goal**: reduce tail latency, OOM, and preemption rate beyond what Llumnix achieves
with whole-request migration.

**Mechanism**: instead of (or in addition to) migrating an entire request from instance
A → B, *split a single request along the layer/PP dimension*:
- A request was running PP=1 on A. Its KV cache spans all L layers, all on A.
- Pick a split point M (1 ≤ M < L). Keep KV[0:M] on A. Migrate KV[M:L] to B.
- From this point on, this request is PP=2 across {A, B}. Forward pass:
  hidden = A.forward(layers[0:M], input); B.forward(layers[M:L], hidden) → token.
- Meanwhile, A and B keep serving their other PP=1 requests. So each forward iteration
  on A is a *hybrid batch* containing PP=1 requests + the first half of PP=2 requests.
  Same on B with the second half.

**Why this could win**:
- *OOM/preempt avoidance*: when A is about to OOM on a long sequence, instead of
  preempting or wholesale migrating, you spill half the KV cache. Half-the-cache
  goes farther than full migration in two ways: (a) you only need ~½ the free memory
  on the destination, and (b) only ½ the bytes need to move during the handover.
- *Defragmentation finer-grained*: you can de-frag without uprooting the request.
- *Workload balancing*: if A's batch is compute-heavy and B has spare cycles,
  resharding shifts compute too, not just memory.
- *Tail latency*: short downtime + smaller bytes-moved should beat full migration on
  long sequences (Llumnix's downtime is constant in seq length, but its *total bytes
  moved* still grows; PP-reshard moves ½ of that).

**Why this is hard** (refined after Round-1 reads):

1. **Layer slicing in stock vLLM is baked in at load time.** `make_layers()` /
   `PPMissingLayer` mean a worker with PP rank `r` literally doesn't have the
   weights for layers outside `[r*L/pp, (r+1)*L/pp)`. ⇒ For our hybrid setting
   we must run with `pipeline_parallel_size=1` so every instance loads the full
   model, and dynamically decide at run time which layers to execute per-request.
   Trade-off: every instance pays full model-weight memory cost. (This is
   acceptable because in practice each Llumnix instance is already PP=1, with
   model fitting in single GPU or TP-sharded across one node.)

2. **Continuous batching assumes one forward graph per step.** The model_runner
   currently calls `model.forward()` once per step for the whole batch.
   For hybrid serving we need either:
   - **Option A (per-request layer mask)**: same single `model.forward()` call,
     but each transformer layer checks `forward_context.layer_subset_mask[req_idx]`
     and either runs or no-ops on that request's row. Intrusive in attention
     metadata building (slot_mapping, block_tables must skip masked rows for
     skipped layers), but minimal in scheduling.
   - **Option B (sub-batching)**: split each step into two sub-batches —
     "full-layer batch" (PP=1 reqs + first half of PP=2 reqs) and
     "upper-layer batch" (PP=2 second-half reqs). Each sub-batch goes through
     its own attn_metadata and forward call. **vllm-latest already has the
     `ubatch_slices` machinery in `gpu_model_runner._determine_batch_execution_and_padding`**
     (per A7) and `step_with_batch_queue()` in EngineCore, so the primitive
     exists. **Option B is likely the right choice — it isolates the strange
     forward-half-the-layers logic in one branch without polluting the kernel.**

3. **Cross-instance hidden state shipping.** When A runs layers 0..M for request R,
   the hidden state must travel to B for layers M..L on the same iteration. This
   is exactly what `KVConnectorBase_V1.start_load_kv` / `wait_for_layer_load` do
   *for KV cache*. Naively reusing them for hidden states is harder because
   hidden states are per-step (not append-only), but the side-channel /
   side-stream pattern is identical. NIXL's `example_hidden_states_connector.py`
   exists — that's a hint hidden-state transport via NIXL is feasible.

4. **Layer-aware KV transfer.** The connector metadata must specify "transfer
   layers M..L of request R" and the worker-side `start_load_kv` must iterate
   only that range. **The existing per-layer iteration loop in vLLM 0.6.3
   migration (`for layer_idx in range(num_layers)` in `RayCol.do_send`) and
   the per-layer hooks in vllm-latest connectors give us this structure
   already.** The change is minor: thread a `(layer_start, layer_end)` tuple
   through the metadata.

5. **Request state machine extension.** Llumnix's `MigrationStatus` enum
   ({SRC, DST, FINISHED, ABORTED_SRC, ABORTED_DST}) needs new states like
   `RESHARDED_AS_HEAD` (this instance keeps layers 0..M) and `RESHARDED_AS_TAIL`
   (this instance owns layers M..L). The `LlumnixRequest` needs new fields
   `pp_reshard_partner: Optional[InstanceId]`, `pp_reshard_split_layer: int`,
   and a phase flag.

6. **Iteration synchronization between A and B.** Stock vLLM PP synchronization
   relies on `pp_broadcast` / `pp_recv` (vllm-latest) within a single
   PP-process-group. After a PP-reshard, A and B are not in the same
   process group (they were originally separate Llumnix instances). We need
   a *side-channel PP* — i.e., an inter-instance NCCL or NIXL pipe used only
   for this resharded subset. **Equivalent: treat the resharded request as
   if it's a "remote prefill is done, now decode" disagg PD step, but per
   iteration.** This is the closest existing pattern.

7. **Merge-back / reverse path.** If B is being drained or A has freed enough
   memory, we want to consolidate the layer-tail back onto A. This is the
   inverse direction of the migration; symmetric implementation.

8. **Refcount / ownership.** Llumnix's `RefCounter` is single-machine. Because
   PP-reshard *splits* a request (different layers on different machines), each
   half lives entirely on one machine; refcount stays single-machine per half.
   ✅ this is easier than full migration in this regard.

9. **Attention kernel correctness.** The kernel reads
   `kv_cache[block_tables[s, t//bsz], t%bsz, ...]`. As long as the upper-layer
   instance has its own consistent `block_tables[req_id]` and `kv_cache[layer_i]`
   for layers `M..L`, it can run its half of the forward independently. The
   tricky part is that A and B must be processing the same iteration of R at
   the same time — synchronization, not memory layout, is the hard part.

10. **Sampling.** Only the last PP rank samples. After PP-reshard, B (the
    upper-layer half) is the last rank. So sampling moves from A to B after
    reshard. This is the same flow as last-rank sampling in stock PP.

I'll keep refining as I do the targeted reads in §99.

### §2.1 First-cut implementation strategy (for discussion with user)

Given Round-1 reads, my best initial design is:

- **Base on vllm-latest's v1 engine** — that's where the user said we'd end
  up anyway, and it already has the layer-pipelined connector primitives.
- **Reuse `KVConnectorBase_V1`** as the transport for both the one-time
  KV-half ship at reshard time *and* the per-iteration hidden-state pipe.
- Implement a new connector `LlumnixPPReshardConnector` (subclass) that
  understands `(request_id, layer_range)` instead of just `(request_id)`.
- Add a new request state in v1 `Request`: `RESHARD_PARTIAL_HEAD` /
  `RESHARD_PARTIAL_TAIL` (plus `WAITING_FOR_RESHARD_KV` for the brief moment
  during which the upper-layer KV is in transit).
- Use the **`ubatch_slices`** machinery in `GPUModelRunner` so that the
  hybrid step has two sub-batches: full-layer and upper-half-only.
- Keep Llumnix's `Manager` + `GlobalScheduler` + `Llumlet` architecture for
  cluster-level control. Add a new policy `PPReshardPolicy` alongside
  `Defrag`/`Balanced`.
- The "merge back" path is symmetric to forward reshard; reuses same plumbing.

Open questions for the user are in §99-Q2.

---

# §3. Findings (filled in as agents return)

## §3.1 reshardLLM (main fork)

> Source for this section: Round-1 Explore agents A1, A2, A3. Cited file paths
> need a direct Read pass to verify before any code changes; flagged with [V] where
> verification is most important. All paths are relative to
> `/home/ubuntu/reshardLLM/reshardLLM/` unless otherwise noted.

### §3.1.1 Top-level layout

```
reshardLLM/
├── benchmark/                       # measurement scripts + result data
│   ├── benchmark_serving.py         # main throughput/latency benchmark
│   ├── plot_instance_metrics.py
│   ├── benchmark.npy                # cached numpy result
│   └── tmp-results/
├── configs/                         # YAML deployment configs
│   ├── bladellm.yml
│   ├── vllm_head.yml
│   └── vllm_worker1.yml
├── llumnix/                         # main Python package (source)
│   ├── __init__.py                  # public API surface
│   ├── manager.py                   # global request Manager (dispatch + migration)
│   ├── scaler.py                    # auto-scaler
│   ├── arg_utils.py                 # CLI / engine-args
│   ├── constants.py
│   ├── envs.py
│   ├── version.py
│   ├── backends/                    # backend abstraction layer
│   │   ├── backend_interface.py     # BackendInterface (ABC) + BackendType enum
│   │   ├── migration_backend_interface.py
│   │   ├── vllm/                    # vLLM 0.x backend impl
│   │   │   ├── llm_engine.py        # LLMEngineLlumnix(_AsyncLLMEngine), BackendVLLM
│   │   │   ├── scheduler.py         # SchedulerLlumnix(Scheduler)
│   │   │   ├── executor.py          # LlumnixRayGPUExecutor(RayGPUExecutorAsync)
│   │   │   ├── worker.py            # MigrationWorker(Worker)
│   │   │   ├── migration_backend.py # RayRpc + RayCol KV transfer impls
│   │   │   └── sequence.py          # SequenceGroupLlumnix wrapper
│   │   ├── vllm_v1/                 # vLLM v1 backend impl (newer; thinner)
│   │   │   └── async_core.py        # async EngineCore wrapper
│   │   └── bladellm/                # BladeLLM backend (gRPC, Alibaba-internal)
│   │       ├── llm_engine.py
│   │       ├── migration_backend.py
│   │       └── proto/
│   ├── entrypoints/                 # API server + CLI entry points
│   │   ├── setup.py                 # Ray cluster bootstrap
│   │   ├── client.py
│   │   ├── api_server_actor.py
│   │   ├── vllm/                    # vLLM serve entry
│   │   │   ├── api_server.py        # FastAPI OpenAI-compatible
│   │   │   ├── client.py
│   │   │   ├── arg_utils.py
│   │   │   ├── serve.py             # CLI main
│   │   │   └── register_service.py
│   │   ├── vllm_v1/
│   │   └── bladellm/
│   ├── global_scheduler/            # cluster-level policies
│   │   ├── global_scheduler.py      # GlobalScheduler (top-level)
│   │   ├── dispatch_scheduler.py
│   │   ├── dispatch_policy.py       # Load / Balanced / Queue / RoundRobin
│   │   ├── migration_scheduler.py
│   │   ├── migration_policy.py      # Balanced / Defrag (paper's "defrag")
│   │   ├── migration_filter.py
│   │   ├── scaling_scheduler.py
│   │   └── scaling_policy.py        # AvgLoad / MaxLoad / MinLoad
│   ├── llumlet/                     # per-instance coordinator
│   │   ├── llumlet.py               # Ray actor wrapper
│   │   ├── migration_coordinator.py # multi-stage state machine
│   │   ├── local_migration_scheduler.py
│   │   └── request.py               # LlumnixRequest (state + migration metadata)
│   ├── instance_info.py             # InstanceInfo dataclass (load report)
│   ├── load_computation.py          # BaseLoad hierarchy (KvBlocksRatio, RemainingSteps, AdaptiveDecodeBatch)
│   ├── internal_config.py           # GlobalSchedulerConfig, MigrationConfig
│   ├── config/
│   │   ├── config.py                # YAML loader (LlumnixConfig)
│   │   ├── default.py               # _C global defaults
│   │   └── utils.py
│   ├── queue/                       # request output queue
│   │   ├── zmq_server.py
│   │   ├── ray_queue_server.py
│   │   ├── zmq_client.py
│   │   └── queue_type.py
│   ├── metrics/
│   ├── logging/
│   └── examples/
├── scripts/
│   ├── simple_bench.sh
│   ├── start_4_tp2_instances.sh
│   └── osdi24_artifact/
├── tests/
│   ├── unit_test/                   # 38 files
│   └── e2e_test/                    # incl. test_migration.py, test_dynamic_pd.py
├── docs/
│   ├── Quickstart.md
│   ├── Arguments.md
│   ├── Prefill-decode_Disaggregation.md   # PD already exists in this fork!
│   └── pdd_design.png
├── tools/docker/
├── requirements/{vllm,vllm_v1,bladellm}.txt
└── setup.py
```

**Public API** (from `llumnix/__init__.py`): `Scaler`, `Manager`, `Llumlet`, the args
dataclasses (`ManagerArgs`, `InstanceArgs`, `LaunchArgs`, `EntrypointsArgs`), enums
`BackendType` (VLLM, VLLM_V1, BLADELLM, SIM_VLLM), `LaunchMode` (LOCAL, GLOBAL),
`QueueType`, queue servers, `ServerInfo`, helpers `launch_ray_cluster()`,
`connect_to_ray_cluster()`, `init_scaler()`.

**vLLM integration pattern**: clean subclassing through `BackendInterface`, no
monkey-patching observed. Concrete subclasses:
- `LLMEngineLlumnix(vllm.engine.async_llm_engine._AsyncLLMEngine)`
- `SchedulerLlumnix(vllm.core.scheduler.Scheduler)`
- `LlumnixRayGPUExecutor(vllm.executor.ray_gpu_executor.RayGPUExecutorAsync)`
- `MigrationWorker(vllm.worker.worker.Worker)`

**BladeLLM**: Alibaba-internal gRPC engine; we'll mostly ignore for the PhD project
(we focus on vLLM path). But the existence of the `BackendInterface` abstraction is
useful — it shows the cleanest seam for adding our PP-reshard backend hooks.

**Fork-vs-upstream signal** (per A1): no `pp_reshard` / `reshard` / `split` strings
in the tree. Recent activity is mostly debugging, latency metrics docs, env params,
and adaptive PD work. The fork hasn't started PP-reshard implementation yet —
**we're starting from a clean Llumnix base + we have v1 hooks already wired**.
*Notable*: `tests/e2e_test/test_dynamic_pd.py` exists, and there's
`docs/Prefill-decode_Disaggregation.md`. So Llumnix has *some* PD-disagg
already — we need to read this doc and that test.

### §3.1.2 Migration mechanism deep-dive (A2)

Files (I'll cite by path / class / role):

| Path | Class / Role |
|---|---|
| `llumnix/llumlet/migration_coordinator.py` | `MigrationCoordinator` — owns multi-stage state machine, handshake, abort logic |
| `llumnix/llumlet/local_migration_scheduler.py` | `LocalMigrationScheduler` — picks which requests to migrate (LCR/LR/SR/FCW/FCWSR) |
| `llumnix/llumlet/llumlet.py` | `Llumlet` — Ray actor wrapper, exposes migration RPCs |
| `llumnix/backends/migration_backend_interface.py` | `MigrationBackendInterface` ABC |
| `llumnix/backends/vllm/migration_backend.py` | `RayRpcMigrationBackend` (numpy/shared-mem) and `RayColMigrationBackend` (Gloo/NCCL collectives) |
| `llumnix/backends/vllm/worker.py` | `MigrationWorker` — `do_send` / `do_recv` / `recv_cache`, seq_group_metadata caching |
| `llumnix/backends/vllm/scheduler.py` | `SchedulerLlumnix` — pre-allocated block tables, incremental-block extraction |
| `llumnix/backends/vllm/llm_engine.py` | `BackendVLLM` + `LLMEngineLlumnix` — `pre_alloc_cache`, `commit_dst_request`, `send_cache` |
| `llumnix/backends/vllm/sequence.py` | `SequenceGroupLlumnix` — bridges vLLM SequenceGroup with `LlumnixRequest` migration state |

**Successful-migration call graph** (per A2; **[V]** verify before coding):

```
GlobalScheduler                     (initiator, Manager-level)
  └─ Llumlet.migrate_out(dst_actor, dst_instance_id)               [Ray remote]
      └─ MigrationCoordinator._migrate_out_multistage()
           Stage 1..N-1 (live, pipelined):
             ├─ backend.get_request_incremental_blocks()           # newly-generated since last stage
             ├─ _dst_pre_alloc_cache(req_id, num_blocks, ...)      [Ray remote → dst]
             │     └─ SchedulerLlumnix.pre_alloc_cache()
             │           creates a BlockTable on dst, reserves GPU blocks
             ├─ _send_cache(src_blocks, dst_blocks)
             │     └─ BackendVLLM.send_cache()                     [Ray remote → dst]
             │           └─ MigrationWorker.recv_cache()
             │                 └─ MigrationWorker.do_recv()        # layer-by-layer, migration_stream
             ⮕ continues until threshold
           Stage N (blocking, suspend):
             ├─ remove_running_request()                           # SUSPEND on src
             ├─ _migrate_out_onestage()                            # final block transfer
             ├─ _dst_commit_dst_request(request, block_table)
             │     └─ commit_dst_request()
             │           binds seq_id, installs block_table, enqueues request
             └─ free_src_request()                                 # cleanup on src
```

**Handshake protocol (PRE-ALLOC / ACK / ABORT / COMMIT) → code**:

| Paper term | Implementation |
|---|---|
| **PRE-ALLOC** | `_dst_pre_alloc_cache()` (migration_coordinator.py ≈ L376–402). Args: `request_id`, `block_num`, `token_ids`, `is_first_stage`. Allocates blocks via `SchedulerLlumnix.pre_alloc_cache()` (scheduler.py ≈ L131–159). Returns dst block IDs. |
| **ACK** | Implicit. Destination returns success in pre-alloc response (its block list). Failure (OOM) returns `success=False` (line 147 scheduler.py) → coordinator sees the error and triggers abort. |
| **TRANSFER** | `_send_cache()` (≈ L421–435). Routes via `BackendVLLM.send_cache()` → `MigrationWorker.recv_cache()`. Last stage also carries `send_worker_metadata=True` to ship sampling state (worker.py ≈ L267–270). |
| **ABORT** | Multiple call sites: status enum `MigrationStatus.ABORTED_SRC` / `ABORTED_DST` / `FINISHED`. Free cleanups: `_dst_free_pre_alloc_cache()` (≈ L210), `reset_migration_states_src()` (request.py ≈ L73–89). |
| **COMMIT** | `_dst_commit_dst_request()` (≈ L446–459). Dst binds seq_id, installs block table, enqueues. Src then frees its blocks. |

**Status transitions** on the request:
- vLLM `RUNNING` → Llumnix `RUNNING_MIGRATING` (scheduler.py ≈ L110)
- back to `RUNNING` on commit, **or** rolled back to running queue on abort.

**KV transfer mechanism**:
- Two backends — chosen in `get_migration_backend()` (migration_backend.py ≈ L414–458).
  - `RayRpcMigrationBackend` — uses a single pinned-memory `dummy_cache`
    (≈ L103–108), packs `num_migration_buffer_blocks` blocks into it, ships as
    numpy via Ray shared memory (≈ L182). Effectively does CPU staging.
  - `RayColMigrationBackend` — Gloo or NCCL collectives via `col.send` / `col.recv`
    (≈ L390, L407). Layer-by-layer per `for layer_idx in range(num_layers)`.
- **Block fusion**: yes (the paper's "block fusion" optimization). Blocks are
  packed into the dummy_cache before transmission rather than sent individually.
- **Stream**: a separate `migration_stream` (CUDA stream) is used; `swap_blocks`
  on this stream and `synchronize` (≈ L181). This is what keeps inference
  uninterrupted.
- **GPU↔CPU staging** is needed for Gloo (CPU-only); not strictly needed for
  NCCL but Llumnix uses it anyway, see paper §5 ("Using Gloo needs to copy the
  KV cache between CPU and GPU memory, which is done in another CUDA stream...").

**Multi-stage pipeline (Stage-0..N)**:
- Loop in `_migrate_out_multistage()` (≈ L259–270):
  ```python
  while stage_count < migration_max_stages:
      status = _migrate_out_onestage(..., is_first_stage=(stage_count==1))
      if MigrationStatus.is_finished(status): return status
  ```
- **Incremental block computation**: `get_request_incremental_blocks(req, pre_stage_num_blocks)`
  (scheduler.py ≈ L97–104) returns `block_table[pre_stage_num_blocks:]`. The
  cumulative count is tracked in `migrate_out_request.stage_num_blocks_list`.
- **Non-last stages** transfer `incremental_blocks[:-1]` (≈ L291); the
  currently-being-written block is excluded (the "append-only safety" exploit).
- **Last stage** suspends, then transfers all remaining (incl. the in-flight block).
- **Downtime window** is therefore the period between `remove_running_request()`
  on the source and `add_running_request()` on the destination — this is what
  the paper says is "negligible regardless of sequence length".
- `MIGRATION_MAX_STAGES = 3` (default in config/default.py L156); bigger pipelines
  abort with `ABORTED_SRC`.

**Edge cases handled** (per A2):
1. Preempted on source: `should_abort_migration()` (request.py ≈ L135–139)
   checks `last_preemption_time`; abort with `ABORTED_SRC`, free dst blocks.
2. EOS during migration: abort, cleanup `reset_migration_states_src()`.
3. Dst OOM mid-migration: `pre_alloc_cache()` returns failure → `ABORTED_DST`,
   src restores request to running queue.
4. Instance death: detected by Ray; `watch_migrate_in_request_wrapper`
   (≈ L64–136) monitors `pending_migrate_in_request_time`; timeout triggers
   cleanup. `asyncio_wait_for_with_timeout()` (≈ L386).
5. Two requests racing for same dst: not explicitly serialized; `pre_alloc_cache`
   ordering provides the implicit serialization.

**Engine integration touchpoints** (the things migration code reaches into):
- `vllm.sequence.SequenceGroup` (wrapped as `SequenceGroupLlumnix`)
- `vllm.core.block_manager.BlockTable` — for pre-alloc on dst
- `vllm.worker.cache_engine.CacheEngine` — `swap_blocks()`, plus `num_attention_layers`,
  `block_size`, `num_kv_heads`, `head_size`, `gpu_cache`
- `vllm.sequence.SequenceGroupMetadata` — sampling/RNG state shipped in last stage

**Append-only exploitation**:
- The slicing `block_table[pre_stage_num_blocks:]` is the explicit code that relies
  on the prefix being immutable.
- Stage timestamps `stage_timestamps.append(time.time())` (≈ L338).
- Excluding the in-flight block until last stage is the safety dance.

**Existing subset-of-state infrastructure** that PP-reshard can lean on (per A2):
1. **Layer-by-layer iteration is already the shape of the loop** in both
   `RayRpc.do_send()` (≈ L178) and `RayCol.do_send()` (≈ L384). For PP-reshard
   we'd just change the loop bounds: instead of `range(num_layers)`, use
   `range(M, num_layers)` for the upper-half ship.
2. Block-level granularity is already the atomic unit (block_table partial transfer).
3. Per-stage seq_group metadata caching (`_stage_seq_group_metadata`, worker.py
   ≈ L218) decouples metadata stages from block stages.
4. Per-migration state reset (`reset_migration_states_dst/src`) is already
   compartmentalized — could be extended for per-layer-range reset.

**Missing for PP-reshard** (per A2; my synthesis):
1. Layer-range fields on `LlumnixRequest` (e.g. `layer_start`, `layer_end`,
   `pp_reshard_partner`).
2. Per-layer block tables (currently one global block_table per request because
   PP=1; but vLLM's `gpu_cache` is *already* `List[Tensor]` indexed by layer).
3. Partial seq_group metadata at boundaries (the second-half instance doesn't
   sample; it just produces hidden states for the last PP rank).
4. Per-step forward-graph awareness so that A.forward(layers[0:M], inputs) and
   B.forward(layers[M:L], hidden) are called in lockstep on the per-iteration
   timeline, with a hidden-state send in between, and only on the resharded
   subset of the batch.
5. A migration state that's "split" rather than "moved" — current state machine
   assumes the request leaves the source. We need a new migration outcome that
   means "the request now lives on both", with refcount-style ownership.

### §3.1.3 Scheduler / policy deep-dive (A3)

Files:

| Path | Class / Role |
|---|---|
| `llumnix/global_scheduler/global_scheduler.py` | `GlobalScheduler` — instance registration, dispatch, migration trigger, scaling |
| `llumnix/global_scheduler/dispatch_scheduler.py` | `DispatchScheduler` — routing logic (with PD-disagg branch) |
| `llumnix/global_scheduler/dispatch_policy.py` | `DispatchPolicy` ABC + `Load`, `Balanced`, `Queue`, `RoundRobin` |
| `llumnix/global_scheduler/migration_scheduler.py` | `MigrationScheduler` — pairing logic |
| `llumnix/global_scheduler/migration_policy.py` | `Balanced`, `Defrag` (paper's defrag) |
| `llumnix/global_scheduler/scaling_scheduler.py` | `ScalingScheduler` — threshold checks |
| `llumnix/global_scheduler/scaling_policy.py` | `AvgLoad`, `MaxLoad`, `MinLoad` |
| `llumnix/llumlet/llumlet.py` | `Llumlet` — reports `InstanceInfo`, hosts `MigrationCoordinator` |
| `llumnix/llumlet/local_migration_scheduler.py` | `LocalMigrationScheduler` — picks which req to migrate (`SR`, `LR`, `LCR`, `FCW`, `FCWSR`) |
| `llumnix/instance_info.py` | `InstanceInfo` dataclass (the per-instance load report) |
| `llumnix/load_computation.py` | `BaseLoad` hierarchy, `KvBlocksRatioLoad`, `RemainingStepsLoad`, `AdaptiveDecodeBatchLoad` |
| `llumnix/manager.py` | `Manager` — orchestrates the whole thing |
| `llumnix/internal_config.py` | `GlobalSchedulerConfig`, `MigrationConfig` |
| `llumnix/config/default.py` | the `_C` defaults registry |

**Per-poll cycle** (default `polling_interval = 0.05s`):
1. `Manager._poll_instance_info_loop()` calls each llumlet's `get_instance_info()`.
2. Llumlet returns an `InstanceInfo` with: `num_running_requests`,
   `num_waiting_requests`, `num_used_gpu_blocks`, `num_total_gpu_blocks`,
   `migration_load_metric`, `num_killed_requests`. **Per-request detail is
   NOT shipped** — this is the scalability trick.
3. `GlobalScheduler.update_instance_infos()` (≈ L85) refreshes its dict.
4. Every `pair_migration_frequency` polls (default 1):
   - `Manager._push_migrations()` → `GlobalScheduler.pair_migration(constraint)`
     (≈ L176)
   - `MigrationScheduler.pair_migration()` filters and asks the policy for pairs.
   - For each pair `(src, dst)`: `llumlet[src].migrate_out(llumlet[dst], dst_id)`.

**Algorithm 1 in code** — the simplification: Llumnix doesn't compute *per-request*
virtual usage. Instead, it computes *instance-level* load metrics that play the
same role:

| Paper concept | Code analog |
|---|---|
| `CalcVirtualUsage` per request | not implemented per-request; instead instance-level load is summed by the load module |
| `GetHeadroom(priority)` | **not in code** — priority/headroom support appears NOT to be implemented in this fork; need to verify [V] |
| `CalcFreeness(instance)` | implicit. `RemainingStepsLoad.compute_instance_load()` (≈ L102–116): `num_available_gpu_blocks / (num_running + num_waiting)` |
| Fake-request-with-∞ for terminating instance | **not in code** — uses `num_killed_requests` counter instead (instance_info.py ≈ L57) |

> **This is a significant deviation from the paper that we should ask the user about.**
> The paper-level virtual-usage concept is not faithfully implemented in this fork.
> Priority support also seems absent. See §99-Q2.

**Policy parameters** (from `llumnix/config/default.py`):

| Param | Default | Meaning |
|---|---|---|
| `MANAGER.DISPATCH_POLICY` | `'load'` | dispatch by load metric |
| `MANAGER.DISPATCH_LOAD_METRIC` | `'remaining_steps'` | metric used for dispatch |
| `MANAGER.PAIR_MIGRATION_POLICY` | `'defrag'` | defrag (paper's primary policy) or balanced |
| `MANAGER.MIGRATE_OUT_THRESHOLD` | `-3.0` | src eligibility (negative-freeness territory) |
| `MANAGER.POLLING_INTERVAL` | `0.05` | seconds between info polls |
| `MANAGER.PAIR_MIGRATION_FREQUENCY` | `1` | run migration check every N polls |
| `MANAGER.SCALE_UP_THRESHOLD` | `-10` | scale-up trigger |
| `MANAGER.SCALE_DOWN_THRESHOLD` | `-60` | scale-down trigger |
| `MANAGER.SCALING_POLICY` | `'avg_load'` | scaling load aggregation |
| `MANAGER.MIGRATION.REQUEST_MIGRATION_POLICY` | `'SR'` | shortest running first |
| `MANAGER.MIGRATION.MIGRATION_MAX_STAGES` | `3` | matches stage cap |
| `MANAGER.MIGRATION.MIGRATION_LAST_STAGE_MAX_BLOCKS` | `16` | when to stop pipelining and suspend |
| `MANAGER.ENABLE_PD_DISAGG` | (boolean) | PD disaggregation toggle (already present!) |
| `MANAGER.ENABLE_ADAPTIVE_PD` | (boolean) | adaptive PD toggle |

**Dispatch path** (new request arrives):
```
Manager.generate(request_id, server_info)
  → GlobalScheduler.dispatch(request_id, dispatch_kwargs)
       (if PD disagg: dispatch_pd; else dispatch_no_constrains)
  → DispatchScheduler.dispatch_no_constrains(instance_infos, num_requests)
  → DispatchPolicy.dispatch(...)
       Load: sort by metric ascending → topk_random_dispatch → pick instance_id
  → Manager → instance_actor.generate.remote(request_id, server_info, expected_steps, ...)
  → Llumlet.generate() → backend_engine.add_request()
```

**Migration trigger path**:
```
Manager._push_migrations()
  → GlobalScheduler.pair_migration(pair_migration_type)
  → MigrationScheduler.pair_migration(instance_infos)
       MigrationFilter.filter_instances():
         src cond: load > migrate_out_threshold OR num_killed > 0
         dst cond: load < migrate_out_threshold AND num_killed == 0
  → PairMigrationPolicy.pair_migration(src_infos, dst_infos)
       Defrag/Balanced: sort, pair (top src) ↔ (top dst)
  → returns [(src_id, dst_id), ...]
  → Manager.llumlet[src].migrate_out.remote(llumlet[dst], dst_id)
       └─ MigrationCoordinator (see §3.1.2)
```

Source request selection: `LocalMigrationScheduler.get_migrate_out_requests()`
(local_migration_scheduler.py ≈ L26–40), policy = `SR`/`LR`/`FCW`/`LCR`/`FCWSR`.
Default `SR` = shortest running request.

**Auto-scaling path**:
- `ScalingScheduler.check_scale()` aggregates load (avg/max/min) across instances.
- Scale-up: simulates adding dummy instance until avg ≤ threshold.
- Scale-down: returns `scale_down_num=1` if load < `scale_down_threshold`.
- Scaler picks instance with fewest running requests for termination.
- **No fake-request mechanism** for draining — requests get migrated via the
  normal migration loop instead. (Differs from paper.)

**Priority support**: **not visible in this fork** (per A3). No priority field
on `LlumnixRequest`, no priority-aware load metric, no priority-weighted
migration selection. Either it was removed/never-ported or it's hidden behind
`request.server_info` extras. **Need to verify** [V].

**What Llumlet reports** to the global scheduler (per polling interval):
```python
InstanceInfo(
  instance_id, instance_type,
  num_running_requests, num_waiting_requests,
  num_used_gpu_blocks, num_total_gpu_blocks,
  migration_load_metric, num_killed_requests
)
```
That's it — O(num_instances), not O(num_requests). This is the scalability trick
in code form. Llumlets do *all* per-request bookkeeping locally.

**Hooks A3 suggested for PP-reshard** (paraphrased & critiqued by me):

| # | Location | A3's argument | My take |
|---|---|---|---|
| H1 | `LocalMigrationScheduler.get_migrate_out_requests()` | flag a request as "PP-reshardable" instead of "fully-migratable" | natural; this is where we'd choose split policy |
| H2 | `Balanced.pair_migration()` / `MigrationScheduler.pair_migration()` | add a third pairing mode for PP-reshard pairs (overloaded src + capacity dst that already has all layers loaded) | yes, but easier if we treat PP-reshard as a sub-mode of migration on the same pair |
| H3 | `MigrationCoordinator.migrate_out()` | branch on `migration_type=="pp_reshard"` | the heavy lifting goes here. Branch point = where the multi-stage state machine forks |
| H4 | `DispatchScheduler.dispatch_*()` | dispatch new requests to PP-reshard-compatible instances | low priority — dispatch is for new requests; PP-reshard is for live requests |
| H5 | `Manager._poll_instance_info_loop()` or new `_check_pp_reshard_merge_loop()` | monitor when to merge a PP-reshard back to PP=1 | yes, we need a merge policy. Could piggyback on existing migration channel. |

### §3.1.4 Things flagged for direct verification later

- [V1] confirm `pre_alloc_cache_dict` and `migrating_out_request_last_stage` fields
  on `SchedulerLlumnix` (scheduler.py L131–159 area)
- [V2] confirm the layer-by-layer iteration in `RayCol.do_send` (migration_backend.py
  L384, L390) — most critical for PP-reshard re-implementation
- [V3] confirm `MIGRATION_MAX_STAGES = 3` default (config/default.py L156)
- [V4] verify priority support absence by reading `LlumnixRequest` and `InstanceInfo`
  in full
- [V5] verify how `ENABLE_PD_DISAGG` is wired — what does dispatch_pd do? Is there
  a "kv_role" equivalent in this fork? (docs/Prefill-decode_Disaggregation.md
  + tests/e2e_test/test_dynamic_pd.py likely reveal this)
- [V6] read `migration_coordinator.py` end-to-end — A2's call graph is plausible
  but I want to see the exact stage transitions and the state enum


## §3.2 vllm 0.6.3 (the version Llumnix currently targets)

> Source: A4 (engine/scheduler/worker/PP) + A5 (KV cache/attention). Paths
> relative to `/home/ubuntu/reshardLLM/vllm/vllm/`.

### §3.2.1 Top-level layout

- `engine/` — `LLMEngine`, `AsyncLLMEngine`
- `core/` — `Scheduler`, block manager, queues
- `worker/` — `Worker`, `ModelRunner`, `CacheEngine`
- `model_executor/` — model loading, layer instantiation, sampler
- `attention/` — backends (FlashAttention, FlashInfer, xFormers, ...), abstract metadata
- `distributed/` — `parallel_state`, `GroupCoordinator`, communication primitives
- `executor/` — `Executor` impls (uniproc, multiproc, Ray)
- `entrypoints/` — OpenAI server, offline LLM API
- `sequence.py` (top-level) — `Sequence`, `SequenceGroup`, `SequenceGroupMetadata`, `SequenceData`
- `inputs/`, `config/`, `usage/`, `lora/`, `prompt_adapter/`, `multimodal/`, `transformers_utils/`, `tracing/`, `compilation/`

### §3.2.2 Engine / scheduler / worker / model_runner

**`LLMEngine.step()`** (`engine/llm_engine.py` ≈ L1263):
1. `scheduler.schedule()` → `SchedulerOutputs`
2. `executor.execute_model()` → `List[SamplerOutput]`
3. `_process_model_outputs()` → detokenize, finalize, stream

**Sync `step()` does NOT support PP** (raises NotImplementedError ≈ L1314).
PP requires `AsyncLLMEngine`.

**`Scheduler.schedule()`** (`core/scheduler.py` ≈ L1211):
- Returns `SchedulerOutputs(scheduled_seq_groups, num_batched_tokens, blocks_to_swap_in/out, blocks_to_copy, num_lookahead_slots)`.
- Maintains `waiting_queue`, `running_queue`, `swapped_queue`.
- Per-step: drain running, admit waiting, return preempted/swapped.
- Two preemption modes: recompute-preempt (kill KV, requeue) vs. swap-out (move blocks to CPU).
- `block_manager.can_allocate()` / `allocate()` decide admission.

**`Worker._execute_model_spmd()`**: cache ops via `CacheEngine.swap_in/out()` →
`ModelRunner.prepare_model_input()` → `ModelRunner.execute_model()`.

**`ModelRunner.execute_model()`** (`worker/model_runner.py` ≈ L1606):
- Input: `ModelInputForGPUWithSamplingMetadata` (positions, slot_mapping, block_tables, attn_metadata).
- Calls `model_executable(input_ids, positions, kv_caches, attn_metadata, intermediate_tensors, ...)`.
- If not last PP rank: returns `IntermediateTensors`.
- If last PP rank: `compute_logits()` → `sample()` → `SamplerOutput`.

### §3.2.3 PP forward path (CRITICAL FOR PP-RESHARD)

**Layer slicing at load time** (`model_executor/models/utils.py:make_layers`,
`distributed/utils.py:get_pp_indices`):
- Default contiguous slicing — rank `r` of `pp_size` gets layers
  `[r*L/pp, (r+1)*L/pp)`.
- Override via env `VLLM_PP_LAYER_PARTITION="20,20,32"`.
- Layers outside the rank's range are replaced with `PPMissingLayer()` placeholders
  to preserve global indexing.

**PP forward at run time** (e.g. `model_executor/models/llama.py:LlamaModel.forward` ≈ L325):
- Non-first rank: receive `IntermediateTensors(hidden_states, residual)`.
- Loop over `[start_layer, end_layer)` apply each layer.
- Non-last rank: return `IntermediateTensors(...)`.
- Last rank: final norm → logits → sample.

**PP send/recv**: not explicit p2p in `model_runner`. `execute_model()` returns
either `IntermediateTensors` or `SamplerOutput`; the *executor* coordinates rank
ordering. PP groups are managed by `GroupCoordinator` (`distributed/parallel_state.py`
≈ L128) and accessed via `get_pp_group()` (≈ L882).

**`initialize_model_parallel(tp_size, pp_size)`** (`distributed/parallel_state.py`
≈ L962): creates TP and PP groups; populates global `_TP` and `_PP`.

**🔴 MAJOR IMPLICATION FOR PP-RESHARD** (synthesized): in stock vLLM, *which* layers
a worker holds is fixed at load time, baked into the model object via
`PPMissingLayer()` placeholders. To support hybrid serving (PP=1 batch + PP=2
sub-batch on the same iteration), each worker must hold *all* layers and decide
at run time which subset to execute. This means we cannot use the
`PPMissingLayer` mechanism. Instead, we'd run `pipeline_parallel_size=1` always,
and at the model_runner level decide per-request which layers to run.

### §3.2.4 KV cache & block manager (A5)

**`CacheEngine`** (`worker/cache_engine.py`):
- `_allocate_kv_cache(num_blocks, device)` (≈ L77) loops `num_attention_layers` and
  appends one `torch.zeros(kv_cache_shape, ...)` per layer.
- Result: `self.gpu_cache: List[torch.Tensor]` of length `L`.
- **One tensor per layer**, indexed by layer.

**KV cache shape** (FlashAttention backend, `attention/backends/flash_attn.py`):
- `(2, num_blocks, block_size, num_kv_heads, head_size)` per layer.
- Dim 0: [key, value] fused — `kv_cache[0]` = K, `kv_cache[1]` = V.
- Per layer, this is one big contiguous tensor.

PagedAttention backend uses a flatter shape:
`(2, num_blocks, block_size * num_kv_heads * head_size)`.

For LLaMA-7B (TP=1, PP=1, block_size=16, FP16), 32 layers, num_kv_heads=8 (GQA),
head_size=128. **Per-layer tensor depends on `num_blocks`** which is auto-computed
from available memory. (A5's "67GB per layer" calc assumed `num_blocks=8192` —
which is unusually large; a more realistic `num_blocks=2048` gives ~16 GB
across all 32 layers, not per layer.) [V] verify with the actual `num_blocks`
heuristic in `cache_engine.py`.

**PP interaction**: `cache_engine.py` ≈ L44 does `num_gpu_blocks //= pp_size`.
Each worker owns `num_blocks/pp_size` × num-of-its-layers tensors.

**Block manager** in 0.6.3 is `SelfAttnBlockSpaceManager`
(`core/block_manager.py`) — uses `CpuGpuBlockAllocator` from `core/block/`.
A5 says "no `block_manager_v1.py` vs `v2.py`" — but Llumnix's `SchedulerLlumnix`
still references the older API; **need to verify which manager is actually
default in 0.6.3** [V].

`BlockTable` (`core/block/block_table.py`):
- Per-sequence list of `Block` objects.
- `allocate(token_ids)` → loops `block_allocator.allocate_immutable_block` per
  block-sized chunk.
- `append_token_ids` → `ensure_num_empty_slots` → `allocate_mutable_block`.
- `fork()` → CoW (copy-on-write) for beam search.
- `free()` → `block.unref()` per block; refcount=0 → return to free pool.

**Prefix caching** (`core/block/prefix_caching_block.py`):
- `PrefixCachingBlockAllocator._cached_blocks: Dict[hash, block_id]`.
- LRU eviction on the cached blocks.
- Computed via `ComputedBlocksTracker` and `LastAccessBlocksTracker`.

**Attention metadata** (`attention/backends/flash_attn.py:FlashAttentionMetadata`):
- `slot_mapping`, `seq_lens`, `seq_lens_tensor`, `block_tables`,
  `query_start_loc`, `seq_start_loc`, `context_lens_tensor`,
  `max_query_len`, `max_decode_query_len`, `max_prefill_seq_len`, `max_decode_seq_len`,
  `use_cuda_graph`.
- `block_tables: torch.Tensor` of shape `[batch_size, max_blocks_per_seq]`.
- `slot_mapping`: flat `[num_tokens]` of physical slot indices.
- `compute_slot_mapping()` (`attention/backends/utils.py` ≈ L74) computes
  `slot = block_tables[seq, block_idx] * block_size + offset`.

**Attention kernel call** (decode):
```python
flash_attn_with_kvcache(
    q=query, k_cache=kv_cache[0], v_cache=kv_cache[1],
    causal=True,
    block_tables=attn_metadata.block_tables,
    seqlen_k=attn_metadata.seq_lens_tensor,
)
```
Kernel reads KV at `kv_cache[block_tables[s, t//bsz], t%bsz, ...]`.

**Swap-in / swap-out** (`worker/cache_engine.py`):
```python
def swap_in(self, src_to_dst):
    for i in range(num_attention_layers):
        attn_backend.swap_blocks(cpu_cache[i], gpu_cache[i], src_to_dst)
```
Calls a custom CUDA kernel `ops.swap_blocks` — NOT `torch.copy_()`. Per-layer
loop, blocking from caller's perspective.

**Block copy** (`copy_blocks(kv_caches, src_to_dsts)`): also a CUDA kernel,
maps `[num_copies, max_dests]` indices over per-layer tensors.

**Per-request state** (`sequence.py`):
- `Sequence` (≈ L379) — seq_id, `data` (`SequenceData`: prompt + output token ids),
  status enum, `output_logprobs`, `output_text`, `block_size`.
- `SequenceGroup` (≈ L648) — request_id, list of `Sequence` (multiple in beam search),
  `sampling_params`, `arrival_time`, `state` (generation step counter, beam state,
  preemption info), optional `lora_request`, `prompt_adapter_request`, `encoder_seq`.
- `SequenceGroupMetadata` is what gets passed to the model runner each step
  and is what migration ships to the destination.

### §3.2.5 Invariants we must respect for PP-reshard (synthesized from A5)

1. **`gpu_cache` is `List[Tensor]` indexed by layer** — natural shape for
   "drop layers M..L from this worker's cache and ship them elsewhere". ✅
2. **Block IDs are global within a worker's pool** (not per-layer). All layers'
   tensors share the same block-ID pool. So if we keep all layers on each worker
   for our hybrid PP design (the right choice for our use case), block IDs
   stay simple. ✅
3. **Model weights and KV cache colocate**: layer-i's KV must live on the GPU
   that holds layer-i's weights. **We satisfy this trivially because each
   worker holds all layers' weights.** ✅
4. **Block tables are integer-ID lists** with no device tag. We need to be
   careful that a block table on instance A doesn't get mixed up with block
   IDs on instance B. ⚠️
5. **RefCounter is single-machine.** A block's refcount is local. We must
   either (a) avoid cross-machine block sharing (each instance owns its own
   blocks separately) — easy because PP-reshard literally puts different
   layers' KV on different machines — or (b) do cross-machine refcount
   tracking (hard). ✅ (we get this for free)
6. **Schedulers in 0.6.3 don't know about PP layer subsets per-request.**
   We'll have to teach the scheduler to admit a "split request" that runs
   layers 0..M here and gets its hidden states sent to the partner instance.

### §3.2.6 Things flagged for direct verification

- [V7] confirm `block_manager.py` is the only manager file in 0.6.3 and that
  `SelfAttnBlockSpaceManager` is what `SchedulerLlumnix` actually subclasses.
- [V8] confirm `make_layers()` location and the `PPMissingLayer` mechanism end
  to end (path ≈ `model_executor/models/utils.py`).
- [V9] read `LlamaModel.forward()` to see exactly how `IntermediateTensors`
  is consumed and produced — this is the function we'd most likely override
  for PP-reshard.
- [V10] verify the actual KV-cache tensor shape numbers via `cache_engine.py`
  — A5's "67GB per layer" looks suspect.
- [V11] read `executor/distributed_gpu_executor.py` (or the Ray executor)
  to see how the executor coordinates which rank runs in which order during
  PP — is there a single master that calls each rank's `execute_model` in
  sequence, or do ranks run in parallel and use NCCL send/recv between them?


## §3.3 vllm-latest

> Source: A6 (top-level), A7 (v1 engine), A8 (KV connector), A9 (disagg PD).
> Paths relative to `/home/ubuntu/reshardLLM/vllm/vllm-latest/vllm/` unless noted.
> **This is the most important borrow target for the project per the user's
> instruction.**

### §3.3.1 Top-level layout

- **`v1/`** — the new engine. **Default in latest, preferred path.**
  - `v1/engine/` — `EngineCore`, `EngineCoreClient` family, `LLMEngine`,
    `AsyncLLM`, `tensor_ipc`, `output_processor`, `detokenizer`, `coordinator`
  - `v1/core/` — scheduler, KV cache management
    - `v1/core/sched/{scheduler.py, async_scheduler.py, output.py, interface.py, request_queue.py}`
    - `v1/core/{kv_cache_manager, kv_cache_coordinator, kv_cache_utils, single_type_kv_cache_manager, block_pool}.py`
  - `v1/executor/` — `Executor` ABC + `uniproc`, `multiproc`, `ray`, `ray_executor_v2`
  - `v1/worker/` — `WorkerBase`, `gpu_worker`, `cpu_model_runner`, `gpu_model_runner`,
    `block_table`, `ubatching`, `kv_connector_model_runner_mixin`, `gpu/pp_utils.py`
  - `v1/sample/` — `Sampler`
  - `v1/attention/` — backends and metadata
  - `v1/request.py`, `v1/outputs.py`, `v1/serial_utils.py`
- **`engine/`** — legacy v0; `llm_engine.py`, `async_llm_engine.py`. Minimal in latest,
  kept for backward compat.
- **`distributed/`** — `parallel_state` (PP/TP/DP groups), `kv_transfer/`,
  `device_communicators/`, `weight_transfer/`, `ec_transfer/`, `eplb/`
- **`distributed/kv_transfer/`** — the KV-connector subsystem (CRITICAL — see §3.3.4)
- **`attention/`** — v0 attention layer (mostly superseded by `v1/attention/`)
- **`model_executor/`** — model loading, layers (still used in v1)
- **`entrypoints/`** — `llm.py` (LLM API), `openai/`, `serve/disagg/`, `cli/`
- `config/` — `vllm.py:VllmConfig`, `parallel.py:ParallelConfig`, `model.py:ModelConfig`,
  `cache.py:CacheConfig`, `scheduler.py:SchedulerConfig`, `kv_transfer.py:KVTransferConfig`,
  `kv_events.py:KVEventsConfig`, `weight_transfer.py:WeightTransferConfig`,
  `ec_transfer.py:ECTransferConfig`
- `forward_context.py` — per-step broadcast context (relevant for our layer-mask injection)
- `lora/`, `compilation/`, `multimodal/`, `kernels/`, `tracing/`, `profiler/`,
  `usage/`, `transformers_utils/`, `platforms/`, `device_allocator/`

**v0 vs v1 parity** (per A6):

| Subsystem | v0 path | v1 path | Default in latest |
|---|---|---|---|
| Engine core | `engine/llm_engine.py` | `v1/engine/core.py` | v1 |
| Scheduler | embedded in v0 LLMEngine | `v1/core/sched/scheduler.py` | v1 |
| Model runner | `worker/model_runner.py` (legacy) | `v1/worker/gpu_model_runner.py` | v1 |
| Executor | `executor/` | `v1/executor/` | v1 |
| Attention | `attention/` | `v1/attention/` | v1 |
| KV cache mgr | scheduler-embedded | `v1/core/kv_cache_manager.py` | v1 |
| PP | per-rank IntermediateTensors | `v1/worker/gpu/pp_utils.py` (broadcast/recv) | v1 |

### §3.3.2 v1 engine architecture (A7)

**`EngineCore`** (`v1/engine/core.py` ≈ L91–230):
- Owns `Executor`, `Scheduler`, `KVCacheManager`, `StructuredOutputManager`.
- Run loop: `step()` or `step_with_batch_queue()` (the latter pipelines batches
  to fight PP bubbles — relevant for our PP-reshard since we want microbatching).
- Runs in own process via `EngineCoreProc` wrapper or in-process when
  `multiprocess_mode=False`.

**`EngineCoreClient` family** (`v1/engine/core_client.py`):
- ABC `EngineCoreClient` + concrete `InprocClient`, `SyncMPClient`, `AsyncMPClient`.
- Transport: ZMQ (inproc for multiproc mode, in-memory otherwise).
- Messages: `EngineCoreRequest` typed (ADD_REQUEST / GET_OUTPUT / ABORT / ...) +
  `EngineCoreOutputs` (responses) — msgpack-serialized.

**`LLMEngine`** (`v1/engine/llm_engine.py`) — sync; **`AsyncLLM`**
(`v1/engine/async_llm.py`) — async.

**Output processor** (`v1/engine/output_processor.py`) — separate thread:
detokenization, logprob aggregation, request finalization.

**`Scheduler`** (`v1/core/sched/scheduler.py`, `Scheduler(SchedulerInterface)`):
- ~600 LOC; chunked prefill + prefix caching + KV-connector hooks.
- `SchedulerOutput` (`v1/core/sched/output.py` L178–239):
  ```python
  scheduled_new_reqs: list[NewRequestData]      # first-time admit
  scheduled_cached_reqs: CachedRequestData      # diff updates
  num_scheduled_tokens: dict[str, int]          # per-request budget
  total_num_scheduled_tokens: int
  finished_req_ids: set[str]
  kv_connector_metadata: KVConnectorMetadata | None    # CRITICAL for disagg
  ```
- Continuous batching: `_schedule_requests()` admits up to a token budget;
  prefill chunks split across steps via `num_scheduled_tokens[req_id]`.
- KV cache: calls `KVCacheManager.allocate(request, num_tokens) → KVCacheBlocks`.
- KV connector: scheduler-side connector contributes
  `kv_connector_metadata` per step (see §3.3.4).

**`Executor`** (`v1/executor/abstract.py:37`):
- `execute_model(scheduler_output) → ModelRunnerOutput` via `collective_rpc()`.
- Concrete: `UniProcExecutor`, `MultiprocExecutor`, `RayDistributedExecutor`,
  `RayExecutorV2`.

**`Worker`** (`v1/worker/gpu_worker.py:Worker(WorkerBase)`):
- Receives `SchedulerOutput`, calls `model_runner.execute_model(output, intermediate_tensors)`.
- Returns `ModelRunnerOutput` (or in PP, `IntermediateTensors` for inter-rank).
- Mixes in `KVConnectorModelRunnerMixin`
  (`v1/worker/kv_connector_model_runner_mixin.py`) for KV transfer hooks.

**`GPUModelRunner`** (`v1/worker/gpu_model_runner.py`, ≈4000 LOC):
- Main forward path:
  ```
  execute_model(scheduler_output, intermediate_tensors=None):
    1. _update_states(scheduler_output)         # request bookkeeping
    2. _prepare_inputs(scheduler_output)        # token IDs, embeddings, positions
    3. _determine_batch_execution_and_padding() # CUDAGraph mode, ubatching
    4. _build_attention_metadata()              # per-layer block tables, slot mappings
    5. _preprocess()                            # token → embedding
    6. set_forward_context(...)                 # broadcast batch info
    7. _model_forward(input_ids, positions, ...)# the actual transformer
    8. sample_tokens(grammar_output)            # last-rank only
  ```
- PP path (≈ L3787–4137): non-last ranks return `IntermediateTensors` (line ≈4072);
  last rank computes logits + samples.
- `set_forward_context()` (≈ L4034) pushes batch metadata into a thread-local
  context that all transformer layers read.
- Attention metadata is built per step via `AttentionMetadataBuilder.make()`.
- Microbatching / DBO (dual-batch overlap): `ubatch_slices` computed in
  `_determine_batch_execution_and_padding()` (≈ L3918–3924). **This already exists
  in v1 and is exactly the kind of "two parallel mini-batches in one step"
  primitive we need** for hybrid PP=1 + PP=2 batching.

**`Sampler`** (`v1/sample/sampler.py`, `Sampler(nn.Module)`): logits → logprobs
→ logit processors → penalties → top-k/p → sampled tokens + logprobs.

**`ModelRunnerOutput`** (`v1/outputs.py` L165–202):
```python
req_ids: list[str]
sampled_token_ids: list[list[int]]
logprobs: LogprobsLists | None
kv_connector_output: KVConnectorOutput | None    # connector feedback
```

**KV cache management** (v1):
- `KVCacheManager` (`v1/core/kv_cache_manager.py`): `allocate(request, num_tokens)`
  → `KVCacheBlocks` (tuple of lists per cache group); `free(request)`; `reset()`.
- `KVCacheCoordinator` (`v1/core/kv_cache_coordinator.py`): multi-group support
  (e.g. encoder + decoder, or full-attn + sliding-window).
- `BlockPool` (`v1/core/block_pool.py`): block allocation pool, prefix-cache hashing.
- `SingleTypeKVCacheManager` (`v1/core/single_type_kv_cache_manager.py`): per-spec
  manager (`FullAttention`, `SlidingWindow`, etc.).

**Attention in v1**:
- Backends: FlashAttention 2/3, Triton, FlashInfer, FlexAttention (+ Mamba, MLA).
- `AttentionMetadata` and `AttentionMetadataBuilder.make()` built once per step.
- Block tables / slot mapping similar to v0 in shape but cleaner construction.

### §3.3.3 PP in vllm-latest

- `pipeline_parallel_size` field on `ParallelConfig` (`config/parallel.py` ≈ L111).
- `distributed/parallel_state.py` ≈ L726–745: `get_pp_group()` returns the PP
  group coordinator with `is_first_rank`, `is_last_rank`, `device_group`.
- `v1/worker/gpu/pp_utils.py`: `pp_broadcast()`, `pp_receive()` for token / sample
  distribution between ranks. Uses `torch.distributed.broadcast()` on
  `get_pp_group()`.
- Model loader still does PP-aware layer slicing (same `make_layers()` /
  `PPMissingLayer` pattern as 0.6.3).

> **🟢 Good news for our project**: the `step_with_batch_queue()` mode in
> `EngineCore` already does pipelined batches across PP iterations to hide
> bubbles. And `ubatch_slices` in `gpu_model_runner` already supports running
> two mini-batches per step. These are the primitives we'd need to extend for
> hybrid PP=1 + PP=2 hosting.

### §3.3.4 KV connector layer (A8) — borrow target #1

Files (path relative to `vllm/distributed/kv_transfer/`):

```
__init__.py
kv_transfer_state.py                   # global connector state holder
kv_connector/__init__.py
kv_connector/base.py                   # KVConnectorBase = KVConnectorBase_V1 alias
kv_connector/factory.py                # KVConnectorFactory.create_connector()
kv_connector/utils.py                  # EngineId, BlockIds, KVOutputAggregator, layout helpers
kv_connector/v1/__init__.py            # exports base + KVConnectorRole + SupportsHMA
kv_connector/v1/base.py                # KVConnectorBase_V1 (the real ABC) + 3 metadata bases

kv_connector/v1/example_connector.py
kv_connector/v1/example_hidden_states_connector.py
kv_connector/v1/lmcache_connector.py                  # LMCacheConnectorV1
kv_connector/v1/lmcache_mp_connector.py
kv_connector/v1/lmcache_integration/
kv_connector/v1/p2p/p2p_nccl_connector.py             # P2pNcclConnector
kv_connector/v1/p2p/p2p_nccl_engine.py
kv_connector/v1/p2p/tensor_memory_pool.py
kv_connector/v1/nixl/connector.py                     # NixlConnector (facade)
kv_connector/v1/nixl/scheduler.py                     # NixlConnectorScheduler
kv_connector/v1/nixl/worker.py                        # NixlConnectorWorker
kv_connector/v1/nixl/metadata.py                      # NixlConnectorMetadata, NixlAgentMetadata, NixlHandshakePayload
kv_connector/v1/nixl/utils.py
kv_connector/v1/nixl/stats.py
kv_connector/v1/mooncake/mooncake_connector.py        # MooncakeConnector
kv_connector/v1/mooncake/mooncake_utils.py
kv_connector/v1/moriio/moriio_connector.py            # MoRIIOConnector
kv_connector/v1/moriio/{moriio_common.py, moriio_engine.py}
kv_connector/v1/offloading_connector.py               # OffloadingConnector
kv_connector/v1/offloading/{scheduler.py, worker.py, common.py, metrics.py}
kv_connector/v1/simple_cpu_offload_connector.py       # SimpleCPUOffloadConnector
kv_connector/v1/flexkv_connector.py                   # FlexKVConnectorV1
kv_connector/v1/decode_bench_connector.py
kv_connector/v1/multi_connector.py                    # MultiConnector
kv_connector/v1/hf3fs/hf3fs_connector.py              # HF3FSKVConnector
kv_connector/v1/hf3fs/{hf3fs_client.py, hf3fs_metadata_server.py, hf3fs_utils/}
kv_connector/v1/metrics.py
kv_connector/v1/ssm_conv_transfer_utils.py
```

**Base class `KVConnectorBase_V1`** (`v1/base.py`):

Roles enum:
```python
class KVConnectorRole(enum.Enum):
    SCHEDULER = 0
    WORKER = 1
```

Per-step metadata abstractions (each connector subclasses):
- `KVConnectorMetadata` — scheduler → worker per-step plan
- `KVConnectorWorkerMetadata` — worker → scheduler per-step feedback
- `KVConnectorHandshakeMetadata` — out-of-band P/D handshake (e.g. NIXL agent discovery)

**Scheduler-side methods** (called from `Scheduler`):

| Method | Purpose | When |
|---|---|---|
| `get_num_new_matched_tokens(request, num_computed_tokens) → (int|None, bool)` | how many tokens of external KV can be loaded? | request scheduling step |
| `update_state_after_alloc(request, blocks, num_external_tokens)` | notify connector of allocated blocks for remote KV | after KVCacheManager.allocate |
| `build_connector_meta(scheduler_output) → KVConnectorMetadata` | build per-step plan for workers | per scheduler step |
| `request_finished(request, block_ids) → (bool, dict|None)` | finalization hook; True = connector takes async ownership of blocks | on finish |
| `take_events() → Iterable[KVCacheEvent]` | yield prefetch/eviction events | periodically |
| `update_connector_output(connector_output)` | absorb worker feedback | after workers return |

**Worker-side methods** (called from worker forward path):

| Method | Purpose | When |
|---|---|---|
| `bind_connector_metadata(metadata)` | inject per-step metadata | start of forward |
| `start_load_kv(forward_context)` | async-start remote receives | start of forward |
| `wait_for_layer_load(layer_name)` | block until layer KV arrived | inside attn layer |
| `save_kv_layer(layer_name, kv, attn_meta)` | async-start remote send | after attn layer |
| `wait_for_save()` | block until all saves done | end of forward |
| `register_kv_caches(kv_caches)` | RDMA pre-register layer tensors | worker init |
| `register_cross_layers_kv_cache(kv_cache, backend)` | register unified cross-layer tensor | worker init |
| `set_host_xfer_buffer_ops(copy_op)` | CPU↔GPU staging ops | worker init |
| `handle_preemptions(metadata)` | abort transfers for evicted reqs | before block recycling |
| `get_finished(finished_req_ids) → (sending, recving)` | poll completion | each step |
| `build_connector_worker_meta() → ...` | feedback to scheduler | each step |

Optional: `get_block_ids_with_load_errors`, `shutdown`, `get_kv_connector_stats`,
`get_kv_connector_kv_cache_events`.

Class methods: `get_required_kvcache_layout(vllm_config)` (e.g. NIXL prefers HND
layout for fast row-major xfer), `requires_piecewise_for_cudagraph(extra_config)`.

`SupportsHMA` ABC: `request_finished_all_groups(request, block_ids: tuple[list[int], ...])`
for hybrid memory allocator (multi-group cache).

**Lifecycle of a KV transfer** (consumer/decode side):
```
Scheduler step (recv side):
  get_num_new_matched_tokens(req) → (seq_len, is_async=True)  # remote KV available
  KVCacheManager.allocate(req, ...)                            # local block reservation
  update_state_after_alloc(req, blocks, num_external_tokens)   # connector notes "recv into these"
  build_connector_meta(scheduler_output)                       # produces NixlConnectorMetadata, etc.

Worker forward:
  bind_connector_metadata(meta)
  start_load_kv(forward_context)                               # async issue NCCL/RDMA receives
  for layer in transformer.layers:
      wait_for_layer_load(f"layer_{i}")                        # pipelined: blocks only if not arrived
      out = layer(x, kv_cache[i], attn_meta)
      save_kv_layer(f"layer_{i}", kv_cache[i], attn_meta)      # if also producing for next stage
  wait_for_save()
  build_connector_worker_meta() → ModelRunnerOutput.kv_connector_output
```

**Per-impl notes** (most relevant for our project):

- **`P2pNcclConnector`**: NCCL p2p; per-request granularity; GPU-direct;
  role-pinned (producer or consumer at startup). Simple, fast. Per the agent,
  `kv_parallel_size=2` for 1P1D.
- **`NixlConnector`**: zero-copy RDMA via NIXL agents, ZMQ side-channel for
  handshake. **Symmetric — any instance can be producer or consumer for any
  request.** GPU-direct or CPU-staged. Per-block, per-layer, or unified-cross-layer.
  Layer-wise pipelining via `start_load_kv` / `wait_for_layer_load` / `save_kv_layer`.
  `NIXL_CONNECTOR_VERSION = 2`. **Most production-ready**, well-aligned with
  what Llumnix migration does. **First borrow target.**
- **`MooncakeConnector`**: Mooncake transfer engine; supports heterogeneous
  TP via `_get_tp_ratio()` (a useful precedent if we ever want different TP
  on src/dst). Bootstrap server registration.
- **`OffloadingConnector`**: CPU/disk offload; not P2P; not directly useful
  for our work but the layer-wise pipelining pattern is identical.
- **`HF3FSKVConnector`**: shared distributed cache with central metadata server;
  conceptually different from migration.

**Worker integration site** (`v1/worker/kv_connector_model_runner_mixin.py` —
the mixin used by `Worker`/`GPUModelRunner`):
```python
with self._get_kv_connector_output(scheduler_output):
    self.bind_connector_metadata(scheduler_output.kv_connector_metadata)
    self.start_load_kv(get_forward_context())
    # ... model forward ...
    self.wait_for_save()
    self.get_finished(finished_req_ids)
    self.build_connector_worker_meta()
    self.clear_connector_metadata()
```

**Connector instantiation** (`vllm/distributed/kv_transfer/kv_transfer_state.py`):
```python
# Scheduler:
self.connector = KVConnectorFactory.create_connector(
    config=vllm_config, role=KVConnectorRole.SCHEDULER, kv_cache_config=...,
)
# Worker (via ensure_kv_transfer_initialized()):
_KV_CONNECTOR_AGENT = KVConnectorFactory.create_connector(
    config=vllm_config, role=KVConnectorRole.WORKER, kv_cache_config=...,
)
```

**Synchronization with compute**: layer-wise pipelining is the central trick.
`start_load_kv` issues NCCL/RDMA receives on a side stream/agent thread; compute
flows on the main stream and only blocks at `wait_for_layer_load(layer_i)` if
that layer's KV isn't ready. `save_kv_layer` is non-blocking; final
`wait_for_save()` synchronizes.

**Failure / abort paths**:
- `get_block_ids_with_load_errors()` → reschedule those requests for recompute
  (or fail per `kv_load_failure_policy`).
- `handle_preemptions(metadata)` cleans up async ops before scheduler frees blocks.
- Instance death → connection timeouts in NIXL handshake → marked failed; revert.

**PP/TP interaction** (per A8):
- TP must typically match between producer and consumer (Mooncake supports
  heterogeneous via tp_ratio).
- Connector is PP-aware in the sense that transfers are per (rank, layer) in
  the TP/PP layout. **For the natural PP-reshard scenario, this means: we want
  to hand off "the upper-half-layer KV from instance A's PP-rank-0 worker to
  instance B's PP-rank-0 worker" → TP=1 in our base case, PP=1 on each
  instance, but the connector still has to be instructed which layers to transfer.**

### §3.3.5 Disaggregated PD (A9) — borrow target #2

Where it lives:
- `vllm/entrypoints/serve/disagg/{protocol.py, serving.py, api_router.py, mm_serde.py}`
  — disagg HTTP serving (Tokens IN/OUT API, GenerateRequest with `kv_transfer_params`).
- `vllm/distributed/kv_transfer/` — the actual transport (KV connectors above).
- `vllm/v1/core/sched/scheduler.py` — scheduler creates `KVConnectorRole.SCHEDULER`
  at L128–132 if `vllm_config.kv_transfer_config` is set.
- `vllm/config/kv_transfer.py:KVTransferConfig` — fields
  `kv_connector` (str), `kv_role` (`"kv_producer" | "kv_consumer" | "kv_both"`),
  `kv_rank` (0 / 1), `kv_parallel_size` (typically 1, 2 for P2P NCCL), and
  connector-specific `kv_connector_extra_config: dict`.
  Plus `kv_load_failure_policy ∈ ["recompute", "fail"]`, `cache_salt`, etc.

Examples (`examples/`):
- `offline_inference/disaggregated-prefill-v1/{prefill_example.py, decode_example.py, run.sh}`
- `online_serving/disaggregated_serving/{disagg_proxy_demo.py, moriio_toy_proxy_server.py, mooncake_connector/mooncake_connector_proxy.py}`
- `online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_example_p2p_nccl_xpyd.sh`
- `others/lmcache/{disagg_prefill_lmcache_v0.py, disagg_prefill_lmcache_v1/}`

Tests (`tests/`):
- `v1/kv_connector/unit/test_remote_prefill_lifecycle.py` — full lifecycle
  (request goes to `WAITING_FOR_REMOTE_KVS`, KVs arrive, transitions to RUNNING)
- `v1/kv_connector/unit/test_remote_decode_lifecycle.py`
- `v1/kv_connector/unit/test_nixl_connector.py`, `test_lmcache_connector.py`,
  `test_mooncake_connector.py`, `test_p2p_nccl_connector.py`
- `v1/kv_connector/nixl_integration/test_disagg_accuracy.py` — E2E correctness
- `v1/kv_connector/unit/test_kv_connector_lifecycle.py`
- `entrypoints/serve/disagg/test_serving_tokens.py`, `test_generate_stream.py`

**End-to-end request flow** (1P1D pattern):
```
USER ──► PROXY ──► PREFILL_INSTANCE (kv_role="kv_producer", max_tokens=1)
                   ├─ runs prefill, samples first token
                   └─ on finish: connector.request_finished() → kv_transfer_params
                        (transfer_id, remote_engine_id, remote_block_ids,
                         remote_tp_size, remote_dp_size, ...)
        ◄── response with first token + kv_transfer_params

PROXY ──► DECODE_INSTANCE (kv_role="kv_consumer", request includes kv_transfer_params)
          ├─ scheduler sees kv_transfer_params → request enters WAITING_FOR_REMOTE_KVS
          ├─ pre-allocates local blocks for remote KV
          ├─ worker connector pulls blocks from prefill instance
          ├─ when finished_recving signal → request → RUNNING
          ├─ generates remaining tokens
          └─ streams tokens back to proxy

USER ◄── STREAM
```

**Pull vs push**: mostly pull. Decode worker fetches KV from prefill instance.
Mooncake uses a ZMQ "ready" signal but actual bytes are pulled layer-by-layer.

**Limitations** (per A9, important for our extension):
1. 1P1D only — one prefill, one decode (P2P NCCL needs `kv_parallel_size=2`).
2. Single consumer per producer (no broadcast).
3. **No mid-generation migration**. Once decode starts, KV stays. → **This is
   exactly the gap PP-reshard fills.**
4. Failure model: recompute-or-fail; not as resilient as Llumnix's per-stage abort.
5. Some attention backends incompatible (NIXL prefers HND layout / FlashAttention
   et al.).
6. No built-in proxy/router — user provides their own.

**Comparison with Llumnix migration** (per A9, well-articulated):
- *Granularity*: disagg PD = at request boundary (after prefill). Llumnix =
  at stage boundary (mid-generation, multi-stage pipelined).
- *Direction*: disagg = pull. Llumnix = push (each stage commits to dst).
- *Lifecycle*: disagg = WAITING_FOR_REMOTE_KVS until full transfer. Llumnix
  = pipelined Stage-0..Stage-N with overlap.
- *Failure*: disagg = recompute-or-fail. Llumnix = handshake with PRE-ALLOC /
  ACK / ABORT / COMMIT, finer-grained.
- *Append-only exploit*: Llumnix yes; disagg no (transfers full blocks).
- *Orchestration*: disagg = simple proxy + 1P1D pair; Llumnix = global
  scheduler picks pairs dynamically.

→ For PP-reshard, we want the *Llumnix* style (live, mid-generation,
pipelined, abort-able) but reusing the *disagg* layer-wise pipelining
primitives in vllm-latest's `KVConnectorBase_V1`. The borrow recipe is roughly:
- Adopt `KVConnectorBase_V1`'s scheduler-side / worker-side hook split.
- Adopt `wait_for_layer_load` / `save_kv_layer` / `wait_for_save` for layer
  pipelining — this gives us the multi-stage pipeline for free.
- Keep Llumnix's scheduling policy (global scheduler picking pairs based on
  load).
- Add a layer-range field to the connector metadata so layers M..L go one
  way and we don't ship layers 0..M.

### §3.3.6 Things flagged for direct verification

- [V12] read `v1/core/sched/scheduler.py` end to end — about 600 LOC, but it's
  the central piece we'll modify. I want to see how `kv_connector_metadata`
  actually flows from scheduler to executor to worker.
- [V13] read `v1/worker/kv_connector_model_runner_mixin.py` — the worker-side
  hook integration.
- [V14] read `v1/distributed/kv_transfer/kv_connector/v1/base.py` — the ABC.
  Need to confirm exact method signatures the agent paraphrased.
- [V15] read `v1/distributed/kv_transfer/kv_connector/v1/nixl/connector.py`,
  `nixl/scheduler.py`, `nixl/worker.py`, `nixl/metadata.py` — most aligned
  borrow target.
- [V16] read `v1/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`
  — simplest borrow target if NIXL is too heavy.
- [V17] read `v1/worker/gpu_model_runner.py` around L3787–4137 (the PP path).
- [V18] read `v1/engine/core.py:step_with_batch_queue()` — for understanding
  the existing PP bubble-hiding loop.
- [V19] read `v1/worker/ubatching.py` — microbatching primitives that
  could be repurposed for hybrid PP=1+PP=2 in same step.
- [V20] read `examples/online_serving/disaggregated_serving/disagg_proxy_demo.py`
  — reference design for the proxy / router pattern.
- [V21] read `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py`
  — best illustration of the request-state machine for KV transfer.
- [V22] read `vllm/v1/request.py` — find the `WAITING_FOR_REMOTE_KVS` state
  and other transitions.
- [V23] read `vllm/config/kv_transfer.py` — exact fields and validation.
- [V24] read `vllm/distributed/kv_transfer/kv_connector/factory.py` — confirm
  registration list and how a custom connector would be registered.


## §3.4 artifact-llumnix (DONE — light pass)

Light map already returned. Summary:

```
artifact-llumnix/llumnix/
├── artifact/
│   ├── 0_getting_started/
│   ├── 62_migration_efficiency/   → Figure 10
│   ├── 63_serving_performance/    → Figure 11
│   ├── 64_support_for_priorities/ → Figure 13
│   ├── 65_auto_scaling/           → Figures 14 & 15
│   ├── README.md
│   └── kill_exp.sh
├── benchmarks/
│   ├── benchmark_serving.py
│   ├── benchmark_latency.py
│   └── benchmark_throughput.py
├── vllm/        (vendored vLLM submodule — older v0.1.x / v0.2.0)
├── llumnix/     (the artifact's vendored Llumnix lib — older than the fork)
├── setup.py
└── requirements.txt
```

Each `6X_*` dir contains:
- `llumnix_exp` (Python driver),
- `run.sh` (orchestrator),
- `config/` (workload params: QPS, distribution, dispatch strategy, migration
  thresholds; uses ShareGPT, BurstGPT, generated power-law),
- `plot/plot.py` (post-processes `*.data` → `*.png` and `*.claim`).

**Artifact vs. fork**: the artifact is vLLM-integrated (patches vLLM v0.1.x / v0.2.0
directly, requires Ray); the main `reshardLLM/` fork is the *standalone* lineage
(separate `llumnix/` Python lib + separate `vllm/` directory). Per the paper Appendix
A, this is the difference between the OSDI prototype and the post-submission release.

**Implication for us**: when reproducing Llumnix figures with `reshardLLM/`, we cannot
reuse the artifact's `run.sh` directly — the entry-point CLI changed. But the
*workload configs* (request rates, CV, length distributions) and the plot scripts are
useful references.

---

# §4. Concrete implementation plan

> Generated 2026-04-27 after a second deep-read pass over the actual code.
> Files I read end-to-end myself (line-level accuracy on names/signatures):
> - `reshardLLM/llumnix/llumlet/migration_coordinator.py` (473 LOC)
> - `reshardLLM/llumnix/backends/vllm/migration_backend.py` (459 LOC)
> - `reshardLLM/llumnix/backends/vllm/scheduler.py` (273 LOC)
> - `reshardLLM/llumnix/llumlet/llumlet.py` (183 LOC)
> - `reshardLLM/llumnix/llumlet/request.py` (140 LOC)
> - `reshardLLM/llumnix/internal_config.py` (91 LOC)
> - `vllm-latest/vllm/distributed/kv_transfer/kv_connector/v1/base.py` (663 LOC)
> - `vllm-latest/vllm/distributed/kv_transfer/kv_connector/factory.py` (228 LOC)
> - `vllm-latest/vllm/distributed/kv_transfer/kv_transfer_state.py` (79 LOC)
> - `vllm-latest/vllm/v1/request.py` (342 LOC)
> - `vllm-latest/vllm/v1/core/sched/output.py` (262 LOC)
> - `vllm-latest/vllm/config/kv_transfer.py` (123 LOC)
>
> Plus 5 deep-dive Explore-agent reports on:
> - Llumnix backend internals (worker.py, llm_engine.py, sequence.py,
>   backend_interface.py ABC, migration_backend_interface.py ABC, vllm_v1/)
> - vllm-latest NIXL connector (connector/scheduler/worker/metadata)
> - vllm-latest P2P-NCCL connector (connector/engine/tensor_pool)
> - vllm-latest v1 GPU model runner / ubatching / KV connector mixin
> - vllm-latest disagg-PD lifecycle tests + examples
> - vllm-latest example/hidden-states/multi connectors
>
> Cited paths are repo-rooted; class/method names are verbatim. Tagged [A?]
> = unverified assumption, addressed in the risk register §4.0.4.

## §4.0 Plan-level decisions

### §4.0.1 Eight design decisions

**D1. Target backend = vLLM v0 first.** The Llumnix fork's v1 backend at
`llumnix/backends/vllm_v1/` is essentially a stub: only `async_core.py`
(269 LOC), and it merely wraps `EngineCore` with one
`disagg.init(vllm_config, self)` hook (line 138). The v0 backend at
`llumnix/backends/vllm/` is complete: `LLMEngineLlumnix`,
`SchedulerLlumnix`, `LlumnixRayGPUExecutor`, `MigrationWorker`,
`SequenceGroupLlumnix`, full migration. We build PP-reshard on v0 first,
then port to v1 as a separate phase (Phase 6).

**D2. Transport for the one-time KV ship at split** = extend the existing
`RayColMigrationBackend.do_send` / `do_recv`
(`migration_backend.py:374` and `:393`) with a `layer_range: Tuple[int,int]`
parameter. Both methods *already* iterate layer-by-layer
(line 384: `for layer_idx in range(self.cache_engine[0].num_attention_layers)`).
The change is to gate the loop body on `layer_idx ∈ [layer_range[0], layer_range[1])`.
Tiny code change; biggest functional payoff.

**D3. Each instance always runs `pipeline_parallel_size = 1`** from vLLM's
view. Both instances load the *full* model. The "PP=2 across instances" is
a Llumnix-level concept implemented by per-request layer masking, not
vLLM's PP machinery. Why: stock vLLM's `make_layers()` (with `PPMissingLayer`
placeholders at `model_executor/models/utils.py`) hardwires the layer slice
at load time, which is incompatible with serving hybrid PP=1 + PP=2 batches
in the same step. Trade-off: each instance pays full model-weight memory
cost — acceptable because Llumnix's deployment assumes the model fits per
instance anyway (TP-sharded within node if needed).

**D4. Per-request layer subset** is a new field
`pp_reshard_role: Optional[PPReshardRole]` on `LlumnixRequest`. Enum:
`HEAD` (this instance owns layers `0..M-1`), `TAIL` (layers `M..L-1`),
`None` (normal full-model request). For the MVP `M` is a global config
constant (e.g. 16 of 32 for LLaMA-7B). Phase 4 generalizes M per-request.

**D5. Cross-instance hidden-state transport** at every iteration:
- Phase 1 (MVP): **Ray RPC** (`backend_engine.send_hidden_state.remote`).
  Same plumbing as Llumnix's existing control plane. Latency ≈ low ms; this
  is fine for correctness validation. Equivalent to the existing
  `dst_instance_actor.execute_migration_method_async.remote(...)` pattern
  used in `MigrationCoordinator._dst_pre_alloc_cache`
  (migration_coordinator.py:386).
- Phase 3 (perf): **NCCL p2p send/recv on a side group**, sub-µs.

**D6. Initial scheduling trigger = manual API.** A new method on `Llumlet`:
`pp_reshard_out(request_id, dst_actor, dst_instance_id, split_layer)`.
No automatic policy until Phase 4. Mirrors how `migrate_out` is exposed
today at `llumlet.py:165`.

**D7. Demo & evaluation target = single-request OOM avoidance.**
LLaMA-7B on a single A10 (24GB). Long input + long output such that the
request would normally hit decode-time preemption (KV cache exhausts free
GPU blocks). Reshard mid-decode to a partner instance with free room.
Verify the output token sequence matches a reference run on a 2×A10 box.

**D8. Each step is testable in isolation.** Many steps are pure refactors
that don't change behavior; we validate by re-running existing tests. The
behavior-changing steps each get an explicit test.

### §4.0.2 Phase overview

| Phase | Goal | Effort | Key risk |
|---|---|---|---|
| 0 | Env, baseline reproduction, smoke | 1 wk | none |
| 1 | MVP: manual reshard E2E on toy model, "stop-the-world" mode | 4–6 wk | model.forward layer-mask plumbing, iteration barrier |
| 2 | Correctness — EOS/preempt/OOM/abort during reshard | 2 wk | combinatorial test cases |
| 3 | Perf — pipelined KV ship + NCCL hidden-state pipe + overlap | 3–4 wk | NCCL stream coordination |
| 4 | Auto-trigger PP-reshard policy | 2–3 wk | choosing M and dst |
| 5 | Merge back to PP=1 | 1–2 wk | symmetric to Phase 1 |
| 6 | Port to v1 backend / vllm-latest with `KVConnectorBase_V1` | 4–6 wk | v1 backend is ~stub today |
| 7 | Evaluation — paper figures | 3–4 wk | trace generation, baselines |

Total nominal: ~5–6 months, but Phase 1's 4–6 wk is the gating critical-path
item and the remaining phases can have more parallelism between us.

### §4.0.3 Test taxonomy used throughout this plan

For each step we say "Test:" and "Verify:". The test-type tags below tell
you what level of test we're talking about.

- **U1 (pure unit, no GPU)** — pytest, run on laptop. Imports + mocks. Fast.
- **U2 (GPU unit, single instance)** — pytest, single A10/A100. Hits the
  vLLM engine but with a tiny model (`facebook/opt-125m` or `llama-1B`).
- **I1 (integration, two instances on one machine)** — two Llumnix
  instances on different GPUs of the same machine, talking via Ray local
  cluster. The bread-and-butter test for PP-reshard.
- **I2 (integration, two machines)** — full Llumnix cluster, real network.
  Phase 3+ only.
- **B (microbenchmark)** — measures latency / throughput / bytes-moved for
  a specific operation. Phase 3+ only.

### §4.0.4 Risk register

Numbered so I can refer back to them in steps.

**R1 [A?]** vLLM v0's `LlamaModel.forward` does not accept a per-request
layer mask. We will have to add the mask in `forward_context` or as an
extra arg, then patch `LlamaModel.forward`'s layer loop to honor it.
*Mitigation*: in §4.2 Step 1.8 we read `vllm/model_executor/models/llama.py`
fully and design the patch. If the patch is too invasive, fall back to a
*subclass* `LlamaModelPPReshard(LlamaModel)` that we register conditionally.

**R2 [A?]** `BackendVLLM` already serializes some operations to step
boundaries via `_step_done_event_queue` (per A20). We will reuse this
pattern to inject the reshard at a step boundary so the engine isn't in
the middle of a forward pass when reshard begins.
*Mitigation*: in §4.2 Step 1.7 we read `BackendVLLM._start_engine_step_loop`
to confirm the queueing semantics.

**R3 [A?]** Two independent step loops (one per instance) need an
iteration-level barrier in PP=2 hybrid mode. They have separate Ray actors
with independent asyncio loops.
*Mitigation*: §4.2 Step 1.7 designs a "head waits for tail's hidden-state
ack" flow per iteration.

**R4 [A?]** `RayColMigrationBackend.dummy_cache` (line 266 of
migration_backend.py) is shape
`(num_migration_buffer_blocks, migration_num_layers, 2, migration_cache_size)`.
For partial-layer transfer we send fewer layers; the buffer math still holds
because we just don't fill all layer slots, but we need to be careful with
the receive side (it expects the same layer count).
*Mitigation*: in §4.2 Step 1.3 we add a `layer_range` to `do_send`/`do_recv`
that is also passed through `recv_cache`'s outer loop.

**R5 [A?]** `BlockManagerLlumnix.add_block_table` (scheduler.py:48) and
`pre_alloc_cache` (scheduler.py:131) work as-is for "this request lives on
this instance starting now" — they don't carry "exclusively". For
PP-reshard, the destination only ever owned layers M..L-1's KV (which is
just blocks; no per-layer block IDs in vLLM, blocks are global within a
worker).
*Mitigation*: confirmed by reading `scheduler.py` — block tables map
seq_id → list[Block], and the same block IDs index into all layer
tensors `gpu_cache[i]` for `i ∈ [0, num_layers)`. Per-layer "ownership" is
not a thing in vLLM's data model. So a TAIL instance allocating blocks for
a request will hold KV for *all* layers' positions for that request, but
will only ever populate / read the layers M..L-1 of those blocks. The KV
slots of layers 0..M-1 in those blocks remain garbage and unread. ✅ no
change needed in BlockManagerLlumnix.

**R6 [A?]** Phase 3 NCCL p2p between instances requires a coordinated init.
Llumnix's `RayColMigrationBackend.init_backend`
(`migration_backend.py:275`) already does ray-collective init via
`col.init_collective_group(world_size, rank, backend, group_name)`. We add
a separate group for the hidden-state pipe.

**R7 [A?]** vLLM v1 has a clean `WAITING_FOR_REMOTE_KVS` state
(`vllm-latest/vllm/v1/request.py:304`). Llumnix v0's `RequestStatus`
(request.py:42-46) only has RUNNING/WAITING/FINISHED/RUNNING_MIGRATING/
WAITING_MIGRATING. We add `RUNNING_PP_RESHARD_HEAD`,
`RUNNING_PP_RESHARD_TAIL`, `WAITING_FOR_RESHARD_KV` (the last is the brief
window during which the upper-layer KV is in transit).

**R8** [A?] AGENTS.md notes uv-based dev environment for vllm-latest. Our
fork uses pip + setup.py for Llumnix and a vendored vLLM. Reconciling
build hygiene with vllm's workflow happens at Phase 6, not now.

---

## §4.1 Phase 0 — Environment & baseline (≈ 1 week)

Goal: dev environment that builds, plus a baseline run we can compare PP-reshard against.

### Step 0.1 — Establish dev environment

- **Goal**: machine where one can `pip install -e reshardLLM/`, import
  llumnix and vllm 0.6.3.post1, and run pytest.
- **Files touched**: none (shell + venv).
- **Action**:
  1. Create venv with Python 3.10 (Ray + cupy + Gloo dependency surface).
     `python3.10 -m venv .venv && source .venv/bin/activate`.
  2. Install vLLM 0.6.3 from the vendored `vllm/vllm/` source (per
     `requirements/requirements_vllm.txt`).
  3. `pip install -e reshardLLM/`.
- **Test (U1)**:
  ```bash
  python -c "import llumnix, vllm; print(llumnix.__version__, vllm.__version__)"
  ```
- **Verify**: prints Llumnix `0.0.1` and vLLM `0.6.3.post1`.

### Step 0.2 — Baseline: existing test_migration.py passes

- **Goal**: confirm Llumnix's own end-to-end migration works on this box.
- **Files**: none (run existing test).
- **Action**: `pytest tests/e2e_test/test_migration.py -v -s`.
- **Test (I1)**: this test launches two Llumnix instances on two GPUs and
  exercises a full migration.
- **Verify**: passes; logs show `migrate done, migrate request [...]`.
  Note any flakes — they will recur in our work, and we want the baseline
  to be stable before we modify anything.

### Step 0.3 — Reproduce Fig. 11-style serving benchmark (optional, but strongly recommended)

- **Goal**: end-to-end traces on a representative workload, so we have a
  before-and-after for the paper.
- **Files**: none (run existing scripts).
- **Action**: `bash scripts/simple_bench.sh` (or
  `scripts/start_4_tp2_instances.sh` if 4 GPUs available). Capture the
  per-request P50/P99 latency JSON / npy output that
  `benchmark/benchmark_serving.py` writes.
- **Test (I1 or I2)**: depending on testbed.
- **Verify**: latency numbers in the same order of magnitude as the
  Llumnix paper's Fig. 11 for the matching configuration.

### Step 0.4 — Pin vllm-latest commit

- **Goal**: a stable target for Phase 6 port.
- **Files**: this notes file (record decision) and **eventually**
  `requirements/requirements_vllm_v1.txt` (don't change yet).
- **Action**: agree on a commit hash on `vllm-project/vllm` main branch.
  Record here in §99.Q2.12.
- **Test**: none.
- **Verify**: noted in §99.

### Step 0.5 — Identify the demo workload for our PP-reshard runs

- **Goal**: a single, repeatable workload that's small enough to debug but
  realistic enough for the paper claim.
- **Action**: pick (model, prompt size, output length, request rate)
  combinations:
  - **Toy** — `facebook/opt-125m`, 16 layers, prompt 64, output 64. Used
    for U2 and I1 sanity tests. Reshards in milliseconds.
  - **Small** — LLaMA-1B (or similar), 24 layers, prompt 1024, output 512.
    Used for I1 correctness validation.
  - **Realistic** — LLaMA-7B, 32 layers, prompt 4096, output 2048. Used
    for the OOM-avoidance demo (D7).
- **Files**: none.
- **Verify**: documented here.

---

## §4.2 Phase 1 — Minimal viable PP-reshard, "stop-the-world" mode (≈ 4–6 weeks)

Goal: a single request can be split across two Llumnix instances, finish
correctly, and produce identical output to a non-reshard reference run.
All tests use the **toy** workload (OPT-125m). No performance tuning yet.

### Big picture

We will execute reshard at a **step boundary** with a "stop-the-world"
protocol:
1. The global controller (or a manual API call) tells SRC to reshard
   request R to DST at split layer M.
2. SRC inserts a marker into its `_step_done_event_queue` so reshard runs
   between step k and step k+1.
3. SRC removes R from its running queue.
4. SRC pre-allocates blocks on DST via `pre_alloc_cache`, ships KV blocks
   for layers `M..L-1` via the extended `RayColMigrationBackend.do_send`,
   ships `SequenceGroupMetadata` (sampling state etc.) like the existing
   migration commit does.
5. DST commits the request as TAIL; SRC commits it as HEAD.
6. From step k+1 onward, SRC executes layers `0..M-1` for R, computes
   hidden state, ships it via Ray RPC to DST. DST executes layers
   `M..L-1`, samples a token, ships token id back to SRC. Both instances
   see R in their own `running` queues, with their own block tables, but
   each only executes "their" layers and references "their" KV.
7. Other (non-resharded) requests on each instance run normally with all
   layers.

The "stop-the-world" simplification: while the KV ship is happening (step 4)
and during step k+1's hand-off, we don't try to overlap with computation.
Phase 3 fixes that.

### Step 1.1 — `LlumnixRequest.pp_reshard_role` field + role enum

- **Goal**: the data type that identifies a request as HEAD or TAIL.
- **Files**: `llumnix/llumlet/request.py` (modify), `llumnix/__init__.py` (export).
- **Action**: in `request.py` after the `RequestStatus` enum (line 41), add:
  ```python
  class PPReshardRole(str, Enum):
      HEAD = "pp_reshard_head"   # this instance owns layers [0, split_layer)
      TAIL = "pp_reshard_tail"   # this instance owns layers [split_layer, num_layers)
  ```
  In `LlumnixRequest.__init__` (line 54), add fields:
  ```python
  self.pp_reshard_role: Optional[PPReshardRole] = None
  self.pp_reshard_split_layer: Optional[int] = None
  self.pp_reshard_partner_instance_id: Optional[str] = None
  self.pp_reshard_partner_actor: Optional[ray.actor.ActorHandle] = None
  self.pp_reshard_iteration_event: Optional[asyncio.Event] = None  # for stop-the-world barrier
  ```
  In `reset_migration_states_dst` and `reset_migration_states_src`: do NOT
  reset these fields (they outlive a migration).
- **Test (U1)**: `tests/unit_test/llumlet/test_request.py` — instantiate
  a `LlumnixRequest`, set the role, serialize-deserialize via Ray pickle
  (use `ray.cloudpickle.loads(ray.cloudpickle.dumps(req))`).
- **Verify**: round-trip preserves the new fields. The `actor` field is a
  Ray actor handle; we test that it pickles (Ray supports this).

### Step 1.2 — Extend `MigrationConfig` with `enable_pp_reshard`

- **Goal**: a feature flag so we can ship code without changing default
  behavior.
- **Files**: `llumnix/internal_config.py` (modify), `llumnix/config/default.py`
  (modify), `llumnix/arg_utils.py` (likely modify — add CLI arg).
- **Action**:
  1. In `internal_config.py:MigrationConfig.__init__` (line 52), add
     `enable_pp_reshard: bool = False` and `pp_reshard_split_layer: int = -1`
     (-1 means "auto: half of num_layers"). Save on self.
  2. In `config/default.py` add corresponding `_C.MANAGER.MIGRATION.ENABLE_PP_RESHARD = False`
     and `_C.MANAGER.MIGRATION.PP_RESHARD_SPLIT_LAYER = -1`.
  3. In `arg_utils.py`, surface as CLI flags.
- **Test (U1)**: a unit test that loads a YAML config with the flag set
  and asserts `migration_config.enable_pp_reshard is True`.
- **Verify**: existing tests still pass (default = False).

### Step 1.3 — `RayColMigrationBackend.do_send` / `do_recv` accept `layer_range`

- **Goal**: ship only KV for layers `[layer_start, layer_end)` instead of
  all layers.
- **Files**: `llumnix/backends/vllm/migration_backend.py` (modify
  `do_send` line 374, `do_recv` line 393, `recv_cache` line 341).
  Also `llumnix/backends/migration_backend_interface.py` to update the ABC.
- **Action**: change signatures to accept an optional
  `layer_range: Optional[Tuple[int, int]] = None`. Inside the loop on
  line 384 (`do_send`):
  ```python
  for layer_idx in range(self.cache_engine[0].num_attention_layers):
      if layer_range is not None and not (layer_range[0] <= layer_idx < layer_range[1]):
          continue
      cache_idx = layer_idx % self.migration_num_layers
      self.cache_engine[virtuel_engine].attn_backend.swap_blocks(...)
      if cache_idx + 1 == self.migration_num_layers or layer_idx + 1 == self.cache_engine[0].num_attention_layers:
          col.send(send_cache, dst_worker_handle, self.group_name)
  ```
  Mirror in `do_recv` line 404 with the same `layer_range` filter on the
  iteration. The dummy_cache buffer logic is unchanged; we just send fewer
  payloads in `do_send` (and skip the matching `col.recv` calls in `do_recv`).
  **Important**: `cache_idx == self.migration_num_layers - 1` triggers the
  `col.send`; with a layer_range, we may finish the range mid-buffer. Need
  to flush the buffer at end of layer_range too. Carefully handle this edge.
- **Also**: extend `RayRpcMigrationBackend.do_send` / `do_recv` (lines 167
  and 185) symmetrically — the existing "recv all layers, copy each into
  GPU cache" logic also needs the layer-range filter.
- **Test (U2 — needs GPU)**:
  ```python
  # Two cache_engines on GPU. Allocate identical blocks.
  # Call do_send(blocks, layer_range=(8, 16)) on src.
  # Call do_recv(blocks, layer_range=(8, 16)) on dst.
  # Assert: dst.gpu_cache[layer i for i in [8,16)] == src.gpu_cache[layer i] for those blocks.
  # Assert: dst.gpu_cache[layer 0..7 and 16..N] is unchanged (zero or whatever it was).
  ```
- **Verify**: layers in range match exactly; layers outside untouched.

### Step 1.4 — `BackendVLLM.send_cache` / `recv_cache` thread `layer_range` through

- **Goal**: plumb `layer_range` from the engine down to the backend.
- **Files**: `llumnix/backends/vllm/llm_engine.py` (BackendVLLM around
  L434, L451 per A20), `llumnix/backends/backend_interface.py` ABC.
- **Action**: add `layer_range: Optional[Tuple[int,int]] = None` to
  `send_cache(...)` and `recv_cache(...)`. Pass it through:
  - `send_cache` calls `dst_instance_actor.execute_migration_method_async.remote("recv_cache", ..., layer_range)`
  - `recv_cache` calls `_run_workers_async("recv_cache", ..., layer_range=layer_range)`
  - Worker's `recv_cache` (`worker.py:131`) passes it to
    `migration_backend.recv_cache(..., layer_range)`.
- **Test (U1)**: mock the migration backend, call BackendVLLM.send_cache
  with a layer_range, verify the kwargs propagate through Ray (using
  `mock.patch` on `dst_instance_actor.execute_migration_method_async.remote`).
- **Verify**: the kwarg arrives at the worker.

### Step 1.5 — `MigrationCoordinator.pp_reshard_out` entry point

- **Goal**: a parallel of `migrate_out` (line 156) that performs a single
  one-stage layer-bounded transfer, plus tags both ends as HEAD/TAIL.
- **Files**: `llumnix/llumlet/migration_coordinator.py` (modify).
- **Action**: add a new method:
  ```python
  async def pp_reshard_out(self,
                          dst_instance_actor: ray.actor.ActorHandle,
                          dst_instance_id: str,
                          request_id: RequestIDType,
                          split_layer: int) -> MigrationStatus:
      # 1. Find the request. Must be RUNNING (not WAITING).
      req = self.backend_engine.get_running_request_by_id(request_id)  # NEW helper
      if req is None or req.finished:
          return MigrationStatus.ABORTED_SRC

      # 2. Like _migrate_out_onestage(is_last_stage=True), but:
      #    - ship KV with layer_range=(split_layer, num_layers)
      #    - DON'T remove req from src's running queue (HEAD keeps running)
      #    - DON'T free_src_request afterward
      #    - Tag req with HEAD role; tell DST to tag its replica as TAIL.

      blocks, token_ids, _ = await self.backend_engine.get_request_incremental_blocks(req, 0)

      response = await self._dst_pre_alloc_cache(
          dst_instance_actor, dst_instance_id, request_id,
          RequestStatus.RUNNING, req.request_arrival_time,
          len(blocks), token_ids, is_first_stage=True)
      if not response.success:
          return MigrationStatus.ABORTED_DST

      dst_blocks = response.return_value

      # Layer-bounded send:
      response = await self.backend_engine.send_cache(
          dst_instance_actor, blocks, dst_blocks, request_id,
          is_last_stage=True,
          layer_range=(split_layer, self.backend_engine.num_attention_layers))
      if not response.success:
          return MigrationStatus.ABORTED_DST

      # Commit on dst as TAIL:
      response = await self._dst_commit_pp_reshard_tail(
          dst_instance_actor, dst_instance_id, req, split_layer)
      if not response.success:
          # rollback: dst aborts; src request continues normally
          await self._dst_free_pre_alloc_cache(dst_instance_actor, dst_instance_id, request_id)
          return MigrationStatus.ABORTED_DST

      # Tag src as HEAD (in-place; no removal from running queue):
      req.pp_reshard_role = PPReshardRole.HEAD
      req.pp_reshard_split_layer = split_layer
      req.pp_reshard_partner_instance_id = dst_instance_id
      req.pp_reshard_partner_actor = dst_instance_actor
      return MigrationStatus.FINISHED
  ```
  And a sister method `commit_dst_pp_reshard_tail` (registered like the
  existing `commit_dst_request`, lines 442–444), which on the DST side:
  1. Allocates a new seq_id
  2. Installs the block table for the dst pre-allocated blocks
  3. Sets `req.pp_reshard_role = PPReshardRole.TAIL`,
     `req.pp_reshard_split_layer = split_layer`,
     `req.pp_reshard_partner_instance_id = src_instance_id`,
     `req.pp_reshard_partner_actor = src_actor`
  4. Adds the request to the running queue
- **Test (U1 with mocks)**: stub out `BackendVLLM.send_cache` and
  `_dst_commit_pp_reshard_tail`. Call `pp_reshard_out`. Assert that
  `req.pp_reshard_role == HEAD` afterward and the call sequence is
  pre_alloc → send_cache → commit.
- **Test (I1)**: full two-instance run; call `pp_reshard_out` from a
  driver script; assert that both instances now see the request with
  matching roles.
- **Verify (I1)**: SRC's `req.pp_reshard_role == HEAD`,
  DST's `req.pp_reshard_role == TAIL`, both have the same
  `split_layer`, the cross-pointers are set.

### Step 1.6 — Llumlet RPC surface: `pp_reshard_out`

- **Goal**: expose the coordinator method as a Ray RPC on the Llumlet
  actor (mirrors `llumlet.migrate_out`, line 165).
- **Files**: `llumnix/llumlet/llumlet.py`.
- **Action**: add
  ```python
  async def pp_reshard_out(self, dst_actor, dst_instance_id, request_id, split_layer):
      return await self.migration_coordinator.pp_reshard_out(
          dst_actor, dst_instance_id, request_id, split_layer)
  ```
  Plus a static helper / driver script under `notes/` that walks two
  Llumlets and triggers reshard manually for a given request ID.
- **Test (U1)**: instantiate two Llumlets locally (without GPU, mocking
  the engine), call `pp_reshard_out` on src; assert it calls coordinator's
  method.
- **Verify**: works.

### Step 1.7 — Iteration-level barrier (the "stop-the-world" execution)

- **Goal**: in HEAD/TAIL hybrid mode, both instances synchronize at every
  iteration: HEAD computes layer 0..M-1's hidden state for R, sends to
  TAIL; TAIL waits for it, runs M..L-1, samples, sends token id back to
  HEAD; both then proceed to the next iteration's batch on their own
  schedules.
- **Files**: `llumnix/backends/vllm/llm_engine.py`,
  `llumnix/backends/vllm/scheduler.py`,
  `llumnix/backends/vllm/worker.py` (the latter is where we'll need to
  hook into model.forward).
- **Action**:
  - Define a new internal control-plane RPC: `LlumletRPC.send_hidden_state(request_id, layer_idx, hidden_states_tensor) -> token_id`.
    Implementation on TAIL: takes the hidden state tensor, *injects it*
    into TAIL's next forward as the input to layer `M`, runs the rest,
    samples, returns the sampled token_id. Implementation on HEAD: at
    forward time, after layer M-1, *do not run more layers*, instead call
    this RPC, then write the returned token id into the request's
    `_output_token_ids`. This must happen at step boundary (engine step
    completes after the RPC returns).
  - Concretely (Phase 1 simplification): we add a "single-request mode"
    where the resharded request R is the *only* request running on both
    instances. Both schedulers see only R in `running`. SRC runs forward,
    intercepts after layer M-1 (see Step 1.8), Ray-RPCs to TAIL with
    hidden state, TAIL runs forward starting at layer M (see Step 1.8),
    samples, returns token id over Ray, SRC stuffs it into the request's
    output, both step loops advance.
  - Hybrid (HEAD batch contains R + other PP=1 requests): more complex —
    deferred to Phase 3. For Phase 1 MVP, *isolate* R: ensure both
    schedulers schedule only R while the reshard window is open. This is
    a non-realistic restriction but it lets us validate correctness
    without juggling sub-batches.
- **Test (I1, toy model)**: launch two instances, OPT-125m, single
  request "Hello world", reshard at layer 8 of 12 (after a few decode
  steps), let it complete, verify the output token sequence equals the
  reference (single-instance, no reshard).
- **Verify**: token-by-token equality with reference. (Stochastic
  sampling needs a fixed RNG seed.)

### Step 1.8 — Per-request layer mask in `model.forward`

- **Goal**: the model's transformer layer loop honors `layer_start..layer_end`.
- **Files**: `vllm/vllm/model_executor/models/llama.py` (or whichever
  model we use for the toy test — opt.py for OPT-125m). Plus the
  `forward_context` that v0 uses (need to verify whether v0 has one).
- **Action plan**: read the chosen model file's `forward()` (probably
  ~50 lines). It iterates `for i, layer in enumerate(self.layers)`. We
  want to:
  - take `layer_start: int = 0, layer_end: Optional[int] = None` as
    optional args (or read them from `IntermediateTensors` extras),
  - if `layer_end is None`, set to `len(self.layers)`,
  - run only `self.layers[layer_start:layer_end]`,
  - return the *intermediate hidden state* if `layer_end < len(self.layers)`
    (i.e., HEAD case),
  - or if `layer_start > 0`, expect the input to *be* a hidden state
    (i.e., TAIL case): skip embedding lookup.
- **Risk R1 mitigation**: the change is small and confined to the model
  class. We may subclass: `LlamaModelPPReshard(LlamaModel)` — keeps
  upstream model unchanged, register conditionally.
- **Plumbing**: `ModelRunner.execute_model` (vllm 0.6.3
  `worker/model_runner.py:1606`) builds `ModelInputForGPUWithSamplingMetadata`.
  We add fields `pp_reshard_layer_start`, `pp_reshard_layer_end` to the
  per-request metadata, and a per-batch reduction (since a Phase 1 batch
  is a single-request batch in the simple mode, the per-batch reduction
  is trivial: it equals the request's). For Phase 3 hybrid mode we'd
  per-row mask the attention computation; deferred.
- **Test (U2)**: tiny model (OPT-125m), call `model.forward` with
  `layer_start=0, layer_end=6`, then with `layer_start=6, layer_end=12`
  using the previous output as input. Compare to a one-shot
  `forward(layer_start=0, layer_end=12)` — must equal.
- **Verify**: outputs match to within numerical precision.

### Step 1.9 — `BackendVLLM` step loop tracks resharded requests

- **Goal**: when `req.pp_reshard_role == HEAD`, instead of running full
  forward, call the partner-RPC after layer M-1. When TAIL, the forward
  starts at layer M with the received hidden state.
- **Files**: `llumnix/backends/vllm/llm_engine.py`
  (`BackendVLLM._start_engine_step_loop`), `llumnix/backends/vllm/worker.py`,
  and the `model_runner` shim if needed.
- **Action**: at engine step time, in the per-request loop within
  `_process_model_outputs` (llm_engine.py:144), detect
  `pp_reshard_role`. The cleanest design is:
  - **HEAD**: when scheduling a step, mark this request to run with
    `pp_reshard_layer_end = split_layer`. After the model runner produces
    intermediate hidden states, call the partner Llumlet RPC
    (`partner_actor.execute_migration_method_async.remote("apply_pp_reshard_tail", request_id, hidden_states)`)
    and *await* the returned token id. Then write that token id into the
    request's output via `seq.append_token_id(...)` paths normally.
  - **TAIL**: do not schedule the request normally — the HEAD will inject
    it via the RPC. So the TAIL llumlet exposes a method
    `apply_pp_reshard_tail(request_id, hidden_states_tensor)` that:
      1. starts the model runner with `layer_start = split_layer` and the
         hidden_states pre-loaded as input,
      2. runs through to last layer + sampling,
      3. updates its block manager / KV cache normally for layers M..L-1,
      4. returns the sampled token id.
- **Caveat**: this couples HEAD and TAIL more tightly than we'd like. A
  cleaner Phase 3 design uses async push from HEAD with a poll-for-done on
  TAIL, instead of synchronous Ray RPC. For Phase 1 MVP correctness, sync
  is fine.
- **Test (I1, toy model)**: same as Step 1.7's test.
- **Verify**: same.

### Step 1.10 — Smoke test: end-to-end PP-reshard, single request

- **Goal**: an automated test that triggers a reshard and validates output.
- **Files**: `tests/e2e_test/test_pp_reshard.py` (new — but only as a
  *plan* here; we don't write the file until permission to do so).
- **Action**: write a pytest that:
  1. Launches two Llumnix instances (OPT-125m).
  2. Submits a single deterministic request (fixed seed, prompt,
     temperature=0).
  3. Lets it run for K=4 decode steps so KV is non-trivial.
  4. Calls `instance0.pp_reshard_out.remote(instance1, instance1_id, req_id, split_layer=6)`.
  5. Lets it run to completion (EOS or max tokens).
  6. Submits the same request to a single instance (no reshard) for reference.
  7. Asserts equality of output token ids.
- **Test (I1)**: 2× A10 or 2 GPUs locally.
- **Verify**: equal output. **This is the Phase 1 success criterion.**

### Phase 1 exit criteria

- [ ] Step 1.1–1.10 all pass on toy model.
- [ ] No regression in existing `tests/unit_test/` and
      `tests/e2e_test/test_migration.py`.
- [ ] Full-model demo (LLaMA-1B, Step 0.5 "small") at least runs end to
      end (correctness exact only on toy; numerical drift acceptable on
      LLaMA-1B due to attention kernel precision).
- [ ] Latency doesn't matter yet; just record it.

---

## §4.3 Phase 2 — Correctness under stress (≈ 2 weeks)

Phase 1 covered the happy path with a single resharded request and no other
load. Phase 2 adds the edges: the request finishes mid-reshard, gets
preempted, the destination OOMs, an instance dies, the reshard is aborted
by the controller. Each of these is one short step with a focused test.

For each sub-step the design pattern is the same: introduce the failure
mode, inspect Llumnix's existing equivalent failure handling for full
migration, mirror it for PP-reshard.

### Step 2.1 — Request hits EOS during reshard

- **Goal**: if R produces EOS while the KV ship is in flight, neither side
  ends up holding a stranded request.
- **Files**: `llumnix/llumlet/migration_coordinator.py` (modify
  `pp_reshard_out`), `llumnix/backends/vllm/scheduler.py`.
- **Reference**: existing migration handles this via
  `migrate_out_request.finished` check (migration_coordinator.py:310:
  `if not found or migrate_out_request.finished: return MigrationStatus.ABORTED_SRC`).
- **Action**: in `pp_reshard_out`, after each await point, check
  `req.finished`. If True, abort: tell DST to free pre-allocated cache via
  `_dst_free_pre_alloc_cache` (line 407), do not commit, leave the request
  on SRC where the engine's normal output processing will publish the EOS.
- **Test (U1 with mocks)**: stub out `pre_alloc_cache` to take a long
  time; meanwhile mark `req.finished = True`; assert the abort path runs
  and DST gets a free call.
- **Verify**: no stranded TAIL request on DST; SRC's request completes
  normally.

### Step 2.2 — Request preempted during reshard

- **Goal**: vLLM's scheduler may preempt R between when reshard was
  triggered and when the KV ship completes (e.g. another higher-priority
  request needs memory). Reshard must abort cleanly.
- **Files**: same as 2.1.
- **Reference**: `LlumnixRequest.should_abort_migration()` (request.py:135)
  detects preemption via `last_preemption_time > begin_time`.
- **Action**: insert `req.should_abort_migration()` checks at the same
  await points in `pp_reshard_out` as in existing
  `_migrate_out_onestage` (migration_coordinator.py:277, 285, 333, 350).
  On abort, free pre-alloc cache on DST; SRC keeps the request, which
  eventually re-runs from the preempted checkpoint.
- **Test (I1)**: deliberately overload SRC with a large second request
  while reshard is in flight; assert the reshard aborts and SRC's
  scheduler resumes R normally after preemption.
- **Verify**: DST's pre_alloc_cache_dict is empty post-abort; SRC's req
  still in `running` (or `swapped`).

### Step 2.3 — Destination OOMs during pre-alloc

- **Goal**: if DST cannot allocate enough free blocks, the reshard fails
  cleanly without leaking state.
- **Files**: `llumnix/backends/vllm/scheduler.py`
  (`SchedulerLlumnix.pre_alloc_cache`, line 131).
- **Reference**: `pre_alloc_cache` already returns
  `MigrationResponse(success=False, return_value=None)` when
  `block_manager.get_free_blocks` returns None (line 147). `pp_reshard_out`
  already handles `not response.success` (per Step 1.5 design).
- **Action**: validate by injecting OOM. Use
  `block_manager.get_num_free_gpu_blocks` patching to force a small free
  block count.
- **Test (U2)**: instantiate one instance, set `num_gpu_blocks=128`,
  attempt to pp_reshard a request requiring 256 blocks worth of KV.
- **Verify**: returns `ABORTED_DST`; SRC's request continues normally.

### Step 2.4 — Source instance dies mid-reshard

- **Goal**: DST notices SRC is gone and frees the pre-allocated blocks.
- **Files**: `llumnix/llumlet/migration_coordinator.py`
  (`watch_migrate_in_request_wrapper` lines 64–136 + the watch loop on
  line 461). This already exists for migration; we must wire pp_reshard
  into it.
- **Action**: decorate `pre_alloc_cache` (from pp_reshard) with
  `@watch_migrate_in_request_wrapper`. Already-decorated for migration —
  we reuse the same machinery. Add a `is_first_stage` semantic for
  pp_reshard: pp_reshard is single-stage, so the wrapper sees `is_start =
  is_first_stage = True` on `pre_alloc_cache` and `is_stop = True` on
  `commit_pp_reshard_tail`. The timeout loop on line 463 takes care of
  the rest.
- **Test (I1)**: kill SRC after pre_alloc_cache succeeds but before
  send_cache. Wait `PENDING_MIGRATE_IN_TIMEOUT` seconds. Verify DST's
  `pre_alloc_cache_dict[request_id]` is empty.
- **Verify**: `block_manager.get_num_free_gpu_blocks()` rebounds.

### Step 2.5 — Destination instance dies after commit but before first hybrid step

- **Goal**: SRC notices TAIL is gone, recovers R as a normal full request.
- **Files**: `llumnix/llumlet/llumlet.py` (add a partner-health check),
  `llumnix/backends/vllm/llm_engine.py` (engine step loop).
- **Action**: when HEAD's RPC to TAIL fails (Ray exception), HEAD sets
  `req.pp_reshard_role = None`, clears partner fields, and runs the full
  model normally on the next step. We rely on the fact that HEAD has KV
  for layers 0..M-1 of R and *had* layers M..L-1 before reshard but did
  not free them in Phase 1. **Wait**: in Phase 1's `pp_reshard_out`, did
  SRC free its layers M..L-1 KV? Looking at our Step 1.5 design — no, we
  didn't free anything. SRC's gpu_cache[layer_idx] for layer_idx ≥ M
  still holds the request's blocks at the same physical block IDs. ✅
  Recovery is straightforward: just unset the role.
- **Test (I1)**: pp_reshard, then kill DST before any hybrid step
  completes. Verify SRC continues R as a normal full request and
  finishes correctly.
- **Verify**: token output equals reference run.

> Caveat: in Phase 3 we may free SRC's layers M..L-1 KV after reshard
> completes (to reclaim memory). Then this recovery becomes harder
> (would have to re-prefill or recompute). For now we keep the SRC's
> upper-half KV alive as a recovery backstop. We'll revisit in §4.4.

### Step 2.6 — Controller-initiated abort of an in-flight reshard

- **Goal**: an admin / scheduler can cancel a reshard before commit.
- **Files**: `llumnix/llumlet/migration_coordinator.py`.
- **Reference**: there's no equivalent in Llumnix's full migration today.
- **Action**: add `MigrationCoordinator.abort_pp_reshard(request_id)`.
  Sets a flag `req._pp_reshard_abort = True`; `pp_reshard_out` polls this
  flag at await points, takes the same abort path as 2.1/2.2.
- **Test (U1)**: stub out send_cache to be slow; in another coroutine
  call `abort_pp_reshard`; assert ABORTED_SRC.
- **Verify**: works.

### Phase 2 exit criteria

- [ ] All 6 sub-steps' tests pass on toy model.
- [ ] All Phase 1 tests still pass (no regressions).
- [ ] Manual stress test: 100 sequential reshards on the same long-running
      request (reshard, immediately reshard back, repeat). No memory leaks
      (`block_manager.get_num_free_gpu_blocks()` returns to baseline).

---

## §4.4 Phase 3 — Performance: pipelined KV + NCCL hidden-state pipe + hybrid batching (≈ 3–4 weeks)

Phase 1 was correctness in single-request, stop-the-world mode. Phase 3
makes PP-reshard fast and concurrent: (a) hidden-state transfer happens on
NCCL p2p instead of Ray RPC, (b) KV ship overlaps with continuing decode
on SRC, (c) HEAD/TAIL share their batches with non-resharded PP=1
requests in the same step.

### Step 3.1 — Multi-stage pipelined KV ship for PP-reshard

- **Goal**: don't pause the request while the upper-half KV is shipping;
  instead use Llumnix's existing multi-stage protocol (Stage-0..Stage-N
  per Fig. 6 of the paper) to overlap with continuing decode on SRC.
- **Files**: `llumnix/llumlet/migration_coordinator.py`.
- **Reference**: `_migrate_out_multistage` (line 254) and
  `_migrate_out_onestage` (line 272) implement this pattern for full
  migration. The append-only KV property is exploited by
  `incremental_blocks[:-1]` (line 291).
- **Action**: replace the single-stage `pp_reshard_out` from Step 1.5
  with a multi-stage version. Each stage:
  1. SRC computes which blocks for layers M..L-1 are *new* since previous
     stage (`get_request_incremental_blocks` already handles "new since
     last stage" — line 97 of scheduler.py — but we need to confirm it
     returns block ids that index into all layers identically [A?]).
  2. SRC ships those blocks for layers M..L-1 (using the layer_range
     do_send from Step 1.3) while continuing to decode normally (R is
     still full-PP=1 on SRC, not yet resharded).
  3. After max_stages or "few enough new blocks" condition, SRC pauses
     R, ships final stage, commits HEAD/TAIL roles. The pause window is
     "one decode step" worth — Llumnix's near-zero downtime claim.
- **The trick**: for PP-reshard, the request *stays running on SRC* during
  pre-stages, generating new tokens normally with full PP=1. Only at the
  *commit* step do we swap into HEAD/TAIL mode. So the per-stage transfer
  is a side-channel KV pre-fetch from SRC's perspective.
- **Risk R4 mitigation in code**: confirm `dummy_cache` reshape math
  works when only some layer slots are filled. Probably do_send needs to
  send a smaller buffer per stage (only the layers in range), and do_recv
  matches.
- **Test (B + I1)**: microbenchmark — measure end-to-end reshard time vs.
  bytes shipped, for sequence lengths 256, 1024, 4096, 8192. Compare:
  - Phase 1's stop-the-world reshard time
  - Phase 3 multi-stage reshard time
  - Llumnix's existing full-migration time (which ships *both* halves)
- **Verify**: Phase 3's reshard time ≈ ½ of Llumnix's migration time
  (since we ship half the layers); downtime-the-request-is-paused window
  is constant in seq length, like the paper.

### Step 3.2 — NCCL p2p side group for cross-instance hidden-state pipe

- **Goal**: replace the Ray-RPC hidden-state hop (Step 1.7) with NCCL p2p.
- **Files**:
  - `llumnix/backends/vllm/migration_backend.py` (extend
    `RayColMigrationBackend.init_backend`, line 275).
  - `llumnix/llumlet/llumlet.py` (carry the new group name).
  - `llumnix/backends/vllm/worker.py` (add `send_hidden_states`,
    `recv_hidden_states` methods).
- **Action**:
  1. Establish a second collective group for hidden-state pipe between
     each pair of instances that may PP-reshard. Use the same
     `ray.util.collective` mechanism as today's KV migration. Group name
     could be `"hidden_state_{src_id}_{dst_id}"`.
  2. On each forward step where R is HEAD on SRC, after layer M-1 the
     model writes hidden states to a contiguous tensor; SRC calls
     `col.send(hidden_states, dst_worker_handle, hs_group)` on a
     dedicated `hidden_state_stream`. TAIL polls or syncs with `col.recv`,
     then runs layers M..L-1, samples, calls `col.send(token_id, src,
     hs_group)`.
  3. Cap the side group's bandwidth use so it doesn't starve the KV-ship
     stream: use a separate CUDA stream so the inference compute stream
     stays unblocked.
- **Test (I1, B)**: microbenchmark hidden-state hop latency vs. Ray RPC.
  Expected: 100× lower for small models, 5–10× lower for large models
  (where serialization is the bottleneck).
- **Verify**: tokens still match the Phase 1 reference output.

### Step 3.3 — Layer-pipelined hidden-state save on TAIL (the real win)

- **Goal**: hide the hidden-state RPC latency by overlapping TAIL's
  layer M's compute with SRC's layer M-1 computation of the *next*
  iteration. Borrow the layer-wise `wait_for_layer_load` pattern from
  vllm-latest's `KVConnectorBase_V1` (we read the abstract methods at
  base.py:317 and 331).
- **Files**: more invasive into the model_runner than Phase 1's design.
  Likely need a small hook inside `LlamaModelPPReshard.forward` that
  yields control after each layer.
- **Action plan**:
  1. Pre-issue: at iteration k, TAIL's worker pre-receives R's hidden
     state from iteration k (the one HEAD just sent). Compute begins.
  2. While TAIL is computing layers M..M+1 of iteration k, HEAD has
     already moved on to iteration k+1's layers 0..M-1.
  3. As soon as iteration k+1's hidden state is ready on HEAD, ship it.
     TAIL receives it concurrently with finishing iteration k.
  4. Net effect: pipelined PP=2 across instances, same shape as in-engine
     PP today. Per-iteration latency = max(HEAD layers 0..M-1, TAIL
     layers M..L-1, network hop), not the sum.
- **Test (B)**: per-token decode latency for HEAD-only vs. full-PP=1 vs.
  PP-reshard at split=L/2.
- **Verify**: PP-reshard per-token latency ≤ 1.2× of full-PP=1 per-token
  latency for layers within bandwidth budget. (Rough target — concrete
  number depends on model size and link.)

### Step 3.4 — Hybrid batching: HEAD batch contains R + non-resharded PP=1 requests

- **Goal**: in a real workload, SRC isn't dedicated to R — it's still
  serving other requests. We need the model_runner to handle a batch
  containing some requests that need `[0, L)` (normal) and others that
  need `[0, M)` (HEAD half).
- **Files**: vLLM's model_runner (which we'll patch) +
  `SchedulerLlumnix._schedule_running` (line 257).
- **Action plan** — two design options:
  - **Option A (preferred): per-request layer mask in `forward`**. The
    transformer loop runs all `L` layers, but for each request *r* we
    consult `r.layer_end` and either skip the per-r contribution (zero
    out the slot for that token at that layer) or stop attention reads
    after layer M-1 for that row. This is non-trivial because the
    attention kernel processes the whole batch atomically. We'd need to
    *split the batch* into "stop-at-M" rows and "stop-at-L" rows for
    each forward call, OR run two sub-batches. With ubatching v0 doesn't
    support layer-aware sub-batches (per A23's analysis of v1 — v0 has
    even less). Conclusion: not feasible without invasive runner changes.
  - **Option B: separate sub-batches**. The scheduler emits two
    `SequenceGroupMetadata` lists per step:
      - "full layers" list — runs `forward(layer_start=0, layer_end=L)`,
      - "head-only" list — runs `forward(layer_start=0, layer_end=M)`.
    Each is a separate forward pass, possibly serialized. CUDA streams
    can run them in parallel if memory allows; but for simplicity start
    serialized.
- **Recommendation**: Option B. Implement in
  `BackendVLLM._schedule_and_execute_async` (or wherever the engine step
  goes from scheduler output to model executor call). When the
  scheduler returns running_seq_groups, partition by
  `pp_reshard_role`. Run each partition through the model with the
  appropriate `layer_end`. Combine outputs.
- **Symmetrically on TAIL**: TAIL has "tail-only" requests + "full" requests.
- **Test (I1)**: 4 requests on SRC: 1 HEAD (resharded), 3 PP=1 (normal).
  Run for 100 decode steps. Verify all 4 produce expected outputs.
- **Verify**: tokens match reference; per-step latency reasonable.

### Step 3.5 — Free SRC's layers M..L-1 KV after reshard commits (memory reclaim)

- **Goal**: the whole point — once reshard is committed, SRC should free
  the upper-half KV blocks for R, freeing memory for other requests.
- **Files**: `llumnix/backends/vllm/scheduler.py` (a new method
  `partial_free_blocks(request_id, block_indices)`).
- **Caveat**: vLLM's block manager treats blocks as atomic (a block holds
  KV for *all* layers' positions in a chunk of 16 tokens). We can't
  free a block partially. So we *don't* free physical blocks; we just
  treat the upper-layer slots as *garbage* memory within those blocks.
  The blocks are still allocated to R on SRC. The memory reclaim
  benefit comes from the fact that *we don't allocate new blocks for
  layers M..L-1 going forward*, since R isn't running them on SRC.
- **Real reclaim**: at commit time, R is HEAD, future tokens added to R
  on SRC append to its block table for layers 0..M-1 (which is the same
  block table — vLLM doesn't have per-layer block tables). So new tokens
  *do* allocate new blocks for layers 0..M-1, but those blocks are also
  used (uselessly) for layers M..L-1 of those positions. **Hmm.**
- **Realization**: this means PP-reshard *doesn't* actually save memory
  for *future* tokens — it only saves for the prefix that's being kept
  on SRC. We must check: does it still help with OOM-avoidance (D7)?
  Yes — if R is about to OOM at moment T, the *current* KV is the
  problem. Future tokens are small and incremental; we have time to
  defrag or migrate. So PP-reshard buys us a one-shot relief but it's
  not asymptotically free.
- **Mitigation idea**: per-layer block tables — i.e., a request's
  block_table is a `dict[layer, list[Block]]` instead of a flat
  `list[Block]`. Then HEAD can stop allocating blocks for layers M..L-1.
  But this is a much bigger vLLM change. **Defer to Phase 6** (when we
  also port to v1, we can choose a friendlier KV layout).
- **Step 3.5 action (Phase 3 minimal)**: simply free the SRC's
  *outermost* set of blocks for R that contain only "garbage" upper-layer
  data. This is pretty much "don't free anything" in MVP — just leave
  the upper-half memory occupied. This step is mostly a thought experiment
  for the design write-up; no code changes.
- **Test**: none.
- **Verify**: documented design decision for the paper.

### Phase 3 exit criteria

- [ ] Steps 3.1–3.4 implemented; 3.5 documented.
- [ ] Microbenchmark numbers recorded (Step 3.1, 3.2, 3.3).
- [ ] Hybrid batching (Step 3.4) handles 16 concurrent requests with up
      to 4 simultaneously resharded.
- [ ] No correctness regressions vs. Phase 2.

---

## §4.5 Phase 4 — Auto-trigger PP-reshard policy (≈ 2–3 weeks, sketched)

Goal: the global scheduler decides when to PP-reshard, when to migrate
fully, and when to do neither. Manual API stays as a debugging tool.

### Step 4.1 — Reshard eligibility on instances (per-poll signal)

Add fields to `InstanceInfo` (the report each Llumlet sends to the global
scheduler each `polling_interval`, every 50ms today):
- `num_pp_reshard_eligible_running_requests: int` — running requests that
  have output_len > threshold and would benefit (defined below).
- `pp_reshard_capacity_blocks: int` — free-block budget that could absorb
  an incoming TAIL.

### Step 4.2 — Reshard decision in `MigrationScheduler`

In `llumnix/global_scheduler/migration_scheduler.py`, add a third
"reshard pair" mode alongside today's defrag pairs (the defrag policy is
in `migration_policy.py`). Decision logic, simplified:
- For each (src, dst) pair Llumnix already considers for migration, also
  ask: *would PP-reshard* be better here than full migration?
- "Better" if: src is at >threshold% memory, the candidate request's KV
  upper half fits on dst, and the request is in decode phase
  (incremental — not prefilling).
- If yes, emit a "pp_reshard" instruction; `Manager` calls
  `llumlet[src].pp_reshard_out.remote(...)` instead of `migrate_out`.

### Step 4.3 — Choosing the split layer M

For Phase 4 MVP: M = num_layers / 2 (config). Future work: choose M
adaptively based on which half of the model has more KV (it's roughly
balanced for transformer LMs anyway).

### Step 4.4 — Choosing the destination instance

Reuse Llumnix's existing freeness-based instance selection for migration.
The only added constraint: dst must have `pp_reshard_capacity_blocks ≥
needed_blocks_for_upper_half`. This is a filter on top of the existing
selection.

### Step 4.5 — Throttling

Cap the number of in-flight PP-reshards globally and per-instance. Without
this, a thundering-herd reshard could oscillate.

---

## §4.6 Phase 5 — Merge back to PP=1 (≈ 1–2 weeks, sketched)

Goal: when src has spare memory again (or dst is being drained), undo the
PP-reshard.

### Step 5.1 — `pp_reshard_merge` operation

Symmetric to `pp_reshard_out`. SRC pulls TAIL's KV for layers M..L-1 back,
re-installs into its own KV cache for R, sets `req.pp_reshard_role = None`,
DST removes R from its running queue and frees R's blocks.

The transfer direction reverses: TAIL is the "src" of the merge ship and
HEAD is the "dst". We can reuse the same `do_send`/`do_recv` (with
layer_range = (M, L)) — the symmetry makes this almost free.

### Step 5.2 — Trigger conditions

- Auto: dst is being terminated (auto-scale down) — drain all PP-reshards.
- Auto: src has freed enough memory.
- Manual: API call.

### Step 5.3 — Correctness tests symmetric to Phase 2

Same matrix: EOS during merge, preempt during merge, src OOM during merge.

---

## §4.7 Phase 6 — Port to v1 backend / vllm-latest (≈ 4–6 weeks, sketched)

Goal: PP-reshard works on top of `KVConnectorBase_V1` and the v1 engine.
The biggest pieces:

### Step 6.1 — Build `LlumnixPPReshardConnector(KVConnectorBase_V1)`

Subclass `KVConnectorBase_V1`. Implement:
- Scheduler-side: `get_num_new_matched_tokens` (return non-zero for
  TAIL-side requests), `update_state_after_alloc`, `build_connector_meta`
  (per-step instructions: "ship layers M..L-1 of these blocks"),
  `request_finished`.
- Worker-side: `register_kv_caches`, `start_load_kv` (for TAIL receive),
  `wait_for_layer_load` (per-layer pipelining), `save_kv_layer`
  (per-layer ship from HEAD), `wait_for_save`, `get_finished`.

This is essentially porting Phase 1+3 to the v1 connector interface. Most
of the heavy lifting is already done by the connector machinery; we're
just providing a layer-range-aware metadata.

### Step 6.2 — Register in factory

Add to `vllm/distributed/kv_transfer/kv_connector/factory.py:149` (the
register block). Or use the external module path mechanism — the factory
already supports `kv_connector_module_path` in
`KVTransferConfig` (config/kv_transfer.py:63), which we lifted from our
own fork to avoid touching upstream code.

### Step 6.3 — Llumnix v1 backend support

Build out `llumnix/backends/vllm_v1/` properly. Today it's just
`async_core.py`. Add:
- A v1 SchedulerLlumnix subclass (subclass of v1's
  `vllm/v1/core/sched/scheduler.py:Scheduler`).
- A v1 LlumnixWorker (likely subclass of `vllm/v1/worker/gpu_worker.py:Worker`).
- A v1 BackendInterface impl.
- Wire `KVConnectorBase_V1` agent (set globally via
  `ensure_kv_transfer_initialized` from
  `vllm/distributed/kv_transfer/kv_transfer_state.py:51`).

### Step 6.4 — Adapt scheduling-side decisions

The v1 scheduler uses `SchedulerOutput` (output.py:178) with
`scheduled_new_reqs` and `scheduled_cached_reqs` and a
`kv_connector_metadata` field. Adapt our policies to populate this.

### Step 6.5 — Migrate Llumnix's existing whole-request migration to v1 too

A bonus: while we're in there, port Llumnix's existing migration to v1
(also via `KVConnectorBase_V1`). The Llumnix project benefits, and we
unify our two paths.

### Step 6.6 — Match upstream test cadence

The vllm-latest AGENTS.md (read in this session) prescribes uv-based env,
pre-commit, and pytest invocation patterns. Our Phase 6 work should
satisfy these so that *if* we ever propose anything upstream (we likely
won't, given the research-experiment nature), it's clean. Concretely:
- `uv venv --python 3.12; source .venv/bin/activate`
- `VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto`
- `.venv/bin/python -m pytest tests/v1/kv_connector/unit/`

---

## §4.8 Phase 7 — Evaluation (≈ 3–4 weeks, sketched)

### Step 7.1 — Reproduce Llumnix's Fig. 11 baseline

Run Llumnix as-is on our testbed; capture P50/P99 latency on the
ShareGPT, BurstGPT, and generated power-law traces. (The artifact's
`62_…/65_…` directories show the harness; we adapt for our fork.)

### Step 7.2 — PP-reshard benefit experiments

Compare four configurations on the same trace:
1. Plain vLLM (no Llumnix)
2. Llumnix with full migration (current state)
3. Llumnix + PP-reshard (Phase 3 / Phase 6)
4. vLLM-latest disagg PD (NIXL connector) — orthogonal baseline

Metrics: tail latency (P50/P99 first-token, per-token), preemption rate,
throughput, fragmentation%, OOM count.

### Step 7.3 — Microbenchmarks

- Reshard time vs. seq length (Phase 3 Step 3.1's data).
- Hidden-state hop latency (Phase 3 Step 3.2's data).
- Pipelined vs. blocking reshard (Phase 3 Step 3.3's data).

### Step 7.4 — Targeted demo (D7)

A single LLaMA-7B request that would OOM on a stand-alone instance,
served without preemption via PP-reshard. Plotted as a "what would
otherwise be impossible" data point.

### Step 7.5 — Failure-mode demo

Show graceful degradation: kill an instance during a hybrid step;
HEAD/TAIL recovers either by completing on remaining instance (if KV
still cached) or by re-scheduling.

### Step 7.6 — Paper artifact

Mirror Llumnix's `artifact/` structure: numbered subdirectories per
figure, `run.sh` and `plot/plot.py` per. Make our results reproducible.

---

## §4.9 Cross-cutting concerns

### §4.9.1 Data structures we'll add (cheat sheet)

| New type | File | Purpose |
|---|---|---|
| `PPReshardRole` (enum) | `llumnix/llumlet/request.py` | HEAD or TAIL |
| Fields on `LlumnixRequest` | same | role, split_layer, partner_id, partner_actor |
| `enable_pp_reshard`, `pp_reshard_split_layer` (config) | `llumnix/internal_config.py` | feature flags |
| `pp_reshard_out` (method) | `llumnix/llumlet/llumlet.py` | RPC entry |
| `pp_reshard_out`, `apply_pp_reshard_tail` | `llumnix/llumlet/migration_coordinator.py` | core protocol |
| `commit_dst_pp_reshard_tail` | `llumnix/backends/vllm/llm_engine.py` (BackendVLLM) | dst commit |
| `do_send(layer_range=...)`, `do_recv(layer_range=...)` | `llumnix/backends/vllm/migration_backend.py` | layer-bounded transfer |
| `LlumnixPPReshardConnector` (class) | `llumnix/backends/vllm_v1/kv_connector_pp_reshard.py` (Phase 6) | v1 connector |

### §4.9.2 Tests we'll add (cheat sheet)

| Test | Type | Phase | What it asserts |
|---|---|---|---|
| `test_pp_reshard_role_roundtrip` | U1 | 1.1 | LlumnixRequest pickles with new fields |
| `test_pp_reshard_config_loads` | U1 | 1.2 | YAML config sets the flag |
| `test_layer_range_kv_transfer` | U2 | 1.3 | Only specified layers' KV moves |
| `test_pp_reshard_propagates_kwargs` | U1 | 1.4 | layer_range reaches the worker |
| `test_pp_reshard_e2e_toy` | I1 | 1.10 | Token output equals reference |
| `test_pp_reshard_eos_during` | U1+I1 | 2.1 | EOS mid-reshard cleans up |
| `test_pp_reshard_preempt_during` | I1 | 2.2 | Preempt mid-reshard aborts |
| `test_pp_reshard_dst_oom` | U2 | 2.3 | OOM aborts cleanly |
| `test_pp_reshard_src_dies` | I1 | 2.4 | Watchdog frees DST |
| `test_pp_reshard_dst_dies` | I1 | 2.5 | SRC recovers as full request |
| `test_pp_reshard_abort_api` | U1 | 2.6 | Controller can abort |
| `bench_reshard_time_vs_seqlen` | B | 3.1 | Reshard time grows linearly with bytes |
| `bench_hidden_state_hop_latency` | B | 3.2 | NCCL ≪ Ray RPC |
| `bench_pipelined_decode_step` | B | 3.3 | PP-reshard step ≤ 1.2× full-PP=1 |
| `test_hybrid_batch_4reqs_1reshard` | I1 | 3.4 | Mixed batches work |

### §4.9.3 Knobs and how to tune them

- `pp_reshard_split_layer` — start at L/2; sweep in evaluation.
- `migration_max_stages` — already 3 in Llumnix today; same for PP-reshard.
- `migration_buffer_blocks` — already config; affects Phase 3 Step 3.1.
- A new throttle: `max_concurrent_pp_reshards_per_instance` (Step 4.5).
- A new threshold: `pp_reshard_min_output_len` — only consider requests
  past N decode tokens, since reshard overhead is wasted on a request
  that's about to finish.

### §4.9.4 Risks I want to revisit before code

In rough priority order:

1. **R1 (model.forward layer mask)** — needs verifying by reading
   `vllm/vllm/model_executor/models/llama.py` end to end. If invasive,
   subclass route. *Block on this before Step 1.8.*
2. **R3 (iteration-level barrier)** — synchronizing two independent
   asyncio loops via Ray RPC may have surprising latency floors. Profile
   in a tiny test before committing to Phase 1 design.
3. **The "memory not actually freed on SRC" insight (Step 3.5)** — the
   value of PP-reshard for the OOM-avoidance demo (D7) hinges on
   freeing being possible. Re-examine whether to redesign block tables
   to be per-layer in Phase 6, OR pivot the demo to "preemption avoidance"
   which is genuinely improved even without per-layer freeing.
4. **R5 (block accounting)** — confirmed mostly fine, but the `n_blocks`
   computation in `prefill_num_blocks` (used in `pre_alloc_cache`) needs
   to match what we're actually shipping. For TAIL, we ship layers
   M..L-1 worth — same number of blocks, but only "valid" data in those
   blocks for the upper half.

### §4.9.5 What we are NOT doing

To bound the project:
- Not supporting PP=k for k>2 (i.e., reshard onto >2 instances). All
  designs above assume one HEAD + one TAIL.
- Not supporting reshard *during prefill* (only during decode). Prefill
  is where the model is fastest in vLLM; resharding mid-prefill is messy
  and unlikely to help.
- Not supporting cross-TP-config reshards (TP1 ↔ TP2). NIXL/Mooncake
  support heterogeneous TP; we'll need it eventually but not for MVP.
- Not changing the core block-allocation semantics in vLLM (per-layer
  block tables — see §4.4 Step 3.5 caveat). We accept that the SRC will
  hold "ghost" upper-layer KV slots after reshard.

---

# §5. Investigation plan: Llumnix goodput collapse on high-end multi-node TP clusters

> Triggered by: department researchers report Llumnix's stable goodput is ~10
> req/s on 12B and ~5 req/s on 70B when deployed on multi-node clusters with
> 4×–8× H100 GPUs running TP-only (no PP), even with input rate 16–100 req/s.
> The cluster is expensive; this section plans how to find the root cause
> with the fewest cluster-hours.

## §5.1 What this symptom tells us before we do anything

**Sanity check on the numbers.** A single H100 80GB at FP16 should deliver
roughly:
- 12B model, 256/256 in/out: ≈ 30–80 req/s with vanilla vLLM in continuous
  batching (depending on length distribution; the headline numbers in vLLM
  benchmarks are higher for short sequences).
- 70B model, TP=4, 256/256 in/out: ≈ 8–20 req/s.
- Multi-node TP doesn't usually help latency; it helps fitting the model
  but adds inter-node NCCL all-reduce per layer.

So 10 req/s on 12B is **at least 3× too low**, and 5 req/s on 70B is
**also low for TP=4 H100, marginal for TP=8** if the workload is heavy.
This is a *gap* problem, not a *limit* problem. There is something
serializing requests or bleeding cycles to overhead. We're looking for a
scapegoat that costs a lot of throughput.

**"Stable" matters.** "Output stable at 5–10 req/s while input is 16–100"
means the system is in saturation: the running/waiting queues are full and
the bottleneck is the *steady-state per-second drain rate of the system*.
That drain rate could be limited by:
1. **Compute** (the GPUs are saturated doing useful work — unlikely given
   the gap).
2. **Synchronization** (GPUs idle waiting for the slowest peer or for a
   lock).
3. **Communication** (NCCL/Gloo/Ray RPCs serialize the hot path).
4. **Scheduling/queuing** (admission, batching, or output-drain bottleneck).
5. **Excess work** (every "useful" iteration is paired with a migration or
   re-prefill that wastes its progress).

We map each hypothesis below to which of these it would manifest as, and
to a cheap experiment that distinguishes it.

## §5.2 Information we need from the researchers BEFORE any new experiment

The plan branches dramatically on the answers to these. None requires
cluster time; they're emails / chats with the people seeing this.

**Workload characterization**:
- Q5.2.1: Average prompt length, average output length, sampling params
  (temperature, max_tokens, ignore_eos), priority distribution if any.
- Q5.2.2: Distribution shape — is it bursty (Gamma high CV) or smooth
  (Poisson)? Llumnix's behavior varies a lot by burstiness.
- Q5.2.3: Are there long requests that would otherwise preempt others?
  (If yes → migration churn hypothesis grows.)

**Llumnix configuration in the bad runs**:
- Q5.2.4: `enable_migration` (`MigrationConfig.enable_migration`,
  `internal_config.py:65`)? Default is True.
- Q5.2.5: `migration_backend` — `rayrpc` / `gloo` / `nccl`? Each has very
  different perf characteristics on H100.
- Q5.2.6: `enable_pd_disagg`, `enable_engine_pd_disagg`, `enable_adaptive_pd`?
  Their PD code path forces a migration after prefill — the source of
  several known-perf-cliff issues.
- Q5.2.7: `dispatch_policy` (default `'load'`), `dispatch_load_metric`
  (default `'remaining_steps'`).
- Q5.2.8: `pair_migration_policy` (default `'defrag'`),
  `migrate_out_threshold` (default `-3.0`),
  `pair_migration_frequency` (default `1`),
  `migration_max_stages` (default `3`).
- Q5.2.9: Number of instances (Llumlets), TP size per instance, total GPUs.
- Q5.2.10: Auto-scaling enabled? `scale_up_threshold`, `scale_down_threshold`.

**Comparison data already collected**:
- Q5.2.11: Has anyone run **vanilla vLLM** (no Llumnix) on the same
  hardware with the same workload? What was the goodput? *This is the
  single most valuable piece of data*; if it doesn't exist yet, that's
  the very first cluster-hour we should spend.
- Q5.2.12: Have other Llumnix users reported similar collapse? Check the
  GitHub issue tracker (`AlibabaPAI/llumnix`) for terms: "throughput",
  "goodput", "TP", "NCCL", "migration storm", "saturation". (Free; do
  before any cluster work.)
- Q5.2.13: Logs from a bad run? Especially the per-instance metrics dumps
  and any migration logs.

**Topology**:
- Q5.2.14: Inter-node fabric: NVLink-connected within node, NDR/HDR
  InfiniBand or 200G/400G Ethernet between nodes? Llumnix's KV migration
  flows over the inter-node fabric and Ray RPC overhead amplifies on
  slower fabrics.
- Q5.2.15: Are Llumnix's Ray actors actually placed on different nodes,
  or has Ray clumped them onto one node? `ray status` would tell us.

## §5.3 Hypotheses, ranked, with how each manifests

I'll number these H-1..H-10 so we can cite them in the experiment plan.
Each has: 1-line claim → manifest signature → cheapest distinguishing
experiment.

### H-1. Migration storm — **TOP SUSPECT**

**Claim**: Llumnix's defrag policy keeps triggering migrations, each one
ships KV blocks across the inter-node fabric, and the steady-state cost
of migration eats most of every step's wallclock.

**Manifest**:
- Migration rate ≫ 0 (visible in logs: "begin migrate out" / "migrate done"
  log lines from `migration_coordinator.py:183` and `:213`).
- Per-instance running queue length oscillates / never grows large.
- `num_killed_requests` (the preempted count proxy in
  `instance_info.py:57`) oscillates.
- Goodput recovers immediately when `enable_migration=False`.
- Worse on 70B than 12B (more KV per token → bigger migration cost).

**Cheap distinguisher**: §5.5.2 — toggle `--enable_migration=False`. If
goodput jumps to expected range, H-1 is confirmed.

**Where in code**:
- `Manager._push_migrations` calls `GlobalScheduler.pair_migration` every
  `PAIR_MIGRATION_FREQUENCY=1` polls, i.e. every 50ms. The `defrag` policy
  pairs source ↔ destination by freeness gap and asks each source to
  migrate one or more requests.
- `MIGRATE_OUT_THRESHOLD = -3.0` (`config/default.py`): with the default
  `RemainingStepsLoad` metric, this fires more aggressively than the
  paper's experiments on A10s.

### H-2. NCCL stall / serialization in the migration path

**Claim**: Llumnix uses `RayColMigrationBackend` with NCCL or Gloo. Its
own paper §5 acknowledges "concurrent invocations of NCCL are known to
be unsafe", which is why they use Gloo. With Gloo on H100 (very high
nominal bandwidth that Gloo's CPU staging cannot achieve), KV migration
becomes the bottleneck.

**Manifest**:
- Each migration takes much longer per stage than expected (e.g. ≫ 100ms
  for an 8k seq even though H100 NVLink is sub-ms).
- `migration_stream` synchronizes (`migration_backend.py:391`) block the
  worker thread.
- Migrations ↑ when the inference stream is busy → effectively
  alternating compute and communication.

**Cheap distinguisher**: in §5.5.4 toggle the migration backend between
`gloo`, `nccl`, and `rayrpc` and measure per-stage time.

### H-3. Manager-actor bottleneck

**Claim**: The Manager Ray actor handles dispatch, polling, migration
triggering, output forwarding. With 16–100 req/s incoming and ~50ms
polling × N instances, the Manager's coroutine loop or its Ray RPC
inbox saturates.

**Manifest**:
- Manager actor CPU pinned at 100%.
- Ray RPC latency from Manager → Llumlet's RTT high (>10ms).
- Goodput insensitive to GPU count (because the bottleneck is CPU, not GPU).
- New requests sit in dispatch waiting >>> step time.

**Cheap distinguisher**: §5.5.6 — single instance Llumnix vs vanilla vLLM.
If Llumnix-1-instance is also slow, it's not migration; it's per-step
overhead in the engine path.

### H-4. Continuous batching broken / running batch always tiny

**Claim**: Some interaction between Llumnix's `SchedulerLlumnix`
(scheduler.py:59) and vLLM's `Scheduler.schedule()` keeps the running
batch artificially small. The override `_schedule_running` (line 257)
filters running by `output_len >= expected_steps`, which in PD-disagg
mode could pull most requests *out* of the running queue every step.

**Manifest**:
- Per-step running batch size very small (e.g. 1–2 requests) regardless
  of how many requests are queued.
- GPU utilization nominal, but tokens-per-step very low.
- Particularly bad if PD-disagg or adaptive PD is on.

**Cheap distinguisher**: §5.5.5 — disable PD-disagg
(`--enable_pd_disagg=False`) and instrument running-batch-size per step.

**Where in code**: `SchedulerLlumnix._schedule_running` (scheduler.py:257)
explicitly filters out seq_groups whose `output_len >= expected_steps`
*before* calling super()._schedule_running. For a "prefill-only"
disagg request `expected_steps == 1`, so after one step it gets pulled.
This is *intentional* but if not paired with proper PD migration could
strand requests.

### H-5. Output queue / detokenization backpressure

**Claim**: Llumnix routes outputs through a `ZmqServer` or `RayQueueServer`
(`queue/`). If the consumer side (e.g., the API server actor) is slow to
drain, outputs back up and the engine stalls.

**Manifest**:
- Tokens generated but not delivered to the user.
- ZMQ socket queue depth high.
- API server actor CPU high.

**Cheap distinguisher**: §5.5.7 — set request output mode to "no-op"
(consume but don't return). If goodput recovers, output drain is the bug.

### H-6. PD disaggregation misconfigured

**Claim**: If `enable_pd_disagg=True` and the prefill-to-decode ratio is
wrong (`pd_ratio` in `PDDConfig`), one role starves and waiting requests
pile up.

**Manifest**:
- One subset of instances 100% busy, the other idle.
- Forced migration after prefill creates a request hop that doubles
  every request's per-token cost.
- Goodput sensitive to `pd_ratio`.

**Cheap distinguisher**: try with PD disagg off; toggle pd_ratio.

### H-7. Memory-pressure preemption oscillation

**Claim**: vLLM's continuous batching admits requests until memory is
near full, then preempts. Llumnix piles migration on top: a preempted
request might also be a migration candidate. The two systems interact
and produce thrash.

**Manifest**:
- High preemption count (`num_killed_requests > 0` for many instances
  most of the time).
- Goodput recovers when `gpu_memory_utilization` is *lowered*.

**Cheap distinguisher**: lower `gpu_memory_utilization` to 0.7 from 0.9.

### H-8. Auto-scaling triggering instance churn

**Claim**: If auto-scaling is enabled, instances are added/removed
mid-experiment. The terminating instance drains via migration → migration
storm during scaling events.

**Manifest**:
- Goodput recovers when auto-scaling thresholds are pushed off-range.
- Number of instances changes during the bad run.

**Cheap distinguisher**: disable auto-scaling.

### H-9. Ray placement: actors on a single node

**Claim**: Ray's default placement put all Llumlet actors on the head
node, ignoring the multi-node intent. NCCL all-reduces still happen
within the node so it "works" but goodput is single-node.

**Manifest**:
- `ray status` shows N actors on node 0, 0 on others.
- Per-node nvidia-smi shows only node 0 GPUs busy.

**Cheap distinguisher**: free; just ssh and check.

### H-10 (long-shot). vLLM version drift

**Claim**: The fork pins `vLLM 0.6.3.post1`. If the cluster install drifted
to a different vLLM version, behavior could change drastically.

**Manifest**: `pip show vllm` reports a different version than expected.

**Cheap distinguisher**: free; check installs.

## §5.4 What to do BEFORE booking the cluster (free)

These are all laptop / GitHub / chat work. Aim: rule out 30% of the
hypothesis surface for $0.

### §5.4.1 Read the metrics that Llumnix already records

- `llumnix/metrics/timestamps.py` defines `set_timestamp(server_info, key, t)`,
  used in many places (e.g. `Llumlet.generate` at `llumlet.py:151` sets
  `'llumlet_generate_timestamp'`). What other timestamps exist? Does the
  trace already capture per-step latency, per-migration time, queue
  lengths?
- `llumnix/metrics/exporters.py` — does Llumnix export Prometheus
  metrics? If yes, the cluster may already have a dashboard we can stare
  at without running anything new.
- `llumnix/metrics/dumper.py` / `llumnix/metrics/manager_metrics.py` —
  what does the Manager log? Could already have migration count and
  per-instance load histories.

**Action**: read all four files, write a one-page "what Llumnix logs by
default" appendix here in §5. Saves us from instrumenting things that
are already instrumented.

### §5.4.2 Search the Llumnix issue tracker

- `gh issue list -R AlibabaPAI/llumnix --search "throughput TP"`
- `gh issue list -R AlibabaPAI/llumnix --search "goodput"`
- `gh issue list -R AlibabaPAI/llumnix --search "migration NCCL"`
- `gh issue list -R AlibabaPAI/llumnix --search "H100"`
- Same in PR list with state both open and closed.

If anyone has reported the same symptom and the maintainers know the
cause, that's our answer for free. If they've reported it and there's no
fix, that constrains our hypothesis space.

### §5.4.3 Read the recent commit history

- `git -C /home/ubuntu/reshardLLM/reshardLLM log --since="6 months ago" --oneline`
- Look for commits touching `manager.py`, `migration_coordinator.py`,
  `migration_backend.py`, `scheduler.py`, `global_scheduler/*.py`.
- A regression introduced recently is a leading hypothesis if the
  goodput numbers got worse after a specific date.

### §5.4.4 Read the hot path end-to-end

I have already read several files in §3 / §4. To pin H-1/H-2/H-3 down,
also read end-to-end:
- `llumnix/manager.py` — the polling loop, dispatch path, migration
  trigger path. Look for any `await` that could block under load.
- `llumnix/global_scheduler/global_scheduler.py` — the pairing
  computation. How expensive is it per call? O(instances²)?
- `llumnix/load_computation.py` — the `RemainingStepsLoad` formula. Is
  the metric noisy under load (e.g. dividing by very small numbers)?

### §5.4.5 Estimate per-poll Manager cost on paper

With:
- N = 16 instances (multi-node 4 nodes × 4 H100s, each TP=1; or 2 nodes ×
  TP=8, etc.)
- Polling interval = 50ms = 20 polls/s
- Per poll: Ray RPC to each Llumlet → 16 round-trips → each ~1ms in
  Ray local cluster, ~5ms in cross-node Ray over IB.

Per second cost: 20 × 16 × 5ms = 1.6 *seconds* per second per Manager
actor. **The Manager would be a bottleneck without any other load.**

This is the kind of paper estimate we should do *before* we book GPUs.
If this number comes out > 1.0 for any candidate config, H-3 is highly
plausible and we should plan a Manager-bypass test (5.5.6).

## §5.5 Single-node experiments (cheap — first to run on cluster)

Each of these is one box, < 1 hour.

### §5.5.1 Vanilla vLLM baseline (CRITICAL — first cluster experiment)

**Goal**: a number that says "this hardware can do X req/s without
Llumnix on this workload".

**Setup**: one node, 4× H100, vLLM directly with `tensor_parallel_size=4`,
no Llumnix in the picture. Same workload generator the researchers used.

**Runtime**: 30 min including warmup.

**Outcome**:
- If vanilla vLLM also gets 5–10 req/s → the issue is **not Llumnix**;
  it's hardware, vLLM, or workload. Re-examine the workload definition
  before going further.
- If vanilla vLLM gets ≥ 30 req/s → the gap is on Llumnix. Continue.

### §5.5.2 Llumnix with `--enable_migration=False`

**Goal**: rule out / confirm H-1 (migration storm) and the bulk of H-2.

**Setup**: same node and workload as 5.5.1, but Llumnix instead of vLLM.
1 instance, TP=4, migration disabled.

**Runtime**: 30 min.

**Outcome**:
- Goodput jumps to vanilla-vLLM levels → H-1 confirmed; the cure is in
  the migration policy / triggers.
- Goodput similar to bad-run levels → H-1 ruled out; move to H-3/H-4.

### §5.5.3 Llumnix with auto-scaling disabled

**Goal**: rule out H-8.

**Setup**: same as 5.5.2 but `enable_migration=True` and N=2 instances
with auto-scaling thresholds pushed off-range (e.g.,
`scale_up_threshold=-1000`, `scale_down_threshold=-10000`).

**Runtime**: 20 min.

**Outcome**: if goodput differs from "auto-scaling on" runs of similar
config, H-8 has weight.

### §5.5.4 Migration backend sweep

**Goal**: distinguish H-2 candidates.

**Setup**: same workload, Llumnix with migration enabled, 2 instances on
1 node (TP=2 each), migrate_out_threshold tweaked so we get N migrations
per second deliberately. Toggle `migration_backend` ∈ {`rayrpc`, `gloo`,
`nccl`}. Measure per-stage time and goodput.

**Runtime**: ~1.5 hours total (3 backends × 30 min).

**Outcome**: which backend is the bottleneck. Tells us whether the issue
is the *fact of migration* or *the chosen migration backend*.

### §5.5.5 PD disaggregation off

**Goal**: rule out H-6 / H-4.

**Setup**: same as 5.5.2 but now with `enable_pd_disagg=False` (probably
already off; check cluster's actual config).

**Outcome**: if researchers had it on and turning it off recovers
goodput, H-6 confirmed.

### §5.5.6 Single-instance Llumnix vs vanilla vLLM

**Goal**: isolate Llumnix's per-step overhead vs vanilla vLLM. Rules out
H-3 if Llumnix-1-instance is fine.

**Setup**: 1 Llumlet (so no migration is even possible), TP=4, vs vanilla
vLLM with TP=4. Identical workload.

**Outcome**:
- Llumnix-1-instance ≈ vanilla → Llumnix's per-engine overhead is fine;
  the gap is in inter-instance coordination (migration, Manager).
- Llumnix-1-instance ≪ vanilla → there's per-step overhead in the
  Llumnix engine wrapper itself (see §5.5.10).

### §5.5.7 Output queue stress

**Goal**: rule out H-5.

**Setup**: replace API server actor with a no-op consumer that just
counts tokens.

**Outcome**: if goodput recovers, H-5 confirmed.

### §5.5.8 Memory-utilization sweep

**Goal**: rule out H-7.

**Setup**: same workload at gpu_memory_utilization ∈ {0.7, 0.8, 0.9, 0.95}.
Record preemption count.

**Outcome**: if 0.7 has dramatically better goodput, H-7 contributes.

### §5.5.9 Migration trigger sensitivity sweep

**Goal**: refine H-1 — find the migrate_out_threshold that maximizes goodput.

**Setup**: keep migration enabled but vary `migrate_out_threshold` ∈
{-3, -10, -50, -100}. -100 effectively disables triggering without
disabling the migration code path.

**Outcome**: a curve. The threshold at which migration *adds* throughput
(the paper's claim) versus where it subtracts (our hypothesis) is
useful to publish back upstream.

### §5.5.10 Per-step latency profile (instrumented)

**Goal**: identify *where* the time goes per step in Llumnix.

**Setup**: instrument the engine step (see §5.7) to record timestamps at:
- Scheduler enter / exit
- Forward enter / exit
- Sample enter / exit
- Output processing enter / exit
- Manager poll receive
- Migration triggered → completed

**Runtime**: short, but adds setup time.

**Outcome**: a stacked-bar plot of step time. The dominant bar is the
bottleneck.

## §5.6 Multi-node experiments (more expensive)

If §5.5 hasn't pinned the bug, escalate.

### §5.6.1 Reproduce the bad-goodput run with full instrumentation

**Goal**: capture every metric in a known-bad configuration so we have a
single dataset everyone can stare at.

**Setup**: exact configuration the researchers reported. With our §5.7
instrumentation enabled. ~1 hour.

**Outcome**: a saved trace file we don't need to re-collect.

### §5.6.2 Ray placement check (free during 5.6.1)

**Goal**: rule out H-9.

**Setup**: during 5.6.1, run `ray status` and `nvidia-smi` on each node.

**Outcome**: are all GPUs being used, or only one node's?

### §5.6.3 Inter-node fabric microbenchmark

**Goal**: bound the migration cost from below.

**Setup**: a script that uses the same NCCL/Gloo group Llumnix uses to
ship a 1 GB tensor between two instances. Time it.

**Outcome**: if migration's per-stage time in 5.5.4 ≫ this, the migration
code has overhead beyond the wire.

### §5.6.4 Targeted intervention test

**Goal**: validate the fix.

If 5.5/5.6 indicate a specific cause, try the fix at multi-node scale
and confirm goodput recovers.

## §5.7 Instrumentation to add upfront (saves repeat experiments)

Add these *before* the cluster runs so we don't repeat them. All are
small Python additions, no kernel changes.

1. **Per-step latency log**. In `BackendVLLM._start_engine_step_loop`
   (`llm_engine.py`), wrap each step with a timestamp dict, log
   line-per-step in JSON. Fields: `instance_id`, `step_id`, `t_schedule`,
   `t_execute`, `t_process`, `running_batch_size`, `waiting_batch_size`,
   `num_killed`, `num_migrating`.

2. **Migration count + per-stage time**. In `MigrationCoordinator._migrate_out_one_request`
   (already logs at line 213), add the `stage_num_blocks_list` and
   `stage_timestamps` to a structured log line.

3. **Manager polling RTT histogram**. In `Manager._poll_instance_info_loop`,
   record per-RPC latency to a histogram.

4. **Output queue depth**. In `ZmqServer` / `RayQueueServer`, expose
   queue size as a metric exported every second.

5. **Per-Ray-RPC latency** (between Manager → Llumlet, between Llumlets
   for migration). Wrap `execute_migration_method.remote(...)` with a
   timer.

6. **GPU utilization** via `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`
   sampled once per second on every node.

All these dump to a per-run directory `traces/{run_id}/` so we can post-
process offline and re-derive plots without re-running.

## §5.8 Decision tree

```
Run 5.4 (free) →
  if Llumnix issues exist on tracker matching our symptom → use their answer.
  else proceed.

Run 5.5.1 (vanilla vLLM baseline) →
  if vanilla ≈ bad-Llumnix → it's vLLM/hardware/workload, not Llumnix.
                              Stop blaming Llumnix; debug vLLM.
  else (vanilla ≫ bad-Llumnix) → continue.

Run 5.5.2 (migration disabled) →
  if goodput recovers → H-1 confirmed.
                        Run 5.5.4, 5.5.9 to characterize migration cost.
                        Fix: tune thresholds + use better migration backend.
  else → migration is not the (sole) bottleneck. Continue.

Run 5.5.6 (single-instance Llumnix vs vanilla) →
  if Llumnix-1-instance ≪ vanilla → engine-wrapper overhead. Profile in 5.5.10.
  else → coordination overhead. Continue 5.5.3, 5.5.5, 5.6.

Run 5.5.10 (per-step profile) →
  read the dominant bar. Match to one of:
    schedule big   → H-4 (broken batching) or scheduler bug.
    execute big    → vLLM kernel issue. Cross-check with vanilla vLLM.
    output_proc big → H-5 (output queue).
    idle big       → H-3 (waiting for Manager) or H-2 (waiting for NCCL).
```

## §5.9 Cluster-hours budget

| Phase | Activity | Hours | Stop condition |
|---|---|---|---|
| 0 | §5.4 free work — code reads, tracker, paper estimates | 0 | none |
| 1 | §5.5.1 vanilla vLLM baseline | 1 | result tells direction |
| 2 | §5.5.2, 5.5.6 (migration off, single-instance) | 1.5 | usually pinned by here |
| 3 | §5.5.3, 5.5.5, 5.5.7, 5.5.8 (toggles) | 1.5 | minor sources cleared |
| 4 | §5.5.4, 5.5.9 (migration backend + threshold sweeps) | 2 | only if H-1/H-2 still alive |
| 5 | §5.5.10 (instrumented profile) | 1 | only if not pinned |
| 6 | §5.6.1, 5.6.2 (multi-node reproduce + check) | 1.5 | only if bug requires multi-node |
| 7 | §5.6.3 (NCCL/Gloo wire bench) | 0.5 | only if H-2 alive |
| 8 | §5.6.4 (intervention validation) | 1 | confirms fix |

**Best case (H-1 confirmed early)**: ~3 cluster-hours.
**Median case**: ~6 cluster-hours.
**Worst case (no easy answer)**: ~10–12 cluster-hours.

## §5.10 Pre-flight checklist (do all before phase 1)

- [ ] All Q5.2 questions answered, written into a "known config" page.
- [ ] §5.4.1 done — list of metrics already recorded by Llumnix.
- [ ] §5.4.2 done — searched Llumnix issue tracker; saved relevant issue
      threads.
- [ ] §5.4.3 done — recent commit log scanned for likely regressions.
- [ ] §5.4.4 done — read manager.py + load_computation.py end-to-end.
- [ ] §5.4.5 done — paper estimate of Manager polling cost for our N.
- [ ] §5.7 instrumentation patched into a branch and unit-tested locally
      with a tiny model (no GPU needed for the instrumentation
      *scaffolding*; just for the GPU-dependent metrics).
- [ ] Dashboard / log-shipping endpoint set up so traces from cluster
      runs land where we can read them without ssh-ing in.
- [ ] Ahead-of-time agreement on which engineer babysits the runs and
      who has the cluster reservation.

## §5.11 Likely punchline

If I had to bet before any data, I'd put weight roughly:

| Hypothesis | Prior |
|---|---|
| H-1 migration storm | 35% |
| H-3 manager-actor bottleneck | 20% |
| H-2 NCCL/Gloo path | 15% |
| H-4 broken batching (esp. PD-disagg interaction) | 10% |
| H-9 ray placement | 10% |
| H-7 preemption oscillation | 5% |
| H-5/H-6/H-8/H-10 | 5% combined |

Reasoning: the gap is *very* large (3–10×); the H100 fabric makes
inter-instance ops cheap if the software lets them be cheap, so the most
likely culprit is something software-side that *doesn't* get cheaper on
H100. Migration triggering and Manager polling don't get cheaper on
better GPUs — they're CPU/RPC-bound. The 70B-is-worse-than-12B signal
fits H-1 (more KV per token to migrate) and H-2 (more bytes on the
fabric). It also fits H-3 mildly (more time per step, so polling overhead
is a smaller proportion — which would predict 70B is *less* affected by
H-3, the opposite of what we see; that's why H-3 is below H-1).

## §5.12 Things to ask the researchers in the dept

These are non-obvious from your message and would change the plan:

- **A.** Has anyone tried Llumnix's own benchmarks
  (`scripts/simple_bench.sh`, `scripts/start_4_tp2_instances.sh`) on
  this cluster? Their numbers would be the canonical "expected" goodput.
- **B.** What workload trace are they using? ShareGPT? BurstGPT?
  Generated power-law? In-house? The plan above assumes a synthetic
  Poisson trace; if it's heavily long-tailed the analysis branches.
- **C.** Is this Llumnix as-is from a recent commit, or was this fork
  modified internally before the bad runs? (Per A1 in our session 1
  finding, the fork has minor benchmark/scripts changes; if your dept
  has *additional* changes beyond what I saw, they could be the cause.)
- **D.** Are they comparing against a target or just observing low
  goodput? "5 req/s" might be acceptable for *some* workloads (long
  in/out, large model). The "low" judgment depends on a target.
- **E.** Did the cluster's Ray and CUDA versions change recently?
- **F.** Is `VLLM_USE_V1=1` set anywhere, accidentally engaging a
  different vLLM code path the Llumnix fork doesn't fully support?
  (The fork's `backends/vllm_v1/` is mostly stub per A20.)

I'd batch-email these before any cluster run. If even half come back
with "we don't know", that's data — it tells us the bad-run config isn't
fully reproducible, and 5.6.1 becomes the priority experiment.

---

# §6. ShareGPT-misuse audit + cross-machine experiment brief

> **What this section is**: a deep audit of two latent flaws in Llumnix's
> ShareGPT benchmarking code, plus a copy-pasteable brief for another
> Claude Code session running on a different machine to design and
> execute an experiment that *quantifies* the impact of fixing these flaws.
>
> Audit target: `/home/ubuntu/reshardLLM/llumnix-eval-vllm/vendor/llumnix-ray/`.
> Cross-agent paths are repo-relative (no `/home/ubuntu/...`).

## §6.0 Summary

Two concrete problems exist in this Llumnix tree's benchmark harness:

**P1. ShareGPT first-turn-only sampling**: `sample_sharegpt_requests` reads
only `conversations[0]` (first user prompt) and `conversations[1]` (first
assistant reply). Every turn from index 2 onward is silently discarded.
ShareGPT-GPT4 conversations are typically multi-turn (2–20 turns); using
only the first turn produces a workload whose KV-cache pressure profile
looks nothing like real chatbot deployment.

**P2. Hidden length filter**: `--max_request_len` (default `16384`) is
used as a *prerequisite filter* — any conversation whose
`prompt_tokens + completion_tokens ≥ max_seqlen` is silently dropped from
the candidate pool *before* the experiment starts (line 756 in
`benchmark_serving.py`). Then any sampled conversation that exceeds the
cap *during* the run gets its response truncated (lines 922–925).
Result: the long-tail conversations that would stress memory hardest
never appear in published metrics.

Both flaws bias results toward configurations that handle short, simple
workloads — exactly the configurations Llumnix promotes. Fixing them is
a precondition for measuring the real preemption rate of vLLM (and
Llumnix) under realistic chatbot load.

## §6.1 Detailed problem analysis

### §6.1.1 P1: first-turn-only ShareGPT sampling

**Code location** (`benchmark/benchmark_serving.py`, file path relative
to `vendor/llumnix-ray/`):

```python
# benchmark/benchmark_serving.py:738-768
def sample_sharegpt_requests(
        dataset_path: str,
        num_requests: int,
        tokenizer,
        max_seqlen: int,
        ):
    # Load the dataset.
    prompts = []
    prompt_lens = []
    response_lens = []
    with open(dataset_path) as f:
        for line in f:
            data = json.loads(line)
            if len(data["conversations"]) >= 2:
                prompt = data["conversations"][0]["value"]   # ← first user turn only
                res    = data["conversations"][1]["value"]   # ← first assistant turn only
                prompt_token_ids = tokenizer(prompt).input_ids
                completion_token_ids = tokenizer(res).input_ids
                if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and \
                        len(prompt_token_ids) > 0 and len(completion_token_ids) > 0:
                    prompts.append(prompt)
                    prompt_lens.append(len(prompt_token_ids))
                    response_lens.append(len(completion_token_ids))
            if len(prompts) > num_requests:
                break
    # ... random sampling from the small pool ...
```

**Why this matters**:

- ShareGPT-GPT4 conversations have a median of ~4–6 turns and a long tail
  (some conversations have 20+ turns).
- A first-turn-only workload's prompt distribution sits in the
  100–1,500-token range. A multi-turn-flattened workload pushes that to
  3,000–15,000+ tokens for the later turns of a conversation, because
  each turn's input contains all prior turns.
- Llumnix's headline numbers (Figure 11 in the OSDI'24 paper) report tail
  latency improvements that are most visible *under memory pressure*.
  Removing 80%+ of the conversational length removes most of the
  pressure. The system "looks great" because it's not being tested.
- This is the classic "benchmark drift" — the harness reports honest
  numbers for the workload it actually feeds the system, but that
  workload is not representative of production multi-turn usage.

**Severity**: high. Anyone consuming Llumnix's published latency numbers
believes they apply to chatbot deployments. They don't. They apply to
*synthetic single-turn QA*, which is a different beast.

**Note on related sampler functions**:
- `sample_burstgpt_request` (lines 770-791) — BurstGPT data is per-prompt,
  not multi-turn, so this is a non-issue here.
- `sample_arxiv_request` (lines 793-815) — same shape as ShareGPT; uses
  `article_text` as the single prompt and `abstract_text` as the single
  response. Not multi-turn by nature so the first-turn problem doesn't
  apply.

### §6.1.2 P2: hidden length filter

**Code locations** (all in `benchmark/benchmark_serving.py`):

1. **The CLI flag**:
   ```python
   # line 846
   parser.add_argument('--max_request_len', type=int, default=16384)
   ```

2. **The pre-filter** in each sampler — drops conversations exceeding the
   cap before they're even candidates:
   ```python
   # benchmark_serving.py:756 (sharegpt sampler)
   if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and \
           len(prompt_token_ids) > 0 and len(completion_token_ids) > 0:
       prompts.append(prompt)
   # benchmark_serving.py:786 (burstgpt sampler)
   if request_tokens[idx] + response_tokens[idx] < max_seqlen and ...
   # benchmark_serving.py:809 (arxiv sampler)
   if len(prompt_token_ids) + len(completion_token_ids) < max_seqlen and ...
   ```

3. **The truncate-during-run path**:
   ```python
   # benchmark_serving.py:920-925
   for i, (prompt_len, gen_len) in enumerate(zip(prompt_lens, response_lens)):
       total = prompt_len + gen_len
       if total > args.max_request_len:
           print(f'truncating long prompt+gen_len {prompt_len=} {gen_len=}')
           gen_len = args.max_request_len - prompt_len
       response_lens[i] = gen_len
   ```

4. **The vLLM/Llumnix server-side cap**, set by `simple_bench.sh`:
   - `MAX_MODEL_LEN=16384` (line 15 of `scripts/simple_bench.sh` in the
     up-to-date LCMigrate copy) → passed as `--max-model-len 16384` to
     both `vllm.entrypoints.openai.api_server` and Llumnix.
   - The older `LCMigrate/scripts/simple_bench.sh` (a stale wrapper) had
     `--max-model-len 2000`, even more aggressive.

5. **The per-request cap**, set in `vllm_server_req_func` (line 84) and
   `bladellm_server_req_func` (line 108): `max_tokens` defaults to the
   sampled response length.

**Why this matters**:

- The pre-filter is the most pernicious form: filtered-out conversations
  *never appear in any output*. There's no log line, no warning, no
  histogram showing the dropped tail. Researchers reading the results
  literally cannot tell that the long tail was excluded.
- The truncate-during-run path *does* print "truncating long prompt+gen_len"
  but that's stderr noise; it doesn't enter any plot or summary.
- Llumnix's own paper §4 motivates the system by pointing to memory
  fragmentation and OOM-induced preemption. Filtering out the
  conversations most likely to cause exactly that behavior is
  self-defeating.

**Severity**: high, and *compounded* by P1. With first-turn-only
sampling, virtually no conversation exceeds 16K, so P2 looks dormant —
but it's lying in wait. The moment you enable multi-turn (Version A or B
below), the long tail appears, and the filter kicks in to silently
remove it again.

### §6.1.3 Combined effect

The two problems together produce a workload whose 99th-percentile
total length is bounded both by *first-turn-only* and *under-16K* —
neither bound is documented in the published numbers. Fixing P1 without
fixing P2 just substitutes truncation for filtering. Both must be fixed
to expose realistic preemption rates.

## §6.2 Llama-70B context-length reference

The other agent will determine which Llama-70B variant its existing
scripts already pin and use that variant's native context length. For
context, here are the official figures for the family:

| Model | Native context | Source |
|---|---|---|
| Llama-2-70B / 70B-Chat | 4,096 | original Meta release |
| Llama-3-70B / Instruct | 8,192 | Meta Llama 3 release |
| **Llama-3.1-70B-Instruct** | **131,072 (128K)** | [HF model card](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [AWS Bedrock model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-meta-llama-3-1-70b-instruct.html) |
| Llama-3.3-70B-Instruct | 131,072 (128K) | [HF model card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |

Practical caveat: vLLM's `--max-model-len` is bounded *both* by the
model's native context and by KV-cache memory budget. On 8×H100 (640 GB
HBM), the model itself takes ~140 GB BF16 (70B params × 2 bytes), plus
overhead, leaving ~480 GB for KV. Llama-3.1-70B has 80 layers × 8 KV
heads (GQA) × 128 head-size × 2 (K+V) × 2 bytes = 327 KB per token. So
a single 128K-context request would require 41 GB of KV, plus the
running batch. The agent should set `--max-model-len` to the model's
native limit (e.g. 131072) but expect that **GPU memory will cap actual
usable context** — that's exactly the regime where preemptions happen
and exactly what we want to expose.

## §6.3 Recommended experimental design (three vLLM versions)

All three runs use **vanilla vLLM** (not Llumnix), via `--server-api openai`
in the modified `benchmark_serving.py`. All three use the **same**
Llama-70B variant the agent has already chosen, with `--max-model-len`
set to that variant's native context length. The differences are
exclusively in *how the workload is sampled and dispatched*.

### Version 0 — "Existing" (baseline)

The agent's most recent vLLM-only baseline run, *before* this brief.
The first-turn-only sampling and 16K filter are still in effect.
Purpose: establish that our changes actually do change something —
without this baseline, we have no anchor.

### Version A — multi-turn flatten + length cap fix

For each ShareGPT conversation, **concatenate all user turns into one
prompt and all assistant turns into one completion**, separated by
`"\n"`:

```python
# pseudocode for sample_sharegpt_requests modification
prompt = "\n".join(turn["value"] for turn in data["conversations"]
                   if turn["from"] == "human")
response = "\n".join(turn["value"] for turn in data["conversations"]
                     if turn["from"] == "gpt")
```

Then:
- Remove the `< max_seqlen` filter (the long ones must be visible).
- Set `--max_request_len` and `--max-model-len` to the model's native
  context (e.g. 131072 for Llama-3.1-70B-Instruct).
- If a sampled conversation still exceeds even this enlarged cap, log
  the count and length distribution of dropped/truncated conversations
  prominently.

This produces a **single-shot offline-style workload** whose
prompt+response length distribution is realistic.

### Version B — multi-turn streaming + post-pause cache-survival probe

For each ShareGPT conversation, send turns *sequentially* over time:
- Turn 1: send user prompt 1, await assistant reply 1.
- Wait **20 seconds** (simulates user reading + typing).
- Turn 2: send `[user1, ai1, user2]` as prompt, await reply 2.
- ... (repeat) ...
- After the last *real* turn (call it `ai_N`):
  - Wait **2 minutes** (120s).
  - Send a **short probe turn**: prompt =
    `[user1, ai1, ..., user_N, ai_N, "ok"]` with `max_tokens=1`. This is
    a fake follow-up turn whose only purpose is to probe whether the
    conversation's prefix-cache blocks survived the pause. Discard the
    probe response content; record its **prefill latency** as the
    diagnostic.

Server config:
- `--enable-prefix-caching` so the engine reuses KV blocks across
  turns of the same conversation.
- `--max-model-len` and `--max_request_len` as in Version A.

**Why the probe** (originally we'd just wait 2 min and move on; this
trick is strictly better): vLLM 0.6.3 has no API to pin or selectively
free KV blocks. Prefix-cache retention is *soft* — blocks can be evicted
by LRU under memory pressure. With just a passive 2-min wait we have to
peek at the block manager to know if the cache survived. With the probe,
the prefill latency at the 2-min mark is a direct, *observable* answer:
- **Cache hit** → prefill latency ≈ tens of ms (only the trailing `"ok"`
  needs to be prefilled). Retention worked.
- **Cache miss** → prefill latency proportional to full prefix length
  (potentially seconds for a 16K-token conversation). The cache was
  evicted under pressure during the pause.

The probe doesn't *actively* free anything — it just measures survival.
The actual freeing still happens via natural LRU eviction as new
conversations arrive and put pressure on memory.

**Implementation note**: prefix-cache hits in vLLM 0.6.3 require
*token-exact prefix matches* (the block-hash chain is over token IDs,
16 tokens per block). Two safe ways to maintain that across turns:
1. Use `/v1/chat/completions` with the full message history each turn —
   vLLM's chat template tokenizes consistently across turns.
2. On `/v1/completions`, store the *exact token IDs* returned by each
   prior turn and feed them via `prompt_token_ids=[...]` rather than
   re-tokenizing the assistant's text from string. Re-tokenization can
   silently introduce BOS/special-token differences and cause
   unintended cache misses.

The probe turn must use the same prompt-construction path as the real
turns. If the cross-agent attempt at token alignment turns out fiddly,
the documented fallback is to drop the probe and revert to the simpler
"wait 2 minutes doing nothing" version — losing the observable
cache-survival measurement but preserving the rest of VB's data.

### What to measure (all three versions)

Per-request, every request:
- arrival timestamp
- prompt length (tokens)
- output length (tokens)
- queue latency (arrival → first token)
- prefill latency (first token timing)
- decode latency (per-token, mean and full distribution)
- e2e latency (arrival → last token)

At each preemption event:
- timestamp
- request_id of preempted sequence
- request's current context length (tokens generated so far)
- preemption reason: `RECOMPUTE` or `SWAP`
- VRAM usage at the preemption moment (read via `nvidia-smi`)
- P99 e2e latency over the most recent N completed requests
  (sliding window — N=50 is reasonable). This is the "P99 over recent
  requests" the user asked for.

Aggregate:
- total preemption count
- preemption rate (preemptions per second, and per request)
- count of conversations dropped/truncated by the length filter
- length distribution of the resulting workload (P50/P95/max for prompt,
  output, and total)

## §6.4 vLLM 0.6.3 preemption mechanics — and why the user's C3 hypothesis is likely wrong

The user's stated hypothesis (C3 in the prior round) was:
> "I believe vLLM sometimes triggers preemption when it finds the recent P99
> is too high."

This is **almost certainly wrong** for vLLM 0.6.3. Here's why:

**Preemption decision logic** (in `vllm/core/scheduler.py:Scheduler._schedule_running`):
- The scheduler iterates over running sequence groups in order.
- For each group it calls `block_manager.can_append_slots(seq_group)` to
  check whether a fresh KV block can be allocated to extend the group's
  decode by one token.
- If `can_append_slots` returns False (i.e. no free GPU blocks), the
  scheduler *frees memory by preempting* the lowest-priority running
  sequence: it calls `_preempt(victim, blocks_to_swap_out, preemption_mode)`.
- Preemption mode is `RECOMPUTE` (drop the victim's KV, return it to the
  waiting queue, will re-prefill from scratch later) or `SWAP` (move the
  victim's KV blocks to CPU memory, can resume later without
  recomputing) — chosen based on configuration and victim size.

**There is no latency-based code path** in this decision. P99, queue
times, "recent requests" — none of those signals enter `_preempt` or
`_schedule_running`. Preemption is purely memory-pressure-driven.

**What actually happens** (the correlation that masquerades as causation):
- Memory pressure builds up → KV blocks become scarce.
- Scarce blocks → continuous batching falls back to smaller running
  batches, slower decode, longer queue waits.
- Slower decode + longer queues → P99 latency goes up.
- The same memory pressure simultaneously causes `can_append_slots` to
  return False → preemption fires.

So: high P99 and preemptions both *follow from* memory pressure. They
correlate but neither causes the other. The data the agent collects will
show this correlation clearly — that's still publishable evidence of "P99
spikes coincide with preemption density" — but the report should not
claim "vLLM uses P99 as a preemption signal".

**Implication for the experiment**: collecting "P99 over recent requests
at the preemption point" is still worth doing — it's a marginal
additional measurement with high diagnostic value — but the agent should
report results as correlation, not as evidence of latency-driven
preemption.

## §6.5 Limitations and caveats

- **Version B's retention is soft, made observable by the probe**.
  vLLM 0.6.3 has no API to pin KV blocks against eviction or to
  selectively free a single conversation's blocks; under sustained
  pressure the `--enable-prefix-caching` blocks will be evicted by the
  LRU allocator. This is *realistic* (real systems also evict under
  pressure). The probe-turn approach (§6.3 Version B) doesn't change
  this — it just makes the eviction *measurable* by yielding a
  prefill-latency reading at the 2-min mark. Expect a bimodal probe-
  latency distribution: a low-ms cluster (hits = cache survived) and a
  seconds-scale cluster (misses = evicted). The hit/miss ratio per
  version is the headline number.
- **Probe-turn token alignment is fiddly**. If the agent re-tokenizes
  assistant responses from strings between turns, the resulting token
  IDs may differ from what was actually generated, causing the probe to
  miss the cache for the wrong reason. The two safe paths are
  `/v1/chat/completions` (vLLM's chat template handles tokenization
  consistently) or feeding `prompt_token_ids=[...]` from previously
  returned token IDs. The cross-agent brief explicitly calls this out
  and provides a fallback (drop the probe, revert to passive 2-min
  wait) if alignment turns out blocking.
- **Llama-3.1-70B's 128K is not always practical**. On 8×H100 the agent
  may need to set a smaller `--max-model-len` to avoid OOM at engine
  startup. If so, the agent should pick the largest practical value
  and document the choice.
- **The vLLM scheduler instrumentation requires a small monkey-patch**.
  Adding a structured log to `Scheduler._preempt` is the cleanest way to
  capture preemption events; the agent should patch in the
  user's vLLM install (editable or source build) or wrap the method.
- **The 5-retry budget is tight**. The agent should prioritize getting
  *some* form of all three versions running, even if it means
  simplifying instrumentation (e.g., parsing existing log lines instead
  of adding structured logging). The detailed brief below explicitly
  enumerates which requirements are core vs droppable.

## §6.6 Cross-agent message — copy-pasteable block

The block below is what to hand verbatim to the other Claude Code session.
It is self-contained: it does not reference this notes file, and uses
relative paths + grep-able strings so the agent can locate the right code
on its machine without your absolute paths.

---

```text
============================================================================
TASK BRIEF: ShareGPT-fix preemption experiment for vLLM 0.6.3.post1
============================================================================

You are receiving this from another Claude Code session that audited the
Llumnix benchmark code on a different machine. Read this entire brief
before changing anything.

---------- Background you need to know ----------

Two latent flaws in Llumnix's benchmark harness make all its published
numbers (and any vLLM numbers run through this harness) unrepresentative
of multi-turn chatbot workload:

  P1. ShareGPT first-turn-only sampling. The function
      `sample_sharegpt_requests` in `benchmark/benchmark_serving.py`
      reads only `data["conversations"][0]` (first user prompt) and
      `data["conversations"][1]` (first assistant reply). Every later
      turn is silently discarded. ShareGPT-GPT4 conversations are
      multi-turn; using only the first turn massively under-represents
      realistic prompt+output length.

  P2. Hidden length filter. Every dataset sampler has the form
        `if prompt_tokens + completion_tokens < max_seqlen: ...`
      Conversations exceeding `--max_request_len` (default 16384) are
      DROPPED FROM THE CANDIDATE POOL with no log line. Any sampled
      conversation that exceeds the cap during the run gets its response
      truncated (line ~922 of benchmark_serving.py). The long-tail
      conversations that would stress memory hardest are silently
      excluded from results.

vLLM 0.6.3's preemption is purely memory-pressure-driven (see
`vllm/core/scheduler.py:Scheduler._schedule_running` →
`Scheduler._preempt`). Preempts fire when `block_manager.can_append_slots`
returns False. Latency, P99, queue time — none of those signals enter
the decision. So when the workload is unrepresentatively short, the
preemption rate is unrepresentatively low.

Your goal: quantify the preemption rate of vanilla vLLM under realistic
multi-turn ShareGPT load.

---------- Your environment ----------

You have already been working on vLLM/Llumnix experiments. Some of your
scripts likely live in an `LCMigrate/`-style wrapper directory and you
may have created NEW scripts in OTHER dirs since. Do this first:

  1. Identify your existing experiment scripts. Start by reading the
     wrapper repo (LCMigrate or similar): top-level `simple_bench.sh`,
     `simple_bench_run.sh`, plus anything under `scripts/`. Then run
     `find ~ -name "*.sh" -newer LCMigrate/scripts/simple_bench.sh`
     (or equivalent) to find scripts you've added since.

  2. Identify which Llama-70B variant your scripts have been using.
     Likely candidates and their native context lengths:
       Llama-3.1-70B-Instruct → 131072
       Llama-3.3-70B-Instruct → 131072
       Llama-3-70B            →   8192
       Llama-2-70B            →   4096
     Pick the variant your scripts already pin; use its native context
     length as `--max-model-len`. If GPU memory makes the native length
     impractical, pick the largest power-of-2 below it that the GPUs
     can serve at startup, and DOCUMENT THE CHOICE.

  3. Locate your Llumnix tree. The benchmark_serving.py with the flaws
     is at `benchmark/benchmark_serving.py` relative to the Llumnix
     repo root. Verify with:
        grep -n 'data\["conversations"\]\[0\]' benchmark/benchmark_serving.py
     The match should be inside `sample_sharegpt_requests`.

---------- The experimental matrix: three vLLM-only runs ----------

You will run vanilla vLLM (NOT Llumnix), exposed via the OpenAI-compatible
server (`vllm.entrypoints.openai.api_server`). The benchmark client
(`benchmark/benchmark_serving.py`) supports `--server-api openai` to
target this.

PRIMARY REQUEST RATE: 12 requests per second (`--qps 12`). Run all three
configurations below at this rate first. The trace generator should be
Poisson (`--distribution poisson`) so inter-arrival times are
exponentially distributed with mean 1/12 = ~83ms.

Three configurations, same model, same trace seeds, same QPS=12:

  V0 (existing) - your most recent vLLM-only baseline run, BEFORE this
                  brief. First-turn-only sampling and 16K filter still
                  in effect. Purpose: anchor for "what changed".
                  If you don't have a clean recent baseline, run the
                  scripts as-they-currently-are once before any
                  modification.

  VA (multi-turn flatten + length cap fix):
     For each ShareGPT conversation, concatenate all user turns into
     one prompt and all assistant turns into one completion, joined by
     "\n":
       prompt   = "\n".join(t["value"] for t in conv if t["from"] == "human")
       response = "\n".join(t["value"] for t in conv if t["from"] == "gpt")
     Remove the `< max_seqlen` pre-filter so long conversations are
     visible. Set `--max-model-len` and `--max_request_len` to the
     model's native context length (e.g. 131072 for Llama-3.1-70B).
     LOG (don't drop) any conversation still exceeding this cap; report
     count and length distribution.

  VB (multi-turn streaming + post-pause cache-survival probe):
     Send turns of each conversation SEQUENTIALLY over time:
       Turn 1: send user_1, await ai_1.
       Wait 20 seconds (simulates user reading + typing).
       Turn 2: send (user_1 + ai_1 + user_2) as the prompt, await ai_2.
       ... repeat for all real turns ...
       After the LAST real turn (call it ai_N):
         Wait 2 minutes (120s).
         Then send a SHORT PROBE TURN: prompt =
            (user_1 + ai_1 + ... + user_N + ai_N + "ok")
         with `max_tokens=1`. This is a fake follow-up turn whose ONLY
         purpose is to probe whether vLLM's prefix-cache blocks for the
         conversation prefix survived the 2-minute pause.
         Discard the probe's response content.
     Server flags: `--enable-prefix-caching` so multi-turn prefixes
     reuse KV blocks. Same `--max-model-len` and `--max_request_len`
     as VA.

     WHY THE PROBE: vLLM 0.6.3 has no API to pin or selectively free
     KV blocks. Prefix-cache retention is SOFT - blocks can be evicted
     by LRU under memory pressure. The probe at the 2-min mark gives
     us an OBSERVABLE measurement: if the probe's prefill latency is
     ~tens of ms (just the "ok" suffix to prefill), the cache survived
     the pause; if the probe's prefill latency is proportional to the
     full prefix length (potentially seconds for a 16K-token
     conversation), the cache was evicted under pressure during the
     pause. Record this as `cache_survival_probe_prefill_latency` per
     conversation and report its distribution per version.

     IMPORTANT IMPLEMENTATION NOTES for the probe:
     1. Prefix-cache hits require TOKEN-EXACT MATCHES in vLLM 0.6.3
        (block hashing is over token IDs, 16 tokens per block). Two
        safe ways to maintain that:
        (i)  Use `/v1/chat/completions` with the full message history
             every turn - vLLM's chat template tokenizes consistently
             across turns of the same conversation.
        (ii) On `/v1/completions`, store the EXACT token IDs returned
             by each prior turn (do not re-tokenize the assistant's
             text from string) and pass them via `prompt_token_ids=[...]`
             in subsequent turns. Re-tokenizing the response string
             can introduce BOS/special-token differences and cause
             unintended cache misses.
     2. The probe MUST use the same prompt-construction path as the
        real turns. If you re-tokenize for the probe but used token
        IDs for real turns (or vice versa), the probe will miss the
        cache for the wrong reason.
     3. `max_tokens=1` works in vLLM 0.6.3; `max_tokens=0` does not.
     4. EXCLUDE the probe from latency aggregates (queue / prefill /
        decode / e2e). The probe is a diagnostic call, not a "real"
        turn; mixing it into per-turn distributions skews them.
        Keep its latency in a separate per-conversation field.

     FALLBACK if the probe approach gets stuck (token-alignment is
     fiddly): drop the probe entirely and revert to the simpler "wait
     2 minutes doing nothing, no probe" version. This loses the
     observable cache-survival measurement but still produces the
     other VB metrics (preemption rate during streaming, length
     distribution). Document this fallback explicitly in the report.

---------- QPS sweep (only if QPS=12 results look sane) ----------

After completing all three versions at QPS=12 and reviewing the data,
sweep the request rate across {4, 8, 16, 20, 24} req/s (i.e. 4 to 24
req/s on an interval of 4, skipping the 12 you already did). Run all
three versions (V0, VA, VB) at each rate. This produces a 3-version x
6-rate matrix.

DO NOT START THE SWEEP if the QPS=12 results look wrong. Specifically,
abort the sweep and report instead if any of these hold at QPS=12:
  - Server crashes, hangs, or OOMs at startup.
  - Latency numbers are absurd (e.g. P50 e2e > 5x what you got at
    QPS=12 in your prior baseline runs of the same model, suggesting
    something broke).
  - Preemption rate is exactly 0 in ALL three versions (suggests the
    instrumentation patch is not actually firing — silent failure).
  - Preemption rate is greater than 50% in V0 (suggests memory was
    misconfigured before any of our changes; not a meaningful baseline).
  - VA or VB drop more than 80% of conversations via the length filter
    (suggests --max-model-len was set too small to absorb multi-turn
    flatten).

If the QPS=12 results look sane, proceed with the sweep. The sweep is
secondary; if you run out of retry budget or wallclock, prioritize the
QPS=12 runs being complete and correct over having more rates.

REPORT ORDERING: produce the QPS=12 report first as a standalone
deliverable. Then, if the sweep runs, append a per-rate table showing
preemption rate vs QPS for each version, plus a brief observation about
where each version starts to break (e.g. "V0 holds up to QPS=20; VA
exceeds 5% preempt rate at QPS=16"). Do not block the QPS=12 deliverable
on the sweep.

---------- What to measure (all three versions) ----------

Per-request, EVERY request:
  - arrival timestamp
  - prompt length (tokens)
  - output length (tokens)
  - queue latency  (arrival -> first token sent to engine)
  - prefill latency (first token returned by engine)
  - decode latency  (mean per-output-token over the response)
  - e2e latency     (arrival -> last token)

At each preemption event:
  - timestamp
  - request_id of preempted sequence
  - request's current context length (tokens generated so far)
  - preemption reason: RECOMPUTE or SWAP (the two enum values vLLM
    0.6.3 emits in `Scheduler._preempt`)
  - VRAM usage (nvidia-smi at the moment, or whatever is cheapest)
  - P99 e2e latency over the last 50 completed requests
    (sliding window). This is for correlation analysis.

Aggregate, per version:
  - total preemption count
  - preemption rate (per second, and per request)
  - count of conversations dropped/truncated by the length filter
  - workload length distribution: P50/P95/max for prompt, output, and
    prompt+output total
  - [VB only] cache-survival-probe prefill-latency distribution
    (P50/P95/max). A bimodal distribution (one cluster of low-ms
    "cache hit" probes and one cluster of seconds-scale "cache miss"
    probes) is the expected diagnostic shape. Report the fraction of
    probes that hit (low-ms cluster) vs missed (high-ms cluster).

---------- How to instrument vLLM 0.6.3 for preemption logging ----------

vLLM 0.6.3's `Scheduler._preempt` does not emit structured records by
default. To get the per-event data, monkey-patch or directly modify
`vllm/core/scheduler.py:Scheduler._preempt` to append a row to a JSONL
file (or print a structured marker line we can grep) containing:
request_id, current `seq.get_len()`, preemption_mode (RECOMPUTE/SWAP),
timestamp, and `len(self.running)` / `len(self.waiting)` / `len(self.swapped)`
as queue snapshots.

Two valid approaches; prefer (a) but fall back to (b) if (a) gets stuck:

  (a) Editable install + small source patch in `vllm/core/scheduler.py`.
      Cleanest. Works if your vLLM is installed via `pip install -e`.
  (b) Subclass `Scheduler` in your benchmark wrapper and use
      `LLMEngine`'s constructor injection, OR patch via runtime
      monkey-patch (`vllm.core.scheduler.Scheduler._preempt = ...`).
      Less clean but doesn't touch installed source.
  (c) [LAST RESORT] Parse vLLM's existing log line:
        "Sequence ... preempted by ... mode."
      Time-stamped from the log line. Lacks queue snapshots and
      victim's exact length. Use only if (a) and (b) fail twice.

Keep the modification SMALL. We are NOT trying to land changes upstream.

---------- Latency logging ----------

Instrument the OpenAI-server request handler (or the benchmark client)
to capture per-request timestamps. The client side is easier and does
NOT require modifying vLLM. Use the existing `MeasureLatency` class in
`benchmark/benchmark_serving.py` as the base and extend it to record
queue/prefill/decode separately. If extracting `prefill` vs `decode`
cleanly is hard, log just queue and e2e and compute decode from output
tokens. DON'T LET LATENCY INSTRUMENTATION BLOCK THE EXPERIMENT.

---------- Workload sampling changes (most important code change) ----------

In `benchmark/benchmark_serving.py:sample_sharegpt_requests`:

  - Add a CLI flag `--multi_turn_mode {first,flatten,streaming}` with
    default `first` (the existing behavior). Wire it through `main()`
    to the sampler.
  - If `flatten`: build prompt and response by joining all turns of the
    matching role with `"\n"`.
  - If `streaming`: emit per-turn requests with the inter-turn waits
    described above. This requires the benchmark client to dispatch
    multiple sequential requests per "conversation" with a 20s wait
    between each, AND (after the last) to hold the conversation slot
    for 120s before another conversation reuses it.
  - Remove or relax the `< max_seqlen` filter for VA and VB. Replace
    with: log conversations that exceed; do not exclude them.

Make the smallest changes that achieve these. Prefer adding a new
sampler function (e.g. `sample_sharegpt_requests_flatten`,
`sample_sharegpt_requests_streaming`) over modifying the existing one,
so V0 still works unchanged.

---------- Retry policy (you have 5 attempts) ----------

If experiments fail, fix carefully and retry. Maximum 5 retries TOTAL
across the whole task. Priority order if you have to drop scope:

  CORE (must complete):
    1. V0, VA, VB all run end-to-end and produce per-request latency logs.
    2. Preemption count per version (even if individual events lack
       VRAM/P99 detail).
    3. Length distribution per version (P50/P95/max).

  STRETCH (drop these first if blocked):
    4. P99-over-recent-50-completed at preemption events.
    5. VRAM usage at preemption events.
    6. Reason = RECOMPUTE vs SWAP (if you can't patch vLLM, derive from
       log lines or skip).
    7. Streaming wait timing in VB exactly matching 20s/120s (if
       harness limitations force coarser timing, document and use
       what works).

If you drop ANY stretch item, EXPLICITLY name what you dropped in the
final report and explain why it doesn't undermine the core finding
(preemption count / rate per version).

---------- Final deliverable ----------

A concise report with:

  1. Methodology block: which model variant, which max-model-len, which
     vLLM version (should be 0.6.3.post1), GPU topology, QPS and
     trace size.
  2. Per-version table: total requests, total preemptions, preemption
     rate, dropped/truncated count, workload length P50/P95/max.
  3. A scatter or correlation plot of "preemption density vs P99 over
     recent 50" if you have the data; otherwise per-version preemption
     timelines (preemption events per minute).
  4. Plain-language conclusion answering: how does preemption rate
     change between V0 and {VA, VB}? Are the rates "low" (< 1% of
     requests preempted), "moderate" (1-10%), or "high" (>10%)? Use
     vLLM-team rule-of-thumb: > 5% preemption usually indicates
     under-provisioned KV memory.
  5. A note clarifying: vLLM preemption is purely memory-bound (not
     latency-driven). The "P99 at preemption point" data is correlation,
     not causation.

---------- Things you should NOT do ----------

  - Do not modify Llumnix's manager.py or any Llumnix-specific scheduler
    code. The whole experiment is vLLM-only; bypass Llumnix entirely.
  - Do not commit the modified benchmark_serving.py to a shared branch
    without flagging that the V0 sampler stays as the default and
    `--multi_turn_mode` switches the new behavior.
  - Do not reduce QPS or num_prompts to make experiments "fit" - that
    eliminates the very memory pressure we want to measure.
  - Do not increase --max-model-len beyond what the model's native
    context allows - it'll silently fail or give wrong results.

---------- Sanity checks before each run ----------

  - `python -c "import vllm; print(vllm.__version__)"` returns 0.6.3.post1.
  - `nvidia-smi` shows expected GPUs and free memory.
  - `python -c "from transformers import AutoTokenizer;
     print(AutoTokenizer.from_pretrained('<model_path>').model_max_length)"`
     returns the expected native context length.
  - For VB: `--enable-prefix-caching` is in your vLLM launch command.
  - For VA/VB: `--max-model-len` matches the model's native context.

============================================================================
END OF BRIEF
============================================================================
```

---

(End of §6. The block above is the deliverable to hand to the other agent.)

---

# §99. Queued for user return

This section grows as I work; it is the punch-list for our next live session.

## Q1. Permissions to grant for the next session

When you come back, please consider granting these for the rest of this
project (or per-session if you prefer):

- **`Read`** — I want to read ~30 specific files myself in Round 2 (list in Q4).
  Read is fully read-only and can never modify code or run commands. Approve
  blanket.
- **`Agent` with `subagent_type=Explore`** — I want ~5 more deep-dive agents
  in Round 3 once I've done Round 2 reads (list in Q5). Each agent runs
  read-only.
- **`Edit` on `notes/00_permission_test.md`** — already auto-allowed; keep as is.

I will NOT need: Bash (we'll use Explore for any searching), Write to other
files, git, network.

## Q2. Open design questions for the user

**About the project's scope and goals:**
- Q2.1: Is the goal of the project (a) reproduce Llumnix's figures and *then*
  add PP-reshard on top, or (b) skip Llumnix reproduction and go straight to
  PP-reshard demonstration? The artifact's run.sh ↔ this fork's CLI mismatch
  affects this — we'd need to port the workload configs.
- Q2.2: For PP-reshard specifically, is the immediate target paper-figure
  comparable to Llumnix's Fig. 11 / Fig. 13, or a bigger paper of its own
  (introducing a new latency knob)?
- Q2.3: What's the targeted GPU testbed for experiments? (4×A10 like the
  paper? H100? B200? This shapes what's worth optimizing.)
- Q2.4: Do we plan to support PP=2 only, or arbitrary PP=k≥2 for very long
  contexts? (Affects state machine complexity.)
- Q2.5: When you say "deeply modify vLLM" — are you OK if the modifications
  are disruptive enough that we keep our own fork rather than upstreaming?

**About scope decisions for the implementation:**
- Q2.6: Are we OK requiring `pipeline_parallel_size=1` on every Llumnix
  instance (so each instance has the full model loaded)? This is the simplest
  path. Alternative: each instance is already PP=k, and we reshard *across*
  PP groups — much harder.
- Q2.7: Are we OK requiring `tensor_parallel_size` to match between any two
  instances that might PP-reshard together? (Per-rank KV layout assumes
  matching TP. Mooncake supports heterogeneous TP, but let's start with
  homogeneous.)
- Q2.8: Do we want PP-reshard to be triggered (a) as a fallback when
  full-migration would OOM the destination, (b) as a primary tool with its
  own scheduler policy, or (c) both? My §2.1 strategy assumes (c).
- Q2.9: How should we handle the "merge back to PP=1" path? Always merge
  when the partner has spare memory (eager merge), or only merge when one
  side is shutting down (lazy merge)? Latter is simpler to start.

**About fork-vs-upstream:**
- Q2.10: Why does this fork's scheduler appear to *lack* paper-faithful
  virtual-usage and priority support? (See §3.1.3.) Is this an intentional
  simplification by upstream Llumnix, or a feature we should reinstate?
  If we want the priority experiments (Fig. 13) we may need to add it back.

**About vllm version targeting:**
- Q2.11: The fork has both `backends/vllm/` (0.6.3) and `backends/vllm_v1/`
  (latest v1) directories, with v1 looking thinner / less developed. Which
  do we target for our PP-reshard work? My recommendation: target v1 (per
  your earlier instruction) but keep working migration on v0 as a baseline
  to compare against.
- Q2.12: vllm-latest moves fast. Do we pin a specific commit / tag, or
  follow tip? Llumnix's PR cadence vs vllm tip will determine pain level.

**Concrete sanity checks on Llumnix's mechanism:**
- Q2.13: Per A2's reading, Llumnix's KV transfer uses NCCL or Gloo via Ray
  collective groups. Why not use the same KV-connector layer that vllm-latest
  introduced? Probably because Llumnix predates it. **Should we propose
  refactoring Llumnix migration to ride on `KVConnectorBase_V1` connectors
  (e.g. NIXL)?** This would unify our PP-reshard with the rest of the
  ecosystem and remove a maintenance burden. But it's a bigger refactor;
  may be deferrable.
- Q2.14: The agent A1 said "no monkey-patches" — but the fork has
  `LLMEngineLlumnix(_AsyncLLMEngine)`, which means Llumnix tightly couples
  to vLLM internals. How fragile is this against vllm-latest API changes?

**About the experimental method:**
- Q2.15: Do we need to recreate the ShareGPT/BurstGPT/power-law trace
  generators, or is there a script we can lift from the artifact?
- Q2.16: What's our comparison baseline going to be? Llumnix as-is?
  vllm-latest disagg PD? Both?

## Q3. Commands I considered but did not run

I did not run any Bash commands. I considered but skipped:
- `find ... -name '*.py' | wc -l` to count source files (could ask Explore
  agent if needed; not blocking).
- `grep -rn 'reshard\|pp_reshard\|split.*layer'` across the fork to confirm
  no in-progress PP-reshard work — A1 covered this functionally; flagging
  here in case you want stricter confirmation.
- `git log --oneline -50` to see recent commits and confirm fork-vs-upstream
  drift — A1 paraphrased commit messages but I haven't read them.

## Q4. Round-2 file reads I want to do (myself, with Read tool)

These are the [V*] flagged items from §3, prioritized. Total ≈ 30 files,
mostly small. Listed by codebase:

**reshardLLM (highest priority — what we'll modify first):**
1. `llumnix/llumlet/migration_coordinator.py` — full file, the multi-stage
   state machine ([V6])
2. `llumnix/llumlet/local_migration_scheduler.py` — full file, request
   selection policies
3. `llumnix/llumlet/llumlet.py` — Ray actor surface, RPC method list
4. `llumnix/llumlet/request.py` — `LlumnixRequest`, migration state fields ([V4])
5. `llumnix/backends/vllm/migration_backend.py` — full file, the actual
   KV transfer code ([V2])
6. `llumnix/backends/vllm/scheduler.py` — `SchedulerLlumnix`, pre_alloc_cache
   logic ([V1])
7. `llumnix/backends/vllm/llm_engine.py` — `BackendVLLM`, `LLMEngineLlumnix`,
   the engine-side migration entry points
8. `llumnix/backends/vllm/worker.py` — `MigrationWorker`, `do_send`/`do_recv`
9. `llumnix/global_scheduler/migration_scheduler.py` + `migration_policy.py`
10. `llumnix/global_scheduler/global_scheduler.py` — top-level
11. `llumnix/manager.py` — orchestration & polling loop
12. `llumnix/instance_info.py` — what's actually reported each poll
13. `llumnix/load_computation.py` — load metric formulas
14. `llumnix/config/default.py` — confirm all defaults ([V3])
15. `llumnix/internal_config.py` — `MigrationConfig` definition
16. `docs/Prefill-decode_Disaggregation.md` — what PD-disagg looks like in
    this fork ([V5])
17. `tests/e2e_test/test_dynamic_pd.py` — the PD test ([V5])
18. `tests/e2e_test/test_migration.py` — the migration test (golden behavior)

**vllm 0.6.3 (so we understand what Llumnix migration depends on):**
19. `vllm/core/scheduler.py` (`Scheduler.schedule`) — the `_schedule_running`,
    `_schedule_prefills`, preemption paths
20. `vllm/core/block_manager.py` — confirm `SelfAttnBlockSpaceManager` is
    default ([V7])
21. `vllm/core/block/block_table.py` — `BlockTable.allocate / append / fork / free`
22. `vllm/worker/cache_engine.py` — the actual `_allocate_kv_cache` and
    `swap_in/out` ([V10])
23. `vllm/model_executor/models/llama.py` — `LlamaModel.forward()` for
    PP intermediates ([V9])
24. `vllm/distributed/utils.py` — `get_pp_indices`
25. `vllm/distributed/parallel_state.py` — `initialize_model_parallel`, `get_pp_group`

**vllm-latest (the eventual base + KV connector borrow targets):**
26. `vllm/v1/core/sched/scheduler.py` ([V12]) — the heart of v1
27. `vllm/v1/core/sched/output.py` — `SchedulerOutput` shape
28. `vllm/v1/worker/kv_connector_model_runner_mixin.py` ([V13])
29. `vllm/distributed/kv_transfer/kv_connector/v1/base.py` ([V14])
30. `vllm/distributed/kv_transfer/kv_connector/v1/nixl/connector.py` ([V15])
31. `vllm/distributed/kv_transfer/kv_connector/v1/nixl/scheduler.py` ([V15])
32. `vllm/distributed/kv_transfer/kv_connector/v1/nixl/worker.py` ([V15])
33. `vllm/distributed/kv_transfer/kv_connector/v1/nixl/metadata.py` ([V15])
34. `vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py` ([V16])
35. `vllm/distributed/kv_transfer/kv_connector/v1/example_hidden_states_connector.py`
    — hidden-state transport precedent
36. `vllm/v1/worker/gpu_model_runner.py` (skim L3787–4137 for PP path) ([V17])
37. `vllm/v1/engine/core.py` (`step_with_batch_queue`) ([V18])
38. `vllm/v1/worker/ubatching.py` ([V19])
39. `vllm/v1/request.py` (find `WAITING_FOR_REMOTE_KVS`) ([V22])
40. `vllm/config/kv_transfer.py` ([V23])
41. `vllm/distributed/kv_transfer/kv_connector/factory.py` ([V24])
42. `vllm/distributed/kv_transfer/kv_transfer_state.py` — global agent
43. `examples/online_serving/disaggregated_serving/disagg_proxy_demo.py` ([V20])
44. `tests/v1/kv_connector/unit/test_remote_prefill_lifecycle.py` ([V21])

**Estimated time** to read all of these ≈ 1.5–2 hours of my context budget,
likely spread across multiple parallel reads.

## Q5. Round-3 Explore agents I want to spawn after Round-2 reads

Each is well-scoped and independent of the others (can run in parallel).

**A10 — End-to-end "what would PP-reshard look like in code" sketch.**
Given files I've now read, write a concrete file-by-file change plan: which
existing classes to subclass, which methods to override, what new classes
to add, in which directories. Make it a diff-able outline.

**A11 — Concrete comparison: Llumnix migration vs vllm-latest disagg PD.**
Side-by-side feature matrix on KV transfer protocol, scheduling, failure
handling, and IPC. Identify which Llumnix features have a clean v1 analog
and which would need to be ported.

**A12 — Hidden-state transport feasibility.**
Read `example_hidden_states_connector.py` plus relevant NIXL/Mooncake parts.
Can we ship hidden states between two instances per iteration with low
latency? What is the minimum round-trip we'd add to the decode step?

**A13 — Test-harness reuse audit.**
Walk `tests/e2e_test/test_migration.py` and the v1 KV-connector tests. List
the test fixtures we can reuse for our PP-reshard correctness tests; flag
the gaps we'd have to fill.

**A14 — Long-term maintenance: vllm-latest commit pinning.**
Look at how Llumnix's `requirements_vllm_v1.txt` pins vllm-latest, and
whether the fork's `backends/vllm_v1/` looks bit-rotted. Estimate the
maintenance burden of keeping our PP-reshard code current with vllm tip.

## Q6. Things I would have wanted to do but couldn't

- Actually run the existing migration test to confirm the codebase works
  (would need permission to run pytest).
- Inspect the deltas between the artifact's vendored vLLM and the standalone
  fork's `vllm/vllm` to understand what Alibaba changed in 0.6.3 specifically
  for Llumnix (would need git/diff).
- Run the workload generator and see what a "real" trace looks like.

## Q7. Status summary

**Done in session 1 (structural mapping):**
- Read Llumnix paper (from your first message) — full understanding.
- Round-1 structural mapping of all four codebases (9 Explore agents).
- Wrote §0–§3 (findings) and §99 (queue) in this file.

**Done in session 2 (this session — deep reads + implementation plan):**
- Direct end-to-end reads of 12 critical files (line-level accuracy):
  migration_coordinator.py, migration_backend.py, scheduler.py (Llumnix vLLM
  backend), llumlet.py, request.py, internal_config.py (Llumnix);
  KVConnectorBase_V1 base.py, factory.py, kv_transfer_state.py, v1/request.py,
  v1/core/sched/output.py, kv_transfer.py (vllm-latest).
- 5 deep-dive Explore agents on Llumnix backend internals (worker.py,
  llm_engine.py, sequence.py, vllm_v1/), NIXL connector, P2P-NCCL connector,
  v1 model runner / ubatching / KV connector mixin, disagg-PD lifecycle
  tests + examples, example/hidden-states/multi connectors.
- Wrote §4 — full step-by-step implementation plan with concrete file
  paths, line numbers, class/method names, test types, and verify
  conditions.

**Not done:**
- Any code changes (not in scope yet).
- Phase 0.1 environment setup (will need permission to run uv/pip/pytest).
- Verification reads of the [A?] tagged risks (R1–R8) before Phase 1
  Step 1.8 in particular.

**Recommended order when you're back:**
1. Read §4.0 (decisions, risks) and §4.2 Phase 1 plan.
2. Answer Q8 (new) — plan-specific questions.
3. Optionally answer Q2 — general design questions still open from session 1.
4. Approve Read + Agent broadly so I can verify the [A?] risks.
5. We commit to the plan, possibly with edits.
6. Begin Phase 0.

## Q8. Plan-specific questions (NEW this session)

**About the targeted backend**:
- Q8.1: Are you OK with the recommendation to **build on vLLM v0 first**
  (Phase 1–5) and port to v1 in Phase 6? The v1 backend in this fork is
  effectively a stub (only `async_core.py` with one `disagg.init` hook),
  so building on v1 first means building on nothing.
- Q8.2: If you'd rather start on v1, that's possible but I'd recommend we
  first complete a Llumnix-style migration on v1 (porting today's full
  migration to `KVConnectorBase_V1`), then add PP-reshard on top. That
  doubles Phase 6 effort but Phase 1's MVP would then be on v1.

**About the PP=1 forever assumption (D3)**:
- Q8.3: Are you OK with the constraint that each Llumnix instance runs
  `pipeline_parallel_size = 1` (full model loaded per instance)? Stock
  vLLM's `make_layers()` + `PPMissingLayer` mechanism makes this a
  significantly easier path than supporting in-instance PP > 1. If you
  want PP > 1 per instance + cross-instance reshard *on top of that*,
  the design gets a lot more complex.

**About the memory-freeing realization (Step 3.5)**:
- Q8.4: vLLM's block model is per-position, not per-layer — a block
  holds KV for *all* layers at one 16-token chunk. So after PP-reshard,
  SRC keeps the upper-layer KV slots allocated (just unused) for already
  generated tokens. The only "real" memory savings are for *future*
  tokens (HEAD doesn't allocate new blocks for upper layers because
  vLLM doesn't actually allocate per layer — see, this is where it gets
  confusing). The clean fix is per-layer block tables, which is a big
  vLLM change, deferred to Phase 6. **Are you OK with the demo claim
  shifting from "memory reclaim" to "preemption avoidance via load
  spreading"?** Or is full memory reclaim a hard requirement?

**About the iteration-level barrier (Step 1.7)**:
- Q8.5: Phase 1's MVP uses Ray RPC for cross-instance hidden-state
  shipping. That's slow (likely 1–10ms RTT for small models). Are you OK
  with that for the correctness-only MVP, with NCCL replacement coming
  in Phase 3? Or do you want to skip directly to NCCL?

**About the hybrid batching feasibility (Step 3.4)**:
- Q8.6: I propose Option B (separate sub-batches: one for full-layer
  requests, one for HEAD-only requests) over Option A (per-row layer
  mask within a single batch) because A is way too invasive to vLLM
  attention kernels. **Are you OK with two forward calls per step on
  instances that host both kinds of requests?** Throughput penalty
  expected.

**About the demo workload (D7 / Step 0.5)**:
- Q8.7: Do you have a target hardware for the demo? My plan assumes A10
  per the Llumnix paper. If we have H100s available, the OOM-avoidance
  demo gets harder (more memory headroom) but is still possible at
  longer contexts.

**About the manual-API trigger in Phase 1 (D6)**:
- Q8.8: My Phase 1 plan only adds a manual `pp_reshard_out` API; the
  global scheduler doesn't decide automatically until Phase 4. **Are
  you OK with that or would you prefer the scheduler integration to
  come earlier**, even at a basic threshold?

**About the AGENTS.md from upstream vllm**:
- Q8.9: vllm-latest's AGENTS.md prescribes uv-based env, pre-commit, etc.
  We won't be contributing back upstream (right?), so most of those rules
  don't apply. But adopting uv for the dev environment is reasonable
  hygiene. **OK to adopt uv for the venv?**

