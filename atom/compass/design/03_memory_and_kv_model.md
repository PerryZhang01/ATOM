# ATOM Compass — Design Topic 3: Memory Model and the KV Pool

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `02_model_runner_and_cost_backend.md` (the runner that must answer
`get_num_blocks` and `allocate_kv_cache`).

**Why this is in scope at all.** Memory decides which configurations exist. Predicting
the speed of a configuration that cannot start is answering the wrong question. It also
carries the **tightest acceptance gate in the project — KV capacity / block count within
5%**, against 10% for everything else.

---

## D13. What actually needs simulating in the KV path

### Problem

The project brief lists "KV cache pool, simulation of the real counterpart from ATOM
including caching abstraction and transfer". Read literally that is a substantial
subsystem. It is worth establishing how much of it must actually be built.

### Finding

**Almost none of it.** ATOM's KV management is pure Python arithmetic over integers,
with exactly one point of GPU contact.

| Component | File | Touches a GPU? |
|---|---|---|
| `BlockManager` — allocation policy, prefix hashing, state-slot attach, checkpoint ladder | `block_manager.py` (1698 lines) | **No** |
| `BlockPool` — free list, ref counts, content index, eviction, `reserve_units`/`retire_top` | `block_pool.py` (393 lines) | **No** |
| `Block` | `kv_block.py` | **No** |
| `StateSlotPool` — stateful-attention slots, fork/checkpoint index | `state_pool.py` (795 lines) | **No** |
| `PagedStateCheckpointCoordinator` | `page_unit_checkpoint.py` (577 lines) | **No** |
| `plan_pools` over `SubPoolSpec` | `model_ops/attentions/sub_pool_spec.py` | **No** |
| Prefix-cache hash (`compute_hash`, xxhash xxh64 chained with the parent) | `block_manager.py::BlockManager.compute_hash` | **No** |
| Prefix-cache scan and claim (`can_allocate`, `allocate`) | `block_manager.py::BlockManager.can_allocate`, `BlockManager.allocate` | **No** |
| Prefix publish, deferred until the forward computed the KV (`hash_blocks`) | `block_manager.py::BlockManager.hash_blocks` | **No** |
| **`allocate_kv_cache`** — creating the actual tensors | **`model_runner.py::ModelRunner.allocate_kv_cache`** | **Yes** |

`BlockManager.__init__` asserts `num_blocks > 0` (`block_manager.py`) and nothing else
about the device.

### Decision

**Do not simulate the KV pool. Run ATOM's real one.**

Two changes only:

1. `allocate_kv_cache` creates no tensor in the simulated runner. It passes an empty
   registry to ATOM's `set_kv_cache_data`, which builds the worker-side KV connector when
   a `compass` `kv_transfer_config` is set (#428).
2. `get_num_blocks` returns a block count produced from a device model rather than from
   device readings. See D14.

Everything else — prefix caching, chunked prefill, preemption, eviction, ref counting,
the hybrid PAGE/STATE split, GDN recurrent state, the V4 compressor ring, Eagle3 draft
KV merged onto the target's block ids by name — runs unmodified and is therefore
**correct by construction rather than by fidelity**.

### Consequences

- Prefix-cache behaviour in a simulated run is the real behaviour, provided the prompts
  presented to it actually share prefixes. That is a workload-harness problem, not a
  simulator problem, and it is not free: the corpus carries no prompt text, only
  per-64-token-block hash ids, so a prompt generator must reproduce the sharing topology.
  Deferred to topic `06`.

#### A cache hit changes the cost of the prefill, and this is where that is handled

A hit is not free and a longer hit is not the same as a shorter one. Three separate
things have to be true for that to come out right, and they live in three places.

**1. The batch already carries the hit — nothing needs to infer it.** ATOM resolves the
prefix match in the scheduler, *before* the forward. It sets `num_computed_tokens` from
the matched block count, and the `ScheduledBatch` that reaches the runner contains only
the **uncached** tokens as query. So for a request of 100k tokens with a 90k-token hit,
the runner is handed `N_Q = 10k` with `N_KV_cached = 90k` — which is what a real run
hands its kernels too. Compass reads the hit off the batch; it never re-derives it.
This is a direct consequence of D13: the real `BlockManager` and prefix index run, so
the hit *is* ATOM's hit, at ATOM's 64-token block granularity, including partial-block
truncation.

**2. The cost model must have a term that separates the two.** A model of the form
`a + b·tokens` cannot express this: it sees 10k tokens and prices them as if there were
no 90k of context behind them, when in fact every one of those 10k queries attends over
100k keys. That is why the prefill form in doc `02` D12 is

```
a + b·tokens + c·Σ_req N_Q²  + d·Σ_req (N_Q · N_KV_cached)
                ^^^^^^^^^^^     ^^^^^^^^^^^^^^^^^^^^^^^^^^
                self-attention  attention against cached context
                within the      — this is the cache-hit term
                new tokens
```

`d` is the cache-hit term. A 90k hit and a 10k hit on the same `N_Q` differ in `d ·
N_Q · ΔN_KV`, linearly, which is the right shape: the query block is fixed, the KV it
scans is not. ATOM already computes these sums itself — `detailed_sqsq`, `detailed_sqsk`
and `detailed_sk` on `scheduler.py::ScheduledBatch` are exactly `Σ N_Q²`, `Σ N_Q·N_KV` and
`Σ N_KV`, so the features are read from the batch rather than recomputed.

**3. The calibration must have *seen* high-hit steps, or the hull refuses.** This is
the consequence with teeth, and it belongs here because it is easy to miss. A
cache-hit step has a shape that a prefix-cache-disabled sweep **never produces**: small
`N_Q`, very large `N_KV_cached`, a ratio far off the `N_Q ≈ N_KV` diagonal that an
uncached chunked prefill walks. Calibrating with prefix caching off and predicting with
it on puts every hit step outside the convex hull of doc `09` D58, and the honest
outcome is a refusal on most of the workload.

So: **doc `07`'s Phase 0 discovery must run with prefix caching enabled**, matching the
cc-traces default, and the discovered shape set must be checked for coverage of the
high-`N_KV`/low-`N_Q` corner specifically — not just for coverage of `tokens` and
`batch` separately, which is precisely the per-feature bounding-box failure doc `09`
D58 spent three iterations abandoning.

**What is still not modelled:** the *lookup* cost. Matching a 100k-token prompt against
the prefix index is host work proportional to block count, and it is charged to nobody
— it currently falls inside the host floor of doc `07` Phase 1c as an unattributed
constant. At cc-traces' p50 input of 88,768 tokens that is ~1,387 blocks hashed and
probed per request. Whether that is 0.1 ms or 10 ms is unmeasured. Recorded as **T49**.
- `hash_blocks` is deliberately called from `Scheduler.postprocess` (`scheduler.py`)
  *after* the forward, because the real engine only publishes a prefix once the KV
  behind it exists. The simulated forward must not disturb that ordering.

### Open issues

- **Resolved by #428.** `model_runner.py::ModelRunner.allocate_kv_cache` also registers
  tensors globally via `set_kv_cache_data`. The Compass override makes the same call with
  an empty registry, so the registry is `{}` (`forward_context.py::set_kv_cache_data`), and no stand-in
  is needed, because nothing reads it under the Compass runner:
  - No module under `atom/compass` or `scripts/compass` reads it.
  - Outside `atom/plugin`, ATOM reads it in three kinds of place. `set_forward_context`
    copies it into each step's context, and is called from
    `model_runner.py::ModelRunner.prepare_inputs`, from graph capture and from
    `Drafter.warmup_draft_graphs`; `UBatchWrapper._make_ubatch_context` copies that
    again per micro-batch. Attention forwards read the copy:
    `PagedAttentionImpl.rope_cache` and `SparseMHAPagedAttentionImpl.rope_cache`
    (`attention_mha.py`), `MLAAttention.forward_impl` (`attention_mla.py`),
    `GatedDeltaNet.forward` (`attention_gdn.py`) and `KimiKDAAttention._forward_impl`
    (`models/kimi_k3.py`). And the drafter reads it outside any model forward, in
    `DSparkProposer._resolve_dtype_q` (`spec_decode/dspark_proposer.py`).
  - `atom/plugin` reads it only in plugin mode, where vLLM, SGLang or RTP-LLM runs ATOM's
    model code. The mode defaults to `"atom"`, which is not a plugin mode
    (`plugin/prepare.py`).
  - The Compass runner reaches none of these: its `forward` and `capture_cudagraph`
    replace ATOM's and run none of `prepare_inputs`, the model, the drafter or a
    capture. Its `_build_and_load_model` also sets the model to an `UnbuiltModel`,
    whose `forward` refuses, and refuses a speculative config.
- `BlockManager.hash_block_size = block_size * dcp_world_size`
  (`block_manager.py`) — decode context parallelism changes the hash granularity.
  Out of scope now; noted so it is not discovered later.

---

## D14. Producing the KV budget without a device

### Problem

`model_runner.py::ModelRunner.get_num_blocks` is **five device readings
plus arithmetic**:

```
# in _read_device_memory, which get_num_blocks calls before any budget arithmetic
free, total        = torch.cuda.mem_get_info()
peak_torch         = max(allocated_bytes.all.peak, .all.current)
non_torch          = max((total - free) - torch.cuda.memory_reserved(), 0)
# in get_num_blocks
cudagraph_overhead = self._estimate_cudagraph_overhead()
safety_margin      = int(total * 0.02)
budget             = int(total * config.gpu_memory_utilization)
available_for_kv   = min(budget - (peak_torch + non_torch + cudagraph_overhead
                                   + safety_margin)
                         - self._kv_budget_extra_reserve(total), free)
plan                = plan_pools(self._sub_pool_specs(), available_for_kv,
                                 config.max_num_seqs)
num_kvcache_blocks  = plan.paged_entries
# under PP: all_reduce MIN across stages
```

`_read_device_memory` is `model_runner.py::ModelRunner._read_device_memory`.

Consumed at `engine_core.py::EngineCore.__init__`, which sets `config.num_kvcache_blocks` before the
`Scheduler` and `BlockManager` are constructed.

Running on an MI308X while modelling an MI355X, the real readings describe the wrong card.

### Options

**A. Substitute the five readings; reuse ATOM's arithmetic.** Compute the readings from a
device model and model geometry, then let ATOM's own budget maths and `plan_pools` run
unmodified.

- *Pros:* the budget formula stays single-sourced. Copying it into Compass is exactly the
  drift the substitution avoids, and upstream will change it (`_kv_budget_extra_reserve`,
  the 2% margin, the clamp). `plan_pools` already gets hybrids right. Prior work reached
  **−0.09% on 27B TP=2** and −0.02/−0.02/+0.07% on 0.6B TP=1/2/4 this way.
- *Cons:* two of the five readings are tables, not laws (see below).

**B. Fully analytical budget owned by Compass.** Cleaner to reason about; a second copy of
a formula upstream will change, and a silent divergence moves the admission cliff with no
test failing.

**C. Replay recorded readings from a real run.** Highest fidelity where a record exists;
cannot size a configuration nobody has run, which is half the tool's purpose.

**D. Analytical with recorded readings as a validation oracle.** Most work; the only one
that makes transfer error measurable.

### Decision

**Option A.** The rule, inherited from the prior work and adopted deliberately:
**substitute the readings, never the arithmetic.**

Where each reading comes from:

| Reading | Source under the device model |
|---|---|
| `total` | the device spec's HBM capacity |
| `peak_torch` | weights + buffers + load residue + persistent + activations, each from the source D16 names |
| `non_torch` | a spec constant per width (D15) |
| `cudagraph_overhead` | mirror ATOM's own `_estimate_cudagraph_overhead`, because that is the number that actually reserves |
| `free` | derived as a **clean box**: `total - peak_torch - non_torch` |

### Two things the device model fixes structurally

- **`non_torch` stops being a device-wide reading.** It is `(total - free) - reserved`,
  and `total - free` counts every process on the card. Six prior runs died at start-up
  with `available_for_kv = -103667.58 MB (budget=57.60GB, peak_torch=2.94GB,
  non_torch=152.01GB, ...)` because a neighbour held 152 GB while this rank had reserved
  2.9 GB. Reading it from a spec removes the contamination by construction.
- **The `min(budget, free)` clamp becomes inert.** With `free` derived as a clean box it
  cannot bind. The prior design needed a `free_was_binding()` guard to refuse records
  where it had; that guard is unnecessary here.

### Declared scope boundary

Compass therefore models a **dedicated** device. It will not predict the OOM that a
shared box produces, and it will not reproduce a neighbour-induced admission cliff. That
is the right thing to model and it is stated here so it is not discovered as a gap.

### The pipeline minimum is not inert

An earlier note here said that under PP the `all_reduce(MIN)` across stages
(in `model_runner.py::ModelRunner.get_num_blocks`) is inert, *because every stage computes
the same number*, and that it *still needs a live process group or a stub*. Both halves are
wrong — and the engine's own comment on that reduce has said so all along:
*"PP stages compute different block counts; block ids must be valid on every stage's KV
tensor, so reduce to the global minimum."* The code said what this document denied.

**The stages do not compute the same number.** The five readings *are* identical across
stages — nothing in the memory model varies with pipeline rank. The layer count is not:
`model_runner.py::ModelRunner._get_total_num_layers` takes a `get_pp_indices` slice, so
each stage sizes its pool from the layers it actually holds. Re-derived from ATOM's own
partitioner at `feature/atomcompass_new` `92f1fdafe`, over the 64-layer hybrid vendored at
`tests/compass/qwen3_5_27b_config.json` — one full-attention layer in four, so 16 of the 64
hold paged KV — at block size 64, `max_num_seqs` 256, and a KV budget fixed at
200,000,000,000 bytes. **That budget is a chosen figure, not a sizing anyone runs at**: the
absolute counts below are illustrative, and only their movement *between* stages is
derived.

| PP | layers held per stage | paged layers per stage | distinct block counts |
|---|---|---|---|
| 2 | 32, 32 | 8, 8 | 1 |
| 3 | 21, 22, 21 | 5, 5, 6 | 2 — 152,587 / 152,587 / 127,156 |
| 4 | 16, 16, 16, 16 | 4, 4, 4, 4 | 1 |
| 5 | 13, 13, 13, 13, 12 | 3, 3, 3, 4, 3 | 2 |
| 6 | 10, 11, 11, 11, 11, 10 | 2, 3, 3, 2, 3, 3 | 2 |
| 7 | 9, 9, 9, 9, 9, 10, 9 | 2, 2, 2, 3, 2, 2, 3 | 2 |
| 8 | 8, 8, 8, 8, 8, 8, 8, 8 | 2, 2, 2, 2, 2, 2, 2, 2 | 1 |

5/5/6 is **two** distinct counts, not three — the two five-layer stages share one.

**What the counts turn on is the *paged* layer count, not the layer count.** A block costs
`paged_layers × block_size × bytes_per_token_per_layer`, so two stages agree exactly when
they hold the same number of paged layers — and an even division of *layers* does not give
one. The counterexample is inside the table: at pp = 6 stages 1 and 3 each hold 11 layers,
yet hold **3** and **2** paged ones, and their block counts differ. Layer division and
paged-layer division coincide at pp = 2, 4 and 8 here only because this model's
full-attention layers fall one in four; on a stack whose paged layers are spaced otherwise
they need not. Over the widths above the reduction binds at pp = 3, 5, 6 and 7.

**The block count a PP deployment gets is the minimum over stages**, set by whichever stage
holds the most paged layers — which is not in general the last one: at pp = 5 it is stage
3 of 5, and at pp = 6 it is four stages of the six.

**Neither a live process group nor a stub is needed.** The reduce is already guarded by
`torch.distributed.is_initialized()` (in `model_runner.py::ModelRunner.get_num_blocks`): with no group it does not
run, and with one it runs ATOM's own code unchanged. Building a stub for it is building
something nothing asks for.

`tests/compass/test_pipeline_minimum.py` re-derives the table and the widths sentence above
from `get_pp_indices` rather than restating them, and reads the guard by parsing the runner
rather than importing it.

### Open issues

- Under PP the guard means the stages agree on a count only when a process group is live.
  **Without one they do not converge at all**: each keeps its own `plan.paged_entries`, so
  a simulated pp = 3 run leaves stage 0 holding 152,587 blocks where the deployment it
  models runs 127,156 on every stage — and the comment cited above says why that is not
  harmless, since block ids must be valid on every stage's KV tensor. Whether a simulated
  PP run has a live group is `01` D1's question about topology and is left there; the
  divergence without one is not a question, and is recorded here.
- **Which width keys `non_torch` under DP-attention is open** (2026-10-10, D15): it waits
  on [#702](https://github.com/jgong5/ATOM/issues/702).
- `gpu_memory_utilization` here is a fraction of **total**, with the non-KV footprint
  subtracted afterwards — the vLLM convention, **not** TRT-LLM's. Comparing the resulting
  block count against a number produced under the other convention is wrong.
- Fragmentation is not modelled, is not planned, and is not modelled by anyone.

---

## D15. The device specification artifact

### Problem

The five readings need numbers that describe a target card and its software stack.
Some are physical (capacity); some belong to the runtime and driver (`non_torch`); some
belong to the libraries (AITER's registered allreduce pools). A model of a card you do
not own cannot measure any of them.

### What depends on what

| Term | Prior measurement | Depends on |
|---|---|---|
| `total` | — | the card. A spec number. |
| weights | exact via a meta build, −0.00/+0.00/−0.02/+0.01% at TP1/2/4/8 | model geometry + quantisation + the group widths it shards over |
| buffers (rotary tables) | recorded exactly; a formula matched the 0.6B and was **4x wrong on the 27B** (partial rotary) | model config |
| activations | liveness walk; held out at the warmup shape to **+0.0% at TP=1/2/4** | model + shape + width. Sourced as D16 says; the walk needs an op graph (topic `04`). |
| invisible scratch | 0.1 KB/token on the 0.6B, **39.6 KB/token on the 27B** | model. One fitted number, deliberately not per-operator; inside the measured activation coefficient (D16). |
| persistent | 118 MiB, flat in width | model |
| load residue | 1.1 MiB at TP1, **2069 MiB flat at TP2/4/8** | AITER `CustomAllreduce` 1 GiB pool + the two-stage kernel's. A *software* constant. |
| `non_torch` | 926 / 6906 / 7266 / 10704 MiB at width 1/2/4/8, +266 MiB model-dependent | HIP context + libraries + RCCL buffers. **"A table, not a law"** — no fixed-plus-per-peer form fits 5980/6340/9138 at widths 2/4/8. |
| graph pool | `91.1 MiB + 0.3033 MiB per captured token` at W=1; a flat **104 MiB** above W=1 (allocated delta byte-identical, 79,692,800, across three widths and three ladders) | capture ladder, width |

### Options considered

1. Per-field provenance tags on the spec, measured where possible and declared transfer
   otherwise.
2. A separate calibration table keyed by (device, ROCm build, topology) — which would make
   explicit that these constants belong to the **software** stack as much as the silicon.
3. Model `non_torch` analytically from its parts. Attempted in the prior work and failed —
   no fixed-plus-per-peer form fits. Open research, not a task.
4. Plain declared configuration, authored outside Compass.

### Decision

**Option 4: a device specification is an input artifact.** A user or a separate tool
authors it beforehand and passes it to Compass; Compass consumes it and does not derive
it.

It carries, in one object serving **both** the memory model and the cost model — the
project requires compute, memory size, bandwidth and interconnect all configurable and
none read from a runtime:

- HBM capacity
- peak compute and memory bandwidth
- interconnect topology, per-link bandwidth and latency
- the runtime/library memory constants above, per TP width
- **derate factors** bridging spec peak numbers and achievable ones — the standard
  roofline practice, since no kernel reaches spec peak

### The one honesty measure retained

**The full resolved spec is echoed into every run artifact.** One line, no machinery. It
means a wrong constant is at least visible beside the number it produced. Given the ≤5%
KV gate, a spec that cannot be recovered from the artifact makes an error unattributable.

### Open issues

- **A wrong constant is otherwise indistinguishable from a right one in the output.** This
  is the accepted cost of option 4. The mitigation is the echo above plus, eventually, a
  per-term comparison against a real run on any card that is available — which costs no
  GPU time, because every hardware run already prints the real breakdown.
- **The runtime constants are assumed to transfer across devices of one software
  generation, and not across software upgrades.** The reading behind the assumption: they
  depend on the ROCm/RCCL/AITER build more than on the die — the +5980 MiB at width > 1 is
  collective buffer sizing, and the 926 MiB at TP1 is HIP context plus libraries. Adopted
  as a working assumption rather than left open, and made *enforceable* by
  `software_pinned_to` (doc `05` D25 rule 3): reusing a spec across a stack change warns by
  default (`check_stack`) and refuses under `validate(strict=True)`, and moving constants
  across stacks, by a transfer or a merge, always refuses. Still untested across dies;
  recorded as **T50** and cheap to settle with one startup on a second card type.
- **`non_torch` and the load residue under DP-attention: which width keys them is open**
  (2026-10-10). Both tables are keyed by `tensor_parallel_size`, which `CoreManager`
  rewrites to 1 under DP-attention, yet every rank there sits in DP-wide and EP-wide
  groups whose device communicators aiter builds (`15` D90, Q4). At the nightly DeepSeek-V4-Pro
  DPA cell the two candidate keys differ by 10 GB per rank:
  `driver_and_collective_reserve_bytes` is 0.97 GB at width 1 and 11.2 GB at width 8
  ([#545](https://github.com/jgong5/ATOM/issues/545)). The answer waits on the spike
  [#702](https://github.com/jgong5/ATOM/issues/702).
- Three topologies of one model is interpolation, not a law. The prior work said so
  explicitly and could not get a third model because the box was offline.
- **The graph-pool width scaling rests on one point above W=1.** Context, since the line
  alone does not carry it: the measured form is `91.1 MiB + 0.3033 MiB per captured token`
  at W=1, and a **flat 104 MiB** at W=2, W=4 and W=8 — where the allocated delta was
  byte-identical (79,692,800) across three widths *and* three capture ladders. So "flat
  above W=1" is well supported as a *shape*; what rests on one point is the claim that the
  transition happens at W=2 rather than being a function that merely looks flat over the
  widths measured. A fourth width would not help; a second *model* would.
- The spec-authoring tool is specified in doc `05` D26 (probe tiers emitting fragments,
  `merge`/`validate`/`explain`) and scheduled as doc `07`'s Phase 2. It probes a real card
  where one is available and falls back to declared datasheet values plus a mandatory
  derate where one is not.

---

## D16. The non-KV memory terms

### Problem

Each non-KV term carries a **≤10%** acceptance gate, individually. A summed check does
not discharge it.

### The rule that this design adopts

**Validate per term, never as a sum.** A prior summed check reported +13.8%, which was
three errors two of which cancelled: weights over-counted by +0.280 GB (a tied head never
resident), activations compared at the wrong shape (−0.015 GB), and −0.084 GB of a
resident term nobody had noticed existed. **The largest single error was 25% of a term and
the sum said 13.8%.** `peak_torch` was subsequently split at source into three
non-subtractive readings: `parameter_bytes`, `weights_torch`, `current_torch`.

Validation costs no GPU time, because every hardware run already prints the real
breakdown.

### The source of each term

**Revised 2026-10-10** after the owner's rulings on
[#545](https://github.com/jgong5/ATOM/issues/545#issuecomment-6094838221) and
[#620](https://github.com/jgong5/ATOM/issues/620#issuecomment-6094839293) and in
[#704](https://github.com/jgong5/ATOM/issues/704), for `16` D95's stages 2 and 3. One
source per term; the bullets below say why each is what it is.

| Term | Source | Issue |
|---|---|---|
| weights | a meta build in the simulated runner's `_build_and_load_model`: the model class under `torch.device("meta")` with ATOM's own quantisation config and groups at the deployment's widths, bytes summed over unique storages, a tied `lm_head` added; the tree is dropped after counting | [#699](https://github.com/jgong5/ATOM/issues/699) |
| buffers | the declared formula over fields the model's `config.json` states, labelled declared, until a recording exists | [#701](https://github.com/jgong5/ATOM/issues/701) |
| activations | one coefficient per model and width, from a machine-spec entry keyed by architecture and width: a real run's `peak - current` after warmup over the warmup tokens, times the deployment's warmup tokens. A model with no entry takes a geometry form, labelled declared. The liveness walk replaces both once capture lands | [#704](https://github.com/jgong5/ATOM/issues/704), [#705](https://github.com/jgong5/ATOM/issues/705) |
| invisible scratch | inside the activation coefficient; a term of its own only under the liveness walk (`04` T4) | [#704](https://github.com/jgong5/ATOM/issues/704) |
| persistent | the spec constant `persistent_forward_buffer_bytes` | — |
| load residue, `non_torch` | the spec's width tables; the width under DP-attention is open (D15) | [#702](https://github.com/jgong5/ATOM/issues/702) |
| graph pool | `graph_pool.reserves()` for the deployment's own `enforce_eager`; when it captures, the runner's `_piecewise_cg_active()` picks the branch, whole-graph from the activations term, piecewise from `piecewise_per_token_bytes` and `capture_token_shapes` over the deployment's ladder | [#700](https://github.com/jgong5/ATOM/issues/700) |

**The per-term check** is `memory/compare.py` with `graph_pool.predicts`: each predicted
term against the same term of a recorded real run, wired for stage 2's paired runs
([#620](https://github.com/jgong5/ATOM/issues/620)).

### Term-by-term approach

- **Weights** — ask a meta build, deduped by storage. Exact at every width on both models
  tested. Two prerequisites: dtype comes from the **config**, or every byte is twice what
  the model holds; and the width must be *simulated*, or asking for a world of two from
  one process hangs. One known correction: a meta-built model has not been through the
  loader, so a **tied `lm_head` is invisible** — worth one embedding, 0.290 GiB on the
  0.6B, which was the whole of a gap once. A declared `parameter_count × dtype / TP`
  is not a substitute: one dtype cannot state FP4 experts with e8m0 scales beside FP8
  and bf16 parameters, and under DP-attention TP is 1 while the experts shard over the
  DP group, so at the nightly DeepSeek-V4-Pro DPA cell it was refused at 10.76x the card
  (#545).
- **Buffers** — the declared formula over fields `config.json` states, labelled declared,
  until a recording exists (`ModelTerms.from_declared_config`,
  [#701](https://github.com/jgong5/ATOM/issues/701)). The label is there because the
  first formula, which matched the 0.6B exactly, was 4x wrong on the 27B (partial rotary).
- **Activations** — until capture lands, the coefficient measured on a real run (the
  table above): the scaling in tokens is validated (`10` D67), and the reading is the one
  ATOM's `_estimate_cudagraph_overhead` takes, so the scratch is inside it. After that, a
  def-use liveness walk over a traced op graph, **not** a footprint sum, which depends on
  topic `04` (model capture). Four faults had
  to be fixed before the walk worked, and they are worth restating because each is a trap:
  deaths inferred from the last read are the wrong event (a residual held across a block
  outlives every read of it — use `weakref.finalize` on the allocator); an operator's
  outputs need not share a life (fused add-and-norm returns one tensor that dies into the
  next gemm and one that carries to the end of the block — one death for the pair held an
  extra tensor per layer, **36% of the term at TP=4**); an address map must forget, since
  the allocator hands a freed address straight back; and `torch.empty` inside a custom
  operator never reaches a dispatch tracer, yet the MLP's silu destination is **13.6 MB a
  layer at TP=1 and is exactly where the high-water mark sits**.
- **Invisible scratch** — inside the measured activation coefficient; under the walk, one
  fitted number of bytes per token, clamped at zero. Explicitly
  *not* per-operator: a per-operator correction reproduces the traced curve exactly, is
  worth nothing at any other shape, and is a recording dressed as a model. On the 27B this
  term is the difference between −35.0% and +3.4% held out.
- **Graph pool** — keep **two** functions deliberately. One mirrors ATOM's own estimator,
  because that is the number that actually reserves the memory; the other predicts the
  real cost. They disagree by 4-19x. ATOM's estimator is `0.2 x peak activations` of the
  *warmup* shape and is blind to the capture ladder — the pool moves over 4x across
  ladders while the estimate does not move at all. Under-reserving is not an OOM: the
  capture loop re-checks free memory per bucket and silently skips what will not fit, so
  the price is **dropped buckets and a decode cliff at those batch sizes**. On a 192 GB
  card nothing was ever dropped, which is why this went unnoticed. The mirror reserves
  for the deployment's own capture mode: installing the eager reading for every
  deployment refuses one that captures, and `--enforce-eager` around that overstates the
  KV pool against a real run that captures (#545, #700).

### The gate that actually matters

Owned by doc `08` D48, not restated here: the gate is not the byte error but whether the
top-1 configuration choice survives. What belongs in *this* document is the consequence
for the model — which is the per-term rule above, not the aggregate.

### Open issues

- Activations are not blocked on topic `04`: until an op graph exists they come from the
  measured coefficient, and a model with no entry takes a labelled geometry form, whose
  error against the coefficient is what #705 measures (`10` T34). **Cross-check against
  `04`, for when the walk lands:** the liveness walk is `04` D22, which resolves observationally
  under fake tensors and treats an opaque leaf's internal scratch as a *declared per-leaf
  constant* rather than a walked one. So this term is not simply "blocked on a graph" —
  it is blocked on a graph **plus** the scratch constants of `04` T4, and the second is
  the one with no law behind it. The measured spread that makes it load-bearing:
  invisible scratch is **0.1 KB/token on the 0.6B and 39.6 KB/token on the 27B** (doc
  `10` D67), i.e. the difference between −35.0% and +3.4% held out. A graph without the
  scratch table does not discharge the ≤10% gate on this term.
- The model generalises to a shape the trace was not taken at, but **not to a model that
  was never traced** — a hybrid needs a real graph. So "size a configuration nobody has
  run" holds for shape and width, not for architecture. The measured coefficient narrows
  that further: it holds for shape, not for a model or a width with no run, which take
  the geometry form.
- A tracing run must reset the allocator's high-water mark around the step it writes a
  graph for, or `peak_torch` belongs to the warmup prefill, whose shape is nobody's
  choice (−14.7% on one configuration).

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D13 | Do not simulate the KV pool. Stub `allocate_kv_cache`; ATOM's real `BlockManager`, `BlockPool`, prefix index and `plan_pools` run unmodified. | 2026-09-18 |
| D14 | Substitute the five device readings, reuse ATOM's budget arithmetic and `plan_pools`. Compass models a dedicated device; that is a declared scope boundary. | 2026-09-18 |
| D15 | The device specification is an input artifact authored outside Compass, carrying capacity, compute, bandwidth, interconnect, runtime memory constants and derate factors. It is echoed into every run artifact. | 2026-09-18 |
| D16 | Validate every non-KV memory term individually, never as a sum. Keep ATOM's graph-pool estimator and the measured one as two separate functions. Revised: one source per term. Weights come from a meta build in the simulated runner at the deployment's group widths. Activations come from a coefficient measured per model and width, with the scratch inside it and a geometry form as fallback, until the liveness walk lands. The graph pool reserves for the deployment's own capture mode, and the per-term check is wired. | 2026-09-18; revised 2026-10-10 |
