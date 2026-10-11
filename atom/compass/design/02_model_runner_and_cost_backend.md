# ATOM Compass — Design Topic 2: Model Runner Seam and Cost Backend

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `01_execution_and_time_model.md` (the clock protocol supplies the
`advance_to` that consumes this document's output).

**Scope.** Where Compass attaches to ATOM, what replaces the forward pass, and the
interface between that replacement and whatever predicts a duration. It does **not**
cover how a *real* cost is derived (operator capture, pricing, analytical models) — that
is topic `04` — nor memory sizing, which is topic `03`.

---

## D10. The attachment point

### Problem

Compass replaces only the forward pass. The attachment must be narrow enough that the
scheduler, block manager, admission logic and prefix cache all run unchanged, and cheap
enough that it does not become a maintenance burden against upstream ATOM.

### Candidate cut points

| Cut | Where | Replaces | Verdict |
|---|---|---|---|
| `ModelRunner.run_model()` | `model_runner.py::ModelRunner.run_model` | `model(input_ids, positions)` + `compute_logits` | **Rejected.** Returns `(logits, hidden_states)` as real tensors that `postprocess` indexes, samples and gathers logprobs from. Faking it still requires a `[bs, vocab]` device allocation. |
| `ModelRunner.forward()` | `model_runner.py::ModelRunner.forward` | the whole forward, sampling included | **Chosen.** Its contract is `ScheduledBatch` in, `ScheduledBatchOutput` out — both pure numpy/list/int, already pickled across the worker boundary. |
| whole-runner replacement via `Config.runner_qualname` | `config.py::Config.runner_qualname` | weights, KV tensors, CUDA graphs, sampling | **Chosen as the delivery mechanism** for the above. |

### Decision

**Subclass `ModelRunner` and inject it with `--runner-qualname`.**

`Config.runner_qualname` (`atom/config.py`) is consumed at `engine_core.py::EngineCore.__init__`
and `async_proc.py::AsyncIOProc.__init__`. It already has two in-tree users —
`atom/rollout/async_engine.py::AsyncLLMEngine.__init__` injects `RLHFModelRunner`, and `Config.__post_init__`
(`config.py`) swaps in `RapidServeModelRunner` automatically. **The injection
itself requires no ATOM change.**

`model_runner.py::RapidServeModelRunner` is a working template for a
non-allocating runner already in the tree. It overrides exactly the memory-owning
methods: `_build_and_load_model`, `_maybe_warmup`, `_kv_budget_extra_reserve`,
`get_num_blocks`, `allocate_kv_cache` and `forward` — and constructs parameters on meta
through `model_runner.py::RapidServeModelRunner._init_weight_params_on_meta`.

### The RPC surface that must be honoured

`AsyncIOProc.busy_loop` dispatches by name (`async_proc.py`). A replacement
runner must answer all of: `get_num_blocks`, `allocate_kv_cache`, `capture_cudagraph`,
`forward`, `dummy_execution`, `exit`, `freeze_gc_heap`, `process_kvconnector_output`,
`async_proc_aggregation`, `start_profiler`, `stop_profiler`, `flush_pp_send`.

**Return contracts are load-bearing across a process boundary.** `engine_core` calls
`capture_cudagraph` with `wait_out=True` and unpacks three values — in the parent, at
`engine_core.py::EngineCore.__init__`, not in the worker. The two ways that contract breaks do not fail
alike. A reply of the wrong **shape** arrives and raises where it is unpacked,
in-process and with a traceback. A reply that never **arrives** queues nothing: a name
the runner does not define is skipped by `busy_loop`, a method that answers `None` is
called and its result declined, and `call_func`'s `outputs_queue.get()` takes no
timeout either way. That silence parks somebody only where somebody is waiting: ten of
the twelve names above have a caller that waits for the reply, and `exit` and
`process_kvconnector_output` do not. `atom/compass/runner/overrides.py` carries the
third case — a method that raises — and the table of which names wait.

**`flush_pp_send` keeps ATOM's own answer, and the control-command worker methods are
refused** (revised 2026-09-28 and 2026-09-30, #443):

- `flush_pp_send` is **answered by ATOM's own, at once**: `ModelRunner.flush_pp_send`
  returns `True` after waiting on any pending `isend`, and a simulated runner posts none.
  The stage loop settles the pending send before the call
  ([`01` D4](01_execution_and_time_model.md#d4-the-interception-contract-which-waits-must-be-touched),
  K5; [`15` D91](15_parallelism_support.md#d91-pp--the-only-strategy-that-adds-logical-processes),
  Q3). The runner gains the two methods that loop calls to move one `pp_data` or `pp_ack`
  frame over the PP CPU group; only the stage loop's hooks call them, never ATOM.
- The runner defines and refuses by name the worker methods the profiler and RL
  control commands reach (`start_profiler`, `stop_profiler`, `update_weights`,
  `update_weights_from_shm`, `update_weights_from_ipc`, `release_memory`,
  `resume_memory`, `clear_kv_cache`, `configure_hidden_states`), as
  [`01` D5](01_execution_and_time_model.md#d5-timers-clock-reads-and-control-commands)
  decides.

### Three semantics `forward()` must reproduce

1. **Deferred output.** `tokenIDProcessor.is_deferred_out` is True by default: the tokens
   returned for step *N* belong to step *N-1*. `Scheduler.postprocess` depends on it
   (`scheduler.py`). Returning the current batch's tokens with
   `is_deferred_out=False` offers them one step early to a loop that cannot yet see the
   sequence, and **never offers them again until a decode batch happens to include the
   request** — on the prior 27B run every request whose prefill completed inside a
   36-step prefill streak waited for step 37.
2. **The deferral unit is one *meaningful* step, not one step.** `ModelRunner.forward`
   returns early for a pure middle chunk of a chunked prefill (`produces_output()` false,
   nothing sampled) with `is_deferred_out` unset. On the 27B that is **4,354 of 4,440
   `postprocess` calls** that defer. Mirroring `is_pure_middle_chunk` reproduced the lag
   step for step (mean 8.99 s vs a real 9.00 s).
3. **`produces_output()`** (`scheduler.py::ScheduledBatch.produces_output`): a pure-middle-chunk prefill batch
   must return an **empty** `token_ids` list with the same `req_ids`, mirroring
   the early `return ScheduledBatchOutput(...)` in
   `model_runner.py::ModelRunner.forward`.

Speculative decoding adds `num_rejected` / `num_bonus` sized `batch.total_seqs_num` and
`draft_token_ids` shaped `[bs, mtp_k]`. ATOM's existing `synthetic_acceptance_rates` path
(`config.py::SpeculativeConfig`, `rejection_sampler.py::rejection_sample`) is the model to copy — it already
forces a chosen acceptance curve instead of computing one.

### Open issues

- `--runner-qualname` exists on `Config` but has **no `EngineArgs` field and no CLI
  flag**. Adding one is the 3-touch recipe (`arg_utils.py` field, `add_cli_args`,
  `Config` field); `_get_engine_kwargs` forwards by name and `LLMEngine.__init__` filters
  by `fields(Config)`, so name-matching suffices.
- `ModelRunner.__init__` warms the model before returning, and warmup drives a forward,
  so anything mode-dependent is consulted **before** a subclass's `__init__` body has run.
  Config must resolve lazily from `self.config`.
- The deferral *unit* is empirical. "One meaningful step" reproduced the 27B exactly, but
  the underlying rule is not derived. Prior work left this open explicitly.

---

## D10.1. Bringing a model into existence without a device

### Problem

Compass needs a **module tree** for weight geometry (the memory model's Class-A term) and
for tracing (`04` D18 walks ATOM's real model code). Neither may read checkpoint bytes
onto a device or allocate weights there. ATOM already has every piece required; what is
missing is a statement of which combination Compass uses, and when.

### What ATOM already provides

| Piece | Where | What it does |
|---|---|---|
| `--load_dummy {empty,zero,xavier}` | `config.py::Config.load_dummy`, `arg_utils.py::EngineArgs.load_dummy`, `EngineArgs.add_cli_args`, `loader.py::load_model_in_plugin_mode`, `load_model` | skips the checkpoint read; `empty` leaves params uninitialised, the others fill them with finite values in place |
| `_init_weight_params_on_meta` | `model_runner.py::RapidServeModelRunner._init_weight_params_on_meta` | wraps `Module.register_parameter` so every `nn.Parameter` is replaced by a meta tensor as it is registered |
| `no_init_weights` | `models/utils.py::no_init_weights` | uses `torch.device("meta")` as a **context manager**, so construction itself lands on meta - no transient, no GPU, and it covers buffers. **Currently unused in ATOM.** |
| `RapidServeModelRunner._build_and_load_model` | `model_runner.py::RapidServeModelRunner._build_and_load_model` | the override point where a runner declines to load |

#### Why `_init_weight_params_on_meta` allocates on the real device, despite its name

The name describes the **end state** — every `nn.Parameter` ends up on meta — not the
mechanism. The mechanism is a hook on `Module.register_parameter`, and by the time that
hook runs the parameter **already exists**: `Module.__setattr__` constructs
`nn.Parameter(torch.empty(...))` first and registers it second. The hook can only replace
what it is handed.

So construction allocates each parameter on the current default device, and the hook
immediately swaps in a meta tensor and drops the original. Its own docstring is precise
about this: *"Each parameter is briefly created on the real device then replaced with a
meta tensor, so the transient peak is one parameter, not the whole model."*

**It has to be that way, and the docstring says why:** the helper deliberately leaves the
default device unchanged *"so init code that explicitly targets CUDA (e.g. aiter RoPE) and
buffers work normally."* Setting the default device to meta would put **buffers** on meta
too — and buffers are not parameters, so the hook never sees them — and would change the
branch taken by init code that targets CUDA explicitly. For its actual use case,
disaggregated decode, both of those must stay real: it fills parameters from prefill over
CUDA IPC and recomputes RoPE caches locally.

**So it is not a bug.** It is a deliberate trade — a one-parameter transient bought in
exchange for real buffers and unchanged init branches — and for its use case the trade is
clearly right. It avoids what the call site calls *"the transient 2x-weights peak that
OOMs at TP=4"* (in `model_runner.py::RapidServeModelRunner._build_and_load_model`), which is the thing that mattered there.

Worth noting for anyone reading that code: the call site says construction *"allocates
zero GPU bytes"* while the helper says one parameter is transiently real. Both are true of
different quantities — zero **persistent**, one parameter **transient**. The docstring is
the precise one.

#### The transient is avoidable, and ATOM already ships the mechanism

There is a second way, and it is strictly better than the `register_parameter` hook on
both counts. **ATOM already has it**, at `atom/models/utils.py::no_init_weights`:

```python
with register_module_module_registration_hook(hook), torch.device("meta"):
    yield
```

`torch.device("meta")` as a **context manager** (torch ≥ 2.0) sets the default device, so
it intercepts **construction** rather than registration. Verified on this stack (torch
2.10.0+rocm7.2.4): `with torch.device("meta"): nn.Linear(4, 4)` yields
`weight.device == meta`. Two consequences:

| | `_init_weight_params_on_meta` | `torch.device("meta")` context |
|---|---|---|
| Transient real allocation | one parameter | **none** |
| Requires a GPU to exist | **yes** | **no** |
| Covers buffers | no — the hook only sees parameters | **yes**, everything a constructor makes |

The buffer row is the one that matters for Compass. A model that allocates a rotary table
or an attention mask in `__init__` allocates it for real under the hook and on meta under
the context.

**Provenance, since this came up as "Transformers does this":** the mechanism is the same
one `accelerate.init_empty_weights` provides — accelerate reaches it by patching
`torch.empty` / `zeros` / `ones` / `full` on older torch, and by this same device context
on torch ≥ 2.0. Checked here: **accelerate is not installed and is not an ATOM dependency**
(`pyproject.toml` lists `transformers==5.12.1` and no accelerate), and transformers 5.12
does not export `no_init_weights` / `init_empty_weights` from `modeling_utils`. So the
in-tree `atom/models/utils.py` version *is* ATOM's own form of it, and reusing it needs no
new dependency.

**It is currently unused** — grep finds no caller. Compass would be its first consumer.

One caveat on reusing the function rather than the mechanism: `no_init_weights` is shaped
for a different job — it takes a `placeholder` callable and swaps submodules out. The part
Compass wants is the `torch.device("meta")` context, which is one line. Calling ATOM's
function would mean supplying a placeholder we do not want.

#### Why Compass prefers `FakeTensorMode` anyway, and where the meta path is still used

Design principle 1 says reuse ATOM's mechanisms, so the bar for not reusing this one has
to be more than a few hundred megabytes of transient.

The meta context solves the transient and the GPU requirement, so the remaining reason is
neither of those. It is the one `04` D18 already gives:

> **Meta has no symbolic shapes, and `device.type` is `'meta'`.**

Both matter, and both are measured:

- **No `ShapeEnv`.** `ShapeEnv` attaches to `FakeTensorMode`, not to a device context. Without
  it there are no symbolic shapes, which puts capture back on the shape-synthesis route
  that mispredicted **3 of 11 dimensions** at a held-out point — including a ceil-division
  **9.5x off**, which two sample points straddling a block boundary hide entirely.
- **Wrong device branch.** ATOM registers its ops at `dispatch_key="CUDA"`
  (`atom/utils/custom_register.py::direct_register_custom_op`) and the model and runner branch on device
  throughout. Under meta that code takes paths nobody runs — and *"meta accepts kernels
  real devices reject"*: AITER's fused qk-rmsnorm takes fp16/bf16 only, and meta traced it
  happily at fp32.

So the split is by **what the caller needs**, not by which mechanism is cheapest:

| Need | Path |
|---|---|
| geometry only | HF config, no module tree |
| **a module tree, no tracing** — parameter enumeration, a structural walk | **`torch.device("meta")` context**, no GPU, no transient |
| **a module tree for tracing** | **`FakeTensorMode`** — the only one with symbolic shapes and the right device branch |

**What is reused in all three:** `--load_dummy empty`, which is how the checkpoint read is
skipped. That is the mechanism doing the real work, and Compass does not replace it.

**The T5 fallback is the meta context, not the hook.** If `FakeTensorMode` construction
turns out not to work on ATOM's model classes, `--load_dummy empty` plus
`torch.device("meta")` gives a weight-free, GPU-free module tree — losing symbolic shapes,
so capture would fall back to concrete traces at the shapes that actually occur (`04`
option C), not to shape synthesis. That is a real degradation and an escalation —
one of `16`'s named escalation points — not a silent substitution.

### Decision

**Two paths, chosen by what the caller needs. Neither reads weights.**

| Need | Path | Device touched |
|---|---|---|
| **geometry only** — M1's fake model, configuration sweeps | **HF config, no module tree at all.** | none |
| **a module tree, no tracing** - parameter enumeration, the weight-bytes term, a structural walk | **`torch.device("meta")` context** with `--load_dummy empty`. ATOM already has this shape at `models/utils.py::no_init_weights`. Weight bytes are Class A, exact from a meta build deduped by storage: measured **-0.00 / +0.00 / -0.02 / +0.01%** at TP 1/2/4/8 (`10` D63); the build runs in the simulated runner (`03` D16, revised 2026-10-10). | none |
| **a module tree for tracing** - tier b, the liveness walk, structure discovery | **Construct the model inside `FakeTensorMode`**, with `--load_dummy empty` so no checkpoint is read. | none |

The second is not an addition to `04` D18 — it *is* D18, stated from the construction side.
Building under the mode means every parameter is a `FakeTensor` at creation, so there is no
transient at all and the `register_parameter` wrapper is unnecessary.

**Why `FakeTensorMode` and not a meta default device**, restating `04` D18's reason in this
context: ATOM's init code explicitly targets CUDA — the docstring above names aiter RoPE —
and buffers are constructed rather than registered as parameters. Under a meta default
those paths take branches nobody runs; under `FakeTensorMode` they produce `FakeTensor`s
carrying a `cuda` device tag and take the real branch.

### Two hazards this inherits

1. **`FakeTensorMode.__enter__` probes the driver** to choose the fake device, so model
   construction can hang on a wedged node despite touching no GPU. Same caveat as `07`
   Phase 1a: portable, not hermetic. The 30-second `rocminfo` check comes first.
2. **Buffers are not parameters.** `--load_dummy` and the meta wrapper both act on
   parameters; a model that allocates a large buffer in `__init__` — a rotary table, an
   attention mask — allocates it for real unless the whole construction happens inside the
   mode. That is the concrete argument for constructing *under* the mode rather than
   wrapping `register_parameter`. Recorded as **T68**: enumerate buffer allocations in the
   two target models and confirm none escapes the mode.

### What this does not decide

Whether a **quantized** checkpoint's geometry is derivable without reading it. Quantized
weight bytes depend on the on-disk layout, and `10` D63 records that checkpoint-header
derivation came out **3.3% low at TP=4** on the hybrid 27B. Out of scope for M1-M3;
recorded as **T69**.
## D11. No modes on the runner; the algorithm comes from a backend

### Problem

A runner with modes — `trace`, `measure`, `predict` — puts a mode check in front of every
override, and makes "which mode produced this artifact" a question that has to be tracked
out-of-band.

### Decision

Per the project's own design note: **the runner has no modes. Every run is a simulation.
The algorithm that produces a duration comes from a modelling backend.**

Two orthogonal flags sit beside it, neither a mode of the runner:

- **`measure`** — additionally perform a real GPU run and cache the measurements for
  later simulation.
- **`trace`** — performed on demand on first use and cached; not something a user selects.

#### Why these two are asymmetric, and how they line up with `compass plan`

The asymmetry is not stylistic. It follows from one property:

> **Tracing is device-free; measuring is not.**

Capture runs under `FakeTensorMode` (doc `04` D18) and allocates nothing on a GPU, so it
can happen *inside a simulated run* at the moment a structure is first asked for.
Measuring runs kernels, so it can never happen inside a simulated run — a simulated run
is defined by not touching a device.

That gives each flag exactly one legitimate place, and the two must agree with doc `07`'s
phases rather than describe a second workflow:

| | `trace` | `measure` |
|---|---|---|
| Needs a device | no | **yes** |
| Offline home | doc `07` **Phase 1a** — trace the structures Phase 0 discovered | doc `07` **Phase 1b/1c/2** — op pricing, in-situ steps, memory constants |
| Allowed during a simulated run | **yes**, lazily, on a structure miss | **never** |
| Who triggers it | the runner, automatically | the operator, via `compass plan`'s emitted commands |
| Artifact written | the IR graph for that structure key | a price / region / memory-constant entry |

**Recommended path is offline for both.** `compass plan` (doc `07` D37) emits the whole
campaign, Phase 1a included; a run that finds every structure already traced does no
lazy work at all, which is also the only way a run is reproducible from its artifacts.

**The lazy path is a fallback with a declared cost, not a convenience.** It exists
because a structure set is discovered, not enumerated (doc `07` Phase 0 is a
best-effort over-cover), and refusing a whole run because one unforeseen shape appeared
is worse than tracing it. Its costs, stated so nobody is surprised:

1. It consumes **wall-clock inside a simulated run** — real seconds that do not
   correspond to virtual time. Harmless to the predicted numbers (the clock does not
   advance during it) but it degrades the ≥5x speed result, so the run artifact records
   lazy-trace count and seconds separately.
2. A newly traced structure has **no price**. The graph exists; the leaves in it may
   not. The backend then refuses that step (per the rule below) rather than guessing.
   So a lazy trace converts "unknown shape" into a *named* refusal with a graph
   attached — which is exactly the input `compass plan` needs to extend the campaign.

**Consequence worth stating plainly:** a lazy trace never rescues a run on its own. It
makes the gap diagnosable. `measure` closes it, offline, on a machine with a card.

### The backend interface

```
CostBackend:
    estimate(batch_view) -> StepCost          # seconds, plus a breakdown
    describe() -> str                          # provenance, for the artifact
```

`batch_view` is a projection of `ScheduledBatch`, not the object itself, so the backend
stays free of ATOM imports and is testable on a laptop. The prior work enforced exactly
this separation (`atom.compass.core` was forbidden from importing anything from ATOM) and
it paid for itself.

`StepCost` carries `seconds` **and a breakdown**. The breakdown is not decoration: every
serious error in the prior effort was found by decomposing a total, and at least six
times an aggregate hid compensating errors — a +13.8% memory sum that was three errors
two of which cancelled; a cc-traces latency within 5% that was +52% TTFT against −30%
decode; a prefill total of +0.04% that was −4.01% pure against +27.28% mixed.

### Provenance vocabulary, carried on every backend

Adopted from the prior work because it prevented a recurring category error:

- `analytical` — computed without measuring the subject. **Nothing was analytical in the
  prior PoC.** The word is reserved, not aspirational.
- `empirical/measured` — the unit itself was timed
- `empirical/fitted` — a form chosen, coefficients regressed over measured steps
- `empirical/interpolated` — no form assumed, nearby measurements looked up
- `empirical/extrapolated` — asked outside the measured range

Two rules: **"priced" is not a species** (a price is `measured`, differing only in unit —
say `measured (op-level)` vs `measured (step-level)`); and **reading a measurement back
from a file is still `measured`** — what changes is whether the key matched exactly
(measured), nearest (interpolated) or outside (extrapolated).

### Open issues

- Whether `measure` and `trace` are CLI flags, env vars, or both. Settled in part by the
  table above: `measure` is an operator-facing switch and belongs on the command line;
  `trace` needs at most a *disable* (`--compass-no-lazy-trace`) for runs that want a
  refusal instead of a stall. The full flag surface is a gap — see `12_open_items.md`,
  "Missing topics".
- A backend asked about a step kind it has no samples of must **raise**, not fall back.
  The prior fallback was the mean of an empty list, i.e. zero — "a confident, precise,
  entirely fictional answer", a TTFT of 0 ms against a real 7.6 s.
- What a refusal *does to the run* is not settled here and is not local to this document:
  abort the run, or mark the step and continue with `provenance=refused`? Five documents
  emit refusals and none of them says. Recorded as **T48**.

---

## D12. The milestone-1 fake model

### Problem

Milestone 1 requires "fake models that can simulate all a real model can do — prefill,
decode, KV cache needs, TP/DP/PP/EP". Read as a modelling requirement this is a large
task. Read correctly it is a **test double**: the cheapest thing that produces a plausible
duration and a correctly-shaped `ScheduledBatchOutput`, so that the time model, the
simulated KV management and the scheduler integration become observable **before any real
cost model exists**.

### What "simulate all a real model can do" actually requires

From the engine's point of view the fake model must be *interface-indistinguishable*, not
*accurate*. Concretely it must supply:

1. **KV geometry** — enough that `get_num_blocks` returns a real number, `plan_pools`
   over `SubPoolSpec` runs for real, and `BlockManager.__init__`'s `assert num_blocks > 0`
   (`block_manager.py`) passes. Derived from the HF config: layer count, KV head count,
   head dim, KV dtype, block size.
2. **A step duration** from the cost backend.
3. **A correctly-shaped output**, honouring the three semantics of D10.

Parallelism support then collapses to arithmetic on the geometry — which dimension
divides by what:

| Axis | Effect on the fake model |
|---|---|
| TP | KV heads shard (GQA-bounded) so KV bytes/token divides; weights shard; collectives appear as a cost term |
| DP | N independent engines, each holding full KV |
| PP | layers split per stage, so KV per stage; and it creates the LP structure of topic `01` |
| EP | experts shard; KV unaffected; an all-to-all appears as a cost term **only at `dp_size > 1`** — at DP 1 ATOM builds no all-to-all at all and the MoE is masked local compute plus the TP all-reduce (`15` D92) |

This is on the order of thirty lines, and the point is that **the rest of the system then
does genuine work**: the real `BlockManager` gets a genuinely different block count per
TP width, and the Clock Authority gets a genuinely different LP count per PP depth.

### Options for the cost backend at M1

**A. Two constants** (`prefill_seconds`, `decode_seconds`). take2's `ConstantCostOracle`,
about 40 lines.

- *Pros:* absolutely minimal; proves the plumbing.
- *Cons:* a fixed-shape oracle hides every shape-dependent bug, and you cannot tell
  whether the scheduler is reacting to the predictions at all. Measured directly in the
  prior work: an offline workload has one shape per kind, so a one-graph-per-kind oracle
  looked exact — and then cost **+47.9% TTFT** the moment shapes varied under real
  serving (3 prefill steps of 16/256/16128 tokens costing 38.6/151.0/177.6 ms, all
  answered as ~47 ms).

**B. Shape-analytic stub.** A declared linear form over fields `ScheduledBatch` already
carries.

- *Pros:* about twenty lines more than A, and it buys a **testable integration property**:
  double the chunk size and every term of the form moves by its own power of the chunk,
  so the step time rises and the prefill streak breaks at a predicted place. That closed
  loop — step cost to queueing to a different batch — is what the whole design rests on,
  and constants cannot exercise it.
  - *Not* "the step time doubles". Measured against the form this section mandates, at
    `a`=2e-5, `b`=5e-7, `c`=2e-11 and a 256→512 single-request *first* chunk — so the
    cross term is zero and `d` does not enter: the step goes **149.31 µs → 281.24 µs**,
    the intercept 1.00x, the token term exactly 2.00x, the quadratic exactly 4.00x, and
    the total **1.8836x**.
    The total doubles only when `a = c = d = 0`, which is the degenerate linear case —
    i.e. only by deleting the quadratic term this same section requires. The property
    that holds for the mandated form is term by term, and the exit criterion of any task
    building this option must be written that way.
- *Cons:* its coefficients mean nothing physically, so it must never be quoted as
  accuracy.

**C. Roofline from the geometry** — FLOPs/compute + bytes/bandwidth.

- *Pros:* physically motivated; uses the configurable compute, bandwidth and interconnect
  values the project requires anyway.
- *Cons:* this is the milestone-2+ analytical backend arriving early. More work than M1
  needs, and it invites judging M1 on accuracy when M1 is about plumbing.

### Decision

**Option B, with a constant mode retained for first bring-up.**

The form, including the quadratic query term:

```
prefill:  a  + b ·tokens + c ·Sum_req N_Q^2 + d ·Sum_req (N_Q * N_KV_cached)
decode:   a' + b'·batch  + c'·Sum ctx       + e'·( rung * max(ctx) - Sum ctx )
```

**ATOM already computes the attention terms.** `ScheduledBatch.detailed_sqsq`,
`detailed_sqsk`, `detailed_sk` (`scheduler.py`) are Sum N_Q^2, Sum N_Q*N_KV and
Sum N_KV, produced by `Scheduler.compute_detailed_aggregates`, currently
gated on `self.profile_active and ATOM_ENABLE_DETAILED_ANNOTATION`.
Ungating them is a flag, not code.

Two properties of this form are free here and cost the prior effort real time to
discover, so they are baked in from the start even though M1 is a stub:

- **The attention terms are summed per request, not computed from batch-collapsed
  scalars.** Collapsing the batch to `tokens x history` and multiplying was a rank
  deficiency; the per-request sum was the repair.
- **Decode is fitted per CUDA-graph rung**, with both coefficients per rung — shared-slope
  gave 8.09% median error against **0.93%** for per-rung intercept and slope. And the
  padding feature is **the rung's rectangle** `rung*max(ctx) - Sum ctx`, not the batch's
  `len(ctx)*max - Sum ctx`. The wrong one read as **-22.6% at rung 16** and survived two
  rounds of evidence-widening, because a wrongly-computed feature is indistinguishable
  from an unconstrained one. Error tracked the ratio exactly: rung 2 ratio 1.00 gave
  -0.34%, rung 16 ratio 5.59 gave -22.64%.

### Geometry source

**Default: the real HF config** (Qwen3.8-27B's actual layer, head and dtype numbers), so
M1 already exercises real geometry and M2 is a backend swap rather than a rebuild.
**Additionally: a synthetic dial-able config as a test fixture**, for stressing the
scheduler at shapes no real model has.

### Open issues

- **A single global quadratic coefficient is wrong for a hybrid.** The 27B target is 48
  gated-DeltaNet layers plus 16 full-attention layers; a DeltaNet decode does not grow
  with history at all. Harmless for a stub; **must not be carried into topic `04`
  unexamined.**
- The stub's coefficients are declared, so every M1 number is a plumbing result and must
  be labelled as such. Nothing from M1 may be presented as accuracy evidence.
- `compute_detailed_aggregates` is currently gated behind profiling. Ungating it in the
  simulated path is trivial; whether it should also be ungated on the real path (it is
  cheap, and having it on both sides makes real-vs-simulated feature comparison possible)
  is open.
- CUDA-graph rung selection: `StepShape.capture_bucket` must be `None` when nothing was
  replayed. A prior bug derived the bucket from batch size alone and so assigned a rung to
  prefill steps too, charging a four-sequence prefill 2 us per launch instead of 67 us per
  operator. Also note ATOM's own padding site takes the **largest** rung where its comment
  says otherwise (`reversed(capture_sizes)` is correct only on a descending list, and
  `capture_cudagraph` re-sorts ascending when it finishes); the rule that actually selects
  the graph is `ForwardMode.decide`.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D10 | Attach at `ModelRunner.forward`, delivered by a `--runner-qualname` subclass; no ATOM change for the injection. Revised: `flush_pp_send` keeps ATOM's own answer and the stage loop settles the pending send before calling it; the control-command refusals are recorded in [`01` D5](01_execution_and_time_model.md#decision-log) | 2026-09-18, revised 2026-09-28 and 2026-09-30 |
| D10.1 | A model comes into existence three ways, none reading weights: HF-config geometry where only geometry is needed; a `torch.device("meta")` context where a module tree is needed without tracing (no transient, no GPU, covers buffers - ATOM already has this shape at `models/utils.py::no_init_weights`, unused); and `FakeTensorMode` for tracing, the only one with symbolic shapes and the right device branch. `--load_dummy` is reused in all three. `_init_weight_params_on_meta` is not a bug but is superseded by the meta context. The T5 fallback is the meta context plus concrete traces. | 2026-09-20 |
| D11 | No modes on the runner. Every run simulates; the algorithm comes from a pluggable cost backend. `measure` and `trace` are orthogonal flags. | 2026-09-18 |
| D12 | M1 fake model = KV/weight geometry from the HF config + a shape-analytic cost stub including the quadratic query term; constant mode retained for bring-up | 2026-09-18 |
