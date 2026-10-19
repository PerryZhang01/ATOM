# ATOM Compass — Design Topic 7: The Calibration and Benchmarking Toolchain

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Revised 2026-10-13 by the owner's rulings on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791) and
[#882](https://github.com/jgong5/ATOM/issues/882#issuecomment-6140285611):** the
toolchain's commands are `13` D80's, and D38 says which phase each runs; planning belongs
to the backend that needs the measurements (D37); the bench is drawn and its window
defined (D39); new artifact kinds and invalidation rows (D41, D43).

**Revised 2026-10-14 by the owner's
[ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104), which
[#851](https://github.com/jgong5/ATOM/issues/851)'s decisions carry:** Phase 2 is
`compass measure memory`, whose `memory_readings` entry a run reads, so the readings leave
the machine spec (D38, D41, D43); the step grid's planner runs inside `measure steps` (D37); no
command starts the client, and grading decides whether a run is gradeable (D38.1).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** two granularities, step-level and
op-level, and a method per term replace the earlier three-way split of cost models (D36);
the backends, artifacts and flags carry the step stub, step linear and step grid names.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489), amended the same day by its
[bench, band and capture ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252):**
the op-level backend is empirical on a measured device. Capture is concrete, one graph
per decode width and prefill (T, B) (Phase 1a). `measure ops` times each IR leaf cold
from its contract inside ATOM's own forward, a stub model in place of the model, at
points planned from the steps of record directories (#1527), recording the kernels each point launched (D38,
D39); collectives are leaves measured across every rank of the layout (D40); `fit ops`
publishes a `price_list` that interpolates only between same-kernel neighbours, as `09`
D60 requires, and nothing else, so region terms and the op-level residual are retired
(D38.1, D41, D43); a leaf the price list declines falls to its analytical law, counted
(D36, D42).

**Depends on:** `04_model_capture_and_cost_ir.md` (what is traced and priced),
`05_machine_spec_and_probes.md` (the machine artifact and its probes).

**Scope.** The offline toolchain that produces the data an online prediction consumes:
what each phase measures, how its output is keyed and digested, when it must be re-run,
and what the user actually types. Empirical modelling only — analytical laws are a separate
design topic (`10`).

**Terminology, fixed here because "priced" is not a method (doc 02 D11):**

| Term | Unit | Produced by |
|---|---|---|
| **op price** | one IR leaf at one point | `measure ops`: the leaf from its contract inside ATOM's forward, CUDA-graph replay over cold copies (D38) |
| **step measurement** | one whole forward | CUDA events in situ |
| **calibration** | the *act* of deriving model terms from either of the above | — |

"Calibration" must not swallow the other two. A prior conflation applied a 2% kernel
stability gate to a prefill region whose published value carries a **59.5% relative range**
as its stated uncertainty, and concluded the region was unmeasurable.

---

## D36. Two granularities, and what each needs

Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952), which retires the earlier three-way
split of cost models: it mixed what a backend prices as one unit with where each number
comes from. The op-level row revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489).

| Granularity | Backends | Needs | Used for |
|---|---|---|---|
| **step-level** | the step stub, linear law and grid (run-file keys `step_stub`, `step_linear`, `step_grid`); the step roofline | measured steps, fitted over features `ScheduledBatch` already carries; the roofline only the machine spec and the modules of the meta build | structure discovery, plumbing, sweeps; day zero, a device never measured, fast configuration sweeps |
| **op-level** | the op graph (`OpGraphBackend`, not built) | a concrete IR graph per decode width and prefill (T, B); each leaf priced from a `price_list`, else by its analytical law; no residual | acceptance cells; on a measured device, a new engine config, layout or shape by measuring only the leaf signatures the price list lacks, with no step recapture |

Two orthogonal axes: **the granularity of the backend asked**, and **the method of each
term it returns** (doc 02 D11): analytical, declared, measured or estimated. Fallback runs
only along method inside one granularity: an op-level backend answers an unpriced leaf by
its analytical law, always on; a step-level run falls to the step roofline only where its
run file composes `StepFallbackBackend`; nothing falls from op-level to step-level. This
document covers what the empirical terms of both need to exist.

**One use beyond the table, by ruling.** Topic `17`'s step grid, a step-level backend, is
graded as a candidate for #627's stage 2 by the owner's ruling on #696 (`17` D96). The
"used for" column is otherwise unchanged.

Scope boundary, stated once: **Compass models a device it has been measured on.**
Configurability — dialling the capacity, bandwidth or interconnect of a measured device —
stays in scope. An unmeasured architecture is predicted only by analytical laws (`10`),
declared while their derates are assumed, and a term with an input observed on another
subject (another platform, width, model or engine config) is declared. Accordingly, doc 05
D26 has no `transfer` probe: no empirical data crosses hardware as measured.

---

## D37. The backend that needs a measurement plans it

### Problem

The toolchain is several artifacts, phases, hardware classes and an invalidation matrix.
Expecting a user to know which of those they need is how coverage rots quietly.

### Decision

**The tool tells the user what to measure. The user does not tell the tool.**

**Revised 2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** no single
command plans the whole campaign. A backend that needs measurements plans them itself,
beside its own code, and owns the coverage check over them. The step grid's planner and
coverage check are the first (`17` D98, D99): `measure steps --grid` runs the planner
over a `shape_population`, and `compass run --coverage` calls the check before the run
starts, refuses a run that fails it, and records the result for grading; a backend with
no check refuses `--coverage`. The commands, with their inputs and outputs, are `13`
D80's.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** the op-level backend's planner is
the second. `measure ops --plan` reads each step of the record directories it is given (#1527) against the
`op_graph` captured at its shape and places each leaf's points over them (D38, Phase 1b),
and its coverage check answers which steps have every leaf at a measured point or between
two same-kernel neighbours.

### The cases the commands cover

| Case | What runs |
|---|---|
| new model, on hardware I have | D39's recommended flow: its steps half for the step-level backends, and its ops half too for the op-level one |
| new workload, same model and hardware | `trace steps` and the coverage check; a GPU only for steps outside what was measured |
| new parallel width | `measure memory` and `measure` at that width, since memory readings and collectives are per width (D40) |
| software upgrade | each entry whose D43 row the change invalidates refuses at load, naming its key; only those are re-measured |
| configuration sweep | `measure memory` and `run` per configuration, then rank |
| day zero, nothing measured | the step roofline (`10`), with no measurement, its terms declared while the derates are assumed; memory from a `memory_readings` entry marked assumed (Phase 2) |

The upgrade case is the one the invalidation matrix (D43) exists to make cheap. Day zero
is why the roofline is designed as a backend and not only as a fallback; today a run
reaches it only as the fallback its run file composes (`StepFallbackBackend`).

### Preparing a measurement so the empirical model works

This is the question users actually have, and the answer is two tool behaviours rather
than user knowledge:

- **Pre-flight.** The backend's planner names the shapes the sweep must reach, derived
  from Phase 0's observed population plus a stated margin.
- **Before a graded run.** The coverage check covers every real step and every Phase 0
  step of the case set, from `record` directories and, for the step grid's, `shape_population` entries, and
  names any step that falls outside what was measured.

Both must use a **convex hull or k-NN distance, never a per-feature bounding box.** The
reasoning, the three iterations it took to abandon the bounding box, and the measured
failures are **doc `09` D58**, which owns coverage geometry because that is a fitting
question. Two consequences the toolchain has to honour, without restating the argument:
the planner's pre-flight and the coverage check both call the same hull test that `09`
specifies, and neither reports coverage for a sweep alone — coverage is a statement about
a **(sweep, workload) pair**.

### Open issues

- What a command should do when the store holds an artifact whose fingerprint cannot be
  recomputed (e.g. the source root is gone) is unresolved. Refusing is safe but may be
  unusable.

---

## D38. The phases

```
 Phase 0   discovery            device-free   shapes ONLY, never timings
 Phase 1a  trace                device-free   one per decode width / prefill (T, B), JIT-able
 Phase 1b  price ops            GPU, each     every leaf from its contract, cold, at planned points
           price collectives    layout        every rank of the layout, the union of signatures
 Phase 1c  in-situ calibrate    GPU, TP1      full-engine steps
           ...plus ONCE         GPU, TP2      not to calibrate, but to TEST transfer
 Phase 2   memory readings      GPU, each cfg one engine start, no workload
 (there is no Phase 3 - see "What is NOT a calibration axis" below)
```

`13` D80's commands run them. Phase 0 is `trace steps`; 1a is `trace ops`; 1b is
`measure ops`, collectives included (D40), then `fit ops`; 1c is `record` for real runs
and `measure steps` on the bench, then `fit steps` and `fit host`; 2 is `measure memory`.
A `measure` command writes measurements per point, and a `fit` command makes the laws,
bundles and leaves from them.

Every phase gets a section. Phases 1a and 2 were named but not elaborated in the first
draft; they are below, after 0, 1b and 1c.

### Phase 0 — discovery

`trace steps` runs ATOM's real scheduler device-free, serving as `run` does, priced by
the backend it names and driven by a client started apart, and writes the step record
(`13` D80): each step's batch, its rows in order with their query and context tokens, and
its rung. It yields the distinct **structure keys**, the
**shape population** actually reached (batch widths, context multisets, raggedness, rung
histogram), and which structures exist only at width, published as one `shape_population`
entry (D41).

**It reads the target workload's shapes and never its timings.** That separation is what
keeps the acceptance corpus held out, and it is auditable — the artifact records exactly
what Phase 0 consumed.

**Never reimplement admission to derive shapes.** A prior attempt re-derived
`Scheduler.schedule()`'s Phase 1/Phase 2 plus `_chunked_prefill_size` and omitted
`_finalize_prefill_chunk`'s `checkpoint_cut`, the `BlockManager`, preemption and the pool —
*"which change which requests coexist, not just chunk sizes"* — so it was not conservative
in either direction. Against the real classes it under-counted prefill steps **3 vs 8** on
one cell and over-counted width **5 seqs vs 4** on another, while landing close in
aggregate.

**Circularity, and how it is closed.** Phase 0's schedule depends on the cost model Phase 1
produces. Rather than iterate: **over-cover with a stated margin and verify the hull
post-hoc.** Cheaper, and it fails closed. The population's key names the backend that
priced it, so populations from two backends coexist and a planner names the one it read.

### Phase 1a — trace capture

**Device-free.** `trace ops` captures one concrete IR graph per decode width and per
prefill (T, B) it is given or that a record directory's steps hold (`13` D80), using the `TorchDispatchMode` + `FakeTensorMode` mechanism of
doc `04` D18, each leaf with its contract (`04` D20). Revised 2026-10-18 by the owner's
ruling on [#1489](https://github.com/jgong5/ATOM/issues/1489): capture is concrete, not
one symbolic graph per structure. Nothing is allocated on a GPU and no kernel runs, which
is what makes this phase schedulable anywhere, and what makes it the one phase that may
also run lazily inside a simulated run (doc `02`).

| | |
|---|---|
| **Input** | the decode widths and prefill (T, B) it is given, and the steps of the record directories it is given, from `record` or `trace steps --out`; the model config; the engine config |
| **Output** | one `op_graph` artifact per shape, keyed by the structure's discrete key and the shape (`04` D19) |
| **Cost** | 7.6 s per shape on Qwen3.8-27B TP1, 2.2 s of it the traced forward ([#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)'s PoC) |
| **Needs a GPU** | **no** — but see the caveat below |

**The caveat that makes this phase less free than it looks.** "Device-free" means no GPU
*compute* and no GPU *allocation*. It does not mean no ROCm. `FakeTensorMode.__enter__`
probes the driver to choose the fake device, so a wedged driver hangs a trace that
touches no GPU — which has happened, and reads exactly like a tracing bug. The probe is
stubbable (claim a device exists so both `is_available` and
`_ensureCUDADeviceGuardSet` paths are skipped), but stubbing has its own cost: anything
that trusts `is_available()` and then really touches the device will hang instead. So
Phase 1a is *portable*, not *hermetic*, and a hang here is diagnosed with the 30-second
`rocminfo` check before it is diagnosed as a capture defect.

**What a trace at TP>1 requires** is the open question, not a detail: `T5` asks whether
ATOM's model classes trace cleanly under `FakeTensorMode` at width, and `T52` is the
mode-induced 8-rank hang that gates it. Phase 1a is device-free at TP1 and **unproven at
TP>1**. If it turns out to need a live process group, this phase stops being schedulable
on a CPU box and the campaign's shape changes — which is why both are gating tasks.

**What it does not produce.** A graph is a structure, not a price. Every leaf in it is
unpriced until Phase 1b. A run with complete Phase 1a coverage and no Phase 1b has no
measured leaf: the op-level backend answers every leaf by its analytical law, and the step
reports those seconds by method (`08` D50.1). That is the intended behaviour and a useful
diagnostic rather than a failure.

### Phase 1b — op pricing: each leaf from its contract, cold

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489) and its
[bench, band and capture ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252).**
`measure ops` times each IR leaf alone, from its contract (`04` D20), inside ATOM's own
forward: the op bench (D39) runs the full model on dummy weights after ATOM's warmup,
with a stub model in place of the model. For each planned point ATOM builds the metadata
and forward context; the stub calls each leaf with activations random by its contract's
spec and stride, and weights, KV and metadata from the live runner. Attribution from
profiled steps is not built.

Two placement constraints put pricing inside the runner process, and the op bench is
that process:

- **AITER registers kernels lazily on first call *in the calling process*.** Importing the
  defining module does not put `gemm_a16w16` in `torch.ops.aiter`; even a process that
  built an engine and generated tokens still reports it missing, because ATOM runs the
  model in a worker subprocess. The stub calls each leaf in the worker that runs ATOM's
  forward; a leaf whose operator cannot run from its contract refuses by name.
- **An operator that walks paged KV needs a KV cache.** The live runner's, with the block
  table ATOM builds for the point's batch.

Weights are ambient rather than random by spec, measured: random GEMM weights read one
signature 5% high, where the layer's own weights match.

Method, settled by elimination:

| Route | Verdict |
|---|---|
| per-operator CUDA events in line | **rejected** — 3.9x instrumentation overhead; 327 operators summed to 45.664 ms against a 3.946 ms replayed step |
| per-call back-to-back loop | **rejected** — measures the host. A flat ~30 µs floor from M=1 to M=256; 298 operators x 30 µs = 8.9 ms against a priced total of 9.24 ms, i.e. the price list was the operator count times the floor |
| one event pair per call | **worst** — the device idles through the host dispatch; 52.20 µs at M=4 against ~9 µs of kernel |
| **capture N calls into a CUDA graph and replay** | **the method.** A replay of K calls reads `kernel + F/K` a call, F the replay's fixed cost: 18.7 and 19.4 µs for the mrope leaf on MI308X between K = 32 and 256 ([#1513](https://github.com/jgong5/ATOM/pull/1513)), so a 2.7 µs leaf still reads 16% high at K = 32. Every graph holds the same K (`atom/compass/bench/ops.py::CALLS`), which adds F/K, about 0.1 µs, to every price |

**Cold, by rotating copies.** The replay rotates over copies of the leaf's inputs and
ambient state whose total exceeds the device's last-level cache, so each repeat reads cold
data. The copies include tensors the leaf reads through module attributes: the mrope
leaf's cos/sin caches are `rotary_emb` attributes, counted in its bytes. The prototype's
mrope reading, 3.3% to 11.5% under the same call timed where ATOM's decode step makes it,
was the fixed replay cost at two call counts (above), not warm caches. The warm-over-cold ratio, the same point replayed on one copy, is reported per point
as a diagnostic and prices nothing. The card's PTL state is read and keyed as
[#1434](https://github.com/jgong5/ATOM/issues/1434) rules.

**Points, planned from the steps of record directories (D37).** Per leaf: every decode rung; prefill rows on
a ladder of chunk remainders; attention's context features chosen per kind by data, under
[#941](https://github.com/jgong5/ATOM/issues/941)'s rule. MoE leaves run on independent
uniform routing. Each point records the kernels the leaf launched there, its kernel
identity.

**The price.** `fit ops` publishes a `price_list`: per leaf, piecewise-linear
interpolation between same-kernel neighbours only, as `09` D60 requires. A query at a
measured point takes that point's price; one between two points that launched the same
kernel is interpolated; any other query, outside the measured range or between two
kernels, is declined with its reason (the leaf, the dimension, the neighbours and their
kernels), and the leaf falls to its analytical law, its share counted (D42, `08` D50.3).

### Phase 1c — and the one extra run

Full-engine steps, TP1. Produces the step-level backends' measurements and the real runs
the host laws are fitted on.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** Phase 1c produces nothing the
op-level backend reads. That backend has no region terms and no residual correction: it
composes its leaves by the IR's joins, and the gap between a composed step and the
measured one is reported per phase and op kind (`08` D50.3), never fitted away. That
**an operator priced alone is not the same operator in a step** is why the gap is
reported: priced kernels at 98.8% coverage summed to **0.749 of the step**; per kernel the ratio clusters 0.83-0.93,
across all kernels it runs **0.434 to 0.989**.

**Revised 2026-10-11 by the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815):** every backend prices device time only
(`02` D11), so a step measurement is device busy time, and the host floor is no op-level
term. Host time is `09` D54's host laws, composed by `01` D4's recurrence.

**Plus one full-engine run at TP2**, not to calibrate but to **test whether 1c transfers**.
It is the cheapest possible test of the one term with no transfer evidence: between TP=1
and TP=2 on the same GPUs, the two overhead constants moved **12% and 7% in opposite
directions** (2.25 -> 1.97 µs/launch, 86.35 -> 92.48 µs/op). If it transfers, TP4/TP8 are
predicted; if not, that is the finding and the recipe grows.

### Phase 2 — memory readings

**One engine start per configuration, no workload at all.**
`compass measure memory --gpu G --machine M --store S --out D -- <ATOM args>` starts the
real engine, reads the readings below on every rank once the KV cache is allocated and
the graphs are captured, publishes one `memory_readings` entry (D41), and exits (`13`
D80); `record` takes none. The entry is keyed by the model, the digest of `--machine`,
the parallel layout and the engine config, its KV offload connector included
(`atom/compass/run/memory.py::key_fields`), and
records the stack the start ran on, which a run's spec must pin. A run reads its rank's
member and refuses when there is no entry, so the readings leave the machine spec (doc
`05` D25). This phase exists because these memory quantities cannot be derived from
geometry and have no law behind them. It is the cheapest phase in the
campaign and the one with the tightest acceptance gate behind it — KV block count at
**≤5%**.

| | |
|---|---|
| **Input** | ATOM's arguments: the model, its widths, and the engine config that fixes the capture ladder |
| **Output** | one `memory_readings` entry for that configuration (D41) |
| **Cost** | one start per configuration, two on a cold inductor cache. Minutes, no workload, no traffic |
| **Needs a GPU** | **yes**, and a **quiet** one |

**What is read, and where it comes from:**

| Reading | Source |
|---|---|
| `capacity_bytes` | the device, once; the capacity a run sizes against stays the machine spec's configured one (`13` D78) |
| `driver_and_collective_reserve_bytes` | ATOM's own `get_num_blocks` `non_torch` term, from a start in which inductor autotuned no kernel: the configs autotuning tries stay loaded, +12 to 26 MiB a rank on a cold cache (Qwen3.8-27B TP2), so a start that autotuned is started again on the cache it warmed, under a free `ATOM_DP_MASTER_PORT`, since the first start to load ATOM's compiled-graph cache autotunes again ([#1428](https://github.com/jgong5/ATOM/issues/1428)) |
| `allocator_retained_after_load_bytes` | torch allocator, after weight load completes, less the model's parameter and buffer storages |
| `persistent_forward_buffer_bytes` | allocated bytes where `get_num_blocks` reads the device, less those after load: `allocate_forward_vars`, attention metadata setup |
| `activation_bytes` over `warmup_tokens` | ATOM's one warmup forward's allocator peak above what stays allocated, as a serving start reads it, from a start in which inductor autotuned no kernel: on a cold compile cache the peak sat in inductor's autotuning tensors, +15.8% (Qwen3-0.6B) and +65.2% (Qwen3-30B-A3B) ([#705](https://github.com/jgong5/ATOM/issues/705)). A second warmup in the same start reads higher than the first, which is the one that sizes the KV pool ([#1481](https://github.com/jgong5/ATOM/issues/1481)) |
| `cudagraph_pool_bytes` | reserved delta across capture, at the configuration's own ladder, as `capture_cudagraph` returns it |

**Three rules that turn this from a reading into evidence:**

1. **Quiet machine, checked before *and* after.** `non_torch` is a **device-wide**
   reading, so a neighbour's allocation lands in it and is indistinguishable from ours.
   Three of the last five prior pilot attempts were lost or degraded this way — two
   refused to start against 141 GB of neighbour, one ran 64% slow. A check that only
   runs before the measurement does not catch a tenant that arrives during it.
2. **Reset the allocator high-water mark around the measured step.** Otherwise
   `peak_torch` belongs to the warmup prefill, whose shape is nobody's choice — measured
   at **−14.7%** on one configuration.
3. **A configuration with no entry refuses rather than defaults.** `run` names it. This
   is the one place with no fallback, because a silently defaulted memory constant
   produces a plausible KV block count against a 5% gate.

**The checkpoint is loaded.** ATOM's `--load_dummy` skips it, and `measure memory`
refuses it until its readings match a real load's on Qwen3.8-27B TP1 on MI308X (T100).

**A device never measured** has no engine to start. Developer tooling outside the CLI
publishes an entry for its configuration, every reading marked assumed, as `16` marks every
MI455 input; `run` reads it as any entry, and every term built on it is declared, its note
naming the assumption.

**Why it is per width and not fitted across widths:** because it does not fit. The
measured `non_torch` sequence is 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8, and
**no fixed-plus-per-peer form reproduces** the deltas 5980 / 6340 / 9138. A table, not
a law — which is also why Phase 2 is per configuration rather than a one-off.

### What is NOT a calibration axis — and why there is no Phase 3

Client count and workload slice. They change **which shapes occur**, not what a shape
costs, and Phase 0 already told us which. There is deliberately no Phase 3: the numbered
list stops at 2 because adding a per-client-count or per-slice phase would multiply the
campaign by a factor that buys nothing. The place client count *does* matter is
validation (doc `08`), not calibration.

### Open issues

- Under the old flat-graph design the per-launch constant was absorbing pricing error
  multiplied by launch count — which is why it worked and why it did not transfer. With no
  residual now, the composed-over-measured gap (`08` D50.3) shows that error; whether
  pricing each leaf at its own points shrinks it is a **hope, not evidence**.
- Phase 1c's sensitivity: prefill priced against a real run's own steps was **+0.10%**
  overall but **+3.06% excluding step 0** — a single cold-start error of **-6.75 s** almost
  exactly offsetting **+6.99 s** spread over 105 chunks. The scheduler never sees the
  total; it sees the running sum.

---

## D39. The standalone measurement bench

### Problem

Driving a whole serving stack to take a measurement is slow to iterate and couples the
measurement to the scheduler. But calling the model directly was tried and failed:
`model(input_ids, positions)` dies at `fwd_ctx.context.is_dummy_run` — attention reads a
forward context only the runner establishes, and building it by hand means reimplementing
`prepare_inputs`.

### Decision

**The bench is `EngineCore` without its scheduler, fed *replayed* batches.** Revised
2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791), which draws
the boundary this decision first left open (T17).

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252): this is
the step bench, and the op bench is ATOM's own forward with a stub model.** The op bench
starts the full model on dummy weights as `EngineCore` starts its runner, through ATOM's
warmup, and puts a stub in place of the model's forward. Each point its planner placed
reaches the runner as a batch, so ATOM builds the metadata and forward context as for any
step; the stub then times each leaf of the point's `op_graph` from its contract, cold
(D38, Phase 1b). Nothing of ATOM's input preparation is copied. At width it starts every
rank of the layout through ATOM's own group initialisation (D40). The step bench is not
reused for ops: it times whole forwards, and it refuses DP.

Measured on [#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)'s
PoC, Qwen3.8-27B TP1: a leaf timed from its contract, with ambient state from ATOM's input
preparation, came within 1.6% of the same call timed in ATOM's decode step for attention,
GDN, six GEMM signatures and `silu_and_mul` at B = 2 and 16, and within 1.1% for nine
prefill leaves at two points. The full model is the one that runs: a 4-layer runner read
GDN 12.6% low at B = 16, its state staying in the last-level cache, and without ATOM's
warmup GDN read about 5% low.

The constraints point at the *runner*, not the server — a `ModelRunner` **is** the worker.
ATOM already shows how to drive one
without a scheduler: `model_runner.py::ModelRunner.dummy_execution` and
`model_runner.py::ModelRunner.warmup_model` both fabricate `ScheduledBatch`es by hand.

- **Every rank's runner, started as `EngineCore` starts it.** The bench starts each TP
  rank's `RecordingModelRunner` (`atom/compass/parity/runner.py`) through `EngineCore`'s
  runner manager, in `EngineCore`'s order — `get_num_blocks`, `allocate_kv_cache`,
  `capture_cudagraph` — and then calls `forward` with replayed batches, with no scheduler.
  Any TP width: every rank measures, and a step is charged to its slowest rank (`15`
  D90). DP above one, expert parallelism and PP refuse until a task builds them.
- **The runner a real run records with.** A measured step and a recorded one share one
  timing method, so a bench point and a real step compare.
- **One window for both step kinds.** A step's value, decode or prefill, is device time
  from an event at the forward's entry to one at its exit, with device work queued ahead
  so the host's launch cannot show in it. A repeat counts only if the entry event is still
  pending when the forward returns; one that is not is discarded, and more work is queued
  ahead. A forward with a host synchronisation inside never qualifies, and its point
  refuses. For decode the window covers the device work the step linear law's label covers in a
  recorded step, device gap and busy window (`09` D54); for prefill it adds, to that label's
  busy window, the device work the step's input preparation queues.
- **The value kind is stated.** A step grid bundle states that value kind beside its
  interval (`02` D10), and loading refuses another (`17` D98).
- **Replayed by rule.** A batch arrives as a step-record line (`13` D80), its rows in
  order with their query and context tokens, and is rebuilt by a fixed, versioned rule:
  row order kept, each row's KV blocks contiguous from a fresh pool in row order, filler
  tokens, a fresh state slot per row that keeps one. Each measurement records the rule's
  version. Whether that reproduces the recorded step's forward context is T18.

```
  record, trace steps;              the step bench (GPU): EngineCore's
  step grid's planner, in measure   runner manager, every TP rank, no scheduler
  +-------------------+  step-     +--------------------------------+
  | batch: its rows   |  record    | RecordingModelRunner, per rank |
  | in order, their   |  lines     |  .forward(replayed by rule)    |--> step measurement
  | tokens, its rung  |----------->|                                |
  +-------------------+            +--------------------------------+
```

**Batches must be replayed, not synthesised.** Hand-built batches get the composition
wrong in ways that matter:

- Row order alone moves measured decode attention across **1.77x** at one fixed 32-row
  context multiset (grouped-descending 276.1 µs, grouped-ascending 292.2 µs, alternating
  376.5 µs, ladder-interleaved 489.5 µs). Hand-built ladders are grouped by construction.
  **Explicit `block_tables` are mandatory**, not optional, or the treatment is
  unreproducible; the replay rule builds them from the recorded row order.
- Raggedness is workload-dependent and a fit is blind to a dimension it never varied: every
  calibration round running one sequence length left raggedness at exactly 1.00 in **all
  2,997 samples** — a column of zeros where a feature should be.

### What each measurement may use

| Measurement | Step bench with replayed batches | Synthetic shapes |
|---|---|---|
| op price | no: the op bench | **yes** — a planned point is a batch ATOM prepares; activations come from the leaf's contract, the rest from the live runner |
| step measurement | **required** | no |

Op pricing on synthetic shapes is what keeps cc-traces off a GPU during calibration.

**One departure, by ruling.** Topic `17`'s step grid measures its baseline grid on synthetic
homogeneous shapes (`17` D98). The residual that carries row order and raggedness is
fitted to real-run step times, to replayed bench measurements, or to both (`17` D100,
[#882](https://github.com/jgong5/ATOM/issues/882#issuecomment-6140285611)). The rule
above holds for every other step measurement.

### The benches do not change D38's phases — they are where two of them run

A bench is a **host**, not a phase. D38's phase list is unchanged; what the benches
settle is *which process* each phase runs in, which was previously implicit.

| Phase | Command (`13` D80) | Runs in | Why there |
|---|---|---|---|
| 0 discovery | `trace steps` | a simulated serve, device-free | drives ATOM's real `Scheduler`, priced by the backend it names |
| 1a trace | `trace ops` | any CPU process, **or** lazily inside a sim run | device-free (with the ROCm caveat above) |
| 1b price ops | `measure ops` | **the op bench**: ATOM's forward with a stub model | ATOM builds each point's metadata and forward context, and the stub times each leaf from its contract; at width every rank of the layout (D40) |
| 1c in-situ | `measure steps`; `record` for real runs | **the step bench**; a real serve | needs whole real steps; the step bench replays recorded and planned batches into them |
| 2 memory | `measure memory` | **a plain engine startup**, not a bench | it measures startup, so a bench that skips parts of startup would measure the wrong thing |

So: the op bench covers 1b and the step bench 1c; Phase 2 is an ordinary startup
`measure memory` runs; Phases 0 and 1a need neither.

### The recommended flow

One canonical sequence per half of the toolchain. For a new (model, machine) pair, where
each server in steps 2, 3 and 6 serves the client runs a client started apart sends
(`13` D80):

```
 Steps half: the step-level backends (16 S2.7)
 1.  compass measure memory -- <ATOM args>                 [GPU, one start]
       -> memory_readings for this configuration
 2.  compass record ... -- <ATOM args>                     [GPU, a server]
       -> step records, split per client run; run.json with the engine's bounds
 3.  compass trace steps <backend> --machine M ... -- <ATOM args>   [CPU, a server]
       -> shape_population, keyed by the backend that priced it
 4.  compass measure steps <records> --grid <populations> ...       [GPU]
       -> the step grid's planner over 3 (17 D98); one step_measurement per distinct batch
 5.  compass fit host <records>; compass fit steps --backend step-linear|step-grid ... [CPU]
       -> host_laws; a step_linear_law or a step grid bundle
 6.  compass run <backend> --machine M [--coverage <populations, records>] ...  [CPU, a server]
       -> the step grid's coverage check reports before it starts, refusing
          no step (17 D99); run.json records what grading reads
 7.  python -m atom.compass.validate; python -m atom.compass.memory.check,
     outside the CLI                                       [CPU]
       -> whether each client run is gradeable, the separable results (08 D44);
          the memory terms, one at a time

 Ops half: the op-level backend (16 Stage 3d), after the steps half's 1, 2, 3 and 5
 1.  compass trace ops --population <records> ...          [CPU, seconds per shape]
       -> op_graph per decode width and prefill (T, B) of the steps in the
          steps half's 3 --out, or 2's records
 2.  compass measure ops --plan <record> [--plan <record>]... ...  [GPU, every rank]
       -> the op-level planner over the steps in the steps half's 3 --out,
          or 2's records (D37); one cold
          measurement per planned point, with the kernels each leaf launched,
          collectives included (D40)
 3.  compass fit ops ...                                   [CPU]
       -> price_list
 then the steps half's 6 and 7 with the op-level backend
```

The ops half is not self-contained. `run` with the op-level backend reads the steps half's
step 1 (`memory_readings`) and step 5 (`host_laws`, which `01` D4 composes with the device
time the backend returns), and grading reads step 2's records.

Only `record` and `measure` need a card. One `record` at TP2 is the whole transfer claim:
it does not calibrate anything, it tests whether Phase 1c generalises across width (T21).
Steps 6 and 7 are what turn a pile of artifacts into a statement about what can and cannot
be predicted.

**Two things the flow is arranged to make hard to get wrong.** `trace steps` runs with the
engine config and prefix caching that the target workload will use, because a shape
population discovered without prefix caching does not contain the high-`N_KV`/low-`N_Q`
corner a cache hit produces (doc `03`), and every hit step will then fall outside the
hull. And the coverage check runs *before* a graded run, not after a prediction looks
wrong.

**Iterating.** When the coverage check or a run reports refusals, the fix is to extend the
shape population and re-run from `trace steps` — not to widen the hull. Re-measuring for
an extended population is incremental: the artifact store is keyed per entry (D41), and a
step measurement by its batch, so only new batches and signatures are measured.

### Open issues

- Whether a batch replayed by rule reproduces the recorded step's *forward context*
  faithfully, including the attention metadata built by `prepare_inputs`, is unverified.
  MoE routing follows token content, so a replayed MoE step may route otherwise than the
  recorded one.

---

## D40. Pricing communication

Comms split four ways, and only one is straightforward.

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** collectives are leaves like any
other, interpolated over message bytes. They are measured on a multi-rank op bench that
starts every rank of the deployed layout through ATOM's own group initialisation, takes a
barrier before each timed replay, and measures the union of all ranks' signatures (rule 1
below). The step bench, which refuses DP, is not reused.

**(a) Dispatcher-visible collectives** — `aiter::all_reduce_` and the fused
`fused_allreduce_rmsnorm_*` family, `qr_all_reduce`. Priced like any leaf, **on the
multi-rank op bench, where the process group is live**. Measured: `[4,1024]` decode
**10.38 µs**, `[1256,1024]` prefill **71.33 µs**; a TP=2 decode graph holds 58 grouped
operators, 57 of them `all_reduce_`, coverage **98.3%**. One known failure: `c10d::broadcast_` takes a
`ProcessGroup` object no JSON artifact can hold — one operator per step.

**(b) Invisible collectives, and the default path is the invisible one.**
`ATOM_USE_CUSTOM_ALL_GATHER` defaults to **1** (`atom/utils/envs.py::environment_variables`); its
registration is *commented out* at `aiter/dist/parallel_state.py:467-468` and
`all_gather_reg` / `all_gather_unreg` are in `NONE_WRAPPED_OP`. Zero dispatcher events end
to end — while the RCCL path **is** visible (`c10d::allgather_into_tensor_`). These need
doc 04 D21's **stated node** plus a dedicated standalone benchmark calling the same entry
point.

**(c) MoE all-to-all (EP)** — MORI, called inside `moe_forward`, invisible at any degree.
Its block counts, ATOM's two TBO paths and the overlap are doc 04 D19's.

**(d) p2p.** PP is `isend`/`recv` on NCCL (`atom/distributed/pp_comm.py::async_send_intermediate_tensors`, `commit_pp_send_work`),
priceable like (a). **KV transfer is not priced at all** — doc 01 D6 simulates it as
`latency + bytes/bandwidth` from the machine spec, which is what makes interconnect
configurable.

### Three rules, all learned by hanging or by being wrong

1. **Price the union of every rank's graphs, not each rank's own.** Signature lists
   differing by a single entry **deadlock rather than fail** — a hang with no error, inside
   the worker.
2. **Anything optional in the pricing loop is optional on every rank or on none.** Pricing
   at TP=4 died in a distributed `recvBytes` with no error of its own, because a
   per-signature kernel breakdown made two extra calls per signature — and for a
   collective, two extra calls are two extra collectives.
3. **Scaling is a table, not a law.** Measured in situ: `cross_device_reduce_1stage`
   9.08 / 10.40 / 10.95 µs at TP 2/4/8 (**+20.6%** across the range) while
   `allgather_lastdim` went 29.00 / 23.73 / **16.39 µs — down 43.5%**. Linear-in-group
   predicts +300%; linear-in-log2 predicts +200%. Both badly wrong. Measure per width.

### The honesty constraint

**A collective's measured duration includes waiting for its peer.** Same shape, two passes
in a fresh container: `cross_device_reduce_1stage` x129 read 14554.5 µs then 3400.9 µs
(**0.23x**) while every compute kernel matched within 4% and the window did not move at
all. So a collective price at TP>1 is an **upper bound on device work**, and any
in-situ-versus-isolated comparison at width carries the same contamination on its in-situ
side.

Also recorded: *"a one-stage reduce on a fully-connected fabric is latency-bound at these
sizes ... which is a claim about this interconnect, not about collectives, and should be
re-measured on a multi-node deployment before being relied on there."*

### The recommended flow for comms — and what it reuses

Only class (a) reuses the ordinary op-pricing flow unchanged. The other three need
something, and the "something" differs per class. This is the answer to whether a
dedicated flow has to be built: **one new tool, shared by (b) and (c); nothing new for
(a) or (d).**

| Class | Command (`13` D80) | Reuses | New work |
|---|---|---|---|
| **(a)** dispatcher-visible | `measure ops` at each layout — the multi-rank op bench, the same graph-capture-and-replay method as any leaf | **everything**, including the CUDA-graph replay method and the artifact schema | none; it is an op with a process group |
| **(b)** invisible collectives | `measure ops` at each width, driven from the **stated-node list** rather than from the captured graph | the op bench, the timing method, the artifact schema | a **stated-entry-point benchmark**: call `all_gather_reg`/`all_gather_unreg` directly at the shapes the graphs imply |
| **(c)** MoE all-to-all | `measure ops` at each width and EP degree | the same stated-entry-point benchmark as (b) | the block count per phase: `min(128, CU)` in prefill, `min(64, CU)` in decode |
| **(d)** p2p / PP | `measure ops` at each width and PP degree | same as (a) — `isend`/`recv` are dispatcher-visible | none |
| **(d)** KV transfer | **not priced** | — | nothing: doc `01` D6 computes it from the machine spec, which is what keeps interconnect configurable |

**So the recommended flow is:** classes (a) and (d)-p2p ride `measure ops` and need no
separate step at all — they appear in the captured graph and are priced with everything
else. Classes (b) and (c) ride `measure ops` too, in the **same multi-rank op bench, in
the same processes**, differing only in where their work list comes from: a stated list of
entry points beside the captured graph. That is why it is one command and not two.

Three constraints carry over into the flow and are not optional:

1. **`measure ops` prices collectives at every deployed width**, not at TP1 with a
   scaling law — rule 3 above, where two collectives moved **+20.6%** and **−43.5%**
   across the same width range.
2. **It prices the union of all ranks' signatures** — rule 1 above, where a
   one-entry difference deadlocks silently.
3. **Its output is labelled an upper bound**, not a price — the honesty constraint above,
   where the same collective read **0.23x** between two passes because the number includes
   peer wait. Doc `02`'s method vocabulary tags these `measured (collective, upper
   bound)` so nothing downstream treats them as device work.

**The open piece:** `c10d::broadcast_` takes a `ProcessGroup` object no JSON artifact can
hold, so it cannot be replayed from a price list — one operator per step, issued in
`ModelRunner.postprocess` after the sampler. Since the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815) (2026-10-11) removed the Phase 1c host
floor, its device time is inside the forward's device busy time, the label the step linear law and
Phase 1c are fitted to (`09` D54), and its launch is inside the `W_post` host law (`01`
D4). Recorded as **T55**.

---

## D41. The artifact store

### Problem

The benchmarking mechanics are largely settled by prior evidence. **Artifact identity is
not**, and it is where this project has repeatedly bled. Eight distinct incidents, all the
same shape — the number was fine, the question of *which artifact answered* was not:

- a price list that could not name the code that produced it — *"not an observation"*
- a bare or merged price file that **states no width and silently prices nothing**
- a hard-coded `sys.path` letting a **stale regions module answer under a current
  registry's name**
- a correct registry digest sitting beside a **stale ATOM package**, because the snapshot
  was `rsync` rather than `git archive`
- **two copies of every script** on one node — a silent divergence that faked two
  modelling bugs
- a head-only price list reporting **2,568 refused operators** when the real number was
  **17**
- a summary field with no range that **dropped a 50.88 µs excursion** from a record
  claiming 2.3% spread
- a notes-only regeneration that **destroyed a reviewed digest**

### The artifacts

Every entry carries a **key**, a content **digest**, a validity **fingerprint**, and a
**provenance** stanza naming the executed source root. Revised 2026-10-13 on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791): `measure`
writes measurements per point and `fit` makes laws and bundles from them, each a kind of
its own, and a population is keyed by the backend that priced it. Revised 2026-10-18 on
[#1489](https://github.com/jgong5/ATOM/issues/1489): the region-term artifact is retired,
`measure ops` writes an `op_measurement` per point, `fit ops` publishes a `price_list`
only, and an `op_graph` is keyed by its shape.

| Artifact | Produced by | Keyed by |
|---|---|---|
| `machine_spec` | doc 05 probes | (device, software stack, PTL state) |
| `shape_population` | Phase 0, `trace steps` | (model, workload, engine config, the pricing backend's digest) |
| `op_graph` | Phase 1a, `trace ops` | (**structure**, shape): the structure's discrete key and a decode width or a prefill (T, B) |
| `op_measurement` | Phase 1b, `measure ops` | (model, parallel layout, machine-spec digest, source-root digest, the point's digest) |
| `price_list` | Phase 1b, `fit ops` | **(model, parallel layout, machine-spec digest, source-root digest)** — a tuple, never a path |
| `memory_readings` | Phase 2, `measure memory`; published with every reading assumed for a device never measured | (model, machine-spec digest, parallel layout, engine config) |
| `coverage_hull` | derived | the price list it was built from |
| `step_measurement` | Phase 1c, `measure steps` | (model, device, engine config, source-root digest, sampling, step kind, the batch's digest) |
| `step_linear_law` | `fit steps --backend step-linear` | (model, machine-spec digest, engine config, source-root digest, form) |
| `host_laws` | `fit host` | (host CPU, ATOM source, model, engine config, machine-spec digest), as the run reads its host key |
| step grid bundle (`step_grid_bundle`) | `fit steps --backend step-grid` | `17` D101 |

The **parallel layout** of an `op_measurement`, `price_list` and `memory_readings` key is
`dp`, `pp`, `pcp`, `tp`, `enable_expert_parallel`, `enable_dp_attention` and
`decode_context_parallel_size` (owner ruling on
[#165](https://github.com/jgong5/ATOM/issues/165)), held in
`atom/compass/artifacts/keys.py::LAYOUT`. No entry answers for another layout: a TP2 price
list does not answer for `-tp 2 -dp 2`, and a transfer between layouts, if one is wanted,
is an explicit record backed by a measurement. A read miss names the fields in which the
nearest entry of its kind differs.

The **machine-spec digest** is the digest of the machine spec's own key, its device,
software stack and PTL state (`atom/compass/artifacts/keys.py::machine_spec_digest`), so
an entry measured on one platform does not answer on another, and a spec edit outside
those three moves no key ([#952](https://github.com/jgong5/ATOM/issues/952) S2). The
host name, the HBM vendor and the PTL format stay out of it (owner ruling on
[#1434](https://github.com/jgong5/ATOM/issues/1434#issuecomment-6245078946)).

The **PTL state** is `device.ptl` in the spec, `disabled` when the spec states none. A
record reads it at capture and at run time from
`/sys/bus/pci/devices/<bus>/ptl/ptl_enable`, with each GPU's bus as `rocm-smi
--showbus` names it (not `/sys/class/drm/card<n>`, whose numbering is not the GPU
index: DRM `card0` can be the BMC's VGA), a card with no `ptl` directory reading
`disabled`, and refuses when the cards of one run disagree, naming each
(`atom/compass/drive/identity.py::_ptl`). A fit refuses records whose PTL state is not
the spec's, as it does their device and stack, and so does `measure memory` before its
start. A record that carries no PTL state, captured before the field existed, refuses
at a fit unless `--ptl enabled|disabled` states it; the fit records the stated value in
its report, and refuses `--ptl` for records that carry a state of their own. PTL is the
one static field that separated hjbog-srdc-39 from node 18 once both pinned one stack,
and it accounts for the gap a synthetic decode step shows between them, about 10%.
[#828](https://github.com/jgong5/ATOM/issues/828)'s gap on real runs, 18 to 20%, is
larger and stays open. The band one key
leaves: same-key cards agree within 0.6% with PTL off; with PTL on, 3 to 5% per card
plus 1 to 2% run-to-run remains.

### Four rules

1. **A key is a tuple, never a path.** *"A bare or merged price file states no width and
   silently prices nothing."*
2. **The source root is a `git archive` digest, not an rsync.** And the audit must digest
   **every** module that can answer, not just the one named — a hard-coded `sys.path` once
   let a stale regions module answer under a current registry's name.
3. **Resolution names its answer.** Every predicted step records which artifacts answered
   it, and a refusal says *which key missed*. "Price refusal is not graph absence" was a
   day's confusion on its own, and an `incomplete: N/2570` line proves less than it looks.
4. **Handed-off artifacts are immutable.** A regeneration that changes only the notes still
   destroys the reviewed digest. Publish a new one.

### Caching keys must carry the binding, not just the shape

A shape-keyed price cache is **unsound**. `signature_of` reads each operator's `context`,
and `context` is where the binder writes this step's `slot_mapping` and state indices. A
second valid allocation for the same shape moved **64 of 2,439** signatures and took a step
from **32.667 ms over 2,424 priced operators to 28.360 ms over 2,376** — 48 operators
priced under one allocation are unpriced under the other. A shape-only cache answers the
first number, with a complete-coverage claim, for a step that is neither.

Measured cost of getting the key right, device-free, against a 32.7 ms modelled step:
**41.7 ms** uncached (~39 ms of it summing prices), **2.3 ms** shape-keyed and unsound,
**4.3 ms** with the allocation carried.

### Cold costs are accounted once, not twice

Factory build is **9.5-11 s** once; the first `graph_for` on a new shape is **0.27 s**.
These must be journalled, not folded into a per-step average.

**Beware the cold-loop artefact:** an average over a loop whose first iteration is a cache
miss reads **3-7x higher** than steady state. Measure a second, warm loop.

### Capture hygiene, as refusals rather than conventions

- **Trace at step >= 2.** Triton autotunes on a kernel's first launch. Step one recorded
  **90,838 operators** against step two's **101**; `chunk_fwd_kernel_o` alone appeared
  **34,269 times**.
- **Warm with identical *token* counts**, plus `--no-enable_prefix_caching` so the second
  pass really re-prefills. Warming with the same *word* count instead produced **50,451
  tuning launches in a prefill graph of 51,179 operators**, ranks disagreeing (51,179 vs
  50,549), coverage reading **2.7%**. The fix took prefill from 51,179 to **1,280**
  operators and coverage to 62.1%.
- **A failed forward writes no graph.** A 101-operator capture of a 64-layer model was a
  crashed forward written from a `finally` block — *"a well-formed artifact from a failed
  run, structurally valid and merely wrong."*
- **What is written is checked against model depth**, since attention runs once per layer.
- **Per-rank artifacts carry every group's coordinates.** A rank-coordinate function that
  reported only `{"tp": rank}` while the topology was TP=2 x DP=2 made **all four ranks
  resolve to one `graph.tp0.json` and write it in turn** — three quarters of the run's
  evidence discarded, the surviving file looking complete at 807 operators with no error.
  *"A single-writer path is indistinguishable from a correct one by inspection."*
- **Write and read sides go through one naming function.** At TP=2 the calibration wrote
  `steps.tp0.jsonl` / `steps.tp1.jsonl` and the run looked for `steps.jsonl`; both workers
  died on a bare `FileNotFoundError` and the manager reported *"Received unexpected
  SHUTDOWN signal from DP rank 0 during initialization"* — no DP in the run, nothing to do
  with initialization. **A single rank cannot expose this**, because at width one no suffix
  is applied.

### Instrument note, to be carried on every artifact

Pricing repeatability, two runs of one graph on one box back to back with identical code:
summed contribution moved **0.96%**, median per signature **1.18%**, **p90 32%**.
High-occupancy signatures repeat to ~1-3%.

**Any residual quoted below about 2% is quoting the instrument, not the model.**

---

## D42. When: offline, JIT, and `--measure`

| Activity | When | GPU? |
|---|---|---|
| machine spec probes | once per (device, software stack) | probe classes 1-3 yes |
| Phase 0 discovery | every run | **no** |
| Phase 1a trace | **JIT on first use of a shape**, then cached | **no** |
| Phase 1b op pricing | offline campaign, or opt-in `--measure` | yes |
| Phase 1c in-situ | offline campaign | yes |
| Phase 2 memory readings | offline, `measure memory`, one start per configuration | yes |

This follows the project's own note: the runner has no modes, every run simulates, trace
happens on demand at first use and is cached, and `--measure` is a separate flag that does
a real GPU run and caches the result.

**A new shape is priced between same-kernel neighbours; anywhere else, and for a new
operator kind, the leaf falls to its analytical law.** Revised 2026-10-18 by the owner's
ruling on [#1489](https://github.com/jgong5/ATOM/issues/1489) and its
[band ruling](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252). A shape
never measured is captured (Phase 1a) and each leaf in it priced by interpolating the
leaf's measured points when its two neighbours launched the same kernel (`09` D60). A
query outside the measured range or between two kernels, or an operator *kind* with no
measurement, is answered by its analytical law (D36), stating that source, and refuses
only without one; the analytical share of priced seconds is counted and reported (`08`
D50.3). Under the prior flat design a new shape meant a new graph meant a refusal, which
is why a fixed-shape oracle cost **+47.9% TTFT** the moment serving varied shapes (3
prefill steps of 16 / 256 / 16,128 tokens costing 38.6 / 151.0 / 177.6 ms, all answered as
~47 ms).

### Two refusals that belong with `--measure`

1. **A `--measure` run can never be an acceptance run.** The instrument changes what it
   measures: measure mode cost ~**11 ms of TTFT on the 27B (4%)**, and a variant that
   synchronised around each forward made the run **33% slower** (TPOT 3.26 -> 4.33 ms;
   fixed by recording CUDA events on the stream and draining with `query()` rather than
   `synchronize()`, which took perturbation from 1.33x to 0.97x). The standing rule
   generalises: *"admission must come from a run instrumented like the one the prediction
   is judged against."*
2. **A price measured during a simulated run is marked as such**, so it is distinguishable
   from one measured in a dedicated campaign.

### And two hygiene rules for any measurement run

- **Never mix profiled and unprofiled numbers.** Being profiled costs ~**0.7-1.05 µs per
  kernel** — **8.1% of a 27B decode step** (951 kernels, +1.002 ms), 8.5% on the 0.6B. A
  profiled run is ~16% slower end to end. A profile is for attribution *within* a step and
  never for an absolute.
- **Never mix machines.** The same unprofiled 27B TP=4 decode step is **9.774 ms on one box
  and 12.365 ms on another — 26%**, larger than the difference between TP=4 and TP=8 on one
  box.

Every GPU command holds both as refusals (`13` D80): it names its GPU and records whether
it profiled and which machine it ran on, and `fit` refuses profiled inputs and inputs from
two machines.

---

## D43. Invalidation

A global fingerprint would force re-measuring everything on a torch bump. Each artifact
records the fingerprint of **its own dependency row**; on load, recompute and compare;
mismatch **refuses** (warn only under an explicit flag).

| artifact \ invalidated by | ROCm / AITER / RCCL | torch | ATOM src | model | device | PTL | host CPU | engine cfg |
|---|---|---|---|---|---|---|---|---|
| `shape_population` | - | - | **X** | **X** | - | - | - | **X** |
| `op_graph` | - | **X** | **X** | **X** | - | - | - | X (level, cudagraph mode) |
| `op_measurement`, `price_list` | **X** | **X** | X | - *(shape-parametric)* | **X** | X | - | - |
| `step_measurement`, `step_linear_law` | X | X | X | X | X | X | - | X |
| step grid bundle (`17` D101) | X | X | X | X | X | X | - | X |
| `host_laws` | - | - | **X** | **X** | - | - | **X** | **X** |
| `memory_readings` | **X** | X | X | X | **X** | X | - | **X** |
| `machine_spec`: capacity | - | - | - | - | **X** | - | - | - |
| `machine_spec`: tokenizer terms | - | - | - | **X** *(tokenizer)* | - | - | - *(T53)* | - |

Consequences worth noting:

- `price_list` does **not** depend on the model once leaves are shape-parametric. That is
  what lets one campaign serve many shapes.
- A `shape_population`'s pricing backend is in its key (D41), not its row: another
  backend addresses another entry rather than invalidating this one.
- PTL invalidates what the device does, except the capacity: it moves the clock, not the
  memory. `memory_readings` still refuse under another PTL state, because their key
  carries the machine-spec digest; a parity run re-measures them.
- `host_laws` time the host, so the host CPU invalidates them and the device does not;
  `step_measurement` and `step_linear_law` hold device time, so the host CPU does not invalidate
  them.
- The memory readings depend on the **library build** as much as the silicon, which is
  why ROCm, AITER and RCCL invalidate them. The +5980 MiB appearing the moment width
  exceeds one is collective buffer sizing; the 926 MiB at TP1 is HIP context plus
  libraries.
- The region-term artifact's row is retired with its kind, 2026-10-18 on
  [#1489](https://github.com/jgong5/ATOM/issues/1489). An entry that still records a
  fingerprint for it refuses at load, as any row the matrix no longer names does.
  `op_measurement` shares `price_list`'s row, since a price list is fitted from them.

### The gate must be verifiable from the artifact, not from the flag

*"A dead gate is worse than no gate."* A prior `PRICE_KERNELS` gate read `WORLD_SIZE`,
which the engine never sets — it spawns its own ranks — so the gate was false in every
worker and per-kernel breakdowns were taken at TP=2 all along. **The proof was in the
artifact: the price list written under the supposedly-off gate carried breakdowns for 164
of its 237 entries.** Worse, an inductor-kernel device fault had been gated "off under
parallelism" on the strength of that gate; five runs then faulted, and each fault was read
as evidence about whatever had changed most recently, because an HSA fault surfaces at the
next synchronise and the signature moves between runs.

So: every gate's state is written into the artifact and checked there.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D36 | Two granularities, step-level and op-level, and a method per term (analytical, declared, measured, estimated) replace the three-way split of cost models; fallback runs only along method inside one granularity. Compass models a device it has been measured on; an input observed on another subject makes a term declared, and doc 05 D26 has no `transfer` probe. The op-level backend is empirical on a measured device, over a concrete IR graph per decode width and prefill (T, B): each leaf priced from a `price_list`, else by its analytical law, with no residual, so a new engine config, layout or shape needs only its missing leaf signatures measured. | 2026-09-18; probe dropped 2026-10-14 (#937); two granularities 2026-10-14 (#952); op level empirical 2026-10-18 (#1489) |
| D37 | The tool tells the user what to measure: a backend that needs measurements plans them beside its own code and owns the coverage check that `compass run --coverage` calls before the run starts; the step grid's planner runs inside `measure steps`, over `shape_population` entries and `record` directories, and the op-level backend's inside `measure ops --plan`, over the steps of record directories from `record` or `trace steps --out`; no single command plans the campaign. Coverage checks use a convex hull or k-NN distance, never a bounding box. | 2026-09-18; revised 2026-10-13 and 2026-10-14 (#851); op-level planner 2026-10-18 (#1489); its input, record directories, 2026-10-18 (#1527) |
| D38 | Phase 0 discovery (shapes only) -> 1a trace -> 1b price -> 1c in-situ (+ one TP2 transfer test) -> Phase 2 memory. Client count and workload are not calibration axes. `13` D80's commands run them: `trace steps`, `trace ops`, `measure ops` and `fit ops`, `record` and `measure steps` then `fit steps\|host`, and `measure memory` for Phase 2, one engine start per configuration whose `memory_readings` a run reads and refuses without; `measure` writes per point, `fit` makes laws, bundles and leaves. Phase 1a captures one concrete graph per decode width and prefill (T, B). Phase 1b times each IR leaf alone from its contract inside ATOM's own forward with a stub model, activations random by spec and weights, KV and metadata from the live runner, by CUDA-graph replay over rotating copies, module-attribute tensors included, whose total exceeds the last-level cache, at points planned from the steps of record directories from `record` or `trace steps --out`, each point recording the kernels its leaf launched, with the warm-over-cold ratio reported and the card's PTL state keyed; MoE leaves on uniform routing; `fit ops` interpolates only between same-kernel neighbours (`09` D60) and declines any other query, which the leaf's analytical law answers, counted. Phase 1c feeds the op-level backend nothing. | 2026-09-18; commands 2026-10-13 (#851); Phase 2 2026-10-14 (#851); Phases 1b and 1c 2026-10-18 (#1489); 1b's points planned from record directories 2026-10-18 (#1527) |
| D39 | The step bench is `EngineCore` without its scheduler: every TP rank's `RecordingModelRunner` started through its runner manager in its order, at any TP width, a step charged to its slowest rank, fed **replayed** batches rebuilt from step records by a versioned rule. One window for both step kinds: device time from an event at the forward's entry to one at its exit, with device work queued ahead, a repeat counting only if the entry event is still pending when the forward returns; whatever is fitted on it states that value kind. Step measurements may not use synthetic shapes, the step grid's excepted (`17` D98). The op bench is ATOM's own forward with a stub model, apart from the step bench: the full model on dummy weights after ATOM's warmup, ATOM building each point's metadata and forward context, and the stub timing each leaf from its contract. | 2026-09-18; revised 2026-10-13 (#851); op bench 2026-10-18 (#1489) |
| D40 | Four classes of communication, priced separately. Scaling is measured per width, never modelled. A collective's measured duration is an upper bound on device work. Collectives are leaves, interpolated over message bytes, measured on a multi-rank op bench that starts every rank of the layout through ATOM's own group initialisation, takes a barrier before each timed replay and measures the union of all ranks' signatures; the step bench is not reused. | 2026-09-18; collectives as leaves 2026-10-18 (#1489) |
| D41 | Artifacts, each with key, digest, fingerprint and provenance. Keys are tuples; source roots are `git archive` digests; resolution names its answer; handed-off artifacts are immutable. `step_measurement`, `step_linear_law` and `host_laws` added, and `shape_population` keyed by its pricing backend. `memory_readings` from `measure memory`, keyed as `step_linear_law` is, or published with every reading assumed for a device never measured. `price_list` and `memory_readings` keyed by the parallel layout, not a scalar width. The region-term artifact retired; `op_measurement` added, one per `measure ops` point, keyed as `price_list` is plus the point; `op_graph` keyed by structure and shape. | 2026-09-18; kinds 2026-10-13 (#851); `memory_readings` 2026-10-14 (#851); parallel layout 2026-10-15 (#165, #537); region terms retired, `op_measurement`, `op_graph` per shape 2026-10-18 (#1489) |
| D42 | Trace is JIT and cached; pricing is offline or opt-in `--measure`. A `--measure` run is never an acceptance run. A new shape is captured, and each leaf in it interpolated between same-kernel neighbours; anywhere else, and for a new operator kind, the leaf falls to its analytical law, counted, and refuses without one. | 2026-09-18; same-kernel interpolation 2026-10-18 (#1489) |
| D43 | Per-artifact invalidation matrix, not a global fingerprint. Every gate's state is written into the artifact and checked there. Rows for `shape_population`, `step_measurement`, `step_linear_law` and `host_laws`, and a host-CPU axis. The region-term row retired; `op_measurement` on `price_list`'s row. | 2026-09-18; rows 2026-10-13 (#851); region-term row retired, `op_measurement` 2026-10-18 (#1489) |
| D38.1 | One recommended flow per half: steps (`measure memory`, `record`, `trace steps`, `measure steps` with the step grid's planner, `fit host` and `fit steps`, `run --coverage`, then grading outside the CLI), each server driven by a client started apart, and ops, after the steps half's `measure memory`, `record`, `trace steps` and `fit host` (`trace ops --population`, `measure ops --plan` over the record directories of `record` or `trace steps --out`, `fit ops` publishing a `price_list` only, then `run` and grading). Phases unchanged; the op bench, ATOM's forward with a stub model, hosts 1b and the step bench 1c, Phase 2 is a plain startup `measure memory` runs. | 2026-09-19; revised 2026-10-13 and 2026-10-14 (#851); ops half 2026-10-18 (#1489); `measure ops --plan` reads record directories 2026-10-18 (#1527) |
| D40.1 | Classes (a) and p2p ride `measure ops` unchanged. Classes (b) and (c) share one new stated-entry-point benchmark inside the same multi-rank op bench, also run by `measure ops`. KV transfer is never priced. Collective prices are labelled upper bounds. | 2026-09-19; one command 2026-10-13 (#851); the multi-rank op bench 2026-10-18 (#1489) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T16~~ | ~~Calibrate the campaign planner's GPU-time estimates~~ — retired 2026-10-13 on #851: no command plans the whole campaign (D37) | needs one campaign |
| ~~T17~~ | ~~Draw the boundary of the standalone `ModelRunner` bench~~ — drawn 2026-10-13 on #851: `EngineCore` without its scheduler (D39) | needs a first implementation |
| T18 | Verify a batch replayed by rule reproduces the recorded step's forward context faithfully, MoE routing included | unverified, and it gates D39 |
| T19 | Decide the artifact store's physical form: directory convention, manifest, or indexed store | no decision taken |
| T20 | Build the stated-node + standalone benchmark for invisible collectives | custom all-gather is the default path and is entirely invisible |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) | it is the recipe's one unproven assumption |
| ~~T22~~ | ~~Analytical laws as their own design topic~~ — **done**, now `10` | — |
| T100 | Compare `measure memory`'s readings under ATOM's `--load_dummy` with a real load's on Qwen3.8-27B TP1 on MI308X; until they match, `measure memory` refuses `--load_dummy` (D38, Phase 2) | needs `measure memory` built |
