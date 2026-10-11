# ATOM Compass — Design Topic 4: Model Capture and the Cost IR

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `02_model_runner_and_cost_backend.md` (the backend plug point this fills),
`03_memory_and_kv_model.md` (the activation term this supplies).

**Scope.** What a model looks like to Compass: how it is captured without a device, what
intermediate representation the cost is computed over, and which parts of a step that
representation cannot see. Covers both tiers of the cost model.

---

## D17. Two tiers, and how they compose

### Problem

A single cost model cannot be both cheap enough to run per step over a 300-second
workload and detailed enough to explain a tile cliff. The project needs both: quick
simulation for exploring a configuration space, and operator-level accuracy for the
acceptance gates.

### Decision

**Two tiers behind one `CostBackend` interface (D11).**

| | tier (a) — coarse | tier (b) — op level |
|---|---|---|
| Input | features `ScheduledBatch` already carries | a symbolic operator graph |
| Form | declared/fitted closed form over `tokens`, `Sum N_Q^2`, `Sum N_Q*N_KV`, `Sum ctx`, rung padding | sum over priced nodes, with a declared per-leaf parameter extractor |
| Construction cost | none | one trace per **structure** |
| Per-step cost | microseconds | target: one layer body, not 2,999 operators |
| Used for | sweeps, structure discovery, M1 plumbing | acceptance cells |

### The composition, which is the point

**Tier (a) is the discovery mechanism for tier (b).** Drive ATOM's real scheduler
device-free with the coarse backend, log the structure key per step, take the distinct
set, and trace exactly those. The step table needed already exists — predict mode records
`num_scheduled_tokens`, `context_lens`, `req_ids` and the scheduler's own decision per
step.

This is also why hand-written admission arithmetic is forbidden. A prior attempt to
re-derive `Scheduler.schedule()`'s Phase 1/Phase 2 plus `_chunked_prefill_size` omitted
`_finalize_prefill_chunk`'s `checkpoint_cut`, the `BlockManager`, preemption and the
pool — which change *which requests coexist*, not just chunk sizes — and so was not
conservative in either direction. Measured against the real classes it under-counted
prefill steps 3 vs 8 on one cell and over-counted width 5 seqs vs 4 on another.

A second consequence: **tier (a) can eventually be *derived* from tier (b)** — take the
symbolic step-cost expression, drop terms below a threshold, and the coarse model has
stated provenance instead of being a separately-fitted thing that can disagree with the
fine one for unknown reasons. Not required for M1; recorded as the intended direction.

### Open issues

- The structure count is expected to be small — prior evidence gives unchunked prefill at
  **319 operators**, chunked prefill at **356**, and decode as **one** structure across
  all eleven CUDA-graph rungs (*"330 operators, 19 distinct kinds, one identical operator
  multiset at every rung"*). But nobody has enumerated it for the hybrid 27B.
- Whether tier (a)'s fitted coefficients and tier (b)'s priced sum should be required to
  agree, and what to do when they do not.

---

## D18. Capture mechanism

### Problem

The IR needs an operator list with shapes, obtained without a GPU, for a configuration
that may never have been run.

### Options

**A. `TorchDispatchMode` + `FakeTensorMode(shape_env=ShapeEnv())` + `torch._C._EnablePythonDispatcher()`.**

**B. `torch.export` + `Dim`.**

**C. Concrete traces at the shapes that actually occur** (decode = the capture ladder,
prefill = the enumerable chunk sizes).

**D. Concrete trace plus shape-family synthesis** — the prior `synth_shapes` rule,
"rewrite any leading dimension equal to the traced token count".

### Evidence gathered

Route A was built and validated on this exact stack (torch 2.10.0+rocm7.2.4). One
symbolic trace at hint T=17, substituted, against a fresh concrete trace at each point:

| T | symbolic prediction | concrete trace | |
|---|---|---|---|
| 2 | 197,632 | 197,632 | exact |
| 17 | 1,745,152 | 1,745,152 | exact |
| 64 | 7,340,032 | 7,340,032 | exact |
| 512 | 117,440,512 | 117,440,512 | exact |
| 4096 | 4,697,620,480 | 4,697,620,480 | exact |

`FLOPs(T) = 256*T^2 + 98304*T`, zero guards installed.

**Route D is unsound, measured.** Two-point fitting mispredicted **3 of 11 dimensions**
at a held-out point: quadratic (`T*T`), **ceil-division (`ceil(T/16)`, 9.5x off — and two
sample points straddling a block boundary hide it entirely, since the fit then reports a
constant)**, and saturating (`min(T, cap)`). All three are load-bearing in paged attention
and chunked prefill. The prior one-point rule is a weaker version of the same thing.

### Decision

**Option A.** `torch.export` is recorded as an unverified alternative in D18.1.

### The four disciplines, all mandatory, all silent when omitted

Three of them keep the *tracing* symbolic and are listed here; the fourth is about
what the **engine around the trace** does to a symbol, needs the FakeTensor material
below to state, and is therefore a section of its own -- "A fourth discipline: the
engine's host arithmetic asks a symbol for a number". A reader who stops at the end of
this numbered list has three of four.

1. **`torch._C._EnablePythonDispatcher()` is not optional.** Without it, `torch.matmul` on
   ndim>2 is a C++ CompositeImplicitAutograd decomposition that calls non-symbolic
   `sizes()`; the ShapeEnv resolves it by installing `Eq(s52, 17)` and the graph is
   concrete from there. No error, no warning. The resulting cost model is **linear where
   the model is quadratic**: **8.44x wrong at T=512, 46x wrong at T=4096.**
2. **Post-trace assertions.** `assert not shape_env.replacements` is the specialization
   detector. Then check every entry of `shape_env.guards` is one you intended, assert the
   output shapes still carry free symbols, and call `shape_env.freeze()` so a later
   accidental guard is an error rather than a silently widened artifact.
3. **Trace at a hint >= 2.** A dimension whose trace-time hint is 1 is **silently
   specialized to a constant**. Decode steps have `num_tokens == 1` per sequence, so
   tracing one yields a fully constant graph with no warning. Trace at >= 2 and
   substitute T=1 afterwards; that evaluates correctly.

Plus one from an earlier era that still holds: **trace at step >= 2.** Triton autotunes on
a kernel's first launch. Tracing step one recorded **90,838 operators** against step two's
**101**; `chunk_fwd_kernel_o` alone appeared **34,269 times**.

### FakeTensor, not bare meta

A `FakeTensor` is a meta tensor plus a device tag, a `ShapeEnv` hookup, stride and
memory-format propagation, and an op cache.

| | `device='meta'` | `FakeTensor` |
|---|---|---|
| `x.device.type` | `'meta'` | **`'cuda'`** |
| Reaches a `register_fake` impl | yes | yes |
| **Symbolic shapes** | **no** | **yes** |
| Mixing with real weights | must convert everything | `allow_non_fake_inputs=True` |
| `__enter__` probes CUDA | no | **yes** |

Symbolic shapes are the decisive reason — `ShapeEnv` attaches to `FakeTensorMode`, and
bare meta puts you back on the unsound route D. The device tag is the practical reason:
ATOM registers its ops at `dispatch_key="CUDA"` (`atom/utils/custom_register.py::direct_register_custom_op`) and
the model and runner branch on device throughout; under meta that code takes paths nobody
runs. The prior work felt this directly — *"meta accepts kernels real devices reject"*
(AITER's fused qk-rmsnorm takes fp16/bf16 only, yet meta traced happily at fp32, so model
dtype has to be pinned from the config, not from torch's default).

**What FakeTensor does not fix: `torch.cuda.*` module-level calls.** The device tag is on
tensors, not on the namespace. This is why a meta ModelRunner was rejected outright —
**62 `torch.cuda.` sites in `model_runner.py`**. Stubs are required:

```python
torch.cuda.is_available  = lambda: True     # must be True, see below
torch.cuda.device_count  = lambda: 1
torch.cuda._lazy_init    = lambda *a, **k: None
torch.cuda.get_rng_state = lambda *a, **k: torch.zeros(16, dtype=torch.uint8)
torch.cuda.set_rng_state = lambda *a, **k: None
```

`is_available()` must report **True**. Reporting False makes `FakeTensorMode.__enter__`
take its `avoid_device_init` path, which calls `_ensureCUDADeviceGuardSet()` and probes
the driver anyway. Measured both ways on a host with a wedged driver: False hangs, True
completes.

Three FakeTensor traps, each of which silently produces a wrong artifact:

1. **Build the real tensor outside the mode, then `from_tensor`.** Allocating inside
   `with fm:` yields an already-fake **static** tensor and `from_tensor` becomes a no-op —
   you get plain `int` shapes with no indication anything went wrong.
2. **Weights need `from_tensor(w, static_shapes=True)`** or they are symbolized too.
3. The torch 2.10 signature is
   `symbolic_context=StatelessSymbolicContext(dynamic_sizes=[...])`, not `dynamic_dims=`.

### A fourth discipline: the engine's host arithmetic asks a symbol for a number

The three disciplines above keep the *tracing* symbolic. They are not enough on a real
engine, and the reason is one line of torch: **`SymInt.__index__` and `SymInt.__int__`
are `guard_int`.** Every host-side use of a step's width — filling a staging buffer's
numpy view, slicing a Python list, checking a staged array's length — asks for a number,
gets the symbol's trace-time hint, and *records the ask as `Eq(s, hint)`*. The graph is
constant from there, with no error and no warning.

Measured on ATOM's decode path for the published 27B: **16 ATOM lines** convert the step's
width to a number during one traced forward — 20 conversions in all — and they are host
fills and slices, nothing else:

| where | lines | conversions |
|---|---|---|
| `aiter_attention.py in prepare_decode` | 1106, 1115, 1121, 1122, 1123, 1131, 1132 | 10 |
| `model_runner.py::ModelRunner.prepare_inputs` | 2468, 2479, 2481 | 4 |
| `model_runner.py::tokenIDProcessor.prepare_input_ids` | 510, 513 | 2 |
| `model_runner.py::ModelRunner.prepare_sample` | 2564 | 1 |
| `backends.py in _mrope_cpu_view` | 398, 400 | 2 |
| `gdn_attn.py in _attach_gdn_decode_metadata` | 1237 | 1 |

A seventeenth sits in `forward_context`'s own `assert_shape_contract`, whose `_rows`
helper takes `int(t.shape[0])`, and an eighteenth solves the width by *comparison* rather
than conversion: `ScheduledBatch.__init__` checks the staged token array's length against
the count. **Repairing them one at a time does not converge** — the two sites this
property was originally recorded at were repaired and `_rows` appeared behind them.

**The conversion is not the defect; the recording is.** A host fill genuinely needs a
number, and the number it needs is the hint, which is the count the engine computed. So a
symbolic capture keeps the conversion and replaces the guard with its own log — every
conversion, with the line it happened on — and asserts that log, as a multiset of
`(line, conversions)`, against a declared set. That last clause is the whole of the
discipline's value and it is the part that is easy to leave out: a log nothing compares
records a conversion at a seventeenth line and says nothing, so the instrument reads as
evidence while behaving as decoration. The capture referenced below declares the set as
`EXPECTED_HOST_RESOLUTIONS` and asserts it at both group widths and both step widths.
**`__bool__` stays untouched**, so a branch on a width still installs its guard and a step
whose shape decides which path the engine takes still records that it did.

Two consequences worth stating, because both were expected the other way round:

- **`copy_to_gpu` needs no change, and neither does `CpuGpuBuffer`.** The specialisation
  recorded there was a *buffer capacity* — `max_model_len // block_size` — being solved
  against the CPU side's constant, and it only existed because that capture symbolised
  every dimension of the staged device tensor. Capacities are engine configuration and are
  not a function of the step. With only the step's width symbolic, the two slices carry
  the same symbol and the copy dispatches with it on both sides.
- **One production line changes**, `_rows`'s `int(t.shape[0])` → `t.shape[0]`.
  `torch.Size.__getitem__` already returns a Python `int` for a tensor with a real size,
  so it is the identity in a served step; it matters only where the size is symbolic, and
  there converting one side of an equality to a number forces the other to become it.

**What the symbol has to be attached to is the batch, not the buffers.** Handing each
staged buffer a bound of its own re-derives widths the engine did not run at. The width
is set on the `ScheduledBatch` and the engine derives the rest — `ForwardMode.decide`
settles both units off it, `prepare_inputs` writes the `cu_seqlens_q` boundary at
`running_bs + 1`, `prepare_decode` uses it as every staged bound. A decode step is one
query row per sequence, so its token count and its sequence count are **one** symbol; two
would have to be equated later, which is the specialisation again by a longer route.

**It does not survive the batch's own constructor, and that is the eighteenth site.**
`ScheduledBatch.__init__` compares the staged token array's length against the count it
was handed, which solves the width by comparison rather than by conversion — the
`__bool__` boundary this discipline deliberately leaves alive. So the capture builds the
batch at the concrete width and rebinds the four count fields afterwards. The symbol
enters the engine's *staging*; it does not enter the engine's batch constructor, and a
reader who takes "the symbol enters at the `ScheduledBatch`" literally will look for it
in the wrong place.

**The evidence that the symbol is free, rather than a hint in disguise.** The engine
computes its host values from the hint, so a graph built this way would still be a graph
about one width if any dimension had taken the hint instead of the symbol — and nothing in
a census would say so. The step is therefore traced at **two** widths, at **each** group
width the result is claimed at, and the inventories compared operator for operator and
shape for shape with the symbol's name set aside. They are identical, which also says that
every dimension that stayed a number is the same number at both widths and so is not a
width in disguise. The digest that carries this covers operator names and tensor shapes
only — not scalar arguments, dtypes or strides — so a value-level difference is found by
comparing the distinct-operator sets instead, which is how the one difference between the
concrete and symbolic passes (`lift_fresh` becoming `scalar_tensor`) was found.

**One artifact is width-dependent, and the comparison should say so rather than omit it.**
The guard set is not the same at two hints: at every hint but 2 a lower bound
`<axis> + 1 > <hint>` appears as well, measured at hints 3, 8 and 16 at TP1
and at hint 8 at TP2, and seen independently at 5, 7 and 64. At a hint of 2 it is
elided, because a size symbol's default range is
`[2, ∞)` and the inequality is vacuous. It is a `__bool__` comparison on a live symbol, so
it is the positive evidence that the boundary is intact — but it means the applicability
statement carries a *lower* bound at any hint but 2, and a two-width table that lists
five rows as identical has to name the sixth that is not.

Pinned by `tests/compass/test_capture_real_model.py`.

### The gating cost turned out to be small

An operator with no fake/meta impl is a **hard stop**, not a degradation:
`RuntimeError: There was no fake impl registered for <CustomOpDef(...)>`.

aiter has **314 `@compile_ops` decorations and only 83 with `gen_fake`**. But **every
ATOM-registered opaque op already carries a `fake_impl`** — `moe_forward`,
`unified_attention_with_output_base`, `linear_attention_with_output_base`,
`maybe_dual_stream_forward`, `v4_attention_with_output`, `v4_attn_compress`,
`v4_qk_norm_rope`, `indexer_score_topk`, `tbo_all_reduce`, both `topK` ops, both
`kimi_k3` ops. ATOM had to write them for Dynamo; Compass inherits them.

That composes with D20: **the trace stops at the opaque leaf**, which is where 60-75% of
the step time lives and exactly where a parameterized price is wanted anyway. The ~231
fake-less aiter ops sit *inside* those leaves and are never reached.

### Collectives at TP>1 — measured, and the group is not what the trace needs

> **Provenance — measured under a capture that has since been withdrawn.** Every count in
> this section was taken with the fake-tensor capture module that PR #10 added under
> `atom/compass/capture/`, driven by scripts that were never committed and writing JSON
> records that no longer exist. That module and its tests have been withdrawn from the
> tree, so **none of the numbers below was reproducible here**. They are kept because they
> are the specification a replacement capture is written from, not because they can be
> re-run; anything that builds on them re-takes them first.
>
> **Partly re-taken since, by `tests/compass/test_capture_real_model.py`.** That test
> traces the 27B at both widths through a group of the honest width, and agrees with the
> 27B row on what it is a claim about rather than on its totals: at TP2 it records
> `aiter.all_reduce_` **129** — 128 row-parallel at `communication_op.py::tensor_model_parallel_all_reduce` plus the
> vocab-parallel one at `embed_head.py::VocabParallelEmbedding.forward`, the 128 predicted from the config's layer
> types and not read off the inventory — one all-gather at `embed_head.py::ParallelLMHead.forward`, and one
> broadcast, plus two `_c10d_functional.wait_tensor` entries that belong to the
> substitution below rather than to ATOM. The raw Triton traffic agrees exactly:
> **33 launches across 3 kernels**. The operator totals do
> **not** match and are not expected to: they are 2,521 at TP1 and 2,662 at TP2 against
> 2,471 and 2,611 here, on a decode step of two sequences at a block size and a batch
> budget this file never recorded, which is the reason a total is not the assertion.
>
> **The TP2 inventory is conditional on two declared substitutions, and every number in
> the paragraph above inherits them.** `ATOM_USE_CUSTOM_ALL_GATHER=0` selects the
> non-custom vocab-parallel gather: it is a **non-default** ATOM path, and a default TP2
> deployment takes the `ca_comm` custom gather, which is not what was traced — with the
> default the forward dies on `ca_comm` after roughly 2,500 operators, so this is the
> difference between a run that refuses part-way and one that completes. Second, the four
> call sites reaching `c10d`'s legacy in-place collectives are routed to their functional
> forms, which is where every `_c10d_functional.*` entry, including both `wait_tensor`s,
> comes from; only the 129 `aiter.all_reduce_` are ATOM's own dispatch.
>
> **The gather's shapes — both arrangements, each read off the live call.** ATOM hands
> `all_gather_into_tensor` an output buffer of `(world_size,) + input_size`, measured as
> `[2, 2, 124160]`; the functional substitute concatenates the `[2, 124160]` input along
> dim 0 into `[4, 124160]`, and the shim reshapes that into ATOM's buffer. `[4, 124160]`
> is therefore the **substitute's** arrangement and not the 27B's. An earlier revision of
> this paragraph gave it as the 27B's output shape, and gave it from the width rather
> than from any record: the dispatched operator carries only its input, so nothing
> recorded a destination shape at all until the test began recording both.

The open issue *"whether ATOM's real model classes trace cleanly under this mode at
TP>1"* is **answered yes**, on two models at both widths, with the collectives in the
inventory rather than substituted away.

> **These are DIAGNOSTIC inventories.** Every run below was taken with raw
> `@triton.jit` launches recorded and **not executed** -- 33 launches across 3 kernels on
> the 27B decode -- so anything downstream of a skipped kernel read uninitialised fake
> memory, and each record carries `diagnostic_inventory: true`. The operator and
> collective counts are an enumeration of what a step reaches. They are **not** a cost
> model input at any width.

| model | TP1 | TP2 | collectives recorded at TP2 |
|---|---|---|---|
| Qwen3.8-27B (hybrid; vision tower, linear attention) | 2,471 ops / 33 distinct | 2,611 / 38 | `aiter.all_reduce_` **129**, functional all-gather 1, broadcast 1 |
| Qwen3-0.6B (dense MHA) | 389 / 17 | 457 / 23 | `aiter.all_reduce_` **57**, functional all-gather 1, broadcast 1 |

Both widths complete a full decode step. The 27B's TP1 inventory is identical
operator-for-operator to the TP1 record taken through `apply_simulated_tp`, so the TP2
difference is attributable to width rather than to a different capture. The 0.6B's 57 is
28 layers x 2 row-parallel linears plus 1 vocab-parallel reduce, which its parameter
geometry predicts independently. A third model with a different collective pattern,
Qwen3-30B-A3B, was attempted and is not reachable on this stack: `AutoConfig` rejects the
checkpoint before any capture code runs.

**Why no process-group substitution is required.** The collective ATOM issues at TP>1
goes through a **registered custom operator with a registered fake implementation** —
aiter's `all_reduce_`. Under `FakeTensorMode` the fake answers and the body that needs a
device communicator is unreachable, so the operator is recorded with its real shapes
having allocated and communicated nothing. It does not need the group to exist at all:
called with a group name that does not, it still returns the input's shape.

**The exception, which is not the group either.** Four call sites reach
`torch.distributed`'s legacy entry points instead — the non-custom `all_gather`,
`gather`, `broadcast`, and the `barrier` in `allocate_kv_cache`. On this stack those
`c10d::*` operators carry a backend kernel and **neither a `Meta` nor a
`CompositeExplicitAutograd` one**, so `FakeTensorMode` raises
`UnsupportedOperatorException` rather than producing a meta operation. The functional
forms do carry a kernel it can run, and give the same shapes — a `[2, 124160]` input
gathers to `[4, 124160]` as one recorded collective. This is the same fact as the
`c10d.broadcast_` gap, measured to be general rather than particular to `broadcast`.

A width-N group inside one process is available from torch's own `fake` backend and needs
no peer. What one process cannot build is the **transport**: the device communicator opens
a collective rendezvous that waits for absent ranks, as does the message-queue
broadcaster, so both have to be declined when the group is built.

Declining the device communicator is **not free**, and the earlier claim that nothing
reaches it once the group exists is wrong. ATOM's default `ATOM_USE_CUSTOM_ALL_GATHER`
takes `embed_head.py::ParallelLMHead.forward`'s vocab-parallel gather down the custom path, which asserts on
`device_communicator.ca_comm` and fails with `'NoneType' object has no attribute
'ca_comm'` after 2,588 operators. The runs above therefore set
**`ATOM_USE_CUSTOM_ALL_GATHER=0`**, selecting the non-custom gather; that is a declared
configuration of the capture and belongs in the record beside the device readings. It is
the whole difference between the run that fails at 2,588 and the one that completes at
2,611.

The test that enumerated both operator sets by name — so that a future torch growing a
meta kernel for one of the legacy forms would fail there rather than leave unexplained
code behind — was withdrawn along with the capture module it exercised. Nothing on this
tree holds either set in place today.

### Open issues

- Whether any op on the forward path **outside** an opaque leaf lacks a fake impl. One
  class of them is now measured -- the legacy `c10d::*` collectives above. The rest is
  expected to be small; unmeasured.

---

## D18.1. `torch.export` — recorded as an unverified alternative

Not chosen. Recorded because it is genuinely attractive and the reasons against it are
specific enough to re-examine later.

**What it would give**

- **Refuses rather than silently specializing.** `UserError: Constraints violated (T3)! ...
  Suggested fixes: T3 = Dim('T3', max=128)`. On the 0/1 case: *"You marked Tg as dynamic
  but your code specialized it to be a constant (1)"*. This is the opposite failure mode
  from route A's, and the safer one.
- The Python dispatcher is enabled for you, so the 46x trap cannot be forgotten.
- `node.meta['val']` is documented per-node shape/dtype metadata; `range_constraints` is a
  documented domain readout; `unbacked_bindings` handles data-dependent symbols.
- An `ExportedProgram` is a serializable, diffable, versionable artifact.
- The FX graph **is** a def-use graph, which is most of what the activation liveness walk
  needs to reconstruct by hand.
- `run_decompositions()` works on this stack once the `torch.cuda` stubs above are in
  place, and custom ops **survive** decomposition as single opaque nodes. Two structurally
  different readouts of the same exported program produced identical FLOP expressions.

**Why it was not chosen**

1. **Decomposition changes the thing being priced.** The objection raised by the project
   owner, and it is sound in two ways. It adds a third representation on top of the
   existing eager-vs-compiled gap, with its own fidelity to establish. More seriously, it
   can **shatter an opaque leaf**: custom ops survive, but aten composites do not, so
   `aten.linear` lowering to `mm` + `add` would price at a different granularity than the
   fused hipBLASLt GEMM that actually runs. The whole leaf-pricing design rests on the
   trace boundary coinciding with the tuned-kernel boundary, and decomposition moves it.
2. **It captures a model call, not an engine step.** ATOM's step is `ModelRunner.forward`
   -> `prepare_model` (attention metadata build, H2D staging, `ForwardMode.decide` with a
   cross-DP all-reduce) -> `run_model` -> `postprocess` (sampler, rejection sampler,
   logprobs). The prior work found tracing had to go **through the runner, not the model**,
   because `model(input_ids, positions)` fails at `fwd_ctx.context.is_dummy_run` —
   attention reads a forward context only the runner establishes.
3. Dynamo must get through the model end to end; a graph break is a hard failure rather
   than a degradation.
4. Stream annotations are not in FX metadata, and whether Dynamo traces stream ops is
   unverified. Route A can record `torch.cuda.current_stream()` per node directly, which
   D19 needs for `Par`.

**When to re-examine:** if route A's three silent-failure disciplines prove hard to keep,
or if the hand-written def-use reconstruction becomes a maintenance burden.

---

## D19. The IR

### Problem

A flat operator list is the obvious IR and it fails three ways. The prior work's flat
`OpGraph` held **2,999 operators** for the 27B, pricing cost ~39 ms per step summing
~2,440 operator prices, and it could only answer for the exact shape it was traced at.

### Decision

**Hierarchical, symbolic, stream-annotated.**

```
Graph   := (applicability, Region)
Region  := Seq[Node] | Repeat(body: Region, count: int) | Par([Region], join: JoinPolicy)
Node    := Op(name, kind, in_shapes: [SymExpr], out_shapes: [SymExpr],
              attrs, stream_id, context_ref)
```

### `Repeat` — the efficiency property

An LLM forward is a prologue, N layer bodies, and an epilogue. `Repeat` prices the body's
operators once and **reuses those prices** for every instance — the same prices, in the
same order, added the same way. What is reused is the body's sequence of per-operator
prices, re-emitted in order once per instance, and not a body total; it does not multiply
a body price by the count. Float addition is not associative, so multiplying re-associates,
and it reproduces the recorded price in **none of the three shapes measured**: eight
identical layers, 3.2e-05 s multiplied against 3.200000000000001e-05 s recorded; a
four-block pattern repeated twenty times, 0.00036 against 0.0003600000000000009; six
instances of 0.1 s, 0.6000000000000001 against 0.6. Reuse reproduces all three exactly.

For Qwen3.8-27B that is 64 layers collapsing to roughly two bodies —
**48 `linear_attention` and 16 `full_attention`**, `full_attention_interval: 4` — plus
embedding and the head.

This is the direct attack on the per-step replay cost, which is what actually threatens
the >=5x speed gate. Measured previously, device-free, against a 32.7 ms modelled step:
41.7 ms uncached, ~39 ms of it summing prices; 2.3 ms with a shape-keyed cache (**and
unsound**, see below); 4.3 ms once the key carried the bound allocation.

> **Why a shape-only price cache is unsound.** `signature_of` reads each operator's
> `context`, and `context` is where the binder writes this step's `slot_mapping` and
> state indices. A second valid allocation for the same shape moved **64 of 2,439**
> signatures and took a step from 32.667 ms over 2,424 priced operators to 28.360 ms over
> 2,376 — 48 operators priced under one allocation are unpriced under the other. A
> shape-only cache answers the first number, with a complete-coverage claim, for a step
> that is neither.

**`Repeat` must be validated, not assumed.** Derive flat, group, and compare the two forms
**term by term, in order** — not total against total. A step cost is reported as a
breakdown, one row per term, so individual prices are read downstream and not only their
sum; two forms agreeing on the sum while disagreeing on a term disagree in what gets
reported. Measured: one instance priced one bit above the others left the two forms
bit-identical in total, so the elementwise comparison refuses that grouping and a
total-only one accepts it. Layer 0 often differs structurally; per-layer quantization
scales differ without differing in cost; a hybrid's layer types interleave rather than
block.

### `Par` — concurrency

The concurrency taxonomy in ATOM is exactly three shapes and only one belongs inside a
graph:

| Shape | Example | IR treatment |
|---|---|---|
| compute \|\| compute, fork/join, graph-captured | dual-stream MoE; V4 compressor + indexer | **`Par`** |
| compute \|\| collective, two micro-batches, CPU-baton-serialized compute | TBO | **not** a `Par` — two graphs interleaved, modelled one level up |
| step-crossing async | D2H sampled ids, PP isend, KV offload | not a node; a latency annotation at the step boundary |

`JoinPolicy` has three values, because `max()` is wrong in two of three regimes:

- **`max`** — branches small, device not saturated. The V4 source states its own case:
  *"side streams ~25us, main Q/KV chain ~87us"*, so the join is free.
- **`resource_bound`** — both branches compute-heavy: `max(t_a, t_b, (w_a + w_b) / peak)`.
- **`exclusive`** — MORI dispatch/combine caps `block_num` at `get_cu_num()` because it
  uses a **grid-wide spin barrier requiring every block co-resident**. It occupies the
  whole device; concurrent work does not overlap it, so cost adds. (On an 80-CU MI308X,
  launching 128 blocks deadlocks.)

**The tracer has to earn a `Par`.** `torch.cuda.stream()` and `wait_stream()` are not
dispatcher events — a dispatch trace records a *serial* operator list for work the device
runs concurrently. So the tracer records `torch.cuda.current_stream()` as a field on every
node and hooks `Event.record` / `wait_event` / `wait_stream` for the edges. `Par` is then
**derived** from stream ids rather than declared.

**Staging.** In M1 and in M2's first gate every node carries one stream id and `Par`
never materialises: Qwen3.8-27B is dense-MLP (no `num_experts`, no `shared_expert`) so
dual-stream MoE cannot fire; it is not MLA so there is no metadata `prep_stream`; TBO
defaults off and needs `--enable-tbo` plus `--enable-dp-attention` plus >=2 GPUs; and PP
asserts `enforce_eager` so it cannot coexist with CUDA graphs. **DeepSeek-V4-Pro populates
`Par` in M2**, on both nodes of each nightly 1P+1D cell in `recipes/mesh/DeepSeek-V4.md`.
Only the DPA cells pass `--enable-dp-attention`, and only their prefill node adds
`--enable-tbo`.

- **Decode, in every cell, runs without TBO.** Inside the replayed graph,
  `DeepseekV4Attention.maybe_compressors_async` forks the compressors onto `alt_stream`
  and `indexer_stream`, and `MoE` runs the shared expert on `alt_stream` against the
  routed experts whenever this rank's tokens <= `ATOM_DUAL_STREAM_MOE_TOKEN_THRESHOLD`
  (1024 by default) — i.e. **every decode step** at the cells' concurrencies. The shared
  expert stays unfused because it is FP8 and the routed experts FP4.
- **Prefill never forks the compressors**: it runs eager, and the compressor fork needs
  `fc.in_hipgraph`, which only decode graph capture sets. A prefill step is one of:
  - **a TBO step** (DPA cells only), taken when some DP rank reaches
    `ATOM_TBO_PREFILL_MIN_TOKENS` (8192 by default) and every rank can split
    (`local_tbo_precompute`, `atom/utils/tbo/ubatching.py`): two microbatches alternate
    between a compute and a comm stream joined by events, and `tbo_active()` keeps the
    shared-expert fork off;
  - **any other step with at most 1024 tokens on this rank**: `maybe_dual_stream_forward`
    forks the shared expert in eager mode, a `Par` on the prefill node;
  - **any other step**: one stream.

So `Par` is defined now and populated at M2, from its first DeepSeek-V4-Pro step.

### Symbolic shapes — what they buy

1. One trace per structure covers every shape in the family **exactly**, replacing an
   unsound synthesis rule.
2. **The cache key becomes the structure, not the shape**, so the ~0.16 s meta forward
   runs a handful of times per model rather than per step.
3. **The cost becomes a closed form.** If each node's cost is a function of its shapes and
   the shapes are expressions in `T`, `B`, `Ctx`, `TP`, then `step_cost(T, B, Ctx)` is one
   expression, simplified once and evaluated in microseconds.

Three things symbolic shapes do **not** buy, stated so nobody expects them:

- **They do not make cost smooth.** GEMMs cliff **2.4x between M=768 and M=1024**
  (`MT256x1...` -> `MT256x2...`), and prefill attention sits on plateaus quantised in
  sequence length with a step of ~2.27 us/token at `ceil(L / 2530)`. A cost function over
  a symbolic shape must be piecewise and dispatch-band aware.
- **`Ctx` is a multiset, and row order is an undeclared treatment.** At one fixed 32-row
  context multiset, row **order alone** moves measured decode attention across **1.77x**
  (grouped-descending 276.1 us, grouped-ascending 292.2 us, alternating 376.5 us,
  ladder-interleaved 489.5 us). KV-pool footprint, co-residency, shape and launch-wave
  packing were all refuted as causes. Every candidate feature is a function of the
  multiset and therefore blind to this. A corpus that mixes orderings is fitting an
  undeclared treatment.
- **Data-dependent shapes stay unknown.** MoE per-expert token counts depend on the
  router. `ShapeEnv.allow_dynamic_output_shape_ops` defaults True so they trace as
  unbacked symbols (`u0`), and branching on one raises rather than guessing
  (`GuardOnDataDependentSymNode`), with `guard_or_false` / `torch._check` as the declared
  escape hatches. A balance assumption has to be declared; ATOM's `fake_eplb` is the hook.

### Applicability: no branches in the IR

A trace is a straight-line record of one path. Its guards declare where that record is
valid, so branch modelling lives **outside** the graph as an applicability predicate, in
two parts:

| Part | Matched by | Covers |
|---|---|---|
| **discrete key** | equality | `ForwardMode`, `has_cached`, `produces_output()`, spec width, replayed vs eager, `tbo_on`, attention backend, `is_dummy_run` |
| **symbolic domain** | evaluation of `shape_env.guards` / `var_to_range` | anything branching on a shape |

Guards only capture branches on symbolic shapes, which is why the discrete key is needed
as well. ATOM already emits most of it: `atom/model_engine/run_labels.py` builds
`prefill[bs=... tok=... ctx=...]` / `decode[bs=117/128 ... spec=...]` and appends `tbo=1`
driven by `forward_context.ubatch_slices is not None`.

Measured example of the domain readout, for `if x.shape[0] > 128`:

```
hint T=17   -> guards ['s52 <= 128'],  var_to_range {s52: VR[2, 128]}      # decode path
hint T=512  -> guards ['s52 > 128'],   var_to_range {s52: VR[129, int_oo]} # prefill path
```

At predict time Compass evaluates these itself: substitute the step's bindings into each
guard; all true means the graph applies; any false means **refuse, naming both the guard
and the binding**. For ~5 structures of ~3 guards each this is microseconds.

**One rule that belongs in the code comment:** evaluate with `sympy.subs` or
`shape_env.size_hint()`, **never** `int()` or a Python comparison on a SymInt — the latter
installs a guard, so the act of checking would mutate the artifact. The stated torch
contract is that semantics stay symbolic and hints are for policy only.

This is strictly better than modelling branches in-IR: the predicates are produced rather
than authored, so they cannot drift from ATOM's control flow, and a step outside every
domain is detected rather than mispriced.

### `Repeat` must nest, and must tolerate a non-contiguous pattern

The obvious form of `Repeat` — one layer class, N identical instances, contiguous — is
what a dense decoder looks like and not what real models are. They break it in two
distinct ways, and the IR has to carry both or it silently flattens back to a linear `Seq`
and loses the compression that makes symbolic pricing cheap.

**Way 1 — prologue and epilogue.** The first and last layer routinely differ: a dense
first layer in an otherwise-MoE stack, a different attention variant on layer 0, a final
norm fused into the last block. These break a naive run-length scan over the whole body.

**Way 2 — hierarchical and non-contiguous repetition.** A model may interleave layer
classes on a period, and repeat *that period*. Concretely:

```
sub-pattern P  =  LayerClassA x3  then  LayerClassB
global pattern =  P x20
```

Nothing about this is exotic — it is how hybrid attention/SSM stacks, MoE-every-k-layers
schedules and shared-expert variants are laid out. A flat run-length encoder sees
`AAABAAAB...` and produces either 80 singleton groups or 20 groups of 4, never the
nested form, and in both cases the `Repeat` body it emits is no longer a single
priceable structure.

**The IR change is small; the detection is the work.** `Repeat` already takes a *body*,
so the representation is sufficient the moment the body is allowed to be a composite and
`Repeat` is allowed to nest:

```
Seq[
  Block(embed),
  Block(layer_0_dense),                    <- prologue, a plain sibling
  Repeat( n=20, body = Seq[                <- outer period
            Repeat( n=3, body=Block(A) ),  <- inner run
            Block(B),
          ] ),
  Block(layer_last),                       <- epilogue
  Block(norm), Block(lm_head),
]
```

Three rules make this well-formed rather than merely expressible:

1. **A `Repeat` body is any node**, including a `Seq` and including another `Repeat`.
   No arity or depth limit; in practice depth 2 covers everything seen.
2. **`Repeat` carries the index binding it varies over**, so a body whose cost depends on
   layer index (KV cache offsets, per-layer expert counts, a sliding-window pattern that
   changes period) prices per instance rather than being assumed uniform. Without this,
   nesting is a lie: `Repeat(20, P)` claims 20 identical `P`s.
3. **Grouping is an optimisation and must be provably free.** `Repeat` is only emitted
   where the flattened and grouped forms price identically, term by term; otherwise the
   node stays a `Seq`. This is already **T6**, and the nested form makes it load-bearing
   rather than a nicety — a wrong nesting is a systematic error multiplied by the repeat
   count.

**Detection algorithm.** Bottom-up run-length encoding over a *canonical block
signature*, not over the module name:

1. Give every top-level block a signature = the hash of its leaf op sequence with
   symbolic shapes, layer index excluded. Two blocks with the same signature are
   interchangeable for pricing.
2. Run-length-encode the signature string. This finds Way 1 for free (the prologue and
   epilogue are runs of length 1 and simply stay as siblings) and finds contiguous runs.
3. Re-encode the *resulting* symbol string. `AAAB AAAB ...` becomes `(A³B)` at step 2
   and `(A³B)²⁰` at step 3. Iterate until a pass changes nothing — bounded by the depth
   of real nesting, which is small.
4. Emit `Repeat` only where step 3 of the rules above holds.

This is standard run-length composition, it is device-free, it runs on the traced
signature string rather than on tensors, and it costs microseconds on an 80-layer model.
The risk is not cost, it is a *near*-miss: two blocks whose signatures differ only in a
constant the pricing ignores will fail to group, costing compression but never
correctness. That is the right way round.

**Recorded as T51:** enumerate the actual layer-pattern shapes for the two target models
(Qwen3.8-27B, DeepSeek-V4-Pro) and confirm the detector reaches the nested form on both.
DeepSeek-V4-Pro is the one that matters. In `atom/models/deepseek_v4.py` every layer is the
same `Block` with an `MoE` and one shared expert, so its shape comes from per-layer
switches: `MoE` routes by hash on layers below `n_hash_layers`,
`DeepseekV4Attention` picks CSA or HCA from `compress_ratios[layer_id]`, and `Block` runs
layer 0's mHC unfused (`enable_fused_hc` is false on layer 0 only, so `Block.fuse_hc` calls
`mhc_post_pre` there and `mhc_fused_post_pre` on every later layer). The DeepSeek-V4-Pro
checkpoint's `config.json` makes the hash-routed layers HCA, HCA, CSA, then alternates HCA and CSA to
the last layer. With the unfused layer 0, no two hash-routed layers are the same
block: a prologue (Way 1) before a period-2 interleave (Way 2), in which a flat
run-length scan finds nothing to group.

### Open issues

- Whether `resource_bound`'s `peak` should come from the device spec's derated compute or
  be fitted.
- Nothing yet validates that a `Par` reconstructed from stream ids matches the real
  fork/join structure.

---

## D20. Opaque leaves

### What they are

An operator the tracer records as **one node whose internal kernels are invisible**,
because ATOM deliberately registers a whole subsystem as a single dispatcher op. The
intent is stated in the `atom/model_ops/module_dispatch_ops.py` module docstring: hide dynamic-shape
internals from Dynamo while leaving CUDA-graph capture transparent. The same property
makes them opaque to any trace.

There are ~19 ATOM-registered ones. The ones that matter: `moe_forward`
(`atom/model_ops/moe.py`), `unified_attention_with_output_base`
(`atom/model_ops/base_attention.py`), `linear_attention_with_output_base`
(`atom/model_ops/base_attention.py`), `gemm_a16w16` (aiter), and the `v4_*` family.

### Why this is good news

**Concentration is extreme.** Sixteen distinct operators cover a step, with half the time
in two — `aiter::gemm_a16w16` at **33.7%** and `aiter::unified_attention_with_output_base`
at **23.8%** — and **twelve kinds cover 99.7%**. Roughly **60-75% of step time sits in
8-20 opaque leaves**.

So the IR's job is not decomposition. It is **pricing ~20 leaves well**: bounded,
enumerable, and reviewable.

### The load-bearing consequence: cost is not a function of arguments

Opaque leaves take `layer_name: str` and look the module up in
`compilation_config.static_forward_context`. The tensors that decide cost —
`block_tables`, `context_lens`, `slot_mapping`, `cu_seqlens_q`, `max_seqlen_q/k` — are
**ambient**, not arguments.

A prior attempt to fix this by widening the operator's signature was **reverted**, and the
reason generalises:

> An op graph records arguments. Arguments cannot carry non-tensor ambient state through
> `torch.compile`. Therefore an op graph cannot, on its own, describe an operator whose
> cost depends on non-tensor ambient state, and no change to that operator's signature
> will make it able to.

Specifically: `md = get_forward_context().attn_metadata` is traced by Dynamo. Tensor reads
become graph inputs and flow per step; everything else is a compile-time constant read
once from whichever forward triggered compilation — `max_seqlen_q` because it is an int,
`block_tables` because it happened to be `None` in the warmup dummy.

**So each leaf needs a declared parameter extractor:**

```
(forward_context, symbol_bindings) -> cost parameters
```

A ~20-entry hand-written table. It is the principal hand-authored asset of this design and
should be visible as such rather than buried in the tracer.

### What getting it wrong costs, both directions, measured

- **7.1x over.** Attention priced at **163.6 us/call against a true 23.0 us**. Cause:
  `unified_attention_with_output_base` rebuilds a forward context from its arguments *only
  when there is not one already*, and after `capture_cudagraph()` there always is —
  `set_forward_context` assigns a module global and nothing clears it. What stood was the
  last rung of the descending capture ladder: **batch 1 at the model's maximum context**,
  so the kernel was handed `context_lens=(1,)[16384]` while four sequences of 155 tokens
  were passed. **26x the KV traffic.** The 163.7 us was real device time in a real kernel.
  Fix: reset the forward context **per signature**, not once, because an operator that
  rebuilds it leaves it behind for whatever is priced next.
- **~100x under.** `linear_attention_with_output_base` priced at **5.793 us/call against
  0.567 ms in situ** — 0.277 ms against 27.214 ms over 48 layers. `attention_gdn.py` opens
  with `if gdn_metadata is None: core_attn_out.zero_(); return`. **The benchmark was timing
  48 `zero_()` calls.** Correcting it moved priced prefill kernels from 296.999 ms to
  **319.310 ms against 318.351 ms in situ — from -6.7% to +0.30%** — and explained a sign
  puzzle: the 0.6B priced 2.28% high and the 27B 6.7% low, not two models disagreeing
  about pricing but one model with an entire layer type missing from its cost.

The second case needed two things beyond copying the first: the recurrent/convolution
state is **not recorded** (it lives in `kv_cache_data`, set once at start-up, so it is
already real in the pricing process), and the convolution metadata is **recomputed, not
recorded** — `nums_dict`, `batch_ptr`, `token_chunk_offset_ptr` are a pure function of the
recorded query start offsets, so the installer calls the engine's own
`compute_causal_conv1d_metadata`.

### They also bound what symbolic buys

Inside a leaf, shapes stop being symbolic: the fake impl gives the output shape, not the
internal work. The leaf's cost model is its own thing — a fitted law, a measured price
table with dispatch bands, or an analytic roofline. **Symbolic shapes feed it; they do not
replace it.** And the boundary is the right one, because a leaf is also where ATOM's own
autotuning happens — one leaf is one tuned kernel family.

### FlyDSL and MORI resolve here

Both are opaque leaves *below* an opaque leaf — they live inside `moe_forward`. They add
**no IR nodes**, and **FlyDSL never needs intercepting** unless one wanted to decompose
`moe_forward`, which this design does not.

Recorded for completeness, since it would otherwise look like an unexamined hole: FlyDSL
is a standalone Python-embedded **MLIR DSL and JIT compiler** (`flydsl` 0.3.1), not part
of aiter and not Triton. The path is `python -> ctypes fnptr -> MLIR-generated native host
code -> mgpuLaunchKernel -> HIP`; arguments are erased to `data_ptr()` before the call, so
even a tensor-subclass trace would see nothing. aiter's integration has **96 `@flyc.kernel`,
132 `@flyc.jit`, 87 `.launch(...)` sites**, and **one `JitFunction.__call__` is 1 to 12
kernel launches** (`moe_sorting_kernel.py` has 12). Under `--moe-backend mega` a single
dispatcher event covers the entire EP-MoE step: dispatch + GEMM1 + quant + GEMM2 + combine.
If it ever *does* need intercepting, the chokepoint is
`flydsl/compiler/jit_executor.py:210` (`CallState.__call__`) for both entry paths, with
`jit_function.py:1357` and `:1632` for named arguments.

### Open issues

- The parameter-extractor table does not exist and is the main hand-written work item.
- Leaf prices are not repeatable to better than about 1%: two runs of one graph, one box,
  back to back, identical code moved the summed contribution **0.96%**, median
  per-signature **1.18%**, **p90 32%**. **Any residual quoted below ~2% is quoting the
  instrument.**

---

## D21. Operator coverage

### Problem

An operator whose cost is invisible is an invisible cost. But coverage-by-count is a
misleading metric, and chasing it made the prior model *worse*.

### What each interception mechanism bought

| Added | Coverage of a 0.6B decode step |
|---|---|
| dispatch mode, tensor arguments only | 54.8% |
| + scalars in the signature | 90.3% |
| + attention via recorded forward context | 98.8% |
| + raw Triton (`JITFunction.run`) | 99.4% |
| + inductor-generated (`CachingAutotuner.run`) | 99.7% |
| + profiler operators excluded from the graph | **100%** on chunked prefill |

### The trap

On a hybrid model **~34% of recorded operators are Triton launches** the dispatcher cannot
see. The reflex is to call that 34% of the model missing. **It is not.** Every one of
those 433 launches in a 27B prefill graph happened *inside* a dispatched operator and was
a duplicate of work already in that operator's price. Recording both **double-charged** —
attention priced 59.958 ms where its in-situ kernels are 47.382 + 5.883 = 53.265 ms.
Dropping them moved priced kernel time by **1 ms out of ~296 ms** while raising
coverage-by-count 25 points.

The rule: **record a Triton launch only when no dispatched operator is on the stack**
(`inside_an_operator()`).

Similarly, `record_function` dispatches `profiler::_record_function_enter_new` / `_exit`,
which run no kernel and take arguments no artifact can hold. Executing them without
recording them took chunked prefill from 365/366 to **364/364**.

### What is genuinely absent

| | cost |
|---|---|
| inductor-generated graph regions, decode | **~4.8% of a step** (112 kernels, 0.465 ms) |
| inductor-generated graph regions, prefill | ~0.6% |
| EP dispatch/combine at any degree | not separable — inside `moe_forward` |
| CUDA-graph replay internals | unobservable in principle |

The 8x prefill/decode spread is structural: the per-layer generated-kernel count is fixed
and a decode step is ~30x shorter.

### Decision: declared nodes, and refusal over silence

1. Do not chase 100% capture. The IR carries **declared nodes** alongside captured ones,
   so something the tracer cannot see becomes an **explicit assumption in the artifact**
   rather than an absence.
2. Record `unpriced{signature: reason}` and report coverage as a fraction, always.
3. **A failed forward writes no graph.** A 101-operator capture of a 64-layer model was a
   crashed forward written from a `finally` block — structurally valid, and merely wrong.
   What is written is additionally checked against model depth, since attention runs once
   per layer.

### Open issues

- The declared-node mechanism has no consumers yet; EP is its first customer.
- At `ep_size == tp_size` EP is close to a no-op — without EP each rank holds all experts
  sharded along the intermediate dimension, with EP each rank holds half the experts
  whole. Same FLOPs, graphs structurally identical (488 operators, 23 kinds either way),
  timings within 3-8%. A prior "EP works" result was withdrawn for exactly this reason.
  **Any EP claim must be made at a degree where EP is not a no-op.**

---

## D22. Activation liveness under fake tensors

### Problem

`03_memory_and_kv_model.md` D16 makes a def-use liveness walk the activation term's source
once capture lands.
The prior implementation was **observational** — `weakref.finalize` firing when the CUDA
allocator reclaimed — and there is no allocator device-free.

### Finding: three of the four prior faults carry over, and one is easier

A `weakref.finalize` on a `FakeTensor` fires at the same program point as on a real
tensor: when Python drops the last reference. A real CUDA tensor's storage is freed by
that same dealloc. So the observational method is preserved.

**Fault 1 — a death is not a last read.** A residual held across a block outlives every
read of it.

```
op index:   1    2    3    4    5   ...  32
            |    |    |    |    |         |
residual ---*----+----+----+----+---------+-->
            ^         ^                   ^
         created   last read          actually freed
                   <- naive death      <- true death
```

**Fault 2 — deaths are per output, not per operator.**

```
fused_add_norm(x, residual) -> (normed, new_residual)
                                  |            |
          normed ------> dies into the next gemm (2 ops later)
          new_residual --------------------> lives to end of block

  one death for the pair -> the residual is held an extra step, every layer
                         -> 36% of the term at TP=4
```

**Fault 3 — the address map must forget.** *Easier* device-free. On hardware the allocator
hands a freed address straight back, so a map keyed on storage address credits the next
tensor to whoever held it before. With fake tensors there are no addresses: key on a
**monotonic creation counter**, never on `id()`, which Python also reuses after GC. The
failure mode is designed out rather than guarded against.

**Fault 4 — this one breaks, and harder.** `torch.empty` inside a custom operator never
reached a dispatch mode on hardware either, but the operator *ran* and the allocator delta
was observable. Device-free, the operator does not run at all — its **fake impl** runs,
returning a shape and allocating nothing.

```
   REAL run                              FAKE run
   --------                              --------
   unified_attention(...)                unified_attention(...)
     |- torch.empty(scratch)  <-105 MB     `- fake_impl(shapes) -> out shape
     |- kernel launches                        allocates nothing
     `- returns out                            runs nothing

   dispatch mode sees : [out]            dispatch mode sees : [out]
   allocator saw      : [out, scratch]   allocator          : does not exist
                              ^
                   recoverable as a delta on hardware;
                   structurally unobservable device-free
```

What that gap looked like on the 27B: a **near-constant -105.8 MB** behind the allocator
through the whole layer stack, opened and closed around
`unified_attention_with_output_base` and `_fused_qk_rmsnorm_group_quant_kernel`, ending
correct at -0.012 MB.

```
bytes
  ^          ,--.      ,--.      ,--.          <- true (allocator)
  |      ,--'    `--,-'    `--,-'    `--,
  |     ,  ,--.      ,--.      ,--.            <- visible walk
  | ,--'  '    `--,-'    `--,-'    `--,        (parallel, 105.8 MB low)
  |       :
  |       :  gap opens at each opaque leaf, closes at the next op
  +-------+--------------------------------> op index
```

Also note the sign of the *other* known divergence: at each attention operator the walk
runs ~27 MB **below** the allocator and recovers at the next, because finalizers for q/k/v
fire before attention dispatches while the allocator still holds the memory (views onto a
storage something else keeps alive). Transient, self-correcting, nowhere near the peak.

### Decision

**Declare a scratch parameter per opaque leaf**, symmetric with the cost parameter
extractor of D20: `bytes = f(shape)` or the single bytes-per-token number the prior work
found generalises. Sourced from kernel source, documentation, or one measurement on
whatever device happens to be available — and in every case recorded as a **declared
constant in the artifact**, not a hidden absence.

Measured magnitudes: **0.1 KB/token on the 0.6B, 39.6 KB/token on the 27B**, worth the
difference between **-35.0% and +3.4%** held out.

Deliberately **not** per-operator attribution from a recorded curve: that reproduces the
traced curve exactly, is worth nothing at any other shape, and is a recording dressed as a
model.

**Revised 2026-10-10: the walk is not the activation term's first source.** Until capture
lands, the term is one coefficient per model and width measured on a real run
(`03` D16, [#704](https://github.com/jgong5/ATOM/issues/704)), and the scratch is inside
that reading, so neither this walk nor the per-leaf constants gate stages 2 and 3. The
walk replaces the coefficient when it lands, and the per-leaf constants are needed from
then (T4). A model with no measured coefficient takes a geometry form, which
[#705](https://github.com/jgong5/ATOM/issues/705) tests (`10` T34).

### Open issues

- Nothing establishes the scratch constants for Qwen3.8-27B under this design yet; they
  are needed once the walk replaces the measured coefficient.
- The fault-4 case that is *not* an opaque leaf — the MLP's silu destination, **13.6 MB a
  layer at TP=1 and exactly where the high-water mark sits** — was handled by an
  out-variant rule: an operator returning no tensor is an out-variant, and what it produces
  is the destination it was handed, recorded as its output when no live tensor already owns
  that storage. Whether that rule works unchanged against fake tensors is unverified.

---

## D23. Eager versus compiled

### Problem

Device-free means the model cannot be compiled: inductor must codegen and autotune for the
target architecture. So tier (b) traces the **eager** operator stream, while production
runs at `--level 3` PIECEWISE with `use_inductor = True` and `--cudagraph-mode FULL`
(`atom/model_engine/arg_utils.py::EngineArgs.add_cli_args`).

### Why this is more sound than it sounds

ATOM registers its expensive operators as opaque custom ops precisely so Dynamo cannot
look inside them, and Dynamo preserves custom ops as opaque calls. **So the eager and
compiled operator streams agree on exactly the 60-75% of step time that matters.**

What differs is the cheap material around them, and it was measured: **386 derived (eager)
against 330 captured (compiled)**. Inductor accounted for **exactly one** operator
(`aten::embedding` -> `inductor::triton_poi_fused_embedding_0`); the other 56 are
`split_with_sizes` and `empty` that inductor resolves into offsets and a buffer plan.
**Compute totals are 283 operators either way.**

So compilation does not thin the graph in a way that would justify capturing at
`--level 0`, which nobody deploys.

### The launch regime belongs to the step, not the trace

| regime | paid per | constant |
|---|---|---|
| replayed (CUDA graph) | kernel launch | **2.02 us** |
| eager | operator | **67.4 us** |

Thirtyfold apart. One eager trace supplies the operator list; the constant is chosen by
whether the *real* step replays. **Prefill is eager** — its token count is not on the
capture ladder, which is captured at one token per sequence — and **decode is replayed**.
Two regimes in one run.

A prior bug derived the CUDA-graph bucket from batch size alone and so assigned a rung to
prefill steps too, charging 2 us per launch instead of 67 us per operator. *"The fix above
it did nothing until this was corrected."* `StepShape.capture_bucket` must be `None`
whenever nothing was replayed.

Related, and ATOM's own defect rather than Compass's: the padding site takes the
**largest** rung where its comment says a 65-request batch should replay the 128 graph —
`reversed(capture_sizes)` is correct only on a descending list, and `capture_cudagraph`
re-sorts ascending when it finishes. The rule that actually selects the graph is
`ForwardMode.decide`. Mirroring the buggy site made every step in a 1,662-row sweep report
bucket 512.

### TODO (not decided)

**The inductor fusion gap.** An eager trace shows N unfused operators where one fused
kernel runs, over-counting launches and mispricing the fusion. Magnitude: **~4.8% of a
decode step, ~0.6% of a prefill step.** Three candidate treatments, none chosen:

1. declare the bias and report it alongside every result;
2. fit a fusion correction per structure;
3. capture the compiled operator list **once** on whatever device is available and diff it
   against the eager trace to build a structural correction — a one-time calibration,
   which the `measure` flag of D11 already sanctions, leaving simulation device-free.

Deferred to future work by decision on 2026-09-18.

### Open issues

- Whether a `TorchDispatchMode` is even entered during a compiled run is unresolved and
  probably moot here, but it bears on option 3 above. Inductor-generated kernels do not go
  through the dispatcher at all — they are reached via
  `torch._inductor.runtime.triton_heuristics.CachingAutotuner.run`, not
  `JITFunction.run`. A prior in-tree precedent
  (`atom/model_loader/online_quant_streaming.py::_CopyCounter`) overrides
  `ignore_compile_internals()` because *"TorchDispatchMode keeps its compile-internal
  state in process-global booleans"*.
- One in-tree hazard for mode-based instrumentation, recorded in a code comment in
  `atom/spec_decode/dspark_scheduler.py::schedule_prefix_lengths_tensor`: `torch.tensor(N, device=...)`
  under an active `DeviceContext` `__torch_function__` guard **hung all 8 ranks on ROCm**.
  The comment records a hang observed while the function was written; no landed version
  makes that call, as the function builds the value with `torch.full_like` on a device
  tensor. The mechanism is unexplained.

  **How much this gates T5.** A `FakeTensorMode` trace should be **GPU-free and
  collective-free**, so a hang whose mechanism is a desynchronised collective should not
  be reachable from it.

  Two things have to hold for that, and both look true:

  1. **No real collective executes under the mode.** ATOM's collectives are either
     dispatcher-visible ops (which have `register_fake` impls — every ATOM opaque op
     does) or invisible ones reached through declared nodes, which the tracer records
     rather than calls. Nothing should reach RCCL.
  2. **The specific hazard is not the same mode.** `dspark_scheduler.py::schedule_prefix_lengths_tensor` hung under
     an active `DeviceContext` **`__torch_function__`** guard doing
     `torch.tensor(N, device=...)` — a
     *real* run with a real device, not a fake-tensor trace.

  So the honest status is: **the hang gates `--measure` runs and any mode-based
  instrumentation of a real execution. It probably does not gate Phase 1a.** "Probably"
  is doing work there — both points above are reasoned from the code, not observed — and
  the cheap way to convert them into evidence is T5 itself, which will either trace
  cleanly at TP>1 or produce the hang and settle the question.

  **It gets root-caused rather than worked around**, for a reason independent of T5: a `__torch_function__` mode (the `DeviceContext` guard) hung all 8 ranks while that function was written, and
  `--measure` is a designed path. A workaround that avoids the one known call site leaves
  the mechanism unexplained and the next call site undiscovered.

  **Ordering:** run T5 *first* and cheaply. If it traces clean at TP>1, T52 drops to
  ordinary priority and gates nothing on the critical path.

  What root-causing means concretely, and why it is tractable: the failure is a
  *collective* hang, so the question is which rank diverges. `torch.tensor(N,
  device=...)` under a `__torch_function__` guard either (a) triggers a H2D copy on a
  stream the other ranks are not on, (b) takes a different branch on one rank because the
  guard intercepts a `.item()` and turns a device value into a host one, or (c) causes a
  lazy-init ordering difference. All three are distinguishable from a single `rocgdb`
  attach reading `info dispatches` per rank, which the in-tree
  `debug-agent-locate-kernel` procedure already automates. Estimated at well under a day
  on a quiet node, and it gates T5.

  Recorded as **T52**, and placed *before* the first tracing work in the execution plan
  rather than beside it.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D17 | Two tiers behind one `CostBackend`; tier (a) discovers the structure set tier (b) traces | 2026-09-18 |
| D18 | Capture with `TorchDispatchMode` + `FakeTensorMode(ShapeEnv)` + `_EnablePythonDispatcher()`, on FakeTensor rather than bare meta; **four** disciplines, not three — the fourth is that the engine's host arithmetic asks a symbol for a number, so the capture keeps the conversion, logs it with the line it happened on, and asserts that log against a declared set | 2026-09-18, fourth discipline 2026-09-22 |
| D18.1 | `torch.export` recorded as an unverified alternative; rejected now on decomposition fidelity and on capturing a model call rather than an engine step | 2026-09-18 |
| D19 | Hierarchical, symbolic, stream-annotated IR: `Seq` / `Repeat` / `Par`. No branches in the IR — applicability is a discrete key plus an evaluated guard domain | 2026-09-18 |
| D20 | Opaque leaves are priced, not decomposed. Each carries a declared parameter extractor. FlyDSL and MORI need no IR node. | 2026-09-18 |
| D21 | Declared nodes over chased coverage; refuse and record `unpriced` reasons; a failed forward writes no graph | 2026-09-18 |
| D22 | Liveness stays observational under fake tensors; internal scratch of an opaque leaf is a declared per-leaf constant. Revised: until capture lands the activation term is a coefficient measured on a real run, scratch inside it, and the walk replaces it after | 2026-09-18; revised 2026-10-10 |
| D23 | Trace eager, price with the regime constant of the real step. The inductor fusion gap (~4.8% decode) is a documented **TODO**, not a decision. | 2026-09-18 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T1 | Inductor fusion correction (D23) | ~4.8% of a decode step; three candidate treatments, none chosen |
| T2 | Enumerate the structure set for Qwen3.8-27B | needs tier (a) running against the real scheduler |
| T3 | Build the per-leaf parameter-extractor table (~20 entries) | the main hand-written asset; needs the leaf list frozen first |
| T4 | Establish scratch constants per leaf for the 27B | needed once the liveness walk replaces the measured activation coefficient (D22); unobservable device-free, so it needs a source or one measurement |
| T5 | Verify ATOM's model classes trace cleanly under FakeTensorMode at TP>1 | needs a non-wedged node |
| T6 | Validate that `Repeat` grouping reproduces the flat prices term by term | needs a first trace |
| T7 | Validate `Par` reconstruction from stream ids | not exercised until M2 (DeepSeek-V4-Pro) |
| T8 | Decide whether tier (a) is fitted independently or derived from tier (b) | tier (b) does not exist yet |
| T9 | Declare a row-ordering treatment for decode attention | 1.77x effect, invisible to every current feature |
