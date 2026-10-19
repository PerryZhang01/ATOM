# ATOM Compass — Design Topic 10: Analytical Laws

**Status:** reviewed and approved, 2026-09-20 — but **more speculative than every other
document here.**

Docs 01-09 are grounded in measurements, most of which were got wrong once before they
were got right. This one has a single existence proof and four recorded failures. Treat
its numbers as targets, not findings.

**Depends on:** `05_machine_spec_and_probes.md` (the device parameters), `04` (the IR whose
shapes feed a law), `07` D36 (the two granularities, and fallback along method).

**Revised 2026-10-10 for `16` D95:** the analytical laws are on M2's critical path, and D68 names the
DeepSeek-V4-Pro operator families whose laws M2 needs.

**Revised 2026-10-14 on [#937](https://github.com/jgong5/ATOM/pull/937):** D67's Class-C
memory constants are a `memory_readings` entry, marked assumed, so declared, for a device
never run (`07` D38).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952), and 2026-10-15 by the owner's ruling
on [#1011](https://github.com/jgong5/ATOM/issues/1011):** the analytical method names what
this document prices (D63); an op-level backend answers an unpriced leaf by its analytical
law, and the step roofline is the step-level fallback, its method counting the run file's
stated inputs (D64); the analytical-to-measured ratio is implementation efficiency, a
self-check that ranks no law above a measurement (D65, D68, D70); D67 gives each memory
term its method; D67.1's goals attach to the backend and are graded against real runs.

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** the analytical method is a
reasonable theoretical bound, priced at attainable peaks probed on the device, at both
granularities: op level from the captured IR, step level from module extractors (D64).
Collectives take an algorithm-free alpha-beta lower bound (D66). The bound is the gate;
the step and end-to-end latency goals are withdrawn and ranking is reported (D67.1).
Implementation efficiency is a diagnostic, never a projection input (D68), and MI455's
peaks are its datasheet times the ratios probed on MI355X (D69).

**Scope.** Cost and memory derived from device parameters and model geometry, with **no
observation of the subject**. Serves three purposes: M2's projection of DeepSeek-V4-Pro
onto MI455, a device never run, at step and op level; the analytical fallback, for what an
empirical source declines; and the step roofline, usable on day zero. The method gives a
reasonable theoretical bound: possibly unreachable, but not far from it. It is neither the
datasheet at derate 1 nor an efficiency measured on ATOM's own kernels (#1392).

---

## D63. What "analytical" means, and the three classes

### The word is reserved

`analytical` means **a form from first principles whose every parameter is model
geometry, deployment config, or a device parameter from a datasheet or a model-independent
probe on this platform** (#952, `02` D11). The prior PoC had no analytical term; the
meta-build weight count is one now (`03` D16). A value read back from a file is `measured`;
a fitted coefficient is `estimated`; a law that reads an assumed parameter, such as an
attainable peak carried from another device (D69), is `declared`, its note naming the
assumption. A peak probed on the device makes a term analytical (#1392).

### The organising principle, from what worked and what didn't

**The one existence proof.** Analytical **weight bytes** — `resident_bytes(model, tied_head)`
asked of a meta build — came out exact at every width: **−0.00 / +0.00 / −0.02 / +0.01%**
at TP=1/2/4/8 on the 0.6B, and 0.00% at TP=2 and TP=4 on the 27B.

**The four failures, and they share a shape.**

| Attempt | Result |
|---|---|
| rotary buffer size from `max_position_embeddings × head_dim × 2` | matched the 0.6B **exactly**, **4× wrong** on the 27B (partial rotary). *Tested on a second model, failed, did not ship.* |
| weight bytes from checkpoint headers (2-D shards, 1-D replicates, tied head dropped) | exact on the dense 0.6B at every width and on the hybrid 27B at TP=2 — **3.3% low** at TP=4 |
| `non_torch` as fixed-plus-per-peer | **no form fits** 5980 / 6340 / 9138 MiB at widths 2/4/8 |
| collective scaling as linear-in-group or linear-in-log₂ | predicts **+300%** / **+200%**; measured **+20.6%** TP2→TP8, and one collective went **down 43.5%** |

> **An analytical law works where the quantity is a pure function of stated geometry. It fails
> where the quantity is decided by a runtime, library or kernel-selection policy.**

### The three classes

| Class | Derivable from | Examples | Analytical treatment |
|---|---|---|---|
| **A — exact** | model geometry + dtype | weight bytes, KV bytes/token, FLOPs per operator, bytes moved per operator, collective message sizes | computed |
| **B — device-parameterised** | A + device spec | GEMM time, norm/elementwise time, attention time | roofline at attainable peaks probed on the device (D64) |
| **C — policy-determined** | nothing | which tuned kernel a shape dispatches to, `non_torch`, load residue, launch overhead, the in-situ gap, fabric scaling | **stated constants**, so declared (doc 05), or accepted error |

Class A comes free from the IR: doc 04's graph is captured at the step's shape (`04` D18),
so each node's FLOPs and bytes moved follow from its argument shapes without any extra
machinery.

---

## D64. The attainable-peak roofline, at step and op level

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392).** This replaces the derated leaf model
first written here.

### Work

Work comes from the contract only. `F`, flops per operand format, and `C`, communication,
come from work primitives (GEMM, grouped GEMM, attention, elementwise, the collectives of
D66). `M` is compulsory traffic: each input read once, each output written once, no
intermediates. All three are Class A.

### One price per pricing unit

```
  t_unit = max( t0,  F / peak_c,  M / BW(M),  t_comm )
```

The peaks are **attainable peaks probed on the device** (`05` D26): `peak_c`, one compute
rate per format (BF16, FP8, FP4); `BW(bytes)`, a bandwidth curve, per size the fastest of read-only, write-only and copy; `t0`, an
empty-kernel floor; and the link latency and bandwidth `t_comm` reads (D66). Probed values
make a term analytical. MI455, with no hardware, takes its datasheet times the ratio
probed on MI355X, declared (D69). A mixed-format operand pair takes the rate of the wider
format.

**MoE** work takes the expectation under independent uniform routing: `E(1-(1-k/E)^B)`
activated experts for `B` rows, each routed to `k` of `E`, and under EP the expected
maximum rank load.

**Graph padding** is priced as executed, at the rows `capture_rung` pads to; a run option
prices the real rows instead.

### Op level

- The op list comes from capture into the IR (`04`).
- A leaf's interface is everything needed to execute it: tensor specs and values plus the
  ambient values it reads. An estimate reads the leaf's contract (`04` D20); the benchmark
  times the same call inside ATOM's own forward, which supplies its weights, KV and
  metadata (`07` D39). Kernel
  selection and environment belong to the benchmark.
- An extractor registry maps an op kind to work primitives. `F` and `C` sum over the
  primitives; `M` is the leaf's own boundary. A fused op is one leaf priced once, so fusion
  is priced as fusion. Aten ops share one generic extractor that reuses
  `torch.utils.flop_counter`. An op kind with no extractor is refused by name.
- Leaves compose by the IR's own joins: `Seq` sums, `Par` joins, TBO overlaps, a DP group
  takes the max. The `resource_bound` join sums the branches' work and takes one roofline.

### Step level

A second projection, independent of capture, as hedge and cross-check.

- Module extractors walk the meta build by ATOM module class, read per-rank parameter
  shapes and the batch view, and emit work primitives. An unclassified module is refused
  by name.
- One `max` and one `t0` per module, summed over modules. Communication inside a module
  adds to its compute, unless a module attribute states fusion (`fused_allreduce`).
- The same probed peaks as op level; nothing is fitted.

**The step roofline.** `backends/step_roofline.py::StepRooflineBackend` is the step-level
projection. A run reaches it as the step-level fallback its run file composes
(`StepFallbackBackend`, `07` D36), turned on by the run-file key `step_roofline`, and
always on under `step_grid` (`17` D102). Its method counts the inputs it reads (owner ruling on
[#1011](https://github.com/jgong5/ATOM/issues/1011), 2026-10-15): spec fields by their
origin, analytical from a datasheet or a probe, declared if assumed; the run file's stated
inputs declared unless they carry an origin.

### Device time only

**Revised 2026-10-11 by the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815): an analytical law prices device time
only.** Every backend prices one forward's device busy time (`02` D11) and host time is
composed per runner, so the host floor the prior work established, `t_step = max(Σ t_leaf,
launches × h)`, leaves both granularities. The `max` it states survives in `01` D4's
recurrence, `end = max(target_end + D_draft, launch_end)`, where `launch_end` plays the
part of `launches × h`. The launch time comes from the measured `W_launch` host law (`09`
D54), not from `host.*`. The evidence below is why that `max` is kept. `t0` is the
device's own floor on one kernel, not this host floor.

The step roofline has no host bound (owner ruling on
[#1047](https://github.com/jgong5/ATOM/issues/1047), 2026-10-15): it prices device time
only, and `run/__init__.py::runner` composes host time with it through the host
recurrence, so launch time is counted once. The run file's former `roofline_floor_s`,
which both set a host floor and turned the fallback on, is refused by name, naming
`step_roofline`.

### Why the host floor is not optional

`step = max(kernel time, launches × h)` at h = **95.8 µs** held four warm prefill shapes
within 2% at a fixed 391 launches:

| tokens | kernels | idle | device window |
|---|---|---|---|
| 794 | 13.384 ms | **23.360 ms (63.6%)** | 36.745 ms |
| 2,294 | 27.037 | 11.195 (29.3%) | 38.232 |
| 6,594 | 110.046 | **0.033 (0.0%)** | 110.080 |
| 15,694 | 418.164 | 0.009 (0.0%) | 418.173 |

The 6,594 row was a **held-out prediction before it ran** and came back at 0.033 ms idle. A
roofline that ignores this term is **63% wrong at small shapes** and right at large ones —
which is exactly the regime a decode step lives in.

And the additive form fails: fitted to the smallest shape it over-predicts the largest by
**+5.6%**; fitted to the largest it under-predicts the smallest by **−63.6%**.

**`h` is a Class-C constant.** It survives a machine change (95.8 vs 95.7 µs on a box whose
GPUs are ~20% faster — *"a floor that ignores a 20% swing in device speed is a floor on the
host side"*) and a width change exactly (TP=4 costs what TP=2 costs: same 505 launches,
same 45.1 ms). It does **not** survive a model change: **95.7 µs on the 0.6B against 125.0
µs on the 27B — 40% apart.**

---

## D65. Four places roofline is known to be wrong, with numbers

These are not speculative. Each was measured, and each is a smooth model meeting a
discontinuous reality.

**1. GEMM tile cliffs.** A 2.4× jump between M=768 (58.07 µs) and M=1024 (137.45 µs) where
the trend says ~77 µs. Twelve distinct tile configurations across twenty values of M. A
roofline is smooth through all of them.

**2. Attention plateaus.** Cost per token sits on flat plateaus quantised in sequence
length — 3.67, 6.28, 6.10, 5.90, 6.01, 8.31, 8.45, 8.17, 8.22, 10.58, 10.41, 10.47, 10.50
µs/token across a 13-point ladder — with the plateau index `ceil(L / 2530)` and steps of
~2.27 µs/token. *"The quadratic term is computed in whole chunks of keys."*

**3. Decode cost falls as batch grows.** At matched total context, batch 1 → 12 costs
3.640 → 3.532 ms (TP=1) and 3.636 → 3.310 ms (TP=2). Twelve sequences cost **less** per
step than one. A roofline predicts growth; the sign is wrong.

**4. Prefill is non-monotone in token count.** 2×400 at 4,588 tokens costs **52.14 ms**
where 4×200 at 4,376 tokens costs **63.16 ms** — fewer tokens, more time, reproducible to
0.03 ms over five repeats.

### Decision

**An analytical law is a smooth bound and is allowed to be.** It does not model these
places: they are where the measured time sits furthest above it, and the
measured-over-analytical report per op kind and per phase (D68) is where they show.

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** a pricing unit has one price (D64),
so the band this decision first required wherever a known discontinuity lay in range is
withdrawn, and with it T39's dispatch-band table. A measured price answers where one
exists; tracking it closely earns a law nothing.

---

## D66. Collectives

The textbook models are refuted on this fabric as *predictions*. Ring all-reduce predicts
`2(N−1)/N · n/BW`, i.e. roughly +75% from TP2 to TP8; linear-in-group predicts +300%;
linear-in-log₂ +200%. **Measured: `cross_device_reduce_1stage` 9.08 → 10.40 → 10.95 µs at
TP 2/4/8 (+20.6%), and `allgather_lastdim` 29.00 → 23.73 → 16.39 µs (−43.5%).**

The explanation on record: *"a one-stage reduce on a fully-connected fabric is latency-bound
at these sizes ... which is a claim about this interconnect, not about collectives, and
should be re-measured on a multi-node deployment before being relied on there."*

### Decision

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** a collective takes an
**algorithm-free alpha-beta lower bound**: one hop of link latency plus the minimum bytes
any algorithm must send, over the link bandwidth.

```
  t_comm = link_latency_s + min_bytes(collective, N, n) / link_bandwidth
```

Both link terms are probed (`05` D26). The spec names no algorithm, so AITER's one-stage
reduce, its quick-reduce, the custom all-gather behind the default
`ATOM_USE_CUSTOM_ALL_GATHER=1` and RCCL are all priced by the same bound, which none of
them can beat. Open item T36, which asked the spec to name the algorithm per code path, is
withdrawn. This replaces the two-regime `max` form, with an algorithm-dependent message
size, first written here. A group rides `intra_node` while it fits in
`intra_node.domain_devices` (`05` D25), whatever `count_per_node` is.

A bound sits below every algorithm, so its distance from the measured time is the
diagnostic of D68, per op kind. Collectives are ~6.6% of kernel time at TP=2, 10.4% at
TP=4, 13.4% at TP=8, and at the sizes ATOM uses they are latency-bound, so the latency term
carries most of the bound there.

---

## D67. Analytical memory

This is where analytical laws are strongest, because most terms are Class A. Each term
carries a method, as cost terms do (#952, `memory/terms.py`):

| Term | Class | Form | Method | Evidence |
|---|---|---|---|---|
| **weights** | A | ask a meta build, dedupe by storage | analytical from the meta build | **exact at every width** — the existence proof |
| **KV bytes/token** | A | layers × kv_heads/TP × head_dim × 2 × dtype | analytical | arithmetic |
| **buffers** | A | the same meta build, dedupe by buffer storage ([`03` D16's source table](03_memory_and_kv_model.md#the-source-of-each-term)) | analytical | **exact** against DeepSeek-V4-Pro's real buffer storages at DPA-TP8 (#1166); a rotary formula before it was **4× wrong** on the 27B |
| **activations** | B | `k_model × tokens`, k read on one real start per configuration (`07` D38) | measured at the entry's warmup tokens, estimated at others; declared from an assumed entry | linear scaling holds — a 3,494-token trace scaled to an independently measured 4,096-token peak at **+0.0% at TP=1/2/4**. `k` is not derivable from geometry within 25% (T34). |
| **invisible scratch** | **C** | inside the measured `k` | that of the activations term | kernel-internal. **0.1 KB/token on the 0.6B, 39.6 KB/token on the 27B** — worth the difference between −35.0% and +3.4% held out |
| **`non_torch`, load residue** | **C** | — | measured where probed at this width; declared if assumed or transferred | no form fits; a `memory_readings` entry, marked assumed for a device never run (`07` D38) |
| **graph pool** | C | `reserves()` mirrors ATOM; `predicts()` reads a pool line | `reserves()` analytical on the eager branch, combined with the activations term on the whole-graph one; `predicts()` estimated if probed, declared if assumed | measured line at W=1, flat 104 MiB above |
| **capacity** | — | the machine spec's | analytical from a datasheet, measured if probed | — |

A real run's readings, the side a term is compared with, are measured. A declared term can
fail its gate but cannot pass it, and a term built from a recording is not passed against
that same recording (#960).

So an analytical memory model gets weights and KV **exactly**, activations **approximately**,
and needs Class-C constants from a `memory_readings` entry for the rest — which is
precisely the arrangement doc 03 already describes. Analytical memory is therefore *not* a new
design; it is the existing memory model (`03` D16) with the activation coefficient
measured or derived rather than walked.

**Revised 2026-10-10 and 2026-10-16: the measured coefficient is the source.** `k_model`
is one real start's `peak - current` after warmup over the warmup tokens, per
configuration, so the scratch is inside it
([#704](https://github.com/jgong5/ATOM/issues/704)). It is not derivable from geometry
within the ≤25% goal of D67.1 ([#705](https://github.com/jgong5/ATOM/issues/705), T34),
so a configuration nobody has run takes an entry marked assumed, the MI455 projection
included (`07` D38).

---

## D67.1. The accuracy goal for the analytical method

### Problem

The empirical backends carry gates on throughput, TPOT, TTFT, each non-KV memory
term and KV block count, at the [acceptance targets](README.md#acceptance-targets).
The analytical method needs its own, stated in advance, or "the projection is available on
day zero" is an unfalsifiable claim. A number with no gate cannot be reported as a result.

### Why its gate cannot be the empirical gate

1. **It is a bound, not an estimate** (D64). A reasonable theoretical bound may be
   unreachable; a latency error against it measures the kernels, not the method.
2. **It is measurably smooth where reality is not** (D65), and the corrections are exactly
   what the empirical campaign supplies.
3. **Its job is different.** It is asked what a device never run could do, and which
   configuration wins there.

### The goal

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392).**

**Gate: on every measured step, at both granularities, analytical is at most measured.**
A step whose analytical price exceeds its measured time is a defect in an extractor, a
join or the spec, named by its op kind or module. It is a bound, so no violation is
tolerated. `08` D50.2 applies it.

**Reported, not gated:** configuration ranking (top-1, top-3) over M2's four P/D cells;
measured over analytical per op kind and per phase, with no threshold yet (D68); and the
method's error from MI308X to MI355X (`08` D50.2).

The step-time goal of 30% and the end-to-end goal of 40% first stated here are withdrawn,
and ranking is no longer the primary gate. The memory goals stand:

| Quantity | Goal | Why this number |
|---|---|---|
| **Memory: KV capacity, block count** | **≤5%**, same as empirical | Class A — exact from geometry. There is no modelling here to be wrong about; a miss is a bug. |
| **Memory: weights** | **≤5%**, *tighter* than empirical's ≤10% | Class A — exact from geometry, so a miss is a bug rather than modelling error. Weights is a non-KV memory term, and the acceptance table gates those at ≤10%; the roofline keeps its own goal at that level anyway, because a looser bound on an exact quantity would not catch the bug the row exists to catch. Tighter than the empirical gate, not the same as it. |
| **Memory: each non-KV term except weights** | **≤25%** | activations are approximate (`k_model` derived from geometry where no run measured it) and invisible scratch is Class C. The spread that sets the number: scratch is 0.1 KB/token on the 0.6B and **39.6 KB/token** on the 27B. |
| **Feasibility (does it fit)** | **no false "fits"** | a configuration the roofline says fits and then OOMs is a hard failure regardless of byte error. One-sided on purpose. |

**On the weights row.** The empirical acceptance table has two memory buckets —
≤5% for KV capacity / block count and ≤10% for each non-KV memory term, of which
weights is one (`03` D16 lists it first). So for KV and block count the roofline goal is
the empirical gate; for weights it is *tighter* than the empirical gate. The roofline keeps
the tighter goal deliberately — the quantity is exact from geometry and the goal
exists to catch a bug, which a bound set for modelling error would not do. What is
withdrawn is only the claim of parity: an earlier wording grouped all three quantities
in one cell and called the number "same as empirical", which is true of KV capacity
and block count and false of weights, and was read in review as evidence that the
empirical weights gate is ≤5%.

### How it is graded

The gate attaches to the backend and is checked against real runs' step records (#952):
the stage-2 MI308X record and the stage-3 MI355X records, at step level and at op level.

What that cannot check is MI455, which has no real run. That gap is irreducible: the
bound is checked *where real runs exist*, and MI455's peaks are carried from MI355X, so
its terms are declared (D69). The method's error from MI308X to MI355X is quoted with
the projection, so a user of an unmeasured device sees a range rather than a promise
(T56).

---

## D68. Implementation efficiency is a diagnostic

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392).**

Implementation efficiency, **measured over analytical**, says how far the kernels ATOM
runs sit from the bound on this device. It reflects the implementation, not whether the
law is right. It is reported per op kind and per phase as a distribution, never as a mean
alone, with no threshold yet. It is **never a projection input**: no measured efficiency
is carried onto another device to scale a price (D69).

### On M2's critical path

`16` D95 puts the analytical method on M2's critical path, for the MI455 projection of
DeepSeek-V4-Pro. Its families, each priced at both granularities:

| Family | Law | Spec fields read (`05` D25) |
|---|---|---|
| MoE expert GEMMs | D64, routing in expectation | each operand pair's wider format (FP4 only where the activations are MXFP4 too); `BW(bytes)`; `t0` |
| attention, with its compressors | D64 | the rate at each leaf's operand format; `BW(bytes)`; `t0` |
| the CSA indexer | D64 | the same |
| mHC | D64 | the same |
| the DP group's gather and reduce-scatter over `T_moe`, EP off | D66 | `interconnect.intra_node.*` |
| MORI dispatch and combine, for the EP8 step | D66 | `interconnect.intra_node.*` |

Mooncake's KV transfer needs no law here: it is charged `latency + bytes/bandwidth` from
`interconnect.inter_node` (`05` D25). Most of these families are opaque leaves, so their
work comes from the extractors of T35: module extractors at step level, op-kind
extractors at op level (`16` *Wave 5*).

### The analytical fallback

An op-level backend answers a leaf no measurement prices by its analytical law, always
(#952); at step level the step roofline is the fallback a run file composes (D64). As the
fallback of an empirical backend, the analytical method answers its own price, reported
apart (#1392), with the method its inputs give it, declared while it reads an assumed
parameter, and its share of seconds is always reported (`08` D50.1). A law never moves
ahead of a measurement: one that tracks its measured price within the instrument's
repeatability (doc 07 D41) still answers only where no measured price does.

### What is reported

Measured over analytical per op kind and per phase, beside the bound gate (D67.1). A run
whose terms are all analytical reports 100% `analytical` in its seconds by method; an
op-level run with gaps reports the mixture, beside the cell's verdict and not gated (doc 08
D50.1).

---

## D69. Device transfer

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392).** Analytical laws are the only route to
a device never measured, and the route has one stated weakness.

**Peaks are attainable peaks probed on the device** (D64). A device with no hardware, as
MI455 in M2, takes **its datasheet times the ratio probed on MI355X**, per peak. The ratio
is observed on another device, so the field is assumed and every term that reads it is
declared (`05` D25, rule 5). The ratio is a property of the probe on the silicon and the
vendor library, not of ATOM's kernels: no implementation efficiency enters it (D68).

> **A projection onto an unmeasured device is as good as its carried ratios, and nothing in
> the procedure validates them there: its terms are declared.**

Mitigations, none of which removes the caveat:

1. **Report sensitivity.** Every projection onto an unmeasured device is reported with its
   carried ratios, and recomputed at ±20% of each, capped at 1, the datasheet
   ([#1466](https://github.com/jgong5/ATOM/issues/1466)). A ranking that survives that band is
   more trustworthy than one that doesn't.
2. **Rank, don't predict.** Configuration ranking is reported beside the projection
   (D67.1). A ratio error that applies uniformly across configurations may leave the
   ranking intact while moving every absolute number.
3. **Measure the method where both sides exist.** The same carrying, MI308X to MI355X on
   Qwen3.8-27B, is reported against real MI355X runs (`08` D50.2).

---

## D70. Expected accuracy, stated in advance

**D67.1 sets the gate.** This decision is the separate and weaker statement: what we
*expect* to observe, per use, so that a result cannot be graded against an expectation
invented after seeing it.

| Use | Expected error | Why |
|---|---|---|
| the projection, a measured device | **below the measured time on every step**; how far is reported per op kind and phase | D67.1's gate; D68's report |
| the projection, MI455 | **declared**; its distance from a real run is estimated by the MI308X-to-MI355X method error | D69 |
| the analytical fallback for a leaf | **below the measured price, by an amount unknown per leaf** — that is why its share is reported apart | D68 |
| analytical memory: weights, KV | **exact** | Class A; D67.1 gates both at ≤5% purely to catch bugs — for KV that is the empirical gate, for weights it is tighter than empirical's ≤10% |
| analytical memory: block count end to end | **unknown** | depends on activations and the Class-C constants |

Single digits on latency are not a target for the analytical method and should not be
claimed. That is what the empirical op-level backend exists for.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D63 | Three classes: A exact from geometry, B device-parameterised, C policy-determined. An analytical law works where the quantity is a pure function of stated geometry and fails where a runtime or library policy decides it. `analytical` is #952's method: a form from first principles over geometry, deployment config and datasheet or probed device parameters; an assumed parameter makes a term declared. Class B prices at attainable peaks probed on the device. | 2026-09-19; revised 2026-10-14 (#952) and 2026-10-17 (#1392) |
| D64 | One pricing unit costs `max(t0, F/peak_c, M/BW(M), t_comm)` at attainable peaks probed on the device: one rate per format, the wider format's for a mixed pair, a bandwidth curve, an empty-kernel floor and link alpha-beta. Work comes from the contract only; MoE in expectation under uniform routing; graph padding as executed. Op level: the captured IR, an extractor registry by op kind, refused by name without one, composed by the IR's joins. Step level: module extractors over the meta build, one `max` and one `t0` per module, summed; refused by name for an unclassified module. Every analytical law prices device time only; the host floor's `max` lives in `01` D4's recurrence, with the launch time from the measured `W_launch` law. The step roofline's method counts the inputs it reads; it has no host bound; the run-file key `step_roofline` turns it on. Replaces the derated leaf model first stated here. | 2026-09-19; revised 2026-10-11 (#815), 2026-10-15 (#1011), 2026-10-15 (#1047) and 2026-10-17 (#1392) |
| D65 | An analytical law is a smooth bound and says so: the known discontinuities are where the measured time sits furthest above it, shown by the measured-over-analytical report. A pricing unit has one price; the band first required here is withdrawn. | 2026-09-19; revised 2026-10-14 (#952) and 2026-10-17 (#1392) |
| D66 | A collective takes an algorithm-free alpha-beta lower bound: one hop of link latency plus the minimum bytes any algorithm must send, over the link bandwidth. The spec names no algorithm (T36 withdrawn). Textbook ring/log models are refuted on this fabric as predictions. Replaces the two-regime form first stated here. | 2026-09-19; revised 2026-10-17 (#1392) |
| D67.1 | The gate is the bound: on every measured step, at both granularities, analytical is at most measured. Reported, not gated: configuration ranking (top-1, top-3) over the four P/D cells, measured over analytical per op kind and per phase, and the MI308X-to-MI355X method error. Memory goals: ≤5% on weights and on KV capacity/block count (the empirical gate for KV, tighter than it for weights), ≤25% each other non-KV memory term, no false "fits". The step and end-to-end latency goals and ranking as the primary gate, first stated here, are withdrawn. | 2026-09-19; revised 2026-10-14 (#952) and 2026-10-17 (#1392) |
| D67 | Analytical memory is the existing memory model with the activation coefficient derived rather than walked. Weights and KV exact; scratch stays a stated constant, and `non_torch` and load residue come from a `memory_readings` entry, marked assumed for a device never run. Revised: the coefficient is measured on one real run per model and width, with the scratch inside it rather than a stated constant; no geometry form stands in for it (T34, 2026-10-16, #965). Each term carries a method, and a declared one cannot pass its gate. | 2026-09-19; revised 2026-10-10, 2026-10-14 (#937) and 2026-10-14 (#952) |
| D68 | Implementation efficiency, measured over analytical, is a diagnostic reported per op kind and per phase as a distribution, with no threshold yet; it is never a projection input. The analytical method is on M2's critical path for the MI455 projection of DeepSeek-V4-Pro's families (`16` D95). An op-level backend answers an unpriced leaf by its analytical law, and as an empirical backend's fallback the method answers its own price, reported apart; no law moves ahead of a measurement. Replaces the efficiency first stated here as a calibration self-check, carried declared to another device. | 2026-09-19; revised 2026-10-10, 2026-10-14 (#952) and 2026-10-17 (#1392) |
| D69 | Peaks are probed on the device; a device with no hardware takes its datasheet times the ratio probed on MI355X, so its terms are declared. Report each carried ratio with a ±20% sensitivity band; prefer ranking claims to absolute ones; measure the method from MI308X to MI355X. Replaces the assumed derate first stated here. | 2026-09-19; revised 2026-10-17 (#1392) |
| D70 | Expected accuracy is stated in advance, per use: below the measured time on a measured device, declared on MI455. Single-digit latency error is not a target for the analytical method. | 2026-09-19; revised 2026-10-17 (#1392) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T34~~ | ~~Test whether the activation coefficient `k_model` is derivable from geometry~~ — closed 2026-10-16 by [#965](https://github.com/jgong5/ATOM/issues/965): not within 25% ([#705](https://github.com/jgong5/ATOM/issues/705)) | — |
| T35 | Derive the work of the opaque leaves: op-kind extractors at op level ([#1402](https://github.com/jgong5/ATOM/issues/1402), [#1403](https://github.com/jgong5/ATOM/issues/1403)) and module extractors at step level ([#1398](https://github.com/jgong5/ATOM/issues/1398), [#1399](https://github.com/jgong5/ATOM/issues/1399)) | Class A is free from the IR for aten ops, but an opaque leaf's internal work must be described by hand (doc 04 D20's parameter extractor, extended) |
| ~~T36~~ | ~~Name the collective algorithm per code path in the machine spec schema~~ — **withdrawn 2026-10-17 by [#1392](https://github.com/jgong5/ATOM/issues/1392)**: collectives take an algorithm-free lower bound (D66) | — |
| ~~T37~~ | ~~Decide whether `h` (the host floor) is derivable or stays a per-model measured constant~~ — **ruled 2026-10-11 on #815**: host time is measured, as host laws keyed by host CPU, `atom` digest, model and engine config, with no transfer (`09` D54) | 40% apart between two models; it is the largest single Class-C term |
| T38 | Build the measured-over-analytical report per op kind and per phase, beside the bound gate ([#1405](https://github.com/jgong5/ATOM/issues/1405)) | needs both granularities' prices and a real run's step records |
| ~~T39~~ | ~~Establish a dispatch-band table from geometry for leaves with no measured bands~~ — **withdrawn 2026-10-17 with D65's band (#1392)** | — |
| T56 | Quote the method's error from MI308X to MI355X with the MI455 projection, so a user of an unmeasured device sees a range rather than a promise ([#1406](https://github.com/jgong5/ATOM/issues/1406)) | needs real MI355X runs and both machines' probes |
