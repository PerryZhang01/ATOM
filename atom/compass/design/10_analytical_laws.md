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

**Scope.** Cost and memory derived from device parameters and model geometry, with **no
observation of the subject**. Serves three purposes: the step roofline, usable on day zero;
the analytical fallback, for an op-level leaf no measurement prices; and the only route to
a device that has never been run.

---

## D63. What "analytical" means, and the three classes

### The word is reserved

`analytical` means **a form from first principles whose every parameter is model
geometry, deployment config, or a device parameter from a datasheet or a model-independent
probe on this platform** (#952, `02` D11). The prior PoC had no analytical term; the
meta-build weight count is one now (`03` D16). A value read back from a file is `measured`;
a fitted coefficient is `estimated`; a law that reads an assumed parameter, such as an
assumed derate, is `declared`, its note naming the assumption.

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
| **B — device-parameterised** | A + device spec | GEMM time, norm/elementwise time, attention time | roofline with derates |
| **C — policy-determined** | nothing | which tuned kernel a shape dispatches to, `non_torch`, load residue, launch overhead, the in-situ gap, fabric scaling | **stated constants**, so declared (doc 05), or accepted error |

Class A comes free from the symbolic IR: doc 04's shapes are sympy expressions, so FLOPs
and bytes-moved are closed-form expressions in `T`, `B`, `Ctx`, `TP` without any extra
machinery. The experiment already demonstrated this — `FLOPs(T) = 256·T² + 98304·T`,
exact against a fresh concrete trace at T = 2 / 17 / 64 / 512 / 4096.

---

## D64. The roofline leaf model

For a leaf with `F` flops and `M` bytes moved, both Class-A expressions over the IR's
symbols:

```
  t_compute = F / (peak_flops  × derate_compute)
  t_memory  = M / (bandwidth   × derate_memory)
  t_leaf    = max(t_compute, t_memory)
```

and at step level, the host floor that the prior work established as a **`max`, not an
addend**:

```
  t_step = max( Σ t_leaf ,  launches × h )
```

All four parameters come from the machine spec (doc 05 D25): `compute.*_flops` at the
leaf's operand precision,
`memory.bandwidth_bytes_per_s`, their `derate`s, and — for `h` — `host.*`.

**Revised 2026-10-11 by the owner's ruling on
[#815](https://github.com/jgong5/ATOM/issues/815): an analytical law prices device time
only, `Σ t_leaf` at op level and the larger of compute and memory in the step roofline.**
Every backend prices one forward's device busy time (`02` D11) and host time is composed
per runner, so the host floor leaves both. The
`max` it states survives in `01` D4's recurrence, `end = max(target_end + D_draft,
launch_end)`, where `launch_end` plays the part of `launches × h`. The launch time comes
from the measured `W_launch` host law (`09` D54), not from `host.*`. The evidence below is why that `max` is kept.

**The step roofline.** `backends/step_roofline.py::StepRooflineBackend` is this law at
step level, without the IR: compute `2 · P · Σ N_Q / flops`, memory `(weight bytes + KV
bytes per token · (Σ N_KV + Σ N_Q)) / bandwidth`. A run reaches it only as
the step-level fallback its run file composes (`StepFallbackBackend`, `07` D36), turned on
by the run-file key `step_roofline`. Its one
term is named by the bound that won, and its method counts the inputs that bound reads
(owner ruling on [#1011](https://github.com/jgong5/ATOM/issues/1011), 2026-10-15): spec
fields by their origin, analytical from a datasheet or a probe, declared if assumed; the
run file's stated inputs declared unless they carry an origin. Every derate is assumed
today and the run file gives `parameter_count` no origin, so no roofline term is
analytical yet.

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

**An analytical law is a smooth model and is allowed to be.** It must not pretend otherwise:

- **It reports a band, not a point**, wherever a known discontinuity lies inside its input
  range. The dispatch-band boundaries are already recorded in the price list (doc 09 D60)
  when one exists; when it doesn't, the band comes from the geometry (tile size, chunk
  size) rather than from measurement.
- **It answers a leaf whose known discontinuities fall inside the queried range with a
  band**, as the analytical fallback too (#952). Tracking a measured price closely earns a
  law nothing: a measured price answers where one exists.

---

## D66. Collectives

The textbook models are refuted on this fabric. Ring all-reduce predicts `2(N−1)/N · n/BW`,
i.e. roughly +75% from TP2 to TP8; linear-in-group predicts +300%; linear-in-log₂ +200%.
**Measured: `cross_device_reduce_1stage` 9.08 → 10.40 → 10.95 µs at TP 2/4/8 (+20.6%), and
`allgather_lastdim` 29.00 → 23.73 → 16.39 µs (−43.5%).**

The explanation on record: *"a one-stage reduce on a fully-connected fabric is latency-bound
at these sizes ... which is a claim about this interconnect, not about collectives, and
should be re-measured on a multi-node deployment before being relied on there."*

### Decision

A two-regime model keyed off the spec's `interconnect` section:

```
  t = max( link_latency_s × hops(topology, N),
           message_bytes(N, algorithm) / (link_bandwidth × derate) )
```

with the **algorithm named in the spec**, not assumed — because AITER's one-stage reduce,
its quick-reduce, the custom all-gather and RCCL all have different `message_bytes(N)`, and
the default path is `ATOM_USE_CUSTOM_ALL_GATHER=1`, which is **entirely invisible to a
dispatch trace** (doc 07 D40). A group rides `intra_node` while it fits in
`intra_node.domain_devices` (doc 05 D25), whatever `count_per_node` is.

Expected accuracy: poor at the sizes ATOM actually uses, because those are latency-bound
and the latency term is the one least constrained by a datasheet. Collectives are ~6.6% of
kernel time at TP=2, 10.4% at TP=4, 13.4% at TP=8 — so a 2× error here is 7–27% of a step
at TP=8.

---

## D67. Analytical memory

This is where analytical laws are strongest, because most terms are Class A. Each term
carries a method, as cost terms do (#952, `memory/terms.py`):

| Term | Class | Form | Method | Evidence |
|---|---|---|---|---|
| **weights** | A | ask a meta build, dedupe by storage; the run file's `parameter_count` where the runner hands in no meta build | analytical from the meta build; declared from `parameter_count` | **exact at every width** — the existence proof |
| **KV bytes/token** | A | layers × kv_heads/TP × head_dim × 2 × dtype | analytical | arithmetic |
| **buffers** | A-ish | the stated formula over stated `config.json` fields, until a recording exists ([`03` D16's source table](03_memory_and_kv_model.md#the-source-of-each-term)) | analytical; declared on the assumed `partial_rotary_factor` branch | the first formula matched the 0.6B and was **4× wrong** on the 27B |
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

## D67.1. The accuracy goal for the roofline

### Problem

The empirical backends carry gates on throughput, TPOT, TTFT, each non-KV memory
term and KV block count, at the [acceptance targets](README.md#acceptance-targets).
A roofline backend, the step roofline today, has had none, which makes "the roofline is
available on day zero" an unfalsifiable claim. A number with no gate cannot be reported
as a result.

### Why the roofline's gate cannot be the empirical gate

Not a matter of ambition. Three structural reasons the empirical gates are the wrong target:

1. **The derate absorbs everything the roofline does not model.** A single scalar stands in for
   occupancy, tail effects, launch gaps, cache behaviour and kernel quality. Its own
   authoring error is comparable to the accuracy being demanded.
2. **Roofline is measurably wrong in four named places** (D65 above), and the corrections
   are exactly what the empirical campaign supplies. A roofline that hit those gates would
   mean the empirical campaign was unnecessary.
3. **The roofline's job is different.** It is asked *which configuration wins* and *does this
   fit*, on a device nobody has measured. Ranking and feasibility are its outputs.
   Latency to the empirical gates is not.

### The goal

Stated in advance, per quantity, because they are not equally hard:

| Quantity | Roofline goal | Why this number |
|---|---|---|
| **Memory: KV capacity, block count** | **≤5%**, same as empirical | Class A — exact from geometry. There is no modelling here to be wrong about; a miss is a bug. |
| **Memory: weights** | **≤5%**, *tighter* than empirical's ≤10% | Class A — exact from geometry, so a miss is a bug rather than modelling error. Weights is a non-KV memory term, and the acceptance table gates those at ≤10%; the roofline keeps its own goal at that level anyway, because a looser bound on an exact quantity would not catch the bug the row exists to catch. Tighter than the empirical gate, not the same as it. |
| **Memory: each non-KV term except weights** | **≤25%** | activations are approximate (`k_model` derived from geometry where no run measured it) and invisible scratch is Class C. The spread that sets the number: scratch is 0.1 KB/token on the 0.6B and **39.6 KB/token** on the 27B. |
| **Step time (prefill and decode, separately)** | **≤30%** | roofline plus a stated derate, against the four known wrongnesses of D65. Anything tighter would be claiming the corrections are unnecessary. |
| **End-to-end TTFT / TPOT / throughput** | **≤40%** | step error compounds through a scheduler with discontinuities (doc `08` D44). Reported, not gated on tighter. |
| **Configuration ranking** | **top-1 must survive; top-3 set must survive** | **this is the primary gate.** The others are diagnostics for it. |
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

### Why ranking is the primary gate and the others are not

The roofline exists for the day-zero and never-measured-this-device cases. In both, the
decision being made is *choose a configuration*, and a model that is uniformly 25% slow
ranks identically to a perfect one. A model that is 10% fast on one configuration and
10% slow on another can invert the ranking while looking better by every latency
statistic. So the gate that matters is ordinal, and the percentage goals above exist to
make a ranking failure diagnosable rather than to be reported on their own.

This is the same argument doc `08` D48 makes for memory and it is adopted here
deliberately — one gate philosophy across both, not two.

### How it is graded

The goals attach to the backend and are graded against real runs (#952): a run the
roofline prices is a cell like any other, compared with its configuration's real runs by
`08` D44's results, ranking first. Implementation efficiency (D68) is a self-check beside
it, not the grade.

What that cannot check is the roofline on a device with no real run, which is its whole
purpose. That gap is irreducible: the goals above are graded *where real runs exist* and
**assumed to transfer**, with the derate carrying the assumption, so such a prediction is
declared. Recorded as **T56**: state the roofline's error observed on each measured device
in the artifact, so a user on an unmeasured device sees the range rather than a promise.

### Open issue

- These numbers are **stated, not derived**. 30% for step time is a judgement informed
  by D65's four measured wrongnesses; one campaign will say whether it was generous or
  harsh, and it should be revised then rather than defended.

---

## D68. Implementation efficiency: the empirical campaign checks the laws

### The structural argument

For every leaf where **both** a measured price and an analytical law exist, the law's time
over the measured time is that leaf family's **implementation efficiency**: how close the
kernel library comes to the roofline on this device. It reflects the implementation, not
whether the law is right, and it is a self-check in the calibration report, off the
critical path (#952). Author the laws after the empirical campaign and there is no way to
read it except by measuring again.

**So the analytical laws are authored during the empirical campaign, not after it.**

### On M2's critical path

`16` D95 makes the laws needed during the campaign as well. M2's MI455 roofline projection
is built from them, so they are an M2 deliverable rather than a side wave, and that is the
only reason they are on it. `16` W5.2
authors the laws for DeepSeek-V4-Pro's operator families below, per precision, alongside
`16` W4.4's MI355X calibration of the same families, which supplies each law's measured
counterpart:

| Family | Law | Spec fields read (`05` D25) |
|---|---|---|
| MoE expert GEMMs | D64 | `compute.fp4_flops` for the FP4 routed experts, the operand precision's rate for the rest; `memory.bandwidth_bytes_per_s` |
| attention, with its compressors | D64, with D65's band where a plateau lies in range | `compute.*_flops` at each leaf's operand precision; `memory.bandwidth_bytes_per_s` |
| the CSA indexer | D64 | the same |
| mHC | D64 | the same |
| the DP group's variable-size gather | D66 | `interconnect.intra_node.*` |
| MORI dispatch and combine, for the EP8 step | D66 | `interconnect.intra_node.*` |

Mooncake's KV transfer needs no law here: it is charged `latency + bytes/bandwidth` from
`interconnect.inter_node` (`05` D25). Most of these families are opaque leaves, so their FLOPs
and bytes come from the hand-written extractors of T35, which W5.2 owes for these families
first. The roofline is graded by D67.1.

### The analytical fallback

An op-level backend answers a leaf no measurement prices by its analytical law, always
(#952); at step level the step roofline is the fallback a run file composes (D64). Either
answers with the method its inputs give it, declared while it reads an assumed parameter,
and its share of seconds is always reported (`08` D50.1). A law never moves ahead of a
measurement: one that tracks its measured price within the instrument's repeatability (doc
07 D41) still answers only where no measured price does.

Carried onto another device, as onto MI455 from MI355X, an implementation efficiency is
declared: it was observed on another subject (`08` D50.2).

### What is reported

Per leaf family: the implementation-efficiency distribution, in the calibration report. A
run whose terms are all analytical reports 100% `analytical` in its seconds by method; an
op-level run with gaps reports the mixture, beside the cell's verdict and not gated (doc 08
D50.1).

---

## D69. Device transfer, and the derate problem

Analytical laws are the only route to a device never measured. They have a specific weakness that must
be stated rather than discovered.

**Peak FLOPs and bandwidth are datasheet numbers. Achieved performance is
`peak × derate`, and the derate is a property of the kernel library's maturity on that
architecture — not of the silicon.** For a device you have never run, the derate is a
guess, and it is the dominant term.

> **A roofline on an unmeasured device is as good as its assumed derate, and nothing in the
> procedure validates that assumption: its terms are declared.**

Two honest mitigations, neither of which removes the caveat:

1. **Report sensitivity.** Every roofline prediction on an unmeasured device is reported with
   its derate, and with the prediction recomputed at ±20% derate. A ranking that survives
   that band is more trustworthy than one that doesn't.
2. **Rank, don't predict.** The gate that matters is whether the top-1 configuration choice
   survives (doc 08 D48). A derate error that applies uniformly across configurations may
   leave the ranking intact while moving every absolute number.

---

## D70. Expected accuracy, stated in advance

**D67.1 sets the goals — what the roofline is graded against.** This decision is the separate
and weaker statement: what we *expect* to observe, per use, so that a result cannot be
graded against an expectation invented after seeing it. Where a use has a goal in D67.1,
that goal governs; the rows below cover the uses D67.1 does not gate.

| Use | Expected error | Why |
|---|---|---|
| the roofline, unmeasured device | **unbounded, derate-dominated** | D69. This is why D67.1's goals are graded only where real runs exist. |
| the analytical fallback for a leaf | **unknown per leaf** — that is why its share is reported apart | the calibration report's implementation efficiency for its family says how far it sits from the measured price (D68) |
| analytical memory: weights, KV | **exact** | Class A; D67.1 gates both at ≤5% purely to catch bugs — for KV that is the empirical gate, for weights it is tighter than empirical's ≤10% |
| analytical memory: block count end to end | **unknown** | depends on activations and the Class-C constants |

Single digits on latency are not a target for the roofline and should not be claimed. That is
what the op-level backend exists for.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D63 | Three classes: A exact from geometry, B device-parameterised, C policy-determined. An analytical law works where the quantity is a pure function of stated geometry and fails where a runtime or library policy decides it. `analytical` is #952's method: a form from first principles over geometry, deployment config and datasheet or probed device parameters; an assumed parameter makes a term declared. | 2026-09-19; revised 2026-10-14 (#952) |
| D64 | Roofline leaf `max(F/flops, M/bandwidth)` with derates, plus the host floor as a step-level `max`, not an addend. Revised: every analytical law, the step roofline included, prices device time only; the `max` moves to `01` D4's recurrence, with the launch time from the measured `W_launch` law. Revised: the step roofline's method counts the inputs the winning bound reads. Revised: the step roofline has no host bound; the run-file key `step_roofline` turns it on. | 2026-09-19; revised 2026-10-11 (#815), 2026-10-15 (#1011) and 2026-10-15 (#1047) |
| D65 | An analytical law is a smooth model and says so: it answers a leaf with a known discontinuity in range with a band, as the analytical fallback too. | 2026-09-19; revised 2026-10-14 (#952) |
| D66 | Two-regime collective model with the algorithm named in the spec. Textbook ring/log models are refuted on this fabric. | 2026-09-19 |
| D67.1 | Roofline accuracy goals stated per quantity: ≤5% on weights and on KV capacity/block count (the empirical gate for KV, tighter than it for weights), ≤25% each other non-KV memory term, ≤30% step time, ≤40% end-to-end. **Configuration ranking (top-1 and top-3 set) is the primary gate**; no false "fits". The goals attach to the backend and are graded against real runs. | 2026-09-19; revised 2026-10-14 (#952) |
| D67 | Analytical memory is the existing memory model with the activation coefficient derived rather than walked. Weights and KV exact; scratch stays a stated constant, and `non_torch` and load residue come from a `memory_readings` entry, marked assumed for a device never run. Revised: the coefficient is measured on one real run per model and width, with the scratch inside it rather than a stated constant; no geometry form stands in for it (T34, 2026-10-16, #965). Each term carries a method, and a declared one cannot pass its gate. | 2026-09-19; revised 2026-10-10, 2026-10-14 (#937) and 2026-10-14 (#952) |
| D68 | The analytical laws are authored **during** the empirical campaign, because that campaign is their only check. A leaf family's law-over-measured time is its implementation efficiency, a self-check in the calibration report, off the critical path, and declared when carried to another device. An op-level backend answers an unpriced leaf by its analytical law; no law moves ahead of a measurement. Revised for `16` D95: the laws for DeepSeek-V4-Pro's operator families are on M2's critical path, for the MI455 projection. | 2026-09-19; revised 2026-10-10 and 2026-10-14 (#952) |
| D69 | On an unmeasured device, the roofline is derate-dominated and declared. Report the derate and a ±20% sensitivity band; prefer ranking claims to absolute ones. | 2026-09-19 |
| D70 | Expected accuracy is stated in advance, per use. Single-digit latency error is not a roofline target. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T34~~ | ~~Test whether the activation coefficient `k_model` is derivable from geometry~~ — closed 2026-10-16 by [#965](https://github.com/jgong5/ATOM/issues/965): not within 25% ([#705](https://github.com/jgong5/ATOM/issues/705)) | — |
| T35 | Derive FLOPs and bytes-moved expressions for the ~20 opaque leaves | Class A is free from the IR for aten ops, but an opaque leaf's internal work must be described by hand (doc 04 D20's parameter extractor, extended) |
| T36 | Name the collective algorithm per code path in the machine spec schema | doc 05 D25 has no field for it yet |
| ~~T37~~ | ~~Decide whether `h` (the host floor) is derivable or stays a per-model measured constant~~ — **ruled 2026-10-11 on #815**: host time is measured, as host laws keyed by host CPU, `atom` digest, model and engine config, with no transfer (`09` D54) | 40% apart between two models; it is the largest single Class-C term |
| T38 | Build the implementation-efficiency report (analytical over measured time per leaf family) as part of the empirical campaign | it must exist before the campaign, not after |
| T39 | Establish a dispatch-band table from geometry for leaves with no measured bands | tile sizes and chunk sizes are knowable; whether the *selection rule* is, is not |
