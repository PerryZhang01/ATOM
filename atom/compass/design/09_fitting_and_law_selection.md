# ATOM Compass — Design Topic 9: Fitting and Law Selection

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `07_calibration_toolchain.md` (the measurements being fitted),
`04_model_capture_and_cost_ir.md` (the leaves a law prices).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** the fitted step-level law is the step
linear law (`backends/step_linear.py::StepLinearBackend`), its terms `estimated` and
validated only inside the hull of its held-out steps, with its support domain stated
(D54, D58); a treatment Compass assumes is an assumed treatment (D61).

**Revised 2026-10-18 by the owner's
[band ruling on #1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252):**
at the op level, `measure ops` records each point's kernel identity, and a leaf query with
no same-kernel bracket is declined by the price list and answered by the leaf's analytical
law, counted (D60).

**Scope.** How measurements become a model: objective, feature construction, outlier
handling, guards, and how a functional form is chosen and shown to be right. This is a
separate concern from the toolchain that produces the data and from the protocol that
judges the result.

**Why it needs its own document.** Nearly every wrong number in the prior effort came from
a fitting decision, not a measurement error — the measurement was usually fine. Four
distinct failures: an objective that weighted by magnitude, a feature computed against the
wrong rectangle, a guard that could not see a hole, and a family validated by a method that
cannot validate families.

---

## D53. Fit relative error, not absolute seconds

### The failure

Ordinary least squares on squared seconds makes a 250 ms sample count roughly **60×** a
32 ms one. The symptom was diagnostic and almost invisible: widening the sweep to cover
decode rungs 32–64 added large prefill samples, dropped the share of sub-1024-token prefill
steps from **17% to 9%**, doubled the median token count to **13,008** — and moved the
512-token prediction from **−13% to −17%**, *against a measurement that did not move*
(32.51 → 32.53 ms).

> **An unweighted fit is a weighted average whose weights are the sample values.** Adding
> evidence in one region degraded prediction in another, which is not something a coverage
> check can catch.

### Decision

**Every equation is divided by its own target before fitting, and the MAD outlier test runs
on the same relative residuals.**

| prefill prediction | absolute fit | relative fit | measured |
|---|---|---|---|
| at 512 tokens | −11.8% | **−5.3%** | 32.53 ms |
| at 2512 tokens | +18.3% | **+3.9%** | 43.78 ms |
| median over 121 rows | 9.51% | **9.03%** | |

---

## D54. What is fitted separately

**Prefill and decode are separate fits.** Prefill is compute-bound in new tokens; decode is
bandwidth-bound in KV history. Total context is **summed** across the batch, never
averaged.

**Decode is fitted per CUDA-graph rung, with *both* coefficients per rung:**

| decode model | median \|err\| | RMSE | worst region |
|---|---|---|---|
| `[1, batch, total_context]` | 5.04% | 0.4216 ms | −3.7% |
| per-rung intercept, shared slope | 8.09% | 0.3952 ms | −12.6% |
| **per-rung intercept and slope** | **0.93%** | **0.1259 ms** | **−1.3%** |

In place over 2,174 measured decode steps: **0.73% median**, per rung 0.46 / 0.45 / 0.69 /
1.67 / 2.11 / 1.57 / 0.25 / 0.37% for rungs 1 → 64.

### A sign that was wrong, not merely a curve that was

**Decode cost *falls* as batch grows.** At matched total context over 1,662 decode steps:

| batch | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| TP=1 | 3.640 | 3.564 | 3.393 | 3.317 | 3.532 ms |
| TP=2 | 3.636 | 3.563 | 3.403 | 3.351 | 3.310 ms |

Twelve sequences cost *less* per step than one. A model assuming monotone growth in batch
is wrong in direction, not just in magnitude.

### Under DP and EP: one per-rank law per engine config

Owner's DP ruling, 2026-10-01
([#470](https://github.com/jgong5/ATOM/issues/470#issuecomment-5933154215)). The step linear law covers
DP and EP with a **per-rank** law: it prices one rank's step, and the group's device end is
the `max` over ranks (`15` D90).

- **Key.** The engine config, the one source of the parallel layout (owner ruling on
  [#973](https://github.com/jgong5/ATOM/issues/973#issuecomment-6210598012), 2026-10-16):
  a law at one layout is no evidence about another (`15` D89), so a run under another
  engine config refuses it, naming each field that differs. A law stating `(dp, ep)`
  is refused, and no store key holding the engine config states `parallel`.
- **Features.** The rank's own batch features (D55), plus two the group decides: the
  DP-unified rung every rank shares, settled by `ForwardMode.decide`, and the MoE row
  count, which the MoE terms read alone. That count is `T_moe` (`BatchView.moe_rows`)
  with EP on or off: with EP off the rows the group's MoE gather carries, with EP on the
  rows the group dispatches to MORI (`15` D90).
- **The capture.** Full-engine steps on a real deployment at the law's layout,
  logging per forward each rank's batch features and the MoE row count beside the
  measured step seconds. Under DP-attention the local segment runs at TP 1, as in the
  TP1-only full-engine calibration of `07` D38; with EP off the MoE segment does not,
  since `flatten_tp_across_dp` shards every expert `dp` ways; with EP on it does, each
  rank running its own experts whole on the rows MORI delivers.
- **The fit: alternating max-affine regression.** The ranks wait for each other inside the
  forward, so a rank's measured step is the group's step and its own cost is never seen:
  the observation is `y = max_r f(x_r)`. The fit alternates two moves until no step
  changes rank: assign each step to the rank with the largest prediction under the
  current coefficients, then refit the one law on the assigned ranks' features, each
  equation divided by its target (D53). This is the least-squares partition heuristic for
  max-affine fitting (Magnani and Boyd, 2009), with the ranks in place of the affine
  pieces and one coefficient vector shared by all of them.

### The label: device time per step kind, and the host laws beside it

Owner ruling on [#815](https://github.com/jgong5/ATOM/issues/815), 2026-10-11, superseding
ruling 2 of
[#627's 2026-10-10 plan](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740).
That ruling labelled a step with its period, from one forward's host entry to the next,
with host gaps in the intercepts. A pure middle chunk's device tail then lands in the next
step's period: about +4% after idle and -8% after a pure middle chunk, at c1 and c4
(#801). That label and the law fitted under it are retired
([#845](https://github.com/jgong5/ATOM/issues/845)): a record without device windows has
no label, and the fit and the cost check refuse it.

**The step linear law's label is one forward's device time**, built from the window from a device
event at `run_model` entry to one at the forward's exit (`01` D4, *Measured, not hooked*),
and the device gap, from the previous forward's exit event to this forward's entry event.
Owner ruling on [#843](https://github.com/jgong5/ATOM/issues/843#issuecomment-6116328961),
2026-10-12, sets it per step kind:

| step kind | label | launch-exposed cut |
|---|---|---|
| decode | device gap + busy window | kept |
| output prefill, middle chunk | busy window | none |

- **Decode.** The gap is device work between forwards that no other term carries; leaving
  it out charged decode about 0.29% short on #824's c=1 record (#841). A forward whose
  device start precedes its host launch end is cut as launch-exposed and counted apart: a
  launch event that finds the device idle completes on enqueue, so the cut also keeps
  device idle out of the gap.
- **Prefill and middle chunks.** No gap: after an idle engine the gap holds the idle time.
  No cut: under an eager launch the device starts the first kernel while the host still
  launches the rest, so device start precedes launch end on every eager forward, and the
  cut would leave prefill unpriced.
- **Accepted approximation.** A host bubble inside a prefill forward stays in its device
  time. Same-host parity absorbs it through `01` D4's `end = max(target_end,
  launch_end)`; skipping a host segment or pricing another host does not.

The R1 wait's exit, a host time at which a known device event has completed, anchors the
host and device clocks to each other. With MTP the label splits at
`forward_done_event` into target and draft
([#829](https://github.com/jgong5/ATOM/issues/829)).

**Host laws** price the host segments of `01` D4: `E`, `W_pre`, `W_launch` and
`W_post`, each linear in `(1, tokens, seqs)`, fitted separately for decode, output
prefill and pure middle chunk. A pure middle chunk returns from `ModelRunner.forward`
before `postprocess`: no sampler launch, no output and no R1 wait, so a prefill law that
mixed the two would misprice both. Its `W_post` is zero and not fitted. On a record
that stamps the MTP status wait, an output forward's `W_pre` ends at that wait and a
further law, `W_inputs`, is fitted from it to the launch
([#829](https://github.com/jgong5/ATOM/issues/829)).
They use the step linear law's solver: D53's relative error and MAD drops, D57's report, D58's hull. An
`E` sample whose gap spans an idle engine is cut as idle and counted apart.

- **Key:** the host CPU, the `atom` source digest, the model, the engine config and the
  machine-spec digest (`07` D41). A mismatch refuses. There is no transfer mode: the MI455 projection prices no host floor
  (`08` D50.2).
- **Home:** beside the step linear law, not in the machine spec's `host.*` (`05` D25).
- **Use:** the runner composes them with any backend's device time by `01` D4's
  recurrence; no backend adds them to its own price.

---

## D55. Feature construction

```
prefill:  [ 1,  tokens,  Σ_req N_Q²,  Σ_req N_Q·(ctx − N_Q) ]
          + history: Σ_req (ctx − N_Q), in `step_linear_fit`'s `history` form
decode:   [ 1,  batch,   Σ ctx ]
per rung: [ 1,  Σ ctx,   rung·max(ctx) − Σ ctx ]
          + rowmod(k), on a rung that adopts one (D61)
```

Two constructions are load-bearing and both were originally wrong.

**1. Attention terms are summed per request**, not computed from batch-collapsed scalars.
Collapsing the batch to `tokens × history` and multiplying was a rank deficiency. ATOM
already computes the correct form: `ScheduledBatch.detailed_sqsq / detailed_sqsk /
detailed_sk` are **Σ N_Q², Σ N_Q·N_KV, Σ N_KV**, produced by `compute_detailed_aggregates`
(`scheduler.py`).

**2. The padding term is the *rung's* rectangle, not the batch's.** `rung·max(ctx) − Σctx`,
**not** `len(ctx)·max(ctx) − Σctx`. Error tracked the ratio between them exactly:

| rung | ratio | error with the wrong rectangle |
|---|---|---|
| 2 | 1.00 | −0.34% |
| 8 | 1.40 | −2.41% |
| 4 | 2.26 | −3.91% |
| **16** | **5.59** | **−22.64%** |

---

## D56. A rank deficiency is not a coverage gap

### The failure, and how it hid

Decode error at rungs 8 and 16 was diagnosed as missing evidence. It was not — it was **a
column of zeros where a feature should be**. Every calibration round ran sequences of one
length, so raggedness was exactly **1.00 in all 2,997 samples**. Filling the *context* gap
moved rung 8 from **−32.6% to −32.1%**. Adding ragged rounds plus the padding feature took
it to **−7.4%**.

### And the second-order trap

> **A wrongly-computed feature reads as an unconstrained one.**

Two rounds of sweep changes were spent on the theory that the padding coefficient was
unconstrained. It was constrained — it was fitted to the wrong number (D55). The symptom
worth remembering: **widening the evidence twice moved rung 16 by 0.07 points.**

### Decision

Before concluding "we need more samples", check the design matrix:

1. **Any feature with no variance in the samples is dropped and its coefficient returned as
   zero**, explicitly, in the fit's description — never silently absorbed into the
   intercept.
2. **Report the condition number and the per-feature variance** with every fit.
3. **If widening the evidence does not move the error, the feature is not unconstrained —
   it is wrong.** That is a diagnostic rule, not a heuristic.

---

## D57. Outliers, and what a fit must report

- **MAD rejection on relative residuals, then refit.** The number dropped is reported in
  the fit's description, never silently.
- **A fit's description names its provenance** (`07` terminology), its sample count, the
  dropped count, the condition number, and the hull it is valid over.
- **Report an interval or report nothing.** A difference smaller than ~5% needs repeats
  before it means anything, and repeats must be spaced (`08` D45).

---

## D58. Guards: refuse an unseen kind, extrapolate outside the hull

### Refuse rather than fall back

An oracle asked about a step kind it has no samples of **raises**. The prior fallback was
the mean of an empty list — zero — producing *"a confident, precise, entirely fictional
answer"*: a TTFT of **0 ms against a real 7.6 s**.

### The extrapolation guard must see the joint distribution

A per-feature guard **cannot see a hole**. The concrete case: batch 8 was covered (1–16
seen), context ~2,650 was covered (57–41,052 seen), and nothing warned — but the sweep's
batch-8 steps run 1,776–2,288 and then jump to 5,360. *"The query sits in a hole, and the
guard reported it as interpolation."*

This cost **three** iterations to abandon:

1. context bounds hid a gap — rung 8's bounds `[264, 327168]`, a step at 33,045 inside them
   *and* inside a gap; 117/117, 512/512 and 1,642/1,642 real steps fell in the hole
2. raggedness bounds hid a gap
3. rung 16 was covered on both axes **separately** and still came out **22.6% low**, holding
   64 samples at raggedness exactly 1.00 against a run at 1.18–1.32

**Decision: a convex hull or k-NN distance with a threshold, never a per-feature bounding
box.** The k-NN oracle already computes the distance. The guard decides the answer's
source and validation, not whether a step is answered: a step outside the hull is priced
by the same law, `estimated` and unvalidated, its `source` naming the extrapolation, and
reported apart, not refused (`08` D50.1; #627's
[ruling 3](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740)).

**The step linear law's support domain** (#952, `backends/step_linear.py`). A discrete
value it has no vector for refuses: a prefill step when the law has no prefill vector, a
decode rung with none. Outside the continuous domain, the fitted hull, it extrapolates as
above. A term is `validated` only inside a second hull, the steps `step_linear_fit`
priced with a law fitted on the other records, whose errors per band it writes into the
law beside that hull; a law without one prices every term unvalidated.

And the dimensions are **workload-dependent** — 0.6B batches ran at raggedness 2.82–3.85,
27B at 1.11–1.42 — so:

> **"The sweep covers this" is a statement about a pair, never about a sweep alone.**

---

## D59. Law selection, and why leave-one-out cannot do it

### The failure

Attention cost was fitted as a power law `a·L^p` on three geometrically spaced lengths:
LOO median **0.75%**, worst 1.08%. Two independently traced graphs at 3,194 and 6,594 then
came in at **−6.2% and −6.7% — the same direction**. Cross-run repeatability is 0.2%, so
this was model bias, not noise.

> **LOO perturbs the sample; it does not test the family.** Where a sample is small and
> geometrically spaced, LOO reports how stable the fit is, not whether the family is right.

### What the data actually looked like

A dense 13-point ladder (2,294 → 10,094 in steps of 600) shows cost per token sitting on
**flat plateaus**: 3.67, 6.28, 6.10, 5.90, 6.01, 8.31, 8.45, 8.17, 8.22, 10.58, 10.41,
10.47, 10.50 µs/token. Four plateaus, steps of ~2.27 µs/token, plateau index
`ceil(L / 2530)`.

**The quadratic term is computed in whole chunks of keys** — attention is quantised in
sequence length. `f(L) = L·(a + b·ceil(L/S))` with a = 1.461 µs, b = 2.272 µs, S = 2530
gives LOO median **1.59%** against 7.78% for the power law, and predicts the independently
traced graphs to **0.13%, 0.62%, 1.21%**.

### And it still does not survive batching

`2×3194` is **+23.9%**. Tracing 3,194 at one and two sequences: **19.967 ms/seq at one,
15.470 ms/seq at two** — a 22.5% drop per sequence from adding a neighbour, while the same
comparison moves 2.8% at 2,294 and 0.9% at 4,694.

> The chunking is a property of the **batch**, not the sequence.

### Decision

1. **A family is validated on independently gathered data, never by LOO.** LOO ranks
   candidates within a family; a held-out measurement decides the family.
2. **Rank at least three candidate laws** and report all of them with their held-out error,
   not only the winner. The prior tool ranked five.
3. **A law that fits one axis must be tested on a second** before it ships — the batching
   test above is the pattern, and so is the rotary-buffer formula that matched the 0.6B
   exactly and was **4× wrong** on the 27B. *Tested on a second model, failed, did not
   ship.*

---

## D60. Cost is piecewise; interpolation is not sound-but-imprecise

### The cliffs are real and reproducible

**GEMM tile cliff.** One gemm at 20 values of M: flat at ~9.2 µs to M=12, then climbing,
then **jumping 58.07 µs at M=768 → 137.45 µs at M=1024 where the trend says ~77**. Each
signature already records which kernels it launched: M=256/384 → `MT128x1…`; 512/768 →
`MT256x1…`; 1024 → `MT256x2…`. **The price list already knows the band.**

**Prefill is non-monotone in token count, reproducibly** (five repeats each, tight to a
tenth of a percent):

| shape | tokens | step |
|---|---|---|
| 8×100 | 3,952 | 55.32 ± 0.07 ms |
| 4×200 | 4,376 | 63.16 ± 0.02 ms |
| 2×400 | 4,588 | **52.14 ± 0.03 ms** |
| 1×800 | 4,694 | 65.13 ± 0.05 ms |

Two independent causes: a gemm tile cliff (M=4,376 at 536.80 µs `MT256x224x64` against
M=4,500 at 259.68 µs `MT256x192x64`) and attention's quadratic. And the extreme case: at
~9–10k tokens, **one long sequence costs 174.5 ms where four shorter ones totalling 9,176
tokens cost 100.3 ms — 74% more time for 10% more tokens.**

### What interpolation measures

| measured shapes | n | median | worst |
|---|---|---|---|
| every other shape | 10 | 5.5% | **57.8%** |
| powers of two | 11 | 2.8% | **60.1%** |
| 1, 8, 64, 512 | 4 | 3.7% | **64.7%** |
| 1, 32, 1024 | 3 | 8.6% | **80.1%** |

And the obvious guard — interpolate only between same-kernel neighbours — **refuses almost
everything**: twelve distinct tile configurations across twenty M values, so 8 of 10
held-out shapes have no same-kernel bracket. The two that do come out at 0.4%.

> **Interpolation is not sound-but-imprecise; it is accurate most of the time with no way to
> tell when it is not. For a tool whose output is a ranking of configurations, an occasional
> silent 60% is worse than a uniform 10%.**

Prefill steps do not interpolate either, even averaged over hundreds of kernels: median
**12.2%**, worst **32.1%**, against 2.1% for a measured ladder.

### Decision

**Laws are piecewise over dispatch bands.** A band boundary is read from the
recorded kernel identity, not inferred from a residual. A query between bands with no
bracket **refuses** rather than interpolating.

**At the op level**, revised 2026-10-18 by the owner's
[band ruling on #1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252):
`measure ops` records the kernels each leaf launched at each point, and `fit ops`
interpolates a leaf only between two neighbours that launched the same kernel (`07`
D38). The refusal is the price list's: the leaf falls to its analytical law, and that
share of priced seconds is counted (`08` D50.3). The cost is the larger analytical share
the measurements above predict, 8 of 10 held-out GEMM shapes with no same-kernel bracket,
and it is reported rather than hidden. Revised 2026-10-19 by the
[ruling on #1493](https://github.com/jgong5/ATOM/issues/1493#issuecomment-6260357438):
a line whose held-out points show a hidden band (a point that launched other kernels
than the neighbours it is priced from) or an error over 5% does not interpolate. Its
measured points still answer; every other query along it goes to the analytical law,
and the price list names each such line with its reason.

---

## D61. Assumed treatments

A treatment is a factor that moves cost and is **invisible to every feature in the model**.
It must be stated and the calibration population matched to the target, or the fit is
silently averaging over it. A treatment Compass assumes rather than observes is an
**assumed treatment**: a property of the query, recorded in the run summary (#952).

**Decode row order: a treatment, then a feature.** At one fixed 32-row context multiset,
row order alone moves measured decode attention across **1.77×** — grouped-descending 276.1 µs,
grouped-ascending 292.2 µs, alternating 376.5 µs, ladder-interleaved 489.5 µs. Refuted as
causes: KV pool footprint (2.42× pool maximum moves the price −0.5%), co-residency, shape,
and launch-wave packing.

> Every feature in every candidate law — `crit_waves`, `work_waves`, `launch_waves` — is a
> function of the context multiset, so all four orders are the same vector. **No
> coefficient over those features can separate them, and a corpus that mixes orderings is
> fitting an unstated treatment.**

Consequence, already in `07` D39: calibration batches are **replayed** from the real
scheduler's step table, with explicit `block_tables`. Hand-built ladders are grouped by
construction.

**Since #941, row order is a decode feature of the step linear law, not a treatment.** `rowmod(k)` is the
largest summed context over the rows `i ≡ r (mod k)` of a batch in scheduled order; `k = 1`
is `Σ ctx`. It names no kernel, split rule, head count or CU count. A rung adopts one `k`
as a fourth column (D55) only when all four hold:

1. the column cuts the rung's scaled MAD, by at least a tenth;
2. shuffling each step's rows takes that cut away, read as the mean MAD over 8 seeded
   shuffles;
3. no calibration cell, held out of a fit on the others, gets a p50 error further from zero;
4. every cell holds at least 100 steps at the rung, and there are two cells or more.

Candidates are the powers of two up to the rung; the qualifying `k` with the smallest MAD
wins, and the law records `k` with the MADs and held-out errors that chose it. A rung
that adopts none keeps its three columns and prices exactly as before. Pricing reads the
rows in the order the scheduler built the batch, the order the real engine's attention
metadata takes them in; a simulated run prices the batch ATOM's own scheduler built.

On the hjbog-srdc-39 law's records the rule adopts `k = 16` at rung 32 alone: scaled MAD
7.59% to 2.21%, held-out p50 −13.7% to −2.2% (c=64) and +11.7% to +3.7% (c=256)
([evidence](https://github.com/jgong5/ATOM/issues/941#issuecomment-6162749853)). At rung 32
the gfx942 decode attention grid places workgroups so that the rows `i mod 16` share a
dispatch class; the fit found 16 from the data alone.

**Assumed treatment: uniform routing** (owner's DP ruling, 2026-10-01). Under DP the
MoE segment is priced under independent uniform routing over the experts, at every
granularity, from `T_moe` with EP on or off (`15` D90); the analytical laws price it in
expectation, under EP at the expected maximum rank load (`10` D64). Real
routing is data-dependent (`15` D92 Q2) and no feature carries it, so whatever skew a
capture's routing has is folded into the fitted MoE terms. `max` over ranks is the
per-layer critical path, by construction with EP off and under this treatment with EP on
(`15` D90); where routing is not uniform under EP, `max` underestimates the step.
Stage 3 keeps this treatment at EP8 (owner ruling on
[#627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6194442788), 2026-10-15):
the step linear law fitted under the EP8 engine config absorbs MORI dispatch and combine
through its prefill `moe_rows` term and its intercepts, and no received-rows feature is
fitted. A decode vector needs no `moe_rows` term, since at a replayed rung `T_moe` is
fixed by the rung and the rung's intercept holds it. The measured received-row
imbalance, whose size `15` D90 states, is reported beside the law's accuracy
([#1108](https://github.com/jgong5/ATOM/issues/1108)).

**Suspected treatments to check before fitting:** raggedness (workload-dependent, and it
was a zero column for 2,997 samples), cached fraction, and the prefill chunk's position in
its streak.

---

## D62. Two biases that must be modelled or excluded, not left to cancel

**1. Leading warmth does not transfer, and charging it is wrong.** The 27B sweep's first
three prefill steps cost **47.5 / 19.5 / 6.1 s** at 8 / 32 / 96 tokens where the same shapes
later cost **0.11 s**. Replaying that onto a simulated run's first step gives **+31.03%** on
the prefill total against a real cold start of 6.87 s. *"A server does a profile run and
captures graphs before its first measured step, so most of the process-level warmth is
spent by then; a sweep meets it head-on."*

**Decision:** the fit **measures and reports** its leading warmth and charges nothing
(`leading warmup in table = 3 steps totalling 72.80 s, not charged`). A per-deployment
`warmup_seconds` constant defaults to 0 — measured at 6.68 s on 27B TP=4 (steady to 1.3%
over four repeats), 0.01 s on the 0.6B.

**2. The prefill bias is shape-dependent, so scaling cannot fix it.** Split by the history
each chunk attends over:

| history band | chunks | error |
|---|---|---|
| 16k–50k | 44 | −1.16% |
| 50k–100k | 50 | +3.74% |
| 100k–150k | 11 | **+13.36%** |

Not extrapolation — the sweep has 16 samples in that band and reaches 258k. *"It is a
balance problem: 77% of the sweep sits in 16k–50k, the workload never goes past 131k, and
one linear coefficient is fitted across the whole range."*

**Decision:** report error **stratified by the axes the law is linear in**, not only in
aggregate. A single coefficient across a range the workload samples unevenly is a design
choice that must be visible.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D53 | Fit relative error, not absolute seconds. MAD outlier test on the same relative residuals. | 2026-09-18 |
| D54 | Prefill and decode fitted separately; decode per CUDA-graph rung with both coefficients per rung. Under DP and EP, the step linear law is one per-rank law per engine config, which states the parallel layout, with the DP-unified rung and the MoE row count as features (`T_moe`, EP on or off), captured on a real deployment at that layout and fitted by alternating max-affine regression. Its label is one forward's device busy time, plus the device gap before it on a decode, with launch-exposed decodes cut and no prefill cut; host laws (`E`, `W_pre`, `W_launch`, `W_post`, and under MTP `W_inputs`, `W_pre` split at the MTP status wait), each `(1, tokens, seqs)` per decode, output prefill and pure middle chunk (a middle chunk has no sampler launch, output or R1 wait, so its `W_post` is zero and not fitted), are fitted beside it, keyed by host CPU, `atom` digest, model and engine config, with no transfer. | 2026-09-18; revised 2026-10-01, 2026-10-09 (`T_moe`, #502) and 2026-10-11 (#815); label per step kind 2026-10-12 (#843); named 2026-10-14 (#952); `W_inputs` 2026-10-16 (#829); EP on 2026-10-15 (#627, #1106); engine config as the key 2026-10-16 (#973) |
| D55 | Attention terms summed per request; the padding term is the rung's rectangle, not the batch's. A decode rung may add `rowmod(k)` (D61). | 2026-09-18; revised 2026-10-14 (#941) |
| D56 | Distinguish rank deficiency from coverage gap: drop zero-variance features explicitly, report the condition number, and treat "widening the evidence does not move the error" as evidence the feature is wrong. | 2026-09-18 |
| D57 | Every fit reports provenance, sample count, dropped count, condition number and validity hull. Report an interval or report nothing. | 2026-09-18 |
| D58 | Refuse rather than fall back on a step kind with no samples. The extrapolation guard is a convex hull or k-NN distance, never a per-feature bounding box; an out-of-hull step is priced `estimated` and unvalidated, its source naming the extrapolation, and reported apart, not refused. The step linear law refuses a discrete value it has no vector for, extrapolates outside its fitted hull, and is validated only inside the hull of its held-out steps. | 2026-09-18; revised 2026-10-11 (#765) and 2026-10-14 (#952) |
| D59 | A family is validated on independently gathered data, never by LOO. Rank at least three candidates and report all held-out errors. A law that fits one axis is tested on a second before it ships. | 2026-09-18 |
| D60 | Laws are piecewise over dispatch bands read from recorded kernel identity. A query with no same-band bracket refuses. At the op level `measure ops` records each point's kernels and `fit ops` interpolates only between same-kernel neighbours, and along no line holding a hidden band or a held-out error over 5%; a declined leaf falls to its analytical law, counted. | 2026-09-18; op level 2026-10-18 (#1489); declined lines 2026-10-19 (#1493) |
| D61 | Treatments are stated and the calibration population matched; an assumed treatment is a property of the query, recorded in the run summary. Decode row order, worth 1.77× at a fixed multiset, is a decode feature, `rowmod(k)` in scheduled order, adopted per rung by a four-condition rule. Uniform routing of the MoE segment under DP and EP is an assumed treatment, at every granularity. | 2026-09-18; revised 2026-10-01, 2026-10-14 (#941, #952) |
| D62 | Leading warmth is measured and reported, never charged. Error is reported stratified by the axes the law is linear in. | 2026-09-18 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T29 | Choose the hull implementation (convex hull vs k-NN distance threshold) and its threshold | needs one calibration corpus to tune against |
| T30 | Enumerate candidate laws per leaf family, with their held-out validation shapes | needs the leaf list frozen (`04` T3) |
| T31 | Test raggedness, cached fraction and chunk-position for treatment status | each needs a controlled experiment like the row-order one |
| T32 | Decide whether the step linear law is derived from the op level's composed prices or fitted independently | `04` T8; if derived, most of this document applies only to leaves |
| T33 | Establish `warmup_seconds` for Qwen3.8-27B under the current stack | per-deployment constant, currently defaulting to 0 |
