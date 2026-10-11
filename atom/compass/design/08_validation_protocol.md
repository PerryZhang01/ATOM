# ATOM Compass — Design Topic 8: The Validation Protocol

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `07_calibration_toolchain.md` (the artifacts being validated),
`06_workload_harness_contract.md` (the workload driving both sides).

**Revised 2026-10-09 for `16` D95:** the acceptance cells are M2's (D50), the TTFT target is
15%, the ranking gate is explicit (D50), and the MI455 projection has its own gate (D50.2).

**Scope.** What counts as evidence: which results are reported, how each is measured, what
tolerance applies and where that tolerance comes from, and what invalidates a run.
Acceptance is paired simulated and real execution of cc-traces proper, so every rule here
applies symmetrically to both sides.

---

## D43.1. ATOM's own test suite is the first validation layer, in two tiers — one of them free

Asked directly: can Compass reuse existing ATOM unit tests? **Yes, and the CPU tier of that
suite is the cheapest correctness evidence in this document — but it validates a different
property than everything else here.**

**Revised 2026-09-20; the original claim was measured false.** This decision used to read
*"187 test files under `tests/`, and ATOM's own `CLAUDE.md` states they need **no GPU** —
AITER and `torch.cuda` are mocked"*. Neither half survived measurement. At `fada7424e` there are **189**
test files under `tests/`, and the **29 of the 159 outside `tests/plugin/`** that
`scripts/compass/cpu_gate_exclude.txt` lists **reach the driver**, each run alone: 28
at *collection* time, via `rocminfo` reached on import, so pytest cannot even build the node
list, plus 1 that collects cleanly and then fails at run time on a pinned-host allocation
(`hipHostMalloc failed: 100`). That numerator is non-plugin, so its denominator must be too —
quoting 29 against 189 is the defect `16` L4 fixes and it is not repeated here. **At the
whole-suite denominator the figure is at least 34 of 189**, because `tests/plugin/` is not
purely a packaging problem: collecting it alone at `236abfd9a`, and again at `fada7424e`, in `xiaobizh_n18_cpu` gives
`153 tests collected, 7 errors`, rc=2, and those 7 decompose as **3 `rocminfo`**
(`test_gdn_target_verify_batched_equiv.py`, `test_rtpllm_forward_context_semantics.py`,
`test_vllm_deepseek_v4_proxy_state_arena_layout.py`), **1 `ModuleNotFoundError: No module
named 'sglang'`**, and **3 `ImportError: cannot import name 'fused_gdn_gating' from
'atom.model_ops.attention_gdn' (unknown location)`** — a module left half-initialised by the
first three, not a fourth cause. With the 2 tier files the table below names, which reach
`rocminfo` run alone at `fada7424e` too, 29 + 2 + 3 = 34 is a floor: the other 23 plugin files collect
here and are never run, so nothing is measured about what they would touch.
`tests/plugin/`'s 30 files at `fada7424e` are dropped whole because the tier targets ATOM and sglang
and vllm are in neither image — the measured reason for 27 of them, not for all 30. The
suite as a whole is **not** GPU-free and never was. What is GPU-free is a *tier* of it, and
that tier is genuinely green.

### The two tiers

| Tier | What runs | Measured result |
|---|---|---|
| **CPU tier** — per task, and green is the bar | Every test file outside `tests/plugin/` and `scripts/compass/cpu_gate_exclude.txt`. Driven by `scripts/compass/gate_cpu.sh`; no driver is touched **as a batch**. That is the weaker and correct claim: `test_postprocess_width.py` and `test_v4_checkpoint_slot_copy.py` are both in the tier, and each reaches `rocminfo` when run *alone* (`no tests collected, 1 error`, rc=2, each, measured at `236abfd9a` in `xiaobizh_n18_cpu`); inside the gate they module-skip on the mock an earlier file installs. Driver-freedom here is a property of the batch, not of every file in it. | Green, `GATE_CPU_RC=0`, read at `bdd244c57` in `xiaobizh_n18_cpu`. The counts are not restated here, because they move with every test the tier gains or loses: `gate_cpu.sh` prints them under the `commit:` line of the tree it ran, and `scripts/compass/README.md` says what the tier runs. |
| **GPU superset** — per wave, judged as a **delta**, never as "green" | `tests/ --ignore=tests/plugin` in the GPU container, driven by `scripts/compass/gate_gpu.sh`. | **4779 passed / 5 failed**, 0 errors, 105 skipped, 3 xfailed, 72.6 s, at `fe9ea043c` on node 18 in `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20 — torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509**; two runs, byte-identical failing sets. The five are pre-existing and unrelated to Compass, and they are **four ULP comparisons plus one bitwise check**, not "five bf16 ULP failures": four `allclose` cases in `tests/test_fused_compress_ragged.py` off by one bf16 ULP (`max\|diff\| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3`), plus `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise`, a `torch.equal` with **no tolerance at all** — a tolerance bump would not move it. All five node-ids are on file verbatim in `scripts/compass/gpu_gate_known_failures.txt` and compared by name, so "5 failed" can now be checked against "the *same* 5 failed". |

`test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` were once excluded and were re-measured CPU-green (`16`); prose
saying those three need the driver is stale.

### The tests that matter to Compass, and which tier each is in

| Test | Tier | Covers |
|---|---|---|
| `test_scheduler.py` | CPU | `Scheduler` public API — admission, `ScheduledBatch`, `ScheduledBatchOutput`, spec stats |
| `test_prefill_scheduler.py`, `test_scheduler_partial_prefill_tail.py` | CPU | chunked prefill boundaries |
| `test_block_manager.py`, `test_block_pool.py` | CPU | the block accounting D13 relies on running unmodified |
| `test_prefix_cache_accuracy.py` | **neither** | nothing. It holds **no test function at all** — it is an `argparse` script that drives a live server on `localhost:8000`, and `pytest` reports `no tests ran` for it in both containers. The prefix-cache behaviour doc `03` declares correct-by-construction is not covered by this file. |
| `test_prefill_prefix_vs_native.py` | **GPU only** | the prefix-cache behaviour doc `03` declares correct-by-construction. It module-skips on the CPU tier ("needs a real GPU") and gives 4 passed in the GPU container (`xiaobizh_n18`, one card, at `cb684287f`), so the prefix-cache row is GPU-tier evidence and nothing else. |
| `test_scheduled_batch_marshal.py` | CPU | the IPC payloads the channel wrappers timestamp (`01` D4, K4/K5) |
| `test_block_table_marshal.py` | **GPU only** | the block-table half of those same payloads — excluded at collection time, so the CPU tier does not see it |
| `test_kv_connector_scheduler.py` | **neither** | the connector factory doc `01` D6 registers a simulated connector into — except that it runs nothing in either tier: `1 skipped`, because ATOM #690 split `kv_transfer_engine` into `moriio` and the test's imports were never updated. |
| `test_disagg_modes.py`, `test_disagg_types.py` | CPU | the PD paths of M1's simulated 1P+1D run and of M2's 1P+1D cells |
| `test_dp_load_balance.py`, `test_dp_metadata.py`, `test_dp_sync_layout.py` | CPU | the DP paths of M2's DP-attention cells |

**The two "neither" rows above are a finding, not a gap in the table**, and there is a
third file in the same position that was never listed here at all.
`test_prefix_cache_accuracy.py`, `test_kv_connector_scheduler.py` and
`test_transfer_engine.py` run nothing in *either* tier — measured per file by P0.2 in both
containers, `no tests ran` for the first and `1 skipped` for the other two. That
measurement was taken at `83daf636d`, a tree superseded for every pass count in this
document; it is retained because it is a statement about whether these files contain
runnable tests, which no later re-measurement has changed. Two of the three were cited here
as coverage Compass keeps, and are not. Raised with ATOM's owners as **T80** in `16`.

### What this does and does not prove

**What it proves, and it is exactly the claim Compass's central design choice rests on:**
that the scheduler, block manager and admission logic still behave identically after
Compass's changes. Every one of doc `01` D4's edits — clock reads on the LP clock (K2), idle points as
`next_event` (K3), timestamped sends and counted receives (K4, K5), timers on the LP clock
(K7) — lands in code these tests cover. A
simulated run "makes the same scheduling decisions as a real one" is not a hope if the CPU
tier passes unchanged and the GPU superset's delta is zero; it is the definition of what the
suite checks.

**What it does not prove:** nothing about accuracy. No ATOM test knows what a step should
*cost*. The three results of D44 below are untouched by it.

### How it is used, as four rules

1. **The suite is a merge gate, run on every Compass change, unmodified.** Not adapted, not
   subsetted by hand. The CPU tier's exclusion list is not an exception to that: it is
   *generated* — `regen_cpu_gate_exclude.sh` iterates collection to a fixed point — and every
   hand-kept entry carries its observed failure above it, so an inconvenient test cannot be
   quietly removed. The moment a Compass change requires editing an ATOM test to keep it
   green, that change has altered ATOM's behaviour and needs justifying on its own terms.
2. **A red test is never "expected under simulation".** If a clock substitution breaks
   `test_scheduler.py`, the substitution is at a business-logic site that changes a
   decision — which is precisely the sorting rule doc `01` D5 exists to apply, and the test
   found a misclassification.
3. **New Compass components get tests in the same suite**, in `tests/compass/`, in ATOM's
   style, so they run in the CPU tier on the same driverless box. The clock protocol, the
   straggler detector and the cost-backend interface are all testable without a device.
   Those tests are the 74 in the decomposition above.
4. **The CPU tier refuses rather than reporting "not required" (principle 6).** Green at the
   CPU tier is not green for the task when the diff lands in what the CPU tier cannot see. That
   blind spot is a file, not a sentence: `scripts/compass/gpu_gate_triggers.txt`, **30 paths**,
   generated rather than hand-written — the counts in its own header included. The rule:
   an `atom` module named by an excluded test is a blind spot unless a CPU-tier test that
   **actually runs** names it too. Imports are read at any indentation on the excluded side
   (190 of this tree's `import atom.*` lines are indented), but coverage is credited only for
   a module-level import in a CPU-tier file that collects at least one test — the only
   imports that provably execute. `regen_gpu_gate_triggers.sh` reproduces it, and a trailing
   `/` matches a subtree.
   `gate_cpu.sh` exits **98** when a changed path matches a trigger and
   `COMPASS_GPU_GATE_DONE=<sha>` is absent or names a different tree — and also when it
   cannot determine the diff at all (no git, no `COMPASS_CHANGED_FILES`) or cannot name the
   tree the attestation would apply to. An unanswerable question is refused, not answered
   "no". The previous form of this rule was a prose list of areas — "EPLB, DP metadata,
   cudagraph bounds, block tables" — which both missed modules and named two the CPU tier
   does cover.

   **The trigger file errs in both directions, so do not cite it as a floor.** Toward
   *firing*: an indented import in a CPU-tier test that does run is not credited, so its
   module can be listed although the CPU tier reaches it. Toward *silence*: imports are read
   as text and not resolved as a graph, so a module reached only transitively is invisible.
   A path **absent** from the file is not a claim that the CPU tier covers it, and a path
   **present** is not proof that it does not. `16`'s measured test and lint baselines say
   which of the two mistakes to prefer — running the GPU tier when in doubt is never the wrong one — and the asymmetry
   above is that preference written down. Cite it that way wherever it is cited.

`16` states the same rule as the per-task gate, under its measured test and lint
baselines; the two must be amended together.

**Baseline caveat.** Take each tier's *current* pass/fail state as the baseline before the
first Compass commit, and record it with the tree, container and versions it was measured on
— the GPU baseline above is a standing example of what an unrecorded torch/ROCm version
costs. A pre-existing failure attributed to Compass costs a day; this has already happened
once on this codebase with a lint baseline.

---

## D44. Three separable results, never one number

### The finding this exists to respect

The prior effort reached, on one run: **prefill seconds within 1.0%, decode within 1.1%,
run length within 1.0% — and median TTFT wrong by 90%.**

> *"Aggregate cost accuracy does not bound schedule accuracy when the scheduler has a
> discontinuity in it."*

TTFT there was decided by two comparisons with **1.5 s and 1.2 s of slack**. Prefill-first
means a decode window opens only in the gap between one request's last prefill chunk and
the next request's arrival; the simulated run reached the same point 3.9 s later, both
margins went negative, both windows closed, and ten requests got a first token at 271.6 s
instead of 213.1 and 231.2.

### Decision

**Three results, reported side by side, always:**

| Result | Question | Measured on |
|---|---|---|
| **cost accuracy** | is a step priced correctly? | the **real** run's own step sequence |
| **schedule agreement** | does the engine do the same things? | both step tables |
| **end-to-end** | TTFT / TPOT / throughput | the engine's own readings |

And the standing prohibition:

> **Aggregate latency is the one number this benchmark must never be judged on.** It would
> have called the first cc-traces pilot a success at −5.0% and the second a regression at
> −48.4%, when the second was the more truthful model.

That −5.0% was **+52% TTFT cancelling −30% decode**. Compensating errors inside an
aggregate have happened here at least six times.

---

## D45. Noise-floor first, and a two-sided admissibility test

### Problem

Every schedule metric needs a tolerance, and an arbitrary one is indefensible. Two *real*
runs do not agree exactly either — so the question is not "how close must simulation be"
but "how close is real to itself".

### Decision

**The real-vs-real spread is the tolerance. Measure it; do not assume it.**

```
  1. run the REAL side N>=3 times, SPACED (not back to back)
  2. metric(real_i, real_j) for all pairs        -> the NOISE FLOOR
  3. metric(sim, real_i) for each i              -> the RESULT
  4. GATE: the result lies inside the noise floor
```

Self-calibrating per cell, and it makes an ungradeable cell visible instead of passed or
failed: **if a cell's real-vs-real spread swamps the metric's acceptance target — 10%, or
15% for TTFT (`README.md`) — that cell demonstrates nothing** for that metric and must be
reported as such.

**Repeats must be spaced or interleaved.** Consecutive repeats reported **−2.2% ± 0.1**
where five separated runs of the same command spanned **−6.1 / −14.0 / −14.7 / −14.4 /
−0.5%, mean −9.9 ± 6.4 — a standard deviation larger than the mean**. *"A tight sd from
consecutive repeats is a lower bound and not a measurement."*

### What the noise floor looked like before, and why it may not hold

| | spread |
|---|---|
| 27B, 4 real runs: streak structure | **identical** (36, 6, 42, 7, 15), breaks at the same two places, margins varying by at most **0.1 s** |
| 27B, 4 real runs: TTFT | 27.54 / 27.57 / 27.61 / 27.62 s — **0%** |
| 27B, 4 real runs: latency | 74.50–74.89 s — **1%** |
| 0.6B, 5 real runs: TTFT | 3.55–3.89 s — **9%** |

The conclusion then was *"the real machine is not on the knife edge; only the simulator
is."* **That was 20 requests from one session under declared arrivals.** Under closed-loop
replay at 256 clients, request *k+1* arrives at `finish(k) + delay`, so step-time jitter
feeds back into arrival times and *which* requests coexist depends on arrival ordering
across independent sessions. Mitigating: within-session inter-arrival p50 is **5.61 s** and
`think_time` p50 is **1.17 s**, against ~1% latency jitter. But that is an argument, not a
measurement, and this protocol replaces it with one.

### A metric is admissible only if it is both stable and sensitive

- **Stable** — small real-vs-real spread. Otherwise it cannot discriminate.
- **Sensitive** — it moves under a known perturbation. Otherwise it cannot grade a model.

**The sensitivity control already exists and is measured.** Re-price prefill at **×0.97**
and re-simulate:

| prefill price | TTFT median | prefill streaks | longest |
|---|---|---|---|
| real | 27.63 s | 5 | 42 |
| ×1.00 | 52.40 s | 6 | 63 |
| **×0.97** | **37.31 s** | **8** | **42** |
| ×1.03 | 54.33 s | 6 | 63 |

At ×0.97 the 63-chunk streak splits back into the real run's exact **42 + 7 + 15**,
breaking where the real run breaks (213.7 vs 213.1 s; 230.6 vs 231.2 s). **Any candidate
metric is run against this counterfactual. One that does not respond to a 3% price change
cannot grade a cost model.**

A metric failing either test is reported **inadmissible**, not silently dropped.

---

## D46. Three metric families, ordered by robustness

An extensible registry. Each metric declares a **family**, an **extractor** (step table →
value) and a **comparator** (value, value → distance). Admission is measured by D45's
two-sided test, not asserted. New metrics are expected; the families and the admission
procedure are the stable part.

### Family 1 — counting invariants. Exact match expected; a mismatch is a defect, not noise.

| Metric | Prior observation |
|---|---|
| prefill step count | 106 on both sides |
| decode step count | 4,346 vs 4,333 |
| total prefill tokens, total decode tokens | pure workload arithmetic — should be exact |
| preemption count | 0 in the runs measured |
| prefill streak count | identical across four real runs |
| requests admitted / completed | — |

### Family 2 — distributional. The family to weight most.

Compare **histograms**, never step-*k*-to-step-*k*: batch-width distribution,
scheduled-token distribution, per-step context-length distribution, cached-fraction
distribution. Graded by a distance (KS or Wasserstein) against the real-vs-real distance
for the same quantity.

Two reasons this family carries the most weight:

1. **It sidesteps alignment drift entirely** — no step has to correspond to a step, which
   is exactly the failure mode a per-step alignment rate suffers from. A per-step rate is
   dominated by long uniform decode runs, so it reads ~99% while the handful of steps that
   decide TTFT all disagree.
2. **The batch-width histogram is literally what determines throughput**, so it is
   meaningful rather than merely computable.

### Family 3 — event timing. Most diagnostic, least robust.

Structural events — prefill streak boundaries, admission points, preemptions — matched in
order, reporting the offset distribution of matched boundaries. Graded against the
real-vs-real offsets, which on the 27B were **0.1 s**.

### What does not need measuring

Scheduler **policy** agreement. **0 of 4,378 real decode steps and 0 of 4,621 simulated
ones** took a decode step while an arrived request still owed prefill. *"There is no
scheduling disagreement to find."* Keep the decision-record diff as a **regression guard**,
not a discriminator.

### Open issues

- The distance function for family 2 is unchosen (KS vs Wasserstein vs something
  workload-aware).
- Family 3's "structural event" needs a definition that survives past prefill-first
  scheduling; streaks may be an artefact of that policy.

---

## D47. Cost accuracy, on a fixed real step sequence

### Method

Apply the cost model to the **real** run's own steps. A within-run comparison that
separates cost error from schedule error by construction. *"Use the real sequence to assess
cost predictions first; use controlled durations to assess the scheduler separately."*

### Reporting rule: never the total alone

The tool reports **three numbers side by side** — on totals, holding out the largest single
contributor, and the median per step:

```
prefill  106 steps   measured 235.56 s   priced 228.20 s
  on totals                 -3.13 %
  holding out step 0        -0.27 %      <- the unmodelled cold start
  median per step           -0.09 %

decode  4346 steps   measured  73.17 s   priced  73.10 s
  on totals                 -0.09 %
  holding out step 148      -0.07 %
  median per step           +0.09 %
```

One step moved the prefill total by **3.0 points**. *The total was describing that step,
not the model.*

### The sign trap, and why the running sum is what matters

Prefill priced against a real run's own steps came to **+0.10% overall** but **+3.06%
excluding step 0** — a single cold-start error of **−6.75 s** almost exactly offsetting
**+6.99 s** spread over the other 105 chunks.

> *"The scheduler never sees the total. It sees the running sum, which is 3% high from the
> second step onward and never gets the 6.75 s back."*

So cost accuracy must be reported **with outliers held out**, and the cold start priced or
excluded deliberately rather than left to cancel a real bias.

---

## D48. Memory: per term, never as a sum

### The rule

A summed check reported **+13.8%**. It was three errors, two of which cancelled: weights
over-counted by +0.280 GB (a tied head never resident), activations compared at the wrong
shape (−0.015 GB), and −0.084 GB of *"a resident term nobody had noticed was there at
all"*. **The largest single error was 25% of a term and the sum said 13.8%.**

`peak_torch` was consequently split at source into three non-subtractive readings —
`parameter_bytes`, `weights_torch`, `current_torch` — so a sum can never hide a term again.

Validation costs **no GPU time**: every hardware run already prints the real breakdown.

### The decision the byte error exists to produce

| | modelled | measured | error |
|---|---|---|---|
| 0.6B TP=1 | 96,033 | 96,051 | **−0.02%** |
| 0.6B TP=2 | 183,709 | 183,745 | **−0.02%** |
| 0.6B TP=4 | 367,614 | 367,364 | **+0.07%** |
| 27B TP=2 | 37,825 | 37,858 | **−0.09%** |

TP=1 and TP=2 land **under**, which is the safe direction. TP=4's +250 blocks is
contamination — the calibration takes the minimum `non_torch` across ranks and the compared
run had 144 MiB more on rank 0. **+0.07% is inside the contamination band and its sign
carries no information.**

### The gate that actually matters

> **The gate is not the byte error, it is whether the top-1 configuration choice survives.
> A byte error of a few percent that never changes which configuration wins is a better
> outcome than a tighter one that does.**

---

## D49. Measurement hygiene, as refusals

Each of these voided real percentages before.

1. **Never mix profiled and unprofiled numbers.** Being profiled costs ~**0.7–1.05 µs per
   kernel** — **8.1% of a 27B decode step** (951 kernels, +1.002 ms) and 8.5% on the 0.6B;
   a profiled run is ~16% slower end to end (TPOT 3.619 vs 3.115 ms). *The cost model is
   validated against the measure table (CUDA events, no profiler). A profile is for
   attribution within a step and never for an absolute.*
2. **Never mix machines.** The same unprofiled 27B TP=4 decode step is **9.774 ms on one
   box and 12.365 ms on another — 26%**, larger than the difference between TP=4 and TP=8
   on one box.
3. **Read the spread before the mean**, and report an interval or report nothing. A
   difference smaller than ~5% needs repeats before it means anything.
4. **Any residual below about 2% is quoting the instrument.** Pricing repeatability across
   two back-to-back runs of one graph: summed **0.96%**, median per signature **1.18%**,
   **p90 32%**.
5. **Check the machine before *and* after, not only before.** Three of the last five prior
   pilot attempts were lost or degraded by other tenants — two refused to start with a
   negative KV budget at 141 GB of neighbour, one ran **64% slow**. A check that only runs
   first cannot see a tenant that arrived mid-run.
6. **Coverage is a hull, not a bounding box**, and **leave-one-out does not validate a
   family**. Both are doc `09` D58/D59 — the geometry of coverage and the choice of a
   functional form are fitting questions and are argued there, with the measurements.
   Named here because both are *refusals a validation run applies*: a step outside the
   hull is reported as refused rather than predicted, and a law whose family was chosen
   by LOO alone is not admissible evidence.
7. **A speculative acceptance cell run at declared acceptance is not acceptance
   evidence.** It is exploration. Acceptance cells use the *measured* per-position
   distribution from the paired real run (`14` D84 tier 1); the artifact names the tier.

---

## D50.1. What a refusal does to a run

### Problem

Seven documents emit refusals — an unpriced leaf, a step outside the hull, a missing
runtime constant, an unmatched tokenizer, a guard domain no binding satisfies — and until
now none of them said what happens next. Abort the run? Continue and mark the step? The
answer changes what a result *is*, so it belongs here rather than in any of the seven.

### Options

| | abort on first refusal | mark and continue | continue, exclude from grading |
|---|---|---|---|
| Bring-up usability | **poor** — one unforeseen shape kills a 300 s run | good | good |
| Diagnostic value | names one refusal | **names all of them, with counts** | names all of them |
| Risk of a partial result being read as complete | none | **real** | real |
| Schedule fidelity | n/a | a refused step still has to consume *some* time, or the schedule diverges from that point on | same |

### Decision

**Mark and continue, with the refused fraction as a reported result and a declared
admissibility threshold.** Concretely:

1. **A refused step is priced by the next rung of the resolver ladder that can answer**,
   down to tier 0, and tagged `provenance=refused(<reason>)` naming the rung that actually
   answered. It is never priced at zero and never skipped — a skipped step changes the
   schedule from that point forward, which would corrupt every number after it, not just
   the refused one.
2. **The run artifact carries the refused count, the refused fraction of steps, the
   refused fraction of predicted *seconds*, and the distinct reasons with counts.** The
   seconds fraction matters more than the step fraction: 2% of steps can be 40% of the
   time.
3. **Admissibility gate: a run with more than 5% of predicted seconds refused is not
   acceptance evidence.** It is a coverage report, and it names exactly what to measure
   next. The number is declared, not derived, and should be revisited after one campaign.
4. **`--compass-on-refusal abort` exists** (`13` D80) for the case where a refusal is a
   bug being chased. It is not the default and is not valid in an acceptance run.

**Why not abort by default.** Refusals are the design's main *diagnostic* output — the
whole point of `compass plan` is that a refusal tells you what to measure. Aborting makes
that one refusal per run instead of a complete list, which turns a single coverage gap
into as many iterations as there are gaps.

**Why the fraction is a gate rather than a note.** Without one, "mark and continue"
degrades into a run that is 30% tier-0 fallback and reports a tidy latency number. The
gate is what keeps the refusal from becoming a silent fallback — which is the exact
failure the *refuse rather than fall back* principle exists to prevent.

---

## D50. Registration and the cell matrix

### Registration

Everything is frozen **before** evaluation and hashed into the artifact: dataset version,
scope, request identities and order, token lengths, arrival pacing, the session split, the
machine spec, every artifact fingerprint, and the gate states of D43.

**The case set may never shrink after errors are seen.** And a development workload may
never be called held-out: the prior pilot corpus was iterated against during development,
so it could not be — *"the held-out axis the user actually wants is configuration."*

### The cell matrix

**Revised for `16` D95.** The cells are the configurations ATOM's nightly runs define, not
widths chosen to span what calibration did not see: M2 is graded at the widths those
configurations run, so width transfer (T21) is not on the acceptance path. The nightly runs
define the configurations, never the real side, which this project runs itself (`16`
P0b.3). Two sets, each driven by cc-traces:

| Set | Model and hardware | Configurations | Client counts |
|---|---|---|---|
| **Stage 2**, M2's first gate | Qwen3.8-27B on MI308X | TP1, aggregated | 1, 4, 16, 64, 256 |
| **M2** | DeepSeek-V4-Pro, 1P+1D on the MI355X pair | the four nightly configurations of [`recipes/mesh/DeepSeek-V4.md`](../../../recipes/mesh/DeepSeek-V4.md) as written, expert parallelism off: TP, DPA, TP MTP3, DPA MTP1 | each configuration's benchmark concurrencies in that recipe |

Flags and concurrencies are the recipe's; this document does not copy them. **M2's EP8
step** adds the two DPA configurations re-run with `--enable-expert-parallel` (`16` W4.11),
graded the same way, and joins them to the ranking gate below. The aggregated DPA+TBO
variant in `.github/benchmark/models.json` is not a cell: D95 grades M2 at its P/D
configurations only.

Client count means **agent session trees**, not requests; in-flight exceeds it during
fan-out. Each cell is one `aiperf profile` invocation — the scenario rejects
comma-separated sweeps.

**One corpus limit to design around:** only **175 of 393** sessions contain any subagent
and only **144** offer a multi-request episode containing a descendant. A 256-client cell
requiring genuine fan-out in every root is not constructible without reusing sessions, and
the recipe's largest DPA concurrency exceeds the corpus's sessions, so the cells at that
concurrency reuse sessions whatever they require. Whatever is done there must be declared,
not discovered (T27).

### The ranking gate

T28 asked whether ranking becomes an explicit gate; for M2 it does. The prior effort
called it the gate that matters, and D48 already makes the top-1 choice the memory gate.
Over M2's cells, for each end-to-end metric of D44 separately — throughput, TPOT, TTFT,
never a combination of them:

1. **The same best configuration as hardware at each concurrency.** Among the
   configurations run at one client count, the simulated best is the real best. Where the
   real best and runner-up differ by less than their real-vs-real spread (D45), that client
   count has no determined best and is reported ungradeable, not passed.
2. **Rank correlation across all cells.** Spearman's ρ between simulated and real values
   over every cell must pass two tests together: ρ ≥ 0.9, and, graded as D45 grades any
   metric, ρ inside the spread of the same ρ taken real-vs-real. A wide real-vs-real spread
   never admits a ρ under 0.9. Client count alone orders much of a matrix spanning one to
   hundreds of clients, so ρ is the weaker check of the two.

Stage 2 runs one configuration, so it has no ranking to grade; it is graded by D44's
results alone.

### The DP cells

The owner's DP ruling of 2026-10-01
([#470](https://github.com/jgong5/ATOM/issues/470#issuecomment-5933154215)) puts DP
acceptance on DeepSeek-V4-Pro's DP-attention configurations on MI355X and no other model;
D95 narrows those to the recipe's DPA and DPA MTP1 configurations above. Each is judged by
cc-traces e2e and by the per-step error spread (D47), never by a total. TBO under DP (#531)
and tier b pricing for V4-Pro are prerequisites, so neither cell is acceptance evidence
before both exist.

The mechanism tests run device-free on a CPU fake-model DP2: both ranks report the same
step seconds, equal to the larger of the two; an idle rank's dummy batch is priced and
enters the `max`; the batch view refuses a DP run that states no `num_tokens_across_dp`
(#470); and the CA refuses members whose call counts differ (#528).

### Repeats

The simulator is **deterministic by construction** under doc 01 D3 (Clock Authority, ties
broken by LP id) — so: **one simulated run plus an asserted reproducibility test**, and
N≥3 **spaced** real runs per cell to establish the noise floor.

---

## D50.2. A projection is sound, not graded

### Problem

M2's MI455 roofline projection (`16` D95) has no real side. D45's tolerance has no
real-vs-real spread to measure and D44's results have nothing to compare against, so any
error bar quoted on the projection would be invented.

### Decision

**The projection is held to three checks instead of an error bar.** A projection that
fails the first two, or ships without the third's measurement, is not M2's result.

1. **The roofline is a lower bound on every measured MI355X step.** The bound prices each
   leaf by `10` D64 at the spec's peaks, every derate at 1, with no host floor, and
   composes the leaves the way the step composes them: each `Par` through its own join
   (`04`, `Par`: concurrency) and TBO's two micro-batches through `16` W4.6's overlap,
   never summed across concurrent work. D64's flat `Σ t_leaf` is not a lower bound on a
   step that overlaps work, such as V4's compressor and indexer on side streams or the DPA
   prefill side's TBO. The `max` join, the `resource_bound` join at derate 1 and the
   `exclusive` sum each still are. Tier 0's own price is not the bound: it is derated and
   floored, a central estimate (`10` D67.1). Applied as D47 applies cost accuracy, on the
   real runs' own unprofiled steps (D49): no step's bound may exceed its measured time. One
   that does is a defect in a law or in the spec, reported with its operator family; it is
   a bound, not a statistic, so none is tolerated. The measured-to-roofline efficiency is
   reported per operator family as a distribution, never as a mean alone: a step's measured
   time has no per-family split, so a family's efficiency is its leaves' roofline time over
   their measured time in `16` W4.4's MI355X calibration tables. `16` W5.3 runs this check.
2. **Every MI455 input is sourced or marked assumed.** A machine-spec field's source is the
   provenance of the fragment that supplied it, as `05` D26's `merge` records; which fields
   are assumed is rule 5 of `05` D25's schema rules. The projection artifact lists the assumed
   fields a tier-0 law reads. A field no fragment supplies is refused (principle 6). `16`
   W5.1 and W5.4 carry it.
3. **The method's error is measured where both sides exist.** The same method —
   measured-to-roofline efficiencies carried from a measured device onto a target spec —
   projects Qwen3.8-27B from MI308X to MI355X and is scored against real MI355X runs by
   D44's three results (`16` W5.5). That score is the error estimate quoted with the MI455
   projection. It estimates the method, not MI455: it is taken on another model and another
   device pair, because DeepSeek-V4-Pro's FP4 experts do not run on gfx942.

Where a claim about MI455 can be a ranking rather than an absolute, prefer the ranking
(`10` D69).

---

## D51. Simulation speed

### What was measured before

| workload | real | simulated | ratio |
|---|---|---|---|
| 64 requests, Poisson 8/s, 0.6B TP1 | 9,426 ms | 639 ms | **14.8×** |
| cc-traces, 20 requests, 27B TP4 | 309 s | 3 s | **~103×** |
| **saturated, 300 requests, 0.6B** | **36 s** | **122 s** | **0.30× — slower** |

The discrete-event jump only pays when there is idle to skip. Partly explained by the
arrival barrier spinning **89,336 scheduling ticks** before the first step — machinery doc
01 D8 removes.

### Rules

- **Measure a saturated cell early**, before the architecture is load-bearing. The
  ≥5× target is negotiable; *faster than real* is not.
- **Account cold costs once.** Factory build is 9.5–11 s; first `graph_for` on a new shape
  is 0.27 s. Journal them rather than folding them into a per-step average.
- **Beware the cold-loop artefact** — an average over a loop whose first iteration is a
  cache miss reads **3–7× high**. Measure a second, warm loop.
- Report the simulator's own CPU cost per step alongside the ratio. Previously **4.3 ms**
  per step with the allocation carried in the cache key, against a 32.7 ms modelled step.

---

## D52. What invalidates a run

Fail closed. Each of these has a prior incident behind it.

| Condition | Why |
|---|---|
| a **coverage hull violation** on any acceptance step | the step was priced outside what was measured |
| **machine contamination** before or after | a tenant compute-bound in 0.7 GB was caught by a health check and would have passed any free-memory test |
| a **`--measure` flag** on an acceptance run | the instrument changes what it measures — ~11 ms of TTFT on the 27B |
| **analytic fraction** above the declared threshold | doc 07 D36's resolver ladder rung 4 is a permission, not a fallback |
| an **artifact fingerprint mismatch** | doc 07 D43 |
| a **gate state** in the artifact disagreeing with the declared configuration | *"a dead gate is worse than no gate"* — a `PRICE_KERNELS` gate read `WORLD_SIZE`, which the engine never sets, so it was false in every worker and the proof was in the artifact: breakdowns present for 164 of 237 entries |
| **saturation** where scheduling fidelity is being claimed | see below |
| any **refusal** raised by the cost model | it refused for a reason; name it |

### Saturation deserves its own note

**Scheduling fidelity is unobservable on a saturated workload.** Every loaded prior
experiment ran at essentially 100% utilisation, *"where the queue term swamps everything
and no scheduling property is observable."* Concretely: both runs executed the same **1,028
steps**, the simulated ones cost **3.8% more**, the engine was **99.6% utilised real and
100.0% simulated** — and *"adding 3.8% to every step is enough to take 99.6% to 100%, and
the wait follows."*

A previously-reported *"constant quarter-second admission delay"* was **withdrawn** on
exactly this basis: per quartile the simulator admits the **first** requests *faster*
(0.000 s vs 0.046 s) and then falls progressively behind. The question as posed —
*"find why the simulator admits requests late"* — had no answer, because it does not.

So a scheduling claim needs a workload **with slack**, or controlled step durations that
remove cost error by construction. Throughput claims at saturation are fine; scheduling
claims there are not.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D44 | Three separable results — cost accuracy, schedule agreement, end-to-end — reported side by side. Aggregate latency is never the judgement. | 2026-09-18 |
| D45 | Noise-floor first: the real-vs-real spread is the tolerance. A metric is admissible only if stable **and** sensitive, the latter tested against the ×0.97 counterfactual. | 2026-09-18 |
| D46 | Three metric families ordered by robustness, as an extensible registry: counting invariants (exact), distributional (weighted most), event timing (most diagnostic). Policy agreement is a regression guard, not a discriminator. | 2026-09-18 |
| D47 | Cost accuracy on the real run's own step sequence; report totals, held-out-worst, and median per step together. | 2026-09-18 |
| D48 | Memory per term, never as a sum. The gate is whether the top-1 configuration choice survives, not the byte error. | 2026-09-18 |
| D49 | Seven hygiene refusals, each with a prior incident behind it. | 2026-09-18 |
| D50.1 | A refusal marks and continues: priced by the next answerable rung, tagged `refused(reason)`, never zero and never skipped. Refused fraction of **seconds** is a reported result; **>5% refused seconds is not acceptance evidence**. | 2026-09-19 |
| D50 | Everything registered and hashed before evaluation; the case set never shrinks. One simulated run plus a reproducibility assertion; N≥3 spaced real runs. The cells are stage 2's Qwen3.8-27B TP1 on MI308X and M2's four nightly P/D configurations of DeepSeek-V4-Pro on MI355X at their concurrencies, then the DPA two at EP8. M2's ranking gate: the same best configuration as hardware at each concurrency, and rank correlation across all cells, per end-to-end metric, each inside its real-vs-real spread, with Spearman's ρ also at least 0.9. DP mechanism tests run on a CPU fake-model DP2. | 2026-09-18; revised 2026-10-01, 2026-10-09, 2026-10-10 |
| D50.2 | The MI455 projection is sound, not graded: the roofline at the spec's peaks, derates at 1, no host floor, composed by the step's own joins and never summed across concurrent work, is a lower bound on every measured MI355X step; every MI455 input is sourced or marked assumed; the method's error is measured by projecting Qwen3.8-27B from MI308X to MI355X. | 2026-10-09 |
| D51 | Measure a saturated cell early. Account cold costs once; report the simulator's own per-step CPU cost alongside the ratio. | 2026-09-18 |
| D52 | Eight fail-closed invalidation conditions. Scheduling claims require slack; throughput claims at saturation are fine. | 2026-09-18 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T23 | Choose the family-2 distance function | needs one cell's real-vs-real data to compare candidates |
| T24 | Define "structural event" for family 3 beyond prefill streaks | streaks may be an artefact of prefill-first scheduling |
| T25 | Measure the real-vs-real noise floor under **closed-loop** replay at high client count | the only prior data is 20 requests, one session, declared arrivals |
| T26 | Assert simulator bit-reproducibility as a test | depends on the Clock Authority existing |
| T27 | Decide the construction of the cells the corpus cannot fill without reusing sessions: 256 clients with fan-out in every root, and the recipe's largest DPA concurrency | doc 06 T14, surfaces again here |
| ~~T28~~ | ~~Establish whether ranking/regret becomes an explicit acceptance gate~~ — **done**: M2's ranking gate, D50 | — |
