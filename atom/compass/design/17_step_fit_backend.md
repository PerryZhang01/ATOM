# ATOM Compass — Design Topic 17: Step Fit Backend (Tier a)

**Status:** reviewed and approved, 2026-10-11. Drafted by an AI assistant from the design
PR #626 proposed and the owner's rulings on
[#696](https://github.com/jgong5/ATOM/issues/696) (2026-10-09 and 2026-10-10), and
reviewed by jgong5 on PR #626. No code has been written against it yet; implementation
follows the execution plan in `16`.

**Depends on:** `02` (the backend interface, the forward's interval, the provenance
species), `07` (the tiers, the bench, artifact identity and invalidation), `08` (stage 2's
protocol and the refusal rule this topic departs from), `09` (the hull and law
validation), `13` (the flag surface), `14` (the cost form), `15` (DP and PP).

**Scope.** One tier-a cost backend, `StepFitBackend` (config id `step_fit`): a sparse grid
of measured whole-forward times, looked up for each batch, plus a residual fitted on
replayed batches that carries what the grid's coordinates cannot see. It is a candidate
backend for [#627](https://github.com/jgong5/ATOM/issues/627)'s stage 2 and gets one
non-graded evaluation on DeepSeek-V4-Pro. It does not cover tier b; the tier-a law of `09`
D54, which stage 2 prices with ([#738](https://github.com/jgong5/ATOM/issues/738)) and
whose DP-attention form is [#529](https://github.com/jgong5/ATOM/issues/529)'s in stage 3;
or KV transfer (`01` D6).

**Deliberate departures**, by owner ruling on #696 unless the row says otherwise. Every
other conflict between #626 and the decided design is resolved in the decided design's
favour, and the decision that adopts it says which.

| Departs from | How | Decision |
|---|---|---|
| `07` D36: tier b is what acceptance cells use | a tier-a backend is graded as a stage-2 candidate | D96 |
| `07` D39: step measurements come from replayed batches | the baseline grid is synthetic and homogeneous; the residual is fitted on replayed batches | D98 |
| `08` D50.1 items 1, 3 and 4, and `13` D80's default that follows them: a refusal marks and continues | a refusal aborts the run, and a graded run is refusal-free by coverage | D102 |
| `13` D79: no Compass config file | step fit is selected by a run-file key, as the code selects every backend; not ruled on #696, and open until the owner settles the run file (T96) | D103 |

---

## D96. Role: a tier-a candidate for stage 2

### What it is

An answer is the grid's value at the batch's coordinates (D99), corrected by a residual
fitted over the batch's own features (D100). That is tier a by `07` D36's definition —
fitted over features `ScheduledBatch` already carries, needing measured steps — and it is
a plain `CostBackend` (`02` D11): `estimate(batch_view)` returns a `StepCost`, `describe()`
one line. No Cost IR and no op-level model.

The name describes the method. A bundle is identified by its key (D101), never by a name
or an experiment round. Each answer's provenance uses `02` D11's species (D99).

### The stage-2 candidacy

Step fit is graded by stage 2's protocol: Qwen3.8-27B TP1 on MI308X, aggregated, at the
client counts of `08` D50's stage-2 row, by `08` D44's three results alone. Stage 2 runs
one configuration, so there is no ranking gate.

This does not make step fit the backend every stage-2 run, or any other work, must use.
Which backend stage 2 is passed with is #627's ruling, and on 2026-10-10
[it ruled](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096190407) that
stage 2 prices with `09` D54's tier-a law and that step fit stays off the stage-2 path for
now. Step fit's stage-2 evaluation (`16` S2.2 to S2.6) therefore runs off stage 2's
critical path: no stage-2 task waits on it, and it takes MI308X time only when the stage-2
node is idle from that path.

**The real runs.** Grading reads the registered real runs of stage 2's exit
([#743](https://github.com/jgong5/ATOM/issues/743)), their per-request and per-step
records, read only. If they do not exist when step fit is ready, step fit runs its own real
side under the same rules (`08` D45, D50), named apart from stage 2's so the two are never
confused.

**The trace split**, #627's
[ruling 1](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740): the
residual, `09` D59's validation and every comparison of grid definitions use only
cc-traces traces 256 to 392; grading uses only traces 0 to 255. Grid planning may read the
shapes of any trace, never its timings (D98). This split is the development and held-out
split of `08` D50 for step fit.

**A use beyond `07` D36.** D36's "used for" column gives tier a structure discovery,
sweeps and plumbing, and gives tier b the acceptance cells. Grading a tier-a backend on
stage 2 goes beyond that column, by the owner's ruling on #696; D36 carries a note.

Before grading, `09` D59's validation is a planned step (`16` S2.5): the residual is
validated on data held out from its fit, at least three candidate families are ranked and
all reported, and the chosen form is tested on a second axis. That data comes from traces
256 to 392, never from the traces grading holds out.

Stage 2 has no MTP. MTP is M2's, on DeepSeek-V4-Pro only (`14`, *Milestone placement*);
step fit refuses a speculative step (D97).

### What it is not

- **Not a stage-3 backend.** Its one DeepSeek-V4-Pro evaluation is not graded and is
  outside M2's ranking gate (D105).
- **Not the source of the per-family MI308X tables** of `08` D50.2 item 3 and `16` W5.5.
  Those are operator-family tables for the projection method; step fit's grid holds whole
  forwards. Both take MI308X time (`16`, *The GPU booking queue*).
- **Not for Kimi-K3 or DCP in v1.** Both come after v1 (`16`, *After v1*). DCP's
  correction is described as an optional recipe, generically (D100).

---

## D97. The pricing boundary: one whole forward, one term

### One term per step

The whole batch is one `StepCost` with a single term covering the forward's interval
(D98): compute, logits, sampling, the in-forward TP collectives and their overlap. The term
is not multiplied by TP, and no collective terms are added beside it, because the
measurement already contains them. The lookup's value, the residual and each member's
output are diagnostics in the provenance detail, never terms, so no term is negative. The
term is in seconds; a bundle's unit is converted once, at that boundary.

### The runner prices, the engine charges

- **One charge per logical forward.** The predicted forward's runner prices its own
  rank's batch with the installed backend, and the cost rides on the step's output to the
  engine, whose clock owner charges it (`01` D4, K1). A real forward carries no cost and
  charges nothing. The backend never touches the clock and never sleeps.
- **The label is the group's step.** A step measurement of a TP group is the group's
  step, which `01` charges to its slowest rank (`15` D90), so a step is priced once, never
  per rank.
- **DP and PP.** Under DP the `15` D90 `max` exchange is the rule, but every bundle this
  topic plans is measured at DP 1, so a DP run mismatches the key and refuses (D101);
  DP-attention's tier a stays #529's. A PP stage refuses to price: `15` D91 prices each
  stage, and a whole-step label cannot price a stage.
- **Only executed batches are priced**, intermediate prefill chunks included: a chunk that
  samples nothing still costs forward time (`02` D10). A forward is atomic: no op-level,
  pipeline or cross-step overlap.
- **The step's output keeps `02` D10's three semantics.** The cost rides on the output,
  not beside it, so the deferred-token protocol is unchanged and nothing visible is
  published before the grant.

### Inputs

- **Rows in the batch's own order**, never sorted or merged; position features read order
  (D100).
- **Per request:** `N_Q`, the query tokens this forward computes, and the cached context,
  the tokens whose KV existed before it. A `BatchView` row carries the visible context,
  cached plus query, so cached context is `context_tokens - query_tokens`.
- **Phases.** Every row prefill: the prefill grid, single-token prefill included. Every
  row decode with `N_Q = 1`: the decode grid. A mix refuses. A decode row with `N_Q > 1`
  (speculative decoding) refuses: the decode grid has no `N_Q` axis.
- **Validated at the boundary, never coerced.** An empty batch, a non-integer or negative
  count, or an unknown phase refuses.
- **The backend imports nothing from ATOM** (`02` D11). What reads ATOM's configuration
  (D101) lives outside the backend package.

### Outside the backend

Queueing emerges from events. KV transfer has its own model (`01` D6). The estimator's
own CPU time is simulator overhead, which counts against the speed target (`08` D51) and
never against the forward. Worker dispatch and IPC are not forward time.

---

## D98. The grid: synthetic, homogeneous, planned from shapes

### The departure from `07` D39

D39 forbids synthesised batches for step measurements: row order moved decode attention
**1.77x** at one context multiset, and a fit is blind to a dimension it never varied. Step
fit departs from that **for the baseline only**. The grid is synthetic and homogeneous
because a grid gives regular, controllable coverage: the plan chooses every corner, so
whether a cell is complete (D99) is something the plan can guarantee, which replayed
batches cannot. D39's two hazards are carried by the residual, fitted on replayed batches
in the first version (D100), so row order and raggedness are covered before grading.

### The grid definition is a calibration input

- **An input to the compass toolchain, never code and never a silent default** (`13` D79
  rule 1): one list of node coordinates per axis, and the planning margin. A grid run
  without either refuses, naming it.
- **Recorded in the bundle and part of its identity** (`07` D41, and D101's key), so
  another grid is another bundle, never a code change.
- **A cell is the box between adjacent nodes on every axis.** That rule stays in code,
  with the coordinate mapping, the complete-cell rule and the interpolation inside a cell
  (D99); the backend reads the cells, and which of them are complete, from the bundle.
- **What a point's measurement means is code, not input.** Warm-up, repeat count and the
  statistic taken over the repeats are fixed in the toolchain and recorded with each
  point; none is configurable.
- **Measurements are stored per point**, keyed by the point's shape and the `07` D43
  fingerprint, as `07` D41 keys every entry. A grid definition selects points, so a new
  definition measures only the points not yet measured.

### Planning: nodes from shapes, cells from the planned population

- **Sparse, and placed from shapes.** A definition's nodes come first from `07` Phase 0's
  population on the registered case set, then from the real runs' step tables. Never from
  timings: a node placed where an error looked large is a node chosen by the answer. The
  shapes may come from any trace, graded ones included; their timings never do. Choosing
  among definitions already written is D100's, on traces 256 to 392 only.
- **Only the cells the planned shapes fall in are measured, plus the declared margin**
  around them, `07` D38's "over-cover with a declared margin". A cell outside both is not
  measured, stays incomplete, and refuses the step that needs it (D99).
- **The feasible region's boundary is in the grid.** Its boundary nodes, the KV capacity
  and the token budget, are grid nodes. The bundle records the configuration bounds its
  grid was planned for (D101 checks them).
- **Order.** The real runs first (D96); then top-up measurement of the cells real
  batches fall in; then grading. Planning waits for the case set's registration (`08`
  D50), which waits for T27.

### Decode: replay on the target's own ladder

Decode is measured in replay mode on the target's own capture ladder, read from ATOM's
config (`05` D24, `13` D78). For each rung `R`, at `B = R` and at `B = previous rung + 1`,
over the context nodes. `B` is interpolated only inside one rung, never across: a batch
replays its rung's graph and pays its rung's time, as `09` D54's per-rung decode fit also
holds. A decode step that replays no graph, being wider than the widest rung, is not on
this grid and refuses (D99).

**The lookup axis follows the graph mode at measurement.** The measurement this rests on
(#626 at `1139bbca`, recomputed from the reference bundle of serving_simulator `2ce181d`):
every decode row was captured under graph replay on a host whose ladder was nearly dense,
so the rung equalled the batch size almost everywhere and a table read as `cost(B)` was
`cost(rung)`. On a sparse production ladder a batch replays a wider rung than its size, so
reading by `B`, or interpolating across rungs, underestimates in one direction, more the
sparser the ladder. The axis is a property of the measurement, so the bundle declares its
graph mode and a mismatched target refuses (D101); it is never inferred at run time.

### Prefill: eager on `(B, N_Q, cached context)`

Measured eager, distinguishing whether a chunk produces output. ATOM skips logits and
sampling for a pure middle chunk (`02` D10), so a chunk that produces no output is never
priced from one that does.

### Scope and timing

The interval is `02` D10's: the whole `ModelRunner.forward()`, sampling included. It is
timed per `07` D42, with CUDA events drained by `query()` and no per-forward synchronize,
under the target workload's sampling settings. A bundle declares its scope, and a narrower
one is refused at load.

The measurement this rests on (#626 at `1139bbca`): the reference bundle measured
`run_model` plus logits under a GPU synchronize, which leaves `ModelRunner.postprocess` —
sampling, the TP broadcast, the device-to-host copy, run serially on the same stream —
covered by no term.

Measurements are unprofiled (`08` D49 item 1) and taken on the device the real side runs
on (`08` D49 item 2). MoE time reflects the routing the measurement saw; uniform routing is
`09` D61's declared treatment.

---

## D99. Lookup: complete cells, or refuse

### Coordinates

A batch maps to grid coordinates by a deterministic function of its ordered rows: an
equivalent homogeneous shape that keeps the batch's token count and its attention work,
`Σ (N_Q·c + N_Q(N_Q+1)/2)` over requests with `c` the cached context, in exact arithmetic
so a lookup reproduces bit for bit. Prefill reads `(B, N_Q, c)` of that shape and whether
the batch produces output; decode reads `(rung, B, c)`. What the equivalent shape loses,
the raggedness and the order of the rows, is the residual's job (D100).

### Decision

- **An answer is an exact hit, or an interpolation inside a cell, only when every corner
  of the enclosing cell was measured.** Otherwise the step refuses, naming the missing
  corners. A cell is the box between adjacent nodes of the bundle's grid definition on
  every axis (D98).
- **Nothing else answers.** No extrapolation, clamp, missing-point recovery, completion
  from another axis, calibrated model or constant fallback (`09` D58; principle 6).
- **A hole does not fail loading.** The bundle records which cells are complete; a hole
  refuses the step that needs it.
- **A pre-run coverage check** covers the grid cells and the residual's hull (D100) for
  every real step and every Phase 0 step of the registered case set. A graded run starts
  only after it passes (D102).
- **Lookups are deterministic**: one batch and one bundle give one bit-identical answer
  (`08` D50's reproducibility assertion, `01` D3.4).

### Species, per `02` D11

A step's term carries the residual's species, `fitted`. The detail records the lookup's
own: `measured` where a homogeneous batch hits a node exactly, `interpolated` otherwise. An
exact hit on a heterogeneous batch's equivalent shape is `interpolated`, not `measured` or
`analytical`: the batch itself was never timed, and nothing was computed without measuring.
Nothing is `extrapolated`, because nothing answers outside a complete cell.

### How this sits with `09` D60

D60 found interpolation between step measurements unsafe (prefill: median 12.2%, worst
32.1%, against 2.1% for a measured ladder) and makes laws piecewise over dispatch bands read
from a leaf's recorded kernel identity. A whole-step measurement records no kernel
identity today, as for `09` D54's tier-a law, so step fit cannot yet see a band inside a
cell. That is why a lookup is never an answer alone: the residual, fitted on replayed
batches that fall inside the cells, carries what interpolation misses, and `09` D59's
validation and the held-out grading measure whether it does. T94 closes the gap: each grid
point records the kernels it ran, and a cell whose corners ran different kernels crosses a
dispatch band and is refused or split.

---

## D100. The residual and its recipes

### Fitted on replayed batches

- **In the first version.** Fitted through the compass toolchain (`07` D37) on replayed
  batches (`07` D39) run on the bench, as relative error over the lookup (`09` D53), and
  reported as `09` D57 requires of every fit.
- **Development and held-out by the trace split** (D96, `08` D50). The residual is fitted
  only on replayed batches of traces 256 to 392, and grading uses only traces 0 to 255.
  Both are registered before evaluation, with the bundle digest (`07` D41).
- **Grid definitions are compared on traces 256 to 392 only**, and every definition tried
  is recorded with its errors, so the one a bundle carries was not chosen on the traces
  grading holds out (`08` D50).
- **Features in `14` D85's form**: tokens, `Σ N_Q²` and `Σ N_Q·N_KV` over requests, the
  aggregates `ScheduledBatch` carries; one form for prefill and decode, a decode step being
  its `N_Q = 1` case. Plus the row order and raggedness that `07` D39 and `09` D61 name as
  treatments. Model, dtype and width select and check the bundle (D101); they never enter
  a feature. The lookup stays split by graph mode (D98): eager and replay are two
  measurements, not two forms.
- **Hull refusal.** The residual answers only inside its fitted hull (`09` D58, `08` D49
  item 6); outside, the step refuses.

### Families and members

- **Candidates per `09` D59**: at least three families ranked on data held out from the
  fit, all reported, the winner tested on a second axis (D96).
- **Members are data.** The bundle lists members, their weights and their output
  transforms. No plugins and no expression language. A member, with its dependencies,
  loads only if the bundle selects it. Nothing trains at run time.
- **Validity, checked explicitly.** Each member's output and the combined answer must be
  finite and positive. A failure refuses the step and names the member: no renormalising
  over the remaining members, no drop to the lookup alone. A load error fails startup,
  naming the member and the artifact.
- **Caching.** Any result cache is keyed by the ordered batch, never a
  permutation-invariant hash, and a hit returns the same provenance.

### Recipes: functions fixed, values injected

A recipe is a set of feature functions and the constants frozen with them (normalisers and
units). It is fixed code: no branch on a model name or a width.

- **Identity.** A recipe is named, versioned and digested over its implementation. A bundle
  fitted under another digest refuses at load.
- **Geometry and hardware constants are injected from configuration**, never written into
  a feature. Injection does not remove the refit: a new table, grid, width or device needs
  new measurements and a new fit, not a new recipe. A recipe's version changes only when a
  feature function changes. The measurement this rests on (#626 at `1139bbca`): the
  reference recipes hard-coded the DCP chunk geometry, the attention chunk budget and three
  candidate CU counts, two of them the CU counts of two GPU generations, inside feature
  functions, so a new width or device meant a code change.
- **A normaliser is not configuration.** A recipe's normalisers are frozen with it; the
  scheduler's token budget is ATOM's. Changing one never changes the other.
- **`device.compute.compute_units`** is added to the machine spec (`05` D25) when a recipe
  needs it, not here.

### Optional recipes

A correction recipe applies when the target's configuration enables what it corrects, the
bundle includes it, and the batch is inside its domain. That is checked after the bundle
matches the target, so a mismatch refuses rather than skipping. A recipe not selected
loads nothing; an inapplicable one is recorded as skipped. DCP's correction is the first
such recipe and comes after v1; it reads DCP's state as ATOM's own DCP helper reads it.

---

## D101. Bundle identity and checks

### Identity

- **A bundle is an artifact in `07` D41's sense**, beside the artifacts that table lists:
  a key tuple plus a content digest, the path only a location (D41 rule 1). The key is
  (model, device, parallel configuration, the engine configuration the grid was planned
  for, the grid definition and its margin (D98), source-root digest).
- **Immutable once handed off** (D41 rule 4). A changed coefficient is a new bundle, and
  accuracy reports are not inherited.
- **#626's manifest** (digests, dependencies, a five-step load) is a proposal for `07`
  T19, not decided here.
- **No matching bundle refuses**, naming the key that missed. A bundle of another width,
  model or device is never reused.

### Invalidation, per `07` D43

The bundle's row in D43's dependency table is invalidated by every column: ROCm, AITER and
RCCL, torch, ATOM source, model, device, and engine config, because a step measurement
covers all of them. A mismatch refuses. In particular a device or architecture mismatch
refuses (`08` D49 item 2), and drift in the ATOM source refuses rather than marking the
bundle unverified.

### Keyed conditions and derived values

- **Keyed conditions refuse on mismatch**: the key, D43's row, the graph mode, the
  ladder, block size, KV dtype, quantization, TP width, the chunk sizes a recipe reads,
  and the declared scope (D98).
- **Derived values**, such as a chunk width computed from keyed conditions, are not
  declared separately and go into the provenance.

#626 had two more grades, both dropped. An asserted grade, where a user declares a value
the bundle cannot evidence, has nothing to cover: a bundle measured by the compass
toolchain records every condition its grid ran under. A record-only grade lets a mismatch
through, which D43 refuses.

The conditions are read from the resolved ATOM config the engine runs with, after ATOM's
own normalisation and deployment rewrites and never from a first parse of the command
line, and are never restated in a Compass setting (`05` D24, `13` D78). The width checked
is the logical TP, never the count of local workers launched.

### At startup

- **Every rung of the target's ladder inside the bundle's measured range was measured**,
  or the run refuses, listing the rungs.
- **The run's configured bounds lie within the bounds the grid was planned for** (D98), or
  the run refuses.

#626 rejected a startup gate on these bounds: they are upper bounds the scheduler can
reach, not values a run will reach, so such a gate refuses runs that never leave the grid.
Here the grid is planned to those bounds, its boundary nodes included (D98), so the check
fires only on a configuration the grid was not planned for. A step inside the bounds can
still fall in a hole; the pre-run coverage check (D99) and D102 handle that.

---

## D102. Refusal aborts the run

### Decision

**`--compass-on-refusal abort` only.** With step fit selected, `abort` is the default and
`mark` refuses at startup, by name. A graded run is refusal-free by coverage, checked
before it starts (D99), and any refusal voids that run.

### The departure from `08` D50.1

D50.1 prices a refused step by the next rung that can answer, down to tier 0 (item 1);
holds a run with more than 5% of predicted seconds refused inadmissible (item 3); and
keeps `abort` out of acceptance runs (item 4). `13` D80's default `mark` follows it. Step
fit departs from all three items. No lower rung exists for Qwen3.8-27B, since no row of
`16` authors Qwen tier-0 laws (W5.2 authors DeepSeek-V4-Pro's), so marking would have
nothing to price a refused step with. And because coverage is checked before grading,
voiding a run is stricter than the 5% gate, not looser. T92 records supporting `mark`
once a tier-0 law exists for the target.

### Against the tier-a law's rule

#627's [ruling 3](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740)
has the tier-a law that stage 2 prices with extrapolate an out-of-hull
step rather than refuse it, outside the 5% gate, never aborting a cell, and reported
apart. Step fit keeps its stricter rule: it answers only inside complete cells (D99),
aborts on a refusal, and a refusal voids the run. T95 records a later trial of the law's
rule for step fit, decided only once step fit's stage-2 evaluation shows how often steps
fall outside complete cells.

### What a refusal leaves

Contracts, not mechanism:

- **The refused batch gets no forward, no output processing and no forward time**, and
  the clock is not rolled back. ATOM's scheduler has already changed request and KV state
  for that batch, which is why there is no rollback and no skip.
- **The run ends as failed**, not with a fabricated final grant. A failure end is
  distinct from a normal end, and every participant (engines, workers, the clock) stops
  waiting.
- **The first refusal's reason, its batch and the bundle key are kept**, the run record
  is flushed before the run ends (D104), and `01` D3.5's run summary is written and names
  the refusal.

---

## D103. Selecting the backend

- **Selected the way the code already selects a backend**: a third key of the run file,
  `step_fit`, exclusive with `law` and `coefficients`
  (`atom/compass/run/__init__.py::runner`). A run file naming more than one, or none,
  refuses, as it does today. No new Compass flag.
- **The bundle is found by its key** (D101) in the artifact store. Under step fit a
  refusal aborts (D102).
- **The run file is not yet in `13`.** `13` D79 rejects a Compass config file and D80's
  flags select no backend, while the code reads a run file named by `--compass-run`.
  T96 records reconciling the two; this decision follows the code.

---

## D104. The run record, and the CPU segments

### The run record is inside `01` D3.5's outputs

- **#626's per-step run record stays, with no `off` level.** Its summary level is D3.5's
  run summary, always written. Its full level is the per-step detail of D3.5's timeline
  log, which is off by default (D3.5 rule 1).
- **Written incrementally**, and flushed before the run ends on a refusal (D102).
- **Per step:** the phase, the bundle key and digest, the recipe, the lookup's species and
  the cell it used, the residual, the optional recipes applied or skipped, and the final
  seconds.
- **From each `estimate()` return**, never from differences of cumulative counters: a
  cache hit skips lookups, so a counter difference miscounts.
- **`describe()`** is one line naming the bundle key, the recipe and the members.

### The scheduler's CPU segments

ATOM's `scheduler.schedule()` and `scheduler.postprocess()` run unchanged, and nothing
prices scheduling decisions (README, *Scope*). "Scheduler postprocess" here means
`scheduler.postprocess()`, not `ModelRunner.postprocess()`, which is inside the forward's
interval (D98).

- **No `measured` mode.** Charging the simulating host's wall clock would make two runs
  of one input disagree (`01` D3.4), and it charges time the simulated machine never spent.
- **Default `off`**: neither segment is charged, as `01` D4 has it today.
- **`fitted`** prices each segment from parameters in the machine spec's `host.*` (`05`
  D25), a host property like the tokenizer terms. It is not usable until it is measured in
  ATOM whether a segment's time adds serially to the forward or overlaps it (T93): `10`
  D64 found the host floor to be a `max`, not an addend, and a serial charge is the addend.
  When `fitted` becomes usable, `01` D4's site table is amended to name the two segments.

---

## D105. One evaluation on DeepSeek-V4-Pro, not graded

After stage 2, step fit gets one evaluation on DeepSeek-V4-Pro (`16` W4.13):

- **The cell.** The nightly TP cell — TP8, no DP-attention, no MTP, expert parallelism off
  — on MI355X, on the prefill node and the decode node.
- **The grid and fit** are planned from that cell's workload, as D98 plans stage 2's.
- **The comparison.** W4.3's per-step tables (each forward's batch and unprofiled
  seconds, `08` D49 item 1), and tier b where it is available, reported per `08` D47:
  on totals, with the largest contributor held out, and as the median per step.
- **Not a stage-3 backend**, and outside M2's ranking gate (`08` D50). No DP-attention,
  MTP, TBO or EP8 rows; #529 keeps DP-attention's tier a.

---

## Open issues

- **A dispatch band inside a cell is invisible to step fit** until each grid point records
  the kernels it ran (D99). Until then the residual's held-out error is the only detector,
  so a band that the replayed batches never straddle is not caught before grading.
  Recorded as **T94**.
- **Supporting `mark`** needs a lower rung for the target. Recorded as **T92**.
- **Whether the scheduler's CPU segments add serially to the forward or overlap it** is
  unmeasured in ATOM, and gates the `fitted` mode. Recorded as **T93**.
- **Whether step fit should extrapolate out-of-cell steps** as the tier-a law does under
  #627's ruling 3 is untried (D102). Recorded as **T95**.
- **`13` does not describe the run file** that selects a backend (D103). Recorded as
  **T96**.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D96 | Step fit (`StepFitBackend`, `step_fit`) is a tier-a `CostBackend`: grid lookup plus a residual. A candidate for #627's stage 2, graded by stage 2's protocol after `09` D59's validation, a use beyond `07` D36 by ruling; which backend passes stage 2 is #627's ruling, which chose `09` D54's law and keeps step fit off the stage-2 path for now, so step fit's evaluation runs off that critical path. Grading reads #743's registered real runs, or step fit's own named apart; the residual, validation and grid comparisons use traces 256 to 392, grading traces 0 to 255. Not a stage-3 backend, not the source of the per-family MI308X tables; Kimi-K3 and DCP after v1. | 2026-10-10 (#696); #627's rulings applied 2026-10-11 (#626) |
| D97 | One term per step over the forward's whole interval, priced once per logical forward by the runner and charged by the engine. Rows in order; mixed phases and speculative decode refuse; a DP run refuses by key and a PP stage refuses. The backend imports nothing from ATOM. | 2026-10-10 (#696) |
| D98 | A synthetic homogeneous grid, departing from `07` D39 for the baseline only. Its definition (a node list per axis and the planning margin) is a calibration input recorded in the bundle's identity; a cell is the box between adjacent nodes; warm-up, repeats and statistic are fixed in code; measurements are stored per point. Nodes are placed from shapes, and only the cells the planned shapes fall in, plus the margin, are measured; boundary nodes sit at the feasible region's edge. Decode in replay on the target's own ladder, per rung at `B = R` and `B = previous rung + 1`; prefill eager on `(B, N_Q, c)` by whether a chunk produces output. `02` D10's interval, `07` D42's timing; a narrower declared scope refuses. | 2026-10-10 (#696); grid definition 2026-10-11 (#626) |
| D99 | An answer only from a complete cell, by exact hit or interpolation; otherwise refuse, naming the missing corners. No extrapolation, clamp, recovery or fallback. A pre-run coverage check over cells and the residual hull; deterministic lookups; `02` D11 species. | 2026-10-10 (#696) |
| D100 | The residual is in the first version, fitted on replayed batches of traces 256 to 392 in `14` D85's form plus order and raggedness, graded on traces 0 to 255, refusing outside its hull; grid definitions compared on traces 256 to 392 only, each one tried recorded. Members are bundle data; each output checked. Recipes are fixed functions with injected geometry and hardware constants, versioned only when a function changes; optional recipes, DCP's first. | 2026-10-10 (#696); grid comparison 2026-10-11 (#626) |
| D101 | A bundle is a `07` D41 key plus a digest, immutable, with a row in `07` D43 invalidated by every column. Keyed conditions refuse; derived values go to provenance; #626's asserted and record-only grades are dropped. Conditions come from the resolved ATOM config. At startup, unmeasured rungs and bounds beyond the planned grid refuse. | 2026-10-10 (#696) |
| D102 | Refusal aborts: `abort` only and by default under step fit, departing from `08` D50.1 items 1, 3 and 4. A graded run is refusal-free by coverage; any refusal voids it. The refused batch gets no forward and no time, the run ends as failed, and the first reason is kept. Stricter than the tier-a law's extrapolation under #627's ruling 3; T95 records trying that rule later. | 2026-10-10 (#696); ruling 3 contrast 2026-10-11 (#626) |
| D103 | Selected as the code selects a backend: a `step_fit` run-file key, exclusive with `law` and `coefficients`; no new Compass flag. The bundle is found by its key in the artifact store. `13` does not yet describe the run file (T96). | 2026-10-10 (#696); revised 2026-10-11 (#626) |
| D104 | The run record lives inside `01` D3.5's outputs with no `off` level, written incrementally and flushed on refusal, per step from each `estimate()` return. The scheduler's CPU segments: no `measured` mode, default `off`, `fitted` from `host.*` once T93 is settled. | 2026-10-10 (#696) |
| D105 | One non-graded evaluation on DeepSeek-V4-Pro's nightly TP cell on MI355X, against W4.3's step tables and tier b, reported per `08` D47; not a stage-3 backend. | 2026-10-10 (#696) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T92 | Support `--compass-on-refusal mark` under step fit once a tier-0 law exists for the target | no row of `16` authors a Qwen tier-0 law |
| T93 | Measure in ATOM whether the scheduler's CPU segments add serially to the forward or overlap it, before `fitted` is usable | needs a real run with per-segment timing |
| T94 | Record the kernels each grid point ran, and treat a cell whose corners ran different kernels as crossing a dispatch band (`09` D60): refused or split | needs the grid measurement (`16` S2.2) to record kernel identity |
| T95 | Trial, not this design: extrapolate an out-of-cell step as #627's ruling 3 has the tier-a law do, by one method only, extending the nearest complete cell's interpolation, tagged `extrapolated` with the distance, reported apart, outside the 5% gate, never aborting. It would replace the abort-only rule and the `08` D50.1 departure (D102) | decide after step fit's stage-2 evaluation shows how often steps fall outside complete cells |
| T96 | Reconcile `13` D79 and D80 with the run file the code selects a backend by (`--compass-run`, `atom/compass/run/__init__.py::runner`): `13` rejects a Compass config file, the code reads one | needs an owner choice between describing the run file in `13` and moving selection to flags |
