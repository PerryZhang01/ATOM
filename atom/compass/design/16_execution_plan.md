# ATOM Compass — Design Topic 16: The Execution Plan

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** all of `01`–`15`, and `17`. This document turns those topics' decisions,
and the register of open items in [`12_open_items.md`](12_open_items.md), into work that
can be allocated. Neither is counted here. The register is owned by `12`, restated once on
`README.md`'s front page, and it moves as tasks land; a third copy in a document that does
not own it has been stale before, and "open items" and "registered items" are two different
numbers that a single figure here cannot distinguish.

**Revised 2026-09-30 by D95, and aligned with the e2e stage order
of [#627](https://github.com/jgong5/ATOM/issues/627).** Priorities changed: M1 is #627's
stages 1 and 1b, and M2 is v1, its stage 3. Phase 0 through Wave 3 stand, with
the rows D95 changes marked in place; Phase 0b is new; Waves 4–6 replace the rough
sketch that followed Wave 3.

**Revised 2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791).** The
toolchain's steps half (`13` D80) is stage 2's, S2.7, built ahead of the ops half, which
stays W4.1's; the module layout names where each piece goes. Revised 2026-10-14 by the
owner's [ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104):
S2.7 adds `measure memory`, and grading decides whether a run is gradeable.

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** tasks and gates carry its names, the
step linear law, the step grid (`17`), the analytical laws, and the CPU and GPU gates.

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** Wave 5 is #627's stage 3c, the
analytical projection at step and op level, its tasks the issues #1392 tracks; the
projection's first check is the analytical bound on every measured step (D95).

**Scope.** How the work is organised, allocated and gated; what happens first; and what
each stage must produce. It is **detailed through M2**. Work after v1 is listed, not
planned — see *After v1*. Phase 0b's answers can still reshape Waves 4–6, and the plan
marks where rather than pretending the DAG is stable across them.

---

## D95. Priority change: M1 is redefined, M2 is v1

### Context

The milestones of `README.md` ran from a fake-model M1 through Qwen3.8-27B on MI308X-class
hardware (M2–M4) to Kimi-K3 at TP8 with DP, PP and EP (M5–M7). Priorities changed: the
focus is now an **MoE model with wide expert parallelism and P/D disaggregation**, with
**full support on MI355X** and **roofline projection for MI455**. Calibration and pricing
move onto MI455 as soon as MI455 hardware is available.

Owner rulings shape the milestones below: Compass work is ordered by the end-to-end
cc-traces stages of [#627](https://github.com/jgong5/ATOM/issues/627); the real side is
re-run on hardware this project books ([#503](https://github.com/jgong5/ATOM/issues/503));
a host that holds a GPU may run a GPU-free stage
([#472](https://github.com/jgong5/ATOM/issues/472)); stage 3 runs first on the nightly
DP-attention cells as written, with expert parallelism off
([#501](https://github.com/jgong5/ATOM/issues/501#issuecomment-6077912835)); and EP8 is
part of v1, added after that
([#646](https://github.com/jgong5/ATOM/issues/646#issuecomment-6078555307)).

### Decision

- **The model is DeepSeek-V4-Pro as ATOM ships it** — routed experts in FP4 (e2m1
  microscaling), the rest in FP8. ATOM already serves it 1P+1D through atomesh and
  Mooncake at TP8, with DP-attention, TBO and MTP, nightly on MI355X
  (`recipes/mesh/DeepSeek-V4.md`), and carries gfx1250 paths for it. No nightly cell
  enables expert parallelism. The nightly runs define the configurations to measure; they
  are not ground truth. The real side runs on the dedicated MI355X pair (P0b.3).
- **M2 runs in two steps, expert parallelism off and then EP8.** First the nightly cells
  as written: TP8, DP-attention where the cell sets it, expert parallelism off (#501).
  There is no MORI all-to-all; the DP group's exchange is the variable-size gather sized by
  the peers' token counts. Then, before v1 is done, DP-attention + EP8 within a node, with
  a MORI routing model, graded the same way; its real side is this project's own run with
  `--enable-expert-parallel`, since no nightly cell passes it. A P/D deployment is one
  prefill node and one decode node. EP beyond one node is out of scope, not projected.
- **M1 is #627's stages 1 and 1b** (`README.md` owns its text): cc-traces end to end on a
  simulated run, at TP1 and then 1P+1D through atomesh, each ended by the finish. The run
  is GPU-free: no compute and no allocation on the device. The scheduling-parity test
  leaves M1 for M2. Its exit criterion is under *Wave 3*.
- **M2 is v1, #627's stage 3**: DeepSeek-V4-Pro 1P+1D on MI355X, in the nightly
  configurations and then at EP8, each step first with the same
  scheduling decisions as the real engine, then within the acceptance targets; and the
  MI455 roofline projection. **Stage 2 is its first gate**: Qwen3.8-27B TP1 on MI308X,
  paired real and simulated cc-traces graded by `08`. Its grading is under *Waves 4–6*.
- **The TTFT target is 15%, not 10%** (`README.md`). TTFT is the hardest serving metric
  to predict in every surveyed tool, and it carries prefill, queueing and P/D transfer at
  once. The other targets stand.
- **The analytical laws move onto the critical path.** `10` D68 authored analytical laws during the
  empirical campaign as a side wave; the MI455 projection makes them an M2 deliverable.
- **A projection is sound, not graded.** MI455 has no ground truth in M2, so the
  projection is held to these checks instead of an error bar (`08` D50.2, revised by
  #1392): on every measured step, at both granularities, analytical is at most measured;
  every MI455 input is sourced or marked assumed; and the same projection method, run
  MI308X → MI355X on Qwen3.8-27B (DeepSeek-V4-Pro's FP4 experts do not run on gfx942), is
  reported against real MI355X runs to give the method an error estimate.
- **The other milestones are retired, not renumbered.** Each maps as follows, so an older
  citation still resolves:

| Old | Where its scope went |
|---|---|
| M1 | Redefined above as #627 stages 1 and 1b; its scheduling-parity test moves to M2, and PP2 leaves that test |
| M2 | M2's first gate, #627 stage 2: Qwen3.8-27B TP1 on MI308X, paired and graded by `08` |
| M3 | Not a milestone: Qwen3.8-27B is the development and CI model and the projection check of M2 |
| M3.5 | Into M2: MTP runs in the nightly configurations |
| M4 | Into M1 (the simulated 1P+1D run) and M2 (accuracy, on DeepSeek-V4-Pro and MI355X) |
| M5, M6 | Replaced by DeepSeek-V4-Pro |
| M7 | DP as DP-attention, then EP8, into M2; PP deferred |

### Consequences

- W3.2 (PP) is deferred. W3.3 (EP) moves to M2's EP8 step as W4.10. W3.5 is M1's 1P+1D
  slice, two containers on one node; W4.9 carries it to DeepSeek-V4-Pro's topology in M2.
  T84 is answered by this decision: EP2 means
  `-tp 2 --enable-dp-attention --enable-expert-parallel`, the form in which the all-to-all
  runs, and is the cheap leg of the EP8 step's parity test.
- The width-transfer question (T21) leaves the critical path: M2 is graded at the widths
  the nightly configurations run, not projected across width.
- Target dates live outside the design; the plan stays sized in lines of code, per
  *What this plan does not contain*.

---

## The module layout, which is also the task boundary

Tasks are cut so each touches one module plus its tests.

```
atom/compass/
  clock/      CA core, grant rule, channel table and path distances, LP registry, both transports
  detect/     straggler check, stall diagnostic, CI clock-source lint
  runner/     CompassModelRunner - the seam, RPC surface, forward semantics
  backends/   CostBackend interface, StepCost, provenance and method; the step-level and op-level backends
  memory/     device-reading substitution, per-term model
  ir/         Cost IR (Seq/Repeat/Par), nested-Repeat detector, opaque leaves
  capture/    FakeTensorMode capture, guard domains
  spec/       machine-spec schema, probes, merge / validate / explain
  artifacts/  store, keys, digests, invalidation matrix
  drive/      serving ATOM for record, trace steps and run; measure memory; run.json; the GPU checks
  bench/      the measurement bench: replay by rule, the window, measure
  fit/        samples for fit and publishing; each backend owns its fit
  validate/   grading: whether a client run is gradeable, the separable results
  metrics/    per-step sampling, backfill
  kv/         simulated KV connector
  cli.py      the `compass` console script, registered in pyproject.toml (13 D80); it parses and dispatches only
tests/compass/          mirrors the above
<out-of-tree package>   the aiperf adapter (never inside ATOM)
```

Edits to ATOM outside `atom/compass/` are deliberately few and are enumerated per task —
clock-read substitutions, the channel wrappers and idle points of `01` D4, the config flags of `13` D80, and
the two ATOM-owned additions (`--preprocess-pool-width`, `--spec-decode-acceptance-rates`).

---

## Phase 0 — de-risking, everything parallel

Nothing here depends on anything else. All seven can be claimed at once, subject to the
five-slot cap.

| ID | Task | Module | Effort | Hardware |
|---|---|---|---|---|
| **P0.1** | Environment: worktree layout, container mounts covering the worktree parent, `PYTHONPATH` wrapper with verification, GPU pre-flight script | tooling | ~150 LOC | — |
| **P0.2** | Baselines: record ATOM suite and ruff pass/fail state | — | ~0 | — |
| **P0.3** | **T10** — can `AgenticReplayStrategy` be subclassed rather than vendored? | spike | ~50 LOC | laptop |
| **P0.4** | **T5** — trace the 27B under `FakeTensorMode` at TP1 **and TP2**; diff the captured structures | spike → `capture/` | ~200 LOC | clean node |
| **P0.5** | **T64** — does ATOM microbatch PP? | reading + 1 run | ~0 | clean node |
| **P0.6** | **T65** — EP group membership per supported configuration | reading | ~0 | — |
| **P0.7** | **T52** — root-cause the `DeviceContext` `__torch_function__` guard 8-rank hang. **Only if P0.4 hits it.** | spike | — | quiet node |

**Why these seven.** Each can invalidate work that would otherwise be built on top of it.
P0.3 swung an estimate by ~2,000 lines for one hour's work; it has since run and the swing
did not happen. P0.4 decides whether the op-level backend has an IR at all, and **it is the task most
likely to reshape Waves 2–4** — if tracing does not work at TP>1, `capture/` and `ir/`
change shape. The plan marks that edge rather than pretending the DAG is stable across it.

**P0.7 is conditional.** `15` and `04` both argue a fake-tensor trace should be GPU-free
and collective-free, so the known hang should not be reachable from P0.4. If P0.4 confirms
that, T52 drops to ordinary priority and gates nothing on the critical path.

**Each of P0.3–P0.7 ends in an escalation, not a decision.** The result plus its options
and their costs go to the project owner; the scope call is theirs (see Escalation points below).

---

## Phase 0b — de-risking for D95, everything parallel

D95 brings in unknowns of the kind Phase 0 exists for: each can invalidate work built on
top of it, and each is cheap to settle first. Same rules as Phase 0 — nothing here depends
on anything else, and each ends in an escalation, not a decision.

| ID | Task | Module | Effort | Hardware |
|---|---|---|---|---|
| **P0b.1** | Trace DeepSeek-V4-Pro under `FakeTensorMode` with `--load_dummy empty` at the nightly shape (TP8 with DP-attention, MTP on), with expert parallelism off as the nightly runs it and at EP8. Name every operator with no fake or meta implementation — the FP4 MoE, the CSA indexer, mHC, MORI dispatch/combine at EP8, the MTP draft path are the expected ones — and whether each needs a shape rule only or more. **Answered on #501** | spike → `capture/` | ~200 LOC | CPU container; a clean node only if model construction reaches the driver |
| **P0b.2** | DP-attention in the fake-model run, expert parallelism off and at EP8: which per-step exchanges the DP group makes (metadata sync, dummy batches for idle ranks, the variable-size gather of the peers' tokens with EP off, the MORI all-to-all at EP8), and whether each is already a classified site in `atom/compass/audit/sync_sites.json` | reading + one CPU run | ~50 LOC | — |
| **P0b.3** | What the real side needs recorded: per-request timings and the exact launch configuration. The nightly P/D runs of `recipes/mesh/DeepSeek-V4.md` define the configurations, not ground truth: the owner ruled on #503 that the paired runs are re-run on the dedicated MI355X pair. What is left is reading what a run records from the upstream workflow (`.github/workflows/atomesh-benchmark.yaml`) and its bench scripts, so W4.3 records the same | reading | ~0 | — |
| **P0b.4** | The MI455 inputs: which fields of the `05` machine spec the analytical laws need for DeepSeek-V4-Pro's operator families (matrix-core rate per precision including FP4 and FP8, HBM and LDS bandwidth, the in-node fabric), and for each one a source or an "assumed" mark | reading + owner input | ~0 | — |

**P0b.1 is the one most likely to reshape Waves 4–6**, for the reason P0.4 was: if the
trace fails, `capture/` and the whole pricing path change shape. **P0b.4 needs the
owner** for its second half, pre-release MI455 numbers, because it is an organisational
question, not an engineering one.

---

## The measured test and lint baselines

P0.2's result, and the reason the agents' original test gate cannot be run as written.
That gate, `08` D43.1 and ATOM's own `CLAUDE.md` all assert the same thing — that the
187-file suite under `tests/` runs GPU-free and green — and P0.2 measured both halves
false. What is GPU-free is the **CPU gate**, run per task; the rest is the GPU gate, a superset, run per
wave as a delta. Re-derived on 2026-09-20 after review: the first derivation iterated
*whole-suite collection* to a fixed point, which measures a property of the collection
order, not of the files.

**Derivation — one `pytest <file>` per fresh process,** over all 157 non-plugin files under
`tests/`, in container `xiaobizh_n18_cpu` on hjbog-srdc-18, against a `git archive` snapshot
of `b963c9411` with `PYTHONPATH` asserted to resolve `atom` under that root (Python 3.12.3,
pytest 9.0.3):

| run alone, the file… | files |
|---|---|
| fails collection — `RuntimeError: Get GPU arch from rocminfo failed` | **30** |
| collects, fails 2 tests `RuntimeError: hipHostMalloc failed: 100`, then **never exits** (killed at 600 s) | 1 — `test_lmcache_offload_disk_integration.py` |
| collects **zero** tests (module-level skip) | 20 |
| runs at least one test | 106 |

One cause for all 30 — but only per file. Collected as a whole suite the same tree gives 37
errors, 7 under `tests/plugin/` and 30 elsewhere, and those 30 wear four faces: 25
`rocminfo`, one `KeyError: 'aiter'` (`test_dcp_topk.py`), one `AttributeError` on
`atom.model_ops.attentions.gdn_attn` (`test_kda_checkpoint_slot_copy.py`), and three with no
exception recorded. Alone, all five resolve: the two "import defects" are `rocminfo` reached
through a half-initialised `aiter` left by an earlier file, not defects in ATOM's mocks; and
`test_dp_metadata`, `test_dp_sync_layout`, `test_forward_mode` are **CPU-green** — 31 passed
in 0.18 s — and were excluded on that non-evidence. The sentence that `08` and this document
both carried, "32 files reach the driver, 28 of them at collection time via `rocminfo`",
joined two real numbers wrongly: the 28 counts `rocminfo` across the *whole* suite,
`tests/plugin/` included (25 + 3), and was attached to a non-plugin set of 32.

- **Per task — the CPU gate, re-measured by P0.1 and superseding the counts above.**
  `tests/` minus `tests/plugin/` (30 files) minus the 29 driver-dependent files in
  `scripts/compass/cpu_gate_exclude.txt`, driven by `scripts/compass/gate_cpu.sh`:
  **130 of the 189 test files at `105ca4197`, 4030 passed, 0 failed, 149 skipped, 3 xfailed,
  rc=0** in `xiaobizh_n18_cpu` on hjbog-srdc-18. `105ca4197` is P0.1's head and landed as
  `4c16792d9`, the same tree. Six runs at two of that branch's commits before its restack, `d737f15e7` and
  `7ff80cc4b`, gave the same counts, and the clock read **25.6-30.8 s of pytest inside
  31.6-36.9 s of wall (`time` real)**, which is a measured spread rather than a bound: it tracks
  what else is on the node. Run against a `git archive` snapshot of the tree, with
  `PYTHONPATH` asserted to resolve `atom` under that root and pytest's own exit status
  captured before any pipe. The 4030 at `105ca4197` is **3956 ATOM tests + 74 `tests/compass/` tests**,
  stated as its parts because a single total cannot show which half moved (principle 7). The
  3956 is P0.1's control, `042aad97d` with P0.1's `scripts/compass/` copied in and, like the
  P0.2 readings above, no `tests/compass/`; `105ca4197` adds `tests/compass/test_cpu_gate_exclude.py`
  and `tests/compass/test_gate_gpu_surplus.py`, so the file count moves 128 → 130. **4005** was the same gate at
  `3afcb4880`, with `tests/compass` at 49 tests. The 49 → 74
  growth is two steps: `tests/compass/test_cpu_gate_exclude.py` parametrises one case per
  entry of `gpu_gate_triggers.txt`, and correcting that file's derivation took it from 13
  entries to 30 (+17); `tests/compass/test_gate_gpu_surplus.py`, added here, is the other 8.
- **Green is the bar, but green is not "exercised".** Of the 130 files handed at `105ca4197`, **22 collect
  no test at all**: 16 declare a device dependency, 3 need PyAV, 2 are dead since ATOM #690
  split `kv_transfer_engine` into `moriio` (`test_kv_connector_scheduler.py`,
  `test_transfer_engine.py`), and `test_prefix_cache_accuracy.py` has **no test function at
  all** — it is an `argparse` script that drives a live server on `localhost:8000`. That
  decomposition was measured per file by P0.2 at `b963c9411`, where the same 22 sat inside a
  128-file gate whose 149 skips were 68 distinct reasons: 66 skipped tests name a device, 83
  do not. Both carry to `105ca4197`: its `tests/` and code differ from `b963c9411`'s only by
  the two `tests/compass/` files above, which collect tests and skip none.
- **Per wave — the GPU superset, re-measured by P0.1.** `tests/ --ignore=tests/plugin` in the
  GPU container, driven by `scripts/compass/gate_gpu.sh`, judged as a **delta**, never as
  "green". The baseline is **4779 passed / 5 failed**, 0 errors, 105 skipped, 3 xfailed,
  72.6 s, measured 2026-09-20 at `fe9ea043c` on node 18 in `xiaobizh_n18` with
  `HIP_VISIBLE_DEVICES=1` — torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip`
  **7.2.53211**, ROCm **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509** — over two runs with
  byte-identical failing sets. It supersedes P0.2's **4730 / 5** at `83daf636d`, whose torch,
  AITER and ROCm versions were unrecorded and which therefore could not be reproduced as
  recorded; **T77**, which tracked that gap, is closed by this measurement. The five are
  pre-existing and unrelated to Compass, and they are **four ULP comparisons plus one bitwise
  check**, not "five bf16 ULP failures": four `allclose` cases in
  `tests/test_fused_compress_ragged.py` off by one bf16 ULP
  (`max|diff| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3`), plus
  `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise`, a
  `torch.equal` with **no tolerance at all** — a tolerance bump would not move it. All five
  node-ids are on file verbatim in `scripts/compass/gpu_gate_known_failures.txt` and compared
  by name, so "5 failed" is checked against "the *same* 5 failed". A review record that
  claims "green" instead of citing the delta has not read the baseline. The pass count moves
  by construction — the superset includes `tests/compass/`, so every test this phase adds
  raises it above 4779, and `gate_gpu.sh` therefore expects
  `4779 + (this tree's tests/compass count - 49)` and refuses a bare "no worse than".

**The exclusion list lives in the tree.** P0.2 derived it here, in prose, because
`scripts/compass/` did not yet exist on this branch and a gate may not name a file its own
tree does not contain. P0.1 mechanises it: `scripts/compass/cpu_gate_exclude.txt`, with
`gate_cpu.sh` to run it and `regen_cpu_gate_exclude.sh` to rewrite the generated half. From
this commit that file is the source of truth and is regenerated rather than edited, and the
prose derivation above is superseded. Its 29 entries are **28 GENERATED + 1 MANUAL**: the
generator iterates batch collection to a fixed point and produces 28, and the single manual
entry — `test_lmcache_offload_disk_integration.py`, which collects cleanly and then fails on
`hipHostMalloc failed: 100` — carries the observed failure text above it, which is the only
evidence such an entry can have. Both reviewers found that the earlier 32-entry list could
not be reproduced by its own generator: four entries had been hand-added to a file whose
header read "never hand-edit", and three of those four — `test_dp_metadata.py`,
`test_dp_sync_layout.py` and `test_forward_mode.py` — collect *and pass* in a driverless
container. Prose saying the list is 32, or that those three need the driver, is stale.

The list is regenerated under the *batch* the CPU gate runs, because that is the configuration
whose green is claimed — and the per-file census above is a separate question, not a
replacement for it. Both are needed: `test_postprocess_width.py` and
`test_v4_checkpoint_slot_copy.py` sit inside the CPU gate, module-skip in the batch
("model_runner imports aiter at module load", "the V4 builder's module imports aiter at
load"), and fail collection with `rocminfo` when run alone; the three CPU-green files above
go the other way. That pair is also the whole difference between the two counts of silent
files — 20 per file, 22 in the CPU gate. Recording only the batch is how three CPU-green files
stayed excluded and how one attributed cause replaced four.

**The CPU gate's blind spot is a file, not a sentence.** P0.2 stated it as prose naming four
areas — EPLB, DP metadata, cudagraph bounds, block tables — which nothing read, and which
named two files (`test_dp_metadata`, `test_dp_sync_layout`) the CPU gate does in fact cover.
It is now `scripts/compass/gpu_gate_triggers.txt`, **30 paths** in the tree committed here,
generated by `regen_gpu_gate_triggers.sh` and never hand-edited — the counts in its own
header included (at `fada7424e`: 189 test files, 130 CPU gate, 108 of them collecting, 47 candidates, 99
covered). The rule it applies: an `atom` module named by an excluded test is a blind spot
**unless a CPU-gate test that actually runs names it too**. A module both gates import is
covered when the CPU gate runs, so triggering on it would make the gate cry wolf. A trailing
`/` matches a subtree. `gate_cpu.sh` matches the diff against that file and **exits 98 unless
`COMPASS_GPU_GATE_DONE` names this tree's own HEAD**; where it can compute neither a diff nor
a supplied file list — a `git archive` snapshot with no `.git` — it refuses rather than
reporting "not required", per principle 6.

The rule has three parts. One was forced by a counter-example measured in this tree, one is
what keeps the headline example, and one currently changes no path:

- *Imports are read at any indentation on the excluded side.* 190 of this tree's
  `import atom.*` lines are indented — inside a function, a `try`, or a
  `skipif(not torch.cuda.is_available())` guard. An earlier draft anchored the match at
  column 0 on both sides and silently dropped `atom/model_ops/topK.py`, whose only
  excluded-side reference is `test_moe_dp_token_capacity.py::test_dpa_capacity_fits_gathered_topk_metadata`, indented under exactly such
  a guard.
- *Coverage is credited only for a module-level import.* An indented import in a CPU-gate
  file is not proof that the CPU gate executes it, and crediting it would let a never-taken
  branch suppress a trigger. **This is the part that keeps
  `atom/model_engine/model_runner.py`** — the module Compass's runner seam replaces.
  `tests/test_mla_index_cache.py` imports `ModelRunner` at
  `tests/test_mla_index_cache.py::test_model_runner_local_total_layers_adds_mtp_only_on_drafter_stage`, indented four spaces inside a test function, so that
  import is never credited whatever the file collects. Measured at `236abfd9a` in
  `xiaobizh_n18_cpu`: crediting coverage at any indentation drops the set 30 → 29, losing
  `topK.py`; doing that *and* crediting non-collecting files drops it 30 → 27, losing
  `model_runner.py`, `aiter_mla.py` and `topK.py`.
- *Coverage is credited only from a CPU-gate file that collects at least one test.* 22 of the
  130 CPU-gate files collect none. On this tree that probe **removes no path**: 30 triggers
  with it, 30 without, difference empty — measured at `236abfd9a` in `xiaobizh_n18_cpu`,
  where it withholds 15 coverage paths, and the only one of those that is also a candidate,
  `atom/model_ops/v4_kernels/state_writes.py`, is absorbed either way by the candidate
  subtree entry above it. It is kept as a forward guard for trees this one does not
  represent, and it is not free: a full `pytest --collect-only` over the CPU gate and one
  more refusal path (exit 97). Whether that guard earns its cost is an open call
  (principle 3), not a settled one.

**That file errs in both directions, so do not cite it as a floor.** Toward *firing*: an
indented import in a CPU-gate test that does run is not credited, so a module can be listed
although the CPU gate reaches it, and coverage is subtracted by exact string, so a candidate
subtree entry is never cancelled by coverage of a file under it. Toward *silence*: imports
are read as text, not resolved as a graph, so a transitive import, an `importlib` call or a
re-export is invisible. A path *absent* from `gpu_gate_triggers.txt` is not a claim that the
CPU gate covers it, and a path *present* is not proof that it does not. Of the two mistakes
this gate prefers the first, because running the GPU gate when in doubt is never the wrong
one. Judging that a change sits outside the blind spot remains the task's own call.

**A task touching EPLB, CUDA-graph capture bounds, block tables, the paged index builders or
the prefix-cache kernels runs the GPU superset as part of its own gate, not at wave end.**
That is where the files running nothing here concentrate — 29 excluded plus 22 exercising
nothing, 51 in all; the silent 22 include `test_pool_index.py`,
`test_prefill_indices_paged.py`, `test_decode_indices_paged.py`, `test_postprocess_width.py`
and `test_prefill_prefix_vs_native.py`, the paged-index and prefix-cache areas `03` D13 and
`01` D6 lean on. `test_dp_metadata` and `test_dp_sync_layout` are **not** among them: they
are CPU-green, measured above, and booking a GPU for them spends the resource the booking
queue below exists to ration.

**Three files are covered by neither gate**, measured in `xiaobizh_n18` at `83daf636d`:
`test_prefix_cache_accuracy.py` (`no tests ran`), `test_kv_connector_scheduler.py` and
`test_transfer_engine.py` (`1 skipped`, ATOM #690). The first and the second are cited by
`08` D43.1 as coverage Compass keeps. See T80 in [`12_open_items.md`](12_open_items.md).

**Lint, same day.** In `xiaobizh_n18_cpu`, `ruff check .` gives **1003 errors, 640 fixable**
and `black --check .` is clean over **660 files** (ruff 0.16.7, black 26.5.1 — the GPU
container has neither, so lint and the GPU superset cannot be run in one place). Both
reproduce at `83daf636d` on 2026-09-20. So the lint bar is "**no new** ruff error, and black
stays clean" — never "ruff is clean", which it has never been.

Every number above is stated here rather than cited. The task record lives outside the tree
and the raw pytest output lives only in the containers that produced it; a citation to either
is not resolvable from a checkout, which is the same defect as naming a script that is not in
the tree. The commits carrying P0.2's measurements are `9c8df1328` and `b963c9411`, landed as
`947d5b282`; the ones that supersede them are on this branch. P0.1's GPU re-measurement has
an **in-tree** record, because a plan of record cannot cite one that is not: the five failing
node-ids verbatim in `scripts/compass/gpu_gate_known_failures.txt`, and the pair, the tree and
the toolchain as the `BASE_*` constants in `scripts/compass/gate_gpu.sh`. P0.2's own task
record predates that measurement and contains neither `4779` nor `fe9ea043c`; it is a source
for the superseded 4730 / 5 above and for nothing else.

---

## Wave 1 — foundations

Ten independent tasks. No task here depends on another in this wave; each depends only on
Phase 0's environment. All CPU-only.

| ID | Task | Module | Effort | Implements / consumes |
|---|---|---|---|---|
| **W1.1** | CA core as the HLA RTI (`01` D3): TAR/NER/TAG, a grant strictly below the lookahead-distance LBTS with in-transit messages counted, a channel table addressed by LP and channel name with the path distances `D(j->i)` computed once by Floyd-Warshall (`atom/compass/clock/channels.py`, #453), LP registry, **both transports** (in-process and socket) from one implementation | `clock/` | 600–900 | implements the clock client API |
| **W1.2** | Synthetic-LP harness modelling ATOM's **verified** topology — the ZMQ/shm/Gloo links of `01` D1 and the K1–K9 mechanisms of D4 — plus adversarial scenarios: a send missing from the sender's send log (induced straggler), a duplicate seq on one channel, a wait the CA cannot see | `tests/compass/clock/` | 400–600 | consumes W1.1 |
| **W1.3** | Causality detectors (`01` D3.2): straggler check on arrival and at each drain (fails the run), `DIAG_S` stall diagnostic (keeps waiting, never aborts), CI clock-source lint | `detect/` | 250–400 | consumes W1.1 |
| **W1.4** | `CompassModelRunner` skeleton: the `--runner-qualname` subclass, the full RPC surface, and the three forward semantics of `02` D10 (deferred output, the meaningful-step unit, `produces_output`) | `runner/` | 400–600 | implements the runner seam |
| **W1.5** | `CostBackend` interface, `StepCost` with its breakdown, the provenance vocabulary and the resolver | `backends/` | 200–300 | implements the backend API |
| **W1.6** | M1 fake model: HF-config geometry plus the step stub of `02` D12, including the quadratic query term | `backends/` | 300–450 | consumes W1.5 |
| **W1.7** | Machine-spec schema, `merge` / `validate` / `explain`. No probes yet | `spec/` | 400–600 | implements the spec artifact |
| **W1.8** | Wire-contract fields on ATOM's real endpoint: one `compass` object each direction (`06` D28) | ATOM entrypoints | 100–200 | implements the wire contract |
| **W1.9** | aiperf adapter package, out of tree: clock client, pacing redirect, latency anchors from response fields | out-of-tree | 450–650, reopened | consumes W1.8 |
| **W1.10** | Cost IR: `Seq` / `Repeat` / `Par`, the **nested** Repeat detector with its index binding, and the provably-free grouping rule (`04` D19) | `ir/` | 500–700 | implements the IR |

**W1.2 is the one that earns its keep.** It is where the distributed CA — the riskiest
component, and the one whose failures are silent — gets exercised without ATOM and without
a GPU, so it runs in CI on every change thereafter.

**W1.9's estimate is no longer conditional on vendoring. P0.3 resolved T10 on
2026-09-20:** subclassing works and nothing is vendored. The spike also found that a
strategy subclass alone is **not sufficient** — further pacing sites live in
`BranchOrchestrator` and `ReplayBarrierCoordinator`, which a subclass never sees.

**Owner decision, 2026-09-20 — rebind *and* assert (option C).** W1.9 rebinds the shared
`LoopScheduler` module global from a Compass bootstrap, **and** registers a strategy
subclass that raises unless the scheduler it receives is already clock-paced (`06` D34).
The subclass does no wrapping; it exists because the rebind's failure mode is silent, and
a silently wall-clocked run is the worst result this design can produce.

**Review of P0.3 then showed the tripwire is load-bearing, not belt-and-braces**, and that
the 450–650 range needs re-costing before W1.9 starts:

| Correction | Effect on W1.9 |
|---|---|
| A bootstrap that precedes the first `PhaseRunner` **does** exist. `discover_plugins()` resolves each entry point with `importlib.util.find_spec`, which on a *dotted* value imports the parent package, so `compass_harness.plugin:plugins.yaml` executes `compass_harness/__init__.py` during discovery. The rebind is deferred from there by a stdlib import hook, because that bootstrap runs inside `aiperf.plugin.plugins`' own module body and cannot import `aiperf` — executed, five combinations, 7/7 in the working shape (`06` D34, T73) | **T73 is a packaging choice plus ~20 lines of import hook, inside the existing "plugin manifest, bootstrap, config glue" row — not a precondition of the seam and not W1.9's first deliverable.** The bootstrap shape that *would* be too late is a side effect of the strategy module: phase 0 on the real clock, phase 1 rebound, smoke test passes. That is why the tripwire ships regardless |
| `runner.py:191` calls `LoopScheduler()` with no arguments, so the rebound class must be no-arg constructible — `ClockPacedLoopScheduler(LoopScheduler)`, a subclass, not the spike's `ClockPacedScheduler(inner)` wrapper | The spike validated option A's shape. "About five lines" was costed against the wrong object, so that component row is **open** |
| One scheduler per `PhaseRunner`, and `seamless=True` keeps two live | T76 — reconcile, or assert `seamless=False` |
| Two `loop.call_later` idle-cap timers the rebind cannot reach, one of which upstream deliberately keeps outside the scheduler | T75 — override the two `_arm_*` methods, or declare the feature unsupported and assert both caps are `None` |

**W1.9's acceptance is a list, not a count:** every one of the nine enumerated pacing calls
in `06` D34 is observed to advance on the virtual clock — plus a stated disposition for the
two timers in T75. Those line numbers are pinned to agentx-harness
`56a0cf70f4c0359454ee4bd15a17770b541a3e3e`; re-verify them against that revision, or
re-derive them from the whole-tree grep in `06` D34, before building the test to them.
"None of the seven advances on the real clock" was unimplementable as written; the seven
added four *methods* to three *call sites*.

It does **not** cover the 32 `asyncio.wait_for` timeout sites, which bypass `LoopScheduler`
entirely (T74).

---

## Wave 2 — integration

Depends on Wave 1. The first three are the vertical slice.

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **W2.1** | **The vertical slice**: one simulated request end to end at TP1 with the co-hosted CA — runner + clock + fake backend + ATOM's **real** scheduler, block manager and prefix cache | `runner/`, wiring | 300–500 | W1.1, W1.4, W1.6 |
| **W2.2** | Memory: substitute the five device readings, reuse ATOM's budget arithmetic and `plan_pools`; per-term validation, never a sum | `memory/` | 400–600 | W1.7 |
| **W2.3** | Simulated KV connector registered into ATOM's existing factory; `latency + bytes/bandwidth` from the spec | `kv/` | 250–400 | W1.7 |
| **W2.4** | Artifact store: keys, digests, source-root fingerprints, the invalidation matrix | `artifacts/` | 500–700 | W1.7 |
| **W2.5** | `FakeTensorMode` capture with the three mandatory disciplines and post-trace assertions; guard-domain evaluation | `capture/` | 500–800 | W1.10, **P0.4** |
| **W2.6** | Metrics under virtual time: per-step sampling, the class taxonomy, OpenMetrics backfill | `metrics/` | 300–500 | W2.1 |
| **W2.7** | Determinism test: byte-diff of two step tables from one configuration, in CI (`01` D3.4) | `tests/compass/` | ~100 | W2.1 |
| **W2.8** | Model construction under `FakeTensorMode` with `--load_dummy empty` (`02` D10.1), including the buffer audit of T68 | `runner/` | 200–350 | W2.5 |

---

## Wave 3 — M1 completion

| ID | Task | Effort | Notes |
|---|---|---|---|
| **W3.1** | LP structure for DP: the DP group is one LP with one member process per rank, joined by the CA (#528); its step is one TAR costing the `max` over ranks of each rank's own cost, no operator-level TAR/NER (`01` D4, `15` D90); both collectives run for real; dummy-batch pricing for idle ranks (`15` D90) | 300–500 | |
| ~~**W3.2**~~ | ~~LP structure for PP: one LP per stage; transfer as a size from the spec; layer split via `get_pp_indices`~~ | ~~400–600~~ | **deferred by D95** — see *After v1* |
| ~~**W3.3**~~ | ~~EP: `exclusive` occupancy honoured in the IR; group membership per P0.6~~ | ~~200–350~~ | **moved to W4.10 by D95**, M2's EP8 step |
| **W3.4** | Standalone CA exercised against the synthetic-LP harness at multi-container topology | 200–300 | consumes W1.2 |
| **W3.5** | The 1P+1D slice: a prefill and a decode container on one node, each a full ATOM deployment with W2.3's KV connector, `atomesh launch --pd-disaggregation` between them, a standalone CA; one request end to end (#637) | 400–600 | consumes W2.1, W2.3 |
| **W3.6** | cc-traces harness driving a simulated run end to end | 300–500 | consumes W1.9 |
| **W3.7** | CA observability: timeline log, stall diagnostic, run summary (`01` D3.5) | 300–450 | |

**M1's exit criterion** is #627's stages 1 and 1b, and needs no cost model: cc-traces
through agentx-harness on a simulated TP1 run (W3.6), then through agentx-harness and
atomesh on a simulated 1P+1D run (W3.5), each ended by the finish and both GPU-free.

**The scheduling-parity test opens M2**, per `15` D94 as revised by D95: **does a
fake-model run reach the same scheduling decisions as the real engine at the same
configuration** — first at TP2 / DP2 on one node, where it is cheap,
`-tp 2 --enable-dp-attention`; then at DeepSeek-V4-Pro's 1P+1D in the nightly DP-attention
configuration, expert parallelism off, where it counts? M2's EP8 step repeats both legs
with expert parallelism on: EP2 in its non-degenerate form,
`-tp 2 --enable-dp-attention --enable-expert-parallel` (T84), then EP8. PP2 left the test
with W3.2. ATOM's own `test_dp_load_balance.py`, `test_dp_metadata.py`,
`test_dp_sync_layout.py` and `test_forward_mode.py` already cover the pieces, CPU-only —
all four are in the CPU gate at `3afcb4880`, which was not true when this was written: the
last three sat in a 32-entry exclusion list until they were re-measured and found green.

**Schedule early: a saturated cell on the PDES runtime.** As soon as the vertical slice
runs on the PDES runtime (#472), time one saturated cell. The prior clock design ran
**0.30x** under saturation — slower than the system it simulates. If the speed target fails
structurally, that is worth knowing while the fake model is still the only thing in play.

---

## Waves 4–6 — M2, which is v1

Detailed now, because M2 overlaps M1: calibration and the analytical laws run alongside Waves 1–3 on
their own hardware, and a rough wave cannot be claimed as an issue. Every row that Phase
0b can reshape says so.

### Stage 2 — Qwen3.8-27B TP1 on MI308X, M2's first gate

#627's stage 2 (D95): paired real and simulated cc-traces, graded by `08`. S2.1, the real
side, serves any candidate backend. #627 ruled on 2026-10-10 that stage 2 prices with
`09` D54's step linear law, whose tasks #627 files, and that the step grid stays off the stage-2 path
for now. The rows after S2.1 are the step grid's stage-2 evaluation (`17` D96), off stage 2's
critical path: no stage-2 issue waits on them, their MI308X measurements run only when node
18 is idle from that path, S2.4 builds on the step linear backend #738 delivered (#759), and none
of S2.2 to S2.6 depends on or edits #741's or #742's code. S2.7 builds the toolchain's
steps half that the step grid's rows run on (`13` D80), ahead of the ops half (W4.1); it edits
`validate/`, #741's and #742's code, through #902 and #920, and no stage-2 issue waits on
it.

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **S2.1** | The real side: Qwen3.8-27B TP1 on MI308X, aggregated, driven by cc-traces at `08` D50's stage-2 client counts; three or more spaced runs per client count (`08` D45), with per-request records and per-step tables, each forward's batch and unprofiled seconds (`08` D49 item 1). Filed under #627's stage-2 plan; the step grid reads #743's registered runs read only, or, if they do not exist when it is ready, runs its own under the same rules, named apart from stage 2's (`17` D96) | `scripts/compass/` | 150–300 | GPU queue |
| **S2.2** | The step grid: a grid definition with its margin, and the boundary nodes the planner adds from the target's bounds (per decode rung, each B node's KV limit and `max_model_len - 1` as c nodes and B nodes where shapes cross the rung's limit; prefill on `(B, T, c)` with the token budget, `max_num_seqs` and `max_model_len - max_num_batched_tokens` as nodes), planned by the step grid's planner, beside its backend and run inside `measure steps` (#909) from Phase 0 on the registered case set, then from the shapes of S2.1's step tables, any trace's shapes and never its timings; the cells the planned shapes fall in measured on MI308X per point by `measure steps` while node 18 is idle from the critical path, then topped up in the cells real batches fall in (`17` D98) | `backends/`, calibration | 300–500 | S2.7, S2.1, T27, GPU queue |
| **S2.3** | The step grid's residual, fitted on real-run step times, bench replays or both, of traces 256 to 392, pooled with no source term and reported by source; the bundle from `fit steps --backend step-grid` (`17` D100; #907, #919) | `backends/`, calibration | 250–400 | S2.2 |
| **S2.4** | `StepGridBackend` with its production wiring, built on the step linear backend #738 delivered (#759): lookup, residual, bundle and startup checks, the run record and the `step_grid` run-file key (`17` D99–D104) | `backends/`, wiring | 500–800 | W2.4, S2.3 |
| **S2.5** | `09` D59's validation of the residual: data held out from the fit within traces 256 to 392, three or more candidate families, a second axis (`17` D96) | calibration | 150–250 | S2.3 |
| **S2.6** | Grading the step grid by stage 2's protocol: `08` D44's three results on traces 0 to 255 of S2.1's real runs, no ranking gate; a run with any refusal is a coverage report (`17` D102) | `validate/` | 150–250 | S2.4, S2.5 |
| **S2.7** | The toolchain's steps half (`13` D80): `record`, `trace steps`, `measure steps` on the bench of `07` D39, `measure memory`, `fit steps` and `fit host`, `run` with `--coverage` and what grading reads, the artifact kinds and rows of `07` D41 and D43, and the one entry; its tasks are #901 to #920 but for #909 (S2.2's) and #907 and #919 (S2.3's), and an issue for `measure memory` | `drive/`, `bench/`, `fit/`, `artifacts/` | 2700–3800 | W2.4, GPU queue |

### Wave 4 — DeepSeek-V4-Pro on MI355X

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **W4.1** | The toolchain's ops half (`13` D80): `trace ops` (Phase 1a), `measure ops` with collectives (`07` D40), `fit ops`, and the ops-half settings of `run`, on S2.7's bench and entry | `capture/`, `bench/`, `backends/` | 400–700 | W2.4, W2.5, S2.7 |
| **W4.2** | DeepSeek-V4-Pro capture: a fake or meta implementation for every operator P0b.1 names — shape and dtype rules, no math — plus #501's launch interceptor for Triton, FlyDSL and MORI and its two host-query rules, with no ATOM change; carried by S3c.9 ([#1401](https://github.com/jgong5/ATOM/issues/1401)) | `capture/`, `ir/` | 600–1000 | **P0b.1** |
| **W4.3** | The real side: the nightly P/D configurations of `recipes/mesh/DeepSeek-V4.md` (TP; DP-attention + TBO; TP + MTP3; DP-attention + MTP1) on the MI355X pair, driven by cc-traces, with per-request records and a real-vs-real repeat per cell for the noise floor (`08` D45) | `scripts/compass/` | 200–350 | **P0b.3**, GPU queue |
| **W4.4** | Calibration Phases 1b / 1c on MI355X for DeepSeek-V4-Pro's operator families: FP4 MoE GEMMs, attention with its compressors, the CSA indexer, mHC, the DP group's variable-size gather | `backends/`, calibration | 400–700 | W4.1, W4.2, GPU queue |
| **W4.5** | Fitting and law selection for those families (`09`) | `backends/` | 300–500 | W4.4 |
| **W4.6** | The DP-attention step: the `max` over the DP group's ranks (W3.1) with the gather priced from W4.4, and TBO's overlap | `backends/` | 300–500 | W3.1, W4.5 |
| **W4.7** | MTP: per-position acceptance measured from W4.3's real runs, never declared (`08`), feeding the spec-decode pricing of `14` | `runner/`, `backends/` | 200–350 | W4.3 |
| **W4.8** | Mooncake transfer time from the spec, checked against the transfer timings W4.3 records | `kv/` | 150–250 | W2.3, W4.3 |
| **W4.9** | DeepSeek-V4-Pro 1P+1D at the nightly topology: W3.5's slice with one container per node, TP8 and DP-attention on each side as the cell sets it | wiring | 150–300 | W3.1, W3.5, W4.2 |
| **W4.10** | M2's EP8 step: MORI dispatch and combine on the comm stream, overlapping the other micro-batch's compute under TBO ([#531](https://github.com/jgong5/ATOM/issues/531)); group membership per P0.6; MORI dispatch/combine carried as the group's collective in W4.6's step, in place of the gather, and priced from W4.12 by a routing model (#501) | `ir/`, `backends/` | 350–600 | **P0.6**, **P0b.2**, W4.6, W4.12 |
| **W4.11** | The EP8 real side: W4.3's DP-attention cells re-run with `--enable-expert-parallel` on the MI355X pair, with the same records and repeat, since no nightly cell passes it | `scripts/compass/` | 50–100 | W4.3, GPU queue |
| **W4.12** | MORI dispatch/combine within a node on MI355X, measured and fitted (`09`) for W4.10's routing model | `backends/`, calibration | 150–300 | W4.1, W4.2, GPU queue |
| **W4.13** | The step grid evaluated on DeepSeek-V4-Pro, not graded (`17` D105): a workload-planned grid and fit on the nightly TP cell (TP8, no DP-attention, no MTP, expert parallelism off) on MI355X, prefill node and decode node, compared with W4.3's per-step tables and with the op-level backend where available, reported per `08` D47. Competes with W4.4 and W4.12 for MI355X | calibration | 200–350 | S2.2–S2.6, W4.3, GPU queue |

### Wave 5 — MI455 by roofline

#627's stage 3c, tracked on [#1392](https://github.com/jgong5/ATOM/issues/1392) and ruled
there on 2026-10-17: the analytical projection (`10` D64) at step and op level, for
DeepSeek-V4-Pro on MI455, built so that a new model needs only new extractors, and checked
by `08` D50.2. Every issue carries `e2e-stage3c`. The step-level rows do not wait on
capture: step level is a second projection, independent of capture, as hedge and
cross-check.

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **S3c.1** | Simplify first: the IR package shrunk to what capture needs ([#1393](https://github.com/jgong5/ATOM/issues/1393)), the spec package to its production callers ([#1394](https://github.com/jgong5/ATOM/issues/1394)) | `ir/`, `spec/` | net negative | — |
| **S3c.2** | The spec fields the attainable-peak roofline reads: the bandwidth curve, the kernel floor, a derate per compute format (`05` D25; [#1407](https://github.com/jgong5/ATOM/issues/1407)) | `spec/` | 40–80 | S3c.1 |
| **S3c.3** | The probes on MI308X and MI355X: per-format GEMM rate, the bandwidth curve, the kernel floor, link latency and bandwidth (`05` D26; [#1396](https://github.com/jgong5/ATOM/issues/1396)) | `scripts/compass/` | 200–300 | S3c.2, GPU queue |
| **S3c.4** | Work primitives and the one price, shared by both granularities: GEMM, grouped GEMM, attention, elementwise, the collectives' lower bounds, MoE in expectation (`10` D64, D66; [#1397](https://github.com/jgong5/ATOM/issues/1397)) | `backends/` | 200–300 | S3c.2 |
| **S3c.5** | The MI455 machine spec in the `05` schema: datasheet peaks times MI355X's probed ratios, every field sourced or marked assumed, with its assumed `memory_readings` entry ([#1291](https://github.com/jgong5/ATOM/issues/1291)) | `spec/`, `scripts/compass/` | 120–220 | **P0b.4**, S3c.3, #1226 |
| **S3c.6** | Step level, dense models: module extractors over the meta build and a step roofline summing one roofline per module ([#1398](https://github.com/jgong5/ATOM/issues/1398)) | `run/`, `backends/` | 250–350 | S3c.4 |
| **S3c.7** | Step level, DeepSeek-V4-Pro's modules: attention with its compressors and CSA indexer, mHC, the FP4 MoE with its shared expert, the DP gather and reduce-scatter over `T_moe`, MORI at EP8 ([#1399](https://github.com/jgong5/ATOM/issues/1399)) | `run/` | 250–350 | S3c.6, #1387, #530 |
| **S3c.8** | Op level, capture into the IR on Qwen3.8-27B ([#1400](https://github.com/jgong5/ATOM/issues/1400)) | `capture/` | 400–500 | S3c.1 |
| **S3c.9** | Op level, DeepSeek-V4-Pro capture at the nightly shapes, which is W4.2 ([#1401](https://github.com/jgong5/ATOM/issues/1401)) | `capture/` | 500–600 | S3c.8, #530 |
| **S3c.10** | The extractor registry for aten and dense opaque op kinds ([#1402](https://github.com/jgong5/ATOM/issues/1402)) | `backends/` | 200–300 | S3c.4, S3c.8 |
| **S3c.11** | Extractors for DeepSeek-V4-Pro's opaque op kinds ([#1403](https://github.com/jgong5/ATOM/issues/1403)) | `backends/` | 250–300 | S3c.10, S3c.9 |
| **S3c.12** | `OpGraphBackend` prices an IR graph analytically, composed by the IR's joins and TBO's overlap ([#1404](https://github.com/jgong5/ATOM/issues/1404)) | `backends/`, `run/` | 250–350 | S3c.10, #531 |
| **S3c.13** | The bound gate and the gap reports at both granularities (`08` D50.2 item 1; [#1405](https://github.com/jgong5/ATOM/issues/1405)) | `validate/` | 150–250 | S3c.6, S3c.12 |
| **S3c.14** | The projection: MI455's artifact for the four P/D cells, EP off and EP8, at both granularities, every assumed input listed; and the method's error, Qwen3.8-27B from MI308X to MI355X (`08` D50.2 items 2 and 3; [#1406](https://github.com/jgong5/ATOM/issues/1406)) | `validate/`, `artifacts/` | 150–250 | S3c.13, S3c.5, S3c.7, S3c.11 |
| **S3c.15** | The projection re-priced at ±20% of each ratio carried from MI355X, capped at the datasheet, with the ranking each gives (`10` D69; [#1466](https://github.com/jgong5/ATOM/issues/1466)) | `validate/` | 40–80 | S3c.14 |

**W5.1–W5.5 are retired by the ruling**, so an older citation resolves here: W5.1 is
S3c.5; W5.2's laws are S3c.4 with the extractors of S3c.6, S3c.7, S3c.10 and S3c.11;
W5.3's soundness check is S3c.13, at probed peaks rather than the spec's at derate 1; W5.4
and W5.5 are S3c.14, whose projection carries no implementation efficiency (`10` D68).

If MI455 hardware arrives during M2, W4.3–W4.5 repeat on it (W4.11 and W4.12 too, for the
EP8 step), and S3c.3's probes run on it, so its peaks are probed rather than carried.

### Wave 6 — M2 acceptance

| ID | Task | Module | Effort | Depends on |
|---|---|---|---|---|
| **W6.1** | Paired acceptance over W4.3's cells, expert parallelism off: the separable results of `08` D44, plus the ranking gate — the same best configuration as hardware at each concurrency, and rank correlation across all cells | `validate/` | 250–400 | W4.6–W4.9 |
| **W6.2** | The speed check on a saturated acceptance cell, against the target in `README.md` | `validate/` | ~100 | W6.1 |
| **W6.3** | M2's EP8 step graded the same way: W6.1 over W4.11's cells, with W4.3's cells in the ranking | `validate/` | ~100 | W4.10, W4.11, W6.1 |

---

## After v1

Listed so it is not mistaken for an omission; each is re-planned when v1 lands.

- PP: W3.2, T64, and PP's accuracy.
- EP beyond one node, and the multi-node rank mapping of T85.
- GLM-5.2, Kimi-K3 and other models; MXFP4 beyond DeepSeek-V4-Pro's own experts.
- DCP (decode context parallelism).
- The step grid for Kimi-K3 with DCP: a grid and fit for that target, with DCP's correction as
  an optional recipe (`17` D100).
- Width transfer (T21) as a claim.
- The upstream RFC for ATOM-core edits.

---

## The GPU booking queue

**M2's hardware, by D95.** An RDMA-connected MI355X pair for the paired runs of W4.3 and W4.11, and a
third MI355X node for W4.4 and W4.12, so that calibration does not queue behind the pairs; an MI308X
node for development, the CPU-and-GPU gates, S3c.3's MI308X probes, stage 2's real side
(S2.1), and the step grid's measurement (S2.2) while it is idle from stage 2's critical
path. Until dedicated nodes
exist, the shared pools below serve, with the pre-flight gate as written.

GPU is the scarce resource; almost everything else is CPU-only by design principle 2.
Only calibration Phases 1b/1c/2, `--measure` runs, the real side of pairings, and the
T52 root-cause need one.

**One queue.** A GPU task declares in its issue body, before it is claimable: what it
measures, which width, how long it needs, and which artifact it writes. A quiet window is
never spent deciding what to run in it.

**Wall-clock is quoted only where it was actually measured.** A TP2 engine run is priced
at ~1 h (`12` T21) because that is what engine runs take, and the long calibration sweep
is priced at six hours because it was measured at six hours — not estimated from
throughput. Effort elsewhere is sized in lines of code, not wall-clock, precisely because
most tasks have no such measurement to quote (`agent-team` plugin, `plan.md`).

### The pre-flight gate — four checks, not one

```
  0.  D-state census               -> thousands in D state? node is WEDGED.
  1.  timeout 25 rocminfo          -> hangs?  node is WEDGED. Do not use.
  2.  rocm-smi --showuse           -> compute busy?
  3.  rocm-smi --showmemuse        -> VRAM held?     <- the one that gets missed
```

**Check 3 is not optional.** A node can show 0% utilisation and ~99% VRAM — a loaded,
idle model. A pre-flight testing only utilisation calls that node free, and `non_torch` is
a **device-wide** reading, so a neighbour's allocation is indistinguishable from ours.

**Check 0 states its own scope, because in a container it cannot answer the question.** It
was added by P0.2 on the reasoning that `timeout` cannot kill a probe already in D state.
But a container's PID namespace is private, so the census counts the container, not the
node. Measured 2026-09-20T08:45Z on hjbog-srdc-39, the same second from both sides: the host
had **10840 processes of which 2236 were in D state**, while the `jgong5_vllm` container on
it saw **3171 and 103** — 4.6% of the D-state processes that were actually there, against a
threshold of 20 calibrated on host-wide readings. `docker inspect
-f '{{.HostConfig.PidMode}}'` is empty and no host procfs is bind-mounted, so there is
nothing to look through. `preflight.sh` therefore reads a host procfs when one *is* mounted
(`/host/proc`, `/hostfs/proc`, `/rootfs/proc`), detects the container positively
(`/.dockerenv` or `/proc/1/cgroup`) rather than inferring it from a low count, prints
`scope:` on every run, and when the scope is container-local and under threshold reports
**PARTIAL** — "nothing is wedged inside this container, the node was not examined" — instead
of a clear (principle 6: an unanswered question is not a "no"). Check 1 is what still sees a
host wedge from inside a container, because `rocminfo` goes through the driver.

**Check before *and* after.** Three of the last five prior pilot attempts were lost or
degraded by other tenants; a check that only runs first cannot see a tenant that arrived
mid-run (`08` D49.5).

### Node status at time of writing — a snapshot, not a fact

| node | compute | VRAM | D-state procs | usable |
|---|---|---|---|---|
| 18 | GPU0 100%, rest idle | GPU0 90% | 0 | 7 GPUs |
| 19 | idle | **0%** | 0 | **yes** |
| 20 | idle | **0%** | 0 | **yes** |
| 21 | idle | 99% | 0 | no — VRAM held |
| 22 | idle | 98% | 0 | no — VRAM held |
| 39 | idle | 71% | **2183** | **no — wedged** |

Node 39's 2,183 D-state processes are the ROCm wedge signature: `rocm-smi` answers
normally, which is precisely why the node looks healthy. The durable artifact here is the
pre-flight script, not this table.

---

## Escalation points

The checks below can each reshape the plan; T10 has been run and did not fire. Each ends in a
decision that belongs to the project owner, not to the agent that ran it.

| Trigger | What the escalation carries |
|---|---|
| ~~**T10** fails~~ | ~~adapter cost rises to ~2,000 vendored lines — vendor, fork, or restrict the harness~~ — **did not fire.** P0.3, 2026-09-20: the subclass works and nothing is vendored. The escalation this row prepared for never arose |
| **T5** fails | the op-level backend has no IR at TP>1 — options and their effect on M2 onward |
| ~~**P0b.1** finds an operator that needs more than a shape rule~~ | **ruled by the owner on #501**: two rows needed more, MORI at EP8 and the DP gather with EP off, both sized by the peers' token counts. W4.2 takes the launch interceptor; M2 runs expert parallelism off first, then EP8 with a MORI routing model (W4.10) |
| ~~**P0b.3** finds the nightly runs unusable~~ | **ruled by the owner on #503**: every paired cell is re-run on the dedicated pair, and W4.3 is booked for it |
| **P0b.4** leaves an MI455 input unsourced | the projection ships with that input marked assumed, or waits — the owner's call |
| ~~**T21** fails~~ | ~~calibration does not transfer across width~~ — **off the critical path by D95**: M2 is graded at the nightly widths |
| **T25** fails | the noise floor swamps 10% at high client count — those cells are **ungradeable**, and the acceptance set needs re-scoping |
| ~~**T64** answers "yes"~~ | ~~ATOM microbatches PP~~ — **deferred with W3.2 by D95** |

**Each escalation is prepared, not improvised.** When a check completes, its result
arrives with the options and their costs already worked out, so the decision is one round
trip rather than a fresh analysis under time pressure.

The stop-and-diagnose rule (`agent-team` plugin, `rules.md`) is the general case: any surprise stops and
is diagnosed, and becomes an escalation only when it needs an owner ruling.

Reaching any of these, like the loop's halt above, is an escalation — so it applies
`need human` too (`agent-team` plugin, `rules.md`), for the same reason: the stop should be visible on
GitHub, not only inside an agent's report.

---

## What this plan does not contain

Stated so it is not mistaken for an omission.

- **Dates.** Effort is LOC; wall-clock appears only where it is machine time with a
  measurement behind it (`agent-team` plugin, `plan.md`).
- **Detail past v1.** Deliberate — see *After v1*.
- **Agent prompts.** Task briefs are written to be close to a prompt and generated from at
  launch, because embedded prompts go stale as tasks move.
- **A test plan separate from `08`.** Validation is `08`; this document schedules it.
- **Upstreaming to ATOM's `main`.** Out of scope until the project agrees a milestone is
  ready; the integration branch is the destination until then.

---

## TODO register

This topic's items only. The consolidated register is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T71~~ | ~~Add Wave 4+ detail as Phase 0 and T21 answers arrive~~ — **done by D95**: Waves 4–6 are detailed | — |
| T72 | Decide whether reviewer agents use ATOM's existing `review-pr` skill or a Compass-specific checklist | needs one review cycle to tell |
| T80 | Raise with ATOM's owners: `tests/test_prefix_cache_accuracy.py` has no test function — it is an `argparse` script driving a live server — and `test_kv_connector_scheduler.py` / `test_transfer_engine.py` have been dead since #690. Measured: all three run nothing in **either** gate | not a Compass change; needs the disaggregation and prefix-cache owners |
