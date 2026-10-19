# ATOM Compass — Open Items: TODO Register, Assumptions, Gaps

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**What this is.** Everything across the design topics that is *not settled*, in one
place. Split out of `README.md` so the front page stays a bird's-eye view rather than a
backlog. Nothing here is a decision; every decision lives in its topic's decision log.

**How to read it.** Four sections, in decreasing order of how much rests on them:

1. **Load-bearing assumptions** — hold up large parts of the design; each has a check plan
2. **Missing topics** — design points nobody has written yet, with a recommendation
3. **TODO register** — per topic. A row struck through or marked closed is done or
   retired, and says by what. The register is the rows of section 3 below and nothing
   else: prose elsewhere in this file names T-numbers that belong to other branches. It is
   allocated across parallel task branches, so not contiguous
4. **Cross-cutting issues and pending amendments**

---

## 1. Load-bearing assumptions, and how each gets checked

Five assumptions hold up large parts of the design. **One has been tested** — T10, resolved
by P0.3 on 2026-09-20; the other four have not. Each row names the check, where it runs,
and roughly what it costs — so these are schedulable work in the execution plan rather than
caveats in a document.

| # | Assumption | If false | The check | Cost |
|---|---|---|---|---|
| **T21** | The in-situ calibration transfers across TP width | the recipe's "calibrate at TP1, predict TP2/4/8" collapses and the campaign multiplies by the number of widths | **Doc `07` Phase 1c's one extra run.** Calibrate at TP1, predict a TP2 full-engine run, compare. Already a designed step of the recommended flow — it is step 6 — so the check is not extra work, it is the reason that step exists. | one TP2 engine run, ~1 h GPU |
| **T25** | The real-vs-real noise floor stays narrow under closed-loop replay at high client count | those cells become ungradeable — not failed, *ungradeable*, which is worse because nothing is proven either way | **Doc `08` D45's step 1, run before any simulated comparison.** N≥3 spaced real repeats at the 64- and 256-client cells; report the pairwise spread. If it swamps 10%, say so and re-scope the acceptance cells. | 3 real cc-traces runs per cell, ~3 h GPU |
| **T5** | ATOM's model classes trace cleanly under `FakeTensorMode` at TP>1 | the op-level backend has no IR, and docs `04`, `07` and `09` rest on it | **Trace the 27B at TP2 under the doc `04` D18 mechanism and diff the captured structure against TP1.** Run this first and cheaply. A fake-tensor trace should be GPU- and collective-free, so the known mode hang should not be reachable from it; T5 is the experiment that settles whether that reasoning holds. Needs a non-wedged node (`rocminfo` under `timeout` before starting). **Answered yes**, on two models at both widths, with the collectives recorded rather than substituted away, and the reasoning holds for the collectives that go through a registered custom op but not for the four raw `torch.distributed` call sites -- see `04` D18, *Collectives at TP>1*. The structure diff is there too. | half a day, one node |
| ~~**T10**~~ | ~~`AgenticReplayStrategy` can be subclassed rather than vendored~~ — **resolved 2026-09-20 by P0.3.** Yes, and it is not needed on its own: the strategy is built by the plugin factory, so an out-of-tree subclass displaces it with no upstream edit — but a subclass reaches only four of the nine pacing sites. All nine share one `LoopScheduler` resolved as a module global, so the adapter **rebinds that global** instead (`06` D34). No vendoring. **W1.9's 450–650 total is reopened, not settled** — see `06` D34's component table: one row was costed against the option-A wrapper, and T75 and T76 are new, uncosted scope. Evidence: six executed claims, `agent_scratch/compass_dev/p0_3/spike_t10.py`, zero edits to agentx-harness. | done |
| **T52** | `TorchDispatchMode` instrumentation does not hang ATOM at width | `--measure` runs and any mode-based instrumentation of a REAL execution are unusable. **Probably does not gate Phase 1a tracing** - the hazard is a `__torch_function__` guard on a real device, not a fake-tensor trace. | **Root-cause the known hang** at `atom/spec_decode/dspark_scheduler.py::schedule_prefix_lengths_tensor`. `rocgdb` attach, `info dispatches` per rank, identify which rank diverges. | <1 day, quiet node |

**Ordering.** T10 came first — no hardware, largest swing per hour — and is done. Then
**T5** — a `FakeTensorMode` trace should be GPU-free and collective-free, so it is both the
cheaper experiment and the one that tells us whether T52 gates anything on the critical
path.
**T52** follows, at ordinary priority unless T5 actually hits the hang. T21 and T25 need
the calibration and harness to exist, so they land later — but both are *designed-in
steps*, not add-ons, and neither should slip to the end.

**What each one costs if it fails.** T21 and T25 can each invalidate a whole acceptance
claim. T5 can invalidate the op-level backend whole. T52 invalidates `--measure` and mode-based
instrumentation of a real run — narrower, but a designed path. Finding out late is the
expensive outcome in every case, which is the argument for putting them early rather than
where they naturally fall.

---

## 2. Missing topics

Design topics identified but not yet written up. Listed with a recommendation rather than
silently carried as gaps.

| # | Topic | Why it is missing, and why it matters | Recommendation |
|---|---|---|---|
| ~~M-a~~ | ~~Configuration surface and CLI~~ | Flags were scattered across five topics with no owner and no precedence rule. | **DONE** - topic `13_configuration_surface.md`, D78-D81. |
| ~~M-b~~ | ~~Refusal semantics end to end~~ | Seven documents emitted refusals and none said what one *does to the run*. | **DONE** - `08` D50.1: a step is refused only when some term is still unanswered after method fallback; a refused step costs 0 s and the run continues, and any refusal makes the run a coverage report. |
| ~~M-c~~ | ~~Model loading without a GPU~~ | 关键技术点 1.4.2. Doc `03` covers memory *sizing*; nothing covers how the module tree comes into existence to be traced. ATOM has the pieces — `--load_dummy {empty,zero,xavier}`, `RapidServeModelRunner._init_weight_params_on_meta` — but no doc names the path or says whether weights are read at all. | **DONE** - `02` D10.1: HF-config geometry where only geometry is needed, construction inside `FakeTensorMode` with `--load_dummy empty` where a module tree is. |
| ~~M-d~~ | ~~TP / DP / PP / EP specifics~~ | DP couples *scheduling decisions* across ranks through a per-forward collective that rewrites the batch - so TP's and DP's LP structure is an M1 deliverable, not an accuracy one. | **DONE** - topic `15_parallelism_support.md`, D88-D94; D94 splits the work between M1 (plumbing, no parity test) and M2 (scheduling parity, then accuracy), with PP after v1. |
| ~~M-e~~ | ~~Determinism and reproducibility~~ | Doc `08` **T26** asks for bit-reproducibility as a test, but nothing designs for it. Under a distributed CA, grant ordering is a function of real-time message arrival unless something pins it. Two runs of one configuration disagreeing would undermine every paired comparison. | **DONE** - `01` D3.4: the `(LP, virtual time, event)` sequence is what must reproduce; CA grants tie-break by LP identity. |
| ~~M-f~~ | ~~Speculative decoding / MTP~~ | Acceptance is a *behaviour* Compass cannot compute - the first quantity in the design that is neither derivable nor measurable. | **IN SCOPE** by decision 2026-09-19; topic `14_speculative_decoding.md`, D82-D87. Placed in **M2**: see [`14` *Milestone placement*](14_speculative_decoding.md#milestone-placement). |
| ~~M-g~~ | ~~Simulated-run observability~~ | Doc `01` D3.1's open issue says the CA should own the global timeline log and the stall diagnostic, and that "its output format is part of the acceptance evidence and should be designed, not improvised". Doc `11` covers Prometheus metrics, which is a different thing. Still improvised. | **DONE** - `01` D3.5: timeline log, stall diagnostic, and a run summary written always, an early end included. |

**All seven are now closed.** M-a `13`; M-b `08` D50.1; M-c `02` D10.1; M-d `15`; M-e `01` D3.4;
M-f `14`; M-g `01` D3.5.

---

## 3. TODO register

### Topic 04 — model capture and cost IR

| # | Item |
|---|---|
| T1 | Inductor fusion correction (~4.8% of a decode step) |
| T2 | Enumerate the structure set for Qwen3.8-27B |
| T3 | Build the per-leaf parameter-extractor table (~20 entries) |
| T4 | Establish scratch constants per leaf for the 27B, once the liveness walk replaces the measured activation coefficient (`03` D16) |
| T5 | Verify ATOM's model classes trace cleanly under FakeTensorMode at TP>1 |
| T6 | Validate that `Repeat` grouping reproduces the flat prices term by term |
| T7 | Validate `Par` reconstruction from stream ids |
| T8 | Decide whether the step-level backend is fitted independently or derived from the op-level one |
| T9 | State a row-ordering treatment for decode attention |
| **T51** | Enumerate the layer-pattern shapes for Qwen3.8-27B and DeepSeek-V4-Pro; confirm the nested-`Repeat` detector reaches the hierarchical form on both |
| **T52** | Root-cause the `DeviceContext` `__torch_function__` guard 8-rank hang at `dspark_scheduler.py::schedule_prefix_lengths_tensor` — gates T5 |
| ~~**T81**~~ | ~~Make the D18 capture *symbolic* on ATOM's real forward, or record that it cannot be.~~ **Closed 2026-10-18 by [#1489](https://github.com/jgong5/ATOM/issues/1489#issuecomment-6249950252)'s capture ruling**: capture is concrete, one graph per decode width and prefill (T, B) (`04` D18), since symbolic decode needed five graphs for B = 2 to 512 and symbolic prefill T specialises in the attention builder. The record follows. **Done for a decode step, at TP1 and TP2, on the published 27B; not attempted for prefill.** A step whose *width* is a free symbol traces through ATOM's own `ModelRunner`, model classes and staging with `ShapeEnv.replacements` **empty** and no symbol solved anywhere. **The census, by operator family** — a total is what made two earlier attempts look like progress while every symbol sat on a view. At TP1, **5,016 of 12,544** shape entries are not plain integers: **514 on 257 `aiter.gemm_a16w16`**, **304 on the 64 attention calls** (`unified_attention_with_output_base`, `linear_attention_with_output_base`) and **802 on 273 normalisation operators** (`_fused_qk_rmsnorm_group_quant_kernel`, `mean.dim`, `pow`, `rsqrt`). At TP2, **5,297 of 13,107**, the same three families plus **266 on the 133 collectives**. The control is the same file with the width left an integer: 0 of 12,544 and 0 of 13,107. Both passes record **the same operators** — 2,521 at TP1 and 2,662 at TP2 — differing in one call that materialises a symbol where it used to lift a constant, so what changed is what the shape entries say and not which operators ran. The breakdown is only worth the bucket that fails: an operator matching no stated family lands in `unclassified` and the test refuses it, and the classifier matches **whole operator names** — it matched by prefix until 2026-09-22, which silently folded `addmm` and `addbmm` into elementwise and `slice_scatter` into views, so the bucket could not have fired for the cases it exists for. No operator on this step is one of those, so the rows above are unchanged by the repair. **The mechanism: the three sites are symptoms of one line of torch.** `SymInt.__index__` and `SymInt.__int__` are `guard_int`, so every host-side use of the width asks for a number, gets the hint, and records the ask as `Eq(s, hint)`. **16 ATOM lines** convert the width during one traced forward — 20 conversions, all host fills and slices in `prepare_decode` (7 lines), `prepare_inputs` (3), `prepare_input_ids` (2), `prepare_sample`, `_mrope_cpu_view` (2) and `gdn_attn` — which is why closing sites one at a time relocates rather than converges. The capture keeps the conversion, to the hint, which is the count ATOM computed, and replaces the *recording* with its own log of every conversion and the line it happened on. **That log is asserted as a multiset against a stated set** (`EXPECTED_HOST_RESOLUTIONS`), identical at TP1 and TP2 and at both step widths, so a conversion at a seventeenth line fails rather than joining the log unremarked; `__bool__` is untouched, so a branch on a width still guards. **One production line changed**, and only one: `_rows` returns `t.shape[0]` rather than `int(t.shape[0])`, which is the identity wherever the size is real. That is site three, and the only one of the three not reachable from a capture-time substitution. **"A symbolic `CpuGpuBuffer`" is not required, measured.** Site two's `-> 16384` is a buffer *capacity* (`max_model_len // block_size`) being solved against the CPU side's constant, and it exists only because that probe symbolises every dimension of the staged device tensor. Capacities are engine configuration, not a function of the step. With only the width symbolic the two slices carry the same symbol and `copy_to_gpu` dispatches unchanged; building the CPU side fake instead costs **38 `prim.device` calls ATOM never makes** and leaves **every family's non-numeric shape-entry count unchanged** — 5,016 either way, family for family — while the *total* entries rise **12,544 → 12,585**, because those 38 calls carry entries of their own. That ablation's code is **not in the tree**: it is a fake, non-numpy-backed CPU side written into the capture's staged allocators, measured, and removed once it had answered, so the row is a measurement whose artifact has to be rebuilt to re-check. **Site three is also not what it was.** With `_rows` no longer converting, the same probe still reaches `forward_context.py::ForwardMode.assert_shape_contract` — another file and a later phase of the step, reached from the `run_model` call in `model_runner.py::ModelRunner.forward` with `prepare_inputs` already returned, not fourteen lines after the `slot_mapping` fill in `aiter_attention.py::AiterAttentionMetadataBuilder.prepare_decode` — because that probe hands the caller a bound that is a *different symbol* from the staged buffer's rows, and ATOM's contract is that the two are one number. Equating them is the contract working. It does not fire when the whole step is one symbol, which is why it is one: a decode step has one query row per sequence, so its token count and its sequence count are the same number for a structural reason. **The symbol is free, shown and not assumed**: the step is traced at two widths, 2 and 8, at **both TP1 and TP2**, and the inventories are identical once the symbol's name is set aside — same operators, same 12,544 entries, same 5,016 non-numeric, same digest, same concrete dimensions by value. That is also what says every dimension that stayed a number is not a width in disguise; there are 23 distinct concrete values at TP1 and every one is the model's or the engine's. Three guards at TP1 and two at TP2, all staging capacities read against the width, none an equality — **and the guard set is hint-dependent**: at every step width but 2 a fourth guard `<axis> + 1 > <hint>` appears, at both TP1 and TP2. It is not a specialisation but a `__bool__` comparison on a live symbol, so it is the positive evidence that `__bool__` was left alive; what it costs is that the applicability carries a *lower* bound at any hint but 2, and that the two-width comparison has one artifact that is not identical between widths. **What the symbol does not survive**: `ScheduledBatch.__init__` compares the staged token array's length against the count, which solves the width by comparison rather than conversion, so the capture builds the batch at the concrete width and rebinds its four count fields afterwards. The symbol enters ATOM's staging, not ATOM's batch constructor. **Still open**: prefill and chunked prefill, speculative decode, any structure where the token count is not the sequence count, and the constructor above — this closes the decode step only. Pinned by `tests/compass/test_capture_real_model.py`, whose four passes are the concrete control, the two probes that record the three sites, and the step-symbol pass that specialises nowhere; the earlier assertions were re-pointed rather than removed. Gates T5 alongside T52. |

### Topic 06 — workload harness contract

| # | Item |
|---|---|
| ~~T10~~ | ~~Verify `AgenticReplayStrategy` can be subclassed rather than vendored~~ — **done**: yes, but the adapter rebinds the shared `LoopScheduler` global instead, which covers all nine pacing sites |
| T11 | Build the per-tokenizer vetted filler-token set |
| T12 | Chase the 32 `asyncio.wait_for` sites under virtual time |
| ~~T13~~ | ~~Decide the simulated KV connector's completion semantic~~ — **done**, `01` D6: Mooncake, one simulated write event per request |
| T14 | Build the client-count matrix given only 144 fan-out-capable sessions |
| ~~T15~~ | ~~Warmup handling in the harness contract~~ — **done**: warmup requests are ordinary requests; the rule is an exclusion window agreed by request id |
| **T54** | Detect warmth that *recurs* mid-run (a new shape reaching autotune at minute 10) under `step_linear_fit.WARM_FACTOR` times its price: the fit and `validate.cost` name a vector's leading slow steps and any step above that factor, not a mid-run step below it |

### Topic 07 — calibration toolchain

| # | Item |
|---|---|
| ~~T16~~ | ~~Calibrate the campaign planner's GPU-time estimates~~ — retired 2026-10-13 on #851: no command plans the whole campaign (`07` D37) |
| ~~T17~~ | ~~Draw the boundary of the standalone `ModelRunner` bench~~ — drawn 2026-10-13 on #851: `EngineCore` without its scheduler (`07` D39) |
| T18 | Verify a batch replayed by rule reproduces the recorded step's forward context faithfully, MoE routing included |
| T19 | Decide the artifact store's physical form |
| T20 | Stated node + standalone benchmark for invisible collectives |
| T21 | Establish whether Phase 1c transfers across width (the TP2 test run) |
| ~~T22~~ | ~~Analytical laws as their own design topic~~ — **done**, now `10` |
| T100 | Compare `measure memory`'s readings under ATOM's `--load_dummy` with a real load's on Qwen3.8-27B TP1 on MI308X; until they match, `measure memory` refuses `--load_dummy` (`07` D38) |
| **T55** | Decide the treatment of `c10d::broadcast_`, which takes a `ProcessGroup` no artifact can hold — working answer since #815's ruling (2026-10-11): its device time is in the device-busy label, its launch in `W_launch` (`07` D40) |
| T86 | **aiter is a second executed source root, and no artifact key names it.** Opened 2026-09-21 by P0.6. D41 gives every artifact a provenance stanza naming *the* executed source root, and keys `price_list` on `(model, parallel layout, machine-spec digest, source-root digest)` — one executed root, ATOM's; the machine-spec digest covers the aiter commit the spec pins, not the checkout that ran. But the EP group is constructed entirely in aiter (`15` D92), the MoE kernels and every collective are aiter's, and `aiter` is installed in the container's writable layer at `/app/aiter-test`, outside `/workspace` and discarded by `teardown.sh`. D43's matrix already invalidates `price_list` and `memory_readings` on an AITER bump — but a fingerprint can only refuse on a version something recorded, and nothing records one. It is **not** a recoverability problem: that checkout is a clean `https://github.com/ROCm/aiter.git` at a commit on `origin/main`, so the hash resolves. It is a provenance gap, and the machinery already exists — `scripts/compass/gate_gpu.sh` resolves aiter's checkout with `atom/compass/artifacts/provenance.py::module_root("aiter")`, which does not import aiter, and takes `git describe --tags --always --dirty`, comparing it against `BASE_AITER`. W2.4 adopts the same call. **Two aiter versions are in circulation on this project right now**: `v0.1.20-103-g23f83724f` in `jgong5_vllm` and `v0.1.21.dev0-49-gf4e7c7509` on node 18, so this is an observed divergence rather than a hypothetical one. The decision half — whether an aiter bump invalidates an EP artifact or only warns — is the owner's, and it is cheap to take before the first artifact exists rather than after. |
| ~~T87~~ | ~~**`07`'s price-list table states the MoE all-to-all's block-count cap as one number, and it is two.**~~ **Closed 2026-10-18 by [#1408](https://github.com/jgong5/ATOM/issues/1408)**: `04` D19 states both caps by symbol, `07` D40's table both values, `15` D92 links to `04`, and the `exclusive` join is gone. Opened 2026-09-21 by P0.6. `_get_dispatch_config` returns `min(128, CU)` blocks at 16 warps for **prefill** and `min(64, CU)` at 4 warps for **decode** (`atom/model_ops/fused_moe/mori_prepare_finalize.py`), so on node 18's 80-CU MI308X the cap is 80 for prefill and 64 for decode, not one figure. aiter's own config-time default is a hard-coded 80 for IntraNode and 32 plus 16 RDMA for InterNodeV1 (`aiter/dist/device_communicators/all2all.py::MoriAll2AllManager._make_all2all_kwargs`), overridden per call. The cell that hides the split is the `exclusive` row of `07`'s calibration table; `04`'s `exclusive` join-policy text has the same single-number phrasing. Reporting one number for two is the aggregate-without-decomposition failure principle 7 names. |

### Topic 08 — validation protocol

| # | Item |
|---|---|
| T23 | Choose the family-2 distance function |
| T24 | Define "structural event" for family 3 beyond prefill streaks |
| T25 | Measure the real-vs-real noise floor under closed-loop replay at high client count |
| T26 | Assert simulator bit-reproducibility as a test — mechanism now `01` D3.4; this item is the CI wiring |
| T27 | Decide the construction of the cells the corpus cannot fill without reusing sessions: 256 clients with fan-out in every root, and the recipe's largest DPA concurrency (`08` D50) |
| T101 | Rebuild grading's sensitivity control when a need appears: a metric that does not move under a known perturbation cannot grade a cost model, and grading admits a metric on stability alone until then. Deferred 2026-10-16 by the owner's [ruling on #1025](https://github.com/jgong5/ATOM/issues/1025#issuecomment-6205795319); `08` D45 keeps what the ×0.97 control measured |
| ~~T28~~ | ~~Establish whether ranking/regret becomes an explicit acceptance gate~~ - **DONE**: `08` D50, M2's ranking gate |
| ~~T48~~ | ~~What a refusal does to a run~~ - **DONE**: `08` D50.1, a refused step (some term unanswered after method fallback) costs 0 s and the run continues as a coverage report |

### Topic 09 — fitting and law selection

| # | Item |
|---|---|
| T29 | Choose the hull implementation (convex hull vs k-NN threshold) and its threshold |
| T30 | Enumerate candidate laws per leaf family, with held-out validation shapes |
| T31 | Test raggedness, cached fraction and chunk-position for treatment status |
| T32 | Decide whether the step-level backend is derived from the op-level one or fitted independently |
| T33 | Establish `warmup_seconds` for Qwen3.8-27B under the current stack |

### Topic 10 — analytical laws

| # | Item |
|---|---|
| ~~T34~~ | ~~Test whether the activation coefficient is derivable from geometry~~ — closed 2026-10-16 by [#965](https://github.com/jgong5/ATOM/issues/965): it is not, within 25% ([#705](https://github.com/jgong5/ATOM/issues/705)); a configuration's `memory_readings` entry is its source |
| T35 | Derive the work of the opaque leaves: op-kind extractors at op level ([#1402](https://github.com/jgong5/ATOM/issues/1402), [#1403](https://github.com/jgong5/ATOM/issues/1403)), module extractors at step level ([#1398](https://github.com/jgong5/ATOM/issues/1398), [#1399](https://github.com/jgong5/ATOM/issues/1399)) |
| ~~T36~~ | ~~Name the collective algorithm per code path in the machine spec schema~~ — withdrawn 2026-10-17 by [#1392](https://github.com/jgong5/ATOM/issues/1392): collectives take an algorithm-free lower bound (`10` D66) |
| ~~T37~~ | ~~Decide whether the host floor is derivable or stays a per-model constant~~ — ruled 2026-10-11 on #815: measured host laws, keyed by host CPU, `atom` digest, model and engine config, with no transfer (`09` D54, `10` D64) |
| T38 | Build the measured-over-analytical report per op kind and per phase, beside the bound gate ([#1405](https://github.com/jgong5/ATOM/issues/1405)) |
| ~~T39~~ | ~~Establish a dispatch-band table from geometry where no measured bands exist~~ — withdrawn 2026-10-17 with `10` D65's band ([#1392](https://github.com/jgong5/ATOM/issues/1392)) |
| **T56** | Quote the method's error from MI308X to MI355X with the MI455 projection, so a user of an unmeasured device sees a range rather than a promise ([#1406](https://github.com/jgong5/ATOM/issues/1406)) |

### Topic 11 — metrics support

| # | Item |
|---|---|
| T40 | Confirm the new histograms are **classic**, not native |
| T41 | Audit every `observe()` site; extend the AST test to observation arguments |
| T42 | Measure `collect_metrics()` per-step cost on the real side; decide decimation |
| T43 | Verify the backfill end to end — one block, loaded, visible in Grafana |
| T44 | Sanity-check histogram bucket ranges against simulated latencies |
| T45 | Tag ATOM's existing metrics with their D77 class |
| T46 | Decide the DP-aggregation rule per class; refuse summaries there |

### Topic 13 — configuration surface

| # | Item |
|---|---|
| ~~T57~~ | ~~Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them~~ — retired 2026-10-14 on #851: there is no environment level (`13` D79) |
| T58 | Decide whether a per-leaf source override is worth the reproducibility cost |
| ~~T96~~ | ~~Reconcile `13` D79 and D80 with the run file the code selects a backend by (`--compass-run`, `atom/compass/run/__init__.py::runner`): `13` rejects a Compass config file, the code reads one (`17` D103)~~ — settled 2026-10-13 on #851: `compass run` takes flags only, and the run file is its internal transport (`13` D79, D80) |
| ~~T99~~ | ~~A simulated server outlives its client (`13` D80): `01` D3 ends a run when no essential work is left, and must end a client run there instead, the clock then waiting for the next client~~ — settled 2026-10-14 on #966: `01` D3, Client runs |

### Topic 14 — speculative decoding and MTP

| # | Item |
|---|---|
| T59 | Capture `ATOM_ENABLE_RELAXED_MTP` in the run fingerprint - it changes acceptance *semantics* (`RELAXED_TOP_N` 1 to 10, `RELAXED_DELTA` 0 to 0.6) and is invisible to every artifact key today |
| T60 | Test whether a draft forward's cost is linear in `K` - serial MTP should be, a real draft stack need not be |
| T61 | Decide how chunked prefill and drafting interact, and what structure that produces |
| T62 | Assert the host acceptance draw and the Triton kernel agree: same stated rates, same seed, same accepted-count distribution over a few thousand draws |
| T63 | Add ATOM flag `--spec-decode-acceptance-rates` (list) - contract 2 has no transport today; the CLI exposes only the two scalars |

### Topic 16 — execution plan

| # | Item |
|---|---|
| ~~T71~~ | ~~Add Wave 4+ detail as Phase 0 and T21 answers arrive~~ — **done by `16` D95**: Waves 4–6 are detailed through M2 |
| T72 | Decide whether reviewer agents use ATOM's `review-pr` skill or a Compass-specific checklist |
| T73 | **Declare the plugin entry point with a dotted module path, and defer the rebind from that package's `__init__.py` with a `sys.meta_path` hook.** Successor to T10, opened 2026-09-20; re-scoped 2026-09-20 after review refuted its premise by execution, and re-measured 2026-09-21. `aiperf/plugin/plugins.py::_PluginRegistry.discover_plugins` calls `importlib.util.find_spec` on the entry-point value, and on a **dotted** value that imports the parent package — so `compass_harness.plugin:plugins.yaml` executes `compass_harness/__init__.py` inside `discover_plugins()`, which runs at import of `aiperf.plugin.plugins` and therefore before any `PhaseRunner` is constructed, with the manifest still resolving and zero edits to agentx-harness. The rebind cannot run *inline* there: the bootstrap fires at the module-level `_registry = _PluginRegistry()` in `aiperf/plugin/plugins.py`, inside that module's own body, so any `import aiperf.…` re-enters `aiperf/plugin/enums.py`, whose module-level `PluginType` calls `plugins.list_categories()`, and raises `AttributeError: partially initialized module`. The bootstrap therefore installs a stdlib-only import hook that rebinds `LoopScheduler` on `aiperf.timing.phase.runner` after that module executes. Executed against `56a0cf70f` over five entry-point/bootstrap combinations, 7/7 in the working shape; decomposed in `06` D34. **A packaging decision plus ~20 lines, not a precondition of the seam** — the earlier "no such bootstrap exists today" was an asserted negative and is false. The tripwire ships regardless, because a bootstrap that silently fails to run has no other detector — but it covers only that half: an inline rebind de-registers the plugin the tripwire lives in, so W1.9 also needs a positive check that the Compass plugin registered at all (`06` D34, `16` W1.9) |
| T74 | **Whether the 32 `asyncio.wait_for(..., timeout=T)` sites need virtual time.** Successor to T10, opened 2026-09-20. They bypass `LoopScheduler`, so the rebind does not reach them; under virtual time they may fire instantly. P0.3 did not examine them (`06` D34, second risk) |
| T75 | **Decide the two `loop.call_later` idle-cap timers.** Opened 2026-09-20. `replay_dependencies.py::ReplayBarrierCoordinator._arm_root_idle_watchdog` and `agentic_replay.py::AgenticReplayStrategy._arm_system_idle_watchdog` arm real-clock timers the `LoopScheduler` rebind cannot reach; upstream's own docstring says the second deliberately lives outside the scheduler. Either override the two `_arm_*_idle_watchdog` methods from the Compass subclass, or state the idle-cap feature unsupported under virtual time and assert both caps are `None`. Related: `agentic_replay.py::AgenticReplayStrategy.enforce_system_idle_cap` derives the virtual-time skip from `time.monotonic()` (`06` D34) |
| T76 | **Reconcile `seamless=True`, which keeps two `PhaseRunner`s and two schedulers live at once.** Opened 2026-09-20. `phase_orchestrator.py::PhaseOrchestrator._execute_phases` builds one runner per phase and tracks `_active_runners`. W1.9 either reconciles two concurrent schedulers against one virtual clock or asserts `seamless=False` (`06` D34) |
| T80 | Raise with ATOM's owners: `tests/test_prefix_cache_accuracy.py` has no test function — it is an `argparse` script driving a live server — and `test_kv_connector_scheduler.py` / `test_transfer_engine.py` have been dead since #690. Measured: all three run nothing in **either** gate |

### Topic 17 — step grid backend

| # | Item |
|---|---|
| ~~T93~~ | ~~Measure in ATOM whether the scheduler's CPU segments add serially to the forward or overlap it; the step grid's `fitted` CPU mode waits on it~~ — ruled 2026-10-11 on #815: serial on the host timeline, inside `01` D4's `E` (`17` D104) |
| T94 | Record the kernels each step grid point ran, and treat a cell whose corners ran different kernels as crossing a dispatch band (`09` D60): refused or split (`17` D99) |
| ~~T95~~ | ~~A later trial, not this design: extrapolate the step grid's out-of-cell steps as #627's ruling 3 has the step linear law do, by one algorithm only (extending the nearest complete cell's interpolation), its terms estimated and unvalidated with the algorithm and the distance named in `source`, reported apart, which is not a refusal. It includes extrapolation within rounding distance: a step outside every complete cell, and every T98 simplex, only by integer or block rounding is answered by extrapolation that far, and a farther one declined. It would answer out-of-cell steps, which the step grid now declines; decided after the step grid's stage-2 evaluation shows how often steps fall outside complete cells (`17` D102)~~ — ruled 2026-10-18 on #1464 and #1465: the step grid extrapolates past its residual's hull and a decode cell past the KV pool, and other out-of-cell steps fall to the step roofline (`17` D99, D100, D102) |
| T98 | Interpolate the step grid's steps whose cell has a corner past a limit, and its prefill steps cached past the top c node, each in the simplex of measured points holding it, those points measured where the cell's edges cross the feasible boundary. Taken up when the pre-run coverage check finds steps the boundary nodes miss, the prefill steps past the top c node counted on a run with prefix caching; #830's counts find none on #740's records, which hold neither kind. Until then a decode one extrapolates past the KV pool and a prefill one falls to the step roofline (`17` D98, D99, D102) |

### Topics 02, 01 — gaps now closed

| # | Item |
|---|---|
| **T77** — closed 2026-09-20 by P0.1 | Re-measure the GPU-gate baseline and record it in full. As stated it was unverifiable: `gate_gpu.sh` judged the superset against `BASE_PASSED=4730 / BASE_FAILED=5` measured by P0.2 at `83daf636d`, with `BASE_TORCH=UNRECORDED`, `BASE_ROCM=UNRECORDED`, no AITER version, a toolchain-drift warning guarded on `BASE_TORCH != UNRECORDED` so the one check that would catch drift could not fire, and — sharper — no failing node-ids on file, so "5 failed" could not be checked against "the *same* 5 failed". **Closed by** a re-measurement on node 18 in `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20, at `fe9ea043c`: **4779 passed / 5 failed**, 0 errors, 105 skipped, 3 xfailed, 72.6 s, two runs with byte-identical failing sets, under torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509**. All five node-ids are written out verbatim in `scripts/compass/gpu_gate_known_failures.txt` and compared by name, and the drift guard now fires because every version field is recorded. The five are pre-existing and unrelated to Compass, and are **four ULP comparisons plus one bitwise check**, not five ULP failures: four `allclose` cases in `tests/test_fused_compress_ragged.py` off by one bf16 ULP (`max\|diff\| = 0.001953125`, exactly 2^-9, against `atol=rtol=1e-3`), plus `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise`, a `torch.equal` with no tolerance at all. What remains open is not T77: the baseline is a *pass count*, so it moves whenever `tests/compass/` grows, and `gate_gpu.sh` handles that by re-deriving the expectation per tree (`4779 + (this tree's tests/compass count - 49)`) rather than by re-measuring. |
| T68 | Enumerate buffer allocations in the two target models and confirm none escapes `FakeTensorMode` - `--load_dummy` and the meta wrapper act on parameters only |
| T69 | Whether a quantized checkpoint's geometry is derivable without reading it; header derivation was 3.3% low at TP=4 on the 27B |
| T70 | Estimate the timeline log's volume at PP degree > 1 - grants scale with stages and microsecond lookahead, so it is largest exactly where it is most wanted |

### Topic 15 — parallelism support

| # | Item |
|---|---|
| T64 | Establish whether ATOM microbatches PP - changes the LP event rate and the bubble model |
| ~~T65~~ | ~~Establish EP's group membership per supported configuration; if EP spans DP, the LP collapse does not hold~~ — **answered 2026-09-21 by P0.6**, stated in `15` D92. The EP group is built in **aiter**, not in ATOM (`aiter/dist/parallel_state.py::initialize_model_parallel`; no ATOM file assigns `_EP`), and is `dp × pcp × tp` within one PP stage — so EP does span DP wherever DP exists, and the caveat's second cross-DP barrier (MoRI dispatch/combine) does appear. **The LP collapse survives anyway**: the group is exactly one PP stage's ranks, and `pp > 1` with `dp > 1` is refused at `engine_core_mgr.py::CoreManager.__init__`, so D93's formula is unchanged and only D92 Q1's stated *reason* was wrong. Three findings were carried out as successors — **T83**, **T84**, **T85** — plus **T86** and **T87** under topic 07. Out of scope and left untouched: PD disaggregation crossed with EP, and what `MORI_SHMEM_MODE=ISOLATION` does to it (`01` D6 already carries a "confirm" on that variable). |
| T83 | **The EP group's size and `moe_parallel_config.ep_size` are two different numbers, and they disagree in two configurations.** Opened 2026-09-21 by P0.6. The group is `dp × pcp × tp` (aiter); the config number is `ep_size = tp_size` (`moe.py::FusedMoEParallelConfig.make`), flattened across DP only when `enable_dp_attention or moe_ep_flatten_tp_across_dp` (`moe.py::FusedMoEParallelConfig.make`) and folded with PCP only under `ATOM_PCP_MOE_MERGE` (`moe.py::FusedMoEParallelConfig.make`). At **`-tp 4 -dp 2 --enable-expert-parallel` without DP-attention** — the command the *MoE throughput scaling* row of `docs/distributed_guide.md`'s *Quick Reference* advertises, and which no recipe uses — the group is 8 while the config number is 4, so MoRI v1 is built over 8 ranks (`num_ep_ranks`, `moe.py::FusedMoEMethodBase._maybe_make_prepare_finalize`) with `num_local_experts` sized for 4 (`moe.py::FusedMoEMethodBase._maybe_make_prepare_finalize`), and its destination rule `expert_id // num_experts_per_rank` then only ever addresses ranks 0-3. MoRI v2 takes both from the group (`mori_v2_prepare_finalize.py::make_mori_v2_prepare_finalize`), so the two transports disagree exactly here. A second instance: `local_ep_size = data_parallel_size_local * tp_size_` (`moe.py::FusedMoEParallelConfig.make`), which reaches MoRI as `gpu_per_node`, omits PCP while the group includes it — unreachable today only because `use_all2all_kernels` needs `dp_size > 1` and PCP with DP-attention is refused (`llm_engine.py::LLMEngine.__init__`), but PCP with plain DP and EP was not found refused anywhere. **Settled by** one 8-GPU startup logging the two values, or by an owner statement that the combination is unsupported — in which case the fix is a refusal at `FusedMoEParallelConfig.make` and a correction to that row. Compass's own consequence either way: carry both numbers and assert they agree, because ATOM does not. |
| ~~T84~~ | **Answered by `16` D95: EP2 means `-tp 2 --enable-dp-attention --enable-expert-parallel`, the form in which the all-to-all runs; `15` D94's line follows in its own PR.** ~~**`15` D94's M1 EP2 leg is degenerate as written.** Opened 2026-09-21 by P0.6. At `dp_size == 1` ATOM builds no all-to-all: `moe.py::FusedMoEParallelConfig.use_all2all_kernels` requires `dp_size > 1` (with `use_ep`, `dp_logical_ratio == 1` and an installed `mori`, found by `find_spec`), it is the only gate on `MoriPrepareAndFinalize` (`moe.py::FusedMoEMethodBase.maybe_make_prepare_finalize`), and without it the layer falls through to `fused_moe(..., expert_mask=...)` (`moe.py::UnquantizedFusedMoEMethod.apply`). So `-tp 2 --enable-expert-parallel` would pass a scheduling-agreement test while exercising nothing EP-specific. The only non-degenerate EP2 is `-tp 2 --enable-dp-attention --enable-expert-parallel`, which `CoreManager` rewrites to `dp=2, tp=1` (`engine_core_mgr.py::CoreManager.__init__`). An owner decision plus one line in D94.~~ |
| T85 | **Multi-node EP rank-to-node mapping is assumed, not verified.** Opened 2026-09-21 by P0.6. MoRI infers node identity as `ep_rank // gpu_per_node`, which is correct only if consecutive EP ranks are physically consecutive GPUs on a node. At `pp == 1` the group is `[0..world-1]` in order, and `ParallelConfig` enforces that a node owns a contiguous DP slice (`config.py::ParallelConfig.__post_init__`), so it holds **on paper**; nothing here measured more than one node. **Settled by** a 2-node DP+EP run logging `all2all_manager.internode` and each rank's EP group. Needs two nodes; not on P0's path. **After v1 by `16` D95**, which takes EP beyond one node out of scope. |
| T66 | Measure whether the Class-C runtime constants move with PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs |
| T78 | **`atom/models/qwen3_5.py::Qwen3_5Model` and `atom/models/glm4_moe.py::Glm4MoeModel` block PP at compilation level >= 2.** Opened 2026-09-20 by P0.5. Both declare `"intermediate_tensors": 0` in `dynamic_arg_dims`. `atom/utils/decorators.py::_support_torch_compile.__call__` raises `ValueError("Unsupported dynamic dimensions ...")` for any non-Tensor argument, and that raise sits **before** the decorator's own `IntermediateTensors` handling, which marks the token dim of each contained tensor. A PP non-first stage receives its activations inside an `IntermediateTensors` container, so the declaration can never be satisfied. Levels 0 and 1 escape it: `decorators.py::_support_torch_compile.__init__` sets `do_not_compile` for `NO_COMPILATION` and `DYNAMO_AS_IS` - 0 and 1 per `atom/config.py::CompilationLevel` - and `decorators.py::_support_torch_compile.__call__` returns before the raise is reached. Observed at the default level 3: `-pp 2 --enforce-eager` on Qwen3.8-27B raised `ValueError: Unsupported dynamic dimensions [0] for argument intermediate_tensors with type <class 'atom.models.utils.IntermediateTensors'>` (node 18, container `xiaobizh_n18`, `/workspace/p05-t64/pp2_27b.log:126`, 2026-09-20); the same invocation with `--level 0` reached KV sizing instead. **Two instances, one class:** `grep -rn '"intermediate_tensors": 0' atom/models/` returns exactly these two files, so an upstream fix scoped to Qwen3.8-27B alone leaves `glm4_moe.py` broken under PP. Other PP-capable models use a bare `@support_torch_compile` and reach the working path. **On no v1 path**: PP comes after v1 (`16` D95), and the PP tests that run today use a fake model - `tests/test_pp.py` mocks `aiter.dist.parallel_state` (`tests/test_pp.py::test_make_layers_places_real_and_missing`) and builds a `MockConfig` (`tests/test_pp.py::_pp_config`), loads no model file, and runs in the CPU gate. It blocks any **real-model** PP measurement, so it is a prerequisite for PP cost accuracy and for T66. Upstream ATOM fix, not Compass; owner unassigned. |
| T79 | **`atom/model_ops/attentions/gdn_attn.py::GDNAttentionMetadataBuilder.sub_pool_specs` mixes a PP-local layer count with a global one.** Opened 2026-09-20 by P0.5. `total = runner._get_total_num_layers()` returns the **PP-local** slice under PP>1 (`atom/model_engine/model_runner.py::ModelRunner._get_total_num_layers`, via `get_pp_indices`), but `num_draft = total - hf_config.num_hidden_layers` subtracts the **global** count, and `runner.num_full_attn` (`gdn_attn.py::GDNStateMixin._init_gdn_state`) is global as well. Measured on Qwen3.8-27B (64 layers, `full_attention_interval` 4, `num_full_attn` 16) on node 18, container `xiaobizh_n18`, 2026-09-20, with `python -m atom.examples.simple_inference --model <Qwen3.8-27B snapshot> -pp {1,2} --enforce-eager --level 0 --max-num-batched-tokens 512`: PP1 gives `num_draft=0`, `n_full=16`, logs `sub-pool kv: entries=80746, entry_bytes=1056768` and exits `EXIT=0` (`/workspace/p05-t64/ctrl_pp1.log:43` and `:410`); PP2 gives `num_draft=-32`, `n_full=-16`, logs `entries=0, entry_bytes=-1056768, num_kvcache_blocks=0` and exits `EXIT=1` (`/workspace/p05-t64/pp2_27b_l0.log:77-80` and `:160`). An exact sign flip at an identical magnitude, so the mechanism is determined rather than inferred. The two invocations also differ in the visible device set (1 vs 1,2) and in `--max-tokens` (8 vs 64); neither enters the `entry_bytes` computation, which is fixed at startup by the layer counts above. On no v1 path, as T78: the fake-model path never reaches this code, and real-model PP comes after v1. Upstream ATOM fix. |
| T82 | **`15` D91 Q2's "no scheduling coupling" is contradicted by the head's decode admission loop.** Opened 2026-09-21 by P0.5. `atom/model_engine/scheduler.py::Scheduler.schedule` skips any running sequence whose id is in `_pp_inflight_token_block` while composing a decode batch, and `mark_pp_inflight` puts exactly the sequences whose token is in flight into that set - its docstring reads "Head: block re-scheduling of seqs whose token is now in flight." So the first stage's batch composition at step N is a function of pipeline in-flight state, which is the coupling D91 Q2 says PP does not add. The mechanism is T64's own finding: the head keeps up to `pp_size` independently scheduled batches in flight, and the block is what stops a sequence decoding against a stale token. The item is the amendment to D91 Q2 and whatever follows for the PP LP model; this row registers it and does not amend the decision. |

### Topics 01, 03, 05 — newly opened

| # | Item | Topic |
|---|---|---|
| **T47** | A wrong lookahead is a calibration error no causality detector sees; only validation against real runs (`08`) or a stated-vs-measured channel-delay check catches it | `01` |
| **T97** | Settle the recurrence check's provisional thresholds, p10 to p90 of the host-period error within 2% per step kind and the run total within 0.5%, once [#827](https://github.com/jgong5/ATOM/issues/827) reports them on [#828](https://github.com/jgong5/ATOM/issues/828)'s records | `01` |
| ~~**T91**~~ | ~~How the DP ranks that make no CA call register sends and get frames released~~ — ruled 2026-10-01, option (b) ([#470](https://github.com/jgong5/ATOM/issues/470#issuecomment-5933154215)); stated in `01` D3 | `01` |
| **T49** | The prefix-index *lookup* cost is charged to nobody — ~1,387 blocks hashed and probed per request at the cc-traces p50, magnitude unmeasured | `03` |
| **T50** | Whether runtime memory constants transfer across dies (the working assumption says yes within a software generation) | `03`, `05` |
| **T53** | Whether tokenizer throughput transfers across CPU classes (the working assumption says yes, adjusted by derate) | `05` |
| **T88** | **D25 says the resolved spec is echoed *verbatim*, and `echo()` is canonical rather than verbatim.** Opened 2026-09-23 by #196 (issue #192). A dotted key resolves to the nested path — deliberately, because a probe fragment writes them — so one field has two spellings and `echo()` rebuilds only the nested one. Measured on `087f85e9d`: the reference document with `host.cpu` written flat reads, `value("host.cpu.cores_physical")` is 96, and `echo()` equals the **nested** document with `digest(flat) == digest(nested)` — while the document read was the flat one. For a digest that is the right answer, and it is why #196 refuses a document stating both spellings rather than picking one. But "verbatim" and "canonically" differ for a fragment a probe wrote flat, and D15's honesty measure is stated as the echo standing beside the number it produced. Either D25 says *canonically, after resolution*, or `echo()` records the spelling it was given. Nothing reads the echo as text today, so it is cheap to settle before the artifact store (#157, #161) takes a content digest over one. | `05` |

---

## 4. Cross-cutting issues

Beyond the per-topic TODOs.

1. **Silent failure is the dominant risk mode.** The always-on causality detectors
   (`01` D3.2), the `01` D3 stall rules (Chandy–Misra recovery for a stall the CA can
   see, one `DIAG_S` diagnostic for one it cannot, never an abort), and the AST
   clock-site test are the design, not decoration.
2. **Simulation speed is unmeasured under this architecture.** The prior design ran
   **0.30×** under saturation — slower than the system it simulates. Measure a saturated
   cell early, not at the end.
3. **Per-step replay CPU cost** was 4.3 ms against a 32.7 ms modelled step, and only with
   the bound allocation in the cache key; a shape-only key is unsound.
4. **Schedule agreement must be reported separately from latency**, and it was established
   two changes *before* the latency numbers were right.
5. **Scheduling fidelity is unobservable at saturation.** Some cell needs deliberate slack.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries the banner. If paired evidence is needed there, the real side may not exist.
7. **A user guide is a deliverable, not documentation debt.** The toolchain is designed
   so the tool tells the user what to measure (`07` D37), which only works if the flows,
   the recommended calibration sequence and the flag surface are written down for a reader
   who was not in these design conversations. Requested during review; belongs in the
   execution plan as its own task, not as a trailing chore.
8. 7. **ATOM's `main` moves while Compass is built.** The seam (`Config.runner_qualname`) has
   two in-tree users so it is unlikely to vanish, but the 212 synchronization sites of
   `01` D4 and the clock-read sites of `11` D72 are ordinary code that upstream will
   touch. The CI clock-source lint is the detector; a rebase cadence is an execution-plan
   question.

---

## 5. Pending amendments

Corrections identified while writing later documents, not yet applied to earlier ones.

| Document | Amendment |
|---|---|
| `01`, `03`–`06`, `09`, `13`–`17` | The naming record of [#952](https://github.com/jgong5/ATOM/issues/952): two granularities and a method per term replace the cost-model split of `04` D17 and the documents that cite it, and the retired backend, resolver and provenance names go; applied to `README`, `02`, `07`, `08`, `10` and `12`, and pending in the others ([#954](https://github.com/jgong5/ATOM/issues/954)) |
| `05` | **Applied** 2026-10-14 (#937): D26 has no `transfer` probe; no cross-hardware transfer of empirical data (`07` D36) |
| `04` | **Applied**; the second half 2026-10-18 (#1489), D18's decision now being concrete capture. D18's open issue *"whether ATOM's real model classes trace cleanly under this mode at TP>1"* is answered yes and the measurement is now in `04` itself. The other half was that the measured trace was **concrete, not symbolic** — 0 non-numeric shape entries of 13,047 — so it did not carry D18's argument against route D; D18 now captures at concrete shapes and no longer needs route D, which closed T81 |
| `04` | D18's five `torch.cuda` stubs are not enough to import ATOM on this stack. Construction needs three more (`get_device_properties`, `current_device`, `get_device_capability` — the first is read at *import* by aiter's Triton attention configs) and running `ModelRunner.__init__` needs fourteen more, tagged `needed_for: "model_runner"` by the capture module's `install_runner_stubs` (the import-path set carried `needed_for: "import"`) and pinned by name in that module's tests — **both the module and those tests have since been withdrawn from the tree**, so the counts stand as measured and nothing here re-takes them. **Corrected since: fourteen were stubbed, nine are needed.** `tests/compass/test_capture_real_model.py` counts every read of every stub over a whole capture, and of its fourteen runner-group stubs a capture reads nine; `empty_cache`, `max_memory_allocated`, `mem_get_info`, `memory_reserved` and `reset_peak_memory_stats` were never read, leave `graph_digest` unchanged at TP1 and TP2 when removed, and are no longer stubbed (#269). "Needs fourteen more" read a stub list as a need, which is what the read count exists to separate. `get_device_properties` and `mem_get_info` are *stated readings* in the `03` D14 sense and must be passed in, never read from a host |
| `04` | **Applied** 2026-10-18 (#1489): D20's leaf contract lifts a raw Triton or FlyDSL launch to its innermost ATOM function. D18 does not cover raw `@triton.jit` launches. They bypass the dispatcher entirely, so `FakeTensorMode` cannot fake them and the first one reached kills the trace in `triton/backends/amd/driver.py::HIPLauncher.__call__`. Any inventory taken with them skipped is a **diagnostic**, not a capture, and must be labelled so wherever it is reported |
| `04` | A TP>1 capture taken through `apply_simulated_tp` at one physical rank both **erases** and **fabricates**: 129 `all_reduce` per forward become the identity and appear nowhere in the inventory, while one `all_gather` becomes six real dispatched ops over a half-zeros tensor. Neither direction is visible in the operator list itself. **Measured since: the substitution is not needed.** At an honest width the same forward completes and records all 129 (`04` D18, *Collectives at TP>1*), where the substituted one refused at 2,477 ops with none. Removing it from the capture path was left as a separate task; the module has since been withdrawn from the tree, so no capture here takes the substituted route |
| `04` | D18 requires `torch.cuda.is_available()` to report **True**, measured on a host whose driver was *wedged*: False hangs inside `_ensureCUDADeviceGuardSet` and True completes. On a host with **no driver at all** the dependency runs the other way, and the same flag is read by two parties that need opposite answers. ATOM needs True, as D18 says. `FakeTensorMode` needs False: that one flag gates `_only_lift_cpu_tensors`, which keeps `torch.tensor` on the host and moves it afterwards — without it ATOM's own `torch.tensor([])` under a CUDA default device is `No HIP GPUs are available`, below anything the mode can intercept — and it gates `_ensureCUDADeviceGuardSet`, which makes CUDA kernels traceable, and the skipping of constant propagation across a device conversion, which otherwise runs the next operator on a small fake **for real** on the destination device. `tests/compass/test_capture_real_model.py` answers both by overriding `FakeTensorMode.avoid_device_init` rather than by choosing one. Two further readings are not `torch.cuda` at all and D18 does not name them: aiter shells out to `rocminfo` at import through `get_gfx_runtime`, which ignores `GPU_ARCHS` and needs `/dev/kfd`, and Triton's active driver asks the live device for its target, after which aiter falls back to a jax import that is not installed. Both are the architecture, and it is configured. |
| `04` | **D18 does not say how `CpuGpuBuffer` is built under the mode, and it has to straddle it.** `CpuGpuBuffer.__init__` allocates a CPU tensor with `pin_memory`, allocates the device side `zeros_like` it, and takes `.numpy()` of the first. Under `FakeTensorMode` all three are faked and `.numpy()` raises `.numpy() is not supported for tensor subclasses`, so **no `ModelRunner` constructs at all** until this is dealt with. `tests/compass/test_capture_real_model.py` stages the three primitives around ATOM's own body rather than replacing the constructor: the host side and the numpy view are taken outside the mode, `pin_memory` is dropped (pinning is a real `hipHostMalloc`, a property of the transfer rather than of the shape, and nothing traced can observe it), and the device side is converted inside the mode. **Two traps on the way, each of which costs a day twice.** *One:* a `.numpy()` taken while the mode is active leaves the real storage marked **not resizable**, and the next operator on the CPU side then fails inside the converter, with a deprecation warning about reading a FakeTensor data pointer as the only clue. *Two:* converting `self.cpu` itself rather than a discarded template memoises the concrete side as a symbolic fake, so ATOM sees a symbolic shape on the side that is supposed to be concrete — `Trying to resize storage that is not resizable`. The two sides have to be two tensors, not two views of one conversion. This is also where T81's site two lives, so a repair here is the next task's; the constructor is executed rather than substituted precisely so that such a repair cannot land unnoticed. A related detail on the same path: `Tensor.numpy` is dispatched through the torch-function mode `set_default_device` installs, so a counter on it reads two per buffer unless it is guarded. |
| `15` | **D92's decision row still reads *"inherits the TP group"* and *"remainder included"***, and P0.6 measured both false — the group is built in aiter and spans DP, and an indivisible expert count is refused rather than rounded (`15` D92, T65). D92's **conclusion** and D93's formula survive unchanged, so this is a correction to the decision's stated reasons, not to the decision. Rewriting a decision is the owner's call; the body of D92 carries the measurement in the meantime. |
| `04`, `07` | **Applied** 2026-10-18 (#1408): both state the two caps. The MoE all-to-all's block-count cap is stated as a single number in `04`'s `exclusive` join-policy text and in `07`'s calibration table; it is `min(128, CU)` for prefill and `min(64, CU)` for decode (**T87**) |
| `05` | D25 rule 4's *"the whole resolved spec is echoed into every run artifact"* — and D15's "echoed verbatim" — describe `echo()` as it is not: it rebuilds the schema's nested spelling, so a fragment written with dotted keys round-trips to the same *digest* but not to the same *text* (**T88**) |
