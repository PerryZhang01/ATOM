# ATOM Compass — Design Topic 14: Speculative Decoding and MTP

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `02` (the cost form), `03` (KV and memory), `04` (structures), `07`
(calibration), `08` (validation). This topic changes something in each of them.

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** acceptance is a stated input carrying
a method, reported in the run summary and propagated into no cost term (D83, D84, D87).

**Scope.** What speculative decoding — Eagle3, DSpark, and MTP heads — changes for a
simulator, and why one of those changes is different in kind from everything else in this
design. The draft path it prices is DeepSeek-V4-Pro's serial MTP, in M2 (D87, *Milestone
placement*); Eagle3 and DSpark are described because ATOM carries them, and are not on
M2's path.

---

## D82. What spec decode actually changes, and the one hard part

### The four changes

| # | What changes | Difficulty |
|---|---|---|
| 1 | **Step structure.** A decode step becomes *draft* (one or more small forwards, or `mtp_k` reuses of one layer) plus *verify* (one target forward carrying `K+1` tokens per sequence) plus a rejection sample. | **Routine.** More structures, discovered by `07` Phase 0 exactly like any other. |
| 2 | **Batch shape.** `num_tokens` per sequence is `K+1`, not 1. A decode batch stops being one-token-per-row. | **Routine.** The cost form already generalises — D85. |
| 3 | **Memory.** Draft model weights, and draft KV layers on top of the target's. | **Routine.** ATOM already has the single source of truth — D86. |
| 4 | **Acceptance.** How many draft tokens are accepted decides how many steps a request needs, and therefore TPOT and throughput directly. | **This is the hard part.** |

### Why acceptance is different in kind

Everything else Compass predicts is a *duration*. Acceptance is a **behaviour** — it comes
out of comparing draft logits with target logits, and Compass computes neither, because it
does not run the model. There is no cost model that can produce it and no amount of
calibration that would help.

This is the first thing in the whole design that Compass cannot derive and cannot measure
its way to. It has to be **stated**, like device bandwidth is stated, and it carries a method like any input (D84).

The good news, and it is substantial: **ATOM already built the mechanism**, for its own
reasons, before Compass existed.

---

## D83. Acceptance is a stated input, and ATOM's existing synthetic path is the mechanism

### What is already in the tree

`SpeculativeConfig` (`atom/config.py`) carries a synthetic-acceptance path added
for benchmarking against a published acceptance figure while a draft head is still
training (ROCm/ATOM#555):

| Field / flag | Meaning |
|---|---|
| `--spec-decode-acceptance-length` (`synthetic_acceptance_length`) | mean acceptance length in `[1, K+1]`, **counting the target's own guaranteed token** — the same unit as vLLM's `synthetic_acceptance_length` and SGLang's `SGLANG_SIMULATE_ACC_LEN` |
| `--spec-decode-acceptance-rate` (`synthetic_acceptance_rate`) | the same target as a mean rate in `[0,1]`, i.e. `(length − 1) / K`. Mutually exclusive with the above. |
| `synthetic_acceptance_rates` | resolved by `__post_init__` into **per-position unconditional** rates: entry `i` is the marginal probability that the first `i+1` draft tokens are all accepted |

The rejection sampler consumes them (`atom/model_ops/rejection_sampler.py::rejection_sample`, `atom/model_ops/rejection_sampler.py::_get_synthetic_cond_rates`),
converting unconditional to conditional rates because the kernel walks positions
sequentially, and force-accepting accordingly.

### Decision

> **Compass does not build an acceptance model. It states acceptance through ATOM's
> existing `--spec-decode-acceptance-*` flags, and applies ATOM's acceptance *policy*.**

### The sampler is a cost and a value, and only one of them can be traced

`rejection_synthetic_sample_kernel` is a **Triton kernel**
(`rejection_sampler.py`). Under Compass no kernel runs. So the sampler has to be
split into two halves that are handled differently:

| Half | What it is | Treatment |
|---|---|---|
| **cost** | the kernel's duration | **Traced and priced like any other operator** — the sampler is dispatcher-visible and goes through `07` Phase 1b with everything else. Nothing special. |
| **value** | `num_bonus_tokens` per sequence, written by `tl.store(num_bonus_tokens_ptr + req_idx, …)` (`rejection_sampler.py::rejection_synthetic_sample_kernel`) | **Cannot be traced.** Compass needs the number itself — the scheduler updates `num_computed_tokens` from it, decides `max_tokens` completion from it, and sizes the next step from it. Without the value the simulation cannot advance. |

So there is exactly one thing Compass must reimplement, and it is small:

> **A device-free draw reproducing the synthetic kernel's semantics: walk positions,
> compare a uniform against the conditional acceptance rate, stop at the first reject.**

That is the whole of `rejection_synthetic_sample_kernel`'s logic
— a sequential walk, no reduction, no tensor algebra. The conditional-rate
conversion it consumes is *already host Python*
(`_get_synthetic_cond_rates`, and `acceptance_length_to_rates`), so only the
walk moves. Estimated at ~15 lines.

**The rank-consistency machinery does not need reproducing, and this is the one place the
simulator is simpler than the engine.** The real kernel needs a dedicated device generator
re-seeded per step (`_SYNTHETIC_RNG_BASE_SEED`) because `sampled_tokens` is broadcast from
rank 0 while `num_bonus_tokens` stays local, and a per-rank `torch.rand` would desync the
two — *"the anchor gather reads a rejected (−1) column → the draft model then embeds an
invalid id (HSA out-of-bounds)."* Compass single-sources the clock and the step from rank 0
already (`01`), so there is one draw, not `world_size` draws that must agree. Seed it from
the step counter and a run is reproducible, which is what `12` M-e asks for elsewhere.

**As built ([#1090](https://github.com/jgong5/ATOM/issues/1090)).** `DeferredTokenStream`
(`atom/compass/runner/step_output.py`) draws one uniform `u` per decode row of a verify
step, from one stream per DP rank seeded by its rank, and accepts `a = #{i : u <
rates[i]}` drafts. Over non-increasing unconditional rates that is the kernel's
distribution, `P(a >= i+1) = rates[i]`, in one draw per row. The row reports `a + 1`
tokens, `num_rejected = K - a` and `num_bonus = a`, and every row reports `K` draft ids,
as ATOM's runner does. A run whose speculative config declares no acceptance is refused by
name. ATOM refuses forced acceptance beside DSpark's confidence schedule
(`Config.__post_init__`), so a DSpark run verifies its full draft block; its
confidence-scheduled truncation is not modelled. The drafter is built on the meta device
and reads no checkpoint; a drafter hook ATOM renames is refused by name.

The realisations differ: the kernel draws per position from a generator re-seeded each
step, the host one uniform per row from a per-rank stream, so a paired run below full
acceptance parts at the first differing draw. Only the distribution is shared; it is
checked against the kernel at K = 1, and at K > 1 by
[#1546](https://github.com/jgong5/ATOM/issues/1546). A speculative step whose price tags
no draft term is refused by name, `cost:<backend>: prices no draft`; step-level backends
tag one with [#1541](https://github.com/jgong5/ATOM/issues/1541).

### What still flows through ATOM's real code

Everything downstream of the value: `SpecStats` accumulation, the bonus-token bookkeeping,
the anchor gather, and the scheduler's `max_tokens` interaction
(`tests/test_scheduler_mtp_max_tokens.py`). The stated acceptance enters at the same
point the kernel's output would, so the *consequences* of an acceptance are ATOM's, not a
reimplementation.

### Why declaring through ATOM's flag is still right

**It is ATOM's flag, not a Compass one** (`13` D80's non-flag audit). A real run can be
driven with the identical argument, which is what makes a paired comparison a pairing —
and it means the *policy* being applied is the same object in both runs even though the
draw happens in different places.

**Recorded as T62:** the host draw and the Triton kernel must agree. Same stated rates,
same seed, same accepted-count distribution over a few thousand draws. A cheap test, and
without it the two halves of the pairing can drift silently.

**What this makes the prediction:** conditional. Compass predicts *"at this acceptance
length, this throughput"*. That is a weaker claim than for a dense model and it must be
reported as one — D87.

---

## D84. The mean is not enough: feed the measured per-position distribution

### The trap

`acceptance_length_to_rates` builds **the minimum-variance schedule**: accept
`floor(length − 1)` positions with certainty and put the remainder on the next one. It is
the right choice for its original purpose — replay a published mean AL — and it is a trap
for Compass.

A real run's acceptance is *ragged*: at a mean of 2.4, some sequences accept 0 and some
accept `K`. The minimum-variance synthetic schedule at the same mean produces a much
narrower distribution. The two agree on total tokens and **disagree on the shape of every
subsequent batch**, because a sequence that accepted 0 needs a step that one which accepted
`K` does not.

This is the same failure mode as the design's central finding, in a new place: matching an
aggregate does not match the schedule. A prior run was within 1.0% on prefill seconds and
90% wrong on median TTFT.

### What a distribution can and cannot buy

Feeding measured *rates* and re-drawing from them reproduces the acceptance
**distribution**, in expectation. It does **not** reproduce:

- **the per-step realisation** — which sequence accepted how much on step *k*;
- **temporal correlation within a sequence** — a request whose continuation is genuinely
  hard to predict has persistently low acceptance, and independent per-step draws erase
  that;
- **cross-sequence correlation** at a step.

All three change batch composition, and batch composition is what the design's central
finding says aggregates do not bound. So a rate-based feed is a **deliberate trade**, not
an equivalence.

### Three possible contracts

| # | Contract | Cost | Reproduces |
|---|---|---|---|
| **1** | The user states acceptance rates from prior knowledge and passes them to Compass | **cheapest** — no real run needed | the stated distribution |
| **2** | The user collects statistical acceptance rates from a real run (via Compass tooling) and passes them | one real run | the *measured* distribution, marginals only |
| **3** | Each request carries a per-token acceptance list, passed like any other HTTP attribute | **heaviest** — per-request payload, and the trace has to exist | the realisation, including correlation |

### Decision

**Contracts 1 and 2 are the design. Contract 3 is recorded as available and is not
built now.**

Rationale: per-step accepted-token counts are too heavy to carry for a 300-second,
multi-thousand-request replay, and the statistics are a good trade for what they cost.
Contract 2 is the one to use for acceptance cells; contract 1 is what makes *"what if the
draft head reaches AL 3.0"* answerable at all.

Contract 3 stays on the shelf rather than being rejected: it is the only way to close the
correlation gap above, and if a result ever turns out to depend on it, the transport is
the same additive-field mechanism `06` D28 already defines.

For contract 2, the flow:

```
  real run  --->  SpecStats.distribution       (dict: accepted_count -> occurrences,
                  scheduler.py)                 maintained per sequence per step)
                        |
                        v
            per-position unconditional rates    P(first i+1 all accepted)
                        |
                        v
  simulated run  --->  speculative_config.synthetic_acceptance_rates
```

`SpecStats` already tracks exactly the histogram this needs — `distribution` is keyed
`0..mtp_k` — so the conversion is a cumulative sum and nothing more.

### The transport for a list does not exist yet

`synthetic_acceptance_rates` is a `list[float]` on `SpeculativeConfig`
(`config.py`), but it is **internal and derived** — filled by `__post_init__` from
`acceptance_length_to_rates(length, n)` (`config.py::SpeculativeConfig._resolve_synthetic_acceptance`). The CLI exposes only the two
**scalars** — `--spec-decode-acceptance-length` and
`--spec-decode-acceptance-rate` (`arg_utils.py::EngineArgs.add_cli_args`). There is no input path for a
list, so contract 2 has nowhere to put its measurement.

**The fix is an ATOM flag, not a Compass one**, by `13` D78's own test: a per-position
acceptance curve is meaningful in a real ATOM run — it is the *same* benchmarking use case
as the scalar, only more faithful, letting a published AL curve be replayed shape and all
rather than collapsed to its mean.

```
--spec-decode-acceptance-rates 0.92,0.71,0.48,0.19
```

- mutually exclusive with the two existing scalars, which already exclude each other;
- validated as non-increasing and in `[0,1]` — they are marginals of a prefix event, so a
  rising entry is a malformed input, not a preference;
- bypasses `acceptance_length_to_rates` and sets `synthetic_acceptance_rates` directly.

Small, useful to ATOM independently of Compass, and it is the same shape of ask as
`--preprocess-pool-width` in `05` D24. Recorded as **T63**.

**Three acceptance inputs, ordered, each with its method** (#952):

| # | Input | When | Method |
|---|---|---|---|
| 1 | measured per-position rates from a paired real run | acceptance cells | `measured` |
| 2 | measured mean AL, expanded by the min-variance schedule | a real run exists but only its mean was recorded | `estimated` |
| 3 | a stated AL from a datasheet or a published figure | exploration, day zero, a draft head that does not exist yet | `declared` |

The run summary reports the input's method. It propagates into no cost term: a step's
terms carry the methods their own inputs give them, whatever the acceptance input's.

Input 3 is legitimate and is what makes *"what if the draft head reaches AL 3.0"* a question
Compass can answer. It is **not** legitimate in an acceptance run, and `08`'s hygiene
refusals gain a row saying so.

---

## D85. Cost: no new form is needed

### Why the existing form already covers it

The prefill form of `02` D12 is

```
a + b·tokens + c·Σ_req N_Q² + d·Σ_req (N_Q · N_KV_cached)
```

and a speculative *verify* step is precisely this with `N_Q = K+1` per sequence rather than
`N_Q = 1`. The quadratic term covers attention among the `K+1` new tokens; the product term
covers them attending over context. Nothing about the algebra is new — what changes is that
decode can no longer be priced by a form that assumes one token per row.

**So the decision is a simplification, not an addition:**

> **Drop the separate decode form. Price every step with the general form, and let a
> non-speculative decode be the `N_Q = 1` case.**

The features are already on the batch: `detailed_sqsq`, `detailed_sqsk` and `detailed_sk`
(`scheduler.py::ScheduledBatch`) are `Σ N_Q²`, `Σ N_Q·N_KV` and `Σ N_KV`, computed by
`compute_detailed_aggregates`.

### What *is* new

| Item | Treatment |
|---|---|
| **draft forwards** | their own structures. Eagle3 and standalone DSpark run a real layer stack per drafting step; serial MTP reuses one layer `mtp_k` times. `07` Phase 0 discovers both. |
| **rejection sampler** | an ordinary operator, priced by `07` Phase 1b. Small, but it is per step and it is not free. |
| **`num_spec_steps`** | on `SpecDecodeMetadata` (`forward_context.py`); part of the structure key, because `K` changes the graph. |
| **CUDA-graph rungs** | the capture ladder is over `running_bs`, and a spec step's token count is `running_bs × (K+1)`. Rung padding is computed from the real ladder, as ever — but the rung set a spec run captures differs from a dense one, so `07` Phase 2's memory readings must be taken **with spec enabled**. |

---

## D86. Memory: two extra terms, and ATOM already owns the layer count

Draft models add:

| Term | Source |
|---|---|
| **draft weights** | Class A, exact from the draft's own HF config. `_MTP_CONFIG` (`config.py`) names the architecture and the `n_predict` attribute per model type. |
| **draft KV layers** | `model_runner.py::ModelRunner._num_draft_kv_layers` — **the single source of truth, and it is already called out as such in the code.** A draft with a real layer stack (Eagle3, standalone DSpark) needs one slot per layer; serial MTP declares `num_nextn_predict_layers`. |

Two properties inherited rather than designed, both consequences of `03` D13's decision to
run ATOM's real block accounting:

1. **Whether draft KV shares the target's pool or owns its own is already decided by
   ATOM**, per `_get_total_num_layers` and the `eagle3_draft_builder` split. Compass
   reads the outcome; it does not model the policy.
2. **Eagle3 draft KV merged onto the target's block ids by name** is `03`'s existing
   statement, and it holds unchanged.

**The one thing to be careful about:** the code comment on `_num_draft_kv_layers` records
that two independent spellings of this count *"silently disagreed for the standalone DSpark
draft, sizing 1 slot while allocating 5"*. Compass must call the method, never re-derive
the count — which is `07` Phase 0's "never reimplement" rule applying to memory.

---

## D87. Validation: the claim is conditional, and must be reported that way

### The rule

> **A throughput or TPOT result for a speculative configuration is a statement
> *conditional on the acceptance input*, and the run summary names that input's method.**

Concretely, for acceptance cells:

1. Run the real side. Record `SpecStats.distribution`.
2. Feed the measured per-position rates to the simulated side (D84 input 1).
3. Report the three results of `08` D44 as usual — **plus** an acceptance-agreement check
   between the two sides, at the right strength — which is weaker than exact agreement,
   for the reason D84 gives. What is checkable:

   | Quantity | Expectation | Status |
   |---|---|---|
   | mean acceptance length | agrees to sampling error over the run | **checkable; a gate** |
   | the accepted-count histogram (`SpecStats.distribution`) | agrees in distribution | **checkable; a gate** |
   | per-step realisation — which sequence accepted what on step *k* | **does not agree, and is not expected to** | not a result |
   | within-sequence temporal correlation | **not reproduced** by a rate-based feed | not a result |

4. **If the two aggregate rows disagree materially, the plumbing is wrong, not the cost
   model** — rates were mis-converted, the seed was not set, or the wrong input was fed.
   That is a refusal, not a finding.
5. **Schedule agreement (`08` D44's second result) is graded more loosely for speculative
   cells than for dense ones**, and the artifact says so. Two runs whose acceptance
   histograms match can still diverge in *which* request accepted what, and that
   divergence is an input property, not a modelling error. Exactly how much looser is
   set by the same noise-floor procedure as everything else (`08` D45) — measure
   real-vs-real on a speculative cell first, and if that spread swamps the target, the
   cell is ungradeable and must be reported as such.

### What this does and does not demonstrate

- **Demonstrated:** that Compass predicts the *time* consumed by a speculative
  configuration whose acceptance is known. That is the question the acceptance targets
  actually ask.
- **Not demonstrated, and not claimed:** that Compass predicts what acceptance a draft head
  will achieve on a workload. Nothing in this design attempts that, and a result that
  appeared to show it would be an artifact of the input.

A new hygiene refusal for `08` D49: **a speculative acceptance cell run on input 3
(declared acceptance) is not acceptance evidence.** It is exploration.

### Milestone placement

**M2, on DeepSeek-V4-Pro** (`16` D95). MTP runs in two of the nightly configurations M2
grades, TP + MTP3 and DP-attention + MTP1 (`recipes/mesh/DeepSeek-V4.md`), so it enters
with the model and is not a milestone of its own. The earlier placement, a mechanism check
on Qwen3.8-27B and the claim on Kimi-K3, was retired with those milestones.

**The draft path priced is DeepSeek-V4-Pro's serial MTP**, as ATOM builds it:

| Item | In ATOM | For this topic |
|---|---|---|
| draft model | `deepseek_v4` maps to `deepseek_v4_mtp` (`SpeculativeConfig._MTP_TYPE_MAP`), built as `DeepseekV4MTPModel` (`atom/models/deepseek_v4_mtp.py`). A DSpark checkpoint is routed elsewhere by `SpeculativeConfig.hf_config_override` | serial MTP, not a draft stack |
| draft steps | `hf_config_override` sets `num_nextn_predict_layers` to 1; the one `MTPBlock` runs `mtp_k` times, `K = --num-speculative-tokens`, 3 or 1 in the nightly cells | one draft structure per drafting step (D85); T60 is the serial case |
| draft forward | `MTPBlock` subclasses V4's `Block` (the target's attention and MoE classes), adds `e_proj`, `h_proj`, three norms and its own `hc_head` parameters, and reads the target's un-reduced mHC residual | traced and priced like any structure (D85) |
| draft weights | the MTP block only: `DeepseekV4MTP.share_with_target` binds the embedding and LM head to the target's | D86's weight term excludes both |
| draft KV | one slot (`_num_draft_kv_layers` returns `num_nextn_predict_layers`) in the target's pool, at index `n_layers` (`mtp_start_layer_idx`) | D86 as written: call the method |

**Every M2 MTP cell is an acceptance cell.** Its per-position rates are measured from the
paired real run, input 1 of D84 (`16` W4.7), and never declared. Input 1 needs T63's
transport, so T63 is on M2's path. For MTP1 the curve has one position and the mean
carries it whole; MTP3 is where its shape matters.

---

## Open issues

- **`ATOM_ENABLE_RELAXED_MTP` changes acceptance semantics**, not just its rate —
  `RELAXED_TOP_N` goes 1 → 10 and `RELAXED_DELTA` 0 → 0.6
  (`rejection_sampler.py`). It is an environment variable, so it is invisible to
  every artifact key today. It must be captured in the run fingerprint or two runs with
  different acceptance semantics will compare as one. Recorded as **T59**.
- **The DSpark hang of `04` T52 was observed in this subsystem** (`dspark_scheduler.py::schedule_prefix_lengths_tensor`,
  while the function was written; the landed function builds that value with `torch.full_like`, which avoids the call that hung).
  Root-causing it is already a gating task (`16` P0.7). DSpark is not the draft path M2
  prices, so the hang is not on this topic's path to M2.
- **Whether a draft forward's cost transfers across `K`** is untested. A serial MTP reusing
  one layer `mtp_k` times should be linear in `K`; a real draft stack need not be. One
  measurement at two values of `K` settles it. Recorded as **T60**.
- **Chunked prefill interacting with spec decode** — whether a partially-prefilled sequence
  drafts, and what structure that produces, is not covered here. Recorded as **T61**.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D82 | Spec decode changes four things; three are routine and acceptance is different in kind, because it is a behaviour Compass cannot compute. | 2026-09-19 |
| D83 | Acceptance is a **stated input**, carrying a method, supplied through ATOM's existing `--spec-decode-acceptance-*` flags. The sampler splits: its **cost** is traced and priced like any op, its **value** (`num_bonus_tokens`) needs a ~15-line device-free host draw, because the synthetic sampler is a Triton kernel and no kernel runs. Compass builds no acceptance *model*. | 2026-09-19 |
| D84 | Feed the **measured per-position distribution**, not the mean - the minimum-variance schedule matches an aggregate and not a schedule. Contracts 1 and 2 (stated or measured statistical rates) are the design; contract 3 (per-request per-token lists) is recorded and not built. Transport for a list does not exist today and needs an ATOM flag (T63). Three acceptance inputs, measured, estimated and declared; the summary reports the method, which propagates into no cost term; input 3 is not acceptance evidence. | 2026-09-19; revised 2026-10-14 (#952) |
| D85 | No new cost form. Drop the separate decode form and price every step with the general one; non-speculative decode is the `N_Q = 1` case. | 2026-09-19 |
| D86 | Draft weights are Class A; draft KV layers come from `_num_draft_kv_layers()` and are never re-derived. | 2026-09-19 |
| D87 | A speculative throughput result is **conditional on its acceptance input**, and the run summary names its method. Mean AL and the accepted-count histogram are gated; per-step realisation and within-sequence correlation are **not reproduced and not claimed**. Schedule agreement is graded more loosely for speculative cells. Predicting acceptance itself is explicitly not attempted. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T59 | Capture `ATOM_ENABLE_RELAXED_MTP` in the run fingerprint — it changes acceptance semantics and is invisible today | needs the artifact fingerprint to be implemented |
| T60 | Test whether a draft forward's cost is linear in `K` | one measurement at two values of `K` |
| T61 | Decide how chunked prefill and drafting interact, and what structure that produces | needs Phase 0 discovery on a spec-enabled run |
| T62 | Assert the host acceptance draw and the Triton kernel agree over a few thousand draws | needs the host draw to exist |
| T63 | Add ATOM flag `--spec-decode-acceptance-rates` (list); contract 2 has no transport today | an ATOM change, sized and specified in D84 |
