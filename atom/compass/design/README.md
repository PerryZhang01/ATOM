# ATOM Compass — Design

**Status: reviewed and approved, 2026-09-20. Design only — no code has been written
against it yet.** Every document carries a matching header.
**Decisions D0–D105, with no gaps, and their sub-decisions** are indexed at the end of this
file. The TODOs are registered in **`12_open_items.md`**, with the load-bearing assumptions
and the cross-cutting issues. Implementation follows the execution plan in `16`.

---

## What Compass is

A performance simulator for ATOM. It answers one question:

> **For this model, this parallelism, and this workload — what latency and throughput, and
> does it fit in memory?**

Memory is part of the question, not an afterthought: it decides which configurations exist
at all, and predicting the speed of a configuration that cannot start is answering the
wrong question.

## The one idea that explains all the others

> **Compass replaces only the forward pass.**

ATOM's real scheduler, real block manager, real admission logic and real prefix cache all
run unchanged. A simulated run therefore makes **the same scheduling decisions** as a real
one; it only substitutes a predicted duration for the work.

Almost every decision in these documents follows from taking that literally.

---

## Design principles

Inherited from the task, and every decision in these documents is traceable to one of them.
Where a decision looks odd, it is usually principle 1 or 2 being applied literally.

| # | Principle | Where it bites |
|---|---|---|
| **1** | **Reuse ATOM's API server and scheduling modules.** Replace only the model layer and the modules it depends on (KV cache management, communication). Modify or refactor the reused parts where discrete time requires it — but do not reimplement them. | `01` D1 keeps ATOM's whole multi-process topology rather than collapsing it. `03` D13 refuses to simulate the KV pool. `07` Phase 0 forbids re-deriving admission to get shapes. |
| **2** | **Simulated execution touches no GPU.** No compute, no device allocation. Compute capability, memory size, bandwidth and interconnect are **configured**, never read from a device runtime. | `05` exists at all. `03` D14 substitutes the five device readings. `04` D18 captures under `FakeTensorMode`. |
| **3** | **Prioritise simplicity.** Add only what is necessary, and nothing more. | `02` D11: the runner has no modes. `01` D4: the K9 synchronization sites are deliberately left outside the model, measured. |
| **4** | **Prefer clean abstractions and refactoring over ad-hoc changes.** | `02`'s `CostBackend` boundary, which forbids ATOM imports. `05` D24 pushes thread-pool width into ATOM's config rather than Compass's. |
| **5** | **Start small; cut work into stages that are individually verifiable.** | M1's fake model exists to validate the clock, the KV path and the harness *before* any real cost model. |
| **6** | **Refuse rather than fall back.** A declined answer with a named reason is a result. A guessed one is a defect. | Everywhere. The archetypal failure is in *Conventions* below. |

Two more that the evidence added, and that are not in the original brief:

| # | Principle | Why it was added |
|---|---|---|
| **7** | **Never report an aggregate without its decomposition.** | Aggregates have hidden compensating errors here at least six times. See finding 2. |
| **8** | **Every claim carries its measurement.** A number without a source is a defect. | Several "known" constants turned out to be one data point. |

---

## Scope: what is simulated, what is reused, what is stubbed

The single most important table here, and the one most easily got wrong. "Compass replaces
only the forward pass" is a statement about *decisions*, not about tensors — several things
ATOM's real code owns still never touch a device.

| Component | Status | What that means concretely |
|---|---|---|
| **Forward pass** | **SIMULATED** | No kernels run. `CompassModelRunner.forward` evaluates a cost model and returns a predicted duration plus a correctly-shaped `ScheduledBatchOutput`. `02` D10 |
| **Model weights** | **STUBBED** | Constructed on meta/fake tensors for geometry; no checkpoint bytes are read and nothing is resident on a device. `02`, and see `12_open_items.md` M-c |
| **KV cache *tensors*** | **STUBBED** | `allocate_kv_cache` allocates no bytes. It passes an empty registry to ATOM's `set_kv_cache_data`, which sets the registry to `{}` and builds the worker-side KV connector for a `compass` `kv_transfer_config`. `03` D13 |
| **KV block accounting** | **REUSED, unmodified** | `BlockManager`, `BlockPool`, the prefix index, `plan_pools`, ref counting, eviction, preemption. It is pure arithmetic over integers, so running the real thing is *more* faithful than simulating it, and free. `03` D13 |
| **KV *transfer* (PD disagg)** | **SIMULATED** | No RDMA, no real bytes on a fabric. A simulated connector registered in ATOM's existing factory charges `latency + bytes/bandwidth` from the machine spec. `allocate_kv_cache` refuses, by name, any `kv_connector` other than `compass`, including a non-empty `kv_transfer_config` with no `kv_connector` key, which reads as `moriio`. A `kv_transfer_config` of `{}` (ATOM's default) or `None` builds no connector and refuses nothing; a config that lacks the field is refused by name. `01` D6 |
| **Prefix caching** | **REUSED, unmodified** | The hit *is* ATOM's hit, at ATOM's 64-token block granularity. Its effect on prefill cost is a cost-model term, not an inference. `03` |
| **Scheduler / admission / chunking** | **REUSED, unmodified** | The whole point. Same decisions as a real run. `01`, `03` |
| **API server, tokenizer** | **REUSED, real** | Real HTTP, real uvicorn, real tokenizer — run for its *effect*, with a modelled duration charged for its *time*. `06` D33 |
| **Device memory readings** | **SUBSTITUTED** | Five readings come from the machine spec instead of the runtime, so an MI308X host can model an MI355X. The budget *arithmetic* around them is ATOM's. `03` D14, `05` |
| **Wall-clock time** | **SUBSTITUTED** | Every clock read and timer the Clock Authority can reach runs on virtual time, failure detectors and metrics cadence included. Only the Atomesh router's bounds and health check, set by its flags, and OS-level process-death detection stay real. `01` D5, `11` D72 |
| **Collectives / comms** | **SIMULATED** (cost only) | Priced from measurements or the spec; no collective actually runs on a device. `07` D40 |
| **Atomesh router** | **REUSED, configured** | No Compass change beyond the `--worker-request-timeout-secs` option (#478); a simulated launch sets it large and switches the health check and circuit breaker off. The simulated timeline rides carriers the router already relays: the `compass` entry of the `tracestate` header on requests, a field in `kv_transfer_params` on the prefill-to-decode forward, and SSE comment lines on streamed output. `01` D7, `06` D30 |
| **Serving *decisions*** | **NOT MODELLED** | Compass predicts the time decisions consume, not the decisions themselves — because it reuses the code that makes them. *(A simple serving simulation is planned as a future part of Compass; explicitly out of scope for this work.)* |

**The rule underneath the table:** anything that is *arithmetic* is reused; anything that is
*a device* is substituted or stubbed. The KV pool is the clearest case of the first and the
one most often assumed to be the second.

### Not modelled at all

Fragmentation. Neighbour contention. A device that has never been measured (except at
tier-0 fidelity). See **Scope and non-goals** below for where each is recorded.

---

## Architecture

Four views. **A** shows where Compass sits inside ATOM; **A2** is Compass alone, layered;
**B** follows one step; **C** splits offline from online; **D** is the cost-tier stack.

### A. Where Compass sits

```
   Dotted (:) borders are Compass. Solid borders are ATOM's own code, or a
   third-party harness, running unmodified.

   harness (any vendor)              ATOM, unmodified
   +----------------+       +-----------------------------+
   |  agentx-harness|  HTTP |  api_server                 |
   |  / aiperf      |------>|  LLMEngine / CoreManager    |
   |                |<------|  Scheduler                  |
   | +............+ |       |  BlockManager / BlockPool   |
   | :  adapter   : |       |  prefix cache, admission    |
   | : clock cli  : |       |  tokenizer (real)           |
   | +............+ |       +--------------+--------------+
   +----------------+                      |
           :                 ScheduledBatch |  ScheduledBatchOutput
           :                                v
           :  +...........................................................+
           :  :                        C O M P A S S                      :
           :  :   +-----------------------------------------------+       :
           :  :   |  CompassModelRunner          <-- THE SEAM     |       :
           :  :   |  --runner-qualname; no ATOM change needed     |       :
           :  :   |  no weights, no KV tensors, no GPU            |       :
           :  :   +-----------------------+-----------------------+       :
           :  :                           |                               :
           :  :   +-----------------------v-----------------------+       :
           :  :   |  CostBackend (tier 0 / a / b)                 |       :
           :  :   |  MemoryModel . Cost IR . ArtifactStore        |       :
           :  :   +-----------------------+-----------------------+       :
           :  :                           |                               :
           :  :   +-----------------------v-----------------------+       :
           :..:   |  Clock Authority                              |       :
   TAR/NER    :   |  grants virtual time to every logical process |       :
   requests,  :   |  co-hosted by default; standalone for 1P+1D   |       :
   TAG grants :   +-----------------------------------------------+       :
              :                                                           :
              :   +-----------------------------------------------+       :
              :   |  simulated KV connector                       |       :
              :   |  latency + bytes/bandwidth, from the spec     |       :
              :   |  (registered into ATOM's connector factory)   |       :
              :   +-----------------------------------------------+       :
              +...........................................................+
```

**Yes, the Clock Authority is a Compass component.** It ships in Compass, it is started by
Compass, and it has no meaning in a real ATOM run. It deploys two ways from one
implementation — co-hosted in the API-server process by default, standalone for the
multi-container 1P+1D runs of M1 and M2 (`01` D3.3).

The seam needs **no ATOM change**: `Config.runner_qualname` already exists and already has
two in-tree users.

### A2. Compass alone, layered

Every component in the detailed design documents appears here exactly once, with the
document that owns it. Nothing in `01`–`11` is outside this diagram.

```
  +=====================================================================+
  |  L5  TOOLING          compass plan | discover | trace |             |
  |                       measure {ops,collectives,steps,memory} |      |
  |                       validate | explain                            |
  |                       spec probes . merge . validate . explain      |
  |                       flags, precedence, the `compass` executable   |
  |                                                docs 07, 05, 13      |
  +=====================================================================+
  |  L4  WORKLOAD         clock client  |  compass stamps on the wire   |
  |                       per-harness adapter (out of tree)             |
  |                                                    doc 06           |
  +=====================================================================+
  |  L3  MODELLING        CostBackend          MemoryModel              |
  |                       +- tier b op-level   +- weights               |
  |                       +- tier a coarse     +- KV (ATOM's arithmetic)|
  |                       +- tier 0 analytic   +- activations           |
  |                       Cost IR: Seq/Repeat/Par, opaque leaves        |
  |                       fitting, law selection, hull guard            |
  |                       spec-decode: draft/verify structures,         |
  |                       declared acceptance, draft KV                 |
  |                                       docs 02,03,04,09,10,14,17     |
  +=====================================================================+
  |  L2  EXECUTION        CompassModelRunner (the seam)                 |
  |                       simulated KV connector                        |
  |                       metrics under virtual time                    |
  |                                             docs 02, 01 D6, 11      |
  +=====================================================================+
  |  L1  TIME             Clock Authority: grant rule, lookahead matrix |
  |                       LP registry . TAR/NER/TAG protocol            |
  |                       causality detectors (straggler check at       |
  |                       receipt and drain, one DIAG_S stall           |
  |                       diagnostic, CI clock lint)                    |
  |                       LP topology per strategy: TP/DP/EP collapse,  |
  |                       PP adds one LP per stage                      |
  |                                                docs 01, 15          |
  +=====================================================================+
  |  L0  ARTIFACTS        machine_spec   op_graph      price_list       |
  |                       region_terms   memory_readings coverage_hull  |
  |                       keys . digests . fingerprints . invalidation  |
  |                                                    docs 05, 07      |
  +=====================================================================+

  Dependencies point downward. L0 is written offline and read by everything;
  L1 is the only layer every other layer talks to at run time.
```

### Component map — where each piece is designed

| Layer | Component | Document | Decisions |
|---|---|---|---|
| L1 | Clock Authority, grant rule, LP collapse | [`01_execution_and_time_model.md`](01_execution_and_time_model.md) | D3, D3.1 |
| L1 | Causality detectors | [`01`](01_execution_and_time_model.md) | D3.2 |
| L1 | CA deployment (co-hosted / standalone) | [`01`](01_execution_and_time_model.md) | D3.3 |
| L1 | Wait interception contract (K1–K9 mechanisms) | [`01`](01_execution_and_time_model.md) | D4, D5 |
| L1 | Arrival gate | [`01`](01_execution_and_time_model.md) | D8 |
| L2 | `CompassModelRunner`, the seam | [`02_model_runner_and_cost_backend.md`](02_model_runner_and_cost_backend.md) | D10, D11 |
| L2 | M1 fake model | [`02`](02_model_runner_and_cost_backend.md) | D12 |
| L2 | Simulated KV connector | [`01`](01_execution_and_time_model.md) | D6 |
| L2 | Atomesh handling | [`01`](01_execution_and_time_model.md), [`06`](06_workload_harness_contract.md) | D7, D30 |
| L2 | Prometheus metrics under virtual time | [`11_metrics_support.md`](11_metrics_support.md) | D71–D77 |
| L3 | `CostBackend` interface, provenance vocabulary | [`02`](02_model_runner_and_cost_backend.md) | D11 |
| L3 | Memory model, KV pool reuse | [`03_memory_and_kv_model.md`](03_memory_and_kv_model.md) | D13–D16 |
| L3 | Model capture, Cost IR | [`04_model_capture_and_cost_ir.md`](04_model_capture_and_cost_ir.md) | D17–D23 (+ D18.1) |
| L3 | Fitting, law selection, hull guard | [`09_fitting_and_law_selection.md`](09_fitting_and_law_selection.md) | D53–D62 |
| L3 | Analytic laws (tier 0) | [`10_analytic_laws.md`](10_analytic_laws.md) | D63–D70 |
| L3 | Step fit backend (tier a), a stage-2 candidate | [`17_step_fit_backend.md`](17_step_fit_backend.md) | D96–D105 |
| L4 | Harness contract, wire fields, adapter | [`06_workload_harness_contract.md`](06_workload_harness_contract.md) | D27–D35 |
| L5 | `compass plan` and the calibration phases | [`07_calibration_toolchain.md`](07_calibration_toolchain.md) | D36–D43 |
| L5 | Machine spec schema and probes | [`05_machine_spec_and_probes.md`](05_machine_spec_and_probes.md) | D24–D26 |
| L0 | Artifact store, keys, invalidation | [`07`](07_calibration_toolchain.md) | D41, D43 |
| — | Validation protocol (judges all of it) | [`08_validation_protocol.md`](08_validation_protocol.md) | D43.1, D44–D52 (+ D50.1, D50.2) |
| L5 | Configuration surface: flags, precedence, the `compass` CLI | [`13_configuration_surface.md`](13_configuration_surface.md) | D78–D81 |
| L2/L3 | Speculative decoding and MTP | [`14_speculative_decoding.md`](14_speculative_decoding.md) | D82–D87 |
| L1/L3 | Parallelism: TP, DP, PP, EP | [`15_parallelism_support.md`](15_parallelism_support.md) | D88–D94 |
| — | Open items, assumptions, gaps | [`12_open_items.md`](12_open_items.md) | — |
| — | How it gets built: tasks, waves, gates, GPU queue | [`16_execution_plan.md`](16_execution_plan.md) | — |

### B. A simulated step, end to end

```
  harness                api_server        Scheduler         CompassModelRunner
     |                       |                 |                     |
     |--- POST /v1/... ----->|                 |                     |
     |                       |-- tokenize ---->|                     |   modelled as a
     |                       |   (real)        |                     |   bounded-width queue
     |                       |--- add_request->|                     |
     |                       |                 |-- schedule() ------>|   REAL decisions:
     |                       |                 |   ScheduledBatch    |   admission, chunking,
     |                       |                 |                     |   prefix hits, preemption
     |                       |                 |                     |
     |                       |                 |                     |-- cost model
     |                       |                 |                     |   -> predicted seconds
     |                       |                 |<-- BatchOutput -----|
     |                       |                 |    + predicted dt   |
     |                       |                 |                     |
     |                       |          advance_to(now + dt) --------------> Clock Authority
     |                       |                 |                     |
     |                       |<-- postprocess -|                     |
     |<-- SSE chunk ---------|                 |                     |
     |    + ": compass a=..  |                 |                     |
     |       s=.." comment   |                 |                     |
     |    line before each   |                 |                     |
     |    event              |                 |                     |
```

Wall-clock time passes while that HTTP round trip happens. **Virtual time does not** — it
advances only for durations the cost model produced.

### C. Offline / online split

This is what makes "no GPU at simulation time" work.

```
  OFFLINE, needs a GPU, run rarely          ONLINE, device-free, every run
  ================================          ==============================

   spec probes      Phase 1b price
   Phase 1c in-situ Phase 2 memory
        |                  |
        v                  v
   +--------------------------------+       +---------------------------+
   |        ARTIFACT STORE          |------>|  Phase 0  discovery       |
   |  key + digest + fingerprint    |       |  Phase 1a trace (JIT)     |
   |                                |<------|  cost model -> seconds    |
   |  machine_spec   price_list     |  JIT  |  memory model -> blocks   |
   |  op_graph       region_terms   | trace |                           |
   |  memory_readings coverage_hull |       |  -> simulated run         |
   +--------------------------------+       +---------------------------+
```

Artifacts are keyed, digested and fingerprinted; a stack or source change **refuses**
rather than answering from stale data.

### D. The three cost tiers

```
  +-------------------------------------------------------------+
  |  tier b   op-level empirical                                 |
  |           symbolic IR + priced leaves + region terms         |
  |           needs: a calibration campaign                      |
  |           used for: acceptance cells                         |
  +-------------------------------------------------------------+
  |  tier a   coarse empirical                                   |
  |           fitted over features ScheduledBatch already carries|
  |           needs: measured steps                              |
  |           used for: structure discovery, sweeps, plumbing    |
  +-------------------------------------------------------------+
  |  tier 0   analytic / roofline                                |
  |           machine spec + model geometry ONLY                 |
  |           needs: nothing                                     |
  |           used for: day one; a device never measured         |
  +-------------------------------------------------------------+
```

Orthogonal to the tier, every individual answer carries its **provenance** — measured,
fitted, interpolated, extrapolated or analytical — and a step reports the mixture.

One tier-a backend goes beyond its "used for" line by ruling: `17`'s step fit is graded as
a candidate for #627's stage 2 (`17` D96, `07` D36).

---

## Acceptance targets

| Requirement | Target |
|---|---|
| Throughput prediction | error ≤ 10% |
| Time per output token (TPOT) | error ≤ 10% |
| Time to first token (TTFT) | error ≤ 15% (`16` D95) |
| Each non-KV memory term | error ≤ 10% |
| KV capacity / block count | **error ≤ 5%** |
| Generalization | predict beyond the configurations used for calibration |
| Simulation speed | aim ≥ 5×; negotiable, but **faster than a real run** is the floor |

Tier 0 has its own, looser, separately-declared goals, with **configuration ranking** as
its primary gate rather than latency error (`10` D67.1).

Final proof is **paired simulated and real execution of cc-traces proper**.

### Milestones

| # | Milestone |
|---|---|
| **M1** | **[#627](https://github.com/jgong5/ATOM/issues/627) stages 1 and 1b: cc-traces end to end on a GPU-free simulated run** (`16` D95). Stage 1 drives cc-traces through agentx-harness at TP1; stage 1b adds atomesh at 1P+1D; each run ends by the finish. Built on the discrete-event foundation, fake models covering prefill, decode and KV need, and the test harness |
| **M2 — v1** | **#627 stage 3: DeepSeek-V4-Pro 1P+1D on MI355X within the targets above**, first in the nightly configurations with expert parallelism off, then at EP8 — each step first the same scheduling decisions as the real engine, then the separable results of `08` D44 and the configuration-ranking gate; **the MI455 roofline projection**, held to the soundness checks of `16` D95. **First gate, #627 stage 2:** Qwen3.8-27B TP1 on MI308X, paired real and simulated cc-traces graded by `08` |
| ~~M3–M7~~ | Retired by `16` D95, which maps each one to where its scope went. PP, EP beyond one node, and further models come after v1 |

Sequencing, dependencies and parallelisable work are in the execution plan (`16`). `00_initial_prompt.md` is the original seed and is
**not** a design document — see *Development history*.

---

## Scope and non-goals

Stated here so they are not discovered at review.

| Not in scope | Where it is recorded |
|---|---|
| A device that has never been measured — except at tier-0 fidelity, which is derate-dominated | `10` D69 |
| Neighbour contention. Compass models a **dedicated** device, so it will not predict an OOM a shared box produces | `03` D14 |
| Memory fragmentation — not modelled, not planned, and nobody models it | `03` D16 |
| **Cancellation** — `status` is `"completed"` on all 1,697 subagent wrappers in both corpora. There is nothing to replay. | `06` D35 |
| Serving *decisions* as a simulated subsystem — a simple serving simulation is planned for a later phase of Compass, not this one | — |

One thing that **is** in scope and is worth stating as a limit rather than a non-goal:

- **Predicting what acceptance rate a speculative draft head will achieve.** Speculative
  decoding and MTP *are* in scope (`14`), but acceptance is a behaviour Compass cannot
  compute — it is a **declared input**, like device bandwidth. A speculative throughput
  result is conditional on that input and the artifact says which tier it came from
  (`14` D83, D87).

### In scope, and worth naming because they are easily assumed otherwise

- **Speculative decoding and MTP.** Now topic `14`. Three of the four things it changes are
  routine — more structures, a `K+1`-token query per sequence, two extra memory terms — and
  the fourth, acceptance, is handled by ATOM's existing `--spec-decode-acceptance-*`
  mechanism rather than by anything Compass builds.
- **Closed-loop arrivals.** These *are* reproduced (`06` D35) and the clock contract makes
  them work: the harness holds a clock client and its pacing is a Clock Authority call, so
  a closed-loop replay runs against the global virtual timeline. What is genuinely
  *reconstructed rather than recorded* is the **join**: no field in the corpus says a parent
  resumed because a child finished, so the harness imposes SPAWN/JOIN linkage. That is a
  fidelity caveat on the workload, not a scope exclusion.
- **Asymmetric parallelism (EP, PP, DP).** DP-attention and EP8 within a node are in M2;
  PP and EP beyond one node come after v1 (`16` D95). `15` designs all three.

---

## Load-bearing assumptions

Five assumptions hold up large parts of the design. **One has been tested** — T10, resolved
by P0.3 on 2026-09-20; the other four have not. Each now carries a named check, a place it
runs, and a cost — in **`12_open_items.md` §1**, so they are schedulable work rather than
caveats.

| # | Assumption | If false |
|---|---|---|
| **T21** | The in-situ calibration transfers across TP width | the "calibrate at TP1, predict TP2/4/8" recipe collapses. Overhead constants have **already** been measured moving 12% and 7% in *opposite directions* between TP1 and TP2 on the same GPUs. |
| **T25** | The real-vs-real noise floor stays narrow under closed-loop replay at high client count | those cells become ungradeable. All prior data is 20 requests, one session, declared arrivals. |
| **T5** | ATOM's model classes trace cleanly under `FakeTensorMode` at TP>1 | tier b has no IR, and docs `04`, `07`, `09` rest on it |
| **T52** | `TorchDispatchMode` instrumentation does not hang ATOM at width | capture is unusable at TP>1; gates T5. A mode-induced 8-rank hang was observed while `dspark_scheduler.py::schedule_prefix_lengths_tensor` was written; no landed version makes the call that hung, and the hang is to be root-caused, not worked around. |
| ~~**T10**~~ | ~~`AgenticReplayStrategy` can be subclassed rather than vendored~~ — **resolved 2026-09-20 by P0.3: yes, and nothing is vendored** | the ~2,000-line consequence does not occur. What the spike found instead is that a subclass reaches only four of the nine pacing calls, so the adapter rebinds the runner's `LoopScheduler` (`06` D34.1) |

**Order to settle the four that remain:** T52 → T5 → T21, T25. T10 came first — an hour, no
hardware, largest swing per hour — and is done. The last two are designed-in steps of the
calibration and validation flows, not extra work — but they can each invalidate an
acceptance claim, so they should not drift to the end.

---

## Development history

Two prior attempts exist on this repository, and the measurements quoted throughout these
documents come from them.

| Branch | Relationship | What it contributed |
|---|---|---|
| `feature/atomcompass_take2` | the pruned PoC baseline; equals PR jgong5/ATOM#2 | the seam, the arrival field, the provenance vocabulary, the first cc-traces pilots |
| `feature/atomcompass` | **not** an earlier abandoned attempt — it is take2 plus ~350 commits | the calibration campaign, the memory evidence, the op-pricing method, and every failure mode listed in *Five findings* |
| `feature/atomcompass_new` | this branch, a clean fork of upstream `main` | design only, so far |

**This is a fresh design.** The prior work is referred to at the level of *design*, never as
a code-port plan: where a prior mechanism is the right answer it is re-derived on its
merits; where it is not, it is not carried. What *is* inherited wholesale is the evidence —
roughly forty measurements, most of which were got wrong once before they were got right,
and which is why several decisions here look more defensive than a first design would.

`00_initial_prompt.md` is preserved as the **original seed**: the task as first written,
sketchy and partly superseded. It is not a design document and is not normative. Where it
disagrees with a design topic, the design topic wins.

---

## Five findings that shaped everything

Anyone reading one section should read this one.

1. **Aggregate cost accuracy does not bound schedule accuracy.** A prior run was within
   **1.0% on prefill seconds, 1.1% on decode, 1.0% on run length — and 90% wrong on median
   TTFT**, because TTFT hung on two comparisons with 1.5 s and 1.2 s of slack. Hence `08`
   D44's three separable results.
2. **Aggregates hide compensating errors — at least six times.** A +13.8% memory sum was
   three errors two of which cancelled, the largest being 25% of one term. A cc-traces
   latency within 5% was **+52% TTFT against −30% decode**. Hence per-term validation
   everywhere.
3. **Every failure mode in this design is silent.** A causality violation, a specialized
   trace, a dead gate, a contaminated reading — none of them crash. They produce a
   plausible table. Hence always-on assertions, loud aborts, and gates whose state is
   written into the artifact rather than inferred from a flag.
4. **Concentration rescues the cost model.** Sixteen operators cover a step, half the time
   in two, twelve kinds cover 99.7%. 60–75% of step time sits in 8–20 **opaque leaves**. So
   the job is pricing ~20 leaves well, not decomposing everything.
5. **The instrument has a floor.** Pricing repeats to 0.96% summed, 1.18% median per
   signature, **p90 32%**. *Any residual quoted below about 2% is quoting the instrument,
   not the model.*

---

## Glossary

The documents use these precisely; a reader will bounce off without them.

| Term | Meaning |
|---|---|
| **provenance** | how an answer was obtained. `analytical` (computed without measuring the subject) / `measured` (the unit itself was timed) / `fitted` (a form chosen, coefficients regressed) / `interpolated` (nearby measurements looked up) / `extrapolated` (asked outside the measured range). *"Priced" is not a species* — a price is `measured`, differing only in unit. |
| **opaque leaf** | an operator the tracer records as one node whose internal kernels are invisible, because ATOM deliberately registers a whole subsystem as a single dispatcher op. ~20 of them hold 60–75% of step time. |
| **logical process (LP)** | the unit the Clock Authority coordinates. *Not* an OS process — a TP group is one LP, a DP group is one LP, because both already have a hardware barrier. |
| **structure** | a distinct operator list, independent of shape. Prefill has two (chunked, unchunked); decode has one across all CUDA-graph rungs. Traces are cached per structure, not per shape. |
| **treatment** | a factor that moves cost and is invisible to every feature in the model. Uniform MoE routing under DP is one; decode **row order** was one until tier a gained `rowmod(k)` (`09` D61). |
| **tier** | which cost model was asked (0 analytic / a coarse / b op-level). Orthogonal to provenance. |
| **refusal** | a declined answer with a named reason. Refusals are results, not errors, and are preferred to fallbacks everywhere. A **refused step** is one no rung of the resolver ladder answered: it costs 0 s, the run continues, and any refusal makes the run a coverage report (`08` D50.1). A rung that declines while a lower one answers is a fall-through, and an analytic or extrapolated answer is reported apart; neither is a refusal. |
| **step fit** | the tier-a backend of `17`: a grid of step measurements looked up per batch, plus a residual, fitted on replayed batches, for what the grid's coordinates cannot see (row order, raggedness). |
| **complete cell** | a cell of step fit's grid whose every corner was measured. Step fit answers only inside one, by exact hit or interpolation; anywhere else it refuses. |
| **recipe** | step fit's feature functions and the constants frozen with them. Configuration supplies geometry and hardware values, a fit supplies coefficients, and a recipe's version changes only when a function does. |
| **bundle** | step fit's artifact: a grid, a recipe and the residual fitted with them, identified by a `07` D41 key and a digest, never by a path. |

---

## Conventions

- Every document opens with a **status header** recording its review state.
- Each file is a **design topic**; the numbered `D*` items inside it are **design points**.
  A topic owns several points. Sub-numbered points (`D3.1`, `D25.1`) extend the
  point they hang off, rather than renumbering everything downstream.
- Decisions are numbered **`D*`**, continuous across documents, with a decision log at the
  end of each. TODOs are **`T*`**, likewise continuous.
- Documents cite each other by number and decision — *"`07` D41"*.
- Claims carry their measurement. A number without a source is a defect.
- **Refuse rather than fall back.** The archetypal failure was an oracle asked about a step
  kind it had no samples of falling back to the mean of an empty list — zero — producing
  *"a confident, precise, entirely fictional answer"*: a TTFT of 0 ms against a real 7.6 s.

---
## Start here

| If you are… | Read |
|---|---|
| new to the project | the top of this file through *Architecture*, then Part I below |
| reviewing a specific decision | the decision map below, then straight to that document |
| about to implement | Part I, then the document owning your area, then `12_open_items.md` |
| wondering what is *not* settled | **`12_open_items.md`** — assumptions, missing topics, TODOs, cross-cutting issues, all in one place |
| looking for the original task | `00_initial_prompt.md` — a seed, not a design doc |

---

## Reading order

### Part I — Architecture: what a simulated run *is*

| Doc | Title | What it settles |
|---|---|---|
| [`01`](01_execution_and_time_model.md) | Execution and Time Model | Keep ATOM's multi-process topology. A central **Clock Authority** acts as the HLA RTI (TAR, NER, TAG); an LP is a process group with one clock owner per member process, the CA joins a DP group's members, and zero-lookahead couplings share one LP. Every synchronization site maps to one PDES mechanism, K1–K9. A straggler fails the run; a stall the Clock Authority cannot see prints one diagnostic and never aborts. KV transfer is simulated, reproducing Mooncake through scheduler-side hooks; the Atomesh router is a configured segment of a channel; arrivals are messages in transit. |
| [`02`](02_model_runner_and_cost_backend.md) | Model Runner Seam and Cost Backend | Attach at `ModelRunner.forward`, delivered by a `--runner-qualname` subclass — **no ATOM change for the injection**. The runner has no modes; the algorithm comes from a pluggable backend. Trace is device-free and may be lazy; measure needs a device and never is. The milestone-1 fake model. |

### Part II — What is modelled

| Doc | Title | What it settles |
|---|---|---|
| [`05`](05_machine_spec_and_probes.md) | Machine Specification and its Probes | The one input artifact describing device, host, interconnect and the software stack it is pinned to. Read this before `03` and `04`, both of which consume it. Probe tools and their contamination refusals. |
| [`03`](03_memory_and_kv_model.md) | Memory Model and the KV Pool | **Do not simulate the KV pool** — ATOM's real one is pure arithmetic. Substitute the five device readings, never the arithmetic. How a prefix-cache hit reaches the cost model. Validate per term, never as a sum. |
| [`04`](04_model_capture_and_cost_ir.md) | Model Capture and the Cost IR | Capture with `TorchDispatchMode` + `FakeTensorMode` + `ShapeEnv`. A hierarchical, symbolic, stream-annotated IR whose `Repeat` nests and tolerates non-contiguous layer patterns. **Opaque leaves** are priced, not decomposed. |
| [`10`](10_analytic_laws.md) | Analytic Laws (Tier 0) | Cost and memory from device parameters and model geometry, with no measurement of the subject. Three classes: exact from geometry, device-parameterised, policy-determined. Declared accuracy goals with **ranking** as the primary gate. **The most speculative document here.** |
| [`17`](17_step_fit_backend.md) | Step Fit Backend (Tier a) | A measured grid of whole forwards, answered only from complete cells, plus a residual fitted on replayed batches. A stage-2 candidate, graded by stage 2's protocol and off the stage-2 path for now (#627). |

### Part III — How the data is made

| Doc | Title | What it settles |
|---|---|---|
| [`07`](07_calibration_toolchain.md) | Calibration and Benchmarking Toolchain | `compass plan` as the single entry point, and one recommended flow it emits. The minimal recipe: one full-engine run at TP1, plus one TP2 transfer test, plus a startup per width. Four classes of communication pricing. The artifact store, its keys and its invalidation matrix. |
| [`09`](09_fitting_and_law_selection.md) | Fitting and Law Selection | How measurements become a model: fit relative error, per-rung decode, hull guards not bounding boxes, and why leave-one-out cannot choose a family. Owns coverage geometry for every other document. |

### Part IV — How it is driven and judged

| Doc | Title | What it settles |
|---|---|---|
| [`06`](06_workload_harness_contract.md) | Workload Harness Contract | A three-part contract, not a bespoke client. agentx-harness reused with **zero edits** via an out-of-tree plugin. One namespaced stamp each direction, audited for minimality: a `tracestate` entry in, an SSE comment line per event out. The timeline rides carriers Atomesh already forwards, so it needs no change. Tokenizer cost is a queue, not a constant. |
| [`08`](08_validation_protocol.md) | Validation Protocol | ATOM's own test suite as the first validation layer, in two tiers: a CPU tier over every test file outside `tests/plugin/` and `cpu_gate_exclude.txt`, driver-free **as a batch** and held to green, and a GPU superset judged as a **delta** against **4779 / 5** (`fe9ea043c`, torch 2.10.0+rocm7.2.4, ROCm 7.2.4, AITER v0.1.21.dev0-49-gf4e7c7509, all five failing node-ids on file). Three separable results, never one number. **The real-vs-real spread is the tolerance.** A metric is admissible only if stable *and* sensitive. |
| [`11`](11_metrics_support.md) | Engine Metrics under Virtual Time | ATOM's Prometheus exporter under a virtual clock. Metrics are classified by the **provenance of their value**, not their type. Sample once per engine step — virtual time is discrete-event. The metrics push and refresh are daemon timers on simulated time. |

### Part V — Cross-cutting

| Doc | Title | What it settles |
|---|---|---|
| [`13`](13_configuration_surface.md) | The Configuration Surface | Three homes for a setting, and the test that assigns them. Precedence is CLI > env > artifact > **refuse**. The engine-side flags, one `compass` executable, and an audit of what is deliberately *not* a flag. |
| [`14`](14_speculative_decoding.md) | Speculative Decoding and MTP | Three of the four changes are routine. Acceptance is a **declared input** through ATOM's existing flags, fed the *measured per-position distribution* rather than a mean. No new cost form — decode becomes the `N_Q = 1` case of the general one. |
| [`15`](15_parallelism_support.md) | Parallelism Support | One frame of four questions per strategy. **Only PP adds logical processes** - TP, DP and EP each sit behind a barrier ATOM already has. DP is the one that couples *scheduling decisions*, through a per-forward collective that rewrites the batch; both its collectives run for real, because both reduce over scheduling metadata rather than model outputs. Explicit M1/M2 split; PP after v1. |

### Part VI — What is not settled

| Doc | Title | What it holds |
|---|---|---|
| [`12`](12_open_items.md) | Open Items | The five load-bearing assumptions and their check plans; the missing-topic register; the TODO register; cross-cutting issues; pending amendments. |

### Part VII — How it gets built

| Doc | Title | What it settles |
|---|---|---|
| [`16`](16_execution_plan.md) | The Execution Plan | Tasks are a **pool**, capped at 5 in flight, with context durable in the task's GitHub issue and PR rather than in an agent. Phase 0 de-risks five assumptions before any build work. Effort in lines of code, not dates. Detailed through Wave 3, deliberately coarse beyond. |

---

## Decision map

| Decisions | Document |
|---|---|
| D0 – D9 (+ D3.1–D3.5, D5.1) | `01` Execution and Time Model |
| D10 – D12 (+ D10.1) | `02` Model Runner Seam and Cost Backend |
| D13 – D16 | `03` Memory Model and the KV Pool |
| D17 – D23 (+ D18.1) | `04` Model Capture and the Cost IR |
| D24 – D26 (+ D25.1) | `05` Machine Specification and its Probes |
| D27 – D35 (+ D34.1) | `06` Workload Harness Contract |
| D36 – D43 (+ D38.1, D40.1) | `07` Calibration and Benchmarking Toolchain |
| D43.1, D44 – D52 (+ D50.1, D50.2) | `08` Validation Protocol |
| D53 – D62 | `09` Fitting and Law Selection |
| D63 – D70 (+ D67.1) | `10` Analytic Laws (Tier 0) |
| D71 – D77 | `11` Engine Metrics under Virtual Time |
| D78 – D81 | `13` The Configuration Surface |
| D82 – D87 | `14` Speculative Decoding and MTP |
| D88 – D94 | `15` Parallelism Support (TP, DP, PP, EP) |
| D95 | `16` The Execution Plan |
| D96 – D105 | `17` Step Fit Backend (Tier a) |

### Headline decisions

| # | Decision |
|---|---|
| D1 | Keep ATOM's multi-process and multi-thread topology; every change is additive |
| D3 | A central **Clock Authority** acting as the HLA RTI; a grant stays strictly below the lookahead-distance LBTS and waits for messages in transit; zero-lookahead couplings collapse into one LP |
| D4 | Every synchronization site maps to one of the PDES mechanisms K1–K9; the boundary is the LP, not the process. A forward's charge composes the backend's device time with measured host time, by one recurrence per runner |
| D10 | Attach at `ModelRunner.forward` via `--runner-qualname`; no ATOM change for the injection |
| D13 | Do not simulate the KV pool; run ATOM's real one |
| D14 | Substitute the five device readings, never the arithmetic |
| D18 | Capture with dispatch mode + FakeTensor + ShapeEnv; `_EnablePythonDispatcher()` is mandatory |
| D19 | Hierarchical, symbolic, stream-annotated IR; no branches in the IR — applicability is a discrete key plus an evaluated guard domain |
| D20 | Opaque leaves are priced, not decomposed; each carries a declared parameter extractor |
| D28 | Stamps on the **real** endpoint, both directions, in carriers a real server ignores: a `tracestate` entry in, SSE comment lines out |
| D30 | Ride the simulated timeline on carriers Atomesh already forwards, `kv_transfer_params` on the prefill-to-decode hop among them; Atomesh needs zero changes |
| D36 | Three tiers: analytic, coarse empirical, op-level empirical. Compass models a device it has been measured on |
| D37 | `compass plan` — the tool tells the user what to measure |
| D45 | The real-vs-real spread is the tolerance; a metric must be stable **and** sensitive |
| D3.2 | Always-on causality detectors: a straggler, checked on receipt and at each drain, fails the run rather than warning; a stall the Clock Authority cannot see waits for ever and prints one `DIAG_S` diagnostic, never aborting; a CI clock-source lint |
| D3.3 | The Clock Authority ships two deployment forms from one implementation: co-hosted by default, standalone for multi-container runs |
| D43.1 | ATOM's suite is a merge gate on every Compass change, unmodified, in two tiers: a CPU tier (every test file outside `tests/plugin/` and `cpu_gate_exclude.txt`, driver-free as a batch, held to green) per change, a GPU superset judged as a delta per wave against 4779 / 5 at `fe9ea043c`, by an equality on a per-tree expectation rather than "no worse than". The CPU tier **exits 98** rather than reporting "GPU not required" when it cannot tell |
| D67.1 | Tier 0 is graded on **configuration ranking** first; its latency goals are diagnostics for that, not the result |
