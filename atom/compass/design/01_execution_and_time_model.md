# ATOM Compass — Design Topic 1: Execution and Time Model

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Branch:** `feature/atomcompass_new` (clean fork of upstream `main` at `0b4f1ddb`).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** the run summary reports predicted
seconds by method (`seconds_by_method`), each with its step count and share (D3.5).

**Revised 2026-10-16 by the owner's ruling on
[#1087](https://github.com/jgong5/ATOM/issues/1087):** D6.1 makes the lmcache MP server an
LP of its own; inside a DP group a park request amends D3's member join rules 1, 2 and 5
(ruling on [#1162](https://github.com/jgong5/ATOM/pull/1162)).

**Scope of this document.** The execution architecture: what a simulated run *is*,
which processes exist, how virtual time advances, and what contract the rest of the
system must honour. It does **not** cover the cost model, the memory model, model
capture, or the workload harness. Those are separate design topics.

---

## 0. Ground rules inherited from the task

- Compass replaces **only the forward pass**. ATOM's real scheduler, real block
  manager, real admission logic run unchanged. A simulated run makes the same
  scheduling decisions as a real one; it substitutes a predicted duration for the work.
- Simulated execution must not compute on a GPU and must not allocate GPU memory.
  Compute capability, memory size, bandwidth and interconnect are **configured**, not
  read from a device runtime.
- Prefer clean abstractions and refactoring over ad-hoc changes. Add only what is
  necessary.
- Final proof is paired simulation and real execution of cc-traces proper.

---

## Relationship to the prior PoC branches

Not a design point — a one-paragraph statement of provenance, recorded here so the
evidence cited throughout this document has an origin. The fuller account is
**Development history** in `README.md`.

Two prior attempts exist (`feature/atomcompass_take2`, `feature/atomcompass`; the
history is linear, the second is the first plus ~350 commits). This design is **fresh,
referring to the prior work at the level of *design* only**. No code-port plan is
defined up front. Where a prior mechanism is the right answer it is described here on
its merits and re-derived; where it is not, it is not carried. Every quantitative claim
below that is attributed to "the prior work" or "a prior run" is a measurement taken on
one of those branches, on this hardware.

### What the prior work established that this design relies on

These are measurements, not opinions, and they are load-bearing below:

- **The seam needs no ATOM change.** `Config.runner_qualname` (`atom/config.py`),
  consumed in `engine_core.py::EngineCore.__init__` and `async_proc.py::AsyncIOProc.__init__`. Two in-tree precedents:
  `RLHFModelRunner` (`atom/rollout/async_engine.py::AsyncLLMEngine.__init__`) and `RapidServeModelRunner`
  (`Config.__post_init__`, `config.py`).
- **Rank-0 single-sourcing of the clock is correct for symmetric TP.** TP=2 over 1727
  steps: per-step rank difference median 0.03%, worst 0.82%, rank 1 slower on 51% of
  steps. TP=4 over 2295 steps: rank totals within ±0.02%; charging every step to its
  **slowest** rank adds **0.06%** to the total.
- **Speedup has two independent sources, and only one of them survives saturation.**
  (i) *Not doing the compute* — the forward pass evaluates a cost model instead of
  running kernels, so every step is cheaper by construction, at every load. (ii) *The
  idle jump* — when every LP is blocked, the clock leaps to the next event instead of
  sleeping through real seconds. Measured: 64 requests Poisson 8/s, 9,426 ms real vs
  639 ms simulated (14.8x); cc-traces 20 requests 27B TP=4, 309 s vs 3 s (~103x). Under
  saturation, where there is no idle to skip, only (i) is left — and it was not enough:
  **122 s vs 36 s, i.e. 0.30x, slower than the system it simulates.**

  **Why source (i) is worth so much less than it sounds, which is the whole explanation
  of that 0.30x.** Removing the kernels does not remove the step. A real decode step runs
  its kernels on the device *while the host is busy* with `prepare_inputs`, block-table
  marshalling, sampling setup and the scheduler's own bookkeeping — the two overlap, and
  on small shapes the host side is the longer of the two. Doc `10` D64 measures exactly
  this from the other direction: at 794 tokens the device window is 36.745 ms of which
  **23.360 ms (63.6%) is idle**, waiting on the host. So deleting the kernel time deletes
  the *shorter* of two overlapped costs on precisely the steps a decode-heavy workload is
  made of, and the host work that remains is not replaced — Compass still runs all of it,
  plus the cost model on top.

  Two consequences: the ≥5x target is a statement about **the arrival process**, not only
  about the cost backend; and the thing worth optimising for speed is the per-step host
  path, which is what doc `08`'s "cost of a simulated step" work measures.
- **Aggregate cost accuracy does not bound schedule accuracy when the scheduler has a
  discontinuity.** The prior run was within 1.0% on prefill seconds, 1.1% on decode,
  1.0% on run length — and **90% wrong on median TTFT**, because TTFT was decided by two
  comparisons with 1.5 s and 1.2 s of slack. A counterfactual at x0.97 on the prefill
  price split a 63-chunk streak back into the real run's exact 42+7+15.
  **Consequence for this design: schedule agreement must be a first-class, separately
  reported result, never inferred from latency error.**
- **The real machine is not on the knife edge; only the simulator is.** All four real
  repeats produced the identical streak structure (36, 6, 42, 7, 15) breaking at the
  same two places, margins varying by at most 0.1 s.

---

## D1. Process and thread model

### Problem

ATOM is a multi-process system. A minimal TP1/DP1 serving run is **3 processes**: one
API server, one `EngineCore`, one `ModelRunner` worker. Wider configurations multiply:
`dp_size x pp_size` engine cores, each with `tp_world_size x pcp_size` workers.

A discrete-event simulator needs a coherent notion of "now". The question is whether to
keep that topology or collapse it.

Verified topology (path and symbol):

| Link | Mechanism |
|---|---|
| API server <-> CoreManager | in-process |
| CoreManager -> EngineCore | ZMQ ROUTER/DEALER (`engine_core_mgr.py::CoreManager.__init__`, `engine_core.py::EngineCore.process_input_sockets`) |
| EngineCore -> CoreManager | ZMQ PUSH/PULL (`engine_core.py::EngineCore.process_output_sockets`) |
| EngineCore -> workers | `aiter.dist.shm_broadcast.MessageQueue`, POSIX shm, 16 MiB chunks (`async_proc.py::AsyncIOProcManager.__init__`, `AsyncIOProcManager.call_func`) |
| worker rank0 -> EngineCore | ZMQ PUSH/PULL (`async_proc.py::AsyncIOProcManager.__init__`, `AsyncIOProcManager.process_output_sockets`) |
| every worker -> EngineCore | per-rank ZMQ PUSH/PULL for KV status (`async_proc.py::AsyncIOProcManager.__init__`, `AsyncIOProcManager.process_kv_output_sockets`) |
| cross-DP | `torch.distributed` Gloo CPU group (`engine_core.py::DPEngineCoreProc._init_data_parallel`, `DPEngineCoreProc._sync_dp_state`, `_sync_shutdown_state`, `_has_global_unfinished_reqs`) |
| PP stage <-> stage | ZMQ metadata + NCCL tensors (`atom/distributed/pp_comm.py`, `pp_transport.py`) |

`multiprocessing` start method is forced to `spawn` (`engine_core_mgr.py::CoreManager.__init__`,
`launch_engine_core`, `atom/utils/__init__.py::get_mp_context`). **There is no in-process / single-process
mode**: `LLMEngine.__init__` unconditionally constructs a `CoreManager`
(`llm_engine.py`) and `EngineCore.__init__` unconditionally constructs an
`AsyncIOProcManager` (`engine_core.py`). Offline examples still go through the full
stack.

### Options

**A. Single process, virtual-time asyncio loop.** Replace `CoreManager` and
`AsyncIOProcManager` with in-process equivalents; every engine core, worker and the
traffic source become coroutines on one loop whose `time()` is virtual. IPC becomes
in-process queues with modelled latency.

- *Pros:* one clock, no races, no distributed time, trivially reproducible. Scales to
  PP/DP/EP/PD because "two nodes" is two coroutine groups. Fastest wall clock.
- *Cons:* large, invasive refactor of ATOM's process layer. Diverges from the design
  principle of reusing ATOM's structure. A simulated run would no longer be "the same
  program with a different forward".

**B. Keep ATOM's multi-process topology; give it coordinated virtual time.**

- *Pros:* minimal intrusion. The simulated deployment *is* the real deployment. Every
  ATOM change is additive (a clock module, clock-read substitutions, channel
  wrappers, config flags).
- *Cons:* requires a time-coordination protocol between processes, and causality
  violations are silent.

**C. Hybrid:** single-process for single-deployment runs, multi-process later.

- *Cons:* defers the hard problem while designing the kernel around assumptions that
  break when it arrives.

### Decision

**Option B — keep ATOM's process and thread design.** No process is removed, no thread
is removed. The changes are:

1. an injectable clock module,
2. clock-read substitution at business-logic sites,
3. wrappers on the cross-LP channels ATOM already has: the channel sockets, the pollers
   that wait on them, and the engine's output queue, which becomes a `RelayQueue` that
   stamps each item when the step loop puts it. Threads no longer declare themselves
   blocked or running; an LP is idle when its clock owner sits at an idle point with
   nothing left to release (D3),
4. the failure-detector treatment of D5,
5. a simulated KV connector and a simulated model runner.

**One substitution that is not a topology change.** A simulated run starts the API server
on a stdlib-asyncio `CompassEventLoop` instead of uvloop, through the `loop=` setting
uvicorn already takes (`api_server.py::main` picks `"uvloop"` or `"auto"` there;
uvicorn also accepts a `"module:Class"` loop factory). Processes, threads and the service
topology are unchanged; only the event loop's implementation differs. The reason is
mechanical: the API server's event loop is an LP clock owner (D3), and virtual time can
only be hooked into a loop written in Python. The stdlib loop decides timer expiry and its
`select` timeout from `self.time()` and blocks in `self._selector.select(timeout)`, so
overriding `time()` puts every timer on LP time and the selector becomes the idle point.
uvloop hands timers to libuv in real milliseconds and blocks inside C (`uv_run`), where
neither hook reaches. Keeping uvloop would take process-wide syscall interception
(`LD_PRELOAD` over `clock_gettime` and `epoll_wait`), which is not an additive change at
an existing site. No fidelity is lost: the frontend's service times come from the model,
not from the wall clock.

Rationale: the design principle is explicit that ATOM's api server and scheduling
modules are reused and only the model layer and its dependencies are replaced. Option A
replaces the process layer, which is neither the model layer nor a dependency of it.
Option B's stated cost — a coordination protocol — is addressed in D3 and turns out to
be small **because ATOM's own structure supplies most of the synchronization already**
(see D3's logical-process collapse).

### Mitigating option B's real cost: causality violations are silent

Option A's safety is structural — one thread cannot race itself. Option B has to *earn*
the same property, and a violation under option B produces a plausible number rather
than a crash. Accepting option B without a detector would mean accepting that class of
error into the acceptance evidence. So the detectors below are always on, sized to be
cheap enough that none of them is a mode anyone can forget to enable.

**What class of error these detect — and it matters which.** They all catch
**implementation defects, not holes in the PDES algorithm.** The grant rule (D3: grant
strictly below `LBTS(i) = min over j≠i of (N[j] + D(j→i))`, with in-transit messages
counted) is conservative by construction: given correct inputs it *cannot* produce a
violation, which is the standard Chandy–Misra–Bryant guarantee and is not in question
here. What the detectors watch for is the inputs being wrong — a send nobody
registered, a channel nobody wrapped, a clock read nobody substituted. Those are
bugs in *our* code and configuration.

This distinction is worth stating plainly because the two cases warrant opposite
responses. If a detector fires, the fix is local: register the send, wrap the
channel, substitute the clock read. **If one fired and none of those explained it, the
mechanism itself would be in question** — and that would be a much larger problem than a
detector, because it would mean the conservative rule is not conservative on this
topology. Nothing observed so far suggests that, and the grant rule is standard rather
than invented here. But the detectors are also the only thing that would *tell us*, which
is a second reason to have them.

**Where violations actually come from.** Each way the inputs go wrong maps to a detector.

| Failure | What it looks like | Detector |
|---|---|---|
| **A send the CA never saw** — an unregistered channel, or a send from a non-owner thread that slipped the thread-identity assertion; no grant counts it | Receiver has already released messages past the arrival time. Silent; the message lands "in the past" | **(1) Straggler check** |
| **A duplicate sequence number** on one channel. Out of order is fine (release is per `(channel, seq)`); a missing one shows as a wait that (2) names | Two frames claim one registered send | **Seq assertion** (D3): fails the run |
| **A cross-LP send off the clock owner** — a `RelayQueue.put` from another thread, or a second thread sending on one channel (I3, I4) | The send would bypass registration | **Thread-identity assertion**: raises at once |
| **A wait the CA cannot see** — an unwrapped channel, or a real collective spanning two LPs; an LP blocks for real while the CA believes it is running | A run that stops making progress. Nothing times out, so nothing is released wrongly, but nothing names the site either | **(2) Coverage audit** |
| **A clock read was missed** — business logic still calls `time.monotonic()` | Two timestamps on one timeline disagree; durations mix scales. Silent | **(3) Clock-source audit** |

**(1) Straggler check — receive side, always on.** Every cross-LP
message passes through the channel wrappers (decision item 3). Each frame carries its
channel, its sequence number and its arrival time `a = t_send + L(ch)`, stamped when the
clock owner produced it. A straggler is a buffered message that no grant has released
(its `(channel, seq)` is in no grant's released set) and whose `a` is below the
receiver's last drain time — the LP time up to which it has already released messages
to ATOM. The wrapper checks this on receipt and again at each drain, over every message
still buffered: a frame read before the receiver passed `a`, and never released because
its send was never registered, would otherwise wait in the buffer for ever with no
straggler and no diagnostic. A message a grant *did* release may arrive with `a` below
the grant time; that is normal TSO delivery (D3) and does not fire the check. A
violation means the receiver has already delivered past the moment this message
arrives, i.e. the local-causality constraint is broken. (It compares the arrival time,
not the send time against the receiver's current clock, which fails correct runs when
`L > 0`.) It is the direct test of the property the whole protocol exists to provide.
On failure: record `(channel, seq, a, drain time, stated lookahead)` and
**fail the run** — not a warning, because a straggler invalidates every number
downstream of it.

A lookahead stated larger than reality cannot fire this check: every arrival is stamped
`t_send + L(ch)` by the sender, and a strict grant never passes an unreported message (its
arrival is at least its sender's `now + L`), so the model just delivers later than reality.
That is a calibration error, invisible to any causality check; validation against real
runs (`08`) or a stated-vs-measured channel-delay check catches it.

**(2) Coverage audit — a stall diagnostic, never an abort.** A wait the CA cannot see is
one no wrapper covers. In a valid run every such wait ends; one that does not can only
be a fault (a bug or a dead process). So nothing waits on a timer: the CA and every LP
wait for as long as it takes, which consumes no virtual time, cannot change the result,
needs no threshold tuned, and cannot kill a valid but slow run. After `DIAG_S` = 30 wall
seconds without progress, one diagnostic is printed (D3.5) and the wait continues.
`DIAG_S` is a reporting threshold, not a model
parameter. Its value is that it names the uncovered wait **during development** rather
than leaving it to be discovered as a hang with no name. A stall the CA *can* see —
every LP waiting at the CA — is not a fault and is not this detector's business: D3's
finish ends the run if it applies, and D3's deadlock recovery handles it otherwise. The
preventive half runs at startup: a process group whose ranks map to more than one LP
raises (D3).

**(3) Clock-source audit — static, run in CI.** D4's categorisation rests on a
by-hand audit of clock-read sites. A grep-level lint over the simulated-path modules
that flags `time.time`, `time.monotonic`, `time.perf_counter`, `datetime.now` and
`asyncio.sleep` outside an allow-list keeps that audit from rotting as ATOM's main
branch moves. The allow-list is the set left on real time deliberately: transport, and
the bounds the CA cannot reach, which D5 sends to configuration. Metrics are not on it:
`11` D72 runs every metric clock on virtual time, so `metrics.py::AtomMetricsExporter.update` is a
substituted clock read like any other. Anything new lands as a CI failure on the day it
is added, not at validation time.

**What this costs.** One float comparison per buffered message at receipt and at each
drain, which is on the per-step path and bounded by the buffer depth; one wall-clock
check in waits that already block; one CI lint. **What it buys:** the statement "no
causality violation occurred" becomes a reported result of every run rather than an
assumption, which is what doc `08` needs in order to treat a simulated number as
evidence at all.

**What it does not cover.** A wrong lookahead: no detector sees it (above). Recorded as
**T47**.

### Open issues

- Simulation wall-clock cost is higher than Option A's. The >=5x target is stated as
  negotiable with a bottom line of "faster than real runs". Under saturation the prior
  design was 0.30x. This must be measured early, not assumed.
- `torch.cuda.set_device` (in `model_runner.py::ModelRunner._setup_device_and_distributed`) and
  `torch.cuda.mem_get_info` (in `model_runner.py::ModelRunner._read_device_memory`) are the two
  hard GPU dependencies a simulated runner must not inherit.

---

## D2. What "PD disaggregation on two nodes" means

### Problem

ATOM contains **three unrelated mechanisms** that are all called prefill/decode
disaggregation. Conflating them invalidates any performance model.

| Name | Scope | Processes | How KV moves |
|---|---|---|---|
| **RapidServe** (`--enable-rapidserve`) | one node, one GPU set | 2 `EngineCore` procs on the **same** devices | **Never moves.** Decode owns the KV tensor; prefill imported it by CUDA IPC and writes directly into decode's buffer. Only the sampled first token crosses ZMQ. `PrefillScheduler.block_manager = None` (`scheduler.py::PrefillScheduler.__init__`). |
| **True PD disagg** (`--kv-transfer-config`) | two node groups | independent full servers | RDMA — MoRI-IO (read/pull) or Mooncake (write/push) |
| **Atomesh** (`atom/mesh/`) | fleet router | Rust binary, **or** `libmesh.so` in-process via PyO3 | Never touches KV. HTTP relay only. |

### Decision

**True PD disaggregation**, with two qualifications from the project owner:

1. The two "nodes" are **two containers on one physical node**. This removes the real
   network, lets both sides share a filesystem, and lets the time-coordination channel
   be local ZMQ/shm rather than a network protocol.
2. **KV transfer is simulated**, not carried by real RDMA. See D6.

### Consequences

- Atomesh is in the loop. Its treatment is D7.
- ATOM's CI-benchmarked multi-node PD path exists (`.github/scripts/atomesh/pd_server_atom.sh`),
  so real-run baselines for pairing are obtainable.
- The ATOM relay is **strictly sequential and blocking**
  (`http_pd_router.rs::PDRouter::dispatch_atom_relay_internal`): POST prefill, await the full JSON, extract
  `kv_transfer_params` (hard error if absent), enrich the decode body, POST decode.
  Unlike the SGLang path (`tokio::join!` on both, `PDRouter::execute_dual_dispatch_internal`) and the vLLM path (detached
  `tokio::spawn`, `PDRouter::dispatch_vllm_mooncake_internal`). **This sequentiality is a gift: the prefill->decode causal
  edge is explicit, one-way, and per-request.**

### Open issues

- Both containers on one physical node contend for the same CPUs. This does not affect
  fidelity (no GPU work) but it does affect simulation wall time, and it means the
  simulator's own CPU cost is on the critical path twice.
- `MORI_SHMEM_MODE=ISOLATION` is required for DP+TP so MoRI (MoE all-to-all) and MoRI-IO
  (KV) heaps stay apart. With a simulated connector this may become moot; confirm.

---

## D3. Virtual time coordination protocol

### Problem

With ATOM's topology kept (D1) and two engine deployments (D2), more than one process
advances time. The local-causality constraint must hold:

> A logical process may not advance its clock past the timestamp of any message it has
> not yet received.

Violating it does not crash. It produces a plausible latency table. Every historically
expensive failure in this project has had exactly that shape.

### Survey of what the field does

Verified against source, not only papers:

- **SimAI** is the only true parallel DES in the LLM/ML simulator space. It achieves it
  by vendoring **UNISON** (EuroSys'24) — conservative, lookahead-based, barrier-windowed
  YAWNS. The safe-window computation, verbatim from
  `ns-3-alibabacloud/simulation/src/mtp/model/logical-process.cc:155`:
  `grantedTime = Min(MtpInterface::GetSmallestTime() + m_lookAhead, MtpInterface::GetNextPublicTime())`.
  Its auto-partitioner cuts p2p links whose delay >= the median — i.e. **it partitions on
  where the lookahead is**.
- **ASTRA-sim** avoids the problem entirely: it has no second clock. `Sys::boostedTick()`
  is a *read* of `comm_NI->sim_get_time()`, and every rank reads rank 0's network
  interface. The system layer never advances time; it only registers callbacks.
- **LLMServingSim** is more extreme still: its frontend parses ASTRA-sim's cycle count
  out of stdout and adopts it.
- **LLMCompass** is not a DES at all — analytical tile-recurrence plus a brute-force
  mapper; its only cycle-level component is memoized into a shipped CSV.
- **No optimism, no rollback, no GVT anywhere in the space.** This is the right call
  here: rolling back a live Python `Scheduler` + `BlockManager` + prefix-cache index is
  not cheap and probably not sound.

Summary: **every system that parallelizes uses conservative barrier windows; every
system that avoids the problem collapses to one clock.**

### The move that makes this small: collapse already-barriered groups into one LP

A *logical process* (LP) is the unit the protocol coordinates. It need not be an OS
process. ATOM already contains hardware barriers that make several processes behave as
one LP:

| ATOM group | OS processes | LPs | Why |
|---|---|---|---|
| TP group | 1 EngineCore + N workers | **1** | Workers are slaved by a blocking RPC (`async_proc.py::AsyncIOProcManager.call_func`) and hold no clock. Rank-0 authority validated at 0.06% (D0). |
| DP group | N EngineCores | **1** | Already `all_reduce`s every step for lockstep (`DPEngineCoreProc._sync_dp_state`) and runs `dummy_execution` on idle ranks (`DPEngineCoreProc._execute_dummy_batch`). Every rank prices its own forward; with host time the ranks share one device end, the latest over ranks, and each leaves at its own release (below). |
| API server | 1 process: uvicorn, `CoreManager` and its threads | **1** (frontend) | Its asyncio event loop is a clock owner of its own: it tokenizes and streams while the engine runs a forward, so it overlaps the step loop in simulated time and cannot be folded into it. Offline, the owner is the loop that calls `get_output`. |
| PD container (prefill or decode) | one full ATOM deployment | **2** | its frontend LP and its engine LP |
| Traffic source | — | **1** | |
| Atomesh router, 1P1D | 1 Rust process | **0**, or **1** under `--dp-aware` | Part of the channels that cross it: with one P and one D it has no choice to make, so it contributes only a fixed per-request forward cost, stated in those channels' lookahead (D7). Under `--dp-aware` (DP attention) it chooses among DP ranks on load that responses move, and is LP `router` (D7). Several P or D instances are deferred. |
| lmcache MP server (`ATOM_KV_OFFLOAD=lmcache_mp`) | 1 process | **1** (`lmcache`) | Its cached chunks decide later hits for every engine that reaches it, so it is a Compass stand-in on the real channels (D6.1). |
| PP stages | N EngineCores | **N** | The only group the collapse does not cover. |

LP counts for the milestones (`16` D95):

| Milestone | LPs |
|---|---|
| M1, #627 stage 1 (one deployment, TP1) | **3** — traffic, frontend, engine |
| M1, #627 stage 1b (atomesh 1P+1D, two containers on one node) | **5** — traffic, frontend-P, engine-P, frontend-D, engine-D |
| M2 (DeepSeek-V4-Pro 1P+1D, one container per node, TP8 on each side, DP-attention where the nightly cell sets it) | **5**; the EP8 step keeps the same LPs |
| After v1: PP | the engine LP becomes one LP per stage |

So each side of M2 is **one** engine LP, not eight: ATOM runs `-tp 8
--enable-dp-attention` as eight `EngineCore`s in one DP group (`15`).

**The partition principle.** Where LP boundaries fall is a modelling choice: a finer
partition is not more correct, it only adds grants and wrapped channels, and the speedup
here comes from skipping idle time, not from parallelism. So: as few LPs as satisfy these
constraints.

1. **Zero-lookahead couplings stay inside one LP.** A per-step barrier or collective, a
   synchronous RPC, or a read of the other side's live state lets one side's action at `t`
   change the other at `t`; no positive lookahead exists. Split across LPs, the protocol
   is still correct (deadlock recovery, below) but fully serial: attention-DP + EP with
   one LP per rank would cost ~960 grants per step (60 layers x 2 collectives x 8 ranks),
   ~48 ms of wall time at ~50 us each against a ~30 ms simulated step. Worse, the real
   collectives would bypass the CA, where it cannot see them, and making them visible
   would mean rewriting them as CA rendezvous, which D1 forbids.
2. **One clock owner per LP process** — a step loop, a PP stage loop, or an asyncio event
   loop; a DP group's LP has one per rank (below). Only the owner calls TAR/NER (below); every other thread is transport (receives into a
   queue, or relays sends the owner registered) or a handler running on a message the
   owner released. Activities that overlap in simulated time and each have a duration
   are either events on the owner's timer list or separate LPs.
3. **LPs interact only through timestamped channels.** A cross-process shared-memory or
   collective interaction is out of model if it touches only replaced code or hardware
   scheduling; otherwise it is a channel with lookahead > 0, or its two sides are one LP.
   A startup assertion enforces the collective case: creating a process group whose ranks
   map to more than one LP raises.

**A DP group is one LP, and its ranks share one device end per forward** (owner ruling
2026-10-01, `15` D90; revised 2026-10-16, #1267). Every rank prices its own batch; with
host time one `all_reduce(MAX)` takes the latest device end over ranks, each rank leaves
at its own release from it, and one `all_gather` on the DP group carries every rank's
step target (the rule is below); `15` D90 owns the `T_moe` and uniform-routing argument. Every rank then calls
the CA as a member of the LP (below). No operator-level TAR/NER. Per-rank LPs are right only if the ranks decouple at step level (no per-step
collective) and interact only with lookahead > 0.

**A DP group's LP has one member process per rank, and the CA joins them** (owner
ruling 2026-10-01, `12` T91, #528). Each DP rank is its own `EngineCore` process with its
own `#dpN` channels and input and output threads, and holds its own LP runtime: its step
loop is that member's clock owner, registers that rank's sends and releases its received
frames. The CA completes the LP's request only when every member has called:

1. Every member makes the same number of CA calls per loop iteration; a per-member call
   counter refuses a mismatch by name.
2. One request kind per round: TAR with equal `T` on every member, or NER with `t`
   and `t_daemon` each the minimum over members. Unequal `T` or mixed kinds are refused.
3. Every member's send log is merged before any grant is computed, so the LP's promised
   time never rises on a partial request.
4. One common grant `G`; each member is released only the messages on its own channels.
5. A member never advances time alone; a lone call only ships its send log.

With `lmcache_mp` above DP width 1, a park request amends rules 1, 2 and 5 (D6.1).

Every member's step end is a round, and so is every member's input pull (owner ruling
2026-10-15, #988; revised 2026-10-16, #1267, and 2026-10-18, #1345).
With host time, one `all_reduce(MAX)` over the DP group, which every forward kind joins, the DP
dummy included, gives each forward the group's device end on the LP clock, as per-layer
collectives hold every rank's device until the slowest, and under MTP the latest target
end beside it. Each rank writes the device end back as its last device end and, on an
output forward, as its output end, the target end in its place under MTP; its forward ends
at its own
release, as a real rank returns at its own launch end. A replay rank's forward ends at its
own replayed return. With no host time, a rank whose forward
waits on its device, one with output other than the DP dummy, ends at the group's `max`;
any other ends at its own seconds, as a real idle rank's forward returns on its own. Where
`out_s` is set, each member then returns at its own pull, its end plus its `out_s`, and
takes in arrivals there before `_sync_dp_state`; the pulls join the ends as rounds
(`group_out`). Each member asks one TAR per distinct round, ascending, returns at its own
pull (its own end where there is no `out_s`) and owes the rest. Until they are paid its sends are stamped at its own end; a TAR before
the last owed round and any NER are refused; and every host collective on its clock owner
thread, a `torch.distributed` collective whose tensors are all on the CPU or that has none,
pays them first, as it returns at the latest arrival. A device collective returns once
enqueued and pays nothing.

One LP per rank is rejected: under NER a rank woken by a message blocks in the lockstep
`all_reduce` on a rank that waits at the CA with no message. Precedents: dist-gem5, where
each process holds its own connection to the synchronising switch that joins them; and
Pham and Bagrodia (WSC 1998), where a parallel federate's time is the minimum over its
members.

### Options for the protocol

**A. Central Clock Authority.** One small process owns global virtual time; LPs request
grants.

- *Pros:* simplest to reason about and to audit. Trivially reproducible. Makes causality
  violations detectable at a single choke point. Degenerates gracefully — with one LP it
  is the prior design's local clock, with zero lookahead it is a single global event loop.
- *Cons:* one RPC round trip per advance; a new component; a single point of failure
  (which is acceptable — its failure is loud).

**B. Conservative barrier window (SimAI/UNISON style).** No central process; each LP
computes its own safe window from a distributed LBTS reduction plus a lookahead matrix.

- *Pros:* what the only true PDES in this field does. No bottleneck.
- *Cons:* more machinery; the LBTS reduction still needs a collective, so at 3-10 LPs it
  buys nothing over A.

**C. Fixed-quantum lockstep.** Barrier every fixed quantum of virtual time.

- *Cons:* the quantum must be smaller than the shortest modelled delay or fidelity is
  lost; at 100 us quanta a 300 s cc-traces run is 3M barriers.

### Decision

**Option A — a central Clock Authority (CA).**

#### State

The CA plays the part of an HLA RTI, and an LP's clock owner uses HLA's time
services:

| LP call | PDES term | Meaning |
|---|---|---|
| `advance_to(T)` | TAR (time advance request) | the owner prices an event of duration `d` (a forward) and asks for `T = now + d`; the grant is exactly `T` |
| `next_event(t, t_daemon)` | NER (next event request) | the owner is idle; `t` is its next essential event and `t_daemon` its next daemon deadline, each or `+inf`; the grant is `min(t, t_daemon, earliest undelivered arrival)`, a daemon deadline held until it is at most the essential horizon `H` (Invariants, below); a one-argument `next_event(t)` means `t_daemon = +inf` |
| grant reply | TAG (time advance grant) | carries `G` and the `(channel, seq)` set it releases |

**Guarantee: a grant to `T` means every message with timestamp `<= T` has been delivered
to that LP.** The one exception is a deadlock-recovery grant (Invariants, below): it
guarantees delivery below `T`, and a message at exactly `T` arrives in that instant's next
round.

Per LP *i*: `now[i]`; a state in `{running, TAR, NER}`; and the target it asked for (`T`,
or `t` and `t_daemon`). Per channel: every registered message as `seq -> arrival`, and
which of them a grant has already released. Static: the channel table (lookahead sources, below), from
which the CA computes once, by Floyd–Warshall, `D(j->i)` — the least total lookahead
over any channel path from *j* to *i*.

#### Grant rule

```
N[j] = now[j]                                                     j running
     = T_j                                                        j waiting in TAR(T_j)
     = min(t_j, t_daemon_j, earliest undelivered arrival into j)  j waiting in NER(t_j, t_daemon_j)

LBTS(i) = min over j != i of ( N[j] + D(j->i) )
a waiting LP i is granted G = N[i]  only if  G < LBTS(i)      -- strictly
```

Waiting LPs are tried in `(N, LP id)` order; a grant changes state, so the check restarts.
PDES term: Ayani's distance-between-objects LBTS (Fujimoto, ch. 3). Each part matters:

- **A running LP contributes `now[j]`, not its next event.** It can produce a message at
  any moment from `now[j]` on. With `next[j]`, LP A can be behind LP B (A had the minimum,
  B ran ahead earlier), and an event A generates at `A.now + 0` lands in `[A.now, B.now)`
  — B's past.
- **An LP waiting in TAR contributes its target.** Its owner is inside the CA call and
  produces nothing (I1, below); messages released to it meanwhile are handled by receiver
  and handler threads, which send nothing across LPs (I4). With `now[j]` instead, two LPs
  in TAR whose targets cross would wait on each other for ever under the strict rule.
- **Distance, not direct neighbours.** In a 1P+1D run, with illustrative lookaheads
  `L(traffic->frontend-P:http) = 0.1` and `L(frontend-P->frontend-D:relay) = 0.5`:
  traffic is running at 9.0, and frontend-P, engine-P and engine-D are idle with `t = inf`
  and nothing undelivered, so each has `N = inf`; frontend-D waits. Its direct
  neighbours are frontend-P and engine-D, both `inf`, but treating the idle frontend-P as
  `inf` is unsafe — traffic's 9.0 request reaches frontend-D through frontend-P at 9.6 —
  and using frontend-P's stale `now` stalls frontend-D for nothing. By distance,
  `D(traffic->frontend-D) = 0.1 + 0.5`, so
  `LBTS(frontend-D) = min(9.0 + 0.6, inf, inf, inf) = 9.6`: correct.
- **Strict.** A message not yet reported to the CA has arrival `>=` its sender's
  `now + L`, so a strict grant never reaches it.

**Messages in transit are counted, not assumed away.** `min(now + L)` alone only
guarantees that no *future* message is earlier. ATOM's sends are asynchronous (ZMQ PUSH,
then a receiver thread), so when the CA sees a sender's clock move, an earlier message may
still be in flight — the case Fujimoto notes the classic rule does not cover. The fix is
the standard counting one (Fujimoto, Mattern):

1. **Registered when produced, reported with the advance.** Only the clock owner registers
   a send, at the moment it produces the message, with `arrival = now + L(ch)`. The
   engine's output thread relays through a `RelayQueue` whose `put` registers on the step
   loop, so a late physical send changes nothing. Inside the simulation window (below) the
   output thread sends every item it takes (its one skip, an all-`EXIT_ENGINE` list at
   `engine_core.py::EngineCore.process_output_sockets`, occurs only at shutdown), so an item it never sends raises
   `UnsentRelayItem` rather than being covered by a null message. The send log
   rides in the same request as the owner's next TAR/NER, so the CA knows every message
   an LP produced before it moves that LP's clock.
2. **The grant carries the expected `(channel, seq)` set**: every registered message into
   *i* with `arrival <= G` not yet released. The receiver waits locally until each has
   physically arrived; nothing is acknowledged back to the CA.
3. **TSO delivery in `_step_through`.** The receive wrappers hold arrived frames back from
   ATOM. Before returning from the CA call, the owner walks the released set in
   `(arrival, channel, seq)` order: it sets `now = arrival`, releases one message, and
   for a thread-received channel waits until the receiving thread is back at its wait
   point before releasing the next. A handler thread (RapidServe's `_recv_prefill_done`)
   therefore runs with the LP clock at the message's timestamp, with nothing else in the
   LP moving.
   Inline channels are taken by the owner at its own receive point.

**The simulation window.** Each LP's window opens once its process is ready to serve (an
engine after it queues READY, `engine_core.py::EngineCore._send_ready_signal`) and closes at that LP's `+inf` grant,
before its process begins to shut down (`LPRuntime.next_event` calls `end_run` on that
grant). Inside it every cross-LP frame is stamped, registered and counted, and the
thread-identity assertion (I3, I4) applies. Frames outside it — READY, SHUTDOWN and the
rest of startup and teardown — are not counted and are handed to ATOM at once; a send
stamped after the `+inf` grant carries arrival `+inf`, which no grant releases.

Counting is exact: release and completion are per `(channel, seq)`, so a channel need not
be FIFO and arrivals on one channel need not be monotone — concurrent HTTP requests do
reorder through the router. The sequence number also asserts no duplicate.

Not chosen: CMB null messages (a standing stream on every idle channel, and the time
promise would have to travel on ATOM's own sockets); RTI message forwarding (ATOM's ZMQ
traffic would route through the CA, against D1); per-message acknowledgement (Samadi:
one extra message per message and a round trip before each grant).

At zero lookahead the strict rule grants nothing once every LP waits, and deadlock
recovery (below) grants one LP at a time: a single global event loop across processes —
correct, serialized, no parallelism. That is why zero-lookahead couplings stay inside one
LP (partition principle, above).

#### Invariants

- **Safety (asserted, always on):** every message registered into LP *j* must have
  `arrival >= now[j]` at the CA; equality is that instant's next round (below). A
  backdated message **aborts the run with a full LP state dump**. It must not be a
  warning and must not be behind a flag.
- **Deadlock: detected and recovered, never aborted.** When every LP waits at the CA and
  the strict rule grants none — possible only with a zero-lookahead channel, once the
  finish (below), which the CA checks first, does not apply — the CA
  grants the LP with the least `(N, LP id)`: no other LP's `N` is smaller, so nothing
  earlier can reach it. PDES term: Chandy–Misra deadlock detection and recovery. A
  message that then arrives at the receiver's current instant counts toward that
  instant's next round, not as a violation: the CA asserts `>=`, and the LP sets
  `now = max(now, arrival)`. A stall the CA cannot see can only be a fault; it waits and
  prints one diagnostic after `DIAG_S` (D1's detector (2)). Never a quiet timeout that
  releases. (The prior arrival barrier's 120 s timeout released on a run that was
  invalid, the client printed "0 failed", and a day's conclusions came off it.)
- **A run finishes when no essential work is left** (#533). Housekeeping timers (the
  metrics push and refresh, uvicorn's server tick, keep-alive, the periodic scrape) are
  daemon deadlines: they fire as usual but do not keep the run alive. The CA keeps an
  essential horizon `H`, the largest TAR target, finite essential NER target or
  registered arrival seen so far, and grants a daemon deadline only once it is at most
  `H`. When every LP waits, nothing is undelivered, no essential target is pending and
  every daemon deadline exceeds `H`, the CA grants every LP `+inf`. The clock cannot tell
  a lost message from a finished run, so on its `+inf` grant the traffic LP raises unless
  every request it sent has its final response, naming those that do not (#534). On a
  server that serves several client runs this test ends a client run, and only the last
  one's end is the finish (Client runs, below).
- **Determinism:** ties at equal timestamps broken by `(time, LP id, channel, seq)`.
  Without this the 126-vs-189-decode-steps nondeterminism returns.

The grant rule relies on these invariants inside each LP:

- **I1** An LP's clock moves only inside its clock owner's CA call (TAR/NER). In a DP
  group's LP each member's clock moves only inside its own owner's call, to the common
  grant (above).
- **I2** In a CA call the owner holds no lock another thread of its LP needs; otherwise a
  handler thread deadlocks. Checked at `f87413a7a` by searching `atom/model_engine`,
  `atom/entrypoints/openai`, `atom/distributed` and `atom/utils` for
  `threading.Lock/RLock/Condition/Semaphore`, the serving-path locks are
  `PrefillScheduler._pending_lock` (`scheduler.py`) and
  `DecodeScheduler._prefill_lock`, both RapidServe;
  `CoreManager._lb_lock` (`engine_core_mgr.py::CoreManager._init_shared_state`) and `_control_send_lock`;
  the metrics exporter's `_lock` (`metrics/exporter.py::AtomMetricsExporter`); and the KV-event
  publisher's `_lock` (`distributed/kv_events.py::ZmqEventPublisher`). Each is held only in a short
  `with` block, none around a CA call site; an AST test asserts no TAR/NER call sits
  lexically inside a `with ...lock` block.
- **I3** One sending thread per channel (a ZMQ socket is not thread-safe either).
- **I4** Receiver and handler threads produce no cross-LP message; only the clock owner
  does, directly or by registering on a `RelayQueue`. Checked: the engine input thread,
  the frontend output thread, `_recv_prefill_done` and `_recv_block_assignments` send
  nothing inside the simulation window, and the simulated KV connector sends only from
  scheduler-side hooks on the step loop (D6).

#### Client runs: termination in a federation open to clients

A server outlives its client and serves client runs one after another, each a client
started apart (`13` D80). The finish above is termination detection, and on such a
server it detects less than the end of the run.

**The PDES concept.** A distributed computation has terminated when every process is
passive and no message is in transit (Dijkstra and Scholten 1980; the counting form,
Mattern 1987). The CA is the centralised detector: every LP's state reaches it in a
request and it counts every message, so the finish's test is that predicate. It means
termination only in a closed system. A server open to clients is not one: after the test
holds, a new client can still send. HLA's federation management is the standard shape
for that case, and Compass takes it as it stands:

- **A federate joins and resigns** (Join and Resign Federation Execution). The traffic
  LP is the run's *client*: the CA is built naming it, and it alone joins and resigns.
  Server LPs join once, at startup.
- **A late joiner regulates time from a point none of its messages can precede** (Enable
  Time Regulation hands it its logical time). A client binding gets in the bind's reply
  the client row's clock and the next sequence number of each channel out of it. Its
  first send arrives at that clock plus the channel's lookahead, in no LP's past, and
  sequence numbers run on across client runs, so counting stays exact and no receiver
  resets anything.
- **A federation ends by an explicit act, not by quiescence** (Destroy Federation
  Execution). Here the act is stated before the run: the run file's `client_runs`, the
  number of client runs the server serves, 1 when absent and `null` for a server that
  serves until it is stopped.
- **A boundary every federate acts on is a synchronization point** (Register, Announce
  and Achieve Synchronization Point). The end of a client run is one.

**The rule.** When every LP waits and none is grantable, the open client run ends at
`E`, the latest LP clock.

1. If it is the last client run the run file declares, the CA finishes the run as above.
2. Otherwise the CA announces the end in one round. The client is granted `+inf`, which
   resigns it, and on which it raises as above unless each request it sent has its final
   response. Every server LP, and every member of one, is granted its own clock with
   nothing released, marked as the end of client run `k`, counted from 0. It hands in its
   part of the summary and asks for time again, which achieves the point. No clock moves.
3. The client's row then runs at `E`, as the first client's runs at 0 before it binds,
   holding every LP at `E + D(client->i)`. The CA cannot find every LP waiting again until
   a client has bound, so it announces each end once and grants nothing between client
   runs: every daemon deadline stays above `H`, and virtual time stands at `E` until the
   next client's first send.
4. The connection of a client granted `+inf` may close and the next client binds the same
   LP. A connection that closes at any other time still ends the run
   (`clock_transport/service.py::_Server._loop`). The bound on simulated time bounds the
   server's clock across client runs.

**What a later client run meets.** Virtual time does not pass between client runs, so
the next starts at `E` with every server LP idle, as a fresh server's first starts at 0
(`13` D80). What carries over is state: the prefix cache earlier client runs left
(`03` D13); the phase of daemon timers, which make no scheduling decision; request ids,
which pick the token a simulated step reports
(`atom/compass/runner/step_output.py::DeferredTokenStream`); and the stream
detokenizer's audit count, which charges a decode to the first stream update of a fresh
server and to every `audit_every`-th after it
(`atom/entrypoints/openai/streaming_dispatch.py::_DeltaReuse`). Traces sharing no prefix
block with an earlier client run take the steps a fresh server takes, shifted by `E` to
rounding; their stream differs from a fresh server's only by the audits that fall
elsewhere.

**Its summary.** Once every server LP has handed in at the end of client run `k`, the
CA's host writes the step table and client run `k`'s summary: its index, start and end;
steps, predicted seconds, the seconds by method, refusals and
`coverage_report` over its own steps, as each LP's hand-in less its hand-in at client run
`k - 1`'s end; and final clocks, grants and wall seconds as the server's at `E` (`13`
D80). The finish also writes the server's summary over its whole life (D3.5). A server
stopped between client runs, by a kill, has written every ended client run's summary.

**Not chosen.**

- *Ending the client run when its connection closes.* A close is not a time, and the
  clock cannot tell a lost message from a finished run (#534); only the test above says
  the client's work is done.
- *Restarting virtual time at 0 for each client run.* Server LPs keep timestamps in their
  state, a request's arrival and their timers among them, which a restart would put in
  the future.
- *Reading the hand-ins only at the finish.* Their counts would not split by client run,
  and a server stopped by a kill would leave no summary of the client runs it served.

#### Lookahead sources

Every one is physical and configurable, which is also a project requirement
(interconnect must be configurable and not read from a device). Lookahead is stated
**per channel**, not per LP pair: one pair of LPs can have several sockets between them,
and one LP can span several processes. A channel is one real communication path from a
sending endpoint to a receiving endpoint, named `src->dst:kind#inst`. Only channels the
real deployment already has: the simulation adds none, and stamps ride in existing
headers or fields. A component the simulation replaces reproduces the real component's
channel — same endpoints, message and trigger. The table is the channel list: wrapping an
unknown channel name raises.

Receive mode: **thread** — a transport or handler thread receives, and the owner releases
each message in `_step_through` and waits for it to be handled; **inline** — the owner
receives it itself, at its own receive point.

| Channel | Sender | Receiver (mode) | Lookahead |
|---|---|---|---|
| `traffic->frontend:http` | Compass traffic source | event loop (inline) | modelled admission delay. Prior work measured 13.7 ms end-to-end, worth ~4 points of TTFT. Path-specific; the values and what each was measured on are in `channels.py::ADMISSION_DELAY_S`. 1P+1D adds the router hop. |
| `frontend->traffic:stream` | event loop, writing the SSE stream | traffic source (inline) | stated return delay |
| `frontend->engine:request#dpN` | event loop, `engine_core_mgr.py::CoreManager._send_request` | engine input thread, `poller.poll()` in `engine_core.py::EngineCore.process_input_sockets` (thread) | stated IPC delay |
| `frontend->engine:control#dpN` | event loop, `engine_core_mgr.py::CoreManager._send_control`. Its second writer, the frontend output thread sending SHUTDOWN (`process_outputs_socket -> CoreManager._shutdown_engine_core_rank`), runs only outside the window | same (thread) | stated IPC delay |
| `engine->frontend:output#dpN` | step loop `put` on the `RelayQueue`; output thread sends (`engine_core.py::EngineCore.process_output_sockets`) | frontend output thread, `poller.poll()` in `engine_core_mgr.py::CoreManager._create_output_thread` (thread) | stated IPC delay |
| `frontend-P->frontend-D:relay` | frontend-P's event loop writes prefill's JSON; the router moves `kv_transfer_params` into the decode request (D2) | frontend-D's event loop (inline) | the router's per-request forward cost (D7) |
| `engine-D->engine-P:kv_write_req` | decode step loop, in the scheduler-side hook `update_state_after_alloc` (`scheduler.py::Scheduler._notify_connector_after_prefill_alloc`); Mooncake's write request, sent from the engine process rather than the worker | prefill step loop, drained in `process_completions` (`scheduler.py::Scheduler._update_from_kv_xfer_finished`) (inline) | stated request latency (D6) |
| PP stage to stage: `meta`, `tokens`, `kv_status` | stage loop, `pp_transport.py::PPStageTransport.send_metadata` / `send_tokens` / `send_kv_status` | the stage loop's own poll and receive (inline) | stated IPC delay: each is one ZMQ message, as on the frontend channels; the head's batch to stage `k` is `meta#pp<k>`. The hidden states between stages are `pp_data` (`15` D91). |
| `engine-P->engine-D:prefill_done` (RapidServe only, in neither M1 nor M2) | prefill step loop, direct send in `engine_core.py::PrefillEngineCore._process_engine_step` | decode handler thread `DecodeEngineCore._recv_prefill_done`, `sock.recv()` (thread) | stated IPC delay |
| `engine-D->engine-P:block_assignment` (RapidServe only, in neither M1 nor M2) | decode step loop, direct send in `engine_core.py::DecodeEngineCore._send_block_assignment` | prefill handler thread `PrefillEngineCore._recv_block_assignments`, `sock.recv()` (thread) | stated IPC delay |
| `engine->lmcache:lookup#dpN`, `engine->lmcache:transfer#dpN.tpK` and their replies (`ATOM_KV_OFFLOAD=lmcache_mp` only) | D6.1 | D6.1 | stated `lmcache_ipc_s` |

The RapidServe rows exist only under `--enable-rapidserve` (D2), whose prefill and decode
`EngineCore`s are two engine LPs of one deployment; they are the only channels received
by a handler thread. No M1 or M2 channel is.

The 1P+1D channel list at DP1, M1's stage 1b (M2's DP-attention sides carry each engine
channel once per rank, `#dp0` to `#dp7`):

- `traffic->frontend-P:http` and `frontend-D->traffic:stream`, both through the router;
- `frontend-P->frontend-D:relay`;
- `frontend-P->engine-P:request#dp0`, `frontend-P->engine-P:control#dp0`,
  `engine-P->frontend-P:output#dp0`;
- `frontend-D->engine-D:request#dp0`, `frontend-D->engine-D:control#dp0`,
  `engine-D->frontend-D:output#dp0`;
- `engine-D->engine-P:kv_write_req`.

Mooncake's write-done message is not a channel: both ends compute its time (D6). With PP,
Mooncake also has a decode-to-prefill `MSG_RELEASE` channel; the simulated connector
refuses PP (D6).

**Declaring a lookahead floor on every channel is a design commitment, not a constant to
tune later.** Zero lookahead is correct under the grant rule above but serializes
everything.

### Sizing

PP8, 27B, ~10 ms step, 300 s modelled run: ~30k steps x 8 stages ~= **240k grants**. At
~50 us per local IPC round trip, ~12 s of overhead against a 300 s real run — still
~25x. For M1 and M2, with 3-5 LPs, the grant traffic is negligible.

**PP is therefore an efficiency concern, not a correctness concern.** It is also
single-node only (every PP address is ZMQ IPC, `engine_core_mgr.py::CoreManager.__init__`), and ATOM
*rejects* PP+DP and multi-node DP+PP (both in `CoreManager.__init__`). Defer it on scope
grounds.

### Open issues

- Grant RPC latency has not been measured on this box. The 50 us figure is an estimate.
- Whether the CA should be a process or a thread in the API-server process is open. A
  separate process is cleaner for the PD case where the API server is not obviously the
  right host.
- PP's microsecond lookahead will make the CA the bottleneck. When PP arrives after v1,
  re-evaluate option B for the PP sub-graph specifically.

---

## D3.1. Clock Authority deployment and scalability

### Problem

The CA is a logical service. Where it runs, and whether one instance can serve a
deployment that grows from one container to many nodes, is a separate decision from the
protocol itself.

### The finding that settles the sizing: LP count does not scale with hardware

A DP group is **one** LP however many ranks it holds, because it already
`all_reduce`s every step (`engine_core.py::DPEngineCoreProc._sync_dp_state`). A TP group is **one** LP however
wide, because its workers are slaved by a blocking RPC (`async_proc.py::AsyncIOProcManager.call_func`) and hold no
clock. Adding GPUs to either does not create a time domain.

Every ATOM deployment is two LPs: its API server's event loop (the frontend LP) and its
engine's step loops, one per DP rank, together the engine LP; plus one traffic LP per
run (D3). Beyond that, only these create an LP:

1. a **PD role boundary** — prefill fleet vs decode fleet, each its own deployment
2. a **PP stage**, which replaces the engine LP with one LP per stage
3. an **independent replica** behind the router. `.github/scripts/atomesh/pd_server_atom.sh`
   deploys *xP* prefill servers and *yD* decode servers, each an independent deployment
   with its own `Scheduler`, none synchronizing with the others. With more than one
   replica per role the router itself is an LP (deferred; in 1P1D it is transport).

| Deployment | GPUs | LPs |
|---|---|---|
| M1, #627 stage 1: TP1, one server | 1 | **3** |
| M1, #627 stage 1b: TP1 prefill + TP1 decode, two containers on one node | 2 | **5** |
| M2: DeepSeek-V4-Pro 1P+1D, one container per node, TP8 on each side, DP-attention where the nightly cell sets it; the EP8 step the same | 16 | **5** |

After v1 (`16` D95), kept as sizing:

| Deployment | GPUs | LPs |
|---|---|---|
| TP8 + PP4, one server | 32 | **6** |
| 8 prefill + 8 decode replicas, each TP8 | 128 | **34** |
| ... the same with PP4 | 512 | **82** |

The replica rows are traffic + router + 16 deployments of 2 LPs (34), or of a frontend
and 4 stages (82).

### Sizing against a measured workload

The prior 27B cc-traces run executed 106 prefill + 4,346 decode steps over 267 s of
modelled time — about **4,450 events per LP**. A frontend LP is counted at the same rate:
it takes one delivery per engine output step plus its requests' HTTP events, so it is
about as busy as its engine. That is an assumption, not a measurement.

| LPs | Grants per run | CA cost at 50 us (local IPC) | at 500 us (cross-node TCP) |
|---|---|---|---|
| 3 | 13k | 0.7 s | 7 s |
| 5 | 22k | 1.1 s | 11 s |
| 34 | 151k | 7.6 s | 76 s |
| 82 | 365k | 18 s | 182 s |

Against the **cost model on the same steps**: previously measured at **4.3 ms per step**
with the bound allocation carried in the cache key (2.3 ms with a shape-only key, but
that key is unsound — a second valid allocation for the same shape moves 64 of 2,439
operator signatures; 41.7 ms with no cache at all). 4,450 steps x 4.3 ms = **~19 s per
LP, running in parallel across LPs.**

**A grant is therefore ~1% of the per-step cost at local IPC and ~10% cross-node.** The
simulator's own pricing dominates by two orders of magnitude. Optimising the time
protocol before the cost model would be optimising the wrong thing.

Throughput headroom: a Python ZMQ ROUTER sustains roughly 50-200k msg/s; the largest case
above is ~19k/s (365k grants over the ~19 s the pricing takes). The CA's own work per
state change is an LBTS for each waiting LP, each a `min` over the other LPs: O(LPs²),
nothing at the milestones' 3-5 LPs but ~6.7k terms at 82, so measure it before the
replica scale. RTTs pipeline across LPs.

### Options

**A. Single CA with a hierarchy-ready interface.** One instance. Deployed as a thread in
the API-server process for the single-node case, or as a standalone process addressed by
`host:port` for multi-node — the same shape as `--data-parallel-master-ip`. The LP-facing
interface is identical in both.

- *Pros:* zero deployment cost single-node; one address to configure multi-node; one
  choke point that knows global state, which is where the safety assertion, the stall
  diagnostic, and a "who is holding up the simulation" query naturally live. A node-CA can
  later slot between an LP and a root CA without either side changing, because a node-CA
  runs the same algorithm as the root — the CA is recursive.
- *Cons:* a single point of failure. Acceptable: its failure is a hang, which is loud,
  and a simulation is a batch job rather than a service.

**B. Hierarchical CA built now.** Node-local CA per physical node, root CA over them.

- *Pros:* cross-node grant traffic drops from O(LPs) to O(nodes). The correct end state
  beyond ~100 LPs across many nodes.
- *Cons:* two levels to debug and a global-min protocol to get right, at a scale no
  milestone reaches. The sizing above says this is premature.

**C. No CA — carry time on a Compass-owned `torch.distributed` collective.**
`all_reduce(MIN)` over a Gloo group built with ATOM's existing
`stateless_init_torch_distributed_process_group` (`utils/distributed/utils.py`).

- *Pros:* no new process; scales exactly as `torch.distributed` scales; the collective
  **is** the barrier, so the safety property is structural rather than asserted.
- *Cons, and the first is probably disqualifying:* a Gloo `all_reduce` requires every
  rank to call it, and **a blocked LP cannot**. An LP parked in `zmq.recv` awaiting a
  peer's message would need a separate time-service thread sharing mutable `now`/`next`
  with its main thread under a lock — a concurrency hazard placed exactly where silent
  failure is most expensive. It also gives up the single choke point for the safety
  assertion and the stall diagnostic.

**D. Single CA with no hierarchy provision.** Least code now; if a hierarchy is ever
needed the LP-facing interface changes, which touches every LP.

### Decision

**Option A — a single CA with a hierarchy-ready interface.**

Requirements that follow, and they are cheap only if honoured from the start:

1. The LP-facing interface must not name the CA's location or its level. An LP knows an
   endpoint, nothing more.
2. The CA's own interface to *its* peers must be the same as its interface to LPs, so a
   node-CA is a CA whose "LPs" are other CAs.
3. The channel table must be addressable by LP and channel name, not by index, so
   inserting a level does not renumber anything.
4. Partition guidance for a future hierarchy is already implied by the topology, and
   matches what SimAI's auto-partitioner does (it cuts p2p links whose delay is at or
   above the median): **PP stages have microsecond lookahead and must stay under one
   node-local CA; PD role boundaries have millisecond lookahead and are the cheap links
   to cross a node.** The hardware already lays out that way.

### Where the CA runs: both, selected by one flag

Settled: **two deployment forms of one implementation, not two implementations.**

| Form | When | How it is reached |
|---|---|---|
| **Co-hosted in the API-server process** (default) | single-container runs — M1's stage 1, M2's first gate (#627 stage 2), and every calibration or debug run | the LP's endpoint resolves to an in-process transport; no socket, no extra process to start or reap |
| **Standalone CA server** | multi-container and multi-node runs — M1's stage 1b, M2, and anything with an Atomesh router between roles | `--compass-clock-endpoint tcp://host:port`; the CA is its own process, started before the engines |

This is cheap precisely because requirement 1 above already forbids the LP-facing
interface from naming the CA's location. The two forms differ only in which transport
the endpoint resolves to. The in-process form is *not* a shortcut that skips the
protocol — the same grant rule, the same channel table, the same `(channel, seq, arrival)`
stamps —
so a bug found in one form is a bug in the other, and the cheap single-container runs
are a real test of the expensive multi-container path.

Default co-hosted rather than always-standalone because the overwhelming majority of
runs are single-container, and a standalone CA there is one more process to start,
supervise and leak. Standalone for 1P+1D because neither container is obviously the
right host and making one of them the clock owner would give the two roles asymmetric
failure behaviour that the real deployment does not have.

**What must be true for this to stay one implementation:** the co-hosted form must not
acquire an in-process fast path that bypasses what detector (1) above reads: the
`(channel, seq, arrival)` stamp on each frame and the `(channel, seq)` set each grant
releases. If either stops travelling in-process, the single-container runs stop testing
the property they are supposed to be testing.

### Open issues

- Grant RPC latency has not been measured on this hardware. The 50 us figure is an
  estimate and should be measured before it is quoted. Note this matters only for the
  standalone form; co-hosted grants are function calls.
- The CA is the only component that knows every LP's virtual time. It should therefore
  own the global timeline log and the stall diagnostic. That makes it an observability
  component as well as a coordination one, which is a benefit but also means its output
  format is part of the acceptance evidence and should be designed, not improvised.

---


---

## D3.4. Determinism: what a simulated run must reproduce, and what it need not

### Problem

`08` T26 asks for bit-reproducibility as a test, and the whole paired-comparison protocol
leans on it: a simulated side that disagrees with *itself* between runs cannot be compared
with a real side at all. But a multi-process simulator has several genuine sources of
non-determinism, and demanding that all of them vanish would mean rebuilding the process
model that D1 deliberately kept.

### The distinction that makes this tractable

> **Wall-clock interleaving may vary between runs. The sequence of
> `(LP, virtual time, event)` may not.**

Nothing downstream reads wall time. Every graded number, every step-table row and every
artifact is a function of virtual time, so reproducibility is a property of the *virtual*
schedule and of nothing else. Two runs may take different real durations, issue grants in
different real orders, and schedule their threads differently, and still be identical
runs.

### The sources, and what each needs

| Source | Deterministic? | What makes it so |
|---|---|---|
| **Grant order at the CA** when two LPs are eligible at the same virtual time | **not by default** — real arrival order decides | **Tie-break by LP identity, never by arrival order.** The CA holds a total order over LP ids and grants in it; messages released at one instant go in `(channel, seq)` order. This is the single most important rule here, and it costs one comparison. |
| **Cost model output** | yes, if the backend is pure | no iteration over a `dict` or `set` whose order depends on insertion or on object identity; a fixed summation order over IR nodes, since float addition is not associative |
| **Speculative acceptance draw** | already handled | `14` D83: one host draw seeded from the step counter, not `world_size` draws that must agree |
| **`dict` iteration** over request or block ids | **yes** | insertion-ordered since Python 3.7, and insertion order is the schedule's order, which is itself deterministic |
| **`set` iteration** | **NO — and this is the trap** | see below |
| **Thread scheduling inside an LP** | irrelevant **only under D3's delivery rules** | without in-transit counting and TSO delivery (D3 grant rule), a race inside the LP decides which drain sees a message and at what logical time a handler thread runs; with them, a handler runs only on a released message, at its timestamp, while the owner waits |
| **Deliberately-real clock reads** (transport; bounds the CA cannot reach) | irrelevant | they never enter the virtual schedule. Metrics are not in this row (D1 detector (3)) |

### `set` is the trap, and it is worse than "unordered"

`dict` preserves insertion order; **`set` does not, and its order is not even stable
between processes.** Iteration order follows hash values and the insertion history that
produced the table's layout. For the ids Compass actually holds:

- **string ids** — request ids, block hashes, LP names — are hashed with
  **`PYTHONHASHSEED` randomisation**, on by default. So a `set` of request ids iterates
  in a *different order in every process*, and two runs of one configuration diverge with
  no code change at all.
- **integer ids** hash to themselves, so a small-int set looks stable — which is worse,
  because it works in testing and then reorders the moment the values spread out or the
  table resizes.

This is the one source in the table that produces a divergence with nothing to point at:
no error, no warning, and a diff between two runs of identical code.

**Rule: no `set` iteration on the simulated path.** Where set *semantics* are wanted, use
a `dict` with `None` values as an ordered set, or sort explicitly at the point of
iteration. Membership tests against a `set` are fine — it is only iteration that leaks
order.

This is mechanically checkable, so it joins the CI clock-source lint of D3.2 as a second
rule in the same check rather than a convention anyone has to remember.

### The rule, and the test

**Rule:** the CA's grant order is a total order over LP identity, and ties at one instant
fall to `(time, LP id, channel, seq)`; the cost backend is a
pure function of its `batch_view`; and no simulated-path code iterates a `set` or any
container whose order depends on object identity.

**Test** (this is `08` T26, now with a mechanism): run the same configuration twice and
diff the step tables byte for byte. It is CPU-only, it needs no GPU, and it belongs in CI
from the first stage that produces a step table — because the failure it catches is one
that gets *much* harder to localise once several tracks are contributing.

**What the test does not cover:** a run that is reproducible and wrong in the same way
twice. Determinism is a precondition for comparison, not evidence of fidelity.

---

## D3.5. Observability: the Clock Authority owns the timeline

### Problem

D3.1 records that the CA is the only component that knows every LP's virtual time, and
that it should therefore own the global timeline log and the stall diagnostic — and that
*"its output format is part of the acceptance evidence and should be designed, not
improvised."* This is that design. It is deliberately small.

Note the boundary with `11`: that topic covers ATOM's **Prometheus engine metrics**, which
describe the *simulated system*. This covers the **simulator itself** — whether the run was
valid, not what the modelled engine did. Different consumers, different lifetimes, no
overlap.

### Three outputs, and nothing else

**1. The timeline log.** One append-only record per granted advance:

```
  lp_id . virtual_time_from . virtual_time_to . event . detail
```

Written by the CA because it is the only place with a consistent global view, and the
only place where the ordering is authoritative. It is what makes a causality report
actionable: when the straggler check (D3.2) fails, the log already contains both LPs'
histories up to the violation.

**2. The stall diagnostic.** A stall the CA can see — every LP waiting at the CA, none
grantable under the strict rule — is not a failure: unless D3's finish applies, it
takes D3's recovery branch, and the timeline marks that grant as a recovery grant, the
one kind whose TAG guarantee is weaker (D3). A stall the CA cannot see can only be a
fault. The run keeps waiting, and after `DIAG_S` = 30 wall seconds without progress the
CA prints once, for every LP: its virtual time, its state (`running` / `TAR` / `NER`)
and target, the registered messages not yet delivered to it, and the `N[j] + D(j->i)`
term that bounds its grant. An LP stuck delivering a message it released prints its
channel, its sequence number and every thread's stack. An LP that has waited at the CA
for `DIAG_S` also prints the buffered frames no grant has released, which is where a
frame whose send was never registered sits when no later drain runs the straggler check.
None of these aborts. This is what makes a hang
diagnosable rather than merely silent.

**3. The run summary**, written once at the end of every run, an early end (an abort on
refusal, an engine that raises) included, and once at the end of every client run (D3,
Client runs), and carried in the run artifact:

| Field | Why |
|---|---|
| `schema_version` | two summaries are compared only at one format |
| `source_root`: ATOM's tree, or a snapshot's stamped commit, read when the authority is built | two summaries from different code are not one result; a tree with neither `.git` nor a stamp is refused before the run |
| grants issued, per LP | the protocol's own cost; the number that says whether PP degree is affordable |
| wall seconds vs simulated seconds | the speed result (`08`), and the only place the ≥5x target is measured |
| lazy traces: count and wall seconds | `02` — they consume real time inside a simulated run and must not silently degrade the speed result |
| causality detector state | straggler count (must be 0), stall diagnostics printed, clock-lint status |
| refusals: count and distinct reasons; predicted seconds by method (`seconds_by_method`), each with its step count and share; `coverage_report` | `08` D50.1: any refusal makes the run a coverage report; analytical and declared seconds are reported, not gated |

### Rules

1. **The timeline log is off by default and costs nothing when off.** It is a debugging
   and evidence artifact, not a per-step tax; at millions of grants it would dominate a
   fast run. The *summary* is always written, an early end included — it is small and it
   is what the acceptance gate reads.
2. **The summary is part of the run artifact, not a log line.** `13` D81's rule applies:
   recorded by value, so a result can be audited without the machine that produced it.
3. **Each quantity is counted in one place.** The summary, the step table and the
   timeline read it and do not count it again. The refusal denominator once counted the
   engine's clock rows while the refusals were counted per priced step.
4. **The summary does not copy the simulated system's latency and throughput.** TTFT,
   ITL and request counts stay in the benchmark's own output and ATOM's metrics (`11`).

### Open issue

- The timeline log's volume at PP degree > 1 is unestimated. Grants scale with PP stages
  and with the microsecond lookahead between them, so the log could be very large exactly
  where it is most wanted. Recorded as **T70**.
## D4. The interception contract: which waits must be touched

### Problem

ATOM's serving path contains many distinct synchronization points: blocking ZMQ
recvs, bounded pollers, queue gets with timeouts, Gloo and NCCL collectives,
`multiprocessing` barriers and joins, busy-waits, and literal sleeps. "Intercept every
blocking call" is the obvious reading of what a virtual clock demands. It is also the
reading that turns this project into reimplementing a scheduler on top of the OS
scheduler.

### The contract

Each site is answered by **the PDES mechanism a simulated run applies there**, not by the
shape of the wait. What decides it is which LP owns the waiting thread and what the wait
means for that LP's clock. Each LP process has one clock owner — its step loop or its
asyncio event loop; a DP group's LP has one per rank — and only a clock owner moves the
LP's clock (D3); every other thread in the LP is transport. Every site maps to one of the mechanisms K1–K9, and **none is left
undecided**.

| Class | Mechanism | PDES rule | Representative sites |
|---|---|---|---|
| K1 | Event cost | TAR: `advance_to(now + d)`, `d` from the cost model or a resource station; for a forward, from the runner's host and device recurrence (below) | the forward, the idle DP rank's dummy batch, KV-transfer completion, tokenization, the DP lockstep `all_reduce`, multimodal preprocessing |
| K2 | Clock read | returns the LP's logical time | the arrive / leave / first-token stamps, `_passed_delay`, the waiting-prefill age |
| K3 | Idle point | NER: `next_event(t, t_daemon)` | the step loops' spin, the PP bounded polls, offline `get_output`, a step-loop wait on an lmcache reply (D6.1) |
| K4 | Channel send | timestamped send, logged on the clock owner | PD direct sends, PP sends, output sends through the relay queue |
| K5 | Channel receive | counted at the wait point, then TSO delivery | the engine input thread, the frontend output thread, the PD handler threads, the PP receives and zero-bound drains, the PP `flush_pp_send` |
| K6 | Wait inside one LP | LP aggregation: never reported to the CA | TP worker RPC and barriers, frontend coroutines awaiting their own process's data, control commands' calls to workers, the DP group's reductions (the `PrefillDelayer` all_reduce, and the device-end `all_reduce(MAX)` and step-target `all_gather` in the predicted forward) |
| K7 | Virtual timer | the timer runs on the LP clock | idle KV drain, metrics push and refresh, Anthropic ping, keep-alive, the silence warning |
| K8 | Real bound | the CA cannot reach it, or it stays real on purpose: configuration | the Rust router's bounds and health check, process-death detection, the control-command reply deadline |
| K9 | Outside the model | outside the simulation window, inside replaced code, cannot park, or a timing only a log line reads | startup and shutdown, the replaced runner and RDMA backends, collectives inside the real forward, text scanners, the step-loop timings that only feed a log line |

Each site's mechanism is the `mechanism` field #476 adds to its row in
`atom/compass/audit/sync_sites.json`, produced by the scanner beside it. A row's
first classification (#53) maps naturally to A→K1, B→K5, C1→K8, C2→K7, ignore→K9; the
rules below say where a site moved off that map.

The rules that settle the boundaries #53's categories left in the wrong place:

- **Inside or across an LP, not the same or another process.** A wait whose peer is in
  the same LP is K6: the LP never reports it to the CA, and its duration belongs to the
  event the clock owner is pricing. The TP group is one LP, so the engine's RPC to its
  workers — `call_func(..., wait_out=True)`, which parks in `self.outputs_queue.get()`
  in `AsyncIOProcManager.call_func` (`atom/model_engine/async_proc.py`) and carries every
  forward — is K6 although its peer is another process; the forward's duration is
  charged by the clock owner's TAR. #53 drew this line at the process and put such waits
  in B. K5 holds only the receives that cross an LP: those, the PP zero-bound drains, and
  the PP idle flushes.
- **Zero-lookahead couplings go inside one LP.** A barrier or collective, a synchronous
  RPC, or a live read of the peer's state lets one side's action at *t* change the other
  at *t*. Split across LPs, every round of such a coupling serializes through the CA, and
  ATOM's real collectives would bypass it ([D3](#d3-virtual-time-coordination-protocol)
  sizes both). So the TP workers and every rank of a lockstep DP group sit in the engine
  LP, and a startup assertion refuses a process group whose ranks map to more than one
  LP.
- **A DP step is one TAR per member release and per member pull; each member returns at
  its own pull** (D3, #988, #1267, #1345). Each rank charges its own forward by the recurrence below, from the group's
  device end, the `max` over ranks of each rank's absolute device end by one
  `all_reduce(MAX)`; the ranks exchange the step targets by one `all_gather` inside the
  predicted forward, and every rank calls the CA as a member of the LP (D3). With no host
  time an output rank's target is the `max` of the step seconds (D3). Both collectives
  are K6 and carry no cost (`15` D90). The lockstep `all_reduce`
  (`DPEngineCoreProc._sync_dp_state`, `atom/model_engine/engine_core.py`) keeps its payload; it is K1 rather than ignored because it has a cost of its own, not
  because the step is decided there. **Second-order bound:** the shared device end
  leaves out each rank's own tail after its last cross-rank collective; with no host
  time a fast rank's output is stamped at the group's `max`, late by at most the
  difference between its own cost and the `max`. The step is split with in-step TAR
  only if that is measured to affect TPOT.
- **A bound in Python is a virtual timer; only what the CA cannot reach is a real
  bound.** #53 split bounded waits by purpose, failure detector (C1) against pacing (C2).
  Both run on the LP clock now (K7, D5). The K8 row above lists what stays real. The C1 bounds inside the TP group
  (`AsyncIOProcManager.process_output_sockets`, `process_kv_output_sockets` and
  `call_func_with_aggregation`) are K6: real-time bounds on in-LP transport that the
  simulated runner answers in real milliseconds, so they never fire and stay as they are.
  The one exception is the non-blocking drain in `call_func_with_aggregation`: a reply
  still in flight would move a transfer's report to a later step, so on a simulated run
  `clock.reply_queue` (`atom/utils/clock.py`) makes each worker's reply queue wait for
  it, up to `REPLY_WAIT_S`.
- **Sends are counted.** #53 ignored sends because they do not park. Under D3 a send is
  one end of the in-transit message count: it carries the timestamp and is logged on the
  clock owner, so the sends are K4. A direct send (the clock owner calls `send`:
  `PrefillEngineCore._process_engine_step` and `DecodeEngineCore._send_block_assignment`
  in `engine_core.py`, the `PPStageTransport` sends in `atom/distributed/pp_transport.py`)
  is stamped by a socket wrapper. The engine's output sends go through its output thread
  (`EngineCore.process_output_sockets`), so `output_queue`, created in
  `EngineCore.__init__`, becomes a relay queue that stamps at `put`, on the step loop:
  the timestamp is the step's, not the moment the output thread wakes. Inside the
  simulation window every item put is sent — the one skip is a list of `EXIT_ENGINE`
  sequences, which exist only at shutdown — so there are no null messages, and an item
  never sent raises `UnsentRelayItem`.

#### K1–K3 — what #53 called category A, as measured

- **K1**, the forward pass: `EngineCore._process_engine_step_inner` (the main step),
  `PrefillEngineCore._process_engine_step` and `DecodeEngineCore._process_engine_step`
  (the two halves of RapidServe) in `atom/model_engine/engine_core.py`, and
  `PPEngineCoreProc._pp_head_step` and `_downstream_busy_loop` (the PP head and a
  downstream stage) in `atom/model_engine/pp_engine_core.py`
- **K1**, the idle rank's empty batch, `DPEngineCoreProc._execute_dummy_batch` — it
  consumes a step and is charged like any other
- **K1**, KV-transfer completion (D6): `EngineCore._poll_kv_transfer_progress`,
  `PPEngineCoreProc._poll_kv_transfer_progress` and `_poll_and_send_kv_status`
- **K1**, **tokenization**, at the `run_in_executor` hand-offs `06` D33 names, in
  `generate_async`, `generate_async_multimodal`, `generate_async_fanout`,
  `setup_streaming_request` and `setup_streaming_request_fanout`
  (`atom/entrypoints/openai/api_server.py`): a multi-server resource station in the
  frontend LP whose completion times come from D33's service model. The multimodal
  preprocessing hand-off in `chat_completions` is K1 too, a service job on the same
  station charged a run-file price per image (#1089). A tokenizer call on the event loop thread itself, such
  as the final `decode` of a non-streaming completion, is no station job: the loop is the
  LP's clock owner, so the call is a TAR on the loop, `advance_to(now + d)` with `d` from
  the same service model, and nothing else on the loop runs until it ends.
- **K1**, the DP lockstep `all_reduce` (`DPEngineCoreProc._sync_dp_state`), for its own
  cost, per the DP rule above
- **K3**, **the idle jump, which is a real site in ATOM even though the name this list
  once gave it is not.** `Scheduler._advance_to_next_arrival` does not exist here — but
  the loops it would have served do, and each spins rather than waits when there is
  nothing to run, because `pull_and_process_input_queue` drains with `get_nowait` and
  nothing else in the turn blocks: `EngineCore.busy_loop`, `DPEngineCoreProc.busy_loop`
  and `PPEngineCoreProc._head_busy_loop`. D8's measurement of the prior design — first
  real step is tick 1, first simulated step is tick **89,336** — is this loop counted.
  Each becomes `next_event(t, t_daemon)`, `t` the LP's earliest essential local event and
  `t_daemon` its next daemon deadline (the metrics push). The PP bounded polls
  (`PPStageTransport.recv_tokens` and `recv_metadata`) follow the owner's ruling (b) on
  #468: a stage holding no pending send and no pending KV work polls on a daemon
  deadline, `next_event(inf, now + bound)`, so an idle stage keeps no run alive; one
  holding either polls on an essential one, `next_event(now + bound)`, so its idle flush
  and KV status report still run on time. The stage loop decides which, since only it
  knows what is pending; that choice is #468's. Until then the shim asks the daemon form
  on every bounded poll, which is the same while the simulated runner holds no pending
  send; and
  the offline driver's `CoreManager.get_output` (`atom/model_engine/engine_core_mgr.py`)
  is `next_event(inf)`.
- **K2**, `Scheduler._passed_delay` / `--scheduler-delay-factor`,
  `Scheduler._oldest_waiting_prefill_age_ms` feeding `PrefillDelayer`, and the stamps the
  result reports: `seq.arrive_time` in `InputOutputProcessor.preprocess_fanout` and
  `req.leave_time` in `InputOutputProcessor.postprocess` (`atom/model_engine/llm_engine.py`),
  `seq.first_token_time` in `Scheduler.postprocess` and `DecodeScheduler.on_prefill_done`
  (`atom/model_engine/scheduler.py`) — D5 lists these as business logic

**One entry of the original list is not ATOM code at all.** The arrival gate (D8) is
something Compass adds: `_arrival_barrier_unmet`, `compass_workload_size` and
`ARRIVAL_BARRIER_TIMEOUT_S` return nothing on the whole `atom/` tree, so there is no
site to intercept, only a mechanism to build.

#### K5 and K6 — what #53 called category B

**Nothing is annotated.** #53's category B wrapped each cross-process wait in
`declare_blocked()` / `declare_running()`. Under D3 an LP's idleness is inferred by the
CA from its clock owner parking at an idle point with nothing left to release, so no
thread declares anything:

- **K6**, `call_func(..., wait_out=True)`, per the first rule above, and the out-of-band
  control commands' calls to the worker in the `EngineUtilityHandler._handle_*` methods
  (`atom/model_engine/engine_utility.py`); D5 says which of those commands a simulated
  run refuses.
- **K6**, the streaming endpoints' collector reads in `stream_chat_response` and
  `stream_chat_response_fanout` (`serving_chat.py`), `stream_completion_response` and
  `stream_completion_response_fanout` (`serving_completion.py`), and the same call in
  `anthropic_messages` (`api_server.py`): a frontend coroutine awaiting data its own
  process produces.
- **K5**, the engine's input thread (`EngineCore.process_input_sockets`), the frontend's
  output thread (`CoreManager._create_output_thread`), and the RapidServe handler threads
  (`PrefillEngineCore._recv_block_assignments`, `DecodeEngineCore._recv_prefill_done`).
  A receiving thread is counted when it returns to its wait point, so every branch is
  counted, including a frame dropped on a decode error; the poller is wrapped where it is
  created, and a handler thread's socket is wrapped. Then **TSO delivery**: a received
  message is held at the wait point until the clock owner's `advance_to` walks past its
  arrival time, and released one at a time in arrival order.
- **K5**, the PP idle flush, `flush_pp_send` in `PPEngineCoreProc._pp_head_step` and
  `_downstream_busy_loop`: it waits on the host for the gloo parts of the stage's
  previous send, whose completion a rendezvous gloo part learns from the next stage. The stage loop receives that send's
  `stage(k+1)->stage(k):pp_ack#dp0` inline before the call; an eager (small) send has a
  local completion and no ack, and the loop advances to it by K1's rule. The forward's
  own wait for the previous send is the same receive, made at the forward call site after
  its compute TAR. The shutdown call in `_downstream_busy_loop` is the same call as the
  idle one and shares its answer; it runs outside the simulation window, so sharing
  changes nothing. The head's shutdown call in `_head_busy_loop` is outside the window
  too and stays K9. The real `isend` and `wait()` inside the replaced runner are K9
  (`15` D91 Q3).
- **K9**, `CoreManager._wait_for_all_ready_signals`, reached only from
  `CoreManager.__init__`: waiting for READY is outside the simulation window. #53 kept
  these as B with the contradiction recorded; the LP rule resolves it.

**On the send side (K4), "it cannot park" is true of most of them and has to be said per
socket, not once.** The PP stage loop also registers its `pp_data` and `pp_ack` sends at
the forward call sites (`15` D91). Of the sends the scanner finds, the ones built by
`make_zmq_socket` (`atom/utils/__init__.py`) carry `SNDHWM=0` and provably cannot park.
The rest are bare `ctx.socket(...)` and keep ZeroMQ's default thousand-message bound: the
`PPStageTransport` sends, the disagg bootstrap sends, and — the two that matter at
runtime — `PrefillEngineCore._process_engine_step` on the **prefill step loop** and
`DecodeEngineCore._send_block_assignment` on the decode one, whose sockets each class's
`_init_disagg` creates bare. Their protocol is one message per sequence and the peer
drains it every tick, so a thousand-message backlog is not reachable in a run; the rows
say that rather than claiming the bound does not exist.

#### Which mistakes fall on the loud side

A channel site this table got wrong — a send or a receive — leaves an LP waiting: a
message logged but never delivered, or an LP stuck in a primitive the CA cannot see. Such
a wait produces no simulated time, so the run waits for it, prints one stall diagnostic
naming the channel and sequence number after `DIAG_S` wall seconds, and keeps waiting
(D3). The opposite failure — advancing past a message — is what the send count and the
strict grant rule rule out. So for sends and receives, mistakes fall on the loud side.
The other mechanisms do not: a missed event cost charges no modelled time, and a clock
read or timer left real stamps wall time, both silent at run time. The clock-source lint
(D9) catches a clock read left real; validation against real runs (`08`) catches a
missed event cost.

### A forward's charge: host and device, composed per runner

Owner ruling on [#815](https://github.com/jgong5/ATOM/issues/815), 2026-10-11. It
supersedes ruling 2 of
[#627's 2026-10-10 plan](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740),
under which the step linear law fitted the step period with host gaps in its intercepts.

**A backend prices device time only.** Its answer is one forward's device busy time `D`,
with no launch gaps (`02` D11, `09` D54). With MTP, `D` splits at `forward_done_event`
into `D_target` and `D_draft`; without it `D_draft = 0`. Host time is priced by the
host laws (`09` D54), never by a backend, and the runner composes the two into the
forward's K1 charge by one recurrence. There is one recurrence per runner: one per PP
stage and per DP rank, with a TP group modelled on its rank 0.

**Host segments**, serial on the host timeline, waits excluded:

- `E`: `scheduler.postprocess()`, `scheduler.schedule()` and the dispatch to the worker,
  measured as the gap from one forward's exit to the next one's entry. It is the
  recurrence's first term.
- `W_pre`: prepare (`ModelRunner.prepare_model`), which waits only at its two ends;
  under MTP, an output forward's prepare up to R1's wait.
- `W_inputs`: under MTP, an output forward's prepare after R1's wait: the input ids,
  `prepare_inputs` and the drafter's staging. A run without MTP never runs it; an MTP
  run whose law lacks it is refused by name.
- `W_launch`: the launch (`ModelRunner.run_model`).
- `W_recv`: on a pipeline stage after the first, the part of `W_launch` that
  `run_model` runs before `recv_intermediate_tensors`; the stage posts its receive
  after it (`15`). It is inside `W_launch`, never charged beside it. A first stage
  never runs it; a later stage whose law lacks it is refused by name.
- `W_post`: the sampler launch and the output (`ModelRunner.postprocess`).

**The recurrence**, for forward `n`, where `now` is the clock owner's time when the
forward is called (the previous forward's exit, or the grant that ended an idle wait) and
`R*` is the release time of a wait rule below:

```
entry      = now + E
h          = max(entry, R4) + W_pre
h          = max(h, R1) + W_inputs             # MTP only: R1 waits before launch
launch_end = max(h, R7 - W_recv) + W_launch    # non-first PP stage: R7, the metadata,
                                               # holds the launch after W_recv
start      = max(end[n-1], h, R7)
target_end = start + D_target
end        = max(target_end + D_draft, launch_end)
exit       = max(launch_end, R1) + W_post      # no MTP: R1 for an output forward
charge     = exit - now                        # the K1 TAR: advance_to(exit)
```

`end` and `target_end` are the runner's device state, kept across forwards; they never
touch the clock. A device-bound forward charges device time and a host-bound one host
time.

**After the return: `E_out`** (owner ruling on
[#1345](https://github.com/jgong5/ATOM/issues/1345), 2026-10-18). `Scheduler.postprocess`,
the output put and the loop's way back to `pull_and_process_input_queue` run after the
forward returns. `clock.charge` advances the reply's `predicted_s`, the forward's exit,
then its `out_s`, `E_out`, so the engine's input pull and output send sit at
`exit + E_out`; the forward's charge stays the forward. `E` stays the recurrence's first
term and, where `out_s` is set, covers the pull to the next entry, `E_in`. A replay sends
a real record's `t_pull_ns - t_exit_ns`; predict sends none until an `E_out` host law is
fitted ([#1480](https://github.com/jgong5/ATOM/issues/1480)), so `E` stays whole and the
pull sits at the exit. Under DP each member's pull, its own release plus its `out_s`,
joins the group's rounds (`group_out`), never the device-end `max`; the member returns
at its own pull and `group_targets` keeps releases only. The put lands at the pull,
after the real put: the records' `t_put_ns` measure that gap.

**What the ruled order implies.**

- **The sampler launch sits after R1.** `ModelRunner.postprocess` launches the sampler,
  and with TP above 1 the `get_tp_group().broadcast` of its tokens, before
  `prepare_sampled_ids` reaches the R1 wait, but the recurrence puts all of `W_post`
  after R1. A device-bound output forward is therefore overcharged by both launches.
  #824's stamps separate the two parts of `W_post`, and #827's per-step residuals detect
  the error.
- **With every host law at zero**, a forward exits at the latest of `now`, R4's release
  and, for an output forward, R1's: in a run of output forwards, forward `n` charges the
  previous output forward's device time and the first output forward charges 0, one
  output forward later than a charge of each forward's own device time.

**The wait rules.** Rule numbers are #815's.

| rule | applies to | where | waits for | ATOM site |
|---|---|---|---|---|
| R1 | output forward, single stage | after launch; before launch with MTP | the previous output forward's `target_end` | `tokenIDProcessor.recv_async_output`, from `prepare_sampled_ids`; with MTP, `tokenIDProcessor.recv_mtp_status_async`, from `prepare_input_ids` in `ModelRunner.prepare_model` |
| R4 | every forward, single stage | forward entry | `end[n-2]` | `ModelRunner._gate_staging_reuse` |
| R6 | every forward, PP | forward entry | this stage's `end[n-pp]` | `ModelRunner._advance_forward_vars`, the `forward_vars` ring |
| R7 | non-first PP stage | device start, host launch | the upstream stage's `end`, and its host send | `GroupCoordinator.recv_tensor_dict`, from `recv_intermediate_tensors` (`atom/distributed/pp_comm.py`) in `ModelRunner.run_model`: the metadata on gloo holds the host, the tensors the device |
| R8 | last PP stage, output batch | after launch | its own `target_end` | `tokenIDProcessor.prepare_sampled_ids`'s `tolist`, on the stage that does not defer output |

The DP barriers are not wait rules: the step-seconds `all_gather` carries each
member's charge, and the CA's member join grants each member's step end in turn (D3, `15` D90). PD's
KV transfer keeps its own model (D6).

Stage 2 builds R1 and R4 ([#825](https://github.com/jgong5/ATOM/issues/825)). R6 and R8
come with the stage sends ([#468](https://github.com/jgong5/ATOM/issues/468)), R7 with
[#1200](https://github.com/jgong5/ATOM/issues/1200): a simulated stage loop receives `pp_data`
at the forward's call, its launch after `W_recv` waits for the metadata's arrival (#1529) and its device start for
the tensors', and a non-last stage sends at its launch end (#1324).
The gate charges a PP record's stages together, step by step, and predicts both waits
between them that ATOM's stage loops hold outside the recurrence: the head's
`recv_tokens` before it schedules, released at the last stage's exit of forward `n - pp`
or of a later output forward holding a request it schedules, and a downstream stage's
`recv_metadata`, released at the head's call ([#1325](https://github.com/jgong5/ATOM/issues/1325)).
On a machine spec's carriers it prices each send as `StageSends` does (`15` D91): the
receive posted at the prepare end plus `W_recv`, the RCCL parts behind the previous
frame's, the receiver's device end and its own send's hold, and a sender's device start
behind the send it last settled. The gate infers idle flushes: the head flushes whenever
it waits for tokens, a later stage once `recv_metadata`'s 100 ms poll times out (#1449,
#1450, #1457). The record stamps each idle `flush_pp_send` that settles a send, and the
gate names, per sending stage, the steps where the two disagree (#1532).
A simulated run prices a speculative step with `D_target` and `D_draft` from the cost
backend, its terms tagged `draft` making the latter, and R1 holds its launch with MTP
([#1090](https://github.com/jgong5/ATOM/issues/1090)). Under DP, each forward's
`target_end` joins the device end's `all_reduce(MAX)` under MTP, and each rank writes the
group's latest back as its output end. Its accepted drafts are `14` D83's draw.

**What-ifs.** The run-file key `host.skip` zeroes the named segments. Such a run is a
what-if: never graded, and refused by the graders.

**Measured, not hooked.** The parity recorder, `RecordingModelRunner`
(`atom/compass/parity/runner.py`), stamps host time around the staging gate, the launch
and the R1 wait (with MTP also the draft copy wait; on a non-first PP stage also the R7
receive, and on the last stage its own token read; on a sending stage an idle
`flush_pp_send` that settles a send), and records device events at
`run_model` entry, with MTP at the target's end, on a non-first PP stage where the
received tensors are in, and at the forward's exit, drained with `query()`, never `synchronize()`
([#824](https://github.com/jgong5/ATOM/issues/824)). It edits no ATOM file. The
recording's engine process stamps the first input pull (`clock.idle`) and output put
(`clock.relay_queue`'s queue) after each reply, `t_pull_ns` and `t_put_ns`, into
`dp<rank>.pulls.jsonl` on the same monotonic clock; `parity.read` joins them to the
steps (#1345).

**The recurrence's own check, provisional.** Fed each real record's measured device
windows and host segments, the recurrence must reproduce the measured host period from
p10 to p90 within 2% per step kind (decode, output prefill, middle chunk, first after
idle), and the run total within 0.5%
([#827](https://github.com/jgong5/ATOM/issues/827)). Without it a wrong wait rule reads as
cost error. T97 records settling the thresholds.

**Accepted approximations**, each added back only if that check shows its error: TP ranks
are symmetric; PP send backpressure; the MTP plus DP draft barrier; the overlaps of the
copy stream, staging and `prep_stream`. One more follows from #815's ruling 2, which
makes `E` the recurrence's first term: `E` is charged inside forward `n`'s TAR, but
`scheduler.postprocess()` and `schedule()` run before the forward is called, at `now`, so
their clock reads (the first-token stamps, the waiting-prefill age, the arrivals a
`schedule()` admits) sit up to `E` early. Where the reply carries an `out_s` they run at
the pull: `postprocess()`'s reads up to `E_out` late, `schedule()`'s up to `E_in` early. **Detected, not modelled:** implicit waits on
pageable copies, and launches blocking on a full hardware queue. The check names each
failing step with its residual against the device work queued at its launch.

**ATOM's sync stalls are fixed in ATOM, not simulated.** Every ATOM inefficiency the
simulator omits is listed here with its ATOM issue and estimated effect:

| ATOM issue | stall | estimated effect |
|---|---|---|
| [#822](https://github.com/jgong5/ATOM/issues/822) | `compute_causal_conv1d_metadata` (`atom/model_ops/attentions/gdn_attn.py`) reads chunk lengths back from the device and copies from pageable memory | on every prefill step of a GDN model (Qwen3.8-27B), the host waits for the queued device work before its own launch, so the device idles through the rest of `W_pre` |
| [#823](https://github.com/jgong5/ATOM/issues/823) | with top-k or top-p, `Sampler._to_tensor_scalar` and `_aiter_sample` (`atom/model_ops/sampler.py`) read device scalars on the host | every decode step loses deferred output's one-step lead, so its host segments add to its device time instead of overlapping it. Every decode step of a stage-2 run takes this path: cc-traces requests carry no sampling fields, so Qwen3.8-27B's `generation_config` (top-k 20, top-p 0.95) applies |

Each issue's named result measures the effect.

### The case that is not a "wait" problem at all

`DecodeEngineCore._recv_prefill_done` blocks in a background thread and calls
`DecodeScheduler.on_prefill_done`, which stamps `seq.first_token_time = time.time()`.
Intercepting the block fixes nothing. The rule is about **when the handler runs**:

> **Handler threads stay where they are. TSO delivery runs each one at its message's
> timestamp, while the clock owner waits.**

Moving `on_prefill_done`'s logic into the step loop instead would change ATOM's thread
model (D1).

Left alone, the handler runs whenever its thread wins the race, so which step's
`schedule()` sees the sequence changes from run to run. TSO delivery fixes that without
touching the handler: the receive wrapper holds the frame, and the clock owner's
`advance_to` walks the pending arrivals in order — set the LP clock to the arrival time,
release one message, wait until the handler thread is back at its wait point, release
the next. `time.time()` inside the handler then reads the arrival time, and `schedule()`
never sees a message from its future. While the handler runs, the clock owner is parked
in its CA call, so one executor handles events in the LP at a time, as with HLA's
single-threaded callbacks. **`on_prefill_done` needs no change.** The cost is that a
simulated run never reproduces the real lock-free race between this handler and the step
loop.

### Open issues

- ~~The counts above are estimates from a synchronization inventory, not from a completed
  pass over the code. The first implementation task should be to produce the exact
  classified list and check it in.~~ **Done, 2026-09-21.** `atom/compass/audit/`.
- ~~The literal `time.sleep(2)` in `DecodeEngineCore._post_model_load_hook` ... it must
  be *checked*, not assumed.~~ **Checked, 2026-09-21, and both halves hold.** It is
  reached only from that hook, and `DecodeEngineCore` is constructed only by
  `DisaggCoreManager`, which `LLMEngine.__init__` selects only under
  `config.enable_rapidserve` — so "RapidServe-only" is exact.
  Its purpose is verified by the code around it: the sleep sits between importing
  decode's weight IPC handles and acknowledging to prefill, and prefill measures free
  VRAM for KV sizing only after that ACK. It must stay on the real clock. Nothing in ATOM
  couples it to whether weights are real, but `Config` keeps a simulated runner from it:
  `--enable-rapidserve` selects `RapidServeModelRunner` only when `runner_qualname` is
  still the default (`Config.__post_init__`), and otherwise `Config` raises `ValueError`
  unless `runner_qualname` is in `RAPIDSERVE_RUNNERS`. The cost, for a runner that list
  names, is two real seconds of startup and no modelled time, because it runs before
  READY and therefore before any arrival.
- The scanner's boundary is a list of exclusions, not a graph. It reads every `.py`
  file under `atom/` except the entries `UNSCANNED_ROOTS` names, each with its reason,
  so a package added anywhere is caught; but a blocking call under an excluded entry is
  invisible to the test, and so is one reached through a call shape the scanner does
  not know.

---

## D5. Timers, clock reads and control commands

### Problem

Some time-dependent behaviour in ATOM changes what the scheduler does; some only decides
when to declare something broken; some is a command from outside the request path. Each
needs a stated answer, and a wrong one either changes the schedule or hides a cost.

### The sorting rule

> **Can the CA reach it?**
> Yes (a Python `time.*` read, an asyncio timer) -> it runs on the LP clock.
> No (a compiled Rust constant, an OS-level wait) -> configuration, or left real on purpose.

This replaces the earlier rule, *"does it change which batch gets scheduled? — if not,
disable it"*. Deterministic-simulation testing (FoundationDB, madsim, turmoil) puts every
timer on the virtual clock and stops sorting them by purpose: a failure detector then
fires in simulated time, only when the simulated system really is that slow, which is
the right behaviour. Disabling it was a second mechanism for no gain.

### Configured, because the CA cannot reach them (K8)

| Thing | Where | Simulated run |
|---|---|---|
| Atomesh health check + circuit breaker | `--disable-health-check --disable-circuit-breaker` | switched off; the flags are already used in ATOM's own CI launcher |
| Atomesh fan-out request bound, default 5 s (`request_timeout` in `atom/mesh/src/core/worker_manager.rs`) | `--worker-request-timeout-secs` (#478) | set large |

That is one Rust bound. The 30 s `DEFAULT_WORKER_HTTP_TIMEOUT_SECS`
(`atom/mesh/src/core/worker.rs`) is never reached: it is the default of a client whose
only request is the health check, which a simulated launch switches off and which sets
its own timeout (#477). The fan-out bound bites whenever the CA holds an LP while wall
time passes, which includes virtual time running **slower** than wall under saturation
(0.30x measured). It is not optional.

### On the LP clock (K7), and why nothing needs a flag

Every Python-side `time.*` read goes through the clock substitution of D9 item 2 and
returns LP time. Every asyncio timer becomes a virtual timer with no code change, because
the frontend's event loop itself runs on LP time (D5.1): the Anthropic SSE ping
(`_ANTHROPIC_PING_INTERVAL_SECONDS`, 5.0 s, read in `anthropic_messages` in
`atom/entrypoints/openai/api_server.py`), uvicorn's `--timeout-keep-alive` (parsed and
applied in that file's `main`), and the stream silence warning (`SILENCE_LOG_SECONDS`,
30.0 s, clock reads in `FrameWait` and `longest_silence_seconds` in
`atom/entrypoints/openai/streaming_dispatch.py`).
Each fires only if the simulated system is really that slow, which is when a real one
would fire it too.

The pacing timers are local next events the clock owner hands to
`next_event(t, t_daemon)` (D4, K3):

- `KV_IDLE_DRAIN_INTERVAL_S = 0.001` (`atom/model_engine/engine_core.py`), gated in
  `EngineCore._advance_idle_kv_transfer`, is essential and joins `t`.
- Daemon deadlines (#533), which join `t_daemon`: the engine's metrics push
  (`EngineCore.busy_loop` and `DPEngineCoreProc.busy_loop`) and the API server's refresh
  loop (`_periodic` running `_refresh_metrics_once` in `api_server.py`), both paced by
  `ATOM_METRICS_UPDATE_INTERVAL_S`. **Metric cadence is virtual time again**,
  as revised `11` D72 also decides (its first version kept it on the real clock): an
  observer in the traffic LP scrapes `/metrics` every `scrape_interval` of simulated
  time. `11` D72 carries the observer and the reasons.
- uvicorn's server tick, a 0.1 s `asyncio.sleep` in `Server.main_loop`, and its
  keep-alive timeout are daemon deadlines too. They are uvicorn's code, not ATOM's, so the
  sync inventory (`atom/compass/audit/sync_sites.json`) has no row for them; the frontend
  declares them with the refresh loop in `DAEMON_TIMERS` (`atom/utils/compass_loop.py`).

### Must read the LP clock — these three change scheduling (K2)

- **`PrefillDelayer`** (`atom/model_engine/prefill_delayer.py`, gated by
  `ATOM_ENABLE_PREFILL_DELAYER`). Its `should_allow_prefill` performs a cross-DP
  `all_reduce(SUM)` every tick on every rank. Its input `oldest_waiting_age_ms` reads
  `time.time()` in `Scheduler._oldest_waiting_prefill_age_ms`. The delayer's own module
  docstring says it is deliberately *tick*-based for determinism; **this is its one
  wall-clock leak, and skew between DP ranks would make them decide differently and
  desynchronize.** Fix the leak, keep the delayer.
- **`Scheduler._passed_delay`** / `--scheduler-delay-factor`.
- **The arrive, leave and first-token stamps** D4 lists under K2.

### Leave on real time, deliberately

- `AsyncIOProcManager.monitor_procs` (`atom/model_engine/async_proc.py`) —
  `multiprocessing.connection.wait` on process sentinels. OS-clock, unreachable from
  Python, and it is a **death detector**, not a timeout. Keeping it real is correct (K8).
- Shutdown joins (`CoreManager.close`, `AsyncIOProcManager.exit`), outside the
  simulation window (K9).

The only requirement is that these stay on the *same* clock as each other:
`CoreManager.close`'s `time.monotonic() + 5` is patchable while `monitor_procs`'s wait
is not, so virtualizing only the former makes the two halves of shutdown disagree.

The control-command reply deadline stays real too (K8, ruling on #964, 2026-10-14; its
reads are in `_KEPT_REAL`, `atom/compass/detect/clock_source.py`): a control command is
out-of-band and produces no simulated event (#886), and the LP clock moves only while
its owner takes a grant, so a deadline on it would never expire while the owner waits.

### D5.1. The simulated run uses stdlib asyncio, not uvloop

The frontend's clock owner is its event loop, and hosting LP time needs two things from
it: timers that expire on the loop's `time()`, and a Python-level point where the loop
blocks, to call `next_event`. The stdlib loop has both — `_run_once` judges timer expiry
and the select timeout from `self.time()`, and blocks in `self._selector.select(timeout)`,
whose selector is replaceable. uvloop has neither: `call_at` becomes a libuv timer counted
in real milliseconds, and the loop blocks inside `uv_run` in C. The only way to keep
uvloop is to intercept `clock_gettime` and `epoll_wait` for the whole process, which is not
an additive change at an existing site. So a simulated run passes uvicorn a stdlib loop
class through `loop="module:Class"`, where `main` in `api_server.py` already chooses the
loop. Only the loop implementation changes; processes, threads and the serving topology
do not (D1), and the frontend's service times come from the model, not from the wall.

### Control commands

The engine registers its out-of-band commands in `EngineUtilityHandler._UTILITY_HANDLERS`
(`atom/model_engine/engine_utility.py`). Each arrives on the control channel, runs on the
step loop between steps, reaches a worker, if at all, through the in-LP RPC (K6), and
replies on the output channel, except the fire-and-forget `abort_request` and
`get_mtp_stats` (`FIRE_AND_FORGET_UTILITY_CMDS`), which never reply.

| Commands | Simulated run | Why |
|---|---|---|
| `abort_request`, `get_mtp_stats`, `get_mtp_statistics`, `get_cache_statistics`, `finish_weight_sync`, `get_weight_sync_status` — call no worker | **run as usual** in the engine; `get_mtp_statistics` and `get_cache_statistics` are refused by the frontend from its clock owner (below), as `/debug/mtp_stats` and `/debug/cache_stats` send them | engine state only, inside the LP. `abort_request` fires when a client disconnects and changes scheduling, so it must stay |
| `start_profile`, `stop_profile` (HTTP routes `start_profile` and `stop_profile` in `api_server.py`), reaching worker methods `start_profiler`, `stop_profiler` | **refused**, by the frontend from its clock owner and by the runner | a real profiler slows every later step; charging it zero is a silent simplification, and a profiled system is not what is simulated |
| `update_weights`, `update_weights_shm`, `update_weights_ipc` (worker methods `update_weights_from_shm`, `update_weights_from_ipc`), `discard_failed_weight_sync`, `release_memory`, `resume_memory`, `clear_kv_cache`, `configure_hidden_states` | **refused** | RL weight sync and sleep/wake. The API server has no route to them; the only senders in the tree are in `atom/rollout/` (`async_engine.py`, `weight_sync.py`), a Python interface for external RL frameworks whose `AsyncLLMEngine` a simulated run refuses at startup as an offline engine. Simulating them needs cost models for weight transfer and memory release that do not exist |
| `collective_rpc` | **as the worker method it names** | it runs a method `COLLECTIVE_RPC_METHODS` (`atom/model_engine/collective_rpc.py`) opens, on every rank; the runner refuses `configure_hidden_states` and answers a name it does not define with ATOM's missing-method reply. Its only sender is `AsyncLLMEngine`, refused as above |

A command is refused by the **simulated runner**, in the worker method it reaches, with a
reply naming that method (`command:<method>`), not an exception, and is counted in the run
summary's refusals (D3.5). An exception would end the engine for a command a client sent
while serving; the reply goes back through ATOM's handler, and the run continues. Any
`command:` refusal makes the run a coverage report, not acceptance evidence, as every
refusal does (`08` D50.1), and for a reason of its own: ATOM's handler still applies its
own half of the command (it marks the profiler active, or the engine asleep), so what runs
after it is no longer the deployment being simulated.
ATOM's handlers are unchanged, the command travels its real channel, and a refusal is
visible rather than a silent zero.

The frontend refuses `CoreManager.broadcast_utility_command_sync` called on its clock
owner (`_refuse_sync_utility_commands`, `atom/compass/run/__init__.py`; ruling on #1076,
2026-10-15): the replies reach it only at that owner's next clock call. The HTTP route
answers 500 naming the command, and the frontend hands in a `clock:` refusal. Off the
clock owner the call is ATOM's.

### Deleted for free by D6

In `atom/kv_transfer/disaggregation/`:

- the MoRI-IO bare-`continue` spin with no sleep and no deadline
  (`MoRIIOConnector.start_load_kv`)
- the Mooncake staging-pool busy-wait (`MooncakeConnector._acquire_staging_slot`)
- Mooncake's `PREFILL_LOOKUP_TIMEOUT = 60` blocking `Condition.wait_for`
  (`MooncakeConnector._wait_for_prefill_data`) and its 2.0 s doubling RDMA retry
  (`_rdma_write_with_retry`)

### Open issues

- A real bound configured off removes a safety net from a long unattended run. The
  simulator should log, once at startup, exactly which K8 bounds it configured, so a hung
  run is diagnosable.

---

## D6. KV transfer: simulated, not carried

### Problem

True PD disaggregation (D2) moves KV between deployments by RDMA. The project
requirement is that simulated execution must not depend on real devices, and that
bandwidth and interconnect are configurable rather than read from hardware. Real RDMA is
therefore out.

### Connector landscape (verified)

`KVConnectorFactory` (`atom/kv_transfer/disaggregation/factory.py`) registers:

| Name | Model | Classes |
|---|---|---|
| `moriio` (**default** when unset, `factory.py::KVConnectorFactory.create_connector`) | RDMA **read**, pull: decode reads from prefill | `moriio_connector.py` |
| `mooncake` | RDMA **write**, push: producer writes into consumer | `mooncake_connector.py` |
| `multi` | fan-out wrapper | |
| `lmcache_offload` | CPU/NVMe offload, **not** P/D | `offload/connector.py` |

There is **no NIXL connector** in this tree.

The ABC is small and has **no `send_kv` / `recv_kv` verb** to fake
(`disaggregation/base.py`):

- worker (`KVConnectorBase`): `register_kv_caches`, `start_load_kv`, `get_finished`,
  `get_finished_recv_blocks`
- scheduler (`KVConnectorSchedulerBase`): `get_num_new_matched_tokens`, `build_connector_meta`,
  `update_state_after_alloc`, `request_finished`, and the send-claim hooks
  `should_defer_free`, `send_finished`, `source_blocks_released`

Another scheduler-side hook is not on the ABC: `process_completions`, which the scheduler
calls on each completion report when the connector defines it
(`scheduler.py::Scheduler._update_from_kv_xfer_finished`).
The offload and `multi` connectors define it; Mooncake and MoRI-IO do not.

### The seam

Every connector's completion reaches the scheduler through **one** method:

```
model_runner.py::ModelRunner.async_proc_aggregation
  -> engine_core.py::EngineCore._poll_kv_transfer_progress
     -> scheduler.py::Scheduler._update_from_kv_xfer_finished
```

That single funnel covers any backend, which is why a simulated connector is cheap.
It is pipelined: `async_proc.py::AsyncIOProcManager.call_func_with_aggregation` keeps one
aggregation in flight, so each poll returns the workers' `get_finished` snapshot taken at
the previous poll, which started that aggregation. A transfer that ends between two polls
is reported at the second poll after it ends, not the first (#1155).

### Design

A `SimulatedKVConnector` is registered through the existing factory. It reproduces
**Mooncake**, the backend ATOM's PD CI deploys (`.github/scripts/atomesh/pd_server_atom.sh`,
`start_prefill` and `start_decode`), and implements **scheduler-side hooks only**: sends
and receives both run on the engine's step loop, and no worker takes part.

- Each decode and prefill rank pair is one FIFO link: a write arriving at `a` starts
  once the pair's earlier writes have moved their bytes, holds the link for
  `bytes / bandwidth`, and completes `latency` after its last byte, at `c`. Alone on
  the link, `c = a + latency + bytes / bandwidth`. FIFO, not processor sharing: on the
  #1152 record, Mooncake TCP finished a wave posted at once smallest-first in posting
  order, the smallest within one step, which processor sharing cannot do (#1157).
  A pair's first write starts `setup` later, the link's optional `link_setup_s` (`05`),
  zero when absent: on node 18 Mooncake TCP every first write of 2 to 7 blocks on a
  pair took 9.0 to 28.1 ms longer than the pair's later writes, and its second write
  was back at their median (#1172). RDMA is unmeasured. `bytes` is exactly computable from block count x
  per-block bytes — no measurement needed. `latency` and `bandwidth` are
  configuration (`05`, the `interconnect` link the run file's `kv_link` names),
  satisfying the "interconnect is configurable" requirement directly.
- **Decode** posts the write request in `update_state_after_alloc` (called from
  `Scheduler._notify_connector_after_prefill_alloc`): channel
  `engine-D->engine-P:kv_write_req`, sent at `t`, arriving at `a = t + request latency`.
  It computes its own ready time `c + notify latency` and reports the request in
  `finished_recving` from `process_completions` as the pipelined funnel would: at the
  call after the first one whose clock reached it. A local event of engine-D.
- **Prefill** records the data-ready time `r` in `request_finished` and takes the write
  request in `process_completions`, which the step loop receives inline.
  `Scheduler._mark_finished` calls `request_finished` once per finished request. There
  the producer takes Mooncake's send claim
  (`mooncake_connector.py::MooncakeConnectorScheduler.request_finished`), owner ruling
  (a) on #858: it adds the request to `_awaiting_send` when the request is not aborted
  and its `kv_transfer_params` carry `do_remote_decode`, which the router sets on every
  request it sends to prefill (`atom/mesh/src/core/placement/backend/atom.rs`).
  `Scheduler.postprocess` parks a finished request in `deferred_free_blocks` only while
  `should_defer_free` finds the claim; any other finished request frees at once. At
  `c`, reported the same way, prefill reports `finished_sending`, and
  `Scheduler._update_from_kv_xfer_finished` retires the claim through `send_finished`
  and frees the parked blocks. `source_blocks_released` does nothing. A local event of
  engine-P. Scheduler code is unchanged.
- **Write-done is not a message.** Real Mooncake sends `MSG_WRITE_DONE` when the write
  ends (`mooncake_connector.py::MooncakeConnector._send_write_done`). Both ends compute
  that time instead, for two reasons: the router forwards strictly in sequence (D2), so
  the decode request, and with it the write request, exists only after prefill has
  returned, i.e. `r <= a`, which prefill asserts and refuses otherwise, so a write starts
  at `a`; and `c` depends only on the writes on this rank pair, which both ends see:
  decode sends them and prefill receives them, in the channel's order, and each `c` is
  fixed at its write's arrival, so both compute the same `c` (owner ruling on #1157). No channel is added; one message whose time both
  ends already know is not simulated.
- **The write request keeps Mooncake's channel** (D3). The real `MSG_WRITE_REQUEST`
  leaves a decode worker (`MooncakeConnector.start_load_kv`) for a prefill worker's
  listener thread (`MooncakeConnector._write_listener`); the simulated one leaves and
  reaches the engine process of the same two LPs, so the LP-level channel is the same
  and no worker ever sends or receives across LPs. A prefill-side local timer would not
  do: when decode posts depends on its own admission, which prefill cannot know, and
  assuming "at once" frees prefill's blocks early exactly when decode is congested.
- **Ceiling.** FIFO per rank pair is measured on Mooncake TCP only; for RDMA it is an
  assumption. Under RDMA the real sharing unit is each prefill rank's NIC: its worker
  writes through the HCA set `_select_ib_devices` resolves for its GPU, and
  `MooncakeConnector._write_listener` hands every write request, from any decode rank,
  to one `_send_executor` of `send_worker_count` (`num_worker_threads`) threads. Writes
  on different rank pairs do not contend here. At decode DP width one a prefill rank's
  NIC carries only the one decode rank's writes, so both ends still compute the same `c`
  under any sharing discipline. Above that, `c` depends on other decode ranks' writes,
  which this decode rank never sees, so a per-prefill-rank model cannot keep both ends
  computing `c` without a message: decode would need Mooncake's own write-done notice
  (`MooncakeConnector._send_write_done` to the consumer's `_notification_listener`) as a
  prefill-to-decode channel (#1179).
- It must still emit the `kv_transfer_params` blob that Atomesh relays, so
  `AtomAdapter` works unmodified. The router hard-errors if it is absent
  (`http_pd_router.rs::PDRouter::dispatch_atom_relay_internal`). The two backends emit
  **different shapes** — thirteen fields and seventeen — and Mooncake's is the
  seventeen; see *The blob, per backend* below. The relay timestamp (D7) is a field
  frontend-P adds to `kv_transfer_params` after the connector has emitted it, so the
  connector's blob stays the seventeen.
- The consumer side must still return `(len(prompt), True)` from
  `get_num_new_matched_tokens` when `do_remote_prefill` is set, i.e. park the request
  (`mooncake_connector.py::MooncakeConnectorScheduler.get_num_new_matched_tokens`), so
  `Scheduler._park_for_remote_load` (`scheduler.py`) and the `WAITING_FOR_REMOTE_KVS`
  state behave identically. It also clears the flag in `update_state_after_alloc`, as
  Mooncake does (`MooncakeConnectorScheduler.update_state_after_alloc`), so every call
  that posts a write request is followed by that park.

### The blob, per backend

Each connector assigns `seq.kv_transfer_params_output` exactly once, so there is no
second site either row below could be describing. Each row's field set is the key
list of that one dict literal, walked out of the AST at `92f1fdafe`.

| Backend | Assignment | Keys | Field set, in source order |
|---|---|---|---|
| `moriio` (pull, the default) | `moriio_connector.py::MoRIIOConnectorScheduler.request_finished` | 13 | `do_remote_prefill`, `do_remote_decode`, `remote_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `remote_handshake_port`, `tp_size`, `dp_rank`, `transfer_id`, `first_token_id`, `draft_token_ids`, `prefix_cache_hit_tokens` |
| `mooncake` (push) | `mooncake_connector.py::MooncakeConnectorScheduler.request_finished` | 17 | `do_remote_prefill`, `do_remote_decode`, `remote_block_ids`, `remote_swa_block_ids`, `remote_engine_id`, `remote_host`, `remote_port`, `remote_handshake_port`, `tp_size`, `dp_rank`, `remote_pp_size`, `hash_block_size`, `transfer_id`, `first_token_id`, `draft_token_ids`, `local_slot_index`, `prefix_cache_hit_tokens` |

The push shape is the pull shape plus four: `remote_swa_block_ids`, `remote_pp_size`,
`hash_block_size`, `local_slot_index`. They are a second backend's blob, not optional
fields of one, and the simulated connector, standing in for `mooncake`, emits the
seventeen.

One of the four is load-bearing rather than descriptive. The push consumer compares
the producer's `hash_block_size` against its own and falls back to a full transfer —
`num_computed_blocks = 0` — whenever it is absent or differs
(`mooncake_connector.py::MooncakeConnectorScheduler.update_state_after_alloc`), so a blob carrying only the thirteen can never
take the incremental path.

`tests/compass/test_kv_blob_site.py` holds the one-site claim above. Nothing re-checks
the field sets: they describe the connectors' code as read at `92f1fdafe`, and where the
two differ the code is right.

### Pros

- Removes RDMA, drivers and the handshake entirely.
- Makes interconnect a first-class configured parameter, which a real RDMA run could not.
- Deletes three busy-waits and two long blocking timeouts (D5).
- Every cross-LP send and receive stays on the engine's step loop, so no LP state has
  to be shared with a worker process.
- The parked-duration gauge `_num_parked_remote_kv` (`scheduler.py::Scheduler._count_inflight_load`, logged in
  `Scheduler.schedule`) is already almost the instrumentation needed to validate it.

### Cons / open issues

- The transfer model is now **unvalidated** — a real RDMA baseline is needed at least
  once to fit `latency` and `bandwidth`, or the numbers are assumed rather than
  measured. This is a calibration task, not a simulator task, but it must be named.
- Mooncake requires **all** `(pp_rank, tp_rank)` pairs to report before a request
  completes (`mooncake_connector.py::MooncakeConnector._record_write_done`). The
  simulated write is one event per request, priced on the whole block table, so ranks
  with unequal shares are not modelled.
- **The simulated connector refuses PP** (`pp_size > 1`), naming the two reasons below.
  No milestone pairs PD with PP: M1 and M2 run PP1, and PP comes after v1 (`16` D95). Under
  PP, Mooncake adds a decode-to-prefill `MSG_RELEASE` channel
  (`MooncakeConnector._send_release`) that this section does not cover, and the release
  above never happens: the PP head's
  `PPEngineCoreProc._poll_kv_transfer_progress` passes worker output to
  `Scheduler._update_from_kv_xfer_finished` only when it is non-empty, and this
  connector's workers report nothing, so `process_completions` is never called, prefill
  never frees and decode never becomes ready. `EngineCore._poll_kv_transfer_progress`
  passes it on every poll, which is why PP1 works. Lifting the refusal takes both:
  the `MSG_RELEASE` channel, and a PP head that passes every poll's output on, empty
  included.

---

## D6.1. KV offload to the lmcache MP server: an LP of its own

Owner ruling on [#1087](https://github.com/jgong5/ATOM/issues/1087), 2026-10-16.

### Problem

`ATOM_KV_OFFLOAD=lmcache_mp` (`arg_utils.py::kv_offload_connector_config`), which the
V4 PD-Max recipe sets on prefill beside Mooncake (a `multi` connector), stores and loads
KV through a separate `lmcache server` process. The scheduler half's adapter
(`mp/deployment.py::_make_scheduler_adapter`) and each worker's
(`_make_worker_adapter`) hold a client that lmcache 0.5.6.dev, the nightly image's,
builds with `lmcache/v1/multiprocess/transport/factory.py::RequestClientFactory.create`:
for a ZMQ URL a `transport/zmq_impl/mq.py::MessageQueueClient`, a DEALER to the server's
ROUTER at `lmcache.mp.host:port`. One `ClientPollingLoop` thread per process sends every
client's queued requests and resolves each reply's future. Stores and loads finish steps
after they are submitted, and the
server's cached chunks decide later hits for every engine that reaches it.
`NonAllocatingRunner.allocate_kv_cache` refuses the connector today.

### Decision

**One LP, `lmcache`, per server process, played by a Compass stand-in.** The server's
state (cached chunks, read locks, its queue) decides what every engine sharing it is
told, which is D7's test for an LP. The real server cannot be instrumented as the router
is: it maps each engine's KV pool over CUDA IPC and copies on the device (README
principle 2). The stand-in reproduces its channel (D3, *Lookahead sources*): it binds
the same URL, decodes and encodes lmcache's own MQ frames, answers the request types
ATOM's adapters send with the reply payloads the real server returns, and takes the
`lmcache server` flags the run names, refusing by name any it does not model. Its clock
owner is its one loop: it receives inline at its own receive point, runs the work on the
stations below, sends each reply when its job completes, and asks only NER, with `t` its
next job completion. No other thread sends.

**State.** A content index keyed as lmcache keys a chunk (model namespace and chunk
token hashes), with the `--l1-size-gb` byte budget, the eviction policy and watermark,
lookup read locks, sessions, and the read TTL. A store's chunks become visible at the
store's completion on the server clock. A TTL expiry is a daemon deadline (`t_daemon`,
#533): it changes later hits but keeps no run alive.

**Channels.** Every one is a real ZMQ path between the real endpoints' identities; the
stamp is its own header frame, as on every wrapped socket
(`clock.py::WrappedSocket`). `N` is the DP rank, `K` the TP rank.

| Channel | Sender (K-class) | Receiver (mode, K-class) | Lookahead |
|---|---|---|---|
| `engine->lmcache:lookup#dpN` | step loop, at `MessageQueueClient.submit_request` (LOOKUP, QUERY_PREFETCH_STATUS, FREE_LOOKUP_LOCKS, END_SESSION); the `ClientPollingLoop` thread sends it as a relay (K4, registered at the put, as the engine output queue is) | stand-in loop (inline, K5) | `lmcache_ipc_s` (below) |
| `lmcache->engine:lookup#dpN` | stand-in loop, at the job's completion (K4) | the `ClientPollingLoop` thread, which sets the future (thread, K5); TSO delivery resolves it at the reply's arrival | `lmcache_ipc_s` |
| `engine->lmcache:transfer#dpN.tpK` | STORE or RETRIEVE of TP rank `K`'s connector, one per writer rank (one per group under `lmcache.mp.tp_rank_collapse`) (K4) | stand-in loop (inline, K5) | `lmcache_ipc_s` |
| `lmcache->engine:transfer#dpN.tpK` | stand-in loop, at the copy's completion (K4) | step loop, in `process_completions` (inline, K5) | `lmcache_ipc_s` |

In a 1P+1D run the engine end is `engine-P`. Every cycle through the server crosses two
channels, so its lookahead is `2 x lmcache_ipc_s`, positive. `lmcache_ipc_s` is a run-file
key defaulting to the run's `ipc_s` (`2**-14` s in `scripts/compass/parity_one_node.sh`):
the recipe runs the server on loopback TCP beside prefill, the same one-host ZMQ hop as
the engine's IPC channels; sample (c) records the adapter's measured LOOKUP round trip
beside it. Startup and teardown frames
(GET_CHUNK_SIZE, REGISTER_KV_CACHE, UNREGISTER_KV_CACHE) fall outside the window and are
not counted (D3).

**The transfer channels end on the step loop, not in the worker**, as D6's write request
does. The worker connector is replaced code (it registers device tensors and records
CUDA IPC events), and its two contacts with the step loop are a fire-and-forget dispatch
(`process_kvconnector_output`) and an aggregation that answers one poll behind
(`AsyncIOProcManager.call_func_with_aggregation`). Neither lets the owner register the
worker's send before its next clock call or release a reply at its arrival, so the
PP pattern (`15` D91: the loop stamps, the worker moves bytes in a call the loop makes)
would need new ATOM call sites. The shell keeps one DEALER per writer rank, so the
server sees the clients it sees in a real run.

**A completion is a stamped message.** On the step loop, the Compass shell around the
scheduler half (below) sends, from `build_connector_meta`, the STORE or RETRIEVE each
writer rank's worker would submit for that metadata (`mp/worker.py::LMCacheMPConnector`
and `native_state_worker.py`, their range arithmetic reused rather than copied). The
server starts the copy at `max(arrival, server free)` and replies at its end `t_c`,
arriving at `t_c + lmcache_ipc_s`. The grant that reaches that arrival releases the
reply in `(arrival, channel, seq)` order; the shell takes it inline in
`process_completions` and reports what TP aggregation of the workers' `get_finished`
would have: `finished_saving` with the store and source-safe completions, or
`finished_loading` / `failed_loading`, once every writer rank's reply is in. It reports
at each call what it held at its previous call, as D6's connector does, because the
aggregation answers one poll behind (#1159). ATOM's scheduler half then runs unchanged.

**Idle points.** The idle offload work (`EngineCore._dispatch_idle_offload_work`, from
`_advance_idle_kv_transfer`) stays where D4 put it: the busy loop's idle point
(`clock.idle` with `_idle_deadline`), the `KV_IDLE_DRAIN_INTERVAL_S` drain a K7 timer
joining `t`. What changes is how a completion is found: a released message, not a
future polled against the wall clock. The other sites in `kv_transfer/offload/mp/` and
`chunked_scheduler.py`, scanned once #1080 drops their exclusion:

- K3: a step-loop wait on a reply, the blocking lookup below.
- K2: the defer clock in `chunked_scheduler.py::lookup_pending`
  (`OFFLOAD_LOOKUP_DEFER_S`), which decides whether admission passes over a request.
- K7: the transfer deadlines (`mp/transfer.py::_enforce_transfer_deadline`, the
  scheduler's `_enforce_transfer_deadlines`); they fire only if the simulated tier is
  that slow.
- K9: the worker connector and its heartbeat, which the simulated runner replaces; the shutdown
  drain (`_drain_kv_work_at_exit`); `perf_counter` reads that only feed counters. The
  stand-in runs no worker reaper: process death is K8.

**Blocking lookups are request and reply.** Admission reaches the blocking
`mp/lookup.py::_MPLookupClient.lookup` under the recipe's normal load, by two triggers:
a waiting request beyond the first `OFFLOAD_ASYNC_LOOKUP_DEPTH`, which
`Scheduler.schedule` never prefetched, and a prefetched lookup deferred past
`OFFLOAD_LOOKUP_DEFER_S`. Its `maybe_submit_lookup_request` (LOOKUP) and
`check_lookup_result` (QUERY_PREFETCH_STATUS) each put a request on the client (K4,
arriving at `now + lmcache_ipc_s`) and wait on its future (K5). The wait is an idle
point, `next_event(+inf, now + mq_timeout)`: the bound is a daemon deadline, not a K7
timer. While a reply is on its way, the server's job end is essential at LP `lmcache`,
so the grant reaches the reply first, as a real 300 s timeout never fires. An essential
bound would itself be work: it keeps the run alive to `now + mq_timeout` and lets LP
`lmcache`'s daemon ticks fire up to it, past the run's `bound_s`. With no reply coming,
the authority grants `+inf` and the wait returns false. The poll loop's `time.sleep`
between QUERY_PREFETCH_STATUS calls is `next_event(now + poll interval)`, an essential target, since the
loop acts then on its own, and `lmcache.mp.lookup_timeout` is a deadline it reads on the
LP clock. The stand-in takes requests in
timestamp order and replies at its job's end `t_c`; the grant that reaches
`t_c + lmcache_ipc_s` releases the reply to the `ClientPollingLoop` thread, which sets
the future before the owner returns. With positive lookahead both ways, the server's
grant to `t_c` needs only the engine's NER target, so at DP width 1 nothing more is
needed.

**In a DP group the members park** (owner ruling on #1162, 2026-10-16). The waiting
member asks NER alone. Its peers have scheduled and dispatched their forwards, and wait
on it in the step-seconds `all_gather` (`overrides.py::_group_step_seconds`;
`chunked_scheduler.py::lookup_pending`: "every DP rank waits in lockstep"), so they make
no CA call. The member join rules alone deadlock here: rule 5 lets the lone NER move no
time, and rule 3 holds the grant for the peers' send logs, which carry their own lookups
and stores to `lmcache`. While the peers count as running, the server's LBTS stays at
the group's `now + lmcache_ipc_s`, below `t_c`, so the reply is never sent.

A *park* request amends rules 1, 2 and 5, at DP width above 1 with `lmcache_mp`:

- **The park.** Each member's last CA call before it dispatches a forward (real or the
  DP dummy) is a park, which ships its send log. A parked member stops regulating time
  until the group's collective completes, since it sends nothing before then: until
  every member is parked, its park stands in every round as NER with `t` and
  `t_daemon` at `+inf`.
- **Rule 1.** A park is one call however many rounds it stands in, so the NER calls a
  waiting member makes while its peers are parked are not a count mismatch.
- **Rule 2.** A round may mix NER and parks; a TAR meeting either is still refused.
  That is why the park is the last call before the dispatch: a K1 TAR between
  `_sync_dp_state` and the dispatch would meet the waiting member's NER in one round.
- **Rule 5.** The waiting member's NER rounds complete with its peers parked and move
  its time alone. A parked member moves no time, and walks the thread-channel messages
  a round releases to it inside its call, so I1 holds.
- **Release.** The round in which the last member parks releases every member at the
  group's time, the latest member's `now`, not its own. A peer parked at `t0` leaves at
  the waiting member's `now` after the reply, so its forward starts when the late
  member dispatches, and #988's step-end rounds keep one base (rule 2's equal `T`).

A parked member's call takes no reply until the release. The authority holds what a
round releases on its channels; at the release the member walks those messages inside
its park call, its clock at each one's arrival in turn, then leaves at the group's
time. Its clock therefore moves only inside its own call, and the step table records
each held release from the clock the releasing grant found. The check refuses any TAR
from the return of `_sync_dp_state` to the park. On an idle DP step,
`_advance_idle_kv_transfer` runs `_poll_kv_transfer_progress` inside that window,
before `_execute_dummy_batch`. D4 lists the poll as K1, but it asks the authority
nothing: a completion is a released message, taken inline. The check therefore passes
an idle step.

`settle_before_collectives` cannot host the park: the collective the peers wait in,
the step-seconds `all_gather`, runs in the runner, not on the clock owner thread. The
park is one call on the step loop before the forward dispatch, and costs one CA round
per step per member.

**Service time.** Two stations in the server LP: the request loop, one server, FIFO,
pricing a LOOKUP by a declared law in prompt chunks and the bookkeeping requests at a
declared constant; and the copy engine, a stated width, pricing a STORE or RETRIEVE as a
`TransferModel` (`kv/transfer.py`) over the machine spec's host link, `bytes` the
chunks moved times the chunk's bytes from the KV geometry. Both are declared until
measured; their accuracy is out of scope here.

**Fake-model duties.**

- The simulated runner's worker half is a null connector, as `compass`'s is: it
  registers nothing, sends nothing and reports nothing; the forward's duration is the
  backend's.
- A Compass shell, built in place of the `lmcache_mp` connector through
  `KVConnectorFactory`, wraps ATOM's real scheduler half (PAGE-only or native-state, as
  `mp/connector.py::LMCacheMPConnectorScheduler.bind_block_manager` picks) and carries
  the transfer channels above. The scheduler adapter is lmcache's own; its client's
  queue and socket get the relay and wrapped-socket treatment at
  `_make_scheduler_adapter`, its one construction site.
- Inside a `multi`, the composite's scheduler half is a Compass subclass of ATOM's
  through the same factory, whose `process_completions` lets `compass`'s scheduler half
  add its writes before the one offload sub-connector reads the output. ATOM's composite
  calls that method on one sub only, since a real P/D connector reports through its
  workers' `get_finished`; the two write disjoint fields.
- The stand-in server replies to every request with the real payload: a hit count from
  its index, a success flag, a chunk size.

**Accepted approximations**, each added back only if the parity sample shows it:

- A STORE waits on a CUDA event recorded at dispatch; the stand-in takes it as ready at
  the send, early by at most one forward's device time.
- Source-safe chunk events are device events the worker polls, not messages; every
  chunk is reported with the store's reply, so early block release comes no earlier
  than real.
- A native-state load's device restore after the copy is charged nothing.

**Refusals that remain.** ATOM's own (`mp/deployment.py::_validate_mp_config`: PP, DCP,
PCP, DP across hosts, `engine_driven`, more than one server) stand as they are. The
stand-in refuses a server flag or request type it does not model, by name, and the shell
refuses a server URL that `RequestClientFactory` hands to gRPC (`grpc://`,
`grpc+unix://`), a carrier with no wrapped socket, by its scheme. A `multi`
connector runs only when each sub-connector is simulated (`compass` and `lmcache_mp`);
`ATOM_KV_OFFLOAD=lmcache` (in-process, no server) and `mooncake_store` stay refused.

**Acceptance**, per #1076's ruling:

- (a) A one-deployment run, and one at DP2 whose ranks share the server, on a workload
  that stores a prefix, evicts it from HBM and loads it back, and holds more than
  `OFFLOAD_ASYNC_LOOKUP_DEPTH` waiting requests on each rank with a lookup deferred past
  `OFFLOAD_LOOKUP_DEFER_S`, so both blocking triggers fire on every rank: each run
  reaches its finish with no refusal, a seed pair gives byte-identical step tables, every
  `(channel, seq)` on every channel is received once and none before its receiver's
  clock, every request gets its full output, and the runs carry at least one load hit,
  the DP2 run one across ranks. The DP2 run reports the CPU cost per step of the park's
  CA round.
- (b) Under the constant step stub and constant station laws, the same workload's
  sends, server job starts and ends, reply arrivals, the parked request's release and,
  at DP2, each member's park and release match a hand-derived timeline event by event.
  (a) and (b) run in the gate that can start a simulated run
  (`atom/compass/detect/scenario.py` checks the channels).
- (c) One scheduling-parity sample on node 18 MI308X (`xiaobizh_n18_next13`), in the
  style of `scripts/compass/parity_one_node.sh`: a real run against a real
  `lmcache server`, then its replay; per-step batches and each request's load and save
  decisions compared.

**Not chosen.**

- *A station on each engine LP* (#1087 option (b)): it cannot express one server shared
  by several engines, which the recipe's eight prefill ranks are.
- *The real server under Compass*: README principle 2.
- *Refusing a blocking lookup above DP width 1* (#1162 option (b)): it refuses the
  recipe, whose prefill runs at DP8.

---

## D7. Atomesh: no virtual clock, three flags and a constant

### Problem

Atomesh is Rust. Every timing primitive in it is `tokio::time` or `std::time::Instant`,
unreachable from a Python virtual clock. In standalone mode it runs a tokio runtime
**inside the Python process** via PyO3 (`src/python.rs::launch_mesh`, `startup_runtime`), so the engine and the
router share an OS process and run on two different clocks.

### Analysis

The ATOM relay is strictly sequential and blocking (D2). Atomesh therefore contributes
exactly two things to a request: a **routing decision**, and **two HTTP round trips of
overhead**. It never overlaps prefill and decode.

Every milestone's PD run is **1P1D**, the only layout ATOM's single-node PD launch supports
(`.github/scripts/atomesh/pd_server_atom.sh` refuses any other `SINGLE_NODE_PD` layout):
one prefill and one decode instance, two containers on one node in M1, one container per
node in M2 (`16` D95). The routing decision then has nothing to choose between, so no load
count, random draw or clock read reaches its result, and the router is **one segment of a
channel**, not an LP. Each of the three channels through it,
`traffic->frontend-P:http`, `frontend-P->frontend-D:relay` and
`frontend-D->traffic:stream`, has a real LP at both ends; D3's channel table names their
senders.

The router's per-request overhead is part of these channels' lookahead. The wall time
it really takes only lengthens a message's time in flight, which the `(channel, seq)`
counting covers (D3, D8). FIFO is not required: concurrent requests through the router's
tokio tasks can arrive out of order, and messages are released by `(channel, seq)`. A
response the router makes itself, such as the error for a missing blob
(`PDRouter::dispatch_atom_relay_internal`) or for a failed decode
(`PDRouter::handle_decode_error_response`), is no message on
`frontend-D->traffic:stream`, since no LP sends it: it is a fault, a stall the CA cannot
see (D3.2).

This holds only while the router sees one worker per role and its limiter is off. A
`--dp-aware` launch, which CI uses with DP attention (`pd_server_atom.sh`,
`start_router`), registers every DP rank as a separate worker, and the policy then
chooses among them; the router is then an LP (below, #449). `--max-concurrent-requests`
defaults to `-1` and the token bucket to unset (`cliargs.rs::CliArgs`), and CI sets
neither; with either on, a wall-time queue or refill decides admission.

So its only time-dependent behaviours are **failure detectors**, which by the D5 rule
get disabled — not virtualized.

### Design

1. **Run mesh-only mode, not standalone.** In standalone, the SSE drain at
   `atom_standalone.rs::AtomStandaloneRouter::run_sse_service_stream` is a `loop { ... if chunks.is_empty() { continue } }`
   paced only by a Python `queue.get(timeout=0.05)`. Under a fast virtual clock that
   degenerates into a GIL-hammering spin. A separate binary keeps the two clocks from
   touching.
2. **Disable the detectors** (D5 table). `--request-timeout-secs` is already 1800.
3. **Model the relay overhead as a stated per-request delay**, exactly as the prior
   work modelled admission (13.7 ms measured; worth ~4 points of TTFT; applied as a
   delay on the request, *not* as time the engine consumes — advancing a global clock
   per admission double-counts concurrent arrivals). The delay is the lookahead of the
   three channels above.
4. Atomesh's routing **decision** is business logic and is reused unchanged. Only its
   latency is modelled.
5. **Timestamps ride carriers the router already forwards**, so the router is not
   changed. Verified against `http_pd_router.rs` and `openai-protocol 1.0.0`, the crate
   it pins:

   | Channel | What the router does | Carrier |
   |---|---|---|
   | `traffic->frontend-P:http` | Parses the body into a typed request and re-serializes it. `CompletionRequest` keeps unknown fields (`#[serde(flatten)]` on `CompletionRequest.other`, `completion.rs`); `ChatCompletionRequest` (`chat.rs`) has no such field, so **an unknown top-level chat field is dropped**. Request headers are forwarded by allow-list, `tracestate` included (`header_utils.rs::should_forward_request_header`), to decode as well. | The `compass` entry of the W3C `tracestate` header, e.g. `tracestate: compass=a:12.345;s:17`, appended after any existing entry so real tracing is unaffected. The same for chat and completion. |
   | `frontend-P->frontend-D:relay` | Takes `kv_transfer_params` from prefill's JSON, lets the ATOM adapter insert fields, and writes it into the decode body (`http_pd_router.rs::PDRouter::dispatch_atom_relay_internal`). | A field inside `kv_transfer_params`, added by frontend-P after the connector has emitted its blob (D6). frontend-D also receives the original `tracestate`, so a request carrying `kv_transfer_params` takes its stamp from there (relay channel), any other from `tracestate` (traffic channel). |
   | `frontend-D->traffic:stream` | Streamed: passes the ATOM stream through byte for byte; `PDRouter::create_streaming_response` rewrites only with `return_logprob` and prefill logprobs, and the ATOM path has neither. Not streamed: returns decode's body unchanged with decode's response headers, less hop-by-hop ones (`header_utils.rs::preserve_response_headers`). | Streamed: an SSE comment line before each event, e.g. `: compass a=12.345 s=18`; SSE clients ignore lines starting with a colon. Not streamed: a `compass` response header, e.g. `compass: a=12.345;s=18`. |

   A single deployment (M1's stage 1 and M2's first gate, no router) uses the same carriers.

**Under `--dp-aware`, or with several instances per role, the router is an LP.** With several
instances per role, or `--dp-aware`, every policy this router ships
(`atom/mesh/src/policies/`) reads state that changes over time:
`power_of_two`, `cache_aware` and `prefix_hash` compare `load()`, which moves as
requests start and finish; `random` draws from an unseeded generator; `round_robin`
counts requests in the order they reach the router, which is wall-clock order under
concurrency; `dp_sticky` reads `Instant::now()`. A component whose time-varying state
decides its output is an LP, so the router would have to become one, instrumented in its
own process with the policy code untouched. It is (owner ruling on
#449, 2026-10-15; several instances per role, #1085).

- **Channels.** A run file with `router_in_s` declares LP `router`
  (`channels.py::prefill_decode_table`): each path through the router is a hop in,
  with lookahead `router_in_s`, and a hop out with the rest of the path's. `pd_sim.sh`
  chooses the router's arguments as `start_router` does: a DP-attention cell is
  `--dp-aware` when it is agentic (as `cctraces_sim.sh` cells are) or maps ranks, and
  then `pd_sim.sh` writes `router_in_s` at the run's `ipc_s`. Any other DP-attention cell
  keeps a router without `--dp-aware`, and ATOM's own DP dispatch picks the rank.
- **Delivery.** `atom/mesh/src/compass_router.rs` binds the LP when
  `ATOM_COMPASS_CLOCK_ENDPOINT` is set. A request (ingress layer), a prefill response
  (`PDRouter::dispatch_atom_relay_internal`) and each decode stream event
  (`PDRouter::create_streaming_response`) wait until released, and are handled one at a
  time in (arrival, channel, seq) order. Each sends exactly one message on, stamped at
  its arrival plus the hop out's lookahead, which keeps every path's total. A request is
  handled once its pair holds load, a prefill response once prefill's load is dropped,
  and a stream's last event once decode's load is dropped. The router prices no work and
  waits with NER only.
- **Policy inputs.** `determinism::enable` runs before the policies are built, with the
  run's seed (`ATOM_COMPASS_SEED`, the cell's `SEED`): the policies read the LP clock, a
  seeded RNG and a fixed hasher. Without the endpoint nothing changes. `power_of_two` is
  refused by name when it is built: its loads come from each worker's `/get_load`,
  polled on a wall-clock interval (`LoadMonitor`), which no channel carries (#1134).
- **Unchanged.** Cancels stay off behind the router; a clock error ends the router
  process.
- **Periodic tasks.** Under the LP, `PeriodicTask` (cache_aware's tree eviction) starts
  no thread: the task's next deadline is the daemon deadline of each NER, and it runs
  before any message arriving at or after that deadline (#1093).
- **Several instances.** A run file's `prefill_instances` and `decode_instances`
  declare LPs `frontend-P<k>` and `engine-P<k>` (`-D<k>` for decode) per deployment
  (`channels.py::prefill_decode_table`), each with its own hops and write-request
  channels; without `router_in_s` the table refuses them. Each server names its index as
  `compass_instance` in its `kv_transfer_config`; `pd_sim.sh` starts them and names each
  worker URL's frontend LP to the router (`ATOM_COMPASS_ROUTER_WORKERS`). A request goes
  on, stamped, to the prefill it was placed on, and a relay to its decode; decode finds
  the prefill deployment by the relayed `remote_engine_id`.

### An unused hook worth knowing about

`mocker/virtual_workers/mock_case.rs` declares
`SimulationFixture { ttft_ms, chunk_interval_ms }`, threads it through
`MockCase.simulation`, and ships it in all 8 fixtures — with **zero consumers**
anywhere in the crate. A purpose-built latency-injection point that nobody wired up. If
router-side latency injection is ever wanted, this is where it goes.

### Open issues

- `worker.rs::DEFAULT_WORKER_HTTP_TIMEOUT_SECS` is the one Atomesh timeout still compiled
  in; changing it means touching the Rust crate, with `ATOM_MESH_BUILD=1` and a Rust
  toolchain in the loop. D5 finds a simulated launch never reaches it, so this stays open
  only if that launch ever keeps the health check on.
  `worker_manager.rs::DEFAULT_WORKER_REQUEST_TIMEOUT_SECS` is only the default of
  `--worker-request-timeout-secs` (#478), which D5 sets large.
- Mesh-only mode has not been exercised by this project. Confirm it serves the ATOM
  relay path correctly against two ATOM servers before depending on it.

---

## D8. Arrivals

### Problem

A discrete-event clock may only advance when it knows no earlier event will still turn
up. An HTTP client posts concurrently, so requests reach the engine in a different order
from the one they were stated in.

Measured cost of getting this wrong (64 requests, Poisson 8/s, Qwen3-0.6B TP=1):

| | real | stated arrivals, no barrier | + barrier |
|---|---|---|---|
| TTFT mean | 42.23 ms | **213.50 ms** | 37.39 ms |
| TTFT max | 82.10 ms | **1653.09 ms** | 67.63 ms |
| wall clock | 9,417 ms | 495 ms | 847 ms |

### The prior mechanism and why it is replaced

take2 solved it with an **arrival barrier**: the engine runs nothing until
`len(waiting) >= compass_workload_size`. It works, and it has four costs:

1. It needs a **count**, so it cannot serve open-ended arrival.
2. It is bounded by `ARRIVAL_BARRIER_TIMEOUT_S = 120.0` on the **real** clock, and on
   timeout the run's latencies are invalid. In one case the client still reported
   "0 failed" and a day's conclusions came off it.
3. It forces a client design of one connection per request, hard-bounded at
   `MAX_IN_FLIGHT = 1024` — a 64-thread pool against a 300-request stated workload
   deadlocks, because the server produces no response until all 300 have arrived.
4. It spins: the real run's first step is scheduling tick 1; the simulated run's is
   tick **89,336**. Virtual time is frozen so it costs no fidelity, but it is real CPU
   and it scales with the workload.

### Design

With a CA, the traffic source is simply **an LP that publishes one thing: "no arrival
before time T."** No LP it feeds can be granted past `T` plus its lookahead distance
from traffic (D3), so none can miss an arrival.

This is the same mechanism every other LP uses. Consequences:

- `_arrival_barrier_unmet`, `compass_workload_size`, the 120 s timeout, and the 89,336
  spinning ticks are all removed.
- **Open-ended serving works**, because the CA needs the next arrival time, never the
  total count.
- The one-connection-per-request constraint relaxes, because the server no longer
  withholds every response until the whole workload has landed.

### The delivery-lag wrinkle

A POST travels uvicorn -> `LLMEngine.preprocess` -> ZMQ -> EngineCore input thread ->
`scheduler.waiting`. The traffic LP must not publish a bound past an arrival that has not
landed in `waiting`.

This is not special to arrivals: it is the general **transient message**, sent but not
yet received, and the send/receive counting of D3 already covers it (PDES term:
Fujimoto/Mattern message counting under an HLA time-advance grant). The traffic LP
registers each POST as a `(channel, seq)` with its arrival time, reported with its next
advance request; a grant past that time carries the `(channel, seq)` set it releases,
and the receiver does not return from the grant until each has arrived. The frontend's
send to the engine is counted the same way, and a frame is held at the receive wrapper
until its grant, so ATOM never sees a request before its arrival time. There is no
enqueue acknowledgement, and **closed and open-ended workloads are handled alike**.

The arrival time rides the `compass` entry of the `tracestate` request header (D7). take2
put it in a body field, `CompletionRequest.compass_arrival`, which the router drops from a
chat request because the typed chat request keeps no unknown top-level field.

### What the corpus cannot supply

The requirement mentions "faithful agentic behaviour: branching, joins, delays,
completion-driven recycling, cancellation". Verified against the corpus:

- **Joins are not represented.** No field says a parent resumed because a child finished.
- **Cancellation is not represented.** `status` is `"completed"` on all 1,697 subagent
  wrappers; `tool_use_count` is `null` on all 1,697; `subagent_type` is `"Subagent"` on
  all 1,697. There is no other value in either corpus.
- **Branching is represented** — 1,697 wrappers across 175 of 393 sessions, up to 10
  concurrent branches, max 153 branches in one session.
- **Delays are present but undefined on the dataset card** (`think_time`, p50 1.17 s,
  p90 10.45 s).
- Arrivals are therefore **open-loop** by necessity, and the prior tooling deliberately
  keeps them so.

**This must be stated in the acceptance scope rather than claimed.** It belongs to the
workload design topic, but it is recorded here because it is what licenses the
stated-arrival model.

### Open issues

- Whether the workload schedule is owned by the client (stated arrivals, take2's
  model) or handed to the engine as an input file. The prior notes concluded the general
  serving case "needs the schedule handed over up front, with HTTP demoted to fetching
  results". The CA makes the client-owned version viable, so this is now a preference
  rather than a forced move — but it should be decided, not drifted into.
- A client stopwatch cannot time a simulated engine. Measurement egress must come from
  the engine's own readings. take2 used `GET /compass/requests` returning
  `{request_id, seq_id, arrive_time, first_token_time, finish_time, ttft, latency}` plus
  a top-level `clock: "virtual"|"wall"`. Something equivalent is required.

---

## D9. The ATOM diff, and how completeness is enforced

### Expected change set

Every item is additive at an existing site; no process and no thread is removed (D1).

1. **`atom/utils/clock.py`** — an injectable clock. take2 designed this in 148 lines:
   a `Clock` protocol, `WallClock`, `VirtualClock`, `get/set/reset_clock`. One detail
   worth preserving: `WallClock.epoch` returns `None`, deliberately not `0.0`, because a
   caller offsetting from the Unix epoch would get a 1970 timestamp and a duration in the
   billions; `epoch is None` then becomes the engine-wide discriminator for "real clock,
   none of this applies". Add an `AuthorityClock` that talks to the CA, and with it the
   LP runtime and the CA's side of D3: the send log, the expected counts, the strict
   grant, and walking pending arrivals in order. **~550**
2. **Clock-read substitution** at the K2 and K7 sites of D4 and D5, including the
   `_last_refresh` stamp in `AtomMetricsExporter.update`
   (`atom/metrics/exporter.py`, `11` D72). take2 did four in `scheduler.py`.
   **~30**
3. **TAR and NER hooks.** `EngineCore._process_engine_step_inner` calls
   `advance_to(now + predicted)` after the forward; take2's `_advance_clock_for` is
   this hook. The predicted duration rides back on a new field of
   `ScheduledBatchOutput`, which is `None` on a real run so the advance is a no-op and a
   real run is untouched. The same at the other forward sites and the idle rank's
   dummy batch (K1).
4. **`Scheduler._advance_to_next_arrival`**, which does not exist, is not added: the step
   loops' spin and the offline `get_output` become `next_event` calls (K3), and the CA
   performs the jump. Items 3 and 4 together: **~100**
5. **Socket wrappers** at the channel creation points, one line each. The message
   header is (channel, arrival time, sequence number); on a ROUTER socket it goes after
   the identity frame (K4, K5). **~150**
6. **The relay queue**, one line replacing `output_queue` in `EngineCore.__init__` (K4).
   **~80**
7. **Poller wrappers** where the receiving threads create their pollers,
   `EngineCore.process_input_sockets` and `CoreManager._create_output_thread` (K5).
   **~40**
8. **Frontend virtual time**: the stdlib event loop class and its selector hook (D5.1);
   the tokenizer thread pool as a resource station and detokenization as one per output
   thread; wrappers on `tokenizer.encode` / `decode`; an HTTP middleware that counts and
   releases requests (`06` D33). **~350**
9. **Configuration** for the K8 bounds of D5 — atomesh's two disable flags and
   `--worker-request-timeout-secs` set large, no Rust change — and uvloop off (D5.1).
   **~20**
10. **The process-group assertion** (D4): a collective group never spans two LPs.
    **~30**
11. **`SimulatedKVConnector`** registered through the existing factory (D6). **~80**
12. **Control-command refusal** in the simulated runner's worker methods that the refused
    commands reach (D5). **~40**
13. **The metrics observer and the end of a run**: a periodic `/metrics` scrape in the
    traffic LP, the engine's push gate on a virtual timer (D5, `11` D72), and the traffic
    LP's end-of-run check (D3, #534). **~60**
14. **A simulated `ModelRunner`** injected via `--runner-qualname` (no ATOM change needed
    for the injection itself).

### Size estimate

Items 1-13 come to ~1,530 lines, and the tests below to ~450: **~1,980 lines**, against
the ~1,200 this section estimated while the change set was status annotation at the
category-B sites. take2's whole clock integration was **+148 new lines and ~771 modified
across 11 files, with 13 deletions** — almost pure addition, no import-time
monkeypatching of any ATOM class.

The tests, all CPU-only:

- **In-transit messages**: two fake LPs with 5 ms of added transport delay; the drain at
  10.2 sees a message sent at 9.6, and two runs choose the same batches.
- **TSO delivery**: a handler thread reads its message's arrival time as the clock, and
  `schedule()` never sees a message from its future.
- **Relay queue contract**: three puts are all sent, received in order, each stamped with
  the LP time of its `put`; an output thread made to skip one raises `UnsentRelayItem` on
  the next `get()`; a `put` from another thread raises.
- **Inventory completeness**: `test_sync_inventory.py` extended so every row carries a
  mechanism consistent with its peer and bound fields.
- **Determinism** (D3.4): two runs of one configuration, byte-diffed step tables.
- **PD release**: two fake engine LPs and the simulated connector, with decode admission
  delayed: prefill frees its KV blocks at the write's completion and decode is ready a
  notify latency later (D6); prefill finishing after the write request arrives raises.
- **I2 guard**: an AST test that no `advance_to` / `next_event` call sits lexically inside
  a `with ...lock` block, so a clock owner never parks in the CA holding a lock another
  thread in its LP needs.

### The enforcement mechanism

Clock-site completeness **cannot be verified by reading.** take2 solved the narrow
version with an AST-walking test (`tests/compass/test_serving_timings.py`) asserting that
no assignment to `first_token_time` / `finish_time` / `arrive_time` in `scheduler.py`
calls `time.*`.

Generalize it: a test that walks the serving path and **fails on any un-allowlisted
`time.time` / `time.monotonic` / `time.perf_counter`**, where the allowlist is exactly
what D5 leaves on real time. That test is what makes this design safe rather than
merely careful, and it should exist before the substitution work starts, not after.

A second test should assert the CA's safety invariant fires: construct a backdated
cross-LP event and assert the run aborts.

---

## Cross-cutting open issues

Ordered by how much they could cost.

1. **Silent failure is the dominant risk mode.** Every failure in D3-D5 produces a
   plausible latency table rather than an exception. The mitigations — always-on
   assertions, the D3 stall rules (recovery, or one `DIAG_S` diagnostic, never an
   abort), the AST test — are the design, not decoration. This
   project's history contains at least four instances of a plausible artifact from a
   broken run being read as a result for a day or more.
2. **Simulation speed is unmeasured under this architecture.** The prior design was
   0.30x under saturation. Multi-process with per-advance RPCs is not obviously faster.
   Measure a saturated cell early, before the architecture is load-bearing.
3. **Per-step replay CPU cost.** Previously measured device-free against a 32.7 ms
   modelled step: 41.7 ms uncached, 2.3 ms with a shape-keyed cache (**unsound** — a
   second valid allocation for the same shape moves 64 of 2439 operator signatures), and
   4.3 ms with the allocation carried in the key. Cold costs are separate: ~9.5-11 s for
   a factory build, 0.27 s for the first graph of a new shape. Beware a cold-loop
   artefact: an average over a loop whose first iteration is a cache miss reads 3-7x
   higher than steady state.
4. **Schedule agreement must be reported separately from latency.** See D0. The prior
   best result reproduced five prefill streaks with identical chunk counts, every break
   within half a second across a 267-second run — and that agreement was established two
   changes *before* the latency numbers were right. It is the more diagnostic metric.
5. **Scheduling fidelity is unobservable on a saturated workload.** Every loaded prior
   experiment ran at essentially 100% utilisation, where the queue term swamps
   everything. A scheduler comparison needs a workload with slack, or controlled step
   durations that remove the cost error by construction.
6. **Multi-node DP is implemented but not hardware-validated** — `docs/distributed_guide.md`
   §9 carries an explicit banner, and the MoRI `InterNodeV1` path and RDMA behaviour are
   unverified. If paired real-vs-simulated evidence is needed there, the real side may
   not exist yet.
7. **CA placement and grant latency** are unresolved (D3).
8. **PP will make the CA the bottleneck** when it arrives after v1 (D3).

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D0 | Fresh design; prior branches referenced at the design level only, not as a code-port plan | 2026-09-17 |
| D1 | Keep ATOM's multi-process / multi-thread topology; additive changes only | 2026-09-17 |
| D2 | "Two nodes" means true PD disaggregation, realised as two containers on one physical node | 2026-09-17 |
| D3 | Central Clock Authority acting as the HLA RTI (`advance_to` = TAR, idle point = NER, grant = TAG); an LP is a process group with one clock owner per process, and zero-lookahead couplings collapse into one LP; the CA grants a DP group's LP only after every rank has called; a grant is strictly below the lookahead-distance LBTS and waits for the messages already in transit (Fujimoto counters); a run finishes when no essential work is left; on a server open to clients that test ends a client run, as HLA's join, resign and synchronization point frame it, and the finish comes after the client runs the run file declares | 2026-09-18; revised 2026-09-28, 2026-10-02 and 2026-10-14 (#966) |
| D3.1 | Single CA with a hierarchy-ready interface; LP count scales with replicas and PP stages, not with GPUs | 2026-09-18 |
| D3.2 | Always-on causality detectors: a straggler check against the receiver's last drain (fails the run); a stall the CA cannot see prints one diagnostic after 30 wall seconds and keeps waiting, never aborting; clock-source CI lint | 2026-09-19; revised 2026-09-28 |
| D3.3 | CA deploys two ways from one implementation: co-hosted in the API-server process by default, standalone server via `--compass-clock-endpoint` for multi-container runs (1P+1D in M1 and M2) | 2026-09-19 |
| D3.4 | Wall-clock interleaving may vary between runs; the `(LP, virtual time, event)` sequence may not. Thread scheduling inside an LP is irrelevant only under in-transit counting and TSO delivery; ties break by **(time, LP id, channel, sequence number), never arrival order**; the cost backend is a pure function of its batch view; **no `set` iteration on the simulated path** - string ids are hashed under `PYTHONHASHSEED` randomisation, so a set of request ids iterates differently in every process. Test is a byte-diff of two step tables, CPU-only, in CI. | 2026-09-20; revised 2026-09-28 |
| D3.5 | The CA owns three outputs: an opt-in timeline log, a stall diagnostic (a stall the CA can see is resolved by deadlock detection and recovery; one it cannot see gets a diagnostic, never an abort), and a run summary written always, an early end included, carrying grants, speed ratio, lazy-trace cost, detector state, the refusals with their reasons, the predicted seconds by method, and `coverage_report` (`08` D50.1). Each quantity is counted in one place, which the summary, the step table and the timeline read; the simulated system's latency and throughput (TTFT, ITL, request counts) stay in the benchmark's output and ATOM's metrics, not the summary. | 2026-09-20; revised 2026-09-28, 2026-10-11 and 2026-10-14 (#952) |
| D4 | Every synchronization site maps to one of the PDES mechanisms K1-K9, none undecided; the boundary is the LP, not the process; a DP member's step is one TAR at its own release from the group's device end, the `max` over ranks by an `all_reduce(MAX)` (with no host time, an output rank's at the `max` of the step seconds), exchanged inside the forward by K6 waits with no cost, and the lockstep `all_reduce` is K1 for its own cost; handler threads stay and TSO delivery runs them at their message's timestamp. A forward's K1 charge is one recurrence per runner over the backend's device busy time and host segments (`E`, `W_pre`, `W_launch`, `W_post`, and under MTP `W_inputs` after R1), with wait rules R1, R4 and R6 to R8; after the return the engine LP advances the host's `out_s` (`E_out`) to its input pull, a DP member's pull joining the group rounds; `host.skip` makes a what-if; ATOM's sync stalls are fixed in ATOM and listed with their issues | 2026-09-18; revised 2026-09-28, 2026-10-01, 2026-10-11 (#815), 2026-10-16 (#829, #1267) and 2026-10-18 (#1345) |
| D5 | Every timer and clock read the CA can reach runs on the LP clock, except the out-of-band control-command reply deadline; only what it cannot reach is configured or left real; metric cadence is virtual time (revising `11` D72); the profiler and RL control commands are refused by the simulated runner with a reply, counted, and make the run a coverage report; a synchronous command from the frontend's clock owner is refused by the frontend | 2026-09-18; revised 2026-09-28, 2026-10-09, 2026-10-14, 2026-10-15 |
| D5.1 | The simulated run uses stdlib asyncio, not uvloop: only a Python event loop can host virtual time | 2026-09-28 |
| D6 | KV transfer is simulated through a connector registered in the existing factory, reproducing Mooncake (what ATOM's PD CI deploys) through scheduler-side hooks only; a rank pair's writes queue on its link | 2026-09-18; revised 2026-09-28; link sharing 2026-10-16 (#1157); first-write setup 2026-10-16 (#1172) |
| D6.1 | The lmcache MP server is LP `lmcache`, a Compass stand-in on the real ZMQ channels with a content index and two stations; a store or load completes as a stamped reply that the step loop takes in timestamp order; a blocking lookup is a request and reply whose wait is an idle point; inside a DP group each member parks before its forward, amending member join rules 1, 2 and 5, and the last park releases every member at the group's time | 2026-10-16 (#1087, #1162) |
| D7 | Atomesh gets no virtual clock: mesh-only mode; in 1P1D the router is one segment of a channel, not an LP, and its relay latency is that channel's lookahead; its bounds are configured (health check and circuit breaker off, `--worker-request-timeout-secs` set large); timestamps ride carriers it already relays; under `--dp-aware`, or with several instances per role, the router is LP `router`, handling requests, prefill responses and stream events in timestamp order (#449, #1085) | 2026-09-18; revised 2026-09-30, 2026-10-15, 2026-10-16 |
| D8 | Arrivals are messages in transit, covered by the send and receive counts for closed and open workloads alike; the arrival timestamp rides the `tracestate` header | 2026-09-18; revised 2026-09-28 |
| D9 | The ATOM diff is enumerated (~1,980 lines with tests) and completeness enforced by test rather than by review; ATOM's own suite is the gate (`08` D43.1) | 2026-09-18; revised 2026-09-28 |

---

## Appendix: verified reference points

Facts this design leans on, with their source, so a later reader can re-check rather than
re-derive.

**The seam**
- `Config.runner_qualname` — `atom/config.py`; consumed `engine_core.py::EngineCore.__init__`,
  `async_proc.py::AsyncIOProc.__init__`
- `model_runner.py::ModelRunner.forward`, whose signature is
  `forward(batch: ScheduledBatch) -> ScheduledBatchOutput`
- the RPC boundary — `engine_core.py::EngineCore._process_engine_step_inner`
- `ScheduledBatch` fields — `scheduler.py::ScheduledBatch.__init__`; notably `detailed_sqsq` /
  `detailed_sqsk` / `detailed_sk`, which are sum(N_Q^2), sum(N_Q * N_KV),
  sum(N_KV) per batch, computed by `Scheduler.compute_detailed_aggregates` and
  currently gated on `profile_active and ATOM_ENABLE_DETAILED_ANNOTATION`
- `ScheduledBatchOutput` — `scheduler.py`; `ScheduledBatch.produces_output()`

**Existing simulation-shaped hooks in ATOM**
- `--load_dummy {empty,zero,xavier}` — `config.py::Config.load_dummy`, `arg_utils.py::EngineArgs.add_cli_args`,
  `loader.py::initialize_dummy_weights`, `load_model`, `loading_core.py::load_weights_into_model`
- meta-device model construction —
  `model_runner.py::RapidServeModelRunner._init_weight_params_on_meta`
- a working non-allocating runner template — `model_runner.py::RapidServeModelRunner`,
  which overrides `_build_and_load_model`, `_maybe_warmup`, `_kv_budget_extra_reserve`,
  `get_num_blocks`, `allocate_kv_cache` and `forward`
- `model_runner.py::ModelRunner.dummy_execution` shows how to hand-build
  a `ScheduledBatch`
- `ScheduledBatch.is_dummy_run` — `scheduler.py::ScheduledBatch.__init__`
- simulated TP (`--fake-eplb`) — `atom/distributed/simulated_tp.py`; explicit precedent
  for a shape-accurate, value-meaningless run
- synthetic speculative acceptance — `config.py::SpeculativeConfig._resolve_synthetic_acceptance`,
  `atom/model_ops/rejection_sampler.py::RejectionSampler`, `rejection_sample`; the closest existing behaviour simulator
- profiler label taxonomy — `atom/model_engine/run_labels.py`; consumed by
  `tools/parse_trace.py`

**Memory sizing (needed because it decides which configurations exist)**
- `model_runner.py::ModelRunner.get_num_blocks`, with its four `torch.cuda`
  reads in `model_runner.py::ModelRunner._read_device_memory`. Five device readings plus
  arithmetic: `mem_get_info`, `allocated_bytes.all.peak`,
  `(total - free) - memory_reserved()`, `_estimate_cudagraph_overhead()`, a 2% safety
  margin, then `min(budget - ..., free)` and `plan_pools`. Consumed
  `engine_core.py::EngineCore.__init__`.
- `BlockManager.__init__` asserts `num_blocks > 0` (`block_manager.py`), so a simulated
  runner must return a plausible count.
- The prior work's rule: **substitute the readings, never the arithmetic.**

**Prefix caching** (default on, `config.py::Config.enable_prefix_caching`)
- hash: `BlockManager.compute_hash` — `block_manager.py`, xxhash xxh64 chained
  with the parent hash
- hit scan: `BlockManager.can_allocate` — `block_manager.py`
- publish, deferred until after the forward computed the KV: `BlockManager.hash_blocks` —
  `block_manager.py`, called from `scheduler.py::Scheduler.postprocess`

**Timing call sites in the serving path**
- request latency uses **`time.time()`** (wall) because `arrive_time` is stamped in the
  API process and `first_token_time` in the EngineCore process: `llm_engine.py::InputOutputProcessor.preprocess_fanout`,
  `scheduler.py::Scheduler.postprocess`, `scheduler.py::DecodeScheduler.on_prefill_done`, `llm_engine.py::InputOutputProcessor.postprocess`; TTFT/TPOT computed in
  `llm_engine.py::InputOutputProcessor.postprocess`
- loop pacing uses `time.monotonic()`; benchmark clients use `time.perf_counter()`
- **the main `EngineCore._process_engine_step_inner` does not time the forward.** Only the
  RapidServe prefill/decode cores do (`engine_core.py::PrefillEngineCore._process_engine_step`, `DecodeEngineCore._process_engine_step`).

**Test harness that a compatible seam inherits**
- `tests/conftest.py::MockConfig` — a GPU-free, download-free stand-in giving
  `BlockManager` / `Scheduler` exactly the fields they read
- `tests/aiter_stub.py::stubbed_aiter` — so `async_proc` imports on a CPU runner
- directly relevant existing tests: `test_scheduler.py`, `test_block_manager.py`,
  `test_block_pool.py`, `test_prefill_scheduler.py`, `test_scheduled_batch_marshal.py`,
  `test_forward_mode.py`, `test_prefill_delayer.py`, `test_dp_load_balance.py`,
  `test_simulated_tp.py`, `test_kv_drain_liveness.py`
