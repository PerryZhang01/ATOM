# ATOM Compass — Design Topic 15: Parallelism Support (TP, DP, PP, EP)

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `01` (the LP structure and the wait contract this applies per strategy),
`03` / `05` (width-keyed memory), `07` D40 (communication pricing), `04` (structures that
exist only at width).

**Scope.** What each parallelism strategy changes for a simulator, answered in one frame
so the four are comparable. It **applies** `01`'s clock protocol per strategy; it does not
re-decide it.

**Why this is not a late topic.** v1 runs DeepSeek-V4-Pro with DP-attention and then EP8
(`16` D95), and DP couples *scheduling decisions*. So TP's and DP's LP structure, clock
topology and memory-accounting shape come first, in **M1**; EP's come with M2's EP8 step.
Scheduling parity and then cost *accuracy* follow in **M2** — D94 splits them. PP comes
after v1.

---

## D88. The frame: four questions, asked of each strategy

A parallelism strategy can affect a simulated run in exactly four ways. Asking the same
four questions of each is what makes them comparable, and three of the four turn out to
have the same answer for most strategies.

| # | Question | Why it matters |
|---|---|---|
| **Q1** | **Does it add logical processes, and at what lookahead?** | LP count and lookahead set the cost of the clock protocol. A microsecond lookahead means many grants; a barriered group means none. |
| **Q2** | **Does it couple *scheduling decisions* across ranks?** | This is the one that can break Compass's central claim. A strategy that only changes cost is routine; one that changes *which batch runs* touches the part Compass reuses rather than models. |
| **Q3** | **What does it change about cost?** | Which structures exist, which collectives appear, how they scale with width. |
| **Q4** | **What does it change about memory?** | Which terms shard, which replicate, which are keyed by width. |

**The headline result, and it is the reason this topic is short:**

> **Only PP adds logical processes.** TP, DP and EP each sit behind a barrier that already
> exists in ATOM, so `01` D3's collapse rule absorbs all three. The clock protocol's cost
> is therefore a function of PP degree alone, not of total GPU count.

---

## D89. TP — the settled case

Nothing here is new; it is gathered so the frame has a worked instance and so the
width-keyed material has one index.

| | |
|---|---|
| **Q1 LPs** | **None added.** A TP group is one LP (`01` D3) — it already barriers on every collective. Rank-0 single-sourcing of the clock is measured correct: TP=2 over 1,727 steps, per-step rank difference median **0.03%**, worst 0.82%; TP=4 over 2,295 steps, rank totals within **±0.02%**, and charging every step to its *slowest* rank adds **0.06%**. |
| **Q2 scheduling** | **None.** One scheduler, one batch. |
| **Q3 cost** | Collectives priced per width, never scaled by a law — measured `cross_device_reduce_1stage` **9.08 / 10.40 / 10.95 µs** at TP 2/4/8 (+20.6%) while `allgather_lastdim` went **29.00 / 23.73 / 16.39 µs (−43.5%)** (`07` D40, `10` D66). Some structures exist only at width. |
| **Q4 memory** | Weights shard; the runtime memory readings are **measured per configuration and do not fit a law** — `non_torch` 926 / 6906 / 7266 / 10704 MiB at 1/2/4/8 (`03` D14, `07` D41's `memory_readings`). |
| **Open** | **T21** — whether the in-situ calibration transfers across width. The one unproven assumption in the recipe. |

**What TP supplies to the others:** the per-width discipline. Every strategy below inherits
the rule that *width is a key, not a parameter* — a number measured at one degree is not
evidence about another until it is shown to transfer.

---

## D90. DP — the step shape is decided by a collective, and that is the design problem

**Revised 2026-10-01 by the owner's DP ruling**
([#470](https://github.com/jgong5/ATOM/issues/470#issuecomment-5933154215)): the step's
cost is the `max` over ranks, below. It replaces the per-layer critical-path pricing `01`
D3 adopted on 2026-09-28. **Revised 2026-10-16**
([#1267](https://github.com/jgong5/ATOM/issues/1267#issuecomment-6215701399)): with host
time the `max` is the group's device end, and each rank leaves at its own charge.

### Q2 first, because it is the one with teeth

DP is the only strategy that couples **scheduling decisions** across ranks. Under DP, the
shape a rank runs is not a function of what that rank scheduled.

`ForwardMode.decide` (`atom/utils/forward_context.py`) runs a DP collective *before* the
step's shape is settled, and settles the shape **from the result**:

```
   rank A schedules its batch  ---+
   rank B schedules its batch  ---+--> sync_dp_metadata()  --> unified_bs
   rank C schedules its batch  ---+    (one packed all_gather    any_rank_has_prefill
                                        of int32 per rank)       max_seqlen_q_across_dp
                                              |
                                              v
                                   the batch is REWRITTEN from the group's
                                   answer, then the ladder, the graph key,
                                   the draft pass and the attention plan are
                                   all read off it
```

The class docstring states the property and the hazard together:

> `running_bs` is agreed on **EVERY** step — it is the reduction on a ladder every rank
> shares — because a captured graph holds its collective at that batch and **a rank
> arriving with another one hangs the group**.

and, on why the chain is not separable:

> `decide` owns the whole chain … because the steps are not separable: the reduced query
> length rewrites the batch, and everything else is read off it afterwards.

`running_tokens` is agreed only when `running_tokens_are_unified`, i.e. when no rank has a
prefill; otherwise each rank runs its own count through the variable-length gather, whose
contract is that the rows it was handed are the rows *this* rank scheduled.

### There are two DP collectives, not one

`01` D3 cites the liveness one and is right about its conclusion, but the shape one is the
load-bearing one for this topic:

| | Where | What it carries | Cadence |
|---|---|---|---|
| **liveness / lockstep** | `DPEngineCoreProc._sync_dp_state` (`atom/model_engine/engine_core.py`) | Gloo CPU `all_reduce(MAX)` over three booleans: `has_unfinished`, `shutdown`, `offloaded` | engine-core loop |
| **step shape** | `sync_dp_metadata` (`atom/utils/tbo/ubatching.py`), called from `ForwardMode.decide` | one packed `all_gather` of `n_fields` int32 per rank: DP token padding, the prefill fan-out, the cross-DP TBO gate, and the DSpark graph-shape MAX | **every forward** |

The second is cheap by construction — *"one all_gather of n_fields int32 values per rank
suffices"*, down from up to three `all_reduce`s, and with TBO off only the first three
fields are exchanged.

### Decision

> **Both DP collectives are *exactly reproducible* under simulation, because both reduce
> over scheduling metadata rather than over model outputs.**

This is the same argument as `03` D13's for the KV pool: the quantity is arithmetic over
integers Compass already has — `scheduled_tokens`, `scheduled_bs`, `is_prefill`, and the
TBO/DSpark scalars — so **running the real reduction is more faithful than modelling it,
and free.** Nothing in either collective depends on a logit, a hidden state, or anything a
device would have produced.

Consequences:

1. **Do not stub `sync_dp_metadata`.** Let it run. The simulated DP group reaches the same
   `unified_bs`, the same `any_rank_has_prefill`, the same ladder rung — by computation,
   not by assumption.
2. **The DP group stays one LP** (`01` D3), and the shape collective is a *second* barrier
   confirming it: with a blocking all-gather at the head of every forward, DP ranks cannot
   drift by more than one step.
3. **Both collectives are internal to the LP, and neither is ignored** (revised
   2026-09-28, #443). Neither is a cross-LP wait the CA sees. The lockstep `all_reduce`
   (`DPEngineCoreProc._sync_dp_state` in `atom/model_engine/engine_core.py`) is an event
   cost (`01` D4, K1). Both have their own cost priced (Q3 below). The
   step-seconds exchange (below) is Compass's own and carries no cost (`01` D4, K6).

### Q1, Q3, Q4

| | |
|---|---|
| **Q1 LPs** | **None added.** One LP per DP group, barriered at each of ATOM's DP collectives and at Compass's step-seconds exchange, every step. Each rank is a member process of that LP with its own LP runtime, and the CA joins the members (`01` D3). |
| **Q3 cost** | ATOM's DP collectives (the table above) are tiny, and the liveness one runs on the **CPU/Gloo** path. Each is priced like any other collective (`07` D40), per width. The step-seconds exchange is not priced: it stands in for the MoE gather and reduce-scatter (the all-to-all under EP), which the MoE segment already prices. The all-gather payload is `n_fields × dp_size` int32 — bandwidth-irrelevant, latency-dominated, which is the regime `10` D66 already models. |
| **Q4 memory** | **Plain DP** (no DP-attention): weights and KV **replicate** per DP rank; nothing shards. So the memory model is per-replica — plain DP is the one strategy that does *not* move the memory readings. **DP-attention is not plain DP** (revised 2026-10-10): `moe.py::FusedMoEParallelConfig.make` flattens TP across DP, so the routed experts shard over the DP group, and aiter builds device communicators for the DP-wide and EP-wide groups. Weights come from a meta build at those widths (`03` D16, [#699](https://github.com/jgong5/ATOM/issues/699)); `non_torch` and the load residue are keyed by the layout, (tp, dp), written `"1x8"` for `-tp 8 --enable-dp-attention` (owner ruling on [#702](https://github.com/jgong5/ATOM/issues/702#issuecomment-6182503618)): each rank runs attention at TP 1 inside 8-wide DP and EP groups, and its readings match neither width 1 nor width 8. With `--enable-expert-parallel` the key gains the EP width, (tp, dp, ep), written `"1x8x8"` (owner ruling on [#627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6194442788)): an EP rank on MORI also holds MORI's symmetric heap outside torch. The `"1x8"` entries measured on node 18 with `--all2all-backend none` (EP on, MORI off) stay under the EP-off key. Each layout is its own engine-start measurement. |

### The group's device end is the `max` over ranks

`01` charges a step to its **slowest rank**, and measured that this costs 0.06% at TP=4
because TP ranks are near-identical. **DP ranks are not.** Different DP ranks hold
different requests and therefore genuinely different batches, so "slowest rank" is not a
rounding correction here — the slowest rank's device end is the step's device end, and it
is what the `unified_bs` padding exists to express. Rank-0 single-sourcing is a TP result and does not
transfer to DP.

- **Combine.** Every rank prices its own batch with its own runner and backend. With host
  time (a `HostDeviceRecurrence`), the group's device end is the `max` over ranks of each
  rank's absolute device end on the LP clock: every forward kind, the DP dummy included,
  joins one `all_reduce(MAX)`, under MTP with the latest target end beside it. Each rank
  writes the device end back as its last device end, and as its output end on an output
  forward, the target end in its place under MTP, then leaves at its own composed charge, as a real rank
  returns at its own launch end while per-layer collectives tie the devices. A replay host
  leaves at its own replayed charge. With no host time, a rank with output other than the
  DP dummy leaves at the `max` over ranks of the step seconds, any other at its own.
- **Exchange.** The `all_reduce(MAX)` (`atom/compass/runner/overrides.py`,
  `_group_step_seconds`) stands in for the cross-rank collectives a real forward makes
  (the MoE gather and reduce-scatter, or the all-to-all under EP), which a predicted
  forward does not run. After the charge, one `all_gather` on the DP group
  `sync_dp_metadata` uses carries every rank's step target. Neither carries a cost of its
  own, because the MoE segment already prices those collectives; each is a wait inside one
  LP (`01` D4, K6). With one rank both are skipped. The lockstep `all_reduce` in `DPEngineCoreProc._sync_dp_state`
  keeps its payload, the three booleans above; the step's cost is not exchanged on it.
- **Two token counts.** `T_own` is the rank's own scheduled token count. `T_moe` is the
  number of rows the group's MoE gather carries, the same on every rank. With EP off, MoE
  is tensor-parallel across the whole group (`flatten_tp_across_dp` in
  `atom/model_ops/moe.py`), so every rank runs a slice of every expert on all `T_moe`
  rows. `ForwardMode.decide` settles both of its inputs from `sync_dp_metadata`, and the
  gather follows `forward_mode.running_tokens_are_unified`, not the CUDA graph:
  - Every rank decoding: the padded `all_gather`, `T_moe = dp x running_bs x
    max_seqlen_q`, where `running_bs` is the smallest capture size at or above
    `max_bs_across_dp`, or `max_bs_across_dp` itself when no capture size is that large;
    under `enforce_eager` nothing is captured, so `running_bs` is `max_bs_across_dp`.
  - Any rank prefilling, and an MTP draft's first pass: the variable-size `all_gatherv`,
    `T_moe = sum(num_tokens_across_dp)`.

  The batch view carries both counts, and refuses a DP run that states no
  `num_tokens_across_dp`. The local segment (attention, dense layers, the shared expert)
  is priced from the rank's own batch; the MoE segment (gather, experts, reduce-scatter)
  reads only `T_moe`, at every granularity (`09` D54 for the step linear law, #530 for the step roofline and the op-level backend).
  With EP on there is no gather: `FusedMoE.forward_impl_graph` skips it under
  `use_all2all_kernels`, and each rank hands MORI (`MoriPrepareAndFinalize.prepare`) the
  rows its forward runs on. When no rank prefills (`ForwardMode.running_tokens_are_unified`),
  `ForwardMode.decide` sets `running_tokens` to the group's height, so every rank, an idle
  rank's dummy batch included, dispatches `T_moe / dp` rows, its own padded to that height,
  whether the decode replays a graph or runs eager (the eager branch of
  `ModelRunner.run_model` pads through `ModelRunner._padded_decode_inputs`, `enforce_eager`
  included). When a rank prefills, each rank dispatches its own rows, decoders beside it
  included, and `T_moe` is the group's sum of them. Either way the group dispatches
  `T_moe` rows, so `T_moe` is the MoE row count with EP on too, and `BatchView.moe_rows`
  carries it unchanged (`atom/compass/runner/projection.py`, `batch_view`). MORI sends a
  row to every rank that holds one of its routed experts; under uniform routing a rank
  receives `T_moe x reach / dp` rows, `reach` being the ranks one row reaches: 4.42 for
  DeepSeek-V4-Pro's 6 routed experts of 384 over 8 ranks, 4.43 measured. These readings
  are [#1106](https://github.com/jgong5/ATOM/issues/1106#issuecomment-6236367874)'s
  EP8 diagnostic cells.
- **Why `max` is the critical path here.** Under DP-attention a layer's barrier-delimited
  segments are the local segment and the MoE segment. With EP off the MoE segment is
  rank-invariant by construction (above). With EP on its priced cost is rank-invariant
  under uniform routing: every rank is priced at the expected maximum rank load (`10` D64).
  When one segment is rank-invariant the sum of per-segment maxima equals the max of
  per-rank sums.
  Rank 0 at attention 3 per layer and rank 1 at 1, both at MoE `c`, give
  `max(3 + c, 1 + c) = max(3, 1) + c`. A gap between the two, such as 240 against 360 over
  60 layers for attention 3 / MoE 1 against 1 / 3, needs MoE cost to differ across ranks,
  which only non-uniform routing under EP produces.
- **Non-uniform routing is a stated limit.** Real routing is data-dependent (D92 Q2).
  With EP off it changes the MoE cost equally on every rank. With EP on it loads ranks
  unevenly: on #1106's cells the busiest rank's received rows over the group mean, per
  step and layer, read 1.36 at p50 and 1.56 at p90 on decode, against 1.24 and 1.37 to
  1.40 from uniform routing of the same sent rows, and 1.27 and 1.49 on prefill, against
  1.00 and 1.01. Uniform routing therefore understates the group's largest received-row count
  by about 10% on decode and 27% on prefill at p50, and where the MoE segment differs
  across ranks `max` underestimates the step. Stage 3 declares uniform routing as an
  assumed treatment (`09` D61; owner ruling on
  [#627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6194442788), 2026-10-15)
  and models neither difference: the step linear law fitted under the EP8 engine config
  absorbs MORI through its `moe_rows` term and intercepts, which take in whatever skew
  the capture's routing had, and the measured imbalance is reported beside the law's
  accuracy ([#1108](https://github.com/jgong5/ATOM/issues/1108)).
- **Idle ranks.** `DPEngineCoreProc._execute_dummy_batch` calls
  `ModelRunner.dummy_execution`, which runs the same `forward` on a dummy batch. So an idle
  rank's dummy batch is priced and enters the `max`; a rank with no work is never free and
  never zero.
- **Second-order effect.** The shared device end leaves out each rank's own tail after its
  last cross-rank collective. With no host time every output rank stamps its outputs at
  the group's `max`, so a fast rank's output is late by at most the difference between its
  own cost and the `max`. The step is split with in-step TAR only if that is measured to affect TPOT.

---

## D91. PP — the only strategy that adds logical processes

### Q1: N LPs at microsecond lookahead

Each PP stage is its own `EngineCore` (`01` D1: `dp_size × pp_size` engine cores), and
stages communicate by point-to-point transfer rather than by a barrier — so `01` D3's
collapse rule does **not** absorb them. PP degree `P` turns each engine LP into `P` LPs.

The lookahead between adjacent stages is the p2p latency, which is **microseconds**. That
is the expensive case for a conservative protocol: small lookahead means frequent grants.
`01` D3.1 already anticipated this and gives the partition rule —

> PP stages have microsecond lookahead and **must stay under one node-local CA**; PD role
> boundaries have millisecond lookahead and are the cheap links to cross a node.

**Decision: adopt that rule as a constraint, not a preference.** A PP stage boundary is
never the cut point for a hierarchical CA, and for a multi-container P/D run the standalone CA sits at the PD
boundary with each role's PP stages under one local authority.

### Q2: no scheduling coupling

Unlike DP, PP does not reduce anything into the batch. The batch is decided once, by the
first stage's scheduler; later stages execute a shape they are handed. So PP adds LPs
without adding a decision coupling — the opposite trade to DP.

### Q3: cost — the transfer is a size, not a tensor

ATOM's PP send (`atom/distributed/pp_comm.py`):

| Call | Nature |
|---|---|
| `async_send_intermediate_tensors` | non-blocking `isend`s: the metadata list through `_async_send_object`, then each tensor |
| `_async_send_object` | two gloo `isend`s on the PP CPU group: a one-element size tensor, then the pickled object |
| `commit_pp_send_work` | `wait()`s on each in-flight `isend`: a gloo part blocks the host until it completes, an RCCL part only makes the current stream wait for it (measured, #1383) |
| `pp_send_allgather_group` | TP group for PP send-allgather, `None` if disabled or tp=1 |

`send_intermediate_tensors`, the blocking variant, has no caller under `atom/`.

**The send is asynchronous, and when it completes can depend on the receiver** (revised
2026-09-30, owner ruling, #443). The stage waits for its last send in two places: before
it issues the next one (`commit_pp_send_work`, called from `ModelRunner.run_model`), and
when its loop is idle (`flush_pp_send`, called from `PPEngineCoreProc._pp_head_step` when
the head launched nothing this round and from `_downstream_busy_loop` when a downstream
stage received nothing). Both wait on the host for the send's gloo parts only; its RCCL
parts hold the stage's stream, so the next send's tensors leave once they are done, and every
forward the stage runs after that wait starts its device work no earlier, a time the ack
carries (#1444)
(#1383). Which clock bounds that completion is decided by the transport's
protocol for the message's size, the standard MPI-level PDES treatment (SimGrid SMPI's
`smpi/send-is-detached-thresh`, SST/macro, CODES):

- **Duration** comes from a LogGP-style cost model per carrier: latency `L`, per-message
  overhead `o`, per-byte time `G`, bytes from geometry (below).
- **Eager** (bytes up to the carrier's `eager_threshold_bytes`): the send completes on the
  sender's clock at `t_send + o + bytes × G`. The receiver plays no part.
- **Rendezvous** (larger): a part completes at `max(t_send, r_i - E × G) + T`, `E` its carrier's threshold:
  the sender moves up to a threshold's bytes before the receive is posted (#1458). `T` is the rendezvous
  duration for its size and `r_i` the time the next stage posts that part's receive, which
  only the next stage's LP knows; the first part's is `t_recv_posted`. The send's `done` is
  the latest completion over its parts (How the parts combine, below).

**A cross-LP dependency arrives as a message** (conservative PDES). Two channels join
adjacent stages, each on a path the real system has: the stage-to-stage send itself, and
the completion that the receiving transport (RCCL or gloo) returns to the sender. PP runs
only with one DP rank (`CoreManager.__init__` rejects PP with DP), so `#dp0` is the only
instance. The `pp_transport.py` channels (`meta`, `tokens`, `kv_status`) are listed
with the others in `01` D3.

| Channel | Sent | Stamped | Received |
|---|---|---|---|
| `stage(k)->stage(k+1):pp_data#dp0` | at stage k's send point | `t_send + L_data` | before stage k+1's forward computes |
| `stage(k+1)->stage(k):pp_ack#dp0` | by stage k+1, sends with a rendezvous gloo part only | `done` of the gloo parts, carrying the RCCL parts' completion | at stage k's pre-send wait and idle flush |

**Lookahead of `pp_ack`.** The receiver can compute `done` only once it holds the data
message and has posted its receive, at `now_r = max(t_recv_posted, t_send + L_data)`, and
the ack is valid only if `done >= now_r + L_ack` for every send. If
`t_recv_posted >= t_send + L_data`, then `done - now_r >= T - E × G`, more than the carrier's
`rendezvous_fixed_s`: a rendezvous part is larger than its threshold. Otherwise
`now_r = t_send + L_data` and `done - now_r >= T - L_data`,
smallest when the receive was posted no later than the send. So the largest valid
lookahead is

    L_ack = T_min - L_data

with `T_min` the smallest `rendezvous_fixed_s` on any carrier.
The two channels' lookaheads sum to `T_min`: the ack makes the stage-to-stage cycle no
tighter than the rendezvous itself. `L_data` is stated as the smallest carrier
`latency_s` (`L`). It moves no modelled time, because the receiver's launch waits for the
metadata's arrival (below), which is at least `t_send + L_data`; it only decides how the
cycle's lookahead is split. The channel table refuses, by name, either channel with a
non-positive lookahead (README principle 6): a zero-lookahead coupling belongs inside one
LP (`01` D4).

- `L_ack <= 0`, i.e. `T_min <= L_data`: the calibration says a rendezvous completes
  before its request could reach the receiver, which no rendezvous does. The refusal
  names both values.
- A carrier's `latency_s <= 0`: `L_data` is then not positive and `pp_data` has no
  lookahead. The refusal names the carrier and its value.

**A send whose gloo parts are eager takes no ack.** An eager completion `t_send + o + bytes × G` can lie before
the receiver's clock (a receiver that posts late), so an ack stamped with it would break
any positive lookahead. The sender computes it itself, and both waits advance to it (K1's
rule).

**The step loop keeps the forward's compute overlap.** On the device, the previous send
runs while the next forward computes; the forward then waits for it and sends. At a
non-last stage's forward call site, after the runner answers the time `c` to its launch end,
the time `p` after it to its exit, and the payload of send `n`:

1. `advance_to(t0 + c)`, the forward's charge (`01` D4), and `e` is its device work's end
   (R7). The previous send's `pp_ack`, if stamped at or before
   `t0 + c`, is released during this TAR, which is the common case.
2. If that send took an ack and its ack is not yet released, the loop receives it
   inline: NER until the grant releases `(pp_ack, n-1)`, its clock reaching `done`. A
   loop that blocked there for `b` goes on `wake_s x (1 - exp(-b / wake_rise_s))` later,
   gloo's wake: a host blocked on a gloo wait wakes later the longer it blocked
   (#1451). If it
   was eager, `advance_to` its local completion when that is later than now.
3. The loop issues send `n` at `t_send`, its clock, now past `done`, its tensors
   leaving at `e`, and registers `(pp_data, n)` (K4), noting whether it is eager (a local completion) or rendezvous (an ack to receive).
4. `advance_to(now + p)`, the rest of its charge: ATOM sends after the launch.

So a send still waits for its forward's launch and the previous send, but a rendezvous
send's completion now comes from the receiver. At a downstream stage's forward
call site, before the runner is called, the loop receives `(pp_data, n)` inline (its
clock reaching `now_r`) and asks the runner for `t_recv_posted`: the forward's prepare end
(`E`, `W_pre` and `W_inputs` run from the call, the entry held by the ring), as ATOM's
`ModelRunner.run_model` calls `recv_intermediate_tensors` after `prepare_model` (#1443),
then `W_recv`, the part of the launch `run_model` runs before that call (#1451).
With non-negative host segments it is never before the call, so the `pp_ack` lookahead derivation above holds. If any gloo
part of the send is rendezvous it registers `(pp_ack, n)` stamped their `done` (K4). The
forward's host runs from the call, its launch after `W_recv` waits for the metadata's arrival and its
device start for the end of the receive, below (`01` D4, R7).

**The idle flushes are channel receives (`01` D4, K5).** At both idle call sites the loop
first settles the pending send: an inline receive of its `pp_ack` if it took one, an
advance to its local completion if it was eager, nothing if no send is pending. ATOM's own
`flush_pp_send` call then runs, and the simulated runner answers it at once (`02` D10). The
shutdown call in `_downstream_busy_loop` is the same call as the idle one and shares its
answer; it runs after the `+inf` grant that closes the simulation window, where
`advance_to` and `next_event` raise (#533), so it settles nothing and sharing changes
nothing. The head's shutdown call
in `_head_busy_loop` runs there too. `commit_pp_send_work`'s own
`wait()` is inside the replaced runner and never runs.

**Carrier.** A simulated runner sends no tensors, so both channels ride gloo
point-to-point on the PP CPU group (`get_pp_group().cpu_group`) between the TP-rank-0
workers of adjacent stages: the group and the endpoints ATOM's `_async_send_object` sends
the metadata on, and aiter's `GroupCoordinator.recv_tensor_dict` reads with
`recv_object`. The stage loop stamps, registers and releases every frame; the worker only
moves bytes, through two simulated-runner methods the loop calls (send one frame, receive
one frame), an engine-to-worker RPC inside the LP like every other (K6). The send only
posts the `isend` and never waits for it, or the loop would wait on the receiving stage
where the CA cannot see it; the receive runs only after a grant released the frame, which
is then registered and in flight, so it waits a bounded time. Chosen because it
is the real path between the real endpoints, adds no socket, process or address, exists in
every PP run, runs on the CPU (README principle 2) and is ordered per pair. Not chosen: a
socket between adjacent stage loops, which ATOM does not have (`PPStageTransport` joins
the head to each stage and each stage to the head), so it would add a channel; and the
`meta` frame, which the head sends before its own forward and so cannot carry stage k's
send time.

**Which sends are which regime.** `async_send_intermediate_tensors` makes one stage send
out of several parts, and each part's carrier is ATOM's:

- `_async_send_object` puts the size tensor and the pickled metadata list on the PP CPU
  group: the gloo carrier, bytes = the pickled length.
- `hidden_states`, `residual` and, when a PP boundary splits a DSA IndexShare group,
  `sparse_kv_indices` are device tensors on `pp.device_group`: the RCCL carrier. Bytes are
  tokens × hidden × dtype size for `hidden_states` and `residual`, and tokens ×
  `_pp_index_topk` int32 indices for `sparse_kv_indices` (`ModelRunner.run_model`). With
  `pp_send_allgather_group` on, a tensor whose element count divides by the TP width is
  sent as its shard, a TP-width fraction of those bytes. A small decode batch can fall
  under the threshold, so the regime is decided per send, not per site.
- a CPU tensor in the list goes on the PP CPU group, the gloo carrier.
- an empty tensor is skipped on both sides and is no part.

**How the parts combine.** The sender posts each gloo part at `t_send` and each RCCL part
at `e`; `t_send` below is the part's own. The receiver,
`GroupCoordinator.recv_tensor_dict`, takes them in order: the size and the metadata through
`recv_object` (blocking gloo receives of the size, then the object), then each tensor in list order, and it posts a
part's receive only when the previous part is done. With `r_1 = t_recv_posted` and parts
`i` in that order:

- an eager part arrives at `a_i = max(r_i, t_send + L + o + bytes × G)`, and its sender
  side is done at `t_send + o + bytes × G`;
- a rendezvous part leaves at `s_i`, `t_send` or the previous part's sender-side completion if
  later, and ends on both sides at `a_i = max(s_i, r_i - E × G) + T`, `E` its carrier's threshold;
- the next receive is posted at `r_(i+1) = a_i + g_i`, where `g_i` is the TP-wide
  all-gather that the receiving stage runs right after receiving a sharded tensor (an
  ordinary priced collective on the receiver's time), else 0. The receiving worker's cost
  backend prices it (`all_gather_s`, per gathered row; the step stub charges its
  collective coefficient once per row). A PP run at TP above 1 with
  `ATOM_PP_SEND_ALLGATHER` on whose backend prices no all-gather is refused by name when
  the worker installs it ([#1182](https://github.com/jgong5/ATOM/issues/1182)).

The receive ends at the last part's `a_i + g_i`. The send takes an ack when any gloo part
is rendezvous, and `done` is the latest sender-side completion over its gloo parts, the
only ones the sender's host waits for. Its RCCL parts leave no earlier than the previous
send's are done, one stream sending both, and the receiver's stream takes them only after
its own earlier device work, so an RCCL part's receive is posted no earlier than the
receiving stage's device end (#1444). A send whose gloo parts are eager and an RCCL part
rendezvous takes no ack, so nothing tells the sender when its stream is free: its next
forward is refused by name. `(pp_data, n)`
carries each part's carrier and bytes, so the receiver computes both. The lookahead
derivation still holds: every `r_i >= t_recv_posted`, so `done` is at least
`max(t_send, t_recv_posted) + T_min`, the only bound it uses.

**The payload is `hidden_states` + `residual` — real tensors that a simulated run never
materialises.** So its size is computed from geometry, as for KV transfer (`01` D6). The
single intra-node latency and bandwidth of the machine spec do not tell carrier from
protocol, so each carrier (`rccl`, `gloo`) gets these fields under
`interconnect.intra_node.pp.<carrier>`: `eager_threshold_bytes`, `latency_s` (`L`),
`overhead_s` (`o`), `per_byte_s` (`G`) and `rendezvous_fixed_s`, the size-independent part
of `T = rendezvous_fixed_s + bytes × G`. So `T_min` is the least `rendezvous_fixed_s`. PP is
single-node (every PP address is ZMQ IPC), hence the intra-node section. The fields are
optional in the schema and required by a PP run; a machine that does not state them refuses
a PP run by name. `scripts/compass/pp_carriers.py` fits them on two GPUs over RCCL and gloo
(README principle 8). A threshold of 0 states that no size is eager: on MI308X a gloo send
completes only once its receive is posted, at every size probed, so each of its parts is
rendezvous and each send takes an ack.

### Q4: memory — layers split, and the split is ATOM's

`get_pp_indices(num_hidden_layers, pp_rank, pp_size)` (`atom/models/utils.py`, used at
`kv_transfer/offload/config.py::scale_cpu_size_for_pp`, `kv_transfer/offload/config.py::build_lmcache_metadata` and in the Mooncake connector) owns the layer
range per stage. `ModelRunner._get_total_num_layers` already consults `get_pp_indices`
when `pp_group.world_size > 1`.

**Compass calls it; it never re-derives a split.** Same rule as `14` D86 for draft KV
layers, and for the same reason: two spellings of one count drift.

**Weights shard by layer range. KV does not.** What separates them is homogeneity, not
per-layer-ness: KV is per-layer too. *Every* layer carries weights, so a stage's share of
the sharded weight terms is `layers_in_stage / total_layers`; only the layers that cache
the whole history carry paged KV, and inside a span of a hybrid stack that count is not
proportional to the span's length. A stage's share of the KV term is
`paged_layers_in_stage / total_paged_layers`, and on a hybrid the two fractions differ.

The weights ratio is itself exact only where the layer kinds are equally sized, and on this
hybrid they are merely close. The two kinds are different modules — `Qwen3NextAttention`
against `Qwen3_5GatedDeltaNet` — and at TP1, counted from the vendored config over their
projection shapes with norms and biases omitted (under 0.01% of a layer), a
`full_attention` layer is **372,244,480** parameters against **383,262,720** for a
`linear_attention` one, **+2.96%**. That moves a stage's true weight share off the layer
ratio by at most **0.242%**, at pp = 7; 0.197% at pp = 6 and 0.104% at pp = 3. Two orders
of magnitude inside the ≤10% target a non-KV memory term carries, so the layer range stays
the weights key here — but it stays one because the two kinds happen to be nearly the same
size, not because weights are per-layer, and on a hybrid whose kinds differ more it would
not. How close that ratio has to be is a weights-term question rather than this decision's.

**The paged count is read, not computed.** It comes from the same two sources as the split
itself — `get_pp_indices` for the span, and the model's own `layer_types` for which layers
inside it are paged (`atom/compass/backends/geometry.py`'s `paged_layers`, which refuses a
kind it does not recognise rather than counting it as ordinary attention). An earlier form
of this paragraph gave the KV share as the layer ratio. That is exactly the re-derivation
the rule above forbids: it recomputed from the layer count something ATOM already holds,
and the two spellings disagree on every hybrid.

**A second spelling of this count already exists in the engine, and it is not the one to
read.** `GDNStateMixin._init_gdn_state` (`atom/model_ops/attentions/gdn_attn.py`) sets
`num_full_attn` by dividing `num_hidden_layers` by `full_attention_interval`, and the
attention sizing in the same file consumes it. That count is **global**, so it cannot
answer what a stage holds at all. It agrees at 16 on this config and not by luck:
`Qwen3_5TextConfig.__init__` (`atom/model_config/qwen3_5.py`) fills `layer_types` *from*
that interval when a config omits it, so anything built through ATOM's own config class is
consistent by construction. A config carrying an explicit non-periodic `layer_types` would
separate the two, and `layer_types` is the one that stays right.

Derived at `feature/atomcompass_new` `92f1fdafe` from `get_pp_indices`, with
`VLLM_PP_LAYER_PARTITION` cleared — left set it overrides the partitioner and a layout out
of the environment reads as ATOM's — on the 64-layer hybrid vendored at
`tests/compass/qwen3_5_27b_config.json`: one `full_attention` layer in four, so 16 of the
64 are paged. The layers that do not divide evenly are added walking back from the
*second-to-last* partition, so the **last stage never takes one** and the rest fill in from
the right; at pp = 5 that reaches stage 0, the first. *The middle stages* is true of 3, 6
and 7 here and false of 5, which is why no row below is the split a reader would write
out:

| PP | `get_pp_indices` spans | layers held | paged layers | stages where `held/64` = `paged/16` |
|---|---|---|---|---|
| 2 | 0-32, 32-64 | 32, 32 | 8, 8 | all |
| 3 | 0-21, 21-43, 43-64 | 21, 22, 21 | 5, 5, 6 | none |
| 4 | 0-16, 16-32, 32-48, 48-64 | 16, 16, 16, 16 | 4, 4, 4, 4 | all |
| 5 | 0-13, 13-26, 26-39, 39-52, 52-64 | 13, 13, 13, 13, 12 | 3, 3, 3, 4, 3 | stage 4 |
| 6 | 0-10, 10-21, 21-32, 32-43, 43-54, 54-64 | 10, 11, 11, 11, 11, 10 | 2, 3, 3, 2, 3, 3 | none |
| 7 | 0-9, 9-18, 18-27, 27-36, 36-45, 45-55, 55-64 | 9, 9, 9, 9, 9, 10, 9 | 2, 2, 2, 3, 2, 2, 3 | none |
| 8 | 0-8, 8-16, 16-24, 24-32, 32-40, 40-48, 48-56, 56-64 | 8, 8, 8, 8, 8, 8, 8, 8 | 2, 2, 2, 2, 2, 2, 2, 2 | all |

**The pp = 6 row falsifies the layer ratio in one line: stages 1 and 3 each hold 11 layers
and carry 3 and 2 paged ones.** `layers_in_stage / total_layers` is 11/64 for both, and
their shares of the KV are 3/16 and 2/16. The ratio understates stage 1 by 8.3% and
overstates stage 3 by 37.5% — against the **≤ 5%** target the KV term carries, the
tightest in the project, where every other memory term is gated at 10%. The sign of the
error changes between two stages of one deployment, so no single correction factor absorbs
it. At pp = 3 the stage holding the *most* layers holds the *fewest* paged ones of the
three: 22 layers carry 5 where 21 carry 6.

**Uniform stacks are unaffected.** Where every layer is paged the two fractions are equal
by construction and the layer ratio is correct — that is the case anyone checks first, and
it is why this stood. The widths above where they agree at every stage, 2, 4 and 8, are
the ones where the paged period of this model, 4, divides every span.

The Class-C memory readings scale by neither key, since HIP context and collective
buffers are per-process; that half of the sentence this replaces was already right.
**Memory readings depend on PP degree as well as TP width**, and `07` D41 keys them by the
whole parallel configuration rather than deriving one degree from another.

The per-stage *block count* these paged counts produce, and the `all_reduce(MIN)` that
picks one of them for the whole deployment, are `03` D14's question and not this one's.

### What is not established

**Whether ATOM microbatches PP.** A grep for `microbatch` / `micro_batch` / `num_micro`
across `pp_comm.py` and `engine_core.py` returns nothing, which suggests stage-to-stage
execution of whole batches rather than a 1F1B-style pipeline. That matters a lot — without
microbatching, PP's bubble is the whole of a stage's idle time and the LP structure is
simple; with it, each microbatch is a separate event and the grant count rises by the
microbatch factor. **Recorded as T64**, and it is the first thing to settle before any PP
work.

---

## D92. EP — invisible communication, overlapped under TBO

**Revised 2026-10-18 by the owner's
[ruling on #531](https://github.com/jgong5/ATOM/issues/531#issuecomment-6244543796):** MORI
dispatch and combine do not occupy the whole device; under TBO they overlap the other
micro-batch's compute on either of ATOM's two paths, the comm stream or split send and
receive (Q3).

### The EP group, per supported configuration

**Measured 2026-09-21 by P0.6, answering T65.** ATOM at `7fc7a5ddd`; aiter at
`23f83724f` (`v0.1.20-103-g23f83724f`) in container `jgong5_vllm`, re-checked against
`f4e7c7509` (`v0.1.21.dev0-49-gf4e7c7509`) in node 18's `xiaobizh_n18` and
`xiaobizh_n18_cpu`, where every aiter line quoted below is byte-identical at the same
numbers. Two aiter versions are in circulation on this project and they agree here; that
they agree is a reading, not a guarantee, and no artifact key currently records which one
answered — **T86**.

**ATOM does not construct the EP group — aiter does.** No ATOM file assigns `_EP`, and
every use site imports `get_ep_group` from `aiter.dist.parallel_state` (`moe.py::FusedMoEMethodBase._maybe_make_prepare_finalize`,
`fused_moe/mori_v2_prepare_finalize.py::_init_cco_comm`, `fused_moe/mori_v2_prepare_finalize.py::make_mori_v2_prepare_finalize`, `fused_moe/flydsl_mega_experts.py::run_mega_moe`,
`eplb.py::EPLBManager._maybe_initialize_runtime`, `models/glm4_moe.py::Glm4MoE.__init__`, `models/qwen3_next.py::Qwen3NextSparseMoeBlock.__init__`,
`model_runner.py::ModelRunner._force_aiter_unreg_capture_for_piecewise`). Both of ATOM's distributed-init paths —
`init_pp_aware_dist_env` (`distributed/pp_comm.py`) when `pp_size > 1`, aiter's
`init_dist_env` (`aiter/ops/communication.py:22`) otherwise, chosen by the
`pipeline_parallel_size > 1` branch of
`model_runner.py::ModelRunner._setup_device_and_distributed` — end in
aiter's `initialize_model_parallel`, which builds the
group at `aiter/dist/parallel_state.py:1926-1945` out of

```python
group_ranks = (
    all_ranks.transpose(1, 2)
    .reshape(-1, data_parallel_size
                 * prefill_context_model_parallel_size
                 * tensor_model_parallel_size)
    .unbind(0)
)
```

over `all_ranks = torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)`
(`parallel_state.py:1833-1839`). The `transpose(1, 2)` swaps DP and PP, so **the EP group
is every rank of one PP stage — it spans DP, PCP and TP, its size is `dp × pcp × tp`, and
there is one group per PP stage.** aiter says so itself:
*"all2all lives in ep group, which is merged from dp and tp group"*
(`dist/device_communicators/base_device_communicator.py:42`).

Membership, from re-executing those expressions verbatim at each configuration ATOM
ships or documents:

| Configuration | world | dp/pp/pcp/tp | EP groups | size |
|---|---|---|---|---|
| `-tp 8` (`recipes/Qwen3-235b.md:24`) | 8 | 1/1/1/8 | `[0…7]` | 8 |
| `-tp 2 --enable-dp-attention` (`recipes/GPT-OSS.md:34`) | 2 | 2/1/1/1 | `[0,1]` | 2 |
| `-tp 8 --enable-dp-attention` (`recipes/TBO.md:63`, `recipes/GLM-5.md:205`, `recipes/DeepSeek-V4.md:126`) | 8 | 8/1/1/1 | `[0…7]` | 8 |
| `-tp 4 -dp 2` (`docs/distributed_guide.md:20`) | 8 | 2/1/1/4 | `[0…7]` | 8 |
| `-tp 2 -pp 2` | 4 | 1/2/1/2 | `[0,1]`, `[2,3]` | 2 |
| `-tp 4 -pcp 2` | 8 | 1/1/2/4 | `[0…7]` | 8 |
| `-tp 2 -pp 2 -dp 2` — **refused** at `engine_core_mgr.py::CoreManager.__init__` | 8 | 2/2/1/2 | `[0,1,4,5]`, `[2,3,6,7]` | 4 |

Under DP-attention `CoreManager` rewrites `dp := dp × tp, tp := 1` before any of this
(`engine_core_mgr.py::CoreManager.__init__`), which is why those rows carry `tp 1`. The torch world is
`dp × pcp × tp` either way: `init_dist_env` passes `world_size = pp × tp × pcp` with `pp`
pinned to 1 (`aiter/ops/communication.py:33-40`) and `init_distributed_environment`
multiplies DP back in (`parallel_state.py:1726-1729`); the PP branch computes the same
index itself in `model_runner.py::ModelRunner._setup_device_and_distributed`.

**Ranks are contiguous whenever `pp == 1`**, which is every reachable EP configuration.
The strided last row is the only non-contiguous case, and ATOM refuses it
(*"Pipeline parallel combined with data parallel is not supported yet."*).

**Membership is not topology-aware.** It is index arithmetic on
`torch.arange(world_size)`; nothing reads the interconnect. *Kernel selection* is:
`All2AllManagerBase` sets
`self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))`
(`base_device_communicator.py:55`), MoRI picks `IntraNode` against `InterNodeV1` from it,
and ATOM deliberately shares that one probe rather than re-deriving it from a width
(`moe.py::FusedMoEMethodBase._maybe_make_prepare_finalize`).

**Two configurations where ATOM does not own the answer at all.** Under the vLLM plugin
ATOM adopts vLLM's group wholesale — `aiter_ps._EP = getattr(vllm_ps, "_EP", None)`
(`plugin/vllm/tp_group_reuse.py::init_aiter_dist_from_vllm`) — so membership is vLLM's decision there, and a
vLLM configuration with no `_EP` installs `None` and fails at the first `get_ep_group()`
(`parallel_state.py:1586`). And EP width is never requested directly: ATOM has no
`--ep-size` flag (`model_engine/arg_utils.py::EngineArgs.add_cli_args` offers only the boolean
`--enable-expert-parallel`), and the SGLang and rtp-llm frontends, which do carry an
`ep_size`, have it reduced to that boolean (`plugin/config.py::_generate_atom_config_from_sglang_config`, `plugin/config.py::_generate_atom_config_from_rtpllm_config`) — so `ep_size=2`
under `-tp 8` is accepted and silently means 8.

### `moe_parallel_config.ep_size` is a second number, and it is not the group size

`FusedMoEParallelConfig.make` computes its own: `ep_size = tp_size; ep_rank = tp_rank`
(`moe.py`), where `tp_size` has been flattened across DP **only if**
`enable_dp_attention or moe_ep_flatten_tp_across_dp` (`moe.py`) and
folded with PCP only under `ATOM_PCP_MOE_MERGE` (`moe.py`). The two numbers agree
in every configuration ATOM ships a recipe for, and disagree in one it documents:

| Configuration | group size (aiter) | `moe_parallel_config.ep_size` | agree |
|---|---|---|---|
| `-tp N`, DP 1 | `N` | `N` | yes |
| `-tp N --enable-dp-attention` | `N` | `N` | yes |
| vLLM plugin `--enable-expert-parallel` (`plugin/config.py::_generate_atom_config_from_vllm_config`) | `dp × tp` | `dp × tp` | yes |
| **`-tp 4 -dp 2 --enable-expert-parallel`, no DP-attention** | **8** | **4** | **no** |
| `-pcp P` without `ATOM_PCP_MOE_MERGE` | `P × tp` | `tp` | **no** |

MoRI v1 is handed `num_ep_ranks` from the group and `num_local_experts` from the config
number (`moe.py::FusedMoEMethodBase._maybe_make_prepare_finalize`); MoRI v2 derives both from the group
(`mori_v2_prepare_finalize.py::make_mori_v2_prepare_finalize`). The two paths therefore disagree exactly where
the two numbers do. **Compass must carry both and assert they agree**, because ATOM
asserts nothing here — recorded as **T83**, which also covers `local_ep_size`
(`moe.py::FusedMoEParallelConfig.make`, MoRI's `gpu_per_node`) omitting PCP while the group includes it.

### EP without DP runs no all-to-all at all

`use_all2all_kernels` requires `dp_size > 1` (`moe.py`) and is the sole gate on
building `MoriPrepareAndFinalize` (`moe.py::FusedMoEMethodBase.maybe_make_prepare_finalize`); without it `self.fused_experts`
stays `None` (`moe.py::FusedMoEMethodBase.init_prepare_finalize`) and the layer falls through to a plain
`fused_moe(…, expert_mask=…)` (`moe.py::UnquantizedFusedMoEMethod.apply`). So under
`-tp 8 --enable-expert-parallel` — `recipes/Qwen3-235b.md:24`, the flagship EP recipe —
the MoE is masked local-expert compute plus the ordinary TP all-reduce
(`moe.py::FusedMoE.forward_impl_graph`), and moves **zero all-to-all bytes**. That is the mechanism behind
`04`'s *"at `ep_size == tp_size` EP is close to a no-op"*, and it is stronger than close.

**A cost model must refuse to price a MoRI all-to-all at `dp_size == 1` rather than price
zero bytes** — a confident, precise, fictional number is the archetypal failure `README`
names. The gate to mirror is `moe.py::FusedMoEParallelConfig.use_all2all_kernels` in full, including `dp_logical_ratio == 1`
and `_has_module("mori")`. One exception: `--moe-backend mega` installs `MegaFusedExperts`
unconditionally (`moe.py::MegaMxfp4MoEMethod.init_prepare_finalize`) and reads the group directly for its rank and world
(`flydsl_mega_experts.py::run_mega_moe`), so it does run peer-to-peer at `dp_size == 1`;
`config.py::Config.__post_init__` refuses `mega` without EP.

### Q1: no LPs added — the conclusion holds, the reason under it did not

**The conclusion stands, and D93's formula is unchanged.** The EP group is exactly the set
of ranks of one PP stage, and PP is the only LP-adding dimension (D88, D93). The one case
where EP's membership could cut across an LP boundary is `pp > 1` together with `dp > 1`,
and `engine_core_mgr.py::CoreManager.__init__` refuses it.

**The reason previously given here was wrong, and is recorded rather than quietly
dropped**, because a future reader who lifts that refusal will need to know which half
survived. It read that EP *"inherits the TP group"*, citing `moe.py::FusedMoEParallelConfig.make`; the cited line is a
comment inside the PCP-merge block explaining that the *integers* `ep_size`/`ep_rank`
inherit `tp_size`/`tp_rank`, and says nothing about the communicator. The caveat below it
was stated conditionally — *"if a deployment configures EP to span the DP dimension"* —
and EP spans DP in every configuration where DP exists. The second cross-DP
synchronisation that caveat predicted **does** appear: MoRI dispatch/combine, over a
membership different from `sync_dp_metadata`'s. ATOM confirms the two groups are distinct
objects in code — `eplb.py::EPLBManager._maybe_initialize_runtime` compares the DP group's global ranks against the EP
group's to decide `_dp_is_migration_group`, a comparison with no purpose if they were the
same group.

`get_max_tokens_across_dispatchers` (`moe.py`) was cited here as hinting at a
cross-dispatcher reduction. It is `def …(input): return input.item()` — no collective —
and a tree-wide `grep -rn` returns the definition and this document. It has no callers.

**D92's decision-log row below is left unamended**: its parentheticals *"inherits the TP
group"* and *"remainder included"* are the two claims this section corrects, and rewriting
a decision is the project owner's call. Registered as a pending amendment in
[`12_open_items.md`](12_open_items.md) §5.

### Q2: no scheduling coupling

Expert assignment is a function of the routing computed inside the layer, not of a
scheduler decision — and **real routing is data-dependent, so it is not derivable from
geometry**. The closed-form flat ring
`expert_ids = (p % ep_size) × L + (p // ep_size) % L` is `init_balance_router_logits`
(`moe.py`), the **synthetic** router built only under `--fake-eplb`
(`moe.py::FusedMoE.__init__`: *"if atom_config.fake_eplb else None"*). Class A under `--fake-eplb`,
and nothing outside it.

### Q3: cost — the hard part, and it is already characterised

From `07` D40 class (c): MoE all-to-all is **MORI, called inside `moe_forward`, invisible
at any degree**. Zero dispatcher events, so it needs `04` D21's *stated node* plus the
stated-entry-point benchmark of `07` D40.1 — the same tool class (b) uses.

MORI dispatch and combine do not hold the device, and under TBO they overlap the other
micro-batch's compute; `04` D19 states their block counts, ATOM's two TBO paths and the
price.

### Q4: memory — experts shard contiguously, and an indivisible count is refused

`determine_expert_map` (`fused_moe/expert_layout.py`) gives rank `r` the
contiguous run `[r×L, (r+1)×L)` with `L = E // ep_size`, and gives any remainder to the
**last** rank (`expert_layout.py::determine_expert_map`) — it is not left unused. Expert weight bytes per
rank are `L × bytes_per_expert`, exact from geometry (Class A).

**There is no remainder for a memory model to reproduce**, because a configuration whose
expert count does not divide by `ep_size` is refused outright:
`assert self.global_num_experts % self.ep_size == 0` whenever `use_ep`
(`moe.py::FusedMoE.__init__`). MoRI derives a token's destination as
`expert_id // num_experts_per_rank` (`distributed/simulated_tp.py::_reject_unsupported`,
`moe.py::FusedMoEParallelConfig.use_all2all_kernels`) and cannot represent an uneven last rank, so ATOM refuses rather than
pads. The *"a remainder is left unused"* comment this section used to quote is
in `moe.py::init_balance_router_logits`, inside the `--fake-eplb` synthetic router, not the real path.

---

## D93. Composition, and what the LP count actually is

Strategies compose, and the LP count is what the clock protocol pays for:

```
  LPs  =  1                                   the traffic source (the harness)
        + (number of deployments)             1 aggregated, 2 for 1P1D (one per PD role)
          x ( 1                               its frontend: the API server's event loop
            + pp_size )                       its engine: one LP per PP stage; a DP group,
                                              and tp_size and ep_size, collapse into it
```

So a TP8/EP8 single-node aggregated deployment is **3 LPs** (traffic, frontend, engine),
the same as TP1; 1P1D is **5** (traffic, frontend-P, engine-P, frontend-D, engine-D); and
a PP4 aggregated deployment is **6** (`01` D3, D3.1). **The protocol's cost tracks PP
degree and PD roles,
not GPU count**, which is the property that makes `01`'s single-CA decision hold as the
milestones widen.

**The composition worth watching** is DP × PP: `dp_size × pp_size` engine cores means the
shape collective of D90 runs *per stage*, across DP, at microsecond-lookahead boundaries.
That is the densest grant traffic any milestone produces and it comes with PP, after v1
(`16` D95). Sizing it before PP work starts is cheaper than discovering it.

---

## D94. What M1 needs, and what M2 adds

The split, because "parallelism support" is otherwise read as one late lump. The
milestones are `16` D95's:

| | **M1 — plumbing, with fake models; no parity test** | **M2 — scheduling parity, then accuracy** |
|---|---|---|
| **TP** | LP collapse; the parallel layout as an artifact key | priced collectives at the widths the nightly configurations run; width transfer (T21) comes after v1 |
| **DP** | both collectives run for real; `max`-over-ranks step duration; dummy-batch pricing for idle ranks | the collectives' own cost, including the gather sized by the peers' token counts |
| **EP** | none: EP's plumbing moved into M2's EP8 step | in the EP8 step, group membership established — it is `dp × pcp × tp` within one PP stage, built in aiter (D92); then the MORI all-to-all within a node absorbed by the step linear law fitted under the EP8 engine config, routing declared uniform (D90); under TBO MORI dispatch and combine overlap the other micro-batch's compute on either of ATOM's two paths (D92 Q3) |

**PP is in neither column.** Its plumbing (one LP per stage, transfer as a size from the
spec, layer split via `get_pp_indices`, memory keyed by PP degree) and its accuracy
(bubble fidelity; microbatching if it exists, T64) come after v1, and so does EP beyond
one node (T85).

**The test that opens M2** is not "does it produce plausible numbers" — the fake model
guarantees it will. It is: **does a fake-model run reach the same scheduling decisions
as the real engine at the same configuration?** Its legs, in order:

1. TP2 / DP2 on one node, `-tp 2 --enable-dp-attention`;
2. DeepSeek-V4-Pro 1P+1D in the nightly DP-attention configuration, expert parallelism
   off;
3. M2's EP8 step repeats both with expert parallelism on: EP2 as
   `-tp 2 --enable-dp-attention --enable-expert-parallel`;
4. then EP8.

EP2 takes that form because `-tp 2 --enable-expert-parallel` at DP 1 runs no all-to-all
at all (D92), so that leg would pass while exercising nothing (T84, answered by `16`
D95). The test is checkable without a cost model, it exercises exactly the couplings
this topic is about, and ATOM's own `test_dp_load_balance.py`, `test_dp_metadata.py`,
`test_dp_sync_layout.py` and `test_forward_mode.py` already cover the pieces on the
CPU-only path (`08` D43.1).

---

## Open issues

- **T64 — does ATOM microbatch PP?** Unestablished, and it changes both the LP event rate
  and the bubble model. Settle before any PP work, which comes after v1 (`16` D95).
- **T65 — EP group membership** per supported configuration. **Answered 2026-09-21 by
  P0.6**, in D92: the group is `dp × pcp × tp` within one PP stage, built in aiter rather
  than in ATOM. EP does span DP wherever DP exists, and the second cross-DP barrier does
  appear — but the LP collapse survives anyway, because the group is exactly one PP
  stage's ranks and PP>1 with DP>1 is refused. Of its three successors, two are open:
  **T83** (the group size and `moe_parallel_config.ep_size` disagree in two
  configurations, one of them documented) and **T85** (multi-node EP rank-to-node
  mapping, unverified; after v1 by `16` D95). **T84** (which EP2 D94's parity test means)
  is answered by `16` D95, and D94 states its answer.
- **PP degree is in the memory readings' key** (`07` D41), and nothing has
  measured whether the Class-C constants move with it. One startup per PP degree settles
  it; recorded as **T66**.
- **The DP `max`-over-ranks rule is unmeasured.** `01`'s 0.06% figure is a TP result. What
  the spread across DP ranks actually is — and therefore how much the padding to
  `unified_bs` costs — has not been measured. **T67**.
- Nothing here covers **PCP / DCP** (`pcp_size`, `dcp_world_size`), which appear in the
  topology and in `03`'s note that `hash_block_size = block_size × dcp_world_size`. They
  are out of scope for v1 (`16` D95 lists DCP after it), and recorded so they are not
  discovered late. `17` D100 describes DCP's correction as an optional recipe of the step
  fit backend, also after v1.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D88 | One frame of four questions per strategy — LPs and lookahead, scheduling coupling, cost, memory. **Only PP adds logical processes**; TP, DP and EP each sit behind an existing barrier. | 2026-09-19 |
| D89 | TP is the settled instance and supplies the per-width discipline: width is a key, not a parameter. | 2026-09-19 |
| D90 | DP's two collectives **run for real** — both reduce over scheduling metadata, never over model outputs, so the real reduction is more faithful than a model and free. The DP group stays one LP, with one member process per rank. With host time the group's device end is the `max` over ranks of each rank's own device end, computed not rank-0-sourced by one `all_reduce(MAX)`, and each rank leaves at its own charge; with no host time an output rank leaves at the `max` of the step seconds. The step targets are exchanged by one `all_gather` inside the predicted forward (#988); both collectives are K6 waits that carry no cost; the lockstep `all_reduce` keeps its payload. With EP off the MoE segment reads `T_moe`, the rows the group's MoE gather carries (`dp x running_bs x max_seqlen_q` when every rank decodes, `sum(num_tokens_across_dp)` otherwise), and is rank-invariant by construction, which makes `max` the per-layer critical path. With EP on the group dispatches the same `T_moe` rows to MORI, padded to the group's height on every rank when no rank prefills (`running_tokens_are_unified`), replayed or eager, and each rank's own when one does; `max` rests on uniform routing, an assumed treatment under which the step linear law fitted under the EP8 engine config absorbs MORI through its `moe_rows` term and intercepts; the measured imbalance is reported beside it. Idle ranks are priced through `dummy_execution`'s own `forward`. Q4's replicated weights and DP-independent width key hold for plain DP only; under DP-attention the routed experts shard over the DP group and the width key is open. | 2026-09-19, revised 2026-09-28, 2026-10-01 (owner's DP ruling, [#470](https://github.com/jgong5/ATOM/issues/470#issuecomment-5933154215)), 2026-10-09 (`T_moe`, [#502](https://github.com/jgong5/ATOM/issues/502#issuecomment-6081138199)), 2026-10-10 (Q4, [#702](https://github.com/jgong5/ATOM/issues/702)), 2026-10-15 (the `all_gather` and each member's own step end, [#988](https://github.com/jgong5/ATOM/issues/988#issuecomment-6183664946)), 2026-10-15 (EP on: uniform routing, [#627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6194442788); the row count, [#1106](https://github.com/jgong5/ATOM/issues/1106#issuecomment-6236367874)) and 2026-10-16 (the device end, [#1267](https://github.com/jgong5/ATOM/issues/1267#issuecomment-6215701399)) |
| D91 | PP is one LP per stage at microsecond lookahead, and PP boundaries are never a hierarchical-CA cut point. The inter-stage transfer is a **size from the machine spec**, like KV transfer. Layer split comes from `get_pp_indices`, never re-derived; weights shard by that range but **KV shards by the paged-layer count inside it**, which on a hybrid is not proportional to it. Memory readings gain a PP-degree key. Revised: a send part completes on the sender's clock when eager and at `max(t_send, r_i - E × G) + T` when rendezvous, a threshold's bytes `E` leaving before the post (#1458), `r_i` the time its receive is posted, a per-carrier size threshold deciding which, and a send's `done` is the latest over its gloo parts, the only ones its host waits for, its RCCL parts holding the next send's (#1383) and, carried by the ack, the device start of the stage's later forwards, and posted no earlier than the receiving stage's device end (#1444), and a receive is posted at the receiving forward's prepare end (#1443); the receiver returns that completion, when a gloo part is rendezvous, on `stage(k+1)->stage(k):pp_ack#dp0`, lookahead `T_min - L_data`, and both waits receive it (K5). Confirmed by the owner on 2026-10-02: the send reaches the next stage on `stage(k)->stage(k+1):pp_data#dp0`, and the TP-rank-0 workers carry both channels' frames over the PP CPU group. | 2026-09-19, revised 2026-09-28 and 2026-09-30, confirmed 2026-10-02, revised 2026-10-18 and 2026-10-19 |
| D92 | EP adds no LPs (inherits the TP group) but its all-to-all is invisible and must be a stated node; under TBO it overlaps the other micro-batch's compute, on the comm stream or as split send and receive (`04` D19). Expert sharding is Class A, remainder included. | 2026-09-19, revised 2026-10-18 ([#531](https://github.com/jgong5/ATOM/issues/531#issuecomment-6244543796)) |
| D93 | LP count = 1 traffic LP + per deployment (1 frontend LP + one engine LP per PP stage): 3 aggregated, 5 for 1P1D, independent of GPU count. The clock protocol's cost tracks PP degree and PD roles, not width. | 2026-09-19, revised 2026-09-28 |
| D94 | Parallelism splits across `16` D95's milestones: TP and DP plumbing (LP structure, couplings, memory shape) at **M1**, with no parity test; at **M2**, scheduling parity and then cost accuracy. The parity test needs no cost model and runs TP2/DP2 as `-tp 2 --enable-dp-attention`, then DeepSeek-V4-Pro 1P+1D in the nightly DP-attention configuration with expert parallelism off; M2's EP8 step, which also carries EP's plumbing, repeats both with expert parallelism on, EP2 as `-tp 2 --enable-dp-attention --enable-expert-parallel`, then EP8. PP, its accuracy, and EP beyond one node come after v1. | 2026-09-19, revised 2026-10-10 (`16` D95) |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T64 | Establish whether ATOM microbatches PP — changes the LP event rate and the bubble model | a grep finds nothing; needs reading `pp_transport.py` and one PP2 run; after v1 with PP (`16` D95) |
| ~~T65~~ | ~~Establish EP's group membership per supported configuration; if EP spans DP, D92's collapse does not hold~~ — **answered 2026-09-21 by P0.6**, in D92 above; successors T83, T84, T85 | — |
| T83 | The EP group's size and `moe_parallel_config.ep_size` disagree at `-tp N -dp M --enable-expert-parallel` without DP-attention (group `N×M`, config `N`) and at `-pcp P` without `ATOM_PCP_MOE_MERGE`; `local_ep_size` also omits PCP while the group includes it | one 8-GPU startup logging `all2all_manager.world_size` against `moe.num_local_experts`; or an owner statement that the combination is unsupported |
| ~~T84~~ | ~~Decide which EP2 D94's scheduling-agreement test means — `-tp 2 --enable-expert-parallel` at DP 1 runs no all-to-all, so that leg exercises nothing~~ — **answered by `16` D95**: EP2 is `-tp 2 --enable-dp-attention --enable-expert-parallel`, stated in D94 | — |
| T85 | Multi-node EP rank-to-node mapping is assumed, not verified: MoRI infers node identity as `ep_rank // gpu_per_node`, which needs consecutive EP ranks to be physically consecutive GPUs | a 2-node DP+EP run logging `all2all_manager.internode` and each rank's EP group; after v1, since `16` D95 takes EP beyond one node out of scope |
| T66 | Measure whether the Class-C runtime constants move with PP degree | one engine startup per PP degree |
| T67 | Measure the step-duration spread across DP ranks, and what padding to `unified_bs` costs | needs a DP2 run with per-rank step timing |
| T78 | `qwen3_5.py::Qwen3_5Model` and `glm4_moe.py::Glm4MoeModel` declare `"intermediate_tensors": 0`, so neither model can run PP at compilation level >= 2 | upstream ATOM fix; PP comes after v1 (`16` D95), so this is on no v1 path |
| T79 | `gdn_attn.py::GDNAttentionMetadataBuilder.sub_pool_specs` mixes a PP-local layer count with a global one; KV sizing sign-flips at PP2 | upstream ATOM fix; blocks real-model PP measurement, which comes after v1 |
| T82 | D91 Q2's "no scheduling coupling" is contradicted by `scheduler.py::Scheduler.schedule`, which skips `_pp_inflight_token_block` seqs inside the decode admission loop | the amendment to D91 and its LP consequence is its own task; this PR registers the contradiction rather than rewriting the decision |
