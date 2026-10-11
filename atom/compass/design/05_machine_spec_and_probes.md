# ATOM Compass — Design Topic 5: The Machine Specification and its Probes

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Depends on:** `03_memory_and_kv_model.md` D14/D15 (which named the artifact), and feeds
`04_model_capture_and_cost_ir.md` (cost) and `01_execution_and_time_model.md` D3
(lookahead floors).

**Revised 2026-10-10 for `16` D95:** the schema holds what the MI455 roofline projection
needs: an FP4 matrix-core rate, the size of the domain the in-node fabric joins, and a
*sourced* or *assumed* status on every field (D25).

**Scope.** The single input artifact that tells Compass what machine it is simulating,
its schema, and the tools that fill it in. A project requirement is that compute, memory
size, bandwidth and interconnect are **configured and not read from a device runtime**, so
this artifact is load-bearing rather than convenient.

---

## D24. The separation rule

### Problem

A simulated run needs numbers describing a GPU, a host CPU, a fabric, and a software
stack. Some of those are properties of hardware; some are properties of the deployment
ATOM was launched with. Mixing them produces a spec that silently contradicts the engine
it is meant to describe.

### Decision

> **The machine spec describes the machine. ATOM's config describes the deployment.
> Anything ATOM could configure belongs in ATOM's config, and Compass reads it.**

Consequences in both directions:

**Stays out of the spec, because ATOM already owns it:** `gpu_memory_utilization`,
`max_num_seqs`, `max_model_len`, `kv_cache_dtype`, `block_size`,
`cudagraph_capture_sizes`, `tensor_parallel_size`, compilation level, cudagraph mode.

**Stays out of the spec, and ATOM does not own it yet — so ATOM should.** Verified on
`feature/atomcompass_new`: there is **no thread configuration anywhere** in `atom/`. No
`set_default_executor`, no `ThreadPoolExecutor`, no `max_workers` under `entrypoints/` or
`model_engine/`; `arg_utils.py` has no thread or worker argument; and
`EngineCore.output_thread` / `input_thread` are hardcoded singletons
(`engine_core.py::EngineCore.__init__`). Tokenization runs on Python's *implicit* default executor
via `await loop.run_in_executor(None, do_preprocess)` (`api_server.py::generate_async`, `generate_async_multimodal`,
`generate_async_fanout`, `setup_streaming_request`, `setup_streaming_request_fanout`), whose width is `min(32, cpu_count + 4)` and is named nowhere.

| Add to ATOM | Why |
|---|---|
| `--preprocess-pool-width`, applied via `loop.set_default_executor(ThreadPoolExecutor(max_workers=N))` | an unnamed implicit default cannot be tuned or reproduced; this is useful to ATOM independent of Compass |
| expose the resolved value on `Config` | Compass reads it rather than re-deriving `min(32, cpu+4)` |

The engine output-thread count stays at 1 — changing it changes ordering semantics — but
it must be **readable**, not inferred.

### Open issues

- Whether `--preprocess-pool-width` belongs on `EngineArgs` or is server-only. It is
  consumed in the API-server process, so server-only is defensible.

---

## D25. Schema

One artifact, four sections, authored outside Compass and **echoed verbatim into every run
artifact**.

```yaml
schema_version: 1
name: mi355x-8gpu-2node

provenance:                      # one block, not per-field tags
  authored_by: <who>
  date: 2026-09-18
  method: datasheet | probed | transferred-from:<spec-name> | assumed | mixed
  fragments: [...]               # filled by `merge`, see D26
  assumed: [...]                 # a `mixed` spec only: its assumed fields, rule 5
  notes: free text

host:
  cpu:
    cores_physical: 96
    cores_logical: 192
  tokenizers:                    # keyed by tokenizer identity, not by model - see below
    - id: qwen3-151k-bpe
      backend: fast              # fast (Rust) vs slow (Python) differ by >10x
      vocab_size: 151936
      fingerprint: sha256:<tokenizer.json>
      applies_to: [Qwen3ForCausalLM, Qwen3MoeForCausalLM]
      encode_fixed_s:          3.0e-4
      encode_tokens_per_s:     2.0e6
      decode_fixed_s:          1.5e-4
      decode_tokens_per_s:     3.0e6
      derate:                  0.85
  ipc:
    zmq_roundtrip_s:           5.0e-5
    shm_broadcast_s:           2.0e-5
  admission_fixed_s:           9.0e-3   # HTTP arrival -> `waiting`, EXCLUDING tokenize

device:
  name: MI355X
  arch: gfx950
  count_per_node: 8
  memory:
    capacity_bytes:            288.0e9
    bandwidth_bytes_per_s:     8.0e12
    derate:                    0.85
  compute:
    bf16_flops:                2.5e15
    fp8_flops:                 5.0e15
    fp4_flops:                 10.0e15  # optional: absent where there is no FP4 matrix core
    derate:                    0.70
  runtime_constants:           # the "table, not a law" terms, keyed by TP width
    # Device memory held OUTSIDE the torch allocator: HIP context, loaded code
    # objects, RCCL/AITER collective staging buffers. Device-wide, not per-process.
    driver_and_collective_reserve_bytes: {1: 970.0e6, 2: 7.2e9, 4: 7.6e9, 8: 11.2e9}
    # Torch-allocator bytes still resident after weight loading finishes, i.e. not
    # weights and not freed: AITER CustomAllreduce pool + its two-stage kernel.
    allocator_retained_after_load_bytes: {1: 1.1e6, 2: 2.17e9, 4: 2.17e9, 8: 2.17e9}
    # Buffers allocated once and alive for the whole run: allocate_forward_vars,
    # attention metadata. Flat in TP width; differs only by model.
    persistent_forward_buffer_bytes: 124.0e6
    # Memory the CUDA-graph private pool holds, on top of the captured activations.
    cudagraph_pool:
      w1_base_bytes:                   95.5e6
      w1_bytes_per_captured_token:     0.318e6
      w_gt1_flat_bytes:                109.0e6
  activations:                 # optional: per model, keyed by architecture and config.json
    - id: qwen3.8-27b
      applies_to: [Qwen3_5ForConditionalGeneration]
      fingerprint: sha256:<config.json>
      # warmup `peak - current` over the warmup tokens, by TP width
      bytes_per_token: {1: 180480}
  software_pinned_to:          # these constants belong to the stack as much as the silicon
    rocm:  "7.2.4"
    aiter: "<commit>"
    rccl:  "<version>"

interconnect:
  intra_node:
    topology: fully_connected
    domain_devices:             8        # devices the fabric joins; may exceed count_per_node
    link_bandwidth_bytes_per_s: 1.0e12
    link_latency_s:             2.0e-6
    derate:                     0.80
  inter_node:
    link_bandwidth_bytes_per_s: 5.0e10
    link_latency_s:             5.0e-6
    derate:                     0.80
  router_relay_s:               1.5e-3   # Atomesh per-hop, PD only
```

### Why each term exists

Names are chosen to say what the quantity *is*, not where it was first observed. Several
of these feed ATOM variables whose own names are terser; the third column records that
mapping so a reader can follow the term into ATOM's code without guessing.

| Term | Feeds (ATOM name) | Evidence it is needed |
|---|---|---|
| `memory.capacity_bytes` | `get_num_blocks` → `total` | the one reading that cannot be derived |
| `runtime_constants.driver_and_collective_reserve_bytes` | `get_num_blocks` → `non_torch` | 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8. **No fixed-plus-per-peer form fits** 5980 / 6340 / 9138. A table, not a law. Note ATOM's `non_torch` is a **device-wide** reading, so on a shared box it includes neighbours — see D14's dedicated-device scope boundary. |
| `runtime_constants.allocator_retained_after_load_bytes` | `peak_torch` | AITER `CustomAllreduce` 1 GiB pool plus the two-stage kernel's: 1.1 MiB at TP1, **2069 MiB flat** at TP2/4/8 |
| `runtime_constants.persistent_forward_buffer_bytes` | `peak_torch` | `allocate_forward_vars` + attention metadata; ~118 MiB, flat in width, differs only by model |
| `runtime_constants.cudagraph_pool` | `graph_pool.predicts`, the per-term check of the pool's real cost; not `cudagraph_overhead`, which mirrors ATOM's estimator (`03` D16) | measured `91.1 MiB + 0.3033 MiB per captured token` at W=1; flat **104 MiB** above W=1, where the allocated delta was byte-identical (79,692,800) across three widths and three ladders |
| `activations[].bytes_per_token` | `peak_torch`, `cudagraph_overhead` | Qwen3.8-27B TP1 on MI308X: **180,480 B/token** fitted at 16384 warmup tokens predicts the held-out 8192 at **+0.000%**; the declared geometry form gives 98,304 (-45.5%). A model, config or width with no entry keeps the declared form |
| `tokenizers[].*` | admission queue | cc-traces p50 input is **88,768 tokens**; at 2 M tok/s that is ~44 ms, **3x take2's entire admission constant**, and it scales with prompt length while a constant does not |
| `ipc.*`, `interconnect.*_latency_s` | **Clock Authority lookahead floors** (doc 01 D3) | a zero lookahead serializes the whole simulation |
| `compute.fp4_flops` | tier-0 roofline (`10` D64) for an FP4 leaf | DeepSeek-V4-Pro's routed experts are FP4 (`16` D95), and MI455X's dense FP4 peak is twice its FP8 peak ([#504](https://github.com/jgong5/ATOM/issues/504#issuecomment-6080974531)); priced at the FP8 rate they would read twice as slow |
| `intra_node.domain_devices` | the link a collective group rides (`10` D66) | MI455X's UALoE fabric joins 72 GPUs across compute trays of 4 (#504), so a TP8 or EP8 group spans two trays without touching the NICs; `count_per_node` alone would send it over `inter_node` |
| every `derate` | cost model | no kernel reaches spec peak; the gap between datasheet and achievable is the user's to declare |

**On `derate`.** The name was kept deliberately after considering `achievable_fraction`
and `efficiency`. It is a fraction in `(0, 1]` multiplying a spec-peak number, it appears
in five places with identical meaning, and it is the term the schema rules make
*mandatory* so that nobody quietly uses a datasheet FLOP as an achievable one. A name
that reads like a neutral measurement would undercut that; `derate` reads like what it is
— a declared haircut the author is responsible for.

**Precisions and links.** Each `*_flops` is the dense matrix-core peak at one operand
precision, never a structured-sparsity figure: `fp8_flops` covers OCP FP8 and MXFP8,
`fp4_flops` is OCP MXFP4. `fp4_flops` is the one optional rate. A device with no FP4
matrix-core path (gfx942) omits it, and a law asked to price an FP4 leaf on that spec
refuses by name rather than borrowing another precision's rate. `intra_node` is the
scale-up fabric: it carries TP collectives, the DP group's gather and MORI's all-to-all,
and `topology` says how it joins its `domain_devices`, `fully_connected` (a link per pair)
or `switched` (MI455X's UALoE). `inter_node` is the scale-out network: it carries
Mooncake's KV transfer between the prefill and the decode node (`01` D6).

### Schema rules

1. **No defaults for `runtime_constants`.** A missing TP width **refuses and names it**.
   These are the terms with no law behind them; a silent default is the worst kind of
   wrong.
2. **`derate` is mandatory wherever a spec-peak number appears**, so nobody can quietly
   use a datasheet FLOP as an achievable one.
3. **`software_pinned_to` is checked, not decorative.** The plausible reading of the
   evidence is that these constants track the ROCm/RCCL/AITER build more than the die —
   the +5980 MiB at width > 1 is collective buffer sizing, the 926 MiB at TP1 is HIP
   context plus libraries. A stack mismatch warns loudly by default and refuses under
   `validate(strict=True)`; constants moved across stacks, by a transfer or a merge,
   always refuse.
4. **The whole resolved spec is echoed into every run artifact.** The KV gate is **≤5%**;
   a number whose spec cannot be recovered from the artifact is unattributable.
5. **Every field is *sourced* or *assumed*, and the document says which.** A fragment
   whose `method` is `assumed` assumes every value it supplies, and its `notes` say why
   each value was chosen. A `mixed` fragment, which is what a saved merge is, assumes
   exactly the fields its `provenance.assumed` names; one that does not state that list
   is refused by name, and no other fragment may state it. A fragment with any other
   method assumes nothing: it is the source of what it supplies, and a `datasheet`
   fragment's `notes` cite the document and the date it was read. A field is assumed when
   every fragment that supplied it assumes it. When the merged method is `mixed`, `merge`
   writes those fields into `provenance.assumed`, keyed as `Merge.sources` keys them and
   empty when there are none, so a saved merge re-merged next week keeps its assumed
   fields assumed although the per-field record is not saved. A derate for a device never
   run is assumed (`10` D69): a vendor's measured fraction of peak informs the choice and
   does not source it. `08` D50.2 says how the projection reports its assumed fields.

### `host.tokenizer` is keyed by tokenizer identity, not by model

The problem was real: as first written, two models with different tokenizers compared on
one spec would silently share one set of terms.

The obvious fix is to key by model type. Rejected, for one reason: **models share
tokenizers, routinely.** Every size in a family ships the same tokenizer, and
distillations and fine-tunes inherit it. Keying by model means measuring the same
tokenizer once per model, storing N copies of one number, and letting them drift — and
drift here is invisible, because nothing ever compares the copies.

So the key is the **tokenizer**, with model names as an index into it — the
`host.tokenizers` list in the schema above. Four fields carry the identity:

| Field | Why it is in the key |
|---|---|
| `id` | author-chosen, stable, human-readable; what `merge` conflicts on |
| `backend` | `fast` (Rust) and `slow` (Python) run an order of magnitude apart on the same `tokenizer.json`, and which one loads depends on the environment, not the model |
| `fingerprint` | sha256 of the `tokenizer.json` actually measured — the check that the numbers belong to this tokenizer |
| `applies_to` | the model architectures that resolve to it; one entry, many models |

Resolution at run time: match the model architecture against `applies_to`; if the loaded
tokenizer's `tokenizer.json` hash does not equal `fingerprint`, **warn and name both**.
No match at all is a **refusal**, not a default — an unmeasured tokenizer on an 88,768-token
p50 prompt is a ~44 ms per-request error that would land entirely inside TTFT.

**What this does not solve:** one spec still describes one *host*. Two tokenizers
measured on different CPUs must not be merged into one spec — that is what
`provenance` and `merge`'s conflict check are for.

### Open issues

- **Runtime-constant transfer across dies is untested** (**T50**). The working assumption
  — transfers within a software generation, not across one — is now recorded in doc `03`
  and enforced by `software_pinned_to`, but it has not been measured on a second card
  type. One engine startup settles it.
- Three topologies of one model is interpolation, not a law. The cudagraph-pool width
  scaling rests on **one** point above W=1: the shape "flat above W=1" is well supported
  (byte-identical 79,692,800 deltas across three widths and three capture ladders), but
  the claim that the transition happens *at* W=2 rather than being a slowly-varying
  function that merely looks flat over 2/4/8 rests on the single W=1→W=2 step. A fourth
  width would not help; a second **model** would.

---

## D26. Probe tools

### Problem

The schema asks users for numbers most of them cannot obtain by reading a datasheet — the
runtime constants in particular are measurable only by starting an engine. Shipping a
schema without the tools to fill it is shipping a blank form.

### Design

A probe **emits a fragment**: a partial spec plus a provenance stanza saying what it
measured, how, and on what. `merge` combines fragments into a spec, recording which
fragment supplied each field. `validate` refuses an incomplete or inconsistent one.

```
compass spec probe <what>  -> fragment.yaml
compass spec merge f1 f2 ... -> machine.yaml
compass spec validate machine.yaml
compass spec explain machine.yaml --term kv_blocks
```

### The probes, by what hardware they need

**Tier 0 — no GPU. Runs anywhere, including the CPU container.**

| Probe | Fills | Method |
|---|---|---|
| `tokenizer --model M` | `host.tokenizers[].*` | encode and decode a length sweep with **ATOM's own loaded tokenizer** (`_load_tokenizer`, `llm_engine.py`), fit `a + b*n`. Must sweep past 200k tokens: the workload's p90 input is 204,288. |
| `ipc` | `host.ipc.*` | round-trip over ATOM's own `make_zmq_socket` and `aiter.dist.shm_broadcast.MessageQueue`, so it measures the transports ATOM actually uses |

**Tier 1 — one GPU of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `device-memory --tp 1` | `capacity_bytes`, `driver_and_collective_reserve_bytes[1]`, `persistent_forward_buffer_bytes`, `cudagraph_pool.w1_*` | start an engine, read the five readings. Sweep >=5 capture ladders for the graph-pool fit; the prior fit predicted a held-out sixth at **+6.4%** |
| `device-compute` | `bandwidth_bytes_per_s`, `*_flops`, their `derate`s | GEMM and bandwidth sweeps; derate is achieved/datasheet |

**Tier 2 — N GPUs of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `device-runtime-constants --tp 2,4,8` | `driver_and_collective_reserve_bytes`, `allocator_retained_after_load_bytes`, `cudagraph_pool.w_gt1_flat_bytes` | one engine start per width |
| `interconnect-intra` | `intra_node.*` | collectives in **real groups**, several sizes |

**Tier 3 — two nodes.**

| Probe | Fills |
|---|---|
| `interconnect-inter` | `inter_node.*`, `router_relay_s` |

**Tier 4 — no hardware at all.**

| Probe | Fills | Provenance stamped |
|---|---|---|
| `from-datasheet` | `capacity_bytes`, `bandwidth`, `flops` | `method: datasheet`, derates left blank so `validate` refuses until declared |
| `transfer --from <spec>` | `runtime_constants.*` | `method: transferred-from:<spec>`, carrying **both** specs' `software_pinned_to` so a later mismatch is visible |

### Two refusals every hardware probe must implement

Both come from failures that cost real time:

1. **`non_torch` is a device-wide reading.** It is `(total - free) - reserved`, and
   `total - free` counts every process on the card. Six prior runs died at start-up with
   `available_for_kv = -103667.58 MB (budget=57.60GB, peak_torch=2.94GB,
   non_torch=152.01GB, ...)` because a neighbour held 152 GB while the rank had reserved
   2.9 GB. **A probe must refuse a reading whose `non_torch` far exceeds what the
   collective terms predict for its width** — the case worth catching is 50x, not 50%.
2. **Rank disagreement is a direct, single-run measurement of contamination.** Ranks of a
   symmetric group do identical work, so a spread in `non_torch` across ranks needs no
   model to interpret: 0 MiB at widths 1-2, **192 MiB at width 4, 640 MiB at width 8**.
   Take the **minimum** across ranks and report the spread; refuse above a threshold.

Also carried over: a probe must record `free` and `total` **separately** and refuse a
reading where `free` was binding in `min(budget, free)` — otherwise the captured budget is
a property of which neighbours happened to be on the box, and the admission cliff moves
with them.

### `validate` refuses on

- a missing required field
- a TP width absent from `runtime_constants` that the deployment will use
- a `derate` missing beside a spec-peak number
- `software_pinned_to` not matching the running stack (warn, or refuse under a strict flag)
- a `transfer` fragment whose source spec pinned a different stack

### `explain`

Given a spec and a predicted quantity, print which spec fields contributed and by how
much. This is not a nicety: the KV gate is 5%, memory validation is **per term and never
as a sum**, and a prior summed check read **+13.8%** while hiding three errors two of
which cancelled — the largest being 25% of a single term.

### Open issues

- The tokenizer probe measures *this* host. **Working assumption: tokenizer throughput
  transfers across CPUs of comparable class, adjusted by the declared `derate`.** Adopted
  rather than left open, because the alternative is refusing every spec authored on a
  machine other than the target, which is most of them. It is weaker than it sounds —
  tokenization is single-threaded, cache-resident, integer work, so it tracks per-core
  clock far more than core count, and the schema records `cpu.cores_physical` so a gross
  mismatch is visible. Untested; recorded as **T53**, settled by one probe on a second
  host class.
- Tier 2 and 3 probes need a quiet machine. Three of the last five prior pilot attempts
  were lost or degraded by other tenants — two refused to start with a negative KV budget
  at 141 GB of neighbour, one ran 64% slow. **Any probe run needs the machine checked
  before and after, not only before.**
- Nothing yet decides what a user does when they have no card of the target type and no
  spec to transfer from. Today the answer is "declare it and accept the error", which is
  honest but unguided.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D24 | The spec describes the machine; ATOM's config describes the deployment. Thread-pool width becomes an ATOM config option, not a Compass one. | 2026-09-18 |
| D25 | Four-section schema: `provenance`, `host`, `device`, `interconnect`. Mandatory derates, no defaults for `runtime_constants`, `software_pinned_to` checked, whole spec echoed into every artifact. Revised for the MI455 projection: an optional dense `fp4_flops`, `intra_node.domain_devices` for a scale-up fabric wider than a node, and every field *sourced* or *assumed* by the fragments that supplied it, which a `mixed` spec records in `provenance.assumed`. | 2026-09-18; revised 2026-10-10 |
| D26 | Probes emit fragments; `merge` / `validate` / `explain` combine and check them. Four hardware tiers plus datasheet and transfer paths. Every hardware probe implements the contamination refusals. | 2026-09-18 |
| D25.1 | Runtime-constant fields renamed to say what they are; ATOM's own term recorded as a mapping column. `host.tokenizer` becomes `host.tokenizers[]`, keyed by tokenizer identity (id + backend + fingerprint) with `applies_to` model architectures. | 2026-09-19 |
