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

**Revised 2026-10-13 by the owner's ruling on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** the probes,
`merge`, `validate` and `explain` are the `spec` package's functions, run by per-machine
scripts, with no `compass` command (D26, `13` D80); `software_pinned_to` states torch, and
a step grid bundle's stack is checked against it (D25).

**Revised 2026-10-14 by the owner's
[ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104), which
[#851](https://github.com/jgong5/ATOM/issues/851)'s decisions carry:** machine specs are
developers', one per supported machine, and a user passes one by `--machine`; the memory
readings of an engine start leave the spec for the `memory_readings` entries
`compass measure memory` takes per configuration (D25, D26, `07` D38).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** a spec states where its values came from,
`provenance.origin`, not how they were produced, and each field's origin, with a probe's
subject, is recorded and kept across a save (D25); probes are grouped by probe class (D26).

**Revised 2026-10-16 by the owner's
[ruling on #1103](https://github.com/jgong5/ATOM/issues/1103#issuecomment-6211106002):** an
optional `host.http_intake` pair, the frontend event loop's charge per HTTP request taken in
(D25), filled by a class-1 `http-intake` probe (D26).

**Revised 2026-10-18 by [#1230](https://github.com/jgong5/ATOM/issues/1230):** the
`host.http_intake` pair is required, now that every committed spec states one probed by
`scripts/compass/http_intake.py`.

**Revised 2026-10-16 by [#1226](https://github.com/jgong5/ATOM/issues/1226):** the schema
drops `device.activations` and the `transferred-from:<spec>` origin with the runtime
constants; the activation coefficient is a `memory_readings` reading (D25).

**Revised 2026-10-17 by the owner's ruling on
[#1392](https://github.com/jgong5/ATOM/issues/1392):** the analytical method reads
attainable peaks probed on the device: one compute rate per format, a bandwidth curve
`BW(bytes)`, an empty-kernel floor and link latency and bandwidth (D25), filled by the
class-1 and class-2 probes (D26). A `derate` is attainable over datasheet, probed; a device
with no hardware carries a probed device's, assumed.

**Revised 2026-10-18 by the owner's
[ruling on #1482](https://github.com/jgong5/ATOM/pull/1482#issuecomment-6248221485):** a
probed number carries the p10 and p90 its probe measured, beside its subject in
`provenance.origins`, and a spec that states one without them is refused by name before it
is written (D25 rule 5, [#1518](https://github.com/jgong5/ATOM/issues/1518)).

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

provenance:                      # the document's origin, and each field's
  authored_by: <who>
  date: 2026-09-18
  origin: datasheet | probed | assumed | mixed
  subject: {model, width, source_root, record}   # a probed document only, rule 5
  origins: {...}                 # each field's origin, subject and band, written by `merge`, rule 5
  fragments: [...]               # filled by `merge`, see D26
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
  http_intake:                 # the event loop's time to take in one HTTP request
    fixed_s:                   2.7e-3
    per_byte_s:                5.6e-9   # per body byte

device:
  name: MI355X
  arch: gfx950
  count_per_node: 8
  memory:
    capacity_bytes:            288.0e9
    bandwidth_bytes_per_s:     8.0e12
    bandwidth_curve:                     # optional: bytes -> attainable bytes/s, STREAM copy, monotone
      - [4096,        1.1e11]
      - [4294967296,  6.4e12]
    derate:                    0.80
  compute:
    bf16_flops:                2.5e15
    fp8_flops:                 5.0e15
    fp4_flops:                 10.0e15  # optional: absent where there is no FP4 matrix core
    derate:                    # attainable over datasheet, one per format
      bf16:                    0.70
      fp8:                     0.65
      fp4:                     0.60
  kernel_floor_s:              2.0e-6   # optional: an empty kernel, replayed in a graph
  # no memory readings, activations included: they are `compass measure memory`'s
  # entries (07 D38, Phase 2)
  software_pinned_to:          # the stack this spec's numbers were taken on
    rocm:  "7.2.4"
    aiter: "<commit>"
    rccl:  "<version>"
    torch: "<version>"

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
    # link_setup_s: optional, once on a rank pair's first KV write; absent is zero
    derate:                     0.80
  router_relay_s:               1.5e-3   # Atomesh per-hop, PD only
```

### Why each term exists

Names are chosen to say what the quantity *is*, not where it was first observed. Several
of these feed ATOM variables whose own names are terser; the third column records that
mapping so a reader can follow the term into ATOM's code without guessing. The
`memory_readings` rows are fields of `07` D41's entry, not of the spec.

| Term | Feeds (ATOM name) | Evidence it is needed |
|---|---|---|
| `memory.capacity_bytes` | `get_num_blocks` → `total` | the one reading that cannot be derived |
| `memory_readings.driver_and_collective_reserve_bytes` | `get_num_blocks` → `non_torch` | 926 / 6906 / 7266 / 10704 MiB at widths 1/2/4/8. **No fixed-plus-per-peer form fits** 5980 / 6340 / 9138. A table, not a law. Note ATOM's `non_torch` is a **device-wide** reading, so on a shared box it includes neighbours — see D14's dedicated-device scope boundary. |
| `memory_readings.allocator_retained_after_load_bytes` | `peak_torch` | AITER `CustomAllreduce` 1 GiB pool plus the two-stage kernel's: 1.1 MiB at TP1, **2069 MiB flat** at TP2/4/8 |
| `memory_readings.persistent_forward_buffer_bytes` | `peak_torch` | `allocate_forward_vars` + attention metadata; ~118 MiB, flat in width, differs only by model |
| `memory_readings.cudagraph_pool_bytes` | `graph_pool.predicts`, the per-term check of the pool's real cost; not `cudagraph_overhead`, which mirrors ATOM's estimator (`03` D16) | measured `91.1 MiB + 0.3033 MiB per captured token` at W=1; flat **104 MiB** above W=1, where the allocated delta was byte-identical (79,692,800) across three widths and three ladders; read at each configuration's own ladder |
| `memory_readings.activation_bytes` over `warmup_tokens` | `peak_torch`, `cudagraph_overhead` | Qwen3.8-27B TP1 on MI308X: **180,480 B/token** fitted at 16384 warmup tokens predicts the held-out 8192 at **+0.000%**; the geometry form gave 98,304 (-45.5%) and is not derivable within 25% ([#705](https://github.com/jgong5/ATOM/issues/705)) |
| `tokenizers[].*` | admission queue | cc-traces p50 input is **88,768 tokens**; at 2 M tok/s that is ~44 ms, **3x take2's entire admission constant**, and it scales with prompt length while a constant does not |
| `http_intake.*` | the frontend event loop's charge per request taken in (`06` D33) | at c=256 first turn the real loop takes a burst in one request at a time over 744 ms; taken in at one instant, step 1 admits 225 against the real 109 ([#1103](https://github.com/jgong5/ATOM/issues/1103)) |
| `ipc.*`, `interconnect.*_latency_s` | **Clock Authority lookahead floors** (doc 01 D3) | a zero lookahead serializes the whole simulation |
| `compute.fp4_flops` | the roofline (`10` D64) for an FP4 leaf | DeepSeek-V4-Pro's routed experts are FP4 (`16` D95), and MI455X's dense FP4 peak is twice its FP8 peak ([#504](https://github.com/jgong5/ATOM/issues/504#issuecomment-6080974531)); priced at the FP8 rate they would read twice as slow |
| `intra_node.domain_devices` | the link a collective group rides (`10` D66) | MI455X's UALoE fabric joins 72 GPUs across compute trays of 4 (#504), so a TP8 or EP8 group spans two trays without touching the NICs; `count_per_node` alone would send it over `inter_node` |
| `memory.bandwidth_curve` | `BW(M)` in the analytical price (`10` D64) | the attainable bandwidth depends on the bytes moved; one rate would price a small transfer at the large-transfer rate (#1392) |
| `kernel_floor_s` | `t0` in the analytical price (`10` D64) | the device's floor on one kernel, below which no pricing unit runs (#1392); the host's launch time is a host law, not this (`10` D64) |
| `*.link_setup_s` | a rank pair's first KV write (`01` D6) | on node 18 Mooncake TCP a pair's first write of 2 to 7 blocks took 9.0 to 28.1 ms longer than its later writes ([#1172](https://github.com/jgong5/ATOM/issues/1172)); absent is zero |
| every `derate` | the analytical price (`10` D64) | no kernel reaches spec peak; the attainable fraction is probed on the device (D26), compute per format, and assumed only where carried from a probed device (`10` D69) |

**On `derate`.** The name was kept deliberately after considering `achievable_fraction`
and `efficiency`. It is a fraction in `(0, 1]` multiplying a spec-peak number, and it is
the term the schema rules make *mandatory* so that nobody quietly uses a datasheet FLOP as
an attainable one. On a device it is attainable over datasheet, probed where the device was
run (D26). It is not ATOM's implementation efficiency, which is never an input (`10` D68);
`efficiency` would read as that, and `derate` reads like what it is.

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

1. **No default for a memory reading.** A configuration with no `memory_readings` entry
   **refuses and names it** (`07` D38). These are the terms with no law behind them; a
   silent default is the worst kind of wrong.
2. **`derate` is mandatory wherever a spec-peak number appears**, so nobody can quietly
   use a datasheet FLOP as an achievable one. `memory.derate` is derived, not probed: the
   bandwidth curve's largest point over `bandwidth_bytes_per_s`.
3. **`software_pinned_to` is checked, not decorative.** Device numbers track the
   ROCm/RCCL/AITER build as well as the die (`07` D43). `compass measure memory` refuses a
   start under a stack the spec does not pin, and constants moved across stacks by a
   merge always refuse. It is the target's stack: a step grid bundle's
   stack, torch included, is checked against it, never against the stack of the process
   that simulates (`17` D101).
4. **The whole resolved spec is echoed into every run artifact.** The KV gate is **≤5%**;
   a number whose spec cannot be recovered from the artifact is unattributable.
5. **Every field has an origin, and the document records it** (`spec/fields.py::ORIGINS`,
   `MachineSpec.origin`): `datasheet`, `probed` or `assumed`.
   A probed field also records its subject, what the probe ran on: model, width, source
   root, and the digest of the run record it read. A probed number, a quantity or a byte
   count, also records its band, `p10` and `p90`: the spread of the value over the
   probe's records, each fitted alone, or over its repeated sweeps; a tokenizer entry
   records one per rate. Readers use the value, never the band. A spec that states a
   probed number without its band is refused by name before it is written
   (`spec/bands.py::refuse_unbanded`). A count or a text is read, not sampled, and so is
   a field the table marks `exact`, the device's capacity: those carry no band. A
   fragment states one
   `provenance.origin` for what it supplies, or `mixed` with each field's origin in
   `provenance.origins` (the tokenizer probe's: the caller's origin on its entries,
   `probed` on the core counts it reads). An
   `assumed` fragment's `notes` say why each
   value was chosen, and a `datasheet` fragment's cite the document and the date it was
   read. `merge` writes each field's origin into `provenance.origins`: a field two
   fragments agree on takes the stronger origin, so it is assumed only when every supplier
   assumed it, and the document's own origin is `mixed` unless every field shares one. A
   saved merge keeps `provenance.origins`, so re-merging it next week keeps each field's
   origin; a `mixed` document without them is refused by name, and so is a document
   that states a method in place of an origin. The origin is not the method: the term that reads a field
   decides that (`02` D11, #952), so a datasheet value used as a parameter is analytical,
   a probed one used as the answer is measured, and an assumed one makes the term
   declared. A derate for a device never run is assumed: MI455's are the ratios probed on
   MI355X (`10` D69, #1392). `08` D50.2 says
   how the projection reports its assumed fields.

### `host.tokenizer` is keyed by tokenizer identity, not by model

The problem was real: as first written, two models with different tokenizers compared on
one spec would silently share one set of terms.

The obvious fix is to key by model type. Rejected, for one reason: **models share
tokenizers, routinely.** Every size in a family ships the same tokenizer, and
distillations and fine-tunes inherit it. Keying by model means probing the same
tokenizer once per model, storing N copies of one number, and letting them drift — and
drift here is invisible, because nothing ever compares the copies.

So the key is the **tokenizer**, with model names as an index into it — the
`host.tokenizers` list in the schema above. Four fields carry the identity:

| Field | Why it is in the key |
|---|---|
| `id` | author-chosen, stable, human-readable; what `merge` conflicts on |
| `backend` | `fast` (Rust) and `slow` (Python) run an order of magnitude apart on the same `tokenizer.json`, and which one loads depends on the environment, not the model |
| `fingerprint` | sha256 of the `tokenizer.json` actually probed — the check that the numbers belong to this tokenizer |
| `applies_to` | the model architectures that resolve to it; one entry, many models |

Resolution at run time: match the model architecture against `applies_to`; if the loaded
tokenizer's `tokenizer.json` hash does not equal `fingerprint`, **warn and name both**.
No match at all is a **refusal**, not a default — an unprobed tokenizer on an 88,768-token
p50 prompt is a ~44 ms per-request error that would land entirely inside TTFT.

**What this does not solve:** one spec still describes one *host*. Two tokenizers
probed on different CPUs must not be merged into one spec — that is what
`provenance` and `merge`'s conflict check are for.

### Open issues

- **Memory readings carried across dies are untested** (**T50**). An entry assumed for a
  device never run copies another device's of the same configuration, and every term
  built on it is declared (`07` D38). The working assumption — the readings transfer
  within a software generation, not across one — is recorded in doc `03` and has not
  been measured on a second card type. One engine startup settles it.

---

## D26. Probe tools

### Problem

The schema asks for numbers a datasheet does not give — achievable rates, tokenizer
throughput, link latencies. Shipping a schema without the tools to fill it is shipping a
blank form.

### Design

A probe **emits a fragment**: a partial spec plus a provenance stanza saying what it
probed, how, and on what. `merge` combines fragments into a spec, recording which
fragment supplied each field. `validate` refuses an incomplete or inconsistent one.

```
a probe                  -> a fragment
merge(fragments)         -> the machine spec, recording which fragment supplied each field
validate(spec or merge)  -> the refusals below
explain(spec, term)      -> each field under a field or block path, with its value and origin
```

Each is a function of the `spec` package, and a per-machine script runs them:
`scripts/compass/mi308x_spec.py` sweeps the tokenizer, merges its fragment with the
device's and the assumed ones, validates the result and prints `explain`'s rows. No
`compass` command wraps them (`13` D80). Machine specs are developers', one per supported
machine (`scripts/compass/machines/`), and a user passes one by `--machine`. The readings
of an engine start are not the spec's: `compass measure memory` takes them per
configuration into a `memory_readings` entry (`07` D38, Phase 2).

### The probes, by what hardware they need

**Class 0 — no GPU. Runs anywhere, including the CPU container.**

| Probe | Fills | Method |
|---|---|---|
| `tokenizer`, per model | `host.tokenizers[].*` | encode and decode a length sweep with **ATOM's own loaded tokenizer** (`_load_tokenizer`, `llm_engine.py`), fit `a + b*n`. Must sweep past 200k tokens: the workload's p90 input is 204,288. |
| `ipc` | `host.ipc.*` | round-trip over ATOM's own `make_zmq_socket` and `aiter.dist.shm_broadcast.MessageQueue`, so it measures the transports ATOM actually uses |

**Class 1 — one GPU of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `device-memory`, at TP1 | `capacity_bytes` | the device's capacity; the readings of an engine start are `measure memory`'s |
| `device-compute` | `compute.derate`, one per format | per format (BF16, FP8, FP4 where the device has it), the best large-GEMM rate the vendor library reaches; derate is attainable over the datasheet `*_flops` (#1392) |
| `device-bandwidth` | `memory.bandwidth_curve` | a STREAM copy sweep over transfer sizes; `memory.derate` is derived from it (schema rule 2) |
| `kernel-floor` | `kernel_floor_s` | an empty kernel replayed in a graph, so the host's launch time is not in it |
| `http-intake` | `host.http_intake.*` | `scripts/compass/http_intake.py`: one real c=256 first-turn burst with the API server's loop stamped. Each gap between consecutive chat-template starts is the next request's intake; `per_byte` is the Theil-Sen slope of the gaps on that request's body bytes, `fixed` keeps the fitted gaps summing to the burst's span. A few gaps of tens of milliseconds hold the loop's other work whatever the size: a least-squares fit of the gaps follows them, and one of each start to running sums of both terms swings sign from burst to burst; the median slope holds steady across bursts and hosts ([#1230](https://github.com/jgong5/ATOM/issues/1230)). Needs a real server, so it runs on a card of the target |

**Class 2 — N GPUs of the target type.**

| Probe | Fills | Notes |
|---|---|---|
| `interconnect-intra` | `intra_node.*` | link latency as half a small-message round trip (one hop), link bandwidth and its `derate` from a large peer copy (#1392); collectives in **real groups**, several sizes, for the empirical pricing of `07` D40 |

**Class 3 — two nodes.**

| Probe | Fills |
|---|---|
| `interconnect-inter` | `inter_node.*`, `router_relay_s` |

**Class 4 — no hardware at all.**

| Probe | Fills | Provenance stamped |
|---|---|---|
| `from-datasheet` | `capacity_bytes`, `bandwidth`, `flops` | `origin: datasheet`, derates left blank so `validate` refuses until stated |

### Two refusals every hardware probe must implement

Both come from failures that cost real time, and `measure memory` holds them too (`07`
D38):

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
- a `derate` missing beside a spec-peak number

### `explain`

Given a spec and the dotted path of a field or a block of fields, print each field under it
with its value and origin, one row per field and never a sum. This is not a nicety: the KV
gate is 5%, memory validation is **per term and never as a sum**, and a prior summed check
read **+13.8%** while hiding three errors two of which cancelled — the largest being 25% of
a single term.

### Open issues

- The tokenizer probe measures *this* host. **Working assumption: tokenizer throughput
  transfers across CPUs of comparable class, adjusted by the stated `derate`.** Adopted
  rather than left open, because the alternative is refusing every spec authored on a
  machine other than the target, which is most of them. It is weaker than it sounds —
  tokenization is single-threaded, cache-resident, integer work, so it tracks per-core
  clock far more than core count, and the schema records `cpu.cores_physical` so a gross
  mismatch is visible. Untested; recorded as **T53**, settled by one probe on a second
  host class.
- Class 2 and 3 probes need a quiet machine. Three of the last five prior pilot attempts
  were lost or degraded by other tenants — two refused to start with a negative KV budget
  at 141 GB of neighbour, one ran 64% slow. **Any probe run needs the machine checked
  before and after, not only before.**
- With no card of the target type, the peaks are its datasheet times the ratios probed on
  a card that was run, assumed, as MI455 takes MI355X's (`10` D69, #1392). For the memory readings nothing yet
  decides it: the answer is "assume it and accept the error", an entry marked assumed
  (`07` D38), which is honest but unguided.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D24 | The spec describes the machine; ATOM's config describes the deployment. Thread-pool width becomes an ATOM config option, not a Compass one. | 2026-09-18 |
| D25 | Four-section schema: `provenance`, `host`, `device`, `interconnect`. Mandatory derates, no memory readings (`measure memory`'s entries, 2026-10-14), activations included (2026-10-16), `software_pinned_to` checked, torch included, as the target's stack, whole spec echoed into every artifact. Revised for the MI455 projection: an optional dense `fp4_flops`, `intra_node.domain_devices` for a scale-up fabric wider than a node, and every field given an origin, `datasheet`, `probed` or `assumed`, `provenance.origin` and per field `provenance.origins` with a probe's subject, kept across a save. Revised: a `host.http_intake` pair, the event loop's charge per HTTP request taken in, required once every committed spec stated it. An optional per-link `link_setup_s`, charged once on a rank pair's first KV write. Revised for the attainable-peak roofline: a `derate` is attainable over datasheet, probed, one per compute format; an optional `memory.bandwidth_curve` and an optional `kernel_floor_s`; a device never run carries a probed device's derates, assumed. A probed number carries its measured p10 and p90 beside its subject, and a spec stating one without them is refused by name before it is written. | 2026-09-18; revised 2026-10-10; torch 2026-10-13 (#851); origin 2026-10-14 (#952); intake 2026-10-16 (#1103), required 2026-10-18 (#1230); link setup 2026-10-16 (#1172); activations and the transfer origin dropped 2026-10-16 (#1226); attainable peaks 2026-10-17 (#1392); `host.admission_fixed_s` dropped 2026-10-17 (#1000); bands 2026-10-18 (#1482, #1518) |
| D26 | Probes emit fragments; `merge` / `validate` / `explain` combine and check them. Four hardware probe classes plus a datasheet path; the transfer path dropped 2026-10-14 (#937). Every hardware probe implements the contamination refusals. All are the `spec` package's functions, run by per-machine scripts, with no `compass` command. Machine specs are developers', one per supported machine, passed by `--machine`; an engine start's memory readings are `measure memory`'s, not the spec's. Revised: class 1 probes the per-format GEMM rate, the bandwidth curve and the kernel floor; class 2 the link latency and bandwidth. | 2026-09-18; revised 2026-10-13 and 2026-10-14 (#851); 2026-10-17 (#1392) |
| D25.1 | Runtime-constant fields renamed to say what they are; ATOM's own term recorded as a mapping column. `host.tokenizer` becomes `host.tokenizers[]`, keyed by tokenizer identity (id + backend + fingerprint) with `applies_to` model architectures. | 2026-09-19 |
