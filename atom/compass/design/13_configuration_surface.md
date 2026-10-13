# ATOM Compass — Design Topic 13: The Configuration Surface

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Revised 2026-10-13 by the owner's rulings on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** D80's
`compass` commands are the toolchain's, each with its inputs and outputs, and the commands
this topic first listed are deprecated; a simulated serve is configured by `compass run`'s
flags, and the run file is the tool's internal transport (D79, settling T96).

**Depends on:** `05_machine_spec_and_probes.md` D24 (the separation rule this extends),
and every other topic, each of which contributed a flag.

**Scope.** What a user types, and where each setting legitimately lives. The original task
asks which key parameters must be specified (关键技术点 1.5); until now the answer was
scattered across five documents with no owner, no precedence rule and no audit of what is
*deliberately* not a flag.

---

## D78. Three homes, and the rule that assigns them

### Problem

A simulated run is configured by three different things — ATOM's own arguments, a machine
specification, and Compass's own switches — and a setting placed in the wrong one is not
merely untidy. A deployment property in the machine spec silently contradicts the engine.
A machine property on the command line is not echoed into the run artifact, so the number
it produced becomes unattributable against a 5% gate.

### The rule

Extends `05` D24, which settled the first boundary. The full form:

> **ATOM's config describes the deployment. The machine spec describes the machine.
> Compass's own flags describe *this run of the simulator* — and nothing else.**

A setting belongs to Compass only if it would be meaningless in a real ATOM run. That is a
sharp test, and it disqualifies most candidates:

| Setting | Meaningful in a real run? | Home |
|---|---|---|
| `tensor_parallel_size`, `max_num_seqs`, `gpu_memory_utilization`, `block_size`, `num_speculative_tokens` | yes | **ATOM config** |
| `--preprocess-pool-width` | yes — ATOM should own it whether or not Compass exists (`05` D24) | **ATOM config** |
| device capacity, bandwidth, derates, runtime memory constants, tokenizer throughput | yes, but read from the device today | **machine spec** — because the requirement is that they are *configured*, not probed |
| where the Clock Authority listens | **no** | **Compass flag** |
| which artifact store to read | **no** | **Compass flag** |
| whether to additionally run on a real GPU and record | **no** | **Compass flag** |

**The consequence worth stating:** Compass's flag surface is *small by construction*. If it
is growing, something has been mis-filed, and the usual mis-filing is an ATOM deployment
property that nobody wanted to add to ATOM.

---

## D79. Precedence, and the refusal that replaces a default

### Problem

Three sources can supply a value — the command line, the environment, and an artifact —
and a silent precedence order is how a run ends up describing a machine nobody chose. The
prior effort's standing lesson is that a plausible default is worse than an error.

### Decision

```
  explicit CLI flag   >   environment variable   >   artifact/spec value   >   REFUSE
```

with three rules on top:

1. **No Compass setting has a silent default if it changes a predicted number.** The
   machine spec, the artifact store and the model are all mandatory; omitting one names it
   and stops. Settings that cannot change a number (log level, output path) may default.
2. **The environment tier exists for one reason** — container and CI plumbing, where a
   command line is fixed by a harness. Every `ATOM_COMPASS_*` variable has a CLI twin, and
   the resolved value records which tier supplied it.
3. **The fully resolved configuration is echoed into every run artifact**, alongside the
   machine spec (`05` D25 rule 4). A number whose configuration cannot be recovered from
   its artifact is unattributable, and the KV gate is 5%.

### Why not a Compass config file

Considered and rejected. A fourth source multiplies the precedence matrix, and everything
that is genuinely file-shaped is *already* a file — the machine spec and the artifact
store. Compass's remaining settings are few enough to be flags, and if they ever are not,
that is the D78 smell rather than an argument for a config file.

**The run file is not one.** `compass run` resolves its flags, store keys included, into a
run file and hands it to ATOM's API server by `--compass-run` (D80). No user writes it, it
holds only what the flags and the store resolved, and `run.json` keeps it, so it adds no
source to the precedence order.

---

## D80. The flag surface

### Engine-side: what `compass run` hands a server

A simulated serve is configured by `compass run`'s flags, never by a file a user writes.
The tool resolves them, store keys included, into a **run file**, its internal transport:
written from the flags, handed to ATOM's API server by `--compass-run`, and kept in
`run.json` (D79, D81). ATOM's own arguments pass after `--`, unchanged.

```
compass run --machine machine.json --step-fit <key> --store ./compass-store \
    [--host-laws <key>] [--coverage <input>...] [--on-refusal mark|abort] \
    --workload-file traces.jsonl --sessions 0:255 --out <dir> \
    -- --model <model> -tp 4 --max-num-seqs 256 ...      # ordinary ATOM, unchanged

# which starts each server it needs as
python -m atom.entrypoints.openai_server <ATOM arguments> \
    --compass-run <run file> [--compass-clock-endpoint tcp://host:port]
```

| `run` flag | Meaning | Default |
|---|---|---|
| `--machine` | the machine specification (`05` D25) | **none — refuses** |
| `--law`, `--coefficients`, `--step-fit` | the backend, exactly one: tier a's law by store key, declared coefficients by file (declared, never fitted), a step fit bundle by store key (`17` D103). The backend is the cost tier; there is no tier flag | **none — refuses**, as does more than one |
| `--store` | the artifact store a key resolves in (`07` D41) | **none — refuses** when a key is given |
| `--host-laws` | the host laws by store key (`09` D54), composed by `01` D4's recurrence | none: the run charges no host time and records that |
| `--on-refusal` | `mark`: a refused step costs 0 s, its reason is recorded, and the run continues. `abort`: stop at the first refusal; the run summary is still written. Either way a refusal makes the run a coverage report (`08` D50.1) | `mark` (`08` D50.1) |
| `--roofline-floor-s` | a tier-0 floor prices the steps the backend refuses | off; given, the run is not gradeable |
| `--coverage` | `shape_population` keys and `record` directories the backend's coverage check reads before the run starts; a failed check refuses the run (`07` D37) | off; a step fit run without it is not gradeable |

`--compass-clock-endpoint` names a standalone Clock Authority (`01` D3.3); absent, it is
co-hosted in the API-server process. The tool sets it, the ports and `--compass-run`, and
refuses each of them among ATOM's arguments. The run file selects the simulated runner
through `Config.runner_qualname`, the seam (`02` D10).

Two settings of tier b are settled with the toolchain's ops half (`16` W4.1): refusing an
unknown structure rather than tracing it in-run (`02`), and `--measure`, which
additionally runs on a real GPU and records, and is **never valid in an acceptance run**
(`07` D42).

### Tool-side: the `compass` CLI

One entry with subcommands. A verb with two halves takes `steps` or `ops`: the steps half
is tier a's and step fit's, built for stage 2 (`16` S2.7); the ops half is tier b's, built
by the toolchain task (`16` W4.1). Each command runs one of `07` D38's phases or fits what
a phase measured.

```
compass record                     [workload] --out D -- <ATOM args>       [GPU]
compass trace steps    <backend>   [workload] --out D --store S -- <ATOM args>
compass trace ops                  (ops half)
compass measure steps  <input>...  <sample> --sampling P --out D --store S \
                                   -- <ATOM args>                          [GPU]
compass measure ops                (ops half)                              [GPU]
compass fit steps      --backend law|step-fit <sample>... --out D --store S
compass fit host       <record>... --out D --store S
compass fit ops                    (ops half)
compass run            <backend>   --machine M [workload] --out D --store S \
                                   -- <ATOM args>

<backend>   exactly one of --law KEY, --coefficients FILE, --step-fit KEY
[workload]  --workload-file F --sessions A:B and the client's flags
<sample>    --sample N --seed S, or --sample all
```

| Command | Phase (`07` D38) | Device | Reads | Writes |
|---|---|---|---|---|
| `record` | 1c's real runs; 2 | GPU | a workload, ATOM's arguments | `--out`: the step record with its cost lines; `run.json` with the engine ATOM resolved and its startup memory readings; the client's export; logs |
| `trace steps` | 0 | none | a backend, a workload, ATOM's arguments | `--store`: one `shape_population`, keyed by the backend that priced it; `--out`: the step record, `run.json`, logs |
| `trace ops` | 1a | none | the structures Phase 0 found | `--store`: one `op_graph` per structure |
| `measure steps` | 1c, on the bench | GPU | batches as step-record lines: `record` and `trace steps` directories, `shape_population` keys, a backend planner's points | `--store`: one `step_measurement` per distinct batch |
| `measure ops` | 1b, collectives included | GPU, each width | the signatures `trace ops` captured; the declared collective entry points (`07` D40) | per-signature measurements |
| `fit steps` | from 1c's measurements | none | `record` directories and `step_measurement` entries, as the backend accepts | `--store`: a `step_law` or a step fit bundle, with its report; `--out`: a readable copy |
| `fit host` | from real runs | none | `record` directories only | `--store`: `host_laws`, with its report |
| `fit ops` | from 1b's and 1c's measurements | none | `measure ops`'s measurements, and 1c's for the region terms | `price_list`, `region_terms` |
| `run` | a simulated serve | none | a backend, the machine spec, a workload, ATOM's arguments | `--out`: the step record with its cost lines, the step table, the run summary, `run.json` with the run file and whether the run is gradeable, the client's export, logs |

**Shared by every command:**

- **ATOM's arguments after `--`**, verbatim: Compass defines no model flags (D78). A
  command records the configuration ATOM resolved, read back from the engine, never from
  a first parse of the command line (D81).
- **The workload by flags.** `--workload-file` names the traces (never `--traces`);
  `--sessions A:B`, inclusive and counted from 0, chooses the ones served; the client's
  settings are flags too. The resolved workload, the traces' digests and every client
  argument included, goes into `run.json`.
- **Two output layers.** Raw outputs, records and logs go to `--out` and are evidence, not
  store entries. What another command consumes and must be fingerprinted is published to
  the artifact store at `--store` (`07` D41), and a consumer names it by key, never by
  path.
- **One step-record format**, the parity record's per-DP-rank lines
  (`atom/compass/parity/`): `record`, `trace steps` and `run` write it, and
  `measure steps` reads its batches in it, a planner's points included.
- **`run.json`** in every `--out`: the full command line, the resolved configuration, the
  workload, the Compass tree's source digest, the host key, and the device and stack.
- **A GPU command refuses rather than warns** (`07` D42): it names its GPU, with no
  default device; it checks that GPU is quiet before and after, keeping both process
  listings; it records whether it was profiled. `fit` refuses profiled inputs, and inputs
  from two machines or two configurations: the configuration travels with the data, and
  the user passes none.

**Per command:**

- **`record`** runs one node, aggregated, at any TP and DP width, one record per DP rank;
  prefill-decode refuses for now. One run per invocation; repeats are separate
  directories. `run.json`'s `engine` entry holds the KV block count, block size and
  ladder, `max_num_seqs`, `max_num_batched_tokens`, `max_model_len`, and the startup
  memory readings. Steps are timed unprofiled; op times come only from a separate profiled
  pass (`--profile-ops`, ops half), which splits a step and never prices one (`07` D42).
- **`trace steps`** takes the backend as `run` does, exactly one with no default, and
  charges host time as `run` does. It reads shapes only, so any sessions may be traced,
  and it records the range it consumed. It needs no GPU device.
- **`measure steps`** takes `--sessions A:B` as a filter on any input, and one of
  `--sample N --seed S` (stratified by step kind and rung, or by prefill token band) or
  `--sample all`. `--sampling`, the workload's request parameters, is required and joins
  the key. A measurement is keyed by its batch view's digest, so a batch is measured once
  and a stopped invocation resumes; it records the batch's origin, a planner or a record
  and its sessions. Its value is `07` D39's bench window.
- **`fit`**: each backend owns its fit and declares the label sources it takes. Tier a's
  law and `fit host` take real records only (`09` D54); step fit takes bench points for
  its grid and real records, bench replays or both for its residual (`17` D98, D100);
  declared coefficients are not fitted, and `--backend coefficients` refuses. A fit copies
  its inputs' provenance into its output, the traces they came from included, and
  identical inputs and arguments give a bit-identical output. A grader refuses a backend
  fitted on the traces it grades (`08` D50).
- **`run`** writes `gradeable`, with its reasons, to `run.json`. A run is not gradeable if
  it skips host segments (`01` D4's what-if), uses `--roofline-floor-s`, prices with a
  backend fitted on its own sessions or on profiled inputs, runs step fit without
  `--coverage`, or refuses any step (`08` D50.1). Grading stays
  `python -m atom.compass.validate`, which grades only gradeable runs. A backend fitted
  under another engine configuration refuses to run: that is a mismatch, not a reason.
  `run` serves the deployments `record` serves.

**Deprecated, and removable.** The commands this topic first listed are deprecated; none
was built, and an entry asked for one refuses, naming where its work went.

| Deprecated | Where its work went |
|---|---|
| `compass plan` | Nowhere as one command (`07` D37). Step fit's grid planning and coverage check sit beside its backend (`17` D98, D99), and `run --coverage` calls a backend's check before a graded run |
| `compass validate` | Grading stays `python -m atom.compass.validate` (`08`); `run` marks in `run.json` whether a run is gradeable |
| `compass explain`, `compass probe`, `compass spec` | The machine spec stays the `spec` package (`05` D26) and per-machine scripts such as `scripts/compass/mi308x_spec.py` |
| `compass measure memory` | `record` writes the engine's startup memory readings (`07` D38, Phase 2) |
| `compass measure collectives` | `measure ops` (`07` D40) |
| `compass discover` | renamed `trace steps` |
| `compass trace`, alone | renamed `trace ops`; there is no bare `trace`, only `trace steps` and `trace ops` |

### What is deliberately NOT a flag

Audited, because an absent flag is a decision and should be a visible one:

| Not a flag | Why |
|---|---|
| a "simulate / real" mode switch | the runner has no modes (`02` D11). A server handed a run file **is** a simulation; `record` and `run` are two commands, not a mode. |
| output length, `ignore_eos` | already in the OpenAI request schema; duplicating them creates state that can disagree (`06` D28) |
| the simulated arrival timeline | per request, on the wire, not per run (`06` D28) |
| virtual-clock enable/disable | there is no simulated run without one. A switch would create an untested second configuration. |
| per-term memory overrides | they belong in the machine spec, where they are echoed and fingerprinted |
| acceptance rate for speculative decoding | **ATOM's own flag** — `--spec-decode-acceptance-length` / `--spec-decode-acceptance-rate` already exist (`14` D83) |
| thread-pool widths | ATOM's, per `05` D24 |
| the run file's lookaheads, bound and admission path | fixed in code: no measurement backs the lookaheads (`atom/compass/clock/channels.py::single_engine_table`), and `run` always drives the API server |
| the model's parameter count, the DP width | read from the model directory and from ATOM's arguments |

---

## D81. Configuration is part of the artifact key

### Problem

Two runs of "the same" configuration that differ in an engine argument are not comparable,
and the difference is invisible after the fact.

### Decision

The **resolved** configuration — post-precedence, with every value tagged by the tier that
supplied it — is written into every run artifact and into every calibration artifact's
fingerprint (`07` D41, D43).

Which parts belong in a *key* rather than merely being recorded differs per artifact and is
already tabulated in `07` D41 — a `price_list` does not depend on `max_num_seqs`, a
`memory_readings` fragment does. This decision adds only the rule that **nothing is
recorded by reference**: the artifact carries the values, not a path to a file that may
have changed.

### Open issues

- Whether a backend flag should accept a per-leaf override for debugging. Useful, and a
  way to produce a number nobody can reproduce. Currently no.
- The `ATOM_COMPASS_*` environment names are not enumerated here; they are mechanical
  twins of the flags and should be generated rather than hand-written.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D78 | Three homes. A setting is Compass's only if it would be meaningless in a real ATOM run — which keeps the surface small by construction. | 2026-09-19 |
| D79 | Precedence is CLI > env > artifact > **refuse**. No silent default for anything that changes a predicted number. The resolved configuration is echoed into every artifact. The run file `compass run` writes is its transport to ATOM, not a config file. | 2026-09-19; run file 2026-10-13 (#851) |
| D80 | A simulated serve is configured by `compass run`'s flags, handed to ATOM's server in a run file by `--compass-run`. One `compass` entry with the toolchain's commands: `record`, `trace steps\|ops`, `measure steps\|ops`, `fit steps\|ops\|host` and `run`, sharing ATOM's arguments after `--`, the workload flags, `--out` and `--store`, one step-record format, `run.json` and the GPU refusals. The commands first listed are deprecated and removable, each with where its work went. Non-flags audited and recorded. | 2026-09-19; revised 2026-10-13 (#851) |
| D81 | The resolved configuration is part of every artifact, by value and never by reference. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T57 | Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them | mechanical; needs the flag table to be final first |
| T58 | Decide whether a per-leaf tier override is worth the reproducibility cost | no demand yet |
| ~~T96~~ | ~~Reconcile D79 and D80 with the run file the code selects a backend by (`--compass-run`, `atom/compass/run/__init__.py::runner`): D79 rejects a Compass config file, the code reads one (`17` D103)~~ — settled 2026-10-13 on #851: `compass run` takes flags only, and the run file is its internal transport (D79, D80) | needs an owner choice between describing the run file here and moving selection to flags |
