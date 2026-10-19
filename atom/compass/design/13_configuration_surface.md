# ATOM Compass — Design Topic 13: The Configuration Surface

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

**Revised 2026-10-13 by the owner's rulings on
[#851](https://github.com/jgong5/ATOM/issues/851#issuecomment-6140384791):** D80's
`compass` commands are the toolchain's, each with its inputs and outputs, and the commands
this topic first listed are deprecated; a simulated serve is configured by `compass run`'s
flags, and the run file is the tool's internal transport (D79, settling T96).

**Revised 2026-10-14 by the owner's
[ruling on #937](https://github.com/jgong5/ATOM/pull/937#issuecomment-6161726104), which
[#851](https://github.com/jgong5/ATOM/issues/851)'s decisions carry:** users reach Compass
only through the `compass` commands, and the environment level goes (D79); one `record` or
`run` serves one instance to any number of client runs, and no command starts the client;
`compass measure memory` takes the memory readings a run reads (D78, D80).

**Revised 2026-10-14 by the naming record in
[#952](https://github.com/jgong5/ATOM/issues/952):** the backend flags follow the run-file
keys, `--step-stub`, `--step-linear` and `--step-grid`, and precedence has levels (D79,
D80).

**Revised 2026-10-18 by the owner's ruling on
[#1489](https://github.com/jgong5/ATOM/issues/1489):** `trace ops` captures one
`op_graph` per shape: each decode width and prefill (T, B) it is given, and each step of a
record directory from `record` or `trace steps --out` (#1539); `measure ops` plans its
points from a `shape_population`,
measures every rank of the layout and writes an `op_measurement` per point; `fit ops`
publishes a `price_list` only (D80).

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
| device capacity, bandwidth, derates, tokenizer throughput | yes, but read from the device today | **machine spec** — because the requirement is that they are *configured*, not probed |
| the runtime and library memory readings (`03` D14) | yes, read from the device | **not a setting**: a `memory_readings` entry per configuration, which `compass measure memory` takes on the target (`07` D38) |
| where the Clock Authority listens | **no** | **Compass flag** |
| which artifact store to read | **no** | **Compass flag** |
| whether to additionally run on a real GPU and record | **no** | **Compass flag** |

**The consequence worth stating:** Compass's flag surface is *small by construction*. If it
is growing, something has been mis-filed, and the usual mis-filing is an ATOM deployment
property that nobody wanted to add to ATOM.

---

## D79. Precedence, and the refusal that replaces a default

### Problem

Two sources can supply a value — the command line and an artifact — and a silent
precedence order is how a run ends up describing a machine nobody chose. The prior
effort's standing lesson is that a plausible default is worse than an error.

### Decision

```
  explicit CLI flag   >   artifact/spec value   >   REFUSE
```

with three rules on top:

1. **No Compass setting has a silent default if it changes a predicted number.** The
   machine spec and the model are mandatory, and so is the artifact store whenever a key
   names an entry in it; omitting one names it and stops. Settings that cannot change a
   number (log level, output path) may default.
2. **There is no environment level** (2026-10-14, #851). Users reach Compass only through
   the `compass` commands' flags (D80), a container or CI harness included. A variable
   the tool sets to carry what it resolved, as the run file does, is not a setting.
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
compass run --machine machine.json --step-grid <key> --store ./compass-store \
    [--host-laws <key>] [--coverage <input>...] [--on-refusal mark|abort] --out <dir> \
    -- --model <model> -tp 4 --max-num-seqs 256 ...      # ordinary ATOM, unchanged

# which starts one instance's server as
python -m atom.entrypoints.openai_server <ATOM arguments> \
    --compass-run <run file> [--compass-clock-endpoint tcp://host:port]
```

| `run` flag | Meaning | Default |
|---|---|---|
| `--machine` | the machine specification (`05` D25) | **none — refuses** |
| `--step-linear`, `--step-stub`, `--step-grid` | the backend, exactly one kind, each named by its run-file key: a step linear law by store key, a step stub's stated coefficients by file (its terms declared, never fitted), a step grid bundle by store key (`17` D103). There is no granularity or method flag. `--step-linear` repeats, one law per engine config: the run file then holds a list, and each process installs the law fitted under its own engine config and refuses when none or several were (the [#973 ruling](https://github.com/jgong5/ATOM/issues/973#issuecomment-6210598012)) | **none — refuses**, as do two kinds |
| `--store` | the artifact store keys resolve in, the configuration's `memory_readings` included (`07` D41) | **none — refuses** |
| `--host-laws` | the host laws by store key (`09` D54), composed by `01` D4's recurrence. Repeats, one entry per engine config, each process installing its own config's, as `--step-linear` does (the [#973 ruling](https://github.com/jgong5/ATOM/issues/973#issuecomment-6210598012)) | none: the run charges no host time and records that |
| `--on-refusal` | `mark`: a refused step costs 0 s, its reason is recorded, and the run continues. `abort`: stop at the first refusal; the run summary is still written. Either way a refusal makes the run a coverage report (`08` D50.1) | `mark` (`08` D50.1) |
| `--step-roofline` | a switch, named by its run-file key `step_roofline`, with no value: the step roofline has no host bound (the owner's ruling on [#1047](https://github.com/jgong5/ATOM/issues/1047)). On, it composes `StepFallbackBackend`: the step roofline (`10` D64) prices the device time of the steps the backend refuses, with the method its inputs give it. A step grid run composes it without the switch (`17` D102) | off; `run.json` records it when on |
| `--coverage` | `shape_population` keys and `record` directories the backend's coverage check reads before the run starts; the step grid's check refuses no step, and counts those it extrapolates or declines to the roofline (`17` D99) | off; `run.json` records the check's result, or that none ran |

`run` reads the `memory_readings` entry for its configuration (`07` D41) and refuses when
there is none, naming the configuration. `compass measure memory` takes it; for a device
never measured, developer tooling states it, marked assumed, and `run.json` records that
(`07` D38).

`--compass-clock-endpoint` names a standalone Clock Authority (`01` D3.3); absent, it is
co-hosted in the API-server process. The tool sets it and `--compass-run`, and refuses
both among ATOM's arguments. How a client started apart is pointed at its run, the run
file's clock endpoint included, is settled by the launch script's own issue (below). The
run file selects the simulated runner through `Config.runner_qualname`, the seam (`02`
D10).

Two settings of the op-level backend are settled with the toolchain's ops half (`16`
Stage 3d): refusing an uncaptured shape rather than tracing it in-run (`02`), and
`--measure`, which additionally runs on a real GPU and records, and is **never valid in an
acceptance run** (`07` D42).

### Tool-side: the `compass` CLI

One entry with subcommands: the `compass` console script, whose module is
`atom/compass/cli.py`, registered in `pyproject.toml`. A verb with two halves takes `steps`
or `ops`: the steps half is the step-level backends', built for stage 2 (`16` S2.7); the ops
half is the op-level backend's, built by stage 3d (`16` Stage 3d). Each command runs one
of `07` D38's phases or fits what a phase measured.

**Users reach Compass only through these commands.** What developers use to check its
quality stays outside the CLI: grading (`python -m atom.compass.validate`, `08`), the
per-term memory check (`python -m atom.compass.memory.check`, `03` D16), and the machine
specs, which developers maintain, one per supported machine (`scripts/compass/machines/`),
with the `spec` package and per-machine scripts (`05` D26). A user passes one by
`--machine`.

```
compass record                                --out D -- <ATOM args>          [GPU]
compass trace steps    <backend> --machine M  --out D --store S -- <ATOM args>
compass trace ops      --model M [--tp N] [--width W]... [--prefill T B]... \
                       [--population D]... --store S
compass measure steps  [<batches>...] [--grid <population>... --grid-definition F] \
                       <sample> --sampling P --out D --store S -- <ATOM args> [GPU]
compass measure ops    [--width W]... [--prefill T B]... [--context C] --llc-mib M \
                       --gpu G --out D --store S -- <ATOM args>               [GPU]
compass measure memory [--beside CMD] --machine M --out D --store S -- <ATOM args> [GPU]
compass fit steps      --backend step-linear [--form FORM] [--seed SEED] \
                       RECORD... --client-run RECORD EXPORT... --machine M \
                       --out D --store S
compass fit steps      --backend step-grid --grid <population>... --grid-definition F \
                       [--sessions A:B] [--seed SEED] <targets>... \
                       --client-run INPUT EXPORT... --machine M --out D --store S
compass fit host       RECORD... --client-run RECORD EXPORT... --machine M \
                       --out D --store S
compass fit ops                   (ops half)
compass run            <backend> --machine M  --out D --store S -- <ATOM args>

<backend>   one of --step-linear KEY, --step-stub FILE, --step-grid KEY; --step-linear
            repeats, one KEY per engine config, as --host-laws does (#973 ruling)
<batches>   record directories and shape_population keys, whose batches it replays
--grid      shape_population keys, and record directories, that the step grid's planner
            places its points over, with the nodes and margin --grid-definition states
<targets>   record directories, and the keys of step_measurement entries of replayed
            batches
<sample>    --sample N --seed S, or --sample all
```

| Command | Phase (`07` D38) | Device | Reads | Writes |
|---|---|---|---|---|
| `record` | 1c's real runs | GPU | ATOM's arguments; the requests clients send | `--out`: the step record with its cost lines; `run.json` with the engine's bounds as ATOM resolved them; logs |
| `trace steps` | 0 | none | `run`'s inputs | `--store`: one `shape_population`, keyed by the backend that priced it; `--out`: what `run` writes |
| `trace ops` | 1a | none | the model at `--tp`; each decode width `--width` names and prefill of T tokens over B sequences `--prefill` names; with `--population`, every step of a record directory from `record` or `trace steps --out`, refusing a `shape_population` key; at least one shape | `--store`: one `op_graph` per shape (`07` D38, Phase 1a) |
| `measure steps` | 1c, on the bench | GPU | its `<batches>`, and `--grid`'s inputs | `--store`: one `step_measurement` per distinct batch |
| `measure ops` | 1b, collectives included | GPU, every rank of the layout | an `op_graph`; with `--plan`, the `shape_population` its planner places each leaf's points over (`07` D37); the stated collective entry points (`07` D40) | `--store`: one `op_measurement` per point, cold, with the kernels each leaf launched and its warm-over-cold ratio (`07` D41) |
| `measure memory` | 2 | GPU | ATOM's arguments; the machine spec | `--store`: one `memory_readings` for the configuration; `--out`: `run.json`, logs, a readable copy |
| `fit steps` | from 1c's measurements | none | the machine spec; `record` directories, each with the exports of the client runs its server served, and `step_measurement` entries, as the backend accepts | `--store`: a `step_linear_law` or a step grid bundle, with its report; `--out`: a readable copy |
| `fit host` | from real runs | none | the machine spec; `record` directories only, each with the exports of the client runs its server served | `--store`: `host_laws`, with its report |
| `fit ops` | from 1b's measurements | none | `measure ops`'s `op_measurement` entries | `--store`: a `price_list`, interpolating only between same-kernel neighbours |
| `run` | a simulated serve | none | a backend, the machine spec, the configuration's `memory_readings`, ATOM's arguments; the requests clients send | `--out`: the step record with its cost lines, the step table, the run summary, `run.json` with the run file and what grading reads (below), logs |

**Shared by every command:**

- **ATOM's arguments after `--`**, verbatim: Compass defines no model flags (D78). A
  command records the configuration ATOM resolved, read back from the engine, never from
  a first parse of the command line (D81).
- **No command starts the client or takes a workload.** A client started apart drives a
  `record`, `trace steps` or `run` server. `measure steps` keeps `--sessions A:B`,
  inclusive and counted from 0, as a filter on the batches it reads.
- **Two output layers.** Raw outputs, records and logs go to `--out` and are evidence, not
  store entries. What another command consumes and must be fingerprinted is published to
  the artifact store at `--store` (`07` D41), and a consumer names it by key, never by
  path.
- **One step-record format**, the parity record's per-DP-rank lines
  (`atom/compass/parity/`): `record`, `trace steps` and `run` write it, and
  `measure steps` reads its batches in it, the planner's points included.
- **`run.json`** in every `--out`: the full command line, the resolved configuration, the
  Compass tree's source digest, the host key, and the device and stack.
- **A GPU command refuses rather than warns** (`07` D42): it names its GPU, with no
  default device; it checks that GPU is quiet before and after, keeping both process
  listings; it records whether it was profiled. `fit` refuses profiled inputs, inputs
  from two machines or two configurations, and inputs from another device, stack or torch
  than its `--machine` spec states: the configuration travels with the data, and the user
  passes none.

**Instances and client runs:**

- **One invocation, one instance.** One `record` or `run` invocation starts one
  instance's server, prefill, decode or aggregated as ATOM's arguments say. `record` is
  `run`'s launch on a real GPU plus recording; `trace steps` is `run`'s launch plus
  publishing. A server outlives any client and serves any number of client runs until it
  is stopped. Several instances are several invocations, started by a script outside the
  CLI that its own issue designs, with how simulated instances share one Clock Authority
  (`01` D3.3) and how a client started apart is pointed at its run.
  A simulated server serves the client runs its run file states, each ending when no
  essential work is left, and the clock waits between them (`01` D3, Client runs).
- **A client run** is one client invocation against a server. Which requests and traces it
  sent comes from the client's export or from the requests (`06`). A step record splits
  per client run by the requests its lines name (`atom/compass/parity/`): a step belongs
  to the client run whose requests it holds, a step holding none to no client run, and a
  step holding two client runs' requests refuses the split. A run summary splits with its
  steps: its step counts (the refusals, the seconds by method,
  `coverage_report`; `01` D3.5) are taken over a client run's steps, while grants, speed
  and detector state stay the server's. Every reader of a record, grading included, takes
  its client runs and traces from this split. A later client run on a server meets the
  prefix cache earlier ones left (`03` D13), so grading pairs each server's first client
  run: every graded run, real or simulated, is a fresh server (`08` D45, D50).

**Per command:**

- **`record`** runs at any TP and DP width, one record per DP rank. `run.json`'s `engine`
  entry holds the engine's bounds: the KV block count, block size and ladder,
  `max_num_seqs`, `max_num_batched_tokens` and `max_model_len`. `record` takes no memory
  readings; `measure memory` does. Steps are timed unprofiled; op times come only from a
  separate profiled pass (`--profile-ops`, ops half), which splits a step and never prices
  one (`07` D42).
- **`trace steps`** takes `run`'s flags, the backend exactly one with no default, and
  publishes the steps its server served as one `shape_population` when it stops. It reads
  shapes only, so any traces may drive it, and the population records the traces it
  served. It needs no GPU device.
- **`measure steps`** replays the batches of its positional inputs, and with `--grid` runs
  the step grid's planner over the shapes of the `shape_population` keys, and the `record`
  directories, it names and measures the points the planner places (`17` D98).
  `--sessions A:B` filters any input; one of `--sample N --seed S` (stratified by step
  kind and rung, or by prefill token band) or `--sample all` draws from the replayed
  batches.
  `--sampling`, the workload's request parameters, is required and joins the key. A
  measurement is keyed by its batch view's digest, so a batch is measured once and a
  stopped invocation resumes; it records the batch's origin, the planner or a record and
  its traces. Its value is `07` D39's bench window.
- **`measure memory`** starts the real engine with ATOM's arguments, reads the memory
  readings once the KV cache is allocated and the graphs are captured, publishes one
  `memory_readings` entry, and exits (`07` D38, Phase 2). `--beside` runs a command the
  start needs, such as the lmcache server an offload connector reaches, between the
  two GPU checks.
- **`fit`**: each backend owns its fit and states the label sources it takes. The step linear
  law and `fit host` take real records only (`09` D54); the step grid takes bench points for
  its grid and real records, bench replays or both for its residual (`17` D98, D100).
  `fit steps --backend step-grid` plans again over the `--grid` inputs and
  `--grid-definition` that `measure steps` planned over, reads each planned point's
  measurement by its batch view, and refuses naming each planned point not measured.
  Its `--sessions A:B` keeps the `--grid` lines `measure steps --sessions A:B` keeps.
  The step stub's coefficients are stated, not fitted, and `--backend step-stub` refuses. A fit copies
  its inputs' provenance into its output, the traces they came from included, and
  identical inputs and arguments give a bit-identical output. A grader refuses a backend
  fitted on the traces it grades (`08` D50).
- **`run`** records in `run.json` what grading reads: whether the step roofline was on,
  whether host segments were skipped (`01` D4's what-if), the refusals, the coverage check's result, and the
  backend's provenance, its fitted traces and profiled inputs included, with the
  assumption of each assumed `memory_readings` member an engine read.
  `python -m atom.compass.validate` decides from them whether a client run is gradeable:
  not when the run skipped host segments, priced with a backend
  fitted on the traces it grades or on profiled inputs, ran the step grid without a
  coverage check, or refused a step of that client run (`08` D50.1). A backend fitted
  under another engine configuration refuses to run: that is a mismatch, not a reason.

**Deprecated, and removable.** The commands this topic first listed are deprecated; none
was built, and an entry asked for one refuses, naming where its work went.

| Deprecated | Where its work went |
|---|---|
| `compass plan` | Nowhere as one command (`07` D37). The step grid's planning and coverage check sit beside its backend (`17` D98, D99): `measure steps --grid` runs the planner over a `shape_population`, and `run --coverage` calls the check before the run starts |
| `compass validate` | Grading stays `python -m atom.compass.validate` (`08`), outside the CLI, and decides whether a run is gradeable from what `run.json` records |
| `compass explain`, `compass probe`, `compass spec` | Machine specs are developers', one per supported machine, maintained with the `spec` package (`05` D26) and per-machine scripts such as `scripts/compass/mi308x_spec.py`; a user passes one by `--machine` |
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
| per-term memory overrides | a memory term comes from the machine spec or a `memory_readings` entry, where it is echoed and fingerprinted |
| a workload, the client's settings | no command starts the client: a client started apart sends them, and grading reads which traces it served from the client run |
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

The **resolved** configuration — post-precedence, with every value tagged by the precedence level that
supplied it — is written into every run artifact and into every calibration artifact's
fingerprint (`07` D41, D43).

Which parts belong in a *key* rather than merely being recorded differs per artifact and is
already tabulated in `07` D41 — a `price_list` does not depend on `max_num_seqs`, a
`memory_readings` entry does. This decision adds only the rule that **nothing is
recorded by reference**: the artifact carries the values, not a path to a file that may
have changed.

### Open issues

- Whether a backend flag should accept a per-leaf override for debugging. Useful, and a
  way to produce a number nobody can reproduce. Currently no.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D78 | Three homes. A setting is Compass's only if it would be meaningless in a real ATOM run — which keeps the surface small by construction. | 2026-09-19 |
| D79 | Precedence is CLI > artifact > **refuse**, with no environment level. No silent default for anything that changes a predicted number. The resolved configuration is echoed into every artifact. The run file `compass run` writes is its transport to ATOM, not a config file. | 2026-09-19; run file 2026-10-13 (#851); no environment level 2026-10-14 (#851) |
| D80 | A simulated serve is configured by `compass run`'s flags, handed to ATOM's server in a run file by `--compass-run`. One `compass` entry, the only way users reach Compass, with the toolchain's commands: `record`, `trace steps\|ops`, `measure steps\|ops\|memory`, `fit steps\|ops\|host` and `run`, sharing ATOM's arguments after `--`, `--out` and `--store`, one step-record format, `run.json` and the GPU refusals; no command starts the client or takes a workload. One `record` or `run` serves one instance to any number of client runs, and a record splits per client run by its requests. `run` records what grading reads, and grading decides whether a run is gradeable. The commands first listed are deprecated and removable, each with where its work went; developers' quality tooling stays outside the CLI. Non-flags audited and recorded. | 2026-09-19; revised 2026-10-13 and 2026-10-14 (#851); backend flags named 2026-10-14 (#952); the step roofline setting made a switch 2026-10-15 (#1047); `trace ops` per shape, `op_measurement`, `fit ops` publishes a `price_list` only 2026-10-18 (#1489) |
| D81 | The resolved configuration is part of every artifact, by value and never by reference. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| ~~T57~~ | ~~Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them~~ — retired 2026-10-14 on #851: there is no environment level (D79) | mechanical; needs the flag table to be final first |
| T58 | Decide whether a per-leaf backend override is worth the reproducibility cost | no demand yet |
| ~~T96~~ | ~~Reconcile D79 and D80 with the run file the code selects a backend by (`--compass-run`, `atom/compass/run/__init__.py::runner`): D79 rejects a Compass config file, the code reads one (`17` D103)~~ — settled 2026-10-13 on #851: `compass run` takes flags only, and the run file is its internal transport (D79, D80) | needs an owner choice between describing the run file here and moving selection to flags |
| ~~T99~~ | ~~A simulated server outlives its client (D80): `01` D3 ends a run when no essential work is left, and must end a client run there instead, the clock then waiting for the next client~~ — settled 2026-10-14 on #966: `01` D3, Client runs | needs a change to `01` D3, not yet designed |
