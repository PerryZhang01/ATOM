# ATOM Compass — Design Topic 13: The Configuration Surface

**Status:** reviewed and approved, 2026-09-20. Drafted by an AI assistant during a design
interview and reviewed by jgong5 across two review rounds on PR #3. No code has been
written against it yet; implementation follows the execution plan in `16`.

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

---

## D80. The flag surface

### Engine-side: what a simulated serve adds to an ordinary ATOM command

```
python -m atom.entrypoints.openai_server \
    --model <model> -tp 4 --max-num-seqs 256 ...    # ordinary ATOM, unchanged
    --runner-qualname atom.compass.runner.CompassModelRunner \
    --compass-spec        machine.yaml \
    --compass-artifacts   ./compass-store \
    [--compass-clock-endpoint tcp://host:port] \
    [--compass-tier        b] \
    [--compass-no-lazy-trace] \
    [--compass-on-refusal  mark|abort]
```

| Flag | Meaning | Default |
|---|---|---|
| `--runner-qualname` | **ATOM's own, unchanged.** The seam. Two in-tree users already (`config.py::Config.runner_qualname`). | ATOM's runner |
| `--compass-spec` | path to the machine specification (`05` D25) | **none — refuses** |
| `--compass-artifacts` | artifact store root (`07` D41) | **none — refuses** |
| `--compass-clock-endpoint` | where the Clock Authority listens. Absent means co-hosted in the API-server process (`01` D3.3). | co-hosted |
| `--compass-tier` | `0`, `a` or `b`. Which cost model is asked; provenance records what each answer actually was. | `b` |
| `--compass-no-lazy-trace` | refuse an unknown structure rather than tracing it in-run (`02`) | lazy tracing on |
| `--compass-on-refusal` | `mark` (record `provenance=refused`, continue) or `abort` | `mark` (`08` D50.1); under step fit `abort`, and `mark` refuses (`17` D102) |
| `--measure` | additionally run on a real GPU and record. **Never valid in an acceptance run** (`07` D42). | off |

Eight flags, one of which is ATOM's. That is the whole engine-side surface.

### Tool-side: the `compass` CLI

One executable, subcommands that mirror `07`'s phases so the plan's output is literally
runnable:

```
compass plan       --model M --spec S --workload W [--width 1,2,4]
compass discover   --model M --workload W                        [CPU]
compass trace      --model M --from-discovery                    [CPU]
compass measure    ops|collectives|steps|memory  --model M --tp W  [GPU]
compass validate   --model M --spec S
compass explain    --spec S --quantity <name>
compass probe      tokenizer|device-memory|device-runtime-constants   [GPU]
compass spec       merge|validate|explain                        [CPU]
```

`plan` is the entry point and the others are steps it names (`07` D37). `probe` and `spec`
are `05` D26's tools, reached through the same executable so a user learns one command.

### What is deliberately NOT a flag

Audited, because an absent flag is a decision and should be a visible one:

| Not a flag | Why |
|---|---|
| a "simulate / real" mode switch | the runner has no modes (`02` D11). A run with `--runner-qualname` pointed at Compass **is** a simulation. |
| output length, `ignore_eos` | already in the OpenAI request schema; duplicating them creates state that can disagree (`06` D28) |
| the simulated arrival timeline | per request, on the wire, not per run (`06` D28) |
| virtual-clock enable/disable | there is no simulated run without one. A switch would create an untested second configuration. |
| per-term memory overrides | they belong in the machine spec, where they are echoed and fingerprinted |
| acceptance rate for speculative decoding | **ATOM's own flag** — `--spec-decode-acceptance-length` / `--spec-decode-acceptance-rate` already exist (`14` D83) |
| thread-pool widths | ATOM's, per `05` D24 |

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

- Whether `--compass-tier` should accept a per-leaf override for debugging. Useful, and a
  way to produce a number nobody can reproduce. Currently no.
- The `ATOM_COMPASS_*` environment names are not enumerated here; they are mechanical
  twins of the flags and should be generated rather than hand-written.

---

## Decision log

| # | Decision | Date |
|---|---|---|
| D78 | Three homes. A setting is Compass's only if it would be meaningless in a real ATOM run — which keeps the surface small by construction. | 2026-09-19 |
| D79 | Precedence is CLI > env > artifact > **refuse**. No silent default for anything that changes a predicted number. The resolved configuration is echoed into every artifact. | 2026-09-19 |
| D80 | Seven Compass flags plus ATOM's `--runner-qualname`; one `compass` executable whose subcommands mirror the calibration phases. Non-flags audited and recorded. `17` D102 makes `abort` the default under step fit. | 2026-09-19; extended 2026-10-10 (`17`) |
| D81 | The resolved configuration is part of every artifact, by value and never by reference. | 2026-09-19 |

---

## TODO register

This topic's items only. The consolidated register across all topics, with the
load-bearing assumptions and their check plans, is [`12_open_items.md`](12_open_items.md).

| # | Item | Why deferred |
|---|---|---|
| T57 | Generate the `ATOM_COMPASS_*` environment twins from the flag table rather than hand-writing them | mechanical; needs the flag table to be final first |
| T58 | Decide whether a per-leaf tier override is worth the reproducibility cost | no demand yet |
| T96 | Reconcile D79 and D80 with the run file the code selects a backend by (`--compass-run`, `atom/compass/run/__init__.py::runner`): D79 rejects a Compass config file, the code reads one (`17` D103) | needs an owner choice between describing the run file here and moving selection to flags |
