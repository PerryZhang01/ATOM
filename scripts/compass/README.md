<!-- SPDX-License-Identifier: MIT -->
# Compass gate and pre-flight scripts

The scripts every task runs around its work: two test gates, a GPU pre-flight, the
snapshot builder a gate runs against, and the two list regenerators. Everything
here runs **inside a container**, which mounts the worktree parent so every task's
tree is reachable, and each script derives its tree from **its own path**, not from
your `$PWD`, so the copy in a worktree exercises that worktree. Run **your**
tree's copy: running another checkout's copy is refused rather than silently acted
on, because until 2026-09-20 it was not — see "Which tree a script acts on" below.

| Script | Where | What |
|---|---|---|
| `gate_cpu.sh` | CPU container | The CPU gate, run per task: every test file outside `tests/plugin/` and `cpu_gate_exclude.txt`, no driver **as a batch**: run alone, a file in it can still reach `rocminfo` (`tests/test_postprocess_width.py` did, through `atom.model_engine.model_runner`'s `from aiter import …`, measured on node 18 at `bdd244c57` while it was in the gate). **Must be green.** Also exits 98 when the diff is in the blind spot below and the GPU gate has not run on *this* tree. Refuses a caller-supplied `-r`. |
| `gate_gpu.sh` | GPU container | The GPU gate, run per wave — and per task for a blind-spot diff. Superset (`--ignore=tests/plugin`), judged as a **delta** against pure upstream `661ef7661` on the nightly 20261013 image (the `BASE_*` constants), with torch, HIP, ROCm and AITER recorded and compared. The failing node-ids are on file in `gpu_gate_known_failures.txt` and compared **by name**. Calls `preflight.sh` itself, before and after. |
| `preflight.sh` | GPU container | `rocminfo` reachability, compute use and VRAM use (checks 1–3), plus a D-state census (check 0), and a list of bookable devices. Run **before and after**. The census states its own scope: in a container it can only see this PID namespace, and says so rather than reporting a clear node. Also worth running for a CPU-only task that imports ATOM's model layer: `import aiter` runs `rocminfo`, and so does any module that imports aiter at load time (`atom.model_engine.model_runner`, `atom.model_ops.linear`), so on a wedged node that import hangs. A bare `import atom` and `atom.compass` run no `rocminfo` and load no torch; `atom.config`, `atom.model_engine.llm_engine` and `atom.entrypoints.openai_server` load torch but run no `rocminfo`. Measured on node 18, 2026-09-23. |
| `regen_cpu_gate_exclude.sh` | CPU container | Regenerate the **GENERATED** section of the exclusion list by iterating `--collect-only` to a fixed point (three passes at `83daf636d`: 37 errors, 1, clean). Preserves the MANUAL section verbatim. |
| `regen_gpu_gate_triggers.sh` | CPU container | Re-derive `gpu_gate_triggers.txt` from the exclusion list and the tree. Run after **any** change to `cpu_gate_exclude.txt` — the two files are derived from the same tree and are wrong separately. Writes the whole file, **header counts included**; nothing in it is maintained by hand. |
| `snapshot.sh` | either | Build the `git archive` tarball a gate runs against (never `rsync`). Stamps `.compass-commit` (so output names its tree, and a `COMPASS_GPU_GATE_DONE` attestation can be checked) and `.compass-changed` (so the blind-spot question is answerable without `.git`). Refuses a dirty tree. |
| `pd_sim.sh` | GPU container | A simulated 1P1D run: the standalone clock authority (`python -m atom.compass.run`), a prefill and a decode API server on the simulated KV connector, and `atomesh` in PD mode between them. `tests/compass/test_pd_slice.py` drives one request through it. Its variables are in its header. |
| `parity_one_node.sh` | GPU container, two devices (eight for `CONFIG=dsv4-dpa8`) | The scheduling-parity test: one workload (`python -m atom.compass.parity.traffic`) on a real `-tp 2 --enable-dp-attention` server (`CONFIG=tp2-plain`: `-tp 2` alone; `CONFIG=dsv4-dpa8`: DeepSeek-V4-Pro's nightly `-tp 8 --enable-dp-attention --enable-tbo` server) with the step record on, then on a simulated one whose steps each end when the real one's did (`atom/compass/parity/replay.py`; `step_stub` prices their device time into the cost record only), and `python -m atom.compass.parity` over the two records. With `STORE` unset it first runs `compass measure memory` on the real run's GPUs, flags and environment into `OUT_DIR/store`, under `LMCACHE=1` with an lmcache server of its own beside it, and the simulated run reads that store. The last lines are the real run's admission (send to response headers: minimum, median, maximum) and, per DP rank, the steps compared, the steps identical and the first divergence. Its variables are in its header. |
| `python -m atom.compass.detect.scenario` | either | The acceptance checks every simulated deployment's runs are held to, over the out_dirs of one run file run under different `PYTHONHASHSEED`: each run reached its finish with no refusal, the step tables are byte-identical, and each channel's messages were released to their receiver once each and none behind its clock. With `--record LP=DIR`, an engine LP's TAR rows must equal the timeline a constant step stub implies, computed from the parity record in `DIR` (`pd_sim.sh` writes one per side when `ATOM_COMPASS_PARITY_RECORD` is set). Prints each failure and exits 1. The vertical, DP, PD and PD-DP slices in `tests/compass/` hold their runs to it, with the traffic's outputs too. |
| `mi308x_spec.py` | GPU container, node 18 or hjbog-srdc-39; any container with `atom` under `--sweep-from` | `python3 scripts/compass/mi308x_spec.py`: rebuild `machines/mi308x-hjbog18.json`, the node-18 MI308X machine spec, by sweeping Qwen3.8-27B's tokenizer on this host and merging it with the device, capacity and stack of one real engine start and the HTTP intake terms `http_intake.py` fitted over one real burst (`INTAKE`), each recorded in the script; with `COMPASS_MACHINE=mi308x-hjbog39`, the same for hjbog-srdc-39 into `machines/mi308x-hjbog39.json`, with the PP carriers `pp_carriers.py` fitted there (`PP`, which names the fit's commit and its record's sha256), and on node 18 the attainable peaks `probe_device.py` read (`PEAKS`, which names the probe's commit and its record's sha256). `--sweep-from DOC` takes the sweep a written document carries instead. Prints `validate`'s verdict and `explain`'s fields grouped by origin; on either host a value no probe fills reads `assumed`. |
| `pp_carriers.py` | GPU container, two devices of one node | `HIP_VISIBLE_DEVICES=a,b torchrun --nproc-per-node 2 scripts/compass/pp_carriers.py OUT`: send one message per size between the two devices over RCCL and gloo, the receive posted first and late, and fit each carrier's eager threshold, overhead, latency, per-byte time and rendezvous fixed part, and how late a gloo sender blocked before the receive is posted goes on (`wake_s`, `wake_rise_s`). Writes OUT, a JSON record of the fit, its error spread, every sample and the script's text. `python3 scripts/compass/pp_carriers.py OUT RECORD...` fits several such records' samples pooled and names each record's sha256 in place of the samples: one probe's wake moves from run to run. `mi308x_spec.py`'s `PP` states a fit and cites its record. Check the two devices are idle before and after. |
| `probe_device.py` | GPU container, two idle devices of one node | `HIP_VISIBLE_DEVICES=a,b python3 scripts/compass/probe_device.py SPEC OUT`: time the best BF16 and FP8 GEMM rate over a grid of large shapes, a copy sweep from 4 KiB to 4 GiB moved, an empty kernel replayed in a graph, a peer copy and a peer ping-pong between the two devices, and print the fragment they make on the machine file SPEC: each format's derate, the bandwidth curve and `memory.derate`, `kernel_floor_s`, the intra-node link's latency and derate, the banded ones with their p10 and p90. Refuses a device that holds VRAM or work, or whose `ptl_enable` is not SPEC's `device.ptl`. Writes OUT, the record of every sample and the script's text; a spec script's `PEAKS` states the readings and cites its sha256. Check the two devices are idle after. |
| `http_intake.py` | GPU container, one real server | `HTTP_INTAKE=STAMPS REAL=1 CONCURRENCY=256 SESSIONS=256 AIPERF_EXTRA_ARGS="--trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0" scripts/compass/cctraces_sim.sh RUN_FILE OUT`, aiperf stopped once the first turns are in, then `python3 scripts/compass/http_intake.py fit STAMPS`: the real server runs under the probe, which stamps each request body its event loop reads and each chat template it renders, and the fit prints `host.http_intake` from the burst; a spec script's `INTAKE` states one burst's fit, cites its STAMPS's sha256, and states each term's p10 and p90 over the host's bursts, each fitted alone. Check the GPUs are idle before and after. |
| `mi355x_spec.py` | any container with `atom`; mi355-gpu-45 to re-sweep | `python3 scripts/compass/mi355x_spec.py --sweep-from scripts/compass/machines/mi355x-gpu45.json`: rebuild `machines/mi355x-gpu45.json`, the mi355-gpu-45 MI355X machine spec, from the tokenizer sweep that document carries and the device, capacity, HTTP intake terms (`INTAKE`), datasheet peaks and assumed values recorded in the script; byte for byte while this script and `mi308x_spec.py` are unchanged. Without `--sweep-from` it re-sweeps DeepSeek-V4-Pro's tokenizer at `/models/DeepSeek-V4-Pro` on mi355-gpu-45. Prints `validate`'s verdict and `explain`'s fields grouped by origin. |
| `cpu_gate_exclude.txt` | — | Excluded test files in two marked sections: **GENERATED** (`# BEGIN GENERATED`, driver-dependent at *collection* time; never hand-edit — `regen_cpu_gate_exclude.sh` rewrites it wholesale) and **MANUAL** (`# BEGIN MANUAL`, collects cleanly then fails on a driver call or for want of a `.git`, so the regenerator cannot see it). The MANUAL section **is** hand-edited; every entry must carry its observed failure above it. |
| `gpu_gate_triggers.txt` | — | The source paths that no *running* CPU-gate test names. Generated, not hand-written; matched by `gate_cpu.sh`. A trailing `/` matches a subtree. |
| `gpu_gate_known_failures.txt` | — | The known-failing GPU node-ids at `gate_gpu.sh`'s `BASE_COMMIT`, verbatim. Its line count and `BASE_FAILED` are two statements of one fact; `gate_gpu.sh` refuses to run if they disagree. |
| `_lib.sh` | — | Tree resolution, `PYTHONPATH`, the `import atom` assertion, commit stamp, and the `tests/compass` pass count the GPU gate derives its allowed surplus from. |

## Gate a tree with its own `scripts/compass/` — four `test_snapshot_ref.py` failures mean you overlaid

Stage a tree with the copy of these scripts **that tree itself carries**, and run
that copy. Do not copy `scripts/compass/` from another branch over it. An older
staging recipe did exactly that — overlaying this directory from
`compass/p0.1-env-and-gates` (`105ca4197`, whose `scripts/compass` is tree
`ddb69e7aa`) so that every tree was gated from one known copy. It predates these
scripts being on the integration branch, and on any tree at or after `186d12829`
it now costs four failures on **every** side of a comparison:

| tree | its own `scripts/compass` | with the `ddb69e7aa` overlay |
|---|---|---|
| `b1dca15da`, tree `00386e887` (the integration head's tree at 2026-09-21T21:00Z) | **4594 passed, 0 failed**, rc=0 | **4590 passed, 4 failed**, `GATE_CPU_RC=1` |
| `83ef2a094` — an earlier head, tree `9091c1dc8` | **4570 passed, 0 failed**, rc=0 | **4566 passed, 4 failed**, `GATE_CPU_RC=1` |
| `cf6429387` — a branch, tree `95cb8358d` | **4501 passed, 0 failed**, rc=0 | **4497 passed, 4 failed**, `GATE_CPU_RC=1` |

149 skipped, 3 xfailed on all six runs, and the total is conserved on every tree
— the four are moved out of passed, not added. Measured node 18, container
`xiaobizh_n18_cpu`, 2026-09-21T20:06-20:09Z and 21:00-21:03Z, staged by
`git archive` + `docker cp`, run sequentially and unpiped, `import atom` asserted
under each root from `/` first.

The four are in `tests/compass/test_snapshot_ref.py`, and they are **correct
failures** — each asserts on *that tree's own* `snapshot.sh` messages, which the
tree has and the overlaid older script does not:

| Failing test | What the overlaid `snapshot.sh` does instead |
|---|---|
| `test_unresolvable_ref_refuses_at_ref_resolution` | prints the one merged `REFUSED: no merge-base with <ref>` for both refusals |
| `test_unrelated_history_refuses_at_merge_base` | same message, so neither test can tell which step refused |
| `test_remote_qualified_ref_resolves_and_names_itself` | has no remote-prefix fallback, so it exits 92 where the tree's own script resolves `fork/feature/atomcompass_new` |
| `test_snapshot_carries_both_stamps` | same: it exits 92 before writing a tarball to inspect |

The fifth test in that file passes either way: a local branch of that name
resolves in both scripts and neither announces a fallback it did not take. **Do
not exclude any of them.** A red gate here is the tests working, and it is the
only signal that says the instrument was swapped.

The overlay costs a second thing on the same path: the older `gate_cpu.sh` it
brings with it prints `Baseline is 4030 passed, 0 failed` when the gate fails, a
figure removed at `186d12829` because a script line cannot name its own commit —
**564** behind the 4594 `b1dca15da` reads with its own scripts, **540** behind
`83ef2a094`'s 4570 and **471** behind `cf6429387`'s 4501. It is stale by a
different amount on every tree, because it was never a statement about the tree it
prints on. So the reader of the false red is handed a stale control as well.

**If two trees being compared carry different copies, say so and name both tree
objects** — `git rev-parse <ref>:scripts/compass`. The gate is then a different
instrument on each side, and the delta is not a measurement of the diff. That
question is live, not settled. Any branch that edits this directory carries its own
tree by construction (`cf6429387` carries `95cb8358d`), and so does the integration
head the moment such a branch lands: it carried `9091c1dc8` at 2026-09-21T20:05Z and
`00386e887` at 20:58Z, when #99 landed. A tree census is therefore a reading, not a
property. Read 2026-09-21T20:58:34Z over `refs/remotes/fork/compass/**` after
`git fetch fork --prune`: **45** `compass/*` branches, **35** carrying this
directory, **six** distinct tree objects between them, **26** of those branches
still on the pre-`186d12829` `ddb69e7aa` the overlay recipe copies from. Read the
first three as readings with their times rather than as a current count — three reads
over the preceding hour gave 41 / 31 / four, 43 / 33 / six and 45 / 35 / seven, and
they move in both directions as branches are pushed and rebased. The `26` is the
exposed population and is the figure that justifies this section. "Every tree is
identical now" is what made the overlay look free.

## Baselines — two gates, two commits, two provenances

Not one baseline. The rows below were measured at different commits by different
tasks, so each row names its own commit and the gate it was measured in.

**Read them as history, not as a current expectation — the CPU total above all.**
It moves whenever `tests/compass/` grows, which is most tasks, so the recorded
figure is stale by construction between one task and the next, and it has twice
been taken as a control by a task that then read the intervening tasks' tests as a
surplus of its own. **A control is measured, not read**: run `gate_cpu.sh` on the
integration head your branch forked from, in the same container, and state both
figures beside their commits. The CPU row below once named no commit, which is
the omission that let it pass for current; the same gate on
`68ef4f329` measured **4380 passed, 149 skipped, 3 xfailed, rc=0** — node 18,
container `xiaobizh_n18_cpu`, 2026-09-22, 36.2 s of pytest inside 43.0 s of wall.

Two runs of the *same* tree can still differ by one: a ±1 in the passed/skipped
split, and — separately — a non-zero `GATE_CPU_RC`, are both outcomes of one
flaky test in ATOM's own suite — see "A red CPU gate that may not be your diff"
below before attributing either to a diff. They are alternatives: the
skip-variant is `rc=0` with the total conserved, the hard failure is `1 failed`
with a non-zero rc and no ±1.

| Gate | Result | Measured |
|---|---|---|
| CPU gate, at `105ca4197` | **4030 passed, 0 failed**, 149 skipped, 3 xfailed, rc=0 — decomposing as **3956 ATOM + 74 `tests/compass`**; the 3956 is PR #6's control, `042aad97d` with PR #6's `scripts/compass/` copied in, since `042aad97d` has none. Six runs at `d737f15e7` and `7ff80cc4b`, before PR #6's restack, read the same counts and **25.6-30.8 s of pytest inside 31.6-36.9 s of wall (`time` real) — a measured spread, not a bound** (`8c0ee374f`) | `105ca4197`, PR #6's head, landed as `4c16792d9` (same tree) — PR #6's gate table; node 18, container `xiaobizh_n18_cpu`, 2026-09-21, against a `git archive` snapshot with `PYTHONPATH` asserted and pytest's own rc captured before any pipe |
| CPU gate, at `186d12829` | **4501 passed, 0 failed**, 149 skipped, 3 xfailed, rc=0 — three runs, identical, 28.8-34.4 s of pytest inside 35-40 s of wall. The **4030** above and the **4380** in the paragraph above are this same gate at earlier trees; all three are history, and this one will be too | `186d12829` — the commit this branch forks from, which is its merge-base with the integration head, **read 2026-09-21T19:21Z**, node 18's own clock — node 18, container `xiaobizh_n18_cpu`, `git archive` snapshot staged by `snapshot.sh`, `PYTHONPATH` asserted, pytest's own rc captured before any pipe |
| GPU superset (`--ignore=tests/plugin`), the baseline until the upstream merge | **4779 passed, 5 failed**, 0 errors, 105 skipped, 3 xfailed, **72.6 s**; two runs, byte-identical failing sets | `fe9ea043c`, node 18, container `xiaobizh_n18`, `HIP_VISIBLE_DEVICES=1`, 2026-09-20, torch **2.10.0+rocm7.2.4.git3d3aa833**, `torch.version.hip` **7.2.53211**, ROCm release **7.2.4**, AITER **v0.1.21.dev0-49-gf4e7c7509** (`git describe`) |
| `ruff check .` | 1003 errors, 640 fixable — the gate is *no new* error, not zero | `83daf636d` |
| `black --check .` | clean, 660 files | `83daf636d` |

**The CPU total moves when `tests/compass/` grows, and one of the things that grows
it is this directory.** `tests/compass/test_cpu_gate_exclude.py` parametrises
`test_every_trigger_path_still_exists` over the entries of `gpu_gate_triggers.txt`,
one case each, so regenerating that file changes the pass count by exactly the
change in the number of entries. The readings in circulation are the same gate
under different exclusion lists and a different `tests/compass`, not discrepancies:
**3925** at `83daf636d`, 32 exclusions and no `tests/compass`; **3956** is the
ATOM-only half, at `042aad97d` above, and 3925 + 31; **4005** at `3afcb4880`, with
`tests/compass` at 49; **4022** at `71d2a1ac2`, with it at 66 — 49 plus the 17 extra
trigger paths the corrected derivation below produces — and **4030** at `105ca4197`,
with it at 74, the 8 cases `test_gate_gpu_surplus.py` adds. The exclusion list went
32 → 29, adding those 31, because `test_dp_metadata.py`, `test_dp_sync_layout.py`
and `test_forward_mode.py` were re-measured **CPU-green**.

**The five GPU failures at `fe9ea043c` were four ULP comparisons and one bitwise check**, and all five are still in `gpu_gate_known_failures.txt` — not
"five bf16 ULP failures", which is what this file and three design documents said
until the review that produced this paragraph:

| Node-id | Character |
|---|---|
| `tests/test_fused_compress_ragged.py::test_kernel_matches_reference_on_ragged_batches[extend0-context0-cut+whole]` | `allclose`, off by one bf16 ULP: `max\|diff\| = 0.001953125`, exactly 2⁻⁹, against `atol=rtol=1e-3` |
| `…[extend1-context1-whole+cut]` | same |
| `…[extend2-context2-resume+fresh]` | same |
| `…[extend4-context4-tiny-then-long]` | same |
| `tests/test_dcp_merge_ops.py::test_row_view_matches_output_slicing_bitwise` | `torch.equal` — **bitwise, no tolerance at all**, so "one ULP" does not describe it and a tolerance bump would not move it |

All five are pre-existing and unrelated to Compass: no importable `atom` module
differs between `fe9ea043c` and the integration base `83daf636d` — the only delta
under `atom/` is markdown documentation, which no test imports. Cite the delta;
do not claim green.

## The compass_harness tests need an agentx-harness wheel

`tests/compass/test_harness_*.py` skip when `compass-harness` or `aiperf` does not
import. On a tree that carries `compass_harness/`, `gate_cpu.sh` installs the tree's
own `compass_harness` into a run-local directory, with `--no-build-isolation`, so
`setuptools` must be installed. `aiperf` comes from the wheel `COMPASS_AGENTX_HARNESS` names, or
from the environment when that is unset. When the tests' skip condition still holds,
the gate exits 96 before pytest and names the reason.

The gate installs with `--no-index`, so it runs offline (inside `unshare -rn` too).
Build the wheel once, where `pip` can fetch `hatchling`, and stage it beside the
snapshot:

```
git -C <agentx-harness> archive 56a0cf70 | tar -x -C <src>
python3 -m pip wheel --no-deps -w <dir> <src>       # aiperf-0.12.0-py3-none-any.whl
COMPASS_AGENTX_HARNESS=<dir>/aiperf-0.12.0-py3-none-any.whl scripts/compass/gate_cpu.sh
```

`compass_harness` checks the installed `aiperf` source against its pinned
functions on import, so a wheel from another commit fails the same way.

## atomesh's cargo tests run offline, from the container's crate cache

`gate_cpu.sh` runs atomesh's unit tests (`cargo test --lib`) and every Compass
cargo target, selected by file name (`atom/mesh/tests/compass_*.rs`, so a new
Compass target must carry the prefix), in `atom/mesh` before pytest, with `CARGO_NET_OFFLINE=true` unless the caller set
it, and fails after pytest if they fail. `tests/compass/test_mesh_clock_client.py`
skips when `cargo metadata` fails, and the gate runs the same probe first: when it
fails and the diff touches `atom/mesh`, the gate exits 96 before pytest and names
cargo's error; otherwise it prints a `mesh:` line and goes on. It skips, by exact name,
tests that fail on a loaded host:
`core::worker::tests::test_load_counter_performance`, a wall-clock throughput floor
(#515), and `policies::tree::tests::test_tree_structure_integrity_after_stress`, a
thread race that records a placeholder tenant (#514).

The tree has no `Cargo.lock` (`atom/mesh/.gitignore`). The gate copies one in from
`COMPASS_CARGO_LOCK`, default `/app/ATOM/atom/mesh/Cargo.lock`, the lockfile the
gate image built atomesh with, unless the tree already has one. The image's cargo
cache holds only what its release build downloaded, so a new gate container needs
the rest fetched once, with the network, outside any netns:

```
docker exec <container> bash -c 'cd /app/ATOM/atom/mesh && cargo fetch --locked'
```

A change to atomesh's `Cargo.toml` that needs a crate the cache lacks makes the
probe fail, and so the gate exit 96, until `cargo fetch` runs in that tree's
`atom/mesh` in the container, with the network.

## A red CPU gate that may not be your diff — one flaky test in ATOM's suite

`tests/entrypoints/test_stream_marker_properties.py::TestTheRegionIsNotCopiedPerChunk`
is non-deterministic. It is ATOM's own test, it is present at **every** control, and
it is not a Compass defect: do not modify it and do not put it in
`cpu_gate_exclude.txt` — excluding it would change what the gate measures on both
sides of the delta. It is written down here because four agents have been warned
about it by hand and one lost a gate run to it.

The class asserts a **timing** property — that a buffered region is not recopied
per chunk — by streaming a 32 KB and a 128 KB payload and comparing cost per KB,
with a linear-loop control arm and a `pytest.skip` noise guard
(`if not 0.6 < control < 1.6`). Wall clock on a shared box is its input, so its
outcome moves with the box's load.

**It is three-way, not two-way**, and the third outcome is the one worth knowing:

| Outcome | What the run looks like | Count | Rate |
|---|---|---|---|
| nominal | passes | 18 | 85.7% |
| skip-variant | passed **−1**, skipped **+1**, total conserved, `rc=0` | 2 | 9.5% |
| **hard failure** | `1 failed`, **non-zero `GATE_CPU_RC`**, `cost per KB grew 1.89x` | 1 | 4.8% |

A skip-variant is recognisable: the pass count moves by one and the skip count
moves back the other way. A hard failure is not — `GATE_CPU_RC` is non-zero, and
**by the number alone that is indistinguishable from a regression**, while the
instinct on a red gate is to look in your own diff. Read the `FAILED ` line the
gate prints first (that is what its `-rf` is for): if it names this class, re-run
rather than hunt.

**Do not pipe the gate.** A pipeline's status is its *last* command's, so a piped
run hands the caller `tail`'s: measured on a forced failure, `gate_cpu.sh 2>&1 |
tail -6` reported **0** while the gate itself exited **1**. This whole section is
addressed to someone reading a non-zero rc, and a piped run does not give them one.
`tail` truncates from the **top**, so it also decides which half survives: `2>&1 |
tail -6` kept 5 of the 9 stderr lines — the test's name among them — and dropped
pytest's `FAILED ` line; `2>/dev/null | tail -6` dropped the paragraph whole and
kept the `FAILED ` line and the counts. Redirect to a file and read `$?`. The gate
does print `GATE_CPU_RC=` on stdout on every path, so the number survives in the
*text* of an untruncated pipe — but only an unpiped run puts it in `$?`.

**What those rates are, and are not.** n=21, node 18, container `xiaobizh_n18_cpu`,
on a box whose load was not controlled. Those conditions, the table and its one test
id, `test_the_cost_per_byte_does_not_grow`, are issue #93's. 19 of the 21 are pinned:
PR #79's review (issue comment 5765186351) ran `b58a48cc2` 19 times, for 17 nominal,
1 skip-variant and the one hard failure. The other 2, 1 nominal and 1 skip-variant
by subtraction from the table, name no commit, so half the skip-variant rate rests
on #93's count alone. #93 attributes the skip-variants to that test, but no
skip-variant run named it (PR #99). Not-nominal combined is ~1 in 7.
The finding is the third outcome, not the rate. The rates are a **lower bound on
the class**, not a measurement of it: the class holds **3 methods / 4 collected
cases** (one is parametrised `buffered-region` and `kimi-incremental`), of which
**2 methods / 3 cases** share the same timing helpers, the same noise guard and the
same `< 1.5` ratio assertion. Only
`test_the_open_region_is_never_scanned_beyond_the_window` does not: it counts scans
of the open region and is deterministic. The sibling
`test_no_format_pays_more_per_byte_as_the_payload_grows[buffered-region]` has been
observed failing the same way — `qwen: cost per KB grew 1.73x from 32 to 128 KB`,
`1 failed, 4495 passed, rc=1`, on the integration head `669dc3f9d`; node 18,
`xiaobizh_n18_cpu`, 2026-09-21, recorded in the round-2 review of PR #67 (issue
comment 5765660762), which hit it on its own gate run. A sequential re-run of that
same tree passed at **4496**, and 1 + 4495 = 4496 accounts for it exactly. Inherited,
not measured here — so the failure belongs to the mechanism and not to the one
method #93 names. Per #93, the two skip-variants both occurred under `gate_cpu.sh`,
and the one hard failure in the counts under direct pytest; comment 5765186351
gives only its mix, 7 gate + 12 direct.

**Run gates sequentially.** Per #93, the hard failure was reproduced when two gate
loops on node 18 overlapped; #93 names neither that run's harness nor whether it
is the one in the counts. Two gates on one box compete for the CPU the control arm is
measuring, which is the condition this test is least able to survive — check for a
running `gate_cpu.sh` before starting one.

**Seen again since, on another tree.** Three `gate_cpu.sh` runs on `354965883`
(node 18, `xiaobizh_n18_cpu`, 2026-09-21T19:05–19:08Z): the first read **4495
passed, 150 skipped**, the next two **4496 passed, 149 skipped** — `rc=0` on all
three and the total conserved at **4645**, so nothing was added or lost and one
test moved from passed to skipped. That is the skip-variant's signature. The gate
prints no skip *reasons* (and refuses a caller's `-r`, which is how you would ask
for them), so that run did not name the test; the counts and the conserved total
are what was observed.

**Gloo timeouts in `tests/compass/test_dp_step_max.py`,
`tests/compass/test_moe_gather_rows.py` and `tests/compass/test_process_group_lp.py`
were DNS, not load.** Their gloo groups
used to resolve the hostname once per rank to pick an address, and node 18's
hostname is not in `/etc/hosts`. One stalled lookup outlasted the group timeout,
and the others failed with `wait timeout after 5000ms`, `Connect timeout` or
`Connection refused` (#726). The fixtures now set `GLOO_SOCKET_IFNAME=lo`, so no
lookup happens; a red in those files is a real failure.

## Which tree a script acts on

Each script resolves its tree from `$BASH_SOURCE`, so it acts on the checkout it
lives in. That is deliberate, and it is not the same question as "the tree the
caller meant".

**EXECUTED 2026-09-20.** From a checkout at `83daf636d` on `feature/atomcompass_new`,
running another worktree's copy —
`bash …/compass-worktrees/p0/scripts/compass/snapshot.sh <outdir>` — wrote
`compass-d78f3bbd3.tar` stamped `commit: d78f3bbd3…` and exited 0, with no warning.
The stamp was truthful about the tree it archived and silent about the tree the
caller was standing in. All five callers of `compass_tree_root` had the same
exposure; for a gate the consequence is worse, since a pytest result would be
attributed to a commit it did not come from, and a `COMPASS_GPU_GATE_DONE`
attestation would then agree with a stamp that describes the wrong branch.

`compass_tree_root` now **refuses** (exit 99) when `$PWD` is inside a *different*
checkout. Resolving from `$PWD` instead was rejected: it would let a script act on
a tree that is not its own. Being invoked by path from outside any checkout is
still allowed — there nothing contradicts the caller, and it is how
`snapshot.sh`'s own extract instructions say to run a gate.

## The CPU gate's blind spot — a file, and a gate

`gpu_gate_triggers.txt` is the blind spot stated as paths a script can match,
generated by `regen_gpu_gate_triggers.sh`. The rule it applies:

> an `atom` module named by an excluded test is a blind spot **unless a CPU-gate
> test that actually runs names it too**.

Four decisions make that sentence operational. Three were forced by a defect or a
measured counter-example; the fourth, collection, changed no path at `236abfd9a`
and is kept as a forward guard:

1. **Indentation.** The parser anchored on `^`, and at `fada7424e` **190** of the
   `import atom.*` lines in non-plugin test files were indented — function-local and
   guarded imports are the dominant idiom here. It was reading about a third of the
   lines it claimed to read. The excluded side now reads any indentation.
2. **Symbols.** `from atom.model_ops import eplb` parsed to the bare package
   `atom.model_ops`, which resolved to the whole subtree and swallowed
   every sibling entry. `from X import y` is now read as `X.y`, resolved to
   `X/y.py`, else `X/y/`, else `X.py`, else `X/__init__.py`. An `atom.*` reference
   that resolves to none of those is a **refusal**, not a warning: either the parser
   misread a line or a test imports something absent, and both make the derived sets
   incomplete by an unknown amount.
3. **Skips.** An indented import is not proof that anything executed it. The
   excluded side reads any indentation, but coverage is credited **only for a
   module-level import**, and that half carries both counter-examples:
   `tests/model_ops/test_moe_dp_token_capacity.py:39` imports `topK` inside a test
   marked `skipif(not torch.cuda.is_available())`, and `tests/test_mla_index_cache.py`
   imports `ModelRunner` at **lines 99–100, indented four spaces inside a test
   function** — not at module level, as this file and three others said until
   2026-09-20. Reading coverage at any indentation drops the set 30 → 29, losing
   `topK.py`; doing that *and* crediting files that collect nothing drops it 30 → 27,
   losing `atom/model_engine/model_runner.py` — the module Compass's runner seam
   replaces — plus `aiter_mla.py` and `topK.py`. Measured at `236abfd9a` in
   `xiaobizh_n18_cpu`.
4. **Collection.** A file that collects nothing covers nothing, so coverage is
   credited only from a CPU-gate file `--collect-only` shows collecting at least one
   test. At `236abfd9a` that probe removed no path: **30 triggers with it, 30
   without, difference empty** (measured in `xiaobizh_n18_cpu`; it
   withheld 15 coverage paths, of which the only candidate,
   `atom/model_ops/v4_kernels/state_writes.py`, was absorbed either way by the
   candidate subtree entry `atom/model_ops/v4_kernels/`). It is a forward guard for
   trees that one does not represent, not the thing that keeps `model_runner.py`, and
   it costs a full `pytest --collect-only` over the CPU gate plus a refusal path
   (exit 97).

So the two sides are **asymmetric on purpose**: candidates come from any import in
an excluded test; coverage is subtracted only for a **module-level** import in a
CPU-gate file that collects at least one test, because those are the only imports
that provably execute (collection imports the module). Making both sides
indentation-tolerant would undo point 3 and re-delete `topK.py`.

**It errs in both directions, so it is not a floor.** Toward *firing*: an indented
import in a CPU-gate test that does run is not credited, so its module can be listed
although the CPU gate reaches it; and coverage subtraction is exact-string, so a
candidate **subtree** entry is never cancelled by coverage of the files beneath it,
nor a candidate file by coverage of its package. Toward *silence*: imports are read as text, not
resolved as a graph, so a module reached only transitively is invisible to both
sides. A path **absent** from the file is not a claim that the CPU gate covers it,
and a path **present** is not proof that it does not. Of the two mistakes, firing
spuriously is the one to prefer: running the GPU gate when in doubt is never the
wrong one, and the asymmetry above is that preference written down.

`gate_cpu.sh` **enforces** that preference rather than restating it. It takes the
changed-file list from `COMPASS_CHANGED_FILES`, else from `git diff` against the
merge-base with `feature/atomcompass_new` (`COMPASS_INTEGRATION_REF`, resolved as
below); with neither it exits **98**, naming the missing stamp, rather than
answering "no". When a changed file matches a trigger,
the run ends 98 unless `COMPASS_GPU_GATE_DONE` discharges it:

| `COMPASS_GPU_GATE_DONE` | Outcome |
|---|---|
| unset | **98** — run `gate_gpu.sh`, then re-run with that tree's HEAD |
| set, but this tree has no commit (no `.git`, no `.compass-commit`) | **98** — an attestation names a tree; accepting an uncheckable one makes the rule bypassable in exactly the snapshot where the gate normally runs. Rebuild with `snapshot.sh`. |
| set to a sha other than this tree's HEAD | **98** — the GPU gate passed on a different tree |
| equal to this tree's HEAD | discharged |

The trigger match is *reported* before pytest and *enforced* after it, so one run
yields both answers instead of trading one for the other. Green at the CPU gate is
not green at the test gate when the diff is in the blind spot.

## Which integration ref resolves, and which step refused

`COMPASS_INTEGRATION_REF` defaults to `feature/atomcompass_new`, which is the
branch's name on the remote and **not** a local branch in a linked worktree or a
fresh clone — there only `fork/feature/atomcompass_new` exists. `_lib.sh`'s
`compass_resolve_ref` therefore tries the bare name first, then the same name
under each configured remote, and reports which one it used: `snapshot.sh` prints
a `ref:` line and names it beside the base, `gate_cpu.sh` names it in the `gpu:`
source. Before that, the default resolved nowhere and `snapshot.sh` exited **92**
in every worktree with nothing set.

A second `ref:` shape appears when the bare name **did** resolve — to a local
branch that has drifted from the remote branch of the same name. Both scripts
print it, in the same position as the one above:

```
ref:    feature/atomcompass_new (1b473e5af) is 2 commit(s) behind fork/feature/atomcompass_new (cae322c86)
        the base is the local branch; COMPASS_INTEGRATION_REF=fork/feature/atomcompass_new uses the remote
```

It reports and does not redirect — the base stays the ref that resolved, in all
three directions (`N commit(s) behind`, `N commit(s) ahead of`, `diverged from
(N ahead, M behind)`), and a local branch level with its remote prints nothing,
which is what makes the line mean something when it appears. Two things to read
it with. `diverged from` is also what **unrelated** histories print, because
`rev-list --left-right --count` returns the whole of each side when there is no
merge-base; it is the merge-base refusal below, not this line, that reports a
pair with no common history at all. And the counterpart is found by **same name** under each remote —
what `compass_resolve_ref` would have picked had the local branch been absent —
so a branch whose configured upstream is a *differently* named remote branch is
either silent or compared against a ref it does not track. That class is the
branches whose `git config --get branch.<name>.merge` names something other
than `refs/heads/<name>`; enumerate it rather than trusting a number, because
`git push -u` moves a branch out of the class the moment it is published — it
creates the same-name ref and rewrites the config in one step — so any count
is stale as soon as anyone pushes. The announcement is not a substitute for
keeping the local integration branch fast-forwarded.

The two ways that can still fail are separate refusals, because one is a
statement about the *ref* and the other about the *history*, and a reader who
confuses them inspects the wrong thing:

| Step | Message | Means |
|---|---|---|
| ref resolution | `REFUSED: ref resolution failed -- <ref> names nothing here…` plus git's own line, verbatim | nothing was compared; set `COMPASS_INTEGRATION_REF` |
| merge-base | `REFUSED: merge-base failed -- <ref> resolved, but shares no commit with HEAD.` | the ref is fine; the histories are unrelated |

Both exit **92**: the exit code is what other scripts key on and did not change,
and the two are told apart by the message. `tests/compass/test_snapshot_ref.py`
covers each path separately so they cannot merge back into one.

A bare `git archive` tree — one extracted without `snapshot.sh`'s stamps — is the
adjacent case, and `gate_cpu.sh` still exits **98** there. The stamps are not
written retroactively: `.compass-changed` is a diff against a base the stamped
tree no longer has any way to compute, so writing one would be a guess in the
shape of a measurement. The refusal names the omission instead.

## What the GPU gate expects, and on a tree that carries no Compass tests

`gate_gpu.sh` judges an **equality**, not a floor: `BASE_PASSED` plus whatever
`tests/compass/` contributes on the tree in front of it, minus what it contributed
at the baseline (`BASE_COMPASS_TESTS`). A floor would be loosened by exactly the
tests each task adds, so a task adding 30 tests while silently losing a 20-test file
would still clear it.

| this tree's `tests/compass/` | expected passes |
|---|---|
| absent | `BASE_PASSED + 0 - BASE_COMPASS_TESTS` — the figure a tree without this phase's tests measures, worked out beside the call in `gate_gpu.sh` |
| present, N passing | `BASE_PASSED + N - BASE_COMPASS_TESTS` |
| present, unreadable | **93** — a surplus with no source is not a measurement |

The absent case is a tree without this phase's tests, as the integration branch
was until `4c16792d9` added the directory. Deriving the surplus with
`pytest tests/compass --collect-only` made that case exit 4 (`file or directory not
found`), left the count empty and refused with `GATE_GPU_RC=93` before running a
test — reproduced on `4da2f3a2d` and on PR #9's branch, so it was a property of the
script and not of any change. An absent directory and an unreadable one are told
apart by asking the filesystem, not by parsing pytest's error text.

Both sides of the arithmetic are **pass** counts. The tree side used to be a
*collected* count, which is the same number only while `tests/compass/` holds no
skip and no xfail; the first Compass test to skip on a GPU host would have made a
green tree read `unaccounted -1` and blamed tests outside `tests/compass/`. Counting
passes costs one pytest run over a CPU-only directory and removes the condition.
`compass_compass_pass_count` in `_lib.sh` holds the derivation, and
`tests/compass/test_gate_gpu_surplus.py` covers each of its branches with a stubbed
`python`, so the absent-directory path is tested without a driver.

## Why the gates refuse a caller-supplied `-r`

Both gates forward `"$@"` to pytest. pytest's `-r` is **store-last-wins**, so a
caller's `-rE` replaces the gate's own `-rf`/`-rfE` and no `FAILED ` lines are
printed at all. `gate_gpu.sh` builds its verdict out of those lines: **EXECUTED on
`d78f3bbd3`, `gate_gpu.sh -rE` compared the five baseline failures against an empty
observed set, reported all five as "no longer failing", and exited
`GATE_GPU_RC=0` while they had in fact failed.** Both gates now refuse the flag
(exit 95) rather than absorb it, and `gate_gpu.sh` treats "baseline failures no
longer failing" as a refusal in its own right — it has three causes (genuinely
fixed, stopped running, could not be parsed) and only the first is a pass.

## Why `import atom` is asserted before every run

A prior run resolved `atom` to a non-git snapshot of a different branch, 72 files
divergent, and failed silently wherever both trees defined the symbol. `_lib.sh`
sets `PYTHONPATH` to the tree and **nothing else** — inherited entries are the
hazard, not a convenience — then asserts the resolved path is under it, and exits
92 if not. The probe runs from `/` on purpose: run in place, Python puts the cwd at
the head of `sys.path` and the assertion passes whatever `PYTHONPATH` says.

## A cc-traces cell on a simulated TP1 run — `cctraces_sim.sh`

`cctraces_sim.sh RUN_FILE OUT_DIR` runs one cell in the GPU container: ATOM's API
server with `--compass-run`, driven by `aiperf profile --scenario
inferencex-agentx-mvp` through compass-harness, on the first `SESSIONS` traces of a
cc-traces `traces.jsonl`. The cell ends when the scenario's `--benchmark-duration`
has passed on the simulated clock; nothing is computed or allocated on the device.
The header of the script lists its environment; `MODEL`, `TRACES` and
`HARNESS_PYTHON` are required.

- `RUN_FILE` is a run file as `atom/compass/run/__init__.py` describes it. The
  script replaces its `out_dir` and `clock_endpoint`, sets `client_runs` to 1, so
  the server leaves at the end of aiperf's run, and adds a `workload` entry:
  the sha256 of `TRACES` and of each replayed trace, the aiperf arguments and the
  pinned benchmark id. The step table's configuration line and the run summary
  carry it.
- Since the served tokenizer is charged, a tokenizer entry of the run file's
  machine spec must name the model's architecture
  (`Qwen3_5ForConditionalGeneration` for a Qwen3.8-27B `config.json`), or the
  server refuses to start.
- `HARNESS_PYTHON` is a Python with agentx-harness 56a0cf70 and compass-harness
  installed; the script puts this tree's `compass_harness` first on `PYTHONPATH`
  and refuses (exit 92) when it resolves elsewhere.
- Each cell writes its aiperf memory-mapped dataset under `OUT_DIR`, so cells run
  side by side.

The last line printed is the cell's result: aiperf's request count and errors,
simulated and wall seconds from the run summary, both prefix-cache hit rates
(`cache_read_pct` is the engine's, `theoretical_hit_pct` is aiperf's oracle), TTFT,
ITL and request throughput in simulated time, and the run's refusals. A
`coverage_report=True` cell is a coverage report, not a result.

A run file with `router_s` and `kv_write_req_s` makes the cell 1P1D: `pd_sim.sh`
deploys it, taking `DECODE_EXEC`, `ATOMESH` and each side's `PREFILL_ARGS` /
`DECODE_ARGS` flags and `PREFILL_ENV` / `DECODE_ENV` variables from the
environment, and aiperf drives the atomesh router. The traffic LP sends to
`frontend-P` and reads the stream from `frontend-D`, as the run file's channel
table names them; with `--enable-dp-attention` in either side's flags the cell
is agentic, its router is `--dp-aware` and LP `router`, and the traffic LP
talks to `router`. The result line adds `kv_transfers` and
`kv_transfer_mean_s`, from the line the prefill engine logs per KV transfer.

`REAL=1` runs the same TP1 cell on a real server, on the GPU and the wall clock,
so a real and a simulated cell at one seed, concurrency and traces hash send the
same requests and warm up on the same ones. The server runs ATOM's own runner
(no `--compass-run`), and aiperf runs as shipped: `HARNESS_PYTHON` must have
agentx-harness without compass-harness (exit 92 otherwise), and the script pins
the benchmark id as compass-harness does. The result line reads aiperf's export
alone and adds `contaminated`, true when `rocm-smi` lists any process on one of the
cell's GPUs before or after it, with each one's pid and VRAM bytes. Both modes
write the TP1 step record to `steps/dp0.jsonl` (`atom/compass/parity`);
`RECORD=0` turns it off. `FIRST_SESSION` skips that many traces, so calibration
and graded cells can replay disjoint ones.

Two runs of a cell with different `PYTHONHASHSEED` must give byte-identical step
tables and step records: `cmp A/step_table.txt B/step_table.txt`.
`tests/compass/test_cctraces_sim_smoke.py` runs that pair on one session for 60
simulated seconds when `ATOM_COMPASS_SLICE_MODEL`, `ATOM_COMPASS_CCTRACES` and
`COMPASS_HARNESS_PYTHON` are set, and with `COMPASS_AIPERF_PYTHON` (no
compass-harness) a real cell on zero dummy weights beside it.

The cell's `OUT_DIR` also holds one hand-in per server LP, `lp-<lp>.json`. The
engine's, `lp-engine.json` (`lp-engine-P.json` and `lp-engine-D.json` for 1P1D),
holds the readings the simulated KV pool was sized from.
Against the server log of the paired real run, with the same deployment flags,
`python -m atom.compass.memory.check OUT_DIR/lp-engine.json REAL_SERVER_LOG`
prints the per-term comparison of the non-KV memory and the graph pool. It takes
the log of one rank.

## The step linear law from real cells — `step_linear_fit`

`python -m atom.compass.backends.step_linear_fit RECORD... --out LAW --machine-spec DIGEST [--form FORM] [--engine FILE]`
fits the law a run file's `step_linear` key holds (`atom/compass/backends/step_linear.py`) from
the step records of real cells, `OUT_DIR/steps/dp0.jsonl` of `REAL=1
cctraces_sim.sh` runs, each with the `dp0.engine.json` the run wrote beside it,
whose engine config the law states and whose `data_parallel_size` its ranks
join at. Records that ran before a run wrote that file, or wrote it without
`data_parallel_size`, take an engine file stated from their `server.log` as
`--engine FILE`. `atom/compass/backends/step_labels.py` states the label
and what is cut, and the command writes each record's labels beside it for
reading, `dp0.cost.jsonl`; the module's docstring states how outliers are
dropped. The written law also carries, per vector, its samples, warm and dropped
counts and condition number, and names any vector it could not fit under
`unfitted`.

`DIGEST` is the `machine_spec_digest` of the capture host's spec in
`scripts/compass/machines/`, which a run's spec must match, printed by
`python -c 'import json, sys; from atom.compass.artifacts import machine_spec_digest as d; from atom.compass.spec import MachineSpec as M; print(d(M.from_mapping(json.load(open(sys.argv[1])))))' SPEC.json`.

Calibration cells replay traces the graded cells do not: `FIRST_SESSION=256
SESSIONS=137` against graded traces 0 to 255, by the
[owner's ruling on #627](https://github.com/jgong5/ATOM/issues/627#issuecomment-6096543740).
`calib_capture.sh RUN_FILE OUT_DIR` captures them: the real cells one at a time,
each with the host load and the GPUs' processes, VRAM and use at its start and end; its header
lists its environment.
Given two or more records, the command also prints each form's error on every
record under a law fitted on the others, per decode rung and prefill token band.

`scripts/compass/laws/qwen3.8-27b-tp1-mi308x-ptl-disabled-661ef7661.json` is the
`history` form with the host laws (`host`), labelled by device time, fitted on
records captured on hjbog-srdc-39 at ce625ddd9 (ROCm/ATOM 661ef7661) in the
nightly-20261013 image, in this order: concurrency 1, 4, 16, 64 and 256 as
`calib_capture.sh` captures them, then first-turn cells at concurrency 64 and
256, which decode at rungs 16 and up, and at 8 and 14, which give it its rung-8
vector. A first-turn cell runs with `AIPERF_EXTRA_ARGS="--trajectory-start-min-ratio 0
--trajectory-start-max-ratio 0"`, which starts every trajectory at its first
turn. The law carries the host key the records carry with its `atom`
digest replaced by the one `host_key` gives a tree that
adds `ScheduledBatchOutput.group_targets`, changes `clock.charge`, makes the
plugin and diffusion bootstraps refuse a simulated run, names a pipeline
stage's channels by its stage, has a downstream stage's loop leave on
the `+inf` grant, moves the lmcache_mp lookup defer, lookup timeout,
transfer deadlines and deferred-save restamp to the LP clock, and gives
lmcache_mp's lookup poll sleep, transfer start stamps and scheduler adapter a
simulated run's LP clock, runs the simulated P/D connector inside a `multi`,
and has `clock.charge` advance a reply's `out_s` and stamp a recording's
input pulls and output puts; none of these changes a real run's host time. Its
engine key also states the fields the records predate, as their `server.log`
engine arguments and the capture's environment give them: DP width 1, no DP
attention, expert parallelism or TBO, and `AITER_LOG_LEVEL=WARNING`; the law
states that engine key as its `engine_config`. The law
and its host key state `machine_spec`, the digest `machine_spec_digest` gives
`scripts/compass/machines/mi308x-hjbog39.json`, so it refuses under any other
spec's device, software stack or PTL state. Its name states that PTL state
rather than the host it was captured on.

`scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-gpu45-{prefill,decode}.json`
are DeepSeek-V4-Pro's DPA-TP8 laws on mi355-gpu-45, one per side of a 1P1D
deployment: the `base` form with host laws, each fitted alone on its side's
capture at ATOM 2d4044b15 by `compass fit steps` and `compass fit host`.
Each side's cells are concurrency 1, 4, 16, 64 and 256 at `cctraces_sim.sh`'s
default `SEED` and 64 and 256 at `SEED=20260708`; the decode side adds 64 at
`SEED=20260709`, the prefill side two more at 64 and `SEED=20260708`. At DP
width 8 only steps every rank ran are fitted, and only the cells at 64 and
above supply decode steps; each decode rung a law fits has such steps in two or more
cells, so each carries a held-out hull. Their host keys carry the `atom`
digest of the tree they were last re-keyed at; `provenance` names the
capture tree's.

## Cost accuracy on a real run's own steps — `validate.cost`

`python -m atom.compass.validate.cost RECORD LAW` prices each step of a real
cell's record (`OUT_DIR/steps` of a `REAL=1 cctraces_sim.sh` run) with a law
file as `step_linear_fit` writes it, and prints the error per kind, per decode rung
and per prefill token band; a record of several DP ranks is joined on step index,
each step priced and banded as its slowest rank. The module's docstring states what each figure is.
Grade a law only on cells whose traces its calibration cells do not replay.

## Schedule and end-to-end grades — `validate.runs`

`python -m atom.compass.validate.runs SIMULATED REAL...` grades a
simulated `cctraces_sim.sh` cell against two or more spaced real runs of it
(`REAL=1`), each metric against the real runs' own spread. The module's
docstring lists the metrics, their distances and the verdicts.

## A cell graded from every source — `python -m atom.compass.validate`

`python -m atom.compass.validate CELL...` joins the commands above and the run
summaries into one table per cell directory: `step_linear/` or `step_grid/`, and two or more
`real*/` runs of one workload. Its docstring lists the rows.

## The stage-2 cell through the `compass` CLI, from an empty store

Qwen3.8-27B TP1 on MI308X, priced by a step linear law `compass fit steps` fits on
this node, in two checks: check 1, that `cctraces_sim.sh` handed the run file a
`compass run` kept schedules the same steps; check 2, the cell graded against real
runs. Run everything from one tree, `$T` below, in the GPU container: host laws are
keyed by ATOM's source, so records, fits and runs from two trees refuse each other.
In the image, `compass` is `PYTHONPATH=$T python3 -c 'from atom.compass.cli import
main; main()'`, run from `/`, so every path it takes is absolute. Below, `$ATOM` is
`--model <Qwen3.8-27B> --max-model-len 262144`, `$M` is `--machine
$T/scripts/compass/machines/mi308x-hjbog39.json`, `$R` is a directory under `/data`
or `/tmp` and `$S` is `--store $R/store`, empty at the start. The raw export keeps
every prompt, about 100 KB to 760 KB a request on this cell.

**A client run.** No command starts one. Each is `aiperf profile` with
`cctraces_sim.sh`'s arguments, its benchmark id pinned as that script pins it for
a real cell, `--input-file` a directory of one file per trace as that script
splits them, `--url` from `OUT/server.json`, `--artifact-dir OUT/artifacts`,
`--export-level raw`, and a warmup on both sides, so the cold first forward falls
outside the graded window: `--trajectory-start-min-ratio 0
--trajectory-start-max-ratio 0 --warmup-requests-per-lane 1`, one one-token
request per lane before profiling. The grace period after the duration stays
aiperf's finite default: `--benchmark-grace-period inf` can hang aiperf's agentic
replay after the cutoff, waiting on a credit the server no longer holds (upstream,
in agentx-harness's aiperf). Each client sets its own
`AIPERF_DATASET_MMAP_BASE_PATH`, such as `OUT`: every client pins one benchmark id,
so two at once would share a dataset directory. A `run` server's client is the
Python `cctraces_sim.sh` takes as `HARNESS_PYTHON`, with
`PYTHONPATH=$T:$T/compass_harness` (with the harness alone it imports the image's
ATOM) and `ATOM_COMPASS_RUN=OUT/run_file.json`. Then
`python -m atom.compass.parity.client_run OUT/artifacts` writes its export,
`OUT/artifacts/client_run.json`, and SIGTERM to the command stops its server.
Until #1301, a real server's environment sets `TRITON_CACHE_AUTOTUNING=1` and one
`TRITON_CACHE_DIR` for every run, as `cctraces_sim.sh` does for a real cell. Until
#1319, take a graded record only in a container that has served one before: aiter
builds its `mha_varlen_fwd` module on a server's first short prefill on a deep
history, about 60 s inside the run, and keeps it in the container's aiter tree,
which no cache directory above holds.

1. `compass measure memory --gpu G $M $S --out $R/mem -- $ATOM`.
2. Calibration, alone on the node: per concurrency 1, 4, 16, 64 and 256,
   `compass record --gpu G --out $R/cal/c<C> -- $ATOM` serving one client run of
   traces 256 to 392, then
   `python -m atom.compass.parity.recurrence $R/cal/c<C>/steps`.
3. `compass fit steps --backend step-linear --form history $R/cal/c* --client-run
   $R/cal/c<C> $R/cal/c<C>/artifacts/client_run.json ... $M $S --out $R/fit` and
   `compass fit host` on the same inputs; each prints its store token.
4. Real runs: per concurrency, two `compass record --gpu G --out
   $R/cell/c<C>/real<k> -- $ATOM`, each serving one client run of traces 0 to 255,
   at most two GPUs at once.
5. `compass run --step-linear LAW --host-laws HOST $M $S --out
   $R/cell/c<C>/step_linear -- $ATOM`, serving the same client run.
6. Check 2: `python -m atom.compass.validate $R/cell/c<C>`.
7. Check 1: `cctraces_sim.sh` with `RUN_FILE` the `run_file` of
   `$R/cell/c<C>/step_linear/run.json`, `FIRST_SESSION=0 SESSIONS=256
   CONCURRENCY=<C>`, `SERVER_ARGS="--max-model-len 262144"`, `AIPERF_EXTRA_ARGS` the
   warmup arguments; then `python -m atom.compass.parity
   $R/cell/c<C>/step_linear/steps OUT/steps`.

### The same cell under the step grid

The step grid reuses steps 1 to 4: the calibration records are its residual's
targets, and the real runs are its coverage check's inputs and its graded side.
Its points are planned from shape populations rather than records: the planner
refuses inputs stating two KV block counts, and two real starts of one
configuration differ by a few blocks (#1349), while every simulated start takes
its count from the one `memory_readings` entry. A population may serve any
traces, the graded ones included, since its shapes are all the plan reads. The
grid definition is a calibration input; `$G` below holds nodes at powers of two
and no margin:

```json
{"prefill": {"B": [1, 2, 4, ..., 256], "T": [1, 2, 4, ..., 16384],
             "c": [0, 1, 2, 4, ..., 131072]},
 "decode": {"c": [0, 1, 2, 4, ..., 131072]}, "margin": 0}
```

8. Per concurrency, `compass trace steps --step-linear LAW --host-laws HOST $M $S
   --out $R/pop/<traces>/c<C> -- $ATOM`, serving one client run as a `run`
   server's client, of traces 256 to 392 and of traces 0 to 255; its stop
   publishes a `shape_population`, `POP` below.
9. `compass measure steps --grid POP... --grid-definition $G --client-run POP
   $R/pop/<traces>/c<C>/artifacts/client_run.json ... --sampling '{"ignore_eos":
   true}' --gpu G $S --out $R/grid -- $ATOM`. The deep-context prefill corners
   take most of its hours. Several GPUs each take a subset of the populations
   into a store of its own, starts a minute apart; copying each store's
   `step_measurement` entries into one, and this step again over every
   population, measures what remains.
10. `compass fit steps --backend step-grid --grid POP... --grid-definition $G
    $R/cal/c* --client-run ...`, naming every population and record, `$M $S --out
    $R/fit-grid`; it prints the bundle's token. A second fit of one definition
    into one store is refused as an overwrite: refit into a fresh store.
11. `compass run --step-grid BUNDLE --host-laws HOST --coverage $R/cell/c<C>/real*
    $M $S --out $R/cell/c<C>/step_grid -- $ATOM`, serving the same client run as
    step 5. The check counts the real steps the bundle covers, extrapolates past
    the residual's hull or the KV pool, and leaves to the step roofline; grading
    takes only a run that ran it.
12. `python -m atom.compass.validate $R/cell/c<C>`, which reads the bundle from
    the store the run file names, by key and digest.
