# SPDX-License-Identifier: MIT
"""`compass run`: one simulated instance's server, serving whatever client runs
reach it until it is stopped, with run.json recording what grading reads.

    compass run <backend> --machine M --store S [--host-laws KEY]
        [--coverage INPUT...] [--on-refusal mark|abort] [--step-roofline]
        --out D -- <ATOM args>

`runfile` builds the run file from the flags, the backend exactly one with no
default, and `serve` starts the server simulated from it with the step record
on. `--out` holds what `serve` writes and what the run writes there: each DP
rank's step record and cost lines, the step table, the run summary and each
LP's hand-in. A prefill or a decode instance refuses: it shares a clock with
its peer, which one invocation does not start.

`--coverage` names record directories and `shape_population` tokens whose steps
the backend's coverage check reads before the server starts: the step grid's
is `step_grid_plan.coverage`, over the bundle `step_grid.load` loads under the
config ATOM's arguments resolve. An input whose DP rank's engine file (a
record's `steps/dp<r>.engine.json`, a population's `engine` member) states
another `SHAPING` field than that config refuses first, naming each field and
both values; a step the check does not cover refuses the run, naming the step.
Another backend has no check, and refuses `--coverage`.

Once the server stops, run.json gains, beside `serve`'s identity: `run_file`,
the run file the server was handed; `step_roofline`, the run file's key,
whether `--step-roofline` was on; `skip`, the host segments the run file's
host entry skips; `refusals`, client run 0's summary's, absent when it has
none, as when a stop came before it ended; `coverage`, the check's report with
the inputs it read, or None when no check ran; and `provenance`, the backend's
digest and fit provenance as the run file states them.
"""

import argparse
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactStore, Kind
from atom.compass.backends.step_grid_plan import BOUNDS, coverage
from atom.compass.drive import runfile, serve
from atom.compass.run import STEP_GRID, STEP_LINEAR, STEP_STUB, step_grid

PROG = "compass run"
#: The engine fields that shape a step's batches: the scheduler's bounds, the
#: capture ladder and the TP width. An input states them as this run resolves
#: them, or refuses.
SHAPING = (*BOUNDS, "ladder", "tensor_parallel_size")


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    atom_args = argv[cut + 1 :]
    if args.coverage and not getattr(args, STEP_GRID):
        key = next(k for k in (STEP_STUB, STEP_LINEAR) if getattr(args, k))
        raise ValueError(
            f"--coverage: --{key.replace('_', '-')} has no coverage check; only "
            "--step-grid's steps are checked"
        )
    transfer = runfile._engine_args(atom_args).kv_transfer_config
    side = compass_run._pd_side(
        SimpleNamespace(kv_transfer_config=json.loads(transfer))
    )
    if side is not None:
        raise ValueError(
            f"--kv-transfer-config {transfer} makes this a "
            f"{'prefill' if side == 'P' else 'decode'} instance, which shares a "
            "clock with its peer; run starts one aggregated instance"
        )
    run = runfile.build(args, atom_args)
    report = check(args.coverage, run, atom_args) if args.coverage else None
    serve.serve(atom_args, args.out, run=run)
    write_grading(args.out, report)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    runfile.add_arguments(p)
    p.add_argument(
        "--coverage",
        nargs="+",
        metavar="INPUT",
        help="record directories and shape_population tokens the check reads",
    )
    p.add_argument(
        "--out", required=True, help="an empty directory for the run's files and logs"
    )
    return p


def check(inputs, run: dict, atom_args) -> dict:
    """The step grid's coverage of the steps `inputs` hold, over the bundle the
    run file `run` names, loaded under the config `atom_args` resolve; an
    input that ran under other `SHAPING` fields, or a step the check does not
    cover, refuses."""
    store, config = ArtifactStore(run["store"]), _config(atom_args)
    here = step_grid.keyed(config) | {name: getattr(config, name) for name in BOUNDS}
    steps = [s for name in inputs for s in _steps(name, store, here)]
    backend, _ = step_grid.load(run[STEP_GRID], config, compass_run._machine(run))
    report = coverage(backend, steps)
    if report["uncovered"]:
        details = [
            f"unmeasured corners of {g}: {c}" for g, c in report["missing"].items()
        ]
        details += [
            f"{r['step']}: {'; '.join(r['reasons'])}" for r in report["refused"]
        ]
        raise ValueError(
            f"--coverage: {len(report['uncovered'])} of {report['steps']} steps "
            "are not covered, so no server starts: "
            + ", ".join(report["uncovered"])
            + "".join(f"\n  {d}" for d in details)
        )
    return {"inputs": list(inputs), **report}


def _steps(name: str, store, here: dict) -> list:
    """The steps of the record directory or `shape_population` token `name`,
    less DP's dummy runs, each named `<name> dp<r> step <n>`; a rank whose
    engine states other `SHAPING` fields than `here` refuses, naming each."""
    path = Path(name)
    if (path / serve.RUN_JSON).is_file():
        records = {
            int(p.stem[2:]): (p.read_text(), p.with_suffix(".engine.json").read_text())
            for p in (path / serve.STEPS).glob("dp*[0-9].jsonl")
        }
    else:
        entry = store.read(store.key_from_token(Kind.SHAPE_POPULATION, name))
        records = {
            rank.dp: (
                entry.read_member("steps", rank, "jsonl").decode(),
                entry.read_member("engine", rank, "json").decode(),
            )
            for rank in entry.topology.ranks()
        }
    for rank, (_, engine) in sorted(records.items()):
        ran = json.loads(engine)
        if differ := [k for k in SHAPING if ran.get(k) != here[k]]:
            raise ValueError(
                f"--coverage: {name} dp{rank} ran under another engine than "
                "ATOM's arguments resolve: "
                + "; ".join(
                    f"{k} {ran.get(k)!r} there, {here[k]!r} here" for k in differ
                )
            )
    return [
        (f"{name} dp{rank} step {line['step']}", line)
        for rank, (text, _) in sorted(records.items())
        for line in map(json.loads, text.splitlines())
        if not line["is_dummy_run"]
    ]


def _config(atom_args):
    """ATOM's `Config` for its arguments, as `LLMEngine` builds one."""
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs

    engine = EngineArgs.from_cli_args(runfile._engine_args(atom_args))
    names = {f.name for f in dataclasses.fields(Config)}
    kwargs = engine._get_engine_kwargs().items()
    return Config(engine.model, **{k: v for k, v in kwargs if k in names})


def write_grading(out, report) -> None:
    """Add what grading reads to the run.json of `out`, a stopped server's
    directory, the coverage check's `report` among it."""
    out = Path(out)
    run = json.loads((out / serve.RUN_FILE).read_text())
    first = out / compass_run.CLIENT_DIR.format(0) / compass_run.SUMMARY_FILE
    written = json.loads((out / serve.RUN_JSON).read_text())
    written |= {
        "run_file": run,
        "step_roofline": run["step_roofline"],
        "skip": [
            s for h in compass_run.entries(run, "host") for s in h.get("skip", [])
        ],
        "coverage": report,
        "provenance": run["backend"],
    }
    if first.is_file():
        summary = json.loads(first.read_text())
        written["refusals"] = summary["schedule"]["refusals"]
    compass_run._write_whole(out / serve.RUN_JSON, json.dumps(written))
