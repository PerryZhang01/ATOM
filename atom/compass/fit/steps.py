# SPDX-License-Identifier: MIT
"""`compass fit steps`: a step-level backend fitted on labelled samples and published.

    compass fit steps --backend step-linear [--form FORM] [--seed S] RECORD...
        --client-run RECORD EXPORT... --machine M --store ROOT --out DIR
    compass fit steps --backend step-grid --grid INPUT... --grid-definition FILE
        [--sessions A:B] [--seed S] TARGET... --client-run INPUT EXPORT...
        --machine M --store ROOT --out DIR

Each backend in `BACKENDS` states the kind it publishes, the label sources it
takes and its fit, which returns the key fields the configuration does not
give, the params and the report; a flag only another backend reads refuses.
The entry is published under the readings a run on the spec `--machine` names
takes, which `publish.inputs` loads each measurement under.

`step-linear` takes real runs only and fits `step_linear_fit.FORMS`' `--form`,
`history` by default, joining each step's ranks at the engine config's
`data_parallel_size`, so a record with a
rank that gave no sample refuses; it holds out each run on the others when
there are two or more, as that module's command does; `--seed` fixes
its shuffles. The law states the configuration's
`machine_spec` and `engine_config`, as that command's does, and the key holds them. The report's
provenance names each run with every trace its steps served. A step stub's
coefficients are stated, not fitted, so `step-stub` refuses.

`step-grid` plans again over `--grid`'s inputs, record directories and
`shape_population` tokens, with the nodes and margin `--grid-definition`
states, as `measure steps` plans, and reads each planned point's
`step_measurement` by its batch view under its targets' configuration,
whichever replay or plan measured it; a planned point the store does not hold
refuses, naming each, and no other measurement is read as a point. Its targets
are records and the tokens of `measure steps`' replayed batches (`TARGET`); a
planned point named as one refuses. `step_grid_fit.bundle` fits the bundle,
keyed by the grid definition's digest; `--seed` fixes its held-out split. The
report's provenance names each target run with every trace it served, and the
plan with the `--grid` inputs it read and the traces their lines served.

A `--grid` input is read with the client runs a `--client-run` names for it,
and one no `--client-run` names refuses, as `measure steps` refuses to plan
over it. `--sessions A:B` keeps the `--grid` lines `measure steps --sessions
A:B` keeps.
"""

import copy
import json
from pathlib import Path

from atom.compass.artifacts import ArtifactStore, GateState, Key, Kind
from atom.compass.backends import step_grid_fit, step_linear_fit
from atom.compass.backends.step_grid_plan import plan
from atom.compass.bench import measure
from atom.compass.fit import publish
from atom.compass.fit.samples import MEMBER

PROG = "compass fit steps"
STUB = "a step stub's coefficients are stated, not fitted; a run takes their file"


def _step_linear(samples, config, args) -> tuple[dict, dict, dict]:
    form = args.form or "history"
    options = {"form": form, "dp": step_linear_fit.join_width(config["engine_config"])}
    law, report = step_linear_fit.fit(samples, options, args.seed)
    if len({s.provenance["run"] for s in samples}) > 1:
        step_linear_fit.mark_held_out(law, report, samples, options, args.seed)
    report["provenance"] = publish.runs(samples)
    law["machine_spec"] = config["machine_spec"]
    law["engine_config"] = config["engine_config"]
    return {"form": form, "machine_spec": law["machine_spec"]}, law, report


def _step_grid(samples, config, args) -> tuple[dict, dict, dict]:
    named = [s.provenance["run"] for s in samples if s.provenance["source"] == "plan"]
    if named:
        raise ValueError(
            f"{', '.join(named)} measured a planned point, and the residual's "
            "targets are records and replayed batches"
        )
    store = ArtifactStore(args.store)
    definition = json.loads(Path(args.grid_definition).read_text())
    engine, sampling = config["engine_config"], config["sampling"]
    read = {}
    for n in args.grid:
        runs = [e for r, e in args.client_run if Path(r) == Path(n)]
        read[n] = measure._read(n, store, runs, args.sessions, engine, sampling)
    steps = [(f"{n} step {x['step']}", x) for n in args.grid for x in read[n][1]]
    planned = plan(definition, steps, {f"{n} dp0": read[n][2] for n in args.grid})
    fields = ("model", "device", "source_root")
    keys = [
        measure.measurement_key(
            {f: config[f] for f in fields},
            engine,
            sampling,
            {k: line[k] for k in measure.VIEW},
        )
        for line in planned["lines"]
    ]
    missing = [str(key) for key in keys if not store.directory_for(key).exists()]
    if missing:
        raise ValueError(
            f"{len(missing)} of the {len(keys)} planned points are not measured:\n"
            + "\n".join(missing)
        )
    entries = [
        store.load(key, conditions=config["readings"], gates=GateState.of())
        for key in keys
    ]
    points = [json.loads(entry.read_member(*MEMBER)) for entry in entries]
    bundle, report = step_grid_fit.bundle(
        definition, planned, points, samples, args.seed
    )
    report["provenance"] = publish.runs(samples) + [
        {
            "source": "plan",
            "run": measure._digest(planned),
            "inputs": sorted({source for source, *_ in read.values()}),
            "traces": sorted({t for _, x in steps for t in x["traces"]}),
        }
    ]
    fields = {"device": config["device"], "grid": measure._digest(definition)}
    return fields, bundle, report


#: Each backend's name, to the kind it publishes, its label sources and its fit.
BACKENDS = {
    "step-linear": (Kind.STEP_LINEAR_LAW, ("record",), _step_linear),
    "step-grid": (Kind.STEP_GRID_BUNDLE, ("record", "replay"), _step_grid),
}
#: The flags only one backend reads, each to that backend.
OWN = {
    "form": "step-linear",
    "grid": "step-grid",
    "grid_definition": "step-grid",
    "sessions": "step-grid",
}


def main(argv: list[str]) -> None:
    parser = publish.parser(PROG, __doc__.split("\n\n")[0])
    parser.add_argument("--backend", required=True, help=", ".join(BACKENDS))
    parser.add_argument(
        "--form", choices=step_linear_fit.FORMS, help="history by default"
    )
    parser.add_argument(
        "--grid",
        nargs="+",
        metavar="INPUT",
        help="record directories and shape_population tokens the planner reads",
    )
    parser.add_argument("--grid-definition", metavar="FILE", help="the grid's nodes")
    parser.add_argument(
        "--sessions",
        type=measure._span,
        metavar="A:B",
        help="the --grid lines measure steps --sessions A:B keeps",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.backend == "step-stub":
        raise ValueError(STUB)
    if args.backend not in BACKENDS:
        raise ValueError(
            f"no backend {args.backend} is fitted; {' and '.join(BACKENDS)} are"
        )
    for flag, owner in OWN.items():
        if getattr(args, flag) is not None and args.backend != owner:
            raise ValueError(
                f"--{flag.replace('_', '-')} is read by {owner}, not {args.backend}"
            )
    if args.backend == "step-grid" and not (args.grid and args.grid_definition):
        raise ValueError(
            "step-grid plans over --grid INPUT... with the nodes and margin "
            "--grid-definition FILE states"
        )
    named = {Path(r) for r, _ in args.client_run}
    if unnamed := [n for n in args.grid or () if Path(n) not in named]:
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, as measure steps planned over it with them"
        )
    kind, sources, fit = BACKENDS[args.backend]
    # A --client-run naming a --grid input and no target is the planner's alone.
    grid_only = {Path(n) for n in args.grid or ()} - {Path(i) for i in args.inputs}
    target_args = copy.copy(args)
    target_args.client_run = [c for c in args.client_run if Path(c[0]) not in grid_only]
    samples, config = publish.inputs(target_args, sources)
    fields, params, report = fit(samples, config, args)
    key = Key.of(
        kind,
        **{f: config[f] for f in ("model", "source_root")},
        engine_config=json.dumps(config["engine_config"], sort_keys=True),
        **fields,
    )
    command = " ".join([PROG, *argv])
    publish.publish(args, command, key, config["readings"], params, report)
