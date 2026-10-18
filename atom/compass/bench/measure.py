# SPDX-License-Identifier: MIT
"""`compass measure steps`: batches replayed or planned, each distinct batch
measured on the bench once and published as one `step_measurement`.

    compass measure steps [INPUT...] [--grid INPUT... --grid-definition FILE]
        [--sample N --seed S | --sample all] [--sessions A:B]
        [--client-run INPUT EXPORT]... --sampling JSON --gpu G --store ROOT
        --out DIR -- <ATOM args>

An input is a record directory, `record`'s output, or a `shape_population`
entry named by the token `Key.token` prints, whose members are, per DP rank,
`steps` (`.jsonl`), the rank's step-record lines, and `engine` (`.json`), its
engine file. Its lines are read as `parity.served` reads a record, each with
the traces the client runs `--client-run` names for it sent. Lines a dummy run
wrote are dropped; under `--sessions A:B` so is every line serving no trace, or
a trace whose session, as its client run's export states it, lies outside A to
B, inclusive; a trace no export gives a session refuses.
An input refuses unless a `--client-run` names it, its kept lines served
requests under `--sampling` alone, and it states the engine ATOM's arguments
resolve, `keyed`'s fields, at data-parallel width 1, the bench's.

The bench measures at tensor parallel 1: above it, a point charges the ranks'
preload skew (#1188), so a deployment of more ranks refuses.

A batch is a line's view: its `rows`, `rung` and `produces_output`. The
positional inputs' distinct views are replayed, origin `replay`, drawn by
`--sample N --seed S`, N from each stratum (a decode rung, or a prefill kind
in a token band of `step_linear_fit.TOKEN_BANDS`), or all of them by `--sample
all`. `--grid` runs the step grid's planner over its inputs' lines for the
engine they state and the nodes and margin `--grid-definition` states, and its
points are measured with origin `plan`, the digest of the plan, which
`<out>/plan.json` holds, and the `--grid` inputs as their sources. A planned
point that is also a replayed view is measured once, as that replay alone: its
origin, sources and traces are the replay's, and it names no plan.

A measurement's key is the configuration (model, device,
`keyed` engine, source root, `--sampling`), the step kind and the view's
digest, so a view the store holds is not measured again and a stopped
invocation resumes; a held entry taken under other readings than this run's
refuses, naming it. Its member `MEMBER` holds the view, `seconds` (the median
of the bench's kept repeats), the repeats, the protocol, the replay rule, the
origin, the inputs it came from (`sources`) and the `traces` their lines
served. A batch the rule or the bench refuses is not published; the command
measures the rest, then fails naming each. `<out>` also holds run.json, with
the warm-up's and the measuring's boundaries, the engine file and the GPU
checks.

The bench holds the largest KV pool its inputs' engines state, whatever this
device's own budget gives, so every batch they ran, and every point planned
for that pool, fits it.
"""

import argparse
import datetime
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Provenance,
    Topology,
    member_name,
    roots_for,
)
from atom.compass.backends.step_grid_plan import plan
from atom.compass.backends.step_labels import digest as file_digest
from atom.compass.backends.step_linear_fit import token_band
from atom.compass.drive.gpu import add_gpu_flag, hip_visible, quiet
from atom.compass.drive.identity import identity, phase
from atom.compass.fit.publish import conditions
from atom.compass.fit.samples import MEMBER, run_fields
from atom.compass.parity.served import served
from atom.compass.run import _write_whole
from atom.compass.run.step_grid import BOUNDS, keyed, keyed_of

PROG = "compass measure steps"
#: What of a step-record line is the batch it measures.
VIEW = ("rows", "rung", "produces_output")


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    _check(args)
    sampling = _sampling(args.sampling)
    os.environ["HIP_VISIBLE_DEVICES"] = hip_visible(args.gpu)
    config = atom_config(argv[cut + 1 :])
    if config.tensor_parallel_size > 1:
        raise ValueError(
            f"tensor parallel size {config.tensor_parallel_size}: above 1 the bench "
            "charges the ranks' preload skew (#1188), so measure steps measures "
            "at tensor parallel 1 until that is fixed"
        )
    engine = keyed(config)
    store, out = ArtifactStore(args.store), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    points, num_blocks = _points(args, store, engine, sampling, out)
    if not points:
        raise ValueError("the inputs hold no batch to measure once filtered")
    engine_file = out / "engine.json"
    resolved = engine | {name: getattr(config, name) for name in BOUNDS}
    engine_file.write_text(json.dumps(resolved))
    command = [*PROG.split(), *argv]
    run, phases = identity(engine_file, config.model, command, args.gpu), {}
    configuration = run_fields(run)
    readings = conditions(
        {**configuration, "engine_config": engine}
        | {k: run[k] for k in ("stack", "host_key", "ptl")}
    )
    todo = _todo(points, configuration, engine, sampling, store, readings)
    held = len(points) - len(todo)
    print(f"{len(points)} distinct batches, {held} in the store", flush=True)
    refused = []
    try:
        if todo:
            with quiet(args.gpu, out):
                with phase(phases, "warm-up"):
                    bench = start_bench(config, num_blocks)
                try:
                    blocks = {"num_kvcache_blocks": config.num_kvcache_blocks}
                    engine_file.write_text(json.dumps(resolved | blocks))
                    with phase(phases, "measuring"):
                        refused = _measure(
                            bench,
                            config,
                            sampling,
                            store,
                            " ".join(command),
                            readings,
                            todo,
                        )
                finally:
                    bench.exit()
    finally:
        _write_whole(out / "run.json", json.dumps({**run, "phases": phases}))
    if refused:
        raise ValueError(
            f"{len(refused)} of {len(todo)} batches refused:\n" + "\n".join(refused)
        )


def _todo(points, configuration, engine, sampling, store, readings) -> list:
    """Each point's key and the point, for the points the store does not hold;
    a held entry measured under other readings than this run's refuses."""
    todo, stale = [], []
    for point in points.values():
        key = measurement_key(configuration, engine, sampling, point["view"])
        if not store.directory_for(key).exists():
            todo.append((key, point))
            continue
        try:
            store.load(key, conditions=readings, gates=GateState.of())
        except ArtifactRefusal as refusal:
            stale.append(f"{key.token()}: {refusal}")
    if stale:
        raise ValueError(
            f"{len(stale)} measurements in the store were taken under other "
            "readings than this run's:\n" + "\n".join(stale)
        )
    return todo


def measurement_key(configuration: dict, engine: dict, sampling, view: dict) -> Key:
    """The key of `view`'s measurement under `configuration` (model, device,
    source root), `engine`, `keyed`'s fields, and `sampling`."""
    return Key.of(
        Kind.STEP_MEASUREMENT,
        **configuration,
        engine_config=json.dumps(engine, sort_keys=True),
        sampling=json.dumps(sampling, sort_keys=True),
        step_kind=_kind(view),
        view=_digest(view),
    )


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument(
        "inputs",
        nargs="*",
        metavar="INPUT",
        help="record directories and shape_population tokens to replay",
    )
    p.add_argument(
        "--grid",
        nargs="+",
        default=[],
        metavar="INPUT",
        help="record directories and shape_population tokens the planner reads",
    )
    p.add_argument("--grid-definition", metavar="FILE", help="the grid's nodes")
    p.add_argument("--sample", type=_count, help="N per stratum, or all")
    p.add_argument("--seed", type=int)
    p.add_argument("--sessions", type=_span, metavar="A:B")
    p.add_argument(
        "--client-run",
        nargs=2,
        action="append",
        default=[],
        metavar=("INPUT", "EXPORT"),
        help="a client run an input's server served, by its export",
    )
    p.add_argument("--sampling", required=True, help="the request parameters, JSON")
    add_gpu_flag(p)
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where run.json and the plan go")
    return p


def _count(text: str):
    if text == "all" or text.isdigit() and int(text) > 0:
        return text if text == "all" else int(text)
    raise argparse.ArgumentTypeError(f"{text!r} is neither a positive count nor all")


def _span(text: str) -> tuple[int, int]:
    a, colon, b = text.partition(":")
    if colon and a.isdigit() and b.isdigit() and int(a) <= int(b):
        return int(a), int(b)
    raise argparse.ArgumentTypeError(f"{text!r} is not A:B, with A at most B")


def _check(args) -> None:
    named = {*args.inputs, *args.grid}
    for refused, why in (
        (not named, "it names no input to replay and no --grid input to plan over"),
        (
            args.inputs and args.sample is None,
            "replayed batches are drawn by --sample N --seed S or --sample all",
        ),
        (args.sample and not args.inputs, "--sample draws from no replayed input"),
        (isinstance(args.sample, int) and args.seed is None, "--sample N needs --seed"),
        (
            args.grid and not args.grid_definition,
            "--grid plans with the nodes and margin --grid-definition FILE states",
        ),
        (args.grid_definition and not args.grid, "--grid-definition plans no --grid"),
    ):
        if refused:
            raise ValueError(why)
    exported = {i for i, _ in args.client_run}
    if stray := sorted(exported - named):
        raise ValueError(f"--client-run names {', '.join(stray)}, which is no input")
    if unnamed := sorted(named - exported):
        raise ValueError(
            f"{', '.join(unnamed)} is named by no --client-run; name the client "
            "runs its server served, so each measurement records the traces they sent"
        )


def _sampling(text: str) -> dict:
    from atom.sampling_params import SamplingParams

    try:
        sampling = json.loads(text)
        SamplingParams(**sampling)
    except (ValueError, TypeError) as error:
        raise ValueError(
            f"--sampling {text!r} is not a JSON object of SamplingParams fields: "
            f"{error}"
        ) from error
    return sampling


def atom_config(atom_args):
    """ATOM's `Config` for its arguments, as `LLMEngine` resolves one, with the
    tokenizer's ids and the end-of-sequence id apart from the stop ids."""
    import dataclasses

    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import _load_tokenizer

    parser = EngineArgs.add_cli_args(argparse.ArgumentParser(prog=f"{PROG} --"))
    engine_args = EngineArgs.from_cli_args(parser.parse_args(atom_args))
    names = {f.name for f in dataclasses.fields(Config)}
    kwargs = engine_args._get_engine_kwargs().items()
    config = Config(engine_args.model, **{k: v for k, v in kwargs if k in names})
    tokenizer = _load_tokenizer(config.model, config.trust_remote_code)
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    config.stop_token_ids = sorted(set(config.stop_token_ids) - {config.eos_token_id})
    return config


def start_bench(config, num_blocks: int):
    """The bench, started on `config` with a pool of `num_blocks` KV blocks;
    imported here, since it needs a driver."""
    from atom.compass.bench.engine import Bench

    return Bench(config, num_blocks)


def _points(args, store, engine: dict, sampling: dict, out: Path) -> tuple[dict, int]:
    """Each distinct view to measure, by its digest: replayed views drawn as
    `--sample` says, then planned points; and the largest KV pool an input's
    engine states."""
    runs = {}
    for name, export in args.client_run:
        runs.setdefault(name, []).append(export)
    read = {
        n: _read(n, store, runs[n], args.sessions, engine, sampling)
        for n in dict.fromkeys([*args.inputs, *args.grid])
    }
    points = {}
    for name in args.inputs:
        source, lines, _ = read[name]
        for line in lines:
            _add(points, line, "replay", [source], line["traces"])
    strata = {}
    for view, point in points.items():
        strata.setdefault(_stratum(point["view"]), []).append(view)
    if isinstance(args.sample, int):
        drawn, rng = set(), random.Random(args.seed)
        for stratum in sorted(strata):
            views = sorted(strata[stratum])
            drawn.update(rng.sample(views, min(args.sample, len(views))))
        points = {v: p for v, p in points.items() if v in drawn}
    if args.grid:
        definition = json.loads(Path(args.grid_definition).read_text())
        steps = [
            (f"{n} step {line['step']}", line) for n in args.grid for line in read[n][1]
        ]
        planned = plan(definition, steps, {f"{n} dp0": read[n][2] for n in args.grid})
        (out / "plan.json").write_text(json.dumps(planned, indent=1))
        digest = _digest(planned)
        traces = {t for _, line in steps for t in line["traces"]}
        for line in planned["lines"]:
            _add(points, line, "plan", [read[n][0] for n in args.grid], traces, digest)
    return points, max(r[2]["num_kvcache_blocks"] for r in read.values())


def _read(name: str, store, client_runs, sessions, engine: dict, sampling: dict):
    """The input `name`: what names it as a source, its kept lines, and the
    engine its rank states."""
    path = Path(name)
    with tempfile.TemporaryDirectory() as unpacked:
        if (path / "run.json").is_file():
            steps = path / "steps"
            files = sorted(steps.glob("dp*[0-9].jsonl"), key=lambda p: int(p.stem[2:]))
            source = ", ".join(map(file_digest, files))
        else:
            key = store.key_from_token(Kind.SHAPE_POPULATION, name)
            entry, steps, source = store.read(key), Path(unpacked), str(key)
            for rank in entry.topology.ranks():
                for stem, extension, file in (
                    ("steps", "jsonl", f"dp{rank.dp}.jsonl"),
                    ("engine", "json", f"dp{rank.dp}.engine.json"),
                ):
                    (steps / file).write_bytes(entry.read_member(stem, rank, extension))
        record = served(steps, client_runs)
        if len(record) != 1:
            raise ValueError(
                f"{name} holds {len(record)} DP ranks, and the bench measures one"
            )
        ((rank, lines),) = record.items()
        stated = json.loads((steps / f"dp{rank}.engine.json").read_text())
    ran = keyed_of(stated)
    differ = [
        k for k in sorted(ran.keys() | engine.keys()) if ran.get(k) != engine.get(k)
    ]
    if differ:
        raise ValueError(
            f"{name} ran under another engine than ATOM's arguments resolve: "
            + "; ".join(
                f"{k} {ran.get(k)!r} there, {engine.get(k)!r} here" for k in differ
            )
        )
    if sessions:
        exports = [json.loads(Path(c).read_text()) for c in client_runs]
        given = {t: s for e in exports for t, s in e.get("sessions", {}).items()}
        traces = sorted({r["trace"] for e in exports for r in e["requests"].values()})
        if unnumbered := [t for t in traces if t not in given]:
            raise ValueError(
                f"--sessions selects traces by the session their client run's "
                f"export states, and {name} served {', '.join(unnumbered)}, which "
                "no export gives one"
            )
        chosen = {t for t in traces if sessions[0] <= given[t] <= sessions[1]}
    lines = [
        line
        for line in lines
        if not line["is_dummy_run"]
        and (not sessions or line["traces"] and set(line["traces"]) <= chosen)
    ]
    wanted = json.dumps(sampling, sort_keys=True)
    if sent := sorted({p for line in lines for p in line["sampling"]} - {wanted}):
        raise ValueError(
            f"{name} served requests under {' and '.join(sent)}, and --sampling "
            f"states {wanted}"
        )
    return source, lines, stated


def _add(points: dict, line: dict, origin: str, sources, traces, plan=None):
    view = {k: line[k] for k in VIEW}
    point = points.setdefault(
        _digest(view),
        {"view": view, "origin": origin, "sources": set(), "traces": set()},
    )
    if point["origin"] != origin:
        return  # a replayed view the plan also places stays the replay's alone
    point["sources"].update(sources)
    point["traces"].update(traces)
    if plan:
        point["plan"] = plan


def _digest(document) -> str:
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _kind(view: dict) -> str:
    """`decode rung R`, `decode eager`, or `prefill with output` or `without`."""
    if all(decode for *_, decode in view["rows"]):
        rung = view["rung"]
        return "decode eager" if rung is None else f"decode rung {rung}"
    return f"prefill {'with' if view['produces_output'] else 'without'} output"


def _stratum(view: dict) -> str:
    kind = _kind(view)
    if kind.startswith("decode"):
        return kind
    return f"{kind} tokens {token_band(sum(query for query, *_ in view['rows']))}"


def _measure(bench, config, sampling, store, command, readings, todo) -> list[str]:
    """Measure and publish each (key, point) of `todo`; what was refused, and why."""
    from atom.compass.bench.replay import ReplayRule

    rule, roots, refused = ReplayRule(config, sampling), None, []
    for key, point in todo:
        try:
            batch = rule.batch(point["view"])
        except ValueError as error:
            refused.append(f"{key}: {error}")
            continue
        got = bench.measure(batch)
        if "refused" in got:
            refused.append(f"{key}: {got['refused']}")
            continue
        member = {
            **point["view"],
            "seconds": got["median_ns"] / 1e9,
            "statistic": "median",
            "repeats": got["repeats"],
            "discarded": got["discarded"],
            "preload_ns": got["preload_ns"],
            "protocol": got["protocol"],
            "replay_rule": batch.replay_rule,
            "origin": point["origin"],
            "sources": sorted(point["sources"]),
            "traces": sorted(point["traces"]),
        } | ({"plan": point["plan"]} if "plan" in point else {})
        roots = roots or roots_for({"atom": "atom", "aiter": "aiter"})
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        store.publish(
            key,
            provenance=Provenance(command, now, roots),
            topology=Topology(),
            conditions=readings,
            gates=GateState.of(),
            members={member_name(*MEMBER): json.dumps(member).encode()},
        )
        ms = got["median_ns"] / 1e6
        print(
            f"measured {key.values['step_kind']} {key.token()}: {ms:.3f} ms", flush=True
        )
    return refused
