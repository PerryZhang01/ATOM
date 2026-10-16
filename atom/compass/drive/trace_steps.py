# SPDX-License-Identifier: MIT
"""`compass trace steps`: one simulated instance's server, whose steps are
published as one `shape_population` when it stops.

    compass trace steps <backend> --machine M --store S --out D -- <ATOM args>

`runfile` builds the run file from `run`'s flags, the backend exactly one with
no default, and `serve` starts the server simulated from it with the step
record on; it serves whatever client runs reach it until the command is
stopped, and `--out` holds what `serve` writes. Only shapes are read, so any
traces may drive it. A prefill or a decode instance refuses: it shares a clock
with its peer, which one invocation does not start.

At the stop, the store gets one `shape_population` at `Topology(dp=D)`, D the
run file's ``data_parallel_size``, with three members per DP rank: ``steps``
(``.jsonl``), the rank's step-record lines; ``engine`` (``.json``), the engine
file its runner wrote; and ``summary`` (``.json``), `summary` of those lines
with ``requests``, the record keys of the requests they name. Its key is the
model (run.json's host key states its config.json's digest), the workload (the
digest of every rank's requests), the engine config (`keyed`'s fields, as the
engine file states them) and the backend's digest, so two backends'
populations at one (model, workload, engine config) are two entries. Its
readings are the machine spec's device and stack, and run.json's source,
model and host CPU with that engine config.
"""

import argparse
import datetime
import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    Provenance,
    Reading,
    Topology,
    machine_spec_key,
    member_name,
    roots_for,
)
from atom.compass.backends.shape import RequestShape
from atom.compass.backends.step_grid_residual import raggedness
from atom.compass.drive import runfile, serve
from atom.compass.run.step_grid import BOUNDS

PROG = "compass trace steps"


def main(argv: list[str]) -> None:
    cut = argv.index("--") if "--" in argv else len(argv)
    args = _parser().parse_args(argv[:cut])
    atom_args = argv[cut + 1 :]
    transfer = runfile._engine_args(atom_args).kv_transfer_config
    side = compass_run._pd_side(
        SimpleNamespace(kv_transfer_config=json.loads(transfer))
    )
    if side is not None:
        raise ValueError(
            f"--kv-transfer-config {transfer} makes this a "
            f"{'prefill' if side == 'P' else 'decode'} instance, which shares a "
            "clock with its peer; trace steps starts one aggregated instance"
        )
    run = runfile.build(args, atom_args)
    serve.serve(atom_args, args.out, run=run)
    key = publish(args.out, run, " ".join([*PROG.split(), *argv]))
    print(f"published {key} as {key.token()}")


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    runfile.add_arguments(p)
    p.add_argument(
        "--out", required=True, help="an empty directory for the run's files and logs"
    )
    return p


def publish(out, run: dict, command: str) -> Key:
    """Publish what the stopped server in `out` served, under the run file's
    mapping `run`, as one `shape_population` in the store the run file names."""
    steps = Path(out) / serve.STEPS
    identity = json.loads((Path(out) / serve.RUN_JSON).read_text())
    host, source = identity["host_key"], identity["source"]
    topology, members, requests = Topology(dp=run["data_parallel_size"]), {}, []
    for rank in topology.ranks():
        record = steps / f"dp{rank.dp}.jsonl"
        lines = [json.loads(line) for line in record.read_text().splitlines()]
        named = sorted(key for line in lines for _, key, *_ in line["named"])
        requests += named
        engine = record.with_suffix(".engine.json")
        members |= {
            member_name("steps", rank, "jsonl"): record.read_bytes(),
            member_name("engine", rank, "json"): engine.read_bytes(),
            member_name("summary", rank, "json"): json.dumps(
                {"requests": named, **summary(lines)}
            ).encode(),
        }
    # Any rank's: every DP rank resolves one engine configuration.
    stated = json.loads((steps / "dp0.engine.json").read_text())
    engine_config = json.dumps(
        {k: v for k, v in stated.items() if k not in (*BOUNDS, "num_kvcache_blocks")},
        sort_keys=True,
    )
    workload = hashlib.sha256(json.dumps(sorted(requests)).encode()).hexdigest()
    key = Key.of(
        Kind.SHAPE_POPULATION,
        model=host["model"],
        workload=f"sha256:{workload}",
        engine_config=engine_config,
        backend=run["backend"]["digest"],
    )
    machine = compass_run._machine(run)
    ArtifactStore(run["store"]).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            roots_for({"atom": "atom", "aiter": "aiter"}),
        ),
        topology=topology,
        conditions=Conditions.of(
            **machine_spec_key(machine).values,
            torch=machine.value("device.software_pinned_to.torch"),
            atom_src=Reading(source["revision_kind"], source["revision"]),
            model=host["model"],
            host_cpu=host["host_cpu"],
            engine_config=engine_config,
        ),
        gates=GateState.of(),
        members=members,
    )
    return key


def summary(lines) -> dict:
    """What the step-record `lines` reached, less DP's dummy runs: each
    histogram a list of [value, steps] pairs in value order, the rungs' over
    the steps replayed from a graph; and ``one_width``, each structure every
    step of which ran at one batch width, to that width."""
    steps = [line for line in lines if not line["is_dummy_run"]]
    shapes = [[RequestShape(*row) for row in step["rows"]] for step in steps]
    structures = [_structure(step) for step in steps]
    widths: dict[str, set] = {}
    for structure, rows in zip(structures, shapes):
        widths.setdefault(structure, set()).add(len(rows))
    return {
        "structures": _histogram(structures),
        "batch_widths": _histogram(len(rows) for rows in shapes),
        "context_multisets": _histogram(
            tuple(sorted(row.context_tokens for row in rows)) for rows in shapes
        ),
        "raggedness": _histogram(float(raggedness(rows)) for rows in shapes),
        "rungs": _histogram(s["rung"] for s in steps if s["rung"] is not None),
        "one_width": {s: min(w) for s, w in sorted(widths.items()) if len(w) == 1},
    }


def _structure(step) -> str:
    """What a step runs, whatever its batch width: rows all ``decode``, all
    ``prefill`` or ``mixed``; replayed from a ``graph`` or run ``eager``; and
    ``no output`` when it samples nothing."""
    decode = {row[2] for row in step["rows"]}
    kind = "mixed" if len(decode) == 2 else "decode" if True in decode else "prefill"
    words = [kind, "eager" if step["rung"] is None else "graph"]
    return " ".join(words + ([] if step["produces_output"] else ["no output"]))


def _histogram(values) -> list:
    return [[value, n] for value, n in sorted(Counter(values).items())]
