# SPDX-License-Identifier: MIT
"""`compass trace ops`: steps of a model, traced with no device, each shape
published as an `op_graph`.

    compass trace ops --model M [--tp N] [--width W ...] [--prefill T B ...]
                      [--population P ...] --store S

A shape is a decode width (`--width`), a prefill of T tokens over B sequences
(`--prefill`), or every shape the steps of a record directory hold
(`--population`): a decode step at its CUDA-graph rung when it ran at one, else
at its width, and any other step as a prefill of its query tokens over its
rows. Each is traced at that concrete shape (`forward`), decoder layers folded
into `Repeat` and forked streams into `Par` (`graph`).

An `op_graph` is keyed by structure, the model's architecture and config.json
digest, the TP width, the phase and eager, and by shape. Its members are
``graph`` (`graph.to_json`) and ``declared``, what the capture stated in place
of a device and the operators it left out of the graph, each with why. A TP
group's ranks run one graph, so rank 0's is the entry's. The declarations are
process-global, so a process runs one capture, of every shape it was given.
"""

import argparse
import datetime
import json
import pathlib
import tempfile
import time


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="compass trace ops", description=__doc__)
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument(
        "--width", type=int, action="append", default=[], help="a decode width"
    )
    p.add_argument(
        "--prefill",
        type=int,
        nargs=2,
        action="append",
        default=[],
        metavar=("T", "B"),
        help="a prefill of T tokens over B sequences",
    )
    p.add_argument(
        "--population",
        action="append",
        default=[],
        help="a record directory; every shape its steps hold",
    )
    p.add_argument("--store", required=True, help="the artifact store")
    args = p.parse_args(argv)

    from atom.compass import artifacts as a

    from .forward import capture, decode, prefill

    store = a.ArtifactStore(args.store)
    shapes = [decode(w) for w in args.width] + [prefill(*s) for s in args.prefill]
    shapes += population_shapes(args.population)
    if not shapes:
        p.error("name a shape: --width, --prefill or --population")
    shapes = list(dict.fromkeys(shapes))
    command = " ".join(["compass", "trace", "ops", *argv])
    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        began = time.perf_counter()  # the first shape's seconds include the build
        for graph, model, declared in capture(args.model, args.tp, shapes, tmpdir):
            seconds = time.perf_counter() - began
            key = publish(store, graph, model, declared, args.tp, command)
            print(f"captured {graph.applicability.shape} in {seconds:.1f} s")
            print(f"published {key} as {key.token()}")
            began = time.perf_counter()


def population_shapes(names):
    """The shape of each step the record directories `names` hold, DP's dummy
    runs aside."""
    from atom.compass.backends.op_graph import step_shape

    from .forward import Shape

    shapes = []
    for name in names:
        path = pathlib.Path(name)
        if not (path / "run.json").is_file():
            raise ValueError(f"{name} is not a record directory: it has no run.json")
        texts = [p.read_text() for p in (path / "steps").glob("dp*[0-9].jsonl")]
        for line in (json.loads(t) for text in texts for t in text.splitlines()):
            if line["is_dummy_run"]:
                continue
            rows = [(row[0], row[2]) for row in line["rows"]]
            try:
                shapes.append(Shape(*step_shape(rows, line["rung"])))
            except ValueError as exc:
                raise ValueError(f"{name} step {line['step']} {exc}") from None
    return shapes


def publish(store, graph, model, declared, tp, command):
    """Publish `graph` of `model` at TP `tp` as one `op_graph` in `store`."""
    import torch

    from atom.compass import artifacts as a
    from atom.compass.artifacts.keys import NO_PTL
    from atom.compass.run import host_cpu

    from .graph import to_json

    applicability = graph.applicability
    key = a.Key.of(
        a.Kind.OP_GRAPH,
        structure=applicability.structure,
        shape=applicability.shape,
    )
    roots = a.roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rank = next(iter(a.Topology().ranks()))
    store.publish(
        key,
        provenance=a.Provenance(command, now, roots),
        topology=a.Topology(),
        conditions=a.Conditions.of(
            software_stack=a.Reading.of_source_root(roots[1]),
            torch=torch.__version__,
            atom_src=a.Reading.of_source_root(roots[0]),
            model=model,
            device=declared["arch"],
            ptl=NO_PTL,
            host_cpu=host_cpu(),
            engine_config=json.dumps({"enforce_eager": True, "tp": tp}),
        ),
        gates=a.GateState.of(),
        members={
            a.member_name("graph", rank, "json"): to_json(graph).encode(),
            a.member_name("declared", rank, "json"): json.dumps(declared).encode(),
        },
    )
    return key
