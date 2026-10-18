# SPDX-License-Identifier: MIT
"""`compass trace ops`: one decode step of a model, traced with no device and
published as an `op_graph`.

    compass trace ops --model M --tp N --store S [--width W]

The step is traced at `--width` one-token sequences (at least 2, default 2),
the width held as the symbol `B` (`forward`); the graph folds decoder layers
into `Repeat` and forked streams into `Par` (`graph`). The store gets one
`op_graph` keyed by structure: the model's architecture and config.json
digest, the TP width, decode, eager. Its members are ``graph`` (``.json``,
`graph.to_json`) and ``declared`` (``.json``), what the capture stated in
place of a device, with every host ask of the width it answered. A TP group's
ranks run one graph, so rank 0's is the entry's one member of each. The
declarations are process-global, so a process captures once.
"""

import argparse
import datetime
import json
import tempfile

PROG = "compass trace ops"


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument("--width", type=int, default=2, help="the traced width's hint")
    p.add_argument("--store", required=True, help="the artifact store")
    args = p.parse_args(argv)

    from .forward import capture

    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        graph, model, declared = capture(args.model, args.tp, args.width, tmpdir)
    key = publish(
        args.store,
        graph,
        model,
        declared,
        args.tp,
        " ".join(["compass", "trace", "ops", *argv]),
    )
    print(f"published {key} as {key.token()}")


def publish(store, graph, model, declared, tp, command):
    """Publish `graph` of `model` at TP `tp` as one `op_graph` in `store`."""
    import torch

    from atom.compass.artifacts import (
        ArtifactStore,
        Conditions,
        GateState,
        Key,
        Kind,
        Provenance,
        Reading,
        Topology,
        member_name,
        roots_for,
    )
    from atom.compass.run import host_cpu

    from .graph import to_json

    key = Key.of(Kind.OP_GRAPH, structure=graph.applicability.structure)
    atom_root, aiter_root = roots_for({"atom": "atom", "aiter": "aiter"})
    rank = next(iter(Topology().ranks()))
    ArtifactStore(store).publish(
        key,
        provenance=Provenance(
            command,
            datetime.datetime.now(datetime.timezone.utc).isoformat(),
            (atom_root, aiter_root),
        ),
        topology=Topology(),
        conditions=Conditions.of(
            software_stack=Reading.of_source_root(aiter_root),
            torch=torch.__version__,
            atom_src=Reading.of_source_root(atom_root),
            model=model,
            device=declared["arch"],
            host_cpu=host_cpu(),
            engine_config=json.dumps({"enforce_eager": True, "tp": tp}),
        ),
        gates=GateState.of(),
        members={
            member_name("graph", rank, "json"): to_json(graph).encode(),
            member_name("declared", rank, "json"): json.dumps(declared).encode(),
        },
    )
    return key
