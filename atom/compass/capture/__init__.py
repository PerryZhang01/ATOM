# SPDX-License-Identifier: MIT
"""`compass trace ops`: one decode step of a model, traced with no device and
published as an `op_graph`.

    compass trace ops --model M [--tp N] [--width W] --store S

The step is traced at `--width` one-token sequences (at least 2), the width
held as the symbol `B` (`forward`); decoder layers fold into `Repeat` and
forked streams into `Par` (`graph`). The `op_graph` is keyed by structure: the
model's architecture and config.json digest, the TP width, decode, eager. Its
members are ``graph`` (`graph.to_json`) and ``declared``, what the capture
stated in place of a device and every host ask of the width it answered. A TP
group's ranks run one graph, so rank 0's is the entry's. The declarations are
process-global, so a process captures once.
"""

import argparse
import datetime
import json
import tempfile


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog="compass trace ops", description=__doc__)
    p.add_argument("--model", required=True, help="a model name or directory")
    p.add_argument("--tp", type=int, default=1, help="tensor-parallel width")
    p.add_argument("--width", type=int, default=2, help="the traced width's hint")
    p.add_argument("--store", required=True, help="the artifact store")
    args = p.parse_args(argv)

    from .forward import capture

    with tempfile.TemporaryDirectory(prefix="compass-capture-") as tmpdir:
        graph, model, declared = capture(args.model, args.tp, args.width, tmpdir)
    command = " ".join(["compass", "trace", "ops", *argv])
    key = publish(args.store, graph, model, declared, args.tp, command)
    print(f"published {key} as {key.token()}")


def publish(store, graph, model, declared, tp, command):
    """Publish `graph` of `model` at TP `tp` as one `op_graph` in `store`."""
    import torch

    from atom.compass import artifacts as a
    from atom.compass.run import host_cpu

    from .graph import to_json

    key = a.Key.of(a.Kind.OP_GRAPH, structure=graph.applicability.structure)
    roots = a.roots_for({"atom": "atom", "aiter": "aiter"})
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    rank = next(iter(a.Topology().ranks()))
    a.ArtifactStore(store).publish(
        key,
        provenance=a.Provenance(command, now, roots),
        topology=a.Topology(),
        conditions=a.Conditions.of(
            software_stack=a.Reading.of_source_root(roots[1]),
            torch=torch.__version__,
            atom_src=a.Reading.of_source_root(roots[0]),
            model=model,
            device=declared["arch"],
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
