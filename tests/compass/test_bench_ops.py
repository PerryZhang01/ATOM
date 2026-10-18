# SPDX-License-Identifier: MIT
"""`compass measure ops`, the parts that run with no device.

Graphs are built by `trace ops`'s own folding (`capture.graph.build`) and
published as `op_graph` entries; a step is a few torch operators on CPU under
`Harvest`, with a stand-in timer in place of the CUDA-graph one.
"""

import json

import pytest
import torch
from test_artifact_store import CONDITIONS, stanza

from atom.compass import cli, ir
from atom.compass.artifacts import (
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    member_name,
)
from atom.compass.bench import ops
from atom.compass.capture.graph import Traced, build, to_json

KEY = "model.layers.{L}.attn"
LAYERS = ["linear", "linear", "linear", "full"] * 2


def _op(name, ins, outs, key=None, dtype="torch.bfloat16"):
    return ir.Op(
        name,
        "opaque_leaf",
        ins,
        outs,
        attrs={
            "in_dtypes": (dtype,) * len(ins),
            "out_dtypes": (dtype,) * len(outs),
        },
        context_ref=key and ir.ContextRef(key),
    )


def _graph() -> dict:
    """A hybrid stack: a GEMM in every layer, attention by the layer's kind."""
    B = ir.SymDim("B", "t")
    items = [(None, _op("aten.embedding.default", [(64, 8), (B,)], [(B, 8)]))]
    for i, kind in enumerate(LAYERS):
        items += [(i, _op("aten.mm.default", [(B, 8), (8, 8)], [(B, 8)]))]
        items += [(i, _op(f"{kind}_attention", [(B, 8)], [(B, 8)], KEY))]
    tail = ir.SymDim("B + 1", "t"), ir.SymDim("512 - B", "t")
    items += [(None, _op("aten.copy_.default", [tail[:1], tail[1:]], [tail[:1]]))]
    return json.loads(to_json(build(items, Traced("x tp=1 decode eager", ()))))


def _sig(graph, name):
    [sig] = [s for s in ops.leaves(graph["region"]) if json.loads(s)["op"] == name]
    return sig


def test_each_leaf_holds_the_context_key_of_every_layer_it_runs_in():
    leaves = ops.leaves(_graph()["region"])

    by_op = {json.loads(sig)["op"]: keys for sig, keys in leaves.items()}
    full = [f"model.layers.{i}.attn" for i, k in enumerate(LAYERS) if k == "full"]
    linear = [f"model.layers.{i}.attn" for i, k in enumerate(LAYERS) if k == "linear"]
    assert by_op == {
        "aten.embedding.default": [],
        "aten.mm.default": [],
        "linear_attention": linear,
        "full_attention": full,
        "aten.copy_.default": [],
    }


def test_a_point_is_its_leaf_s_call_at_its_binding():
    graph = _graph()
    leaves = ops.leaves(graph["region"])
    point = {"leaf": _sig(graph, "aten.copy_.default"), "bind": {"B": 4}}

    assert ops.concrete({**point, "context": 9}, leaves) == (
        "aten.copy_.default",
        ((5,), (508,)),
        ((5,),),
        ("torch.bfloat16", "torch.bfloat16"),
        ("torch.bfloat16",),
    )


@pytest.mark.parametrize(
    "change, refusal",
    [
        ({"leaf": '{"op":"nowhere"}'}, "the graph has no leaf"),
        ({"bind": {"C": 4}}, "names no context or no width B"),
        ({"context": None}, "names no context"),
    ],
)
def test_a_point_the_graph_cannot_answer_refuses(change, refusal):
    graph = _graph()
    point = {"leaf": _sig(graph, "aten.mm.default"), "bind": {"B": 4}, "context": 9}

    with pytest.raises(ValueError, match=refusal):
        ops.concrete(point | change, ops.leaves(graph["region"]))


def test_a_symbol_the_point_does_not_bind_refuses():
    graph = _graph()
    point = {"leaf": _sig(graph, "aten.mm.default"), "bind": {"B": 4}, "context": 9}
    leaf = json.loads(point["leaf"])
    leaf["in"][0][0] = "B + K"
    leaves = {ops.signature({**leaf, "context": None}): []}

    with pytest.raises(ValueError, match=r"binds no \['K'\]"):
        ops.concrete(point | {"leaf": next(iter(leaves))}, leaves)


def _step(width):
    """A step: an embedding, two GEMMs of the leaf's shape, one of another."""
    table, ids = torch.zeros(64, 8), torch.zeros(width, dtype=torch.long)
    x = torch.nn.functional.embedding(ids, table)
    w = torch.ones(8, 8)
    torch.mm(torch.mm(x, w), w)
    torch.mm(torch.ones(width, 4), torch.ones(4, 8))


F32 = "torch.float32"


def test_harvest_times_a_wanted_call_once_and_offers_nothing_inside_the_timer():
    graph = _graph()
    point = {"leaf": _sig(graph, "aten.mm.default"), "bind": {"B": 3}, "context": 9}
    # The leaf's shapes, in the dtype `_step` runs in.
    mm = ops.concrete(point, ops.leaves(graph["region"]))[:3] + ((F32, F32), (F32,))
    relu = ("aten.relu.default", ((3, 8),), ((3, 8),), (F32,), (F32,))
    timed = []

    def time(fn, args, kwargs, first):
        timed.append((fn, [tuple(a.shape) for a in args], first))
        x = torch.ones(3, 8)
        harvest.offer("aten.relu.default", fn, (x,), {}, [x])  # the timer's own
        return {"cold_ns": 1.0}

    harvest = ops.Harvest({mm: [point], relu: [{"leaf": "relu"}]}, time)
    with harvest:
        _step(3)

    assert timed == [(torch.ops.aten.mm.default, [(3, 8), (8, 8)], point)]
    assert harvest.got == {id(point): {"cold_ns": 1.0}}
    assert list(harvest.wanted) == [relu]


def test_a_leaf_the_step_never_calls_stays_unanswered():
    graph = _graph()
    leaves = ops.leaves(graph["region"])
    point = {"leaf": _sig(graph, "aten.mm.default"), "bind": {"B": 5}, "context": 9}
    wanted = {ops.concrete(point, leaves)[:3] + ((F32, F32), (F32,)): [point]}
    harvest = ops.Harvest(wanted, lambda *a: pytest.fail("offered at the wrong width"))

    with harvest:
        _step(3)

    assert harvest.got == {} and list(harvest.wanted.values()) == [[point]]


def test_copies_keep_each_alias_and_rotate_the_context_key_over_its_layers():
    base = torch.arange(24.0)
    a, b = base[2:8], base.view(4, 6)[:, 1:3]
    keys = ["model.layers.3.attn", "model.layers.7.attn"]

    made = ops.copies((a, b, keys[0], 5), {"scale": base[:1]}, 3, keys)

    assert [args[2] for args, _ in made] == [keys[0], keys[1], keys[0]]
    for (x, y, _, n), kwargs in made:
        storage = x.untyped_storage().data_ptr()
        assert storage != base.untyped_storage().data_ptr()
        assert y.untyped_storage().data_ptr() == storage
        assert kwargs["scale"].untyped_storage().data_ptr() == storage
        assert (y.stride(), y.storage_offset(), n) == (
            b.stride(),
            b.storage_offset(),
            5,
        )
        assert torch.equal(x, a) and torch.equal(y, b)
    assert len({args[0].untyped_storage().data_ptr() for args, _ in made}) == 3


def _publish(root, graph) -> str:
    store = ArtifactStore(str(root))
    key = Key.of(Kind.OP_GRAPH, structure=graph["structure"])
    rank = next(iter(Topology().ranks()))
    store.publish(
        key,
        provenance=stanza("compass trace ops"),
        topology=Topology(),
        conditions=CONDITIONS,
        gates=GateState.of(),
        members={member_name("graph", rank, "json"): json.dumps(graph).encode()},
    )
    return key.token()


@pytest.mark.parametrize(
    "argv, refusal",
    [
        (["--width", "4", "--context", "9", "--gpu", "0,1"], "names 2 GPUs"),
        (["--width", "4", "--gpu", "0"], "names no context"),
        (["--points", "{points}", "--gpu", "0"], "the graph has no leaf"),
    ],
)
def test_measure_ops_refuses_before_it_touches_a_gpu(tmp_path, argv, refusal):
    token = _publish(tmp_path / "store", _graph())
    points = tmp_path / "points.json"
    points.write_text(json.dumps([{"leaf": "{}", "bind": {"B": 2}, "context": 9}]))
    argv = [a.replace("{points}", str(points)) for a in argv]
    argv += ["--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]

    with pytest.raises(SystemExit, match=f"^compass measure ops: .*{refusal}"):
        cli.main(["measure", "ops", token, *argv])
    assert not (tmp_path / "out").exists()
