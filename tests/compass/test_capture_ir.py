# SPDX-License-Identifier: MIT
"""`compass trace ops`: a decode step of the published Qwen3.8-27B, traced with
no device into the IR and published as an `op_graph`, at TP1 and TP2.

Each capture runs the console script's `main` in a fresh interpreter, because
what a capture declares in place of a device is process-global. The folding
into `Repeat` and `Par` is also exercised on its own, on op streams written
here.
"""

import collections
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest
import sympy
from test_capture_real_model import EXPECTED_GUARDS, EXPECTED_HOST_RESOLUTIONS

from atom.compass import cli, ir
from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.capture.graph import Fork, Join, Traced, build

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
LAYER_TYPES = json.loads(CONFIG.read_text())["text_config"]["layer_types"]
ATTENTION = {
    "linear_attention": "aiter.linear_attention_with_output_base.default",
    "full_attention": "aiter.unified_attention_with_output_base.default",
}


def _op(name, key=None, stream=0, dims=((2,),)):
    ref = key and ir.ContextRef(key)
    return ir.Op(name, "opaque_leaf", dims, dims, stream_id=stream, context_ref=ref)


def _bound(region, env=None):
    """Every op of `region` in run order, with its context key bound."""
    env = env or {}
    if isinstance(region, ir.Op):
        ref = region.context_ref
        names = ref.index_names if ref else ()
        yield region, ref and ref.bind(**{n: env[n] for n in names}).key
    elif isinstance(region, ir.Repeat):
        for value in region.index_values():
            yield from _bound(region.body, {**env, region.index: value})
    else:
        for item in getattr(region, "items", getattr(region, "branches", ())):
            yield from _bound(item, env)


# --- folding, on op streams written here -------------------------------------


def test_a_hybrid_stack_nests_its_period_and_binds_every_layer():
    items = [(None, _op("embed"))]
    for i, kind in enumerate(LAYER_TYPES):
        key = "model.layers.{L}.attn"
        items += [(i, _op("norm")), (i, _op(kind, key)), (i, _op("mlp"))]
    graph = build(items + [(None, _op("head"))], Traced("x", ()))

    _, outer, _ = graph.region.items
    assert (outer.count, outer.step) == (16, 4)
    assert [type(r) for r in outer.body.items] == [ir.Repeat, ir.Seq]
    assert outer.body.items[0].count == 3
    keyed = [(op.name, key) for op, key in _bound(graph.region) if key]
    assert keyed == [(k, f"model.layers.{i}.attn") for i, k in enumerate(LAYER_TYPES)]


def test_ops_between_a_fork_and_its_join_are_a_par_one_branch_per_stream():
    a, b, c, d = _op("a"), _op("b", stream=1), _op("c"), _op("d", stream=1)
    stream = [a, Fork(1), b, c, d, Join(1), _op("e")]
    region = build([(None, item) for item in stream], Traced("x", ())).region

    par = region.items[1]
    assert par == ir.Par((ir.Seq((b, d)), c), ir.JoinPolicy.RESOURCE_BOUND)


def test_a_stream_the_step_never_joins_leaves_the_graph_for_left_out():
    stream = [_op("a"), Fork(1), _op("copy", stream=1), _op("b")]
    items = [(None, item) for item in stream]
    items += [(0, _op("x")), (0, _op("late", stream=1))]
    left_out = []
    region = build(items, Traced("x", ()), left_out).region

    assert [op.name for op, _ in _bound(region)] == ["a", "b", "x"]
    assert [op.name for op in left_out] == ["copy", "late"]


@pytest.mark.parametrize(
    "layers, where",
    [
        ((0, 1), "forks in layer 0 and joins in layer 1"),
        (
            (None, 0, None),
            "forks in segment 0, outside the layers and joins in segment 2",
        ),
    ],
)
def test_a_fork_joined_in_another_segment_is_refused_by_name(layers, where):
    items = [
        (layers[0], _op("a")),
        (layers[0], Fork(1)),
        (layers[0], _op("s", stream=1)),
    ]
    items += [(layer, _op("b")) for layer in layers[1:]]
    items += [(layers[-1], Join(1))]

    with pytest.raises(ValueError, match=f"stream 1 {where}"):
        build(items, Traced("x", ()))


def test_a_context_key_names_a_layer_as_a_sum_of_indices():
    ref = ir.ContextRef("layers.{outer+inner}.attn.{outer+3}")
    assert ref.index_names == ("outer", "inner")
    assert ref.bind(outer=4, inner=1).key == "layers.5.attn.7"


@pytest.mark.parametrize("key", ["l.{3}", "l.{a+3+b}", "l.{a+}", "l.{a:02}"])
def test_a_hole_that_is_not_a_sum_of_names_then_an_offset_is_refused(key):
    with pytest.raises(ValueError):
        ir.ContextRef(key)


# --- the published 27B, captured --------------------------------------------

_CAPTURES: dict = {}


def capture(tp, width, tmp_path_factory):
    """The graph `compass trace ops` published at (tp, width), and its declarations."""
    if (tp, width) in _CAPTURES:
        return _CAPTURES[tp, width]
    root = tmp_path_factory.mktemp(f"tp{tp}w{width}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model"), "--tp", str(tp)]
    argv += ["--width", str(width), "--store", str(root / "store")]
    done = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from atom.compass.cli import main; main(sys.argv[1:])",
            *argv,
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=1800,
        cwd=TREE,
        env={**os.environ, "PYTHONPATH": str(TREE)},
    )
    published = re.search(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert published, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store = ArtifactStore(str(root / "store"))
    entry = store.read(store.key_from_token(Kind.OP_GRAPH, published.group(1)))
    rank = next(iter(Topology().ranks()))
    graph, declared = (
        json.loads(entry.read_member(stem, rank, "json"))
        for stem in ("graph", "declared")
    )
    _CAPTURES[tp, width] = graph, declared
    return graph, declared


def _layers(region):
    """A graph's decoder layers, each as its ops in run order (repeats expanded)."""
    if "repeat" in region:
        return [
            layer for _ in range(region["count"]) for layer in _layers(region["repeat"])
        ]
    if "seq" in region and any("seq" in i or "repeat" in i for i in region["seq"]):
        return [layer for item in region["seq"] for layer in _layers(item)]
    return [_ops(region)]


def _ops(region):
    if "op" in region:
        return [region]
    if "repeat" in region:
        return [op for _ in range(region["count"]) for op in _ops(region["repeat"])]
    return [op for item in region.get("seq", region.get("par")) for op in _ops(item)]


def _decoder(graph):
    return [
        layer
        for item in graph["region"]["seq"]
        if "op" not in item
        for layer in _layers(item)
    ]


def _at(graph, width):
    """Every op of `graph` with its shapes evaluated at `B = width`."""
    B = sympy.Symbol("B")

    def dim(d):
        return d if isinstance(d, int) else int(sympy.sympify(d).subs(B, width))

    return [
        (op["op"], [[dim(d) for d in s] for s in op["in"] + op["out"]])
        for op in _ops(graph["region"])
    ]


@pytest.mark.parametrize("tp", [1, 2])
def test_trace_ops_folds_the_27b_by_its_layer_types(tp, tmp_path_factory):
    graph, _ = capture(tp, 2, tmp_path_factory)
    layers = _decoder(graph)

    assert len(layers) == len(LAYER_TYPES)
    for layer, kind in zip(layers, LAYER_TYPES):
        [attention] = [op for op in layer if "attention_with_output" in op["op"]]
        assert attention["op"] == ATTENTION[kind]
    # Three linear-attention layers, then a full one, repeated: the period nests.
    assert any(
        "repeat" in inner
        for outer in graph["region"]["seq"]
        if "repeat" in outer and "seq" in outer["repeat"]
        for inner in outer["repeat"]["seq"]
    )


def test_each_layer_s_context_key_binds_to_that_layer(tmp_path_factory):
    graph, _ = capture(1, 2, tmp_path_factory)
    keys = []

    def walk(region, env):
        if "op" in region:
            if region["context"]:
                ref = ir.ContextRef(region["context"])
                keys.append(ref.bind(**{n: env[n] for n in ref.index_names}).key)
        elif "repeat" in region:
            for i in range(region["count"]):
                value = region["start"] + region["step"] * i
                walk(region["repeat"], {**env, region["index"]: value})
        else:
            for item in region.get("seq", region.get("par")):
                walk(item, env)

    walk(graph["region"], {})
    numbers = [int(re.search(r"layers\.(\d+)\.", key).group(1)) for key in keys]
    assert numbers == list(range(len(LAYER_TYPES)))


def test_tp2_adds_two_row_parallel_reduces_to_each_layer(tmp_path_factory):
    one = _decoder(capture(1, 2, tmp_path_factory)[0])
    two = _decoder(capture(2, 2, tmp_path_factory)[0])

    for at_one, at_two in zip(one, two, strict=True):
        names = [op["op"] for op in at_two]
        # The MLP's down projection and the attention's output projection.
        assert names.count("aiter.all_reduce_.default") == 2
        rest = [n for n in names if n != "aiter.all_reduce_.default"]
        assert rest == [op["op"] for op in at_one]


def test_the_graph_traced_at_two_widths_gives_each_width_s_shapes(tmp_path_factory):
    two, eight = capture(1, 2, tmp_path_factory)[0], capture(1, 8, tmp_path_factory)[0]

    for width in (2, 8):
        assert _at(two, width) == _at(eight, width)
    assert _at(two, 2) != _at(two, 8)
    for layer in _decoder(two):
        for op in layer:
            if op["op"].startswith(("aiter.gemm", "aiter.unified", "aiter.linear")):
                assert op["in"][0][0] == "B", op


def test_trace_ops_refuses_a_width_torch_would_specialise(tmp_path):
    with pytest.raises(SystemExit, match="--width 1 traces no symbol"):
        cli.main(["trace", "ops", "--model", "m", "--width", "1", "--store", "s"])


@pytest.mark.parametrize("tp, width", [(1, 2), (2, 2), (1, 8)])
def test_trace_ops_publishes_the_host_asks_and_guards_it_traced_under(
    tp, width, tmp_path_factory
):
    graph, declared = capture(tp, width, tmp_path_factory)

    asks = collections.Counter(declared["host_resolutions"])
    assert asks == EXPECTED_HOST_RESOLUTIONS
    guards = {guard.replace("<axis>", "B") for guard in EXPECTED_GUARDS[tp]}
    assert set(graph["guards"]) == guards


def test_trace_ops_records_torch_s_operators_and_hides_aiter_s(tmp_path_factory):
    ops = _ops(capture(1, 2, tmp_path_factory)[0]["region"])

    kinds = collections.defaultdict(set)
    for op in ops:
        kinds[op["op"].split(".")[0]].add(op["kind"])
    assert (kinds["aten"], kinds["aiter"]) == ({"captured"}, {"opaque_leaf"})


def test_tp2_s_collective_stand_ins_leave_only_the_collective(tmp_path_factory):
    one = _ops(capture(1, 2, tmp_path_factory)[0]["region"])
    graph, declared = capture(2, 2, tmp_path_factory)
    two = _ops(graph["region"])

    added = collections.Counter(op["op"] for op in two)
    added.subtract(op["op"] for op in one)
    added = {name for name, n in added.items() if n > 0}
    assert {name.split(".")[0] for name in added} == {"aiter", "_c10d_functional"}
    stand_ins = [op["op"] for op in declared["left_out"] if op["why"] == "stand-in"]
    assert "aten.copy_.default" in stand_ins
