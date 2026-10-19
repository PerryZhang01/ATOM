# SPDX-License-Identifier: MIT
"""`compass trace ops`: steps of the published Qwen3.8-27B, traced with no
device into the IR at decode widths and prefill shapes, each published as an
`op_graph`, at TP1 and TP2.

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
import torch

from atom.compass import cli, ir
from atom.compass.artifacts import ArtifactStore, Kind, Topology
from atom.compass.capture import forward, population_shapes
from atom.compass.capture.graph import Fork, Join, Traced, build

TREE = pathlib.Path(__file__).resolve().parents[2]
CONFIG = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
LAYER_TYPES = json.loads(CONFIG.read_text())["text_config"]["layer_types"]
ATTENTION = {
    "linear_attention": "aiter.linear_attention_with_output_base.default",
    "full_attention": "aiter.unified_attention_with_output_base.default",
}
GEMM = "aiter.gemm_a16w16.default"
MROPE = "atom.model_ops.triton_mrope:try_mrope_qk_fused"
KV_INDICES = "atom.utils.block_convert:kv_indices_generate_triton"
KV_INDICES_METADATA = ("block_tables", "kv_indptr_convert", "kv_indices_convert")


def _op(name, key=None, stream=0, dims=((2,),)):
    ref = key and ir.ContextRef(key)
    return ir.Op(name, "opaque_leaf", (), dims, stream_id=stream, context_ref=ref)


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


# --- lifting a raw launch, on functions written here ---------------------------


_RECORDER: dict = {}


@pytest.fixture
def recorder(monkeypatch):
    """A recording recorder, which the functions below launch through, that
    takes this file for ATOM's own."""
    monkeypatch.setitem(forward._IS_ATOM, __file__, True)
    recorder = forward._Recorder({})
    recorder.on = True
    monkeypatch.setitem(_RECORDER, "now", recorder)
    return recorder


def _scaled(x, scale):
    y = torch.empty_like(x)
    _RECORDER["now"].launch("triton._scaled_kernel")
    _RECORDER["now"].launch("triton._scaled_tail_kernel")
    return y.view(-1)


def _rescaled(x, scale):
    scale = scale * 2
    _RECORDER["now"].launch("triton._scaled_kernel")


@torch.library.custom_op("compass_test::leaf", mutates_args=())
def _leaf(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


_leaf.register_fake(torch.empty_like)


def _wraps_a_leaf(x):
    _leaf(x)
    _RECORDER["now"].launch("triton._scaled_kernel")


def test_a_raw_launch_lifts_to_its_function_with_what_it_dispatched(recorder):
    x = torch.empty(4, 8, device="meta")
    with recorder:
        x.add(1)
        _scaled(x, 0.5)
        x.mul(2)

    names = [op.name for _, op in recorder.items]
    assert names == ["aten.add.Tensor", f"{__name__}:_scaled", "aten.mul.Tensor"]
    lifted = recorder.items[1][1]
    assert lifted.kind is ir.NodeKind.OPAQUE_LEAF
    x_arg = ir.Tensor((4, 8), "torch.float32", (8, 1), ir.Role.METADATA)
    assert dict(lifted.args) == {"x": x_arg, "scale": 0.5}


def test_a_function_that_reassigns_a_parameter_before_its_launch_is_refused(
    recorder,
):
    with recorder, pytest.raises(ValueError, match=r"_rescaled.*\['scale'\]"):
        _rescaled(torch.empty(2, device="meta"), 1.0)


def test_a_function_that_dispatches_a_leaf_is_refused_unless_a_host_query(
    recorder, monkeypatch
):
    x = torch.empty(2, device="meta")
    with recorder, pytest.raises(ValueError, match="compass_test.leaf.default"):
        _wraps_a_leaf(x)

    monkeypatch.setattr(forward, "HOST_QUERIES", ("compass_test.leaf.default",))
    recorder.start()
    with recorder:
        _wraps_a_leaf(x)
    assert [op.name for _, op in recorder.items] == [f"{__name__}:_wraps_a_leaf"]


# --- shapes a population holds ---------------------------------------------------


def _record(tmp_path, lines):
    (tmp_path / "steps").mkdir()
    (tmp_path / "run.json").write_text("{}")
    rows = [{"is_dummy_run": False, "rung": None, **line} for line in lines]
    text = "".join(json.dumps({"step": i, **r}) + "\n" for i, r in enumerate(rows))
    (tmp_path / "steps" / "dp0.jsonl").write_text(text)
    return str(tmp_path)


def test_a_population_names_each_step_s_shape(tmp_path):
    record = _record(
        tmp_path,
        [
            {"rows": [[1, 90, True]] * 3, "rung": 4},
            {"rows": [[1, 90, True]] * 2},
            {"rows": [[100, 0, False], [1, 50, True]]},
            {"rows": [[7, 0, False]], "is_dummy_run": True},
        ],
    )
    shapes = population_shapes([record])
    assert list(map(str, shapes)) == ["decode B=4", "decode B=2", "prefill T=101 B=2"]


def test_a_speculative_decode_step_is_refused_by_name(tmp_path):
    record = _record(tmp_path, [{"rows": [[1, 9, True], [3, 9, True]]}])
    with pytest.raises(ValueError, match="step 0 decodes 3 tokens in a row"):
        population_shapes([record])


def test_a_population_is_a_record_directory_not_a_token():
    with pytest.raises(ValueError, match="not a record directory"):
        population_shapes(["shape_population:0123abcd"])


# --- the published 27B, captured ----------------------------------------------

_CAPTURES: dict = {}


def capture(tp, shapes, tmp_path_factory):
    """`compass trace ops` at TP `tp` over the `shapes` arguments, in one
    process: each shape's published graph and declarations, by shape."""
    if (tp, shapes) in _CAPTURES:
        return _CAPTURES[tp, shapes]
    root = tmp_path_factory.mktemp(f"tp{tp}")
    (root / "model").mkdir()
    (root / "model" / "config.json").write_bytes(CONFIG.read_bytes())
    argv = ["trace", "ops", "--model", str(root / "model"), "--tp", str(tp)]
    argv += [*shapes.split(), "--store", str(root / "store")]
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
    tokens = re.findall(r"^published .* as (\w+)$", done.stdout, re.MULTILINE)
    assert tokens, f"rc {done.returncode}\n{done.stderr[-4000:]}"
    store = ArtifactStore(str(root / "store"))
    rank = next(iter(Topology().ranks()))
    graphs = {}
    for token in tokens:
        entry = store.read(store.key_from_token(Kind.OP_GRAPH, token))
        graph, declared = (
            json.loads(entry.read_member(stem, rank, "json"))
            for stem in ("graph", "declared")
        )
        assert entry.key.values["shape"] == graph["shape"]
        graphs[graph["shape"]] = graph, declared
    _CAPTURES[tp, shapes] = graphs
    return graphs


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


WIDTH_2 = "--width 2"
WIDTHS = "--width 256 --width 2 --width 1 --width 16"


@pytest.mark.parametrize("tp", [1, 2])
def test_trace_ops_folds_the_27b_by_its_layer_types(tp, tmp_path_factory):
    graph, _ = capture(tp, WIDTH_2, tmp_path_factory)["decode B=2"]
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
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
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
    one = _decoder(capture(1, WIDTH_2, tmp_path_factory)["decode B=2"][0])
    two = _decoder(capture(2, WIDTH_2, tmp_path_factory)["decode B=2"][0])

    for at_one, at_two in zip(one, two, strict=True):
        names = [op["op"] for op in at_two]
        # The MLP's down projection and the attention's output projection.
        assert names.count("aiter.all_reduce_.default") == 2
        rest = [n for n in names if n != "aiter.all_reduce_.default"]
        assert rest == [op["op"] for op in at_one]


def test_each_width_is_its_own_graph_whatever_was_captured_beside_it(
    tmp_path_factory,
):
    graphs = capture(1, WIDTHS, tmp_path_factory)
    alone = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]

    assert graphs["decode B=2"][0] == alone[0]
    for width in (1, 2, 16, 256):
        graph, _ = graphs[f"decode B={width}"]
        gemms = [op for op in _ops(graph["region"]) if op["op"] == GEMM]
        assert gemms and all(op["args"]["A"]["shape"][0] == width for op in gemms)


@pytest.mark.parametrize("width", [16, 256])
def test_mrope_is_one_leaf_per_full_attention_layer_at_either_kernel(
    width, tmp_path_factory
):
    graph, _ = capture(1, WIDTHS, tmp_path_factory)[f"decode B={width}"]
    ops = _ops(graph["region"])

    mrope = [op for op in ops if op["op"] == MROPE]
    assert len(mrope) == LAYER_TYPES.count("full_attention")
    assert not [op for op in ops if op["op"].startswith(("triton.", "flydsl."))]
    for op in ops:
        if ":" in op["op"]:
            assert op["op"].startswith("atom.") and op["kind"] == "opaque_leaf"
            module = op["op"].split(":")[0].replace(".", "/")
            assert (TREE / f"{module}.py").is_file()


def test_each_tensor_argument_carries_where_its_storage_came_from(tmp_path_factory):
    graph, _ = capture(1, WIDTHS, tmp_path_factory)["decode B=16"]
    ops = _ops(graph["region"])

    [indices] = [op for op in ops if op["op"] == KV_INDICES]
    roles = {slot: indices["args"][slot]["role"] for slot in KV_INDICES_METADATA}
    assert roles == dict.fromkeys(KV_INDICES_METADATA, "metadata")
    for op in (op for op in ops if op["op"] == GEMM):
        assert (op["args"]["A"]["role"], op["args"]["B"]["role"]) == (
            "activation",
            "weight",
        )
    [mrope, *_] = [op for op in ops if op["op"] == MROPE]
    assert mrope["args"]["positions"]["role"] == "metadata"
    assert re.fullmatch(
        r"\S+\.layers\.\S+\.rotary_emb", mrope["args"]["rotary_emb"]["module"]
    )


def test_a_decode_integer_derived_from_the_context_is_affine_in_it(
    tmp_path_factory,
):
    graphs = capture(1, WIDTHS, tmp_path_factory)

    for width in (2, 16):
        ops = _ops(graphs[f"decode B={width}"][0]["region"])
        affine = {
            (op["op"], slot): value["affine"]
            for op in ops
            for slot, value in op["args"].items()
            if isinstance(value, dict) and "affine" in value
        }
        # ATOM passes it its rows' longest context.
        assert affine == {(KV_INDICES, "max_ctx_len"): [1, 0]}


def _at(n):
    """A decode step of one leaf taking `n`, beside an integer that holds."""
    return [(0, ir.Op("aiter.leaf", "opaque_leaf", (("n", n), ("ratio", 1)), ()))]


def test_an_integer_that_moves_with_the_context_is_affine_or_refused_by_slot():
    c0, c1 = forward.CONTEXTS

    [(_, op)] = forward._affine("decode B=2", _at(3 * c0 + 5), _at(3 * c1 + 5))

    assert dict(op.args) == {"n": ir.Affine(3, 5), "ratio": 1}
    blocks = [-(-c // 16) for c in (c0, c1)]
    with pytest.raises(
        ValueError,
        match=rf"aiter.leaf at decode B=2 takes n={blocks[0]} at context {c0} and "
        rf"{blocks[1]} at {c1}: not affine",
    ):
        forward._affine("decode B=2", *map(_at, blocks))
    with pytest.raises(ValueError, match="dispatches 1 operators at context"):
        forward._affine("decode B=2", _at(1), [])


def test_a_prefill_is_captured_at_its_tokens_and_sequences(tmp_path_factory):
    graphs = capture(1, "--prefill 512 2 --prefill 300 3", tmp_path_factory)

    assert set(graphs) == {"prefill T=512 B=2", "prefill T=300 B=3"}
    for tokens, seqs in ((512, 2), (300, 3)):
        graph, _ = graphs[f"prefill T={tokens} B={seqs}"]
        assert "prefill" in graph["structure"]
        gemms = [op for op in _ops(graph["region"]) if op["op"] == GEMM]
        # Every layer's projections take each token; the head, each sequence's last.
        assert {op["args"]["A"]["shape"][0] for op in gemms} == {tokens, seqs}


def test_trace_ops_refuses_a_width_with_no_sequence(tmp_path):
    with pytest.raises(SystemExit, match="a decode width is at least 1"):
        cli.main(["trace", "ops", "--model", "m", "--width", "0", "--store", "s"])


def test_trace_ops_records_torch_s_operators_and_hides_aiter_s(tmp_path_factory):
    ops = _ops(capture(1, WIDTH_2, tmp_path_factory)["decode B=2"][0]["region"])

    kinds = collections.defaultdict(set)
    for op in ops:
        kinds[op["op"].split(".")[0]].add(op["kind"])
    assert (kinds["aten"], kinds["aiter"]) == ({"captured"}, {"opaque_leaf"})


def _tensors(value):
    if isinstance(value, dict) and "dtype" in value:
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _tensors(item)


def test_no_decoder_tensor_falls_back_to_torch_s_default_dtype(tmp_path_factory):
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
    dtypes = {
        (op["op"], slot, tensor["dtype"])
        for layer in _decoder(graph)
        for op in layer
        for slot, value in op["args"].items()
        for tensor in _tensors(value)
    }
    assert not [d for d in dtypes if d[2] == "torch.float32"]


def test_asking_a_tensor_its_device_is_no_operator(tmp_path_factory):
    prefills = capture(1, "--prefill 512 2 --prefill 300 3", tmp_path_factory)
    graphs = {**capture(1, WIDTH_2, tmp_path_factory), **prefills}
    for graph, _ in graphs.values():
        assert not [
            op for op in _ops(graph["region"]) if op["op"] == "prim.device.default"
        ]


def test_tp2_leaves_its_stand_ins_copies_out_and_says_so(tmp_path_factory):
    one = _ops(capture(1, WIDTH_2, tmp_path_factory)["decode B=2"][0]["region"])
    graph, declared = capture(2, WIDTH_2, tmp_path_factory)["decode B=2"]
    two = _ops(graph["region"])

    added = collections.Counter(op["op"] for op in two)
    added.subtract(op["op"] for op in one)
    # ATOM's legacy collectives write in place; the stand-ins copy back.
    assert added["aten.copy_.default"] == 0
    assert added["_c10d_functional.wait_tensor.default"] == 2
    left_out = collections.defaultdict(list)
    for op in declared["left_out"]:
        left_out[op["why"]].append(op["op"])
    assert set(left_out) == {"h2d", "host", "stand-in", "step-crossing"}
    assert left_out["stand-in"].count("aten.copy_.default") == 2
