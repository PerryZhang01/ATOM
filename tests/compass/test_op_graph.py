# SPDX-License-Identifier: MIT
"""The op-level backend: a step priced from the graph captured at its shape.

Each leaf is one roofline over its work; a `Seq` adds, a `Repeat` re-emits its
body's prices, a `max` `Par` takes its longest branch, and a
`resource_bound` `Par` costs one roofline over the branches' summed work. A step runs at its rung's
graph, or its width's under ``padding: none``, a prefill at its tokens and
rows; attention reads its context in the run's KV format. What cannot be
priced is refused by name, each reason once. The Qwen3.8-27B TP1 capture prices leaf by leaf, and `run.runner` installs the
backend from a run file's ``op_graph`` over the store's graphs.
"""

import json
from types import SimpleNamespace

import pytest
from test_backend_step_linear import _installed, law, view
from test_backend_step_roofline import COMPUTE, MEMORY, _spec
from test_capture_ir import CONFIG, WIDTH_2, _ops, capture
from test_op_work import leaf, t
from test_work import SPEC

from atom.compass.backends import BatchView, CostRefused, Method, RequestShape
from atom.compass.backends.cost import fold_seconds
from atom.compass.backends.op_graph import NONE, OpGraphBackend
from atom.compass.backends.op_work import work
from atom.compass.backends.provenance import Refusal
from atom.compass.backends.work import attention, gemm, price
from atom.compass.run import _op_graphs

GEMM = "aiter.gemm_a16w16.default"
ATTENTION = "aiter.unified_attention_with_output_base.default"
DECODE = view((1, 900, True), (1, 300, True), rung=2)


def g(m, n, k):
    """A GEMM leaf of `[m, k] @ [n, k]^T`."""
    return leaf(GEMM, [[m, n]], A=t([m, k]), B=t([n, k], role="weight"), bias=None)


def backend(region, machine=SPEC, **kwargs):
    return OpGraphBackend(machine, lambda *shape: region, **kwargs)


def by_name(cost):
    return {term.name: term.seconds for term in cost.terms}


def test_a_step_is_one_roofline_per_leaf_composed_by_the_graph_s_joins():
    view_leaf = leaf("aten.view.default", [[2, 64]], self=t([2, 64]), size=[2, 64])
    region = {
        "seq": [
            g(2, 64, 64),
            view_leaf,
            {"repeat": g(2, 128, 64), "count": 3, "index": "l0", "start": 0, "step": 1},
            {"par": [g(2, 2048, 4096), {"seq": [g(2, 4096, 4096)]}], "join": "max"},
            {"par": [g(512, 512, 512), g(512, 256, 512)], "join": "resource_bound"},
        ]
    }
    cost = backend(region).price(DECODE)
    one, body = price(gemm(2, 64, 64), SPEC), price(gemm(2, 128, 64), SPEC)

    assert by_name(cost) == {
        f"op.{GEMM}": fold_seconds([one, body, body, body]),
        "par.max": price(gemm(2, 4096, 4096), SPEC),
        "par.resource_bound": price(gemm(512, 512, 512) + gemm(512, 256, 512), SPEC),
    }


@pytest.mark.parametrize(
    "batch,padded,shape",
    [
        (view(*[(1, 90, True)] * 3, rung=4), True, ("decode", 4, 4)),
        (view(*[(1, 90, True)] * 3, rung=4), False, ("decode", 3, 3)),
        (view((1, 90, True), (1, 9, True)), True, ("decode", 2, 2)),
        (view((100, 100, False), (1, 50, True)), True, ("prefill", 101, 2)),
    ],
    ids=["rung", "padding-none", "eager-decode", "mixed"],
)
def test_a_step_reads_the_graph_captured_at_its_shape(batch, padded, shape):
    asked = []

    def graph(*at):
        asked.append(at)
        return g(1, 8, 8)

    OpGraphBackend(SPEC, graph, padded=padded).price(batch)

    assert asked == [shape]


@pytest.mark.parametrize(
    "batch,graph,named",
    [
        (view((2, 9, True)), g(1, 8, 8), "a step that decodes 2 tokens in a row"),
        (view((1, 9, True)), None, "no op_graph holds a decode step of 1 tokens"),
        (
            BatchView((RequestShape(9, 9, False),), tbo_tokens=((5,), (4,))),
            g(1, 8, 8),
            "two-batch-overlap",
        ),
        (
            BatchView((RequestShape(9, 9, False),), image_tokens=5),
            g(1, 8, 8),
            "5 image tokens",
        ),
    ],
    ids=["speculative", "uncaptured", "tbo", "images"],
)
def test_a_step_with_no_graph_to_price_it_is_refused_by_name(batch, graph, named):
    with pytest.raises(CostRefused, match=named):
        backend(graph).price(batch)


def test_each_reason_a_graph_refuses_is_named_once():
    region = {
        "seq": [
            leaf("foo.bar.default"),
            {
                "repeat": {"seq": [leaf("foo.bar.default"), g(2, 8, 8)]},
                "count": 4,
                "index": "l0",
                "start": 0,
                "step": 1,
            },
            leaf("foo.baz.default"),
        ]
    }
    with pytest.raises(CostRefused) as refused:
        backend(region).price(DECODE)

    assert refused.value.declined == (
        Refusal("op-work", "no extractor for the op kind foo.bar.default"),
        Refusal("op-work", "no extractor for the op kind foo.baz.default"),
    )


def test_a_spec_field_a_leaf_reads_and_the_spec_lacks_refuses_naming_it():
    lacking = SimpleNamespace(values={}, value=SPEC.value, origin=SPEC.origin)

    with pytest.raises(CostRefused, match=f"{GEMM} reads device.kernel_floor_s,"):
        backend(g(2, 8, 8), lacking).price(DECODE)


def test_a_collective_wider_than_the_scale_up_domain_refuses_naming_it():
    reduce = leaf("aiter.all_reduce_.default", [[2, 64]], tensor=t([2, 64]))

    with pytest.raises(CostRefused, match="over 16 ranks leaves the scale-up"):
        backend(reduce, tp=16).price(DECODE)


@pytest.mark.parametrize("kv", [None, "fp8"])
def test_attention_reads_its_context_in_the_run_s_kv_format(kv):
    attn = leaf(ATTENTION, [[2, 6144]], q=t([2, 24, 256]), k=t([2, 4, 256]))
    long = view((1, 200_000, True), (1, 100_000, True), rung=2)
    cost = backend(attn, kv=kv).price(long)
    expected = attention([1, 1], [200_000, 100_000], 24 * 256, 1, 4 * 256, "bf16", kv)

    assert by_name(cost) == {f"op.{ATTENTION}": price(expected, SPEC)}


@pytest.mark.parametrize(
    "batch,reads",
    [(view((1, 900, True)), MEMORY), (view((8192, 8192, False)), COMPUTE)],
    ids=["memory", "compute"],
)
@pytest.mark.parametrize("origin", ["probed", "assumed"])
def test_a_term_is_declared_iff_a_field_its_winning_bound_reads_is_stated(
    batch, reads, origin
):
    big = g(sum(r.query_tokens for r in batch.requests), 4096, 4096)
    (term,) = backend(big, _spec(origin)).price(batch).terms
    note = term.provenance.note

    if origin == "probed":
        assert term.provenance.method is Method.ANALYTICAL and not note
    else:
        assert term.provenance.method is Method.DECLARED
        assert [p for p in MEMORY + COMPUTE if p in note] == list(reads)
        assert note.endswith(NONE)


# --- the published 27B -----------------------------------------------------------


def test_the_27b_s_captured_decode_step_prices_leaf_by_leaf(tmp_path_factory):
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
    cost = backend(graph["region"]).price(DECODE)
    flat: dict = {}
    for op in _ops(graph["region"]):
        w = work(op, DECODE, 1)
        if w is not None:
            flat.setdefault(f"op.{op['op']}", []).append(price(w, SPEC))

    assert by_name(cost) == {name: fold_seconds(s) for name, s in flat.items()}
    assert {f"op.{GEMM}", f"op.{ATTENTION}"} <= by_name(cost).keys()


def test_the_run_s_lookup_reads_the_eager_graph_at_a_shape(tmp_path_factory):
    graph, _ = capture(1, WIDTH_2, tmp_path_factory)["decode B=2"]
    base = tmp_path_factory.getbasetemp()
    hf = SimpleNamespace(architectures=json.loads(CONFIG.read_text())["architectures"])
    found = []
    for store in base.glob("tp1*/store"):
        config = SimpleNamespace(
            model=str(store.parent / "model"),
            hf_config=hf,
            tensor_parallel_size=1,
            enforce_eager=False,
        )
        lookup = _op_graphs(str(store), config)
        found.append((lookup("decode", 2, 2), lookup("decode", 3, 3)))

    assert (graph["region"], None) in found


@pytest.mark.parametrize("padding,padded", [({}, True), ({"padding": "none"}, False)])
def test_a_run_file_op_graph_installs_the_op_level_backend(
    monkeypatch, tmp_path, padding, padded
):
    installed = _installed(monkeypatch, tmp_path, op_graph=True, **padding)

    assert isinstance(installed, OpGraphBackend)
    assert (installed.tp, installed.kv, installed.padded) == (1, "bf16", padded)


def test_a_run_file_op_graph_that_is_not_true_is_refused(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="op_graph is 'yes'; it is true"):
        _installed(monkeypatch, tmp_path, op_graph="yes")


def test_a_run_file_with_op_graph_and_another_backend_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="has step_linear and op_graph"):
        _installed(monkeypatch, tmp_path, step_linear=law(), op_graph=True)
