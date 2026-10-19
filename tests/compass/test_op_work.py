# SPDX-License-Identifier: MIT
"""A captured leaf's work, by op kind.

The GEMM and attention leaves match hand counts at a decode and a prefill
shape, their arguments as the Qwen3.8-27B capture records them; each other
extractor matches a hand count on one shape; a kind with no extractor, or a
dtype with no operand format, is refused by name; and every leaf of the
27B's captures at TP1 and TP2 has work that `price` prices.
"""

import pytest
from test_capture_ir import WIDTH_2, capture
from test_work import SPEC

from atom.compass.backends.op_work import work
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.shape import BatchView, RequestShape
from atom.compass.backends.work import Collective, Work, price
from atom.compass.bench.ops import leaves

BF16, I32, F32, F16 = "torch.bfloat16", "torch.int32", "torch.float32", "torch.float16"


def t(shape, dtype=BF16, stride=None, role="activation"):
    if stride is None:
        stride = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            stride[i] = stride[i + 1] * shape[i + 1]
    return {"shape": list(shape), "dtype": dtype, "stride": stride, "role": role}


def leaf(name, outs=(), /, **args):
    return {
        "op": name,
        "kind": "opaque_leaf",
        "args": args,
        "out": [list(s) for s in outs],
    }


def rows(*pairs, decode=True):
    return BatchView(tuple(RequestShape(q, kv, decode) for q, kv in pairs))


DECODE = rows((1, 5), (1, 9))
PREFILL = rows((3, 3), (2, 6), decode=False)
ONE = rows((1, 1))


@pytest.mark.parametrize("m, bias", [(256, 0), (8192, 16480)])
def test_a_gemm_is_2mnk_over_its_operands_bias_and_output(m, bias):
    a = t([m, 5120])
    b = t([16480, 5120], role="weight")
    c = bias and t([bias], role="weight")
    op = leaf("aiter.gemm_a16w16.default", [[m, 16480]], A=a, B=b, bias=c, otype=BF16)

    assert work(op, ONE, 1, "bf16") == Work(
        {"bf16": 2 * m * 16480 * 5120},
        (m * 5120 + 16480 * 5120 + m * 16480 + bias) * 2,
    )


@pytest.mark.parametrize(
    "view, tokens, pairs, keys, kv, width",
    [
        # Rows (1 of 5) and (1 of 9): 5 + 9 query-key pairs, 14 keys read.
        (DECODE, 2, 14, 14, "bf16", 2),
        # Rows (3 of 3) and (2 of 6): (1 + 2 + 3) + (5 + 6) pairs, 9 keys read.
        (PREFILL, 5, 17, 9, "bf16", 2),
        # An fp8 cache: the cached keys and values at its width, not `k`'s.
        (DECODE, 2, 14, 14, "fp8", 1),
    ],
)
def test_attention_counts_the_view_s_rows_over_the_leaf_s_head_widths(
    view, tokens, pairs, keys, kv, width
):
    q, k, v = t([tokens, 6144]), t([tokens, 1024]), t([tokens, 1024], stride=[14336, 1])
    ref = {"context": "language_model.model.layers.3.self_attn"}
    op = leaf(
        "aiter.unified_attention_with_output_base.default",
        [[tokens, 6144]],
        q=q,
        q_scale=None,
        k=k,
        v=v,
        positions=None,
        layer_name=ref,
        use_mla=False,
        qkv=None,
    )

    # 24 heads x 256 and 4 KV heads x 256; Q read and O written, K and V read.
    assert work(op, view, 1, kv) == Work(
        {"bf16": 4 * 6144 * pairs}, tokens * 2 * 6144 * 2 + keys * 2 * 1024 * width
    )


def test_linear_attention_counts_the_recurrence_and_each_row_s_state():
    op = leaf(
        "aiter.linear_attention_with_output_base.default",
        [[2, 48, 128]],
        mixed_qkv=t([2, 10240], stride=[16480, 1]),
        b=t([2, 48], stride=[16480, 1]),
        a=t([2, 48], stride=[16480, 1]),
        core_attn_out=t([2, 48, 128]),
        layer_name={"context": "language_model.model.layers.0.linear_attn"},
    )

    # 48 value heads of 128, 16 key heads of 128; both rows hold history, so
    # each state of 48 x 128 x 128 is read and written.
    per_token = 2 * 16 * 128 + 2 * 48 * 128 + 2 * 48
    state = 48 * 128 * 128
    assert work(op, DECODE, 1, "bf16") == Work(
        {"bf16": 4 * 2 * state}, (2 * per_token + 4 * state) * 2
    )


@pytest.mark.parametrize(
    "name, table, index",
    [
        ("aiter.masked_embedding.default", "weight", "x"),
        ("aten.embedding.default", "weight", "indices"),
        ("aten.index.Tensor", "self", "indices"),
    ],
)
def test_a_lookup_counts_the_indices_and_the_rows_it_writes(name, table, index):
    args = {table: t([124160, 5120], role="weight"), index: t([64], I32)}
    op = leaf(name, [[64, 5120]], **args)

    assert work(op, ONE, 1, "bf16") == Work({}, 64 * 4 + 64 * 5120 * 2)


def test_a_broadcast_dimension_is_read_once():
    op = leaf(
        "aiter.mixed_sample_outer_exponential.default",
        out=t([4], I32),
        input=t([4, 248320]),
        exponentials=t([4, 248320], F32, stride=[0, 1]),
        temperature=t([4], F32),
    )

    assert work(op, ONE, 1, "bf16").bytes == 4 * 4 + 4 * 248320 * 2 + 248320 * 4 + 4 * 4


def test_an_in_place_leaf_writes_what_it_reads():
    q, k = t([2, 24, 256], stride=[14336, 256, 1]), t(
        [2, 4, 256], stride=[14336, 256, 1]
    )
    w = t([256], role="weight")
    op = leaf(
        "atom.model_ops.layernorm:fused_qk_norm",
        q=q,
        k=k,
        q_weight=w,
        k_weight=w,
        eps=1e-6,
        add_unit_offset=True,
    )

    # The strided views count their own elements, not the qkv rows they sit in.
    assert work(op, ONE, 1, "bf16") == Work({}, (2 * 2 * (24 + 4) * 256 + 2 * 256) * 2)


def test_an_aten_operator_takes_torch_s_flop_formula_and_its_boundary():
    mm = leaf("aten.mm.default", [[4, 6]], self=t([4, 8]), mat2=t([8, 6]))
    copy = leaf("aten.copy_.default", [[257]], self=t([257], I32), src=t([257], I32))
    cast = leaf("aten.to.dtype", [[4]], self=t([4], I32), dtype="torch.int64")
    eq = leaf("aten.eq.Scalar", [[2**20]], self=t([2**20], "torch.int64"), other=0)

    assert work(mm, ONE, 1, "bf16") == Work({"bf16": 2 * 4 * 6 * 8}, (32 + 48 + 24) * 2)
    # In place: the destination is written once, not read and written again.
    assert work(copy, ONE, 1, "bf16") == Work({}, 2 * 257 * 4)
    # A cast aliases nothing it returns: it reads int32 and writes int64.
    assert work(cast, ONE, 1, "bf16") == Work({}, 4 * 4 + 4 * 8)
    # No `dtype` slot: each output in the dtype a meta call returns, bool here.
    assert work(eq, ONE, 1, "bf16") == Work({}, 2**20 * 8 + 2**20 * 1)


@pytest.mark.parametrize(
    "op",
    [
        leaf("aten.view.default", [[2, 3]], self=t([6]), size=[2, 3]),
        leaf(
            "aten.split_with_sizes.default", [[2], [4]], self=t([6]), split_sizes=[2, 4]
        ),
        leaf("aten.empty.memory_format", [[6]], size=[6], dtype=BF16),
        leaf("_c10d_functional.wait_tensor.default", [[6]], tensor=t([6])),
    ],
)
def test_a_leaf_that_launches_no_kernel_has_no_work(op):
    assert work(op, ONE, 2, "bf16") is None


def test_a_collective_spans_its_group():
    tensor = t([64, 5120])
    reduce = leaf(
        "aiter.all_reduce_.default", [[64, 5120]], tensor=tensor, group_name="tp:0"
    )
    gather = leaf(
        "_c10d_functional.all_gather_into_tensor.default",
        [[8, 124160]],
        input=t([4, 124160]),
        group_size=2,
        group_name="3",
    )
    cast = leaf("_c10d_functional.broadcast.default", [[64]], input=t([64], I32), src=0)
    buffer = 64 * 5120 * 2

    assert work(reduce, ONE, 2, "bf16") == Work(
        {}, 2 * buffer, (Collective(2, buffer),)
    )
    assert work(reduce, ONE, 1, "bf16").comm == ()
    assert work(gather, ONE, 1, "bf16").comm == (Collective(2, 4 * 124160 * 2),)
    assert work(cast, ONE, 2, "bf16").comm == (Collective(2, 64 * 4),)


def test_a_kind_with_no_extractor_is_refused_by_name():
    op = leaf("aiter.gemm_a8w8.default", [[4, 6]], XQ=t([4, 8], "torch.float8_e4m3fn"))

    with pytest.raises(
        CostRefused, match="no extractor for the op kind aiter.gemm_a8w8"
    ):
        work(op, ONE, 1, "bf16")


def test_kv_indices_counts_its_row_offsets_only():
    op = leaf(
        "atom.utils.block_convert:kv_indices_generate_triton",
        block_tables=t([256, 4096], I32, role="metadata"),
        kv_indices_convert=t([1 << 20], I32, role="metadata"),
        kv_indptr_convert=t([257], I32, role="metadata"),
        ratio=16,
        max_ctx_len=65536,
    )

    assert work(op, ONE, 1, "bf16") == Work({}, 257 * 4)


def test_a_dtype_with_no_operand_format_is_refused_by_kind():
    op = leaf("aten.mm.default", [[4, 6]], self=t([4, 8], F16), mat2=t([8, 6], F16))

    with pytest.raises(CostRefused, match="aten.mm.default: no operand format for"):
        work(op, ONE, 1, "bf16")


def _view(shape: str) -> BatchView:
    """The capture's own rows, each decode row over 1024 tokens of context."""
    sizes = dict(part.split("=") for part in shape.split()[1:])
    if shape.startswith("decode"):
        return rows(*[(1, 1024)] * int(sizes["B"]))
    per, extra = divmod(int(sizes["T"]), int(sizes["B"]))
    counts = [per + (i < extra) for i in range(int(sizes["B"]))]
    return rows(*[(n, n) for n in counts], decode=False)


@pytest.mark.parametrize(
    "tp, shapes", [(1, WIDTH_2), (2, WIDTH_2), (1, "--prefill 512 2 --prefill 300 3")]
)
def test_every_leaf_of_the_27b_captures_has_work_that_prices(
    tp, shapes, tmp_path_factory
):
    kinds, priced = set(), set()
    for shape, (graph, _) in capture(tp, shapes, tmp_path_factory).items():
        view = _view(shape)
        for op, _ in leaves(graph["region"]).values():
            kinds.add(op["op"])
            w = work(op, view, tp, "bf16")
            if w is not None and price(w, SPEC) > 0:
                priced.add(op["op"])

    assert {"aiter.gemm_a16w16.default", "aten.mul.Tensor"} <= priced
    assert len(priced) < len(kinds)  # views among them launch no kernel
