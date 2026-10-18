# SPDX-License-Identifier: MIT
"""Work primitives and the attainable-peak roofline price.

Each primitive matches a hand count on one shape; the price is monotone in each
of its terms, takes a mixed pair's wider rate, reads the bandwidth curve and the
links the spec states, and refuses by name a format the spec has no rate for,
a missing curve, and a collective wider than the scale-up domain.
The named result prices one TP2 decode layer by hand and by `price`.
"""

import pytest
from test_spec_schema import document

from atom.compass.backends.work import (
    Collective,
    Work,
    allgather,
    allreduce,
    attention,
    bounds,
    elementwise,
    gemm,
    price,
)
from atom.compass.spec import MachineSpec
from atom.compass.spec.rules import SpecRefusal

#: The node-18 MI308X spec's numbers, written here so a probe that moves the
#: shipped file does not move the hand count.
BF16 = 344e12 * 0.7
CURVE = [[4096, 1.1e11], [4294967296, 4.505e12]]
FLOOR = 2e-6
LINK, LATENCY = 64e9 * 0.8, 2e-6


def machine(curve=True, **compute):
    written = document()
    device = written["device"]
    device["compute"].update(bf16_flops=344e12, fp8_flops=688e12, **compute)
    device["compute"]["derate"] = {"bf16": 0.7, "fp8": 0.7}
    if "fp4_flops" in compute:
        device["compute"]["derate"]["fp4"] = 0.6
    if curve:
        device["memory"]["bandwidth_curve"] = CURVE
    device["kernel_floor_s"] = FLOOR
    written["interconnect"]["intra_node"]["link_bandwidth_bytes_per_s"] = 64e9
    return MachineSpec.from_mapping(written)


SPEC = machine()


def test_each_primitive_matches_a_hand_count():
    # [4, 8] bf16 @ [8, 6] fp8 -> [4, 6] bf16: 2*4*6*8 flops at the bf16 rate.
    assert gemm(4, 6, 8, "bf16", "fp8") == Work({"bf16": 384}, 64 + 48 + 48)
    # Rows (q=1, kv=5) and (q=3, kv=4): 5 + (2 + 3 + 4) = 14 query-key pairs.
    # Q and O: 4 queries * 4 heads * 8 * 2 B each; K and V: 9 keys * 2 heads * 8 * 1 B each.
    assert attention([1, 3], [5, 4], 4, 8, kv_heads=2, kv_fmt="fp8") == Work(
        {"bf16": 4 * 4 * 8 * 14}, 2 * 4 * 4 * 8 * 2 + 2 * 9 * 2 * 8
    )
    assert elementwise(10, reads=2) == Work({}, 10 * 3 * 2)


def test_each_collective_sends_the_least_any_algorithm_sends():
    assert allreduce(4, 1000).comm == (Collective(4, 1500),)
    assert allgather(4, 1000).comm == (Collective(4, 750),)
    assert allreduce(1, 1000) == Work()


def test_works_add_by_format():
    total = gemm(1, 1, 1) + gemm(1, 1, 1, "fp8", "fp8") + allreduce(2, 8)
    assert total == Work({"bf16": 2, "fp8": 2}, 6 + 4, (Collective(2, 8),))


def test_a_mixed_pair_takes_the_wider_rate():
    m, n, k = 4096, 4096, 4096
    assert bounds(gemm(m, n, k, "bf16", "fp8"), SPEC)["compute"] == pytest.approx(
        2 * m * n * k / BF16
    )
    assert bounds(gemm(m, n, k, "fp8", "fp8"), SPEC)["compute"] == pytest.approx(
        2 * m * n * k / (688e12 * 0.7)
    )


def test_the_memory_term_reads_the_curve():
    def seconds(nbytes, spec=SPEC):
        return bounds(Work({}, nbytes), spec)["memory"]

    # Below and above the curve, its end rate; between, time linear in bytes.
    assert seconds(1024) == pytest.approx(1024 / 1.1e11)
    assert seconds(2**33) == pytest.approx(2**33 / 4.505e12)
    lo, hi = 4096 / 1.1e11, 4294967296 / 4.505e12
    mid = (4096 + 4294967296) // 2
    assert seconds(mid) == pytest.approx((lo + hi) / 2)
    with pytest.raises(SpecRefusal, match="device.memory.bandwidth_curve"):
        price(Work(), machine(curve=False))


def test_a_collective_rides_its_links():
    # TP8 on the fully connected domain: one hop, the bytes over seven links.
    assert bounds(allreduce(8, 8e6), SPEC)["comm"] == pytest.approx(
        LATENCY + 2 * 7 / 8 * 8e6 / (7 * LINK)
    )
    # Sixteen ranks leave the eight-device scale-up domain.
    with pytest.raises(ValueError, match="interconnect.intra_node.domain_devices"):
        price(allreduce(16, 8e6), SPEC)


@pytest.mark.parametrize(
    "grow",
    [
        lambda w: w + gemm(64, 64, 64),
        lambda w: w + Work({"fp8": 10**15}),
        lambda w: w + elementwise(10**9),
        lambda w: w + allreduce(2, 10**6),
    ],
    ids=["compute", "fp8", "memory", "comm"],
)
@pytest.mark.parametrize(
    "base",
    [Work(), gemm(8192, 8192, 8192), elementwise(10**10), allreduce(8, 10**9)],
    ids=["empty", "compute-bound", "memory-bound", "comm-bound"],
)
def test_the_price_is_monotone_in_each_term(base, grow):
    assert price(grow(base), SPEC) >= price(base, SPEC)
    for name, seconds in bounds(grow(base), SPEC).items():
        assert seconds >= bounds(base, SPEC)[name]
    assert price(Work(), SPEC) == FLOOR


def test_a_format_the_spec_lacks_is_refused_by_name():
    with pytest.raises(SpecRefusal, match="fp4_flops"):
        price(gemm(8, 64, 64, "fp4", "fp4"), SPEC)
    with pytest.raises(SpecRefusal, match="fp32_flops"):
        price(gemm(8, 8, 8, "fp32", "bf16"), SPEC)
    with pytest.raises(ValueError, match="'int4'"):
        gemm(8, 8, 8, "int4", "bf16")
    fp4 = machine(fp4_flops=1e15)
    assert bounds(gemm(8, 8, 8, "fp4", "fp4"), fp4)["compute"] == 1024 / (1e15 * 0.6)


def test_one_tp2_decode_layer_priced_by_hand():
    """A dense layer at Qwen3.8-27B's widths, its full-attention layer without
    the output gate: hidden 5120, intermediate 17408, 24 query heads and 4 KV
    heads of 256, all bf16. TP2 halves the heads and the intermediate; eight
    decode rows, each over 4096 tokens of context."""
    rows, hidden, inter, heads, kv_heads, dim = 8, 5120, 17408 // 2, 12, 2, 256
    qkv = (heads + 2 * kv_heads) * dim
    units = {
        "input_norm": elementwise(rows * hidden) + elementwise(hidden, writes=0),
        "qkv_proj": gemm(rows, qkv, hidden),
        "attention": attention([1] * rows, [4096] * rows, heads, dim, kv_heads),
        "o_proj": gemm(rows, hidden, heads * dim),
        "o_allreduce": allreduce(2, rows * hidden * 2),
        "post_norm": elementwise(rows * hidden) + elementwise(hidden, writes=0),
        "gate_up_proj": gemm(rows, 2 * inter, hidden),
        "silu_and_mul": elementwise(rows * inter, reads=2),
        "down_proj": gemm(rows, hidden, inter),
        "down_allreduce": allreduce(2, rows * hidden * 2),
    }

    def memory(nbytes):  # the curve's two points, time linear in bytes
        lo, hi = 4096 / 1.1e11, 4294967296 / 4.505e12
        return lo + (nbytes - 4096) * (hi - lo) / (4294967296 - 4096)

    # (flops, bytes, collective bytes per rank), counted by hand.
    hand = {
        "input_norm": (0, 40960 * 2 * 2 + 5120 * 2, 0),
        "qkv_proj": (2 * 8 * 4096 * 5120, (8 * 5120 + 5120 * 4096 + 8 * 4096) * 2, 0),
        "attention": (
            4 * 12 * 256 * 8 * 4096,
            8 * 2 * 12 * 256 * 2 + 8 * 4096 * 2 * 2 * 256 * 2,
            0,
        ),
        "o_proj": (2 * 8 * 5120 * 3072, (8 * 3072 + 3072 * 5120 + 8 * 5120) * 2, 0),
        "o_allreduce": (0, 0, 81920),
        "post_norm": (0, 40960 * 2 * 2 + 5120 * 2, 0),
        "gate_up_proj": (
            2 * 8 * 17408 * 5120,
            (8 * 5120 + 5120 * 17408 + 8 * 17408) * 2,
            0,
        ),
        "silu_and_mul": (0, 69632 * 3 * 2, 0),
        "down_proj": (2 * 8 * 5120 * 8704, (8 * 8704 + 8704 * 5120 + 8 * 5120) * 2, 0),
        "down_allreduce": (0, 0, 81920),
    }
    by_hand = {
        name: max(
            FLOOR,
            flops / BF16,
            memory(nbytes) if nbytes else 0,
            LATENCY + sent / LINK if sent else 0,
        )
        for name, (flops, nbytes, sent) in hand.items()
    }
    by_price = {name: price(work, SPEC) for name, work in units.items()}
    assert by_price == pytest.approx(by_hand, rel=1e-12)
    assert sum(by_price.values()) == pytest.approx(sum(by_hand.values()), rel=1e-12)
