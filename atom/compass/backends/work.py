# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Work primitives, and the attainable-peak roofline price of a piece of work.

A `Work` is what one pricing unit does, counted from its shapes alone: flops
per operand format, the compulsory bytes it moves (each input read once, each
output written once, nothing in between), and the collectives it joins. Works
add, so a unit's work is the sum of its primitives'. The price is

    max(t0, sum over formats of F / peak, M / BW(M), t_comm)

at the machine spec's attainable peaks: a format's rate is its `*_flops` times
its own derate, `BW(M)` reads the bandwidth curve, `t0` is the kernel floor,
and each collective costs one hop of link latency plus the least bytes a rank
must send under any algorithm, over its link bandwidth. A matrix product takes
the rate of its wider operand format. A format the spec states no rate for is
refused by the spec, naming the field. A collective wider than the scale-up
domain is refused.
"""

from __future__ import annotations

import bisect
import dataclasses
from collections.abc import Mapping, Sequence

#: Bytes per element, and so the width that decides a mixed pair's rate.
ELEMENT_BYTES = {"fp32": 4, "bf16": 2, "fp8": 1, "fp4": 0.5}


def _size(fmt: str) -> float:
    if fmt not in ELEMENT_BYTES:
        raise ValueError(
            f"no element size for format {fmt!r}; known: {', '.join(ELEMENT_BYTES)}"
        )
    return ELEMENT_BYTES[fmt]


def _wider(a: str, b: str) -> str:
    return a if _size(a) >= _size(b) else b


@dataclasses.dataclass(frozen=True)
class Collective:
    """One collective: the ranks in its group, and the least bytes one rank
    sends under any algorithm."""

    ranks: int
    bytes: float


@dataclasses.dataclass(frozen=True)
class Work:
    """Flops by operand format, compulsory bytes, and the collectives joined."""

    flops_by_format: Mapping[str, float] = dataclasses.field(default_factory=dict)
    bytes: float = 0
    comm: tuple[Collective, ...] = ()

    def __add__(self, other: Work) -> Work:
        flops = dict(self.flops_by_format)
        for fmt, f in other.flops_by_format.items():
            flops[fmt] = flops.get(fmt, 0) + f
        return Work(flops, self.bytes + other.bytes, self.comm + other.comm)


def gemm(m: int, n: int, k: int, a="bf16", b="bf16", out="bf16") -> Work:
    """`[m, k] @ [k, n]`, operands in formats `a` and `b`."""
    moved = m * k * _size(a) + k * n * _size(b) + m * n * _size(out)
    return Work({_wider(a, b): 2 * m * n * k}, moved)


def attention(
    q_lens: Sequence[int],
    kv_lens: Sequence[int],
    heads: int,
    head_dim: int,
    kv_heads: int | None = None,
    fmt="bf16",
    kv_fmt: str | None = None,
) -> Work:
    """Causal attention, one row per request: its query tokens are the last
    `q` of its `kv` tokens, so query `i` of `q` attends to `kv - q + i + 1`.
    QK and PV are each two flops per multiply-add; Q is read and O written in
    `fmt`, K and V read once in `kv_fmt`."""
    kv_heads = kv_heads or heads
    kv_fmt = kv_fmt or fmt
    pairs = 0
    for q, kv in zip(q_lens, kv_lens, strict=True):
        if not 0 <= q <= kv:
            raise ValueError(f"a row attends {q} queries over {kv} keys")
        pairs += q * kv - q * (q - 1) // 2
    moved = sum(q_lens) * 2 * heads * head_dim * _size(fmt) + sum(
        kv_lens
    ) * 2 * kv_heads * head_dim * _size(kv_fmt)
    return Work({_wider(fmt, kv_fmt): 4 * heads * head_dim * pairs}, moved)


def elementwise(elements: int, reads=1, writes=1, fmt="bf16") -> Work:
    """`reads` inputs and `writes` outputs of `elements` each; no matrix flops."""
    return Work({}, elements * (reads + writes) * _size(fmt))


def _collective(ranks: int, sent: float) -> Work:
    return Work(comm=(Collective(ranks, sent),)) if ranks > 1 else Work()


def allreduce(ranks: int, nbytes: float) -> Work:
    """Each rank's `nbytes` buffer reduced into every rank: a rank sends at
    least `2 (N-1)/N` of it."""
    return _collective(ranks, 2 * (ranks - 1) / ranks * nbytes)


def allgather(ranks: int, nbytes: float) -> Work:
    """A gathered output of `nbytes` on every rank, `1/N` of it from each."""
    return _collective(ranks, (ranks - 1) / ranks * nbytes)


def _memory_seconds(nbytes: float, machine) -> float:
    """`nbytes / BW(nbytes)`: the curve's time at each point, linear in bytes
    between points, each end's rate held beyond it."""
    sizes, rates = zip(*machine.value("device.memory.bandwidth_curve"))
    i = bisect.bisect_left(sizes, nbytes)
    if i == 0 or i == len(sizes):
        return nbytes / rates[min(i, len(sizes) - 1)]
    lo, hi = sizes[i - 1] / rates[i - 1], sizes[i] / rates[i]
    return lo + (nbytes - sizes[i - 1]) * (hi - lo) / (sizes[i] - sizes[i - 1])


def _comm_seconds(c: Collective, machine) -> float:
    """One hop plus the bytes over the rank's scale-up links: a fully connected
    group spreads them over its `N-1` links."""
    domain = machine.value("interconnect.intra_node.domain_devices")
    if c.ranks > domain:
        raise ValueError(
            f"a collective over {c.ranks} ranks leaves the scale-up domain of "
            f"{domain} devices (`interconnect.intra_node.domain_devices`); "
            "only a collective inside it is priced"
        )
    link = "interconnect.intra_node"
    lanes = 1
    if machine.value(f"{link}.topology") == "fully_connected":
        lanes = c.ranks - 1
    rate = machine.value(f"{link}.link_bandwidth_bytes_per_s")
    rate *= machine.value(f"{link}.derate") * lanes
    return machine.value(f"{link}.link_latency_s") + c.bytes / rate


def bounds(work: Work, machine) -> dict[str, float]:
    """Each term of the price, by name."""
    compute = sum(
        f
        / (
            machine.value(f"device.compute.{fmt}_flops")
            * machine.value(f"device.compute.derate.{fmt}")
        )
        for fmt, f in work.flops_by_format.items()
    )
    return {
        "floor": machine.value("device.kernel_floor_s"),
        "compute": compute,
        "memory": _memory_seconds(work.bytes, machine),
        "comm": sum(_comm_seconds(c, machine) for c in work.comm),
    }


def price(work: Work, machine) -> float:
    """The seconds one pricing unit takes: the largest of its bounds."""
    return max(bounds(work, machine).values())
