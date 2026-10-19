# SPDX-License-Identifier: MIT
"""From a recorded op stream to an `ir.Graph`: streams to `Par`, layers to `Repeat`.

A recorded op inside decoder layer *i* names its context key with the layer's
own number written `{L}`. Two layers whose regions are equal are
interchangeable for pricing; the layer sequence is run-length encoded, then the
result again, until a pass groups nothing, so `AAAB AAAB ...` nests as
`((A x3) B) x n`. A layer is one region, its ops a `Seq` (or the op alone),
and what runs outside every layer sits around them. Each `Repeat` binds its
own index to its instances' first
layer, and a key names its layer as the sum of the indices around it and an
offset (`{l0+l1}`, `{l0+3}`). A module argument's path names its layer alike.

A side stream that waits on the main one forks; the main one waiting on it
joins. What both ran in between is a `Par` of one branch per stream, joined
`resource_bound`: the branches' work shares one device, and an unsaturated one
prices as their max anyway. A stream the step forks and never joins carries
work that crosses the step boundary (the sampled ids' copy to the host): its
ops from that fork on leave the graph for the caller's `left_out`. Whether a
fork is joined is decided over the whole step, and a fork joined in another
segment (another layer, or across the layers) is refused, since a `Par` cannot
cross a layer's edge.
"""

import dataclasses
import json
from typing import NamedTuple

from atom.compass import ir

#: The placeholder a recorded key writes its own layer's number as.
LAYER = "L"


class Fork(NamedTuple):
    stream: int


class Join(NamedTuple):
    stream: int


@dataclasses.dataclass(frozen=True, slots=True)
class Traced(ir.Applicability):
    """Where a traced step applies: its discrete key, and the shape it ran at."""

    structure: str
    shape: str

    def describe(self) -> str:
        return f"{self.structure} at {self.shape}"


@dataclasses.dataclass(frozen=True, slots=True)
class _Run:
    body: tuple
    count: int


def build(items, applicability: Traced, left_out=None) -> ir.Graph:
    """`items`, `(layer or None, Op | Fork | Join)` in dispatch order, as a graph.

    The ops of a stream the step forks and never joins are appended to
    `left_out`.
    """
    segments = []  # (layer or None, items), consecutive items of one layer
    for layer, item in items:
        if not segments or segments[-1][0] != layer:
            segments.append((layer, []))
        segments[-1][1].append(item)
    unjoined, crossing = _unjoined(segments), set()
    left_out = [] if left_out is None else left_out
    tokens = [
        (layer is not None, tuple(_regions(seg, n, unjoined, crossing, left_out)))
        for n, (layer, seg) in enumerate(segments)
    ]
    return ir.Graph(applicability, _seq(_place(_fold(tokens), (), 0)))


def _unjoined(segments):
    """`(segment, position)` of each fork that no later join of its stream answers."""
    pending = {}  # stream: (segment, position) of its earliest unanswered fork
    for n, (layer, seg) in enumerate(segments):
        for i, item in enumerate(seg):
            if isinstance(item, Fork):
                pending.setdefault(item.stream, (n, i))
            elif isinstance(item, Join) and item.stream in pending:
                where = pending.pop(item.stream)[0]
                if where != n:
                    raise ValueError(
                        f"stream {item.stream} forks in {_segment(segments, where)} "
                        f"and joins in {_segment(segments, n)}: a Par cannot cross "
                        "a layer's edge"
                    )
    return set(pending.values())


def _segment(segments, n):
    layer = segments[n][0]
    return f"layer {layer}" if layer is not None else f"segment {n}, outside the layers"


def _regions(items, n, unjoined, crossing, left_out):
    out, span, open_ = [], [], set()
    for i, item in enumerate(items):
        if isinstance(item, Fork):
            if (n, i) in unjoined or item.stream in crossing:
                crossing.add(item.stream)
            else:
                open_.add(item.stream)
        elif isinstance(item, Join):
            if item.stream in open_:
                open_.discard(item.stream)
                if not open_:
                    out += _par(span)
                    span = []
        elif item.stream_id in crossing:
            left_out.append(item)
        else:
            (span if open_ else out).append(item)
    return out


def _par(span):
    branches = {}
    for op in span:
        branches.setdefault(op.stream_id, []).append(op)
    if len(branches) < 2:
        return span
    seqs = tuple(_seq(ops) for ops in branches.values())
    return [ir.Par(seqs, ir.JoinPolicy.RESOURCE_BOUND)]


def _fold(tokens):
    """Group repeated runs, the shortest period first, until nothing groups."""
    while True:
        for period in range(1, len(tokens) // 2 + 1):
            out, i, grouped = [], 0, False
            while i < len(tokens):
                body, count = tokens[i : i + period], 1
                while tokens[i + count * period : i + (count + 1) * period] == body:
                    count += 1
                if count > 1:
                    out.append(_Run(tuple(body), count))
                    i += count * period
                    grouped = True
                else:
                    out.append(tokens[i])
                    i += 1
            if grouped:
                tokens = out
                break
        else:
            return tokens


def _layers(token) -> int:
    if isinstance(token, _Run):
        return token.count * sum(map(_layers, token.body))
    return int(token[0])


def _place(tokens, names, offset):
    """Regions for `tokens`, whose first layer is `sum(names) + offset`."""
    out = []
    for token in tokens:
        if isinstance(token, _Run):
            index = f"l{len(names)}"
            body = _place(token.body, (*names, index), 0)
            step = _layers(token) // token.count
            out.append(ir.Repeat(_seq(body), token.count, index, offset, step))
        else:
            terms = [*names, str(offset)] if offset or not names else list(names)
            number = "{" + "+".join(terms) + "}" if names else str(offset)
            regions = [_rekey(region, number) for region in token[1]]
            out += [_seq(regions)] if token[0] else regions
        offset += _layers(token)
    return out


def _rekey(region, number):
    if isinstance(region, ir.Op):
        if LAYER not in region.free_indices:
            return region
        args = tuple((slot, _renumber(v, number)) for slot, v in region.args)
        ref = _renumber(region.context_ref, number)
        return dataclasses.replace(region, args=args, context_ref=ref)
    if isinstance(region, ir.Par):
        return ir.Par(tuple(_rekey(b, number) for b in region.branches), region.join)
    return ir.Seq(tuple(_rekey(item, number) for item in region.items))


def _renumber(value, number):
    if isinstance(value, ir.ContextRef):
        return type(value)(value.key.replace("{" + LAYER + "}", number))
    if isinstance(value, tuple):
        return tuple(_renumber(item, number) for item in value)
    return value


def _seq(regions):
    return regions[0] if len(regions) == 1 else ir.Seq(tuple(regions))


def to_json(graph: ir.Graph) -> str:
    """The graph as JSON."""
    applicability = graph.applicability
    return json.dumps(
        {
            "structure": applicability.structure,
            "shape": applicability.shape,
            "region": _json(graph.region),
        }
    )


def _arg(value):
    """An argument as JSON: a tensor, module, context key or integer affine in
    the context as an object."""
    if isinstance(value, ir.Affine):
        return {"affine": [value.slope, value.intercept]}
    if isinstance(value, ir.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": value.dtype,
            "stride": list(value.stride),
            "role": value.role.value,
        }
    if isinstance(value, ir.ContextRef):
        kind = "module" if isinstance(value, ir.ModuleRef) else "context"
        return {kind: value.key}
    if isinstance(value, tuple):
        return [_arg(item) for item in value]
    return value


def _json(region):
    if isinstance(region, ir.Op):
        return {
            "op": region.name,
            "kind": region.kind.value,
            "args": {slot: _arg(value) for slot, value in region.args},
            "out": [list(s) for s in region.out_shapes],
            "stream": region.stream_id,
            "context": region.context_ref and region.context_ref.key,
        }
    if isinstance(region, ir.Seq):
        return {"seq": [_json(item) for item in region.items]}
    if isinstance(region, ir.Par):
        return {"par": [_json(b) for b in region.branches], "join": region.join.value}
    return {
        "repeat": _json(region.body),
        "count": region.count,
        "index": region.index,
        "start": region.start,
        "step": region.step,
    }
