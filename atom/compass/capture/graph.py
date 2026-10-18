# SPDX-License-Identifier: MIT
"""From a recorded op stream to an `ir.Graph`: streams to `Par`, layers to `Repeat`.

A recorded op inside decoder layer *i* names its context key with the layer's
own number written `{L}`. Two layers whose regions are equal are
interchangeable for pricing; the layer sequence is run-length encoded, then the
result again, until a pass groups nothing, so `AAAB AAAB ...` nests as
`((A x3) B) x n`. Each `Repeat` binds its own index to its instances' first
layer, and a key names its layer as the sum of the indices around it and an
offset (`{l0+l1}`, `{l0+3}`).

A side stream that waits on the main one forks; the main one waiting on it
joins. What both ran in between is a `Par` of one branch per stream, joined
`resource_bound`: the branches' work shares one device, and an unsaturated one
prices as their max anyway.
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
    """Where a traced step applies: its discrete key, and the trace's guards."""

    structure: str
    guards: tuple[str, ...]

    def describe(self) -> str:
        return " and ".join([self.structure, *self.guards])


@dataclasses.dataclass(frozen=True, slots=True)
class _Run:
    body: tuple
    count: int


def build(items, applicability: Traced) -> ir.Graph:
    """`items`, `(layer or None, Op | Fork | Join)` in dispatch order, as a graph."""
    segments = []  # (layer or None, items), consecutive items of one layer
    for layer, item in items:
        if not segments or segments[-1][0] != layer:
            segments.append((layer, []))
        segments[-1][1].append(item)
    tokens = [(layer is not None, tuple(_regions(seg))) for layer, seg in segments]
    return ir.Graph(applicability, _seq(_place(_fold(tokens), (), 0)))


def _regions(items):
    out, span, open_ = [], [], set()
    for item in items:
        if isinstance(item, Fork):
            open_.add(item.stream)
        elif isinstance(item, Join):
            if item.stream in open_:
                open_.discard(item.stream)
                if not open_:
                    out += _par(span)
                    span = []
        else:
            (span if open_ else out).append(item)
    if open_:
        raise ValueError(f"streams {sorted(open_)} forked and never joined")
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
            out += [_rekey(region, number) for region in token[1]]
        offset += _layers(token)
    return out


def _rekey(region, number):
    if isinstance(region, ir.Op):
        if region.context_ref is None or LAYER not in region.context_ref.index_names:
            return region
        key = region.context_ref.key.replace("{" + LAYER + "}", number)
        return dataclasses.replace(region, context_ref=ir.ContextRef(key))
    if isinstance(region, ir.Par):
        return ir.Par(tuple(_rekey(b, number) for b in region.branches), region.join)
    return ir.Seq(tuple(_rekey(item, number) for item in region.items))


def _seq(regions):
    return regions[0] if len(regions) == 1 else ir.Seq(tuple(regions))


def to_json(graph: ir.Graph) -> str:
    """The graph as JSON: a str dimension is symbolic, an int concrete."""
    applicability = graph.applicability
    return json.dumps(
        {
            "structure": applicability.structure,
            "guards": list(applicability.guards),
            "region": _json(graph.region),
        }
    )


def _json(region):
    if isinstance(region, ir.Op):
        return {
            "op": region.name,
            "kind": region.kind.value,
            "in": [[_dim(d) for d in s] for s in region.in_shapes],
            "out": [[_dim(d) for d in s] for s in region.out_shapes],
            "attrs": dict(region.attrs),
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


def _dim(dim):
    return dim.text if isinstance(dim, ir.SymDim) else dim
