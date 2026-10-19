# SPDX-License-Identifier: MIT
"""`compass fit ops`: a `price_list` from `measure ops`'s measurements.

    compass fit ops DIR... --store S --out D

Each DIR is a `measure ops --out` directory; each point its ops.jsonl lists as
measured is read back from S, where `measure ops` published it. Every point
must name one `price_list` key, its model, parallel layout, machine-spec
digest and source root, and one torch; the price list is published at that key
under the readings the points ran under.

A leaf's family is its signature with the integers that follow the shape
masked. Among the points of one phase whose signatures match with every
integer masked, an integer position follows the shape when it holds one value
at each shape, a value not the same at every shape, and that value is the
shape's token or sequence count or one more. Any other integer names the leaf:
the GEMMs of a decode step differ in N and K and share a family per weight.

A family's points sit at the shape's counts, B and the context for a decode,
T and B for a prefill. `interpolate` prices a query at a measured point as
that point; between points, multilinearly over the neighbours that bracket it
along each dimension, when they all launched the same kernels. It declines
any other query: outside the measured range, with a neighbour missing, or
between neighbours that launched different kernels, naming the dimension, the
neighbours and their first differing kernel.

The report holds, for each point and dimension along which the point has a
measured neighbour on both sides, the price the other points of that line
give it and its error. A point that launched other kernels than the neighbours
it is priced from is a hidden band: no check of the neighbours' kernels can
see it. A line holding a hidden band or an error over `LIMIT` does not
interpolate: its family lists it under `declined` with its reason, its
measured points still answer, and a query priced along it declines. Per phase
and operator the report gives the median and largest error on the lines that
interpolate.
"""

import argparse
import itertools
import json
import re
import statistics
from pathlib import Path

from atom.compass.artifacts import ArtifactStore, Key, Kind
from atom.compass.fit import publish
from atom.compass.fit.samples import run_fields

PROG = "compass fit ops"
#: What a masked integer reads as in a family's signature.
MASK = "#"
#: The largest held-out error a line that interpolates may hold.
LIMIT = 0.05


class Declined(ValueError):
    """A query the price list does not answer, and why."""


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(prog=PROG, description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+", metavar="DIR", help="measure ops --out")
    p.add_argument("--store", required=True, help="the artifact store's root")
    p.add_argument("--out", required=True, help="where the readable copy goes")
    p.set_defaults(ptl=None)  # every point states its PTL state
    args = p.parse_args(argv)
    points, config, keyed = read(ArtifactStore(args.store), map(Path, args.dirs))
    params = fit(points)
    rows = held_out(params)
    report = {"points": len(points), "families": len(params["families"])}
    report |= {"by_op": summary(rows), "held_out": rows}
    for op, line in report["by_op"].items():
        print(f"{op}: {line}")
    key = Key.of(Kind.PRICE_LIST, **keyed)
    command = " ".join([PROG, *argv])
    publish.publish(args, command, key, publish.conditions(config), params, report)


def read(store, dirs) -> tuple[list[dict], dict, dict]:
    """The measured points the ops.jsonl of `dirs` list, as `store` holds them,
    their configuration, and the `price_list` key fields they name."""
    points, named, config = {}, {}, None
    for d in dirs:
        run = json.loads((d / "run.json").read_text())
        engine = json.loads((d / "engine.json").read_text())
        config = {**run_fields(run), "engine_config": engine}
        config |= {k: run[k] for k in ("stack", "host_key", "ptl")}
        for line in (d / "ops.jsonl").read_text().splitlines():
            record = json.loads(line)
            if "refused" in record:
                continue
            # ponytail: a token is a scan of the kind's folder; index it if a
            # campaign's thousands of points make the fit slow
            key = store.key_from_token(Kind.OP_MEASUREMENT, record["token"])
            entry = store.read(key)
            points[key] = json.loads(entry.read_member("point", publish.RANK, "json"))
            fields = {k: v for k, v in key.values.items() if k != "point"}
            torch = publish.platform(config)["torch"]
            named.setdefault(json.dumps([fields, torch], sort_keys=True), d)
    if not points:
        raise ValueError("these directories list no measured point")
    if len(named) > 1:
        raise ValueError(
            f"the points name {len(named)} price lists, by (key fields, torch): "
            + "; ".join(f"{k} in {d}" for k, d in named.items())
        )
    return list(points.values()), config, json.loads(next(iter(named)))[0]


def _walk(x, f, n):
    """`x` with each integer replaced by `f(position, value)`, positions counted
    over keys in sorted order."""
    if isinstance(x, bool) or not isinstance(x, (int, dict, list)):
        return x
    if isinstance(x, int):
        n[0] += 1
        return f(n[0] - 1, x)
    if isinstance(x, dict):
        return {k: _walk(x[k], f, n) for k in sorted(x)}
    return [_walk(v, f, n) for v in x]


def _ints(sig) -> list[int]:
    got: list[int] = []
    _walk(sig, lambda i, v: got.append(v), [0])
    return got


def _text(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"))


def skeleton(shape: str, leaf: str) -> str:
    """The phase of `shape` and `leaf`'s signature with every integer masked."""
    return _text([shape.split()[0], _walk(json.loads(leaf), lambda i, v: MASK, [0])])


def family(shape: str, leaf: str, free) -> str:
    """The phase of `shape` and `leaf`'s signature with the positions `free`
    holds masked: the integers that follow the shape."""
    sig = _walk(json.loads(leaf), lambda i, v: MASK if i in free else v, [0])
    return _text([shape.split()[0], sig])


def coords(shape: str, context) -> dict:
    """A point's dimensions: the counts `shape` states, then the context."""
    at = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", shape)}
    return at | ({} if context is None else {"context": context})


def fit(points) -> dict:
    """The price list's params: per skeleton, the integer positions that follow
    the shape, and per family, its dimensions and its points, each its
    coordinates, its cold time in ns and its kernels."""
    groups: dict[str, list] = {}
    for p in points:
        groups.setdefault(skeleton(p["shape"], p["leaf"]), []).append(p)
    free_of, families = {}, {}
    for skel, members in groups.items():
        at: dict[str, list] = {}
        for p in members:
            at.setdefault(p["shape"], []).append(_ints(json.loads(p["leaf"])))
        counts = {
            s: {int(c) + d for c in re.findall(r"=(\d+)", s) for d in (0, 1)}
            for s in at
        }
        free_of[skel] = [
            i
            for i in range(len(next(iter(at.values()))[0]))
            if all(len({v[i] for v in vs}) == 1 for vs in at.values())
            and all(vs[0][i] in counts[s] for s, vs in at.items())
            and len({vs[0][i] for vs in at.values()}) > 1
        ]
        for p in members:
            dims = coords(p["shape"], p["context"])
            f = families.setdefault(
                family(p["shape"], p["leaf"], free_of[skel]),
                {"op": p["op"], "dims": list(dims), "points": []},
            )
            f["points"].append([list(dims.values()), p["cold_ns"], p["kernels"]])
    return {"free": free_of, "families": families}


def interpolate(points: dict, at: tuple, dims, declined=None, _path=()):
    """The time at `at` over `points`, {coordinates: (ns, kernels)}, and the
    kernels it was priced from, multilinear over `dims` in order. A corner
    priced along a line in `declined`, {(dimension, the other coordinates):
    reason}, declines with that reason."""
    if not dims:
        corner = tuple(v for _, v, _ in _path)
        for k, (dim, _, moved) in enumerate(_path):
            why = moved and (declined or {}).get((dim, corner[:k] + corner[k + 1 :]))
            if why:
                raise Declined(f"the line along {dim} declines: {why}")
        return points[()]
    x, values = at[0], sorted({c[0] for c in points})
    lo = max((v for v in values if v <= x), default=None)
    hi = min((v for v in values if v >= x), default=None)
    if lo is None or hi is None:
        raise Declined(
            f"{dims[0]} = {x} is outside the measured {values[0]} to {values[-1]}"
        )
    ends = []
    for v in sorted({lo, hi}):
        line = {c[1:]: p for c, p in points.items() if c[0] == v}
        try:
            step = _path + ((dims[0], v, lo != hi),)
            ends.append(interpolate(line, at[1:], dims[1:], declined, step))
        except Declined as why:
            raise Declined(f"at {dims[0]} = {v}, {why}") from None
    if len(ends) == 1:
        return ends[0]
    (a, ka), (b, kb) = ends
    if ka != kb:
        first = next(p for p in itertools.zip_longest(ka, kb) if p[0] != p[1])
        raise Declined(
            f"{dims[0]} = {x} lies between {lo} and {hi}, which launched different "
            f"kernels: {first[0]} and {first[1]}"
        )
    w = (x - lo) / (hi - lo)
    return a + (b - a) * w, ka


def held_out(params) -> list[dict]:
    """Each point priced along each dimension by the other points of its line,
    where it has a neighbour on both sides; a hidden band is a point whose
    kernels are not the ones its neighbours launched. Each family gets its
    `declined` lines, [dimension, the other coordinates, reason]: those
    holding a hidden band or an error over `LIMIT`."""
    rows = []
    for name, f in params["families"].items():
        points = {tuple(c): (ns, k) for c, ns, k in f["points"]}
        mine, why = [], {}
        for c, (ns, own) in points.items():
            for i, dim in enumerate(f["dims"]):
                rest = c[:i] + c[i + 1 :]
                line = {
                    (x[i],): points[x]
                    for x in points
                    if x != c and x[:i] + x[i + 1 :] == rest
                }
                xs = [x for (x,) in line]
                if not (xs and min(xs) < c[i] < max(xs)):
                    continue
                row = {"op": f["op"], "phase": json.loads(name)[0], "dim": dim}
                row |= {"at": list(c), "measured_ns": ns}
                try:
                    priced, kernels = interpolate(line, (c[i],), [dim])
                    row |= {
                        "priced_ns": priced,
                        "error": priced / ns - 1 if ns else None,
                        "hidden_band": kernels != own,
                    }
                except Declined as e:
                    row["declined"] = str(e)
                bad = row.get("hidden_band") and "launched other kernels than"
                if not bad and abs(row.get("error") or 0) > LIMIT:
                    bad = f"reads {row['error']:+.1%} off"
                if bad:
                    why.setdefault((dim, rest), []).append(
                        f"{dim} = {c[i]} {bad} its neighbours"
                    )
                mine.append((row, (dim, rest)))
        rows += [row | {"line_declined": line in why} for row, line in mine]
        f["declined"] = [[d, list(r), "; ".join(w)] for (d, r), w in why.items()]
    return rows


def summary(rows) -> dict:
    """Per phase and operator: points held out, how many were declined between
    kernels or sit on a declined line, and the median and largest absolute
    error on the lines that interpolate."""
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(f"{r['phase']} {r['op']}", []).append(r)
    out = {}
    for name, rs in sorted(by.items()):
        kept = [
            abs(r["error"])
            for r in rs
            if r.get("error") is not None and not r["line_declined"]
        ]
        out[name] = {"held_out": len(rs), "declined": sum("declined" in r for r in rs)}
        out[name] |= {"on_declined_lines": sum(r["line_declined"] for r in rs)}
        if kept:
            out[name] |= {"median_error": statistics.median(kept)}
            out[name] |= {"max_error": max(kept)}
    return out
