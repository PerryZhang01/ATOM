# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A simulated run's non-KV memory against a real run's, one term at a time.

    python -m atom.compass.memory.check PREDICTED SERVER_LOG

PREDICTED is an engine's ``lp-<engine>.json`` in a simulated run's ``out_dir``,
whose memory section holds the readings its KV pool was sized from, the warmup shape, and
what the graph pool really costs. SERVER_LOG is the log of one real ATOM
server with the same deployment.

A real server logs each reading once, in its ``Memory budget`` line, so the
terms compared are ``peak_torch`` and ``non_torch``. Each predicted reading
enters as one term, whose method combines its own terms' methods, and its
terms are printed under the table, each with its method. The
graph pool is compared apart, both of its numbers against the pool the
real server's capture reserved.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from pathlib import Path

from atom.compass.backends.provenance import Method
from atom.compass.memory import graph_pool
from atom.compass.memory.budget import SUBTRACTED
from atom.compass.memory.compare import (
    Predicted,
    Recorded,
    Shape,
    compare,
    compare_graph_pool,
    footprint_terms,
)
from atom.compass.memory.readings import DeviceReadings, MemoryRefusal
from atom.compass.memory.terms import Reading, Term

#: Both sides are taken at the warmup forward, ATOM's and the simulated one.
PHASE = "warmup"
#: The readings compared as footprint terms; the activations are in the first.
COMPARED = ("peak_torch", "non_torch")
SHAPED = frozenset({"peak_torch"})

_BUDGET = re.compile(r"Memory budget: .*\bpeak_torch=([\d.]+)GB, non_torch=([\d.]+)GB")
_WARMUP = re.compile(r"warmup_model [\d.]+ seconds with \d+ reqs (\d+) tokens")
_POOL = re.compile(
    r"CUDA graph capture memory: \d+ graphs \| pool\(reserved\)=([\d.]+)GB"
)


def _rows(reading: Reading) -> list[dict]:
    return [dataclasses.asdict(t) | {"method": t.method.value} for t in reading.terms]


def _reading(name: str, rows: list[dict]) -> Reading:
    if any("basis" in row for row in rows):
        raise MemoryRefusal(
            f"the {name} reading records each term's basis, a vocabulary this "
            "check no longer reads",
            "rerun the simulated run on this tree, which records each method",
        )
    return Reading(
        name, tuple(Term(**row | {"method": Method(row["method"])}) for row in rows)
    )


def record(readings: DeviceReadings, *, tokens: int, memory) -> dict:
    """What a simulated run writes for this command, from the readings it
    installed and the run's `memory_readings` member `memory`."""
    pool = graph_pool.predicts(memory)
    return {
        "label": f"spec {readings.spec_digest} at TP{readings.tp_width}",
        "tokens": tokens,
        "readings": {name: _rows(readings.as_dict()[name]) for name in SUBTRACTED},
        "predicts": None if pool is None else _rows(pool),
        "assumed": memory.assumed,
    }


def _one(pattern: re.Pattern, text: str, log: Path) -> tuple[str, ...]:
    found = pattern.findall(text)
    if len(found) != 1:
        raise MemoryRefusal(
            f"{log} has {len(found)} lines matching {pattern.pattern!r}, and "
            "one rank of one real server logs one",
            "pass the log of a real server at one rank",
        )
    return found[0] if isinstance(found[0], tuple) else (found[0],)


def _gib(text: str) -> int:
    return round(float(text) * (1 << 30))


def recorded(log: Path) -> tuple[Recorded, Term | None]:
    """The real server's two readings, and the pool its capture reserved.

    ATOM logs GiB to two places, so each recorded term is good to 5 MiB.
    The run is named by the log's path and its record by the log's digest.
    The pool is None when the server captured no graph.
    """
    data = log.read_bytes()
    text = data.decode(errors="replace")
    peak_torch, non_torch = _one(_BUDGET, text, log)
    (tokens,) = _one(_WARMUP, text, log)
    run = Recorded(
        run=str(log),
        shape=Shape(int(tokens), PHASE),
        # `ModelRunner.warmup_model` resets the peak before its forward.
        high_water_reset=True,
        terms=footprint_terms(
            {"peak_torch": _gib(peak_torch), "non_torch": _gib(non_torch)},
            source="the Memory budget line",
        ),
        at_shape=SHAPED,
        record="sha256:" + hashlib.sha256(data).hexdigest(),
    )
    pools = _POOL.findall(text)
    if not pools:
        return run, None
    pool = Term(
        "pool(reserved)",
        _gib(pools[0]),
        Method.MEASURED,
        "the CUDA graph capture memory line",
    )
    return run, pool


def parts(predicted_file: Path, log: Path) -> list:
    """The per-term comparison, the predicted readings, and the graph pool's."""
    if not (saved := json.loads(predicted_file.read_text()).get("memory")):
        what = f"{predicted_file} has no memory section"
        raise MemoryRefusal(what, "pass an engine's hand-in, lp-engine*.json")
    readings = {name: _reading(name, rows) for name, rows in saved["readings"].items()}
    predicted = Predicted(
        saved["label"],
        Shape(saved["tokens"], PHASE),
        tuple(readings[name].as_term() for name in COMPARED),
        at_shape=SHAPED,
    )
    run, pool = recorded(log)
    out = [compare(predicted, run)]
    out += [readings[name] for name in COMPARED]
    if pool is None or saved["predicts"] is None:
        out.append(
            "graph pool not compared: the simulated run captures "
            f"{'no' if saved['predicts'] is None else 'a'} graph and the real "
            f"run {'no' if pool is None else 'a'} graph"
        )
    else:
        reserves = readings[graph_pool.RESERVES]
        predicts = _reading(graph_pool.PREDICTS, saved["predicts"])
        out.append(compare_graph_pool(pool, reserves=reserves, predicts=predicts))
    return out


def check(predicted_file: Path, log: Path) -> str:
    """The per-term table, the predicted readings' own terms, and the graph pool."""
    out = parts(predicted_file, log)
    return "\n".join(p if isinstance(p, str) else p.table() for p in out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.memory.check",
        description="A simulated run's non-KV memory against a real run's, per term.",
    )
    parser.add_argument("predicted", type=Path, help="a simulated engine's hand-in")
    parser.add_argument("log", type=Path, help="the real server's log")
    args = parser.parse_args(argv)
    print(check(args.predicted, args.log))


if __name__ == "__main__":
    main()
