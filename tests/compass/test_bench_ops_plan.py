# SPDX-License-Identifier: MIT
"""`compass measure ops --plan` and `--coverage`, with no device.

The bench is a stand-in whose kernels switch where a real one's tile or split
choice would: a GEMM at 100 tokens, an attention at 5000 tokens of context.
Graphs are built by `trace ops`'s own folding; points are published to and
read back from a real store.
"""

import json
from types import SimpleNamespace

import pytest
import torch
from test_bench_ops import KEY, TREE, _graph, _leaf, _t

from atom.compass import ir
from atom.compass.artifacts import ArtifactStore
from atom.compass.bench import ops
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED

CONFIG = SimpleNamespace(
    capture_sizes=[1, 2, 4],
    max_num_seqs=4,
    max_num_batched_tokens=8192,
    speculative_config=None,
)


def _graph_at(store, config, shape):
    rows = shape.tokens if shape.phase == "prefill" else shape.seqs
    attn = _leaf(
        "aiter.attn",
        ("q", _t([rows, 8], "activation")),
        ("layer", ir.ContextRef(KEY)),
        key=KEY,
    )
    gemm = _leaf("aiter.gemm", ("x", _t([rows, 8], "activation")))
    return _graph(attn, gemm, layers=2)


def _bench(fn, config, todo, llc):
    """`measure` as a device would time each point: the GEMM by its tokens, the
    attention by its context, each kernel switching at its threshold."""
    assert fn is ops.measure
    _bench.calls += 1
    got = []
    for shape, context, found in todo:
        for sig, (op, _) in found.items():
            if shape.phase == "prefill":
                n = shape.tokens
                kernel = "p" if op["op"] == "aiter.attn" else f"gemm{int(n >= 100)}"
            elif op["op"] == "aiter.attn":
                n, kernel = context, f"attn{int(context >= 5000)}"
            else:
                n, kernel = 1, "gemm"
            got.append(
                {"shape": str(shape), "context": context, "op": op["op"], "leaf": sig}
                | {"kernels": [kernel], "cold_ns": float(n)}
            )
    return got


def _records(path, *steps):
    """A record directory of `steps`, each `(rows, rung)`."""
    (path / "steps").mkdir(parents=True)
    (path / "run.json").write_text("{}")
    lines = [
        {"step": i, "is_dummy_run": False, "rows": rows, "rung": rung}
        for i, (rows, rung) in enumerate(steps)
    ]
    (path / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(line) + "\n" for line in lines)
    )
    return str(path)


@pytest.fixture
def bench(tmp_path, monkeypatch):
    """A store, the key of a point measured on hjbog-srdc-39, the run, and
    the stand-in bench and graphs in place."""
    spec = MachineSpec.from_mapping(
        json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
    )
    run = {
        "device": [{"index": 6, "name": spec.value("device.name")}],
        "stack": {c: spec.value(f"device.software_pinned_to.{c}") for c in PINNED},
        "ptl": "disabled",
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": "a" * 40},
    }
    run["device"][0]["arch"] = spec.value("device.arch")
    layout = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
    layout |= {"enable_expert_parallel": False, "enable_dp_attention": False}
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: layout)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})
    monkeypatch.setattr(ops, "graph", _graph_at)
    monkeypatch.setattr(ops, "_child", _bench)
    _bench.calls = 0
    root = str(tmp_path / "store")
    return ArtifactStore(root), root, ops.keys(None, run), run


def _plan(bench, tmp_path, population):
    store, root, key, run = bench
    args = SimpleNamespace(plan=[population], store=root, llc_mib=256)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    ops.plan(store, key, CONFIG, run, args, "compass measure ops --plan", out)
    return [json.loads(x) for x in (out / "plan.jsonl").read_text().splitlines()]


def test_a_plan_densifies_each_leaf_to_adjacent_points_where_its_kernel_switches(
    bench, tmp_path
):
    population = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
        ([[30, 30, False], [30, 30, False]], None),
    )

    records = _plan(bench, tmp_path, population)

    assert ops._dependent(records) == {"aiter.attn": True, "aiter.gemm": False}
    at = {}
    for r in records:
        at.setdefault((r["op"], r["shape"].split()[0]), set()).add(
            r["context"] if r["shape"].startswith("decode") else r["shape"]
        )
    # The context-free GEMM keeps the ends of the steps' contexts at every rung.
    assert at["aiter.gemm", "decode"] == {2000, 12000}
    # Attention takes the ladder of means and maxes, then halves to 4999 | 5000.
    attn = sorted(at["aiter.attn", "decode"])
    assert attn[:2] == [2000, 3000] and attn[-1] == 12000
    assert {4999, 5000} <= set(attn) and len(attn) < 20
    # GEMM rows over every B: 40, 60, 150, then halves down to 99 | 100.
    tokens = sorted(
        int(s.split("T=")[1].split()[0]) for s in at["aiter.gemm", "prefill"]
    )
    assert tokens[:2] == [40, 60] and tokens[-1] == 150
    assert {99, 100} <= set(tokens) and len(tokens) < 12
    # Attention depends on context, so its prefill points keep their rows.
    assert at["aiter.attn", "prefill"] == {
        "prefill T=40 B=1",
        "prefill T=150 B=1",
        "prefill T=60 B=2",
    }

    calls = _bench.calls
    assert _plan(bench, tmp_path, population) == records
    assert _bench.calls == calls  # every point is read back from the store


def test_coverage_counts_a_step_only_when_every_leaf_sits_in_one_kernel_s_band(
    bench, tmp_path
):
    plan = _records(
        tmp_path / "plan",
        ([[1, 1000, True], [1, 3000, True]], 2),
        ([[1, 12000, True]], 1),
        ([[40, 40, False]], None),
        ([[150, 150, False]], None),
    )
    records = _plan(bench, tmp_path, plan)
    graded = _records(
        tmp_path / "graded",
        ([[1, 3500, True], [1, 4500, True]], 2),  # inside attention's first band
        ([[1, 3000, True], [1, 6000, True]], 2),  # mean 4500, max across 5000
        ([[1, 20000, True]], 1),  # past the largest context measured
        ([[120, 120, False]], None),  # GEMM between 100 and 150
        ([[120, 1120, False]], None),  # a cached row
        ([[40, 40, False]] * 3, None),  # attention never measured at three rows
    )
    store, root, _, _ = bench

    report = ops.coverage(store, root, CONFIG, [graded], records)

    assert (report["steps"], report["covered"]) == (6, 2)
    assert report["by phase"] == {
        "decode": {"steps": 3, "covered": 1},
        "prefill": {"steps": 3, "covered": 1},
    }
    assert report["left out, by leaf kind"] == {
        "aiter.attn": {
            "between two kernels": 1,
            "outside the measured range": 1,
            "a cached row, measured on fresh rows only": 1,
            "no measured point": 1,
        }
    }


@pytest.mark.parametrize(
    "lo, hi, why",
    [
        (10, 10, ""),  # on a point
        (12, 18, ""),  # between two of one kernel
        (18, 22, "between two kernels"),
        (5, 10, "outside the measured range"),
        (30, 31, "outside the measured range"),
        (30, 30, "between two kernels"),  # two points there disagree
    ],
)
def test_a_bracket_holds_only_between_points_of_one_kernel(lo, hi, why):
    ats, kernels = [10, 20, 25, 30, 30], ["a", "a", "b", "b", "c"]

    assert ops._bracket(ats, kernels, lo, hi) == why


@pytest.mark.parametrize(
    "rows, rung, lo",
    [
        ([4000] * 5, 8, 2500),  # three rows of padding: the rung's mean
        ([6, 10, 6], 3, 6),  # rows 0 and 2 against row 1: 12 over two
        ([1000, 3000], 2, 2000),  # the mean
        ([3], 8, 1),  # at least a token
    ],
)
def test_a_decode_step_s_lowest_context_is_over_the_rung_s_rows(rows, rung, lo):
    line = {"rows": [[1, c, True] for c in rows]}

    assert ops._rows(line, rung) == (lo, max(rows))


def test_a_ladder_keeps_values_a_step_apart_and_the_largest():
    assert ops._ladder([16384, 1, 2, 3, 4, 5, 100, 101, 16383]) == [
        1,
        2,
        3,
        4,
        5,
        100,
        16383,
        16384,
    ]


def test_a_kind_depends_on_context_when_its_times_or_its_kernels_move():
    def point(op, context, ns, kernel):
        record = {"shape": "decode B=2", "context": context, "op": op, "leaf": op}
        return record | {"cold_ns": ns, "kernels": [kernel]}

    records = [
        *(point("state", c, ns, "s") for c, ns in ((1, 10.0), (9, 20.0))),
        *(point("attn", c, 10.0, k) for c, k in ((1, "a"), (9, "b"))),
        *(point("gemm", c, ns, "g") for c, ns in ((1, 10.0), (9, 10.5))),
        point("once", 1, 1.0, "o"),
    ]

    assert ops._dependent(records) == {"state": True, "attn": True, "gemm": False}


def test_an_integer_activation_indexes_a_dimension_of_one():
    spec = {"shape": [64], "dtype": "torch.int32", "stride": [1]}

    index = ops._random(spec, "cpu")

    assert torch.zeros(1, 3)[index].shape == (64, 3)
