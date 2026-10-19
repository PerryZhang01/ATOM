# SPDX-License-Identifier: MIT
"""`compass fit ops`: a price_list from measured points, and what it declines.

Points are published by `measure ops`'s own `publish` into the store the fit
reads, beside the run.json and engine.json `measure ops` writes.
"""

import json
import pathlib

import pytest

from atom.compass import cli
from atom.compass.artifacts import ArtifactRefusal, ArtifactStore, GateState, Key, Kind
from atom.compass.bench import ops as bench
from atom.compass.fit import ops, publish
from atom.compass.spec import MachineSpec
from atom.compass.spec.fields import PINNED

TREE = pathlib.Path(__file__).resolve().parents[2]
SPEC = MachineSpec.from_mapping(
    json.loads((TREE / "scripts/compass/machines/mi308x-hjbog39.json").read_text())
)
LAYOUT = {"dp": 1, "pp": 1, "pcp": 1, "tp": 1, "decode_context_parallel_size": 1}
LAYOUT |= {"enable_expert_parallel": False, "enable_dp_attention": False}
BF16 = "torch.bfloat16"


def _leaf(m, n, cu=False):
    """A GEMM's signature at `m` rows into `n` outputs, as `measure ops` keys it;
    with `cu`, also an `m + 1` offsets vector, as attention's signatures hold."""
    x = {"shape": [m, 64], "stride": [64, 1], "dtype": BF16, "role": "activation"}
    w = {"shape": [n, 64], "stride": [64, 1], "dtype": BF16, "role": "weight"}
    sig = {"op": "aiter.gemm", "args": {"x": x, "w": w}, "out": [[m, n]]}
    if cu:
        sig["args"]["cu"] = {"shape": [m + 1], "dtype": "torch.int32"}
    return json.dumps(sig, sort_keys=True, separators=(",", ":"))


def _point(b, ns, kernel="k", n=128, context=1024, cu=False):
    return {
        "shape": f"decode B={b}",
        "context": context,
        "op": "aiter.gemm",
        "leaf": _leaf(b, n, cu),
        "cold_ns": ns,
        "kernels": [kernel],
    }


def _run(ptl="disabled", root="a" * 40):
    stack = {c: SPEC.value(f"device.software_pinned_to.{c}") for c in PINNED}
    card = {"index": 6, "name": SPEC.value("device.name")}
    return {
        "device": [card | {"arch": SPEC.value("device.arch")}],
        "stack": stack,
        "ptl": ptl,
        "host_key": {"model": "sha256:" + "0" * 64, "host_cpu": "cpu"},
        "source": {"revision_kind": "tree", "revision": root},
    }


@pytest.fixture
def measured(tmp_path, monkeypatch):
    """A `measure ops --out` directory of `records`, published to the store."""
    monkeypatch.setattr("atom.compass.run.memory.key_fields", lambda c: LAYOUT)
    monkeypatch.setattr("atom.compass.run.step_grid.keyed", lambda c: {"level": 0})

    def make(name, records, **run):
        run, out = _run(**run), tmp_path / name
        out.mkdir()
        (out / "run.json").write_text(json.dumps(run))
        (out / "engine.json").write_text(json.dumps({"level": 0}))
        store = ArtifactStore(str(tmp_path / "store"))
        key = bench.keys(None, run)
        bench.publish(store, key, None, run, "compass measure ops", records, out)
        return out, key

    return make


def _families(points):
    got = ops.fit(points)["families"].values()
    return sorted((f["dims"], [p[:2] for p in f["points"]]) for f in got)


def test_a_family_masks_the_rows_and_keeps_the_weight():
    points = [_point(2, 10.0), _point(8, 16.0), _point(2, 20.0, n=256)]
    points += [_point(8, 26.0, n=256), _point(2, 30.0, cu=True)]
    points += [_point(8, 36.0, cu=True)]

    assert _families(points) == [
        (["B", "context"], [[[2, 1024], 10.0], [[8, 1024], 16.0]]),
        (["B", "context"], [[[2, 1024], 20.0], [[8, 1024], 26.0]]),
        (["B", "context"], [[[2, 1024], 30.0], [[8, 1024], 36.0]]),
    ]


def test_two_leaves_each_measured_at_one_width_are_two_families():
    # N differs between the two, and N is no count of either shape
    points = [_point(2, 10.0, n=128), _point(8, 26.0, n=256)]

    assert _families(points) == [
        (["B", "context"], [[[2, 1024], 10.0]]),
        (["B", "context"], [[[8, 1024], 26.0]]),
    ]


LINE = {(2,): (10.0, ["a"]), (8,): (16.0, ["a"]), (32,): (40.0, ["b"])}


@pytest.mark.parametrize("at, ns", [(2, 10.0), (4, 12.0), (8, 16.0), (32, 40.0)])
def test_a_query_takes_a_measured_price_or_interpolates_one_kernel(at, ns):
    assert ops.interpolate(LINE, (at,), ["B"]) == (ns, ["b" if at == 32 else "a"])


@pytest.mark.parametrize(
    "at, why",
    [
        (1, "B = 1 is outside the measured 2 to 32"),
        (33, "B = 33 is outside the measured 2 to 32"),
        (16, "B = 16 lies between 8 and 32, which launched different kernels: a and b"),
    ],
)
def test_a_query_outside_the_hull_or_between_kernels_declines_with_its_range(at, why):
    with pytest.raises(ops.Declined, match=f"^{why}$"):
        ops.interpolate(LINE, (at,), ["B"])


def test_a_grid_interpolates_multilinearly_and_declines_a_missing_corner():
    grid = {(2, 1024): (10.0, ["a"]), (8, 1024): (16.0, ["a"])}
    grid |= {(2, 2048): (20.0, ["a"]), (8, 2048): (32.0, ["a"])}
    dims = ["B", "context"]

    assert ops.interpolate(grid, (4, 1536), dims) == (18.0, ["a"])
    del grid[8, 2048]
    with pytest.raises(ops.Declined, match="^at B = 8, context = 1536 is outside"):
        ops.interpolate(grid, (4, 1536), dims)


def _fit(out, store, *dirs):
    cli.main(["fit", "ops", *map(str, dirs), "--store", str(store), "--out", str(out)])
    return json.loads((out / "price_list.json").read_text())


def test_fit_ops_publishes_its_points_key_and_their_held_out_error(measured, tmp_path):
    # 4 sits inside one band; 12 launched x between two k neighbours
    points = [_point(2, 10.0), _point(4, 12.5), _point(8, 16.0)]
    points += [_point(12, 9.0, kernel="x"), _point(16, 24.0), _point(32, 50.0, "b")]
    d, key = measured("m", points + [_point(4, 1.0, n=512) | {"refused": "no"}])

    got = _fit(tmp_path / "fit", tmp_path / "store", d)

    fields = key("decode B=2", 1024, _leaf(2, 128)).values
    assert got["key"] == {k: v for k, v in fields.items() if k != "point"}
    assert got["report"]["by_op"] == {
        "decode aiter.gemm": {"held_out": 4, "declined": 2, "on_declined_lines": 4}
    }
    declined = [r["declined"] for r in got["report"]["held_out"] if "declined" in r]
    assert declined == [
        "B = 8 lies between 4 and 12, which launched different kernels: k and x",
        "B = 16 lies between 12 and 32, which launched different kernels: x and b",
    ]
    (f,) = got["params"]["families"].values()
    assert f["declined"] == [
        ["B", [1024], "B = 12 launched other kernels than its neighbours"]
    ]


def test_a_line_holding_an_error_over_5pct_either_way_is_declined_with_its_reason():
    # held out along B, B = 4 reads -4% at context 1024 and -20% at 2048;
    # along context at B = 2, context 2048 reads +25%
    points = [_point(2, 10.0), _point(4, 12.5), _point(8, 16.0)]
    points += [_point(b, ns, context=2048) for b, ns in ((2, 20), (4, 30), (8, 32))]
    points += [_point(2, 40.0, context=3072)]
    params = ops.fit(points)
    rows = ops.held_out(params)
    (f,) = params["families"].values()

    assert f["declined"] == [
        ["context", [2], "context = 2048 reads +25.0% off its neighbours"],
        ["B", [2048], "B = 4 reads -20.0% off its neighbours"],
    ]
    assert ops.summary(rows)["decode aiter.gemm"]["max_error"] == pytest.approx(0.04)


@pytest.mark.parametrize(
    "moved, cell",
    [({"ptl": "disabled"}, "PTL"), ({"source_root": "tree-" + "b" * 40}, "ATOM src")],
)
def test_a_price_list_refuses_at_load_on_another_ptl_state_or_source_root(
    measured, tmp_path, moved, cell
):
    d, _ = measured("m", [_point(2, 10.0), _point(8, 16.0)], ptl="enabled")
    got = _fit(tmp_path / "fit", tmp_path / "store", d)
    store = ArtifactStore(str(tmp_path / "store"))
    key = Key.of(Kind.PRICE_LIST, **got["key"])
    _, config, _ = ops.read(store, [d])
    gates = GateState.of()

    assert store.load(key, conditions=publish.conditions(config), gates=gates)
    with pytest.raises(ArtifactRefusal, match=f"`price_list` x `{cell}`: recorded"):
        store.load(key, conditions=publish.conditions(config | moved), gates=gates)


def test_points_naming_two_price_lists_refuse(measured, tmp_path):
    a, _ = measured("a", [_point(2, 10.0)])
    b, _ = measured("b", [_point(8, 16.0)], root="b" * 40)

    with pytest.raises(SystemExit, match="^compass fit ops: the points name 2 price"):
        _fit(tmp_path / "fit", tmp_path / "store", a, b)
    assert not (tmp_path / "store" / "price_list").exists()
