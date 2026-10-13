# SPDX-License-Identifier: MIT
"""`tools/compass/step_fit_plan.py`: step fit's grid points planned from batch
shapes, and steps checked for coverage against a published bundle.

Steps are ATOM's `Scheduler`'s, recorded through the parity step record by
`test_parity_record._drive` at a 64-token budget, with the simulated runner
replaying decode on the ladder [1, 2, 4, 8]. Bundles are published to an
`ArtifactStore` and loaded by `run.step_fit.load` under `test_run_step_fit`'s
deployment, at the same scheduler's bounds and ladder.
"""

import importlib.util
import json
from pathlib import Path

import pytest
from test_backend_step_fit import fitted, linear, point
from test_memory_readings import DOCUMENT
from test_parity_record import _drive
from test_run_step_fit import KEY, deployment, publish

from atom.compass.runner import overrides

TOOL = Path(__file__).resolve().parents[2] / "tools" / "compass" / "step_fit_plan.py"
_spec = importlib.util.spec_from_file_location("step_fit_plan", TOOL)
tool = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tool)

DEFINITION = {
    "prefill": {"B": [1, 2], "T": [8, 16, 32, 64], "c": [0, 32, 64, 128, 192]},
    "decode": {"c": [0, 64, 128, 192, 256]},
    "margin": 0,
    "max_num_seqs": 8,
    "max_num_batched_tokens": 64,
    "max_model_len": 2048,
    "kv_cache_block_size": 16,
    "num_kvcache_blocks": 4096,
    "capture_sizes": [1, 2, 4, 8],
}
LADDER = DEFINITION["capture_sizes"]
#: Prompt lengths. Their decode steps replay rung 4: MIXED's at B = 3 on
#: cached context 90 to 92, SHORT's at B = 4 on 42.5 to 44.5.
MIXED, SHORT = (200, 20, 50), (40, 30, 90, 10)


def record(tmp_path, monkeypatch, *lengths, **drive):
    """One run of prompts of these lengths, each its own, as its step record;
    `drive` goes to `_drive`."""
    directory = tmp_path / "-".join(map(str, lengths))
    with monkeypatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        _drive(
            directory,
            patch,
            {0: range(len(lengths))},
            prompt=lambda i: list(range(5 + 1000 * i, 5 + 1000 * i + lengths[i])),
            **drive,
        )
    return directory


def run(capsys, *argv):
    code = tool.main([str(a) for a in argv])
    return code, json.loads(capsys.readouterr().out)


def plan(tmp_path, capsys, *records, **changes):
    path = tmp_path / "definition.json"
    path.write_text(json.dumps({**DEFINITION, **changes}))
    code, out = run(capsys, "plan", path, *records)
    assert code == 0
    return out


def measured(planned, residual=None, drop=None):
    """A bundle of the plan's grid, each point but `drop` timed, `drop` being
    a (list, point) pair."""
    grid = {"graph_mode": planned["graph_mode"], "scope": planned["scope"]}
    for phase, lists in (("prefill", ("output", "no_output")), ("decode", ("points",))):
        grid[phase] = {"nodes": planned[phase]["nodes"]}
        for name in lists:
            grid[phase][name] = [
                point(*at, time=100 + sum(at))
                for at in planned[phase][name]
                if (name, at) != drop
            ]
    return {
        "grid": {**grid, "unit": "us"},
        "residual": residual or fitted({"unit": linear()}, {"unit": 1.0}),
        "bounds": planned["bounds"],
    }


def coverage(tmp_path, capsys, member, *records, grid=KEY["grid"], **config):
    """`coverage` of `records` against `member`, published under the deployment
    at the scheduler's bounds and ladder, or `config`, and named by grid `grid`."""
    bounds = {k: DEFINITION[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(tmp_path, **{**bounds, **config}, capture_sizes=LADDER)
    publish(tmp_path, r, member=member, grid=grid)
    files = tmp_path / "run.json", tmp_path / "atom_config.json"
    spec = {"store": str(tmp_path / "store"), **KEY, "grid": grid}
    files[0].write_text(json.dumps({"step_fit": spec, "machine": DOCUMENT}))
    c = r.config
    fields = {name: getattr(c, name, None) for name in tool.CONFIG}
    fields["data_parallel_size"] = c.parallel_config.data_parallel_size
    fields["quantization"] = c.quant_config.compute_hash()
    files[1].write_text(json.dumps(fields))
    return run(capsys, "coverage", *files, *records)


def test_planned_points_measured_and_published_cover_every_step_and_name_a_dropped_one(
    tmp_path, monkeypatch, capsys
):
    records = [record(tmp_path, monkeypatch, *n) for n in (MIXED, SHORT)]
    planned = plan(tmp_path, capsys, *records)
    assert planned["infeasible"] == planned["unplaced"] == []

    code, out = coverage(tmp_path, capsys, measured(planned), *records)

    assert code == 0
    assert out == {
        "steps": 15,
        "covered": 15,
        "share": 1.0,
        "missing": {},
        "refused": [],
        "prefill_above_top_c": 0,
    }
    for i, (name, group) in enumerate(
        [("output", "prefill output"), ("no_output", "prefill no_output")]
        + [("points", "decode rung 4")]
    ):
        phase = "decode" if name == "points" else "prefill"
        at = planned[phase][name][0]
        code, out = coverage(
            tmp_path,
            capsys,
            measured(planned, drop=(name, at)),
            *records,
            grid=f"minus-{i}",
        )
        assert code == 1
        assert out["missing"] == {group: [at[1:] if phase == "decode" else at]}
        assert out["refused"] == [] and out["covered"] < 15


def test_a_step_outside_the_residuals_hull_is_refused_and_fails_coverage(
    tmp_path, monkeypatch, capsys
):
    records = [record(tmp_path, monkeypatch, *MIXED)]
    planned = plan(tmp_path, capsys, *records)
    # The default box, its first facet tightened to tokens / 2^10 <= 32 / 2^10.
    box = fitted({"unit": linear()}, {"unit": 1.0})["hull"]["facets"]
    box[0][-1] = -32 / 2**10
    hull = fitted({"unit": linear()}, {"unit": 1.0}, hull={"facets": box})

    code, out = coverage(tmp_path, capsys, measured(planned, residual=hull), *records)

    assert code == 1 and out["missing"] == {}
    # Every step but the final 8 + 20 token chunk and the decode steps.
    refused = {r["step"].rsplit(" ", 1)[1]: r["reasons"] for r in out["refused"]}
    assert sorted(refused, key=int) == ["0", "1", "2", "4"]
    assert all(
        reasons[0].startswith("the step at tokens=")
        and "outside the residual's hull" in reasons[0]
        for reasons in refused.values()
    )
    assert out["covered"] == out["steps"] - 4


def test_prefill_is_planned_by_whether_the_chunk_produces_output(
    tmp_path, monkeypatch, capsys
):
    planned = plan(tmp_path, capsys, record(tmp_path, monkeypatch, *MIXED))

    # The three middle chunks of the 200-token prompt, on nodes.
    assert planned["prefill"]["no_output"] == [[1, 64, 0], [1, 64, 64], [1, 64, 128]]
    # 8 + 20 tokens over 2 rows is (2, 28, 393/7); 50 tokens is (1, 50, 0).
    assert planned["prefill"]["output"] == [
        [1, 32, 0],
        [1, 64, 0],
        [2, 16, 32],
        [2, 16, 64],
        [2, 32, 32],
        [2, 32, 64],
    ]
    assert planned["counts"] == {
        "decode rung 4": 2,
        "prefill no_output": 3,
        "prefill output": 6,
    }


def test_decode_is_planned_on_the_targets_ladder_and_the_margin_widens_every_axis(
    tmp_path, monkeypatch, capsys
):
    records = [record(tmp_path, monkeypatch, *MIXED)]
    # 16 is past max_num_seqs, so not on the ladder the scheduler can produce.
    narrow = plan(tmp_path, capsys, *records, capture_sizes=LADDER + [16])
    wide = plan(tmp_path, capsys, *records, margin=1)

    assert narrow["decode"]["nodes"]["B"] == {
        "1": [1],
        "2": [2],
        "4": [3, 4],
        "8": [5, 8],
    }
    # B = 3 on rung 4, c 90 to 92 between the nodes 64 and 128.
    assert narrow["decode"]["points"] == [[4, 3, 64], [4, 3, 128]]
    # One node outward on B and c, and the rungs either side.
    assert wide["decode"]["points"] == [
        [rung, b, c]
        for rung, bs in ((2, [2]), (4, [3, 4]), (8, [5, 8]))
        for b in bs
        for c in (0, 64, 128, 192)
    ]
    assert [1, 64, 192] in wide["prefill"]["no_output"]


def test_a_rung_between_planned_rungs_is_planned_so_the_bundle_loads(
    tmp_path, monkeypatch, capsys
):
    """Two decode rows replay rung 2 and five replay rung 8; none replays 4.
    The five 8-token prompts prefill as one step at (5, 40, 0)."""
    records = [record(tmp_path, monkeypatch, *n) for n in ((200, 200), (8,) * 5)]
    prefill = {**DEFINITION["prefill"], "B": [1, 2, 5], "T": [8, 16, 32, 40, 64]}
    planned = plan(tmp_path, capsys, *records, prefill=prefill)

    decode = planned["decode"]["points"]
    assert sorted({rung for rung, _, _ in decode}) == [2, 4, 8]
    c_nodes = sorted({c for _, _, c in decode})
    assert [p for p in decode if p[0] == 4] == [
        [4, b, c] for b in (3, 4) for c in c_nodes
    ]
    code, out = coverage(tmp_path, capsys, measured(planned), *records)
    assert code == 0 and out["share"] == 1.0


def test_a_rung_between_planned_rungs_is_planned_on_its_own_c_nodes(
    tmp_path, monkeypatch, capsys
):
    """The records above at a 30-block pool, whose KV limits put c nodes 239 on
    rung 2 and 47 on rung 8 alone; rung 4 is planned on its own nodes."""
    records = [record(tmp_path, monkeypatch, *n) for n in ((200, 200), (8,) * 5)]
    prefill = {**DEFINITION["prefill"], "B": [1, 2, 5], "T": [8, 16, 32, 40, 64]}
    planned = plan(tmp_path, capsys, *records, prefill=prefill, num_kvcache_blocks=30)

    rung4 = planned["decode"]["nodes"]["c"]["4"]
    assert {47, 239}.isdisjoint(rung4)
    assert {c for r, _, c in planned["decode"]["points"] if r == 4} <= set(rung4)
    code, out = coverage(tmp_path, capsys, measured(planned), *records)
    assert code == 0 and out["share"] == 1.0


def test_the_margin_reads_a_neighbouring_rungs_own_c_nodes(
    tmp_path, monkeypatch, capsys
):
    """The step's cell on rung 4 is c 64..128. At a 12-block pool rung 2's c
    nodes are 0, 64, 95 (its KV limit), 128, ..., so one node of margin spans
    0 to 192 there, and 128 and 192 do not fit."""
    directory = record(tmp_path, monkeypatch, *MIXED)
    planned = plan(tmp_path, capsys, directory, margin=1, num_kvcache_blocks=12)

    assert [p for p in planned["decode"]["points"] if p[0] == 2] == [
        [2, 2, 0],
        [2, 2, 64],
        [2, 2, 95],
    ]


# The token budget, max_num_seqs and max_model_len are nodes, so no corner
# passes them; the KV pool bounds B and c together, and a corner off N_Q's
# integers is no homogeneous batch.
@pytest.mark.parametrize(
    "changes,grid,dropped,reason,kept",
    [
        (
            {"num_kvcache_blocks": 12},
            "prefill no_output",
            [2, 64, 128],
            "20 KV blocks are above num_kvcache_blocks",
            [2, 64, 64],
        ),
        # 3 decode rows of 64 cached + 1 tokens hold 5 blocks of 16 each;
        # counting 65 // 16 = 4 a row would fit the 12. 63 is c_max(3).
        (
            {"num_kvcache_blocks": 12},
            "decode rung 4",
            [4, 3, 64],
            "15 KV blocks are above num_kvcache_blocks",
            [4, 3, 63],
        ),
        (
            {"prefill": {**DEFINITION["prefill"], "B": [1, 3]}},
            "prefill output",
            [3, 16, 32],
            "N_Q = T/B = 16/3 is not an integer",
            [1, 16, 32],
        ),
        (
            {"prefill": {**DEFINITION["prefill"], "T": [1, 64]}},
            "prefill output",
            [2, 1, 32],
            "T 1 is below B 2, under one token a row",
            [2, 64, 32],
        ),
    ],
    ids=["kv-blocks", "decode", "non-integer-n-q", "below-one-token"],
)
def test_a_point_the_target_cannot_run_is_listed_and_not_planned(
    tmp_path, monkeypatch, capsys, changes, grid, dropped, reason, kept
):
    planned = plan(
        tmp_path, capsys, record(tmp_path, monkeypatch, *MIXED), margin=1, **changes
    )

    phase, key = ("decode", "points") if "rung" in grid else grid.split(" ")
    points = planned[phase][key]
    assert {"grid": grid, "at": dropped, "reason": reason} in planned["infeasible"]
    assert dropped not in points and kept in points


def test_the_limits_are_nodes_and_each_c_range_ends_on_one(
    tmp_path, monkeypatch, capsys
):
    """Nodes past a limit are dropped and the limit added: the token budget on
    T, max_num_seqs on B, max_model_len less the budget on prefill c and
    max_model_len - 1 on decode c; and, per rung, each B node's KV limit
    16 * (200 // B) - 1 on c, where it is below that."""
    target = {**DEFINITION, "num_kvcache_blocks": 200}
    planned = plan(
        tmp_path,
        capsys,
        record(tmp_path, monkeypatch, *MIXED),
        prefill={"B": [1, 2, 16], "T": [8, 64, 128], "c": [0, 64, 192, 4096]},
        decode={"c": [0, 64, 128, 4096]},
        num_kvcache_blocks=200,
    )

    nodes = planned["prefill"]["nodes"]
    assert nodes == {"B": [1, 2, 8], "T": [8, 64], "c": [0, 64, 192, 1984]}
    assert planned["decode"]["nodes"]["c"] == {
        "1": [0, 64, 128, 2047],
        "2": [0, 64, 128, 1599, 2047],
        "4": [0, 64, 128, 799, 1055, 2047],
        "8": [0, 64, 128, 399, 639, 2047],
    }
    assert planned["infeasible"] == []
    # Each top runs on the planner's own test, and one past it does not.
    b, t, c = (nodes[axis][-1] for axis in ("B", "T", "c"))
    assert not tool._infeasible(target, b, b, 0, prefill=True)
    assert tool._infeasible(target, b + 1, b + 1, 0, prefill=True)
    assert not tool._infeasible(target, 1, t, c, prefill=True)
    assert tool._infeasible(target, 1, t + 1, 0, prefill=True)
    assert tool._infeasible(target, 1, t, c + 1, prefill=True)
    for rung, bs in planned["decode"]["nodes"]["B"].items():
        cs = planned["decode"]["nodes"]["c"][rung]
        for b in bs:
            runs = [c for c in cs if not tool._infeasible(target, b, b, c, False)]
            assert tool._infeasible(target, b, b, runs[-1] + 1, False)


def test_a_rung_gains_a_b_node_where_steps_reach_past_the_rungs_kv_limit(
    tmp_path, monkeypatch, capsys
):
    """Six 200-token prompts decode on rung 8 at B = 6 on 200 to 201 cached,
    past c_max(8) = 16 * (100 // 8) - 1 = 191 and below c_max(6) = 255."""
    directory = record(tmp_path, monkeypatch, *[200] * 6)
    planned = plan(tmp_path, capsys, directory, num_kvcache_blocks=100)

    decode = planned["decode"]
    assert decode["nodes"]["B"]["8"] == [5, 6, 8]
    assert decode["nodes"]["c"]["8"] == [0, 64, 128, 191, 192, 255, 256, 319, 2047]
    assert decode["points"] == [[8, 6, 192], [8, 6, 255]]
    assert planned["infeasible"] == []
    code, out = coverage(tmp_path, capsys, measured(planned), directory)
    assert code == 0 and out["share"] == 1.0


def test_coverage_counts_apart_the_prefill_steps_cached_past_the_top_c_node(
    tmp_path, monkeypatch, capsys
):
    """At a 48-token budget prefill's top c node is 2048 - 48 = 2000; a
    2040-token prompt's last chunk, 24 tokens on 2016 cached, is past it."""
    directory = record(tmp_path, monkeypatch, 2040, budget=48)
    planned = plan(tmp_path, capsys, directory, max_num_batched_tokens=48)
    last = f"{directory} dp0 step 42"
    reason = "c=2016 is outside the prefill output grid's nodes 0..2000"
    assert planned["unplaced"] == [{"step": last, "reason": reason}]

    code, out = coverage(
        tmp_path, capsys, measured(planned), directory, max_num_batched_tokens=48
    )

    assert code == 1
    assert out["prefill_above_top_c"] == 1
    assert out["refused"] == [{"step": last, "reasons": [reason]}]
    assert out["covered"] == out["steps"] - 1


def test_a_step_off_the_targets_ladder_is_listed_unplaced(
    tmp_path, monkeypatch, capsys
):
    directory = record(tmp_path, monkeypatch, *MIXED)
    planned = plan(tmp_path, capsys, directory, capture_sizes=[1, 2, 8])

    assert planned["unplaced"] == [
        {"step": f"{directory} dp0 step {n}", "reason": "no decode rung 4"}
        for n in (5, 6, 7)
    ]
    assert planned["decode"]["points"] == []


def test_the_plan_reads_shapes_never_timestamps(tmp_path, monkeypatch, capsys):
    directory = record(tmp_path, monkeypatch, *MIXED)
    timed = tmp_path / "timed"
    timed.mkdir()
    lines = [
        json.loads(s) for s in (directory / "dp0.jsonl").read_text().split("\n") if s
    ]
    for i, line in enumerate(lines):
        line.update(t_enter_ns=10**9 * i, t_exit_ns=10**9 * i + 7 ** (i % 5))
    (timed / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))

    assert plan(tmp_path, capsys, timed) == plan(tmp_path, capsys, directory)


@pytest.mark.parametrize(
    "changes,named",
    [({k: None}, f"states no {k}") for k in DEFINITION]
    + [({"margin": 0.5}, "0.5 is not a count")]
    + [
        (
            {"prefill": {"B": [1], "N_Q": [8], "c": [0]}},
            "prefill nodes are on B, N_Q, c, not B, T, c",
        )
    ],
    ids=list(DEFINITION) + ["fractional", "prefill-n-q"],
)
def test_every_grid_definition_input_is_required_and_a_count(
    tmp_path, monkeypatch, capsys, changes, named
):
    definition = {k: v for k, v in {**DEFINITION, **changes}.items() if v is not None}
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(definition))
    directory = record(tmp_path, monkeypatch, *MIXED)

    with pytest.raises(SystemExit, match=named):
        tool.main(["plan", str(path), str(directory)])


def test_a_record_with_no_steps_is_refused(tmp_path, capsys):
    path = tmp_path / "definition.json"
    path.write_text(json.dumps(DEFINITION))

    with pytest.raises(SystemExit, match="record no step"):
        tool.main(["plan", str(path), str(tmp_path)])
