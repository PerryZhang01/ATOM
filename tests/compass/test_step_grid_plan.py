# SPDX-License-Identifier: MIT
"""`atom/compass/backends/step_grid_plan.py`: the step grid's points planned
from batch shapes, and steps checked for coverage against a loaded backend.

Steps are ATOM's `Scheduler`'s, recorded through the parity step record by
`test_parity_record._drive` at a 64-token budget, with the simulated runner
replaying decode on the ladder [1, 2, 4, 8], and planned for the engine file
each record's rank wrote. A plan's lines are written as a step record, read
back by `parity.read` and located by the plan's grid. Bundles are published to
an `ArtifactStore` and loaded by `run.step_grid.load` under
`test_run_step_grid`'s deployment, at the same scheduler's bounds and ladder.
"""

import copy
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pytest
import test_parity_record
from test_backend_step_grid import fitted, linear, point
from test_memory_readings import DOCUMENT
from test_parity_record import _drive
from test_run_step_grid import KEY, deployment, publish

from atom.compass import parity
from atom.compass.backends import BatchView, RequestShape, StepGrid, step_grid_plan
from atom.compass.run.step_grid import load
from atom.compass.runner import overrides
from atom.compass.spec import MachineSpec

DEFINITION = {
    "prefill": {"B": [1, 2], "T": [8, 16, 32, 64], "c": [0, 32, 64, 128, 192]},
    "decode": {"c": [0, 64, 128, 192, 256]},
    "margin": 0,
}
LADDER = test_parity_record.LADDER
#: The target fields of the engine `_drive`'s runner states.
ENGINE = {
    "max_num_seqs": 8,
    "max_num_batched_tokens": 64,
    "max_model_len": 2048,
    "kv_cache_block_size": 16,
    "num_kvcache_blocks": 4096,
    "ladder": LADDER,
}
#: Prompt lengths. Their decode steps replay rung 4: MIXED's at B = 3 on
#: cached context 90 to 92, SHORT's at B = 4 on 42.5 to 44.5.
MIXED, SHORT = (200, 20, 50), (40, 30, 90, 10)


def record(tmp_path, monkeypatch, *lengths, ladder=LADDER, **drive):
    """One run of prompts of these lengths, each its own, as its step record,
    by a runner whose capture sizes are `ladder`; `drive` goes to `_drive`."""
    directory = tmp_path / "-".join(map(str, lengths))
    with monkeypatch.context() as patch:
        patch.setattr(overrides, "_group_step_seconds", lambda *_: 0.0)
        patch.setattr(test_parity_record, "LADDER", ladder)
        _drive(
            directory,
            patch,
            {0: range(len(lengths))},
            prompt=lambda i: list(range(5 + 1000 * i, 5 + 1000 * i + lengths[i])),
            **drive,
        )
    return directory


def steps(*records):
    """Each record's lines, named, and the engine each of its ranks states."""
    named, engines = [], {}
    for directory in records:
        for dp, lines in parity.read(directory).items():
            rank = f"{directory} dp{dp}"
            engines[rank] = json.loads((directory / f"dp{dp}.engine.json").read_text())
            named += [(f"{rank} step {line['step']}", line) for line in lines]
    return named, engines


def plan(*records, engine=(), **changes):
    """The plan for `records`, the definition changed by `changes` and each
    rank's engine by `engine`."""
    named, engines = steps(*records)
    engines = {rank: {**e, **dict(engine)} for rank, e in engines.items()}
    return step_grid_plan.plan({**DEFINITION, **changes}, named, engines)


def located(tmp_path, planned):
    """The plan's lines, written as a step record and read back by
    `parity.read`, each at the node the plan's grid locates it on: prefill's
    `[B, T, c]` by grid, decode's `[rung, B, c]` under ``decode``.

    Each line is asserted a homogeneous batch whose cell is one corner, the
    node its own rows state."""
    directory = Path(tempfile.mkdtemp(prefix="plan-", dir=tmp_path))
    lines = planned["lines"]
    (directory / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    grid = StepGrid({**planned, "unit": "s"})
    out, counts = defaultdict(list), Counter()
    for line in parity.read(directory).get(0, []):
        rows, rung = line["rows"], line["rung"]
        (n_q, context, decode), b = rows[0], len(rows)
        assert line["origin"] == "plan" and all(row == rows[0] for row in rows)
        own = (b, context - 1) if decode else (b, b * n_q, context - n_q)
        view = BatchView(
            tuple(RequestShape(*row) for row in rows),
            capture_rung=rung,
            produces_output=line["produces_output"],
        )
        cells, at = grid.locate(view)
        assert cells.corners(view, at) == [(own, 1)]
        counts[cells.name] += 1
        out["decode" if decode else cells.name].append(
            [rung, *own] if decode else list(own)
        )
    assert counts == planned["counts"]
    return dict(out)


def measured(planned, points, residual=None, drop=None):
    """A bundle of the plan's grid, each located point but `drop` timed, `drop`
    being a (grid, point) pair as `located` names them."""
    keys = ("graph_mode", "scope", "prefill", "decode")
    grid = copy.deepcopy({k: planned[k] for k in keys})
    lists = {
        "prefill output": grid["prefill"]["output"],
        "prefill no_output": grid["prefill"]["no_output"],
        "decode": grid["decode"]["points"],
    }
    for name, ats in points.items():
        lists[name] += [
            point(*at, time=100 + sum(at)) for at in ats if (name, at) != drop
        ]
    return {
        "grid": {**grid, "unit": "us"},
        "residual": residual or fitted({"unit": linear()}, {"unit": 1.0}),
        "bounds": planned["bounds"],
    }


def coverage(tmp_path, member, *records, grid=KEY["grid"], **config):
    """`coverage` of `records` over `member`, published under the deployment at
    the scheduler's bounds and ladder, or `config`, named by grid `grid`, and
    loaded as a run loads it."""
    bounds = {k: ENGINE[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(tmp_path, **{**bounds, **config}, capture_sizes=LADDER)
    publish(tmp_path, r, member=member, grid=grid)
    spec = {"store": str(tmp_path / "store"), **KEY, "grid": grid}
    backend, _ = load(spec, r.config, MachineSpec.from_mapping(DOCUMENT))
    return step_grid_plan.coverage(backend, steps(*records)[0])


def test_every_planned_point_read_back_is_a_homogeneous_batch_on_its_own_node(
    tmp_path, monkeypatch
):
    records = [record(tmp_path, monkeypatch, *n) for n in (MIXED, SHORT)]
    for engine in steps(*records)[1].values():
        assert {k: engine[k] for k in ENGINE} == ENGINE
    planned = plan(*records, margin=1)

    points = located(tmp_path, planned)

    assert sum(map(len, points.values())) == len(planned["lines"]) > 0
    assert [line["step"] for line in planned["lines"]] == list(
        range(len(planned["lines"]))
    )
    assert {
        (line["dp_rank"], line["is_dummy_run"], line["origin"])
        for line in planned["lines"]
    } == {(0, False, "plan")}


def test_planned_points_measured_and_published_cover_every_step_and_name_a_dropped_one(
    tmp_path, monkeypatch
):
    records = [record(tmp_path, monkeypatch, *n) for n in (MIXED, SHORT)]
    planned = plan(*records)
    points = located(tmp_path, planned)
    assert planned["infeasible"] == planned["unplaced"] == []

    out = coverage(tmp_path, measured(planned, points), *records)

    assert out == {
        "steps": 15,
        "covered": 15,
        "share": 1.0,
        "missing": {},
        "refused": [],
        "prefill_above_top_c": 0,
    }
    for i, name in enumerate(["prefill output", "prefill no_output", "decode"]):
        at = points[name][0]
        out = coverage(
            tmp_path,
            measured(planned, points, drop=(name, at)),
            *records,
            grid=f"minus-{i}",
        )
        if name == "decode":
            name, at = f"decode rung {at[0]}", at[1:]
        assert out["missing"] == {name: [at]}
        assert out["refused"] == [] and out["covered"] < 15


def test_a_step_outside_the_residuals_hull_is_refused_and_fails_coverage(
    tmp_path, monkeypatch
):
    records = [record(tmp_path, monkeypatch, *MIXED)]
    planned = plan(*records)
    # The default box, its first facet tightened to tokens / 2^10 <= 32 / 2^10.
    box = fitted({"unit": linear()}, {"unit": 1.0})["hull"]["facets"]
    box[0][-1] = -32 / 2**10
    hull = fitted({"unit": linear()}, {"unit": 1.0}, hull={"facets": box})
    member = measured(planned, located(tmp_path, planned), residual=hull)

    out = coverage(tmp_path, member, *records)

    assert out["missing"] == {}
    # Every step but the final 8 + 20 token chunk and the decode steps.
    refused = {r["step"].rsplit(" ", 1)[1]: r["reasons"] for r in out["refused"]}
    assert sorted(refused, key=int) == ["0", "1", "2", "4"]
    assert all(
        reasons[0].startswith("the step at tokens=")
        and "outside the residual's hull" in reasons[0]
        for reasons in refused.values()
    )
    assert out["covered"] == out["steps"] - 4


def test_prefill_is_planned_by_whether_the_chunk_produces_output(tmp_path, monkeypatch):
    planned = plan(record(tmp_path, monkeypatch, *MIXED))
    points = located(tmp_path, planned)

    # The three middle chunks of the 200-token prompt, on nodes.
    assert points["prefill no_output"] == [[1, 64, 0], [1, 64, 64], [1, 64, 128]]
    # 8 + 20 tokens over 2 rows is (2, 28, 393/7); 50 tokens is (1, 50, 0).
    assert points["prefill output"] == [
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
    tmp_path, monkeypatch
):
    # 16 is past max_num_seqs, so not on the ladder the scheduler can produce,
    # which is the ladder the engine states.
    records = [record(tmp_path, monkeypatch, *MIXED, ladder=LADDER + [16])]
    narrow = plan(*records)
    wide = located(tmp_path, plan(*records, margin=1))

    assert narrow["decode"]["nodes"]["B"] == {
        "1": [1],
        "2": [2],
        "4": [3, 4],
        "8": [5, 8],
    }
    # B = 3 on rung 4, c 90 to 92 between the nodes 64 and 128.
    assert located(tmp_path, narrow)["decode"] == [[4, 3, 64], [4, 3, 128]]
    # One node outward on B and c, and the rungs either side.
    assert wide["decode"] == [
        [rung, b, c]
        for rung, bs in ((2, [2]), (4, [3, 4]), (8, [5, 8]))
        for b in bs
        for c in (0, 64, 128, 192)
    ]
    assert [1, 64, 192] in wide["prefill no_output"]


def test_a_rung_between_planned_rungs_is_planned_so_the_bundle_loads(
    tmp_path, monkeypatch
):
    """Two decode rows replay rung 2 and five replay rung 8; none replays 4.
    The five 8-token prompts prefill as one step at (5, 40, 0)."""
    records = [record(tmp_path, monkeypatch, *n) for n in ((200, 200), (8,) * 5)]
    prefill = {**DEFINITION["prefill"], "B": [1, 2, 5], "T": [8, 16, 32, 40, 64]}
    planned = plan(*records, prefill=prefill)
    points = located(tmp_path, planned)

    decode = points["decode"]
    assert sorted({rung for rung, _, _ in decode}) == [2, 4, 8]
    c_nodes = sorted({c for _, _, c in decode})
    assert [p for p in decode if p[0] == 4] == [
        [4, b, c] for b in (3, 4) for c in c_nodes
    ]
    out = coverage(tmp_path, measured(planned, points), *records)
    assert out["share"] == 1.0


def test_a_rung_between_planned_rungs_is_planned_on_its_own_c_nodes(
    tmp_path, monkeypatch
):
    """The records above at a 30-block pool, whose KV limits put c nodes 239 on
    rung 2 and 47 on rung 8 alone; rung 4 is planned on its own nodes."""
    records = [record(tmp_path, monkeypatch, *n) for n in ((200, 200), (8,) * 5)]
    prefill = {**DEFINITION["prefill"], "B": [1, 2, 5], "T": [8, 16, 32, 40, 64]}
    planned = plan(*records, prefill=prefill, engine={"num_kvcache_blocks": 30})
    points = located(tmp_path, planned)

    rung4 = planned["decode"]["nodes"]["c"]["4"]
    assert {47, 239}.isdisjoint(rung4)
    assert {c for r, _, c in points["decode"] if r == 4} <= set(rung4)
    out = coverage(tmp_path, measured(planned, points), *records)
    assert out["share"] == 1.0


def test_the_margin_reads_a_neighbouring_rungs_own_c_nodes(tmp_path, monkeypatch):
    """The step's cell on rung 4 is c 64..128. At a 12-block pool rung 2's c
    nodes are 0, 64, 95 (its KV limit), 128, ..., so one node of margin spans
    0 to 192 there, and 128 and 192 do not fit."""
    directory = record(tmp_path, monkeypatch, *MIXED)
    planned = plan(directory, margin=1, engine={"num_kvcache_blocks": 12})

    assert [p for p in located(tmp_path, planned)["decode"] if p[0] == 2] == [
        [2, 2, 0],
        [2, 2, 64],
        [2, 2, 95],
    ]


# The token budget, max_num_seqs and max_model_len are nodes, so no corner
# passes them; the KV pool bounds B and c together, and a corner off N_Q's
# integers is no homogeneous batch.
@pytest.mark.parametrize(
    "engine,changes,grid,dropped,reason,kept",
    [
        (
            {"num_kvcache_blocks": 12},
            {},
            "prefill no_output",
            [2, 64, 128],
            "20 KV blocks are above num_kvcache_blocks",
            [2, 64, 64],
        ),
        # 3 decode rows of 64 cached + 1 tokens hold 5 blocks of 16 each;
        # counting 65 // 16 = 4 a row would fit the 12. 63 is c_max(3).
        (
            {"num_kvcache_blocks": 12},
            {},
            "decode rung 4",
            [4, 3, 64],
            "15 KV blocks are above num_kvcache_blocks",
            [4, 3, 63],
        ),
        (
            {},
            {"prefill": {**DEFINITION["prefill"], "B": [1, 3]}},
            "prefill output",
            [3, 16, 32],
            "N_Q = T/B = 16/3 is not an integer",
            [1, 16, 32],
        ),
        (
            {},
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
    tmp_path, monkeypatch, engine, changes, grid, dropped, reason, kept
):
    directory = record(tmp_path, monkeypatch, *MIXED)
    planned = plan(directory, margin=1, engine=engine, **changes)

    points = located(tmp_path, planned)["decode" if "rung" in grid else grid]
    assert {"grid": grid, "at": dropped, "reason": reason} in planned["infeasible"]
    assert dropped not in points and kept in points


def test_the_limits_are_nodes_and_each_c_range_ends_on_one(tmp_path, monkeypatch):
    """Nodes past a limit are dropped and the limit added: the token budget on
    T, max_num_seqs on B, max_model_len less the budget on prefill c and
    max_model_len - 1 on decode c; and, per rung, each B node's KV limit
    16 * (200 // B) - 1 on c, where it is below that."""
    target = {**ENGINE, "num_kvcache_blocks": 200}
    planned = plan(
        record(tmp_path, monkeypatch, *MIXED),
        prefill={"B": [1, 2, 16], "T": [8, 64, 128], "c": [0, 64, 192, 4096]},
        decode={"c": [0, 64, 128, 4096]},
        engine={"num_kvcache_blocks": 200},
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
    infeasible = step_grid_plan._infeasible
    b, t, c = (nodes[axis][-1] for axis in ("B", "T", "c"))
    assert not infeasible(target, b, b, 0, prefill=True)
    assert infeasible(target, b + 1, b + 1, 0, prefill=True)
    assert not infeasible(target, 1, t, c, prefill=True)
    assert infeasible(target, 1, t + 1, 0, prefill=True)
    assert infeasible(target, 1, t, c + 1, prefill=True)
    for rung, bs in planned["decode"]["nodes"]["B"].items():
        cs = planned["decode"]["nodes"]["c"][rung]
        for b in bs:
            runs = [c for c in cs if not infeasible(target, b, b, c, False)]
            assert infeasible(target, b, b, runs[-1] + 1, False)


def test_a_rung_gains_a_b_node_where_steps_reach_past_the_rungs_kv_limit(
    tmp_path, monkeypatch
):
    """Six 200-token prompts decode on rung 8 at B = 6 on 200 to 201 cached,
    past c_max(8) = 16 * (100 // 8) - 1 = 191 and below c_max(6) = 255."""
    directory = record(tmp_path, monkeypatch, *[200] * 6)
    planned = plan(directory, engine={"num_kvcache_blocks": 100})
    points = located(tmp_path, planned)

    decode = planned["decode"]
    assert decode["nodes"]["B"]["8"] == [5, 6, 8]
    assert decode["nodes"]["c"]["8"] == [0, 64, 128, 191, 192, 255, 256, 319, 2047]
    assert points["decode"] == [[8, 6, 192], [8, 6, 255]]
    assert planned["infeasible"] == []
    out = coverage(tmp_path, measured(planned, points), directory)
    assert out["share"] == 1.0


def test_coverage_counts_apart_the_prefill_steps_cached_past_the_top_c_node(
    tmp_path, monkeypatch
):
    """At a 48-token budget prefill's top c node is 2048 - 48 = 2000; a
    2040-token prompt's last chunk, 24 tokens on 2016 cached, is past it."""
    directory = record(tmp_path, monkeypatch, 2040, budget=48)
    planned = plan(directory)
    last = f"{directory} dp0 step 42"
    reason = "c=2016 is outside the prefill output grid's nodes 0..2000"
    assert planned["bounds"]["max_num_batched_tokens"] == 48
    assert planned["unplaced"] == [{"step": last, "reason": reason}]

    out = coverage(
        tmp_path,
        measured(planned, located(tmp_path, planned)),
        directory,
        max_num_batched_tokens=48,
    )

    assert out["prefill_above_top_c"] == 1
    assert out["refused"] == [{"step": last, "reasons": [reason]}]
    assert out["covered"] == out["steps"] - 1


def test_a_step_off_the_targets_ladder_is_listed_unplaced(tmp_path, monkeypatch):
    directory = record(tmp_path, monkeypatch, *MIXED)
    planned = plan(directory, engine={"ladder": [1, 2, 8]})

    assert planned["unplaced"] == [
        {"step": f"{directory} dp0 step {n}", "reason": "no decode rung 4"}
        for n in (5, 6, 7)
    ]
    assert "decode" not in located(tmp_path, planned)


def test_the_plan_reads_shapes_never_timestamps(tmp_path, monkeypatch):
    directory = record(tmp_path, monkeypatch, *MIXED)
    timed = tmp_path / "timed"
    timed.mkdir()
    lines = [
        json.loads(s) for s in (directory / "dp0.jsonl").read_text().split("\n") if s
    ]
    for i, line in enumerate(lines):
        line.update(t_enter_ns=10**9 * i, t_exit_ns=10**9 * i + 7 ** (i % 5))
    (timed / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    (timed / "dp0.engine.json").write_text((directory / "dp0.engine.json").read_text())

    assert plan(timed) == plan(directory)


def test_a_definition_stating_a_target_field_must_state_the_engines_value(
    tmp_path, monkeypatch
):
    directory = record(tmp_path, monkeypatch, *MIXED)

    assert plan(directory, **ENGINE) == plan(directory)
    with pytest.raises(
        ValueError,
        match=r"^the grid definition states max_num_seqs 16 and the engine 8; "
        r"ladder \[1, 2\] and the engine \[1, 2, 4, 8\]$",
    ):
        plan(directory, max_num_seqs=16, ladder=[1, 2])
    with pytest.raises(ValueError, match="states capture_sizes, which plan does not"):
        plan(directory, capture_sizes=LADDER)


@pytest.mark.parametrize(
    "drive,engine,field",
    [
        ({"budget": 48}, {}, "max_num_batched_tokens"),
        ({}, {"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
    ],
    ids=["target-field", "keyed-field"],
)
def test_inputs_whose_ranks_state_two_engines_are_refused(
    tmp_path, monkeypatch, drive, engine, field
):
    """The second record is driven by `drive` and its engine file changed by
    `engine`; a field `plan` does not plan for still makes another engine."""
    records = [
        record(tmp_path, monkeypatch, *MIXED),
        record(tmp_path, monkeypatch, *SHORT, **drive),
    ]
    path = records[1] / "dp0.engine.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **engine}))

    with pytest.raises(
        ValueError,
        match=f"^{re.escape(f'{records[0]} dp0 and {records[1]} dp0')} state two "
        f"engines, which differ on {field}$",
    ):
        plan(*records)


def test_a_record_from_a_run_with_a_drafter_is_refused_at_its_first_such_step(
    tmp_path, monkeypatch
):
    """A drafter's decode row also verifies its draft, so it computes two
    tokens: MIXED's decode steps 5 and 6 as a drafter's run would record them."""
    named, engines = steps(record(tmp_path, monkeypatch, *MIXED))
    for _, line in named[5:7]:
        assert line["rung"] == 4
        line["rows"] = [[2, context + 1, True] for _, context, _ in line["rows"]]

    with pytest.raises(
        ValueError,
        match=f"^{re.escape(named[5][0])} decodes more than one token in a row",
    ):
        step_grid_plan.plan(DEFINITION, named, engines)


@pytest.mark.parametrize(
    "changes,named",
    [({k: None}, f"the grid definition states no {k}$") for k in DEFINITION]
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
    tmp_path, monkeypatch, changes, named
):
    definition = {k: v for k, v in {**DEFINITION, **changes}.items() if v is not None}
    lines, engines = steps(record(tmp_path, monkeypatch, *MIXED))

    with pytest.raises(ValueError, match=named):
        step_grid_plan.plan(definition, lines, engines)


@pytest.mark.parametrize("field", step_grid_plan.TARGET)
def test_every_target_field_is_required_of_the_engine_and_a_count(
    tmp_path, monkeypatch, field
):
    lines, engines = steps(record(tmp_path, monkeypatch, *MIXED))
    ((rank, engine),) = engines.items()
    odd = {**engine, field: [0.5] if field == "ladder" else 0.5}
    del engine[field]

    with pytest.raises(ValueError, match=f"^{re.escape(rank)} states no {field}$"):
        step_grid_plan.plan(DEFINITION, lines, engines)
    with pytest.raises(ValueError, match="0.5 is not a count"):
        step_grid_plan.plan(DEFINITION, lines, {rank: odd})


def test_a_target_whose_model_length_is_below_the_token_budget_is_refused(
    tmp_path, monkeypatch
):
    """Prefill's top c node, max_model_len less the budget, would be -32."""
    directory = record(tmp_path, monkeypatch, *MIXED)

    with pytest.raises(ValueError, match="max_model_len is below its max_num_"):
        plan(directory, engine={"max_model_len": 32})


def test_no_step_is_refused(tmp_path):
    with pytest.raises(ValueError, match="there is no step to read"):
        plan(tmp_path)
    with pytest.raises(ValueError, match="there is no step to read"):
        step_grid_plan.coverage(None, [])
