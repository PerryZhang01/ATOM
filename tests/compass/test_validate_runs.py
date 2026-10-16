# SPDX-License-Identifier: MIT
"""Run grades: cells laid out as `cctraces_sim.sh` writes them, and as `record`
and `run` do.

Each cell is a step record in `steps/dp0.jsonl`, the JSON lines a runner
writes, read by `runs._steps`, and aiperf's export in
`artifacts/profile_export_aiperf.json`. `_recorded` adds what a `record` or
`run` directory holds beside them: run.json's identity and grading entries,
and the client run's export.
"""

import collections
import json

import pytest

from atom.compass.validate.runs import CLIENT_RUN, _requests, grade, main

EXPORT = {
    "request_count": {"unit": "requests", "avg": 3.0},
    "time_to_first_token": {"unit": "ms", "avg": 1000.0, "p50": 900.0},
    "inter_token_latency": {"unit": "ms", "avg": 25.0, "p50": 24.0},
    "request_throughput": {"unit": "requests/sec", "avg": 0.5},
    "output_token_throughput": {"unit": "tokens/sec", "avg": 100.0},
}


def _schedule(ids=(7, 8)):
    """Request `a` prefills in two chunks, `b` in one, both decode; `a` is preempted and prefills again."""
    a, b = ids
    return [
        [(a, 100, 100, False)],
        [(a, 100, 200, False), (b, 50, 50, False)],
        [(a, 1, 201, True), (b, 1, 51, True)],
        [(a, 1, 202, True), (b, 1, 52, True)],
        [(a, 203, 203, False)],
        [(a, 1, 204, True)],
    ]


def _cell(directory, schedule, keys, **export):
    """A cell directory: `keys` maps a request id to the key its final chunk is named by."""
    (directory / "steps").mkdir(parents=True)
    (directory / "artifacts").mkdir()
    lines, named = [], set()
    for i, step in enumerate(schedule):
        final = [r for r, _q, _c, d in step if not d and r not in named]
        named.update(final)
        lines.append(
            {
                "step": i,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [[r, q, c] for r, q, c, _ in step],
                "named": [[r, keys[r]] for r in final if r in keys],
                "rows": [[q, c, d] for _, q, c, d in step],
                "rung": None,
            }
        )
    lines.insert(2, {**lines[0], "is_dummy_run": True, "batch": [], "named": []})
    (directory / "steps" / "dp0.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines)
    )
    out = json.loads(json.dumps(EXPORT))
    for name, (stat, value) in export.items():
        out[name][stat] = value
    (directory / "artifacts" / "profile_export_aiperf.json").write_text(json.dumps(out))
    return directory


def test_counts_match_across_runs_whose_request_ids_differ(tmp_path):
    keys = {7: "ka", 8: "kb", 1: "ka", 2: "kb"}
    real = [_cell(tmp_path / f"r{i}", _schedule(), keys) for i in range(3)]
    simulated = _cell(tmp_path / "sim", _schedule((1, 2)), keys)
    report = grade(real, simulated)

    counts = report["counts"]
    assert {k: g["simulated"] for k, g in counts.items()} == {
        "prefill steps": 3,
        "decode steps": 3,
        "prefill tokens": 453,
        "decode tokens": 5,
        "preemptions": 1,
        "prefill streaks": 2,
        "requests completed": 3.0,
    }
    assert {g["verdict"] for g in counts.values()} == {"exact"}
    widths = report["distributions"]["batch width"]
    assert widths["spread"] == 0 and widths["result"] == [0, 0, 0]


def test_a_metric_is_graded_against_the_real_spread(tmp_path):
    keys = {7: "ka", 8: "kb"}
    ttft = [1000.0, 1050.0, 1100.0]  # spread 10%, inside the 15% target
    tpot = [25.0, 25.5, 30.0]  # spread 20%, beyond 10%
    real = [
        _cell(
            tmp_path / f"r{i}",
            _schedule(),
            keys,
            time_to_first_token=("avg", t),
            inter_token_latency=("avg", p),
        )
        for i, (t, p) in enumerate(zip(ttft, tpot))
    ]

    def simulated(name, t, schedule=None, tpot=25.0):
        return _cell(
            tmp_path / name,
            schedule or _schedule(),
            keys,
            time_to_first_token=("avg", t),
            inter_token_latency=("avg", tpot),
        )

    report = grade(real, simulated("at", 1040.0))
    g = report["end to end"]["ttft mean"]
    assert g["result"] == pytest.approx([0.04, 1 - 1040 / 1050, 1 - 1040 / 1100])
    assert g["spread"] == pytest.approx(0.1)
    assert g["verdict"] == "inside the real spread"
    assert report["end to end"]["tpot mean"]["verdict"] == "ungradeable"
    assert report["end to end"]["ttft p50"]["verdict"] == "inside the real spread"

    report = grade(real, simulated("high", 1140.0))
    assert report["end to end"]["ttft mean"]["verdict"] == "within target"
    report = grade(real, simulated("higher", 1300.0))
    assert report["end to end"]["ttft mean"]["verdict"] == "outside"

    # One more decode step: a count off by one where the real runs agree.
    longer = _schedule() + [[(7, 1, 205, True)]]
    report = grade(real, simulated("long", 1040.0, longer))
    assert report["counts"]["decode steps"]["verdict"] == "outside"
    assert (
        report["distributions"]["batch width"]["result"] == [pytest.approx(1 / 14)] * 3
    )


def test_the_command_prints_every_metric_and_needs_two_real_runs(tmp_path, capsys):
    keys = {7: "ka", 8: "kb"}
    ttft = [1000.0, 1000.0, 1000.0, 1200.0]  # the real spread, 20%, swamps 15%
    cells = [
        _cell(tmp_path / f"c{i}", _schedule(), keys, time_to_first_token=("avg", t))
        for i, t in enumerate(ttft)
    ]
    main([str(c) for c in cells])
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "family | metric | spread | result | verdict"
    assert "counts | preemptions | 0 | 0 0 0 | exact" in out
    assert "end to end | ttft mean | 0.2 | 0 0 0.1667 | ungradeable" in out
    assert any(row.startswith("end to end | output token throughput |") for row in out)
    with pytest.raises(ValueError, match="two or more real runs"):
        grade(cells[:1], cells[1])


def test_prefill_tokens_grade_the_requests_both_runs_issued(tmp_path):
    def run(name, ids, hit=0, extra=False):
        """Requests of 100 and 50 prompt tokens, `hit` of the second cached;
        `extra` issues a third, of the first one's length, after them."""
        a, b, c = ids
        schedule = [
            [(a, 100, 100, False)],
            [(a, 1, 101, True), (b, 50 - hit, 50, False)],
            [(a, 1, 102, True), (b, 1, 51, True)],
        ]
        if extra:
            schedule.append([(c, 100, 100, False)])
        return _cell(tmp_path / name, schedule, {i: f"k{i}" for i in ids})

    real = [run("r0", (7, 8, 9)), run("r1", (7, 8, 9)), run("r2", (5, 6, 9), hit=10)]
    simulated = run("sim", (21, 22, 23), hit=10, extra=True)
    report = grade(real, simulated)

    g = report["counts"]["prefill tokens"]
    assert g["simulated"] == 240 and g["real"] == [150, 150, 140]
    assert g["result"] == [10, 10, 0] and g["spread"] == 10
    assert g["verdict"] == "inside the real spread"
    one = report["one run"]["prefill tokens"]
    assert one["result"] == [100, 100, 100] and one["spread"] == 0
    assert one["verdict"] == "not gated"


def test_requests_keys_by_the_final_chunks_context_not_a_later_decode_row():
    """A decode row after the final chunk must not move the request's key away
    from its prompt length (100), which `named` fixes at the prefill row."""
    steps = [
        {"named": [[1, "k1"]], "batch": [[1, 100, 100]], "rows": [[100, 100, False]]},
        {"named": [], "batch": [[1, 1, 101]], "rows": [[1, 101, True]]},
    ]
    assert _requests(steps) == collections.Counter({(100, 0): 100})


#: A simulated run's run.json entries that grading reads, as `run` writes
#: them, for a run it grades: host laws on, no roofline, no refusal, and a
#: backend fitted on a trace no cell here sends.
GRADED = {
    "run_file": {"step_linear": {}, "host": {"laws": {}}},
    "step_roofline": False,
    "skip": [],
    "refusals": {"count": 0, "reasons": []},
    "coverage": None,
    "provenance": {
        "digest": "sha256:0",
        "provenance": {"inputs": [{"run": "fit.jsonl", "traces": ["trace-kz"]}]},
    },
}


def _recorded(cell, sent=None, **run):
    """`cell` as `record` lays it out, or with `run`'s entries as `run` does:
    each named request also named by its response id `id-<key>`, and the
    export naming those whose keys are in `sent` (all by default), each from
    trace `trace-<key>`."""
    path = cell / "steps" / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        line["named"] = [[r, k, f"id-{k}"] for r, k, *_ in line["named"]]
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    keys = sorted({k for line in lines for _, k, *_ in line["named"]})
    requests = {
        f"id-{k}": {"trace": f"trace-{k}", "sampling": {}}
        for k in keys
        if sent is None or k in sent
    }
    (cell / CLIENT_RUN).write_text(json.dumps({"requests": requests}))
    (cell / "run.json").write_text(json.dumps({"format_version": 2, **run}))
    return cell


def _four(tmp_path, schedule=None, keys=None):
    """Three real cells and a simulated one, as `cctraces_sim.sh` lays them out."""
    keys = keys or {7: "ka", 8: "kb"}
    names = ("r0", "r1", "r2", "sim")
    return [_cell(tmp_path / n, schedule or _schedule(), keys) for n in names]


def test_a_cell_record_and_run_wrote_grades_as_the_one_cctraces_sim_wrote(tmp_path):
    *real, simulated = _four(tmp_path)
    report = grade(real, simulated)
    for cell in real:
        _recorded(cell)
    _recorded(simulated, **GRADED)

    assert grade(real, simulated) == report


@pytest.mark.parametrize(
    "entries, reason",
    [
        ({"skip": ["E"]}, r"it skips host segments \['E'\], a what-if run"),
        (
            {"run_file": {"step_linear": {}}},
            "its run file has no host laws, so it prices no host time",
        ),
        ({"step_roofline": True}, "the step roofline was on"),
        (
            {"refusals": {"count": 1, "reasons": [["clock:late", 1]]}},
            r"it refused 1: \[\['clock:late', 1\]\]",
        ),
        (
            {"run_file": {"step_grid": {}, "host": {}}},
            "it ran the step grid without a passed coverage check",
        ),
        (
            {
                "run_file": {"step_grid": {}, "host": {}},
                "coverage": {"uncovered": ["s"]},
            },
            "it ran the step grid without a passed coverage check",
        ),
        (
            {
                "provenance": {
                    "provenance": {
                        "inputs": [{"run": "p", "traces": [], "profiled": True}]
                    }
                }
            },
            r"its backend was fitted on profiled inputs: \['p'\]",
        ),
        (
            {"run_file": {"step_grid": {}, "host": {}}, "coverage": {"uncovered": []}},
            None,
        ),
        ({"provenance": {"digest": "sha256:stub", "provenance": None}}, None),
    ],
)
def test_a_simulated_run_grading_does_not_take_is_refused_naming_why(
    tmp_path, entries, reason
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **(GRADED | entries))
    if reason is None:
        grade(real, simulated)
        return
    with pytest.raises(ValueError, match=f"sim is not graded: {reason}$"):
        grade(real, simulated)


def test_a_simulated_run_fitted_on_a_trace_the_cell_served_is_refused_naming_it(
    tmp_path,
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    inputs = [{"run": "fit.jsonl", "traces": ["trace-kb", "trace-kz"]}]
    leaked = {"digest": "sha256:0", "provenance": {"inputs": inputs}}
    _recorded(simulated, **(GRADED | {"provenance": leaked, "step_roofline": True}))

    fitted = r"its backend was fitted on traces it served: \['trace-kb'\]"
    with pytest.raises(ValueError, match=f"the step roofline was on; {fitted}$"):
        main([str(simulated), *map(str, real)])


def test_a_client_run_its_server_served_after_another_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    # Request `ka` prefills alone first, and this client run sent only `kb`.
    _recorded(real[1], sent={"kb"})

    with pytest.raises(ValueError, match="r1 is not graded: steps of requests it did"):
        grade(real, simulated)


def test_the_steps_of_a_later_client_run_are_not_graded(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    report = grade(real, simulated)
    later = _schedule() + [[(9, 30, 30, False)], [(9, 1, 31, True)]]
    _cell(tmp_path / "later", later, {7: "ka", 8: "kb", 9: "kc"})
    real[0] = _recorded(tmp_path / "later", sent={"ka", "kb"})

    assert grade(real, simulated) == report


def test_runs_that_served_different_traces_are_not_one_cell(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    export = json.loads((real[2] / CLIENT_RUN).read_text())
    export["requests"]["id-kb"]["trace"] = "trace-other"
    (real[2] / CLIENT_RUN).write_text(json.dumps(export))

    beyond = f"'{real[2]}': ['trace-other'], '{simulated}': ['trace-kb']}}"
    with pytest.raises(ValueError, match="sent different traces") as refused:
        grade(real, simulated)
    assert str(refused.value).endswith(beyond)
