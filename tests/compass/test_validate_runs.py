# SPDX-License-Identifier: MIT
"""Run grades: cells laid out as `cctraces_sim.sh` writes them, and as `record`
and `run` do.

Each cell is a step record in `steps/dp0.jsonl`, the JSON lines a runner
writes, read by `runs._steps`, and aiperf's export in
`artifacts/profile_export_aiperf.json`. `_recorded` adds what a `record` or
`run` directory holds beside them: run.json's identity and grading entries,
and the client run's export. `_ran` has `compass run` write a simulated cell's
grading entries, over the summaries a server that served its client runs
wrote (`test_client_run_summary._serve`).
"""

import collections
import json

import pytest
from test_client_run_summary import CLIENT_RUNS, _serve
from test_stop_between_client_runs import CLIENT_RUN as UNREFUSED

from atom.compass import run as compass_run
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve
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


#: The traces every client run here was given.
GIVEN = ["trace-ka", "trace-kb", "trace-kc", "trace-kw"]
#: The arguments every client run here was given, as its export names them.
ARGUMENTS = {
    "phases.profiling.concurrency": 4,
    "phases.profiling.duration": 1800.0,
    "phases.profiling.warmup_requests_per_lane": 1,
    "datasets.main.random_seed": 20260707,
    "benchmark_id": "000000000000",
}


def _recorded(
    cell, sent=None, warmup=("kw",), given=GIVEN, held=True, args=ARGUMENTS, **run
):
    """`cell` as `record` lays it out, or with `run`'s entries as `run` does:
    each named request also named by its response id `id-<key>`, and the
    export naming those whose keys are in `sent` (all but its warmup's by
    default), each from trace `trace-<key>`; its warmup naming those whose
    keys are in `warmup`, by default `kw`, each that no step holds served
    first in a prefill step of its own unless `held` is False; and the traces
    and arguments its client was given, `given` and `args`. With `warmup`,
    `given` or `args` None the export has no such entry."""
    path = cell / "steps" / "dp0.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    for line in lines:
        line["named"] = [[r, k, f"id-{k}"] for r, k, *_ in line["named"]]
    keys = sorted({k for line in lines for _, k, *_ in line["named"]})
    # Request ids 0 and up arrived before every request a cell here holds.
    for r, k in enumerate(sorted(set(warmup or ()) - set(keys)) if held else ()):
        warm = {"batch": [[r, 30, 30]], "named": [[r, k, f"id-{k}"]]}
        lines.insert(r, lines[0] | warm | {"rows": [[30, 30, False]]})
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))

    def named(mine):
        return {f"id-{k}": {"trace": f"trace-{k}", "sampling": {}} for k in mine}

    sent = set(keys) - set(warmup or ()) if sent is None else sent
    export = {"requests": named(k for k in keys if k in sent)}
    if warmup is not None:
        export["warmup"] = named(warmup)
    if given is not None:
        export["traces"] = given
    if args is not None:
        export["arguments"] = args
    (cell / CLIENT_RUN).write_text(json.dumps(export))
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
            "it ran the step grid without a coverage check",
        ),
        # The step grid prices below it by the roofline, switch or none, and a
        # check that counts steps extrapolated or falling to it refuses none.
        (
            {
                "run_file": {"step_grid": {}, "host": {}},
                "step_roofline": True,
                "coverage": {"extrapolated": 3, "fallback": 2, "declined": [{}] * 2},
            },
            None,
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
            {"run_file": {"step_grid": {}, "host": {}}, "coverage": {"declined": []}},
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


def test_the_command_states_the_steps_extrapolated_and_priced_by_a_fallback(
    tmp_path, capsys
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(simulated, **GRADED)
    schedule = {"refusals": {"steps": 9}, "extrapolated_steps": 3, "fallback_steps": 2}
    first = simulated / compass_run.CLIENT_DIR.format(0)
    first.mkdir()
    (first / compass_run.SUMMARY_FILE).write_text(json.dumps({"schedule": schedule}))

    main([str(simulated), *map(str, real)])

    out = capsys.readouterr().out.splitlines()
    assert out[-1] == "of 9 steps priced: extrapolated 3, fallback 2"


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


@pytest.mark.parametrize("target", ["trace-kz", "trace-kb"])
def test_a_step_grid_planned_on_the_traces_a_run_served_grades_unless_a_target_did(
    tmp_path, target
):
    """The provenance `fit steps --backend step-grid` reports: a residual
    target, a replay that served `target`, and the plan, whose points were
    placed from shapes of every trace the cell served."""
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    plan = {
        "source": "plan",
        "run": "sha256:plan",
        "inputs": ["grid"],
        "traces": [*GIVEN, "trace-kz"],
    }
    replay = {"source": "replay", "run": "sha256:replay", "traces": [target]}
    provenance = {"digest": "sha256:0", "provenance": {"inputs": [replay, plan]}}
    grid = {"run_file": {"step_grid": {}, "host": {}}, "coverage": {"uncovered": []}}
    _recorded(simulated, **(GRADED | grid | {"provenance": provenance}))

    if target == "trace-kz":
        grade(real, simulated)
        return
    fitted = r"its backend was fitted on traces it served: \['trace-kb'\]"
    with pytest.raises(ValueError, match=f"sim is not graded: {fitted}$"):
        grade(real, simulated)


def test_a_client_run_its_server_served_after_another_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    # Request `ka` prefills alone first, and this client run sent only `kb`.
    _recorded(real[1], sent={"kb"})

    with pytest.raises(ValueError, match="r1 is not graded: steps of requests it did"):
        grade(real, simulated)


#: Request `kw` warms the server up, two steps of its own before `_schedule`'s.
WARMED = [[(6, 30, 30, False)], [(6, 1, 31, True)]] + _schedule()


def test_a_client_run_s_warmup_steps_precede_its_own_and_are_not_graded(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path / "one_step"))
    report = grade(real, simulated)
    keys = {6: "kw", 7: "ka", 8: "kb"}
    *real, simulated = (
        _recorded(c, sent={"ka", "kb"}, warmup={"kw"})
        for c in _four(tmp_path / "two_steps", WARMED, keys)
    )

    assert grade(real, simulated) == report


@pytest.mark.parametrize(
    "step", [[(9, 30, 30, False)], [(6, 1, 31, True), (9, 30, 30, False)]]
)
def test_a_step_of_a_request_neither_its_own_nor_its_warmup_s_before_them_refuses(
    tmp_path, step
):
    keys = {6: "kw", 7: "ka", 8: "kb", 9: "kx"}
    schedule = WARMED[:1] + [step] + WARMED[2:]
    *real, simulated = (
        _recorded(c, sent={"ka", "kb"}, warmup={"kw"})
        for c in _four(tmp_path, schedule, keys)
    )

    with pytest.raises(ValueError, match="r0 is not graded: steps of requests it did"):
        grade(real, simulated)


@pytest.mark.parametrize("warmup", [None, ()])
def test_a_client_run_whose_export_names_no_warmup_is_refused(tmp_path, warmup):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[2], warmup=warmup)

    cold = "its export names no warmup, so its first steps may be cold"
    with pytest.raises(ValueError, match=f"r2 is not graded: {cold}$"):
        grade(real, simulated)


#: Request `kw` warms the server up only after the client run's first step.
LATE = [_schedule()[0], [(6, 30, 30, False)], *_schedule()[1:]]


@pytest.mark.parametrize("schedule", [_schedule(), LATE], ids=["none", "late"])
def test_a_client_run_whose_warmup_no_step_before_its_first_holds_is_refused(
    tmp_path, schedule
):
    cells = _four(tmp_path, schedule, {6: "kw", 7: "ka", 8: "kb"})
    *real, simulated = (_recorded(c, held=False) for c in cells)

    unheld = "no step before its own first holds its warmup, so its first steps"
    with pytest.raises(ValueError, match=f"r0 is not graded: {unheld} may be cold$"):
        grade(real, simulated)


#: Request `kw` prefills in the client run's first step.
SHARED = [[(6, 30, 30, False), *_schedule()[0]], *_schedule()[1:]]
#: Request `kw` prefills alone first, and decodes on into the client run's second step.
LONG = [WARMED[0], _schedule()[0], [WARMED[1][0], *_schedule()[1]], *_schedule()[2:]]


@pytest.mark.parametrize(
    "schedule, reasons",
    [
        (
            SHARED,
            (
                "no step before its own first holds its warmup, so its first steps "
                "may be cold; a request of its warmup is in 1 of its own steps, from "
                "step 0"
            ),
        ),
        (LONG, "a request of its warmup is in 1 of its own steps, from step 2"),
    ],
    ids=["first", "later"],
)
def test_a_client_run_any_of_whose_own_steps_holds_its_warmup_is_refused(
    tmp_path, schedule, reasons
):
    keys = {6: "kw", 7: "ka", 8: "kb"}
    cells = [_recorded(c, warmup={"kw"}) for c in _four(tmp_path, schedule, keys)]

    into = "so its warmup runs on into the steps it grades"
    with pytest.raises(ValueError, match=f"r0 is not graded: {reasons}, {into}$"):
        grade(cells[:3], cells[3])


def test_the_steps_of_a_later_client_run_are_not_graded(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    report = grade(real, simulated)
    later = _schedule() + [[(9, 30, 30, False)], [(9, 1, 31, True)]]
    _cell(tmp_path / "later", later, {7: "ka", 8: "kb", 9: "kc"})
    real[0] = _recorded(tmp_path / "later", sent={"ka", "kb"})

    assert grade(real, simulated) == report


def test_runs_given_different_traces_are_not_one_cell(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[2], given=["trace-other", *GIVEN[1:]])

    beyond = f"'{real[2]}': ['trace-other'], '{simulated}': ['trace-ka']}}"
    with pytest.raises(ValueError, match="were given different traces") as refused:
        grade(real, simulated)
    assert str(refused.value).endswith(beyond)


def test_runs_given_the_same_traces_pair_though_their_finished_requests_differ(
    tmp_path,
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    report = grade(real, simulated)
    # Request `kb` was in flight at every real run's cutoff, and aiperf
    # records nothing of it: only the simulated run's export names it.
    for cell in real:
        _recorded(cell, sent={"ka"})

    assert grade(real, simulated) == report


def test_a_client_run_whose_export_names_no_traces_given_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[1], given=None)

    unknown = "its export names no traces its client was given"
    with pytest.raises(ValueError, match=f"r1 is not graded: {unknown}$"):
        grade(real, simulated)


@pytest.mark.parametrize(
    "other",
    [
        {"phases.profiling.concurrency": 16},
        {"phases.profiling.duration": 600.0, "datasets.main.random_seed": 1},
        {"phases.profiling.warmup_requests_per_lane": None},
        {"endpoint.timeout": 60.0},
        {"benchmark_id": "0f8c2b6a"},
    ],
)
def test_runs_given_the_same_traces_and_other_arguments_are_not_one_cell(
    tmp_path, other
):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    # None: the export names no such argument.
    args = {k: v for k, v in (ARGUMENTS | other).items() if v is not None}
    _recorded(real[1], args=args)

    cells = [str(c) for c in (*real, simulated)]
    differ = {
        name: {c: ARGUMENTS.get(name) for c in cells} | {str(real[1]): value}
        for name, value in sorted(other.items())
    }
    with pytest.raises(ValueError, match="were given different arguments") as refused:
        grade(real, simulated)
    assert str(refused.value).endswith(f": {differ}")


def test_a_client_run_whose_export_names_no_arguments_given_is_refused(tmp_path):
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    _recorded(real[1], given=None, args=None)

    unknown = "its export names no {} its client was given"
    both = f"{unknown.format('traces')}; {unknown.format('arguments')}"
    with pytest.raises(ValueError, match=f"r1 is not graded: {both}$"):
        grade(real, simulated)


def _ran(monkeypatch, tmp_path, served):
    """Three real cells, and a simulated one whose run.json `compass run`
    wrote once its server, serving until stopped, served one client run per
    entry of `served` and was stopped during the last."""
    sim = _serve(monkeypatch, tmp_path / "sim", None, served, stop="bound")
    run = json.loads((sim / "run.json").read_text()) | GRADED["run_file"]
    run |= {"step_roofline": False, "backend": GRADED["provenance"]}
    (sim / serve.RUN_FILE).write_text(json.dumps(run))
    *real, simulated = (_recorded(c) for c in _four(tmp_path))
    drive_run.write_grading(simulated, None)
    return real, simulated


def test_a_later_client_runs_refusal_does_not_refuse_the_first(monkeypatch, tmp_path):
    real, simulated = _ran(monkeypatch, tmp_path, [UNREFUSED, CLIENT_RUNS[1]])

    server = json.loads((simulated / compass_run.SUMMARY_FILE).read_text())
    reasons = dict(server["schedule"]["refusals"]["reasons"])
    assert "ended:frontend stopped by SIGTERM" in reasons and len(reasons) == 4
    grade(real, simulated)


def test_a_stop_during_the_first_client_run_refuses_it(monkeypatch, tmp_path):
    real, simulated = _ran(monkeypatch, tmp_path, [UNREFUSED])

    unknown = "its first client run has no summary, so its refusals are unknown"
    with pytest.raises(ValueError, match=f"sim is not graded: {unknown}$"):
        grade(real, simulated)
