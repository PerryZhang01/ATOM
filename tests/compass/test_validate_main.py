# SPDX-License-Identifier: MIT
"""Grading a cell: `python -m atom.compass.validate` on runs laid out as `cctraces_sim.sh` writes them,
and as `record` and `run` do (`_as_recorded`).

Every run's step record is `test_validate_cost`'s, periods at its planted law's
price; the export and the server log are as `test_validate_runs` and
`test_memory_check` write them. `validate.runs`, `validate.cost` and
`memory.check` read them as they read a real cell.
"""

import json

import pytest
from test_memory_check import GIB, _log
from test_validate_cost import LAW, _record, _steps
from test_validate_runs import ARGUMENTS, EXPORT

from atom.compass.backends.cost import GROUPS
from atom.compass.run import LP_FILE
from atom.compass.validate.__main__ import main
from atom.compass.validate.runs import CLIENT_RUN

TERM = {"method": "measured", "source": "planted"}
SHARE = {"steps": 3, "fraction_of_predicted_seconds": 0.4}


def _run(
    d,
    ttft,
    *,
    refusals=(),
    non_torch=1.0,
    speed=True,
    workload=1,
    stretch=None,
    steps=None,
    idle_after=(),
):
    """One run. `ttft` is (uncached, cached, ms) per profiling request; a warmup
    request with 9000 uncached tokens precedes them. `refusals` None makes the
    run real, a list of (source, count) makes it simulated with those refusals."""
    d.mkdir(parents=True)
    _record(d / "steps", steps or _steps(), stretch=stretch, idle_after=idle_after)
    (d / "artifacts").mkdir()
    export = json.loads(json.dumps(EXPORT))
    export["time_to_first_token"]["avg"] = ttft[0][2]
    (d / "artifacts" / "profile_export_aiperf.json").write_text(json.dumps(export))
    lines = [("warmup", 9000, 0, 1.0)] + [("profiling", *r) for r in ttft]
    (d / "artifacts" / "profile_export.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "metadata": {"benchmark_phase": phase},
                    "metrics": {
                        "time_to_first_token": {"value": ms},
                        "usage_prompt_tokens": {"value": u + c},
                        "usage_prompt_cache_read_tokens": {"value": c},
                    },
                }
            )
            + "\n"
            for phase, u, c, ms in lines
        )
    )
    (d / "run.json").write_text(
        json.dumps(
            {"workload": {"traces": workload}, "step_linear": LAW, "host": {"laws": {}}}
        )
    )
    if refusals is None:
        _log(d / "server.log", peak=50 * GIB, non_torch=non_torch * GIB)
        return
    readings = {"peak_torch": 50 * GIB, "non_torch": GIB}
    memory = {
        "label": "planted",
        "tokens": 8192,
        "readings": {k: [TERM | {"name": k, "nbytes": v}] for k, v in readings.items()},
        "predicts": None,
    }
    hand_in = {"refusals": [], "priced": {}, "memory": memory}
    (d / LP_FILE.format("engine")).write_text(json.dumps(hand_in))
    summary = {
        "coverage_report": bool(refusals),
        "cost": {
            "speed_ratio": 20.0 if speed else None,
            "speed_refused": None if speed else "no wall time was measured",
            "simulated_seconds": 900.0,
            "wall_seconds": 45.0 if speed else 0.0,
            "meets_speed_target": True if speed else None,
            "speed_target_ratio": 5.0,
        },
        "schedule": {
            "refusals": {
                "count": sum(n for _, n in refusals),
                "by_source": [list(r) for r in refusals],
                "steps": 20,
                "fraction_of_steps": dict(refusals).get("cost", 0) / 20,
            },
            "seconds_by_method": {g: SHARE for g in GROUPS},
        },
    }
    (d / "summary.json").write_text(json.dumps(summary))


def _rows(capsys, tmp_path, law=(), **law_run):
    """The cell's rows; real0's step 5 ran ten times its price, `law` is the
    simulated run's refusals and `law_run` sets its other arguments."""
    for i, (ms, non_torch) in enumerate(((100, 1.0), (110, 1.5), (105, 1.0))):
        _run(
            tmp_path / "c4" / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            non_torch=non_torch,
            stretch={5: 10.0} if i == 0 else None,
        )
    _run(
        tmp_path / "c4" / "step_linear",
        [(100, 1000, 120.0)],
        refusals=law,
        **law_run,
    )
    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    return {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}


def test_a_cell_is_graded_from_every_source_and_its_method_shares_grade_nothing(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path)
    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    # Uncached tokens only, graded against the real spread of 10%.
    assert rows["end to end", "ttft p50 0-512 uncached"] == [
        "spread 0.1; result 0.2 0.09091 0.1429",
        "outside",
    ]
    assert rows["end to end", "ttft p50 512-2048 uncached"][0] == (
        "no request in ['step_linear']"
    )
    # The warmup request is in no band.
    assert rows["end to end", "ttft p50 8192+ uncached"][0].startswith("no request")
    assert rows["end to end", "ttft mean"][1] == "outside"
    assert rows["cost real0", "decode rung 1 validated"][0].endswith(
        "within 10% 100.0%"
    )
    # Every step the cost rows leave out is named, a warm one with its real seconds.
    assert rows["cost real0", "left out"][0] == (
        'not labelled {"exposed": 0, "mixed": 0, "unsplit": 0}; warm steps, real s {5: 0.81}'
    )
    assert rows["cost real0", "prefill validated"][0].startswith("9 steps")
    assert rows["cost real1", "left out"][0].endswith("warm steps, real s {}")
    assert rows["memory real0", "non_torch"][1] == "pass"
    assert rows["memory real1", "non_torch"] == ["-33.33% (-536,870,912 B)", "fail"]
    for g in GROUPS:
        assert rows["summary step_linear", g] == [json.dumps(SHARE), "not gated"]
    assert rows["summary step_linear", "speed"] == [
        "20.0x: 900 simulated s in 45 wall s",
        "meets 5x",
    ]
    assert rows["summary step_linear", "refusals"] == [
        "0 {}; cost refused 0.0% of 20 steps",
        "none",
    ]


def test_a_clock_refusal_makes_the_cell_a_coverage_report(capsys, tmp_path):
    # A clock refusal refuses no step; it is counted under its source, not as a step.
    rows = _rows(capsys, tmp_path, law=[("clock", 1)], speed=False)
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary step_linear", "refusals"] == [
        '1 {"clock": 1}; cost refused 0.0% of 20 steps',
        "coverage report",
    ]
    assert rows["summary step_linear", "speed"] == [
        "no wall time was measured: 900 simulated s in 0 wall s",
        "no result for 5x",
    ]


def test_runs_of_different_workloads_are_not_one_cell(capsys, tmp_path):
    with pytest.raises(ValueError, match="ran different workloads"):
        _rows(capsys, tmp_path, workload=2)


def test_a_cost_refusal_makes_the_cell_a_coverage_report(capsys, tmp_path):
    rows = _rows(capsys, tmp_path, law=[("cost", 2)])
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary step_linear", "refusals"] == [
        '2 {"cost": 2}; cost refused 10.0% of 20 steps',
        "coverage report",
    ]


def test_counts_but_preemptions_and_prefill_tokens_are_not_gated(capsys, tmp_path):
    # A decode step of another request splits the law run's prefill chunks: one
    # prefill streak and one decode token more than every real run.
    steps = _steps()
    steps.insert(5, ([(1, 2000, True)], 1))
    rows = _rows(capsys, tmp_path, steps=steps, idle_after={4, 5})
    assert rows["counts", "prefill streaks"] == [
        "spread 0; result 1 1 1",
        "outside, not gated",
    ]
    assert rows["counts", "decode tokens"][1] == "outside, not gated"
    assert rows["counts", "prefill tokens"][1] == "exact"
    assert rows["counts", "preemptions"][1] == "exact"


def test_a_run_that_ended_without_its_memory_reads_refused_and_the_cell_goes_on(
    capsys, tmp_path
):
    whole = _rows(capsys, tmp_path / "whole")
    _rows(capsys, tmp_path, law=[("ended", 1)])
    law = tmp_path / "c4" / "step_linear"
    ended = {"refusals": ["ended:engine raised RuntimeError('the step raised')"]}
    (law / LP_FILE.format("engine")).write_text(json.dumps(ended))
    summary = json.loads((law / "summary.json").read_text())
    summary["schedule"].update(seconds_by_method=None)
    summary["schedule"]["refusals"].update(steps=None, fraction_of_steps=None)
    (law / "summary.json").write_text(json.dumps(summary))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    for real in ("real0", "real1", "real2"):
        assert rows[f"memory {real}", "memory"] == [
            f"{law / LP_FILE.format('engine')} has no memory section",
            "refused",
        ]
    memory = {k for k in whole if k[0].startswith("memory ")}
    assert set(rows) - {k for k in rows if k[1] == "memory"} == set(whole) - memory
    assert rows["summary step_linear", "unvalidated"] == ["null", "not gated"]
    assert rows["summary step_linear", "refusals"] == [
        '1 {"ended": 1}; steps not counted',
        "coverage report",
    ]
    assert rows["summary step_linear", "speed"][1] == "meets 5x"


def test_a_cell_of_renamed_names_is_refused_naming_their_names_now(capsys, tmp_path):
    _rows(capsys, tmp_path)
    cell = tmp_path / "c4"
    run = cell / "real0" / "run.json"
    run.write_text(json.dumps(json.loads(run.read_text()) | {"law": LAW}))
    with pytest.raises(ValueError, match=r"renamed: law \(now step_linear\)"):
        main([str(cell)])
    (cell / "step_linear").rename(cell / "law")
    with pytest.raises(ValueError, match="now named step_linear/"):
        main([str(cell)])


def _as_recorded(cell, warmup=()):
    """Lay `cell`'s runs out as `record` and `run` write them: each request
    named at its last prefill row, by key `k<id>` and response id `id-<id>`;
    an export naming them all, from trace `t<id>`, those in `warmup` as its
    warmup's, else request `w`, a one-step prefill before every other step,
    numbered below them, and the same traces and arguments given to every run's
    client; and run.json's identity,
    with `run`'s grading entries for `step_linear/`, its law and host laws in
    its run file and its refusals its summary's."""
    for d in sorted(cell.iterdir()):
        path = d / "steps" / "dp0.jsonl"
        lines = [json.loads(x) for x in path.read_text().splitlines()]
        last = {}
        for i, line in enumerate(lines):
            for (r, *_), (_q, _c, decode) in zip(line["batch"], line["rows"]):
                last[r] = i if not decode else last.get(r, i)
        for r, i in last.items():
            lines[i]["named"].append([r, f"k{r}", f"id-{r}"])
        if not warmup:
            r, step = max(last) + 1, lines[0]["step"] - 1
            w = {"step": step, "batch": [[r, 1, 1]], "named": [[r, "kw", "id-w"]]}
            lines.insert(0, lines[0] | w | {"rows": [[1, 1, False]], "rung": None})
        path.write_text("".join(json.dumps(x) + "\n" for x in lines))
        export = {"requests": {}, "warmup": {}, "traces": ["t0", "t1", "tw"]}
        export["arguments"] = ARGUMENTS
        for r in last:
            part = "warmup" if r in warmup else "requests"
            export[part][f"id-{r}"] = {"trace": f"t{r}", "sampling": {}}
        if not export["warmup"]:
            export["warmup"]["id-w"] = {"trace": "tw", "sampling": {}}
        (d / CLIENT_RUN).write_text(json.dumps(export))
        run = {"format_version": 2}
        if d.name == "step_linear":
            summary = json.loads((d / "summary.json").read_text())
            run |= {
                "run_file": {"step_linear": LAW, "host": {"laws": {}}},
                "step_roofline": False,
                "skip": [],
                "refusals": summary["schedule"]["refusals"] | {"reasons": []},
                "coverage": None,
                "provenance": {"digest": "sha256:0", "provenance": None},
            }
        (d / "run.json").write_text(json.dumps(run))


def _graded(capsys, cell):
    main([str(cell)])
    out = capsys.readouterr().out.splitlines()
    return {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}


def test_a_cell_record_and_run_wrote_grades_as_the_one_cctraces_sim_wrote(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")

    graded = _graded(capsys, tmp_path / "c4")
    stated = [graded.pop(k) for k in list(graded) if k[0].startswith("warmup ")]
    assert graded == rows
    # Each run's warmup is request `w`'s step; a `cctraces_sim.sh` run reads none.
    assert stated == [["requests 1, steps 1", "not graded"]] * 4


def test_a_refusal_in_the_client_run_record_and_run_wrote_refuses_the_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path, law=[("clock", 1)])
    _as_recorded(tmp_path / "c4")

    with pytest.raises(ValueError, match="step_linear is not graded: it refused 1"):
        main([str(tmp_path / "c4")])


def test_runs_record_and_run_wrote_that_were_given_other_traces_are_not_one_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    export = tmp_path / "c4" / "real1" / CLIENT_RUN
    other = {"traces": ["tx"]}
    export.write_text(json.dumps(json.loads(export.read_text()) | other))

    with pytest.raises(ValueError, match=r"given different traces; .*real1': \['tx'\]"):
        main([str(tmp_path / "c4")])


def test_runs_record_and_run_wrote_at_another_concurrency_are_not_one_cell(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    _as_recorded(tmp_path / "c4")
    export = tmp_path / "c4" / "real1" / CLIENT_RUN
    other = json.loads(export.read_text())
    other["arguments"] = ARGUMENTS | {"phases.profiling.concurrency": 16}
    export.write_text(json.dumps(other))

    concurrency = r"\{'phases.profiling.concurrency': \{.*real1': 16, .*\}\}$"
    with pytest.raises(ValueError, match=f"given different arguments: {concurrency}"):
        main([str(tmp_path / "c4")])


def _warmed(capsys, cell, slow):
    """A cell `record` and `run` wrote, request 0 every run's warmup: two steps
    of its own before request 1's, `slow` times their price in every real run."""
    steps = [([(200, 200, False)], None), ([(1, 200, True)], 1), *_steps()]
    for i, ms in enumerate((100, 110, 105)):
        _run(
            cell / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            steps=steps,
            idle_after={1},
            stretch={0: slow, 1: slow},
        )
    _run(cell / "step_linear", [(100, 1000, 120.0)], steps=steps, idle_after={1})
    _as_recorded(cell, warmup={0})
    return _graded(capsys, cell)


def test_a_warmed_run_is_graded_on_its_profiling_steps_and_slow_warmup_moves_nothing(
    capsys, tmp_path
):
    rows = _warmed(capsys, tmp_path / "at price", 1.0)

    assert _warmed(capsys, tmp_path / "slow", 1000.0) == rows
    for run in ("real0", "real1", "real2", "step_linear"):
        assert rows[f"warmup {run}", "warmup"] == ["requests 1, steps 2", "not graded"]
    # Cost prices the profiling steps alone: ten prefill chunks, ten decode steps.
    assert rows["cost real0", "prefill validated"][0].startswith("10 steps")
    assert rows["cost real0", "decode validated"][0].startswith("10 steps")
    assert rows["cost real0", "left out"][0].endswith("warm steps, real s {}")


def test_a_run_file_of_a_law_per_engine_config_costs_real_runs_at_their_own(
    capsys, tmp_path
):
    cell = tmp_path / "c4"
    _rows(capsys, tmp_path)
    _as_recorded(cell)
    rows = _graded(capsys, cell)
    ours, theirs = {"enable_tbo": False}, {"enable_tbo": True}
    other = json.loads(json.dumps(LAW))
    other["prefill"]["coefficients"]["step"] *= 2
    run = json.loads((cell / "step_linear" / "run.json").read_text())
    laws = [other | {"engine_config": theirs}, LAW | {"engine_config": ours}]
    run["run_file"]["step_linear"] = laws
    (cell / "step_linear" / "run.json").write_text(json.dumps(run))
    for real in ("real0", "real1", "real2"):
        engine = ours | {"num_kvcache_blocks": 8, "max_num_seqs": 4}
        (cell / real / "steps" / "dp0.engine.json").write_text(json.dumps(engine))

    assert _graded(capsys, cell) == rows


def test_the_steps_extrapolated_or_priced_by_a_fallback_are_stated_and_graded(
    capsys, tmp_path
):
    _rows(capsys, tmp_path)
    law = tmp_path / "c4" / "step_linear"
    summary = json.loads((law / "summary.json").read_text())
    summary["schedule"].update(extrapolated_steps=3, fallback_steps=2)
    (law / "summary.json").write_text(json.dumps(summary))

    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    rows = {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}

    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    assert rows["summary step_linear", "extrapolated steps"] == ["3 of 20", "not gated"]
    assert rows["summary step_linear", "fallback steps"] == ["2 of 20", "not gated"]
