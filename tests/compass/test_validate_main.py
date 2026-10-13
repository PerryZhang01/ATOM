# SPDX-License-Identifier: MIT
"""Grading a cell: `python -m atom.compass.validate` on runs laid out as `cctraces_sim.sh` writes them.

Every run's step record is `test_validate_cost`'s, periods at its planted law's
price; the export and the server log are as `test_validate_runs` and
`test_memory_check` write them. `validate.runs`, `validate.cost` and
`memory.check` read them as they read a real cell.
"""

import json

import pytest
from test_memory_check import GIB, _log
from test_validate_cost import LAW, _record, _steps
from test_validate_runs import EXPORT

from atom.compass.run import LP_FILE
from atom.compass.validate.__main__ import main

TERM = {"basis": "spec", "source": "planted"}
SPECIES = {"steps": 3, "fraction_of_predicted_seconds": 0.4}


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
        json.dumps({"workload": {"traces": workload}, "law": LAW})
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
            "analytical": SPECIES,
            "extrapolated": SPECIES,
        },
    }
    (d / "summary.json").write_text(json.dumps(summary))


def _rows(capsys, tmp_path, law=(), law_run=None, **p97):
    """The cell's rows; real0's step 5 ran ten times its price, `law` is that run's
    refusals, `law_run` sets that run's other arguments and `p97` sets that run."""
    for i, (ms, non_torch) in enumerate(((100, 1.0), (110, 1.5), (105, 1.0))):
        _run(
            tmp_path / "c4" / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            non_torch=non_torch,
            stretch={5: 10.0} if i == 0 else None,
        )
    _run(tmp_path / "c4" / "law", [(100, 1000, 120.0)], refusals=law, **(law_run or {}))
    _run(tmp_path / "c4" / "p97", [(100, 1000, 118.0)], **p97)
    main([str(tmp_path / "c4")])
    out = capsys.readouterr().out.splitlines()
    return {tuple(r.split(" | ")[1:3]): r.split(" | ")[3:] for r in out[1:]}


def test_a_cell_is_graded_from_every_source_and_its_extrapolated_share_grades_nothing(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path)
    assert rows["cell", "verdict"] == ["3 real runs", "graded"]
    # Uncached tokens only, graded against the real spread of 10%.
    assert rows["end to end", "ttft p50 0-512 uncached"] == [
        "spread 0.1; result 0.2 0.09091 0.1429; moved 0.01667",
        "outside, inadmissible",
    ]
    assert rows["end to end", "ttft p50 512-2048 uncached"][0] == (
        "no request in ['law', 'p97']"
    )
    # The warmup request is in no band.
    assert rows["end to end", "ttft p50 8192+ uncached"][0].startswith("no request")
    assert rows["end to end", "ttft mean"][1] == "outside, inadmissible"
    assert rows["cost real0", "decode rung 1 in hull"][0].endswith("within 10% 100.0%")
    # Every step the cost rows leave out is named, a warm one with its measured seconds.
    assert rows["cost real0", "left out"][0] == (
        'not labelled {"idle": 0, "mixed": 0, "last": 1}; warm steps, measured s {5: 0.81}'
    )
    assert rows["cost real0", "prefill in hull"][0].startswith("9 steps")
    assert rows["cost real1", "left out"][0].endswith("warm steps, measured s {}")
    assert rows["memory real0", "non_torch"][1] == "pass"
    assert rows["memory real1", "non_torch"] == ["-33.33% (-536,870,912 B)", "fail"]
    assert rows["summary law", "extrapolated"] == [json.dumps(SPECIES), "not gated"]
    assert rows["summary p97", "speed"] == [
        "20.0x: 900 simulated s in 45 wall s",
        "meets 5x",
    ]


def test_a_refusal_in_either_simulated_run_makes_the_cell_a_coverage_report(
    capsys, tmp_path
):
    # A clock refusal refuses no step; it is counted under its source, not as a step.
    rows = _rows(capsys, tmp_path, refusals=[("clock", 1)], speed=False)
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary p97", "refusals"] == [
        '1 {"clock": 1}; cost refused 0.0% of 20 steps',
        "coverage report",
    ]
    assert rows["summary law", "refusals"] == [
        "0 {}; cost refused 0.0% of 20 steps",
        "none",
    ]
    assert rows["summary p97", "speed"] == [
        "no wall time was measured: 900 simulated s in 0 wall s",
        "no result for 5x",
    ]


def test_runs_of_different_workloads_are_not_one_cell(capsys, tmp_path):
    with pytest.raises(ValueError, match="ran different workloads"):
        _rows(capsys, tmp_path, workload=2)


def test_a_cost_refusal_in_the_law_run_alone_makes_the_cell_a_coverage_report(
    capsys, tmp_path
):
    rows = _rows(capsys, tmp_path, law=[("cost", 2)])
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary law", "refusals"] == [
        '2 {"cost": 2}; cost refused 10.0% of 20 steps',
        "coverage report",
    ]


def test_counts_but_preemptions_and_prefill_tokens_are_not_gated(capsys, tmp_path):
    # A decode step of another request splits the law run's prefill chunks: one
    # prefill streak and one decode token more than every real run.
    steps = _steps()
    steps.insert(5, ([(1, 2000, True)], 1))
    rows = _rows(capsys, tmp_path, law_run={"steps": steps, "idle_after": {4, 5}})
    assert rows["counts", "prefill streaks"] == [
        "spread 0; result 1 1 1; moved 1",
        "outside, admissible, not gated",
    ]
    assert rows["counts", "decode tokens"][1] == "outside, admissible, not gated"
    assert rows["counts", "prefill tokens"][1] == "exact, inadmissible"
    assert rows["counts", "preemptions"][1] == "exact, inadmissible"


def test_a_run_that_ended_without_its_memory_reads_refused_and_the_cell_goes_on(
    capsys, tmp_path
):
    whole = _rows(capsys, tmp_path / "whole")
    _rows(capsys, tmp_path, law=[("ended", 1)])
    law = tmp_path / "c4" / "law"
    ended = {"refusals": ["ended:engine raised RuntimeError('the step raised')"]}
    (law / LP_FILE.format("engine")).write_text(json.dumps(ended))
    summary = json.loads((law / "summary.json").read_text())
    summary["schedule"].update(analytical=None, extrapolated=None)
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
    assert rows["summary law", "extrapolated"] == ["null", "not gated"]
    assert rows["summary p97", "speed"][1] == "meets 5x"
