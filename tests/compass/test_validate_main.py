# SPDX-License-Identifier: MIT
"""Grading a cell: `python -m atom.compass.validate` on runs laid out as `cctraces_sim.sh` writes them.

Every run's step record is `test_validate_cost`'s, periods at its planted law's
price; the export and the server log are as `test_validate_runs` and
`test_memory_check` write them. `validate.runs`, `validate.cost` and
`memory.check` read them as they read a real cell.
"""

import json

from test_memory_check import GIB, _log
from test_validate_cost import LAW, _record, _steps
from test_validate_runs import EXPORT

from atom.compass.validate.__main__ import main

TERM = {"basis": "spec", "source": "planted"}
SPECIES = {"steps": 3, "fraction_of_predicted_seconds": 0.4}


def _run(d, ttft, *, refusals=0, non_torch=1.0):
    """One run. `ttft` is (uncached, cached, ms) per profiling request; a warmup
    request with 9000 uncached tokens precedes them. `refusals` None makes the
    run real, a count makes it simulated with that many refusals."""
    d.mkdir(parents=True)
    _record(d / "steps", _steps())
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
    (d / "run.json").write_text(json.dumps({"workload": {"traces": 1}, "law": LAW}))
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
    (d / "memory-dp0.json").write_text(json.dumps(memory))
    summary = {
        "coverage_report": bool(refusals),
        "cost": {
            "speed_ratio": 20.0,
            "simulated_seconds": 900.0,
            "wall_seconds": 45.0,
            "meets_speed_target": True,
            "speed_target_ratio": 5.0,
        },
        "schedule": {
            "refusals": {"count": refusals, "steps": 20},
            "analytical": SPECIES,
            "extrapolated": SPECIES,
        },
    }
    (d / "summary.json").write_text(json.dumps(summary))


def _rows(capsys, tmp_path, p97_refusals=0):
    for i, (ms, non_torch) in enumerate(((100, 1.0), (110, 1.5), (105, 1.0))):
        _run(
            tmp_path / "c4" / f"real{i}",
            [(100, 1000, ms), (600, 0, 1.0)],
            refusals=None,
            non_torch=non_torch,
        )
    _run(tmp_path / "c4" / "law", [(100, 1000, 120.0)])
    _run(tmp_path / "c4" / "p97", [(100, 1000, 118.0)], refusals=p97_refusals)
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
    rows = _rows(capsys, tmp_path, p97_refusals=1)
    assert rows["cell", "verdict"] == ["3 real runs", "coverage report"]
    assert rows["summary p97", "refusals"] == ["1 of 20 steps", "coverage report"]
