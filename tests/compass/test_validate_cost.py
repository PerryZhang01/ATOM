# SPDX-License-Identifier: MIT
"""Cost accuracy: a known law priced on a record whose periods it wrote.

Each record is the JSON lines a real runner writes, its `t_enter_ns` laid out
so each step's period is the law's price, times a stretch where one is
planted. It is read by `parity.read` and priced by `CoarseBackend.estimate`.
"""

import json

import pytest

from atom.compass.validate.cost import cost_accuracy, main

# Prefill hull: tokens 100-1000, Sum N_Q^2 1e4-1e6, Sum N_Q (N_KV - N_Q) 0-1e7.
BOX = [[-1, 0, 0, 100], [1, 0, 0, -1000], [0, -1, 0, 1e4], [0, 1, 0, -1e6]]
LAW = {
    "provenance": "planted",
    "prefill": {
        "coefficients": {
            "step": 0.02,
            "tokens": 1e-4,
            "query_square": 1e-8,
            "query_cached": 1e-9,
        },
        "hull": {"facets": BOX + [[0, 0, -1, 0], [0, 0, 1, -1e7]]},
    },
    "decode": {
        "1": {
            "coefficients": {"step": 0.02, "context": 1e-7, "graph_padding": 0.0},
            "hull": {"equal": {"graph_padding": 0}, "facets": [[-1, 100], [1, -1e5]]},
        }
    },
}


def _price(rows):
    if rows[0][2]:
        return 0.02 + 1e-7 * sum(n for _, n, _ in rows)
    return sum(1e-4 * q + 1e-8 * q * q + 1e-9 * q * (n - q) for q, n, _ in rows) + 0.02


def _steps():
    """Ten prefill chunks of one request, then ten decode steps of it."""
    out = [
        ([(q, q + 900 * i, False)], None) for i, q in enumerate(range(100, 1000, 90))
    ]
    return out + [([(1, 2000 + 5000 * i, True)], 1) for i in range(10)]


def _record(directory, steps, stretch=None, idle_after=()):
    """Write `steps` as rank 0's record. A step in `idle_after` shares no request with the next."""
    directory.mkdir()
    stretch = stretch or {}
    t, lines, owner = 0, [], 0
    for i, (rows, rung) in enumerate(steps):
        if i - 1 in idle_after:
            owner += 1
        lines.append(
            {
                "step": i,
                "dp_rank": 0,
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [[owner, q, n] for q, n, _ in rows],
                "named": [],
                "rows": [list(r) for r in rows],
                "rung": rung,
                "t_enter_ns": t,
                "t_exit_ns": t + 1,
            }
        )
        t += round(_price(rows) * stretch.get(i, 1.0) * 1e9)
    (directory / "dp0.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return directory


def test_the_step_that_moves_the_total_is_held_out_and_the_spread_reported(tmp_path):
    # Step 4 ran in a fifth of its price: it moves the total most, and step 9 is longest.
    record = _record(tmp_path / "steps", _steps(), stretch={4: 0.2})
    report = cost_accuracy(record, LAW)

    assert report["cut"] == {"idle": 0, "mixed": 0, "last": 1}
    assert report["extrapolated"] == {} and report["refused"] == {}
    prefill = report["fitted"]["prefill"]
    assert prefill["steps"] == 10
    assert prefill["on_totals"] > 0.05
    assert prefill["held_out_step"] == 4
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert prefill["median"] == pytest.approx(0.0, abs=1e-9)
    low = prefill["bands"]["prefill tokens 0-512"]
    assert low["steps"] == 5 and low["within_10pct"] == 0.8
    assert low["p90"] > 0.2 and low["p50"] == pytest.approx(0.0, abs=1e-9)
    assert prefill["bands"]["prefill tokens 512-2048"]["within_10pct"] == 1.0
    decode = report["fitted"]["decode"]
    assert decode["steps"] == 9 and set(decode["bands"]) == {"decode rung 1"}
    assert decode["measured_s"] == pytest.approx(decode["priced_s"], rel=1e-9)

    # Two under-priced steps: step 5 is short by the most seconds, step 0 by the largest ratio.
    record = _record(tmp_path / "under", _steps(), stretch={0: 2.2, 5: 1.5})
    assert cost_accuracy(record, LAW)["fitted"]["prefill"]["held_out_step"] == 5


def test_extrapolated_and_refused_steps_are_reported_apart(tmp_path):
    steps = _steps()
    wide = ([(5000, 5000, False)], None)  # tokens beyond the hull's 1000
    rung2 = ([(1, 3000, True), (1, 4000, True)], 2)  # no vector for rung 2
    steps[3:3] = [wide]
    steps[14:14] = [rung2]
    record = _record(tmp_path / "steps", steps, stretch={3: 2.0}, idle_after={8})
    report = cost_accuracy(record, LAW)

    assert report["cut"] == {"idle": 1, "mixed": 0, "last": 1}
    (extrapolated,) = report["extrapolated"].values()
    assert extrapolated["steps"] == 1
    assert extrapolated["measured_s"] == pytest.approx(2 * _price(wide[0]), rel=1e-9)
    assert extrapolated["on_totals"] == pytest.approx(-0.5, rel=1e-9)
    assert extrapolated["bands"]["prefill tokens 2048-8192"]["within_10pct"] == 0.0
    fitted = report["fitted"]["prefill"]
    assert fitted["steps"] == 9  # step 9, after the idle gap, is fitted
    assert fitted["on_totals"] == pytest.approx(0.0, abs=1e-9)
    assert report["refused"] == {
        "decode rung 2": {"steps": 1, "measured_s": pytest.approx(_price(rung2[0]))}
    }


def test_the_command_prints_the_report_and_refuses_two_ranks(tmp_path, capsys):
    record = _record(tmp_path / "steps", _steps(), stretch={4: 0.2})
    law = tmp_path / "law.json"
    law.write_text(json.dumps(LAW))
    main([str(record), str(law)])

    out = capsys.readouterr().out
    assert "holding out step 4 " in out and "law planted" in out
    (record / "dp1.jsonl").write_text((record / "dp0.jsonl").read_text())
    with pytest.raises(ValueError, match="2 DP ranks"):
        cost_accuracy(record, LAW)


def test_a_one_off_slow_step_is_named_warm_mid_record_and_enters_no_total(
    tmp_path, capsys
):
    # Step 5 ran at 40 times its price, a first touch; step 6 at 4 times is scatter.
    steps = _steps()
    record = _record(tmp_path / "steps", steps, stretch={5: 40.0, 6: 4.0})
    report = cost_accuracy(record, LAW)

    assert report["warm"] == [
        {
            "step": 5,
            "band": "prefill tokens 512-2048",
            "measured_s": pytest.approx(40 * _price(steps[5][0]), rel=1e-6),
            "priced_s": pytest.approx(_price(steps[5][0]), rel=1e-9),
        }
    ]
    prefill = report["fitted"]["prefill"]
    assert prefill["steps"] == 9 and prefill["held_out_step"] == 6
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert report["extrapolated"] == {}

    law = tmp_path / "law.json"
    law.write_text(json.dumps(LAW))
    main([str(record), str(law)])
    assert "warm step 5 (prefill tokens 512-2048)" in capsys.readouterr().out


def test_a_leading_slow_step_under_the_factor_is_warm_and_mid_record_is_not(tmp_path):
    # At twice their price: prefill steps 0 and 1 lead the record, step 10
    # leads the decode steps; step 4 ran after prefill steps at their price.
    steps = _steps()
    stretch = {0: 2.0, 1: 2.0, 4: 2.0, 10: 2.0}
    report = cost_accuracy(_record(tmp_path / "steps", steps, stretch), LAW)

    assert [w["step"] for w in report["warm"]] == [0, 1, 10]
    assert report["warm"][0]["measured_s"] == pytest.approx(
        2 * _price(steps[0][0]), rel=1e-6
    )
    prefill, decode = report["fitted"]["prefill"], report["fitted"]["decode"]
    assert prefill["steps"] == 8 and prefill["held_out_step"] == 4
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert decode["steps"] == 8
    assert decode["measured_s"] == pytest.approx(decode["priced_s"], rel=1e-9)
