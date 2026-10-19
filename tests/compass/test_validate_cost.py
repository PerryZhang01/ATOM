# SPDX-License-Identifier: MIT
"""Cost accuracy: a known law priced on a record whose device windows it wrote.

Each record is the JSON lines a real runner writes, each step's device busy
time (`d_busy_ns`) the law's price, times a stretch where one is planted. It
is read by `parity.read` and priced by `StepLinearBackend.price`.
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
        "held_out": {"hull": {"facets": BOX + [[0, 0, -1, 0], [0, 0, 1, -1e7]]}},
    },
    "decode": {
        "1": {
            "coefficients": {"step": 0.02, "context": 1e-7, "graph_padding": 0.0},
            "hull": {"equal": {"graph_padding": 0}, "facets": [[-1, 100], [1, -1e5]]},
            "held_out": {
                "hull": {
                    "equal": {"graph_padding": 0},
                    "facets": [[-1, 100], [1, -1e5]],
                }
            },
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


def _record(directory, steps, stretch=None, idle_after=(), rank=0, group=None):
    """Write `steps` as rank `rank`'s record. A step in `idle_after` shares no request with the next.
    With `group`, every rank's steps, a step runs for the most any rank's batch costs.
    """
    directory.mkdir(exist_ok=True)
    stretch = stretch or {}
    t, lines, owner = 0, [], 0
    for i, (rows, rung) in enumerate(steps):
        if i - 1 in idle_after:
            owner += 1
        at = [g[i][0] for g in group or [steps] if i < len(g)]
        busy = round(max(map(_price, at)) * stretch.get(i, 1.0) * 1e9)
        lines.append(
            {
                "step": i,
                "dp_rank": rank,
                "moe_rows": sum(q for r in at for q, _, _ in r),
                "is_dummy_run": False,
                "produces_output": True,
                "batch": [[owner, q, n] for q, n, _ in rows],
                "named": [],
                "rows": [list(r) for r in rows],
                "rung": rung,
                "t_enter_ns": t,
                "t_launch_ns": [t, t],
                "t_recv_ns": [t, t],
                "d_busy_ns": busy,
                "t_exit_ns": t + busy,
            }
        )
        t += busy
    (directory / f"dp{rank}.jsonl").write_text(
        "".join(json.dumps(x) + "\n" for x in lines)
    )
    return directory


def test_the_step_that_moves_the_total_is_held_out_and_the_spread_reported(tmp_path):
    # Step 4 ran in a fifth of its price: it moves the total most, and step 9 is longest.
    record = _record(tmp_path / "steps", _steps(), stretch={4: 0.2})
    report = cost_accuracy(record, LAW)

    assert report["cut"] == {"exposed": 0, "mixed": 0, "unsplit": 0}
    assert report["unvalidated"] == {} and report["refused"] == {}
    prefill = report["validated"]["prefill"]
    assert prefill["steps"] == 10
    assert prefill["on_totals"] > 0.05
    assert prefill["held_out_step"] == 4
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert prefill["median"] == pytest.approx(0.0, abs=1e-9)
    low = prefill["bands"]["prefill tokens 0-512"]
    assert low["steps"] == 5 and low["within_10pct"] == 0.8
    assert low["p90"] > 0.2 and low["p50"] == pytest.approx(0.0, abs=1e-9)
    assert prefill["bands"]["prefill tokens 512-2048"]["within_10pct"] == 1.0
    decode = report["validated"]["decode"]
    assert decode["steps"] == 10 and set(decode["bands"]) == {"decode rung 1"}
    assert decode["real_s"] == pytest.approx(decode["priced_s"], rel=1e-9)

    # Two under-priced steps after one at its price, so neither leads: step 5
    # is short by the most seconds, step 1 by the largest ratio.
    record = _record(tmp_path / "under", _steps(), stretch={1: 2.2, 5: 1.8})
    assert cost_accuracy(record, LAW)["validated"]["prefill"]["held_out_step"] == 5


def test_unvalidated_and_refused_steps_are_reported_apart(tmp_path):
    steps = _steps()
    wide = ([(5000, 5000, False)], None)  # tokens beyond the hull's 1000
    rung2 = ([(1, 3000, True), (1, 4000, True)], 2)  # no vector for rung 2
    steps[3:3] = [wide]
    steps[14:14] = [rung2]
    record = _record(tmp_path / "steps", steps, stretch={3: 2.0})
    report = cost_accuracy(record, LAW)

    assert report["cut"] == {"exposed": 0, "mixed": 0, "unsplit": 0}
    (unvalidated,) = report["unvalidated"].values()
    assert unvalidated["steps"] == 1
    assert unvalidated["real_s"] == pytest.approx(2 * _price(wide[0]), rel=1e-9)
    assert unvalidated["on_totals"] == pytest.approx(-0.5, rel=1e-9)
    assert unvalidated["bands"]["prefill tokens 2048-8192"]["within_10pct"] == 0.0
    fitted = report["validated"]["prefill"]
    assert fitted["steps"] == 10
    assert fitted["on_totals"] == pytest.approx(0.0, abs=1e-9)
    assert report["refused"] == {
        "decode rung 2": {
            "steps": 1,
            "real_s": pytest.approx(_price(rung2[0])),
            "reasons": {"step-linear: no decode vector for rung 2": 1},
        }
    }


def test_the_command_prints_the_report(tmp_path, capsys):
    record = _record(tmp_path / "steps", _steps(), stretch={4: 0.2})
    law = tmp_path / "law.json"
    law.write_text(json.dumps(LAW))
    main([str(record), str(law)])

    out = capsys.readouterr().out
    assert "holding out step 4 " in out and "law planted" in out


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
            "real_s": pytest.approx(40 * _price(steps[5][0]), rel=1e-6),
            "priced_s": pytest.approx(_price(steps[5][0]), rel=1e-9),
        }
    ]
    prefill = report["validated"]["prefill"]
    assert prefill["steps"] == 9 and prefill["held_out_step"] == 6
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert report["unvalidated"] == {}

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
    assert report["warm"][0]["real_s"] == pytest.approx(
        2 * _price(steps[0][0]), rel=1e-6
    )
    prefill, decode = report["validated"]["prefill"], report["validated"]["decode"]
    assert prefill["steps"] == 8 and prefill["held_out_step"] == 4
    assert prefill["held_out"] == pytest.approx(0.0, abs=1e-9)
    assert decode["steps"] == 9
    assert decode["real_s"] == pytest.approx(decode["priced_s"], rel=1e-9)

    # A leading step that ran in half its price is fast, not warm: it stays in the totals.
    report = cost_accuracy(_record(tmp_path / "fast", steps, {10: 0.5}), LAW)
    assert report["warm"] == []
    assert report["validated"]["decode"]["steps"] == 10


def test_a_leading_slow_step_is_judged_by_the_law_s_scatter_not_the_record_s(
    tmp_path,
):
    # Prefill scatters by 25% after step 0, which ran at 1.5 times its price.
    # Against this record's own scatter step 0 is ordinary; against the narrow
    # scatter the law was fitted on, it leads slow.
    stretch = {i: (1.25, 0.75)[i % 2] for i in range(1, 10)} | {0: 1.5}
    record = _record(tmp_path / "steps", _steps(), stretch)
    assert cost_accuracy(record, LAW)["warm"] == []

    law = json.loads(json.dumps(LAW))
    law["prefill"]["fit"] = {"centre": 0.0, "mad": 0.01}
    assert [w["step"] for w in cost_accuracy(record, law)["warm"]] == [0]


def test_a_law_without_held_out_errors_reports_every_priced_step_unvalidated(tmp_path):
    record = _record(tmp_path / "steps", _steps())
    bare = json.loads(json.dumps(LAW))
    for vector in (bare["prefill"], bare["decode"]["1"]):
        del vector["held_out"]
    report = cost_accuracy(record, bare)

    assert report["validated"] == {}
    assert {k: v["steps"] for k, v in report["unvalidated"].items()} == {
        "prefill": 10,
        "decode": 10,
    }


def test_the_report_labels_the_record_and_never_reads_the_table_beside_it(tmp_path):
    record = _record(tmp_path / "steps", _steps(), stretch={4: 0.2})
    report = cost_accuracy(record, LAW)

    (record / "dp0.cost.jsonl").write_text('{"version": 1}\nnot a row\n')
    assert cost_accuracy(record, LAW) == report


def _bigger(rows):
    """`rows` with more to price: longer prefill chunks, longer decode contexts."""
    return [(1, n + 1000, d) if d else (q + 50, n + 50, d) for q, n, d in rows]


def _two_ranks():
    """Rank 1 has the bigger batch on even steps, rank 0 on odd ones."""
    small = _steps()
    big = [(_bigger(rows), rung) for rows, rung in small]
    return [
        [(big, small)[i % 2 == 0][i] for i in range(20)],
        [(small, big)[i % 2 == 0][i] for i in range(20)],
    ]


def test_a_group_step_prices_as_its_slowest_rank_against_the_group_s_time(tmp_path):
    ranks = _two_ranks()
    for r, steps in enumerate(ranks):
        _record(tmp_path / "steps", steps, rank=r, group=ranks)
    report = cost_accuracy(tmp_path / "steps", LAW)

    assert report["cut"] == {"exposed": 0, "mixed": 0, "unsplit": 0, "ranks": 0}
    assert report["warm"] == [] and report["refused"] == {}
    for kind in ("prefill", "decode"):
        s = report["validated"][kind]
        assert s["steps"] == 10
        assert s["on_totals"] == pytest.approx(0.0, abs=1e-9)
        assert s["median"] == pytest.approx(0.0, abs=1e-9)
    # Every step ran for the slower rank's price, which neither rank alone matches.
    slowest = sum(max(_price(a[0]), _price(b[0])) for a, b in zip(*ranks))
    total = sum(report["validated"][k]["real_s"] for k in ("prefill", "decode"))
    assert total == pytest.approx(slowest, rel=1e-9)
    assert all(total > sum(_price(rows) for rows, _ in s) for s in ranks)

    # Rank 1's step 10 lies outside the held-out hull, though rank 0 prices it.
    ranks[1][10] = ([(1, 50, True)], 1)
    for r, steps in enumerate(ranks):
        _record(tmp_path / "outside", steps, rank=r, group=ranks)
    report = cost_accuracy(tmp_path / "outside", LAW)
    assert report["unvalidated"]["decode"]["steps"] == 1


def test_ranks_differing_by_more_than_a_trailing_step_refuse_naming_their_lengths(
    tmp_path,
):
    ranks = _two_ranks()
    stopped = [ranks[0], ranks[1][:-1]]  # rank 1's last step was in flight
    for r, steps in enumerate(stopped):
        _record(tmp_path / "stopped", steps, rank=r, group=stopped)
    report = cost_accuracy(tmp_path / "stopped", LAW)
    assert report["cut"]["ranks"] == 1
    assert report["validated"]["decode"]["steps"] == 9

    apart = [ranks[0], ranks[1][:-2]]
    for r, steps in enumerate(apart):
        _record(tmp_path / "apart", steps, rank=r, group=apart)
    with pytest.raises(ValueError, match=r"'dp0.jsonl': 20, 'dp1.jsonl': 18"):
        cost_accuracy(tmp_path / "apart", LAW)


def test_a_group_step_is_banded_by_its_costliest_rank_not_rank_0(tmp_path):
    # Rank 1, the costliest on its 9.95e6 of history, runs 500 tokens; rank 0
    # runs 515 and rank 2, the last, 520 tokens, both cheaper and in band 512-2048.
    ranks = [_steps(), _steps(), _steps()]
    for r, (q, n) in enumerate([(515, 515), (500, 20400), (520, 520)]):
        ranks[r][:10] = [([(q, n, False)], None)] * 10
    for r, steps in enumerate(ranks):
        _record(tmp_path / "steps", steps, rank=r, group=ranks)

    report = cost_accuracy(tmp_path / "steps", LAW)
    prefill = report["validated"]["prefill"]["bands"]
    assert {band: b["steps"] for band, b in prefill.items()} == {
        "prefill tokens 0-512": 10
    }

    # A law with no prefill vector refuses them, banded by the rank of most tokens,
    # not rank 0, once the costliest is rank 0.
    ranks[:2] = ranks[1::-1]
    for r, steps in enumerate(ranks):
        _record(tmp_path / "swapped", steps, rank=r, group=ranks)
    report = cost_accuracy(
        tmp_path / "swapped", {k: v for k, v in LAW.items() if k != "prefill"}
    )
    assert {band: r["steps"] for band, r in report["refused"].items()} == {
        "prefill tokens 512-2048": 10
    }
