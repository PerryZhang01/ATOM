# SPDX-License-Identifier: MIT
"""The recurrence's own check, on a hand-built record of one request.

Two middle chunks, the output prefill, then two decodes, each 10 ms of device
work and 1 ms per host segment. The stamps are what the recurrence's wait
rules give: the third forward's gate waits for the first one's device end,
and each decode waits to receive the previous output forward's.
"""

import json

import pytest

from atom.compass.parity import recurrence

MS = 1_000_000


def line(step, rows, output, enter, gate, launch, recv, exit_):
    s = {
        "step": step,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": output,
        "batch": [[7, rows[0][0], rows[0][1]]],
        "rows": rows,
        "t_enter_ns": enter * MS,
        "t_gate_ns": [g * MS for g in gate],
        "t_launch_ns": [t * MS for t in launch],
        "t_exit_ns": exit_ * MS,
        "d_busy_ns": 10 * MS,
    }
    if recv:
        s["t_recv_ns"] = [t * MS for t in recv]
    return s


def record(r1=41):
    """The last decode's recv wait ends at `r1` ms; 41 is step 3's device end."""
    return [
        line(0, [[50, 50, False]], False, 0, (0, 0), (1, 2), None, 3),
        line(1, [[50, 100, False]], False, 4, (4, 4), (5, 6), None, 7),
        line(2, [[50, 150, False]], True, 8, (8, 11), (12, 13), (13, 13), 14),
        line(3, [[1, 151, True]], True, 15, (15, 21), (22, 23), (23, 31), 32),
        line(4, [[1, 152, True]], True, 33, (33, 33), (34, 35), (35, r1), r1 + 1),
    ]


def test_measured_inputs_reproduce_a_record_the_wait_rules_wrote():
    report = recurrence.check(record())
    assert set(report["kinds"]) == {"idle", "middle", "prefill", "decode"}
    for n, p10, p50, p90, r1, ok in report["kinds"].values():
        assert (p10, p90) == pytest.approx((0, 0), abs=1e-9) and ok
    assert report["kinds"]["decode"][4] == pytest.approx(0, abs=1e-6)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]


def test_a_late_r1_release_fails_its_kind_and_names_the_step():
    report = recurrence.check(record(r1=45))
    assert not report["kinds"]["decode"][-1]
    assert report["kinds"]["prefill"][-1]
    assert report["total"] == pytest.approx(100 * (42 / 46 - 1))
    assert not report["pass"]
    # step 4 launches at 34 behind step 3's device work, which ends at 41
    [(step, _, err, residual, queued, r1)] = report["failures"]
    assert (step, err) == (4, pytest.approx(100 * (10 / 14 - 1)))
    assert (residual, queued, r1) == pytest.approx((-4, 7, -4))


def test_the_script_reads_a_record_directory(tmp_path, capsys):
    (tmp_path / "dp0.jsonl").write_text(
        "".join(json.dumps(s | {"named": []}) + "\n" for s in record(r1=45))
    )
    assert recurrence.main(str(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "decode: 1 steps outside the band" in out
    assert "step 4: -28.57%" in out


def test_every_kind_can_pass_while_the_run_total_still_fails():
    """Five decodes, each receiving 0.1 ms later than the record's first two:
    within every kind's step band, but their sum breaches the tighter total one."""
    steps, exit_ = record()[:4], 32
    for i in range(5):
        enter, launch = exit_ + 1, (exit_ + 2, exit_ + 3)
        recv_end = enter + 8.1
        steps.append(
            line(
                4 + i,
                [[1, 152 + i, True]],
                True,
                enter,
                (enter, enter),
                launch,
                (launch[1], recv_end),
                recv_end + 1,
            )
        )
        exit_ = recv_end + 1
    report = recurrence.check(steps)
    assert all(k[-1] for k in report["kinds"].values()) and not report["failures"]
    assert abs(report["total"]) > recurrence.TOTAL_BAND
    assert not report["pass"]


def test_a_record_without_device_windows_is_refused():
    steps = record()
    del steps[2]["d_busy_ns"]
    with pytest.raises(ValueError, match="step 2 has no device window"):
        recurrence.check(steps)


def test_a_decode_charges_the_device_gap_before_it_and_a_prefill_does_not():
    """A 4 ms gap before step 3 ends its device work at 45 ms, where step 4
    receives it; the same gap before the output prefill moves nothing."""
    steps = record(r1=45)
    steps[2]["d_gap_ns"] = steps[3]["d_gap_ns"] = 4 * MS
    report = recurrence.check(steps)
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert report["pass"] and not report["failures"]
