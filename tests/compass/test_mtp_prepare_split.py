# SPDX-License-Identifier: MIT
"""Under MTP, the prepare after the MTP status wait is its own host segment,
`MTP_SEGMENT`: charged after the wait, labelled and fitted on a record that
stamps the wait, and gated with no device gap fed on an MTP decode."""

import json
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import HOST, _steps, _timed_record
from test_host_device_recurrence import cost_lines, streak  # noqa: F401
from test_parity_recurrence import MS, line

from atom.compass import run as compass_run
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.parity import recurrence
from atom.compass.runner.overrides import (
    FORWARD_KINDS,
    HOST_SEGMENTS,
    MTP_SEGMENT,
    HostDeviceRecurrence,
)

#: Seconds on (1, tokens, seqs) of prepare after the status wait, per output kind.
INPUTS = {"decode": [1e-4, 2e-6, 0.0], "prefill": [2e-4, 3e-8, 3e-5]}


def _three_decodes(laws, mtp):
    """Three decodes' exits, 1 s per host segment unless `laws` says otherwise,
    10 s of target and 4 of draft."""
    seg = {**{s: [1, 0, 0] for s in HOST_SEGMENTS}, **laws}
    clock = [0.0]
    host = HostDeviceRecurrence({"decode": seg}, (), lambda: clock[0], mtp=mtp)
    b = SimpleNamespace(
        total_seqs_num_prefill=0,
        total_tokens_num=2,
        total_seqs_num=1,
        produces_output=lambda: True,
    )
    out = []
    for _ in range(3):
        clock[0] += host.charge(b, 10, 4)
        out.append(clock[0])
    return out


def test_under_mtp_the_prepare_after_the_status_wait_is_charged_after_it():
    """The second forward's wait releases at the first one's target end (12), so
    half a second of its prepare after the wait moves its launch and exit by
    that much."""
    laws = {"W_pre": [0.5, 0, 0], MTP_SEGMENT: [0.5, 0, 0]}
    assert _three_decodes(laws, True) == pytest.approx([4, 14.5, 28.5])


def test_under_mtp_a_law_without_the_segment_is_refused_and_without_mtp_charges_0():
    with pytest.raises(
        CostRefused, match=re.escape(f"no law for decode.{MTP_SEGMENT}")
    ):
        _three_decodes({}, True)
    assert _three_decodes({}, False) == pytest.approx([4, 13, 27])


def test_a_skipped_mtp_segment_is_charged_0_and_listed_skipped(
    streak, tmp_path, monkeypatch  # noqa: F811
):
    seg = {s: [1e-3, 0, 0] for s in (*HOST_SEGMENTS, MTP_SEGMENT)}
    laws = dict.fromkeys(FORWARD_KINDS, seg)
    lines, _ = cost_lines(streak[:4], tmp_path, monkeypatch, laws, [MTP_SEGMENT])
    assert [(x[MTP_SEGMENT], x["skipped"]) for x in lines] == [(0.0, [MTP_SEGMENT])] * 4


def _mtp_record(path):
    """`_timed_record`'s record with an MTP status wait of no width in each
    output forward's prepare, `INPUTS` before its launch."""
    record = _timed_record(path, _steps(14, n=400), {60, 201})[0]
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    for s in lines:
        if s["produces_output"]:
            x = (1, sum(r[0] for r in s["rows"]), len(s["rows"]))
            kind = "decode" if s["rows"][0][2] else "prefill"
            m = s["t_launch_ns"][0] - round(float(np.dot(INPUTS[kind], x)) * 1e9)
            s["t_mtp_ns"] = [m, m]
    record.write_text("".join(json.dumps(s) + "\n" for s in lines))
    return record


def test_an_mtp_record_fits_w_pre_up_to_the_wait_and_the_segment_after_it(tmp_path):
    host, report = fit_host(label_records([_mtp_record(tmp_path / "dp0.jsonl")])[0])
    assert report["unfitted"] == {}
    assert MTP_SEGMENT not in host["laws"]["middle"]
    for kind, c in INPUTS.items():
        laws = host["laws"][kind]
        assert laws[MTP_SEGMENT] == pytest.approx(c, rel=1e-4, abs=1e-12)
        pre = np.subtract(HOST[kind]["W_pre"], c)
        assert laws["W_pre"] == pytest.approx(pre, rel=1e-4, abs=1e-12)
    compass_run._check_host("run.json", {**host, "skip": [MTP_SEGMENT]})

    plain = _timed_record(tmp_path / "plain.jsonl", _steps(14, n=400), {60, 201})[0]
    host = fit_host(label_records([plain])[0])[0]
    assert all(MTP_SEGMENT not in laws for laws in host["laws"].values())


def test_an_mtp_record_with_prepare_after_the_wait_passes_with_no_device_gap_fed():
    """Four MTP decodes of one request, written by the wait rules: 1 ms per host
    segment, the segment after the status wait included, 10 ms of target and
    0.5 of draft. Each target starts at its launch, 0.5 ms after the draft
    before it ends: the device gap the record states, which the prepare after
    the wait already holds."""
    steps = []
    for n, (enter, mtp_end) in enumerate([(0, 1), (5, 12), (16, 23), (27, 34)]):
        launch = (mtp_end + 1, mtp_end + 2)
        s = line(n, [[2, 150 + n, True]], True, enter, (enter,) * 2, launch, None, 0)
        s["t_mtp_ns"] = [(enter + 1) * MS, mtp_end * MS]
        s["t_recv_ns"] = s["t_draft_recv_ns"] = [launch[1] * MS] * 2
        s["t_exit_ns"] = (launch[1] + 1) * MS
        s["d_busy_ns"], s["d_target_ns"] = 10.5 * MS, 10 * MS
        if n:
            s["d_gap_ns"] = MS // 2
        steps.append(s)
    report = recurrence.check(steps)
    assert report["kinds"]["decode"][1:4] == pytest.approx((0, 0, 0), abs=1e-9)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
