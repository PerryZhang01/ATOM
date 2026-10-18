# SPDX-License-Identifier: MIT
"""A later PP stage posts its receive `W_recv` into its launch, and a sending
stage that blocked on its previous send's gloo parts goes on gloo's wake after
they complete."""

import json
import math
import re
from types import SimpleNamespace

import numpy as np
import pytest
from test_backend_step_linear_fit import _steps, _timed_record
from test_dp_step_max import prefill
from test_pp_carriers import SPEC
from test_pp_launch_send import LAWS, _runner
from test_pp_send import ACK, DATA, L_DATA, C, _sent, _stage

from atom.compass import run as compass_run
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.step_linear_fit import fit_host, label_records
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import (
    HOST_SEGMENTS,
    PP_RECV_SEGMENT,
    HostDeviceRecurrence,
    forward_shape,
)
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.utils import clock

#: Seconds on (1, tokens, seqs) from the launch to the receive, per output kind.
RECV = {"decode": [6e-5, 1e-7, 0.0], "prefill": [9e-5, 2e-9, 2e-5]}


def test_a_later_stages_record_fits_its_launch_up_to_the_receive(tmp_path):
    record = _timed_record(tmp_path / "pp1.dp0.jsonl", _steps(14, n=400), {60, 201})[0]
    lines = [json.loads(x) for x in record.read_text().splitlines()]
    for s in lines:
        kind = "decode" if s["rows"][0][2] else "prefill"
        x = (1, sum(r[0] for r in s["rows"]), len(s["rows"]))
        r0 = s["t_launch_ns"][0] + round(float(np.dot(RECV[kind], x)) * 1e9)
        s["t_pp_recv_ns"] = [r0, r0 + 10**6]  # a 1 ms receive, its entry labels it
    record.write_text("".join(json.dumps(s) + "\n" for s in lines))
    host, report = fit_host(label_records([record])[0])
    assert report["unfitted"] == {}
    for kind, c in RECV.items():
        got = host["laws"][kind][PP_RECV_SEGMENT]
        assert got == pytest.approx(c, rel=1e-4, abs=1e-12)
    compass_run._check_host("run.json", host)


def test_a_later_stage_posts_its_receive_its_launch_part_after_its_prepare_end(
    monkeypatch,
):
    """`W_recv` moves the post past the prepare end; skipped, it is 0."""
    stage = _runner(monkeypatch, 1, lambda: 0.0)
    recv = 2.0**-9
    laws = {"prefill": {**LAWS["prefill"], PP_RECV_SEGMENT: (recv, 0, 0)}}
    shape = forward_shape(prefill(8))
    e, pre = LAWS["prefill"]["E"][0], LAWS["prefill"]["W_pre"][0]
    for skip, posted in (((), e + pre + recv), ((PP_RECV_SEGMENT,), e + pre)):
        stage.compass_host = HostDeviceRecurrence(laws, skip, lambda: 0.0, (1, 2))
        assert stage.pp_recv_posted(0.0, *shape)[0] == posted


def test_a_later_stage_whose_law_lacks_its_receive_part_is_refused_by_name():
    """The first stage runs no `W_recv`; a later one needs a law for it or a skip."""
    laws = {"prefill": dict.fromkeys(HOST_SEGMENTS, (0, 0, 0))}
    for stage, skip in (((0, 2), ()), ((1, 2), (PP_RECV_SEGMENT,))):
        HostDeviceRecurrence(laws, skip, lambda: 0.0, stage).charge(
            prefill(8), 1.0, 0.0
        )
    with pytest.raises(
        CostRefused, match=re.escape("(host: no law for prefill.W_recv)")
    ):
        HostDeviceRecurrence(laws, (), lambda: 0.0, (1, 2)).charge(prefill(8), 1.0, 0.0)


def test_a_sender_blocked_on_its_previous_sends_ack_goes_on_its_wake_after_it():
    """Ack 0 is released inside the second forward's compute: no wait, no wake.
    Ack 1 comes at 4 s, 4 - 3C after the third forward's launch end: the loop
    goes on `W (1 - exp(-b / rise))` later and sends then."""
    wake = (2.0**-10, 2.0**-3)
    late = wake[0] * (1 - math.exp(-(4.0 - 3 * C) / wake[1]))
    rt, run, sends, _ = _stage(
        0,
        (C, {}),
        (2 * C, {ACK: [(0, C + 2.0**-7)]}),
        (3 * C, {}),
        (4.0, {ACK: [(1, 4.0)]}),
        (4.0 + late, {}),
        frames=[
            {"seq": 0, "tensors_done": -math.inf},
            {"seq": 1, "tensors_done": -math.inf},
        ],
    )
    sends.wake = wake
    for _ in range(3):
        run("forward")
    assert _sent(rt) == [
        ("TAR", C, []),
        ("TAR", 2 * C, [(DATA, 0, C + L_DATA)]),
        ("TAR", 3 * C, [(DATA, 1, 2 * C + L_DATA)]),
        ("NER", math.inf, []),
        ("TAR", 4.0 + late, []),
    ]
    assert rt.send_log == [(DATA, 2, 4.0 + late + L_DATA)]


def test_a_stage_takes_gloos_wake_from_the_machine_spec(monkeypatch):
    monkeypatch.setattr(clock, "_installed", _stage(0)[0])
    engine = SimpleNamespace(
        runner_mgr=SimpleNamespace(call_func=None),
        pp_rank=0,
        pp_size=2,
        has_pending_kv_work=lambda: False,
    )
    document = json.loads(SPEC.read_text())
    pp_send.install(engine, MachineSpec.from_mapping(document))
    gloo = document["interconnect"]["intra_node"]["pp"]["gloo"]
    assert engine.runner_mgr.call_func.wake == (gloo["wake_s"], gloo["wake_rise_s"])
    del gloo["wake_rise_s"]
    with pytest.raises(SpecRefusal, match="wake_rise_s"):
        pp_send.install(engine, MachineSpec.from_mapping(document))
