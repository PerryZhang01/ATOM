# SPDX-License-Identifier: MIT
"""A pipeline send's RCCL parts and the device streams they wait on: the
sender's stream holds every forward after the wait for that send at the RCCL
parts' completion, and the receiver's stream takes them only after its own
device work, as ATOM's ``Work.wait`` and ``torch.distributed.recv`` do on the
device group.

The stage loops are real `StageSends` on an `LPRuntime` whose clock authority
is the scripted fake connection of the LP runtime tests; the first two drive
the runner's own forward and host recurrence.
"""

import dataclasses
from types import SimpleNamespace

import pytest
import test_pp_launch_send as launch
from test_dp_step_max import prefill
from test_pp_launch_send import CARRIER, LAUNCH, POST, PRE, E
from test_pp_stage_input import ACKED, DATA, SIZE, C, T, _decode
from test_pp_stage_input import _stage as fake_stage

from atom.compass.runner import pp_send

#: Every part rendezvous, as on a machine whose eager thresholds are 0.
ALL_RDV = {
    c: dataclasses.replace(CARRIER, eager_threshold_bytes=0) for c in pp_send.CARRIERS
}
ACK = "engine-pp1->engine-pp0:pp_ack#dp0"


def _stage(monkeypatch, *args, **kw):
    """`test_pp_launch_send`'s stage, and its runner."""
    made, real = [], launch._runner
    monkeypatch.setattr(launch, "_runner", lambda *a: made.append(real(*a)) or made[-1])
    return *launch._stage(monkeypatch, *args, **kw), made[0]


@pytest.mark.parametrize("pp", [2, 4])
def test_a_sending_stage_starts_its_device_work_after_the_send_it_waited_for(
    monkeypatch, pp
):
    """The stage after acks send 0 with its RCCL parts done at `held`. The
    second forward runs before that ack is read, so only the third's device
    work starts there; its host does not wait."""
    held, launch_end = 4.0, E + PRE + LAUNCH
    t1 = launch_end + POST
    ta = t1 + launch_end + 2.0**-3  # ack 0 arrives during the second send's wait
    t2 = ta + POST
    tb = t2 + launch_end + 2.0**-3
    rt, run, sent, _ = _stage(
        monkeypatch,
        0,
        (launch_end, {}),
        (t1, {}),
        (t1 + launch_end, {}),
        (ta, {ACK: [(0, ta)]}),
        (t2, {}),
        (t2 + launch_end, {}),
        (tb, {ACK: [(1, tb)]}),
        (tb + POST, {}),
        frames=[{"seq": 0, "tensors_done": held}, {"seq": 1, "tensors_done": 5.0}],
        pp=pp,
        models=ALL_RDV,
    )
    batches = [prefill(8) for _ in range(3)]
    for batch in batches:
        run("forward", batch)
    assert [f["t_ready"] for f in sent[:2]] == [launch_end, t1 + launch_end]
    assert held < sent[2]["t_ready"] < held + LAUNCH
    assert batches[2].compass_pp_release == held
    assert not hasattr(batches[1], "compass_pp_release")
    assert ("TAR", t2 + launch_end) in [m[:2] for m in rt.conn.sent]


@pytest.mark.parametrize("pp", [2, 4])
def test_a_receiving_stage_takes_a_rendezvous_tensor_after_its_own_device_work(
    monkeypatch, pp
):
    """The last stage's device is busy until `busy` with an earlier forward:
    the tensor, rendezvous, moves after it, and the ack of the metadata, still
    stamped at the metadata's completion, carries the tensor's."""
    parts = [("gloo", 8), ("gloo", 64), ("rccl", 512)]
    frame = {"seq": 0, "t_send": 0.0, "t_ready": 0.0, "parts": parts}
    done, meta, _ = pp_send.arrivals(ALL_RDV, frame, E + PRE)
    busy = E + PRE + 1.0
    assert meta < busy
    data = f"engine-pp{pp - 2}->engine-pp{pp - 1}:pp_data#dp0"
    ack = f"engine-pp{pp - 1}->engine-pp{pp - 2}:pp_ack#dp0"
    rt, run, sent, stage = _stage(
        monkeypatch,
        pp - 1,
        (launch.L_DATA, {data: [(0, launch.L_DATA)]}),
        (4.0, {}),
        frames=[frame],
        pp=pp,
        models=ALL_RDV,
    )
    stage.compass_host.charge(prefill(8), 1.0, 0.0)
    assert stage.compass_host.ends[-1] == busy
    batch = prefill(8)
    run("forward", batch)
    tensor = busy + ALL_RDV["rccl"].rendezvous_s(512)
    assert batch.compass_pp_release == tensor
    assert sent == [{"seq": 0, "tensors_done": tensor}]
    assert rt.conn.sent[1][2] == [(ack, 0, done)]  # stamped where it was


@pytest.mark.parametrize("pp", [2, 4])
def test_an_eager_tensor_holds_the_stream_at_its_local_end_and_an_unacked_one_refuses(
    pp,
):
    """With no ack, an eager tensor's completion is the sender's own; a
    rendezvous one depends on the receiver, so the next forward is refused."""
    eager = [("gloo", 8), ("rccl", 512)]
    rt, run, _ = fake_stage(0, (C, {}), (C + SIZE, {}), ready=4 * C, parts=eager, pp=pp)
    run("forward", SimpleNamespace())
    run("flush_pp_send")
    batch = SimpleNamespace()
    rt.conn.replies.append((5 * C, {}))
    run("forward", batch)
    assert batch.compass_pp_release == 4 * C + 2.0**-12 + 512 * 2.0**-20

    rdv = [("gloo", 8), ("rccl", 2048)]
    rt, run, _ = fake_stage(0, (C, {}), (C + SIZE, {}), ready=4 * C, parts=rdv, pp=pp)
    run("forward", SimpleNamespace())
    run("flush_pp_send")
    with pytest.raises(
        RuntimeError, match="no ack says when the last send.s rccl parts end"
    ):
        run("forward", SimpleNamespace())


def test_a_middle_stage_takes_its_input_after_the_send_its_stream_waited_for():
    """Stage 1 of four: the idle flush reads the ack of its send, whose RCCL
    parts end at `held`; its next input's tensor, rendezvous, waits for them
    on the same stream, and so does its forward's device work."""
    held = 10.0
    frames = [
        {"seq": i, "t_send": t, "t_ready": t, "parts": ACKED}
        for i, t in enumerate((0.25, 1.0))
    ]
    ack_in = "engine-pp2->engine-pp1:pp_ack#dp0"
    _, run, calls = fake_stage(
        1,
        (0.25, {DATA: [(0, 0.25)]}),
        (0.25 + C, {}),
        (0.5, {ack_in: [(0, 0.5)]}),
        (1.0, {DATA: [(1, 1.0)]}),
        (1.0 + C, {}),
        frames=[frames[0], {"seq": 0, "tensors_done": held}, frames[1]],
        parts=ACKED,
        pp=4,
    )
    run("forward", _decode())
    run("flush_pp_send")
    batch = _decode()
    run("forward", batch)
    # Up since 1.0, the tensor moved its threshold's 1024 bytes before the post.
    end = held - 1024 * 2.0**-20 + T
    assert batch.compass_pp_release == end
    assert calls[-3][2] == {"seq": 1, "tensors_done": end}
