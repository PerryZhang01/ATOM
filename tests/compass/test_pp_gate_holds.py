# SPDX-License-Identifier: MIT
"""The PP gate prices a stage's sends as `StageSends` does: a later stage posts
its receive at its prepare end plus `W_recv`, its tensors leave once the
previous frame's are done and are taken once its own device work and its own
send's hold are, and a sending stage's device starts after the tensors of the
send it last waited for."""

import pytest
from test_pp_carriers import MODELS, _late, _shift
from test_pp_stage_input import MS, _line, pp2_record

from atom.compass.parity import recurrence
from atom.compass.runner.pp_send import CarrierModel

FREE = CarrierModel(10**9, 0.0, 0.0, 0.0, 0.0)
#: A frame's two 500-byte tensors: rendezvous, 0.5 ms each in turn.
RDV = CarrierModel(0, 0.0, 0.0, 1e-6, 0.0)


def _step(n, enter, busy, recv=0):
    """Forward `n` of one token, entered at `enter` ms: 1 ms to its launch, a
    launch of `recv` ms to its receive and 1 ms after it, `busy` ms of device
    work, no output."""
    l0 = enter + 1
    s = _line(n, enter, (enter, enter), (l0, l0 + 1 + recv), l0 + 1 + recv, busy)
    return s | {"produces_output": False, "t_pp_recv_ns": [(l0 + recv) * MS] * 2}


def _times(steps, stage, upstream=None, downstream=None, held=None, **carriers):
    """Each forward's predicted (send, device end, call, exit, tensors' end),
    in ms, on `FREE` gloo and `RDV` RCCL unless `carriers` names others."""
    models = {"gloo": FREE, "rccl": RDV} | carriers
    up = upstream and [[t * MS for t in u] for u in upstream]
    *times, _ = recurrence._charged(steps, stage, up, held, (models, 500), downstream)
    return [[t / MS for t in ts] for ts in times]


@pytest.mark.parametrize("pp", [2, 4])
def test_a_later_stage_posts_its_receive_at_its_prepare_end_after_w_recv(pp):
    """Posted at 1 ms plus the 2 ms its launch runs before the receive, the
    metadata, 1 ms a part on gloo, arrives at 5 ms: 10 ms of device work end at 15."""
    slow = CarrierModel(0, 0.0, 0.0, 0.0, 1e-3)
    [t] = _times([_step(0, 0, 10, recv=2)], (1, pp), [(0, 0)], gloo=slow, rccl=FREE)
    assert t[1] == pytest.approx(15)


@pytest.mark.parametrize("pp", [2, 4])
def test_a_later_stage_takes_its_tensors_after_its_own_device_work(pp):
    """The second frame leaves at 3 ms, while the first forward's device work
    runs to 12 ms: its 1 ms transfer ends at 13 ms."""
    steps = [_step(0, 0, 10), _step(1, 2, 1)]
    t = _times(steps, (1, pp), [(0, 0), (2, 3)])
    assert [t[1][4], t[1][1]] == pytest.approx([13, 14])


@pytest.mark.parametrize("pp", [2, 4])
def test_a_frames_tensors_leave_once_the_frame_befores_are_done(pp):
    """Eager, each tensor is done on the sender 2 ms after it leaves: the
    second frame, ready at 1 ms, leaves when the first's are done at 2 ms."""
    eager = CarrierModel(10**9, 0.0, 0.0, 4e-6, 0.0)
    steps = [_step(0, 0, 1), _step(1, 2, 1)]
    t = _times(steps, (1, pp), [(0, 0), (1, 1)], rccl=eager)
    assert [t[1][4], t[1][1]] == pytest.approx([4, 5])


@pytest.mark.parametrize("pp", [2, 4])
@pytest.mark.parametrize("tokens, start", [(None, 50), (60, 80)])
def test_a_sending_stage_starts_after_the_send_it_last_waited_for(pp, tokens, start):
    """The next stage receives the first two sends' tensors by 50 and 80 ms.
    The third forward's device waits for the first's; a head that waited for
    its tokens flushed the second, and waits for that one."""
    steps = [_step(n, 2 * n, 1) for n in range(3)]
    sends = [[0, 0, 0, 0, t * MS] for t in (50, 80)]
    held = [False, False, tokens and (0, tokens * MS)]
    t = _times(steps, (0, pp), downstream=sends, held=held)
    assert t[2][1] == pytest.approx(start + 1)


@pytest.mark.parametrize("call, end", [(52, 55), (152, 502)])
def test_a_middle_stage_flushes_once_its_metadata_poll_times_out(call, end):
    """Stage 1 of four, its first send received by 500 ms. Called 50 ms
    after its last exit it has not flushed, so it holds nothing; called
    150 ms after, it has, and takes its next tensors after that send's."""
    steps = [_step(0, 0, 1), _step(1, 2, 1)]
    held = [0, (2 * MS, call * MS)]
    sends = [[0, 0, 0, 0, 500 * MS]]
    t = _times(steps, (1, 4), [(0, 0), (1, 1)], sends, held)
    assert [t[1][4], t[1][1]] == pytest.approx([end - 1, end])


def test_a_heads_forward_after_its_flush_holds_the_next_stages_input():
    """`test_pp_carriers`'s script record before its fourth forwards move:
    the head waited for its tokens and flushed its third send, so its fourth
    forward starts its device work once those tensors are in, 1 ms after its
    launch, and the last stage's input arrives 1 ms later than recorded."""
    head, last = pp2_record()
    head[2:] = _shift(head[2:], MS)
    last = _late(last, MS)
    last[2:] = _shift(last[2:], MS)
    reports = recurrence.pipeline([head, last], (MODELS, 500))
    assert not reports[0]["failures"]
    assert [(r[0], round(r[3], 6)) for r in reports[1]["failures"]] == [(3, 1.0)]
