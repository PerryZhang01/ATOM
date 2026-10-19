# SPDX-License-Identifier: MIT
"""A pipeline stage's input from the stage before holds its host at the
metadata's arrival and its device at the tensors', in a simulated run, in the
parity record and in the recurrence's gate.
"""

import json
import math
import sys
from functools import partial
from types import SimpleNamespace

import numpy as np
import pytest
from test_parity_record import GRAPHABLE, LADDER, _config, _drive

from atom.compass.clock import LpId, single_engine_table
from atom.compass.parity import ENV, StepRecording, read, recurrence
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.pp_send import CarrierModel, StageSends
from atom.utils.clock import LPRuntime
from tests.compass.clock.test_lp_runtime import FakeConn, _owner

CARRIER = CarrierModel(1024, 2.0**-10, 2.0**-12, 2.0**-20, 2.0**-8)
MODELS = {"rccl": CARRIER, "gloo": CARRIER}
L_DATA, L_ACK = pp_send.lookaheads(MODELS)
C = 2.0**-6
DATA, ACK = "engine-pp0->engine-pp1:pp_data#dp0", "engine-pp1->engine-pp0:pp_ack#dp0"
SIZE = 2.0**-12 + 8 * 2.0**-20  # the eager size part, sender side
#: The size on gloo, eager, then one tensor on RCCL, rendezvous.
RDV = [("gloo", 8), ("rccl", 2048)]
#: Both parts rendezvous: the size is acked.
ACKED = [("gloo", 2048), ("rccl", 2048)]
#: One rendezvous part's duration.
T = 2.0**-8 + 2048 * 2.0**-20


def _stage(rank, *grants, frames=(), ready=C, parts=(), pp=2, post=0.0):
    """Stage `rank` of `pp`, whose forward takes `C` of host time and ends its
    device work `ready` after the call, and whose `W_post` is `post`; its
    runner calls by name and frame."""
    table = single_engine_table(
        admission_path="serving",
        ipc_s=0.5,
        stream_s=0.5,
        pp=pp,
        pp_links=(L_DATA, L_ACK),
    )
    thread, on_owner = _owner()
    rt = LPRuntime(LpId(f"engine-pp{rank}"), table, FakeConn(*grants), owner=thread)
    rt.start_run()
    frames, calls = list(frames), []

    def call(name, *args, wait_out=False):
        calls.append((name, *args))
        if name == "forward":
            rt.advance_to(rt.now + C)
            return SimpleNamespace(pp_send=(ready, post, parts))
        if name == "pp_recv_posted":
            return args[0], -math.inf  # no host laws: posted at the call
        return frames.pop(0) if name == "recv_pp_frame" else True

    return rt, partial(on_owner, StageSends(rt, call, rank, pp, MODELS)), calls


def test_a_downstream_forward_is_called_at_the_arrival_its_launch_held_to_the_metadata():
    t_send, t_ready = 0.25, 0.5
    frame = {"seq": 0, "t_send": t_send, "t_ready": t_ready, "parts": RDV}
    _, meta, end = pp_send.arrivals(MODELS, frame, 0.0)
    assert meta == t_send + SIZE + L_DATA
    # The tensor is posted at the metadata's arrival and sent at the device's end.
    assert end == t_ready + T
    rt, run, calls = _stage(
        1,
        (t_send + L_DATA, {DATA: [(0, t_send + L_DATA)]}),
        (t_send + L_DATA + C, {}),
        frames=[frame],
    )
    batch = _decode()
    run("forward", batch)
    assert [m[:3] for m in rt.conn.sent] == [
        ("NER", math.inf, []),
        ("TAR", t_send + L_DATA + C, []),  # an eager size needs no ack
    ]
    assert (batch.compass_pp_host, batch.compass_pp_release) == ((0.0, meta), end)
    assert [c[0] for c in calls] == ["recv_pp_frame", "pp_recv_posted", "forward"]


def test_a_send_stamps_its_metadata_at_the_host_and_its_tensors_at_the_device_end():
    parts = [("gloo", 8), ("rccl", 512)]  # both eager
    rt, run, calls = _stage(0, (C, {}), (C + SIZE, {}), ready=4 * C, parts=parts)
    run("forward", SimpleNamespace())
    [(_, kind, frame)] = [c for c in calls if c[0] == "send_pp_frame"]
    assert (kind, frame["t_send"], frame["t_ready"]) == ("pp_data", C, 4 * C)
    assert rt.send_log == [(DATA, 0, C + L_DATA)]
    # The idle flush waits for the size, not the tensor.
    run("flush_pp_send")
    assert [m[:2] for m in rt.conn.sent] == [("TAR", C), ("TAR", C + SIZE)]


def test_a_middle_stage_receives_then_sends_and_acks_both_ways():
    """Stage 1 of four: its forward is called at the stage before's frame,
    acking that send's size, its launch held to the metadata; it sends the
    stage after its own frame at the launch end and then runs its `W_post`;
    the idle flush settles on the stage after's ack of its size."""
    t_send, t_ready = 0.25, 0.5
    frame = {"seq": 0, "t_send": t_send, "t_ready": t_ready, "parts": ACKED}
    done, meta, end = pp_send.arrivals(MODELS, frame, 0.0)
    arrived = t_send + L_DATA
    data_out = "engine-pp1->engine-pp2:pp_data#dp0"
    ack_in = "engine-pp2->engine-pp1:pp_ack#dp0"
    sent = {
        "seq": 0,
        "t_send": arrived + C,
        "t_ready": arrived + 4 * C,
        "parts": ACKED,
    }
    # The stage after posts its receive when the frame arrives.
    acked = pp_send.arrivals(MODELS, sent, arrived + C + L_DATA)[0]
    rt, run, calls = _stage(
        1,
        (arrived, {DATA: [(0, arrived)]}),
        (arrived + C, {}),
        (arrived + 2 * C, {}),
        (acked, {ack_in: [(0, acked)]}),
        frames=[frame, {"seq": 0, "tensors_done": -math.inf}],
        ready=4 * C,
        parts=ACKED,
        pp=4,
        post=C,
    )
    batch = _decode()
    run("forward", batch)
    assert (batch.compass_pp_host, batch.compass_pp_release) == ((0.0, meta), end)
    run("flush_pp_send")
    assert [m[:3] for m in rt.conn.sent] == [
        ("NER", math.inf, []),
        ("TAR", arrived + C, [(ACK, 0, done)]),
        ("TAR", arrived + 2 * C, [(data_out, 0, arrived + C + L_DATA)]),
        ("NER", math.inf, []),
    ]
    assert calls == [
        ("recv_pp_frame", "pp_data"),
        ("pp_recv_posted", 0.0, "decode", (1, 1, 1)),
        ("send_pp_frame", "pp_ack", {"seq": 0, "tensors_done": end}),
        ("forward", batch),
        ("send_pp_frame", "pp_data", sent),
        ("recv_pp_frame", "pp_ack"),
        ("flush_pp_send",),
    ]
    assert rt.now == acked


@pytest.mark.parametrize("pp", [2, 4])
def test_a_sending_stage_waits_for_its_previous_sends_size_and_not_its_tensor(pp):
    """The tensor is rendezvous, and the stage after has not acked it: the
    idle flush still ends at the size's local end, as ATOM's host waits on
    gloo while the RCCL wait only holds its stream."""
    parts = [("gloo", 8), ("rccl", 2048)]
    rt, run, calls = _stage(0, (C, {}), (C + SIZE, {}), ready=4 * C, parts=parts, pp=pp)
    run("forward", SimpleNamespace())
    run("flush_pp_send")
    assert [m[:2] for m in rt.conn.sent] == [("TAR", C), ("TAR", C + SIZE)]
    assert ("recv_pp_frame", "pp_ack") not in calls


@pytest.mark.parametrize("pp", [2, 4])
# A rendezvous tensor is done where it arrives; an eager one is done when it
# leaves the sender, a latency before it arrives.
@pytest.mark.parametrize(
    "tensor, step, latency", [(2048, T, 0.0), (512, 3 * 2.0**-12, L_DATA)]
)
def test_the_last_stage_acks_each_size_and_holds_each_tensor_behind_the_last(
    pp, tensor, step, latency
):
    """Two sends whose device work ends at the same time: each ack is stamped
    where the size's receive ends, and the second tensor leaves once the
    first is done, one stream sending both."""
    data = f"engine-pp{pp - 2}->engine-pp{pp - 1}:pp_data#dp0"
    ack = f"engine-pp{pp - 1}->engine-pp{pp - 2}:pp_ack#dp0"
    parts = [ACKED[0], ("rccl", tensor)]
    frames = [
        {"seq": i, "t_send": t, "t_ready": 0.5, "parts": parts}
        for i, t in enumerate((0.0, 0.25))
    ]
    rt, run, _ = _stage(
        pp - 1,
        (L_DATA, {data: [(0, L_DATA)]}),
        (L_DATA + C, {}),
        (0.25 + L_DATA, {data: [(1, 0.25 + L_DATA)]}),
        (0.25 + L_DATA + C, {}),
        frames=frames,
        pp=pp,
    )
    batches = [_decode(), _decode()]
    for batch in batches:
        run("forward", batch)
    assert [m[:3] for m in rt.conn.sent] == [
        ("NER", math.inf, []),
        ("TAR", L_DATA + C, [(ack, 0, T)]),
        ("NER", math.inf, []),
        ("TAR", 0.25 + L_DATA + C, [(ack, 1, 0.25 + T)]),
    ]
    releases = [b.compass_pp_release for b in batches]
    assert releases == [0.5 + step + latency, 0.5 + 2 * step + latency]


def _decode(release=None):
    batch = SimpleNamespace(
        total_seqs_num_prefill=0, total_tokens_num=1, total_seqs_num=1
    )
    batch.produces_output = lambda: True
    if release is not None:
        batch.compass_pp_release = release
    return batch


def test_a_stage_device_starts_at_its_input_and_the_last_returns_at_its_end():
    host = HostDeviceRecurrence({}, (), lambda: 0.0, (1, 2))
    assert host.charge(_decode(5.0), 1.0, 0.0) == 6.0
    assert host.charge(_decode(), 1.0, 0.0) == 7.0  # after the one before
    assert (
        HostDeviceRecurrence({}, (), lambda: 0.0, (1, 2)).charge(_decode(), 1.0, 0.0)
        == 1.0
    )


def test_a_stage_prepares_its_inputs_before_it_waits_for_the_metadata():
    """Under MTP the last stage's 1 s input prepare runs before its wait for
    the metadata at 5 s, so its 1 s forward ends at 6 s."""
    law = dict.fromkeys(("E", "W_pre", "W_launch", "W_post", "W_recv"), (0, 0, 0))
    law["W_inputs"] = (1.0, 0, 0)
    host = HostDeviceRecurrence({"decode": law}, (), lambda: 0.0, (1, 2), mtp=True)
    batch = _decode()
    batch.compass_pp_host = (0.0, 5.0)
    assert host.charge(batch, 1.0, 0.0) == 6.0


# --- the record and its gate -------------------------------------------------

MS = 1_000_000


def _line(n, enter, gate, launch, exit_, busy, **kw):
    s = {
        "step": n,
        "dp_rank": 0,
        "is_dummy_run": False,
        "produces_output": True,
        "batch": [[7, 1, 151 + n]],
        "rows": [[1, 151 + n, True]],
        "named": [],
        "t_enter_ns": enter * MS,
        "t_gate_ns": [g * MS for g in gate],
        "t_launch_ns": [t * MS for t in launch],
        "t_exit_ns": exit_ * MS,
        "d_busy_ns": busy * MS,
    }
    if n:
        s["d_gap_ns"] = 0
    return s | {
        k: [t * MS for t in v] if isinstance(v, tuple) else v * MS
        for k, v in kw.items()
    }


def staged(steps, stage, upstream=None, carriers=None):
    """Stage `stage = (rank, pp)` charged alone on the stage before's predicted
    (send, device end) per forward, `upstream`: its report and its own."""
    *times, report = recurrence._charged(steps, stage, upstream, None, carriers)
    return report, [t[:2] for t in times]


def pp2_record():
    """Two stages written by the wait rules, 1 ms per host segment, a request
    per forward. The head computes 10 ms per forward and enters its third and
    fourth at the last stage's exit two forwards back, which returns their
    tokens; the last stage computes 8 ms from its input's arrival, the head's
    end, and returns at its own end. Its first launch waits 1 ms for the
    head's send, and its first device start 8 ms for the head's end."""
    head = [
        _line(0, 0, (0, 0), (1, 2), 3, 10),
        _line(1, 3, (3, 3), (4, 5), 6, 10),
        _line(2, 20, (20, 20), (21, 22), 23, 10),
        _line(3, 30, (30, 30), (31, 32), 33, 10),
    ]
    for s in head:
        s["batch"][0][0] += s["step"]
    last = [
        _line(
            0,
            0,
            (0, 0),
            (1, 3),
            20,
            18,
            t_pp_recv_ns=(1, 2),
            t_recv_ns=(3, 19),
            d_recv_ns=10,
        )
    ] + [
        _line(
            n,
            t,
            (t, t),
            (t + 1, t + 2),
            t + 10,
            8,
            t_pp_recv_ns=(t + 1, t + 1),
            t_recv_ns=(t + 2, t + 9),
            d_recv_ns=0,
        )
        for n, t in ((1, 20), (2, 30), (3, 40))
    ]
    return head, last


def test_the_last_stage_is_charged_from_the_heads_predicted_sends():
    head, last = pp2_record()
    first, sends = staged(head, (0, 2))
    assert first["pass"] and first["total"] == pytest.approx(0, abs=1e-9)
    assert sends == [
        [t * MS for t in p] for p in ((2, 11), (5, 21), (22, 31), (32, 41))
    ]
    report, _ = staged(last, (1, 2), sends)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    # Without the head, its device start and the first launch are not held.
    alone, _ = staged(last, (1, 2))
    [(step, kind, err, *_)] = alone["failures"]
    assert (step, kind, err) == (0, "idle", pytest.approx(-50))


def test_the_script_gates_every_stage_of_a_pipeline_record(tmp_path, capsys):
    for name, steps in zip(("dp0", "pp1.dp0"), pp2_record()):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in steps)
        )
    assert recurrence.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert "dp0 pp0 kind" in out and "dp0 pp1 kind" in out


def test_a_stage_entered_before_its_release_is_refused_by_name():
    head, last = pp2_record()
    head[2]["t_enter_ns"] -= MS
    with pytest.raises(ValueError, match="stage 0 step 2 enters 1.000 ms before"):
        recurrence.pipeline([head, last])


def _downstream(n, enter, entry, wait, d_recv, end=None):
    """A downstream stage's forward `n`, 1 ms per host segment: entered at
    `enter`, its entry released at `entry`, its launch waiting `wait` for the
    stage before's send, its device 10 ms of work after `d_recv` waiting for
    the input; on the last stage, its token read waits for its device `end`."""
    l1 = entry + wait + 2
    kw = {"t_pp_recv_ns": (entry + 1, entry + 1 + wait), "d_recv_ns": d_recv}
    if end is not None:
        kw["t_recv_ns"] = (l1, end)
    exit_ = (end or l1) + 1
    return _line(n, enter, (enter, entry), (entry + 1, l1), exit_, 10 + d_recv, **kw)


def pp4_record():
    """Four stages written by the wait rules, 10 ms of device work per forward,
    a request per forward. The head's fifth and sixth entries wait for the
    last stage's exit four forwards back, which returns their tokens, and
    every downstream entry for the head's. Each downstream stage launches at
    the stage before's send and starts its device at that stage's end; the
    middle stages return at their launch, the last at its own end."""
    head = [
        _line(n, enter, (enter, entry), (entry + 1, entry + 2), entry + 3, 10)
        for n, (enter, entry) in enumerate(
            ((0, 0), (3, 3), (6, 6), (9, 9), (42, 42), (54, 54))
        )
    ]
    for s in head:
        s["batch"][0][0] += s["step"]
    rest = (
        ((0, 0, 1, 10), (4, 4, 0, 0), (7, 7, 0, 0))
        + ((10, 10, 0, 0), (42, 42, 1, 10), (54, 54, 1, 10)),
        ((0, 0, 2, 20), (5, 5, 0, 0), (8, 8, 0, 0))
        + ((11, 11, 0, 0), (42, 42, 2, 20), (54, 54, 2, 20)),
        ((0, 0, 3, 30, 41), (42, 42, 0, 0, 53), (54, 54, 0, 0, 65))
        + ((66, 66, 0, 0, 77), (78, 78, 0, 0, 89), (90, 90, 0, 0, 101)),
    )
    return [head] + [[_downstream(n, *f) for n, f in enumerate(s)] for s in rest]


#: Each non-last stage's predicted (send, device end) per forward, in ms.
PP4_SENDS = (
    ((2, 11), (5, 21), (8, 31), (11, 41), (44, 53), (56, 65)),
    ((3, 21), (6, 31), (9, 41), (12, 51), (45, 63), (57, 75)),
    ((4, 31), (7, 41), (10, 51), (13, 61), (46, 73), (58, 85)),
)


def test_four_stages_are_each_charged_from_the_stage_befores_sends(tmp_path, capsys):
    stages = pp4_record()
    for k, steps in enumerate(stages):
        name = f"pp{k}.dp0" if k else "dp0"
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in steps)
        )
    for k, steps in enumerate(stages):
        own = [{f: v for f, v in s.items() if f != "named"} for s in steps]
        assert read(tmp_path, k) == {0: own}
    assert read(tmp_path, 4) == {}
    assert recurrence.main(str(tmp_path)) == 0
    out = capsys.readouterr().out
    assert all(f"dp0 pp{k} kind" in out for k in range(4))
    sends = [None]
    for k, steps in enumerate(stages):
        report, own = staged(steps, (k, 4), sends[k])
        assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
        sends.append(own)
        if k:  # unreleased by the stage before, its first forward runs early
            assert not staged(steps, (k, 4))[0]["pass"]
    assert sends[1:4] == [[[t * MS for t in p] for p in s] for s in PP4_SENDS]
    # Two forwards back would hold the head's third entry.
    assert not staged(stages[0], (0, 2))[0]["pass"]
    # The last stage waits for its own device end, a middle stage does not.
    assert not staged(stages[3], (2, 4), sends[3])[0]["pass"]
    assert not staged(stages[2], (3, 4), sends[2])[0]["pass"]
    # The last stage's first forward 2 ms late holds the head's fifth entry,
    # four forwards on, and so every middle stage's.
    stages[3][0]["d_busy_ns"] += 2 * MS
    late = [[r[0] for r in rep["failures"]] for rep in recurrence.pipeline(stages)]
    assert late == [[4], [4], [4], [0]]


def serial_record(pp):
    """`pp` stages on one decoding request, written by the wait rules, 1 ms per
    host segment and 10 ms of device work per stage after its input. The head
    enters 1 ms after the last stage's exit, whose tokens it waits for; stage
    `i` enters `2 i + 1` ms after the head's call, after the stage before has
    sent, so only its entry waits."""
    p = 10 * pp + 3
    head = [
        _line(n, n * p, (n * p,) * 2, (n * p + 1, n * p + 2), n * p + 3, 10)
        for n in range(4)
    ]
    rest = [
        [
            _downstream(
                n,
                n * p + 2 * i,
                n * p + 2 * i,
                0,
                8 * i,
                (n + 1) * p - 2 if i == pp - 1 else None,
            )
            for n in range(4)
        ]
        for i in range(1, pp)
    ]
    return [head] + rest


@pytest.mark.parametrize("pp", [2, 4])
def test_each_stage_waits_for_the_other_stages_predicted_times(pp):
    """With the last stage's second forward predicted 2 ms late, the head's
    next entry follows it, and so does every middle stage's at the head's
    call; the last stage's own next forward is charged as measured."""
    stages = serial_record(pp)
    for report in recurrence.pipeline(stages):
        assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    stages[-1][1]["d_busy_ns"] += 2 * MS
    late = [
        [(r[0], round(r[3], 6)) for r in rep["failures"]]
        for rep in recurrence.pipeline(stages)
    ]
    assert late == [[(2, 2.0)]] * (pp - 1) + [[(1, 2.0)]]


def test_the_head_waits_for_the_tokens_its_batch_needs():
    """Rows are (request, decode): a request's next chunk waits for nothing,
    its first decode, or any request of a batch that decoded, for that batch,
    and every forward for the one `pp` back if that one produced output."""
    head = [
        {
            "produces_output": out,
            "batch": [[k] for k, _ in rows],
            "rows": [[d] for _, d in rows],
        }
        for out, rows in (
            (True, [("A", False), ("B", False)]),
            (False, [("B", False)]),
            (True, [("A", True), ("B", False)]),
            (False, [("B", False)]),
            (True, [("C", False)]),
            (True, [("D", False)]),
        )
    ]
    for pp, waits in (
        (2, [None, None, 0, 2, 2, None]),
        (4, [None, None, 0, 2, 0, None]),
    ):
        assert [recurrence._waited(head, n, pp) for n in range(6)] == waits


class _Group:
    is_first_rank = False

    def recv_tensor_dict(self, src, all_gather_group=None):
        return {"hidden_states": src}


class _Tokens:
    """The last stage's processor: it reads each forward's tokens itself."""

    is_deferred_out = False

    def recv_async_output(self, handle):
        raise AssertionError(
            "a stage that does not defer output has no copy to wait on"
        )

    def prepare_sampled_ids(self, batch):
        return {}


def _record_last_stage(tmp_path, monkeypatch, pp):
    """A simulated run's batches, recorded on the last of `pp` stages by a
    runner whose processor reads each output forward's tokens itself; the
    batches and the record directory."""
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0]})[0][0]
    group = _Group()
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_pp_group=lambda: group),
    )
    config = _config(pipeline_parallel_size=pp)
    config.parallel_config.pipeline_parallel_rank = pp - 1
    times = iter(range(0, 10**9, 1000))  # each device event 1 us after the last

    class _Event:
        def __init__(self):
            self.t = next(times)

        def query(self):
            return True

        def elapsed_time(self, other):
            return (other.t - self.t) / 1e6

    class _Base:
        def __init__(self, rank, config):
            self.rank, self.config = rank, config
            self.capture_sizes_np, self.enforce_eager = np.array(LADDER), False
            self.attn_metadata_builder = GRAPHABLE
            self.tokenID_processor = _Tokens()

        def _gate_staging_reuse(self):
            pass

        def run_model(self, *args):
            assert group.recv_tensor_dict(src=0, all_gather_group=None) == {
                "hidden_states": 0
            }
            return "logits"

        def forward(self, batch):
            self._gate_staging_reuse()
            self.run_model([], batch)
            if batch.produces_output():
                self.tokenID_processor.prepare_sampled_ids(batch)
            return "ran"

        def exit(self):
            pass

    class _Recorded(StepRecording, _Base):
        clock = iter(range(1, 10**9))

        def step_clock_ns(self):
            return next(self.clock)

        @staticmethod
        def device_event():
            return _Event()

    monkeypatch.setenv(ENV, str(tmp_path / "real"))
    runner = _Recorded(0, config)
    for b in batches:
        runner.forward(b)
    runner.exit()
    return batches, tmp_path / "real"


def test_a_downstream_stage_writes_its_own_file_with_its_receive_stamped(
    tmp_path, monkeypatch
):
    batches, real = _record_last_stage(tmp_path, monkeypatch, 2)
    assert read(real) == {}
    steps = read(real, 1)[0]
    assert len(steps) == len(batches)
    for step, batch in zip(steps, batches):
        # Events: the launch, the receive, the exit, 1 us apart.
        assert (step["d_recv_ns"], step["d_busy_ns"]) == (1000, 2000)
        l0, l1 = step["t_launch_ns"]
        assert l0 < step["t_pp_recv_ns"][0] < step["t_pp_recv_ns"][1] < l1
        assert ("t_recv_ns" in step) == batch.produces_output()
    assert any(b.produces_output() for b in batches)


def test_a_runner_off_a_pipeline_leaves_its_token_read_unstamped(tmp_path, monkeypatch):
    """A P/D producer of a recurrent-state model reads its tokens itself, as
    the last stage does; off a pipeline the gate releases that wait at the
    forward's device end, so the record stamps no receive."""
    batches, real = _record_last_stage(tmp_path, monkeypatch, 1)
    steps = read(real)[0]
    assert len(steps) == len(batches)
    assert not any({"t_recv_ns", "t_pp_recv_ns"} & step.keys() for step in steps)


def test_a_single_stage_token_read_is_released_at_the_device_end():
    """Decodes whose host waits inside `W_post` for their own device end, 1 ms
    per host segment and 10 ms of device: the gate holds them unstamped, and a
    receive stamped around that wait would take it out of `W_post`."""
    steps = [
        _line(n, 14 * n, (14 * n,) * 2, (14 * n + 1, 14 * n + 2), 14 * n + 13, 10)
        | ({"d_gap_ns": 3 * MS} if n else {})
        for n in range(6)
    ]
    report = recurrence.check(steps)
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    for s in steps:
        s["t_recv_ns"] = [s["t_enter_ns"] + 2 * MS, s["t_enter_ns"] + 12 * MS]
    assert not recurrence.check(steps)["pass"]


def test_a_stage_gap_after_an_idle_device_does_not_delay_its_device():
    """The head waits 100 ms for the last stage before its second forward, so
    that forward's 102 ms device gap is idle; its device still ends at 105 ms,
    and the fourth forward's entry, two back, is not held."""
    head = [
        _line(0, 0, (0, 0), (1, 2), 3, 1),
        _line(1, 103, (103, 103), (104, 105), 106, 1, d_gap_ns=102),
        _line(2, 107, (107, 107), (108, 109), 110, 1, d_gap_ns=3),
        _line(3, 111, (111, 111), (112, 113), 114, 1, d_gap_ns=3),
    ]
    report, sends = staged(head, (0, 2))
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    assert [end for _, end in sends] == [t * MS for t in (2, 105, 109, 113)]


def test_a_stage_device_idles_its_gap_while_its_host_runs_ahead():
    """The head's host launches each forward 3 ms after the last, its device
    takes 10 ms and idles 2 ms before each forward after the first, so its
    device ends at 11, 23, 35 and 47 ms, and the fourth forward's entry waits
    for the second's end."""
    head = [
        _line(0, 0, (0, 0), (1, 2), 3, 10),
        _line(1, 3, (3, 3), (4, 5), 6, 10, d_gap_ns=2),
        _line(2, 6, (6, 11), (12, 13), 14, 10, d_gap_ns=2),
        _line(3, 14, (14, 23), (24, 25), 26, 10, d_gap_ns=2),
    ]
    report, sends = staged(head, (0, 2))
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    assert [end for _, end in sends] == [t * MS for t in (11, 23, 35, 47)]


def test_a_middle_chunk_launches_once_the_head_has_sent():
    """The head sends at 2 ms; the last stage reaches its launch at 1 ms and
    receives until 2, so its middle chunk exits at 4, not 3."""
    _, sends = staged([_line(0, 0, (0, 0), (1, 2), 3, 10)], (0, 2))
    last = [_line(0, 0, (0, 0), (1, 3), 4, 18, t_pp_recv_ns=(1, 2), d_recv_ns=10)]
    last[0]["produces_output"] = False
    assert staged(last, (1, 2), sends)[0]["total"] == pytest.approx(0, abs=1e-9)
    assert staged(last, (1, 2))[0]["total"] == pytest.approx(-25)
