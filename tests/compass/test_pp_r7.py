# SPDX-License-Identifier: MIT
"""R7 on a pipeline: a stage's input from the stage before holds its host at
the metadata's arrival and its device at the tensors', in a simulated run, in
the parity record and in the recurrence's gate.
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


def _stage(rank, *grants, frames=(), ready=C, parts=()):
    """Stage `rank` of two, whose forward takes `C` of host time and ends its
    device work `ready` after the call; its runner calls by name and frame."""
    table = single_engine_table(
        admission_path="serving",
        ipc_s=0.5,
        stream_s=0.5,
        pp=2,
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
            return SimpleNamespace(pp_send=(ready, parts))
        return frames.pop(0) if name == "recv_pp_frame" else True

    return rt, partial(on_owner, StageSends(rt, call, rank, 2, MODELS)), calls


def test_a_downstream_forward_starts_at_the_metadata_and_its_device_at_the_tensors():
    t_send, t_ready = 0.25, 0.5
    frame = {"seq": 0, "t_send": t_send, "t_ready": t_ready, "parts": RDV}
    done, meta, end = pp_send.arrivals(MODELS, frame, 0.0)
    assert meta == t_send + SIZE + L_DATA
    # The tensor is posted at the metadata's arrival and sent at the device's end.
    assert done == end == t_ready + 2.0**-8 + 2048 * 2.0**-20
    rt, run, calls = _stage(
        1,
        (t_send + L_DATA, {DATA: [(0, t_send + L_DATA)]}),
        (meta, {}),
        (meta + C, {}),
        frames=[frame],
    )
    batch = SimpleNamespace()
    run("forward", batch)
    assert [m[:3] for m in rt.conn.sent] == [
        ("NER", math.inf, []),
        ("TAR", meta, [(ACK, 0, done)]),
        ("TAR", meta + C, []),
    ]
    assert batch.compass_pp_release == end
    assert done - (t_send + L_DATA) >= L_ACK
    assert [c[0] for c in calls] == ["recv_pp_frame", "send_pp_frame", "forward"]


def test_a_send_stamps_its_metadata_at_the_host_and_its_tensors_at_the_device_end():
    parts = [("gloo", 8), ("rccl", 512)]  # both eager
    done = 4 * C + 2.0**-12 + 512 * 2.0**-20
    rt, run, calls = _stage(0, (C, {}), (done, {}), ready=4 * C, parts=parts)
    run("forward", SimpleNamespace())
    [(_, kind, frame)] = [c for c in calls if c[0] == "send_pp_frame"]
    assert (kind, frame["t_send"], frame["t_ready"]) == ("pp_data", C, 4 * C)
    assert rt.send_log == [(DATA, 0, C + L_DATA)]
    # The idle flush waits for the tensor, which leaves at the device's end.
    run("flush_pp_send")
    assert [m[:2] for m in rt.conn.sent] == [("TAR", C), ("TAR", done)]


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
    assert host.charge(_decode(5.0), 1.0) == 6.0
    assert host.charge(_decode(), 1.0) == 7.0  # after the one before
    assert (
        HostDeviceRecurrence({}, (), lambda: 0.0, (1, 2)).charge(_decode(), 1.0) == 1.0
    )


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


def pp2_record():
    """Two stages written by the wait rules, 1 ms per host segment. The head
    computes 10 ms per forward and its entry waits two forwards back; the last
    stage computes 8 ms from its input's arrival, the head's end, and returns
    at its own end. Its first launch waits 1 ms for the head's send, and its
    first device start 8 ms for the head's end."""
    head = [
        _line(0, 0, (0, 0), (1, 2), 3, 10),
        _line(1, 3, (3, 3), (4, 5), 6, 10),
        _line(2, 6, (6, 11), (12, 13), 14, 10),
        _line(3, 14, (14, 21), (22, 23), 24, 10),
    ]
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
    first = recurrence.check(head, (0, 2))
    assert first["pass"] and first["total"] == pytest.approx(0, abs=1e-9)
    assert first["sends"] == [
        [t * MS for t in p] for p in ((2, 11), (5, 21), (13, 31), (23, 41))
    ]
    report = recurrence.check(last, (1, 2), first["sends"])
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    # Without the head, its device start and the first launch are not held.
    alone = recurrence.check(last, (1, 2))
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


def test_a_downstream_stage_writes_its_own_file_with_its_receive_stamped(
    tmp_path, monkeypatch
):
    batches = _drive(tmp_path / "simulated", monkeypatch, {0: [0]})[0][0]
    group = _Group()
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(get_pp_group=lambda: group),
    )
    config = _config(pipeline_parallel_size=2)
    config.parallel_config.pipeline_parallel_rank = 1
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
    assert read(tmp_path / "real") == {}
    steps = read(tmp_path / "real", 1)[0]
    assert len(steps) == len(batches)
    for step, batch in zip(steps, batches):
        # Events: the launch, the receive, the exit, 1 us apart.
        assert (step["d_recv_ns"], step["d_busy_ns"]) == (1000, 2000)
        l0, l1 = step["t_launch_ns"]
        assert l0 < step["t_pp_recv_ns"][0] < step["t_pp_recv_ns"][1] < l1
        assert ("t_recv_ns" in step) == batch.produces_output()
    assert any(b.produces_output() for b in batches)


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
    report = recurrence.check(head, (0, 2))
    assert report["pass"] and report["total"] == pytest.approx(0, abs=1e-9)
    assert [end for _, end in report["sends"]] == [t * MS for t in (2, 105, 109, 113)]


def test_a_middle_chunk_launches_once_the_head_has_sent():
    """The head sends at 2 ms; the last stage reaches its launch at 1 ms and
    receives until 2, so its middle chunk exits at 4, not 3."""
    head = recurrence.check([_line(0, 0, (0, 0), (1, 2), 3, 10)], (0, 2))
    last = [_line(0, 0, (0, 0), (1, 3), 4, 18, t_pp_recv_ns=(1, 2), d_recv_ns=10)]
    last[0]["produces_output"] = False
    report = recurrence.check(last, (1, 2), head["sends"])
    assert report["total"] == pytest.approx(0, abs=1e-9)
    assert recurrence.check(last, (1, 2))["total"] == pytest.approx(-25)
