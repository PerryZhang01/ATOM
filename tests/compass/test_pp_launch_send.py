# SPDX-License-Identifier: MIT
"""A pipeline stage's send leaves at its forward's launch end, and the next
stage runs its schedule and prepare before the wait for it, as ATOM's do.

Each stage loop is a real `StageSends` on an `LPRuntime` whose clock authority
is the scripted fake connection of the LP runtime tests. Its forward is the
runner's own, priced by a stage's step stub and composed with host time by a
`HostDeviceRecurrence`, and charged as the engine's `call_func` charges it.
"""

import collections
import math
import sys
from functools import partial
from types import SimpleNamespace

import pytest
import torch
from test_dp_step_max import prefill, runner

from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.clock import LpId, single_engine_table
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import HostDeviceRecurrence
from atom.compass.runner.pp_send import CarrierModel, StageSends
from atom.utils import clock
from atom.utils.clock import LPRuntime
from tests.compass.clock.test_lp_runtime import FakeConn, _owner

CARRIER = CarrierModel(1 << 20, 2.0**-10, 2.0**-12, 2.0**-20, 2.0**-8)
MODELS = {"rccl": CARRIER, "gloo": CARRIER}
L_DATA, L_ACK = pp_send.lookaheads(MODELS)
DATA = "engine-pp0->engine-pp1:pp_data#dp0"
E, PRE, LAUNCH, POST = 2.0**-6, 2.0**-7, 2.0**-1, 2.0**-5
LAWS = {"prefill": {"E": (E, 0, 0), "W_pre": (PRE, 0, 0), "W_launch": (LAUNCH, 0, 0)}}
LAWS["prefill"]["W_post"] = (POST, 0, 0)
#: Stands in for aiter's, which `_pp_send` pickles into the metadata part.
TensorMetadata = collections.namedtuple("TensorMetadata", "device dtype size")


def _stage(monkeypatch, rank, *grants, frames=()):
    """Stage `rank` of two: its runtime, its loop's runner calls through
    `StageSends`, and the frames it sent."""
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
    monkeypatch.setattr(clock, "_installed", rt)
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(TensorMetadata=TensorMetadata),
    )
    stage = runner(StepStubBackend(stage_share=(0.5, 0.5)))
    stage.config.pipeline_parallel_size = 2
    stage.config.parallel_config.pipeline_parallel_rank = rank
    stage.config.torch_dtype = torch.bfloat16
    stage.hf_text_config = SimpleNamespace(hidden_size=8)
    stage._pp_send_needs_sparse = False
    stage._setup_pp_shared_indexer = lambda: None
    stage.compass_host = HostDeviceRecurrence(LAWS, (), lambda: rt.now, (rank, 2))
    frames, sent = list(frames), []

    def call(name, *args, wait_out=False):
        if name == "forward":
            return clock.charge(stage.forward(*args))
        if name == "send_pp_frame":
            sent.append(args[1])
        return frames.pop(0) if name == "recv_pp_frame" else True

    return rt, partial(on_owner, StageSends(rt, call, rank, 2, MODELS)), sent


def test_a_stage_sends_at_its_launch_end_and_then_runs_its_postprocess(monkeypatch):
    launch_end = E + PRE + LAUNCH
    rt, run, sent = _stage(monkeypatch, 0, (launch_end, {}), (launch_end + POST, {}))
    run("forward", prefill(8))
    [frame] = sent
    assert (frame["t_send"], frame["t_ready"]) == (launch_end, launch_end)
    # The next stage's receive is released at the launch end plus L_data.
    assert [m[:3] for m in rt.conn.sent] == [
        ("TAR", launch_end, []),
        ("TAR", launch_end + POST, [(DATA, 0, launch_end + L_DATA)]),
    ]


@pytest.mark.parametrize("sent_at", [0.0, 2.0**-5])
def test_a_downstream_stage_prepares_before_its_input_arrives_and_launches_at_it(
    monkeypatch, sent_at
):
    """Sent at 0, the metadata arrives inside the schedule and prepare, which
    start at the call, so the launch waits for them; sent later, for it."""
    parts = [("gloo", 8), ("gloo", 64), ("rccl", 512)]
    frame = {"seq": 0, "t_send": sent_at, "t_ready": sent_at, "parts": parts}
    _, meta, end = pp_send.arrivals(MODELS, frame, 0.0)
    arrival = sent_at + L_DATA
    launch_end = max(E + PRE, meta) + LAUNCH
    rt, run, _ = _stage(
        monkeypatch,
        1,
        (arrival, {"engine-pp0->engine-pp1:pp_data#dp0": [(0, arrival)]}),
        (launch_end + POST, {}),
        frames=[frame],
    )
    batch = prefill(8)
    run("forward", batch)
    assert (batch.compass_pp_host, batch.compass_pp_release) == ((0.0, meta), end)
    assert [m[:2] for m in rt.conn.sent] == [
        ("NER", math.inf),
        ("TAR", launch_end + POST),
    ]
