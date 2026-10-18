# SPDX-License-Identifier: MIT
"""A pipeline stage's send leaves at its forward's launch end, and the next
stage runs its schedule and prepare before the wait for it, as ATOM's do.

Each stage loop is a real `StageSends` on an `LPRuntime` whose clock authority
is the scripted fake connection of the LP runtime tests. Its forward is the
runner's own, priced by a stage's step stub and composed with host time by a
`HostDeviceRecurrence`, and charged as the engine's `call_func` charges it.
"""

import collections
import dataclasses
import math
import sys
from functools import partial
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from test_dp_step_max import prefill, runner

from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.clock import LpId, single_engine_table
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import HostDeviceRecurrence, forward_shape
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
LAWS["prefill"]["W_recv"] = (0, 0, 0)  # a later stage receives at its launch start
#: Stands in for aiter's, whose import needs a driver.
TensorMetadata = collections.namedtuple("TensorMetadata", "device dtype size")


def split_tensor_dict(tensors):
    """aiter's `_split_tensor_dict`, for a dict of tensors."""
    meta = [
        (k, TensorMetadata(t.device.type, t.dtype, t.size()))
        for k, t in tensors.items()
    ]
    return meta, list(tensors.values())


def pp_comm(monkeypatch):
    """ATOM's `pp_comm`, splitting by aiter's rule, imported over a stand-in
    aiter when no earlier import has bound it."""
    for name in ("aiter.dist.parallel_state", "aiter.ops.communication"):
        if "atom.distributed.pp_comm" not in sys.modules and name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, MagicMock())
    from atom.distributed import pp_comm

    monkeypatch.setattr(pp_comm, "_split_tensor_dict", split_tensor_dict)
    return pp_comm


def _runner(monkeypatch, rank, now, pp=2):
    """Stage `rank` of `pp`'s runner, its host recurrence reading `now`."""
    pp_comm(monkeypatch)
    stage = runner(StepStubBackend(stage_share=(0.5, 0.5)))
    stage.config.pipeline_parallel_size = pp
    stage.config.parallel_config.pipeline_parallel_rank = rank
    stage.config.torch_dtype = torch.bfloat16
    stage.hf_text_config = SimpleNamespace(hidden_size=8)
    stage._pp_send_needs_sparse = False
    stage._setup_pp_shared_indexer = lambda: None
    stage.compass_host = HostDeviceRecurrence(LAWS, (), now, (rank, pp))
    return stage


def _stage(monkeypatch, rank, *grants, frames=(), pp=2, models=MODELS):
    """Stage `rank` of `pp`: its runtime, its loop's runner calls through
    `StageSends`, and the frames it sent."""
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
    monkeypatch.setattr(clock, "_installed", rt)
    stage = _runner(monkeypatch, rank, lambda: rt.now, pp)
    frames, sent = list(frames), []

    def call(name, *args, wait_out=False):
        if name == "forward":
            return clock.charge(stage.forward(*args))
        if name == "pp_recv_posted":
            return stage.pp_recv_posted(*args)
        if name == "send_pp_frame":
            sent.append(args[1])
        return frames.pop(0) if name == "recv_pp_frame" else True

    return rt, partial(on_owner, StageSends(rt, call, rank, pp, models)), sent


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
    _, meta, end = pp_send.arrivals(MODELS, frame, E + PRE)
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


def test_a_tp_rank_above_zero_sends_at_its_device_end_and_charges_no_post(
    monkeypatch,
):
    """Only TP rank 0 charges the stage's host recurrence, so another rank's
    send keeps the seconds it was given and leaves `W_post` to rank 0."""
    stage = _runner(monkeypatch, 0, lambda: 0.0)
    stage.rank = 1
    seconds, post, parts = stage._pp_send(prefill(8), 0.25)
    assert (seconds, post) == (0.25, 0.0)
    assert [carrier for carrier, _ in parts] == ["gloo", "gloo", "rccl", "rccl"]


#: Every gloo part rendezvous, as on MI308X, so each send's size is acked.
GLOO_RDV = {**MODELS, "gloo": dataclasses.replace(CARRIER, eager_threshold_bytes=0)}
INPUTS = 2.0**-8


@pytest.mark.parametrize("pp", [2, 4])
def test_a_downstream_stage_posts_its_receive_at_its_prepare_end(monkeypatch, pp):
    """Called at 0 with its input sent at 0, the last stage posts its receive
    after its schedule and prepare, as ATOM's `run_model` receives after
    `prepare_model`: the size and the metadata complete, and are acked, a
    rendezvous each after that, not after the call."""
    parts = [("gloo", 8), ("gloo", 64), ("rccl", 512)]
    frame = {"seq": 0, "t_send": 0.0, "t_ready": 0.0, "parts": parts}
    done, meta, end = pp_send.arrivals(GLOO_RDV, frame, E + PRE)
    assert done == meta == E + PRE + 2 * CARRIER.rendezvous_fixed_s + 72 * 2.0**-20
    data = f"engine-pp{pp - 2}->engine-pp{pp - 1}:pp_data#dp0"
    ack = f"engine-pp{pp - 1}->engine-pp{pp - 2}:pp_ack#dp0"
    rt, run, _ = _stage(
        monkeypatch,
        pp - 1,
        (L_DATA, {data: [(0, L_DATA)]}),
        (meta + LAUNCH + POST, {}),
        frames=[frame],
        pp=pp,
        models=GLOO_RDV,
    )
    batch = prefill(8)
    run("forward", batch)
    assert (batch.compass_pp_host, batch.compass_pp_release) == ((0.0, meta), end)
    assert [m[:3] for m in rt.conn.sent] == [
        ("NER", math.inf, []),
        ("TAR", meta + LAUNCH + POST, [(ack, 0, done)]),
    ]


@pytest.mark.parametrize("pp", [2, 4])
def test_a_stage_posts_its_receive_after_its_ring_wait_and_its_input_prepare(
    monkeypatch, pp
):
    """Its entry waits for the device work `pp` forwards back, which ends at
    `E + PRE + INPUTS + 1`, and under MTP it prepares its inputs before the
    receive too."""
    stage = _runner(monkeypatch, pp - 1, lambda: 0.0, pp)
    laws = {"prefill": {**LAWS["prefill"], "W_inputs": (INPUTS, 0, 0)}}
    host = HostDeviceRecurrence(laws, (), lambda: 0.0, (pp - 1, pp), mtp=True)
    stage.compass_host = host
    batch = prefill(8)
    for _ in range(pp):
        host.charge(batch, 1.0, 0.0)
    posted, _ = stage.pp_recv_posted(0.0, *forward_shape(batch))
    assert posted == E + 2 * (PRE + INPUTS) + 1.0
