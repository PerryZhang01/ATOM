# SPDX-License-Identifier: MIT
"""A pipeline stage at TP above 1 sends each tensor as its shard and the next
stage all-gathers it, as ATOM's `async_send_intermediate_tensors` and aiter's
`recv_tensor_dict` do with `pp_send_allgather_group` on: the receive of each
part after a sharded one is posted once its all-gather ends.

The run test needs a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import json
from types import SimpleNamespace

import pytest
import test_pp_launch_send as launch
from test_dp_slice import (  # noqa: F401
    MAX_TOKENS,
    device_classes_put_back,
    run_dp_slice,
)
from test_dp_step_max import prefill
from test_memory_readings import META_8B
from test_pp_launch_send import PRE, E
from test_pp_stream_holds import ALL_RDV
from test_pp_stream_holds import _stage as real_stage
from test_run_step_grid import deployment, run_file
from test_vertical_slice import NEEDS_A_RUN

from atom.compass import run as compass_run
from atom.compass.backends.step_stub import Coefficients, StepStubBackend
from atom.compass.detect.scenario import check_run
from atom.compass.runner import pp_send

#: The step stub's seconds per row of a collective: one all-gather of 8 rows.
G = 2.0**-10


def _sharding(monkeypatch, on: bool, width: int):
    """ATOM's TP group `width` wide, and `ATOM_PP_SEND_ALLGATHER` `on`."""
    comm = launch.pp_comm(monkeypatch)
    group = SimpleNamespace(world_size=width, rank_in_group=0)
    monkeypatch.setattr(comm, "get_tp_group", lambda: group)
    monkeypatch.setenv("ATOM_PP_SEND_ALLGATHER", "1" if on else "0")


def _sent(monkeypatch, rank, pp, on=True, width=2):
    """Stage `rank` of `pp`'s parts for an 8-token prefill at TP `width`, sent
    by TP rank 1."""
    _sharding(monkeypatch, on, width)
    stage = launch._runner(monkeypatch, rank, lambda: 0.0, pp)
    stage.config.tensor_parallel_size, stage.rank = width, 1  # no host charge
    return stage._pp_send(prefill(8), 0.25)[2]


def test_a_tensor_the_tp_width_divides_is_sent_as_its_shard(monkeypatch):
    """8 tokens x hidden 8 in bf16 is 128 bytes a tensor: at TP 2 each is
    sent as a 64-byte shard; 64 elements do not divide by 3, and with the
    send-allgather off ATOM sends whole tensors."""
    tensors = _sent(monkeypatch, 0, 2)[2:]
    assert tensors == [("rccl", 64, True)] * 2
    assert _sent(monkeypatch, 0, 2, width=3)[2:] == [("rccl", 128)] * 2
    assert _sent(monkeypatch, 0, 2, on=False)[2:] == [("rccl", 128)] * 2


def test_a_sharded_parts_all_gather_holds_the_next_receive_not_the_sender():
    """Rendezvous parts: the second part's receive is posted at the first's
    end plus its all-gather, and the receive ends after the second's; the
    sender is done at the second part's end."""
    t = ALL_RDV["rccl"].rendezvous_s(64)
    parts = [("rccl", 64, True)] * 2
    assert pp_send.recv_done(ALL_RDV, 0.0, 0.0, parts, G) == (2 * t + G, 2 * t + 2 * G)
    whole = [("rccl", 64)] * 2
    assert pp_send.recv_done(ALL_RDV, 0.0, 0.0, whole, G) == (2 * t, 2 * t)


@pytest.mark.parametrize("pp", [2, 4])
def test_each_receive_ends_after_its_shard_transfers_and_its_all_gathers(
    monkeypatch, pp
):
    """At TP 2, the last of `pp` stages receives stage `pp - 2`'s send, each
    tensor a 64-byte shard, and each all-gather priced by its own step stub
    at 8 rows; its device starts after both shards and both gathers."""
    parts = _sent(monkeypatch, pp - 2, pp)
    frame = {"seq": 0, "t_send": 0.0, "t_ready": 0.0, "parts": parts}
    _, meta, _ = pp_send.arrivals(ALL_RDV, frame, E + PRE)
    data = f"engine-pp{pp - 2}->engine-pp{pp - 1}:pp_data#dp0"
    _, run, _, stage = real_stage(
        monkeypatch,
        pp - 1,
        (launch.L_DATA, {data: [(0, launch.L_DATA)]}),
        (4.0, {}),
        frames=[frame],
        pp=pp,
        models=ALL_RDV,
    )
    stub = StepStubBackend(Coefficients(collective_token_layer=G), stage_share=(1, 1))
    stage.compass_backend = stub
    sends, call = run.args[0], run.args[0].call
    sends.call = lambda name, *a, wait_out=False: (
        stage.pp_all_gather_s(*a)
        if name == "pp_all_gather_s"
        else call(name, *a, wait_out=wait_out)
    )
    batch = prefill(8)
    run("forward", batch)
    shard = ALL_RDV["rccl"].rendezvous_s(64)
    assert batch.compass_pp_release == meta + 2 * (shard + 8 * G)


def _stub_run(tmp_path, monkeypatch, **keys):
    """A PP2 TP2 stage's runner and a run file pricing it with the step stub."""
    r = deployment(
        tmp_path, tensor_parallel_size=2, tp_world_size=2, pipeline_parallel_size=2
    )
    run = run_file(tmp_path, monkeypatch, step_stub={}, **keys)
    del run["step_grid"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    return r


def test_a_backend_that_prices_no_all_gather_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_PP_SEND_ALLGATHER", "1")
    r = _stub_run(tmp_path, monkeypatch)
    compass_run.runner(r, META_8B)
    assert (
        r.compass_backend.all_gather_s(8) == 8 * Coefficients().collective_token_layer
    )

    r = _stub_run(tmp_path, monkeypatch, step_roofline=True)
    with pytest.raises(ValueError, match="StepFallbackBackend prices no all-gather"):
        compass_run.runner(r, META_8B)
    monkeypatch.setenv("ATOM_PP_SEND_ALLGATHER", "0")
    compass_run.runner(r, META_8B)


@NEEDS_A_RUN
def test_a_tp2_pipeline_meets_the_run_properties(tmp_path, monkeypatch):
    monkeypatch.setenv("ATOM_PP_SEND_ALLGATHER", "1")
    runs = [
        run_dp_slice(tmp_path / s, s, ["-pp", "2", "-tp", "2"], dp=1, pp=2)
        for s in ("1", "2")
    ]
    print(f"\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs], MAX_TOKENS, [line for _, line in runs[0]["events"]]
    )
    assert not failed, "\n".join(failed)
    rows = [line.split() for line in runs[0]["table"].splitlines()[1:]]
    released = {r[4] for r in rows if r[3] == "release"}
    assert "engine-pp0->engine-pp1:pp_data#dp0" in released
