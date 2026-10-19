# SPDX-License-Identifier: MIT
"""A simulated run at pipeline width above one: each stage is an engine LP.

ATOM's API server at ``-pp 2`` or ``-pp 4`` runs one engine-core process per
stage. The head schedules, sends each downstream stage its batch on
``meta#pp<k>``, and the last stage sends the head its output on ``tokens``;
each stage's worker starts on the host with its groups, the PP group among
them, on gloo. Each stage hands the next its forward's output on ``pp_data``,
and the next returns a rendezvous send's completion on ``pp_ack``.

The run test needs a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import json
import sys
from types import SimpleNamespace

import pytest
from test_dp_slice import (  # noqa: F401
    DP_MACHINE,
    MAX_TOKENS,
    _deployment,
    device_classes_put_back,
    run_dp_slice,
)
from test_vertical_slice import CONSTANT, NEEDS_A_RUN, _run_file

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.detect.determinism import StepTable
from atom.compass.detect.scenario import check_run, pipeline_timeline
from atom.compass.runner import pp_send
from atom.compass.runner.overrides import start_on_host
from atom.compass.spec import MachineSpec, SpecRefusal
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

IPC = 2.0**-14
L_DATA, L_ACK = pp_send.lookaheads(
    pp_send.carrier_models(MachineSpec.from_mapping(DP_MACHINE))
)
META = "engine-pp0->engine-pp1:meta#pp1"
DATA, ACK = "engine-pp0->engine-pp1:pp_data#dp0", "engine-pp1->engine-pp0:pp_ack#dp0"


@pytest.fixture(scope="module", params=[2, 4])
def pipeline(request, tmp_path_factory):
    pp = request.param
    root = tmp_path_factory.mktemp(f"pp{pp}")
    return pp, [
        run_dp_slice(root / s, s, ["-pp", str(pp)], dp=1, pp=pp) for s in ("1", "2")
    ]


@NEEDS_A_RUN
def test_a_pipeline_meets_the_run_properties_and_uses_every_stage_channel(pipeline):
    pp, runs = pipeline
    print(f"\n{runs[0]['table']}")
    failed = check_run(
        [r["out"] for r in runs], MAX_TOKENS, [line for _, line in runs[0]["events"]]
    )
    assert not failed, "\n".join(failed)
    rows = [line.split() for line in runs[0]["table"].splitlines()[1:]]
    released = {r[4] for r in rows if r[3] == "release"}
    last = f"engine-pp{pp - 1}"
    assert f"{last}->engine-pp0:tokens" in released
    assert {f"engine-pp0->engine-pp{k}:meta#pp{k}" for k in range(1, pp)} <= released
    # Every stage priced its forwards.
    assert {r[0] for r in rows if r[3] == "TAR"} >= {f"engine-pp{k}" for k in range(pp)}
    # The head keeps up to pp batches in flight, so it starts forward n + pp
    # only once forward n's tokens are back: after every stage's compute of
    # forward n, the whole model's step, and the tokens' IPC to the head.
    starts = sorted(float(r[2]) - IPC for r in rows if r[4].endswith(":meta#pp1"))
    gaps = [b - a for a, b in zip(starts, starts[pp:])]
    print(f"pp{pp} head forward n to n + pp: {gaps}")
    assert min(gaps) >= CONSTANT["decode_step"] + IPC
    # Each rendezvous send from the head: t_send, t_recv_posted, now_r, done.
    data, ack = ({int(r[5]): r for r in rows if r[4] == ch} for ch in (DATA, ACK))
    sends = []
    for n in sorted(ack):
        posted, arrival = float(data[n][1]), float(data[n][2])
        sends.append((arrival - L_DATA, posted, max(posted, arrival), float(ack[n][2])))
        print("t_send {!r} t_recv_posted {!r} now_r {!r} done {!r}".format(*sends[-1]))
    least = min(done - now_r for *_, now_r, done in sends)
    print(f"least done - now_r {least!r}, L_ack {L_ACK!r}")
    assert len(sends) == len(data) and least >= L_ACK


# --- without a driver ---------------------------------------------------------


def test_the_stages_join_the_head_with_a_positive_lookahead_each_way():
    table = single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=IPC, pp=4)
    head, *rest = [LpId(f"engine-pp{k}") for k in range(4)]
    assert set(table.registry.ids()) == {
        LpId("traffic"),
        LpId("frontend"),
        head,
        *rest,
    }
    for stage in rest:
        assert table.distance(head, stage) == table.distance(stage, head) == IPC
    # Two downstream stages hear of each other only through the head.
    assert table.distance(rest[0], rest[1]) == 2 * IPC
    with pytest.raises(ValueError, match="ipc_s is 0.0"):
        single_engine_table(admission_path="serving", ipc_s=0.0, stream_s=IPC, pp=2)


def test_a_stage_names_its_channels_by_the_index_its_frontend_gives_it(monkeypatch):
    table = single_engine_table(admission_path="serving", ipc_s=IPC, stream_s=IPC, pp=2)
    rt = LPRuntime(LpId("engine-pp1"), table, None)
    monkeypatch.setattr(clock, "_installed", rt)
    pc = SimpleNamespace(
        pipeline_parallel_rank=1,
        pp_meta_addrs=["m0", "m1"],
        pp_token_addr="t",
        pp_kv_status_addr="k",
    )
    config = SimpleNamespace(
        parallel_config=pc,
        pipeline_parallel_size=2,
        disagg_p2d_addr="",
        disagg_d2p_addr="",
    )
    relay = clock.relay_queue(0)
    clock.name_endpoints(0, "in", "ctl", "out", config)
    assert relay.wsock.ch == "engine-pp1->frontend:output#dp1"
    assert [rt.endpoints[a] for a in ("in", "ctl", "out", "m1")] == [
        "request#dp1",
        "control#dp1",
        "output#dp1",
        "meta#pp1",
    ]


def _stage(rank, pp=2):
    deployment = _deployment()
    deployment.pipeline_parallel_size = pp
    deployment.parallel_config.pipeline_parallel_rank = rank
    return deployment


def test_a_pipeline_whose_machine_states_no_send_carrier_is_refused_by_name(
    monkeypatch, tmp_path
):
    run = _run_file(tmp_path, pipeline_parallel_size=2)
    monkeypatch.setenv(compass_run.ENV, str(run))
    with pytest.raises(SpecRefusal, match="interconnect.intra_node.pp.rccl"):
        compass_run.frontend(_stage(0))
    assert clock.installed() is None


def test_a_pipeline_at_tp_two_that_sends_shards_is_refused(monkeypatch, tmp_path):
    run = _run_file(tmp_path, pipeline_parallel_size=2)
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setenv("ATOM_PP_SEND_ALLGATHER", "1")
    deployment = _stage(0)
    deployment.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="set ATOM_PP_SEND_ALLGATHER=0"):
        compass_run.frontend(deployment)
    assert clock.installed() is None


def _two_stage_table(*rows) -> str:
    """A PP2 step table: the head's forward 0 starts at 1.0, and `rows`."""
    run = {
        "admission_path": "serving",
        "ipc_s": IPC,
        "stream_s": IPC,
        "pipeline_parallel_size": 2,
        "machine": DP_MACHINE,
    }
    table = StepTable(json.dumps(run))
    table.record("engine-pp0", 1.0, 1.0, "TAR")
    table.record("engine-pp1", 0.5, 1.0 + IPC, "release", META, 0)
    for row in rows:
        table.record(*row)
    return table.text()


#: Forward 0's output leaves the head at 1.5 and is released to the next stage
#: at `IN`; its rendezvous completes at 1.6.
IN = 1.5 + L_DATA
SENT = ("engine-pp1", 1.0 + IPC, IN, "release", DATA, 0)
ACKED = ("engine-pp0", 1.0, 1.6, "release", ACK, 0)


def test_the_pipeline_check_holds_a_stage_to_the_release_of_its_input():
    # The forward starts at the release; the rendezvous completes inside it.
    computes = ("engine-pp1", IN, 1.8, "TAR")
    assert pipeline_timeline(_two_stage_table(SENT, ACKED, computes)) == []
    # A stage busy at the release starts the forward when it frees.
    busy, late = ("engine-pp1", 1.4, 1.7, "TAR"), ("engine-pp1", 1.7, 1.9, "TAR")
    assert pipeline_timeline(_two_stage_table(SENT, ACKED, busy, late)) == []
    # A stage that forwards as its batch arrives, as before the stage sends,
    # fails whether the send was rendezvous or eager.
    early = ("engine-pp1", 1.0 + IPC, 1.2, "TAR")
    for rows in ((SENT, ACKED, early), (SENT, early)):
        assert pipeline_timeline(_two_stage_table(*rows)) == [
            f"engine-pp1 starts no forward 0 at {IN}"
        ]
    assert pipeline_timeline(_two_stage_table(early)) == [
        f"engine-pp0 never sent forward 0's output on {DATA}"
    ]
    # Input released before the batch: the forward starts at the batch.
    sent_early = ("engine-pp1", 0.5, 0.9 + L_DATA, "release", DATA, 0)
    at_input = ("engine-pp1", 0.9 + L_DATA, 0.95, "TAR")
    sent_before = "engine-pp0 sent forward 0 at 0.9, before 1.0"
    failed = pipeline_timeline(_two_stage_table(sent_early, ACKED, at_input))
    assert failed == [sent_before, f"engine-pp1 starts no forward 0 at {1.0 + IPC}"]
    at_batch = ("engine-pp1", 1.0 + IPC, 1.8, "TAR")
    failed = pipeline_timeline(_two_stage_table(sent_early, ACKED, at_batch))
    assert failed == [sent_before]


def test_a_run_file_stating_a_pipeline_the_deployment_does_not_run_is_refused(
    monkeypatch, tmp_path
):
    run = _run_file(tmp_path, pipeline_parallel_size=2)
    monkeypatch.setenv(compass_run.ENV, str(run))
    with pytest.raises(ValueError, match="pipeline_parallel_size is 2"):
        compass_run.frontend(_stage(0, pp=1))
    assert clock.installed() is None


def test_each_stage_is_an_engine_lp_of_its_own(monkeypatch, tmp_path):
    run = _run_file(tmp_path, pipeline_parallel_size=2, machine=DP_MACHINE)
    monkeypatch.setenv(compass_run.ENV, str(run))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(utils, "LP_OF_RANK", None, raising=False)
    monkeypatch.setattr(compass_run, "_member", None)
    joined = []
    monkeypatch.setattr(
        compass_run, "connect", lambda lp, endpoint, member: joined.append((lp, member))
    )
    with compass_run.engine(_stage(1)):
        pass
    assert joined == [(LpId("engine-pp1"), None)]
    assert utils.LP_OF_RANK == {0: LpId("engine-pp1")}


def test_a_stage_starts_on_the_host_after_the_stages_before_it(monkeypatch):
    calls = {}

    def record(name):
        return lambda *a, **k: calls.setdefault(name, (a, k))

    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.parallel_state",
        SimpleNamespace(
            init_distributed_environment=record("world"),
            initialize_model_parallel=record("groups"),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.utils",
        SimpleNamespace(get_distributed_init_method=lambda ip, p: f"tcp://{ip}:{p}"),
    )
    config = SimpleNamespace(
        tensor_parallel_size=2,
        tp_world_size=2,
        decode_context_parallel_size=1,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=4,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=1,
            data_parallel_rank=0,
            pipeline_parallel_rank=2,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
            _managed_distributed_store=False,
        ),
    )
    start_on_host(SimpleNamespace(), 1, config)
    world = calls["world"][1]
    assert (world["world_size"], world["rank"], world["backend"]) == (8, 5, "gloo")
    assert calls["groups"][0] == (2, 4)
    assert calls["groups"][1]["custom_group_config"] == {}
