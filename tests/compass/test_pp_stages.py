# SPDX-License-Identifier: MIT
"""A simulated run at pipeline width above one: each stage is an engine LP.

ATOM's API server at ``-pp 2`` or ``-pp 4`` runs one engine-core process per
stage. The head schedules, sends each downstream stage its batch on
``meta#pp<k>``, and the last stage sends the head its output on ``tokens``;
each stage's worker starts on the host with its groups, the PP group among
them, on gloo. The hidden states one stage hands the next inside the forward
are not modelled here.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import sys
from types import SimpleNamespace

import pytest
from test_dp_slice import (  # noqa: F401
    MAX_TOKENS,
    _deployment,
    device_classes_put_back,
    run_dp_slice,
)
from test_vertical_slice import NEEDS_A_RUN, _run_file

from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.detect.scenario import check_run
from atom.compass.runner.overrides import start_on_host
from atom.utils import clock
from atom.utils.clock import LPRuntime
from atom.utils.distributed import utils

IPC = 2.0**-14


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


def test_a_deployment_whose_pipeline_the_run_file_does_not_state_is_refused(
    monkeypatch, tmp_path
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    with pytest.raises(ValueError, match="pipeline_parallel_size is 1"):
        compass_run.frontend(_stage(0))
    assert clock.installed() is None


def test_each_stage_is_an_engine_lp_of_its_own(monkeypatch, tmp_path):
    run = _run_file(tmp_path, pipeline_parallel_size=2)
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
