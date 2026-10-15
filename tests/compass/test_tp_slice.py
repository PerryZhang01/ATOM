# SPDX-License-Identifier: MIT
"""A simulated run of one engine several ranks wide: its ranks are one engine LP.

ATOM's API server at ``-tp 2`` runs one engine-core process and two worker
processes. Each worker starts on the host with its groups on gloo; only TP
rank 0 prices a step and replies, and the engine LP is the engine core's step
loop, as at TP 1. ``-pcp 2`` and ``-tp 2 -dcp 2`` start the same way.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`, and the prefill-decode one atomesh, as in
`test_pd_slice`; the rest run anywhere.
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from test_dp_slice import MAX_TOKENS, device_classes_put_back, run_dp_slice  # noqa: F401
from test_dp_step_max import prefill, runner
from test_pd_slice import NEEDS_A_PD_RUN, run_pd
from test_vertical_slice import MAX_TOKENS as ONE_REQUEST
from test_vertical_slice import NEEDS_A_RUN

from atom.compass.backends.step_stub import StepStubBackend
from atom.compass.detect.scenario import check_run
from atom.compass.runner.overrides import start_on_host


@pytest.fixture(scope="module")
def tp2(tmp_path_factory):
    root = tmp_path_factory.mktemp("tp2")
    return [run_dp_slice(root / s, s, ["-tp", "2"], dp=1) for s in ("1", "2")]


@NEEDS_A_RUN
def test_a_tp2_engine_meets_every_scenario_check(tp2):
    print(f"\n{tp2[0]['table']}")
    failed = check_run(
        [r["out"] for r in tp2],
        MAX_TOKENS,
        [line for _, line in tp2[0]["events"]],
        {"engine": tp2[0]["out"] / "parity"},
    )
    assert not failed, "\n".join(failed)


@NEEDS_A_RUN
@pytest.mark.parametrize("width", [["-pcp", "2"], ["-tp", "2", "-dcp", "2"]])
def test_a_context_parallel_engine_meets_the_run_properties(tmp_path, width):
    runs = [run_dp_slice(tmp_path / s, s, width, dp=1) for s in ("1", "2")]
    failed = check_run([r["out"] for r in runs])
    assert not failed, "\n".join(failed)


@NEEDS_A_PD_RUN
def test_a_tp2_prefill_and_decode_meet_every_scenario_check(tmp_path, monkeypatch):
    monkeypatch.setenv("SERVER_ARGS", "-tp 2 --enforce-eager --max-model-len 2048")
    runs = [run_pd(tmp_path / s, s) for s in ("1", "2")]
    parity = runs[0]["out"] / "parity"
    failed = check_run(
        [r["out"] for r in runs],
        [ONE_REQUEST],
        [line for _, line in runs[0]["events"]],
        {"engine-P": parity / "prefill", "engine-D": parity / "decode"},
    )
    assert not failed, "\n".join(failed)


# --- without a driver ---------------------------------------------------------


def _config(tp=1, pcp=1, dcp=1, dp=1, dp_rank=0, managed=False):
    return SimpleNamespace(
        tensor_parallel_size=tp,
        tp_world_size=tp,
        prefill_context_parallel_size=pcp,
        decode_context_parallel_size=dcp,
        pipeline_parallel_size=1,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_rank=dp_rank,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
            _managed_distributed_store=managed,
        ),
    )


@pytest.fixture
def aiter_calls(monkeypatch):
    """The calls `start_on_host` makes into aiter and torch.distributed, in order."""
    calls = []

    def record(name):
        return lambda *a, **k: calls.append((name, a, k)) or name

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
    monkeypatch.setattr(torch.distributed, "TCPStore", record("store"))
    monkeypatch.setattr(torch.distributed, "init_process_group", record("group"))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)
    return calls


def test_a_tp_pcp_dcp_rank_builds_every_group_with_no_device_communicator(
    aiter_calls,
):
    runner = SimpleNamespace()
    start_on_host(runner, 3, _config(tp=2, pcp=2, dcp=2))
    assert runner.device == torch.device("cpu")
    world = {
        "world_size": 4,
        "rank": 3,
        "distributed_init_method": "tcp://127.0.0.1:29501",
        "backend": "gloo",
        "local_rank": 0,
        "data_parallel_size": 1,
        "data_parallel_rank": 0,
    }
    groups = {
        "decode_context_model_parallel_size": 2,
        "data_parallel_size": 1,
        "prefill_context_model_parallel_size": 2,
        "custom_group_config": {},
    }
    assert aiter_calls == [("world", (), world), ("groups", (2, 1), groups)]


def test_a_tp_rank_of_a_dp_rank_joins_the_managed_store_at_its_global_rank(
    aiter_calls,
):
    start_on_host(SimpleNamespace(), 1, _config(tp=2, dp=2, dp_rank=1, managed=True))
    group = {"backend": "gloo", "store": "store", "rank": 3, "world_size": 4}
    assert aiter_calls[1] == ("group", (), group)
    assert aiter_calls[2][2]["world_size"] == 2
    assert aiter_calls[3][1] == (2, 1)


class _Unpriceable(StepStubBackend):
    def price(self, batch_view):
        raise AssertionError("a rank other than TP rank 0 priced a step")


def test_only_tp_rank_0_prices_a_step():
    other = runner(_Unpriceable())
    other.rank = 1
    assert other.forward(prefill(8)).predicted_s == 0.0
