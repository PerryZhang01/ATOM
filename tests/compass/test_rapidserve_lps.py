# SPDX-License-Identifier: MIT
"""A simulated RapidServe run: its prefill and decode engines are engine LPs.

ATOM's API server with ``--enable-rapidserve`` runs a prefill and a decode
engine-core process behind one frontend. Decode sends prefill each request's
blocks on ``block_assignment``, and prefill sends decode each prefill's end on
``prefill_done``; a thread on each side takes them.

The run tests need a driver and `ATOM_COMPASS_SLICE_MODEL`, as in
`test_vertical_slice`; the rest run anywhere.
"""

import argparse
import dataclasses
import pickle
from types import SimpleNamespace

import pytest
import test_dp_slice
from test_dp_slice import MAX_TOKENS, run_dp_slice
from test_memory_readings import publish_readings
from test_vertical_slice import MODEL, NEEDS_A_RUN, _run_file

from atom import config as config_module
from atom.compass import parity
from atom.compass import run as compass_run
from atom.compass.clock import LpId, single_engine_table
from atom.compass.detect.scenario import check_run, expected_timeline
from atom.compass.runner import COMPASS_RAPIDSERVE_QUALNAME
from atom.compass.runner.overrides import (
    HostDeviceRecurrence,
    RapidServeRpcs,
    SampledIds,
)
from atom.model_engine.engine_core_mgr import CoreManager, DisaggCoreManager
from atom.utils import clock
from atom.utils.clock import LPRuntime

IPC = 2.0**-14
P, D = LpId("engine-P"), LpId("engine-D")
DONE_CH, BLOCKS_CH = (
    "engine-P->engine-D:prefill_done",
    "engine-D->engine-P:block_assignment",
)
#: The server flags of the run test, beside the slice's own.
FLAGS = ["--enable-rapidserve"]


def _table():
    return single_engine_table(
        admission_path="serving", ipc_s=IPC, stream_s=IPC, rapidserve=True
    )


def test_the_two_engines_join_the_frontend_and_each_other():
    table = _table()
    assert set(table.registry.ids()) == {LpId("traffic"), LpId("frontend"), P, D}
    assert (
        table.channel(DONE_CH).receive == table.channel(BLOCKS_CH).receive == "thread"
    )
    assert table.distance(P, D) == table.distance(D, P) == IPC
    names = {c.name for c in table.channels_from(LpId("frontend"))}
    assert "frontend->engine-D:request#dp1" in names
    with pytest.raises(ValueError, match="ipc_s is 0.0"):
        single_engine_table(
            admission_path="serving", ipc_s=0.0, stream_s=IPC, rapidserve=True
        )


def test_the_run_file_key_builds_the_table_every_lp_reads():
    run = {"admission_path": "serving", "ipc_s": IPC, "stream_s": IPC}
    assert compass_run.channel_table({**run, "rapidserve": True}).channel(BLOCKS_CH)
    with pytest.raises(KeyError):
        compass_run.channel_table(run).channel(BLOCKS_CH)


def _engine_config(decode: bool):
    pc = SimpleNamespace(
        pipeline_parallel_rank=0,
        pp_meta_addrs=[],
        pp_token_addr="",
        pp_kv_status_addr="",
    )
    return SimpleNamespace(
        parallel_config=pc,
        disagg_is_decode=decode,
        disagg_p2d_addr="p2d",
        disagg_d2p_addr="d2p",
    )


@pytest.mark.parametrize("lp, decode, tag", [(P, False, "#dp0"), (D, True, "#dp1")])
def test_each_engine_names_its_channels_by_the_index_its_frontend_gives_it(
    monkeypatch, lp, decode, tag
):
    rt = LPRuntime(lp, _table(), None)
    monkeypatch.setattr(clock, "_installed", rt)
    relay = clock.relay_queue(0)
    clock.name_endpoints(0, "in", "ctl", "out", _engine_config(decode))
    assert relay.wsock.ch == f"{lp.name}->frontend:output{tag}"
    kinds = [rt.endpoints[a] for a in ("in", "out", "p2d", "d2p")]
    assert kinds == [
        f"request{tag}",
        f"output{tag}",
        "prefill_done",
        "block_assignment",
    ]
    assert clock.socket_channels(rt, "prefill_done") == (DONE_CH,)
    assert clock.socket_channels(rt, "block_assignment") == (BLOCKS_CH,)


def _deployment(rapid=True, runner=compass_run.RAPIDSERVE_RUNNER, constrained=False):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(data_parallel_size=1, data_parallel_rank=0),
        tensor_parallel_size=1,
        enable_dp_attention=False,
        pipeline_parallel_size=1,
        enable_rapidserve=rapid,
        disagg_constrained=constrained,
        runner_qualname=runner,
        kv_transfer_config={},
    )


@pytest.mark.parametrize(
    "rapidserve, deployment, refusal",
    [
        (False, _deployment(), "enable_rapidserve is True and the run file's"),
        (True, _deployment(rapid=False), "enable_rapidserve is False and the run"),
        (True, _deployment(constrained=True), "--disagg-constrained has decode"),
        (
            True,
            _deployment(runner=compass_run.ATOM_RUNNER),
            f"prices its steps with {COMPASS_RAPIDSERVE_QUALNAME}",
        ),
    ],
)
def test_a_rapidserve_deployment_the_run_file_does_not_state_is_refused(
    monkeypatch, tmp_path, rapidserve, deployment, refusal
):
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path, rapidserve=rapidserve)))
    with pytest.raises(ValueError, match=refusal):
        compass_run.frontend(deployment)
    assert clock.installed() is None


def test_a_real_rapidserve_run_records_through_its_recording_runner(
    monkeypatch, tmp_path
):
    monkeypatch.delenv(compass_run.ENV, raising=False)
    monkeypatch.setenv(parity.ENV, str(tmp_path))
    for cls in (CoreManager, DisaggCoreManager):
        monkeypatch.setattr(cls, "add_request", cls.add_request)
    config = _deployment()
    with compass_run.frontend(config):
        pass
    assert config.runner_qualname == parity.RAPID


@pytest.mark.parametrize(
    "config, where",
    [
        (SimpleNamespace(enable_rapidserve=False), "r"),
        (SimpleNamespace(enable_rapidserve=True, disagg_is_decode=False), "r/prefill"),
        (SimpleNamespace(enable_rapidserve=True, disagg_is_decode=True), "r/decode"),
    ],
)
def test_each_rapidserve_engine_records_apart(tmp_path, config, where):
    assert parity.record_dir(tmp_path / "r", config) == tmp_path / where


def test_the_runner_answers_every_rpc_the_rapidserve_cores_wait_on():
    assert set(config_module.RAPIDSERVE_RPCS) <= set(vars(RapidServeRpcs))
    assert COMPASS_RAPIDSERVE_QUALNAME in config_module.RAPIDSERVE_RUNNERS


class _Base:
    def allocate_kv_cache(self, n):
        self.allocated = n
        return True


class _Runner(RapidServeRpcs, _Base):
    def __init__(self, decode):
        self.config = SimpleNamespace(disagg_is_decode=decode)


def test_decode_takes_its_block_count_with_the_pool_and_allocates_none_before():
    decode = _Runner(decode=True)
    assert decode.allocate_kv_cache(0) is True
    assert not hasattr(decode, "allocated")
    assert decode.import_kv_cache_ipc_handle(decode.export_kv_cache_ipc_handle(), 7)
    assert decode.allocated == 7
    prefill = _Runner(decode=False)
    assert prefill.allocate_kv_cache(5) and prefill.allocated == 5


def test_a_prefill_reply_is_a_token_per_request_with_the_step_price(monkeypatch):
    runner = _Runner(decode=False)
    out = SimpleNamespace(predicted_s=0.25, out_s=None, group_targets=None)
    monkeypatch.setattr(_Base, "forward", lambda self, batch: out, raising=False)
    # As `RapidServeModelRunner.__init__` binds it on a prefill engine.
    runner.forward = runner.prefill_forward
    runner._token_stream = SimpleNamespace(token_ids=[10, 11, 12])
    reply = runner.prefill_forward(SimpleNamespace(req_ids=[4, 2, 3]))
    sent = pickle.loads(pickle.dumps(reply))
    assert isinstance(sent, SampledIds) and sent == [11, 12, 10]
    assert (sent.predicted_s, sent.out_s, sent.group_out) == (0.25, None, None)


def _batch(output):
    return SimpleNamespace(
        total_seqs_num_prefill=1,
        total_tokens_num=8,
        total_seqs_num=1,
        produces_output=lambda: output,
    )


@pytest.mark.parametrize(
    "sync, output, charges",
    [
        # The previous output forward's device end, or the launch.
        (False, True, [0.0, 1.0]),
        (False, False, [0.0, 0.0]),
        # Its own device end, output or middle chunk.
        (True, True, [1.0, 1.0]),
        (True, False, [1.0, 1.0]),
    ],
)
def test_a_synchronizing_forward_leaves_at_its_own_device_end(sync, output, charges):
    clock = [0.0]
    host = HostDeviceRecurrence({}, (), lambda: clock[0], sync=sync)
    got = []
    for _ in charges:
        got.append(host.charge(_batch(output), 1.0, 0.0))
        clock[0] += got[-1]
    assert got == charges


def test_the_constant_stub_timeline_ends_each_synchronized_forward_at_its_device_end():
    step = {"rows": [[5, 5, False]], "produces_output": True, "is_dummy_run": False}
    record = {0: [step, step]}
    tars = [(0.0, 0.5), (0.5, 1.0)]
    assert expected_timeline(record, 0.5, 0.25, tars, sync=True) == tars
    assert expected_timeline(record, 0.5, 0.25, tars) == [(0.0, 0.0), (0.5, 0.5)]


# --- with a driver: the run -------------------------------------------------


def _rapidserve_run_file(out, **overrides):
    """The slice's run file with ``rapidserve``, and the memory readings of the
    server's configuration planted in its store."""
    from atom.config import Config
    from atom.model_engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser()
    EngineArgs.add_cli_args(parser)
    args = parser.parse_args(
        ["--model", MODEL, *FLAGS, "--enforce-eager", "--max-model-len", "2048"]
    )
    kwargs = EngineArgs.from_cli_args(args)._get_engine_kwargs()
    names = {f.name for f in dataclasses.fields(Config)}
    config = Config(MODEL, **{k: v for k, v in kwargs.items() if k in names})
    publish_readings(out / "store", config, document=overrides["machine"])
    return _run_file(out, rapidserve=True, **overrides)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rapidserve")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(test_dp_slice, "_run_file", _rapidserve_run_file)
        # The server sets it, and the readings are keyed by it.
        mp.setenv("AITER_LOG_LEVEL", "WARNING")
        return [run_dp_slice(root / s, s, FLAGS, dp=1) for s in ("1", "2")]


@NEEDS_A_RUN
def test_rapidserve_meets_every_scenario_check_on_both_engines(runs):
    run = runs[0]
    print(f"\n{run['table']}")
    parity_dir = run["out"] / "parity"
    # A real RapidServe server streams each request every token but the one
    # prefill samples, and its usage counts what it streamed.
    failed = check_run(
        [r["out"] for r in runs],
        [n - 1 for n in MAX_TOKENS],
        [line for _, line in run["events"]],
        {"engine-P": parity_dir / "prefill", "engine-D": parity_dir / "decode"},
    )
    assert not failed, "\n".join(failed)
    rows = [line.split() for line in run["table"].splitlines()[1:]]
    released = [r[4] for r in rows if r[3] == "release"]
    # One block assignment and one prefill end per request.
    assert released.count(BLOCKS_CH) == released.count(DONE_CH) == len(MAX_TOKENS)
    assert {r[0] for r in rows if r[3] == "TAR"} >= {"engine-P", "engine-D"}
