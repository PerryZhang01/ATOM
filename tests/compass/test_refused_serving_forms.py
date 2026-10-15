# SPDX-License-Identifier: MIT
"""Serving forms a simulated run cannot drive are refused where they start, by name.

An offline `LLMEngine`, the Atomesh entrypoints, a sync utility command on the
frontend's clock owner, and a second instance of one role at the authority.
"""

import dataclasses
import json
import threading
from types import SimpleNamespace

import pytest
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.compass.clock import ClockAuthority, LpId, prefill_decode_table
from atom.compass.clock_transport import connect, serve, service
from atom.entrypoints.openai import api_server
from atom.model_engine import llm_engine
from atom.model_engine.engine_core_mgr import CoreManager
from atom.utils import clock
from atom.utils.distributed import utils


@pytest.fixture
def run_file(monkeypatch, tmp_path):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": f"inproc:{tmp_path}",
        "out_dir": str(tmp_path),
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.delenv("USE_ATOMESH_ENTRYPOINTS", raising=False)
    for module, name in [
        (compass_run, "_authority"),
        (compass_run, "_member"),
        (clock, "_installed"),
        (utils, "LP_OF_RANK"),
    ]:
        monkeypatch.setattr(module, name, None)
    # The frontend wraps this method for the process; put ATOM's back after.
    monkeypatch.setattr(
        CoreManager,
        "broadcast_utility_command_sync",
        CoreManager.broadcast_utility_command_sync,
    )
    yield run
    # A frontend that got as far as co-hosting the authority: end its run, so
    # neither the early-end poller nor the serve loop outlives the test.
    if compass_run._authority is not None:
        compass_run._authority.done.set()
        compass_run._early_end.join()
        service._SERVED[run["clock_endpoint"]].close()


@dataclasses.dataclass
class _Config:
    """What `LLMEngine.__init__` reads of its config before the frontend."""

    model: str
    runner_qualname: str = compass_run.ATOM_RUNNER
    stop_token_ids: list = dataclasses.field(default_factory=list)
    parallel_config: SimpleNamespace = dataclasses.field(
        default_factory=lambda: SimpleNamespace(data_parallel_size=1)
    )
    tensor_parallel_size: int = 1
    enable_dp_attention: bool = False
    enable_expert_parallel: bool = False
    enable_tbo: bool = False
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    kv_cache_block_size: int = 16
    enable_rapidserve: bool = False
    kv_transfer_config: dict = dataclasses.field(default_factory=dict)


def _build_engine(**config):
    return llm_engine.LLMEngine(
        "model", tokenizer=SimpleNamespace(bos_token_id=1, eos_token_id=2), **config
    )


@pytest.fixture
def engine_parts(monkeypatch):
    monkeypatch.setattr(llm_engine, "Config", _Config)
    monkeypatch.setattr(llm_engine, "InputOutputProcessor", lambda *_: None)


def test_an_offline_engine_is_refused(run_file, engine_parts):
    with pytest.raises(ValueError, match="an offline LLMEngine has no client"):
        _build_engine()
    assert clock.installed() is None


def test_the_engine_the_api_server_builds_is_not_refused_as_offline(
    run_file, engine_parts
):
    """Built under a function at the API server's ``main`` file and name, the
    frontend goes on to its next check, here the runner's."""
    main = api_server.main.__code__
    scope = {"build": lambda: _build_engine(runner_qualname="my.Runner")}
    code = compile(f"def {main.co_name}():\n    build()\n", main.co_filename, "exec")
    exec(code, scope)  # noqa: S102 - defines a stand-in at the server's main
    with pytest.raises(ValueError, match="runner 'my.Runner' is named"):
        scope[main.co_name]()


def test_the_atomesh_entrypoints_are_refused(run_file, engine_parts, monkeypatch):
    """The Atomesh service builds its engine outside the API server's ``main``,
    so this reason, not the offline one, names it."""
    monkeypatch.setenv("USE_ATOMESH_ENTRYPOINTS", "1")
    with pytest.raises(ValueError, match="USE_ATOMESH_ENTRYPOINTS=1 serves through"):
        _build_engine()
    assert clock.installed() is None


@pytest.mark.parametrize(
    "call, cmd",
    [
        (llm_engine.LLMEngine.start_profile, "start_profile"),
        (llm_engine.LLMEngine.stop_profile, "stop_profile"),
        (llm_engine.LLMEngine.get_mtp_statistics, "get_mtp_statistics"),
        (llm_engine.LLMEngine.get_cache_statistics, "get_cache_statistics"),
    ],
)
def test_a_sync_utility_command_on_the_clock_owner_is_refused(run_file, call, cmd):
    compass_run.frontend(_deployment())
    rt = clock.installed()
    engine = SimpleNamespace(core_mgr=CoreManager.__new__(CoreManager))
    with pytest.raises(RuntimeError, match=f"sync utility command '{cmd}' on the"):
        call(engine)
    assert rt.refusals == [f"clock:sync utility command '{cmd}' on the clock owner"]
    # Off the owner the call is ATOM's, which reads state this stub lacks.
    raised = []

    def off_owner():
        try:
            call(engine)
        except Exception as e:  # noqa: BLE001 - the test reads which one
            raised.append(e)

    worker = threading.Thread(target=off_owner)
    worker.start()
    worker.join()
    assert isinstance(raised[0], AttributeError), raised
    assert len(rt.refusals) == 1


def test_a_second_instance_of_one_role_is_refused_naming_xpyd(tmp_path):
    table = prefill_decode_table(
        admission_path="serving",
        ipc_s=0.001,
        stream_s=0.002,
        router_s=0.003,
        kv_write_req_s=0.004,
    )
    server = serve(ClockAuthority(table), f"inproc:{tmp_path}")
    try:
        connect(LpId("frontend-P"), server.endpoint)
        with pytest.raises(KeyError, match="frontend-P is already bound.*xPyD"):
            connect(LpId("frontend-P"), server.endpoint)
    finally:
        server.close()
