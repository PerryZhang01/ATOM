# SPDX-License-Identifier: MIT
"""The bootstrap of a prefill-decode deployment on the simulated KV connector.

An engine process wrapped in `compass_run.engine` builds the scheduler half of
its connector the way `Scheduler` does, against a clock authority served in
this process over the run file's `prefill_decode_table`.
"""

import copy
import json
import uuid

import pytest
from conftest import atom_config_double
from test_kv_simulated_connector import CONFIG_JSON
from test_memory_readings import DOCUMENT
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.backends import KvGeometry
from atom.compass.clock import ClockAuthority, LpId
from atom.compass.clock_transport import serve
from atom.compass.kv import WRITE_REQ_KEY, SimulatedKVConnectorScheduler
from atom.kv_transfer.disaggregation.factory import KVConnectorFactory
from atom.utils import clock
from atom.utils.distributed import utils as dist_utils

BLOCK_SIZE = 64
BLOCKS = 10


@pytest.fixture
def hf():
    return PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )


@pytest.fixture
def run_file(tmp_path, monkeypatch):
    run = {
        "clock_endpoint": f"inproc:test-run-kv-{uuid.uuid4().hex}",
        "bound_s": 60.0,
        "admission_path": "serving",
        "ipc_s": 2.0**-14,
        "stream_s": 2.0**-12,
        "router_s": 2.0**-12,
        "kv_write_req_s": 2.0**-10,
        "kv_link": "inter_node",
        "machine": copy.deepcopy(DOCUMENT),
        "out_dir": str(tmp_path),
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(path))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(dist_utils, "LP_OF_RANK", None)
    server = serve(
        ClockAuthority(compass_run.channel_table(run), bound_s=run["bound_s"]),
        run["clock_endpoint"],
    )
    yield run
    server.close()


def _config(hf, role):
    kv = {
        "kv_connector": "compass",
        "kv_role": role,
        WRITE_REQ_KEY: f"inproc://kv-write-req-{uuid.uuid4().hex}",
    }
    return atom_config_double(
        kv_transfer_config=kv, hf_config=hf, kv_cache_block_size=BLOCK_SIZE
    )


@pytest.mark.parametrize(
    "role, lp", [("kv_producer", "engine-P"), ("kv_consumer", "engine-D")]
)
def test_a_bootstrapped_engine_builds_its_connector_priced_from_the_run_file(
    run_file, hf, role, lp
):
    config = _config(hf, role)
    with compass_run.engine(config):
        half = KVConnectorFactory.create_connector(config, role="scheduler")
    assert isinstance(half, SimulatedKVConnectorScheduler)
    assert clock.installed().me == LpId(lp)
    assert dist_utils.LP_OF_RANK == {0: LpId(lp)}
    link = DOCUMENT["interconnect"]["inter_node"]
    block = KvGeometry.from_hf_config(hf, block_size=BLOCK_SIZE).bytes_per_block
    assert half._transfer.duration_s(BLOCKS) == pytest.approx(
        link["link_latency_s"]
        + BLOCKS * block / (link["link_bandwidth_bytes_per_s"] * link["derate"])
    )
    half._sock.raw.close(linger=0)


def test_the_co_hosting_frontend_refuses_a_prefill_decode_deployment(run_file, hf):
    config = _config(hf, "kv_producer")
    config.runner_qualname = compass_run.ATOM_RUNNER
    with pytest.raises(ValueError, match="prefill-decode run spans two API servers"):
        compass_run.frontend(config)
    assert clock.installed() is None
