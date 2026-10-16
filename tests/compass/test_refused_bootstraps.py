# SPDX-License-Identifier: MIT
"""Deployments a run file cannot simulate are refused at their bootstrap, by name.

The vLLM, SGLang and rtpllm plugins, the diffusion engine, and a multi-node
data-parallel group.
"""

import json

import pytest
from test_dp_slice import _deployment

from atom.compass import run as compass_run
from atom.config import ParallelConfig
from atom.diffusion.engine import diffusion_engine
from atom.plugin import prepare
from atom.utils import clock


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    run = {
        "admission_path": "serving",
        "ipc_s": 0.001,
        "stream_s": 0.002,
        "bound_s": 10.0,
        "clock_endpoint": f"inproc:{tmp_path}",
        "out_dir": str(tmp_path),
        "data_parallel_size": 2,
    }
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    monkeypatch.setattr(clock, "_installed", None)
    monkeypatch.setattr(prepare, "_CURRENT_FRAMEWORK", prepare._CURRENT_FRAMEWORK)


@pytest.mark.parametrize("framework", ["vllm", "sglang", "rtpllm"])
def test_a_plugin_is_refused(run_env, framework):
    with pytest.raises(ValueError, match=f"the {framework} plugin builds its model"):
        prepare._set_framework_backbone(framework)
    assert not prepare.is_plugin_mode()


def test_server_mode_is_not_refused(run_env):
    prepare._set_framework_backbone("atom")
    assert not prepare.is_plugin_mode()


def test_the_diffusion_engine_is_refused(run_env):
    with pytest.raises(ValueError, match="the diffusion engine has no simulated"):
        diffusion_engine.DiffusionEngine(None)


def _dp_two(**slice_):
    """Two DP ranks, with --fake-eplb, the check after the multi-node one."""
    deployment = _deployment(dp=2)
    deployment.parallel_config = ParallelConfig(data_parallel_size=2, **slice_)
    deployment.fake_eplb = True
    return deployment


@pytest.mark.parametrize("rank", [0, 1])
def test_a_multinode_dp_group_is_refused(run_env, rank):
    """Each node's slice of two DP ranks, one per node."""
    deployment = _dp_two(data_parallel_size_local=1, data_parallel_rank=rank)
    with pytest.raises(ValueError, match="part of a multi-node data-parallel group"):
        compass_run.frontend(deployment)
    assert clock.installed() is None


def test_single_node_dp_goes_on_to_its_next_check(run_env):
    with pytest.raises(ValueError, match="--fake-eplb"):
        compass_run.frontend(_dp_two())
