# SPDX-License-Identifier: MIT
"""The simulated runner's start sets no device and builds nothing on one.

`start_on_host` runs with aiter stood in, because importing aiter needs a
driver; what it is checked against is ATOM's own `CpuGpuBuffer` and torch's own
`torch.cuda.Stream`, the two things `ModelRunner.__init__` builds after it.
"""

import ast
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

from atom.compass.runner.overrides import HostStream, RunnerRefusal, start_on_host
from atom.utils import CpuGpuBuffer

REPO = pathlib.Path(__file__).resolve().parents[2]


def _config(tp=1, dp=1, simulated_tp=None):
    return SimpleNamespace(
        tp_world_size=tp,
        tensor_parallel_size=tp if simulated_tp is None else simulated_tp,
        prefill_context_parallel_size=1,
        pipeline_parallel_size=1,
        master_addr="127.0.0.1",
        port=29500,
        parallel_config=SimpleNamespace(
            data_parallel_size=dp,
            data_parallel_master_ip="127.0.0.1",
            data_parallel_base_port=29501,
        ),
    )


@pytest.fixture
def started(monkeypatch):
    """Run `start_on_host`, recording the group call; undo what it replaces."""
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(init_dist_env=lambda *a, **k: calls.append((a, k))),
    )
    monkeypatch.setitem(
        sys.modules,
        "aiter.dist.utils",
        SimpleNamespace(
            get_distributed_init_method=lambda ip, port: f"tcp://{ip}:{port}"
        ),
    )
    monkeypatch.setattr(torch.cuda, "Stream", torch.cuda.Stream)
    monkeypatch.setattr(CpuGpuBuffer, "__init__", CpuGpuBuffer.__init__)
    monkeypatch.delenv("MASTER_ADDR", raising=False)
    monkeypatch.delenv("MASTER_PORT", raising=False)

    def no_device(*a, **k):
        raise AssertionError("the start set a device")

    monkeypatch.setattr(torch.cuda, "set_device", no_device)

    def start(config):
        runner = SimpleNamespace()
        start_on_host(runner, 0, config)
        return runner, calls

    return start


def test_the_start_sets_the_host_as_the_device_and_builds_groups_on_gloo(started):
    runner, calls = started(_config())
    assert runner.device == torch.device("cpu")
    ((args, kwargs),) = calls
    assert args == (1,)
    assert kwargs["backend"] == "gloo"
    assert kwargs["distributed_init_method"] == "tcp://127.0.0.1:29501"


def test_atoms_buffers_built_after_it_are_unpinned_and_on_the_host(started):
    started(_config())
    buffer = CpuGpuBuffer(8, dtype=torch.int32, device=torch.device("cpu"))
    assert not buffer.cpu.is_pinned()
    assert buffer.gpu.device == torch.device("cpu")
    assert not buffer.clone().cpu.is_pinned()


def test_a_stream_built_after_it_is_a_host_stand_in_that_cannot_queue_work(started):
    started(_config())
    stream = torch.cuda.Stream(torch.device("cpu"))
    assert isinstance(stream, HostStream)
    with pytest.raises(AttributeError):
        stream.synchronize()


@pytest.mark.parametrize("width", [{"tp": 2}, {"simulated_tp": 2}])
def test_a_start_wider_than_one_rank_is_refused_before_anything_is_replaced(
    started, width
):
    stream = torch.cuda.Stream
    with pytest.raises(RunnerRefusal, match="2 ranks wide"):
        started(_config(**width))
    assert torch.cuda.Stream is stream


def test_atoms_init_builds_streams_and_buffers_only_after_the_replaced_setup():
    """What the start relies on, read from ATOM's `ModelRunner.__init__`."""
    tree = ast.parse((REPO / "atom/model_engine/model_runner.py").read_text())
    init = next(
        f
        for c in tree.body
        if isinstance(c, ast.ClassDef) and c.name == "ModelRunner"
        for f in c.body
        if isinstance(f, ast.FunctionDef) and f.name == "__init__"
    )
    body = [ast.unparse(s) for s in init.body]

    def first(text):
        return next(i for i, s in enumerate(body) if text in s)

    setup = first("self._setup_device_and_distributed(")
    for later in ("tokenIDProcessor(", "torch.cuda.Stream(", "allocate_forward_vars("):
        assert setup < first(later), later
