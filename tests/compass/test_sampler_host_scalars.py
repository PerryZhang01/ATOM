# SPDX-License-Identifier: MIT
"""Top-k/top-p sampling reads nothing back from the device.

`ModelRunner.prepare_sample` hands a k or p shared by the whole batch to the
sampler as a host scalar, and the sampler passes it to aiter as the scalar
argument. The sampler tests run on the meta device, where reading a value
raises, so any host-blocking read on the path fails them.
"""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import atom
from atom.utils import CpuGpuBuffer

_AITER = (
    "aiter",
    "aiter.ops",
    "aiter.ops.triton",
    "aiter.ops.triton.softmax",
    "aiter.ops.triton.topk",
    "aiter.ops.sampling",
)


@pytest.fixture
def sampler(monkeypatch):
    """`atom/model_ops/sampler.py` loaded against a fake aiter, and its op calls.

    Loaded under a private name, so the real module, where one is importable,
    is left as it was.
    """
    for name in _AITER:
        stub = types.ModuleType(name)
        stub.__getattr__ = lambda _attr: MagicMock()
        monkeypatch.setitem(sys.modules, name, stub)
    path = pathlib.Path(atom.__file__).parent / "model_ops" / "sampler.py"
    spec = importlib.util.spec_from_file_location("_sampler_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    calls = []

    def op(name):
        def run(probs, *args, **kwargs):
            calls.append((name, args))
            return torch.empty(probs.shape[0], 1, dtype=torch.long, device=probs.device)

        return run

    monkeypatch.setattr(
        module,
        "aiter_ops",
        SimpleNamespace(
            top_k_top_p_sampling_from_probs=op("k+p"),
            top_p_sampling_from_probs=op("p"),
        ),
    )
    monkeypatch.setattr(module, "AITER_TOPK_TOPP_AVAILABLE", True)
    return module, calls


def _meta(*shape):
    return torch.empty(*shape, device="meta")


def test_uniform_k_and_p_reach_aiter_as_the_host_scalars(sampler):
    module, calls = sampler
    out = module.Sampler()(_meta(4, 32), _meta(4), top_ks=20, top_ps=0.95)
    assert out.shape == (4,)
    assert calls == [("k+p", (None, None, 20, None, 0.95))]


def test_per_row_k_stays_a_tensor_and_is_not_read(sampler):
    module, calls = sampler
    top_ks = torch.empty(4, dtype=torch.int32, device="meta")
    module.Sampler()(_meta(4, 32), _meta(4), top_ks=top_ks, top_ps=0.95)
    [(name, args)] = calls
    assert name == "k+p"
    assert args[1] is top_ks and args[2:] == (0, None, 0.95)


def test_top_p_alone_reads_nothing_back(sampler):
    module, calls = sampler
    module.Sampler()(_meta(2, 32), _meta(2), top_ps=0.9)
    assert calls == [("p", (None, None, 0.9))]


def test_the_native_fallback_takes_host_scalars(sampler, monkeypatch):
    module, _ = sampler
    monkeypatch.setattr(module, "AITER_TOPK_TOPP_AVAILABLE", False)
    logits = torch.randn(3, 16)
    with pytest.warns(UserWarning, match="native"):
        out = module.Sampler()(logits, torch.ones(3), top_ks=1, top_ps=0.5)
    assert out.tolist() == logits.argmax(dim=-1).tolist()


def test_prepare_sample_hands_over_a_shared_k_and_p_as_host_scalars():
    mod = pytest.importorskip(
        "atom.model_engine.model_runner",
        reason="model_runner imports aiter at module load",
        exc_type=ImportError,
    )
    runner = object.__new__(mod.ModelRunner)
    kw = {"device": "cpu", "pin_memory": False}
    runner.forward_vars = {
        "temperatures": CpuGpuBuffer(4, dtype=torch.float, **kw),
        "top_ks": CpuGpuBuffer(4, dtype=torch.int32, **kw),
        "top_ps": CpuGpuBuffer(4, dtype=torch.float, **kw),
    }

    def batch(top_ks, top_ps):
        return SimpleNamespace(
            total_seqs_num=len(top_ks),
            temperatures=np.ones(len(top_ks), dtype=np.float32),
            top_ks=np.asarray(top_ks, dtype=np.int32),
            top_ps=np.asarray(top_ps, dtype=np.float32),
        )

    _, top_ks, top_ps, _, _ = runner.prepare_sample(batch([20], [0.95]))
    assert (type(top_ks), top_ks) == (int, 20)
    assert (type(top_ps), top_ps) == (float, float(np.float32(0.95)))

    _, top_ks, top_ps, _, _ = runner.prepare_sample(batch([20, 40], [0.95, 0.95]))
    assert top_ks.tolist() == [20, 40]
    assert type(top_ps) is float
