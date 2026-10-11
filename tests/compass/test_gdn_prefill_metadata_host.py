# SPDX-License-Identifier: MIT
# GDN prefill metadata is built from host lengths, not read back from the device.
#
# The device copy of `cu_seqlens_q` is the one the step's H2D has not
# necessarily landed yet; reading it back makes the host wait for every kernel
# queued before it. Here the device copy holds the previous step's lengths, so
# a build that reads it gets the wrong program map.

import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch

from atom.utils import CpuGpuBuffer


class _AiterStandIn(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Every `aiter` module as a stand-in. `gdn_attn` imports aiter at module
    scope, and without a driver aiter's import raises; nothing here calls it."""

    def find_spec(self, name, path, target=None):
        if name == "aiter" or name.startswith("aiter."):
            return importlib.machinery.ModuleSpec(name, self, is_package=True)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        module.__getattr__ = lambda _attr: MagicMock()
        return module

    def exec_module(self, module):
        pass


if torch.cuda.is_available():
    from atom.model_ops.attentions.gdn_attn import (
        GDNStateMixin,
        compute_causal_conv1d_metadata,
    )
else:
    _finder = _AiterStandIn()
    sys.meta_path.insert(0, _finder)
    try:
        from atom.model_ops.attentions.gdn_attn import (
            GDNStateMixin,
            compute_causal_conv1d_metadata,
        )
    finally:
        sys.meta_path.remove(_finder)
        for _name in [n for n in sys.modules if n.split(".")[0] == "aiter"]:
            del sys.modules[_name]

BLOCK_M = 8


def buf(n):
    return CpuGpuBuffer(n, dtype=torch.int32, device="cpu", pin_memory=False)


def build(host_cu, stale_device_cu):
    var = {
        "cu_seqlens_q": buf(8),
        "gdn_batch_ptr": buf(64),
        "gdn_token_chunk_offset_ptr": buf(64),
    }
    var["cu_seqlens_q"].np[: len(host_cu)] = host_cu
    stub = SimpleNamespace(
        model_runner=SimpleNamespace(forward_vars=var),
        use_spec_decode=False,
        replayssm=False,
        device="cpu",
        prepare_state_indices=lambda batch, with_spec: None,
        non_spec_state_indices_tensor=buf(8),
        non_spec_state_indices_in_tensor=buf(8),
    )
    n, tokens = len(host_cu) - 1, int(host_cu[-1])
    batch = SimpleNamespace(
        total_seqs_num_decode=0,
        total_seqs_num_prefill=n,
        total_tokens_num_decode=0,
        total_tokens_num_prefill=tokens,
        total_seqs_num=n,
        total_tokens_num=tokens,
    )
    attn_metadata = SimpleNamespace(
        cu_seqlens_q=torch.tensor(stale_device_cu, dtype=torch.int32),
        num_cached_tokens=None,
    )
    return stub, batch, attn_metadata, var


def test_program_map_follows_the_host_lengths():
    # Lengths 17, 8, 1: three chunks, one, one. The device copy says 8, 8, 8.
    stub, batch, attn_metadata, _ = build([0, 17, 25, 26], [0, 8, 16, 24, 24])
    md = GDNStateMixin.prepare_gdn_metadata(
        stub, batch, attn_metadata, is_prefill=True, prepare_block_tables=False
    )
    entry = md.nums_dict[BLOCK_M]
    assert entry["tot"] == entry["mlist_len"] == 5
    assert md.batch_ptr.tolist() == [0, 0, 0, 1, 2]
    assert md.token_chunk_offset_ptr.tolist() == [0, 1, 2, 0, 0]
    assert entry["batch_ptr"] is md.batch_ptr


def test_program_map_is_staged_in_the_runner_buffers():
    """The device tensors are views of forward_vars, which the pipeline ring
    clones per in-flight slot, not a fresh allocation per step."""
    stub, batch, attn_metadata, var = build([0, 3, 20], [0, 3, 20])
    md = GDNStateMixin.prepare_gdn_metadata(
        stub, batch, attn_metadata, is_prefill=True, prepare_block_tables=False
    )
    assert md.batch_ptr.data_ptr() == var["gdn_batch_ptr"].gpu.data_ptr()
    assert (
        md.token_chunk_offset_ptr.data_ptr()
        == var["gdn_token_chunk_offset_ptr"].gpu.data_ptr()
    )
    np.testing.assert_array_equal(md.batch_ptr.numpy(), [0, 1, 1, 1])
    np.testing.assert_array_equal(md.token_chunk_offset_ptr.numpy(), [0, 0, 1, 2])


def test_plugins_build_the_map_from_the_device_lengths_alone():
    """The SGLang and RTP-LLM plugins pass only their `query_start_loc`."""
    cu = torch.tensor([0, 3, 20], dtype=torch.int32)
    nums_dict, batch_ptr, token_chunk_offset_ptr = compute_causal_conv1d_metadata(cu)
    assert nums_dict[BLOCK_M]["tot"] == 4
    assert batch_ptr.dtype == token_chunk_offset_ptr.dtype == torch.int32
    assert batch_ptr.device == cu.device
    assert batch_ptr.tolist() == [0, 1, 1, 1]
    assert token_chunk_offset_ptr.tolist() == [0, 0, 1, 2]
