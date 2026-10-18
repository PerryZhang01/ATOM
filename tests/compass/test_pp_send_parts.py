# SPDX-License-Identifier: MIT
"""A simulated PP send's parts are what ATOM's `async_send_intermediate_tensors`
sends: the stage output's `_PP_PROXY_KEYS`, split by `_split_tensor_dict`."""

import pickle
from types import SimpleNamespace

import pytest
import torch
from test_pp_launch_send import _runner, pp_comm

N, TOPK = 8, 4


def _stage(monkeypatch, extra=None):
    """Stage 0 of 2, TP rank 1 (no host recurrence charge), carrying the
    sparse indices; `extra` is added to its output under the key ``extra``."""
    stage = _runner(monkeypatch, 0, lambda: 0.0)
    stage.rank, stage._pp_send_needs_sparse, stage._pp_index_topk = 1, True, TOPK
    if extra is not None:
        output = stage._pp_output
        stage._pp_output = lambda n: {**output(n), "extra": extra}
    return stage


def _parts(stage):
    return stage._pp_send(SimpleNamespace(total_tokens_num=N), 0.25)[2]


def test_a_send_carries_every_key_atom_sends_split_as_atom_splits_it(monkeypatch):
    """With every optional key carried, the send splits exactly ATOM's keys,
    in ATOM's order: the size, the metadata's pickle, then each tensor."""
    comm, split = pp_comm(monkeypatch), []
    stage = _stage(monkeypatch)
    real = comm._split_tensor_dict
    monkeypatch.setattr(
        comm, "_split_tensor_dict", lambda d: split.append(d) or real(d)
    )
    parts = _parts(stage)
    [sent] = split
    assert tuple(sent) == tuple(comm._PP_PROXY_KEYS)
    meta = real(sent)[0]
    assert parts == [
        ("gloo", 8),
        ("gloo", len(pickle.dumps(meta))),
        ("rccl", N * 8 * 2),
        ("rccl", N * 8 * 2),
        ("rccl", N * TOPK * 4),
    ]


def test_a_key_in_the_output_and_in_atoms_keys_is_priced(monkeypatch):
    comm = pp_comm(monkeypatch)
    base = _parts(_stage(monkeypatch))
    monkeypatch.setattr(comm, "_PP_PROXY_KEYS", (*comm._PP_PROXY_KEYS, "extra"))
    parts = _parts(_stage(monkeypatch, torch.empty(3, device="meta")))
    assert parts[2:] == [*base[2:], ("rccl", 12)]
    assert parts[1][1] > base[1][1]


def test_a_key_only_in_the_output_is_not_priced(monkeypatch):
    base = _parts(_stage(monkeypatch))
    assert _parts(_stage(monkeypatch, torch.empty(3, device="meta"))) == base


def test_a_key_atom_sends_that_compass_does_not_size_is_refused_by_name(monkeypatch):
    comm = pp_comm(monkeypatch)
    monkeypatch.setattr(comm, "_PP_PROXY_KEYS", (*comm._PP_PROXY_KEYS, "extra"))
    with pytest.raises(RuntimeError, match=r"cannot size.*\['extra'\]"):
        _parts(_stage(monkeypatch))
