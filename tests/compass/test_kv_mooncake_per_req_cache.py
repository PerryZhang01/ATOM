# SPDX-License-Identifier: MIT
"""Mooncake's consumer takes the full transfer for a request with per-request
cache state, even when the producer's `hash_block_size` matches its own.

A block-only delta does not carry per-request state such as the SWA ring slot,
so `has_per_req_cache` alone turns the incremental offset off.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


@pytest.mark.parametrize("has_per_req_cache, expected", [(False, 4), (True, 0)])
def test_per_request_cache_forces_the_full_transfer(
    has_per_req_cache, expected, monkeypatch
):
    monkeypatch.setitem(sys.modules, "aiter.dist.parallel_state", MagicMock())
    mc = importlib.import_module(
        "atom.kv_transfer.disaggregation.mooncake.mooncake_connector"
    )
    own = 16
    scheduler = object.__new__(mc.MooncakeConnectorScheduler)
    scheduler.is_producer = False
    scheduler.block_size, scheduler.dcp_size, scheduler.hash_block_size = own, 1, own
    scheduler._reqs_need_recv = {}
    scheduler.transfer_id_to_request_id = {}
    scheduler.request_id_to_transfer_id = {}
    seq = SimpleNamespace(
        id=0,
        kv_transfer_params={"do_remote_prefill": True, "hash_block_size": own},
        block_table=list(range(8)),
        has_per_req_cache=has_per_req_cache,
        num_cached_tokens=4 * own,
    )
    scheduler.update_state_after_alloc(seq)
    assert seq.kv_transfer_params["num_computed_blocks"] == expected
