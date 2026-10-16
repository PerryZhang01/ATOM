# SPDX-License-Identifier: MIT
"""The lmcache_mp scheduler half's timers and its admission defer read the LP clock.

Under an installed LP runtime, the lookup defer, the lookup timeout, the
transfer deadlines and a dispatched save's deferred-free stamp follow the
runtime's clock, which the tests move by hand while the wall clock barely
moves. No driver and no lmcache install.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.compass.audit import sync_scan
from atom.kv_transfer.offload import chunked_scheduler
from atom.kv_transfer.offload.mp import lookup as mp_lookup
from atom.kv_transfer.offload.mp import scheduler as mp_scheduler
from atom.kv_transfer.offload.mp import transfer
from atom.utils import clock, envs


class _LP:
    """An installed runtime as `clock.now` sees it: a clock the test sets. It is
    outside a run, so `clock.sleep` sleeps on the wall clock."""

    t = 0.0
    in_run = False

    def read_clock(self) -> float:
        return self.t


@pytest.fixture
def lp():
    runtime = _LP()
    clock.install(runtime)
    yield runtime
    clock.install(None)


def test_the_scanner_reads_mp_and_the_chunked_scheduler():
    tree = sync_scan.repo_root_from_here()
    scanned = set(sync_scan.iter_scanned_files(tree))
    offload = tree / "atom/kv_transfer/offload"
    wanted = {str(p.relative_to(tree)) for p in (offload / "mp").glob("*.py")} | {
        "atom/kv_transfer/offload/chunked_scheduler.py"
    }
    assert wanted <= scanned, sorted(wanted - scanned)


def test_the_lookup_defer_expires_on_the_lp_clock(lp):
    client = SimpleNamespace(is_pending=lambda sid: True, poll=lambda sid: False)
    sched = object.__new__(chunked_scheduler.ChunkedOffloadSchedulerBase)
    sched._lookup_client = client
    sched._lookup_defer_since = {}
    seq = SimpleNamespace(id=7)

    assert sched.lookup_pending(seq) is True
    lp.t = envs.OFFLOAD_LOOKUP_DEFER_S
    assert sched.lookup_pending(seq) is True
    lp.t = envs.OFFLOAD_LOOKUP_DEFER_S + 1e-3
    assert sched.lookup_pending(seq) is False


def test_a_transfer_deadline_fires_on_the_lp_clock(lp):
    lp.t = 1e9 + 9.0
    transfer._enforce_transfer_deadline("op", 1e9, 10.0)
    lp.t = 1e9 + 10.0
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        transfer._enforce_transfer_deadline("op", 1e9, 10.0)


def test_the_scheduler_dates_a_live_transfer_on_the_lp_clock(lp):
    sched = object.__new__(mp_scheduler.LMCacheMPConnectorScheduler)
    sched._transfer_seen_at = {}
    sched._scheduler_deadline_s = 10.0
    sched._save_inflight = {"r": "save-op"}
    sched._save_operation_owner = {}
    sched._active_load_operations = {}

    lp.t = 1e9
    sched._enforce_transfer_deadlines()
    assert sched._transfer_seen_at == {"save-op": 1e9}
    lp.t = 1e9 + 9.0
    sched._enforce_transfer_deadlines()
    lp.t = 1e9 + 10.0
    with pytest.raises(transfer.LMCacheTransferUnprovable):
        sched._enforce_transfer_deadlines()


def test_a_dispatched_save_restamps_its_deferred_free_on_the_lp_clock(lp):
    sched = object.__new__(mp_scheduler.LMCacheMPConnectorScheduler)
    seq = SimpleNamespace(_deferred_save_at=0.0)

    lp.t = 100.0
    sched._refresh_save_reclaim_clock(seq)
    assert seq._deferred_save_at == 100.0


class _Silent:
    """A server that never answers; each look at it moves the LP clock 1 s."""

    lmcache_tokens_per_chunk = 4

    def __init__(self, lp: _LP) -> None:
        self.lp = lp
        self.looks = 0

    def _look(self) -> None:
        self.looks += 1
        assert self.looks < 50, "the lookup timeout is not on the LP clock"
        self.lp.t += 1.0

    def maybe_submit_lookup_request(self, request_id, token_ids) -> None:
        pass

    def check_lookup_result(self, request_id) -> None:
        self._look()


@pytest.mark.parametrize("submitted", [False, True])
def test_a_blocking_lookup_times_out_on_the_lp_clock(lp, monkeypatch, submitted):
    monkeypatch.setattr(mp_lookup.time, "sleep", lambda _s: None)
    adapter = _Silent(lp)
    config = SimpleNamespace(kv_transfer_config={})
    monkeypatch.setattr(mp_lookup, "_mp_session_id", lambda _c, rid: rid)
    client = mp_lookup._MPLookupClient(
        adapter, config=config, timeout=5.0, poll_interval=1e-3
    )
    if submitted:
        never = SimpleNamespace(query=lambda: adapter._look() or False)
        client._async["r"] = ["r", 8, "lookup", never]

    assert client.lookup([1] * 8, lookup_id="r") is None
    assert adapter.looks == 5
