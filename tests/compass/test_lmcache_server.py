# SPDX-License-Identifier: MIT
"""The lmcache MP stand-in against lmcache's own client and adapter."""

import threading

import msgspec
import pytest
import zmq
from lmcache.integration.atom import AtomMPParallelConfig, AtomMPSchedulerAdapter
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.zmq_impl.wire import encode_operation

from atom.compass.kv.lmcache_server import LMCacheRefusal, LMCacheServer

CHUNK = 4
GIB = 1 << 30
FLAGS = ["--chunk-size", str(CHUNK), "--l1-size-gb", "1", "--eviction-policy", "LRU"]
TOKENS = list(range(1, 4 * CHUNK + 3))  # four whole chunks and a tail
WHOLE = 4 * CHUNK


def _key(worker_id=None, start=0, end=WHOLE, rid="r0", readers=1):
    return IPCCacheServerKey(
        model_name="m",
        world_size=2,
        worker_id=worker_id,
        token_ids=tuple(TOKENS),
        start=start,
        end=end,
        request_id=rid,
        num_kv_readers=readers,
    )


def _call(server, op, *payloads, now=0.0):
    """Send one request in the frames lmcache's DEALER sends; return the
    decoded reply and the effect due at the job's end."""
    frames = [b"id", msgspec.msgpack.encode(7), encode_operation(op)]
    frames += [msgspec.msgpack.encode(p) for p in payloads]
    reply, at_end = server.handle(frames, now)
    assert reply[:3] == frames[:3]
    return (msgspec.msgpack.decode(reply[3]) if len(reply) > 3 else None), at_end


def _do(server, op, *payloads, now=0.0):
    reply, at_end = _call(server, op, *payloads, now=now)
    at_end(now)
    return reply


def _store(server, worker_id, start=0, end=WHOLE):
    return _do(server, "store", _key(worker_id, start, end), 1, [[0]], b"")


def _hit(server, rid, now=0.0):
    _do(server, "lookup", _key(rid=rid), 1, now=now)
    return _do(server, "query_prefetch_status", rid, now=now)


@pytest.fixture
def served():
    """A stand-in behind a ROUTER that lmcache's real client reaches."""
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    ctx = zmq.Context()
    router = ctx.socket(zmq.ROUTER)
    url = "inproc://lmcache-stand-in"
    router.bind(url)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            if router.poll(50):
                reply, at_end = server.handle(router.recv_multipart(), 0.0)
                at_end(0.0)
                router.send_multipart(reply)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    yield server, ctx, url
    stop.set()
    t.join()
    router.close(linger=0)
    ctx.term()


def test_every_request_type_round_trips_lmcache_frames(served):
    server, ctx, url = served
    adapter = AtomMPSchedulerAdapter(
        server_url=url,
        context=ctx,
        model_name="m",
        block_size=CHUNK,
        parallel_config=AtomMPParallelConfig(world_size=2, worker_id=0, tp_size=2),
        mq_timeout=5.0,
    )
    client = RequestClientFactory.create(url, context=ctx)
    try:
        assert adapter.lmcache_tokens_per_chunk == CHUNK  # get_chunk_size
        for w in (0, 1):
            assert client.store(_key(w), 1, [[0]], b"").result(5.0) == (b"", True)
        adapter.maybe_submit_lookup_request("r0", TOKENS)  # lookup
        assert adapter.check_lookup_result("r0") == WHOLE  # query_prefetch_status
        assert client.retrieve(_key(0), 1, [[0]], b"", 0).result(5.0) == (b"", True)
        adapter.free_lookup_locks(TOKENS, 0, WHOLE, "r0")
        adapter.end_session("r0")
        assert client.query_prefetch_status("r0").result(5.0) == 0
        assert client.retrieve(_key(0, rid="r1"), 1, [[0]], b"", 0).result(5.0) == (
            b"",
            False,
        )
    finally:
        client.close()
        adapter.shutdown()
    assert sorted(server.request_types) == sorted(
        [
            "get_chunk_size",
            "lookup",
            "query_prefetch_status",
            "free_lookup_locks",
            "end_session",
            "store",
            "retrieve",
        ]
    )


def test_a_hit_is_the_leading_run_present_on_every_rank():
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    assert _hit(server, "a") == 0
    _store(server, 0)
    assert _hit(server, "b") == 0  # rank 1 holds nothing yet
    _store(server, 1, end=2 * CHUNK)
    assert _hit(server, "c") == 2
    _store(server, 1, start=3 * CHUNK)
    assert _hit(server, "d") == 2  # chunk 2 is still missing on rank 1


def test_a_store_is_visible_only_once_its_job_ends():
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    _store(server, 0)
    reply, admit = _call(server, "store", _key(1), 1, [[0]], b"")
    assert reply == [b"", True]
    assert _hit(server, "a") == 0
    admit(1.0)
    assert _hit(server, "b", now=1.0) == 4


def test_a_store_that_does_not_fit_stores_nothing():
    server = LMCacheServer(FLAGS, chunk_bytes=GIB // 6)  # six objects fit
    assert _store(server, 0) == [b"", True]
    assert _store(server, 1) == [b"", True]  # four more do not
    assert len(server.objects) == 4
    assert _hit(server, "a") == 0


def test_eviction_skips_read_locks_until_retrieve_frees_them_or_the_ttl_ends():
    flags = FLAGS + ["--l1-read-ttl-seconds", "10", "--eviction-ratio", "0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 8)  # eight objects fit
    _store(server, 0, end=2 * CHUNK)
    _store(server, 1, end=2 * CHUNK)
    assert server.evict(0.0) == []  # half full, below the 0.8 watermark
    _store(server, 0, start=2 * CHUNK)
    _store(server, 1, start=2 * CHUNK)
    assert _hit(server, "a") == 4  # every object now read-locked
    assert server.evict(0.0) == []
    assert _do(server, "retrieve", _key(0, rid="a"), 1, [[0]], b"", 0) == [b"", True]
    gone = server.evict(0.0)  # rank 0's locks are released
    assert {k[2] for k in gone} == {0} and len(gone) == 4
    assert _hit(server, "b", now=1.0) == 0
    server.capacity = 4 * GIB // 8  # full again with rank 1's four
    assert server.evict(5.0) == []  # still locked by lookup "a"
    assert len(server.evict(10.0)) == 2  # the TTL has ended


def test_free_lookup_locks_releases_only_the_hit_prefix():
    flags = FLAGS + ["--eviction-ratio", "1.0"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 8)  # eight objects fit
    for w in (0, 1):
        _store(server, w)
    _do(server, "lookup", _key(rid="a", end=2 * CHUNK), 1)
    assert _do(server, "query_prefetch_status", "a") == 2
    assert _hit(server, "b") == 4
    _do(server, "free_lookup_locks", _key(rid="a"), 2)  # whole range, hit of 2
    assert server.evict(0.0) == []  # "b" still holds all eight
    _do(server, "free_lookup_locks", _key(rid="b"), 2)
    assert len(server.evict(0.0)) == 8


def test_end_session_makes_its_chunks_most_recent():
    flags = FLAGS + ["--eviction-ratio", "0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, end=2 * CHUNK)
    _store(server, 1, end=2 * CHUNK)
    _do(server, "free_lookup_locks", _key(rid="x"), 2)  # no session: a no-op
    _store(server, 0, end=CHUNK)  # already resident: nothing changes
    _do(server, "end_session", "r0")  # created by the first store
    survivors = set(server.objects) - set(server.evict(0.0))
    assert {(k[2], k[3]) for k in survivors} == {
        (0, server._hashes(_key(), 0, CHUNK)[0]),
        (1, server._hashes(_key(), 0, CHUNK)[0]),
    }


@pytest.mark.parametrize(
    "extra, named",
    [
        (["--l2-adapter", "{}"], "--l2-adapter"),
        (["--eviction-policy", "ARC"], "--eviction-policy"),
        (["--supported-transfer-mode", "engine_driven"], "--supported-transfer-mode"),
    ],
)
def test_an_unmodelled_flag_is_refused_by_name(extra, named):
    with pytest.raises(LMCacheRefusal, match=named):
        LMCacheServer(FLAGS + extra, chunk_bytes=1)


def test_a_missing_budget_is_refused_by_name():
    with pytest.raises(LMCacheRefusal, match="--l1-size-gb"):
        LMCacheServer(["--eviction-policy", "LRU"], chunk_bytes=1)


@pytest.mark.parametrize("op", ["ping", "register_kv_cache", "wait_prefetch_status"])
def test_an_unmodelled_request_type_is_refused_by_name(op):
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    with pytest.raises(LMCacheRefusal, match=op):
        _call(server, op, 1)
