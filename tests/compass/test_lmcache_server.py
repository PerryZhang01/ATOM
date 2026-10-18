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


def _key(worker_id=None, start=0, end=WHOLE, rid="r0", readers=1, **kw):
    return IPCCacheServerKey(
        model_name=kw.get("model", "m"),
        world_size=kw.get("world", 2),
        worker_id=worker_id,
        token_ids=tuple(TOKENS),
        start=start,
        end=end,
        request_id=rid,
        num_kv_readers=readers,
        cache_salt=kw.get("salt", ""),
    )


def _call(server, op, *payloads, now=0.0):
    """Send one request in the frames lmcache's DEALER sends; return the
    decoded reply and the effect due at the job's end."""
    frames = [b"id", msgspec.msgpack.encode(7), encode_operation(op)]
    frames += [msgspec.msgpack.encode(p) for p in payloads]
    reply, at_end, _ = server.handle(frames, now)
    assert reply[:3] == frames[:3]
    return (msgspec.msgpack.decode(reply[3]) if len(reply) > 3 else None), at_end


def _do(server, op, *payloads, now=0.0):
    reply, at_end = _call(server, op, *payloads, now=now)
    at_end(now)
    return reply


def _store(server, worker_id, start=0, end=WHOLE, **kw):
    return _do(server, "store", _key(worker_id, start, end, **kw), 1, [[0]], b"")


def _hit(server, rid, now=0.0, **kw):
    _do(server, "lookup", _key(rid=rid, **kw), 1, now=now)
    return _do(server, "query_prefetch_status", rid, now=now)


def _chunks(server, keys):
    """(rank, chunk index) of each object key."""
    hashes = server._hashes(_key(), 0, WHOLE)
    return {(k[2], hashes.index(k[3])) for k in keys}


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
                reply, at_end, _ = server.handle(router.recv_multipart(), 0.0)
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
    _, ctx, url = served
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


@pytest.mark.parametrize("kw", [{"model": "n"}, {"salt": "s"}, {"world": 1}])
def test_a_chunk_is_not_shared_across_models_world_sizes_or_salts(kw):
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    for w in (0, 1):
        _store(server, w)
    assert _hit(server, "a") == 4
    assert _hit(server, "b", **kw) == 0


def test_the_later_chunks_of_a_store_are_evicted_first():
    flags = FLAGS + ["--eviction-ratio", "0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert _chunks(server, server.evict(0.0)) == {(0, 2), (0, 3)}


def test_a_hit_makes_its_chunks_most_recent():
    flags = FLAGS + ["--eviction-ratio", "0.5", "--l1-read-ttl-seconds", "1"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, end=2 * CHUNK, world=1)
    _store(server, 0, start=2 * CHUNK, world=1)
    assert _hit(server, "a", end=2 * CHUNK, world=1) == 2
    assert _chunks(server, server.evict(1.0)) == {(0, 2), (0, 3)}  # locks ended


def test_a_store_in_flight_takes_its_share_of_the_budget():
    server = LMCacheServer(FLAGS, chunk_bytes=GIB // 6)  # six objects fit
    _, admit = _call(server, "store", _key(0), 1, [[0]], b"")
    assert _store(server, 1) == [b"", True]  # four more do not fit beside it
    admit(0.0)
    assert len(server.objects) == 4


def test_staged_chunks_count_toward_the_watermark_and_the_victim_count():
    flags = FLAGS + ["--eviction-ratio", "0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 8)  # eight objects fit
    _store(server, 0)
    assert server.evict(0.0) == []  # half full
    _call(server, "store", _key(1), 1, [[0]], b"")  # four more staged
    assert len(server.evict(0.0)) == 4  # half of the eight tracked


@pytest.mark.parametrize("ratio, evicted", [("0", 0), ("0.2", 1)])
def test_a_ratio_above_zero_evicts_at_least_one_and_zero_evicts_none(ratio, evicted):
    flags = FLAGS + ["--eviction-ratio", ratio]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert len(server.evict(0.0)) == evicted


def test_a_retrieve_holds_its_locks_until_its_copy_ends():
    flags = FLAGS + ["--eviction-ratio", "1.0"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert _hit(server, "a", world=1) == 4
    reply, done = _call(server, "retrieve", _key(0, rid="a", world=1), 1, [[0]], b"", 0)
    assert reply == [b"", True]
    assert server.evict(0.0) == []
    done(0.0)
    assert len(server.evict(0.0)) == 4


def test_a_failed_retrieve_releases_the_locks_it_found():
    flags = FLAGS + ["--eviction-ratio", "1.0"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 2)  # two objects fit
    _store(server, 0, end=2 * CHUNK, world=1)
    assert _hit(server, "a", world=1) == 2
    key = _key(0, rid="a", world=1)
    assert _do(server, "retrieve", key, 1, [[0]], b"", 0) == [b"", False]
    assert len(server.evict(0.0)) == 2


def test_a_lock_past_its_ttl_restarts_its_reader_count():
    flags = FLAGS + ["--eviction-ratio", "1.0", "--l1-read-ttl-seconds", "10"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert _hit(server, "a", world=1) == 4  # never freed; its locks end at 10
    assert _hit(server, "b", now=20.0, world=1) == 4
    _do(server, "free_lookup_locks", _key(rid="b", world=1), 1, now=20.0)
    assert len(server.evict(20.0)) == 4


def test_a_new_lookup_forgets_the_previous_hit():
    flags = FLAGS + ["--eviction-ratio", "1.0"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert _hit(server, "a", end=2 * CHUNK, world=1) == 2
    _do(server, "lookup", _key(rid="a", world=1), 1)  # hit of 4, not yet read
    _do(server, "free_lookup_locks", _key(rid="a", world=1), 1)  # whole range
    assert _chunks(server, server.evict(0.0)) == {(0, 2), (0, 3)}


@pytest.mark.parametrize(
    "sweeps, end, victims",
    [
        ((0.0, 600.0), 600.0, {(0, 2), (0, 3)}),  # exactly at its TTL
        ((0.0, 600.0), 659.0, {(0, 2), (0, 3)}),  # past its TTL, before a sweep
        ((0.0, 600.0, 660.0), 660.0, {(0, 0), (0, 1)}),
    ],
)
def test_a_session_lives_until_the_first_sweep_past_its_ttl(sweeps, end, victims):
    flags = FLAGS + ["--eviction-ratio", "0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    key = _key(0, end=2 * CHUNK, rid="a", world=1)
    _do(server, "store", key, 1, [[0]], b"", now=0.0)
    _store(server, 0, start=2 * CHUNK, rid="z", world=1)  # LRU: 1, 0, 3, 2
    for t in sweeps:
        server.expire_sessions(t)
    _do(server, "end_session", "a", now=end)  # touches only a live session
    assert _chunks(server, server.evict(end)) == victims


def test_free_lookup_locks_of_a_dropped_session_releases_the_whole_range():
    flags = FLAGS + ["--eviction-ratio", "1.0", "--l1-read-ttl-seconds", "1000"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 8)  # eight objects fit
    for w in (0, 1):
        _store(server, w)
    _do(server, "lookup", _key(rid="a", end=2 * CHUNK), 1)
    assert _do(server, "query_prefetch_status", "a") == 2
    assert _hit(server, "b", now=1.0) == 4
    server.expire_sessions(600.5)  # drops "a", keeps "b"
    _do(server, "free_lookup_locks", _key(rid="a"), 2, now=600.5)  # hit unknown
    gone = server.evict(600.5)  # "b"'s locks on chunks 2 and 3 were released
    assert _chunks(server, gone) == {(r, c) for r in (0, 1) for c in (2, 3)}


def test_a_sweep_says_when_it_drops_which_sessions(capsys):
    server = LMCacheServer(FLAGS, chunk_bytes=1)
    _hit(server, "a", now=0.0)
    _hit(server, "b", now=31.0)
    server.expire_sessions(600.0)
    assert capsys.readouterr().err == ""
    server.expire_sessions(630.5)
    assert capsys.readouterr().err == (
        "lmcache stand-in at 630.500000 s: Cleaned up 1 expired sessions ['a']\n"
    )


def test_a_negative_ratio_evicts_none():
    flags = FLAGS + ["--eviction-ratio", "-0.5"]
    server = LMCacheServer(flags, chunk_bytes=GIB // 4)  # four objects fit
    _store(server, 0, world=1)
    assert server.evict(0.0) == []


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


def _native_store(server, worker_id, end, state_at, **kw):
    """A store of chunks [0, `end`) as ATOM's native-state layout sends it:
    attention blocks on every chunk, recurrent state on chunks `state_at`."""
    n = end // CHUNK
    ids = [[7] * n, [5 if i in state_at else -1 for i in range(n)]]
    return _do(server, "store", _key(worker_id, 0, end, **kw), 1, ids, b"")


@pytest.mark.parametrize(
    "separate, hits", [(True, [4, 0, 2]), (False, [4, 3, 3])], ids=["separate", "one"]
)
def test_a_hit_ends_at_a_chunk_that_keeps_its_recurrent_state(separate, hits):
    flags = FLAGS + ["--null-block-id", "-1"]
    server = LMCacheServer(
        flags + ["--separate-object-groups"] * separate, chunk_bytes=1
    )
    for w in (0, 1):
        _native_store(server, w, WHOLE, {3})
    got = [_hit(server, "a"), _hit(server, "b", end=3 * CHUNK)]
    for w in (0, 1):
        _native_store(server, w, 2 * CHUNK, {1})
    assert got + [_hit(server, "c", end=3 * CHUNK)] == hits


def _copied(server, op, *payloads):
    """Serve one request and apply its effect; the chunks its copy is priced at."""
    frames = [b"id", msgspec.msgpack.encode(7), encode_operation(op)]
    frames += [msgspec.msgpack.encode(p) for p in payloads]
    _, at_end, chunks = server.handle(frames, 0.0)
    at_end(0.0)
    return chunks


@pytest.mark.parametrize(
    "separate, state, copied, hit",
    [(True, GIB // 8, 6, 4), (False, GIB // 8, 12, 4), (True, GIB, 4, 0)],
    ids=["its-own-object", "in-every-chunk", "does-not-fit"],
)
def test_a_recurrent_state_takes_its_bytes_in_the_budget_and_the_copy(
    separate, state, copied, hit
):
    flags = FLAGS + ["--null-block-id", "-1"] + ["--separate-object-groups"] * separate
    server = LMCacheServer(flags, chunk_bytes=GIB // 16, state_bytes=state)
    ids = [[7] * 4, [-1, -1, -1, 5]]
    assert _copied(server, "store", _key(0, world=1), 1, ids, b"") == copied
    assert _hit(server, "a", world=1) == hit
    if hit:
        key = _key(0, rid="a", world=1)
        assert _copied(server, "retrieve", key, 1, ids, b"", 0) == copied


def test_a_recurrent_state_is_evicted_apart_from_its_chunk_and_ends_a_hit_earlier():
    flags = FLAGS + ["--null-block-id", "-1", "--separate-object-groups"]
    server = LMCacheServer(
        flags + ["--eviction-ratio", "0.2"],
        chunk_bytes=GIB // 16,
        state_bytes=GIB // 10,
    )
    for end, state_at in ((2 * CHUNK, {1}), (WHOLE, {3})):
        for w in (0, 1):
            _native_store(server, w, end, state_at)
    assert _hit(server, "a") == 4
    assert _hit(server, "b", end=2 * CHUNK) == 2  # chunk 1's state is now newer
    for rid in ("a", "b"):
        _do(server, "free_lookup_locks", _key(rid=rid), 2)
    last = server._hashes(_key(), 0, WHOLE)[3]
    states = {(*server._okey(_key(), r, last), 1) for r in (0, 1)}
    assert set(server.evict(0.0)) == states  # 90% full: two of twelve objects
    assert _hit(server, "c") == 2  # every attention chunk is still here


def test_end_session_makes_its_recurrent_states_most_recent_too():
    flags = FLAGS + ["--null-block-id", "-1", "--separate-object-groups"]
    server = LMCacheServer(
        flags + ["--eviction-ratio", "0.5"], chunk_bytes=GIB // 8, state_bytes=GIB // 8
    )
    for rid, salt in (("r0", ""), ("r1", "x")):
        for w in (0, 1):
            _native_store(server, w, CHUNK, {0}, rid=rid, salt=salt)
    _do(server, "end_session", "r0")
    assert {k[4] for k in server.evict(0.0)} == {"x"}  # full: four of eight


def test_an_eviction_says_when_and_how_many_objects(capsys):
    server = LMCacheServer(FLAGS, chunk_bytes=GIB // 8)  # eight objects fit
    _store(server, 0, end=2 * CHUNK)
    _store(server, 1)
    server.evict(1.0)
    assert capsys.readouterr().err == ""  # six objects, below the 0.8 watermark
    _store(server, 0)
    server.evict(2.0)
    assert capsys.readouterr().err == (
        "lmcache stand-in at 2.000000 s: L1 memory usage 1.00 above watermark "
        "0.80; triggering eviction of 1 objects\n"
    )
