# SPDX-License-Identifier: MIT
"""A stand-in for the lmcache MP server: its wire protocol and its chunk index.

It answers the request types ATOM's lmcache adapters send, in lmcache's own MQ
frames, with the reply payload the real server returns, and keeps the state
that decides later hits: which chunks are resident, their LRU order, their read
locks and the lock TTL, the byte budget and its eviction, and the per-request
sessions. It holds no bytes and binds no socket; the caller owns the loop, the
clock and the socket, and passes the time of each event in.

`handle` takes the frames a ROUTER socket receives (identity, request id,
operation, payloads) at a job's start and returns the reply frames plus a
callable to run at the job's end: a store's chunks become visible, and a
retrieve's read locks are released, when its copy completes. `evict` is one
tick of the server's eviction loop, which wakes every `EVICTION_INTERVAL_S`.

What it does not reproduce. Every object group of a chunk is one entry, so
`--separate-object-groups` and sliding-window groups price and hit as full
attention, and all-null chunks are stored. The byte budget is
`--l1-size-gb` taken whole, so a lazy pool's growth and allocator alignment
are not modelled. Stores and retrieves need no prior REGISTER_KV_CACHE,
because a simulated worker registers no device memory. A lookup's prefetch is
complete at its own job's end, so QUERY_PREFETCH_STATUS never answers
pending. Sessions do not expire.

A flag or a request type outside these is refused by name.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

EVICTION_INTERVAL_S = 1.0


class LMCacheRefusal(Exception):
    """A server flag or a request type the stand-in does not model."""


class _Flags(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # type: ignore[override]
        raise LMCacheRefusal(f"lmcache server flags: {message}")


def _flag_parser() -> argparse.ArgumentParser:
    p = _Flags(prog="lmcache server", add_help=False, allow_abbrev=False)
    a = p.add_argument
    a("--host", default="localhost")
    a("--port", type=int, default=5555)
    a("--chunk-size", type=int, default=256)
    a("--hash-algorithm", default="blake3")
    a("--l1-size-gb", type=float, required=True)
    a("--l1-read-ttl-seconds", type=int, default=300)
    a("--eviction-policy", choices=["LRU"], required=True)
    a("--eviction-trigger-watermark", type=float, default=0.8)
    a("--eviction-ratio", type=float, default=0.2)
    a("--supported-transfer-mode", choices=["lmcache_driven"], default="lmcache_driven")
    # Accepted, and read by nothing here: the index keeps one entry per chunk
    # (null blocks, object groups), the budget is the configured size (lazy
    # pool), and the rest set workers and endpoints no reply depends on.
    a("--null-block-id", type=int, default=0)
    a("--separate-object-groups", action="store_true")
    a("--l1-use-lazy", action=argparse.BooleanOptionalAction, default=True)
    a("--max-workers", type=int, default=1)
    a("--max-gpu-workers", type=int)
    a("--max-cpu-workers", type=int)
    a("--http-host")
    a("--http-port", type=int)
    return p


def parse_server_flags(argv: list[str]) -> argparse.Namespace:
    """Parse `lmcache server` flags, refusing any this module does not model."""
    args, unknown = _flag_parser().parse_known_args(argv)
    if unknown:
        raise LMCacheRefusal(
            f"lmcache stand-in does not model server flag(s) {unknown}"
        )
    return args


@dataclass
class _Obj:
    readers: int = 0
    lock_until: float = float("-inf")

    def locked(self, now: float) -> bool:
        return self.readers > 0 and now < self.lock_until


@dataclass
class _Session:
    tokens: tuple[int, ...] = ()
    key: Any = None
    hit: int = -1


def _noop(now: float) -> None:
    pass


class LMCacheServer:
    """The state and replies of one `lmcache server` process."""

    def __init__(self, argv: list[str], *, chunk_bytes: int) -> None:
        # lmcache brings torch with it, so importing this module stays free of both.
        from lmcache.v1.multiprocess.rpc import get_rpc_spec
        from lmcache.v1.multiprocess.token_hasher import TokenHasher
        from lmcache.v1.multiprocess.transport.zmq_impl import mq, wire

        self._spec, self._mq, self._wire = get_rpc_spec, mq, wire
        self.args = parse_server_flags(argv)
        self.chunk_size = self.args.chunk_size
        self.chunk_bytes = chunk_bytes
        self.capacity = int(self.args.l1_size_gb * (1 << 30))
        self._ttl = self.args.l1_read_ttl_seconds
        self._hasher = TokenHasher(self.chunk_size, self.args.hash_algorithm)
        self.objects: OrderedDict[tuple, _Obj] = OrderedDict()  # LRU first
        self._staged: set[tuple] = set()
        self._prefetch: dict[str, int] = {}
        self._sessions: dict[str, _Session] = {}
        self._handlers: dict[str, Callable[..., tuple[Any, Callable]]] = {
            "get_chunk_size": self._get_chunk_size,
            "lookup": self._lookup,
            "query_prefetch_status": self._query_prefetch_status,
            "free_lookup_locks": self._free_lookup_locks,
            "end_session": self._end_session,
            "store": self._store,
            "retrieve": self._retrieve,
        }

    @property
    def url(self) -> str:
        return f"tcp://{self.args.host}:{self.args.port}"

    @property
    def request_types(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def handle(
        self, frames: list[bytes], now: float
    ) -> tuple[list[bytes], Callable[[float], None]]:
        """Serve one request at its job's start; return its reply frames and
        the effect to apply at its job's end."""
        identity, b_uid, b_op, *payloads = frames
        op = self._wire.decode_operation(b_op)
        handler = self._handlers.get(op)
        if handler is None:
            raise LMCacheRefusal(f"lmcache stand-in does not model request type {op!r}")
        spec = self._spec(op)
        args = self._mq.unwrap_request_payloads(payloads, list(spec.payload_types))
        response, at_end = handler(*args, now=now)
        reply = [identity, b_uid, b_op]
        if response is not None:
            reply.append(self._mq.msgspec_encode(response, cls=spec.response_type))
        return reply, at_end

    def evict(self, now: float) -> list[tuple]:
        """One eviction-loop tick: above the watermark, drop the configured
        share of resident objects, least recently used first, skipping
        read-locked ones."""
        used = (len(self.objects) + len(self._staged)) * self.chunk_bytes
        if (
            not self.capacity
            or used / self.capacity < self.args.eviction_trigger_watermark
        ):
            return []
        target = int(len(self.objects) * self.args.eviction_ratio) or min(
            1, len(self.objects)
        )
        victims = [k for k, o in self.objects.items() if not o.locked(now)][:target]
        for k in victims:
            del self.objects[k]
        return victims

    # Keys and LRU order, as lmcache's ObjectKey and its LRU policy.

    def _hashes(self, key: Any, start: int, end: int) -> list[bytes]:
        return self._hasher.compute_chunk_hashes(
            list(key.token_ids), start=start, end=end
        )

    @staticmethod
    def _okey(key: Any, rank: int, h: bytes) -> tuple:
        return (key.model_name, key.world_size, rank, h, key.cache_salt)

    def _ranks(self, key: Any) -> range:
        if key.worker_id is None:
            return range(key.world_size)
        return range(key.worker_id, key.worker_id + 1)

    def _touch(self, keys: list[tuple]) -> None:
        for k in reversed(keys):
            if k in self.objects:
                self.objects.move_to_end(k)

    def _lock(self, keys: list[tuple], n: int, now: float) -> None:
        for k in keys:
            o = self.objects[k]
            if not o.locked(now):
                o.readers = 0
            o.readers += n
            o.lock_until = now + self._ttl

    def _unlock(self, keys: list[tuple], n: int) -> None:
        for k in keys:
            if k in self.objects:
                o = self.objects[k]
                o.readers = max(0, o.readers - n)

    def _session(self, key: Any) -> _Session:
        s = self._sessions.setdefault(key.request_id, _Session())
        s.tokens = key.token_ids
        return s

    # Request types, as lmcache's LookupModule, ManagementModule and
    # LMCacheDrivenTransferModule answer them.

    def _get_chunk_size(self, *, now: float) -> tuple[Any, Callable]:
        return self.chunk_size, _noop

    def _lookup(self, key: Any, tp_size: int, *, now: float) -> tuple[Any, Callable]:
        readers = key.require_num_kv_readers()
        hashes = self._hashes(key, 0, key.end)
        ranks = self._ranks(key)
        hit = 0
        while hit < len(hashes) and all(
            self._okey(key, r, hashes[hit]) in self.objects for r in ranks
        ):
            hit += 1
        held = [self._okey(key, r, h) for r in ranks for h in hashes[:hit]]
        self._lock(held, readers, now)
        self._touch(held)
        s = self._session(key)
        s.key, s.hit = key, -1
        self._prefetch[key.request_id] = hit
        return None, _noop

    def _query_prefetch_status(
        self, request_id: str, *, now: float
    ) -> tuple[Any, Callable]:
        hit = self._prefetch.pop(request_id, None)
        if hit is None:
            return 0, _noop
        self._sessions.setdefault(request_id, _Session()).hit = hit
        return hit, _noop

    def _free_lookup_locks(
        self, key: Any, tp_size: int, *, now: float
    ) -> tuple[Any, Callable]:
        if key.start >= key.end:
            return None, _noop
        hit = self._sessions.setdefault(key.request_id, _Session()).hit
        first = key.start // self.chunk_size
        hashes = self._hashes(key, key.start, key.end)
        if hit >= 0:
            hashes = hashes[: max(0, hit - first)]
        held = [self._okey(key, r, h) for h in hashes for r in self._ranks(key)]
        self._unlock(held, key.require_num_kv_readers())
        return None, _noop

    def _end_session(self, request_id: str, *, now: float) -> tuple[Any, Callable]:
        s = self._sessions.pop(request_id, None)
        if s is not None and s.key is not None:
            hashes = self._hasher.compute_chunk_hashes(list(s.tokens))
            ranks = range(s.key.world_size)
            self._touch([self._okey(s.key, r, h) for h in hashes for r in ranks])
        return None, _noop

    def _worker_keys(self, key: Any) -> list[tuple]:
        s = self._session(key)
        if s.key is None:
            s.key = key
        return [
            self._okey(key, key.worker_id, h)
            for h in self._hashes(key, key.start, key.end)
        ]

    def _store(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event: bytes,
        *,
        now: float,
    ) -> tuple[Any, Callable]:
        wanted = [
            k
            for k in self._worker_keys(key)
            if k not in self.objects and k not in self._staged
        ]
        used = (len(self.objects) + len(self._staged) + len(wanted)) * self.chunk_bytes
        if used > self.capacity:
            wanted = []  # one allocation for the whole store: all or nothing
        self._staged.update(wanted)

        def admit(now: float) -> None:
            self._staged.difference_update(wanted)
            fresh = [k for k in wanted if k not in self.objects]
            for k in fresh:
                self.objects[k] = _Obj()
            for k in reversed(fresh):
                self.objects.move_to_end(k)

        return (b"", True), admit

    def _retrieve(
        self,
        key: Any,
        instance_id: int,
        block_ids: list[list[int]],
        event: bytes,
        skip_first_n_tokens: int,
        *,
        now: float,
    ) -> tuple[Any, Callable]:
        keys = self._worker_keys(key)
        readable = [
            k for k in keys if k in self.objects and self.objects[k].locked(now)
        ]
        if len(readable) < len(keys):
            self._unlock(readable, 1)
            return (b"", False), _noop
        return (b"", True), lambda now: self._unlock(readable, 1)
