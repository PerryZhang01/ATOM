# SPDX-License-Identifier: MIT
"""ATOM's `lmcache_mp` connector on a simulated run: its lmcache client and its
workers' transfers on the step loop's LP clock.

`substitute` makes the connector factory build, for `lmcache_mp`, the scheduler
half `LMCacheMPShell` and the worker half of `compass`, which registers, sends
and reports nothing; a `multi` that lists `lmcache_mp` builds them too, and
its scheduler half is `MultiShell`.

`LMCacheMPShell` is ATOM's own scheduler half (`mp.connector`, PAGE-only or
native-state as its `bind_block_manager` picks) with the workers' transfers
moved onto the step loop. It keeps one ATOM worker connector per TP rank, with
no device: `build_connector_meta` hands each rank the metadata, and the rank's
adapter sends each STORE or RETRIEVE it submits to LP ``lmcache`` on its
writer's ``transfer`` channel, stamped now. A native-state rank takes its
checkpoint units as named and restores a load at no cost.
`process_completions` takes, inline, every reply released so far and reports
what the ranks' `get_finished`, aggregated over the TP group, gave at the
previous call, since the engine's aggregation answers one poll behind.

`clocked_adapter` builds ATOM's scheduler adapter with lmcache's client on the
``lookup`` channel of this engine's DP rank. The step loop stamps each request
as it puts it on the client's queue; lmcache's polling thread sends it, takes
each reply once a grant releases it and sets its future. A wait on that future
on the step loop is an idle point until the reply or the wait's bound. Frames
outside the run go unstamped.

A server URL lmcache reaches over gRPC is refused by its scheme. Importing this
module loads no third-party package; using it loads lmcache, zmq and torch.
"""

from __future__ import annotations

import dataclasses
import itertools
import logging
import math
import pickle
import threading
from types import SimpleNamespace

from atom.compass.kv.connector import SimulatedKVConnectorScheduler
from atom.kv_transfer.disaggregation.multi.multi_connector import (
    MultiConnectorScheduler,
)

logger = logging.getLogger(__name__)

#: The URL schemes lmcache's request client carries over gRPC.
GRPC_SCHEMES = ("grpc", "grpc+unix")


def substitute() -> None:
    """Build this module's halves for every `lmcache_mp` connector from now on."""
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    registry = KVConnectorFactory._registry
    registry["lmcache_mp"] = registry["lmcache_mp"] | {
        "worker_module": "atom.compass.kv.connector",
        "worker_class": "SimulatedKVConnector",
        "scheduler_module": __name__,
        "scheduler_class": LMCacheMPShell.__name__,
    }
    registry["multi"] = registry["multi"] | {
        "scheduler_module": __name__,
        "scheduler_class": MultiShell.__name__,
    }


class MultiShell(MultiConnectorScheduler):
    """ATOM's `multi` scheduler half, the simulated P/D connector's completions
    added before its one offload sub-connector reads them.

    A real P/D connector's completions reach the scheduler from its workers'
    `get_finished`; the simulated one adds them in its scheduler half's
    `process_completions`, which ATOM's composite calls on one sub-connector
    only. The two write disjoint fields of the output."""

    def process_completions(self, output):
        subs = self._connectors
        pd = [c for c in subs if isinstance(c, SimulatedKVConnectorScheduler)]
        for c in pd:
            output = c.process_completions(output)
        self._connectors = [c for c in subs if c not in pd]
        try:
            return super().process_completions(output)
        finally:
            self._connectors = subs


#: A native-state rank's checkpoint layout: the units it is given, as named.
_UNITS_AS_NAMED = SimpleNamespace(validate_unit_ids=tuple)


def _restored(pending) -> bool:
    """A native-state rank's restore after a load: done, and at no cost."""
    pending.restore_succeeded = True
    pending.restore_event = SimpleNamespace(query=lambda: True)
    return True


class LMCacheMPShell:
    """ATOM's `lmcache_mp` scheduler half, its workers' transfers on the step loop."""

    def __init__(self, config) -> None:
        from atom.kv_transfer.offload.mp import connector, deployment
        from atom.utils import clock

        url = deployment._server_urls(config)[0]
        if url.partition("://")[0].lower() in GRPC_SCHEMES:
            raise ValueError(
                f"the lmcache server URL {url!r} is one lmcache reaches over gRPC, "
                "a stream no wrapped socket carries; a simulated run reaches LP "
                "lmcache over a ZMQ URL (tcp, ipc)"
            )
        self._inner = connector.LMCacheMPConnectorScheduler(config)
        self._config, self._url, self._rt = config, url, clock.installed()
        self._ranks: list = []
        self._socks: dict = {}  # reply channel -> its DEALER, one per writer
        self._futures: dict = {}  # (reply channel, request id) -> its future
        self._read: dict = {}  # (reply channel, seq) -> (arrival, frames), untaken
        self._ids = itertools.count()
        self._held = None  # what the ranks reported at the previous call

    def __getattr__(self, name: str):
        inner = self.__dict__.get("_inner")
        if inner is None:
            raise AttributeError(name)
        return getattr(inner, name)

    def bind_block_manager(self, block_manager) -> None:
        from atom.kv_transfer.disaggregation.aggregator import KVOutputAggregator
        from atom.kv_transfer.offload.mp.deployment import _tp_replication_factor
        from atom.kv_transfer.offload.mp.native_state_scheduler import (
            NativeStateLMCacheMPConnectorScheduler,
        )
        from atom.kv_transfer.offload.mp.native_state_worker import (
            NativeStateLMCacheMPConnector,
        )
        from atom.kv_transfer.offload.mp.worker import LMCacheMPConnector

        self._inner.bind_block_manager(block_manager)
        if self._ranks:
            return
        impl, config = self._inner._impl, self._config
        native = isinstance(impl, NativeStateLMCacheMPConnectorScheduler)
        tp, group = config.tensor_parallel_size, _tp_replication_factor(config)
        dp = config.parallel_config.data_parallel_rank
        for k in range(tp):
            rank = (NativeStateLMCacheMPConnector if native else LMCacheMPConnector)(
                config
            )
            rank.chunk_size, rank._is_kv_writer = impl.chunk_size, k % group == 0
            rank._adapter = _Rank(self, f"transfer#dp{dp}.tp{k // group}", k // group)
            if native:
                rank._native_layout, rank._begin_restore = _UNITS_AS_NAMED, _restored
            self._ranks.append(rank)
        self._tp = KVOutputAggregator(world_size=tp)

    def build_connector_meta(self):
        """ATOM's metadata, after each rank has submitted its share of it."""
        meta = self._inner.build_connector_meta()
        for rank in self._ranks:
            for req in getattr(meta, "requests", ()):
                if req.load_spec is not None and rank._do_load:
                    rank._submit_load(req, None)
                if req.save_spec is not None and rank._do_save:
                    rank._submit_save(req, None)
        return meta

    def process_completions(self, output):
        """Take the replies released so far; report what the ranks gave at the
        previous call, then let ATOM's scheduler half read it."""
        self._take()
        held = self._tp.aggregate([rank.get_finished() for rank in self._ranks])
        held, self._held = self._held, held
        if held is not None and not held.is_empty():
            logger.info(
                "compass: lmcache_mp reports at %r: saved %s, loaded %s, failed %s",
                self._rt.now,
                sorted(map(str, held.finished_saving)),
                sorted(map(str, held.finished_loading)),
                sorted(map(str, held.failed_loading)),
            )
        for field in dataclasses.fields(held) if held is not None else ():
            value = getattr(held, field.name)
            if isinstance(value, set):
                getattr(output, field.name).update(value)
        return self._inner.process_completions(output)

    def has_pending_work(self) -> bool:
        held = self._held is not None and not self._held.is_empty()
        return bool(self._futures) or held or self._inner.has_pending_work()

    def _send(self, kind: str, op: str, *payloads):
        """Send a request on ``<engine>->lmcache:<kind>``, stamped now; its future."""
        import zmq
        from lmcache.v1.multiprocess.futures import MessagingFuture
        from lmcache.v1.multiprocess.rpc import get_rpc_spec
        from lmcache.v1.multiprocess.transport.zmq_impl import mq, wire

        rt, me = self._rt, self._rt.me.name
        ch, back = f"{me}->lmcache:{kind}", f"lmcache->{me}:{kind}"
        if back not in self._socks:
            self._socks[back] = zmq.Context.instance().socket(zmq.DEALER)
            self._socks[back].connect(self._url)
        uid, sock = next(self._ids), self._socks[back]
        types = get_rpc_spec(op).payload_types
        frames = [mq.encode_request_uid(uid), wire.encode_operation(op)]
        frames += [
            mq.msgspec_encode(p, t) for p, t in zip(payloads, types, strict=True)
        ]
        sock.send_multipart([pickle.dumps((ch, *rt.stamp_send(ch))), *frames])
        future = MessagingFuture()
        self._futures[back, uid] = (future, op, payloads[0].request_id)
        return future

    def _take(self) -> None:
        """Take every transfer reply released so far, in arrival order, first
        waiting for any released one still on the wire."""
        import msgspec
        import zmq

        rt = self._rt
        for back, sock in self._socks.items():
            while True:
                while True:
                    try:
                        hdr, *frames = sock.recv_multipart(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                    ch, arrival, seq = pickle.loads(hdr)
                    rt.check_arrival(ch, arrival, seq)
                    self._read[ch, seq] = (arrival, frames)
                with rt.lock:
                    if rt.released[back] <= rt.arrived[back]:
                        break
                sock.poll()
        with rt.lock:
            due = sorted(
                (a, ch, seq)
                for (ch, seq), (a, _) in self._read.items()
                if seq in rt.released[ch]
            )
            for _, ch, seq in due:
                rt.count_done_locked(ch, seq)
        for _, ch, seq in due:
            arrival, (uid, _, response) = self._read.pop((ch, seq))
            future, op, request = self._futures.pop((ch, msgspec.msgpack.decode(uid)))
            # A STORE or RETRIEVE answers (device event, succeeded).
            ok = msgspec.msgpack.decode(response)[1]
            future.set_result(ok)
            logger.info(
                "compass: lmcache_mp %s of %s %s: arrived at %r, taken at %r",
                op,
                request,
                "succeeded" if ok else "failed",
                arrival,
                rt.now,
            )


class _Rank:
    """One TP rank's lmcache worker adapter: each STORE and RETRIEVE it is given
    goes to LP ``lmcache`` on its writer's channel, as its worker would send it."""

    def __init__(self, shell: LMCacheMPShell, kind: str, worker_id: int) -> None:
        self.shell, self.kind, self.worker_id = shell, kind, worker_id

    def _key(self, request_id: str, spec):
        key = self.shell._inner._impl._mp_adapter._create_key(
            token_ids=list(spec.token_ids),
            start=spec.start,
            end=spec.end,
            request_id=request_id,
            worker_id=self.worker_id,
        )
        return dataclasses.replace(key, num_kv_readers=1)

    def submit_store_request(self, request_id, spec, event):
        return self.shell._send(
            self.kind,
            "store",
            self._key(request_id, spec),
            self.worker_id,
            spec.block_ids,
            b"",
        )

    def submit_retrieve_request(self, request_id, spec, event):
        return self.shell._send(
            self.kind,
            "retrieve",
            self._key(request_id, spec),
            self.worker_id,
            spec.block_ids,
            b"",
            0,
        )


def clocked_adapter(cls, config, **kwargs):
    """ATOM's scheduler adapter `cls`, built with `kwargs`, its lmcache client on
    the ``lookup`` channel of this engine's DP rank."""
    import zmq
    from lmcache.v1.multiprocess.futures import MessagingFuture

    from atom.utils import clock

    rt = clock.installed()
    dp = config.parallel_config.data_parallel_rank
    ch = f"{rt.me.name}->lmcache:lookup#dp{dp}"

    class Socket(zmq.Socket):
        """lmcache's client socket: a request carries the stamp its put took,
        and a reply is handed to lmcache once released."""

        relay = staged = None
        held = ()  # (arrival, channel, seq, frames) read, not handed over

        def send_multipart(self, frames, *args, **kw):
            stamp = self.relay.take_stamp() if rt.in_run else (None, None)
            super().send_multipart([pickle.dumps((ch, *stamp)), *frames], *args, **kw)

        def recv_multipart(self, *args, **kw):
            if self.staged is not None:
                frames, self.staged = self.staged, None
                return frames
            hdr, *frames = super().recv_multipart(*args, **kw)
            if pickle.loads(hdr)[1] is not None:
                raise RuntimeError(f"{ch}: a stamped reply read past the clock")
            return frames

        def pull(self) -> None:
            while True:
                try:
                    hdr, *frames = super().recv_multipart(zmq.NOBLOCK)
                except zmq.Again:
                    return
                back, arrival, seq = pickle.loads(hdr)
                if arrival is not None:
                    rt.check_arrival(back, arrival, seq)
                self.held = [*self.held, (arrival, back, seq, frames)]

    Socket.ch = ch  # the channel `clock.RelayQueue` stamps on

    class Context(zmq.Context):
        _socket_class = Socket

    class Future(MessagingFuture):
        def wait(self, timeout=None):
            if not rt.in_run or threading.current_thread() is not rt.owner:
                return super().wait(timeout)
            end = rt.now + (math.inf if timeout is None else timeout)
            while not self.query() and rt.now < end:
                rt.next_event(math.inf, end)  # the bound keeps no run alive
            return self.query()

    adapter = cls(context=Context(), **kwargs)
    client = adapter._client._message_queue_client
    sock, receive, submit = client.socket, client.process_inbound, client.submit_request
    client.input_queue = clock.RelayQueue(rt, sock)
    wake = clock._Wakeup()

    def inbound() -> None:
        """Hand lmcache each reply read and released, in arrival order."""
        wake.clear()
        sock.pull()
        # Membership only: the step loop adds to `released` and nothing removes.
        ready = [f for f in sock.held if f[0] is None or f[2] in rt.released[f[1]]]
        sock.held = [f for f in sock.held if f not in ready]
        for arrival, back, seq, frames in ready:
            sock.staged = frames
            if arrival is None:
                receive()
                continue
            rt.handed_over(back, seq)
            try:
                receive()
            finally:
                rt.back_at_wait_point()

    def submit_request(operation, payloads):
        future = submit(operation, payloads)
        future.__class__ = Future
        return future

    client.process_inbound, client.submit_request = inbound, submit_request
    # A release wakes lmcache's polling thread as a reply on the socket would.
    waker = SimpleNamespace(
        socket=wake.fd, process_inbound=inbound, process_outbound_task=lambda: None
    )
    rt.wakes[wake] = wake  # on the step loop, the one thread that reads `wakes`
    client._polling_loop.register(waker)
    return adapter
