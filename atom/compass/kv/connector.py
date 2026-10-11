# SPDX-License-Identifier: MIT
"""A KV connector that moves no bytes and charges for the ones it would have.

Registered in the engine's connector factory under the name `compass`. It
stands in for Mooncake, the push backend: decode asks prefill to write a
prompt's KV into decode's blocks, and both learn when the write is done. Only
the scheduler half takes part, on the engine's step loop and the LP clock of
the `LPRuntime` installed in the engine's process:

- Decode, in `update_state_after_alloc`, sends the write request on the
  ``kv_write_req`` channel (arrival ``a``) and is ready at ``a + T + notify``,
  where the notify latency is that channel's lookahead.
- Prefill records a finished request's data-ready time ``r`` in
  `request_finished`, takes the write requests inline in `process_completions`
  and is done at ``max(a, r) + T``. A write request for a request that has not
  finished (``r > a``) is refused: the router forwards to decode only after
  prefill has returned.

Both report from `process_completions`, which the engine calls at every step
and every idle drain tick, once the clock reaches their time. Write-done is
not a message, because both ends already know its time. ``T`` is the
`TransferModel` bound at ``kv_transfer_config[TRANSFER_KEY]``; the write
request travels over the zmq address at ``kv_transfer_config[WRITE_REQ_KEY]``,
which prefill binds and decode connects to.

The worker half reports nothing. The engine still builds and polls it.
"""

from __future__ import annotations

import logging
import pickle

from atom.compass.kv.handoff import transfer_params, whole_number
from atom.compass.kv.transfer import TransferModel
from atom.kv_transfer.disaggregation.base import (
    KVConnectorBase,
    KVConnectorSchedulerBase,
)
from atom.kv_transfer.disaggregation.types import ConnectorMetadata

logger = logging.getLogger(__name__)

#: Where the harness binds the priced transfer model.
TRANSFER_KEY = "compass_transfer"

#: The zmq address the write requests travel over.
WRITE_REQ_KEY = "compass_kv_write_req"


class UnboundSeam(RuntimeError):
    """The harness did not bind something the connector cannot invent."""


def _kv_config(config) -> dict:
    return getattr(config, "kv_transfer_config", {}) or {}


def _is_producer(kv_config: dict) -> bool:
    return kv_config.get("kv_role", "kv_producer") == "kv_producer"


def _bound_transfer(kv_config: dict) -> TransferModel:
    model = kv_config.get(TRANSFER_KEY)
    if not isinstance(model, TransferModel):
        raise UnboundSeam(
            f"kv_transfer_config[{TRANSFER_KEY!r}] holds {model!r}, where this "
            "connector needs a TransferModel priced from the machine spec's "
            "interconnect and the KV geometry of the model being simulated"
        )
    return model


class SimulatedKVConnector(KVConnectorBase):
    """Worker side: no device bytes, nothing to start, nothing to report."""

    def __init__(self, config) -> None:
        pass

    def register_kv_caches(
        self, kv_caches, transfer_tensors=None, num_blocks=None
    ) -> None:
        """Nothing to register: no KV tensor exists to expose to a remote."""

    def start_load_kv(self, metadata: ConnectorMetadata) -> None:
        pass

    def get_finished(self) -> tuple[set, set]:
        return set(), set()


class SimulatedKVConnectorScheduler(KVConnectorSchedulerBase):
    """Scheduler side: Mooncake's write request, and both ends' completion times."""

    def __init__(self, config) -> None:
        import zmq

        from atom.utils import clock

        kv_config = _kv_config(config)
        self.is_producer = _is_producer(kv_config)
        # Checked here so a malformed width refuses the connector before any
        # request is served, not at the first request_finished.
        parallel = config.parallel_config
        self._tp_size = whole_number("tp_size", config.tensor_parallel_size)
        self._dp_rank = whole_number("dp_rank", parallel.data_parallel_rank)
        self._pp_size = whole_number("pp_size", config.pipeline_parallel_size)
        if self._pp_size > 1:
            raise ValueError(
                f"pipeline_parallel_size is {self._pp_size}, which this connector "
                "refuses: a PP head passes worker KV output to the scheduler only "
                "when it is non-empty, so process_completions never runs there, "
                "and Mooncake's MSG_RELEASE channel is not modelled"
            )
        self._hash_block_size = (
            config.kv_cache_block_size * config.decode_context_parallel_size
        )
        self._transfer = _bound_transfer(kv_config)
        self._rt = clock.installed()
        if self._rt is None:
            raise UnboundSeam("no LP runtime is installed to time transfers on")
        address = kv_config.get(WRITE_REQ_KEY)
        if not isinstance(address, str):
            raise UnboundSeam(
                f"kv_transfer_config[{WRITE_REQ_KEY!r}] holds {address!r}, where "
                "this connector needs the zmq address of its write requests"
            )
        channel = clock.channel_of(self._rt, "kv_write_req")
        # The write request's latency, and the write-done notice's.
        self._latency_s = self._rt.table.lookahead(channel)
        raw = zmq.Context.instance().socket(zmq.PULL if self.is_producer else zmq.PUSH)
        (raw.bind if self.is_producer else raw.connect)(address)
        self._sock = clock.WrappedSocket(self._rt, raw, channel)
        self._ready: dict = {}  # prefill: request -> r
        self._due: dict = {}  # request -> when it is reported finished

    def get_num_new_matched_tokens(self, seq) -> tuple[int, bool]:
        """Claim a remote-filled prompt whole, so the engine parks it."""
        params = seq.kv_transfer_params or {}
        if params.get("do_remote_prefill") and not getattr(
            seq, "kv_async_tagged", False
        ):
            return len(seq.prompt_token_ids), True
        return 0, False

    def build_connector_meta(self) -> ConnectorMetadata:
        return ConnectorMetadata()

    def update_state_after_alloc(self, seq) -> None:
        """Decode: send the write request and clear the flag, as Mooncake does.

        The engine parks the request right after this call, so every write
        request has a parked request waiting for it.
        """
        params = seq.kv_transfer_params or {}
        if not params.get("do_remote_prefill"):
            return
        if self.is_producer:
            raise ValueError(
                f"request {seq.id!r} asks to be filled from another deployment "
                "while this connector was built as the producer side. Only the "
                "decoding side takes a remote fill on; a producer queueing one "
                "would wait for a transfer it is itself supposed to serve"
            )
        params["do_remote_prefill"] = False
        computed = 0
        if params.get("hash_block_size") == self._hash_block_size > 0 and not (
            seq.has_per_req_cache
        ):
            computed = seq.num_cached_tokens // self._hash_block_size
        blocks = len(seq.block_table) - computed
        # The arrival `stamp_send` gives this send.
        a = self._rt.now + self._latency_s
        self._sock.send(pickle.dumps((params["transfer_id"], blocks, a)))
        self._due[seq.id] = a + self._transfer.duration_s(blocks) + self._latency_s

    def request_finished(self, seq) -> None:
        """Record prefill's data-ready time, once, and attach the relayed blob.

        The engine calls this twice for one finished request; the first call
        is the time the data was ready.
        """
        if self.is_producer:
            self._ready.setdefault(seq.id, self._rt.read_clock())
        seq.kv_transfer_params_output = transfer_params(
            seq,
            tp_size=self._tp_size,
            dp_rank=self._dp_rank,
            pp_size=self._pp_size,
            hash_block_size=self._hash_block_size,
        )

    def process_completions(self, output):
        """Take the write requests released so far; report what the clock reached."""
        while self.is_producer and self._sock.poll(0):
            req_id, blocks, a = pickle.loads(self._sock.recv())
            r = self._ready.pop(req_id, None)
            if r is None or r > a:
                raise ValueError(
                    f"the write request for {req_id!r} arrived at {a}, before "
                    f"its prefill finished (at {r}); the router sends a request "
                    "to decode only after prefill has returned"
                )
            t = self._transfer.duration_s(blocks)
            self._due[req_id] = max(a, r) + t
            logger.info(
                "compass: KV transfer %s, %d blocks, %r simulated s", req_id, blocks, t
            )
        now = self._rt.read_clock()
        done = [req_id for req_id, t in self._due.items() if t <= now]
        for req_id in done:
            del self._due[req_id]
        if self.is_producer:
            output.finished_sending.update(done)
        else:
            output.finished_recving.update(done)
        return output

    def has_pending_work(self) -> bool:
        """Keeps an idle decode engine polling until its parked requests are ready."""
        return bool(self._due)
