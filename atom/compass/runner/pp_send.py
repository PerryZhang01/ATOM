# SPDX-License-Identifier: MIT
"""A pipeline stage's send to the next stage, and when it completes on whose clock.

A send is several parts, each on one carrier: ``gloo`` for the size and the
pickled metadata, ``rccl`` for each device tensor, in the order the receiver
takes them. A part up to its carrier's eager threshold completes on the
sender's clock at ``t_send + o + bytes x G``; a larger one is rendezvous and
completes on both sides at ``max(t_send, r) + T``, ``r`` the time the receiver
posts its receive and ``T = rendezvous_fixed_s + bytes x G``. The receiver
posts each part's receive when the part before it has arrived.

`StageSends` is the stage loop's side. Before a downstream stage's forward it
receives that forward's input on ``pp_data`` and, for a rendezvous send,
returns the completion on ``pp_ack`` stamped at it; after a non-last stage's
forward it waits for its previous send's completion, then sends. An idle
flush waits for the previous send the same way. The frames move between the
stages' TP-rank-0 workers over the PP CPU group.
"""

import math
from dataclasses import dataclass, fields

from atom.utils import clock

CARRIERS = ("rccl", "gloo")
PREFIX = "interconnect.intra_node.pp"


@dataclass(frozen=True)
class CarrierModel:
    eager_threshold_bytes: int
    latency_s: float
    overhead_s: float
    per_byte_s: float
    rendezvous_fixed_s: float

    def eager(self, nbytes: int) -> bool:
        return nbytes <= self.eager_threshold_bytes

    def rendezvous_s(self, nbytes: int) -> float:
        return self.rendezvous_fixed_s + nbytes * self.per_byte_s


def carrier_models(spec) -> dict[str, CarrierModel]:
    """Each carrier's model from a `MachineSpec`; a missing field is refused by name."""
    return {
        c: CarrierModel(
            *(spec.value(f"{PREFIX}.{c}.{f.name}") for f in fields(CarrierModel))
        )
        for c in CARRIERS
    }


def lookaheads(models: dict[str, CarrierModel]) -> tuple[float, float]:
    """``(L_data, L_ack)``: the least carrier latency, and the least rendezvous
    duration, at a carrier's threshold, less it. A non-positive either is refused."""
    for c, m in models.items():
        if not m.latency_s > 0:
            raise ValueError(f"the {c} carrier's latency_s is {m.latency_s!r}")
    l_data = min(m.latency_s for m in models.values())
    t_min = min(m.rendezvous_s(m.eager_threshold_bytes) for m in models.values())
    if not t_min > l_data:
        raise ValueError(
            f"T_min {t_min!r} is not above L_data {l_data!r}: a rendezvous would "
            "complete before its request reaches the receiver, so pp_ack has no "
            "positive lookahead"
        )
    return l_data, t_min - l_data


def is_rendezvous(models, parts) -> bool:
    return any(not models[c].eager(n) for c, n in parts)


def local_done(models, t_send: float, parts) -> float:
    """An eager send's completion on the sender's clock: its latest part."""
    return max(
        t_send + models[c].overhead_s + n * models[c].per_byte_s for c, n in parts
    )


def recv_done(models, t_send: float, t_recv_posted: float, parts):
    """``(done, end)``: the send's latest sender-side completion, and the
    receive's end, folding the parts in the receiver's order."""
    r, done = t_recv_posted, -math.inf
    for c, n in parts:
        m = models[c]
        if m.eager(n):
            sent = t_send + m.overhead_s + n * m.per_byte_s
            r = max(r, sent + m.latency_s)
            done = max(done, sent)
        else:
            r = max(t_send, r) + m.rendezvous_s(n)
            done = max(done, r)
    return done, r


class StageSends:
    """One stage loop's ``pp_data`` and ``pp_ack``, around its runner calls.

    `call` is the runner manager's own `call_func`; `__call__` stands in for
    it, and acts only on ``forward`` and ``flush_pp_send`` inside the run.
    """

    def __init__(self, rt, call, rank: int, pp: int, models) -> None:
        self.rt, self.call, self.models = rt, call, models

        def ch(a, b, kind):
            return f"engine-pp{a}->engine-pp{b}:{kind}#dp0"

        self.data_in = rank > 0 and ch(rank - 1, rank, "pp_data")
        self.ack_out = rank > 0 and ch(rank, rank - 1, "pp_ack")
        self.data_out = rank < pp - 1 and ch(rank, rank + 1, "pp_data")
        self.ack_in = rank < pp - 1 and ch(rank + 1, rank, "pp_ack")
        self.received = self.sent = 0
        self.pending = None  # the previous send: ("ack", seq) or ("eager", done)

    def __call__(self, name, *args, wait_out=False):
        if not self.rt.in_run or name not in ("forward", "flush_pp_send"):
            return self.call(name, *args, wait_out=wait_out)
        if name == "flush_pp_send":
            self.settle()
            return self.call(name, *args, wait_out=wait_out)
        if self.data_in:
            self.receive()
        t0 = self.rt.now
        reply = self.call(name, *args, wait_out=wait_out)
        if self.data_out:
            ready_s, parts = reply.pp_send
            self.send(t0 + ready_s, parts)
        return reply

    def _frame(self, kind: str, ch: str, seq: int) -> dict:
        """Receive (`ch`, `seq`) inline, then read its frame off the worker."""
        self.rt.receive(ch, seq)
        frame = self.call("recv_pp_frame", kind, wait_out=True)
        if frame["seq"] != seq:
            raise RuntimeError(f"{ch}: read frame {frame['seq']}, released {seq}")
        return frame

    def receive(self) -> None:
        seq, posted = self.received, self.rt.now
        self.received += 1
        frame = self._frame("pp_data", self.data_in, seq)
        parts = frame["parts"]
        done, end = recv_done(self.models, frame["t_send"], posted, parts)
        if is_rendezvous(self.models, parts):
            self.call("send_pp_frame", "pp_ack", {"seq": seq}, wait_out=True)
            self.rt.stamp_send(self.ack_out, done)
        # ponytail: the host waits for the whole receive, where ATOM's waits only
        # for the metadata on gloo and its device for the tensors; split the two
        # once host laws make launch time matter here.
        if end > self.rt.now:
            self.rt.advance_to(end)

    def send(self, ready: float, parts) -> None:
        """Send once the forward's device work, at `ready`, and the previous
        send are done; the loop itself waits only for the previous send."""
        self.settle()
        seq, t_send = self.sent, max(ready, self.rt.now)
        self.sent += 1
        frame = {"seq": seq, "t_send": t_send, "parts": parts}
        self.call("send_pp_frame", "pp_data", frame, wait_out=True)
        self.rt.stamp_send(
            self.data_out, t_send + self.rt.table.lookahead(self.data_out)
        )
        if is_rendezvous(self.models, parts):
            self.pending = ("ack", seq)
        else:
            self.pending = ("eager", local_done(self.models, t_send, parts))

    def settle(self) -> None:
        """Wait for the previous send: its ack if rendezvous, else its local end."""
        pending, self.pending = self.pending, None
        if pending is None:
            return
        kind, at = pending
        if kind == "ack":
            self._frame("pp_ack", self.ack_in, at)
        elif at > self.rt.now:
            self.rt.advance_to(at)


def install(engine, spec) -> None:
    """Put `StageSends` in front of a pipeline stage's runner calls, and make
    the stage's bounded polls essential while it holds a send or KV work."""
    rt = clock.installed()
    mgr = engine.runner_mgr
    sends = StageSends(
        rt, mgr.call_func, engine.pp_rank, engine.pp_size, carrier_models(spec)
    )
    mgr.call_func = sends
    rt.holds_work = lambda: sends.pending is not None or engine.has_pending_kv_work()
