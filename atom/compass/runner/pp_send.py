# SPDX-License-Identifier: MIT
"""A pipeline stage's send to the next stage, and when it completes on whose clock.

A send is several parts, each on one carrier: ``gloo`` for the size and the
pickled metadata, ``rccl`` for each device tensor, in the order the receiver
takes them. A part up to its carrier's eager threshold completes on the
sender's clock at ``t_send + o + bytes x G``; a larger one is rendezvous and
completes on both sides at ``max(t_send, r - E x G) + T``, ``r`` the time the
receiver posts its receive, ``E`` the threshold and ``T = rendezvous_fixed_s +
bytes x G``: up to a threshold's bytes leave before the receive is posted. A
part leaves when the part before it is done on the sender. The receiver
posts each part's receive when the part before it has arrived. The ``gloo``
parts leave at ``t_send``, the sender's host send; the ``rccl`` parts at
``t_ready``, when the sender's device work ends, or when the previous send's
``rccl`` parts are done, whichever is later; their receive waits for the receiver's device end.
At TP above 1 a tensor sent as its shard is a part ``(carrier, bytes, True)``:
the receiver all-gathers it over its TP group before it posts the next
part's receive.

`StageSends` is the stage loop's side. Before a downstream stage's forward it
receives that forward's input on ``pp_data``, the call and the metadata's arrival
as the batch's ``compass_pp_host`` and
the tensors' as the batch's ``compass_pp_release``, and, when the ``gloo`` parts
are rendezvous, returns their completion, and the ``rccl`` parts', on ``pp_ack`` stamped at it; at a
non-last stage's forward's launch end it waits for its previous send's ``gloo``
parts, then sends. A host that blocked on them for ``b`` seconds wakes
``wake_s x (1 - exp(-b / wake_rise_s))`` after they complete, gloo's wake.
The ``rccl`` parts hold the sender's stream, not its host: its
forwards after that wait start their device work no earlier. An idle
flush waits for the previous send the same way. The frames move between the
stages' TP-rank-0 workers over the PP CPU group.
"""

import math
from dataclasses import dataclass, fields

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
    fixed part, which a rendezvous outlasts after its receive is posted, less
    it. A non-positive either is refused."""
    for c, m in models.items():
        if not m.latency_s > 0:
            raise ValueError(f"the {c} carrier's latency_s is {m.latency_s!r}")
    l_data = min(m.latency_s for m in models.values())
    t_min = min(m.rendezvous_fixed_s for m in models.values())
    if not t_min > l_data:
        raise ValueError(
            f"T_min {t_min!r} is not above L_data {l_data!r}: a rendezvous would "
            "complete before its request reaches the receiver, so pp_ack has no "
            "positive lookahead"
        )
    return l_data, t_min - l_data


def gloo_wake(spec) -> tuple[float, float]:
    """``(wake_s, wake_rise_s)``, gloo's wake from a `MachineSpec`; a missing
    field is refused by name."""
    return tuple(spec.value(f"{PREFIX}.gloo.{f}") for f in ("wake_s", "wake_rise_s"))


def woken(wake, block: float) -> float:
    """How long after its gloo parts complete a host that blocked on them for
    `block` seconds goes on."""
    return wake[0] * -math.expm1(-block / wake[1]) if block > 0 else 0.0


def is_rendezvous(models, parts) -> bool:
    return any(not models[c].eager(n) for c, n, *_ in parts)


def recv_done(models, t_send: float, t_recv_posted: float, parts, gather=0.0):
    """``(done, end)``: the send's latest sender-side completion, and the
    receive's end, folding the parts in the receiver's order; a sharded part's
    all-gather takes `gather` seconds after it arrives."""
    r, done = t_recv_posted, -math.inf
    for c, n, *sharded in parts:
        m = models[c]
        if m.eager(n):
            sent = t_send + m.overhead_s + n * m.per_byte_s
            r = max(r, sent + m.latency_s)
            done = max(done, sent)
        else:
            pushed = m.eager_threshold_bytes * m.per_byte_s
            r = max(t_send, done, r - pushed) + m.rendezvous_s(n)
            done = max(done, r)
        if sharded:
            r += gather
    return done, r


def split(parts) -> tuple[list, list]:
    """A frame's ``gloo`` parts and its ``rccl`` parts."""
    return [p for p in parts if p[0] == "gloo"], [p for p in parts if p[0] != "gloo"]


def arrivals(models, frame, t_recv_posted: float):
    """``(sent, meta, end)`` of a ``pp_data`` frame: the ``gloo`` parts'
    sender-side completion, their arrival, and the ``rccl`` parts' after it,
    which are posted when the metadata has arrived."""
    gloo, tensors = split(frame["parts"])
    sent, meta = recv_done(models, frame["t_send"], t_recv_posted, gloo)
    return sent, meta, recv_done(models, frame["t_ready"], meta, tensors)[1]


class StageSends:
    """One stage loop's ``pp_data`` and ``pp_ack``, around its runner calls.

    `call` is the runner manager's own `call_func`; `__call__` stands in for
    it, and acts only on ``forward`` and ``flush_pp_send`` inside the run.
    """

    def __init__(self, rt, call, rank: int, pp: int, models, wake=(0.0, 1.0)) -> None:
        self.rt, self.call, self.models, self.wake = rt, call, models, wake

        def ch(a, b, kind):
            return f"engine-pp{a}->engine-pp{b}:{kind}#dp0"

        self.data_in = rank > 0 and ch(rank - 1, rank, "pp_data")
        self.ack_out = rank > 0 and ch(rank, rank - 1, "pp_ack")
        self.data_out = rank < pp - 1 and ch(rank, rank + 1, "pp_data")
        self.ack_in = rank < pp - 1 and ch(rank + 1, rank, "pp_ack")
        self.received = self.sent = 0
        self.pending = None  # the last send: ("ack", seq, _) or ("eager", done, held)
        self.tensors_done = -math.inf  # the previous frame received's rccl parts
        self.held = -math.inf  # the rccl parts' end of the send last waited for

    def __call__(self, name, *args, wait_out=False):
        if not self.rt.in_run or name not in ("forward", "flush_pp_send"):
            return self.call(name, *args, wait_out=wait_out)
        if name == "flush_pp_send":
            self.settle()
            return self.call(name, *args, wait_out=wait_out)
        if self.held is None:  # an eager send's rendezvous rccl part
            raise RuntimeError(
                f"{self.data_out}: no ack says when the last send's rccl parts end"
            )
        release = self.held
        if self.data_in:  # the tensors' arrival, no earlier than `held`
            args[0].compass_pp_host, release = self.receive(args[0])
        if release > -math.inf:
            args[0].compass_pp_release = release
        t0 = self.rt.now
        reply = self.call(name, *args, wait_out=wait_out)
        if self.data_out:
            ready_s, post_s, parts = reply.pp_send
            self.send(t0 + ready_s, parts)
            if post_s > 0:
                self.rt.advance_to(self.rt.now + post_s)
        return reply

    def _frame(self, kind: str, ch: str, seq: int) -> dict:
        """Receive (`ch`, `seq`) inline, then read its frame off the worker."""
        self.rt.receive(ch, seq)
        frame = self.call("recv_pp_frame", kind, wait_out=True)
        if frame["seq"] != seq:
            raise RuntimeError(f"{ch}: read frame {frame['seq']}, released {seq}")
        return frame

    def receive(self, batch) -> tuple:
        """``(called, meta)`` and the tensors' arrival, posted at `batch`'s prepare end.
        The tensors wait for the stage's device end, and leave once the previous
        frame's are done: one stream sends both. Each sharded tensor's
        all-gather is priced by the worker's cost backend."""
        from atom.compass.runner.overrides import forward_shape

        seq, called = self.received, self.rt.now
        self.received += 1
        frame = self._frame("pp_data", self.data_in, seq)
        shape = forward_shape(batch)
        posted, device = self.call("pp_recv_posted", called, *shape, wait_out=True)
        gloo, tensors = split(frame["parts"])
        sent, meta = recv_done(self.models, frame["t_send"], posted, gloo)
        ready = max(frame["t_ready"], self.tensors_done)
        posted = max(meta, device, self.held)
        gather = 0.0
        if any(len(p) > 2 for p in tensors):
            rows = batch.total_tokens_num
            gather = self.call("pp_all_gather_s", rows, wait_out=True)
        self.tensors_done, end = recv_done(self.models, ready, posted, tensors, gather)
        if is_rendezvous(self.models, gloo):
            ack = {"seq": seq, "tensors_done": self.tensors_done}
            self.call("send_pp_frame", "pp_ack", ack, wait_out=True)
            self.rt.stamp_send(self.ack_out, sent)
        return (called, meta), end

    def send(self, ready: float, parts) -> None:
        """Send once the previous send's ``gloo`` parts are done: the metadata
        now, the tensors once the forward's device work, at `ready`, is done."""
        self.settle()
        seq, t_send = self.sent, self.rt.now
        self.sent += 1
        frame = {"seq": seq, "t_send": t_send, "t_ready": ready, "parts": parts}
        self.call("send_pp_frame", "pp_data", frame, wait_out=True)
        self.rt.stamp_send(
            self.data_out, t_send + self.rt.table.lookahead(self.data_out)
        )
        gloo, tensors = split(parts)
        if is_rendezvous(self.models, gloo):
            self.pending = ("ack", seq, None)
        else:
            done = None  # a rendezvous rccl part ends at the receiver's post
            if self.held is not None and not is_rendezvous(self.models, tensors):
                done = recv_done(self.models, max(ready, self.held), ready, tensors)[0]
            self.pending = ("eager", arrivals(self.models, frame, t_send)[0], done)

    def settle(self) -> None:
        """Wait for the previous send's ``gloo`` parts: their ack if
        rendezvous, else their local end, and take its rccl parts' end as `held`."""
        pending, self.pending = self.pending, None
        if pending is None:
            return
        kind, at, self.held = pending
        if kind == "ack":
            entry = self.rt.now
            self.held = self._frame("pp_ack", self.ack_in, at)["tensors_done"]
            if late := woken(self.wake, self.rt.now - entry):
                self.rt.advance_to(self.rt.now + late)
        elif at > self.rt.now:
            self.rt.advance_to(at)


def install(engine, spec) -> None:
    """Put `StageSends` in front of a pipeline stage's runner calls, and make
    the stage's bounded polls essential while it holds a send or KV work."""
    from atom.utils import clock

    rt = clock.installed()
    mgr = engine.runner_mgr
    sends = StageSends(
        rt,
        mgr.call_func,
        engine.pp_rank,
        engine.pp_size,
        carrier_models(spec),
        gloo_wake(spec),
    )
    mgr.call_func = sends
    rt.holds_work = lambda: sends.pending is not None or engine.has_pending_kv_work()
