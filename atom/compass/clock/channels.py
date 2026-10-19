# SPDX-License-Identifier: MIT
"""The channels between logical processes, and how far apart that puts them.

A channel is one real communication path from a sending endpoint to a
receiving endpoint, named ``src->dst:kind#inst``. Its lookahead is a floor on
the delay any message on it suffers before it takes effect on the receiver,
and its receive mode says who takes a message off it: ``thread`` when a
transport or handler thread receives and the clock owner releases each message
and waits for it to be handled, ``inline`` when the owner receives it itself.

Lookahead is per channel, not per pair of processes: a frontend and an engine
are joined by several channels, and two processes with no channel between them
have no lookahead at all -- declaring one would describe a path that does not
exist. What a grant needs is the distance ``D(j->i)``: the least summed
lookahead over every channel path from *j* to *i*, infinite when there is none.
Lookahead is static, so the distances are computed once, by Floyd-Warshall,
and again only if a channel is declared afterwards.
"""

import math
from dataclasses import dataclass
from enum import StrEnum

from .identity import LpId
from .registry import LpRegistry


class ReceiveMode(StrEnum):
    THREAD = "thread"
    INLINE = "inline"


#: Measured admission delay from the traffic source into the serving stack, per
#: path, in seconds; a run charges tokenization on top. ``serving`` was measured
#: on hjbog-srdc-18 with the parity workload (`atom/compass/parity/traffic.py`,
#: Qwen3-30B-A3B at TP2 with DP attention): the median, over its requests, of
#: send to response headers less the prompt's encode time on that host.
#: ``offline_batch`` is from earlier work, which measured 13.7 ms end-to-end and
#: found the two paths differ enough that one number would misprice one of them.
ADMISSION_DELAY_S = {
    "offline_batch": 13.0e-3,
    "serving": 1.6e-3,
}


@dataclass(frozen=True, slots=True)
class Channel:
    """One declared channel. Frozen: the accessors hand it out."""

    name: str
    source: LpId
    target: LpId
    lookahead_s: float
    receive: ReceiveMode


class ChannelTable:
    """The channel list of a run, addressed by channel name."""

    def __init__(self, registry: LpRegistry) -> None:
        self.registry = registry
        self._channels: dict[str, Channel] = {}
        self._distances: dict[tuple[LpId, LpId], float] | None = None

    def declare(
        self,
        name: str,
        source: LpId,
        target: LpId,
        lookahead_s: float,
        receive: str,
    ) -> Channel:
        """Declare one channel. A zero lookahead is accepted; it serializes the two ends."""
        self.registry.require(source)
        self.registry.require(target)
        if source == target:
            raise ValueError(
                f"{name!r} joins {source} to itself; a channel crosses two logical processes"
            )
        if name in self._channels:
            raise ValueError(f"{name!r} is already declared as {self._channels[name]}")
        lookahead = float(lookahead_s)
        if not math.isfinite(lookahead) or lookahead < 0.0:
            raise ValueError(
                f"the lookahead of {name!r} must be a finite number of seconds and "
                f"not negative, got {lookahead_s!r}"
            )
        if receive not in tuple(ReceiveMode):
            raise ValueError(
                f"the receive mode of {name!r} must be one of "
                f"{', '.join(ReceiveMode)}, got {receive!r}"
            )
        channel = Channel(name, source, target, lookahead, ReceiveMode(receive))
        self._channels[name] = channel
        self._distances = None
        return channel

    def channel(self, name: str) -> Channel:
        """The channel called `name`, or a refusal listing every declared name."""
        found = self._channels.get(name)
        if found is None:
            declared = ", ".join(sorted(self._channels)) or "<none>"
            raise KeyError(f"{name!r} is not a declared channel; declared: {declared}")
        return found

    def channels_into(self, lp: LpId) -> tuple[Channel, ...]:
        """Every channel whose receiver is `lp`, by name."""
        self.registry.require(lp)
        return tuple(c for _, c in sorted(self._channels.items()) if c.target == lp)

    def channels_from(self, lp: LpId) -> tuple[Channel, ...]:
        """Every channel whose sender is `lp`, by name."""
        self.registry.require(lp)
        return tuple(c for _, c in sorted(self._channels.items()) if c.source == lp)

    def lookahead(self, name: str) -> float:
        return self.channel(name).lookahead_s

    def recv_mode(self, name: str) -> ReceiveMode:
        return self.channel(name).receive

    def distance(self, source: LpId, target: LpId) -> float:
        """``D(source->target)``: zero to itself, infinite with no channel path."""
        self.registry.require(source)
        self.registry.require(target)
        if source == target:
            return 0.0
        if self._distances is None:
            self._distances = self._floyd_warshall()
        return self._distances.get((source, target), math.inf)

    def _floyd_warshall(self) -> dict[tuple[LpId, LpId], float]:
        # Absent pairs are infinite. Diagonal entries are never read.
        d: dict[tuple[LpId, LpId], float] = {}
        for c in self._channels.values():
            d[c.source, c.target] = min(
                d.get((c.source, c.target), math.inf), c.lookahead_s
            )
        ids = self.registry.ids()
        for k in ids:
            for j in ids:
                for i in ids:
                    via = d.get((j, k), math.inf) + d.get((k, i), math.inf)
                    if via < d.get((j, i), math.inf):
                        d[j, i] = via
        return d


def _table(lps: tuple[str, ...], rows: list) -> ChannelTable:
    registry = LpRegistry()
    for lp in lps:
        registry.register(LpId(lp))
    table = ChannelTable(registry)
    for src, dst, kind, lookahead_s, receive in rows:
        table.declare(
            f"{src}->{dst}:{kind}", LpId(src), LpId(dst), lookahead_s, receive
        )
    return table


def _frontend_engine(frontend: str, engine: str, ipc_s: float, dp: int = 1) -> list:
    """Request and control in, output back, for each of `dp` data-parallel ranks."""
    return [
        row
        for r in range(dp)
        for row in (
            (frontend, engine, f"request#dp{r}", ipc_s, "thread"),
            (frontend, engine, f"control#dp{r}", ipc_s, "thread"),
            (engine, frontend, f"output#dp{r}", ipc_s, "thread"),
        )
    ]


def _pipeline(pp: int, ipc_s: float, links=()) -> tuple[tuple[str, ...], list]:
    """The engine LPs ``engine-pp<k>`` of a `pp`-stage pipeline and their channels.

    Each stage is an engine of its own to the frontend, tagged with the index
    ATOM's engine core manager gives it, ``#dp<k>``. The head sends each
    downstream stage its batch on ``meta#pp<k>``, the last stage sends the
    head its output on ``tokens``, and each downstream stage reports KV status
    to the head on ``kv_status``. These are ZMQ IPC messages, so each takes
    `ipc_s`, which must be positive: head to stage and back is a cycle.

    With `links`, ``(L_data, L_ack)``, each stage sends the next its forward's
    output on ``pp_data#dp0`` and the next returns a rendezvous send's
    completion on ``pp_ack#dp0``, with those lookaheads.
    """
    if not ipc_s > 0.0:
        raise ValueError(
            f"ipc_s is {ipc_s!r}; the {pp} pipeline stages send to the head and "
            "the head to them, a cycle that needs a positive lookahead"
        )
    stages = tuple(f"engine-pp{k}" for k in range(pp))
    head, last = stages[0], stages[-1]
    rows = [(last, head, "tokens", ipc_s, "inline")]
    for k, stage in enumerate(stages):
        rows += [
            ("frontend", stage, f"request#dp{k}", ipc_s, "thread"),
            ("frontend", stage, f"control#dp{k}", ipc_s, "thread"),
            (stage, "frontend", f"output#dp{k}", ipc_s, "thread"),
        ]
        if k:
            rows += [
                (head, stage, f"meta#pp{k}", ipc_s, "inline"),
                (stage, head, "kv_status", ipc_s, "inline"),
            ]
            if links:
                rows += [
                    (stages[k - 1], stage, "pp_data#dp0", links[0], "inline"),
                    (stage, stages[k - 1], "pp_ack#dp0", links[1], "inline"),
                ]
    return stages, rows


def _rapidserve(ipc_s: float) -> tuple[tuple[str, ...], list]:
    """RapidServe's prefill engine LP ``engine-P`` and decode engine LP
    ``engine-D`` and their channels.

    Each is an engine of its own to the frontend, tagged with the index ATOM's
    disaggregated core manager gives it, ``#dp0`` and ``#dp1``. Decode sends
    prefill each request's blocks on ``block_assignment`` and prefill sends
    decode each prefill's end on ``prefill_done``; a thread on each side takes
    them. These are ZMQ IPC messages, so each takes `ipc_s`, which must be
    positive: the two engines send to each other, a cycle.
    """
    if not ipc_s > 0.0:
        raise ValueError(
            f"ipc_s is {ipc_s!r}; RapidServe's two engines send to each other, "
            "a cycle that needs a positive lookahead"
        )
    engines = ("engine-P", "engine-D")
    rows = [
        ("engine-P", "engine-D", "prefill_done", ipc_s, "thread"),
        ("engine-D", "engine-P", "block_assignment", ipc_s, "thread"),
    ]
    for k, engine in enumerate(engines):
        rows += [
            ("frontend", engine, f"request#dp{k}", ipc_s, "thread"),
            ("frontend", engine, f"control#dp{k}", ipc_s, "thread"),
            (engine, "frontend", f"output#dp{k}", ipc_s, "thread"),
        ]
    return engines, rows


def single_engine_table(
    *,
    admission_path: str,
    ipc_s: float,
    stream_s: float,
    dp: int = 1,
    pp: int = 1,
    pp_links=(),
    rapidserve: bool = False,
) -> ChannelTable:
    """Traffic, one frontend, one engine, and the channels of each of its `dp`
    data-parallel ranks, ``#dp<rank>``; above pipeline width one, an engine
    LP per stage instead (`_pipeline`), and under `rapidserve` its prefill and
    decode engine LPs (`_rapidserve`).

    The request channel's lookahead is the measured admission delay of
    `admission_path` (``serving`` or ``offline_batch``). The other lookaheads
    have no measurement behind them, so the caller declares them.
    """
    engines, rows = ("engine",), _frontend_engine("frontend", "engine", ipc_s, dp)
    if pp > 1:
        engines, rows = _pipeline(pp, ipc_s, pp_links)
    if rapidserve:
        engines, rows = _rapidserve(ipc_s)
    return _table(
        ("traffic", "frontend", *engines),
        [
            (
                "traffic",
                "frontend",
                "http",
                ADMISSION_DELAY_S[admission_path],
                "inline",
            ),
            ("frontend", "traffic", "stream", stream_s, "inline"),
        ]
        + rows,
    )


def add_lmcache(
    table: ChannelTable,
    engine: str,
    lmcache_ipc_s: float,
    dp: int = 1,
    writers: int = 1,
) -> None:
    """LP ``lmcache``, the lmcache MP server, and its channels to and from `engine`.

    Per DP rank, ``lookup#dp<r>`` carries the scheduler adapter's requests and
    their replies, which lmcache's client polling thread takes; per writer rank
    ``k``, ``transfer#dp<r>.tp<k>`` carries a STORE or RETRIEVE from the step
    loop and its reply back to it. The server's loop takes every request
    itself. Each is one ZMQ hop of `lmcache_ipc_s`, which must be positive:
    engine to server and back is a cycle.
    """
    if not lmcache_ipc_s > 0.0:
        raise ValueError(
            f"lmcache_ipc_s is {lmcache_ipc_s!r}; the engine sends to the lmcache "
            "server and the server back, a cycle that needs a positive lookahead"
        )
    eng, lm = LpId(engine), table.registry.register(LpId("lmcache"))
    kinds = [(f"lookup#dp{r}", "thread") for r in range(dp)]
    kinds += [
        (f"transfer#dp{r}.tp{k}", "inline") for r in range(dp) for k in range(writers)
    ]
    for kind, back in kinds:
        table.declare(f"{engine}->lmcache:{kind}", eng, lm, lmcache_ipc_s, "inline")
        table.declare(f"lmcache->{engine}:{kind}", lm, eng, lmcache_ipc_s, back)


def prefill_decode_table(
    *,
    admission_path: str,
    ipc_s: float,
    stream_s: float,
    router_s: float,
    kv_write_req_s: float,
    dp: int | tuple[int, int] = 1,
    router_in_s: float | None = None,
    prefill: int = 1,
    decode: int = 1,
) -> ChannelTable:
    """`prefill` prefill and `decode` decode deployments behind the router,
    each with the channels of its data-parallel ranks: `dp` of them on each
    side, or ``(prefill width, decode width)``.

    Requests enter at frontend-P and the stream leaves from frontend-D. Three
    paths cross the router -- the request, the stream and the relay from
    frontend-P to frontend-D -- and each carries its per-request forward cost
    `router_s`. Without `router_in_s` the router has no choice to make, so it is
    not a process of its own and each path is one channel. With it the router
    is LP ``router``, and each path is a hop in, whose lookahead is
    `router_in_s`, and a hop out, which keeps the rest of the path's lookahead;
    both are positive, so `router_in_s` lies strictly between 0 and `router_s`.
    The router receives each hop in on a thread and handles it before the next.
    Decode posts the KV write request to prefill engine to engine, above width
    one on either side on a channel per decode and prefill rank pair,
    ``kv_write_req#dp<decode rank>.dp<prefill rank>``.

    Above one deployment of a role, deployment ``k`` of it is the LPs
    ``frontend-P<k>`` and ``engine-P<k>`` (``-D<k>`` for decode), each with its
    own hops through the router, which chooses among them and so must be an
    LP, and each decode and prefill engine pair has its own write-request
    channels.
    """
    if max(prefill, decode) > 1 and router_in_s is None:
        raise ValueError(
            f"{prefill} prefill and {decode} decode deployments put a choice in "
            "the router, which makes it an LP: set router_in_s"
        )
    ps = [f"P{k}" if prefill > 1 else "P" for k in range(prefill)]
    ds = [f"D{k}" if decode > 1 else "D" for k in range(decode)]
    paths = [
        (
            "traffic",
            f"frontend-{p}",
            "http",
            ADMISSION_DELAY_S[admission_path] + router_s,
        )
        for p in ps
    ]
    paths += [(f"frontend-{d}", "traffic", "stream", stream_s + router_s) for d in ds]
    paths += [
        (f"frontend-{p}", f"frontend-{d}", "relay", router_s) for p in ps for d in ds
    ]
    lps = ("traffic", *(f"{lp}-{s}" for s in ps + ds for lp in ("frontend", "engine")))
    if router_in_s is None:
        hops = [(src, dst, kind, total, "inline") for src, dst, kind, total in paths]
    elif 0.0 < router_in_s < router_s:
        lps += ("router",)
        # Paths sharing an end share that end's hop.
        hops = dict.fromkeys(
            hop
            for src, dst, kind, total in paths
            for hop in (
                (src, "router", kind, router_in_s, "thread"),
                ("router", dst, kind, total - router_in_s, "inline"),
            )
        )
    else:
        raise ValueError(
            f"router_in_s is {router_in_s!r} and router_s {router_s!r}; each hop "
            "through the router LP needs a positive lookahead, so 0 < router_in_s "
            "< router_s"
        )
    dp_p, dp_d = (dp, dp) if isinstance(dp, int) else dp
    pairs = [f"#dp{d}.dp{p}" for d in range(dp_d) for p in range(dp_p)]
    pairs = [""] if pairs == ["#dp0.dp0"] else pairs
    width = dict.fromkeys(ps, dp_p) | dict.fromkeys(ds, dp_d)
    return _table(
        lps,
        [*hops]
        + [
            (
                f"engine-{d}",
                f"engine-{p}",
                "kv_write_req" + pair,
                kv_write_req_s,
                "inline",
            )
            for d in ds
            for p in ps
            for pair in pairs
        ]
        + [
            row
            for s in ps + ds
            for row in _frontend_engine(f"frontend-{s}", f"engine-{s}", ipc_s, width[s])
        ],
    )
