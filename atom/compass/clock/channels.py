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
#: path, in seconds. Earlier work measured 13.7 ms end-to-end -- worth around
#: four points of time-to-first-token -- and found the two paths differ enough
#: that one number for both would misprice whichever was not measured.
ADMISSION_DELAY_S = {
    "offline_batch": 13.0e-3,
    "serving": 9.0e-3,
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


def single_engine_table(
    *, admission_path: str, ipc_s: float, stream_s: float, dp: int = 1
) -> ChannelTable:
    """Traffic, one frontend, one engine, and the channels of each of its `dp`
    data-parallel ranks, ``#dp<rank>``.

    The request channel's lookahead is the measured admission delay of
    `admission_path` (``serving`` or ``offline_batch``). The other lookaheads
    have no measurement behind them, so the caller declares them.
    """
    return _table(
        ("traffic", "frontend", "engine"),
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
        + _frontend_engine("frontend", "engine", ipc_s, dp),
    )


def prefill_decode_table(
    *,
    admission_path: str,
    ipc_s: float,
    stream_s: float,
    router_s: float,
    kv_write_req_s: float,
) -> ChannelTable:
    """One prefill and one decode deployment behind the router.

    Requests enter at frontend-P and the stream leaves from frontend-D. The
    router has no choice to make with one of each, so it is not a process of its
    own: its per-request forward cost `router_s` is added to every channel that
    crosses it -- the request, the stream and the relay from frontend-P to
    frontend-D. Decode posts the KV write request to prefill engine to engine.
    """
    return _table(
        ("traffic", "frontend-P", "engine-P", "frontend-D", "engine-D"),
        [
            (
                "traffic",
                "frontend-P",
                "http",
                ADMISSION_DELAY_S[admission_path] + router_s,
                "inline",
            ),
            ("frontend-D", "traffic", "stream", stream_s + router_s, "inline"),
            ("frontend-P", "frontend-D", "relay", router_s, "inline"),
            ("engine-D", "engine-P", "kv_write_req", kv_write_req_s, "inline"),
        ]
        + _frontend_engine("frontend-P", "engine-P", ipc_s)
        + _frontend_engine("frontend-D", "engine-D", ipc_s),
    )
