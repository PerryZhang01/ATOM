# SPDX-License-Identifier: MIT
"""The Clock Authority: which waiting process may move its clock, to when, and
with which messages.

A logical process (LP) asks for time with one of two requests:

* ``TAR`` -- advance to ``t``: its owner has priced an event, and the grant is
  exactly ``t``;
* ``NER`` -- idle until ``t``, which may be ``+inf``: the grant is ``t``, its
  daemon deadline, or the earliest message registered for it, whichever comes
  first. A daemon deadline is a housekeeping timer (a metrics push, a scrape)
  that fires as usual but does not keep the run alive.

Each request carries the requester's send log since its previous request, as
``(channel, seq, arrival)``. The log is registered before the requester's state
changes, so every message an LP produced is known before its clock moves.

``N[j]`` is the earliest time LP *j* could still produce a message: its clock
while it runs, its target while it waits in TAR (its owner is inside this call
and sends nothing), and the least of its target, its daemon deadline and its
earliest unreleased message while it waits in NER. A waiting LP *i* is granted
exactly ``N[i]``, and only when ``N[i] < min over j != i of (N[j] + D(j->i))``.
``D`` is the channel table's least summed lookahead over every channel path, so
an idle LP between two others does not hide the first from the third. The comparison is strict: a
message not yet reported arrives no earlier than its sender's ``N`` plus that
distance, so a strict grant never reaches it. Waiting LPs are tried in
``(N, name)`` order and the scan restarts after every grant.

The essential horizon ``H`` is the largest TAR target, finite NER target or
registered arrival seen so far; it only grows. A waiting LP is grantable only
while ``N <= H``, so one whose ``N`` is its daemon deadline is held until some
essential time or arrival reaches that deadline. A held deadline delays no
essential grant: its ``N`` exceeds ``H``, and every essential ``N`` is at most
``H``. Because ``H`` is a function of the schedule, not of which running LP
reports first, a daemon deadline fires under every request order or none.

When every LP waits and the strict rule grants none, which only a
zero-lookahead cycle allows, the grantable LP with the least ``(N, name)`` is
granted: nothing is earlier, so nothing earlier can reach it. A message that
then lands at its receiver's current instant belongs to that instant's next
round. When every LP waits and none is grantable, nothing is undelivered and
no essential target is pending: the run is finished, every LP is granted
``+inf``, and the daemon deadlines still held never fire.

An LP may be declared with members: processes that each hold a runtime for it,
as the ranks of a data-parallel group do. A member names itself in each call,
and each channel into or out of the LP is owned by exactly one member, as the
caller declares; a member may log sends only on channels it owns. The members'
calls are joined into one request per round: each member makes the same number
of calls, all ask one kind, every TAR asks the same target, and the joined
NER's target and daemon deadline are each the least over the members. A call
before the round is complete registers its send log and nothing else: the LP
stays running, so its ``N`` stays at its clock. Every member gets the common
grant, carrying the releases on its own channels only.

A finite grant past the simulated-time bound aborts the run with the LP table.
A housekeeping timer nobody declared daemon keeps raising ``H`` and would
otherwise keep the run alive forever.

Nothing here reads a clock, opens a socket or starts a thread. The caller
carries requests in and replies out.
"""

import math
from dataclasses import dataclass

from .channels import ChannelTable
from .identity import LpId

TAR, NER = "TAR", "NER"
RUNNING = "running"


@dataclass(frozen=True, slots=True)
class LpRow:
    """One LP as the authority sees it."""

    lp: LpId
    state: str  # running, TAR or NER
    now: float
    target: float | None
    n: float  # the earliest time it could still produce a message
    undelivered: tuple[tuple[str, int, float], ...]  # (channel, seq, arrival)
    row: tuple[tuple[LpId, float], ...]  # (j, N[j] + D(j->lp)), every other LP
    binding: LpId | None  # the first least term of `row`, None if none is finite


class ClockAbort(Exception):
    """The run cannot be trusted any further. Carries every LP's row."""

    def __init__(self, reason: str, table: tuple[LpRow, ...]) -> None:
        super().__init__("\n".join([reason, *map(repr, table)]))
        self.reason = reason
        self.table = table


class BackdatedEvent(ClockAbort):
    """A request that would put a message or a clock into some LP's past."""


def _seconds(value, finite: bool, what: str, *args) -> float:
    """`value` as seconds; `what.format(*args)` names it in the refusal, and is
    only formatted for one."""
    seconds = float(value)
    if math.isnan(seconds) or seconds == -math.inf or (finite and seconds == math.inf):
        kind = "a finite number" if finite else "a number"
        raise ValueError(
            f"{what.format(*args)} must be {kind} of seconds, got {value!r}"
        )
    return seconds


class ClockAuthority:
    """Every LP's clock and state, and the rule that grants them time.

    `timeline`, when given, receives one ``record(lp, from, to, kind,
    recovered)`` call per reply, in issue order. `grants` counts each LP's
    finite grants by name. `final_clocks` is every LP's clock just before the run
    finished, and ``None`` until it has. `members` maps an LP to
    ``{member: names of the channels into or out of it that the member owns}``;
    an LP not in it is its own single caller. `bound_s` is the simulated-time
    bound, ``+inf`` for none.
    """

    def __init__(
        self,
        channels: ChannelTable,
        timeline=None,
        members=None,
        bound_s: float = math.inf,
    ) -> None:
        self._channels = channels
        self._bound = _seconds(bound_s, False, "the simulated-time bound")
        self._ids = channels.registry.ids()
        self._now = dict.fromkeys(self._ids, 0.0)
        self._state = dict.fromkeys(self._ids, RUNNING)
        self._target = dict.fromkeys(self._ids)
        self._daemon = dict.fromkeys(self._ids, math.inf)
        self._horizon = -math.inf  # H: the latest essential time seen
        self._into = {
            i: tuple(c.name for c in channels.channels_into(i)) for i in self._ids
        }
        # Per channel, seq -> arrival of every registered message not yet
        # released, in registration order.
        self._undelivered = {
            name: {} for names in self._into.values() for name in names
        }
        self._next_seq = dict.fromkeys(self._undelivered, 0)
        # N[j] of every LP, kept current by each call that changes one of its
        # inputs, and D(j->i) of every pair: a grant scan reads both, never
        # recomputing them.
        self._nv = dict.fromkeys(self._ids, 0.0)
        self._lp = {i: i for i in self._ids}
        self._from = {
            i: tuple((j, channels.distance(j, i)) for j in self._ids if j != i)
            for i in self._ids
        }
        self.timeline = timeline
        self.grants = {i.name: 0 for i in self._ids}
        self.final_clocks = None
        self._members = {}  # LP -> its member names, sorted
        self._owner = {}  # LP -> channel into or out of it -> the member owning it
        self._round = {}  # LP -> member -> (kind, t, daemon) of the round being joined
        for lp, owned in (members or {}).items():
            channels.registry.require(lp)
            if not owned:
                raise ValueError(f"{lp} is declared with an empty member list")
            touching = dict.fromkeys(
                [*self._into[lp], *(c.name for c in channels.channels_from(lp))]
            )
            listed = dict.fromkeys(name for names in owned.values() for name in names)
            self._owner[lp] = {}
            for name in sorted(touching | listed):
                owners = sorted(m for m, names in owned.items() if name in names)
                if name not in touching:
                    raise ValueError(
                        f"{' and '.join(owners)} of {lp} owns {name}, which "
                        f"neither goes into nor comes out of {lp}"
                    )
                if len(owners) != 1:
                    raise ValueError(
                        f"{name} goes into or out of {lp}, and "
                        + (
                            f"its members {' and '.join(owners)} each own it"
                            if owners
                            else f"none of its members {', '.join(sorted(owned))} owns it"
                        )
                    )
                self._owner[lp][name] = owners[0]
            self._members[lp] = tuple(sorted(owned))
            self._round[lp] = {}

    def on_request(
        self,
        lp: LpId,
        kind: str,
        t: float,
        log,
        t_daemon: float = math.inf,
        member: str | None = None,
    ) -> list:
        """Register `lp`'s send log, record its request, and grant what is due.

        `t_daemon` is an NER's daemon deadline, ``+inf`` for none. `member`
        names the caller when `lp` has members, and is ``None`` otherwise.
        Returns the replies made due, as ``(lp, G, released)`` in the order
        issued, where ``released`` maps each channel into that LP to its newly
        released ``(seq, arrival)`` pairs; a member's reply is addressed
        ``(lp, member)`` and carries its own channels only. The requester is
        absent when its reply is held. Once the run is finished, every LP has
        had its ``+inf`` reply, and a request changes nothing and returns none.
        """
        if self.final_clocks is not None:
            return []
        # The registered object itself: every lookup below then matches by
        # identity rather than by calling LpId.__eq__.
        lp = self._lp.get(lp) or self._channels.registry.require(lp)
        members = self._members.get(lp, ())
        if member not in (members or (None,)):
            raise KeyError(
                f"{lp} was called by member {member!r}; its members: "
                + (", ".join(members) or "none")
            )
        if self._state[lp] != RUNNING:
            raise BackdatedEvent(
                f"{lp} sent {kind} while waiting in {self._state[lp]}; only a "
                "running LP can produce a request",
                self.lp_table(),
            )
        if kind not in (TAR, NER):
            raise ValueError(f"{kind!r} is not one of {TAR}, {NER}")
        t = _seconds(t, kind == TAR, "the {} target of {}", kind, lp)
        daemon = _seconds(t_daemon, False, "the daemon deadline of {}", lp)
        if kind == TAR and daemon != math.inf:
            raise ValueError(
                f"{lp} sent TAR with daemon deadline {daemon}; only NER has one"
            )
        now = self._now[lp]
        if t < now or daemon < now:
            asked = f"{kind}({t})" if t < now else f"daemon deadline {daemon}"
            raise BackdatedEvent(
                f"{lp} asked for {asked} behind its own clock at {now}",
                self.lp_table(),
            )
        if members:
            self._refuse_unjoinable(lp, member, kind, t)
        for name, seq, arrival in log:
            self._register(lp, name, seq, arrival, member)
        if members:
            joined = self._round[lp]
            joined[member] = (kind, t, daemon)
            if len(joined) < len(members):
                return []
            t = min(asked for _, asked, _ in joined.values())
            daemon = min(deadline for _, _, deadline in joined.values())
            joined.clear()
        if t < math.inf:
            self._horizon = max(self._horizon, t)
        self._state[lp], self._target[lp], self._daemon[lp] = kind, t, daemon
        self._nv[lp] = t if kind == TAR else self._n(lp)
        return self._address(self._grant_due())

    def _refuse_unjoinable(self, lp: LpId, member: str, kind: str, t: float) -> None:
        joined = self._round[lp]
        reason = None
        if member in joined:
            behind = ", ".join(m for m in self._members[lp] if m not in joined)
            reason = (
                f"{lp}: {member} called again before {behind} called; every "
                "member calls once per round"
            )
        elif joined:
            other, (other_kind, other_t, _) = next(iter(joined.items()))
            if other_kind != kind:
                reason = f"{lp}: {member} asked {kind} while {other} asked {other_kind}"
            elif kind == TAR and other_t != t:
                reason = (
                    f"{lp}: {member} asked TAR({t}) while {other} asked TAR({other_t})"
                )
        if reason is not None:
            raise ClockAbort(reason, self.lp_table())

    def _address(self, replies: list) -> list:
        """One reply per member for an LP with members, each with its own channels."""
        return [
            (
                (i, g, released)
                if m is None
                else (
                    (i, m),
                    g,
                    {c: r for c, r in released.items() if self._owner[i][c] == m},
                )
            )
            for i, g, released in replies
            for m in self._members.get(i, (None,))
        ]

    def lp_table(self) -> tuple[LpRow, ...]:
        """Every LP's row, in name order."""
        rows = []
        for i in self._ids:
            row = self._row(i)
            least = min(row, key=lambda term: term[1], default=(None, math.inf))
            rows.append(
                LpRow(
                    i,
                    self._state[i],
                    self._now[i],
                    self._target[i],
                    self._n(i),
                    self._pending(i),
                    row,
                    least[0] if least[1] < math.inf else None,
                )
            )
        return tuple(rows)

    def _register(
        self, lp: LpId, name: str, seq: int, arrival: float, member: str | None
    ) -> None:
        channel = self._channels.channel(name)
        a = _seconds(arrival, True, "the arrival of {} seq {}", name, seq)
        expected = self._next_seq[name]
        floor = self._now[lp] + channel.lookahead_s
        if channel.source != lp:
            refusal = f"{lp} logged a send on {name}, whose sender is {channel.source}"
        elif member is not None and self._owner[lp][name] != member:
            refusal = (
                f"{lp}: {member} logged a send on {name}, which "
                f"{self._owner[lp][name]} owns"
            )
        elif seq != expected:
            refusal = (
                f"{lp} logged seq {seq} on {name}, which expects seq {expected} "
                "next; a sender numbers each channel consecutively"
            )
        elif a < self._now[channel.target]:
            refusal = (
                f"{name} seq {seq} arrives at {a}, behind {channel.target}'s "
                f"clock at {self._now[channel.target]}"
            )
        elif a < floor:
            refusal = (
                f"{name} seq {seq} arrives at {a}, before {lp}'s clock "
                f"{self._now[lp]} plus the channel's lookahead {channel.lookahead_s}"
            )
        else:
            self._undelivered[name][seq] = a
            self._horizon = max(self._horizon, a)
            self._next_seq[name] = seq + 1
            if self._state[channel.target] == NER:
                self._nv[channel.target] = self._n(channel.target)
            return
        raise BackdatedEvent(refusal, self.lp_table())

    def _pending(self, i: LpId) -> tuple[tuple[str, int, float], ...]:
        return tuple(
            (name, seq, a)
            for name in self._into[i]
            for seq, a in self._undelivered[name].items()
        )

    def _n(self, j: LpId) -> float:
        if self._state[j] == RUNNING:
            return self._now[j]
        if self._state[j] == TAR:
            return self._target[j]
        n = min(self._target[j], self._daemon[j])
        for name in self._into[j]:
            for a in self._undelivered[name].values():
                n = min(n, a)
        return n

    def _row(self, i: LpId) -> tuple[tuple[LpId, float], ...]:
        nv = self._nv
        return tuple((j, nv[j] + d) for j, d in self._from[i])

    def _lbts(self, i: LpId) -> float:
        return min((term for _, term in self._row(i)), default=math.inf)

    def _grant_due(self) -> list:
        replies = []
        state, nv = self._state, self._nv
        while True:
            # `state` iterates in name order and the sort is stable, so equal
            # N stay in name order without comparing two LpIds.
            waiting = sorted(
                ((nv[i], i) for i, s in state.items() if s != RUNNING),
                key=lambda waiter: waiter[0],
            )
            grantable = [(n, i) for n, i in waiting if n <= self._horizon]
            due = next(((i, n) for n, i in grantable if n < self._lbts(i)), None)
            if due is not None:
                replies.append(self._grant(*due))
            elif len(waiting) < len(self._ids):
                return replies
            elif not grantable:
                return replies + self._finish()
            else:
                replies.append(self._recover(grantable[0][1], grantable[0][0]))

    def _recover(self, i: LpId, g: float) -> tuple:
        # The one grant whose guarantee is weaker: a message may still land at g.
        return self._grant(i, g, recovered=True)

    def _grant(self, i: LpId, g: float, recovered: bool = False) -> tuple:
        if g > self._bound:
            raise ClockAbort(
                f"a grant to {g} for {i} passes the simulated-time bound "
                f"{self._bound}; either the bound is shorter than the workload, "
                "or something essential, such as a housekeeping timer not "
                "declared daemon, keeps the run alive",
                self.lp_table(),
            )
        released = {}
        for name in self._into[i]:
            pending = self._undelivered[name]
            released[name] = [(seq, a) for seq, a in pending.items() if a <= g]
            for seq, _ in released[name]:
                del pending[seq]
        if self.timeline is not None:
            self.timeline.record(i, self._now[i], g, self._state[i], recovered)
        self.grants[i.name] += 1
        # g is N[i], so N[i] is unchanged: a running LP's N is its clock.
        self._now[i], self._state[i], self._target[i] = g, RUNNING, None
        return (i, g, released)

    def _finish(self) -> list:
        self.final_clocks = tuple(self._now.items())
        for i in self._ids:
            if self.timeline is not None:
                self.timeline.record(i, self._now[i], math.inf, self._state[i], False)
            self._now[i], self._state[i], self._target[i] = math.inf, RUNNING, None
            self._nv[i] = math.inf
        return [(i, math.inf, {name: [] for name in self._into[i]}) for i in self._ids]
