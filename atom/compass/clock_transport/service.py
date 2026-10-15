# SPDX-License-Identifier: MIT
"""The Clock Authority's serve loop, and the in-process carrier to it.

`serve` starts one thread that takes ``(address, request)`` off a single queue,
in arrival order, and passes it to `ClockAuthority.on_request`. An address is
an LP, or ``(lp, member)`` for a member of an LP declared with members; each
member binds its own connection. Each reply that call returns goes to its
address's own reply slot, whether or not it is waiting on it. A grant only ever
goes to an LP waiting in TAR or NER; the finish at the end of a run comes when
every LP waits, and answers each one's request with ``+inf``. A requester whose reply is held gets nothing until a later request
makes it due, and nothing times it out. A bind is answered with what the LP
starts from (`ClockAuthority.on_bind`).

The end of a client run before the last answers the client with ``+inf``: it
has resigned, so its address is unbound, its connection may end without ending
the run, and the next client binds the same LP.

A `KeyError` or `ValueError` from the authority refuses the request: the
requester's `recv` raises it as the same type, and the loop goes on serving. A
`ClockAbort`, such as `BackdatedEvent`, ends the run: its refusal goes to every
bound address's slot, waiting or not, and answers every later request. Any other
exception from the authority, or while framing a reply, ends the run the same
way, as a `ClockAbort` that names it; its table is empty when the LP table
cannot be read or framed. So does the end of a bound socket connection, as a
`ClockAbort` naming its address: no finish can come without it. A resigned
client's connection is no longer bound, and its end ends nothing.

Frames and closes reach the loop: a carrier hands the serve side encoded frames
and the serve side decodes them onto the queue, so the in-process carrier here
moves the same bytes a socket would; a socket's end reaches it as a ``None``
request. An LP waits for its reply on its own slot and under no lock shared with
another connection, so one parked LP holds up no other.
"""

import math
import queue
import threading

from atom.compass.clock import ClockAbort, LpId

from .wire import (
    BIND,
    GRANT,
    REFUSALS,
    REFUSED,
    REQUESTS,
    MalformedMessage,
    decode,
    encode,
)

IN_PROCESS_SCHEME = "inproc"
DEFAULT_ENDPOINT = f"{IN_PROCESS_SCHEME}:clock"

#: The authorities served in this process, by endpoint.
_SERVED: dict[str, "_Server"] = {}


class _Server:
    """The serve side of one endpoint. It takes in frames and closes."""

    def __init__(self, authority, endpoint: str) -> None:
        self.endpoint = endpoint
        self._authority = authority
        self._lps = [row.lp for row in authority.lp_table()]
        self._slots = {}
        self._requests = queue.Queue()
        self._thread = threading.Thread(
            target=self._loop, name=f"clock {endpoint}", daemon=True
        )
        self._thread.start()

    def bind(self, frame: bytes) -> tuple:
        """Bind a connection to the LP, or LP and member, its first frame names:
        that addressee, its reply slot and what it starts from. A member's name
        is checked by the authority at its first request."""
        kind, *rest = decode(frame)
        if kind != BIND:
            raise MalformedMessage(
                f"the first frame on a connection binds it to an LP, got {kind}"
            )
        lp, *member = rest
        address = (lp, *member) if member else lp
        slot = queue.Queue()
        if lp not in self._lps or self._slots.setdefault(address, slot) is not slot:
            state = "already bound" if lp in self._lps else "not a participant"
            raise KeyError(
                f"{' member '.join(map(str, rest))} is {state} at {self.endpoint}; "
                "participants: " + ", ".join(map(str, self._lps))
            )
        return address, slot, self._authority.on_bind(lp)

    def submit(self, address, frame: bytes) -> None:
        """Queue one request frame from the connection bound to `address`."""
        message = decode(frame)
        if message[0] not in REQUESTS:
            raise MalformedMessage(
                f"{address} sent {message[0]}; a bound connection sends only "
                + ", ".join(REQUESTS)
            )
        self._requests.put((address, message, None))

    def _closed(self, address, slot) -> None:
        """The connection bound to `address`, replying through `slot`, closed."""
        self._requests.put((address, None, slot))

    def close(self) -> None:
        """Stop the loop and give up the endpoint."""
        if _SERVED.get(self.endpoint) is self:
            del _SERVED[self.endpoint]
        self._requests.put(None)
        self._thread.join()

    def _loop(self) -> None:
        ended = None  # the refusal frame that ended the run, once one has
        while (item := self._requests.get()) is not None:
            address, request, slot = item
            lp, member = address if isinstance(address, tuple) else (address, None)
            replies, resigned = [(address, ended)], []
            if request is None:
                replies = []
                if ended is None and self._slots.get(address) is slot:
                    name = f"{lp} member {member}" if member else f"{lp}"
                    ended = self._ending(
                        f"{name} closed its connection, so the run ends here"
                    )
                    replies = [(i, ended) for i in list(self._slots)]
            elif ended is None:
                kind, t, log, t_daemon = request
                try:
                    try:
                        grants = self._authority.on_request(
                            lp, kind, t, log, t_daemon, member
                        )
                    except (KeyError, ValueError) as refused:
                        replies = [(address, encode(_refusal(refused)))]
                    else:
                        # Outside the refusing branch: a grant that cannot be
                        # framed ends the run, and no other grant goes out.
                        replies = [(i, encode((GRANT, *reply))) for i, *reply in grants]
                        # A client run ended: its client, granted +inf, resigned.
                        if any(len(reply) > 3 for reply in grants):
                            resigned = [i for i, g, *_ in grants if g == math.inf]
                except Exception as fault:  # noqa: BLE001 - ends the run, named
                    ended = self._ending(fault)
                    # A copy: a carrier thread may bind a new slot meanwhile.
                    replies = [(i, ended) for i in list(self._slots)]
            for i, frame in replies:
                self._slots[i].put(frame)
            for i in resigned:
                del self._slots[i]

    def _ending(self, fault: Exception | str) -> bytes:
        """The refusal frame that ends the run on `fault`, an exception or a
        reason. Anything but a `ClockAbort` travels as one naming it, with the
        LP table, or with an empty table naming why the table could not be read
        or framed."""
        if isinstance(fault, ClockAbort):
            return encode(_refusal(fault))
        reason = (
            fault if isinstance(fault, str) else f"the clock authority raised {fault!r}"
        )
        try:
            return encode(_refusal(ClockAbort(reason, self._authority.lp_table())))
        except Exception as unread:  # noqa: BLE001 - the run still ends, named
            return encode(
                _refusal(ClockAbort(f"{reason}; no LP table: {unread!r}", ()))
            )


def _refusal(refused: Exception) -> tuple:
    error = next(name for name, cls in REFUSALS.items() if isinstance(refused, cls))
    if isinstance(refused, ClockAbort):
        return REFUSED, error, refused.reason, refused.table
    return REFUSED, error, " ".join(map(str, refused.args)), None


def _reply(frame: bytes) -> tuple:
    """The message `frame` carries; a refusal raises as its own type."""
    kind, *rest = decode(frame)
    if kind == REFUSED:
        error, reason, table = rest
        cls = REFUSALS[error]
        raise cls(reason, table) if issubclass(cls, ClockAbort) else cls(reason)
    return kind, *rest


class _Connection:
    """One LP's or member's connection: ``send((kind, t, log, t_daemon))``,
    ``recv() -> (G, released)``, ``close()``."""

    def __init__(self, server: _Server, lp: LpId, member: str | None = None) -> None:
        self._server = server
        # `joined`: the clock and next sequence numbers the LP starts from.
        self._lp, self._slot, self.joined = server.bind(encode((BIND, lp, member)))

    def send(self, request: tuple) -> None:
        self._server.submit(self._lp, encode(request))

    def recv(self) -> tuple:
        """Block until this LP's reply exists, ``(G, released)`` with the client
        run's index appended when it marks one's end; a refusal raises."""
        _, *reply = _reply(self._slot.get())
        return tuple(reply)

    def close(self) -> None:
        """Nothing to release: the reply slot is the LP's, not the connection's."""


def serve(authority, endpoint: str = DEFAULT_ENDPOINT) -> _Server:
    """Serve `authority` at `endpoint`. `close()` on the result stops it."""
    if endpoint in _SERVED:
        raise ValueError(
            f"{endpoint} is already served in this process; two authorities at "
            "one endpoint would grant time from two states"
        )
    _SERVED[endpoint] = _Server(authority, endpoint)
    return _SERVED[endpoint]


def connect(
    lp: LpId, endpoint: str = DEFAULT_ENDPOINT, member: str | None = None
) -> _Connection:
    """`lp`'s connection, or its `member`'s, to the authority served at `endpoint`."""
    server = _SERVED.get(endpoint)
    if server is None:
        raise KeyError(
            f"nothing is served at {endpoint} in this process; served here: "
            + (", ".join(sorted(_SERVED)) or "<none>")
        )
    return _Connection(server, lp, member)
