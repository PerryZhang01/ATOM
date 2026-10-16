# SPDX-License-Identifier: MIT
"""The frames between a logical process (LP) and the Clock Authority.

A message is a tuple, one of these shapes:

* ``("BIND", lp)`` or ``("BIND", lp, member)`` -- the first frame on a
  connection; every later request on that connection is `lp`'s, made by
  `member` when `lp` is declared with members;
* ``("JOINED", now, seqs)`` -- the reply to a bind: the LP's clock and the next
  sequence number of each channel out of it, ``{channel: seq}``;
* ``(kind, t, log, t_daemon)`` with `kind` ``TAR`` or ``NER`` -- a request,
  where `log` is ``[(channel, seq, arrival)]``, the requester's sends since its
  previous request, and `t_daemon` an NER's daemon deadline, ``+inf`` for none;
* ``("GRANT", G, released)`` -- the reply, `released` being
  ``{channel: [(seq, arrival)]}``; ``("GRANT", G, released, k)`` also marks
  the end of client run `k`;
* ``("REFUSED", error, reason, table)`` -- the request was refused; `error` names
  the exception the requester raises, and `table` is the LP table a `ClockAbort`
  carries, or ``None``.

A frame is JSON with sorted keys and no spacing, so one message is one byte
string in every process. ``+inf`` and ``-inf`` travel as those two strings: an
idle LP asks for ``NER(+inf)``, and the bare ``Infinity`` Python would write is
not JSON any other reader accepts. A NaN has no spelling and is refused where it
would be written.
"""

import json
import math

from atom.compass.clock import (
    NER,
    TAR,
    BackdatedEvent,
    ClockAbort,
    LpId,
    LpRow,
)

BIND, JOINED, GRANT, REFUSED = "BIND", "JOINED", "GRANT", "REFUSED"
REQUESTS = (TAR, NER)
UNBOUNDED = {math.inf: "+inf", -math.inf: "-inf"}
BOUNDS = {name: value for value, name in UNBOUNDED.items()}
#: What a refusal can raise at the requester, most specific first.
REFUSALS = {
    cls.__name__: cls for cls in (BackdatedEvent, ClockAbort, KeyError, ValueError)
}


class MalformedMessage(ValueError):
    """A frame, or a value about to be framed, that is not one of the messages."""


def encode(message: tuple) -> bytes:
    """The frame for `message`. The same message gives the same bytes every time."""
    try:
        body = _ENCODE[message[0]](*message)
        return json.dumps(
            body, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (AttributeError, LookupError, TypeError, ValueError) as fault:
        raise MalformedMessage(f"{message!r} is not a message: {fault}") from fault


def decode(frame: bytes) -> tuple:
    """The message `frame` carries. Refuses anything that is not one."""
    if not isinstance(frame, (bytes, bytearray)):
        raise MalformedMessage(
            f"a frame is bytes, got {type(frame).__name__}; a carrier moves "
            "encoded frames, never message objects"
        )
    try:
        body = json.loads(frame, parse_constant=_refuse_bare_constant)
        return _DECODE[body["kind"]](body)
    except (LookupError, TypeError, ValueError) as fault:
        raise MalformedMessage(f"{bytes(frame)!r} is not a message: {fault}") from fault


def _out(seconds) -> float | str:
    value = float(seconds)
    return UNBOUNDED.get(value, value)


def _in(value) -> float:
    return BOUNDS[value] if isinstance(value, str) else float(value)


def _refuse_bare_constant(name: str):
    raise ValueError(f"bare {name} is not JSON; write one of {sorted(BOUNDS)}")


def _entry(channel, seq, arrival) -> tuple[str, int, float]:
    if not isinstance(channel, str) or type(seq) is not int:
        raise TypeError(f"{[channel, seq, arrival]!r} is not (channel, seq, arrival)")
    return channel, seq, _in(arrival)


def _row_out(row: LpRow) -> list:
    return [
        row.lp.name,
        row.state,
        _out(row.now),
        None if row.target is None else _out(row.target),
        _out(row.n),
        [[ch, seq, _out(a)] for ch, seq, a in row.undelivered],
        [[j.name, _out(term)] for j, term in row.row],
        None if row.binding is None else row.binding.name,
    ]


def _row_in(fields: list) -> LpRow:
    lp, state, now, target, n, undelivered, row, binding = fields
    return LpRow(
        LpId(lp),
        state,
        _in(now),
        None if target is None else _in(target),
        _in(n),
        tuple(_entry(*e) for e in undelivered),
        tuple((LpId(j), _in(term)) for j, term in row),
        None if binding is None else LpId(binding),
    )


def _request_out(kind, t, log, t_daemon) -> dict:
    return {
        "kind": kind,
        "t": _out(t),
        "log": [[c, s, _out(a)] for c, s, a in log],
        "t_daemon": _out(t_daemon),
    }


def _error(name: str) -> str:
    if name not in REFUSALS:
        raise ValueError(f"{name!r} is not a refusal; one of {list(REFUSALS)}")
    return name


def _refused_out(_, error, reason, table) -> dict:
    return {
        "kind": REFUSED,
        "error": _error(error),
        "reason": str(reason),
        "table": None if table is None else [_row_out(r) for r in table],
    }


def _refused_in(body) -> tuple:
    table = body["table"]
    return (
        REFUSED,
        _error(body["error"]),
        str(body["reason"]),
        None if table is None else tuple(_row_in(r) for r in table),
    )


def _member(name) -> str:
    if not isinstance(name, str):
        raise TypeError(f"a member is named by a str, got {name!r}")
    return name


def _bind_out(_, lp, member=None) -> dict:
    body = {"kind": BIND, "lp": lp.name}
    if member is not None:
        body["member"] = _member(member)
    return body


def _bind_in(body) -> tuple:
    bind = (BIND, LpId(body["lp"]))
    return (*bind, _member(body["member"])) if "member" in body else bind


def _grant_out(_, g, released, *ended) -> dict:
    body = {
        "kind": GRANT,
        "G": _out(g),
        "released": {
            ch: [[seq, _out(a)] for seq, a in pairs] for ch, pairs in released.items()
        },
    }
    if ended:
        (body["ended"],) = ended
    return body


def _grant_in(body) -> tuple:
    released = {
        ch: [(seq, _in(a)) for seq, a in pairs]
        for ch, pairs in body["released"].items()
    }
    ended = (body["ended"],) if "ended" in body else ()
    return (GRANT, _in(body["G"]), released, *ended)


_ENCODE = {
    BIND: _bind_out,
    JOINED: lambda _, now, seqs: {"kind": JOINED, "now": _out(now), "seqs": seqs},
    GRANT: _grant_out,
    REFUSED: _refused_out,
    **dict.fromkeys(REQUESTS, _request_out),
}

_DECODE = {
    BIND: _bind_in,
    JOINED: lambda body: (JOINED, _in(body["now"]), dict(body["seqs"])),
    GRANT: _grant_in,
    REFUSED: _refused_in,
    **dict.fromkeys(
        REQUESTS,
        lambda body: (
            body["kind"],
            _in(body["t"]),
            [_entry(*e) for e in body["log"]],
            _in(body["t_daemon"]),
        ),
    ),
}
