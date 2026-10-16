# SPDX-License-Identifier: MIT
"""`zmq` for the modules whose sockets carry Compass channels.

Every name is pyzmq's own except `Context`, `Poller` and `Socket`, which are
chosen on each read: pyzmq's classes while no LP runtime is installed, so a real
run makes exactly the objects it did. With a runtime installed, a socket made
by `Context` is pyzmq's until `bind` or `connect` names an address recorded by
`clock.name_endpoints`, and from then on it is that channel's `WrappedSocket`;
`Poller` makes a `WrappedPoller`.
"""

import zmq as _zmq
from zmq import asyncio  # noqa: F401 - imported so `zmq.asyncio` is bound below

from atom.utils import clock

_SWAPPED = ("Context", "Poller", "Socket")
globals().update(
    (k, v)
    for k, v in vars(_zmq).items()
    if not k.startswith("__") and k not in _SWAPPED
)


def __getattr__(name: str):
    if name not in _SWAPPED:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    rt = clock.installed()
    if rt is None:
        return getattr(_zmq, name)
    return {
        "Context": _Context,
        "Poller": lambda: clock.WrappedPoller(rt),
        "Socket": _Socket,
    }[name]


class _Socket:
    """A socket of a Compass run: pyzmq's own until it names a channel address."""

    # The wrapper's own fields; every other attribute is read from and set on `raw`.
    _OWN = ("raw", "rt", "ch", "into", "relay", "buf", "__class__")

    def __init__(self, raw: _zmq.Socket) -> None:
        self.raw = raw

    def __getattr__(self, name: str):
        return getattr(self.raw, name)

    def __setattr__(self, name: str, value) -> None:
        if name in self._OWN:
            object.__setattr__(self, name, value)
        else:
            setattr(self.raw, name, value)

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.raw.close()

    def bind(self, addr: str) -> None:
        self.raw.bind(addr)
        self._name(addr)

    def connect(self, addr: str) -> None:
        self.raw.connect(addr)
        self._name(addr)

    def _name(self, addr: str) -> None:
        rt = clock.installed()
        kind = None if rt is None else rt.endpoints.get(addr)
        if kind is None or isinstance(self, clock.WrappedSocket):
            return
        # The object ATOM already holds becomes the channel's socket in place.
        self.__class__ = _ChannelSocket
        clock.WrappedSocket.__init__(
            self, rt, self.raw, *clock.socket_channels(rt, kind)
        )
        relay = rt.relays.get(self.ch)
        if relay is not None:
            relay.wsock, self.relay = self, relay


class _ChannelSocket(_Socket, clock.WrappedSocket):
    """A `WrappedSocket` that is still the `_Socket` ATOM made."""


class _Context(_zmq.Context):
    _instance = None  # `instance()` keeps its own, not pyzmq's

    def socket(self, socket_type: int, **kw) -> _Socket:
        return _Socket(super().socket(socket_type, **kw))
