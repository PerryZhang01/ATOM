# SPDX-License-Identifier: MIT
"""The ``tcp`` carrier: the Clock Authority's serve loop reached over a socket.

``serve(authority, "tcp://host:port")`` listens there; port 0 binds a free one,
and the result's ``endpoint`` names the port it got. Each accepted connection
gets one thread. Its first frame binds it to an LP through the serve loop, which
answers by sending that frame back, or a refusal. After that, for each request
frame, the thread queues it with the loop and writes back the next reply from
that LP's slot. The thread waits on its own LP's slot under no lock, so an LP
whose reply is held stops no other connection. When a bound connection ends,
however it ends, the thread tells the loop, which ends the run.

``connect(lp, "tcp://host:port")`` returns the in-process carrier's connection
with a socket in place of the loop: the same ``send``, ``recv`` and ``close``,
and the same frames, each preceded by its length as four big-endian bytes, plus
``fileno`` to wait for a reply alongside other sockets. A
connection that ends inside a frame is a `MalformedMessage`. A frame the loop
will not queue, such as a reply kind sent as a request, is refused at ``recv``
here and at ``send`` in-process, with the same reason, and reaches no authority.
"""

import contextlib
import socket
import threading

from .service import _Connection, _refusal, _reply, _Server
from .wire import MalformedMessage, encode

STREAM_SCHEME = "tcp"
#: A frame is a few hundred bytes. A length far above that is a peer that is not
#: speaking this protocol, and is refused before anything is allocated for it.
LARGEST_FRAME = 1 << 22


def _send(stream, frame: bytes) -> None:
    stream.write(len(frame).to_bytes(4, "big") + frame)
    stream.flush()


def _receive(stream) -> bytes | None:
    """The next frame, or `None` when the peer closed between frames."""
    header = stream.read(4)
    if not header:
        return None
    size = int.from_bytes(header, "big")
    if size > LARGEST_FRAME:
        raise MalformedMessage(f"a {size}-byte frame is over {LARGEST_FRAME}")
    frame = stream.read(size)
    if len(header) < 4 or len(frame) < size:
        raise MalformedMessage("the connection ended inside a frame")
    return frame


def _end(sock: socket.socket) -> None:
    """Wake any thread reading `sock`, then close it."""
    with contextlib.suppress(OSError):
        sock.shutdown(socket.SHUT_RDWR)
    sock.close()


class _StreamServer:
    """A serve loop that LPs reach over TCP."""

    def __init__(self, authority, host: str, port: int) -> None:
        self._listener = socket.create_server((host, port))
        host, port = self._listener.getsockname()[:2]
        self.endpoint = f"{STREAM_SCHEME}://{host}:{port}"
        self._server = _Server(authority, self.endpoint)
        self._open = []
        threading.Thread(target=self._admit, daemon=True).start()

    def close(self) -> None:
        """Stop listening, end every connection and stop the loop."""
        for sock in [self._listener, *self._open]:
            _end(sock)
        self._server.close()

    def _admit(self) -> None:
        while True:
            try:
                sock, _ = self._listener.accept()
            except OSError:
                return
            self._open.append(sock)
            threading.Thread(target=self._carry, args=(sock,), daemon=True).start()

    def _carry(self, sock: socket.socket) -> None:
        # ponytail: a thread parked in `slot.get()` when the server closes stays
        # parked until the process exits; wake it with a sentinel if servers churn.
        slot = None
        # The writer flushes again on leaving the block, so a dead peer can
        # raise there too.
        try:
            with sock, sock.makefile("rwb") as stream:
                while (frame := _receive(stream)) is not None:
                    try:
                        if slot is None:
                            address, slot = self._server.bind(frame)
                            reply = frame
                        else:
                            self._server.submit(address, frame)
                            reply = slot.get()
                    except (KeyError, ValueError) as refused:
                        reply = encode(_refusal(refused))
                    _send(stream, reply)
        except (MalformedMessage, OSError):
            pass
        finally:
            if slot is not None:
                self._server._closed(address)


class _Remote:
    """The LP's end of a TCP connection, in the server-and-slot shape
    `_Connection` drives."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.create_connection((host, port))
        self._stream = self._sock.makefile("rwb")

    def bind(self, frame: bytes) -> tuple:
        _send(self._stream, frame)
        try:
            _reply(self.get())
        except Exception:
            self.close()
            raise
        return None, self

    def submit(self, _lp, frame: bytes) -> None:
        _send(self._stream, frame)

    def get(self) -> bytes:
        frame = _receive(self._stream)
        if frame is None:
            raise MalformedMessage("the clock closed the connection with no reply")
        return frame

    def close(self) -> None:
        self._stream.close()
        _end(self._sock)


class _StreamConnection(_Connection):
    def fileno(self) -> int:
        """The socket's, readable once a reply is on its way."""
        return self._server._sock.fileno()

    def close(self) -> None:
        """End the socket."""
        self._server.close()


def serve(authority, endpoint: str) -> _StreamServer:
    """Serve `authority` at ``tcp://host:port``."""
    return _StreamServer(authority, *_address(endpoint))


def connect(lp, endpoint: str, member: str | None = None) -> _StreamConnection:
    """`lp`'s connection, or its `member`'s, to the authority served at
    ``tcp://host:port``."""
    return _StreamConnection(_Remote(*_address(endpoint)), lp, member)


def _address(endpoint: str) -> tuple[str, int]:
    prefix = f"{STREAM_SCHEME}://"
    host, _, port = endpoint.removeprefix(prefix).rpartition(":")
    if not endpoint.startswith(prefix) or not host or not port.isdigit():
        raise ValueError(f"{endpoint!r} is not {prefix}<host>:<port>")
    return host, int(port)
