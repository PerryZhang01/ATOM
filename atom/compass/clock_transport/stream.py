# SPDX-License-Identifier: MIT
"""The ``tcp`` carrier: the Clock Authority's serve loop reached over a socket.

``serve(authority, "tcp://host:port")`` listens there; port 0 binds a free one,
and the result's ``endpoint`` names the port it got. Each accepted connection
gets a reader thread. Its first frame binds it to an LP through the serve loop,
which answers with what the LP starts from, or a refusal. After that, the reader
queues each request frame with the loop, and a writer thread sends each reply
from that LP's slot. The writer waits on its own LP's slot under no lock, so an
LP whose reply is held stops no other connection. The reader waits for no reply,
so when a bound connection ends, however it ends and even while its LP waits for
a grant, it tells the loop then, which ends the run.

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
from .wire import JOINED, MalformedMessage, encode

STREAM_SCHEME = "tcp"
#: A frame is a few hundred bytes. A length far above that is a peer that is not
#: speaking this protocol, and is refused before anything is allocated for it.
LARGEST_FRAME = 1 << 22


def _framed(frame: bytes) -> bytes:
    return len(frame).to_bytes(4, "big") + frame


def _send(stream, frame: bytes) -> None:
    stream.write(_framed(frame))
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
            threading.Thread(target=self._read, args=(sock,), daemon=True).start()

    def _read(self, sock: socket.socket) -> None:
        slot = None
        try:
            with sock.makefile("rb") as stream:
                while (frame := _receive(stream)) is not None:
                    try:
                        if slot is None:
                            address, slot, joined = self._server.bind(frame)
                            threading.Thread(
                                target=_write,
                                args=(sock, slot, (JOINED, *joined)),
                                daemon=True,
                            ).start()
                        else:
                            self._server.submit(address, frame)
                    except (KeyError, ValueError) as refused:
                        reply = encode(_refusal(refused))
                        if slot is None:
                            sock.sendall(_framed(reply))
                        else:
                            slot.put(reply)
        except (MalformedMessage, OSError):
            pass
        finally:
            if slot is None:
                _end(sock)
            else:
                self._server._closed(address, slot)
                slot.put(None)


def _write(sock: socket.socket, slot, joined: tuple) -> None:
    """Send `joined`, then each reply `slot` gets, on `sock` until its reader
    puts `None`; then end `sock`."""
    with contextlib.suppress(MalformedMessage, OSError):
        reply = encode(joined)
        while reply is not None:
            sock.sendall(_framed(reply))
            reply = slot.get()
    _end(sock)


class _Remote:
    """The LP's end of a TCP connection, in the server-and-slot shape
    `_Connection` drives."""

    def __init__(self, host: str, port: int) -> None:
        self._sock = socket.create_connection((host, port))
        self._stream = self._sock.makefile("rwb")

    def bind(self, frame: bytes) -> tuple:
        _send(self._stream, frame)
        try:
            _, *joined = _reply(self.get())
        except Exception:
            self.close()
            raise
        return None, self, tuple(joined)

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
