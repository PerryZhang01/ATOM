# SPDX-License-Identifier: MIT
"""The parity workload: distinct completions sent on a schedule, to a real or a simulated run.

    python -m atom.compass.parity.traffic PORT MODEL RECORD [REQUESTS [GAP_S]]

Request k's prompt starts with k, and it asks for a fixed number of tokens
with ``ignore_eos``, streamed. On a real run (`ATOM_COMPASS_RUN` unset)
request k is sent GAP_S * k wall seconds after the first and not before the
server has admitted request k - 1, each on its own thread; the worker clock's
reading as each was sent goes to `ARRIVALS` in RECORD, the real run's parity
record directory. On a simulated run this process is the traffic LP: it sends
request k at that reading less the first one, on the run's clock, and takes
every stream event the authority releases until the finish.
"""

import http.client
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atom.compass import run as compass_run
from atom.compass.carriers import sse_stamp, tracestate_with
from atom.compass.clock_transport import connect
from atom.compass.parity import ARRIVALS
from atom.utils.clock import LPRuntime

DONE = "data: [DONE]"
WALL_S = 3600.0
WORDS = ["parity", "replays", "each", "real", "step", "on", "the", "clock"]


def bodies(model: str, n: int) -> list[dict]:
    """Request k: about 48 + 31k words starting with k, 16 to 96 output tokens."""
    return [
        {
            "model": model,
            "prompt": f"{k} "
            + " ".join(WORDS[(k + i) % len(WORDS)] for i in range(48 + 31 * k)),
            "max_tokens": 16 + 16 * (k % 6),
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
        }
        for k in range(n)
    ]


def post(port: int, body: dict, headers: dict):
    """The streamed response's lines, once the server has admitted the request
    (its headers follow the engine's `add_request`); raises unless it answers 200."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WALL_S)
    headers = {"content-type": "application/json", **headers}
    conn.request("POST", "/v1/completions", json.dumps(body), headers)
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    return (raw.decode().rstrip("\r\n") for raw in response)


def real(port: int, requests: list[dict], gap_s: float, record: Path) -> None:
    """Send request k at `gap_s` * k once k - 1 is admitted, read each to its
    end, write `ARRIVALS`."""
    sent, start = [0] * len(requests), time.monotonic()
    admitted = [threading.Event() for _ in requests]

    def one(k: int) -> None:
        time.sleep(max(0.0, start + k * gap_s - time.monotonic()))
        try:
            if k:
                admitted[k - 1].wait()
            sent[k] = time.monotonic_ns()
            lines = post(port, requests[k], {})
        finally:
            admitted[k].set()
        if DONE not in lines:
            raise RuntimeError(f"request {k} ended before {DONE}")

    with ThreadPoolExecutor(len(requests)) as pool:
        list(pool.map(one, range(len(requests))))
    (record / ARRIVALS).write_text(json.dumps({"sent_ns": sent}))


class Traffic:
    """The traffic LP of a simulated run: request k at `offsets[k]` on its clock."""

    def __init__(self, run: dict, port: int, requests: list, offsets: list) -> None:
        table = compass_run.channel_table(run)
        (http,) = table.channels_from(compass_run.TRAFFIC)
        (stream,) = table.channels_into(compass_run.TRAFFIC)
        self.http, self.stream, self.port = http.name, stream.name, port
        conn = connect(compass_run.TRAFFIC, run["clock_endpoint"])
        self.rt = LPRuntime(compass_run.TRAFFIC, table, conn)
        self.requests, self.offsets = requests, offsets
        self.cv = threading.Condition()
        self.frames: dict[int, str] = {}  # stream seq -> its data line, read, not taken
        self.errors: list[BaseException] = []
        self.done = 0

    def run(self) -> None:
        rt = self.rt
        rt.start_run()
        for body, at in sorted(zip(self.requests, self.offsets), key=lambda r: r[1]):
            while rt.now < at:
                rt.next_event(at)
                self._take()
            stamp = tracestate_with(None, *rt.stamp_send(self.http))
            threading.Thread(target=self._read, args=(body, stamp), daemon=True).start()
        while rt.next_event(math.inf) != math.inf:
            self._take()
        rt.close()
        if self.done != len(self.requests):
            raise RuntimeError(f"{len(self.requests) - self.done} requests never ended")

    def _read(self, body: dict, tracestate: str) -> None:
        try:
            stamp = None
            for line in post(self.port, body, {"tracestate": tracestate}):
                stamp = sse_stamp(line) or stamp
                if line.startswith("data:") and stamp is not None:
                    self.rt.check_arrival(self.stream, *stamp)
                    with self.cv:
                        self.frames[stamp[1]] = line
                        self.cv.notify_all()
                    stamp = None
        except BaseException as e:  # noqa: BLE001 - the clock owner raises it
            with self.cv:
                self.errors.append(e)
                self.cv.notify_all()

    def _take(self) -> None:
        """Take every released stream event, waiting for any not read yet."""
        with self.rt.lock:
            due = sorted(self.rt.released[self.stream] - self.rt.handled[self.stream])
        for seq in due:
            with self.cv:
                if not self.cv.wait_for(
                    lambda seq=seq: seq in self.frames or self.errors, WALL_S
                ):
                    raise RuntimeError(f"stream seq {seq} released and never read")
                if self.errors:
                    raise self.errors[0]
                self.done += self.frames.pop(seq) == DONE
            with self.rt.lock:
                self.rt.count_done_locked(self.stream, seq)


if __name__ == "__main__":
    port, model, record, *rest = sys.argv[1:]
    run, record = compass_run.spec(), Path(record)
    if run is None:
        n, gap_s = int(rest[0]) if rest else 64, float(rest[1]) if rest[1:] else 0.05
        real(int(port), bodies(model, n), gap_s, record)
    else:
        sent = json.loads((record / ARRIVALS).read_text())["sent_ns"]
        offsets = [(t - min(sent)) / 1e9 for t in sent]
        Traffic(run, int(port), bodies(model, len(sent)), offsets).run()
