# SPDX-License-Identifier: MIT
"""The parity workload: distinct completions sent on a schedule, to a real or a simulated run.

    python -m atom.compass.parity.traffic PORT MODEL RECORD [REQUESTS [GAP_S [PREFIX_WORDS [IMAGES]]]]

Request k's prompt starts with PREFIX_WORDS words every request shares (none
by default), then k; with IMAGES it is a chat message after that many images
(`image`). It asks for a fixed number of tokens with
``ignore_eos``, streamed. On a real run (`ATOM_COMPASS_RUN` unset)
request 0 is sent alone and read to its end; then request k is sent
GAP_S * (k - 1) wall seconds after request 1 and not before the server has
admitted request k - 1, each on its own thread; the worker clock's
reading as each was sent, and as its response headers arrived (the server
answers once it has handed the request to the engine), goes to `ARRIVALS` in
RECORD, the real run's parity record directory. On a simulated run this
process is the traffic LP: it sends request k at its send reading less the
first one, on the run's clock, and takes every stream event the authority
releases until the finish.
"""

import base64
import http.client
import io
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


def image(k: int) -> str:
    """Request k's image: a 224 x 224 PNG of one colour, as a data URL."""
    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (224, 224), (37 * k % 256, 91 * k % 256, 53 * k % 256)).save(
        out, "PNG"
    )
    return "data:image/png;base64," + base64.b64encode(out.getvalue()).decode()


def bodies(model: str, n: int, prefix_words: int = 0, images: int = 0) -> list[dict]:
    """Request k: `prefix_words` words every request shares, then about
    48 + 31k words starting with k; 16 to 96 output tokens. With `images`, a
    chat message whose text follows that many copies of `image(k)`."""
    prefix = "".join(f"{WORDS[i % len(WORDS)]} " for i in range(prefix_words))
    out = []
    for k in range(n):
        body = {
            "model": model,
            "prompt": f"{prefix}{k} "
            + " ".join(WORDS[(k + i) % len(WORDS)] for i in range(48 + 31 * k)),
            "max_tokens": 16 + 16 * (k % 6),
            "ignore_eos": True,
            "temperature": 0,
            "stream": True,
        }
        if images:
            part = {"type": "image_url", "image_url": {"url": image(k)}}
            text = {"type": "text", "text": body.pop("prompt")}
            body["messages"] = [{"role": "user", "content": [part] * images + [text]}]
        out.append(body)
    return out


def post(port: int, body: dict, headers: dict):
    """The streamed response's lines, once the server has admitted the request
    (its headers follow the engine's `add_request`); raises unless it answers 200."""
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=WALL_S)
    headers = {"content-type": "application/json", **headers}
    path = "/v1/chat/completions" if "messages" in body else "/v1/completions"
    conn.request("POST", path, json.dumps(body), headers)
    response = conn.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:500]!r}")
    return (raw.decode().rstrip("\r\n") for raw in response)


def real(
    port: int,
    requests: list[dict],
    gap_s: float,
    record: Path,
    wait: bool = True,
    prefix_words: int = 0,
    images: int = 0,
    warmup: bool = False,
) -> None:
    """Send request k at `gap_s` * k, once k - 1 is admitted when `wait`, read
    each to its end, write `ARRIVALS` with the requests' `prefix_words` and
    `images`. With `warmup`, request 0 is read to its end alone and the gaps
    count from then, so the rest arrive into running batches rather than all
    during the first forward.
    Through a prefill-decode router a request is admitted only once its
    prefill returns, so `wait` keeps prefills apart."""
    sent, start = [0] * len(requests), time.monotonic()
    admitted = [threading.Event() for _ in requests]
    admitted_ns = [0] * len(requests)

    def one(k: int) -> None:
        time.sleep(max(0.0, start + k * gap_s - time.monotonic()))
        try:
            if k and wait:
                admitted[k - 1].wait()
            sent[k] = time.monotonic_ns()
            lines = post(port, requests[k], {})
            admitted_ns[k] = time.monotonic_ns()
        finally:
            admitted[k].set()
        if DONE not in lines:
            raise RuntimeError(f"request {k} ended before {DONE}")

    first = 0
    if warmup:
        one(0)
        first, start = 1, time.monotonic() - gap_s
    with ThreadPoolExecutor(len(requests)) as pool:
        list(pool.map(one, range(first, len(requests))))
    arrivals = {
        "sent_ns": sent,
        "admitted_ns": admitted_ns,
        "prefix_words": prefix_words,
        "images": images,
    }
    (record / ARRIVALS).write_text(json.dumps(arrivals))


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
        prefix, images = (int(rest[i]) if rest[i:] else 0 for i in (2, 3))
        requests = bodies(model, n, prefix, images)
        real(
            int(port),
            requests,
            gap_s,
            record,
            prefix_words=prefix,
            images=images,
            warmup=True,
        )
    else:
        arrivals = json.loads((record / ARRIVALS).read_text())
        sent, prefix = arrivals["sent_ns"], arrivals.get("prefix_words", 0)
        requests = bodies(model, len(sent), prefix, arrivals.get("images", 0))
        offsets = [(t - min(sent)) / 1e9 for t in sent]
        Traffic(run, int(port), requests, offsets).run()
