# SPDX-License-Identifier: MIT
"""Fit the API server event loop's charge per HTTP request from one real burst.

    python3 scripts/compass/http_intake.py serve STAMPS API_SERVER_ARGS...
    python3 scripts/compass/http_intake.py fit STAMPS [N]

`serve` runs `atom.entrypoints.openai.api_server` in this process, as
``python3 -m`` does, and appends one line to STAMPS for each request body read
off the socket (starlette's `Request.body`; a later read of the cached body is
not stamped) and each chat template rendered (`apply_chat_template`), on the
monotonic clock:

    <end ns> <thread id> body <start ns> <bytes>
    <end ns> <thread id> tmpl <start ns> <length>

`cctraces_sim.sh` runs a real server this way when HTTP_INTAKE names STAMPS.
The burst is N first-turn requests sent at once: concurrency N, every
trajectory at its first turn.

`fit` reads the burst from STAMPS: from the first body read's start, the first
N template starts, and the first N bodies in the order their reads ended,
which is the order their templates start in. The loop takes the burst in one
request at a time, so the gap from one template start to the next is the next
request's intake, ``fixed_s + (its body bytes) x per_byte_s``. A few gaps also
hold the loop's other work, tens of milliseconds whatever the request's size,
so `per_byte_s` is the median slope over every pair of gaps (Theil-Sen), and
`fixed_s` makes the fitted gaps sum to the measured ones, so the burst is taken
in over its real span. A least-squares fit of each start to the running sums of
both terms leaves the per-byte term free to swing sign from burst to burst, as
the two sums rise together. Prints both terms as JSON, with the rms residual of
a gap in seconds, N, the burst's body bytes and STAMPS's sha256. N defaults to
256.
"""

import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import runpy
import sys
import threading
import time
from pathlib import Path

import numpy as np


def serve(stamps: str, args: list[str]) -> None:
    lock = threading.Lock()

    def stamp(kind: str, start: int, size: int) -> None:
        line = f"{time.monotonic_ns()} {threading.get_ident()} {kind} {start} {size}\n"
        with lock:
            out.write(line)

    def requests(module) -> None:
        read = module.Request.body

        async def body(self):
            cached, start = hasattr(self, "_body"), time.monotonic_ns()
            got = await read(self)
            if not cached:
                stamp("body", start, len(got))
            return got

        module.Request.body = body

    def encoders(module) -> None:
        render = module.apply_chat_template

        def apply_chat_template(*a, **k):
            start = time.monotonic_ns()
            got = render(*a, **k)
            stamp("tmpl", start, len(got))
            return got

        module.apply_chat_template = apply_chat_template

    # Each module is patched once it has run, before anything imports from it.
    patches = {
        "starlette.requests": requests,
        "atom.entrypoints.openai.chat_encoders": encoders,
    }

    class Patcher(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name not in patches:
                return None
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            run = spec.loader.exec_module

            def exec_module(module):
                run(module)
                patches[name](module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, Patcher())
    sys.argv[1:], sys.path[0] = args, os.getcwd()
    with open(stamps, "a", buffering=1) as out:
        runpy.run_module(
            "atom.entrypoints.openai.api_server", run_name="__main__", alter_sys=True
        )


def fit(stamps: str, n: int = 256) -> dict:
    """`host.http_intake` from the burst in STAMPS, as the docstring says."""
    data = Path(stamps).read_bytes()
    rows = [line.split() for line in data.decode().splitlines()]
    reads = sorted((int(r[0]), int(r[3]), int(r[4])) for r in rows if r[2] == "body")
    t0 = min(start for _, start, _ in reads)
    sizes = [size for *_, size in reads[:n]]
    starts = sorted(int(r[3]) for r in rows if r[2] == "tmpl" and int(r[3]) >= t0)
    starts = starts[:n]
    if min(len(sizes), len(starts)) < n:
        raise SystemExit(
            f"{stamps}: {len(sizes)} bodies and {len(starts)} templates, not {n}"
        )
    gaps, nbytes = np.diff(starts) / 1e9, np.array(sizes[1:])
    i, j = np.triu_indices(len(gaps), 1)
    pair = nbytes[i] != nbytes[j]
    per_byte = np.median((gaps[j] - gaps[i])[pair] / (nbytes[j] - nbytes[i])[pair])
    fixed = gaps.mean() - per_byte * nbytes.mean()
    return {
        "fixed_s": float(fixed),
        "per_byte_s": float(per_byte),
        "rms_s": float(np.sqrt(np.mean((fixed + per_byte * nbytes - gaps) ** 2))),
        "requests": n,
        "bytes": sum(sizes),
        "record": f"sha256:{hashlib.sha256(data).hexdigest()}",
    }


if __name__ == "__main__":
    verb, stamps, *rest = sys.argv[1:]
    if verb == "serve":
        serve(stamps, rest)
    else:
        print(json.dumps(fit(stamps, *map(int, rest))))
