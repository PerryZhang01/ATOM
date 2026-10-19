# SPDX-License-Identifier: MIT
"""Fit the two carriers of a PP send, RCCL and gloo, on two GPUs of one node.

    HIP_VISIBLE_DEVICES=a,b torchrun --nproc-per-node 2 scripts/compass/pp_carriers.py OUT
    python3 scripts/compass/pp_carriers.py OUT RECORD...

Rank 0 sends one message per size to rank 1 with `torch.distributed.isend`, a
device tensor on an RCCL group and a CPU tensor on a gloo group, as ATOM's PP
send does; rank 1 posts its receive before the send, or after running for
`LATE` seconds. Each side stamps its completion on the monotonic clock both processes
share. An RCCL send is queued behind device work, as ATOM queues it behind
the forward, and is timed from that work's end; RCCL stamps are device
events polled to completion, so no host call or synchronize is in them. A
size is eager when its sender completes before a late receive is posted,
its threshold the largest eager size, 0 when there is none. With the
receive posted first, rendezvous sizes fit ``rendezvous_fixed_s + bytes x
per_byte_s`` from the send, weighted by relative error; eager sizes, or every
size when none is eager, take that ``per_byte_s`` and fit ``overhead_s`` on the
sender and ``latency_s`` from its completion to the arrival.

A gloo sender blocked in its wait completes later the longer it was blocked
before the receive was posted. Rank 1 then runs for each of `BLOCKS` seconds
before it posts, rank 0 waiting on an 8-byte gloo send, as ATOM's next stage
runs its own forward while the stage before waits on the send's size. The
sender's completion past the rendezvous price fits ``wake_s x (1 - exp(-b /
wake_rise_s))`` at block ``b``. OUT is JSON: each
carrier's fields, the model's error on the arrival in seconds per posting order (min, p10,
p50, p90, max), every sample, and this file's text. Given RECORDs this script
wrote, it fits their samples pooled instead of probing, and OUT names their
sha256 in place of the samples: one probe's wake moves with the run. Its
`band` is each field's p10 and p90 over the records, each fitted alone.
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from atom.compass.runner.pp_send import CarrierModel, recv_done, woken

LATE, WARM, REPS = 0.05, 3, 20
BLOCKS = (0.002, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
SPIN = 10**6  # device cycles the RCCL send waits behind, well past its host call
SIZES = {"rccl": [2**k for k in range(3, 30)], "gloo": [2**k for k in range(3, 25)]}
#: The stamps of one sample: rank 0's send and completion, rank 1's post and end.
STAMPS = ("send", "done", "post", "end")


def fit_wake(samples: list[dict], price: float) -> tuple[float, float]:
    """``(wake_s, wake_rise_s)`` from `samples` with a ``block``, each sender's
    completion less the post and `price`, the median per block, fitted in
    microseconds: in seconds the rise barely moves from its first guess."""
    from scipy.optimize import curve_fit

    by = {}
    for s in samples:
        if s.get("block") is not None:
            by.setdefault(s["block"], []).append(s["done"] - s["post"] - price)
    b = np.array(sorted(by))
    y = np.array([np.median(by[k]) for k in b]) * 1e6
    (w, rise), _ = curve_fit(
        lambda b, w, r: w * -np.expm1(-b / r),
        b,
        y,
        p0=(max(y[-1], 1.0), np.median(b)),
        bounds=(1e-12, np.inf),
        x_scale="jac",
    )
    return float(w) * 1e-6, float(rise)


def _eager(samples: list[dict]) -> bool:
    """Whether a late-order sender completes before the post, at the median."""
    d, p = (np.median([s[k] - s["send"] for s in samples]) for k in ("done", "post"))
    return d < p


def threshold(sizes: list[int], eager):
    """`sizes`, then a bisection past the largest `eager` one to 1/32 of it."""
    lo = 0
    for n in sizes:
        yield n
        lo = n if eager(n) else lo
    hi, step = 2 * lo or sizes[0], max(lo // 32, 1)
    while hi - lo > step:
        yield (mid := (lo + hi) // 2)
        lo, hi = (mid, hi) if eager(mid) else (lo, mid)


def fit(samples: list[dict]) -> dict:
    """Each carrier's fields and residual spread from `samples`; gloo's wake
    when any sample has a ``block``."""
    out, got = {}, {}
    blocked = [s for s in samples if s.get("block") is not None]
    for s in samples:
        if s.get("block") is None:
            got.setdefault((s["carrier"], s["bytes"], s["late"]), []).append(s)
    for c, sizes in SIZES.items():
        med = {
            (n, late): {
                k: np.median([s[k] - s["send"] for s in got[c, n, late]])
                for k in STAMPS
            }
            for n in sizes
            for late in (False, True)
        }
        eager = [n for n in sizes if med[n, True]["done"] < med[n, True]["post"]]
        late = sorted(n for k, n, lt in got if k == c and lt)
        seen = [n for n in late if _eager(got[c, n, True])]
        if seen != late[: len(seen)] or eager == sizes:
            raise SystemExit(
                f"{c}: eager late sizes {seen} are not a prefix below {sizes[-1]}"
            )
        rdv = sizes[len(eager) :]
        early = {n: med[n, False] for n in sizes}
        t = np.array([early[n]["end"] for n in rdv])
        g, fixed = np.polyfit(rdv, t, 1, w=1 / t)
        # With no eager size, o and L are the sender's completion and the
        # arrival after it at every size, the receive posted first.
        base = eager or sizes
        o = np.median([early[n]["done"] - n * g for n in base])
        lat = np.median([early[n]["end"] - early[n]["done"] for n in base])
        out[c] = {
            "eager_threshold_bytes": seen[-1] if seen else 0,
            "latency_s": float(lat),
            "overhead_s": float(o),
            "per_byte_s": float(g),
            "rendezvous_fixed_s": float(fixed),
        }
        priced = {c: CarrierModel(**out[c])}
        wake = (0.0, 1.0)
        if c == "gloo" and blocked:
            wake = fit_wake(blocked, fixed + 8 * g)
            out[c]["wake_s"], out[c]["wake_rise_s"] = wake
        # The model's receive end less the measured, in seconds, per posting order.
        for late in (False, True):
            m = [med[n, late] for n in sizes]
            model = [
                recv_done(priced, 0.0, s["post"], [(c, n)])[1]
                + (0.0 if n in eager else woken(wake, s["post"]))
                for n, s in zip(sizes, m)
            ]
            err = [p - s["end"] for p, s in zip(model, m)]
            spread = np.percentile(err, (0, 10, 50, 90, 100))
            out[c]["late" if late else "early"] = [float(p) for p in spread]
    return out


def _event():
    import torch

    event = torch.cuda.Event()
    event.record()
    return event


def _polled(event) -> float:
    """When `event` is first seen complete, on the monotonic clock."""
    while not event.query():
        pass
    return time.monotonic()


def _run(seconds: float) -> None:
    """Keep the host busy for `seconds`, as a stage running its forward."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        pass


def probe() -> list[dict]:
    """This rank's stamps over every carrier, size, order and repeat, then
    over every block of an 8-byte gloo send."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    groups = {
        "rccl": dist.new_group([0, 1], backend="nccl"),
        "gloo": dist.new_group([0, 1], backend="gloo"),
    }
    stamps = []
    for c, sizes in SIZES.items():
        dev = "cuda" if c == "rccl" else "cpu"

        def eager(n: int, c: str = c) -> bool:
            both, mine = [None, None], [s for s in stamps if s[:3] == (c, n, True)]
            dist.all_gather_object(both, mine, group=groups["gloo"])
            return _eager([dict(zip(STAMPS, a[4:] + b[4:])) for a, b in zip(*both)])

        for late in (False, True):
            for n in threshold(sizes, eager) if late else sizes:
                x = torch.ones(n, dtype=torch.uint8, device=dev)
                for i in range(WARM + REPS):
                    dist.barrier(groups["gloo"])
                    if rank == 0:
                        time.sleep(0 if late else 0.002)  # an early receive is up first
                        t0 = time.monotonic()
                        if dev == "cuda":  # queued behind device work, as ATOM's send
                            torch.cuda._sleep(SPIN)
                            ready = _event()
                        dist.isend(x, 1, group=groups[c]).wait()
                    else:
                        _run(LATE if late else 0)
                        t0 = time.monotonic()
                        dist.irecv(x, 0, group=groups[c]).wait()
                    if dev == "cuda":
                        end = _event()
                        if rank == 0:
                            t0 = _polled(ready)
                        t1 = _polled(end)
                    else:
                        t1 = time.monotonic()
                    if i >= WARM:
                        stamps.append((c, n, late, None, t0, t1))
    x = torch.ones(8, dtype=torch.uint8)
    for i in range(WARM + REPS):  # round robin, so a drift reaches every block
        for block in BLOCKS:
            dist.barrier(groups["gloo"])
            if rank == 0:
                t0 = time.monotonic()
                dist.isend(x, 1, group=groups["gloo"]).wait()
            else:
                _run(block)
                t0 = time.monotonic()
                dist.irecv(x, 0, group=groups["gloo"]).wait()
            if i >= WARM:
                stamps.append(("gloo", 8, True, block, t0, time.monotonic()))
    dist.barrier(groups["gloo"])
    both = [None, None]
    dist.all_gather_object(both, stamps, group=groups["gloo"])
    dist.destroy_process_group()
    if rank:
        return []
    return [
        {
            "carrier": c,
            "bytes": n,
            "late": late,
            "block": b,
            "send": s,
            "done": d,
            "post": p,
            "end": e,
        }
        for (c, n, late, b, s, d), (*_, p, e) in zip(*both)
    ]


def main(out: str, *records: str) -> int:
    if records:
        read = [Path(p).read_bytes() for p in records]
        samples = [s for b in read for s in json.loads(b)["samples"]]
        about = {"records": [hashlib.sha256(b).hexdigest() for b in read]}
        # Each field's p10 and p90 over the records, each fitted alone.
        alone = [fit(json.loads(b)["samples"]) for b in read]
        about["band"] = {
            c: {
                k: [float(np.percentile([a[c][k] for a in alone], q)) for q in (10, 90)]
                for k, v in alone[0][c].items()
                if not isinstance(v, list)
            }
            for c in SIZES
        }
    else:
        samples = probe()
        if not samples:
            return 0
        import torch

        about = {
            "gpus": os.environ.get("HIP_VISIBLE_DEVICES"),
            "torch": torch.__version__,
            "rccl": ".".join(map(str, torch.cuda.nccl.version())),
            "samples": samples,
        }
    carriers = fit(samples)
    print(json.dumps(carriers, indent=2))
    record = {"carriers": carriers, **about, "source": Path(__file__).read_text()}
    Path(out).write_text(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
