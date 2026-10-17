# SPDX-License-Identifier: MIT
"""Fit the two carriers of a PP send, RCCL and gloo, on two GPUs of one node.

    HIP_VISIBLE_DEVICES=a,b torchrun --nproc-per-node 2 scripts/compass/pp_carriers.py OUT

Rank 0 sends one message per size to rank 1 with `torch.distributed.isend`, a
device tensor on an RCCL group and a CPU tensor on a gloo group, as ATOM's PP
send does; rank 1 posts its receive before the send, or `LATE` seconds after
it. Each side stamps its completion on the monotonic clock both processes
share. A size is eager when its sender completes before a late receive is
posted, its threshold the largest eager size, 0 when there is none. With the
receive posted first, rendezvous sizes fit ``rendezvous_fixed_s + bytes x
per_byte_s`` from the send, weighted by relative error; eager sizes, or every
size when none is eager, take that ``per_byte_s`` and fit ``overhead_s`` on the
sender and ``latency_s`` from its completion to the arrival. OUT is JSON: each
carrier's fields, the model's error on the arrival in seconds per posting order (min, p10,
p50, p90, max), every sample, and this file's text.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

LATE, WARM, REPS = 0.05, 3, 20
SIZES = {"rccl": [2**k for k in range(3, 30)], "gloo": [2**k for k in range(3, 25)]}
#: The stamps of one sample: rank 0's send and completion, rank 1's post and end.
STAMPS = ("send", "done", "post", "end")


def fit(samples: list[dict]) -> dict:
    """Each carrier's five fields and residual spread from `samples`."""
    out, got = {}, {}
    for s in samples:
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
        eager = [n for n in sizes if med[n, True]["done"] < LATE]
        if eager != sizes[: len(eager)] or len(eager) == len(sizes):
            raise SystemExit(f"{c}: eager sizes {eager} are not a proper prefix")
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
            "eager_threshold_bytes": eager[-1] if eager else 0,
            "latency_s": float(lat),
            "overhead_s": float(o),
            "per_byte_s": float(g),
            "rendezvous_fixed_s": float(fixed),
        }
        # The model's receive end less the measured, in seconds, per posting order.
        for late in (False, True):
            m = [med[n, late] for n in sizes]
            model = [
                (
                    max(s["post"], o + n * g + lat)
                    if n in eager
                    else max(0.0, s["post"]) + fixed + n * g
                )
                for n, s in zip(sizes, m)
            ]
            err = [p - s["end"] for p, s in zip(model, m)]
            spread = np.percentile(err, (0, 10, 50, 90, 100))
            out[c]["late" if late else "early"] = [float(p) for p in spread]
    return out


def probe() -> list[dict]:
    """This rank's stamps over every carrier, size, order and repeat."""
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
        for late in (False, True):
            for n in sizes:
                x = torch.ones(n, dtype=torch.uint8, device=dev)
                for i in range(WARM + REPS):
                    dist.barrier(groups["gloo"])
                    if rank == 0:
                        time.sleep(0 if late else 0.002)  # an early receive is up first
                        t0 = time.monotonic()
                        dist.isend(x, 1, group=groups[c]).wait()
                    else:
                        time.sleep(LATE if late else 0)
                        t0 = time.monotonic()
                        dist.irecv(x, 0, group=groups[c]).wait()
                    torch.cuda.synchronize()
                    if i >= WARM:
                        stamps.append((c, n, late, t0, time.monotonic()))
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
            "send": s,
            "done": d,
            "post": p,
            "end": e,
        }
        for (c, n, late, s, d), (*_, p, e) in zip(*both)
    ]


def main(out: str) -> int:
    samples = probe()
    if not samples:
        return 0
    import torch

    carriers = fit(samples)
    print(json.dumps(carriers, indent=2))
    record = {
        "carriers": carriers,
        "gpus": os.environ.get("HIP_VISIBLE_DEVICES"),
        "torch": torch.__version__,
        "rccl": ".".join(map(str, torch.cuda.nccl.version())),
        "samples": samples,
        "source": Path(__file__).read_text(),
    }
    Path(out).write_text(json.dumps(record) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
