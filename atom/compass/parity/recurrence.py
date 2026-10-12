# SPDX-License-Identifier: MIT
"""Gate the host/device recurrence on a real record's measurements.

    python -m atom.compass.parity.recurrence RECORD_DIR

`HostDeviceRecurrence` charges each forward with `d_busy_ns` as its device
time and its line's host segments: `E` from the previous exit to the entry,
`W_pre` to the launch less the staging gate's wait, `W_launch`, and `W_post`
from the launch to the exit less the recv wait. Each forward is called at the
exit predicted for the one before, so only the waits and device times move it.

Per kind (decode, output prefill, middle chunk, and idle: sharing no request
with the forward before), p10 and p90 of charge over measured exit-to-exit
period, less one, in percent, lie within `STEP_BAND`, the run total within
`TOTAL_BAND`.
A failing kind's steps outside the band are named with their residual, the
device work queued at their launch, and their recv release less the measured.
"""

import sys
from types import SimpleNamespace

import numpy as np

from atom.compass.parity import read
from atom.compass.runner.overrides import FORWARD_KINDS, HostDeviceRecurrence

STEP_BAND, TOTAL_BAND, SHOWN = 2, 0.5, 10  # bands in percent
KINDS = ("decode", "prefill", "middle", "idle")


def _kind(step, prev) -> str:
    # ponytail: a DP dummy shares no request, so it and the forward after it
    # count as idle; split them out once a DP record is gated.
    held = {r[0] for r in prev["batch"]} if prev else set()
    if not held & {r[0] for r in step["batch"]}:
        return "idle"
    if not step["produces_output"]:
        return "middle"
    return "decode" if all(d for *_, d in step["rows"]) else "prefill"


def check(steps: list[dict]) -> dict:
    """One DP rank's steps charged on their measured inputs: the gate's report."""
    now, prev, rows = 0.0, None, []
    rec = HostDeviceRecurrence({}, (), lambda: now)
    for s in steps:
        if "d_busy_ns" not in s:
            raise ValueError(f"step {s['step']} has no device window to charge")
        (g0, g1), (l0, l1) = s["t_gate_ns"], s["t_launch_ns"]
        r0, r1 = s.get("t_recv_ns", (l1, l1))
        last = prev["t_exit_ns"] if prev else s["t_enter_ns"]
        e, pre = s["t_enter_ns"] - last, l0 - s["t_enter_ns"] - (g1 - g0)
        post = s["t_exit_ns"] - l1 - (r1 - r0)
        law = {"E": e, "W_pre": pre, "W_launch": l1 - l0, "W_post": post}
        law = {g: (v * 1e-9, 0, 0) for g, v in law.items()}
        rec.laws = {k: law for k in FORWARD_KINDS}
        queued = rec.ends[1] - max(now + e * 1e-9, rec.ends[0]) - pre * 1e-9
        batch = SimpleNamespace(total_tokens_num=0, total_seqs_num=0)
        batch.total_seqs_num_prefill = sum(not d for *_, d in s["rows"])
        batch.produces_output = lambda out=s["produces_output"]: out
        charge = rec.charge(batch, s["d_busy_ns"] * 1e-9)
        period = (s["t_exit_ns"] - last) * 1e-9
        r1_ms = (charge - (post + r1 - last) * 1e-9) * 1e3 if "t_recv_ns" in s else None
        res, queued = (charge - period) * 1e3, max(queued, 0.0) * 1e3
        rows.append(
            (s["step"], _kind(s, prev), 100 * (charge / period - 1), res, queued, r1_ms)
        )
        now, prev = now + charge, s
    kinds = {}
    for k in KINDS:
        if err := [r[2] for r in rows if r[1] == k]:
            r1 = [r[5] for r in rows if r[1] == k and r[5] is not None]
            p10, p50, p90 = np.percentile(err, (10, 50, 90))
            ok = bool(-STEP_BAND <= p10 and p90 <= STEP_BAND)
            kinds[k] = (len(err), p10, p50, p90, np.median(r1) if r1 else None, ok)
    span = (steps[-1]["t_exit_ns"] - steps[0]["t_enter_ns"]) * 1e-9
    total = 100 * (now / span - 1)
    return {
        "kinds": kinds,
        "total": total,
        "pass": abs(total) <= TOTAL_BAND and all(v[-1] for v in kinds.values()),
        "failures": [r for r in rows if not kinds[r[1]][-1] and abs(r[2]) > STEP_BAND],
    }


def main(record: str) -> int:
    """Print the gate per DP rank of `record`; 1 if any rank fails."""
    ok = True
    for rank, steps in read(record).items():
        report = check(steps)
        ok &= report["pass"]
        print(f"dp{rank} kind      n      p10      p50      p90  recv p50 ms  gate")
        for k, (n, p10, p50, p90, r1, good) in report["kinds"].items():
            r1 = "-" if r1 is None else f"{r1:.3f}"
            gate = "pass" if good else "FAIL"
            print(
                f"    {k:8} {n:6} {p10:+.2f}% {p50:+.2f}% {p90:+.2f}% {r1:>9}  {gate}"
            )
        gate = "pass" if abs(report["total"]) <= TOTAL_BAND else "FAIL"
        print(f"    total {report['total']:+.3f}% {gate}")
        bad = sorted(report["failures"], key=lambda r: -abs(r[2]))
        for k in KINDS:
            if mine := [r for r in bad if r[1] == k]:
                print(f"    {k}: {len(mine)} steps outside the band, worst first")
            for step, _, err, res, queued, r1 in mine[:SHOWN]:
                r1 = "-" if r1 is None else f"{r1:+.3f}"
                print(
                    f"      step {step}: {err:+.2f}%, residual {res:+.3f} ms, "
                    f"queued {queued:.3f} ms, recv {r1} ms"
                )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]) if len(sys.argv) == 2 else __doc__)
