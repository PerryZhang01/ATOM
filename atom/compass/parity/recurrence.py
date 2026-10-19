# SPDX-License-Identifier: MIT
"""Gate the host/device recurrence on a real record's measurements.

    python -m atom.compass.parity.recurrence RECORD_DIR

`HostDeviceRecurrence` charges each forward with `d_busy_ns` as its device
time, plus `d_gap_ns` on a decode, and its line's host segments: `E` from the previous exit to the entry,
`W_pre` to the launch less the staging gate's and MTP status waits, `W_launch` to the
recv wait (the sampler's launch included), and `W_post` from the recv wait to the exit
less the draft copy wait; with
`d_target_ns`, under MTP, the rest of `d_busy_ns` is draft, `W_inputs` is the part of
`W_pre` after the MTP status wait, and no `d_gap_ns` is fed. Each forward is called at the
exit predicted for the one before, so only the waits and device times move it.
A pipeline record is gated step by step, each stage charged with the predicted
times of the others (`pipeline`), its sends priced on a machine spec's PP
carriers when one is given:

    python -m atom.compass.parity.recurrence RECORD_DIR MACHINE_JSON ROW_BYTES

`ROW_BYTES` is one token's `hidden_states` in bytes, the hidden size times
the model's dtype size.

Per kind (decode, output prefill, middle chunk, and idle: sharing no request
with the forward before), p10 and p90 of charge over measured exit-to-exit
period, less one, in percent, lie within `STEP_BAND`, the run total within
`TOTAL_BAND`.
A failing kind's steps outside the band are named with their residual, the
device work queued at their launch, and their recv release less the measured.
A sending stage names the steps where the flush it infers and the record's
`t_flush_ns` disagree.
"""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from atom.compass.parity import read
from atom.compass.runner.overrides import FORWARD_KINDS, HostDeviceRecurrence
from atom.compass.runner.pp_send import carrier_models, recv_done
from atom.compass.spec import MachineSpec

STEP_BAND, TOTAL_BAND, SHOWN = 2, 0.5, 10  # bands in percent
KINDS = ("decode", "prefill", "middle", "idle")
#: A send's gloo parts: the size, and aiter's metadata of `hidden_states` and
#: `residual` pickled, 179 bytes at bf16 from 256 to 65535 tokens.
# ponytail: one length for every batch; it moves by a few bytes, about a
# nanosecond on gloo, so pickle aiter's metadata here only if that changes.
META = [("gloo", 8), ("gloo", 179)]
#: A later stage flushes its last send once `recv_metadata` has waited this
#: long (`_downstream_busy_loop`'s poll); the head flushes whenever it waits.
IDLE_FLUSH_S = 0.1


def _kind(step, prev) -> str:
    # ponytail: a DP dummy shares no request, so it and the forward after it
    # count as idle; split them out once a DP record is gated.
    held = {r[0] for r in prev["batch"]} if prev else set()
    if not held & {r[0] for r in step["batch"]}:
        return "idle"
    if not step["produces_output"]:
        return "middle"
    return "decode" if all(d for *_, d in step["rows"]) else "prefill"


def _waited(steps: list[dict], n: int, pp: int):
    """The head's last forward before `n` whose tokens `n` is scheduled after,
    or None: `n - pp`, which leaves the in-flight queue only with its tokens,
    or a later one holding a request of `n`'s, every request of a batch with a
    decode in it, else the requests `n` decodes."""
    mine = {k: d for (k, *_), (*_, d) in zip(steps[n]["batch"], steps[n]["rows"])}
    for m in range(n - 1, max(n - pp, 0) - 1, -1):
        s, decode = steps[m], any(d for *_, d in steps[m]["rows"])
        held = any(k in mine and (decode or mine[k]) for k, *_ in s["batch"])
        if s["produces_output"] and (m == n - pp or held):
            return m
    return None


def pipeline(stages: list[list[dict]], carriers=None) -> list[dict]:
    """One DP rank's stages charged step by step, `check`'s report for each:
    the head's entry released at the last stage's exit of the forward whose
    tokens it waits for, each downstream entry at the head's call. With
    `carriers`, ``(models, row_bytes)``, each send rides the carriers
    (`_charged`)."""
    if len({len(s) for s in stages}) > 1:
        raise ValueError(f"stages of {[len(s) for s in stages]} steps")
    pp, head, tail = len(stages), stages[0], stages[-1]
    out, held = [[] for _ in stages], [[] for _ in stages]
    runs = [
        _charged(s, (i, pp), out[i - 1] if i else None, held[i], carriers, nxt)
        for i, (s, nxt) in enumerate(zip(stages, [*out[1:], None]))
    ]
    for n in range(len(head)):
        w = _waited(head, n, pp)
        tokens = w is not None and (tail[w]["t_exit_ns"], out[-1][w][3])
        held[0].append(tokens)
        out[0].append(next(runs[0]))
        call = n and (max(head[n - 1]["t_exit_ns"], tokens and tokens[0]), out[0][n][2])
        for i in range(1, pp):
            held[i].append(call)
            out[i].append(next(runs[i]))
    return [next(r) for r in runs]


def check(steps: list[dict]) -> dict:
    """One DP rank's steps charged on their measured inputs: the gate's report."""
    *_, report = _charged(steps, None, None, None)
    return report


def _charged(steps, stage, upstream, held, carriers=None, downstream=None):
    """`check`, yielding each step's predicted send, device end, call, exit and
    received tensors' end before the report. On pipeline stage `stage = (rank, pp)`, the launch
    after `W_recv` waits for the stage before's predicted send in `upstream[n]` and the
    device for its end, in place of the measured `t_pp_recv_ns` and
    `d_recv_ns`. With `carriers`, ``(models, row_bytes)``, the send's
    `hidden_states` and `residual` ride the carriers, posted at the prepare
    end plus `W_recv`: the launch's rest waits for the metadata's arrival and the
    device for the tensors', which leave once the previous frame's are done
    and are taken once this stage's device work and its own send's hold are.
    A stage's device starts no earlier than that hold, the tensors' end of
    its send before last in `downstream`, or of its last if it waited idle
    to flush it: the head for its tokens, a later stage `IDLE_FLUSH_S`.
    The report's `flush` holds a sending stage's steps that flushed so and
    those stamped `t_flush_ns`, None on the last stage.
    Step `n` is called no earlier than `held[n] = (measured, predicted)`'s
    `predicted`, its `E` from `measured` if after the last exit; a step
    entered before `measured` is refused."""
    now, prev, rows, done = 0.0, None, [], -math.inf
    flush = None if downstream is None else ([], [])  # steps inferred, stamped
    origin = steps[0]["t_enter_ns"]
    mtp = any("d_target_ns" in s for s in steps)
    rec = HostDeviceRecurrence({}, (), lambda: now, stage, mtp=mtp)
    for n, s in enumerate(steps):
        if "d_busy_ns" not in s:
            raise ValueError(f"step {s['step']} has no device window to charge")
        (g0, g1), (l0, l1) = s["t_gate_ns"], s["t_launch_ns"]
        r0, r1 = s.get("t_recv_ns", (l1, l1))
        m0, m1 = s.get("t_mtp_ns", (l0, l0))
        d0, d1 = s.get("t_draft_recv_ns", (0, 0))
        p0, p1 = s.get("t_pp_recv_ns", (l0, l0))
        last = prev["t_exit_ns"] if prev else s["t_enter_ns"]
        e, called = s["t_enter_ns"] - last, now
        if held and held[n]:  # E runs from the release, called at its prediction
            if (e := s["t_enter_ns"] - max(last, held[n][0])) < 0:
                raise ValueError(
                    f"stage {stage[0]} step {s['step']} enters {-e * 1e-6:.3f} ms "
                    "before the release it waits for"
                )
            called = max(now, (held[n][1] - origin) * 1e-9)
        pre = l0 - s["t_enter_ns"] - (g1 - g0) - (m1 - m0)
        post = s["t_exit_ns"] - r1 - (d1 - d0)
        entry = max(called + e * 1e-9, rec.ends[0])
        # The last send settled: the one before the last, or the last if flushed.
        flushed = called - now > (IDLE_FLUSH_S if stage and stage[0] else 0)
        settled = n - 2 + flushed
        if flush is not None:
            flush[0].extend([s["step"]] * flushed)
            flush[1].extend([s["step"]] * ("t_flush_ns" in s))
        hold = -math.inf
        if downstream and settled >= 0:
            hold = (downstream[settled][4] - origin) * 1e-9
        sent = ready = -math.inf
        if upstream:  # the stage before's send holds the launch
            sent, ready = ((t - origin) * 1e-9 for t in upstream[n][:2])
            if carriers:
                models, width = carriers
                posted = entry + (pre + p0 - l0) * 1e-9  # prepare end, then W_recv
                sent = recv_done(models, sent, posted, META)[1]
                tensors = [("rccl", sum(r[0] for r in s["rows"]) * width)] * 2
                posted = max(sent, rec.ends[-1], hold)
                done, ready = recv_done(models, max(ready, done), posted, tensors)
        launch = r0 - l0 - (p1 - p0)
        law = {"E": e, "W_pre": pre - (l0 - m1), "W_launch": launch, "W_post": post}
        law |= {"W_inputs": l0 - m1, "W_recv": p0 - l0}
        law = {g: (v * 1e-9, 0, 0) for g, v in law.items()}
        rec.laws = {k: law for k in FORWARD_KINDS}
        queued = rec.ends[1] - max(entry + pre * 1e-9, sent)
        # Under MTP, the wait for the previous output's MTP status sits inside
        # W_pre: released where it starts or where that output's target ends.
        head = (m0 - s["t_enter_ns"] - (g1 - g0)) * 1e-9
        due = max(entry + head, rec.output_end)
        batch = SimpleNamespace(total_tokens_num=0, total_seqs_num=0)
        batch.compass_pp_host = (called, sent)
        batch.total_seqs_num_prefill = sum(not d for *_, d in s["rows"])
        batch.produces_output = lambda out=s["produces_output"]: out
        gap = 0 if batch.total_seqs_num_prefill or mtp else s.get("d_gap_ns", 0)
        if stage is not None:
            # A stage's host may trail its device, so the gap may be idle: it
            # follows the last forward's device end. The tensors' arrival holds it too.
            start = rec.ends[-1] + gap * 1e-9
            batch.compass_pp_release = max(ready, start, hold)
            gap = 0
        busy = s["d_busy_ns"]
        target = s.get("d_target_ns", busy)
        own = gap + target - s.get("d_recv_ns", 0)  # the receive is a wait, not work
        charge = rec.charge(batch, own * 1e-9, (busy - target) * 1e-9)
        times = (now + charge - post * 1e-9, rec.ends[-1], called, now + charge, done)
        period = (s["t_exit_ns"] - last) * 1e-9
        r1_ms = None
        if "t_mtp_ns" in s:  # released before the launch
            r1_ms = (due - now - (m1 - last) * 1e-9) * 1e3
        elif "t_recv_ns" in s:
            r1_ms = (charge - (post + r1 - last) * 1e-9) * 1e3
        res, queued = (charge - period) * 1e3, max(queued, 0.0) * 1e3
        rows.append(
            (s["step"], _kind(s, prev), 100 * (charge / period - 1), res, queued, r1_ms)
        )
        now, prev = now + charge, s
        yield [origin + round(t * 1e9) if t > -math.inf else t for t in times]
    kinds = {}
    for k in KINDS:
        if err := [r[2] for r in rows if r[1] == k]:
            r1 = [r[5] for r in rows if r[1] == k and r[5] is not None]
            p10, p50, p90 = np.percentile(err, (10, 50, 90))
            ok = bool(-STEP_BAND <= p10 and p90 <= STEP_BAND)
            kinds[k] = (len(err), p10, p50, p90, np.median(r1) if r1 else None, ok)
    span = (steps[-1]["t_exit_ns"] - steps[0]["t_enter_ns"]) * 1e-9
    total = 100 * (now / span - 1)
    yield {
        "kinds": kinds,
        "total": total,
        "pass": abs(total) <= TOTAL_BAND and all(v[-1] for v in kinds.values()),
        "failures": [r for r in rows if not kinds[r[1]][-1] and abs(r[2]) > STEP_BAND],
        "flush": flush,
    }


def main(record: str, machine=None, row_bytes=None) -> int:
    """Print the gate per DP rank and pipeline stage of `record`; 1 if any fails."""
    carriers = None
    if machine:
        spec = MachineSpec.from_mapping(json.loads(Path(machine).read_text()))
        carriers = carrier_models(spec), int(row_bytes)
    ok, stages = True, [read(record)]
    while more := read(record, len(stages)):
        stages.append(more)
    pp, runs = len(stages), []
    for rank, steps in stages[0].items():
        reports = (
            pipeline([d[rank] for d in stages], carriers) if pp > 1 else [check(steps)]
        )
        runs += [(i, rank, report) for i, report in enumerate(reports)]
    for i, rank, report in runs:
        ok &= report["pass"]
        name = f"dp{rank} pp{i}" if pp > 1 else f"dp{rank}"
        print(f"{name} kind      n      p10      p50      p90  recv p50 ms  gate")
        for k, (n, p10, p50, p90, r1, good) in report["kinds"].items():
            r1 = "-" if r1 is None else f"{r1:.3f}"
            gate = "pass" if good else "FAIL"
            print(
                f"    {k:8} {n:6} {p10:+.2f}% {p50:+.2f}% {p90:+.2f}% {r1:>9}  {gate}"
            )
        gate = "pass" if abs(report["total"]) <= TOTAL_BAND else "FAIL"
        print(f"    total {report['total']:+.3f}% {gate}")
        if report["flush"] is not None and not report["flush"][1]:
            print("    flush: the record stamps no flush")
        elif report["flush"] is not None:
            inferred, stamped = map(set, report["flush"])
            off = sorted(inferred ^ stamped)
            print(
                f"    flush: {len(off)} steps disagree, {len(inferred)} inferred, "
                f"{len(stamped)} stamped"
            )
            for step in off[:SHOWN]:
                side = "inferred" if step in inferred else "stamped"
                print(f"      step {step}: {side} only")
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
    sys.exit(main(*sys.argv[1:]) if len(sys.argv) in (2, 4) else __doc__)
