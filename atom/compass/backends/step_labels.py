# SPDX-License-Identifier: MIT
"""A real run's recorded steps, labelled one way for every reader.

A record is a real run's step record (`steps/dp<r>.jsonl`, written with
`ATOM_COMPASS_PARITY_RECORD`), read as JSON lines. `label` makes each forward
that is not a dummy run a `Sample`: its batch view, its device label, its host
labels and its provenance. `step_linear_fit` fits samples and `validate.cost`
prices them; neither reads a step's stamps.

A step's device label is its device busy time, `d_busy_ns`, plus for a decode
step the device work queued since the previous forward's exit, `d_gap_ns`. A
decode step whose device start precedes its host launch end is launch-exposed
and cut; a prefill step is never cut, so a host bubble inside it stays in its
label. On an MTP record, one with `d_target_ns`, the device label is the
target's, `d_target_ns` with no gap, and the draft label is the rest of
`d_busy_ns`. The exit of each wait for the previous output's token copy (`t_recv_ns`)
bounds the host clock's lead over the device's (that forward had finished by
then); the tightest of the last `ANCHORS` bounds is the lead. The copy itself
lands on its own stream after the exit event the bound reads, by 14 to 29 us
measured on one host (median 17 us), so each bound carries that much more lead
than the clocks actually have; on the step linear law's records the gap never
moves a decode step across the launch-exposed cut, so it is left in the bound
rather than corrected by a host-specific number with no record or law of its
own to carry it. A record without device windows is refused. A step mixing
prefill and decode rows is cut.
A step of an MTP record without `d_target_ns` is cut as `unsplit`: its device
label would hold a draft the law prices again.

A step's host labels are its segments: `E` from the previous forward's exit,
cut as idle when the two batches share no request; `W_pre` to the launch, less
the staging gate's wait; `W_launch` from the launch to the token copy wait,
the sampler's launch included, or to the launch's end on a forward with no
such wait; `W_post` from that wait's end to the exit, less the draft copy wait,
and a middle chunk has none. With an MTP status wait,
`W_pre` ends at it and `W_inputs` runs from it to the launch. On a later
pipeline stage, `W_recv` runs from the launch to the receive of its input.

`write_table` writes a record's labels beside it, `dp<r>.cost.jsonl`, for
people: a first line `{"version": TABLE_VERSION}`, then per step `step`,
`d_busy_s`, `d_gap_s`, `launch_exposed`, `idle`, the segments, `label_s`,
`draft_s` and `cut`, the reason its device label was cut; seconds throughout, null where
the step has none.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from atom.compass.backends.shape import BatchView, RequestShape

#: Token copy wait exits whose tightest bound is a step's host clock lead.
ANCHORS = 32
#: Why a device label is cut: a launch-exposed decode, mixed rows, no MTP target end.
CUTS = ("exposed", "mixed", "unsplit")
HOST_SEGMENTS = ("E", "W_pre", "W_launch", "W_post")
TABLE_VERSION = 1


@dataclass(frozen=True)
class Sample:
    """One real forward, labelled.

    `device_s` is its device label, None when cut, and `cut` names why (`CUTS`).
    `draft_s` is its draft label, None when cut or on a step with no draft.
    `host` holds the segments it has, in seconds: no `W_post` on a middle chunk,
    no `E` on the record's first forward or after an idle engine (`idle`).
    `provenance` holds `source`, `run` (the run's digest), `requests` (the
    request ids the step served) and `config` (the host key the record states,
    or None).
    """

    step: int
    rank: int
    view: BatchView
    device_s: float | None
    cut: str | None
    host: dict
    idle: bool
    provenance: dict
    draft_s: float | None = None


def digest(record) -> str:
    """The record's file name and the head of its bytes' SHA-256."""
    path = Path(record)
    return f"{path.name}:{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def host_kind(view: BatchView) -> str:
    """`decode`, `prefill`, or `middle` for a step that produces no output."""
    if not view.produces_output:
        return "middle"
    return "prefill" if view.prefill else "decode"


def _launch_exposed(lines) -> set[int]:
    """The steps whose device start precedes their host launch end."""
    device, output_end, timed = 0, None, []
    for s in lines:
        if "d_busy_ns" not in s:
            continue
        start = device + s.get("d_gap_ns", 0)
        device = start + s["d_busy_ns"]
        bound = None
        if "t_recv_ns" in s and output_end is not None:
            bound = s["t_recv_ns"][1] - output_end
        if s["produces_output"]:
            output_end = device
        timed.append((s, start, bound))
    bounds = [b for _, _, b in timed if b is not None]
    if not bounds:
        raise ValueError(
            "no token copy wait exit (`t_recv_ns`) ties the record's host and device clocks"
        )
    # Steps before the first anchor read the first ones.
    first, recent, exposed = min(bounds[:ANCHORS]), deque(maxlen=ANCHORS), set()
    for s, start, bound in timed:
        if bound is not None:
            recent.append(bound)
        if start + min(recent, default=first) < s["t_launch_ns"][1]:
            exposed.add(s["step"])
    return exposed


def _shared(step, other) -> bool:
    """Whether two steps' batches hold a request in common."""
    return bool({r[0] for r in step["batch"]} & {r[0] for r in other["batch"]})


def label(record, run: str | None = None) -> tuple[list[Sample], list[dict]]:
    """The record's samples, in step order, and its table's rows.

    `run` names the run in each sample's provenance, the record's `digest` by
    default.
    """
    path = Path(record)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    steps = [s for s in lines if not s["is_dummy_run"]]
    if not steps or "t_enter_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no timed steps; a fit needs a real run's record"
        )
    if "d_busy_ns" not in steps[0]:
        raise ValueError(
            f"{record} has no device windows (`d_busy_ns`); a fit labels a step "
            "by its device time"
        )
    exposed = _launch_exposed(lines)
    mtp = any("d_target_ns" in s for s in steps)
    key = path.with_suffix(".key.json")
    config = json.loads(key.read_text()) if key.exists() else None
    run = run or digest(record)
    samples, table, prev = [], [], None
    for s in lines:
        if s["is_dummy_run"]:
            prev = s
            continue
        view = BatchView(
            tuple(RequestShape(q, c, bool(d)) for q, c, d in s["rows"]),
            capture_rung=s["rung"],
            moe_rows=s.get("moe_rows"),
            produces_output=s["produces_output"],
        )
        cut = None
        if view.prefill and view.decode:
            cut = "mixed"
        elif not view.prefill and s["step"] in exposed:
            cut = "exposed"
        elif mtp and "d_target_ns" not in s:
            cut = "unsplit"
        gap = 0 if view.prefill else s.get("d_gap_ns", 0)
        device = None if cut else (gap + s["d_busy_ns"]) * 1e-9
        draft = None
        if "d_target_ns" in s:  # an MTP record: the target alone, no gap
            device = None if cut else s["d_target_ns"] * 1e-9
            draft = (s["d_busy_ns"] - s["d_target_ns"]) * 1e-9
        enter, (l0, l1) = s["t_enter_ns"], s["t_launch_ns"]
        g0, g1 = s.get("t_gate_ns", (l0, l0))
        r0, r1 = s.get("t_recv_ns", (l1, l1))
        m0, m1 = s.get("t_mtp_ns", (l0, l0))
        host = {"W_pre": g0 - enter + m0 - g1, "W_launch": r0 - l0}
        if "t_mtp_ns" in s:
            host["W_inputs"] = l0 - m1
        if "t_pp_recv_ns" in s:
            host["W_recv"] = s["t_pp_recv_ns"][0] - l0
        if host_kind(view) != "middle":
            d0, d1 = s.get("t_draft_recv_ns", (0, 0))
            host["W_post"] = s["t_exit_ns"] - r1 - (d1 - d0)
        idle = prev is not None and not _shared(s, prev)
        if prev is not None and not idle:
            host["E"] = enter - prev["t_exit_ns"]
        host = {k: v * 1e-9 for k, v in host.items()}
        provenance = {
            "source": "record",
            "run": run,
            "requests": [r[0] for r in s["batch"]],
            "config": config,
        }
        samples.append(
            Sample(
                s["step"],
                s["dp_rank"],
                view,
                device,
                cut,
                host,
                idle,
                provenance,
                None if cut else draft,
            )
        )
        gap_ns = s.get("d_gap_ns")
        table.append(
            {
                "step": s["step"],
                "d_busy_s": s["d_busy_ns"] * 1e-9,
                "d_gap_s": None if gap_ns is None else gap_ns * 1e-9,
                "launch_exposed": s["step"] in exposed,
                "idle": idle,
                **{k: host.get(k) for k in (*HOST_SEGMENTS, "W_inputs")},
                "label_s": device,
                "draft_s": draft,
                "cut": cut,
            }
        )
        prev = s
    return samples, table


def write_table(record, table: list[dict]) -> Path:
    """Write `table`, `label`'s rows for `record`, beside it; return its path."""
    path = Path(record).with_suffix(".cost.jsonl")
    lines = [{"version": TABLE_VERSION}, *table]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path
