# SPDX-License-Identifier: MIT
"""The acceptance checks every simulated deployment's runs are held to.

    python -m atom.compass.detect.scenario OUT_DIR [OUT_DIR ...] [--record LP=DIR ...]

Each OUT_DIR holds one run of one run file, each under its own
`PYTHONHASHSEED`, with the step table and run summary the authority wrote.
`check_run` returns one line per check that fails, and none when all hold:

- each run reached its finish with no refusal: every LP's last row ends at
  ``inf``, the summary counts no refusal and is no coverage report;
- every run's step table is the first's, byte for byte;
- each channel's messages were released to its receiver exactly once, seqs 0
  to n-1, and none before the receiver's clock: a release row runs from the
  receiver's clock as the grant found it to the message's arrival;
- with `sent`, the max_tokens of each request the traffic LP sent, and
  `taken`, the SSE data lines it took: each request has its ``[DONE]`` and a
  usage chunk counting every token it asked for;
- with `records`, an engine LP's name to the parity record its run wrote
  (`atom.compass.parity`): the LP's TAR rows are `expected_timeline`'s.
- above pipeline width one, the order of the stages' forwards and sends: a
  stage starts a forward no earlier than its input is released to it, and
  sends on no earlier than that forward starts and its previous send is done
  (`pipeline_timeline`, which recomputes no time).

The CLI checks the runs and the records it is given; the traffic's view is a
harness's, so `sent` is for callers that hold one.
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import TAR
from atom.compass.detect.determinism import CONFIGURATION_PREFIX, compare_step_tables
from atom.compass.parity import read
from atom.compass.run import (
    RAPIDSERVE,
    STEP_TABLE_FILE,
    SUMMARY_FILE,
    channel_table,
    entries,
)

DONE = "data: [DONE]"


def expected_timeline(
    record, prefill_s: float, decode_s: float, tars, sync: bool = False
) -> list:
    """One engine LP's TAR rows, ``(from, to)``, under a constant step stub.

    `record` maps each DP rank to its steps (`atom.compass.parity.read`).
    `tars` is the LP's TAR rows as the step table has them, read only for the
    clock each step starts at; everything after the start is computed here.

    A forward's device seconds are `prefill_s` when a row prefills plus
    `decode_s` when a row decodes. With no host laws it enters at its start or
    once the device work two forwards back ends, whichever is later, and
    returns then, or, when it produces output, once the previous output
    forward's device work ends. Above DP width one the group's device end is
    the latest over ranks, which each rank takes as its last device end, and
    as its output end on an output forward; each rank returns at its own exit,
    and the step is one TAR per distinct return, ascending. With `sync`, every
    forward returns at the group's device end, as one that synchronizes its
    stream before it returns does.

    Raises ValueError when the ranks hold different step counts, or a step
    starts after the TAR rows end.
    """
    ranks = sorted(record)
    counts = {r: len(record[r]) for r in ranks}
    if len(set(counts.values())) > 1:
        raise ValueError(f"the record's ranks hold {counts} steps")
    # Per rank: device end of forwards n-2 and n-1, and of the last output one.
    device = {r: (-math.inf, -math.inf, -math.inf) for r in ranks}
    want, i = [], 0
    for k in range(len(record[ranks[0]])):
        if i >= len(tars):
            raise ValueError(f"record step {k} has no TAR row")
        now = tars[i][0]
        returns, ends = set(), {}
        for r in ranks:
            step = record[r][k]
            rows = [row[-1] for row in step["rows"]]
            busy = prefill_s * (not all(rows)) + decode_s * any(rows)
            two_back, last, output = device[r]
            host = max(now, two_back)
            ends[r] = max(last, host) + busy
            returns.add((max(host, output) if step["produces_output"] else host) - now)
        end = max(ends.values())
        if sync:
            returns = {end - now}
        for r in ranks:
            _, last, output = device[r]
            device[r] = (last, end, end if record[r][k]["produces_output"] else output)
        start = now
        for t in sorted(returns):
            want.append((start, now + t))
            start, i = now + t, i + 1
    return want


def _run_properties(table: str, summary: dict) -> list[str]:
    lines = table.splitlines()
    channels = channel_table(json.loads(lines[0].removeprefix(CONFIGURATION_PREFIX)))
    rows = [line.split() for line in lines[1:]]
    failed = []
    last = {r[0]: r for r in rows}  # each LP's last row
    for lp in sorted(lp.name for lp in channels.registry.ids()):
        if lp not in last or last[lp][2] != "inf":
            failed.append(
                f"{lp} did not reach the finish: its last row is {last.get(lp)}"
            )
    refusals = summary["schedule"]["refusals"]
    if refusals["count"] or summary["coverage_report"]:
        failed.append(f"the run refused: {refusals['reasons']}")
    seqs = defaultdict(list)
    for lp, clock, arrival, event, ch, seq, _ in rows:
        if event != "release":
            continue
        seqs[ch].append(int(seq))
        target = channels.channel(ch).target.name
        if lp != target:
            failed.append(f"{ch} seq {seq} was released to {lp}, not to {target}")
        if float(arrival) < float(clock):
            failed.append(
                f"{ch} seq {seq} arrived at {arrival}, behind {lp}'s clock at {clock}"
            )
    for ch, got in sorted(seqs.items()):
        if sorted(got) != list(range(len(got))):
            failed.append(f"{ch} released seqs {sorted(got)}, not each of 0..n-1 once")
    return failed


def pipeline_timeline(table: str) -> list[str]:
    """Each stage's forward n against the stage before's send n, one failure
    a line; none at pipeline width one. It checks order on the step table's
    rows and recomputes no time: a receive or a forward of the wrong length
    passes.

    The head's forward n starts when it sends ``meta#pp1`` seq n. A stage's
    send n leaves no earlier than its forward n starts and its send n-1
    completes, read from the ``pp_ack`` the stage after returned. The stage
    after takes its batch n on ``meta#pp<k>`` and then receives its
    ``pp_data`` n, so it starts its forward n where the later of the two is
    released, or, when that falls inside one of its TARs, at that TAR's end;
    the forward's own waits for the metadata and the tensors come after.
    """
    lines = table.splitlines()
    run = json.loads(lines[0].removeprefix(CONFIGURATION_PREFIX))
    pp, channels = run.get("pipeline_parallel_size", 1), channel_table(run)
    tars, arrived = defaultdict(list), defaultdict(dict)
    for lp, t0, t1, event, ch, seq, _ in map(str.split, lines[1:]):
        if event == TAR:
            tars[lp].append((float(t0), float(t1)))
        elif event == "release":
            arrived[ch][int(seq)] = float(t1)
    meta = "engine-pp0->engine-pp1:meta#pp1"
    starts = {n: a - channels.lookahead(meta) for n, a in arrived[meta].items()}
    failed = []
    for k in range(pp):
        lp, data_ch = f"engine-pp{k}", f"engine-pp{k}->engine-pp{k + 1}:pp_data#dp0"
        data = arrived[data_ch]
        ack = arrived[f"engine-pp{k + 1}->engine-pp{k}:pp_ack#dp0"]
        for n, start in sorted(starts.items()):
            span = (
                b for a, b in tars[lp] if a < start < b and not math.isclose(a, start)
            )
            start = next(span, start)
            if not any(math.isclose(a, start) for a, _ in tars[lp]):
                failed.append(f"{lp} starts no forward {n} at {start}")
            elif k < pp - 1 and n not in data:
                failed.append(f"{lp} never sent forward {n}'s output on {data_ch}")
            elif k < pp - 1:
                least = max(start, ack.get(n - 1, -math.inf))
                sent = data[n] - channels.lookahead(data_ch)
                if sent < least and not math.isclose(sent, least):
                    failed.append(f"{lp} sent forward {n} at {sent}, before {least}")
        batch = arrived[f"engine-pp0->engine-pp{k + 1}:meta#pp{k + 1}"]
        starts = {n: max(a, batch.get(n, a)) for n, a in data.items()}
    return failed


def _outputs(sent, taken) -> list[str]:
    chunks = [json.loads(line[len("data: ") :]) for line in taken if line != DONE]
    tokens = sorted(c["usage"]["completion_tokens"] for c in chunks if "usage" in c)
    done = sum(line == DONE for line in taken)
    if done == len(sent) and tokens == sorted(sent):
        return []
    took = f"took {done} [DONE] and usage counts {tokens}"
    return [f"the traffic sent requests for {sorted(sent)} tokens and {took}"]


def _timeline(table: str, lp: str, record) -> list[str]:
    lines = table.splitlines()
    run = json.loads(lines[0].removeprefix(CONFIGURATION_PREFIX))
    stub = Coefficients(**run.get("step_stub", {}))
    if not stub.shape_blind or any(h.get("laws") for h in entries(run, "host")):
        return [
            f"{lp}: a timeline is computed for a constant step stub with no host laws"
        ]
    tars = [
        (float(f[1]), float(f[2]))
        for f in map(str.split, lines[1:])
        if f[0] == lp and f[3] == TAR
    ]
    # RapidServe's prefill_forward synchronizes its stream before it returns.
    sync = run.get(RAPIDSERVE, False) and lp == "engine-P"
    try:
        want = expected_timeline(
            record, stub.prefill_step, stub.decode_step, tars, sync
        )
    except ValueError as e:
        return [f"{lp}: {e}"]
    for i, (w, got) in enumerate(zip_longest(want, tars)):
        if w is None or got is None or not all(map(math.isclose, w, got)):
            return [f"{lp}: TAR row {i} is {got}; the constant stub implies {w}"]
    return []


def check_run(out_dirs, sent=None, taken=(), records=None) -> list[str]:
    """Every failed check over the runs in `out_dirs`, one line each."""
    out_dirs = [Path(d) for d in out_dirs]
    tables = [(d / STEP_TABLE_FILE).read_text() for d in out_dirs]
    failed = []
    for d, table in zip(out_dirs, tables):
        summary = json.loads((d / SUMMARY_FILE).read_text())
        failed += [f"{d}: {f}" for f in _run_properties(table, summary)]
        failed += [f"{d}: {f}" for f in pipeline_timeline(table)]
    for d, table in zip(out_dirs[1:], tables[1:]):
        code, report = compare_step_tables(tables[0], table, str(out_dirs[0]), str(d))
        if code:
            failed.append(report)
    if sent is not None:
        failed += _outputs(sent, taken)
    for lp, directory in (records or {}).items():
        failed += _timeline(tables[0], lp, read(directory))
    return failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.detect.scenario", description=__doc__
    )
    parser.add_argument("out_dirs", nargs="+", help="each run's out_dir")
    parser.add_argument(
        "--record",
        action="append",
        default=[],
        metavar="LP=DIR",
        help="an engine LP and the parity record of the first run",
    )
    args = parser.parse_args(argv)
    records = dict(r.split("=", 1) for r in args.record)
    failed = check_run(args.out_dirs, records=records)
    for line in failed:
        print(f"scenario: FAIL {line}")
    if not failed:
        print(f"scenario: every check holds over {len(args.out_dirs)} run(s)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
