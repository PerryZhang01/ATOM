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
from atom.compass.run import STEP_TABLE_FILE, SUMMARY_FILE, channel_table

DONE = "data: [DONE]"


def expected_timeline(record, prefill_s: float, decode_s: float, tars) -> list:
    """One engine LP's TAR rows, ``(from, to)``, under a constant step stub.

    `record` maps each DP rank to its steps (`atom.compass.parity.read`).
    `tars` is the LP's TAR rows as the step table has them, read only for the
    clock each step starts at; everything after the start is computed here.

    A forward's device seconds are `prefill_s` when a row prefills plus
    `decode_s` when a row decodes. With no host laws it enters at its start or
    once the device work two forwards back ends, whichever is later, and
    returns then, or, when it produces output, once the previous output
    forward's device work ends. Above DP width one a rank producing output,
    not a dummy, returns at the group's latest return and any other at its own;
    the step is one TAR per distinct return, ascending.
    """
    ranks = sorted(record)
    # Per rank: device end of forwards n-2 and n-1, and of the last output one.
    device = {r: (-math.inf, -math.inf, -math.inf) for r in ranks}
    want, i = [], 0
    for k in range(len(record[ranks[0]])):
        if i >= len(tars):
            break
        now = tars[i][0]
        returns = []
        for r in ranks:
            step = record[r][k]
            rows = [row[-1] for row in step["rows"]]
            busy = prefill_s * (not all(rows)) + decode_s * any(rows)
            two_back, last, output = device[r]
            host = max(now, two_back)
            end = max(last, host) + busy
            back = host
            if step["produces_output"]:
                back, output = max(host, output), end
            device[r] = (last, max(end, host), output)
            waits = step["produces_output"] and not step["is_dummy_run"]
            returns.append((back - now, waits))
        top = max(t for t, _ in returns)
        start = now
        for t in sorted({top if waits else t for t, waits in returns}):
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
    if not stub.shape_blind or run.get("host", {}).get("laws"):
        return [
            f"{lp}: a timeline is computed for a constant step stub with no host laws"
        ]
    tars = [
        (float(f[1]), float(f[2]))
        for f in map(str.split, lines[1:])
        if f[0] == lp and f[3] == TAR
    ]
    want = expected_timeline(record, stub.prefill_step, stub.decode_step, tars)
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
