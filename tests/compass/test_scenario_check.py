# SPDX-License-Identifier: MIT
"""`check_run` over a run the recording authority writes, with no driver.

One request through traffic, frontend and engine, its LPs scripted against
`_RecordingAuthority`: three engine forwards under the slices' constant stub,
at the ends the stub implies. The run passes every check; each seeded
mutation of one artifact turns the check it names red, and no other. The DP
group's timeline is checked on a hand-written two-rank record.
"""

import json
import math
import shutil

import pytest
from test_vertical_slice import CONSTANT, _run_file

from atom.compass import run as compass_run
from atom.compass.clock import NER, TAR, LpId
from atom.compass.detect.scenario import check_run, expected_timeline, main

P, D = CONSTANT["prefill_step"], CONSTANT["decode_step"]
HTTP, REQUEST = "traffic->frontend:http", "frontend->engine:request#dp0"
OUTPUT, STREAM = "engine->frontend:output#dp0", "frontend->traffic:stream"
TAKEN = ['data: {"choices": [], "usage": {"completion_tokens": 3}}', "data: [DONE]"]
# The engine's forwards: a prefill, then two decodes, each producing output.
STEPS = [[[5, 5, False]], [[1, 6, True]], [[1, 7, True]]]


def _run(out):
    run = json.loads(_run_file(out, step_stub=CONSTANT).read_text())
    ca = compass_run._RecordingAuthority(run)
    traffic, frontend, engine = map(LpId, ("traffic", "frontend", "engine"))
    ipc = run["ipc_s"]
    ca.on_request(traffic, NER, math.inf, [(HTTP, 0, 1.6e-3)])
    ca.on_request(engine, NER, math.inf, [])
    ca.on_request(frontend, NER, math.inf, [])  # granted the request at 1.6e-3
    r = 1.6e-3 + ipc
    ca.on_request(frontend, NER, math.inf, [(REQUEST, 0, r)])
    for t in (r, r + P, r + P + D):  # each forward's end under the stub
        ca.on_request(engine, TAR, t, [])
    ca.on_request(engine, NER, math.inf, [(OUTPUT, 0, r + P + D + ipc)])
    end = r + P + D + ipc + run["stream_s"]
    ca.on_request(frontend, NER, math.inf, [(STREAM, 0, end)])
    ca.on_request(traffic, NER, math.inf, [])  # the stream taken, nothing is left
    assert ca.final_clocks is not None, "the scripted run did not finish"
    (out / compass_run.STEP_TABLE_FILE).write_text(ca.steps.text())
    summary = {"schedule": {"refusals": {"count": 0, "reasons": []}}}
    (out / compass_run.SUMMARY_FILE).write_text(
        json.dumps(summary | {"coverage_report": False})
    )
    record = out / "parity"
    record.mkdir()
    (record / "dp0.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "step": k,
                    "dp_rank": 0,
                    "is_dummy_run": False,
                    "produces_output": True,
                    "batch": [],
                    "named": [],
                    "rows": rows,
                }
            )
            + "\n"
            for k, rows in enumerate(STEPS)
        )
    )


@pytest.fixture
def runs(tmp_path):
    first, twin = tmp_path / "seed1", tmp_path / "seed2"
    first.mkdir()
    _run(first)
    shutil.copytree(first, twin)
    return first, twin


def _check(runs, taken=TAKEN):
    first, _ = runs
    return check_run(runs, [3], taken, {"engine": first / "parity"})


def test_a_scripted_run_passes_every_check(runs):
    print("\n" + (runs[0] / compass_run.STEP_TABLE_FILE).read_text())
    assert _check(runs) == []
    assert main([*map(str, runs), "--record", f"engine={runs[0] / 'parity'}"]) == 0


def _edit(path, old, new):
    text = path.read_text()
    assert old in text, f"{old!r} is not in {path}"
    path.write_text(text.replace(old, new, 1))


def _row(table, event, channel=""):
    return next(
        r for r in table.read_text().splitlines()[1:] if f" {event} {channel}" in r
    )


def _drop_finish(first, twin):
    table = first / compass_run.STEP_TABLE_FILE
    _edit(table, "\n" + table.read_text().splitlines()[-1], "")


def _refuse(first, twin):
    _edit(first / compass_run.SUMMARY_FILE, '"count": 0', '"count": 1')


def _diverge(first, twin):
    table = twin / compass_run.STEP_TABLE_FILE
    row = _row(table, "release", STREAM)
    _edit(table, row, row.replace(" 0 -", " 0 recovered"))


def _release_twice(first, twin):
    table = first / compass_run.STEP_TABLE_FILE
    row = _row(table, "release", REQUEST)
    _edit(table, row, f"{row}\n{row}")


def _release_into_the_past(first, twin):
    table = first / compass_run.STEP_TABLE_FILE
    row = _row(table, "release", REQUEST).split()
    _edit(table, " ".join(row), " ".join([row[0], "1.0", *row[2:]]))


def _release_to_another_lp(first, twin):
    table = first / compass_run.STEP_TABLE_FILE
    row = _row(table, "release", REQUEST)
    _edit(table, row, row.replace("engine", "frontend", 1))


def _lose_a_token(first, twin):
    return [TAKEN[0].replace(": 3}", ": 2}"), TAKEN[1]]


def _misprice_a_forward(first, twin):
    _edit(first / "parity" / "dp0.jsonl", "[[1, 6, true]]", "[[1, 6, false]]")


def _record_an_unclocked_step(first, twin):
    record = first / "parity" / "dp0.jsonl"
    step = json.loads(record.read_text().splitlines()[-1]) | {"step": 3}
    record.write_text(record.read_text() + json.dumps(step) + "\n")


def _record_a_short_rank(first, twin):
    first_step = (first / "parity" / "dp0.jsonl").read_text().splitlines()[0]
    (first / "parity" / "dp1.jsonl").write_text(first_step + "\n")


@pytest.mark.parametrize(
    "mutate, named",
    [
        (_drop_finish, "traffic did not reach the finish"),
        (_refuse, "the run refused"),
        (_diverge, "produced different records"),
        (_release_twice, f"{REQUEST} released seqs [0, 0]"),
        (_release_into_the_past, "behind engine's clock at 1.0"),
        (_release_to_another_lp, "released to frontend, not to engine"),
        (_lose_a_token, "usage counts [2]"),
        (_misprice_a_forward, "engine: TAR row 2"),
        (_record_an_unclocked_step, "engine: record step 3 has no TAR row"),
        (_record_a_short_rank, "engine: the record's ranks hold {0: 3, 1: 1} steps"),
    ],
)
def test_each_seeded_mutation_turns_its_check_red(runs, mutate, named):
    taken = mutate(*runs) or TAKEN
    if mutate is not _diverge:  # a mutation of one run's table, made in both
        shutil.copy(*(r / compass_run.STEP_TABLE_FILE for r in runs))
    failed = _check(runs, taken)
    print("\n" + "\n".join(failed))
    assert failed and all(named in line for line in failed)
    assert main([*map(str, runs), "--record", f"engine={runs[0] / 'parity'}"]) == (
        0 if mutate is _lose_a_token else 1
    )


# Two output ranks. Rank 0 prefills, then decodes twice. Rank 1 decodes while
# rank 0 prefills, then runs a mixed batch or a decode, then decodes.
PREFILL, DECODE = [[5, 5, False]], [[1, 6, True]]


@pytest.mark.parametrize(
    "rank1_step1, last",
    [
        # Rank 1's mixed batch starts at the group's device end P and costs
        # P + D, so the last step returns at 2P + D.
        (PREFILL + DECODE, 2 * P + D),
        # Both decodes start at the group's device end P.
        (DECODE, P + D),
    ],
    ids=["mixed", "decode"],
)
def test_a_dp_step_returns_once_at_its_last_output_rank(rank1_step1, last):
    record = {
        r: [
            {"rows": rows, "produces_output": True, "is_dummy_run": False}
            for rows in steps
        ]
        for r, steps in (
            (0, [PREFILL, DECODE, DECODE]),
            (1, [DECODE, rank1_step1, DECODE]),
        )
    }
    # Step 1 returns once, at rank 0's prefill end, not first at rank 1's D.
    tars = [(0.0, 0.0), (0.0, P), (P, last)]
    assert expected_timeline(record, P, D, tars) == tars
