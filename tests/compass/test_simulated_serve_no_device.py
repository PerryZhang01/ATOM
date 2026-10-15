# SPDX-License-Identifier: MIT
"""A simulated serve where there is no driver, with nothing stood in.

The run is `test_vertical_slice`'s: ATOM's API server with ``--compass-run``,
its engine core and TP1 worker, and one streamed request from the traffic LP.
Where there is no driver, every call that needs one fails the serve, so this
test reads the server log and names the first: what it raised, the ATOM call
it came through and the frame that raised it.

It runs only where there is no driver, which is what it probes, and needs
`ATOM_COMPASS_SLICE_MODEL` as `test_vertical_slice` does.
"""

import re

import pytest
import torch
from test_vertical_slice import MODEL, run_slice

TRACEBACK = "Traceback (most recent call last):"
CHAINED = ("During handling of the above exception", "The above exception was")
FRAME = re.compile(r'  File "(?P<path>[^"]+)", line (?P<line>\d+), in (?P<func>.+)')
#: Where a frame's path is cut, so the name reads the same from any install.
ROOTS = re.compile(r".*?/(?=(?:atom|aiter|triton|torch)/)")


def _frame(match: re.Match) -> str:
    path = ROOTS.sub("", match["path"], count=1)
    return f"{path}:{match['line']} in {match['func']}"


def first_failure(log: str) -> str:
    """The first traceback in a server log, chained ones read as one.

    Named by the exception the chain began with, the deepest ATOM frame of any
    of its stacks and the frame that raised it.
    """
    lines = log.splitlines()
    if TRACEBACK not in lines:
        return "no traceback in the server log"
    start = lines.index(TRACEBACK)
    raised = atom = raiser = None
    for i, line in enumerate(lines[start:], start):
        if match := FRAME.match(line):
            frame = _frame(match)
            raiser = raiser if raised else frame
            atom = frame if frame.startswith("atom/") else atom
        elif line[:1].strip() and not line.startswith((TRACEBACK, *CHAINED)):
            raised = raised or line
            following = next((s for s in lines[i + 1 :] if s.strip()), "")
            if not following.startswith(CHAINED):
                break
    return f"{raised}, through {atom}, raised at {raiser}"


@pytest.mark.skipif(
    torch.cuda.is_available() or not MODEL,
    reason="probes a serve where there is no driver, and needs "
    "ATOM_COMPASS_SLICE_MODEL naming a model directory",
)
def test_a_simulated_serve_completes_with_no_driver(tmp_path):
    out = tmp_path / "run"
    try:
        run = run_slice(out, "1")
    except Exception as e:  # noqa: BLE001 - the server log says why
        log = (out / "server.log").read_text()
        pytest.fail(f"the serve needs a driver: {first_failure(log)}\n({e!r})")
    assert run["events"][-1][1] == "data: [DONE]"
