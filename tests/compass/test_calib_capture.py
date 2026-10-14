# SPDX-License-Identifier: MIT
"""`calib_capture.sh`: one cell at a time, stopped after its first recorded step,
never on a GPU that holds a process.

`cctraces_sim.sh` and `rocm-smi` are stubbed, so these run with no GPU: the stub
cell writes a step and then runs a process named as aiperf's controller is,
which only the script's stop ends.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
BASH = shutil.which("bash")
FREE = "No KFD PIDs currently running\n"
BUSY = "PID 1 is using 1 DRM device(s):\n6\n"
CELL = """#!/bin/bash
mkdir -p "$2/steps" && echo '{}' > "$2/steps/dp0.jsonl"
bash -c 'exec -a "aiperf system_controller" sleep 60'
echo "c=$CONCURRENCY done" >> "$2/../order"
"""


def _capture(tmp_path, smi, cells, gpus="6"):
    tree = tmp_path / "ATOM"
    (tree / "scripts" / "compass").mkdir(parents=True)
    for name in ("calib_capture.sh", "_lib.sh"):
        shutil.copy(COMPASS / name, tree / "scripts" / "compass" / name)
    (tree / "scripts" / "compass" / "cctraces_sim.sh").write_text(CELL)
    (tree / ".compass-commit").write_text("cafe\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "rocm-smi").write_text(f"#!/bin/sh\nprintf '{smi}'\n")
    (bindir / "rocm-smi").chmod(0o755)
    (tmp_path / "run.json").write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        CELLS=cells,
        STEPS_S="0",
        HIP_VISIBLE_DEVICES=gpus,
    )
    script = "scripts/compass/calib_capture.sh"
    r = subprocess.run(
        [BASH, script, str(tmp_path / "run.json"), "cap"],
        cwd=tree,
        env=env,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    return r, tree / "cap"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_each_cell_is_stopped_and_logged_in_turn(tmp_path):
    r, cap = _capture(tmp_path, FREE, "4 1")
    assert r.returncode == 0, r.stderr
    assert (cap / "order").read_text() == "c=4 done\nc=1 done\n"
    for c in (4, 1):
        log = (cap / f"c{c}.log").read_text()
        assert log.startswith(f"cafe\nCONCURRENCY={c} GPU=6 ")
        assert "stopped aiperf" in log and "CELL_RC=0" in log
        assert log.count("uptime:") == 2


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_gpu_holding_a_process_runs_no_cell(tmp_path):
    r, cap = _capture(tmp_path, BUSY, "1")
    assert r.returncode == 3
    assert "GPU 6 is in use" in (cap / "c1.log").read_text()
    assert not (cap / "c1").exists()


EIGHT = "0,1,2,3,4,5,6,7"


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_process_on_the_last_named_gpu_runs_no_cell(tmp_path):
    r, cap = _capture(tmp_path, "PID 1 is using 1 DRM device(s):\n7\n", "1", EIGHT)
    assert r.returncode == 3
    assert "GPU 7 is in use" in (cap / "c1.log").read_text()
    assert not (cap / "c1").exists()


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_process_on_an_unnamed_gpu_leaves_the_cell_running(tmp_path):
    r, cap = _capture(tmp_path, "PID 1 is using 2 DRM device(s):\n6 7\n", "1", "0,1")
    assert r.returncode == 0, r.stderr
    assert (cap / "c1.log").read_text().startswith("cafe\nCONCURRENCY=1 GPU=0,1 ")
