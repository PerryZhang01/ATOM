# SPDX-License-Identifier: MIT
"""`calib_capture.sh`: one cell at a time, stopped after its first recorded step,
never on a GPU that holds a process.

`cctraces_sim.sh` and `rocm-smi` are stubbed, so these run with no GPU: the stub
cell runs a process named as aiperf's controller is, which writes a step and
waits; as aiperf does, SIGINT ends it with status 0, and SIGTERM with 143. The
cell exits with its status, as cctraces_sim.sh exits with aiperf's.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from test_drive_gpu import GPU45, SHOWHW

COMPASS = Path(__file__).resolve().parents[2] / "scripts" / "compass"
BASH = shutil.which("bash")
FREE = "No KFD PIDs currently running\n"
BUSY = "PID 1 is using 1 DRM device(s):\n6\n"
AIPERF = """import signal, sys, time
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
open(sys.argv[1], "w").write("{}")
time.sleep(60)"""
CELL = f"""#!/bin/bash
mkdir -p "$2/steps" && echo '{{"workload": 1}}' > "$2/run.json"
(exec -a "aiperf system_controller" {sys.executable} -c '{AIPERF}' "$2/steps/dp0.jsonl")
rc=$?
echo "c=$CONCURRENCY done" >> "$2/../order"
exit $rc
"""
#: `identity` as a stub returns what it was given.
IDENTITY = """def identity(engine_file, model, argv, gpus):
    return {"format_version": 2, "engine": str(engine_file), "model": model,
            "argv": argv, "gpus": list(gpus)}
"""


def _capture(tmp_path, smi, cells, gpus="6", showhw=SHOWHW, **env):
    tree = tmp_path / "ATOM"
    (tree / "scripts" / "compass").mkdir(parents=True)
    drive = tree / "atom" / "compass" / "drive"
    drive.mkdir(parents=True)
    for d in (drive, drive.parent, drive.parent.parent):
        (d / "__init__.py").write_text("")
    (drive / "identity.py").write_text(IDENTITY)
    for name in ("calib_capture.sh", "_lib.sh"):
        shutil.copy(COMPASS / name, tree / "scripts" / "compass" / name)
    (tree / "scripts" / "compass" / "cctraces_sim.sh").write_text(CELL)
    (tree / ".compass-commit").write_text("cafe\n")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "showhw").write_text(showhw)
    (bindir / "rocm-smi").write_text(
        f'#!/bin/sh\ncase "$*" in *--showhw*) cat {bindir}/showhw;;\n'
        f"*) printf '{smi}';;\nesac\n"
    )
    (bindir / "rocm-smi").chmod(0o755)
    (tmp_path / "run.json").write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{bindir}:{os.environ['PATH']}",
        CELLS=cells,
        STEPS_S="0",
        HIP_VISIBLE_DEVICES=gpus,
        **env,
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


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
@pytest.mark.parametrize("gpu", ["3", "7"])
def test_a_process_on_any_named_gpu_runs_no_cell(tmp_path, gpu):
    smi = f"PID 1 is using 1 DRM device(s):\n{gpu}\n"
    r, cap = _capture(tmp_path, smi, "1", "0,1,2,3,4,5,6,7")
    assert r.returncode == 3
    assert f"GPU {gpu} is in use" in (cap / "c1.log").read_text()
    assert not (cap / "c1").exists()


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_process_on_an_unnamed_gpu_leaves_the_cell_running(tmp_path):
    r, cap = _capture(tmp_path, "PID 1 is using 2 DRM device(s):\n6 7\n", "1", "0,1")
    assert r.returncode == 0, r.stderr
    assert (cap / "c1.log").read_text().startswith("cafe\nCONCURRENCY=1 GPU=0,1 ")


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
@pytest.mark.parametrize("gpu, rc", [("1", 0), ("3", 3)])
def test_the_guard_reads_hip_devices_0_and_1_as_rocm_smi_gpus_3_and_0(
    tmp_path, gpu, rc
):
    smi = f"PID 1 is using 1 DRM device(s):\n{gpu}\n"
    r, cap = _capture(tmp_path, smi, "1", "0,1", GPU45)
    assert r.returncode == rc, r.stderr
    log = (cap / "c1.log").read_text()
    assert log.startswith("cafe\nCONCURRENCY=1 GPU=0,1 ROCM_SMI_GPU=3,0 ")
    assert ("rocm-smi GPU 3 is in use" in log) == (rc == 3)


@pytest.mark.skipif(BASH is None, reason="no bash on PATH")
def test_a_stopped_cell_ends_with_aiperf_s_exit_and_a_record_s_identity(tmp_path):
    r, cap = _capture(tmp_path, FREE, "1", "0,1", GPU45, MODEL="/models/m")
    assert r.returncode == 0, r.stderr
    log = (cap / "c1.log").read_text()
    assert "stopped aiperf" in log and "CELL_RC=0" in log, log
    run = json.loads((cap / "c1" / "run.json").read_text())
    tree = tmp_path / "ATOM"
    assert run == {
        "workload": 1,
        "format_version": 2,
        "engine": f"{cap}/c1/steps/dp0.engine.json",
        "model": "/models/m",
        "argv": [
            f"{tree}/scripts/compass/cctraces_sim.sh",
            str(tmp_path / "run.json"),
            f"{cap}/c1",
        ],
        "gpus": [3, 0],
    }
