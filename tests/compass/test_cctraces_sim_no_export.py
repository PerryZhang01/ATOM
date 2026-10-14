# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` names a cell whose aiperf wrote no
``profile_export_aiperf.json`` and exits with aiperf's code, 1 if that was 0.

The script runs end to end with two stand-ins on ``PATH``: a ``python3`` whose
API server says it is ready and exits 0, and a harness Python whose aiperf
exits at once with the given code and writes nothing. Nothing needs a driver
or aiperf.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"


@pytest.mark.parametrize("aiperf_rc, cell_rc", [(3, 3), (0, 1)])
def test_a_cell_with_no_export_names_it_and_keeps_aiperfs_code(
    tmp_path, aiperf_rc, cell_rc
):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        '    echo "Uvicorn running"; sleep 1; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        # The import probe: this tree's compass_harness, as the script requires.
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        f"exit {aiperf_rc}\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(harness),
        WAIT_S="30",
    )
    done = subprocess.run(
        ["bash", str(SCRIPT), str(run_file), str(tmp_path / "cell")],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == cell_rc, done.stdout + done.stderr
    assert done.stdout.splitlines()[-1] == (
        f"cctraces_sim: rc={aiperf_rc}, aiperf wrote no export"
    )
    assert "Traceback" not in done.stderr
