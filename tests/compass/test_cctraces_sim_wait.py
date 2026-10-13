# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` stops a simulated cell whose deployment
has left while aiperf is still up, ``WAIT_S`` seconds after it left.

The script runs end to end with two stand-ins on ``PATH``: a ``python3`` whose
API server says it is ready and exits, and a harness Python whose aiperf never
returns. Nothing needs a driver or aiperf.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
WAIT_S = 3


def test_aiperf_outliving_the_deployment_ends_the_cell_within_wait_s(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    left = tmp_path / "left"
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        f'    echo "Uvicorn running"; sleep 1; date +%s.%N >{left}; exit 1\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        # The import probe: this tree's compass_harness, as the script requires.
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exec sleep 600\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text(json.dumps({}))
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(harness),
        WAIT_S=str(WAIT_S),
    )
    done = subprocess.run(
        ["bash", str(SCRIPT), str(run_file), str(tmp_path / "cell")],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    ended = time.time()
    assert done.returncode == 1, done.stdout + done.stderr
    assert "aiperf is still up 3 s after the deployment left" in done.stderr
    # The script polls once a second; the rest is slack for a loaded host.
    assert ended - float(left.read_text()) < WAIT_S + 5
