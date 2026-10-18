# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` on the run file `compass run` keeps, whose
server serves until stopped, ends its cell with aiperf's client run and prints
its result line.

The run file is `drive.runfile`'s, on a step stub. The script runs a simulated
cell end to end with two stand-ins on ``PATH``: a ``python3`` whose API server
leaves at the finish as ATOM's simulated one does, and a harness Python whose
aiperf writes an export. Nothing needs a driver or aiperf.
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from test_cctraces_sim_client_run import METRICS
from test_drive_runfile import build

from atom.compass.backends.step_stub import Coefficients

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
# The finish is the end of the last client run the run file declares, 1 when
# absent; with more declared, or none (null), the server serves on.
SERVER = """\
import json, sys, time
from pathlib import Path

run = json.loads(Path(sys.argv[sys.argv.index("--compass-run") + 1]).read_text())
out = Path(run["out_dir"])
print("Uvicorn running", flush=True)
# aiperf's artifacts mark the end of its client run.
while run.get("client_runs", 1) != 1 or not (out / "artifacts").exists():
    time.sleep(0.1)
(out / "summary.json").write_text(json.dumps({
    "cost": {"simulated_seconds": 1.0, "wall_seconds": 1.0},
    "schedule": {"refusals": {"count": 0}},
    "coverage_report": False,
}))
"""


def test_a_compass_run_run_file_ends_the_cell_with_its_result_line(tmp_path):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    run_file = tmp_path / "run_file.json"
    run_file.write_text(json.dumps(build(tmp_path, "--step-stub", str(stub))))
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (tmp_path / "server.py").write_text(SERVER)
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        f'    shift 2; exec {sys.executable} {tmp_path / "server.py"} "$@"\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {k: {"avg": 1} for k in METRICS}
            | {"benchmark_id": "000000000000", "error_summary": []}
        )
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        f"mkdir -p artifacts && cp {export} artifacts/profile_export_aiperf.json\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(harness),
        WAIT_S="3",
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

    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.splitlines()[-1].startswith("cctraces_sim: rc=0 ")


def test_a_cells_server_takes_a_free_dp_master_port(tmp_path, monkeypatch):
    """The port keys ATOM's compiled-graph cache, and the first start that
    loads that cache autotunes again, so the cell does not pass the caller's."""
    port = tmp_path / "port"
    monkeypatch.setenv("ATOM_DP_MASTER_PORT", "29500")
    monkeypatch.setitem(
        globals(),
        "SERVER",
        f"import os\nopen({str(port)!r}, 'w').write(os.environ['ATOM_DP_MASTER_PORT'])\n"
        + SERVER,
    )
    test_a_compass_run_run_file_ends_the_cell_with_its_result_line(tmp_path)
    assert port.read_text() not in ("", "29500")
