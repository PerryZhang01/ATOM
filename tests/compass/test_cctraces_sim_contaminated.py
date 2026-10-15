# SPDX-License-Identifier: MIT
"""A ``REAL=1`` cell of ``scripts/compass/cctraces_sim.sh`` is contaminated by a
process on any GPU in ``HIP_VISIBLE_DEVICES``, and not by one on another GPU.

The script runs end to end with three stand-ins on ``PATH``: a ``python3``
whose API server says it is ready and serves until stopped, a harness Python
whose aiperf writes an export, and a ``rocm-smi`` listing one process on a
middle named GPU and one on an unnamed GPU. Nothing needs a driver or aiperf.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
LISTING = (
    "KFD process information:\n"
    "PID    \tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY\t\n"
    "4242   \tUNKNOWN     \t1     \t481734656\t0        \t0           \t\n"
    "1337009\tUNKNOWN     \t1     \t1024     \t0        \t0           \t\n"
    "=== GPUs Indexed by PID ===\n"
    "PID 4242 is using 1 DRM device(s):\n2 \n"
    "PID 1337009 is using 1 DRM device(s):\n5 \n"
)


def test_a_process_on_a_middle_named_gpu_contaminates_the_cell(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        '    echo "Uvicorn running"; exec sleep 600\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    metrics = (
        "request_count",
        "overall_usage_prompt_cache_read_pct",
        "theoretical_prefix_cache_hit",
        "time_to_first_token",
        "inter_token_latency",
        "request_throughput",
    )
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {k: {"avg": 1} for k in metrics}
            | {"benchmark_id": "000000000000", "error_summary": []}
        )
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        # The import probe names this tree's compass_harness; the plugin probe
        # names no plugin; aiperf writes its export.
        'case "$2" in\n'
        '*"import compass_harness"*) exec echo "${PYTHONPATH%%:*}/compass_harness/x.py" ;;\n'
        "*entry_points*) exit 0 ;;\n"
        "esac\n"
        f"mkdir -p artifacts && cp {export} artifacts/profile_export_aiperf.json\n"
    )
    smi = bin_ / "rocm-smi"
    smi.write_text(f"#!/bin/sh\ncat <<'EOF'\n{LISTING}EOF\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text("{}")
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        REAL="1",
        HIP_VISIBLE_DEVICES="1,2,3",
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
    assert done.returncode == 0, done.stdout + done.stderr
    result = done.stdout.splitlines()[-1]
    fields = dict(f.split("=", 1) for f in result.split()[1:])
    assert fields["contaminated"] == "True", result
    assert fields["gpu_before"] == fields["gpu_after"] == "4242:481734656", result
