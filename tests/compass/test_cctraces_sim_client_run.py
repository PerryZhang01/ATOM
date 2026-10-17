# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` runs aiperf at ``--export-level raw`` and
writes the client run's export from its raw records, or exits non-zero with
its result line when the export refuses.

The script runs a ``REAL=1`` cell end to end with three stand-ins on ``PATH``:
a ``python3`` whose API server says it is ready and serves until stopped, a
harness Python whose aiperf writes an export and raw records, and a
``rocm-smi`` listing no process. Nothing needs a driver or aiperf.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_parity_client_run import ARGUMENTS, CONFIG, PARAMETERS, raw, sse

from atom.compass.parity.client_run import NAME, RAW

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
METRICS = (
    "request_count",
    "overall_usage_prompt_cache_read_pct",
    "theoretical_prefix_cache_hit",
    "time_to_first_token",
    "inter_token_latency",
    "request_throughput",
)


@pytest.mark.parametrize("ids, cell_rc", [(["a", "b"], 0), (["a", "a"], 1)])
def test_a_cell_writes_its_client_run_s_export(tmp_path, ids, cell_rc):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        '    echo "Uvicorn running"; exec sleep 600\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    # The input config of the trace files the script gives aiperf.
    given = str(tmp_path / "cell" / "traces")
    dataset = {"name": "main", "type": "file", "path": given, "format": "weka_trace"}
    dataset |= {"random_seed": 20260707}
    export = tmp_path / "export.json"
    export.write_text(
        json.dumps(
            {k: {"avg": 1} for k in METRICS}
            | {"benchmark_id": "000000000000", "error_summary": []}
            | {"input_config": CONFIG | {"datasets": [dataset]}}
        )
    )
    records = tmp_path / "raw.jsonl"
    records.write_text(
        "".join(
            json.dumps(raw([sse(("data", json.dumps({"id": i})))], f"t{k}")) + "\n"
            for k, i in enumerate(ids)
        )
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        'case "$2" in\n'
        '*"import compass_harness"*) exec echo "${PYTHONPATH%%:*}/compass_harness/x.py" ;;\n'
        "*entry_points*) exit 0 ;;\n"
        "esac\n"
        'echo "$@" > aiperf_args.txt\n'
        f"mkdir -p artifacts && cp {export} artifacts/profile_export_aiperf.json\n"
        f"cp {records} artifacts/{RAW}\n"
    )
    (bin_ / "rocm-smi").write_text("#!/bin/sh\nexit 0\n")
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
        MODEL=str(tmp_path),
        TRACES=str(traces),
        HARNESS_PYTHON=str(harness),
        WAIT_S="30",
    )
    cell = tmp_path / "cell"
    done = subprocess.run(
        ["bash", str(SCRIPT), str(run_file), str(cell)],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == cell_rc, done.stdout + done.stderr
    assert done.stdout.splitlines()[-1].startswith(f"cctraces_sim: rc={cell_rc} ")
    assert " --export-level raw " in (cell / "aiperf_args.txt").read_text()
    # What aiperf writes is not the workload it sends.
    workload = json.loads((cell / "run.json").read_text())["workload"]
    assert "--export-level" not in workload["aiperf_args"]
    if cell_rc:
        assert "two requests got response id a" in done.stderr
        assert not (cell / "artifacts" / NAME).exists()
    else:
        assert json.loads((cell / "artifacts" / NAME).read_text()) == {
            "requests": {
                "a": {"trace": "t0", "sampling": PARAMETERS},
                "b": {"trace": "t1", "sampling": PARAMETERS},
            },
            "warmup": {},
            "traces": [0],
            "arguments": ARGUMENTS,
        }
