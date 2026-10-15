# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh`` keeps every other port it picks out of
the ``data_parallel_size`` squared write-request ports from KV_WRITE_REQ_PORT.

The script runs end to end with the stand-ins of
``test_cctraces_sim_no_export``, and its port pick gets planted ports first:
PORT is one above KV_WRITE_REQ_PORT. Nothing needs a driver or aiperf.
"""

import os
import socket
import subprocess
import sys
from pathlib import Path

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"
DP = 2

# Binds to port 0 take the planted ports, in order, while any are left.
SITECUSTOMIZE = """\
import os, socket
planted = [int(p) for p in os.environ["PLANTED_PORTS"].split()]
bind = socket.socket.bind
socket.socket.bind = lambda s, a: bind(s, (a[0], planted.pop(0)) if a[1] == 0 and planted else a)
"""


def _free_pair():
    """A free port whose successor is free too."""
    while True:
        with socket.socket() as a, socket.socket() as b:
            a.bind(("127.0.0.1", 0))
            p = a.getsockname()[1]
            try:
                b.bind(("127.0.0.1", p + 1))
            except OSError:
                continue
            return p


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_no_other_port_lands_in_the_write_request_block(tmp_path):
    kv = _free_pair()
    # CLOCK_PORT PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT
    planted = [_free_port(), kv + 1, _free_port(), _free_port(), kv, _free_port()]
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    picked = tmp_path / "picked"
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        'if [ "$1" = - ]; then\n'
        f'    PYTHONPATH={site}:$PYTHONPATH {sys.executable} "$@" | tee -a {picked}\n'
        "    exit\n"
        "fi\n"
        'if [ "$2" = atom.entrypoints.openai.api_server ]; then\n'
        '    echo "Uvicorn running"; sleep 1; exit 0\n'
        "fi\n"
        f'exec {sys.executable} "$@"\n'
    )
    harness = bin_ / "harness-python"
    harness.write_text(
        "#!/bin/sh\n"
        '[ "$1" = -c ] && exec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
        "exit 3\n"
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id": 0}\n')
    run_file = tmp_path / "run.json"
    run_file.write_text(f'{{"data_parallel_size": {DP}}}')
    env = dict(
        os.environ,
        PATH=f"{bin_}:{os.environ['PATH']}",
        PLANTED_PORTS=" ".join(map(str, planted)),
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
    # The port pick is the first of the two `python3 -` runs.
    ports = [int(p) for p in picked.read_text().splitlines()[0].split()[1:]]
    assert len(ports) == 6, done.stdout + done.stderr
    kv_port = ports.pop(4)
    # The planted picks held, so the block is clear because KV_WRITE_REQ_PORT moved.
    assert ports == planted[:4] + planted[5:], (planted, kv_port, ports)
    block = range(kv_port, kv_port + DP * DP)
    assert not [p for p in ports if p in block], (planted, kv_port, ports)
