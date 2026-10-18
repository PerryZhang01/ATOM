# SPDX-License-Identifier: MIT
"""`pd_sim.sh`, run directly or through `cctraces_sim.sh`, starts each side
under its own flags and environment, and from what each server was started
with, `keyed` picks that side's entry of DeepSeek-V4-Pro's two DPA8 laws.

`python3` is stubbed on `PATH`: an API server prints its arguments and its
environment to its log and waits, and the clock authority only listens. Nothing
needs a driver.
"""

import ast
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.benchmarks.results.metadata import PERFORMANCE_KEYS
from atom.compass import run as compass_run
from atom.compass.run.step_grid import keyed

TREE = Path(__file__).resolve().parents[2]
LAWS = [
    json.loads(
        (
            TREE / f"scripts/compass/laws/deepseek-v4-pro-dpa8-mi355x-gpu45-{side}.json"
        ).read_text()
    )
    for side in ("prefill", "decode")
]
PREFILL_ENV = {"ATOM_NUMA_BIND": "1", "GPU_MAX_HW_QUEUES": "5"}
SHARED_ENV = {
    "AITER_BF16_FP8_MOE_BOUND": "0",
    "AITER_LOG_LEVEL": "WARNING",
    "ATOM_MOE_GU_ITLV": "1",
}
DECODE_SIZES = "[1,2," + ",".join(map(str, range(4, 257, 4))) + "]"
#: `EngineArgs.cudagraph_capture_sizes`, the ladder without the flag.
DEFAULT_SIZES = "[1,2,3,4,5,6,7,8,16,32,48,64,128,256,512]"
PYTHON3 = f"""#!/bin/sh
case "$2" in
atom.entrypoints.openai_server) echo "$@"; env; exec sleep 60 ;;
atom.compass.run) exec {sys.executable} -c 'import socket, sys, time
s = socket.create_server(("127.0.0.1", int(sys.argv[1].rsplit(":", 1)[1])))
time.sleep(60)' "$6" ;;
esac
exec {sys.executable} "$@"
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def started(tmp_path, script) -> dict:
    """Each side's (arguments, environment) as `script` started its server."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "python3").write_text(PYTHON3)
    # The import probe: this tree's compass_harness, as cctraces_sim.sh requires.
    (bin_ / "harness").write_text(
        '#!/bin/sh\necho "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    (out / "run.json").write_text(json.dumps({"kv_write_req_s": 1e-5}))
    (tmp_path / "traces.jsonl").write_text('{"id": 0}\n')
    env = {k: v for k, v in os.environ.items() if k not in PERFORMANCE_KEYS}
    env.update(
        SHARED_ENV,
        PATH=f"{bin_}:{os.environ['PATH']}",
        MODEL=str(tmp_path),
        TRACES=str(tmp_path / "traces.jsonl"),
        HARNESS_PYTHON=str(bin_ / "harness"),
        WAIT_S="2",
        **{p: str(free_port()) for p in ("CLOCK_PORT", "PREFILL_PORT", "DECODE_PORT")},
        PREFILL_ARGS="--enable-tbo",
        DECODE_ARGS=f"--cudagraph-capture-sizes {DECODE_SIZES}",
        PREFILL_ENV=" ".join(f"{k}={v}" for k, v in PREFILL_ENV.items()),
    )
    args = [out / "run.json"] + ([out] if script == "cctraces_sim.sh" else [])
    done = subprocess.run(
        ["bash", str(TREE / "scripts/compass" / script), *map(str, args)],
        env=env,
        cwd=TREE,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 1, done.stdout + done.stderr
    sides = {}
    for side in ("prefill", "decode"):
        argv, *lines = (out / f"{side}.log").read_text().splitlines()
        sides[side] = argv.split(), dict(e.split("=", 1) for e in lines if "=" in e)
    return sides


def side_config(argv) -> SimpleNamespace:
    """The engine config `keyed` reads: the side's own flags over the layout
    both laws share."""
    shared = LAWS[1]["engine_config"]
    flag = "--cudagraph-capture-sizes"
    sizes = argv[argv.index(flag) + 1] if flag in argv else DEFAULT_SIZES
    return SimpleNamespace(
        enforce_eager=False,
        max_num_seqs=512,
        max_num_batched_tokens=16384,
        capture_sizes=ast.literal_eval(sizes),
        kv_cache_block_size=shared["kv_cache_block_size"],
        kv_cache_dtype=shared["kv_cache_dtype"],
        quant_config=SimpleNamespace(compute_hash=lambda: shared["quantization"]),
        tensor_parallel_size=1,
        parallel_config=SimpleNamespace(data_parallel_size=8),
        enable_dp_attention=True,
        enable_expert_parallel=False,
        enable_tbo="--enable-tbo" in argv,
        enable_tbo_decode=False,
    )


@pytest.mark.parametrize("script", ["pd_sim.sh", "cctraces_sim.sh"])
def test_each_side_starts_under_its_own_flags_and_picks_its_own_law(
    monkeypatch, tmp_path, script
):
    sides = started(tmp_path, script)
    (p_argv, p_env), (d_argv, d_env) = sides["prefill"], sides["decode"]
    assert "--enable-tbo" in p_argv and "--cudagraph-capture-sizes" not in p_argv
    assert "--enable-tbo" not in d_argv and DECODE_SIZES in d_argv
    assert (
        PREFILL_ENV.items() <= p_env.items() and not PREFILL_ENV.keys() & d_env.keys()
    )

    for law, (argv, env) in zip(LAWS, sides.values()):
        for name in PERFORMANCE_KEYS:
            monkeypatch.delenv(name, raising=False)
        for name in PERFORMANCE_KEYS & env.keys():
            monkeypatch.setenv(name, env[name])
        here = keyed(side_config(argv))
        assert (
            compass_run._fitted_here(LAWS, here, lambda e: e["engine_config"], "law")
            is law
        )
        hosts = [e["host"] for e in LAWS]
        assert (
            compass_run._fitted_here(hosts, here, lambda e: e["key"]["engine"], "host")
            is law["host"]
        )
