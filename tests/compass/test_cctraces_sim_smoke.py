# SPDX-License-Identifier: MIT
"""``scripts/compass/cctraces_sim.sh``: one cc-traces session through
agentx-harness on a simulated TP1 run, twice, with different ``PYTHONHASHSEED``.

Each run is the script end to end: ATOM's API server with ``--compass-run``,
``aiperf profile`` with the AgentX scenario and compass-harness, ended by a
short simulated duration. The two step tables must be byte-identical and carry
the workload the script hashed into the run file.

Skips without a driver (the engine core and the TP1 worker import aiter), and
without ``ATOM_COMPASS_SLICE_MODEL`` (a model directory holding a config and a
tokenizer), ``ATOM_COMPASS_CCTRACES`` (a cc-traces ``traces.jsonl``) and
``COMPASS_HARNESS_PYTHON`` (a Python with agentx-harness 56a0cf70 and
compass-harness installed).

``test_a_1p1d_pair_answers_through_the_router_and_transfers_kv`` runs the same
pair as a 1P1D cell, through `pd_sim.sh` and atomesh, which must be on PATH;
``ATOM_COMPASS_PD_DECODE_EXEC`` runs decode in another container, as
``test_pd_slice.py`` does.

``test_a_real_cell_warms_up_on_the_simulated_cells_requests`` runs the cell
once more with ``REAL=1`` on zero dummy weights, which needs the GPU free of
anything else, and ``COMPASS_AIPERF_PYTHON``, a Python with agentx-harness
56a0cf70 and without compass-harness.
"""

import copy
import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from test_memory_readings import DOCUMENT

from atom.compass import run as compass_run
from atom.compass.backends.shape import Coefficients
from atom.compass.detect.determinism import CONFIGURATION_PREFIX

MODEL = os.environ.get("ATOM_COMPASS_SLICE_MODEL")
TRACES = os.environ.get("ATOM_COMPASS_CCTRACES")
HARNESS_PYTHON = os.environ.get("COMPASS_HARNESS_PYTHON")
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL and TRACES and HARNESS_PYTHON),
    reason="needs a driver, ATOM_COMPASS_SLICE_MODEL, ATOM_COMPASS_CCTRACES "
    "and COMPASS_HARNESS_PYTHON",
)

TREE = Path(__file__).resolve().parents[2]
SCRIPT = TREE / "scripts/compass/cctraces_sim.sh"


def _template(path: Path, **extra) -> Path:
    """A run file whose declared tokenizer also covers the model's architecture."""
    machine = copy.deepcopy(DOCUMENT)
    arch = json.loads((Path(MODEL) / "config.json").read_text())["architectures"][0]
    machine["host"]["tokenizers"][0]["applies_to"].append(arch)
    run = {
        "bound_s": 1e7,
        "admission_path": "serving",
        "ipc_s": 2.0**-14,
        "stream_s": 2.0**-12,
        "coefficients": dataclasses.asdict(Coefficients()),
        "machine": machine,
        "parameter_count": 8_000_000_000,
        **extra,
    }
    path.write_text(json.dumps(run))
    return path


def _pair(root: Path, template: Path, **extra_env) -> list:
    """The script run twice, `PYTHONHASHSEED` 1 and 2: each cell and its result line."""
    out = []
    for seed in ("1", "2"):
        cell = root / f"seed{seed}"
        env = dict(
            os.environ,
            **extra_env,
            PYTHONHASHSEED=seed,
            MODEL=MODEL,
            TRACES=TRACES,
            HARNESS_PYTHON=HARNESS_PYTHON,
            SESSIONS="1",
            DURATION_S="60",
        )
        done = subprocess.run(
            ["bash", str(SCRIPT), str(template), str(cell)],
            cwd="/",
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
        out.append((cell, done.stdout.strip().splitlines()[-1]))
    return out


@pytest.fixture(scope="module")
def cells(tmp_path_factory):
    root = tmp_path_factory.mktemp("cctraces")
    return _pair(root, _template(root / "template.json"))


def test_both_runs_answer_every_request_and_refuse_nothing(cells):
    for _, result in cells:
        fields = dict(f.split("=", 1) for f in result.split()[1:])
        assert int(fields["requests"]) > 0, result
        assert (fields["errors"], fields["refusals"]) == ("0", "0"), result
        assert fields["coverage_report"] == "False", result


def test_the_step_records_are_byte_identical_and_carry_no_timestamps(cells):
    (left, _), (right, _) = cells
    record = (left / "steps/dp0.jsonl").read_bytes()
    assert record == (right / "steps/dp0.jsonl").read_bytes()
    steps = [json.loads(line) for line in record.splitlines()]
    assert steps and all("rows" in s and "t_enter_ns" not in s for s in steps)


def _warmup(cell: Path) -> list:
    """(conversation, turn) of each request aiperf sent in its warmup phase."""
    export = (cell / "artifacts/profile_export.jsonl").read_text().splitlines()
    meta = [json.loads(line)["metadata"] for line in export]
    return sorted(
        (m["conversation_id"], m["turn_index"])
        for m in meta
        if m["benchmark_phase"] == "warmup"
    )


@pytest.mark.skipif(
    not os.environ.get("COMPASS_AIPERF_PYTHON"),
    reason="a real cell needs COMPASS_AIPERF_PYTHON",
)
def test_a_real_cell_warms_up_on_the_simulated_cells_requests(cells, tmp_path):
    (sim, _), _ = cells
    # A rocm-smi that lists pid 4242 on the cell's GPU and pid 1337009 on another,
    # padded as rocm-smi pads a pid shorter than the PID column.
    gpu = (os.environ.get("HIP_VISIBLE_DEVICES") or "0").split(",")[0]
    listing = (
        "KFD process information:\n"
        "PID    \tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY\t\n"
        "4242   \tUNKNOWN     \t1     \t481734656\t0        \t0           \t\n"
        "1337009\tUNKNOWN     \t1     \t1024     \t0        \t0           \t\n"
        "=== GPUs Indexed by PID ===\n"
        f"PID 4242 is using 1 DRM device(s):\n{gpu} \n"
        f"PID 1337009 is using 1 DRM device(s):\n{int(gpu) + 1} \n"
    )
    stub = tmp_path / "bin/rocm-smi"
    stub.parent.mkdir()
    stub.write_text(f"#!/bin/sh\ncat <<'EOF'\n{listing}EOF\n")
    stub.chmod(0o755)
    env = dict(
        os.environ,
        PATH=f"{stub.parent}:{os.environ['PATH']}",
        REAL="1",
        MODEL=MODEL,
        TRACES=TRACES,
        HARNESS_PYTHON=os.environ["COMPASS_AIPERF_PYTHON"],
        SESSIONS="1",
        DURATION_S="60",
        SERVER_ARGS="--enforce-eager --max-model-len 262144 --load_dummy=zero",
    )
    real = tmp_path / "real"
    done = subprocess.run(
        ["bash", str(SCRIPT), str(sim.parent / "template.json"), str(real)],
        cwd="/",
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    result = done.stdout.strip().splitlines()[-1]
    fields = dict(f.split("=", 1) for f in result.split()[1:])
    assert int(fields["requests"]) > 0 and fields["errors"] == "0", result
    assert fields["contaminated"] == "True", result
    assert fields["gpu_before"] == fields["gpu_after"] == "4242:481734656", result
    assert not (real / compass_run.STEP_TABLE_FILE).exists()
    workload = [
        json.loads((c / "run.json").read_text())["workload"] for c in (sim, real)
    ]
    assert workload[0] == workload[1]
    steps = [json.loads(line) for line in (real / "steps/dp0.jsonl").open()]
    assert steps and all(s["t_exit_ns"] >= s["t_enter_ns"] for s in steps)
    assert _warmup(real) == _warmup(sim) != []


def test_the_step_tables_are_byte_identical_and_name_the_workload(cells):
    (left, _), (right, _) = cells
    table = (left / compass_run.STEP_TABLE_FILE).read_bytes()
    assert table == (right / compass_run.STEP_TABLE_FILE).read_bytes()
    first = table.decode().splitlines()[0]
    workload = json.loads(first[len(CONFIGURATION_PREFIX) :])["workload"]
    trace = (left / "traces/00000.json").read_bytes()
    assert workload["traces_sha256"] == [hashlib.sha256(trace).hexdigest()]
    assert "--benchmark-duration" in workload["aiperf_args"]


@pytest.mark.skipif(not shutil.which("atomesh"), reason="a 1P1D cell needs atomesh")
def test_a_1p1d_pair_answers_through_the_router_and_transfers_kv(tmp_path):
    template = _template(
        tmp_path / "template.json",
        router_s=2.0**-12,
        kv_write_req_s=2.0**-10,
        kv_link="intra_node",
    )
    exec_ = os.environ.get("ATOM_COMPASS_PD_DECODE_EXEC", "")
    (left, result), (right, _) = _pair(tmp_path, template, DECODE_EXEC=exec_)
    fields = dict(f.split("=", 1) for f in result.split()[1:])
    assert int(fields["requests"]) > 0, result
    assert (fields["errors"], fields["refusals"]) == ("0", "0"), result
    assert fields["coverage_report"] == "False", result
    assert int(fields["kv_transfers"]) >= int(fields["requests"]), result
    table = (left / compass_run.STEP_TABLE_FILE).read_bytes()
    assert table == (right / compass_run.STEP_TABLE_FILE).read_bytes()
    assert b" release engine-D->engine-P:kv_write_req " in table
