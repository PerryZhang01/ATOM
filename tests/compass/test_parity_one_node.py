# SPDX-License-Identifier: MIT
"""`parity_one_node.sh`: what each CONFIG starts both servers with, the DP
width the simulated run file names, and the memory measurement it starts first
when STORE is unset.

`python3` is stubbed for the measurement, the servers, the traffic and the
compare, so these run with no GPU and no model; the rest of the script runs as written.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from test_drive_gpu import GPU45, SHOWHW

ROOT = Path(__file__).resolve().parents[2]
STUB = """#!/bin/bash
case "$1 $2" in
"-m atom.entrypoints.openai.api_server")
    echo "$* GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES-} ATOM_NUMA_BIND=${ATOM_NUMA_BIND-}" >> "$STUB_LOG"
    echo "Uvicorn running"
    case "$*" in *--compass-run*) exit 0 ;; esac
    exec sleep 60 ;;
"-m atom.compass.parity.traffic")
    [ -n "${ATOM_COMPASS_RUN-}" ] && exit 0
    mkdir -p "$5" && echo '{"sent_ns": [0], "admitted_ns": [2000000]}' > "$5/arrivals.json"
    exit 0 ;;
"-m atom.compass.parity")
    echo '{"steps": {"0": {"compared": 3, "identical": 3}}, "first_divergence": {"0": null}}'
    exit 0 ;;
"-c from atom.compass.cli import main; main()")
    echo "${*//$'\\n'/ } GPU_MAX_HW_QUEUES=${GPU_MAX_HW_QUEUES-} ATOM_NUMA_BIND=${ATOM_NUMA_BIND-}" >> "$STUB_LOG"
    echo "${ATOM_KV_OFFLOAD-}" > "$STUB_LOG.offload"
    exit "${MEASURE_RC:-0}" ;;
esac
exec "$REAL_PYTHON" "$@"
"""


def _parity(tmp_path, showhw=SHOWHW, **env):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "python3").write_text(STUB)
    (bindir / "python3").chmod(0o755)
    (tmp_path / "showhw").write_text(showhw)
    (bindir / "rocm-smi").write_text(f"#!/bin/sh\ncat {tmp_path}/showhw\n")
    (bindir / "rocm-smi").chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"architectures": ["DeepseekV4ForCausalLM"]}')
    inherited = (
        "CONFIG",
        "MACHINE",
        "SERVER_ARGS",
        "GPU_MAX_HW_QUEUES",
        "ATOM_NUMA_BIND",
        "HIP_VISIBLE_DEVICES",
    )
    env = {k: v for k, v in os.environ.items() if k not in inherited} | dict(
        PATH=f"{bindir}:{os.environ['PATH']}",
        REAL_PYTHON=sys.executable,
        STUB_LOG=str(tmp_path / "servers.log"),
        MODEL=str(model),
        WAIT_S="30",
        **{"STORE": str(tmp_path / "store")} | env,
    )
    env = {k: v for k, v in env.items() if v is not None}
    r = subprocess.run(
        ["bash", "scripts/compass/parity_one_node.sh", str(tmp_path / "out")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    log = tmp_path / "servers.log"
    servers = log.read_text().splitlines() if log.exists() else []
    return r, servers, tmp_path / "out" / "sim" / "run.json"


def test_tp2_is_the_default(tmp_path):
    r, servers, run = _parity(tmp_path)
    assert r.returncode == 0, r.stderr
    assert len(servers) == 2 and "--compass-run" in servers[1]
    for line in servers:
        assert " -tp 2 --enable-dp-attention --enforce-eager " in line
        assert line.endswith("GPU_MAX_HW_QUEUES= ATOM_NUMA_BIND=")
    assert json.loads(run.read_text())["data_parallel_size"] == 2
    assert r.stdout.splitlines()[-1] == (
        "parity_one_node: dp0 compared=3 identical=3 first_divergence=None"
    )


def test_tp2_plain_drops_dp_attention_everywhere(tmp_path):
    r, servers, run = _parity(tmp_path, CONFIG="tp2-plain", STORE=None)
    assert r.returncode == 0, r.stderr
    assert len(servers) == 3 and " measure memory " in servers[0]
    for line in servers:
        assert " -tp 2 --enforce-eager " in line
        assert "--enable-dp-attention" not in line
    assert json.loads(run.read_text())["data_parallel_size"] == 1


def test_dsv4_dpa8_runs_the_nightly_prefill_server_at_eight_ranks(tmp_path):
    machine = ROOT / "scripts/compass/machines/mi308x-hjbog18.json"
    r, servers, run = _parity(tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(machine))
    assert r.returncode == 0, r.stderr
    assert len(servers) == 2 and "--compass-run" in servers[1]
    for line in servers:
        assert " -tp 8 --enable-dp-attention --trust-remote-code " in line
        assert (
            "--kv_cache_dtype fp8 --block-size 16 --gpu-memory-utilization 0.85" in line
        )
        assert "--max-num-seqs 512 --enable-tbo" in line
        assert "--enforce-eager" not in line and "--enable-expert-parallel" not in line
        assert line.endswith("GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1")
    spec = json.loads(run.read_text())
    assert spec["data_parallel_size"] == 8
    assert spec["store"] == str(tmp_path / "store")
    assert spec["machine"]["name"] == json.loads(machine.read_text())["name"]


def test_dsv4_dpa8_without_a_machine_starts_nothing(tmp_path):
    r, servers, _ = _parity(tmp_path, CONFIG="dsv4-dpa8")
    assert r.returncode != 0
    assert "needs MACHINE" in r.stderr
    assert servers == []


def test_a_spec_that_measured_the_models_tokenizer_names_it_once(tmp_path):
    arch = "DeepseekV4ForCausalLM"
    machine = json.loads(
        (ROOT / "scripts/compass/machines/mi308x-hjbog18.json").read_text()
    )
    machine["host"]["tokenizers"][0]["applies_to"].append(arch)
    (tmp_path / "spec.json").write_text(json.dumps(machine))
    r, _, run = _parity(
        tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(tmp_path / "spec.json")
    )
    assert r.returncode == 0, r.stderr
    tokenizers = json.loads(run.read_text())["machine"]["host"]["tokenizers"]
    assert [a for t in tokenizers for a in t["applies_to"]].count(arch) == 1


def test_unset_store_measures_first_with_the_servers_flags_and_environment(tmp_path):
    machine = ROOT / "scripts/compass/machines/mi308x-hjbog18.json"
    r, servers, run = _parity(
        tmp_path, CONFIG="dsv4-dpa8", MACHINE=str(machine), STORE=None
    )
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    measure, real, sim = servers
    assert (
        f" measure memory --gpu 0,1,2,3,4,5,6,7 --machine {machine} "
        f"--store {out}/store --out {out}/memory -- --model {tmp_path}/model -tp 8 "
    ) in measure
    flags = measure.split(" -tp ", 1)[1]
    assert real.split(" -tp ", 1)[1] == flags
    assert sim.split(" -tp ", 1)[1].replace(f" --compass-run {run}", "") == flags
    assert json.loads(run.read_text())["store"] == f"{out}/store"


def test_the_measurement_takes_the_real_runs_gpus(tmp_path):
    r, servers, _ = _parity(tmp_path, STORE=None, HIP_VISIBLE_DEVICES="4,5")
    assert r.returncode == 0, r.stderr
    assert " measure memory --gpu 4,5 " in servers[0]
    assert len(servers) == 3


def test_the_measurement_names_the_real_runs_hip_devices_as_rocm_smi_gpus(tmp_path):
    r, servers, _ = _parity(tmp_path, GPU45, STORE=None, HIP_VISIBLE_DEVICES="0,1")
    assert r.returncode == 0, r.stderr
    assert " measure memory --gpu 3,0 " in servers[0]


def test_a_failed_measurement_starts_no_server(tmp_path):
    r, servers, run = _parity(tmp_path, STORE=None, MEASURE_RC="3")
    assert r.returncode != 0
    assert len(servers) == 1 and " measure memory " in servers[0]
    assert not run.exists()


@pytest.mark.parametrize("config, writers", [("tp2", 1), ("tp2-plain", 2)])
def test_lmcache_writes_from_each_dp_rank_under_dp_attention_else_each_tp_rank(
    tmp_path, monkeypatch, config, writers
):
    lm = tmp_path / "lm"
    lm.mkdir()
    (lm / "lmcache").write_text("#!/bin/bash\nexec sleep 60\n")
    (lm / "lmcache").chmod(0o755)
    monkeypatch.setenv("PATH", f"{lm}:{os.environ['PATH']}")
    r, _, run = _parity(tmp_path, CONFIG=config, LMCACHE="1")
    assert r.returncode == 0, r.stderr
    assert json.loads(run.read_text())["lmcache"]["writers"] == writers


def test_under_lmcache_the_measurement_offloads_to_a_server_of_its_own(
    tmp_path, monkeypatch
):
    lm = tmp_path / "lm"
    lm.mkdir()
    (lm / "lmcache").write_text("#!/bin/bash\nexec sleep 60\n")
    (lm / "lmcache").chmod(0o755)
    monkeypatch.setenv("PATH", f"{lm}:{os.environ['PATH']}")
    r, servers, run = _parity(tmp_path, LMCACHE="1", STORE=None)
    assert r.returncode == 0, r.stderr
    measure = servers[0]
    beside = measure.split(" --beside ", 1)[1].split(" --store ", 1)[0].split()
    server = json.loads(run.read_text())["lmcache"]["server"]
    assert beside == ["lmcache", "server", *server]
    assert (tmp_path / "servers.log.offload").read_text() == "lmcache_mp\n"


def test_rapidserve_measures_a_plain_engine_and_compares_each_engine(tmp_path):
    r, servers, run = _parity(tmp_path, CONFIG="rapidserve", STORE=None)
    assert r.returncode == 0, r.stderr
    measure, real, sim = servers
    flags = "-tp 1 --enforce-eager --level 0 --max-model-len 8192"
    assert f" {flags} " in measure and "--enable-rapidserve" not in measure
    for line in (real, sim):
        assert f" {flags} " in line and " --enable-rapidserve" in line
    assert json.loads(run.read_text())["rapidserve"] is True
    record = tmp_path / "out" / "real" / "record"
    assert all((record / e / "arrivals.json").exists() for e in ("prefill", "decode"))
    assert r.stdout.splitlines()[-2:] == [
        f"parity_one_node: {e} dp0 compared=3 identical=3 first_divergence=None"
        for e in ("decode", "prefill")
    ]
