# SPDX-License-Identifier: MIT
"""A GPU command's checks before and after its run: `preflight.sh` runs for
real, with `rocm-smi` and `rocminfo` planted first on PATH, so these need no
GPU. The planted `rocm-smi` prints its N-th process listing on its N-th call
and the last one after that."""

import argparse
import os
import shutil
import subprocess

import pytest

from atom.compass.drive.gpu import (
    PREFLIGHT,
    GpuRefusal,
    add_gpu_flag,
    hip_visible,
    quiet,
)

FREE = "No KFD PIDs currently running\n"
#: As node 18's rocm-smi prints it, a process on GPU 7 added; a process on no
#: GPU has no list after its line.
LISTING = """\
KFD process information:
PID    \tPROCESS NAME\tGPU(s)\tVRAM USED\tSDMA USED\tCU OCCUPANCY\t
3362894\tpython      \t0     \t0        \t0        \t0           \t
4242\tpython      \t1     \t1073741824\t0        \t0           \t
================================== GPUs Indexed by PID ===================================
PID 3362894 is using 0 DRM device(s)
PID 4242 is using 1 DRM device(s):
7
"""
EIGHT = (0, 1, 2, 3, 4, 5, 6, 7)
#: `rocm-smi --showhw` on node 18 (MI308X), whose GPUs' KFD nodes run in
#: rocm-smi's order, so each GPU's HIP ordinal is its rocm-smi index.
SHOWHW = """\


====================================== ROCm System Management Interface ======================================
=========================================== Concise Hardware Info ============================================
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS  VBIOS             BUS           PARTITION ID
0    2     0x74a2  63420  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0000:0A:00.0  0
1    3     0x74a2  61531  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0000:80:00.0  0
2    4     0x74a2  60224  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0000:A4:00.0  0
3    5     0x74a2  50797  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0000:C8:00.0  0
4    6     0x74a2  61984  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0001:0B:00.0  0
5    7     0x74a2  62919  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0001:81:00.0  0
6    8     0x74a2  61148  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0001:A5:00.0  0
7    9     0x74a2  50161  gfx942   ENABLED  ENABLED   ENABLED  113-M3080202-101  0001:C9:00.0  0
==============================================================================================================
============================================ End of ROCm SMI Log =============================================
"""
#: The same on mi355-gpu-45 (MI355X), where they do not: HIP device 0 is
#: rocm-smi GPU 3 (bus 0x75, KFD node 2), and rocm-smi GPUs 0 and 1 are HIP
#: devices 1 and 3.
GPU45 = """\


======================================== ROCm System Management Interface ========================================
============================================= Concise Hardware Info ==============================================
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS  VBIOS                 BUS           PARTITION ID
0    3     0x75a3  42583  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:05:00.0  0
1    5     0x75a3  27226  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:15:00.0  0
2    4     0x75a3  36479  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:65:00.0  0
3    2     0x75a3  17010  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:75:00.0  0
4    7     0x75a3  1590   gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:85:00.0  0
5    9     0x75a3  51771  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:95:00.0  0
6    8     0x75a3  11806  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:E5:00.0  0
7    6     0x75a3  57875  gfx950   ENABLED  ENABLED   ENABLED  113-M355-01-1K1-030A  0000:F5:00.0  0
==================================================================================================================
=============================================== End of ROCm SMI Log ===============================================
"""

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _plant(tmp_path, monkeypatch, *listings, rocminfo_rc=0):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for i, text in enumerate(listings):
        (bindir / f"listing{i}").write_text(text)
    calls = bindir / "calls"
    (bindir / "showhw").write_text(SHOWHW)
    (bindir / "rocm-smi").write_text(f"""#!/bin/bash
case "$*" in *--showhw*) cat {bindir}/showhw;;
*--showpids*)
    n=$(cat {calls} 2>/dev/null || echo 0); echo $((n + 1)) > {calls}
    f={bindir}/listing$n; [ -e "$f" ] || f={bindir}/listing{len(listings) - 1}
    cat "$f";;
esac
""")
    (bindir / "rocminfo").write_text(f"#!/bin/sh\nexit {rocminfo_rc}\n")
    for tool in ("rocm-smi", "rocminfo"):
        (bindir / tool).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    out = tmp_path / "out"
    out.mkdir()
    return out


def test_a_foreign_pid_on_gpu_7_refuses_before_the_run_starts(tmp_path, monkeypatch):
    out = _plant(tmp_path, monkeypatch, LISTING)
    started = []
    with pytest.raises(GpuRefusal) as refused, quiet(EIGHT, out):
        started.append(True)
    assert str(refused.value) == (
        "GPU 7 has PID 4242 on it before the run, so it was not started"
    )
    assert not started
    assert (out / "gpu_before.txt").read_text() == LISTING
    assert "PREFLIGHT_RC=0" in (out / "preflight_before.txt").read_text()
    assert not (out / "gpu_after.txt").exists()


def test_a_process_on_an_unnamed_gpu_leaves_the_run_alone(tmp_path, monkeypatch):
    out = _plant(tmp_path, monkeypatch, LISTING)
    with quiet((0, 6), out):
        pass
    assert (out / "gpu_before.txt").read_text() == LISTING
    assert (out / "gpu_after.txt").read_text() == LISTING


def test_a_process_on_a_named_gpu_at_the_stop_fails_the_run(tmp_path, monkeypatch):
    out = _plant(tmp_path, monkeypatch, FREE, LISTING)
    ran = []
    after = "^GPU 7 has PID 4242 on it after the run, so it fails$"
    with pytest.raises(GpuRefusal, match=after), quiet((6, 7), out):
        ran.append(True)
    assert ran
    assert (out / "gpu_before.txt").read_text() == FREE
    assert (out / "gpu_after.txt").read_text() == LISTING


def test_the_stop_is_checked_however_the_run_ends(tmp_path, monkeypatch):
    out = _plant(tmp_path, monkeypatch, FREE)
    with pytest.raises(KeyboardInterrupt), quiet((7,), out):
        raise KeyboardInterrupt
    assert (out / "gpu_after.txt").read_text() == FREE


def test_a_failed_preflight_refuses_before_any_listing(tmp_path, monkeypatch):
    out = _plant(tmp_path, monkeypatch, FREE, rocminfo_rc=1)
    failed = "^preflight.sh failed before the run"
    with pytest.raises(GpuRefusal, match=failed), quiet((7,), out):
        pytest.fail("the run started")
    assert "PREFLIGHT_RC=1" in (out / "preflight_before.txt").read_text()
    assert not (out / "gpu_before.txt").exists()


def test_the_gpu_flag_is_required_and_has_no_default(capsys):
    parser = argparse.ArgumentParser()
    add_gpu_flag(parser)
    assert parser.parse_args(["--gpu", "6,7"]).gpu == (6, 7)
    for argv in ([], ["--gpu", ""], ["--gpu", "7,7"], ["--gpu", "a"]):
        with pytest.raises(SystemExit):
            parser.parse_args(argv)
    assert "--gpu" in capsys.readouterr().err


@pytest.mark.parametrize(
    "table, gpus, hip",
    [(SHOWHW, (0, 1), "0,1"), (GPU45, (0, 1), "1,3"), (GPU45, (7, 3), "4,0")],
)
def test_a_gpu_s_hip_ordinal_is_its_kfd_node_s_rank(
    tmp_path, monkeypatch, table, gpus, hip
):
    _plant(tmp_path, monkeypatch, FREE)
    (tmp_path / "bin" / "showhw").write_text(table)
    assert hip_visible(gpus) == hip


def test_a_gpu_rocm_smi_does_not_list_refuses(tmp_path, monkeypatch):
    _plant(tmp_path, monkeypatch, FREE)
    with pytest.raises(GpuRefusal, match=r"^rocm-smi --showhw lists no GPU \[8\]$"):
        hip_visible((0, 8))


def test_preflight_exports_the_hip_devices_of_the_free_rocm_smi_gpus(
    tmp_path, monkeypatch
):
    _plant(tmp_path, monkeypatch, FREE)
    bindir = tmp_path / "bin"
    (bindir / "showhw").write_text(GPU45)
    # rocm-smi GPUs 0 and 1 idle and empty, the rest busy.
    use = "".join(f"GPU[{i}]\t\t: GPU use (%): {0 if i < 2 else 100}\n" for i in EIGHT)
    (bindir / "use").write_text(use)
    (bindir / "memuse").write_text(
        "".join(f"GPU[{i}]\t\t: GPU Memory Allocated (VRAM%): 0\n" for i in EIGHT)
    )
    smi = (bindir / "rocm-smi").read_text()
    smi = smi.replace(
        "*--showhw*)",
        f"*--showuse*) cat {bindir}/use;;\n*--showmemuse*) cat {bindir}/memuse;;\n"
        "*--showhw*)",
    )
    (bindir / "rocm-smi").write_text(smi)
    flight = subprocess.run(
        ["bash", str(PREFLIGHT)], capture_output=True, text=True, check=True
    )
    assert "idle and empty: 0 1\nexport HIP_VISIBLE_DEVICES=1,3\n" in flight.stdout
