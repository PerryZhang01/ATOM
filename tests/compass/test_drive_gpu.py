# SPDX-License-Identifier: MIT
"""A GPU command's checks before and after its run: `preflight.sh` runs for
real, with `rocm-smi` and `rocminfo` planted first on PATH, so these need no
GPU. The planted `rocm-smi` prints its N-th process listing on its N-th call
and the last one after that."""

import argparse
import os
import shutil

import pytest

from atom.compass.drive.gpu import GpuRefusal, add_gpu_flag, quiet

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

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _plant(tmp_path, monkeypatch, *listings, rocminfo_rc=0):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for i, text in enumerate(listings):
        (bindir / f"listing{i}").write_text(text)
    calls = bindir / "calls"
    (bindir / "rocm-smi").write_text(f"""#!/bin/bash
case "$*" in *--showpids*)
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
