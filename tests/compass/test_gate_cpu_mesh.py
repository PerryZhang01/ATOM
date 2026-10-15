# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs atomesh's Compass cargo tests offline, or refuses when a
diff touching atom/mesh would leave them skipped.

Each case runs the real script over the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py`` plus an ``atom/mesh/Cargo.toml``, with
a stand-in ``cargo`` first on PATH that logs its environment and arguments and
exits as told. Nothing needs cargo or a network.
"""

from test_gate_cpu_verdict_through_pipe import BLIND, _run, _tree

CARGO = """#!/bin/sh
echo "$CARGO_NET_OFFLINE $*" >> "$0.log"
case "$1" in metadata) exit {metadata};; test) exit {test};; esac
"""
TARGETS = (
    "--test compass_clock_test --test compass_router_test"
    " --test compass_pd_router_test --test policy_determinism_test"
)


def _gate(tmp_path, changed, metadata=0, test=0):
    root = _tree(tmp_path / "ATOM", stamped=True)
    (root / ".compass-changed").write_text(changed + "\n")
    (root / "atom" / "mesh").mkdir()
    (root / "atom" / "mesh" / "Cargo.toml").write_text("")
    (tmp_path / "Cargo.lock").write_text("# image lock\n")
    cargo = tmp_path / "bin" / "cargo"
    cargo.parent.mkdir()
    cargo.write_text(CARGO.format(metadata=metadata, test=test))
    cargo.chmod(0o755)
    out = _run(
        root,
        f"COMPASS_CARGO_LOCK={tmp_path}/Cargo.lock PATH={cargo.parent}:$PATH {{gate}}",
    )
    log = cargo.with_suffix(".log")
    log = log.read_text().splitlines() if log.exists() else []
    return out, out.stdout.splitlines()[-1], log, root


def test_a_mesh_diff_whose_cargo_tests_would_skip_is_refused(tmp_path):
    out, verdict, log, _ = _gate(tmp_path, "atom/mesh/src/lib.rs", metadata=101)
    assert out.returncode == 96, out.stdout + out.stderr
    assert "touches atom/mesh and its cargo tests would skip" in verdict, verdict
    assert "pytest: rc=" not in out.stdout, out.stdout
    assert log == ["true metadata --format-version 1"], log


def test_an_unchanged_mesh_that_cannot_resolve_is_reported_not_refused(tmp_path):
    out, _, log, _ = _gate(tmp_path, BLIND, metadata=101)
    # 98 is the GPU-gate answer the stamped BLIND diff ends on, after pytest.
    assert out.returncode == 98, out.stdout + out.stderr
    assert "mesh:   cargo tests skipped, atom/mesh unchanged" in out.stdout
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert len(log) == 1, log


def test_the_compass_cargo_tests_run_offline_under_the_image_lock(tmp_path):
    out, _, log, root = _gate(tmp_path, "atom/mesh/src/lib.rs")
    assert out.returncode == 0, out.stdout + out.stderr
    assert log == ["true metadata --format-version 1", f"true test {TARGETS}"], log
    assert (root / "atom" / "mesh" / "Cargo.lock").read_text() == "# image lock\n"


def test_a_failing_cargo_test_fails_the_gate_after_pytest(tmp_path):
    out, verdict, _, _ = _gate(tmp_path, "atom/mesh/src/lib.rs", test=101)
    assert out.returncode == 101, out.stdout + out.stderr
    assert "pytest: rc=0" in out.stdout, out.stdout
    assert "cargo test failed in atom/mesh" in verdict, verdict
