# SPDX-License-Identifier: MIT
"""`gate_cpu.sh`'s verdict has to survive being piped, and say why.

A pipeline's status is its last command's, so `gate_cpu.sh | tail -6` hands
the caller tail's 0 whatever the gate exited. The script cannot change that.
What it controls is the text: its last line is the verdict, and that line has
to carry the reason, because the three common pipes keep different halves of
the output -- `2>/dev/null | tail -6` drops every stderr line, which on a run
that exits 98 leaves the verdict number under pytest's green summary.

Each case runs the real script over a throwaway tree: a one-test suite, an
empty exclusion list, and a trigger list that the stamped diff touches, so the
run is green and then exits 98 for the GPU gate. Nothing needs a driver.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
BLIND = "atom/blind.py"

pytestmark = pytest.mark.skipif(
    BASH is None or shutil.which("git") is None, reason="needs bash and git"
)


def _tree(root, stamped):
    """A minimal checkout the gate accepts, green, touching one GPU trigger."""
    shutil.copytree(REPO / "scripts" / "compass", root / "scripts" / "compass")
    (root / "scripts" / "compass" / "cpu_gate_exclude.txt").write_text("")
    (root / "scripts" / "compass" / "gpu_gate_triggers.txt").write_text(BLIND + "\n")
    (root / "atom").mkdir()
    (root / "atom" / "__init__.py").write_text("")
    (root / "tests").mkdir()
    (root / "tests" / "test_ok.py").write_text("def test_ok():\n    pass\n")
    if stamped:
        (root / ".compass-changed").write_text(BLIND + "\n")
        (root / ".compass-commit").write_text("0" * 40 + "\n")
    return root


def _run(root, command):
    env = {k: v for k, v in os.environ.items() if not k.startswith("COMPASS_")}
    env["PATH"] = os.path.dirname(sys.executable) + os.pathsep + env["PATH"]
    # A checkout above the temp dir must not make the bare tree look like git.
    env["GIT_CEILING_DIRECTORIES"] = str(root.parent)
    return subprocess.run(
        [BASH, "-c", command.format(gate="scripts/compass/gate_cpu.sh")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


@pytest.fixture(scope="module")
def gpu_tree(tmp_path_factory):
    return _tree(tmp_path_factory.mktemp("gate") / "ATOM", stamped=True)


def test_unpiped_the_gate_exits_98(gpu_tree):
    # The control: the status every piped case below loses.
    out = _run(gpu_tree, "{gate}")
    assert out.returncode == 98, out.stdout + out.stderr


def test_the_verdict_key_is_printed_exactly_once(gpu_tree):
    # A reader taking the first match must land on the verdict, not on prose
    # that mentions the key.
    out = _run(gpu_tree, "{gate} 2>&1")
    keyed = [line for line in out.stdout.splitlines() if "GATE_CPU_RC=" in line]
    assert len(keyed) == 1, keyed


@pytest.mark.parametrize(
    "pipe", ["| tail -6", "2>&1 | tail -6", "2>/dev/null | tail -6"]
)
def test_a_piped_run_still_reads_as_not_passed_and_says_why(gpu_tree, pipe):
    out = _run(gpu_tree, "{gate} " + pipe)
    assert out.returncode == 0, "the pipe kept the gate's status; recheck the premise"
    kept = out.stdout.splitlines()
    assert kept, "the pipe kept nothing"
    verdict = kept[-1]
    assert verdict.startswith("GATE_CPU_RC=98 NOT PASSED"), kept
    assert "gate_gpu.sh" in verdict, (
        f"`{pipe}` keeps a verdict with no reason: {verdict!r}. Beside the "
        "pipe's own 0 and pytest's green summary, a bare 98 is all the reader "
        "has, so the reason has to travel on the verdict line."
    )


@pytest.mark.parametrize(
    "orphan, cause",
    [(False, "resolves to no commit"), (True, "shares no commit with HEAD")],
)
def test_a_git_checkout_is_not_called_unstamped(tmp_path, orphan, cause):
    root = _tree(tmp_path / "ATOM", stamped=False)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(git + ["init", "-q"], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "t"], check=True)
    if orphan:
        # An integration branch with no history in common with HEAD.
        tree = git + ["commit-tree", "HEAD^{tree}", "-m", "o"]
        sha = subprocess.run(tree, check=True, capture_output=True, text=True)
        branch = ["branch", "feature/atomcompass_new", sha.stdout.strip()]
        subprocess.run(git + branch, check=True)
    out = _run(root, "{gate}")
    gpu = [line for line in out.stdout.splitlines() if line.startswith("gpu:")]
    assert out.returncode == 98 and gpu, out.stdout + out.stderr
    assert "never stamped" not in gpu[0], gpu[0]
    assert f"feature/atomcompass_new {cause}" in gpu[0], gpu[0]


def test_a_tree_with_no_git_and_no_stamp_is_still_called_unstamped(tmp_path):
    out = _run(_tree(tmp_path / "ATOM", stamped=False), "{gate}")
    gpu = [line for line in out.stdout.splitlines() if line.startswith("gpu:")]
    assert out.returncode == 98 and gpu, out.stdout + out.stderr
    assert "never stamped" in gpu[0], gpu[0]
