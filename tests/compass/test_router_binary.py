# SPDX-License-Identifier: MIT
"""A simulated run under the router LP names the router binary it ran, and
refuses one not built from the tree's ``atom/mesh``."""

import json
import os
import socket
import subprocess
from pathlib import Path

import pytest

from atom.compass.run import router_binary

TREE = Path(__file__).resolve().parents[2]


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def commit(repo, path, text) -> str:
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(text)
    git(repo, "add", "-A")
    git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", path)
    return git(repo, "rev-parse", "--short", "HEAD")


def binary(tmp_path, commit, status="clean") -> str:
    """A stand-in for atomesh that prints `--version-verbose` as atomesh does."""
    exe = tmp_path / f"atomesh-{commit}-{status}"
    exe.write_text(
        "#!/bin/sh\ncat <<'EOF'\natomesh 0.1.0\n\nVersion Control:\n"
        f"  Git Branch: main\n  Git Commit: {commit}\n  Git Status: {status}\n"
        "\nCompiler:\n  rustc 1.94.0\nEOF\n"
    )
    exe.chmod(0o755)
    return str(exe)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A tree whose router source changes in its third commit, not its second."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repo = tmp_path / "ATOM"
    repo.mkdir()
    git(repo, "init", "-q")
    a = commit(repo, "atom/mesh/src/lib.rs", "fn a() {}\n")
    b = commit(repo, "README.md", "elsewhere\n")
    return repo, a, b


def test_a_binary_built_at_an_earlier_commit_with_the_same_router_source_is_named(
    repo, tmp_path
):
    repo, a, _ = repo
    exe = binary(tmp_path, a)
    record = router_binary(exe, str(repo / "atom/mesh"))
    print("record:", record)
    assert record == {"path": exe, "commit": a, "status": "clean"}


def test_a_binary_whose_router_source_the_tree_has_since_changed_is_refused(
    repo, tmp_path
):
    repo, a, _ = repo
    commit(repo, "atom/mesh/src/lib.rs", "fn c() {}\n")
    exe = binary(tmp_path, a)
    with pytest.raises(ValueError, match=f"router binary {exe} .*differs") as e:
        router_binary(exe, str(repo / "atom/mesh"))
    print("refusal:", e.value)


def test_an_uncommitted_edit_to_the_trees_router_source_refuses_the_binary(
    repo, tmp_path
):
    repo, _, b = repo
    (repo / "atom/mesh/src/lib.rs").write_text("fn edited() {}\n")
    with pytest.raises(ValueError, match="differs"):
        router_binary(binary(tmp_path, b), str(repo / "atom/mesh"))


@pytest.mark.parametrize(
    "built, why",
    [
        (("{b}", "dirty"), "built from an uncommitted tree"),
        (("unknown", "clean"), "names no commit"),
        (("0123456789", "clean"), "router source at"),
    ],
)
def test_a_binary_built_dirty_or_at_a_commit_the_tree_lacks_is_refused_by_name(
    repo, tmp_path, built, why
):
    repo, _, b = repo
    exe = binary(tmp_path, built[0].format(b=b), built[1])
    with pytest.raises(ValueError, match=f"router binary {exe} .*{why}"):
        router_binary(exe, str(repo / "atom/mesh"))


def test_a_snapshot_matches_its_own_commit_only(repo, tmp_path):
    repo, a, b = repo
    snap = tmp_path / "snap"
    (snap / "atom/mesh").mkdir(parents=True)
    (snap / ".compass-commit").write_text(git(repo, "rev-parse", "HEAD") + "\n")
    assert router_binary(binary(tmp_path, b), str(snap / "atom/mesh"))["commit"] == b
    with pytest.raises(ValueError, match="not this snapshot's commit"):
        router_binary(binary(tmp_path, a), str(snap / "atom/mesh"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def pd_sim(tmp_path, exe, **cell) -> tuple[int, str]:
    run_file = tmp_path / "out" / "run.json"
    run_file.parent.mkdir()
    run_file.write_text("{}")
    env = dict(
        os.environ,
        MODEL=str(tmp_path),
        ATOMESH=exe,
        CLOCK_PORT=str(free_port()),
        WAIT_S="30",
        **cell,
    )
    done = subprocess.run(
        ["bash", str(TREE / "scripts/compass/pd_sim.sh"), str(run_file)],
        cwd=TREE,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    return done.returncode, done.stdout + done.stderr


def test_pd_sim_refuses_a_dirty_router_for_the_router_lp_before_starting_anything(
    tmp_path,
):
    exe = binary(tmp_path, "0123456789", "dirty")
    code, out = pd_sim(
        tmp_path,
        exe,
        SERVER_ARGS="--enable-dp-attention",
        BENCHMARK_KIND="aiperf_agentic",
    )
    print(out)
    assert code != 0
    assert f"pd_sim: router binary {exe}" in out
    assert not (tmp_path / "out" / "authority.log").exists()


def test_pd_sim_without_the_router_lp_takes_any_router(tmp_path):
    exe = binary(tmp_path, "0123456789", "dirty")
    code, out = pd_sim(tmp_path, exe, SERVER_ARGS="--enforce-eager")
    print(out)
    # It went on to the authority, which refuses the empty run file.
    assert code != 0
    assert "router binary" not in out
    assert "pd_sim: the authority exited" in out
    assert json.loads((tmp_path / "out" / "run.json").read_text()) == {}
