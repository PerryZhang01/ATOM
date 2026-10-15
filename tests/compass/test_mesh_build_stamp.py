# SPDX-License-Identifier: MIT
"""atomesh's build stamp follows the git state at every incremental build.

`atom/mesh/build.rs` embeds the git commit and status in the binary, and
`atom/compass/run` refuses a router whose stamp is not clean or not the tree's.
The test builds a small crate with that build script, in a subdirectory of a
git repository of its own, and reads the stamp back after each change. Skips
where cargo cannot resolve the script's build dependencies offline.
"""

import pathlib
import shutil
import subprocess

import pytest

MESH = pathlib.Path(__file__).resolve().parents[2] / "atom" / "mesh"

CARGO_TOML = """\
[package]
name = "stamp"
version = "0.1.0"
edition = "2021"

[features]
ext-proc = []

[build-dependencies]
chrono = { version = "0.4", features = ["clock"] }
toml = "0.9"
"""

MAIN_RS = """\
fn main() {
    println!("{} {}", env!("ATOM_MESH_GIT_COMMIT"), env!("ATOM_MESH_GIT_STATUS"));
}
"""


def _git(repo, *args):
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_an_incremental_build_restamps_after_an_edit_and_after_a_commit(tmp_path):
    if not (shutil.which("cargo") and shutil.which("git")):
        pytest.skip("no cargo or git here")
    repo, crate = tmp_path, tmp_path / "mesh"
    (crate / "src").mkdir(parents=True)
    (crate / "Cargo.toml").write_text(CARGO_TOML)
    (crate / "src" / "main.rs").write_text(MAIN_RS)
    shutil.copy(MESH / "build.rs", crate / "build.rs")
    if (MESH / "Cargo.lock").exists():
        shutil.copy(MESH / "Cargo.lock", crate / "Cargo.lock")
    (repo / ".gitignore").write_text("target/\nCargo.lock\n")
    resolved = subprocess.run(
        ["cargo", "metadata", "--offline", "--format-version", "1"],
        cwd=crate,
        capture_output=True,
        check=False,
        timeout=600,
    )
    if resolved.returncode:
        pytest.skip("cargo cannot resolve build.rs's dependencies offline here")
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "first")

    def stamp():
        """The built binary's (commit, status), and whether build.rs ran."""
        run = subprocess.run(
            ["cargo", "run", "--offline", "-v"],
            cwd=crate,
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
        return tuple(run.stdout.split()), "build-script-build`" in run.stderr

    first = _git(repo, "rev-parse", "--short", "HEAD")
    assert stamp() == ((first, "clean"), True)
    assert stamp() == ((first, "clean"), False), "nothing changed, build.rs reran"

    (crate / "src" / "probe.txt").write_text("probe\n")
    assert stamp() == ((first, "dirty"), True)

    (repo / "elsewhere.txt").write_text("outside the crate\n")
    _git(repo, "add", "mesh")
    _git(repo, "commit", "-q", "-m", "second")
    second = _git(repo, "rev-parse", "--short", "HEAD")
    assert second != first
    assert stamp() == ((second, "clean"), True)
