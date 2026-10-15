# SPDX-License-Identifier: MIT
"""The Rust router pass: clean over atom/mesh/src, red when a substitution goes."""

import os
import shutil

import pytest

import atom
from atom.compass.detect import rust_router
from atom.compass.detect.rust_router import ALLOW_LIST, check, hits

MESH_SRC = os.path.join(os.path.dirname(atom.__file__), "mesh", "src")

#: The three reads the determinism module replaced, put back as the host
#: reads they were: (file, routed text, host text).
REVERTS = {
    "dp_sticky clock": (
        "policies/dp_sticky.rs",
        "crate::determinism::now()",
        "Instant::now()",
    ),
    "cache_aware rng": (
        "policies/cache_aware.rs",
        "crate::determinism::rng()",
        "rand::rng()",
    ),
    "tree hasher": (
        "policies/tree.rs",
        "DashMap::with_hasher_and_shard_amount(determinism::hasher(), NODE_SHARD_COUNT)",
        "DashMap::with_shard_amount(NODE_SHARD_COUNT)",
    ),
}


def test_the_router_is_clean_today():
    """This assertion is the gate."""
    code, report = check(MESH_SRC)
    assert code == 0, report


@pytest.mark.parametrize("name", sorted(REVERTS))
def test_reverting_a_determinism_substitution_turns_it_red(tmp_path, name):
    path, routed, host = REVERTS[name]
    root = shutil.copytree(MESH_SRC, tmp_path / "src")
    target = root / path
    text = target.read_text(encoding="utf-8")
    assert routed in text
    target.write_text(text.replace(routed, host, 1), encoding="utf-8")
    code, report = check(str(root))
    assert code == 1
    assert f"  UNKEPT {path}:" in report, report


def test_a_new_hit_in_a_kept_file_fails_with_the_kept_ones(tmp_path):
    path = "core/worker_builder.rs"
    count = ALLOW_LIST[(path, "RandomState")][2]
    (tmp_path / "core").mkdir()
    (tmp_path / path).write_text(
        "fn f() {\n" + "    let m = HashMap::new();\n" * (count + 1) + "}\n"
    )
    code, report = check(str(tmp_path))
    assert code == 1
    assert report.count(f"  UNKEPT {path}:") == count + 1


def test_routing_a_kept_decision_hit_leaves_a_stale_entry_that_fails(tmp_path):
    root = shutil.copytree(MESH_SRC, tmp_path / "src")
    target = root / "policies" / "random.rs"
    text = target.read_text(encoding="utf-8")
    target.write_text(
        text.replace("use rand::Rng;", "use rand_core::RngCore as _;").replace(
            "rand::rng()", "crate::determinism::rng()"
        ),
        encoding="utf-8",
    )
    code, report = check(str(root))
    assert code == 1
    assert "0 unkept, 1 stale entries" in report
    assert "  STALE policies/random.rs  rand::" in report


SOURCE = r"""
use std::time::Instant;
// Instant::now() in a comment
/* nested /* Instant::now() */ still a comment */
fn f() -> char {
    let s = "rand::rng() in a string";
    let r = r#"SystemTime::now()"#;
    let c = '"';
    let l: &'static str = "";
    tokio::select! { biased; _ = a => {} }
    tokio::select! { _ = a => {} }
    'x'
}
#[cfg(test)]
mod tests {
    fn g() { let t = Instant::now(); }
}
#[cfg(test)]
mod load_tests;
fn h() { let t = std::time::Instant::now(); }
"""


def test_literals_comments_biased_select_and_test_code_are_not_hits():
    assert hits(SOURCE) == (
        [(11, "unbiased select!"), (20, "Instant::now")],
        ["load_tests"],
    )


def test_a_test_module_file_is_skipped(tmp_path):
    (tmp_path / "router.rs").write_text("#[cfg(test)]\nmod load_tests;\n")
    (tmp_path / "router").mkdir()
    (tmp_path / "router" / "load_tests.rs").write_text("fn t() { Instant::now(); }\n")
    assert rust_router.scan(str(tmp_path)) == []


def test_a_root_holding_no_rust_file_is_refused(tmp_path):
    (tmp_path / "README.md").write_text("# docs\n")
    assert check(str(tmp_path)) == (
        1,
        f"rust router lint: no .rs file under {tmp_path}, nothing checked",
    )
