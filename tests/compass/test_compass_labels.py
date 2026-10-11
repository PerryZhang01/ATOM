# SPDX-License-Identifier: MIT
"""The `compass-labels` skill's path and title rules, against the head's real
`atom/compass/` tree -- not a fixture, per the skill's own rule that the
module set is read from the tree, never hard-coded. The init-only exclusion
alone uses a scratch tree: which real directories are init-only changes as
modules ship.
"""

import importlib.util
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / ".claude"
    / "skills"
    / "compass-labels"
    / "compass_labels.py"
)
_spec = importlib.util.spec_from_file_location("compass_labels", MODULE_PATH)
compass_labels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compass_labels)


def test_the_tree_s_modules_exclude_init_only_and_pycache_dirs(tmp_path):
    # A directory carrying only `__init__.py` ships nothing: no label.
    for d, f in (("shipped", "x.py"), ("init_only", "__init__.py")):
        (tmp_path / d).mkdir()
        (tmp_path / d / f).touch()
    assert compass_labels.discover_modules(tmp_path) == ["shipped"]
    modules = compass_labels.discover_modules()
    assert "__pycache__" not in modules
    for expected in (
        "artifacts", "audit", "backends", "clock", "clock_transport",
        "design", "detect", "ir", "kv", "memory", "parity", "runner", "spec",
    ):
        assert expected in modules


def test_an_atom_compass_path_gets_its_module_label():
    assert compass_labels.labels_for(["atom/compass/clock/authority.py"], "") == {
        "module: compass-clock"
    }


def test_a_flat_test_file_picks_the_longest_matching_module():
    assert compass_labels.labels_for(
        ["tests/compass/test_clock_transport_wire.py"], ""
    ) == {"module: compass-clock-transport"}
    assert compass_labels.labels_for(["tests/compass/test_clock_foo.py"], "") == {
        "module: compass-clock"
    }


def test_a_nested_test_file_is_labelled_by_its_directory_not_its_name():
    # Lives under tests/compass/clock/, though its own name says "transport".
    assert compass_labels.labels_for(
        ["tests/compass/clock/test_clock_transport.py"], ""
    ) == {"module: compass-clock"}


def test_a_flat_test_file_matching_no_module_gets_no_label():
    assert compass_labels.labels_for(["tests/compass/qwen3_5_27b_config.json"], "") == set()


def test_scripts_and_tools_get_their_fixed_module_labels():
    assert compass_labels.labels_for(["scripts/compass/gate_cpu.sh"], "") == {
        "module: compass-script"
    }
    assert compass_labels.labels_for(["tools/compass/detect_loose_refusals.py"], "") == {
        "module: compass-tools"
    }


def test_a_gate_script_change_never_also_gets_dev_process():
    assert "dev-process" not in compass_labels.labels_for(
        ["scripts/compass/gate_cpu.sh"], "compass(script): tighten the exclude list"
    )


def test_dev_process_paths():
    assert compass_labels.labels_for(["AI_DEV_RULES.md"], "") == {"dev-process"}
    assert compass_labels.labels_for([".claude/agent-team.md"], "") == {"dev-process"}
    # "any CLAUDE.md" is a basename match, not just the repo-root one.
    assert compass_labels.labels_for(["gpu_docker/CLAUDE.md"], "") == {"dev-process"}


def test_dev_process_titles():
    assert "dev-process" in compass_labels.labels_for([], "compass(rules): x")
    assert "dev-process" in compass_labels.labels_for([], "compass(process): x")
    assert "dev-process" not in compass_labels.labels_for([], "compass(runner): x")
    assert "dev-process" in compass_labels.labels_for([], "Process: tidy the gate")
    assert "dev-process" in compass_labels.labels_for([], "agent-team: add a rule")


def test_a_path_outside_every_rule_gets_no_label():
    assert compass_labels.labels_for(["atom/model_engine/scheduler.py"], "") == set()


def test_several_paths_union_their_labels():
    assert compass_labels.labels_for(
        ["atom/compass/clock/authority.py", "atom/compass/kv/connector.py"], ""
    ) == {"module: compass-clock", "module: compass-kv"}
