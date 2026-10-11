# SPDX-License-Identifier: MIT
"""`gate_cpu.sh` runs the compass_harness tests or refuses; it never lets them skip.

Each case runs the real script over the throwaway tree of
``test_gate_cpu_verdict_through_pipe.py``, plus a stand-in ``compass_harness``
package. Its import either raises what the real one raises with no aiperf
installed, or records which file was imported. Nothing needs a driver or aiperf.
"""

import importlib.util

import pytest
from test_gate_cpu_verdict_through_pipe import _run, _tree

PYPROJECT = """\
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"
[project]
name = "compass-harness"
version = "0.1.0"
[tool.setuptools]
packages = ["compass_harness"]
"""
NO_AIPERF = (
    'raise RuntimeError("compass_harness needs aiperf, and it is not installed")\n'
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("setuptools") is None,
    reason="the gate builds compass_harness with --no-build-isolation",
)


def _harness_tree(root, init):
    _tree(root, stamped=True)
    package = root / "compass_harness" / "compass_harness"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(init)
    (root / "compass_harness" / "pyproject.toml").write_text(PYPROJECT)
    return root


def test_without_aiperf_the_gate_refuses_before_pytest(tmp_path):
    out = _run(_harness_tree(tmp_path / "ATOM", NO_AIPERF), "{gate}")
    verdict = out.stdout.splitlines()[-1]
    assert out.returncode == 96, out.stdout + out.stderr
    assert "compass_harness tests would skip" in verdict, verdict
    assert "needs aiperf" in verdict and "COMPASS_AGENTX_HARNESS" in verdict, verdict
    assert "pytest: rc=" not in out.stdout, out.stdout


def test_the_tests_import_the_trees_own_compass_harness(tmp_path):
    root = _harness_tree(tmp_path / "ATOM", "MARK = 'tree'\n")
    # aiperf stand-in, so the probe passes; the suite asserts what it imported.
    (root / "aiperf.py").write_text("")
    (root / "tests" / "test_harness.py").write_text(
        "import compass_harness\n"
        "def test_installed_copy():\n"
        "    assert compass_harness.MARK == 'tree'\n"
        "    assert '/site/compass_harness/' in compass_harness.__file__\n"
    )
    out = _run(root, "{gate}")
    # 98 is the GPU-tier answer every _tree run ends on, after a green pytest.
    assert out.returncode == 98, out.stdout + out.stderr
    assert "pytest: rc=0" in out.stdout and "2 passed" in out.stdout, out.stdout
