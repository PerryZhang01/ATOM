# SPDX-License-Identifier: MIT
"""Importing `atom.compass.kv` loads no device runtime, measured by running it.

The package says that it, and the engine's connector interface and factory it
implements, pull in no device runtime, so a transfer can still be priced where
no driver exists. This test
process cannot check that: the suite has imported torch long before it gets
here. So each import runs in a fresh interpreter, which reports what the import
added to `sys.modules`.

The import is run rather than read. A scan of this package's source sees an
`import torch` written here, and nothing reached through `atom.kv_transfer` or
`atom.model_engine`, which is where most of what the package loads lives.

What counts as a device runtime is stated as its complement: the packages from
outside this repository and the standard library that the import may add are
exactly `ALLOWED`, and anything else fails. Listing the runtimes instead --
torch, triton, aiter, a HIP binding -- passes the one nobody listed.

The standard library is told apart by where a module was loaded from, not by
name. `sys.stdlib_module_names` misses modules the standard library creates
under names of their own, such as `__mp_main__` and `_sysconfigdata_*`, and
counting those would fail this on an import that reaches no device.
"""

import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "atom" / "compass" / "kv"

ALLOWED = set()

# What the package docstring names on each side of the boundary. Asserting these
# were loaded is what stops the checks below passing on an import that did
# nothing.
BOUNDARY = {
    "atom.compass.kv.connector",
    "atom.compass.kv.handoff",
    "atom.compass.kv.transfer",
    "atom.kv_transfer.disaggregation.base",
    "atom.kv_transfer.disaggregation.factory",
    "atom.kv_transfer.disaggregation.types",
}

CHILD = """
import importlib, json, sys
before = set(sys.modules)
module = importlib.import_module(sys.argv[1])
result = eval(sys.argv[2])
print(json.dumps({
    "file": module.__file__,
    "added": {
        name: getattr(sys.modules[name], "__file__", None)
        for name in set(sys.modules) - before
    },
    "result": result,
}))
"""


def _import_in_a_fresh_interpreter(module, then="None", path=()):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), *map(str, path)])
    done = subprocess.run(
        [sys.executable, "-c", CHILD, module, then],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert done.returncode == 0, f"importing {module} failed:\n{done.stderr}"
    return json.loads(done.stdout)


STDLIB = [Path(sysconfig.get_paths()[key]) for key in ("stdlib", "platstdlib")]


def _in_the_stdlib(file):
    """Whether *file* was loaded from the standard library.

    A module with no file is built in, frozen, or an alias of `__main__`. The
    install directories can sit inside a standard library directory -- in a
    venv `platstdlib` is `lib/python3.X` and every package is below it in
    `site-packages` -- so being under one is not enough on its own.
    """
    if file is None:
        return True
    path = Path(file)
    return any(path.is_relative_to(root) for root in STDLIB) and not {
        "site-packages",
        "dist-packages",
    } & set(path.parts)


def _third_party(added):
    tops = {
        name.partition(".")[0] for name, f in added.items() if not _in_the_stdlib(f)
    }
    return tops - {"atom"}


def test_the_child_sees_a_package_reached_through_another(tmp_path):
    """The instrument, on an import whose answer is known.

    `outer` imports `inner` and nothing else names `inner`, so reporting it is
    reporting a package reached transitively -- the only way a device runtime
    would ever arrive here.
    """
    (tmp_path / "probe_outer.py").write_text("import probe_inner\n")
    (tmp_path / "probe_inner.py").write_text("")
    seen = _import_in_a_fresh_interpreter("probe_outer", path=[tmp_path])
    assert _third_party(seen["added"]) == {"probe_outer", "probe_inner"}


def _kv_modules():
    # rglob, so a module added under the package is covered the day it lands.
    return sorted(PACKAGE.rglob("*.py"))


def _module_name(path):
    """The dotted name *path* is imported under, subpackages included."""
    return ".".join(path.relative_to(REPO).with_suffix("").parts).removesuffix(
        ".__init__"
    )


def test_the_package_was_found():
    assert _kv_modules(), f"no modules under {PACKAGE}"


def test_the_guard_finds_nothing_when_the_root_moves(monkeypatch, tmp_path):
    """The control for the guard above, which otherwise only proves it is alive.

    Pointed at a root that does not resolve, the listing must come back empty.
    The sibling module one level out catches a derivation that widened past its
    own root and so kept the parametrisation non-empty.
    """
    (tmp_path / "sibling.py").write_text("")
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path / "moved")
    assert not _kv_modules()


def test_a_module_in_a_subpackage_keeps_its_subpackage():
    sub = PACKAGE / "sub"
    assert _module_name(sub / "x.py") == "atom.compass.kv.sub.x"
    assert _module_name(sub / "__init__.py") == "atom.compass.kv.sub"
    assert _module_name(PACKAGE / "__init__.py") == "atom.compass.kv"


@pytest.mark.parametrize(
    "path",
    _kv_modules(),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_importing_it_loads_no_device_runtime(path):
    seen = _import_in_a_fresh_interpreter(_module_name(path))
    assert Path(seen["file"]).is_relative_to(REPO)
    assert BOUNDARY <= set(seen["added"])
    assert _third_party(seen["added"]) == ALLOWED


def test_a_price_is_computed_without_one():
    """The property itself: a transfer priced, then `sys.modules` read."""
    seen = _import_in_a_fresh_interpreter(
        "atom.compass.kv",
        then="module.TransferModel(1e-6, 1e9, 1000).release_at(1.0, 3)",
    )
    assert Path(seen["file"]).is_relative_to(REPO)
    assert seen["result"] == pytest.approx(1.0 + 1e-6 + 3e-6)
    assert BOUNDARY <= set(seen["added"])
    assert _third_party(seen["added"]) == ALLOWED
