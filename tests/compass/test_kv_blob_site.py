# SPDX-License-Identifier: MIT
"""Each disaggregation connector builds its `kv_transfer_params` blob at
exactly one site.

The derivation is an AST walk, not an import: the connectors pull in RDMA
backends, and these tests run with no driver. The modules are read out of this
checkout by path, so the tree under test is the tree the test file lives in.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TREE = Path(__file__).resolve().parents[2]

# The attribute the scheduler reads back and the router relays.
BLOB_ATTR = "kv_transfer_params_output"

CONNECTORS = {
    "moriio": TREE / "atom/kv_transfer/disaggregation/moriio/moriio_connector.py",
    "mooncake": TREE / "atom/kv_transfer/disaggregation/mooncake/mooncake_connector.py",
}


def _blob_site(path: Path) -> list[ast.Assign]:
    """Every `<something>.kv_transfer_params_output = ...` in one module.

    Narrow on purpose: plain `ast.Assign` to an `ast.Attribute`, which is the
    form both connectors use. Every other write of the name is what
    `_writes_not_read` returns, and the one-site test refuses those, so a
    second site written another way fails there by name instead of passing
    unseen. An assignment of `None` builds no blob -- an aborted producer
    request advertises nothing -- so it is not a site.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Attribute) and t.attr == BLOB_ATTR for t in node.targets
        )
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    ]


def _writes_not_read(path: Path) -> list[str]:
    """Every write of the name `_blob_site` does not read, by line and source.

    An annotated or augmented assignment, the name inside a tuple target, the
    name as a string anywhere in the module -- and any mention of `setattr`,
    `__setattr__`, `__dict__` or `vars` at all, since a name built at run time
    cannot be read and neither connector writes attributes that way.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    read = {
        id(t) for n in ast.walk(tree) if isinstance(n, ast.Assign) for t in n.targets
    }
    return [
        f"{path.name}:{n.lineno}: {source.splitlines()[n.lineno - 1].strip()}"
        for n in ast.walk(tree)
        if (
            isinstance(n, ast.Attribute)
            and n.attr == BLOB_ATTR
            and not isinstance(n.ctx, ast.Load)
            and id(n) not in read
        )
        or (isinstance(n, ast.Constant) and n.value == BLOB_ATTR)
        or getattr(n, "id", getattr(n, "attr", None))
        in ("setattr", "__setattr__", "__dict__", "vars")
    ]


@pytest.mark.parametrize("backend", sorted(CONNECTORS))
def test_each_backend_builds_the_blob_at_exactly_one_site(backend):
    """A second site would mean the backend emits one of two shapes."""
    unread = _writes_not_read(CONNECTORS[backend])
    assert not unread, f"{BLOB_ATTR} is written in a form not read here: {unread}"
    sites = _blob_site(CONNECTORS[backend])
    assert len(sites) == 1, (
        f"{CONNECTORS[backend].name} assigns {BLOB_ATTR} at "
        f"{[s.lineno for s in sites]}; one site per backend is expected"
    )
