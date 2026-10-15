# SPDX-License-Identifier: MIT
"""The classified list of blocking calls on the serving path, kept honest.

A simulated run substitutes predicted durations for real work, so it has to
know every call that parks a thread on the real clock. The list of them is in
`atom/compass/audit/sync_sites.json`; deciding what each one is was a reading
job and cannot be re-derived. What *can* be re-derived is the set of call sites
that exist, and these tests assert the two agree -- so a blocking call added to
ATOM later fails here instead of quietly widening the blind spot.

Two failures are worth telling apart, and each has its own test:

* a call site the list does not classify, or a listed site that has gone away
  (`test_every_scanned_site_is_classified`) -- read it and classify it;
* a pinned piece of text that is no longer exactly once in its symbol
  (`test_anchors_are_still_where_they_say`) -- the same, for the points that
  are not a call.

Neither records a line number, so an edit above a site or an anchor changes
nothing here.

Separately, each row's mechanism is checked against what the rest of the row
says (`test_every_mechanism_agrees_with_the_rest_of_its_row`).

A row is only safe to keep across an edit if it cannot silently change which
call it describes, which is what `test_inserting_a_call_above_another_leaves_
its_neighbour_alone` pins: an earlier identity keyed on the callee alone, so
inserting one `call_func("flush_pp_send", ...)` above a `call_func("forward",
...)` moved every later row onto the wrong site.

No driver, and no import of ATOM's serving modules: the scanner parses them.
"""

from __future__ import annotations

import ast
from collections import Counter

import pytest

from atom.compass.audit import sync_scan

TREE = sync_scan.repo_root_from_here()
INVENTORY = sync_scan.load_inventory()
ROWS = INVENTORY["sites"] + INVENTORY["anchors"]
CATEGORIES = set(INVENTORY["categories"])
MECHANISMS = set(INVENTORY["mechanisms"])

# The mechanism each first-classification category named. A row whose
# mechanism is another one has to say why in `mechanism_why`.
NAMED_BY_CATEGORY = {"A": "K1", "B": "K5", "C1": "K8", "C2": "K7", "ignore": "K9"}


@pytest.fixture(scope="module")
def scanned():
    return {site.id: site for site in sync_scan.scan(TREE)}


@pytest.fixture(scope="module")
def listed():
    return {row["id"]: row for row in INVENTORY["sites"]}


def test_every_scanned_site_is_classified(scanned, listed):
    """The two sets are equal, and the message says which way they differ."""
    missing = sorted(set(scanned) - set(listed))
    stale = sorted(set(listed) - set(scanned))
    assert not missing, (
        f"{len(missing)} call site(s) on the serving path are not classified. "
        "Read each one, decide what a simulated run does about it, and add a "
        f"row to {sync_scan.INVENTORY_PATH.name}: " + ", ".join(missing)
    )
    assert not stale, (
        f"{len(stale)} classified site(s) no longer exist in the tree; remove "
        "their rows: " + ", ".join(stale)
    )


def _key(row: dict) -> str:
    return row.get("id") or f"{row['file']}::{row.get('symbol', '')}::{row['anchor']}"


def _anchor_problem(row: dict, source: str) -> str | None:
    """Why `row`'s anchor text is not exactly once in its symbol, or None.

    A Python anchor names its enclosing symbol as a site id does; a file-wide
    match is not enough, because `while True:` repeats in `engine_core.py`.
    An anchor with no symbol (the Rust ones) is counted over the whole file.
    """
    lines = source.splitlines()
    if "symbol" in row:
        visitor = sync_scan._ScopedVisitor(row["file"], source)
        visitor.visit(ast.parse(source))
        spans = visitor.spans.get(row["symbol"])
        if not spans:
            return f"{_key(row)}: no symbol {row['symbol']!r} in the file"
    else:
        spans = [(1, len(lines))]
    found = sum("\n".join(lines[a - 1 : b]).count(row["anchor"]) for a, b in spans)
    if found != 1:
        return f"{_key(row)}: the anchor occurs {found} times, not once"
    return None


def test_anchors_are_still_where_they_say():
    wrong = [
        problem
        for row in INVENTORY["anchors"]
        if (problem := _anchor_problem(row, (TREE / row["file"]).read_text()))
    ]
    assert not wrong, "; ".join(wrong)


def _row_anchored(anchor: str, symbol: str | None) -> dict:
    """A real anchor row, re-pointed at `symbol`, or at the whole file for None."""
    row = dict(next(r for r in INVENTORY["anchors"] if r["anchor"] == anchor))
    row.pop("symbol")
    return row if symbol is None else {**row, "symbol": symbol}


@pytest.mark.parametrize(
    "row,refusal",
    [
        (_row_anchored("def _passed_delay", "Scheduler.postprocess"), "occurs 0 times"),
        (_row_anchored("while True:", "NoSuchClass.busy_loop"), "no symbol"),
        (_row_anchored("while True:", None), "times, not once"),
    ],
)
def test_a_seeded_misplaced_anchor_is_refused(row, refusal):
    problem = _anchor_problem(row, (TREE / row["file"]).read_text())
    assert problem is not None and refusal in problem, problem


def test_an_anchor_twice_in_its_symbol_is_refused():
    row = {"file": "made/up.py", "symbol": "Loop.run", "anchor": "while True:"}
    once = "class Loop:\n    def run(self):\n        while True:\n            pass\n"
    twice = once + "        while True:\n            pass\n"
    assert _anchor_problem(row, once) is None
    assert "occurs 2 times" in _anchor_problem(row, twice)


def test_every_row_carries_a_mechanism_and_a_reason():
    for row in ROWS:
        key = _key(row)
        assert row["mechanism"] in MECHANISMS, f"{key}: {row['mechanism']}"
        assert isinstance(row["mechanism_why"], str), key
        assert row.get("category") in CATEGORIES | {None}, key
        assert len(row["why"]) > 20, f"{key}: the justification is too short"
        assert row["peer"] in {"none", "thread", "process", "deployment"}, key


def _bounded(row: dict) -> bool:
    """A pinned line (a gate, a constant, a flag), a sleep, or a timed call."""
    shape = row.get("shape", "anchor")
    return shape in ("anchor", "sleep") or "timeout" in row.get("expr", "")


def _inconsistencies(row: dict) -> list[str]:
    """What a row's mechanism contradicts in the rest of the row.

    `peer` names a thread, a process or a deployment, not an LP, so it can
    refute a mechanism but not confirm one: a channel receive waits on another
    process, and another deployment is never inside the waiter's own LP.
    """
    wrong = []
    mechanism, peer = row["mechanism"], row["peer"]
    if mechanism == "K5" and peer not in ("process", "deployment"):
        wrong.append(f"a channel receive waits on another process, not a {peer}")
    if mechanism == "K6" and peer == "deployment":
        wrong.append("a wait inside one LP cannot wait on another deployment")
    if mechanism == "K7" and not _bounded(row):
        wrong.append("a virtual timer needs a bound to put on the LP clock")
    named = NAMED_BY_CATEGORY.get(row.get("category"))
    if mechanism != named and not row["mechanism_why"]:
        wrong.append("the mechanism is not the one its category named; say why")
    return wrong


def test_every_mechanism_agrees_with_the_rest_of_its_row():
    wrong = [f"{_key(row)}: {w}" for row in ROWS for w in _inconsistencies(row)]
    assert not wrong, "; ".join(wrong)


@pytest.mark.parametrize(
    "mechanism,change,refusal",
    [
        ("K5", {"peer": "thread"}, "a channel receive waits on another process"),
        ("K6", {"peer": "deployment"}, "a wait inside one LP cannot"),
        ("K7", {"expr": "q.get()", "shape": "queue_get"}, "a virtual timer needs"),
        ("K4", {"mechanism_why": ""}, "the mechanism is not the one"),
        ("K4", {"category": None, "mechanism_why": ""}, "the mechanism is not the one"),
    ],
)
def test_a_seeded_inconsistent_row_is_refused(mechanism, change, refusal):
    row = next(r for r in INVENTORY["sites"] if r["mechanism"] == mechanism)
    assert _inconsistencies(row) == [], _key(row)
    wrong = _inconsistencies({**row, **change})
    assert len(wrong) == 1 and wrong[0].startswith(refusal), wrong


def test_mechanism_crosstab_against_the_first_classification():
    """Rows per mechanism, and per first-classification category, printed.

    Rows added after the first classification carry no category and are
    counted under `-`.
    """
    cells = Counter((row.get("category", "-"), row["mechanism"]) for row in ROWS)
    mechanisms = sorted(MECHANISMS)
    categories = [*INVENTORY["categories"], "-"]
    print("\ncategory", *mechanisms, "total", sep="\t")
    for c in categories:
        line = [cells[(c, m)] for m in mechanisms]
        print(c, *line, sum(line), sep="\t")
    totals = Counter(row["mechanism"] for row in ROWS)
    print("total", *(totals[m] for m in mechanisms), len(ROWS), sep="\t")


def test_calls_that_share_an_ordinal_are_answered_the_same_way(listed):
    """Two sites can share an ordinal only by being the same call, written the
    same way, in one function. The ordinal between them is positional, so the
    inventory must not use it to say two different things."""
    by_text: dict[tuple, set] = {}
    for row in listed.values():
        key = (row["file"], row["symbol"], row["expr"], row["shape"])
        by_text.setdefault(key, set()).add(
            (row.get("category"), row["why"], row["mechanism"], row["mechanism_why"])
        )
    split = {k: v for k, v in by_text.items() if len(v) > 1}
    assert not split, (
        "identical calls in one function are classified differently, so their "
        "ordinals carry meaning they cannot keep across an edit: "
        + "; ".join(f"{k[0]}::{k[1]}::{k[2]}" for k in split)
    )


def test_every_excluded_root_exists():
    for rel, _why in sync_scan.UNSCANNED_ROOTS:
        assert (TREE / rel).exists(), f"{rel} is excluded in the scanner but is gone"


def test_a_new_package_under_atom_fails_until_classified_or_excluded(
    tmp_path, listed, monkeypatch
):
    """The scanned set is read off the tree: a package nobody named is scanned,
    so its blocking call is a site the inventory does not classify."""
    for rel, body in {
        "atom/brand_new/loop.py": "def run(q):\n    return q.get()\n",
        "atom/models/m.py": "def f(q):\n    return q.get()\n",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    unclassified = {s.file for s in sync_scan.scan(tmp_path) if s.id not in listed}
    assert unclassified == {"atom/brand_new/loop.py"}

    named = (("atom/brand_new/", "a reason"),)
    monkeypatch.setattr(sync_scan, "UNSCANNED_ROOTS", sync_scan.UNSCANNED_ROOTS + named)
    assert sync_scan.scan(tmp_path) == []


def test_no_scanned_file_is_also_declared_unscanned():
    """No file under an excluded entry is scanned."""
    scanned_files = set(sync_scan.iter_scanned_files(TREE))
    for rel, _why in sync_scan.UNSCANNED_ROOTS:
        inside = {f for f in scanned_files if f == rel or f.startswith(rel)}
        assert not inside, f"{rel} is excluded but these are scanned: {sorted(inside)}"


# --- the shape rules, against the forms they are written to tell apart ------


def _first_call(src: str) -> ast.Call:
    return next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call))


def _scan_source(src: str):
    visitor = sync_scan._CallVisitor("made/up.py", src)
    visitor.visit(ast.parse(src))
    return sync_scan._number(visitor.sites)


@pytest.mark.parametrize(
    "src,blocking",
    [
        ("q.get()", True),
        ("q.get(timeout=5)", True),
        ("q.get(False)", True),
        ('d.get("key")', False),
        ('d.get("key", None)', False),
    ],
)
def test_a_mapping_get_is_not_a_queue_get(src, blocking):
    assert sync_scan._no_positional(_first_call(src)) is blocking


@pytest.mark.parametrize(
    "src,blocking",
    [
        ("t.join()", True),
        ("t.join(5)", True),
        ("t.join(timeout=1)", True),
        ('",".join(parts)', False),
        ("os.path.join(a, b)", False),
    ],
)
def test_a_string_join_is_not_a_thread_join(src, blocking):
    assert sync_scan._join_form(_first_call(src)) is blocking


@pytest.mark.parametrize(
    "src,blocking",
    [
        ('mgr.call_func("forward", batch, wait_out=True)', True),
        ('mgr.call_func("process_kvconnector_output", meta)', False),
        ('mgr.call_func("x", wait_out=False)', False),
        ('mgr.call_func_with_aggregation("async_proc_aggregation")', True),
    ],
)
def test_a_worker_call_parks_only_when_it_asks_for_the_reply(src, blocking):
    assert sync_scan._worker_rpc_form(_first_call(src)) is blocking


def test_a_new_blocking_call_is_reported():
    """What the completeness test sees the day someone adds one."""
    src = "def handler(self):\n    item = self._inbox.get()\n    return item\n"
    sites = _scan_source(src)
    assert [(s.symbol, s.call, s.shape) for s in sites] == [
        ("handler", "self._inbox.get", "queue_get")
    ]


def test_inserting_a_call_above_another_leaves_its_neighbour_alone():
    """The defect this identity scheme exists to prevent.

    Keyed on the callee alone, the two calls below are indistinguishable and
    the ordinal that separated them is positional -- so inserting the flush
    renamed the forward's row onto the flush and said only that a line moved.
    """
    forward = '    self.mgr.call_func("forward", batch, wait_out=True)\n'
    flush = '    self.mgr.call_func("flush_pp_send", wait_out=True)\n'
    before = {s.id for s in _scan_source("def step(self):\n" + forward)}
    after = {s.id for s in _scan_source("def step(self):\n" + flush + forward)}
    assert before < after, "the surviving call's identity changed under an insert"
    assert len(after - before) == 1


def test_a_loop_that_calls_nothing_that_parks_is_reported_as_a_spin():
    spin = (
        "def f(self):\n"
        "    while self.pending:\n"
        "        if self.ready:\n"
        "            continue\n"
        "        break\n"
    )
    visitor = sync_scan._SpinVisitor("made/up.py", spin)
    visitor.visit(ast.parse(spin))
    assert [s.shape for s in visitor.sites] == ["spin_loop"]

    paced = (
        "import time\n"
        "def f(self):\n"
        "    while self.pending:\n"
        "        time.sleep(0.1)\n"
        "        continue\n"
    )
    visitor = sync_scan._SpinVisitor("made/up.py", paced)
    visitor.visit(ast.parse(paced))
    assert visitor.sites == []
