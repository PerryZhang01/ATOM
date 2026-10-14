# SPDX-License-Identifier: MIT
"""`scripts/compass/cpu_gate_exclude.txt`: the list the CPU test gate trusts.

The list names the ATOM tests that reach the driver, so `gate_cpu.sh` can skip
them and stay green without a GPU.

It has two sections, and the split is the point. The GENERATED section is
produced by iterating collection to a fixed point and is reproducible from the
tree. The MANUAL section is for files that *collect* cleanly and fail later on a
driver call -- invisible to a generator that greps collection errors, so they
cannot be derived and must be asserted, with the failure text that justifies
each one. Before the split, four entries had been hand-added to a file whose
header read "never hand-edit", and the list could not be reproduced from the
tree it described. These tests exist so that cannot recur silently.

The failure being guarded against is drift, and drift here is silent in the
direction that matters. A renamed or deleted ATOM test leaves a stale
`--ignore=` that pytest accepts without complaint, so the entry stops excluding
anything and the gate starts erroring at collection on a file nobody listed --
which reads as "your change broke collection".

Two of the parametrisations below take their argvalues from a file rather than
from a literal, and a parametrisation of nothing is a pass -- so each carries
something that states its own non-empty case, without which this file reports
green with its pins asserting nothing. The two statements are deliberately
different assertions. The trigger list must *exist*: a tree with no
`gpu_gate_triggers.txt` has lost the record of what the CPU gate cannot see, so
that guard refuses outright. The manual section is *allowed* to be empty --
nothing obliges a tree to hand-exclude anything -- so its guard pins the count,
which is what separates an emptiness somebody chose from one that arrived.

What is *not* asserted: that every listed file genuinely needs a driver. Proving
that means collecting each one in a driver-free container, which is
`regen_cpu_gate_exclude.sh`'s job, not a unit test's.
"""

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
EXCLUDE = REPO / "scripts" / "compass" / "cpu_gate_exclude.txt"
TRIGGERS = REPO / "scripts" / "compass" / "gpu_gate_triggers.txt"

GEN_BEGIN, GEN_END = "# BEGIN GENERATED", "# END GENERATED"
MAN_BEGIN, MAN_END = "# BEGIN MANUAL", "# END MANUAL"


def _lines(path):
    return path.read_text().splitlines()


def _paths(lines):
    return [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]


def _section(begin, end):
    """Lines strictly between two markers, markers excluded.

    Returns [] when a marker is missing rather than raising, so the dedicated
    marker test reports that failure once instead of every test erroring.
    """
    lines = _lines(EXCLUDE)
    if begin not in lines or end not in lines:
        return []
    return lines[lines.index(begin) + 1 : lines.index(end)]


def _entries():
    return _paths(_lines(EXCLUDE))


def _manual_entries():
    return _paths(_section(MAN_BEGIN, MAN_END))


def _trigger_entries():
    # Absence returns [] rather than raising, for the reason _section gives: one
    # named refusal beats every test in this module erroring at collection. That
    # is only safe because the refusal exists -- test_the_trigger_list_exists_
    # and_is_not_empty below. Without it the [] is a fallback standing where a
    # refusal belongs, and the parametrisation it feeds is a pass over nothing.
    return _paths(_lines(TRIGGERS)) if TRIGGERS.is_file() else []


def test_the_list_exists_and_is_not_empty():
    assert EXCLUDE.is_file(), f"{EXCLUDE} is missing; run regen_cpu_gate_exclude.sh"
    assert _entries(), "an empty exclusion list means the CPU gate runs driver tests"


def test_both_section_markers_are_present():
    # gate_cpu.sh tolerates their absence -- it reads every non-comment line --
    # but regen_cpu_gate_exclude.sh refuses without them, so losing a marker
    # turns a reproducible list back into a hand-maintained one at the next
    # regeneration, quietly.
    lines = _lines(EXCLUDE)
    for marker in (GEN_BEGIN, GEN_END, MAN_BEGIN, MAN_END):
        assert marker in lines, f"{marker} missing from {EXCLUDE.name}"
    assert lines.index(GEN_BEGIN) < lines.index(GEN_END) < lines.index(MAN_BEGIN)
    assert lines.index(MAN_BEGIN) < lines.index(MAN_END)


@pytest.mark.parametrize("entry", _entries())
def test_every_excluded_path_still_exists(entry):
    # Parametrized so a rename names the offending file, rather than reporting
    # "1 of 29 paths is wrong" and leaving the reader to diff.
    assert (REPO / entry).is_file(), (
        f"{entry} is listed but does not exist -- ATOM renamed or removed it. "
        "Regenerate with scripts/compass/regen_cpu_gate_exclude.sh."
    )


def test_no_plugin_tests_are_listed():
    # gate_cpu.sh excludes tests/plugin/ by directory, for a different reason:
    # sglang and vllm are in neither image. Listing one here too would assert it
    # is a driver problem, and would survive the day the images gain sglang.
    strays = [e for e in _entries() if e.startswith("tests/plugin/")]
    assert not strays, f"excluded by directory already, drop from the list: {strays}"


def test_the_generated_section_is_sorted_and_unique():
    # The generator writes `sort -u`. Anything else means it was edited by hand,
    # which is how the list gets one batch of collection errors and stops.
    # Asserted per section, not over the file: the manual entries are appended
    # after the generated block and are not expected to interleave with it.
    entries = _paths(_section(GEN_BEGIN, GEN_END))
    assert entries, "the generated section is empty; run regen_cpu_gate_exclude.sh"
    assert entries == sorted(set(entries)), "not sorted+unique: regenerate, do not edit"


def test_the_manual_section_is_sorted_and_unique():
    entries = _manual_entries()
    assert entries == sorted(set(entries)), f"sort the manual entries: {entries}"


def test_no_path_appears_in_both_sections():
    # A duplicate is harmless to pytest and fatal to the reader: it makes the
    # generated section look like it discovered something the generator cannot
    # see, which is exactly the confusion the split removes.
    gen = set(_paths(_section(GEN_BEGIN, GEN_END)))
    man = set(_manual_entries())
    assert not gen & man, f"listed twice: {sorted(gen & man)}"


def test_the_manual_section_holds_the_entries_it_is_pinned_to_hold():
    """A count, because this section is allowed to be empty.

    The generated section next door can say `assert entries` -- a tree with no
    collection-time driver failures is not a tree anybody has. Nothing says the
    same about hand-added entries: removing the last one is a legitimate end
    state, which is exactly why "something states the non-empty case" cleared
    this site once and should not have. *Allowed* to be empty is not *asserted*
    non-empty, and with both markers kept and the section emptied the
    parametrisation below collects nothing, the two sibling assertions that read
    this section go vacuous without moving a count, and the file reports green.

    So the number is stated here instead. Reaching zero stays available; it
    costs an edit to this line, which is the declaration the file cannot make
    for itself.
    """
    entries = _manual_entries()
    assert len(entries) == 14, (
        f"the manual section holds {entries}, not the 14 entries pinned here. "
        "Add or remove one deliberately and update this count -- it is what "
        "stands between an empty section and a parametrisation of nothing."
    )


def test_the_manual_guard_finds_nothing_when_the_section_empties(monkeypatch, tmp_path):
    """The control for the pin above, which otherwise is only ever seen passing.

    A pin that has never been seen with its input at zero is a liveness check:
    it holds today because somebody hand-excluded something years ago. Pointed
    at a list whose markers are both present and whose MANUAL block is empty --
    the exact shape `regen_cpu_gate_exclude.sh` leaves behind, since it copies
    the block verbatim -- the derivation must come back empty, so the pin is
    shown failing rather than merely alive. The entry parked outside the markers
    is there so a derivation that read the whole file instead of its own section
    would be caught here.
    """
    moved = tmp_path / "cpu_gate_exclude.txt"
    moved.write_text(
        f"{GEN_BEGIN}\ntests/test_one.py\n{GEN_END}\n"
        f"{MAN_BEGIN}\n{MAN_END}\ntests/test_outside_the_markers.py\n"
    )
    monkeypatch.setitem(globals(), "EXCLUDE", moved)
    assert not _manual_entries()


@pytest.mark.parametrize("entry", _manual_entries())
def test_every_manual_entry_states_why(entry):
    # A manual entry is an assertion no script can check, so the reason is the
    # only evidence it carries. Without one it is indistinguishable from a test
    # somebody found inconvenient -- which is what three of the four original
    # hand-added entries turned out to be: they passed on CPU.
    section = _section(MAN_BEGIN, MAN_END)
    above = section[: section.index(entry)]
    reason = [ln for ln in reversed(above) if ln.strip()]
    assert reason and reason[0].lstrip().startswith("#"), (
        f"{entry} has no comment above it. State the observed failure -- the "
        "error text and where it was measured -- or move it to the generated "
        "section by making regen_cpu_gate_exclude.sh find it."
    )


def test_the_trigger_list_exists_and_is_not_empty():
    # Not a count, unlike the manual section: there is no legitimate tree in
    # which this file is absent or says nothing. It records what the CPU gate
    # cannot see, and gate_cpu.sh matches a diff against it to decide whether
    # the GPU gate is required. Empty, that decision is always "not required".
    assert TRIGGERS.is_file(), (
        f"{TRIGGERS} is missing; regenerate with "
        "scripts/compass/regen_gpu_gate_triggers.sh"
    )
    assert _trigger_entries(), (
        f"{TRIGGERS.name} names no path, so gate_cpu.sh answers 'gpu: not "
        "required' for every diff. Regenerate with "
        "scripts/compass/regen_gpu_gate_triggers.sh."
    )


def test_the_trigger_guard_finds_nothing_when_the_file_moves(monkeypatch, tmp_path):
    """The control for the guard above, which otherwise only proves it is alive.

    A rename of `gpu_gate_triggers.txt` breaks no import and no script the
    suite reads, so nothing else in this tree reddens for it; the guard is the
    whole protection and has to be shown failing, not just present. Pointed at
    a path that does not resolve, the derivation must come back empty -- and the
    real file sitting one directory up is there so a derivation that searched
    past its own path would be caught here instead of quietly keeping the
    parametrisation non-empty.
    """
    (tmp_path / TRIGGERS.name).write_text("atom/model_engine/model_runner.py\n")
    monkeypatch.setitem(globals(), "TRIGGERS", tmp_path / "moved" / TRIGGERS.name)
    assert not _trigger_entries()


@pytest.mark.parametrize("entry", _trigger_entries())
def test_every_trigger_path_still_exists(entry):
    # Same drift, one file over. A trigger naming a moved module matches
    # nothing, and gate_cpu.sh then reports "gpu: not required" for a diff that
    # needs it -- a false green, which is worse than the stale --ignore= above.
    target = REPO / entry
    assert target.is_dir() if entry.endswith("/") else target.is_file(), (
        f"{entry} is listed in {TRIGGERS.name} but does not exist. "
        "Regenerate with scripts/compass/regen_gpu_gate_triggers.sh."
    )
