# SPDX-License-Identifier: MIT
"""What a spec is checked for, and why every reason is reported at once.

A spec is authored by hand against a schema most of whose terms cannot be looked
up anywhere, so the document a person first writes is nearly always incomplete.
The whole value of checking it is that an incomplete one is refused rather than
quietly completed: a term nobody measured has no default, because a default here
is a guess with a number's authority, and it will be discovered by a run rather
than by the person who could still go and measure it.

Refusals are collected, not raised one at a time. Reading a document stops at
the first thing wrong with it, which is right for a reader that has to produce a
value and wrong for a check whose reader is filling in a form: told one missing
field per attempt, they measure one term, run again, and find the next. So the
answer is the list, in the schema's own order, and a caller that wants an
exception raises the first.

The check has two halves. The first asks whether the document is a complete
instance of the schema -- every required field present and every value the shape
its field declares. The second asks whether what it describes is consistent:
whether the widths the deployment will actually use were measured, whether the
stack the constants were taken on is the stack now loaded, and whether constants
carried over from another spec came from one pinned to the same stack.

**The second half is asked of whatever resolved, and not only of a complete
spec.** Those questions are not intrinsically about a resolved document, only
their implementation was: both width tables are present and schema-valid in a
document that the first half refuses for a single missing derate, and that
derate is a value its author types in at a desk. Running the second half only
when the first came back clear therefore hid the expensive refusal -- a width
nobody measured, which costs an eight-GPU reservation to fix -- behind the cheap
one, and reintroduced at the halfway line exactly the one-refusal-per-attempt
this module opens by rejecting. So each question is asked of the fields that did
check out, and a question that reads a field which did not resolve says so by
name rather than being passed over in silence.

**That holds for the schema half too, so the walk over the document collects.**
A key the table does not declare, or a block holding a scalar where the schema
has fields, is one refusal about one key -- and reading the document stops
there, which would leave every other field unchecked and every consistency
question with nothing resolved to be asked of. A typo in a hand-authored
document is the cheapest fix there is, and it would have hidden the width
nobody measured behind itself. So the walk yields its refusals and carries on,
and the document is checked field by field whatever it holds elsewhere.

**The same record makes the opt-in conditions legible.** Four of the six in
`CONDITIONS` need something from the caller -- two of them `tp_widths=`, one
`observed_stack=`, and one a `Merge` rather than a document -- and a clear
result that does not say what it declined to ask is the shape this package
exists to refuse. The sharpest case is the transfer: its source's pin is in no
field of the merged document, by the decision below, so `validate(document)`
can never ask that condition however the document was built, and the same spec
is refused as a `Merge` and clear as a document. That is not a wrong number,
but it must be visible, because a caller that holds only the document cannot
ask it. Merging the document again does not recover the pin: a merge whose
fragments disagree on an origin states `mixed`, so a `Merge` holding the saved
document reports the transfer as not asked too, or as asked only in part when
another fragment states a transfer and a stack pin resolved. No command
validates a machine file: a per-machine script such as
`scripts/compass/mi308x_spec.py` merges its fragments and calls this module,
and nothing in this package reads a spec file. `CONDITIONS` is what a count of
reach is a count of; a condition added to the check set and not to it is one
no result can report on.

**The reach of a document is `ASKABLE_OF_A_DOCUMENT`, which is a value and not
a sentence.** Writing the number down in prose here puts a person between the
check set and the statement about it: the set changes, nobody rereads the
paragraph, and the package goes on claiming a reach it no longer has. So the
statement is the tuple below, a run's own `not_asked` is asserted against it,
and a condition that becomes askable of a document -- or one that stops being
-- fails a test rather than quietly contradicting a docstring.

**What was asked of only part of its fields is reported apart from what was not
asked at all.** A question whose fields all failed to resolve was not asked and
must not be counted as reach. A question asked of two width tables where only
one resolved *was* asked, and can already have earned a refusal; filing it under
"not asked" would report a refusal the run did make as a question it did not,
and would subtract it from the count of what the run reached. Both are results
and neither is the other, so they are two fields of the record.

**A width nobody measured is two questions, not one.** The first is whether the
document carries it, and the remedy that refusal offers is to go and measure it.
The second is whether anything here would measure it, and for one entry -- the
allocator's retained bytes on a single card -- the answer is no: the single-card
probe fills the other width-keyed term, and the multi-rank one starts above
width one. So for that entry the first remedy cannot be followed with the tools
this package ships, and a check that stopped at the first question would leave
an author hunting a probe that does not exist. The second is asked only where
the entry is absent: a spec that carries the number, measured by hand, is a good
spec and is not refused for how it was obtained.

Three of those questions need something the document does not carry, and the two
about widths need the same thing.

**The widths a deployment will use are ATOM's, not the spec's.** The spec
describes a machine and says nothing about how the engine was launched, so the
widths come in as an argument, from the engine's own configuration. A spec that
measured widths 1 and 2 is complete and correct, and is still the wrong spec for
a run at width 8 -- which is a fact about the pair, not about the document.

**A transferred constant is checked against its source's stack pin.** The
fragment that carried it declares the stack the source spec was pinned to, which
is why the merge keeps that declaration aside instead of contributing it. Here
the two are compared, and a transfer out of a differently-pinned spec is refused
by name. A transfer that declares no stack at all is refused too: the evidence
that these constants move with the compute stack is the whole reason a transfer
is allowed to be cheap, and a transfer that does not say what it was measured
against cannot be checked by anyone, ever. A document saved from a merge of
transfers from one spec reads back as such a transfer, since a merge keeps a
transfer's pin out of what it writes, so its refusal names that merge rather
than its author.

A stack mismatch on the running machine is the one finding that is not fatal by
default. The numbers are still measurements, taken on a stack that has since
moved, and whether that matters is a judgement about how far it moved -- so it
warns and names both versions, and refuses only when the caller asks for that.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fields import PINNED, SCHEMA, Kind, check
from .machine import MachineSpec, _missing, _survey
from .merge import PINNED_BLOCK, Merge
from .probes import FILLED_BY, probe_for
from .rules import Rule, SpecRefusal
from .tokenizers import TokenizerTable

#: The conditions a spec is checked for. The first two are asked of anything
#: that is a mapping at all and name themselves in `not_asked` when the subject
#: is not one; each of the rest names itself there when a run could not reach it.
MISSING = "whether every required field is present and every value its declared shape"
DERATES = "whether a spec-peak number carries the derate it obliges"
WIDTHS = "whether the widths this deployment will use were measured"
PROBES = "whether a probe fills the widths this deployment uses and nobody measured"
STACK = "whether the constants' stack pin is the stack now loaded"
TRANSFERS = "whether a transferred constant came from a spec pinned to this stack"
#: The whole check set, in the order a run asks it. What `not_asked` is measured
#: against: a condition missing from here is one no result can report on.
CONDITIONS = (MISSING, DERATES, WIDTHS, PROBES, STACK, TRANSFERS)
#: What a document-taking run can reach, which is every condition but the
#: transfer: a transfer's source pin is in no field of a document, however the
#: document was built. This is the package's statement of its own reach, held to
#: a run's `not_asked` by a test rather than written out in prose that nothing
#: reads. A caller that passes `validate` only a document states its reach from
#: here, or states one nothing checks.
ASKABLE_OF_A_DOCUMENT = tuple(
    condition for condition in CONDITIONS if condition != TRANSFERS
)
#: The fields each consistency question reads before it can be asked at all.
WIDTH_TABLES = tuple(field.path for field in SCHEMA if field.kind is Kind.WIDTH_TABLE)
#: The width tables the probe question can say anything about, which is not all
#: of them. It reports a width a document is missing that no probe would fill
#: either, so a table every probe covers is one it can only ever be silent
#: about -- and a question registered as reading a field it cannot speak for
#: reports a run that could not ask it and a run that asked and found nothing
#: as the same record. Derived from `FILLED_BY`, so a probe given the width
#: that has none empties this with no second list to remember. A table no
#: probe is named for at all contributes nothing here rather than failing the
#: derivation: this runs while the module is being imported, so a subscript
#: would answer a table nobody has entered yet by denying every caller of the
#: package, including the ones with no interest in probes, and the reader would
#: be told which import failed rather than which term is missing. The naming is
#: left to `probe_for`, which refuses such a term by name, and the two lists
#: are held against each other by a test.
PROBE_TABLES = tuple(
    path for path in WIDTH_TABLES if None in FILLED_BY.get(path.rsplit(".", 1)[-1], ())
)
STACK_PINS = tuple(f"{PINNED_BLOCK}.{component}" for component in PINNED)


@dataclass(frozen=True, slots=True)
class Validation:
    """Everything wrong with a spec, rather than the first thing wrong with it."""

    spec: MachineSpec | None
    refusals: tuple[SpecRefusal, ...]
    stack_differences: tuple[tuple[str, str, Any], ...]
    not_asked: tuple[str, ...] = ()
    asked_in_part: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether the spec may be used, which is not whether it was silent."""
        return not self.refusals

    def raise_first(self) -> MachineSpec:
        """The resolved spec, or the first refusal, for a caller that wants one."""
        if self.refusals:
            raise self.refusals[0]
        return self.spec

    def __str__(self) -> str:
        """The verdict, every reason for it, every stack difference it found,
        every question left unasked, and every one asked of only part of what
        it reads."""
        return "\n".join(
            [f"{'ok' if self.ok else 'refused'}: {len(self.refusals)} refusal(s)"]
            + [f"  {refusal}" for refusal in self.refusals]
            + [
                f"  stack moved: the constants were measured against "
                f"{component} {pinned!r} (now {seen!r})"
                for component, pinned, seen in self.stack_differences
            ]
            + [f"  not asked: {condition}" for condition in self.not_asked]
            + [f"  asked in part: {condition}" for condition in self.asked_in_part]
        )


def _unreached(
    condition: str, reads: Sequence[str], resolved: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A condition the fields it reads put out of reach, split by how far.

    Two results, not one. Where nothing it reads resolved the condition was not
    asked, and the count of what a run reached must not include it. Where some
    of what it reads resolved it *was* asked, of those -- it can have earned a
    refusal already -- and the fields it could not be asked of are named
    separately so neither the wording nor the count says it went unasked.
    """
    absent = [path for path in reads if path not in resolved]
    if not absent:
        return (), ()
    named = ", ".join(f"`{path}`" for path in absent)
    if len(absent) == len(reads):
        return (
            f"{condition} -- {named} did not resolve, so it could not be asked at all",
        ), ()
    asked_of_the_rest = (
        f"{condition} -- {named} did not resolve, so it was asked of the rest "
        "and not of those"
    )
    return (), (asked_of_the_rest,)


def _reach(
    resolved: Mapping[str, Any],
    tp_widths: Sequence[int],
    observed_stack: Mapping[str, str] | None,
    merged: Merge | None,
    read: bool,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The conditions this run did not ask, and those it asked of only part of
    the fields they read. Each carries its reason."""
    unasked: list[str] = []
    partial: list[str] = []

    def reached(condition: str, reads: Sequence[str]) -> None:
        none, some = _unreached(condition, reads, resolved)
        unasked.extend(none)
        partial.extend(some)

    if not read:
        why = "the subject is not a mapping, so there was no document to ask it of"
        unasked.append(f"{MISSING} -- {why}")
        unasked.append(f"{DERATES} -- {why}")
    if not tp_widths:
        unasked.append(f"{WIDTHS} -- no `tp_widths=` was given")
        unasked.append(f"{PROBES} -- no `tp_widths=` was given")
    else:
        reached(WIDTHS, WIDTH_TABLES)
        reached(PROBES, PROBE_TABLES)
    if observed_stack is None:
        unasked.append(f"{STACK} -- no `observed_stack=` was given")
    else:
        reached(STACK, STACK_PINS)
    if merged is None:
        unasked.append(
            f"{TRANSFERS} -- the subject is a document, and a transfer's source "
            "pin is in no field of one; ask this of the `Merge` while the "
            "fragments are still in hand"
        )
    else:
        if merged.transfers:
            reached(TRANSFERS, STACK_PINS)
        # Where a transfer is stated and a pin did not resolve, `reached` has
        # already reported the condition once.
        hidden = [repr(f.source) for f in merged.fragments if f.origin == "mixed"]
        if hidden and not any(c.startswith(TRANSFERS) for c in unasked + partial):
            why = (
                f"{TRANSFERS} -- origin `mixed` in {', '.join(hidden)} does not "
                "say whether a transfer went into it, and a transfer's source "
                "pin is in no field of a document"
            )
            if merged.transfers:
                partial.append(f"{why}; it was asked of the transfers stated")
            else:
                unasked.append(why)
    return tuple(unasked), tuple(partial)


def _complete(
    document: object, resolved: dict[str, Any]
) -> tuple[list[SpecRefusal], bool]:
    """Every refusal the schema itself earns, and whether the subject was a
    document the schema could be asked of at all.

    A subject that is not a mapping is the one case where the two schema
    conditions go unasked, and the caller needs to know that rather than infer
    it from a refusal: a result that reported them as asked would count two
    questions nothing answered.
    """
    if not isinstance(document, Mapping):
        return [
            SpecRefusal(
                Rule.SHAPE,
                f"a spec is a mapping of its sections, not {type(document).__name__}",
                "load the document before checking it as a spec",
            )
        ], False
    found: dict[str, Any] = {}
    refusals = list(_survey(document, "", found))
    for field in SCHEMA:
        try:
            if field.path not in found:
                if field.required:
                    _missing(field)
            else:
                resolved[field.path] = check(field, found[field.path], field.path)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    return refusals, True


def _widths(spec: MachineSpec, tp_widths: Sequence[int], resolved: Mapping[str, Any]):
    for path in WIDTH_TABLES:
        if path not in resolved:
            continue
        for width in tp_widths:
            try:
                spec.runtime_constant(path.rsplit(".", 1)[-1], width)
            except SpecRefusal as refusal:
                yield refusal


def _probes(tp_widths: Sequence[int], resolved: Mapping[str, Any]):
    """The widths a document is missing that no probe here would fill either.

    Asked of the tables a probe can fall short on, which is the same set the
    condition registers as what it reads: walking a table every probe covers
    would ask a question whose answer is fixed.
    """
    for path in PROBE_TABLES:
        if path not in resolved:
            continue
        name = path.rsplit(".", 1)[-1]
        for width in tp_widths:
            if width in resolved[path]:
                continue
            try:
                probe_for(name, width)
            except SpecRefusal as refusal:
                yield refusal


def _transfers(resolved: Mapping[str, Any], merged: Merge):
    for fragment in merged.transfers:
        declared = {
            component: fragment.values[f"{PINNED_BLOCK}.{component}"]
            for component in PINNED
            if f"{PINNED_BLOCK}.{component}" in fragment.values
        }
        listed = fragment.values.get("provenance.fragments")
        if not declared and listed:
            yield SpecRefusal(
                Rule.PINNED_STACK,
                f"{fragment.stanza()} carried constants over from "
                f"{fragment.transferred_from!r}, and its provenance names the "
                "fragments an earlier merge built it from; a merge keeps a "
                "transfer's source stack pin out of the document it writes, so "
                "this one cannot say which stack they were measured against",
                "merge the fragments it was built from "
                f"({', '.join(map(repr, listed))}) in its place, since a "
                "transfer states its source's stack pin there and nowhere else; "
                "any of them that is itself a saved document with no pin is "
                "refused the same way",
            )
            continue
        if not declared:
            yield SpecRefusal(
                Rule.PINNED_STACK,
                f"{fragment.stanza()} carried constants over from "
                f"{fragment.transferred_from!r} without saying which stack they "
                "were measured against",
                "a transfer carries both specs' stack pins; these constants "
                "move with the compute stack, so one that cannot name its own "
                "stack cannot be checked against this machine's",
            )
            continue
        for component, version in declared.items():
            if f"{PINNED_BLOCK}.{component}" not in resolved:
                continue
            here = resolved[f"{PINNED_BLOCK}.{component}"]
            if version != here:
                yield SpecRefusal(
                    Rule.PINNED_STACK,
                    f"{fragment.stanza()} carried constants over from "
                    f"{fragment.transferred_from!r}, measured against "
                    f"{component} {version!r}, into a spec pinned to "
                    f"{component} {here!r}",
                    "transfer from a spec pinned to this stack, or measure the "
                    "constants on it; the untested assumption is transfer "
                    "within a software generation, not across one",
                )


def validate(
    subject: object,
    *,
    tp_widths: Sequence[int] = (),
    observed_stack: Mapping[str, str] | None = None,
    strict: bool = False,
) -> Validation:
    """Check a document, or a merge of fragments, and report every refusal."""
    merged = subject if isinstance(subject, Merge) else None
    document = merged.document if merged is not None else subject
    resolved: dict[str, Any] = {}
    refusals, read = _complete(document, resolved)
    spec = None
    if not refusals:
        try:
            spec = MachineSpec.from_mapping(document)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    # The consistency questions go through the fields that checked out, whether
    # or not they add up to a spec. Where there is no spec this stands in for
    # one, and it never leaves this function: a partial document is not a
    # machine, and `Validation.spec` must not offer it as one.
    asked_of = spec if spec is not None else MachineSpec(resolved, TokenizerTable(()))
    refusals += _widths(asked_of, tp_widths, resolved)
    refusals += _probes(tp_widths, resolved)
    differences: tuple = ()
    if observed_stack is not None:
        differences = asked_of.check_stack(observed_stack, carried_only=True)
        if differences and strict:
            refusals.append(
                SpecRefusal(
                    Rule.PINNED_STACK,
                    "this spec's constants were measured against "
                    + ", ".join(
                        f"{component} {pinned!r} (now {seen!r})"
                        for component, pinned, seen in differences
                    ),
                    "measure them on the stack that is loaded, or drop the "
                    "strict check and take responsibility for the difference",
                )
            )
    if merged is not None:
        refusals += _transfers(resolved, merged)
    unasked, in_part = _reach(resolved, tp_widths, observed_stack, merged, read)
    return Validation(spec, tuple(refusals), differences, unasked, in_part)
