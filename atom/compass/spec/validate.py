"""What a spec is checked for, and why every reason is reported at once.

A spec is authored by hand against a schema most of whose terms cannot be looked
up anywhere, so the document a person first writes is nearly always incomplete.
The whole value of checking it is that an incomplete one is refused rather than
quietly completed: a term nobody probed has no default, because a default here
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
its field declares. The second asks whether the stack the spec's numbers were
taken on is the stack now loaded.

**The second half is asked of whatever resolved, and not only of a complete
spec.** The stack pin is present and schema-valid in a document that the first
half refuses for a single missing derate, and that derate is a value its author
types in at a desk. So the question is asked of the fields that did check out,
and a pin that did not resolve is named rather than passed over in silence.

**That holds for the schema half too, so the walk over the document collects.**
A key the table does not declare, or a block holding a scalar where the schema
has fields, is one refusal about one key -- and reading the document stops
there, which would leave every other field unchecked and the stack question
with nothing resolved to be asked of. So the walk yields its refusals and
carries on, and the document is checked field by field whatever it holds
elsewhere.

**The same record makes the opt-in condition legible.** The stack question
needs `observed_stack=` from the caller, and a clear result that does not say
what it declined to ask is the shape this package exists to refuse. So a run
names, in `not_asked`, each of `CONDITIONS` it could not ask and why, and in
`asked_in_part` each it asked of only part of the fields it reads: a question
asked of two pins where one resolved was asked, and can already have earned a
refusal, so filing it under "not asked" would report a refusal the run did make
as a question it did not. No command validates a machine file: a per-machine
script such as `scripts/compass/mi308x_spec.py` merges its fragments and calls
this module, and nothing in this package reads a spec file.

A stack mismatch on the running machine is the one finding that is not fatal by
default. The numbers are still measurements, taken on a stack that has since
moved, and whether that matters is a judgement about how far it moved -- so it
warns and names both versions, and refuses only when the caller asks for that.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fields import PINNED, SCHEMA, check
from .machine import MachineSpec, _missing, _survey
from .merge import PINNED_BLOCK
from .rules import Rule, SpecRefusal
from .tokenizers import TokenizerTable

#: The conditions a spec is checked for. The first two are asked of anything
#: that is a mapping at all and name themselves in `not_asked` when the subject
#: is not one; the last names itself there when a run could not reach it.
MISSING = "whether every required field is present and every value its declared shape"
DERATES = "whether a spec-peak number carries the derate it obliges"
STACK = "whether the spec's stack pin is the stack now loaded"
#: The whole check set, in the order a run asks it. What `not_asked` is measured
#: against: a condition missing from here is one no result can report on.
CONDITIONS = (MISSING, DERATES, STACK)
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
                f"  stack moved: the constants were probed against "
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
    resolved: Mapping[str, Any], observed_stack: Mapping[str, str] | None, read: bool
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The conditions this run did not ask, and those it asked of only part of
    the fields they read. Each carries its reason."""
    unasked: list[str] = []
    partial: tuple[str, ...] = ()
    if not read:
        why = "the subject is not a mapping, so there was no document to ask it of"
        unasked.append(f"{MISSING} -- {why}")
        unasked.append(f"{DERATES} -- {why}")
    if observed_stack is None:
        unasked.append(f"{STACK} -- no `observed_stack=` was given")
    else:
        none, partial = _unreached(STACK, STACK_PINS, resolved)
        unasked.extend(none)
    return tuple(unasked), partial


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


def validate(
    document: object,
    *,
    observed_stack: Mapping[str, str] | None = None,
    strict: bool = False,
) -> Validation:
    """Check a document and report every refusal."""
    resolved: dict[str, Any] = {}
    refusals, read = _complete(document, resolved)
    spec = None
    if not refusals:
        try:
            spec = MachineSpec.from_mapping(document)
        except SpecRefusal as refusal:
            refusals.append(refusal)
    # The stack question goes through the fields that checked out, whether or
    # not they add up to a spec. Where there is no spec this stands in for one,
    # and it never leaves this function: a partial document is not a machine,
    # and `Validation.spec` must not offer it as one.
    asked_of = spec if spec is not None else MachineSpec(resolved, TokenizerTable(()))
    differences: tuple = ()
    if observed_stack is not None:
        differences = asked_of.check_stack(observed_stack, carried_only=True)
        if differences and strict:
            refusals.append(
                SpecRefusal(
                    Rule.PINNED_STACK,
                    "this spec's constants were probed against "
                    + ", ".join(
                        f"{component} {pinned!r} (now {seen!r})"
                        for component, pinned, seen in differences
                    ),
                    "measure them on the stack that is loaded, or drop the "
                    "strict check and take responsibility for the difference",
                )
            )
    unasked, in_part = _reach(resolved, observed_stack, read)
    return Validation(spec, tuple(refusals), differences, unasked, in_part)
