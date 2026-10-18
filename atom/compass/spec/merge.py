# SPDX-License-Identifier: MIT
"""Fragments combined into one spec document, refusing what one host cannot be.

A probe measures part of a machine and emits a **fragment**: a partial document
plus the stanza saying which machine it is for, by whom, when and how.

Nothing in a fragment's shape says which host it came from, so fragments that
name different machines are refused, and so are fragments that name one machine
and disagree about a field, with both readings and both stanzas. The tokenizer
probe writes the core counts of the processor it ran on, so a tokenizer swept
on another host disagrees there.

The tokenizer table is combined per entry, keyed by id and backend; one id over
two fingerprints is refused. The provenance block is rebuilt: it names every
fragment, those a saved merge names included, carries the latest date, and
records each field's origin in `provenance.origins`, a field two fragments
agree on taking the stronger origin. The document states one origin only when
every field has it, else `mixed`.

The result is a document, not a spec; `validate` says whether it is complete.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fields import SCHEMA, check
from .machine import TOKENIZERS, _walk, check_subject, entry_label, field_origin
from .rules import Rule, SpecRefusal

#: The block holding the stack the spec's numbers were taken on.
PINNED_BLOCK = "device.software_pinned_to"
#: What every fragment states, whatever else it carries.
STATED = (
    "schema_version",
    "name",
    "provenance.authored_by",
    "provenance.date",
    "provenance.origin",
)
#: Which origin a field two fragments agree on takes, strongest first.
STRENGTH = ("probed", "datasheet", "assumed")


@dataclass(frozen=True, slots=True)
class Fragment:
    """What one probe read, and the stanza saying which machine it is for."""

    source: str
    values: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, document: object, source: str) -> "Fragment":
        """Read a partial document, checking every field it does carry."""
        if not isinstance(document, Mapping):
            raise SpecRefusal(
                Rule.SHAPE,
                f"fragment {source!r} is {type(document).__name__}, not a mapping",
                "a fragment is a partial spec document, shaped like the spec it "
                "contributes to",
            )
        found: dict[str, Any] = {}
        _walk(document, "", found)
        values = {
            field.path: check(field, found[field.path], field.path)
            for field in SCHEMA
            if field.path in found
        }
        for path in STATED:
            if path not in values:
                raise SpecRefusal(
                    Rule.ONE_MACHINE,
                    f"fragment {source!r} does not state `{path}`",
                    "a fragment names the machine it is authored for and how "
                    "its numbers were obtained, because nothing in the numbers "
                    "themselves says which host produced them",
                )
        check_subject(values, f"fragment {source!r}")
        return cls(source, values)

    @property
    def machine(self) -> str:
        """The machine this fragment is authored for, which is the `name` it states."""
        return self.values["name"]

    @property
    def origin(self) -> str:
        """Where it came from: a datasheet, a probe, an assumption, or several
        of those."""
        return self.values["provenance.origin"]

    def stanza(self) -> str:
        """Who probed what, for which machine and when, as a refusal names it."""
        return (
            f"{self.source!r} (machine {self.machine!r}, {self.origin}, "
            f"by {self.values['provenance.authored_by']} "
            f"on {self.values['provenance.date']})"
        )


@dataclass(frozen=True, slots=True)
class Merge:
    """The document the fragments make, and which fragment supplied each field."""

    document: Mapping[str, Any]
    sources: Mapping[str, tuple[str, ...]]


def _one_machine(fragments: tuple[Fragment, ...]) -> None:
    claimed: dict[str, Fragment] = {}
    for fragment in fragments:
        claimed.setdefault(fragment.machine, fragment)
    if len(claimed) > 1:
        first, second = list(claimed.values())[:2]
        raise SpecRefusal(
            Rule.ONE_MACHINE,
            f"{first.stanza()} and {second.stanza()} are authored for "
            f"different machines, {first.machine!r} and {second.machine!r}",
            "one spec describes one host; numbers from two hosts in one "
            "document describe neither, and nothing in their shape would ever "
            "say so. Author a spec per machine",
        )


def _conflict(
    where: str, held: Any, first: Fragment, value: Any, second: Fragment
) -> None:
    stack = where.startswith(PINNED_BLOCK)
    raise SpecRefusal(
        Rule.PINNED_STACK if stack else Rule.ONE_MACHINE,
        f"`{where}` is {held!r} in {first.stanza()} and {value!r} in "
        f"{second.stanza()}",
        (
            "these constants track the compute stack as much as the die, so "
            "fragments pinned to different stacks are not measurements of one "
            "machine"
            if stack
            else "two fragments that name one machine cannot disagree about it; "
            "settle which reading belongs to this host before combining them"
        ),
    )


def _identity(raw: object, fragment: Fragment, index: int) -> tuple:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("id"), str):
        raise SpecRefusal(
            Rule.TOKENIZER_IDENTITY,
            f"`{TOKENIZERS}[{index}]` in {fragment.source!r} states no id",
            "a tokenizer entry carries an author-chosen id; it is what two "
            "fragments conflict on, so an entry without one combines with "
            "nothing",
        )
    return raw["id"], raw.get("backend")


def _strength(found: dict) -> int:
    return STRENGTH.index(found["origin"])


def _provenance(fragments: tuple[Fragment, ...], origins: dict) -> dict:
    stated = dict.fromkeys(
        found["origin"] for found in origins.values()
    ) or dict.fromkeys(fragment.origin for fragment in fragments)
    authors = dict.fromkeys(
        fragment.values["provenance.authored_by"] for fragment in fragments
    )
    notes = [
        fragment.values["provenance.notes"]
        for fragment in fragments
        if "provenance.notes" in fragment.values
    ]
    named: dict[str, None] = {}
    for fragment in fragments:
        for earlier in fragment.values.get("provenance.fragments", ()):
            named.setdefault(earlier, None)
        named.setdefault(fragment.source, None)
    block = {
        "authored_by": ", ".join(authors),
        "date": max(fragment.values["provenance.date"] for fragment in fragments),
        "origin": next(iter(stated)) if len(stated) == 1 else "mixed",
        "origins": origins,
        "fragments": list(named),
    }
    if notes:
        block["notes"] = " | ".join(notes)
    return block


def merge(fragments: Sequence[Fragment]) -> Merge:
    """Combine fragments into one document, refusing what one host cannot be."""
    fragments = tuple(fragments)
    if not fragments:
        raise SpecRefusal(
            Rule.ONE_MACHINE,
            "there are no fragments to merge",
            "a spec is built from fragments; with none there is nothing "
            "to build it from",
        )
    _one_machine(fragments)
    values: dict[str, Any] = {}
    supplied: dict[str, Fragment] = {}
    sources: dict[str, tuple[str, ...]] = {}
    origins: dict[str, dict] = {}

    def claim(where: str, holder: dict, key: Any, value: Any, by: Fragment) -> None:
        if where != "schema_version":
            found = field_origin(by.values, where, by.stanza())
            origins[where] = min(origins.get(where, found), found, key=_strength)
        if key in holder:
            if holder[key] != value:
                _conflict(where, holder[key], supplied[where], value, by)
            sources[where] += (by.source,)
            return
        holder[key] = value
        supplied[where] = by
        sources[where] = (by.source,)

    for fragment in fragments:
        for path, value in fragment.values.items():
            if path == "name" or path == TOKENIZERS or path.startswith("provenance."):
                continue
            claim(path, values, path, value, fragment)

    entries: dict[tuple, Any] = {}
    files: dict[str, tuple[Any, Fragment]] = {}
    for fragment in fragments:
        for index, raw in enumerate(fragment.values.get(TOKENIZERS, ())):
            identity = _identity(raw, fragment, index)
            named = files.setdefault(identity[0], (raw.get("fingerprint"), fragment))
            if named[0] != raw.get("fingerprint"):
                raise SpecRefusal(
                    Rule.TOKENIZER_IDENTITY,
                    f"{identity[0]!r} is {named[0]!r} in {named[1].stanza()} and "
                    f"{raw.get('fingerprint')!r} in {fragment.stanza()}",
                    "one identity names one probed file; two fingerprints "
                    "under one id are two revisions nothing will ever compare",
                )
            claim(entry_label(*identity), entries, identity, raw, fragment)
    if entries:
        values[TOKENIZERS] = list(entries.values())

    values["name"] = fragments[0].machine
    for leaf, stated in _provenance(fragments, origins).items():
        values[f"provenance.{leaf}"] = stated
    document: dict[str, Any] = {}
    for field in SCHEMA:
        if field.path not in values:
            continue
        *blocks, leaf = field.path.split(".")
        node = document
        for block in blocks:
            node = node.setdefault(block, {})
        node[leaf] = values[field.path]
    return Merge(document, sources)
