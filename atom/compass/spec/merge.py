# SPDX-License-Identifier: MIT
"""Fragments combined into one spec, and the check that is the point of it.

A probe measures part of a machine and emits a **fragment**: a partial document
plus the stanza that says which machine it is for, by whom, when and how.
Combining fragments is arithmetic over dotted paths and would need very little
said about it. The conflict check is why this module exists.

**One spec describes one host, and nothing about a fragment's shape says which
host it came from.** Two tokenizers probed on two different CPUs contradict
nothing: they have different identities, they occupy different entries, no field
of one overlaps a field of the other, and a merge that only compared values
would combine them into a single document that describes no machine at all. The
error is then invisible for as long as the spec lives, and every number derived
from it is wrong by however far the two hosts differ. So a fragment states the
machine it claims to be for, the merge refuses two fragments that name different
machines, and the refusal prints both stanzas rather than the bare names -- what
a reader needs is which two measurements are being combined, not that a string
comparison failed.

**What that arm compares is the declared name, not the host.** `name` is the
machine the spec is being *authored for* -- `mi355x-8gpu-2node` is a class, not
a host -- and no field of a fragment records where a probe actually ran. So a
tokenizer fragment probed on a laptop and a device fragment probed on the
node, both authored `name: node-18`, pass this arm: it compares what they claim
and a claim is not evidence. The claim it can carry is therefore narrower than
the one above -- the fragments agree about which machine they are *for*,
corroborated by nothing except their overlap. The overlap is where that pair is
caught, and it is the second arm's business: the tokenizer probe writes the core
counts of the processor it ran on into two fields the schema already required,
so the laptop and the node now disagree about `host.cpu.cores_physical`, 8
against 96, and are refused with both readings. Until the probe emitted them the
pair shared no field at all, so the second arm had nothing to contradict and the
exact pair this module opens by naming merged into a document describing neither
host. Closing that residue took no schema field, and it is closed from the
probe's side rather than here.

The second arm catches the same hazard from the other side. Two fragments that
name one machine and disagree about one of its fields cannot both be true of it,
whatever their stanzas say, so a differing value is refused with both readings
and both stanzas. Between them the arms cover the mislabelled fragment and the
unlabelled contradiction, which are the two ways a multi-host spec is authored.

Three kinds of field are combined rather than compared, because contributing
part of one is what a probe does:

* a **runtime constant keyed by width** is combined per width, since the widths
  come from different probe runs -- one rank's engine start fills width 1, and
  the multi-rank run fills the rest. Two fragments that probed the same width
  to different values are still a conflict.
* the **tokenizer table** is combined per entry, keyed by the identity and the
  implementation the rates were probed on. Two entries sharing an identity
  must agree about the file they probed; one name over two files is the drift
  that keying by tokenizer exists to prevent.
* the **provenance** block is rebuilt rather than merged. It names every
  fragment that went in, carries the latest date, and records each field's
  origin in `provenance.origins`: the supplying fragment's, or the entry a saved
  merge recorded for it, with a probe's subject on a probed field. A field two
  fragments agree on takes the stronger origin, so it is assumed only when every
  supplier assumed it. The document states one origin only when every field
  has it -- a spec built from a datasheet and an engine run is honestly `mixed`
  and not either of them. Because a merged document is
  itself a legal fragment, incremental authoring -- merge, save, merge next
  week's probe into it -- would otherwise drop the names of the fragments that
  built the saved document, in the one block whose job is to say where the
  numbers came from. So any `provenance.fragments` a fragment already states is
  carried forward ahead of its own source name. "Latest" is the lexicographic
  maximum of the dates as written: the field is text, ISO-shaped dates order
  correctly under it, and `'2026-9-9'` does not.

Constants that were transferred from another spec are the exception to combining
values. What was transferred is read from each field's own origin, not from the
fragment's `provenance.origin`, and a fragment is a transfer when a field it
supplies outside `host.` is `transferred-from:`. Such a fragment states the
stack the *source* spec was pinned to, as pin fields of that same origin, which
is provenance about the constants and not a claim about this machine's stack, so
those are kept aside for checking rather than contributed. Merging them as
though they described this host would install the source's ROCm version as this
one's and silence the check that exists to catch exactly that. A transferred
`host.` field, such as a tokenizer swept on another machine, is recorded as
transferred and makes nothing a transfer: it was measured on a processor, not on
a device stack. A fragment whose fields name two source specs is refused, since
it has one pin block to state a source stack in.

The result is a document, not a spec. A merge produces what the fragments
happened to cover, and whether that is a complete and consistent machine is a
separate question with its own answer.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .fields import BY_PATH, SCHEMA, TRANSFERRED, Kind, check
from .machine import TOKENIZERS, _walk, check_subject, entry_label, field_origin
from .rules import Rule, SpecRefusal

#: The block holding the stack the device constants were probed against.
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
STRENGTH = ("probed", "datasheet", TRANSFERRED, "assumed")


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
        stated = [values["provenance.origin"]]
        stated += [
            found["origin"] for found in values.get("provenance.origins", {}).values()
        ]
        named = sorted(
            {o[len(TRANSFERRED) :] for o in stated if o.startswith(TRANSFERRED)}
        )
        if len(named) > 1:
            raise SpecRefusal(
                Rule.PINNED_STACK,
                f"fragment {source!r} carries fields transferred from "
                f"{' and '.join(map(repr, named))}",
                "a transfer is checked against its source's stack pin, and a "
                "fragment states one; give each source spec its own fragment",
            )
        return cls(source, values)

    @property
    def machine(self) -> str:
        """The machine this fragment is authored for, which is the `name` it states."""
        return self.values["name"]

    @property
    def origin(self) -> str:
        """Where it came from: a datasheet, a probe, an assumption, another
        spec, or several of those."""
        return self.values["provenance.origin"]

    def carried_from(self, label: str) -> str | None:
        """The spec the field under `label` was carried over from, if it was."""
        stated = field_origin(self.values, label, self.stanza())["origin"]
        return stated[len(TRANSFERRED) :] if stated.startswith(TRANSFERRED) else None

    @property
    def transferred_from(self) -> str | None:
        """The spec this fragment's device or interconnect constants were
        carried over from, if any were; a transferred `host.` field asks for
        no stack pin and does not count."""
        for path, value in self.values.items():
            if not path.startswith(("device.", "interconnect.")):
                continue
            wide = BY_PATH[path].kind is Kind.WIDTH_TABLE
            for label in [f"{path}[{w}]" for w in value] if wide else [path]:
                if source := self.carried_from(label):
                    return source
        return None

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
    fragments: tuple[Fragment, ...]

    @property
    def transfers(self) -> tuple[Fragment, ...]:
        """The fragments whose constants came from another spec."""
        return tuple(f for f in self.fragments if f.transferred_from is not None)


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
    stated = found["origin"]
    return STRENGTH.index(TRANSFERRED if stated.startswith(TRANSFERRED) else stated)


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
            if path.startswith(PINNED_BLOCK) and fragment.carried_from(path):
                continue
            if BY_PATH[path].kind is Kind.WIDTH_TABLE:
                table = values.setdefault(path, {})
                for width, measured in value.items():
                    claim(f"{path}[{width}]", table, width, measured, fragment)
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
    return Merge(document, sources, fragments)
