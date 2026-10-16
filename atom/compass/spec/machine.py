# SPDX-License-Identifier: MIT
"""The machine spec: one document read against the field table, and its readers.

A spec is authored outside Compass, so the only thing that stands between a
mistyped document and a run built on it is this reader. It holds the document as
a flat map from dotted path to checked value. That shape is deliberate and it is
chosen for what comes next: combining two documents, reporting on one, and
explaining where a number came from are all per-field operations, and a nested
mapping makes every one of them a recursive walk written again each time.

Four properties are enforced here rather than described.

**The schema is closed.** A key with no row in the field table is refused where
it sits. That, and not a list of excluded knobs, is how the spec stays a
description of the machine: a tensor-parallel width or a block size is refused
because it is not a machine property, with no rule needed that names it.

The stated version is read before any of that, and that ordering is the point.
A document written to a later schema states fields this reader does not
declare, and a closed schema refuses those wherever they sit, so a version
checked after the fields never gets to speak in the one case it exists for: the
author of a newer document would be told their new field is illegitimate rather
than that this reader is old.

**A field has one value, however it is spelled.** A dotted key resolves to the
nested path, so a document may write `host.cpu.cores_physical` flat -- a probe
fragment does -- or nested, and the two are the same field. Writing it both
ways is refused and the path is named.

The rule is about the shape and not about the two values, which is the part
worth stating because the narrower rule looks equivalent. The echo is rebuilt
nested from the field table, so it cannot equal a document that also states
the field flat, whatever those two keys hold -- refusing only where they
disagree would leave a document whose echo is missing a key it carries, which
is the honesty measure gone either way. Where they do disagree there is a
second cost on top: keeping one of the two would keep whichever the mapping
yielded last, silently, and differently on a document whose keys were written
in another order.

**A stack pin is checked.** `check_stack` compares the versions the spec's
numbers were taken on with the versions now loaded and warns, naming both, for
every component that differs.

**The resolved spec is total, and it is echoed.** `echo` rebuilds the whole
document from the checked values, and `digest` fingerprints it. A run artifact
carries the echo, so every number it reports can be traced back to the spec that
produced it; a partial echo would break that, so the echo is built from the
field table rather than from whatever the reader happened to keep. Totality is
`from_mapping`'s doing rather than the dataclass's: the checking verb builds one
out of the fields a document did resolve, deliberately, so that it can ask its
consistency questions of a document that is not yet a spec.

The walk over the document has two forms for the same reason. `from_mapping` is
a reader and raises the first thing wrong; `_survey` under it yields every
refusal and keeps walking, so a checking caller still gets the fields that sit
beside a mistyped key. One unrecognised key would otherwise empty the whole
document of resolved values, and the questions asked over them with it.

Totality is also why `value` declines a path it cannot resolve in three
different ways. Against a spec read from a document, a path that does not
resolve is either a key the schema does not declare, whose reader belongs at
the field table, or a block, whose reader is one dotted segment away from what
they wanted. A spec assembled from parts -- a probe fragment, or an object built
by hand in a test -- can instead lack a field the schema does declare, and
sending that reader to the table sends them to find the field sitting in it and
stop.
"""

import hashlib
import json
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NoReturn

from .fields import BLOCKS, BY_PATH, PINNED, SCHEMA, SCHEMA_VERSION, Field, Kind, check
from .rules import (
    Rule,
    SpecRefusal,
    StackMismatch,
    refuse_absent_field,
    refuse_block,
    refuse_unknown_key,
)
from .tokenizers import Backend, TokenizerEntry, TokenizerTable, table

#: The tokenizer table, combined and given an origin per entry.
TOKENIZERS = "host.tokenizers"


def entry_label(identity: str, backend: object) -> str:
    """How one tokenizer entry is named, by the two fields that select it."""
    return f"{TOKENIZERS}[{identity} {backend}]"


def field_origin(values: Mapping[str, Any], label: str, holder: str) -> dict:
    """Where the value under `label` came from: the origin recorded for it in
    `provenance.origins`, or the document's own when it states one origin."""
    recorded = values.get("provenance.origins", {}).get(label)
    if recorded is not None:
        return dict(recorded)
    stated = values["provenance.origin"]
    if stated == "mixed":
        raise SpecRefusal(
            Rule.TOTALITY,
            f"{holder} is `mixed` and records no origin for `{label}`",
            "rebuild it with `merge`, which writes `provenance.origins`; a "
            "document of several origins cannot say which one a field has",
        )
    return {"origin": stated, **values.get("provenance.subject", {})}


def check_subject(values: Mapping[str, Any], holder: str) -> None:
    """Refuse a subject on anything but a probed document."""
    stated = values["provenance.origin"]
    if "provenance.subject" in values and stated != "probed":
        raise SpecRefusal(
            Rule.SHAPE,
            f"{holder} states `provenance.subject` and its origin is {stated!r}",
            "a subject is what a probe ran on; state it on a probed document only",
        )


def _check_domain(values: Mapping[str, Any]) -> None:
    """Refuse a scale-up fabric that joins fewer devices than one node holds."""
    joined = values["interconnect.intra_node.domain_devices"]
    per_node = values["device.count_per_node"]
    if joined < per_node:
        raise SpecRefusal(
            Rule.SHAPE,
            f"`interconnect.intra_node.domain_devices` is {joined}, fewer than "
            f"`device.count_per_node` {per_node}",
            "write the number of devices the intra-node fabric joins, every "
            "device of the node and any it reaches beyond it",
        )


def _missing(field: Field) -> None:
    if field.kind is Kind.DERATE:
        raise SpecRefusal(
            Rule.DERATE,
            f"`{field.path}` is missing, and its block states a spec peak",
            "declare the fraction of that peak a kernel actually reaches, so "
            "nobody spends a datasheet number as an achievable one",
        )
    raise SpecRefusal(
        Rule.SHAPE,
        f"`{field.path}` is missing",
        "the spec describes one machine completely; a term left out would be "
        "discovered by a run rather than by the person writing the document",
    )


def _holds(block: str) -> tuple[str, ...]:
    """What a block groups, one segment down, in the order the schema declares."""
    depth = block.count(".") + 1
    names = dict.fromkeys(
        path.split(".")[depth] for path in BY_PATH if path.startswith(f"{block}.")
    )
    return tuple(names)


def _unresolved(path: str) -> NoReturn:
    """The refusal a path earns when the schema declares no field at it."""
    if path in BLOCKS:
        refuse_block(path, _holds(path))
    refuse_unknown_key(path)


def _version(document: Mapping) -> None:
    """Check the version the document states, before the fields it governs.

    A document that states no version is left to the field table, which reports
    it missing along with every other term the schema requires.
    """
    field = BY_PATH["schema_version"]
    if field.path not in document:
        return
    stated = check(field, document[field.path], field.path)
    if stated != SCHEMA_VERSION:
        raise SpecRefusal(
            Rule.VERSION,
            f"this document states schema_version {stated}, and this reader "
            f"understands version {SCHEMA_VERSION}",
            "read it with a Compass that knows that version; what a later "
            "schema adds or renames is not this reader's to judge, so nothing "
            "in the document is read past the version itself",
        )


def _survey(node: Mapping, prefix: str, found: dict):
    """Every key of a document, keeping what the table declares and yielding a
    refusal for each key that it does not, or that states a field a second time.

    The walk carries on past a key it refuses, because one unrecognised key
    says nothing about the rest of the document: the block beside it holds the
    capacity and the stack pin, and stopping here would take those out of
    reach of every question asked further on. A reader that has to produce a
    value wants the first refusal and `_walk` below raises it; a check wants
    them all.
    """
    for key, value in node.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if path in BY_PATH:
            if path in found:
                yield SpecRefusal(
                    Rule.SHAPE,
                    f"`{path}` is stated twice by this document, under two "
                    "spellings that resolve to the same field",
                    "a dotted key resolves to the nested path, so `a.b` and a "
                    "nested `b` under `a` are one field; state it once, "
                    "because the echo every run artifact carries is rebuilt "
                    "nested and cannot equal a document that states the field "
                    "flat as well -- and where the two spellings disagree, "
                    "keeping one of them would keep whichever the mapping "
                    "yielded last",
                )
                continue
            found[path] = value
        elif path in BLOCKS:
            if isinstance(value, Mapping):
                yield from _survey(value, path, found)
            else:
                yield SpecRefusal(
                    Rule.SHAPE,
                    f"`{path}` holds {value!r}, where the schema has a block",
                    "write the block's fields there",
                )
        elif path == "provenance.method":
            yield SpecRefusal(
                Rule.SHAPE,
                "`provenance.method` is now `provenance.origin`",
                "rename the key; `merge` writes each field's own origin under "
                "`provenance.origins`",
            )
        else:
            try:
                refuse_unknown_key(path)
            except SpecRefusal as refusal:
                yield refusal


def _walk(node: Mapping, prefix: str, found: dict) -> None:
    for refusal in _survey(node, prefix, found):
        raise refusal


@dataclass(frozen=True, slots=True)
class MachineSpec:
    """One machine, as a document checked against the schema."""

    values: Mapping[str, Any]
    tokenizers: TokenizerTable

    @classmethod
    def from_mapping(cls, document: object) -> "MachineSpec":
        """Read a document, or refuse with the rule that declined and the fix."""
        if not isinstance(document, Mapping):
            raise SpecRefusal(
                Rule.SHAPE,
                f"a spec is a mapping of its sections, not {type(document).__name__}",
                "load the document before reading it as a spec",
            )
        _version(document)
        found: dict[str, Any] = {}
        _walk(document, "", found)
        values = {}
        for field in SCHEMA:
            if field.path not in found:
                if field.required:
                    _missing(field)
                continue
            values[field.path] = check(field, found[field.path], field.path)
        check_subject(values, "this spec")
        _check_domain(values)
        return cls(values, table(values[TOKENIZERS]))

    def value(self, path: str) -> Any:
        """One checked field by its dotted path, or the refusal that fits."""
        if path not in self.values:
            field = BY_PATH.get(path)
            if field is None:
                _unresolved(path)
            refuse_absent_field(path, field.required)
        return self.values[path]

    def origin(self, path: str) -> dict:
        """Where one field came from: its origin and, when probed, the subject
        the probe ran on. A tokenizer entry is asked by its label,
        `host.tokenizers[<id> <backend>]`."""
        if path in ("name", "schema_version") or path.startswith("provenance."):
            raise SpecRefusal(
                Rule.ADDRESSING,
                f"`{path}` describes the document, so it has no origin",
                "ask for a field of host, device or interconnect",
            )
        labels = [
            entry_label(e.key.id, e.key.backend.value) for e in self.tokenizers.entries
        ]
        if (
            path == TOKENIZERS
            or path.startswith(f"{TOKENIZERS}[")
            and path not in labels
        ):
            raise SpecRefusal(
                Rule.ADDRESSING,
                f"`{path}` is not one tokenizer entry of this spec",
                f"a tokenizer entry has its own origin; ask for one of {labels}",
            )
        if path not in labels:
            self.value(path)
        return field_origin(self.values, path, "this spec")

    def echo(self) -> dict:
        """The whole resolved spec, for a run artifact to carry verbatim."""
        document: dict = {}
        for field in SCHEMA:
            if field.path not in self.values:
                continue
            *blocks, leaf = field.path.split(".")
            node = document
            for block in blocks:
                node = node.setdefault(block, {})
            value = self.values[field.path]
            node[leaf] = list(value) if isinstance(value, tuple) else value
        return document

    def digest(self) -> str:
        """A fingerprint of the echo, so an artifact can name the spec it used."""
        canonical = json.dumps(self.echo(), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(canonical.encode()).hexdigest()

    def tokenizer_for(
        self, architecture: str, backend: Backend, fingerprint: str | None = None
    ) -> TokenizerEntry:
        """The probed tokenizer for a model architecture, or a refusal."""
        return self.tokenizers.resolve(architecture, backend, fingerprint)

    def check_stack(
        self, observed: Mapping[str, str], *, carried_only: bool = False
    ) -> tuple:
        """Compare the pinned stack with the loaded one; warn on every difference.

        Each pin is read through `value`, so one this spec holds no value for is
        declined the way any absent field is. With `carried_only`, a component
        this document does not carry is not compared instead, so a partial one
        can still be asked this much of the question; the checking verb asks it
        that way.
        """
        found = []
        for component in PINNED:
            path = f"device.software_pinned_to.{component}"
            if carried_only and path not in self.values:
                continue
            pinned = self.value(path)
            seen = observed.get(component)
            if seen != pinned:
                found.append((component, pinned, seen))
        differences = tuple(found)
        if differences:
            warnings.warn(
                "this spec's constants were probed against "
                + ", ".join(
                    f"{component} {pinned} (now {seen!r})"
                    for component, pinned, seen in differences
                )
                + "; they track the compute stack as much as the die",
                StackMismatch,
                stacklevel=2,
            )
        return differences
