# SPDX-License-Identifier: MIT
"""The machine spec: one document read against the field table, and its readers.

The spec is held as a flat map from dotted path to checked value, so merging,
validating and explaining are per-field operations.

**The schema is closed.** A key with no row in the field table is refused where
it sits. The stated version is read first, so a document written to a later
schema is told this reader is old rather than that its new field is unknown.

**A field has one value, however it is spelled.** A dotted key resolves to the
nested path, so `host.cpu.cores_physical` may be written flat or nested;
writing it both ways is refused, because the nested echo could not equal the
document.

**The resolved spec is total, and it is echoed.** `echo` rebuilds the whole
document from the checked values and `digest` fingerprints it, so an artifact
can name the spec that produced it.

`from_mapping` raises the first thing wrong; `_survey` under it yields every
refusal and keeps walking, for `validate`.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .fields import BLOCKS, BY_PATH, SCHEMA, SCHEMA_VERSION, Field, Kind, check
from .rules import Rule, SpecRefusal, refuse_unknown_key
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


def _stated(field: Field, found: Mapping) -> bool:
    """Whether the document states `field`, refusing a field it owes and leaves
    out, and the derate of an optional peak stated without that peak."""
    if field.of and field.path in found and field.of not in found:
        raise SpecRefusal(
            Rule.DERATE,
            f"`{field.path}` is stated and `{field.of}`, the peak it derates, "
            "is not",
            "state that peak, or drop its derate",
        )
    if field.path in found:
        return True
    if field.required or field.of in found:
        _missing(field)
    return False


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
                    "nested `b` under `a` are one field; state it once",
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
            if _stated(field, found):
                values[field.path] = check(field, found[field.path], field.path)
        check_subject(values, "this spec")
        _check_domain(values)
        return cls(values, table(values[TOKENIZERS]))

    def value(self, path: str) -> Any:
        """One checked field by its dotted path, or a refusal."""
        if path in self.values:
            return self.values[path]
        if path not in BY_PATH:
            refuse_unknown_key(path)
        raise SpecRefusal(
            Rule.TOTALITY,
            f"`{path}` is declared by this schema as optional, and this spec "
            "states no value for it",
            "write it in the document if a reader needs it",
        )

    def origin(self, path: str) -> dict:
        """Where one field came from: its origin and, when probed, the subject
        the probe ran on. A tokenizer entry is asked by its label,
        `host.tokenizers[<id> <backend>]`."""
        labels = [
            entry_label(e.key.id, e.key.backend.value) for e in self.tokenizers.entries
        ]
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
