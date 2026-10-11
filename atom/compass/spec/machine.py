# SPDX-License-Identifier: MIT
"""The machine spec: one document read against the field table, and its readers.

A spec is authored outside Compass, so the only thing that stands between a
mistyped document and a run built on it is this reader. It holds the document as
a flat map from dotted path to checked value. That shape is deliberate and it is
chosen for what comes next: combining two documents, reporting on one, and
explaining where a number came from are all per-field operations, and a nested
mapping makes every one of them a recursive walk written again each time.

Five properties are enforced here rather than described.

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

**A runtime constant has no default.** The widths that were measured are the
widths that can be asked for; a width that was not measured is refused by name,
because these terms fit no law and there is nothing to interpolate along.

**A stack pin is checked.** `check_stack` compares the versions the constants
were measured against with the versions now loaded and warns, naming both, for
every component that differs. The evidence says these constants track the
compute stack at least as much as the die, so a spec transferred across a stack
upgrade is a plausible-looking wrong answer.

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

from .fields import (
    BLOCKS,
    BY_PATH,
    PINNED,
    RUNTIME_CONSTANTS,
    SCHEMA,
    SCHEMA_VERSION,
    Field,
    Kind,
    check,
)
from .rules import (
    Rule,
    SpecRefusal,
    StackMismatch,
    refuse_absent_field,
    refuse_block,
    refuse_unknown_key,
)
from .tokenizers import Backend, TokenizerEntry, TokenizerTable, table

ACTIVATIONS = "device.activations"
#: An activation entry: the model it was measured on, keyed by its
#: architectures and its config.json's sha256 (one architecture serves every
#: size of a family), and per tensor-parallel width one real run's warmup
#: `peak - current` allocated bytes over its warmup tokens, the reading ATOM's
#: `ModelRunner._estimate_cudagraph_overhead` takes.
ACTIVATION_FIELDS = {
    "id": Kind.TEXT,
    "fingerprint": Kind.TEXT,
    "applies_to": Kind.NAMES,
    "bytes_per_token": Kind.WIDTH_TABLE,
}


def _check_activations(entries) -> None:
    """Refuse an entry that is not one, and a model two entries answer for."""
    seen = set()
    for index, entry in enumerate(entries):
        where = f"{ACTIVATIONS}[{index}]"
        if not isinstance(entry, Mapping) or set(entry) != set(ACTIVATION_FIELDS):
            raise SpecRefusal(
                Rule.SHAPE,
                f"`{where}` holds {entry!r}, which is not an activation entry",
                f"write a mapping of exactly {', '.join(ACTIVATION_FIELDS)} there",
            )
        for name, kind in ACTIVATION_FIELDS.items():
            check(Field(name, kind), entry[name], f"{where}.{name}")
        for claim in ((a, entry["fingerprint"]) for a in entry["applies_to"]):
            if claim in seen:
                raise SpecRefusal(
                    Rule.SHAPE,
                    f"{claim[0]!r} with config {claim[1]} resolves to two entries",
                    "one model resolves to one entry; merge their widths into it",
                )
            seen.add(claim)


def _missing(field: Field) -> None:
    if field.path.startswith(RUNTIME_CONSTANTS):
        raise SpecRefusal(
            Rule.NO_DEFAULTS,
            f"`{field.path}` is missing",
            "measure it on the engine and write it down; there is no law "
            "behind this term, so a default would be a guess wearing a number",
        )
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
    width tables and the stack pin, and stopping here would take those out of
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
        _check_activations(values.get(ACTIVATIONS, ()))
        return cls(values, table(values["host.tokenizers"]))

    def value(self, path: str) -> Any:
        """One checked field by its dotted path, or the refusal that fits."""
        if path not in self.values:
            field = BY_PATH.get(path)
            if field is None:
                _unresolved(path)
            refuse_absent_field(path, field.required)
        return self.values[path]

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

    def runtime_constant(self, name: str, tp_width: int | None = None) -> Any:
        """A measured constant, refusing a width nobody measured."""
        path = f"{RUNTIME_CONSTANTS}.{name}"
        field = BY_PATH.get(path)
        if field is None:
            _unresolved(path)
        if field.kind is not Kind.WIDTH_TABLE:
            return self.value(path)
        measured = self.value(path)
        if tp_width is None:
            raise SpecRefusal(
                Rule.NO_DEFAULTS,
                f"`{name}` is measured per tensor-parallel width and none was given",
                f"ask for one of the measured widths {sorted(measured)}",
            )
        if tp_width not in measured:
            raise SpecRefusal(
                Rule.NO_DEFAULTS,
                f"`{name}` was not measured at tensor-parallel width {tp_width}; "
                f"this spec has {sorted(measured)}",
                "measure the engine at that width and add the entry; the "
                "measured widths fit no fixed-plus-per-peer form, so there is "
                "nothing here to interpolate along",
            )
        return measured[tp_width]

    def tokenizer_for(
        self, architecture: str, backend: Backend, fingerprint: str | None = None
    ) -> TokenizerEntry:
        """The measured tokenizer for a model architecture, or a refusal."""
        return self.tokenizers.resolve(architecture, backend, fingerprint)

    def activations_for(
        self, architecture: str | None, fingerprint: str | None, tp_width: int
    ) -> tuple[str, float] | None:
        """The id of the entry measured for this model and its activation bytes
        per token at this width, or None when nothing was measured for it."""
        for entry in self.values.get(ACTIVATIONS, ()):
            if (
                architecture in entry["applies_to"]
                and fingerprint == entry["fingerprint"]
                and tp_width in entry["bytes_per_token"]
            ):
                return entry["id"], entry["bytes_per_token"][tp_width]
        return None

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
                "this spec's constants were measured against "
                + ", ".join(
                    f"{component} {pinned} (now {seen!r})"
                    for component, pinned, seen in differences
                )
                + "; they track the compute stack as much as the die",
                StackMismatch,
                stacklevel=2,
            )
        return differences
