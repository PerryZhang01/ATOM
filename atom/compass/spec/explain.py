# SPDX-License-Identifier: MIT
"""Where a number came from: the spec fields under it, and the origin of each.

Each field is reported on its own line, in its own unit, with the fragment that
supplied it and its origin; nothing is summed, since a summed check once hid
three errors, two of which cancelled. A spec-peak number is shown with its
derate and the derated value beside it.

A term is a field or block by dotted path. Tokenizer rates explain per entry
and per rate.
"""

from dataclasses import dataclass
from typing import Any

from .fields import BY_PATH, SCHEMA, Kind
from .machine import MachineSpec, entry_label
from .merge import Merge
from .rules import Rule, SpecRefusal
from .tokenizers import ENTRY_FIELDS


@dataclass(frozen=True, slots=True)
class Contribution:
    """One spec field under a term: what it holds and where it came from."""

    path: str
    value: Any
    derate: float | None
    effective: Any
    supplied_by: tuple[str, ...]
    #: Where the field came from, as `MachineSpec.origin` answers it; None for
    #: the provenance block, which is the record rather than a value.
    origin: dict | None = None

    def __str__(self) -> str:
        line = f"{self.path} = {self.value!r}"
        if self.derate is not None:
            line += f" x {self.derate!r} = {self.effective!r}"
        if self.supplied_by:
            line += f"  [{', '.join(self.supplied_by)}]"
        if self.origin is not None:
            subject = ", ".join(
                f"{k} {v}" for k, v in self.origin.items() if k != "origin"
            )
            line += f"  {self.origin['origin']}" + (f" on {subject}" if subject else "")
        return line


@dataclass(frozen=True, slots=True)
class Explanation:
    """The fields under a term, and the spec that is stated in."""

    term: str
    digest: str
    contributions: tuple[Contribution, ...]

    def __str__(self) -> str:
        return "\n".join(
            [f"{self.term}, from spec {self.digest}"]
            + [f"  {row}" for row in self.contributions]
        )


def _supplied(origin: Merge | None, *labels: str) -> tuple[str, ...]:
    if origin is None:
        return ()
    for label in labels:
        if label in origin.sources:
            return origin.sources[label]
    return ()


def _row(
    path: str,
    value: Any,
    derate: float | None,
    supplied: tuple[str, ...],
    origin: dict | None = None,
) -> Contribution:
    effective = value if derate is None else value * derate
    return Contribution(path, value, derate, effective, supplied, origin)


def _rows(spec: MachineSpec, path: str, origin: Merge | None) -> list[Contribution]:
    field = BY_PATH[path]
    value = spec.values[path]
    derate = spec.values.get(field.derate) if field.peak else None
    if field.kind is Kind.TOKENIZERS:
        rows = []
        for entry in spec.tokenizers.entries:
            label = entry_label(entry.key.id, entry.key.backend.value)
            for rate in ENTRY_FIELDS:
                if rate.kind is not Kind.QUANTITY:
                    continue
                rows.append(
                    _row(
                        f"{label}.{rate.path}",
                        getattr(entry, rate.path),
                        entry.derate if rate.peak else None,
                        _supplied(origin, label, path),
                        spec.origin(label),
                    )
                )
        return rows
    document = path in ("name", "schema_version") or path.startswith("provenance.")
    found = None if document else spec.origin(path)
    return [_row(path, value, derate, _supplied(origin, path), found)]


def explain(
    spec: MachineSpec, term: str, *, origin: Merge | None = None
) -> Explanation:
    """The spec fields under a term, each with its value and source."""
    paths = tuple(
        field.path
        for field in SCHEMA
        if field.path == term or field.path.startswith(f"{term}.")
    )
    held = [path for path in paths if path in spec.values]
    if not held:
        raise SpecRefusal(
            Rule.ADDRESSING,
            f"`{term}` names no field this spec states",
            "ask again by the dotted path of a field or a block of fields the "
            "spec states",
        )
    rows = tuple(row for path in held for row in _rows(spec, path, origin))
    return Explanation(term, spec.digest(), rows)
