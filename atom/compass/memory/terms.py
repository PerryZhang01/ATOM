# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A memory reading and the terms it was composed of, which it cannot shed.

The rule this file carries is about arithmetic that has already gone wrong
once: a summed non-KV memory check read +13.8% while holding three errors,
two of which cancelled, and the largest of them was 25% of its own term. The
conclusion drawn is that every term is validated individually, never as a sum.
A reading object that can be printed only as a total reproduces that failure by
design, so this one cannot be:

- `Reading` holds its terms and derives the total on every read, so the two
  cannot drift apart and there is no constructor that takes a total.
- It defines no `__int__` and no `__index__`. A reading cannot be spent as a
  number by accident; a caller that wants the number asks for `.total` and the
  call site says so.
- `__str__` is the per-term table. Printing a reading prints its decomposition,
  which is the rule against reporting an aggregate without its decomposition,
  made structural rather than remembered.

Each term carries the `Method` a cost term carries (`backends/provenance.py`),
with the same rules: `validated` only on an estimated term, and a declared term
holds a note stating its origin and what replaces it, so the record of what is
still owed is written where the number is.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass

from atom.compass.backends.provenance import Method, Provenance, combine


@dataclass(frozen=True, slots=True)
class Term:
    """One named quantity of bytes, how it was obtained, and where it came from.

    `source` is the dotted path of the spec field, the config fields that were
    read, or the readings that were composed -- specific enough that a reader
    who doubts the number knows where to go and look. `nbytes` may be negative:
    a clean box subtracts, and naming the subtraction as a term is what keeps
    the derivation visible in the table rather than folded into one figure.

    `records` names the runs the bytes were taken from, each by the digest of
    its record, so a comparison can tell a run that tests the term from one
    that built it.
    """

    name: str
    nbytes: int
    method: Method
    source: str
    note: str = ""
    validated: bool = False
    records: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a term is named, or the table it sits in is unreadable")
        if isinstance(self.nbytes, bool) or not isinstance(self.nbytes, int):
            raise TypeError(
                f"{self.name}: bytes are whole, got {self.nbytes!r}; round at the "
                "call site so the rounding is visible where it happens"
            )
        if not self.source.strip():
            raise ValueError(
                f"{self.name} states no source; a number without one is a "
                "defect, and this is the field that carries it"
            )
        try:
            Provenance(self.method, validated=self.validated, note=self.note)
        except (TypeError, ValueError) as e:
            raise type(e)(f"{self.name}: {e}") from None

    @property
    def mib(self) -> float:
        return self.nbytes / (1 << 20)

    @property
    def method_label(self) -> str:
        """The method, and for an estimated term whether it is validated."""
        if self.method is not Method.ESTIMATED:
            return str(self.method)
        return f"estimated, {'' if self.validated else 'un'}validated"


def spec_term(spec, name: str, path: str, nbytes: int) -> Term:
    """A spec field used as a term, its method read off the field's origin.

    A datasheet value is analytical, a probed one measured, and an assumed one
    declared, with a note naming the origin. A probe subject's record goes on
    the term.
    """
    found = spec.origin(path)
    origin = found["origin"]
    method = {"datasheet": Method.ANALYTICAL, "probed": Method.MEASURED}.get(
        origin, Method.DECLARED
    )
    note = ""
    if method is Method.DECLARED:
        note = (
            f"{path} is {origin} in the machine spec {spec.value('name')}; a "
            "probe of this platform replaces it"
        )
    records = (found["record"],) if "record" in found else ()
    return Term(name, nbytes, method, path, note, records=records)


@dataclass(frozen=True, slots=True)
class Reading:
    """One of the readings `get_num_blocks` takes off a card, and its terms."""

    name: str
    terms: tuple[Term, ...]

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("a reading is named after the reading it substitutes")
        if not isinstance(self.terms, tuple) or not self.terms:
            raise ValueError(
                f"{self.name} has no terms; a reading is its decomposition, so "
                "one with nothing in it is a total wearing a name"
            )
        seen = set()
        for term in self.terms:
            if not isinstance(term, Term):
                raise TypeError(f"{self.name}: not a term: {term!r}")
            if term.name in seen:
                raise ValueError(
                    f"{self.name} carries two terms called {term.name!r}; the "
                    "table would show one number twice and name it once"
                )
            seen.add(term.name)

    @property
    def total(self) -> int:
        """The sum of the terms, folded on every read so it cannot drift."""
        return sum(term.nbytes for term in self.terms)

    @property
    def declared(self) -> tuple[str, ...]:
        """The terms still standing on a declared input."""
        return tuple(t.name for t in self.terms if t.method is Method.DECLARED)

    def as_term(self, name: str = "", *, sign: int = 1, source: str = "") -> Term:
        """This reading as one term: its terms' methods combined, records joined."""
        method = combine(t.method for t in self.terms)
        note = ""
        if method is Method.DECLARED:
            note = (
                f"It sums {', '.join(self.declared)}, which are declared; each "
                f"states its origin and successor in the {self.name} table"
            )
        return Term(
            name or self.name,
            sign * self.total,
            method,
            source or " + ".join(t.name for t in self.terms),
            note,
            records=tuple(sorted({r for t in self.terms for r in t.records})),
        )

    def table(self) -> str:
        """The per-term table: what this reading is, never only what it totals."""
        rows = [(t.name, f"{t.mib:,.1f}", t.method_label, t.source) for t in self.terms]
        rows.append((f"= {self.name}", f"{self.total / (1 << 20):,.1f}", "", ""))
        widths = [max(len(row[col]) for row in rows) for col in range(4)]
        lines = [f"{self.name}  (MiB)"]
        for name, mib, method, source in rows:
            lines.append(
                f"  {name:<{widths[0]}}  {mib:>{widths[1]}}  "
                f"{method:<{widths[2]}}  {source}".rstrip()
            )
        for term in self.terms:
            if term.note:
                wrapped = textwrap.wrap(term.note, 72)
                lines.append(f"  * {term.name}: {wrapped[0]}")
                lines.extend(f"      {line}" for line in wrapped[1:])
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.table()
