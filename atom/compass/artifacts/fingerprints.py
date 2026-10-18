# SPDX-License-Identifier: MIT
"""What a row's fingerprint is, and what a mismatch in one is allowed to say.

A fingerprint is not one digest, and there is no method here that makes one.
It is **the readings of the axes the row depends on, kept apart**, because an
artifact that refuses with a single moved hash says only that something
changed, and nothing about which axis did. Keeping the cells apart is what lets
a refusal say `memory_readings x model`, and what lets a `price_list` be shown
surviving the same change.

**A reading states what kind of thing it is, and two kinds are not compared.**
`SourceRoot.revision` holds a git *tree* on the primary path and a git
*commit* on the stamp path, so two entries published from byte-identical
source can carry different text in that field. Comparing across kinds is not
a mismatch, it is a question that cannot be asked, and the two arrive as
different refusals: `NOT_COMPARABLE` says the comparison is unavailable,
`INVALIDATED` says it was made and failed. Returning `False` for the first
would report a change nobody observed.

**The conditions are stated, never probed.** Nothing in this module reads a
device, a driver or an environment variable; a caller hands in one reading per
axis and this module compares them, so a host with one ROCm can express a ROCm
bump.
That is also why the warning flag is narrow: `OnMismatch.WARN` downgrades *the conditions moved* to a
warning, and does not touch *the comparison could not be made*, which has no
answer to downgrade.
"""

import enum
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

from .matrix import MATRIX, Axis, Row, axes_of
from .rules import ArtifactRefusal, Rule


class OnMismatch(enum.Enum):
    """Whether a mismatch refuses or warns. Refusing is the default."""

    REFUSE = "refuse"
    WARN = "warn"


class StaleArtifact(UserWarning):
    """A refusal the explicit flag downgraded. Never raised without the flag."""


@dataclass(frozen=True, slots=True)
class Reading:
    """One axis's observed value, and what kind of thing the value is."""

    kind: str
    value: str

    def __post_init__(self) -> None:
        for field in ("kind", "value"):
            if (
                not isinstance(getattr(self, field), str)
                or not getattr(self, field).strip()
            ):
                raise ArtifactRefusal(
                    Rule.INVALIDATED,
                    f"a reading states {getattr(self, field)!r} for its {field}",
                    "a fingerprint compares text, and an axis that cannot say "
                    "what it read cannot be checked at all",
                )

    @classmethod
    def stated(cls, value: object) -> "Reading":
        """A configured reading: something a person or a config file declared.

        A scalar, and nothing else. `str()` of an arbitrary object records
        its heap address, which differs between two processes and then
        refuses a change nobody made -- the mirror of the mismatch this
        module refuses to invent when two readings are of different kinds.
        `str(None)` is worse and quieter: it records an axis nobody
        measured as the word `None`, indistinguishable from a device
        actually called that, and certifies clean forever after. A bool is
        refused with the rest because `True` and the string `"True"` would
        record the same text, which is why `Key.of` refuses one too.
        """
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"an axis was handed {value!r}, which is not a stated reading",
                "write the text or the number that was configured; str() of "
                "an object records a heap address that moves between "
                "processes, and str(None) records an axis nobody measured "
                "as the word None",
            )
        return cls("stated", str(value))

    @classmethod
    def of_source_root(cls, root) -> "Reading":
        """A reading taken from a provenance row, carrying its `revision_kind`.

        The kind travels with the value on purpose: a `tree` and a `commit`
        are different objects in one field, and the comparison refuses rather
        than reporting a change between them.
        """
        return cls(root.revision_kind, root.revision)

    def as_json(self) -> dict:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_json(cls, document: object) -> "Reading":
        if not isinstance(document, Mapping):
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"{document!r} is not a recorded reading",
                "a recorded cell states the kind of thing it read and the "
                "value it read; republish the entry",
            )
        return cls(kind=document.get("kind", ""), value=document.get("value", ""))


@dataclass(frozen=True, slots=True)
class Conditions:
    """What every axis reads here and now. Every axis, or none of them.

    All are required even though no row depends on all of them: an entry can
    only be certified against conditions that were stated, and a caller that
    omitted an axis would silently certify against whatever the row happened
    not to ask for.
    """

    readings: tuple[tuple[Axis, Reading], ...]

    @classmethod
    def of(cls, **values: object) -> "Conditions":
        """Conditions from every axis name, or a refusal naming them all."""
        wanted = {axis.field: axis for axis in Axis}
        unknown = sorted(set(values) - set(wanted))
        missing = sorted(set(wanted) - set(values))
        if unknown or missing:
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                "these conditions state "
                + (f"no {', '.join(missing)}" if missing else "")
                + (" and " if missing and unknown else "")
                + (f"an unknown {', '.join(unknown)}" if unknown else ""),
                "state every axis: " + ", ".join(sorted(wanted)),
            )
        readings = {
            wanted[name]: (
                value if isinstance(value, Reading) else Reading.stated(value)
            )
            for name, value in values.items()
        }
        return cls(tuple((axis, readings[axis]) for axis in Axis))

    def reading(self, axis: Axis) -> Reading:
        return dict(self.readings)[axis]

    def with_reading(self, axis: Axis, value: object) -> "Conditions":
        """The same conditions with one axis moved -- how a bump is expressed."""
        return Conditions.of(
            **{
                other.field: (value if other is axis else self.reading(other))
                for other in Axis
            }
        )


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """One row's dependency cells, as they read when the entry was published."""

    row: Row
    cells: tuple[tuple[Axis, Reading], ...]

    @property
    def axes(self) -> tuple[Axis, ...]:
        return tuple(axis for axis, _ in self.cells)

    def as_json(self) -> dict:
        return {
            "row": self.row.field,
            "cells": {axis.field: read.as_json() for axis, read in self.cells},
        }

    @classmethod
    def from_json(cls, document: object) -> "Fingerprint":
        if not isinstance(document, Mapping) or "cells" not in document:
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"{document!r} is not a recorded fingerprint",
                "a fingerprint states its row and one cell per axis the row "
                "depends on; an entry without one cannot be checked on load",
            )
        rows = {row.field: row for row in Row}
        row = rows.get(document.get("row"))
        if row is None:
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"`{document.get('row')}` is not a row of the invalidation matrix",
                "the matrix rows " + ", ".join(str(known) for known in Row),
            )
        cells = document["cells"]
        unknown = sorted(set(cells) - {axis.field for axis in Axis})
        if unknown:
            raise ArtifactRefusal(
                Rule.INVALIDATED,
                f"this fingerprint records `{', '.join(unknown)}`, which the "
                "invalidation matrix has no column for",
                "the matrix moved under this entry; re-measure it, because a "
                "recorded cell nothing compares is a check that silently stopped",
            )
        return cls(
            row,
            tuple(
                (axis, Reading.from_json(cells[axis.field]))
                for axis in Axis
                if axis.field in cells
            ),
        )


@dataclass(frozen=True, slots=True)
class Mismatch:
    """One cell that moved, or one that cannot be compared. Never a total."""

    row: Row
    axis: Axis
    recorded: Reading
    current: Reading
    comparable: bool

    @property
    def text(self) -> str:
        cell = f"`{self.row}` x `{self.axis}`"
        if not self.comparable:
            return (
                f"{cell}: recorded a {self.recorded.kind} and this run reads a "
                f"{self.current.kind}"
            )
        note = MATRIX[self.row][self.axis].note
        because = f" ({note})" if note else ""
        return (
            f"{cell}{because}: recorded {self.recorded.value!r}, this run reads "
            f"{self.current.value!r}"
        )


def fingerprint(row: Row, conditions: Conditions) -> Fingerprint:
    """A row's fingerprint: the axes the matrix rows it against, and no others."""
    return Fingerprint(
        row, tuple((axis, conditions.reading(axis)) for axis in axes_of(row))
    )


def differences(recorded: Fingerprint, current: Fingerprint) -> tuple[Mismatch, ...]:
    """Every cell that moved between two fingerprints of one row.

    Every cell, not the first: a load that names one moved cell when three
    moved has reported an aggregate without its decomposition.
    """
    if recorded.row is not current.row:
        raise ArtifactRefusal(
            Rule.NOT_COMPARABLE,
            f"a `{recorded.row}` fingerprint was compared with a `{current.row}` one",
            "compare a row against itself; different rows depend on different "
            "axes, which is the whole reason this is not one fingerprint",
        )
    if recorded.axes != current.axes:
        raise ArtifactRefusal(
            Rule.NOT_COMPARABLE,
            f"`{recorded.row}` was fingerprinted over "
            f"{', '.join(str(axis) for axis in recorded.axes) or 'nothing'}, but "
            "the matrix now rows it over "
            f"{', '.join(str(axis) for axis in current.axes) or 'nothing'}",
            "the matrix moved under this entry; re-measure it, because an "
            "entry checked against a column it never recorded is unchecked",
        )
    found = []
    for (axis, was), (_, now) in zip(recorded.cells, current.cells):
        if was.kind != now.kind:
            found.append(Mismatch(recorded.row, axis, was, now, False))
        elif was.value != now.value:
            found.append(Mismatch(recorded.row, axis, was, now, True))
    return tuple(found)


def verify(
    recorded: Mapping[Row, Fingerprint],
    conditions: Conditions,
    on_mismatch: OnMismatch = OnMismatch.REFUSE,
) -> tuple[Mismatch, ...]:
    """Check every recorded row against the conditions in force.

    Refuses by default and warns only under the explicit flag -- which reaches
    the mismatches only. A comparison that could not be made is refused either
    way, because there is no answer to downgrade; the refusal still names the
    cells that *did* compare, so a caller who re-takes the reading does not
    then discover a second cell that had already moved.
    """
    found: list[Mismatch] = []
    for row, was in recorded.items():
        found.extend(differences(was, fingerprint(row, conditions)))
    unavailable = [moved for moved in found if not moved.comparable]
    if unavailable:
        raise ArtifactRefusal(
            Rule.NOT_COMPARABLE,
            "; ".join(moved.text for moved in found),
            "a git tree and a git commit are different objects in one field, "
            "so two entries from byte-identical source compare unequal; "
            "re-take the reading the same way the entry did, or re-measure. "
            "Any other cell named above had already moved, and will refuse "
            "again once this one can be compared",
        )
    if not found:
        return ()
    what = "; ".join(moved.text for moved in found)
    remedy = (
        "this entry was measured under other conditions, and these cells are "
        "the ones that decide it; re-measure, or publish a new entry"
    )
    if on_mismatch is OnMismatch.REFUSE:
        raise ArtifactRefusal(Rule.INVALIDATED, what, remedy)
    warnings.warn(
        f"{Rule.INVALIDATED.value}: {what}. {remedy}", StaleArtifact, stacklevel=2
    )
    return tuple(found)
