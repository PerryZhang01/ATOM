# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""A block count, carrying the readings ATOM's budget arithmetic sized it from.

`get_num_blocks` answers one integer, and by the time the scheduler reads it
the five readings and their terms are gone. An integer that came out of a
declared input looks exactly like one that came off a card. `SizedKVPool` is
the count kept beside its inputs, so the decomposition survives the arithmetic
instead of being recoverable only from a log line nobody kept.

**No arithmetic ATOM owns is repeated here.** The budget formula, the 2%
margin, the `min(budget, free)` clamp and `plan_pools` produced the count; this
module receives it. What it computes is over the readings only -- which of them
were subtracted, and how many of those bytes each method accounts for -- and
none of that enters the count.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from atom.compass.backends.provenance import Method
from atom.compass.memory.readings import DeviceReadings, MemoryRefusal
from atom.compass.memory.terms import Reading

#: The readings ATOM subtracts from the utilisation budget before it sizes the
#: pool. `total` scales the budget rather than being subtracted from it, and
#: `free` only clamps -- and `free` is derived from the other three, so counting
#: it here would count those three twice and halve every fraction below.
SUBTRACTED = ("peak_torch", "non_torch", "cudagraph_overhead")


@dataclass(frozen=True, slots=True)
class SizedKVPool:
    """A block count, and how much of what sized it was declared.

    `entries` is the whole per-class entry table the plan published, not the
    paged count alone: a state class that took its floor first is why the paged
    count is what it is, and a record that dropped it could not say so.
    """

    num_kvcache_blocks: int
    entries: Mapping[str, int]
    readings: DeviceReadings

    def __post_init__(self) -> None:
        if isinstance(self.num_kvcache_blocks, bool) or not isinstance(
            self.num_kvcache_blocks, int
        ):
            raise TypeError(
                f"a block count is a whole number, got {self.num_kvcache_blocks!r}"
            )
        if self.num_kvcache_blocks <= 0:
            raise MemoryRefusal(
                f"{self.num_kvcache_blocks} blocks is not a pool, so this is a "
                "record of a run that did not start rather than of one that was "
                "sized; the block manager asserts a positive count before it "
                "builds anything",
                "read the refusal the sizing itself raised -- it names the term "
                "that did not fit -- rather than recording the count it never "
                "reached",
            )

    @property
    def subtracted(self) -> tuple[Reading, ...]:
        """The readings that were taken off the budget, in the order they are."""
        readings = self.readings.as_dict()
        return tuple(readings[name] for name in SUBTRACTED)

    @property
    def subtracted_bytes(self) -> int:
        """The non-KV footprint: what the pool did not get."""
        return sum(reading.total for reading in self.subtracted)

    @property
    def bytes_by_method(self) -> dict[Method, int]:
        """The subtracted bytes, summed by the method of each term."""
        found: dict[Method, int] = {}
        for reading in self.subtracted:
            for term in reading.terms:
                found[term.method] = found.get(term.method, 0) + term.nbytes
        return found

    @property
    def declared_terms(self) -> tuple[str, ...]:
        """Every subtracted term still standing on a declared coefficient."""
        return tuple(
            f"{reading.name}.{term.name}"
            for reading in self.subtracted
            for term in reading.terms
            if term.method is Method.DECLARED
        )

    @property
    def declared_bytes(self) -> int:
        """How many of the subtracted bytes those terms account for."""
        return self.bytes_by_method.get(Method.DECLARED, 0)

    @property
    def declared_fraction(self) -> float:
        """The declared share of the non-KV footprint, or 0.0 if there is none.

        A footprint of no bytes is a card with nothing on it, which is not a
        state this runs in; the guard is here so the ratio cannot be the thing
        that raises while a caller is trying to print why something refused.
        """
        subtracted = self.subtracted_bytes
        return self.declared_bytes / subtracted if subtracted else 0.0

    def declared_line(self) -> str:
        """One sentence a reader cannot take the block count without."""
        if not self.declared_terms:
            return "declared: none of the subtracted footprint is a coefficient."
        return (
            f"declared: {self.declared_bytes:,} of {self.subtracted_bytes:,} B "
            f"({self.declared_fraction:.2%}) of the footprint subtracted from "
            "the budget is a coefficient somebody wrote down, not a "
            "measurement of this card: " + ", ".join(self.declared_terms)
        )

    def table(self) -> str:
        """The count, the entry table, the five readings, and bytes by method."""
        entries = ", ".join(f"{name}={count:,}" for name, count in self.entries.items())
        return "\n".join(
            [
                f"KV pool: {self.num_kvcache_blocks:,} paged blocks"
                + (f"; entries {entries}" if entries else ""),
                self.readings.table(),
                "by method: "
                + ", ".join(
                    f"{method} {nbytes:,} B"
                    for method, nbytes in sorted(
                        self.bytes_by_method.items(), key=lambda i: i[0].value
                    )
                ),
                self.declared_line(),
            ]
        )

    def __str__(self) -> str:
        return self.table()
