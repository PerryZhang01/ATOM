# SPDX-License-Identifier: MIT
"""The spread check on a reading taken on every rank of one symmetric group.

`compass measure memory` publishes each rank's own reading and calls this only
to judge how quiet the machine was. Ranks of a symmetric group do identical
work, so the spread across them measures contamination: 0 MiB at widths 1 and
2, 192 MiB at 4 and 640 MiB at 8 on a quiet machine, against the 50-fold case
of a neighbour holding 152 GB beside a rank's 2.9 GB. `SPREAD_LIMIT` sits well
above the first and far below the second. The check is differential, so it
cannot see contamination common to every rank; `measure memory` also checks the
card quiet before and after its start.
"""

from collections.abc import Sequence

from .fields import Field, Kind, check
from .rules import Rule, SpecRefusal

#: The widest cross-rank spread accepted, as a fraction of the smallest reading.
SPREAD_LIMIT = 0.25


def across_ranks(name: str, readings: Sequence[float]) -> None:
    """Refuse readings no spec could hold, or a spread above `SPREAD_LIMIT`."""
    for rank, reading in enumerate(readings):
        check(Field(name, Kind.QUANTITY), reading, f"{name}[rank {rank}]")
    low, high = min(readings), max(readings)
    if (high - low) / low > SPREAD_LIMIT:
        raise SpecRefusal(
            Rule.RANK_AGREEMENT,
            f"`{name}` reads {low!r} on one of the {len(readings)} ranks and "
            f"{high!r} on another, a spread of {(high - low) / low:.1%} where "
            f"this accepts {SPREAD_LIMIT:.1%}",
            "ranks of one symmetric group do identical work, so a spread this "
            "wide is another process on the card rather than the engine; find "
            "what else holds memory there and measure again",
        )
