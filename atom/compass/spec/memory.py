# SPDX-License-Identifier: MIT
"""The memory a probe reads off a card, and the readings a spec cannot be built on.

A probe that starts an engine to measure a runtime constant reads four numbers
on every rank, and this is what it keeps them in. **They are kept apart.**
Reducing them on the way in -- carrying forward only the one term the spec wants
-- is what makes the checks below impossible to ask afterwards, because each of
them is a question about a number that the reduction has already thrown away.

**The term being measured is a reading of the device, not of this process.**
It is `(total - free) - reserved`: everything resident on the card that the
torch allocator did not reserve. `total - free` counts every process on the
card, so a neighbouring container's weights are charged to it exactly as this
engine's driver context and collective buffers are. The number is a property of
the card at the moment it was read, and `ABSOLUTE_LIMIT` is therefore a
statement about **how much of the card may be somebody else's** -- not a bound
on what this engine allocated, which this reading cannot see and does not
constrain. A reading that crosses it says the card was busy, and says nothing
whatever about the process that read it.

**Four readings of one card are checked as four readings of one card.** Keeping
them apart is half the job and asking whether they agree is the other half.
`free`, `total` and `reserved` describe one device, so a set of them no device
could have produced -- more free memory than the card has, a negative reserve --
is a failed reading rather than a busy card. The difference between them comes
out at some plausible-looking number either way, which is the argument for
holding them apart turned around: a check handed only the difference cannot
tell an impossible card from a quiet one. So each rank's readings are asked
about each other first, where the rank that produced them can still be named.

**The limit is a multiple, because the case worth catching is a multiple.** Six
engine starts died on a negative cache budget with a neighbour holding 152.01 GB
while the rank itself had reserved 2.94 GB. The record does not say which width
that ran at, so the most forgiving reading of it is against the largest reserve
any probed width predicts, 11.2e9, which it exceeds by a factor of thirteen;
against the single-card prediction it is a factor of a hundred and fifty-six.
The honest readings sit just above their own predictions: 926 MiB at width 1,
6906 at 2, 7266 at 4 and 10704 at 8, against 970.0e6, 7.2e9, 7.6e9 and 11.2e9.

**Those two anchors bound an interval, and 2.0 is a choice inside it.** The
quantity compared below is the minimum across ranks, and the widest honest
minimum is 1.0058 times its width's prediction -- 7241465856.0 against 7.2e9,
at width 2. The widest single rank of any width reads higher, 1.06 times its
own prediction, so an argument made from that rank is the more conservative of
the two and is about a statistic this check never sees. Either way every honest
reading is below 1.06 and the cheapest reading of the hazard is above 13.6, so
any limit between them separates them and nothing measured picks a point in
that range. 2.0 is a round multiple taken from the quiet end of it: it leaves
the widest honest minimum a factor of 1.99 of headroom and declines the
cheapest reading of the hazard by nearly seven. It is a parameter for the same
reason the cross-rank one is: how busy a card may be before its readings stop
describing this engine is a judgement about the machine, and whoever is running
it is better placed to make it than this module.

**The constant is provisional, because every anchor behind it is one box.** The
four honest widths are four readings of one machine, and the hazard is a single
incident whose width was never written down. Nothing here establishes whether
2.0 is a property of this class of hardware or of that machine, and a second
machine is what would settle it -- so until one has been read this is a working
figure rather than a measured one, and it should move when one is.

**What the limit lets through is what the spec then carries.** It is a
contamination check and not an accuracy bound on the number that survives it.
The cross-rank limit can say that of itself safely, because taking the minimum
has already discarded the rank a neighbour inflated; this one cannot, because
the minimum is exactly what it hands on. A reading just inside it -- at width
8, anything below 22.4e9 against a prediction of 11.2e9 -- is written into the
spec as that width's reserve, so the number the spec carries can be up to twice
the one that is true of this engine. The engine takes that reserve out of the
cache budget one for one, so the whole of the excess -- up to another 11.2e9
bytes at width 8, about 3.9% of the 288.0e9-byte cards these readings were
taken on -- comes out of the cache on every run that reads the spec, and
nothing downstream asks the question a second time.

**This is the half that makes the cross-rank check sufficient.** The spread
across ranks is differential, so it cannot see common-mode contamination: eight
ranks each reading 2.5 GB high agree to the byte and carry the 2.5 GB into the
spec unremarked. This check is absolute and sees exactly that case. It can only
be asked because the reading was kept as `free` and `total` rather than as the
difference between them -- the difference alone is consistent with a quiet card
and a large engine, and with a small engine on a crowded one, and a check that
receives only the difference cannot tell those apart.

**The prediction is the caller's, and there is one case with nothing to supply
it.** What the collective terms predict for the width comes from a spec: the
table already written for this machine, or a transferred one for a machine of
the same type. On a card nobody has probed -- which is the case an engine is
being started for -- that number is the one the probe is about to produce, and
there is nothing to pass. This function neither answers that nor hides it: the
prediction is a required argument with no default, so a caller with nothing to
pass is stopped here rather than handed a value invented on their behalf.
Taking it from a neighbouring width of the spec being built is a judgement about
how the term moves with width, and it belongs to whoever is building the spec.

**A reading whose free memory was binding describes the neighbours.** The engine
clamps its cache to what is free on the card, `min(budget, free)`, so when free
is the smaller of the two the cache it built is the size of the gap somebody
else left, and every constant probed beside it moves with whoever was on the
box. That is a reading to take again rather than a number to write down, and it
is asked per rank, where the rank that produced it can still be named.

Four checks, then, narrowing: whether one rank's readings describe a card at
all, whether the cache that rank built was sized by its own budget, the ranks
against each other, and the one number that survives them. A contaminated run
can fail more than one, and **each check is asked of every rank before the next
check is asked**, so the one raised is the first in that order. Asking them the
other way round -- both questions of rank 0, then both of rank 1 -- would make
the refusal a function of the order the ranks were listed in: two readings, one
that is no card and one whose cache a neighbour sized, earn a different refusal
depending on which of them is rank 0. The two name different things to repair,
so a run sent after the wrong one can fix what it was told to fix and see the
same refusal again. The rank a refusal names is still the first that failed the
check that fired, which is a question about which reading came in where and has
no other answer.

One difference from the engine's own arithmetic is deliberate: the engine floors
this term at zero and this does not. A floor turns an impossible reading into a
plausible one, and a probe wants the impossible reading declined while the ranks
that produced it are still in hand.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from .fields import Field, Kind, check
from .ranks import SPREAD_LIMIT, RankSpread, across_ranks
from .rules import Rule, SpecRefusal

#: What the reading is called wherever it is reported, so a refusal, a spread
#: and the engine's own start-up line all name one quantity.
NON_TORCH = "non_torch"

#: The largest multiple of the reserve predicted for a width that a device-wide
#: reading may be. Not a bound on this process: the reading counts the whole
#: card, so this says how much of the card may belong to somebody else before
#: nothing probed on it can be trusted. Provisional -- every reading it is
#: calibrated against came from one machine, and the readings bound it to an
#: interval rather than to this point. What it accepts, it hands on: a reading
#: just under it becomes the width's reserve with no further question asked.
ABSOLUTE_LIMIT = 2.0


@dataclass(frozen=True, slots=True)
class DeviceMemory:
    """One rank's readings, kept apart rather than reduced on the way in."""

    free_bytes: float
    total_bytes: float
    reserved_bytes: float
    kv_budget_bytes: float

    @property
    def non_torch(self) -> float:
        """What the card holds that the torch allocator did not reserve."""
        return (self.total_bytes - self.free_bytes) - self.reserved_bytes

    @property
    def free_was_binding(self) -> bool:
        """Whether free memory, not the budget, set the size of the cache."""
        return self.free_bytes < self.kv_budget_bytes

    @property
    def impossible(self) -> str:
        """What about these readings no single card could have produced."""
        if self.total_bytes <= 0:
            return f"a card of {self.total_bytes!r} bytes"
        if self.free_bytes < 0:
            return f"{self.free_bytes!r} bytes free"
        if self.free_bytes > self.total_bytes:
            return f"{self.free_bytes!r} bytes free of {self.total_bytes!r}"
        if self.reserved_bytes < 0:
            return f"{self.reserved_bytes!r} bytes reserved"
        if self.reserved_bytes > self.total_bytes:
            return f"{self.reserved_bytes!r} bytes reserved of {self.total_bytes!r}"
        return ""

    def __str__(self) -> str:
        return (
            f"{self.free_bytes!r} free of {self.total_bytes!r}, "
            f"{self.reserved_bytes!r} reserved, budget {self.kv_budget_bytes!r}, "
            f"{NON_TORCH} {self.non_torch!r}"
        )


def non_torch_across_ranks(
    tp_width: int,
    readings: Sequence[DeviceMemory],
    predicted_bytes: float,
    *,
    limit: float = ABSOLUTE_LIMIT,
    spread_limit: float = SPREAD_LIMIT,
) -> RankSpread:
    """The reserve a spec takes from one engine start, or the reason it cannot."""
    readings = tuple(readings)
    if tp_width < 1:
        raise SpecRefusal(
            Rule.RANK_AGREEMENT,
            f"tensor-parallel width {tp_width!r} has no ranks to read a card on",
            "ask this of a width of one or more; a reduction over no readings "
            "has no minimum to carry into a spec, and no engine was started at "
            "this width for it to describe",
        )
    check(Field("predicted", Kind.QUANTITY), predicted_bytes, "the predicted reserve")
    for rank, reading in enumerate(readings):
        if reading.impossible:
            raise SpecRefusal(
                Rule.DEVICE_WIDE,
                f"rank {rank} reports {reading.impossible}, which is not a "
                f"reading of a card ({reading})",
                "these are one rank's view of one device, so this is a failed "
                "reading rather than a busy card; take them again and find out "
                "what produced them, because the difference between them comes "
                "out plausible whether or not the readings themselves can be "
                "true together",
            )
    # Every rank is asked the first question before any is asked the second, so
    # which of the two a run is refused by is decided by the readings and not by
    # the position a failing rank happened to arrive in.
    for rank, reading in enumerate(readings):
        if reading.free_was_binding:
            raise SpecRefusal(
                Rule.DEVICE_WIDE,
                f"rank {rank} had {reading.free_bytes!r} free against a cache "
                f"budget of {reading.kv_budget_bytes!r}, so what was free on "
                f"the card set the cache size and not the budget ({reading})",
                "a cache sized by the gap a neighbour left is a property of "
                "the neighbour, and every constant probed beside it moves "
                "when they do; take the readings again on a card this engine "
                "has to itself",
            )
    spread = across_ranks(
        NON_TORCH, tp_width, [reading.non_torch for reading in readings], spread_limit
    )
    ceiling = limit * predicted_bytes
    if spread.minimum > ceiling:
        raise SpecRefusal(
            Rule.DEVICE_WIDE,
            f"`{NON_TORCH}` is {spread.minimum!r} on the quietest of the "
            f"{tp_width} ranks, where the collective terms predict "
            f"{predicted_bytes!r} at this width and this accepts up to "
            f"{ceiling!r} ({limit:g}x)",
            "this reading counts every process on the card, so the excess is "
            "resident memory that is not this engine's; the ranks agreeing "
            "about it means they share the company, not that the reading is "
            "clean -- find what else holds memory there and measure again",
        )
    return spread
