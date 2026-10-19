# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""The projection a cost backend prices: one row per request, and its sums.

**Every sum runs over requests, one at a time.** `BatchView` holds one row per
request and every shape sum is a function of those rows: no constructor takes
a sum in place of the rows, and no supplied sum can disagree with the rows it
was computed from. That is the point rather than fastidiousness: collapsing a
batch to `tokens x history` and multiplying makes the quadratic term and the
cross term fixed multiples of each other over every batch a scheduler can
build. A fit cannot then separate them, and a feature computed wrongly is
indistinguishable from one that is merely unconstrained. Summing per request
was the repair.

State the property precisely, because a stronger claim was made here and it
was measured false. The type does not make the collapsed form unreachable.
A one-row `BatchView` *is* `tokens x history` -- `RequestShape(512, 2048)`
alone sums to 1048576 where the two 256-token rows it collapses sum to
524288 -- and nothing in the type ties the row count to the number of
requests a scheduler scheduled. Shadowing a reader is refused outright (see
`_refuse_shadowing`); a subclass that adds a field and redefines nothing is
accepted, and its sums still come from its rows. That closes the two routes
that fabricate a reading from a field a subclass added. The row count it does
not close and cannot: what closes that is the projection -- one row per
scheduled request, built by whoever holds the scheduled batch -- and that
builder does not live in this package yet.

`capture_rung` is a batch-level scalar, correctly so, because a rung is a
property of the replayed graph rather than of any row. Two disagreements with
the rows are refused: a rung with no decode rows names a graph that replayed
nothing, and a rung narrower than the decode row count makes the padding
negative. Both are contradictions the rows settle by themselves. A rung far
wider than the rows is not -- rows cannot contradict a width, only a ladder of
captured widths can, and `StepStubBackend` has no ladder -- so such a rung is
priced and labelled rather than refused: `capture_rung=10**9` on one decode
row prices `decode.graph_padding` at 99.9999999 s, and that term's provenance
says the width was supplied and nothing here bounds it. A test prices that
batch, so a successor who bounds the rung fails a test rather than leaving
this paragraph stale.

The decode padding feature is the *rung's* rectangle, `rung.max - Sum`, and
not the batch's `len.max - Sum`. A replayed graph runs its rung's worth of
rows whatever the batch brought, so the batch's rectangle understates the
padding by exactly the ratio between the two -- which reads as a small
correction at a full rung and a large one at an empty one, and survives
evidence-widening because it is wrong in proportion to something nobody is
plotting. A rung on a step with no decode work is refused outright: that is
the same error in its other direction, a graph charged for a prefill that
replayed nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


def _refuse_shadowing(owner: type, cls: type) -> None:
    """Refuse a subclass of `owner` that redefines anything `owner` reports.

    The shape features are functions of the rows, which is the property the
    whole form rests on -- but a subclass is accepted by the `isinstance`
    check the backend makes, and a subclass that redefines a reader can hand
    back a number that came from a field it added rather than from the rows.
    `cached_tokens` fabricates the cross term directly; `prefill` synthesises
    a single row holding a whole batch's tokens against a whole batch's
    history, which is the collapsed form the summing exists to keep out.
    Refusing that is the same call `StepCost` in this package already makes,
    for the same reason.

    The protected set is read off `owner` rather than listed, so a reader
    added later is covered the day it is added. `__post_init__` is added to
    it by name because it is the only non-public member whose replacement
    also changes what the object reports: it is where a row's integers and a
    rung's width are checked, and a subclass that drops it admits shapes the
    sums are not defined over.
    """
    protected = {
        name
        for name in (*vars(owner), *owner.__annotations__)
        if not name.startswith("_")
    } | {"__post_init__"}
    shadowed = sorted(protected & set(cls.__dict__))
    if shadowed:
        raise TypeError(
            f"{cls.__name__} redefines {', '.join(shadowed)} of "
            f"{owner.__name__}; the shape sums are functions of the rows, and "
            "a reader that answers from anything else is what summing per "
            "request exists to prevent"
        )


@dataclass(frozen=True)
class RequestShape:
    """One request's share of one step, as the scheduler sized it.

    `query_tokens` is what this step computes for the request: a prefill chunk,
    or the one-or-more tokens a decode step verifies. `context_tokens` is the
    KV that attention reads, which for a prefill chunk is what was cached plus
    the chunk itself. `decode` is given rather than inferred, because a
    one-token prefill and a one-token decode are the same pair of integers and
    different work.
    """

    query_tokens: int
    context_tokens: int
    decode: bool

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_shadowing(RequestShape, cls)

    def __post_init__(self) -> None:
        if self.query_tokens < 1:
            raise ValueError(
                f"a scheduled request computes at least one token, got "
                f"{self.query_tokens}"
            )
        if self.context_tokens < self.query_tokens:
            raise ValueError(
                f"{self.context_tokens} context tokens is fewer than the "
                f"{self.query_tokens} being computed; attention reads at least "
                "what this step writes"
            )

    @property
    def cached_tokens(self) -> int:
        """The history this step did not compute: N_KV minus N_Q."""
        return self.context_tokens - self.query_tokens


def _per_request(
    rows: Sequence[RequestShape], quantity: Callable[[RequestShape], int]
) -> int:
    """Add one quantity over rows, evaluating it on a row before adding.

    Every shape feature below comes through here. The alternative is to
    collapse the batch to a token count and a history length first and
    multiply those, which produces features that are fixed multiples of one
    another and cannot be told apart afterwards.
    """
    total = 0
    for row in rows:
        total += quantity(row)
    return total


def sum_tokens(rows: Sequence[RequestShape]) -> int:
    """Query tokens this step computes."""
    return _per_request(rows, lambda r: r.query_tokens)


def sum_query_square(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q^2 -- the same quantity the scheduler publishes as `sqsq`."""
    return _per_request(rows, lambda r: r.query_tokens * r.query_tokens)


def sum_query_context(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q.N_KV -- the scheduler's `sqsk`."""
    return _per_request(rows, lambda r: r.query_tokens * r.context_tokens)


def sum_query_cached(rows: Sequence[RequestShape]) -> int:
    """Sum N_Q.(N_KV - N_Q), which is `sqsk` minus `sqsq` and not a new reading."""
    return _per_request(rows, lambda r: r.query_tokens * r.cached_tokens)


def sum_context(rows: Sequence[RequestShape]) -> int:
    """Sum N_KV -- the scheduler's `sk`."""
    return _per_request(rows, lambda r: r.context_tokens)


#: The refusal of a backend with no vision encoder in what it prices from.
IMAGES = "{} image tokens, and no vision encoder in what this prices from"


@dataclass(frozen=True)
class BatchView:
    """The projection a cost backend is handed: one row per scheduled request.

    This is the type the backend seam deliberately leaves to whoever prices a
    step, and it lives in this package because the no-engine-imports property
    belongs to a package rather than to a type: the scan that enforces it
    reads the files here and nothing else, so a projection written beside the
    engine would be asserted by nobody. Nothing in this module names an engine
    class. Whoever holds a scheduled batch reads the integers off it and
    builds these rows.

    Every shape sum is a function of the rows: no constructor argument takes a
    sum in place of them, so a caller cannot supply a collapsed pair of
    scalars instead, and cannot supply a sum that disagrees with the rows it
    was supposedly computed from. `capture_rung`, `moe_rows` and
    `produces_output` are the batch-level scalars: a rung is a property of the
    replayed graph, and `moe_rows` is the rows the data-parallel group's MoE
    gather carries, which the MoE segment is priced from. It is None on one
    rank. `draft_moe_rows` is the rows the same gather carries on an MTP
    draft's first pass, which runs each rank's own token count; None on one
    rank or with no such drafter. `moe_tokens` is the real tokens among
    `moe_rows`: a padded gather fills the rest with pad rows that all carry
    token 0 at position 0. None on one rank. `produces_output` says whether
    the step samples, which a pure middle prefill chunk does not; None when
    the caller did not say.
    `tbo_tokens` is the two-batch-overlap split the data-parallel group
    agreed: per micro-batch, each rank's token count. It is None when the
    step is not split, and on one rank. `image_tokens` counts the scheduled
    tokens whose embedding the step's vision encoder supplies; a backend that
    has no basis for pricing that encoder refuses a step with any.

    What this does not do is make the collapsed form unreachable: one row is
    a legal batch, and a one-row batch is `tokens x history` by construction.
    The row count is the projection's guarantee, not the type's.
    """

    requests: tuple[RequestShape, ...]
    capture_rung: int | None = None
    moe_rows: int | None = None
    draft_moe_rows: int | None = None
    produces_output: bool | None = None
    tbo_tokens: tuple[tuple[int, ...], ...] | None = None
    image_tokens: int = 0
    moe_tokens: int | None = None

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        _refuse_shadowing(BatchView, cls)

    def __post_init__(self) -> None:
        rows = tuple(self.requests)
        object.__setattr__(self, "requests", rows)
        if not rows:
            raise ValueError(
                "a step with no requests has no shape to price; the engine runs "
                "no forward for an empty batch, so one arriving here is a "
                "projection built from the wrong thing"
            )
        for row in rows:
            if not isinstance(row, RequestShape):
                raise TypeError(f"not a request shape: {row!r}")
        if self.capture_rung is None:
            return
        decode = self.decode
        if not decode:
            raise ValueError(
                f"capture rung {self.capture_rung} on a step with no decode "
                "requests: a rung names a replayed graph, and a step that "
                "replayed nothing has none"
            )
        if self.capture_rung < len(decode):
            raise ValueError(
                f"capture rung {self.capture_rung} is smaller than the "
                f"{len(decode)} decode requests it pads, which would make the "
                "padding negative"
            )

    @property
    def prefill(self) -> tuple[RequestShape, ...]:
        return tuple(row for row in self.requests if not row.decode)

    @property
    def decode(self) -> tuple[RequestShape, ...]:
        return tuple(row for row in self.requests if row.decode)

    @property
    def graph_padding(self) -> int:
        """`rung.max N_KV - Sum N_KV` over the decode rows; 0 with no rung."""
        rows = self.decode
        if self.capture_rung is None or not rows:
            return 0
        widest = max(row.context_tokens for row in rows)
        return self.capture_rung * widest - sum_context(rows)
