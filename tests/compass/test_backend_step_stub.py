# SPDX-License-Identifier: MIT
"""The shape-reading stand-in backend, and the loop it exists to close.

Two kinds of test here, and the second is the reason the first is worth
having.

The first kind checks the form: that a step's price is the declared
coefficients multiplied into the shapes of that step, that the breakdown names
every part, that the attention sums are the ones ATOM itself publishes, and
that every shape sum is a function of the rows rather than something a caller
can supply -- including what that does *not* close, which is the row count.

The second kind drives ATOM's real `Scheduler` -- chunked prefill, real
admission, real block manager -- with a virtual clock advanced by nothing but
this backend's predictions, and watches the batch it builds change. That is
the property constants cannot exercise: a fixed price per kind makes every
step the same length, so the clock reaches an arrival at a time that has
nothing to do with the work, and a run cannot show whether the schedule
depends on the model at all.

Nothing below is an accuracy check. The coefficients were declared, and a test
that compared them to a measured duration would be asserting that somebody
guessed well.
"""

import math
from dataclasses import dataclass, fields, replace
from itertools import count
from types import SimpleNamespace

import pytest
from conftest import MockConfig

from atom.compass.backends import (
    BatchView,
    Coefficients,
    CostBackend,
    Granularity,
    KvGeometry,
    Method,
    Parallelism,
    RequestShape,
    StepStubBackend,
    fold_step,
)
from atom.compass.backends.shape import (
    sum_context,
    sum_query_cached,
    sum_query_context,
    sum_query_square,
)
from atom.compass.backends.step_stub import (
    CANDIDATE,
    DECLARED_NOTE,
    DEPTH_STANDS_IN,
    PER_STACK_LAYER,
    UNCHECKED_RUNG,
)
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence, SequenceType
from atom.sampling_params import SamplingParams

# The two switches that publish ATOM's attention aggregates. Both are settable
# from outside the scheduler -- one an env flag read once into
# `_detailed_annotation_enabled`, the other the public `profile_active`, which
# is a plain attribute -- so a stand-in for them is a stand-in for two
# assignments and not for any code that would have to be written. Assign the
# attribute; do not reach for the start-profile RPC that also flips it, because
# that call starts a real torch profiler on the runner first and writes trace
# files, which is not something a run with no device should be doing.
PUBLISHING = SimpleNamespace(profile_active=True, _detailed_annotation_enabled=True)


def prefill(query_tokens, cached=0):
    return RequestShape(query_tokens, cached + query_tokens, False)


def decode(context, query_tokens=1):
    return RequestShape(query_tokens, context, True)


def named(step):
    return {name: seconds for name, seconds, _ in step.rows()}


# ── the form ────────────────────────────────────────────────────────────────


class TestTheForm:
    def test_a_prefill_step_is_its_four_named_parts(self):
        c = Coefficients()
        step = StepStubBackend().price(
            BatchView((prefill(256), prefill(64, cached=128)))
        )
        assert named(step) == {
            "prefill.step": c.prefill_step,
            "prefill.tokens": (256 + 64) * c.prefill_token,
            "prefill.query_square": (256**2 + 64**2) * c.prefill_query_square,
            "prefill.query_cached": (64 * 128) * c.prefill_query_cached,
        }
        assert step.seconds == pytest.approx(sum(named(step).values()))

    def test_a_decode_step_is_its_four_named_parts(self):
        c = Coefficients()
        view = BatchView((decode(1000), decode(600)), capture_rung=4)
        step = StepStubBackend().price(view)
        assert named(step) == {
            "decode.step": c.decode_step,
            "decode.requests": 2 * c.decode_request,
            "decode.context": 1600 * c.decode_context,
            "decode.graph_padding": (4 * 1000 - 1600) * c.decode_padding,
        }

    def test_a_mixed_step_pays_both_groups(self):
        step = StepStubBackend().price(BatchView((prefill(128), decode(900))))
        groups = {name.split(".")[0] for name in named(step)}
        assert groups == {"prefill", "decode"}

    def test_a_step_names_only_the_groups_it_ran(self):
        step = StepStubBackend().price(BatchView((prefill(128),)))
        assert not [name for name in named(step) if name.startswith("decode")]

    def test_the_total_is_folded_from_the_parts_and_not_stored(self):
        step = StepStubBackend().price(BatchView((prefill(300), decode(50))))
        refolded = 0.0
        for _, seconds, _ in step.rows():
            refolded = fold_step(refolded, seconds)
        assert refolded == step.seconds

    def test_a_step_with_no_requests_is_refused(self):
        with pytest.raises(ValueError, match="no shape to price"):
            BatchView(())

    def test_the_backend_answers_at_step_granularity(self):
        backend = StepStubBackend()
        assert isinstance(backend, CostBackend)
        assert backend.granularity is Granularity.STEP

    def test_something_that_is_not_a_projection_is_refused(self):
        with pytest.raises(TypeError, match="prices a BatchView"):
            StepStubBackend().price({"tokens": 512})

    def test_a_coefficient_that_is_not_a_duration_is_refused(self):
        with pytest.raises(ValueError, match="prefill_token"):
            Coefficients(prefill_token=-1.0)
        with pytest.raises(ValueError, match="decode_context"):
            Coefficients(decode_context=math.inf)


# ── the sums are over requests ──────────────────────────────────────────────


class TestPerRequestSums:
    """Summed per request, made structural rather than asked for in prose."""

    def test_no_constructor_takes_a_sum_in_place_of_the_rows(self):
        """Every sum is a function of the rows, so none can be supplied and
        none can disagree with the rows it came from.

        `capture_rung`, `moe_rows`, `produces_output`, `tbo_tokens`,
        `image_tokens` and `moe_tokens` are batch-level values deliberately: a
        rung is a property of the replayed graph, the MoE gather rows, their
        real tokens and the agreed two-batch-overlap split of the data-parallel
        group, and whether it
        samples and what its vision encoder fills of the step, not of any row.
        """
        assert [f.name for f in fields(BatchView)] == [
            "requests",
            "capture_rung",
            "moe_rows",
            "draft_moe_rows",
            "produces_output",
            "tbo_tokens",
            "image_tokens",
            "moe_tokens",
        ]
        assert [f.name for f in fields(RequestShape)] == [
            "query_tokens",
            "context_tokens",
            "decode",
        ]

    def test_one_row_is_a_legal_batch_and_is_the_collapsed_form(self):
        """What the type does not close, said here rather than claimed away.

        A single row carrying a whole batch's tokens against a whole batch's
        history is `tokens x history` exactly. Nothing in the type ties the
        row count to the number of requests a scheduler scheduled; the
        projection does, and the projection is not in this package yet.
        """
        collapsed = BatchView((prefill(512, cached=1536),))
        honest = BatchView((prefill(256, cached=768), prefill(256, cached=768)))
        assert sum_query_context(collapsed.requests) == 512 * 2048
        assert sum_query_context(honest.requests) == 2 * 256 * 1024
        backend = StepStubBackend()
        assert backend.price(collapsed).seconds != backend.price(honest).seconds

    def test_a_projection_subclass_that_redefines_a_reader_is_refused(self):
        """Route two of the collapsed form, closed at the type.

        A frozen subclass adding a batch-level field and overriding `prefill`
        passes the backend's `isinstance` check and prices the collapsed form
        exactly. `StepCost` in this package already refuses the same shape.
        """
        with pytest.raises(TypeError, match="functions of the rows"):

            @dataclass(frozen=True)
            class Collapsed(BatchView):
                tokens: int = 0
                history: int = 0

                @property
                def prefill(self):
                    return (RequestShape(self.tokens, self.history, False),)

    def test_a_row_subclass_that_redefines_a_reader_is_refused(self):
        """Route three: `cached_tokens` fabricates the cross term directly."""
        with pytest.raises(TypeError, match="functions of the rows"):

            @dataclass(frozen=True)
            class Fabricated(RequestShape):
                @property
                def cached_tokens(self) -> int:
                    return 10**9

    def test_the_guard_names_what_the_subclass_redefined(self):
        """A refusal that does not name the member leaves a reader guessing."""
        with pytest.raises(TypeError) as refused:

            @dataclass(frozen=True)
            class Silent(BatchView):
                @property
                def graph_padding(self) -> int:
                    return 0

        assert "graph_padding" in str(refused.value)

    def test_dropping_the_row_checks_is_refused_too(self):
        """`__post_init__` is where a row's integers and a rung's width are
        checked; a subclass that drops it admits shapes the sums are not
        defined over."""
        with pytest.raises(TypeError, match="__post_init__"):

            @dataclass(frozen=True)
            class Unchecked(RequestShape):
                def __post_init__(self) -> None:
                    return None

    def test_a_subclass_that_redefines_nothing_is_accepted(self):
        """The guard refuses shadowing, not subclassing, and this is which.

        Adding a field changes nothing about where the sums come from, so it
        is allowed; the sums still run over the rows.
        """

        @dataclass(frozen=True)
        class Tagged(BatchView):
            label: str = ""

        view = Tagged((prefill(256, cached=768), prefill(256, cached=768)), label="x")
        assert sum_query_context(view.requests) == 2 * 256 * 1024
        assert StepStubBackend().price(view).seconds == pytest.approx(
            StepStubBackend()
            .price(BatchView((prefill(256, cached=768), prefill(256, cached=768))))
            .seconds
        )

    def test_two_batches_that_collapse_alike_are_priced_apart(self):
        """The rank deficiency, shown rather than described.

        Both batches compute 512 tokens against 2048 of context, so a form
        built from `tokens x history` prices them identically. They are not
        the same work: one request attending over 1024 is not two attending
        over 512 each.
        """
        one_big = BatchView((prefill(256, cached=768), prefill(256, cached=768)))
        lopsided = BatchView((prefill(448, cached=576), prefill(64, cached=960)))
        assert sum_context(one_big.requests) == sum_context(lopsided.requests)
        assert sum(r.query_tokens for r in one_big.requests) == sum(
            r.query_tokens for r in lopsided.requests
        )
        backend = StepStubBackend()
        assert backend.price(one_big).seconds != backend.price(lopsided).seconds

    def test_the_sums_are_the_ones_the_scheduler_publishes(self):
        """Checked against ATOM's own method, not against a re-derivation."""
        rows = (prefill(4, cached=2), prefill(3), decode(10))
        batch = SimpleNamespace(
            num_scheduled_tokens=[4, 3, 1],
            detailed_sqsq=None,
            detailed_sqsk=None,
            detailed_sk=None,
        )
        seqs = {
            0: SimpleNamespace(
                type=SequenceType.PREFILL, num_tokens=6, num_cached_tokens=2
            ),
            1: SimpleNamespace(
                type=SequenceType.PREFILL, num_tokens=3, num_cached_tokens=0
            ),
            2: SimpleNamespace(
                type=SequenceType.DECODE, num_tokens=10, num_cached_tokens=9
            ),
        }

        Scheduler.compute_detailed_aggregates(PUBLISHING, batch, seqs)

        assert sum_query_square(rows) == batch.detailed_sqsq
        assert sum_query_context(rows) == batch.detailed_sqsk
        assert sum_context(rows) == batch.detailed_sk

    def test_the_cached_cross_term_is_a_difference_of_two_published_sums(self):
        """`Sum N_Q.N_KV_cached` is `sqsk - sqsq`, so it needs no new reading."""
        rows = (prefill(128, cached=896), decode(2000), prefill(7))
        assert sum_query_cached(rows) == sum_query_context(rows) - sum_query_square(
            rows
        )


# ── the graph rung ──────────────────────────────────────────────────────────


class TestTheRung:
    def test_the_padding_is_the_rungs_rectangle_not_the_batchs(self):
        rows = (decode(1000), decode(200), decode(200))
        view = BatchView(rows, capture_rung=16)
        batchs_own = len(rows) * 1000 - 1400
        assert view.graph_padding == 16 * 1000 - 1400
        assert view.graph_padding > batchs_own

    def test_a_step_that_replayed_nothing_has_no_rung(self):
        with pytest.raises(ValueError, match="no decode requests"):
            BatchView((prefill(256),), capture_rung=8)

    def test_a_rung_narrower_than_the_batch_is_refused(self):
        with pytest.raises(ValueError, match="smaller than the 3 decode"):
            BatchView((decode(10), decode(10), decode(10)), capture_rung=2)

    def test_without_a_rung_the_padding_term_is_zero_and_still_named(self):
        step = StepStubBackend().price(BatchView((decode(10), decode(10))))
        assert named(step)["decode.graph_padding"] == 0.0

    def test_a_rung_no_ladder_would_hold_is_priced_and_labelled(self):
        """The one rung disagreement the rows cannot settle, pinned.

        A rung narrower than its rows is refused because the padding would go
        negative; a rung wider than any ladder would hold is not refusable
        from the rows, so it is charged at face value and the term says the
        width came from the caller. A successor that bounds the rung breaks
        this test rather than leaving the module's paragraph stale.
        """
        view = BatchView((decode(100),), capture_rung=10**9)
        assert view.graph_padding == 10**9 * 100 - 100
        step = StepStubBackend().price(view)
        assert named(step)["decode.graph_padding"] == pytest.approx(99.9999999)
        padding = {name: prov for name, _, prov in step.rows()}
        assert UNCHECKED_RUNG in padding["decode.graph_padding"]

    def test_with_no_rung_the_padding_term_carries_no_rung_qualifier(self):
        """Nothing was supplied, so there is no unchecked width to declare."""
        step = StepStubBackend().price(BatchView((decode(10), decode(10))))
        padding = {name: prov for name, _, prov in step.rows()}
        assert UNCHECKED_RUNG not in padding["decode.graph_padding"]
        assert DECLARED_NOTE in padding["decode.graph_padding"]


# ── constant mode ───────────────────────────────────────────────────────────


class TestConstantMode:
    def test_constant_pricing_is_a_coefficient_set_not_a_mode(self):
        flat = Coefficients.constant(prefill_seconds=0.05, decode_seconds=0.002)
        backend = StepStubBackend(coefficients=flat)
        short = backend.price(BatchView((prefill(16),)))
        long = backend.price(BatchView((prefill(16128),)))
        assert short.seconds == long.seconds == 0.05
        assert set(named(short)) == set(named(long))
        assert flat.shape_blind

    def test_constant_pricing_still_names_every_part(self):
        flat = Coefficients.constant(prefill_seconds=0.05, decode_seconds=0.002)
        step = StepStubBackend(coefficients=flat).price(BatchView((prefill(16128),)))
        assert named(step) == {
            "prefill.step": 0.05,
            "prefill.tokens": 0.0,
            "prefill.query_square": 0.0,
            "prefill.query_cached": 0.0,
        }

    def test_the_default_reads_the_shape(self):
        assert not Coefficients().shape_blind
        backend = StepStubBackend()
        assert (
            backend.price(BatchView((prefill(16),))).seconds
            != backend.price(BatchView((prefill(256),))).seconds
        )


# ── what a reader of the output is told ─────────────────────────────────────


class TestWhatTheOutputSays:
    """The disclaimer has to survive the trip into an artifact."""

    def test_every_term_says_the_coefficient_was_declared(self):
        step = StepStubBackend().price(BatchView((prefill(128), decode(500))))
        for name, _, provenance in step.rows():
            assert DECLARED_NOTE in provenance, name
            assert "not an accuracy claim" in provenance

    def test_every_term_says_which_count_and_which_coefficient(self):
        step = StepStubBackend().price(BatchView((prefill(128),)))
        rows = {name: provenance for name, _, provenance in step.rows()}
        assert "5e-07 s x 128" in rows["prefill.tokens"]
        assert "x 16384" in rows["prefill.query_square"]

    def test_every_term_is_declared_with_its_origin_and_successor(self):
        """A coefficient nobody measured is an assumed input: the method is
        declared, and the note names the step linear law as its successor."""
        step = StepStubBackend().price(BatchView((prefill(128), decode(500))))
        for term in step.terms:
            assert term.provenance.method is Method.DECLARED, term.name
            assert term.provenance.note == DECLARED_NOTE
        assert "successor: the step linear law" in DECLARED_NOTE

    def test_the_description_says_it_and_which_pricing_produced_it(self):
        assert DECLARED_NOTE in StepStubBackend().describe()
        assert "shape-read" in StepStubBackend().describe()
        flat = Coefficients.constant(prefill_seconds=0.05, decode_seconds=0.002)
        assert "constant bring-up" in StepStubBackend(coefficients=flat).describe()


# ── collectives ─────────────────────────────────────────────────────────────


class TestCollectives:
    # One worker, described by two counts: 64 layers run and 16 of them hold a
    # cache of every past token. A collective is charged on the first and a
    # block is sized from the second, so each construction below states the
    # one it needs and neither number stands in for the other.
    GEOMETRY = KvGeometry(
        layers=16, kv_heads=4, head_dim=256, element_bytes=2, block_size=64
    )
    STACK_LAYERS = 64

    def test_a_single_rank_deployment_is_charged_for_no_collective(self):
        step = StepStubBackend(
            parallelism=Parallelism(expert_parallel=True), geometry=self.GEOMETRY
        ).price(BatchView((prefill(128),)))
        assert not [name for name in named(step) if name.startswith("collective")]

    def test_the_charged_collectives_are_the_ones_the_widths_produce(self):
        widths = Parallelism(tp_size=2, dp_size=2, expert_parallel=True)
        step = StepStubBackend(
            parallelism=widths,
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).price(BatchView((prefill(128),)))
        charged = [name for name in named(step) if name.startswith("collective")]
        assert charged == [f"collective.{n}" for n in widths.collectives()]
        assert named(step)["collective.tp-all-reduce"] == pytest.approx(
            128 * self.STACK_LAYERS * Coefficients().collective_token_layer
        )

    def test_a_collective_with_no_stack_depth_to_price_it_is_refused(self):
        """And a geometry does not answer it: what it counts is the subset."""
        with pytest.raises(ValueError, match="how many layers"):
            StepStubBackend(parallelism=Parallelism(tp_size=2))
        with pytest.raises(ValueError, match="how many layers"):
            StepStubBackend(parallelism=Parallelism(tp_size=2), geometry=self.GEOMETRY)

    def test_a_span_shallower_than_its_paged_layers_is_refused(self):
        """The two counts crossed: a subset cannot be larger than the set."""
        with pytest.raises(ValueError, match="have been crossed"):
            StepStubBackend(
                parallelism=Parallelism(tp_size=2),
                geometry=self.GEOMETRY,
                stack_layers=4,
            )

    @pytest.mark.parametrize("depth", [64.7, "64", 64.0])
    def test_a_depth_that_is_not_a_whole_number_of_layers_is_refused(self, depth):
        """Coercing it would be a guessed count inside the refusing constructor.

        `int(64.7)` is a precise 64 from a source that was already wrong, and
        `int("64")` is a count that was never stated as one. The wording is
        the one this package already uses for a dialled count, so the two
        constructors answer the same question the same way.
        """
        with pytest.raises(TypeError, match="whole number of at least 1"):
            StepStubBackend(
                parallelism=Parallelism(tp_size=2),
                geometry=self.GEOMETRY,
                stack_layers=depth,
            )

    def test_a_bool_is_the_one_layer_span_it_equals(self):
        """A `bool` is an `int` to Python, and this takes it as the 1 it is.

        Declared rather than special-cased. Refusing it only here would be a
        second convention for the question the test above settles, and the
        count it yields is the one the value equals rather than a guess at
        what a caller meant by it. `False` is refused by the `< 1` check,
        which is the same guard a stated 0 meets.

        `== 1` alone does not say this happened: `True == 1`, so that
        assertion passes equally against a value kept as the `bool` it
        arrived as. The difference shows in the one place a count is written
        out -- the record line, which exists here because the seconds do not
        carry the count -- so the type and that line are asserted too.
        """
        taken = StepStubBackend(parallelism=Parallelism(tp_size=2), stack_layers=True)
        assert taken.stack_layers == 1
        assert "on 1 layers" in taken.describe()
        assert not isinstance(taken.stack_layers, bool)
        with pytest.raises(ValueError, match="at least one layer"):
            StepStubBackend(parallelism=Parallelism(tp_size=2), stack_layers=False)

    def test_a_collective_is_priced_with_no_geometry_paired_to_it(self):
        """The geometry is not what a collective is charged on, so it is optional.

        Nothing that computes seconds reads it: the charge is the tokens times
        the stack depth times a declared coefficient. Refusing for want of it
        would decline a price this makes identically either way, which is why
        the refusal above is keyed on the depth alone. The cost of that is the
        crossed-count check having one count to compare and the record having
        one to write -- pinned here and in the test below.
        """
        step = StepStubBackend(
            parallelism=Parallelism(tp_size=2), stack_layers=self.STACK_LAYERS
        ).price(BatchView((prefill(128),)))
        assert named(step)["collective.tp-all-reduce"] == pytest.approx(
            128 * self.STACK_LAYERS * Coefficients().collective_token_layer
        )

    def test_the_record_says_when_no_paged_count_was_stated(self):
        """One count written where two describe a worker, said rather than implied.

        A lone `on 64 layers` reads as the whole description of the worker,
        and on a hybrid it is a quarter of one. The line names the count it
        does not have instead.
        """
        paired = StepStubBackend(
            parallelism=Parallelism(tp_size=2),
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).describe()
        alone = StepStubBackend(
            parallelism=Parallelism(tp_size=2), stack_layers=self.STACK_LAYERS
        ).describe()
        assert "on 64 layers, 16 paged" in paired
        assert "on 64 layers, paged count not stated" in alone

    def test_a_charged_collective_says_it_is_a_candidate(self):
        """The widths rule one out conclusively and rule it in conditionally.

        Two of the four conditions that gate the peer-to-peer path are not
        deployment widths, so a presence in the list is what the widths admit
        rather than what a step ran. A term charged for one says which of the
        two it is where the charge is read, since the seconds are indis-
        tinguishable from a confirmed collective's once they are in a total.
        """
        step = StepStubBackend(
            parallelism=Parallelism(tp_size=2, dp_size=2, expert_parallel=True),
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).price(BatchView((prefill(128),)))
        for name, _, provenance in step.rows():
            assert (CANDIDATE in provenance) is name.startswith("collective.")

    def test_a_charged_collective_says_which_layer_count_it_used(self):
        """Two counts describe the worker and the seconds carry neither.

        A charge on the paged layers of this geometry and one on its stack are
        both a number of seconds in a total, so the term states the count it
        multiplied where a reader of the record finds it.
        """
        step = StepStubBackend(
            parallelism=Parallelism(tp_size=2),
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).price(BatchView((prefill(128),)))
        for name, _, provenance in step.rows():
            assert (PER_STACK_LAYER in provenance) is name.startswith("collective.")

    def test_only_the_collective_whose_count_is_argued_claims_it(self):
        """One count for both charges, and an argument behind one of them.

        An all-reduce runs on every layer the worker runs. An expert
        all-to-all runs on the layers holding experts, which is a third count
        and one nothing in this package supplies, so it is charged on the
        depth in hand and its term says that is what happened. Both are the
        same seconds; only the record tells them apart.
        """
        step = StepStubBackend(
            parallelism=Parallelism(tp_size=2, dp_size=2, expert_parallel=True),
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).price(BatchView((prefill(128),)))
        stands_in = {
            name: DEPTH_STANDS_IN in provenance for name, _, provenance in step.rows()
        }
        assert stands_in["collective.tp-all-reduce"] is False
        assert stands_in["collective.moe-all-to-all"] is True
        assert not any(
            value for name, value in stands_in.items() if not name.startswith("coll")
        )
        charged = named(step)
        assert charged["collective.moe-all-to-all"] == pytest.approx(
            charged["collective.tp-all-reduce"]
        )

    def test_the_stand_in_said_of_an_unargued_collective_is_about_that_one(
        self, monkeypatch
    ):
        """The default is attached by absence, so it may not describe a name.

        Today the only unargued name is the expert all-to-all, and a sentence
        about expert-bearing layers is exact for it -- which is why today's
        output cannot tell a general statement apart from that collective's
        own. A second unargued name can. A point-to-point send between
        pipeline stages is charged once per layer like the rest and the depth
        stands in for its count the same way, and nothing about it is
        expert-bearing; the set exists so that such a name is labelled a
        stand-in rather than inheriting an argument made for a different
        collective, so the sentence the absence attaches has to be one that
        name can carry.
        """
        monkeypatch.setattr(
            Parallelism,
            "collectives",
            lambda self: ("tp-all-reduce", "pp-send-recv"),
        )
        step = StepStubBackend(
            parallelism=Parallelism(tp_size=2),
            geometry=self.GEOMETRY,
            stack_layers=self.STACK_LAYERS,
        ).price(BatchView((prefill(128),)))
        said = {name: provenance for name, _, provenance in step.rows()}
        assert DEPTH_STANDS_IN in said["collective.pp-send-recv"]
        assert DEPTH_STANDS_IN not in said["collective.tp-all-reduce"]
        assert "expert" not in said["collective.pp-send-recv"]


# ── chunk size ──────────────────────────────────────────────────────────────


class TestChunkSize:
    """Doubling the chunk, term by term, because the total is not one number's
    story: the token term doubles, the quadratic term quadruples, and the
    launch intercept does not move. Reporting only the ratio of the totals
    would hide all three."""

    @staticmethod
    def _terms(n):
        return named(StepStubBackend().price(BatchView((prefill(n),))))

    def test_the_token_term_doubles_exactly(self):
        assert (
            self._terms(512)["prefill.tokens"] == 2 * self._terms(256)["prefill.tokens"]
        )

    def test_the_quadratic_term_quadruples_exactly(self):
        assert (
            self._terms(512)["prefill.query_square"]
            == 4 * self._terms(256)["prefill.query_square"]
        )

    def test_the_launch_term_does_not_move(self):
        assert self._terms(512)["prefill.step"] == self._terms(256)["prefill.step"]

    def test_the_step_doubles_exactly_where_the_form_is_linear(self):
        """The literal property, in the regime where it can hold: with the
        quadratic and cross coefficients at zero and no launch intercept, the
        form is `b.tokens` and doubling the chunk doubles the step."""
        linear = Coefficients(
            prefill_step=0.0,
            prefill_query_square=0.0,
            prefill_query_cached=0.0,
        )
        backend = StepStubBackend(coefficients=linear)
        short = backend.price(BatchView((prefill(256),))).seconds
        long = backend.price(BatchView((prefill(512),))).seconds
        assert long == 2 * short


# ── the closed loop ─────────────────────────────────────────────────────────

CHUNK = 256
PROMPT_A = 2048
PROMPT_B = 512
ARRIVAL = 7.48e-4


def _project(batch, seqs):
    """Read one batch's shapes off the engine, as the caller of a backend does.

    This lives in the test rather than in the package because the package
    names no engine type: the two branches below -- a prefill chunk's context
    is what was cached plus the chunk, a decode's is the whole sequence -- are
    exactly the ones `compute_detailed_aggregates` takes, and the tests above
    check that the sums come out equal to what it publishes.
    """
    rows = []
    for seq, scheduled in zip(seqs.values(), batch.num_scheduled_tokens):
        query = int(scheduled)
        is_decode = seq.type == SequenceType.DECODE
        context = (
            int(seq.num_tokens) if is_decode else int(seq.num_cached_tokens) + query
        )
        rows.append(RequestShape(query, context, is_decode))
    return BatchView(tuple(rows))


def _drive(coefficients, steps=10, arrival=ARRIVAL, chunk=CHUNK):
    """Run the real scheduler with a clock that only this backend advances.

    The token budget is two chunks wide, so a second request that has become
    admittable can share the step rather than having to wait for the first to
    finish. That is what makes the streak of single-request prefills break
    where the clock says and not where the prompt runs out.
    """
    Sequence.counter = count()
    scheduler = Scheduler(
        MockConfig(
            kv_cache_block_size=16,
            num_kvcache_blocks=1024,
            enable_chunked_prefill=True,
            max_num_seqs=8,
            max_num_batched_tokens=2 * chunk,
            long_prefill_token_threshold=chunk,
            max_model_len=4096,
        )
    )
    backend = StepStubBackend(coefficients=coefficients)
    params = SamplingParams(max_tokens=200)
    scheduler.add(Sequence(list(range(1, PROMPT_A + 1)), 16, sampling_params=params))
    waiting = [
        (arrival, Sequence(list(range(1, PROMPT_B + 1)), 16, sampling_params=params))
    ]
    clock = 0.0
    trace = []
    for _ in range(steps):
        while waiting and waiting[0][0] <= clock:
            scheduler.add(waiting.pop(0)[1])
        batch, seqs = scheduler.schedule()
        Scheduler.compute_detailed_aggregates(PUBLISHING, batch, seqs)
        view = _project(batch, seqs)
        assert sum_query_square(view.requests) == batch.detailed_sqsq
        assert sum_query_context(view.requests) == batch.detailed_sqsk
        assert sum_context(view.requests) == batch.detailed_sk
        trace.append((tuple(batch.req_ids), view, clock))
        clock = fold_step(clock, backend.price(view).seconds)
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(7,) for _ in batch.req_ids],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )
    return trace


def _predicted_break(coefficients, arrival=ARRIVAL, chunk=CHUNK):
    """Which step the second request first appears in, from the form alone.

    Nothing here consults the scheduler. Every step until the arrival is one
    chunk of the first prompt against the context it has accumulated, so the
    clock is a closed sum over the declared coefficients, and the step whose
    batch is assembled after the clock has passed the arrival is the one the
    second request can be in.
    """
    clock = 0.0
    step = 1
    while clock < arrival:
        cached = (step - 1) * chunk
        clock = fold_step(
            clock,
            coefficients.prefill_step
            + coefficients.prefill_token * chunk
            + coefficients.prefill_query_square * chunk * chunk
            + coefficients.prefill_query_cached * chunk * cached,
        )
        step += 1
    return step


def _observed_break(trace):
    for index, (req_ids, _, _) in enumerate(trace, start=1):
        if len(req_ids) > 1:
            return index
    raise AssertionError("the streak never broke")


class TestClosedLoop:
    """Step cost, to a clock, to admission, to a different batch.

    The scheduler is ATOM's own and is not told anything about the cost model.
    The only channel between them is the clock: a step that this backend
    prices as longer is a step during which more of the arrival schedule
    elapses, so a request becomes admittable earlier or later in the streak,
    and the batch the scheduler then builds is a different batch.
    """

    def test_the_streak_of_single_request_prefills_breaks_where_predicted(self):
        coefficients = Coefficients()
        trace = _drive(coefficients)
        assert _observed_break(trace) == _predicted_break(coefficients) == 6
        assert all(len(req_ids) == 1 for req_ids, _, _ in trace[:5])
        assert all(not row.decode for _, view, _ in trace[:6] for row in view.requests)

    def test_dropping_one_term_moves_the_break_by_a_predicted_step(self):
        """The loop is real rather than decorative.

        Over the five steps before the arrival the cross term is worth 2.62 us
        against 749.18 us elapsed -- 0.35% -- and 0.35% is the whole
        difference between the second request making the sixth batch and
        missing it: without the term the clock stands at 746.55 us there,
        1.45 us short of the arrival, and with it at 749.18 us, 1.18 us past.
        """
        without = replace(Coefficients(), prefill_query_cached=0.0)
        assert _predicted_break(without) == 7
        assert _observed_break(_drive(without)) == 7

    def test_the_break_moves_when_each_step_does_twice_the_work(self):
        """Doubling the chunk halves the streak and raises the step by 1.88x
        -- the token term doubles, the quadratic quadruples and the intercept
        does not move -- so the same arrival lands two batches earlier."""
        coefficients = Coefficients()
        assert (
            _observed_break(_drive(coefficients, chunk=512))
            == _predicted_break(coefficients, chunk=512)
            == 4
        )

    def test_constant_pricing_is_blind_to_that(self):
        """Not a defect in constant mode -- the reason it is not the default.

        The same two runs under a constant price break at the same step,
        because the time a step takes is the constant and the work in it does
        not enter. The schedule is then a property of the number somebody
        typed, and a run cannot show that anything reacted to the model.
        """
        flat = Coefficients.constant(prefill_seconds=2.5e-4, decode_seconds=2.5e-4)
        narrow = _observed_break(_drive(flat, chunk=CHUNK))
        wide = _observed_break(_drive(flat, chunk=512))
        assert narrow == wide == _predicted_break(flat) == 4

    def test_a_longer_chunk_costs_longer_in_the_real_streak(self):
        trace = _drive(Coefficients())
        backend = StepStubBackend()
        first, joined = trace[0][1], trace[5][1]
        assert sum(r.query_tokens for r in joined.requests) == 2 * sum(
            r.query_tokens for r in first.requests
        )
        assert backend.price(joined).seconds > backend.price(first).seconds
