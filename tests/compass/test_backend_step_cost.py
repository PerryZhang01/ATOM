# SPDX-License-Identifier: MIT
"""A step's total, and the parts it is folded from.

Three properties are asserted structurally rather than by inspection. The
total is not stored: it is folded from the terms on every read, and the terms
are re-checked on every read, so an object edited past its constructor fails
when it is used rather than reporting a tidy number. A subclass cannot shadow
the total, because overriding a property needs no bypass at all and is the one
route to a plausible number that has nothing to do with its breakdown. And the
fold order is fixed and observable -- the same terms in a different order are
a different number, bit for bit.

The empty-sample refusal is asserted at both layers it can appear. Inside a
step it is an empty breakdown; over a run it is a fraction with no data behind
it, which matters more, because "refused nothing" and "recorded nothing" would
otherwise be the same number at the point where that number decides whether a
run is worth anything.
"""

import math

import pytest

from atom.compass.backends import (
    CostTerm,
    Method,
    Provenance,
    ProvenanceMix,
    Refusal,
    StepCost,
    fold_seconds,
    fold_step,
)

MEASURED = Provenance(Method.MEASURED, "step-level")
ANALYTICAL = Provenance(Method.ANALYTICAL, "roofline")
UNPRICED = Refusal("price_list", "unpriced leaf")
STOOD_IN = ANALYTICAL.resolved("analytical_law", [UNPRICED])


def term(name, seconds, provenance=MEASURED):
    return CostTerm(name, seconds, provenance)


def test_the_total_is_the_fold_of_the_parts():
    step = StepCost(
        [term("attention", 0.004), term("mlp", 0.006), term("collective", 0.001)]
    )
    assert step.seconds == fold_seconds([0.004, 0.006, 0.001])
    assert step.seconds == pytest.approx(0.011)


def test_grouping_re_associates_so_the_mixture_is_not_a_second_total():
    """The rows re-fold to the total; the method totals are a different sum."""
    step = StepCost(
        [
            term("a", 0.1),
            CostTerm("b", 0.2, ANALYTICAL),
            term("c", 0.15),
        ]
    )
    assert step.seconds == 0.45000000000000007
    assert fold_seconds(step.seconds_by_method().values()) == 0.45
    assert fold_seconds(step.seconds_by_method().values()) != step.seconds
    assert fold_seconds(seconds for _, seconds, _ in step.rows()) == step.seconds


def test_a_reader_can_check_the_total_against_the_rows():
    """The artifact's rows re-fold to the artifact's total, bit for bit."""
    step = StepCost([term("a", 0.1), term("b", 0.2), term("c", 0.3)])
    rows = step.rows()
    assert fold_seconds(seconds for _, seconds, _ in rows) == step.seconds


def test_the_fold_order_is_part_of_the_answer():
    """0.1 + 0.2 + 0.3 is not 0.3 + 0.2 + 0.1 in binary floating point."""
    forward = StepCost([term("a", 0.1), term("b", 0.2), term("c", 0.3)])
    reverse = StepCost([term("c", 0.3), term("b", 0.2), term("a", 0.1)])
    assert forward.seconds == 0.6000000000000001
    assert reverse.seconds == 0.6
    assert forward.seconds != reverse.seconds


def test_a_running_total_is_the_same_fold_as_a_batched_one():
    """So an accumulator and a re-check of it cannot disagree in the last bit."""
    values = [0.1, 0.2, 0.3, 1e-17]
    running = 0.0
    for value in values:
        running = fold_step(running, value)
    assert running == fold_seconds(values)


def test_an_aggregate_with_no_decomposition_cannot_be_built():
    with pytest.raises(ValueError, match="no decomposition"):
        StepCost([])


def test_a_total_read_from_an_edited_step_refuses_rather_than_answering_zero():
    """The check is on every read, not only at construction."""
    step = StepCost([term("attention", 0.004)])
    object.__setattr__(step, "terms", ())
    with pytest.raises(ValueError, match="no decomposition"):
        _ = step.seconds


def test_a_subclass_cannot_shadow_the_total_or_its_parts():
    """Overriding `seconds` needs no bypass and defeats every other check."""
    with pytest.raises(TypeError, match="redefines seconds"):

        class Drifting(StepCost):
            @property
            def seconds(self):
                return 99.0


PUBLIC_READERS = sorted(
    name
    for name in (*vars(StepCost), *StepCost.__annotations__)
    if not name.startswith("_")
)


def test_every_reader_is_covered_not_just_the_obvious_one():
    """`fell_back` is the dangerous one: hiding it empties the fallback count."""
    assert set(PUBLIC_READERS) >= {
        "fallback_seconds",
        "fell_back",
        "refusals",
        "rows",
        "seconds",
        "seconds_by_method",
        "terms",
    }


@pytest.mark.parametrize("reader", PUBLIC_READERS)
def test_no_reader_can_be_shadowed_by_a_subclass(reader):
    """Derived from the class, so a reader added later is covered on arrival."""
    with pytest.raises(TypeError, match=f"redefines {reader}"):
        type("Drifting", (StepCost,), {reader: property(lambda self: None)})


def test_hiding_the_fallback_flag_would_empty_the_fallback_count():
    """The defect the derived set exists to stop, stated as the thing it breaks."""
    mix = ProvenanceMix()
    mix.record(StepCost([CostTerm("step", 0.096, STOOD_IN)]))
    assert mix.fallback_step_fraction == 1.0
    with pytest.raises(TypeError, match="redefines fell_back"):

        class Quiet(StepCost):
            @property
            def fell_back(self):
                return False


def test_terms_are_read_by_name_so_names_are_unique():
    with pytest.raises(ValueError, match="duplicate term"):
        StepCost([term("attention", 0.001), term("attention", 0.002)])


def test_a_term_must_carry_a_provenance():
    with pytest.raises(TypeError):
        CostTerm("attention", 0.001)


@pytest.mark.parametrize("seconds", [-1e-9, float("nan"), math.inf])
def test_a_term_refuses_a_duration_that_is_not_one(seconds):
    with pytest.raises(ValueError):
        term("attention", seconds)


def test_a_refused_term_cannot_be_priced_at_zero():
    """A free stand-in drops the step out of the schedule while looking priced."""
    with pytest.raises(ValueError, match="priced at"):
        CostTerm("attention", 0.0, STOOD_IN)


def test_a_zero_term_is_fine_when_nothing_refused():
    """No collective on this step is a fact about the step, not a missing model."""
    assert StepCost([term("mlp", 0.006), term("collective", 0.0)]).seconds == 0.006


def test_a_step_reports_its_mixture_and_its_fallback_share():
    step = StepCost([term("mlp", 0.006), CostTerm("attention", 0.004, STOOD_IN)])
    assert step.fell_back
    assert step.fallback_seconds == pytest.approx(0.004)
    assert step.seconds_by_method() == {
        (Method.MEASURED, False): pytest.approx(0.006),
        (Method.ANALYTICAL, False): pytest.approx(0.004),
    }
    assert step.refusals == (UNPRICED,)


def test_a_step_reports_every_refusal_behind_it():
    """Two sources declined on one term; both are the record, and both count."""
    chained = ANALYTICAL.resolved(
        "analytical_law", [UNPRICED, Refusal("nearest_key", "outside the range")]
    )
    step = StepCost([CostTerm("attention", 0.004, chained)])
    assert [r.source for r in step.refusals] == ["price_list", "nearest_key"]


def test_fallback_steps_are_counted_three_ways_over_a_run():
    """Few steps can be most of the time, so both fractions are reported."""
    clean = StepCost([term("step", 0.001)])
    refused = StepCost([CostTerm("step", 0.096, STOOD_IN)])
    mix = ProvenanceMix()
    for step in (clean, clean, clean, refused):
        mix.record(step)

    assert not mix.is_empty
    assert mix.steps == 4
    assert mix.fallback_steps == 1
    assert mix.fallback_step_fraction == pytest.approx(0.25)
    assert mix.seconds == pytest.approx(0.099)
    assert mix.fallback_seconds == pytest.approx(0.096)
    assert mix.fallback_second_fraction == pytest.approx(0.096 / 0.099)
    assert mix.reasons() == {("price_list", "unpriced leaf"): 1}


def test_a_run_with_no_steps_refuses_its_fractions():
    """0.0 would make "recorded nothing" read as "nothing fell back"."""
    mix = ProvenanceMix()
    assert mix.is_empty
    with pytest.raises(ValueError, match="no steps were recorded"):
        _ = mix.fallback_step_fraction
    with pytest.raises(ValueError, match="no predicted seconds"):
        _ = mix.fallback_second_fraction


def test_a_run_of_free_steps_refuses_the_seconds_fraction_only():
    """Steps exist, so their fraction is answerable; the seconds are not."""
    mix = ProvenanceMix()
    mix.record(StepCost([term("step", 0.0)]))
    assert mix.fallback_step_fraction == 0.0
    with pytest.raises(ValueError, match="over 1 step"):
        _ = mix.fallback_second_fraction
