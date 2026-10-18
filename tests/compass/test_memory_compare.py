# SPDX-License-Identifier: MIT
"""Two instruments on one breakdown: the per-term gate, and the sum that hid it.

The per-term rule comes from an incident. A summed non-KV memory check read
**+13.8%** and was three errors, two of which cancelled: weights over by
**+0.280 GB**, activations compared at the wrong shape (**-0.015 GB**), and
**-0.084 GB** of a resident term nobody had noticed existed. **The largest
single error was 25% of its own term.**

`HISTORICAL` below is that breakdown, and
`test_two_instruments_on_one_breakdown` runs the two instruments on it side by
side -- the per-term comparator naming three failures, and the summed check
reading +13.8% and passing.

**How the fixture's totals were reconstructed, since the incident was reported
as deltas and ratios rather than totals.** Three deltas are given (+0.280, -0.015,
-0.084 GB), and two ratios: the largest error is 25% of its term, and the sum
is +13.8%. The largest error by bytes is the weights one, so the recorded
weights follow exactly: `0.280 / 0.25 = 1.120 GB`. The three deltas sum to
+0.181 GB, so the recorded terms sum to `0.181 / 0.138`, and the rounding of
13.8% pins that sum to `(1.30686, 1.31636]` GB -- leaving the recorded
activations in `(0.10286, 0.11236]` GB. The fixture takes **0.110 GB**, the
only figure the incident's numbers do not determine, and the resulting sum is
+13.77%, which is the +13.8% the incident reported. Nothing else in the fixture is chosen.

Every test in this file runs with both device readings patched to raise, for the
same reason the readings tests do: a comparator that needed a card would be
useless for sizing a card nobody has. The import-graph check that settles it
for every branch at once lives in `test_memory_readings.py` and globs the
package, so `compare.py` is inside it already.
"""

import copy
import hashlib
import json

import pytest
import torch
from transformers import PretrainedConfig

from atom.compass.backends.geometry import dtype_bytes as element_bytes
from atom.compass.backends.provenance import Method
from atom.compass.memory import (
    DISCHARGES,
    NON_KV_TERM_GATE,
    Comparison,
    MemoryRefusal,
    ModelTerms,
    PiecewiseCapture,
    Predicted,
    Recorded,
    Shape,
    SummedCheck,
    Term,
    TermComparison,
    Verdict,
    capture_token_shapes,
    check,
    compare,
    compare_graph_pool,
    device_readings,
    footprint_terms,
    piecewise_per_token_bytes,
    predicts,
    reserves,
)
from atom.compass.spec import MachineSpec
from tests.compass.test_memory_readings import (
    CAPTURE_SIZES,
    CONFIG_JSON,
    DOCUMENT,
    GPU_MEMORY_UTILIZATION,
    MAX_NUM_BATCHED_TOKENS,
    PARAMETERS,
    WARMUP_TOKENS,
    measured,
    meta_tree,
    spec_of,
)

MIB = 1 << 20
CAPACITY = "device.memory.capacity_bytes"


@pytest.fixture(autouse=True)
def no_device_readings(monkeypatch):
    """Both device readings raise, for every test in this file."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "a device reading was taken; a comparator that needs a card cannot "
            "be run against the card it is sizing for"
        )

    monkeypatch.setattr(torch.cuda, "mem_get_info", refuse)
    monkeypatch.setattr(torch.cuda, "memory_stats", refuse)


@pytest.fixture(scope="module")
def spec():
    # Probed, so the constants it supplies are measured terms.
    return spec_of("probed")


@pytest.fixture(scope="module")
def qwen():
    raw = json.loads(CONFIG_JSON.read_text())
    return PretrainedConfig.from_dict(raw["text_config"])


# --- the historical breakdown, as data ---------------------------------------

#: The shape the incident's activations belonged to. Both sides of the fixture
#: state it, which is the whole of what the -0.015 GB needed and did not have.
HISTORICAL_SHAPE = Shape(tokens=4096, phase="prefill")

#: The terms taken at that shape. Stated, never inferred from a name.
AT_SHAPE = frozenset({"activations"})

#: The incident does not name the third term. It is named here for the two
#: things known about it: it is resident, and the model that missed it
#: attributed nothing to it.
UNATTRIBUTED = "resident (unattributed)"

HISTORICAL_RECORDED = Recorded(
    run="the summed non-KV check that read +13.8%",
    shape=HISTORICAL_SHAPE,
    high_water_reset=True,
    terms=footprint_terms(
        {
            "weights": 1_120_000_000,
            "activations": 110_000_000,
            UNATTRIBUTED: 84_000_000,
        },
        source="the breakdown the run printed",
    ),
    at_shape=AT_SHAPE,
)

HISTORICAL_PREDICTED = Predicted(
    label="the prediction the incident's summed check passed",
    shape=HISTORICAL_SHAPE,
    terms=(
        Term(
            "weights",
            1_400_000_000,
            Method.ANALYTICAL,
            "a meta build, deduped by storage, that had not been through the loader",
        ),
        Term(
            "activations",
            95_000_000,
            Method.MEASURED,
            "a liveness walk of the traced graph",
        ),
    ),
    at_shape=AT_SHAPE,
)


def test_two_instruments_on_one_breakdown(capsys):
    """Three named term errors, beside a sum that reads +13.8% and passes.

    The per-term table and the summed check are printed here rather than only
    asserted, because the argument this file makes is that the two disagree on
    one breakdown and the disagreement is the finding.
    """
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    # No band for a sum exists. Both bands in use -- the 10% a non-KV term
    # carries, and the 25% allowed one on a device nobody has measured -- are
    # per term, and taking one of them for a sum is the substitution
    # the incident is made of. 25% is the loosest, so it flatters the sum most.
    summed = comparison.summed(band=0.25)
    with capsys.disabled():
        print()
        print(summed.table())

    errors = {t.name: t for t in comparison.compared}
    assert set(errors) == {"weights", "activations", UNATTRIBUTED}
    assert errors["weights"].delta_bytes == 280_000_000
    assert errors["activations"].delta_bytes == -15_000_000
    assert errors[UNATTRIBUTED].delta_bytes == -84_000_000
    assert all(t.verdict is Verdict.FAIL for t in comparison.compared)
    assert len(comparison.failures()) == 3

    # The two orderings disagree on this fixture, which is why there are two
    # accessors and both name their unit. The gate is a fraction of the
    # recorded term, so `worst` is the unattributed one at -100%; the
    # incident's own "25% of its term" is a statement about bytes.
    assert comparison.worst().name == UNATTRIBUTED
    assert comparison.worst().relative == -1.0
    largest = comparison.largest_by_bytes()
    assert largest.name == "weights"
    assert largest.relative == pytest.approx(0.25, abs=5e-5)
    # The one the sum cannot see at all: nothing predicted it, so it is absent
    # from the prediction and from any sum over the prediction.
    assert errors[UNATTRIBUTED].predicted is None

    assert summed.delta_bytes == 181_000_000
    assert summed.relative == pytest.approx(0.13775, abs=5e-5)
    assert f"{summed.relative:+.1%}" == "+13.8%"
    assert summed.passed


def test_at_the_per_term_band_the_sum_is_red_and_the_table_names_all_three():
    """The contrast, asserted as a contrast rather than as an absence.

    The objection to a summed check is not that its band was wrong. Given the
    per-term band, the sum is red and still says only that something is wrong;
    the per-term instrument on the same breakdown at the same band names all
    three terms and their verdicts. That pair is the evidence. An assertion
    that the aggregate line contains no term name would be pinned by the
    format string rather than measured, so it is a regression pin below and
    not the argument.
    """
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    summed = comparison.summed(band=NON_KV_TERM_GATE)
    assert not summed.passed
    assert summed.table().splitlines()[0].endswith("10% band -- fails")
    assert {t.name: t.verdict for t in comparison.compared} == {
        "weights": Verdict.FAIL,
        "activations": Verdict.FAIL,
        UNATTRIBUTED: Verdict.FAIL,
    }
    # The pin, stated as a pin: nothing folds a term name into the headline.
    aggregate = summed.table().splitlines()[0]
    for name in ("weights", "activations", UNATTRIBUTED):
        assert name not in aggregate


def test_the_two_compensating_errors_are_what_the_sum_folds():
    """+0.280 and -0.099 GB of error make a sum of +0.181 GB, which is the point."""
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    over = sum(t.delta_bytes for t in comparison.compared if t.delta_bytes > 0)
    under = sum(t.delta_bytes for t in comparison.compared if t.delta_bytes < 0)
    assert over == 280_000_000
    assert under == -99_000_000
    assert over + under == comparison.summed(band=0.25).delta_bytes


# --- an aggregate never without its decomposition ---------------------------


def test_a_comparison_has_no_total_and_a_sum_cannot_be_built_without_one():
    assert not hasattr(Comparison, "total")
    assert not hasattr(Comparison, "__int__")
    assert not hasattr(TermComparison, "__int__")
    # The only state a SummedCheck holds is the comparison and the band: there
    # is no constructor that takes two totals, so the figure cannot exist
    # without the table that produced it.
    assert set(SummedCheck.__dataclass_fields__) == {"comparison", "band"}


@pytest.mark.parametrize(
    "omit, names",
    [
        (lambda c, r: SummedCheck(comparison=c), ("SummedCheck.__init__", "'band'")),
        (lambda c, r: c.summed(), ("Comparison.summed", "'band'")),
        (
            lambda c, r: Predicted.from_readings(r, shape=HISTORICAL_SHAPE),
            ("Predicted.from_readings", "'at_shape'"),
        ),
    ],
    ids=["SummedCheck.band", "Comparison.summed-band", "from_readings-at_shape"],
)
def test_a_value_the_caller_must_state_is_refused_when_omitted(omit, names, spec, qwen):
    # A band for a sum, and which terms move with the shape, are the caller's
    # to state. Omitting one is a TypeError naming the signature and the field.
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    with pytest.raises(TypeError) as omitted:
        omit(comparison, live_readings(spec, qwen))
    for name in names:
        assert name in str(omitted.value)


def test_printing_a_sum_prints_the_terms_it_folded():
    summed = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).summed(band=0.25)
    rendered = str(summed)
    for name in ("weights", "activations", UNATTRIBUTED):
        assert name in rendered
    assert "+13.8%" in rendered


def test_both_orderings_come_back_as_their_term_not_as_a_number():
    comparison = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED)
    for largest in (comparison.worst(), comparison.largest_by_bytes()):
        assert isinstance(largest, TermComparison)
    by_bytes = comparison.largest_by_bytes()
    assert by_bytes.recorded.nbytes == 1_120_000_000
    assert by_bytes.predicted is not None
    # Ranked in the gate's unit it is the other term, and that is the point of
    # having two accessors rather than one word covering both.
    assert comparison.worst().name != by_bytes.name


def test_a_comparison_with_no_comparable_term_refuses_a_largest_error():
    # Every recorded term was taken at a shape and the run did not reset the
    # peak, so every one of them refuses and nothing is left to be largest.
    recorded = Recorded(
        run="a tracing run that did not reset the peak",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=footprint_terms({"activations": 110_000_000}, source="the printout"),
        at_shape=AT_SHAPE,
    )
    predicted = Predicted(
        label="a prediction of that one term",
        shape=HISTORICAL_SHAPE,
        terms=(Term("activations", 95_000_000, Method.MEASURED, "a liveness walk"),),
        at_shape=AT_SHAPE,
    )
    comparison = compare(predicted, recorded)
    assert not comparison.compared
    with pytest.raises(MemoryRefusal) as refusal:
        comparison.worst()
    assert "no largest error" in str(refusal.value)


# --- trap 1: a term the run does not record ----------------------------------


def ladder():
    return capture_token_shapes(
        CAPTURE_SIZES, max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS
    )


def reserved(qwen, total_bytes):
    return reserves(
        piecewise=PiecewiseCapture(
            per_token_bytes=piecewise_per_token_bytes(
                hidden_size=qwen.hidden_size,
                layers=qwen.num_hidden_layers,
                dtype_bytes=element_bytes(qwen.dtype),
            ),
            token_shapes=ladder(),
            budget_bytes=int(total_bytes * GPU_MEMORY_UTILIZATION),
        )
    )


def live_readings(spec, qwen, tp_width=1):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    return device_readings(
        spec,
        memory=measured(tp_width),
        tp_width=tp_width,
        model=ModelTerms.from_tree(
            meta_tree(PARAMETERS * element_bytes(qwen.dtype) // tp_width),
            activations=measured(tp_width).activations(WARMUP_TOKENS),
        ),
        cudagraph_overhead=reserved(qwen, total_bytes),
    )


def live_prediction(spec, qwen, tp_width=1):
    return Predicted.from_readings(
        live_readings(spec, qwen, tp_width),
        shape=Shape(tokens=WARMUP_TOKENS, phase="prefill"),
        at_shape=AT_SHAPE,
    )


def agreeing_recording(prediction, *, drop=("buffers",)):
    """A run whose every recorded term is the predicted byte count exactly.

    The strongest form of the discharge question: with the two sides equal to
    the byte, whether a term's gate is discharged turns on nothing but where
    the predicted number came from.

    `buffers` is dropped because the split at source is three
    non-subtractive readings -- `parameter_bytes`, `weights_torch`,
    `current_torch` -- and none of them is buffers. Buffers are not parameters,
    so they fall inside `weights_torch - parameter_bytes` together with every
    other resident non-parameter allocation; isolating them needs a recording
    of buffers on their own, which is a separate instrument.
    """
    return Recorded(
        run="fixture: every recorded term set to the predicted byte count",
        shape=prediction.shape,
        high_water_reset=True,
        terms=footprint_terms(
            {t.name: t.nbytes for t in prediction.terms if t.name not in drop},
            source="the breakdown the run printed",
        ),
        at_shape=prediction.at_shape - set(drop),
    )


def test_a_term_the_run_does_not_record_refuses_and_carries_its_own_reason(
    spec, qwen, capsys
):
    """`buffers` refuses by name, and the refusal quotes the reason.

    A real run's split does not isolate buffers, so a comparator cannot hold
    them to a 10% gate, and the refusal is the result.
    """
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))
    with capsys.disabled():
        print()
        print(comparison.table())

    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"buffers"}
    message = str(refused["buffers"])
    assert "records no 'buffers'" in message
    assert "not a term of zero bytes" in message
    assert "buffers" not in {t.name for t in comparison.compared}


def test_an_absent_term_is_not_a_term_of_zero_bytes(spec, qwen):
    # The failure mode this refusal exists to stop: a comparator that filled in
    # a zero would report buffers as -100% and pass a reader a number.
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))
    assert all(t.name != "buffers" for t in comparison.compared)
    assert comparison.summed(band=0.25).predicted_total == sum(
        t.predicted_bytes for t in comparison.compared
    )


def test_every_term_agreeing_exactly_discharges_its_gate(spec, qwen):
    """Every recorded term equals its prediction, and every gate closes.

    `weights` is read off a meta build and the rest off the entry, so no term
    is declared; the entry's record is not the recording's.
    """
    prediction = live_prediction(spec, qwen)
    comparison = compare(prediction, agreeing_recording(prediction))

    assert all(t.delta_bytes == 0 for t in comparison.compared)
    assert not comparison.not_discharged()
    assert not comparison.failures()
    assert {t.name for t in comparison.compared if t.verdict is Verdict.PASS} == {
        "weights",
        "load residue",
        "persistent",
        "activations",
        "driver and collective reserve",
    }


def test_a_declared_term_can_fail_its_gate_even_though_it_cannot_pass_one():
    # The asymmetry is deliberate: a recording that contradicts a coefficient
    # is evidence, and agreement with one run is not a measurement.
    declared = Term(
        "activations",
        200_000_000,
        Method.DECLARED,
        "a coefficient over one live layer",
        "a liveness walk over a traced op graph",
    )
    predicted = Predicted(
        label="a declared activation term",
        shape=HISTORICAL_SHAPE,
        terms=(declared,),
        at_shape=AT_SHAPE,
    )
    recorded = Recorded(
        run="a run that contradicts it",
        shape=HISTORICAL_SHAPE,
        high_water_reset=True,
        terms=footprint_terms({"activations": 100_000_000}, source="the printout"),
        at_shape=AT_SHAPE,
    )
    assert compare(predicted, recorded).compared[0].verdict is Verdict.FAIL


# --- trap 2: compared at the wrong shape -------------------------------------


def test_a_pair_taken_at_two_shapes_refuses_and_names_both():
    # 3,494 tokens is the trace the analytical memory law scales to an
    # independently measured 4,096-token peak. Comparing the one against the other without saying so
    # is the -0.015 GB of the incident.
    at_3494 = Recorded(
        run="the 3,494-token trace",
        shape=Shape(tokens=3494, phase="prefill"),
        high_water_reset=True,
        terms=HISTORICAL_RECORDED.terms,
        at_shape=AT_SHAPE,
    )
    comparison = compare(HISTORICAL_PREDICTED, at_3494)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"activations"}
    message = str(refused["activations"])
    assert "3494 tokens, prefill" in message
    assert "4096 tokens, prefill" in message
    # The terms that do not move with the shape are still compared, so a shape
    # disagreement costs the activation row and nothing else.
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_the_table_states_the_shape_of_each_side():
    rendered = compare(HISTORICAL_PREDICTED, HISTORICAL_RECORDED).table()
    assert "predicted at 4096 tokens, prefill" in rendered
    assert "recorded at 4096 tokens, prefill" in rendered


def test_two_shapes_and_no_side_naming_a_shaped_term_refuses_outright():
    # A comparison that cannot say which of its terms move with the shape
    # cannot say whether the two sides measured the same thing at all.
    predicted = Predicted(
        label="a prediction that names no shaped term",
        shape=Shape(tokens=8192, phase="prefill"),
        terms=HISTORICAL_PREDICTED.terms,
    )
    recorded = Recorded(
        run="a run at another shape",
        shape=Shape(tokens=4096, phase="prefill"),
        high_water_reset=True,
        terms=HISTORICAL_RECORDED.terms,
    )
    with pytest.raises(MemoryRefusal) as refusal:
        compare(predicted, recorded)
    assert "neither names a term it took at one" in str(refusal.value)


@pytest.mark.parametrize(
    "at_shape", [AT_SHAPE, frozenset()], ids=["per-term", "outright"]
)
@pytest.mark.parametrize(
    "mine, theirs",
    [("prefill", "decode"), ("decode", "Decode")],
    ids=["prefill-decode", "two-spellings"],
)
def test_a_decode_side_refuses_a_prefill_side_at_the_same_token_count(
    at_shape, mine, theirs
):
    # The token counts agree, so only the phase separates the two sides. Both
    # places that read shape agreement are driven: the per-term refusal when a
    # shaped term is named, and the outright one when neither side names one.
    # Phases compare exactly, so two spellings of one phase refuse as well.
    predicted = Predicted(
        label="a prediction",
        shape=Shape(tokens=HISTORICAL_SHAPE.tokens, phase=mine),
        terms=HISTORICAL_PREDICTED.terms,
        at_shape=at_shape,
    )
    decode = Recorded(
        run="a decode step at the prediction's token count",
        shape=Shape(tokens=HISTORICAL_SHAPE.tokens, phase=theirs),
        high_water_reset=True,
        terms=HISTORICAL_RECORDED.terms,
        at_shape=at_shape,
    )
    both = f"predicted at 4096 tokens, {mine}, recorded at 4096 tokens, {theirs}"
    if not at_shape:
        with pytest.raises(MemoryRefusal) as refusal:
            compare(predicted, decode)
        assert f"taken at different shapes -- {both} -- " in str(refusal.value)
        return
    comparison = compare(predicted, decode)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"activations"}
    assert f"the two disagree -- {both}" in refused["activations"].what
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_a_side_that_claims_a_shaped_term_it_does_not_carry_is_rejected():
    with pytest.raises(ValueError) as bad:
        Recorded(
            run="a run",
            shape=HISTORICAL_SHAPE,
            high_water_reset=True,
            terms=footprint_terms({"weights": 1}, source="the printout"),
            at_shape=frozenset({"activations"}),
        )
    assert "records no such term" in str(bad.value)


# --- trap 3: the allocator's high-water mark ---------------------------------


def test_a_run_that_did_not_reset_the_peak_refuses_every_shaped_term():
    not_reset = Recorded(
        run="a tracing run that did not reset the peak",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=HISTORICAL_RECORDED.terms,
        at_shape=AT_SHAPE,
    )
    comparison = compare(HISTORICAL_PREDICTED, not_reset)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"activations"}
    message = str(refused["activations"])
    assert "warmup prefill" in message
    assert "-14.7%" in message
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_whether_the_peak_was_reset_has_no_default():
    with pytest.raises(TypeError):
        Recorded(
            run="a run that does not say",
            shape=HISTORICAL_SHAPE,
            terms=HISTORICAL_RECORDED.terms,
        )


def test_a_recorded_term_that_was_not_read_off_a_card_is_rejected():
    with pytest.raises(ValueError) as bad:
        Recorded(
            run="a run whose terms are formulas",
            shape=HISTORICAL_SHAPE,
            high_water_reset=True,
            terms=(Term("weights", 1, Method.ANALYTICAL, "a spec field"),),
        )
    assert "which is measured" in str(bad.value)


def test_the_peak_guard_holds_when_only_the_prediction_names_a_shaped_term():
    """The guard reads the union of the two sides, not the recording's alone.

    `at_shape` is the one field on a recording that defaults, so a caller who
    answers `high_water_reset=False` and leaves it empty was buying nothing
    from a guard that read only their side -- the activation term would have
    been compared against the warmup prefill's peak with no refusal at all.
    The two shape guards now read one definition.
    """
    recorded = Recorded(
        run="a tracing run that did not reset the peak and names no shaped term",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=HISTORICAL_RECORDED.terms,
    )
    assert recorded.at_shape == frozenset()
    comparison = compare(HISTORICAL_PREDICTED, recorded)
    assert {r.name for r in comparison.refused} == {"activations"}
    assert "warmup prefill" in str(comparison.refused[0])
    assert {t.name for t in comparison.compared} == {"weights", UNATTRIBUTED}


def test_a_peak_that_was_not_reset_with_no_shaped_term_anywhere_refuses_outright():
    # A recording that says its peak is wrong and then declines to say which
    # terms it is wrong for leaves the guard nothing to refuse.
    predicted = Predicted(
        label="a prediction that names no shaped term",
        shape=HISTORICAL_SHAPE,
        terms=HISTORICAL_PREDICTED.terms,
    )
    recorded = Recorded(
        run="a run that reset nothing and names nothing",
        shape=HISTORICAL_SHAPE,
        high_water_reset=False,
        terms=HISTORICAL_RECORDED.terms,
    )
    with pytest.raises(MemoryRefusal) as refusal:
        compare(predicted, recorded)
    assert "neither side names a term it took at a shape" in str(refusal.value)


# --- a term the run records as zero bytes ------------------------------------


def zero_byte_recording(*, also_predicted):
    predicted_terms = [Term("weights", 100, Method.ANALYTICAL, "a meta build")]
    if also_predicted:
        predicted_terms.append(Term("ghost", 5, Method.ANALYTICAL, "a meta build"))
    return (
        Predicted(
            label="a prediction beside a term the run measured as empty",
            shape=HISTORICAL_SHAPE,
            terms=tuple(predicted_terms),
        ),
        Recorded(
            run="a run that found one term empty",
            shape=HISTORICAL_SHAPE,
            high_water_reset=True,
            terms=footprint_terms({"ghost": 0, "weights": 100}, source="the printout"),
        ),
    )


@pytest.mark.parametrize("also_predicted", [True, False])
def test_a_term_recorded_as_zero_bytes_refuses_by_name(also_predicted):
    """No bare `ZeroDivisionError`, on either path, and a named reason instead.

    Zero is not exotic: an eager-mode run holds no rotary cache and prints
    `buffers: 0`, and so does any term a run instruments and finds empty. The
    gate's unit is a fraction of the recorded term, so there is no error to
    state -- which is a refusal, not an exception.
    """
    predicted, recorded = zero_byte_recording(also_predicted=also_predicted)
    comparison = compare(predicted, recorded)
    refused = {r.name: r for r in comparison.refused}
    assert set(refused) == {"ghost"}
    assert "zero bytes" in str(refused["ghost"])
    assert {t.name for t in comparison.compared} == {"weights"}


@pytest.mark.parametrize("also_predicted", [True, False])
def test_one_zero_byte_term_does_not_take_the_table_down(also_predicted):
    # The decomposition is this module's product, so a term that cannot state
    # a relative error must not destroy the rows that can.
    predicted, recorded = zero_byte_recording(also_predicted=also_predicted)
    rendered = compare(predicted, recorded).table()
    assert "weights" in rendered
    assert "ghost" in rendered
    assert "+0.00%" in rendered


def test_the_named_refusal_for_a_zero_recorded_term_is_still_reachable():
    # Kept on `TermComparison` for a pair built by hand, and asserted so that
    # it is not a dead branch that a reader has to guess about.
    pair = TermComparison(
        name="ghost",
        predicted=None,
        recorded=Term("ghost", 0, Method.MEASURED, "the printout"),
        gate=NON_KV_TERM_GATE,
        verdict=Verdict.FAIL,
    )
    with pytest.raises(MemoryRefusal) as refusal:
        _ = pair.relative
    assert "zero bytes" in str(refusal.value)


# --- bytes are whole, and the rounding stays at the call site ----------------


def test_a_printout_carrying_a_float_byte_count_refuses_rather_than_truncating():
    # The machine specification this package reads writes its byte counts as
    # floats, so a breakdown arriving with 1.1e6 in it is the expected shape.
    with pytest.raises(TypeError) as bad:
        footprint_terms({"load residue": 1.1e6}, source="the printout")
    assert "bytes are whole" in str(bad.value)


# --- which methods discharge a gate -----------------------------------------


@pytest.mark.parametrize(
    "method, expected",
    [
        (Method.ANALYTICAL, Verdict.PASS),
        (Method.MEASURED, Verdict.PASS),
        (Method.ESTIMATED, Verdict.PASS),
        (Method.DECLARED, Verdict.NOT_DISCHARGED),
    ],
)
def test_every_method_but_declared_can_pass_a_gate(method, expected):
    note = "a successor" if method is Method.DECLARED else ""
    predicted = Predicted(
        label=f"a prediction whose term is {method}",
        shape=HISTORICAL_SHAPE,
        terms=(Term("persistent", 100, method, "wherever it came from", note),),
    )
    recorded = Recorded(
        run="a run that agrees to the byte",
        shape=HISTORICAL_SHAPE,
        high_water_reset=True,
        terms=footprint_terms({"persistent": 100}, source="the printout"),
    )
    assert compare(predicted, recorded).compared[0].verdict is expected


def test_only_declared_is_left_out_of_the_set_that_discharges():
    assert set(Method) - DISCHARGES == {Method.DECLARED}


# --- the graph pool: two numbers, both reported ------------------------------

#: The measured pool at this ladder: 91.1 MiB + 0.3033 MiB per
#: captured token at width 1, and flat 104 MiB above it. This is the recording
#: the two functions are compared against; `ENTRY_POOL` is the pool an entry's
#: start read at each width, the authored, rounded form of the same line, which
#: is why `predicts()` lands on top of it and `reserves()` does not.
RECORDED_POOL_W1 = int(91.1 * MIB + 0.3033 * MIB * 1023)
RECORDED_POOL_W_GT1 = 104 * MIB
ENTRY_POOL = {1: 420_814_000, 2: 109_000_000}


def recorded_pool(nbytes, where):
    return Term("recorded pool", int(nbytes), Method.MEASURED, where)


def entry_pool(tp_width, **member):
    """What `predicts()` reads off the entry member at `tp_width`."""
    pool = ENTRY_POOL[tp_width]
    return predicts(measured(tp_width, cudagraph_pool_bytes=pool, **member))


@pytest.mark.parametrize(
    "tp_width, pool, where",
    [
        (1, RECORDED_POOL_W1, "the measured line: 91.1 MiB + 0.3033 MiB x 1023 tokens"),
        (2, RECORDED_POOL_W_GT1, "the measured line: flat 104 MiB above width 1"),
    ],
)
def test_both_graph_pool_numbers_are_reported_against_the_recorded_pool(
    spec, qwen, tp_width, pool, where, capsys
):
    """Neither function is picked, and the one that reserves is labelled.

    The two stay apart because they disagree by 4-19x. A comparator that
    reported one of them would reconcile the two numbers `graph_pool` keeps
    apart, so this one has a row for each and no accessor for *the* error.
    """
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    comparison = compare_graph_pool(
        recorded_pool(pool, where),
        reserves=reserved(qwen, total_bytes),
        predicts=entry_pool(tp_width),
    )
    with capsys.disabled():
        print()
        print(comparison.table())

    assert comparison.reserves.total == 1_877_213_184
    # The measured predictor lands on the recording; the one that reserves does
    # not, and it is the one ATOM subtracts from the budget.
    assert abs(comparison.predicts_relative) < 0.001
    assert comparison.reserves_relative > 3.0
    rendered = comparison.table()
    assert "reserves the memory" in rendered
    assert "reserves nothing" in rendered
    assert f"{comparison.reserves_relative:+.2%}" in rendered
    assert f"{comparison.predicts_relative:+.2%}" in rendered


def test_each_graph_pool_row_prints_its_method(spec, qwen):
    # An assumed entry member makes predicts() declared, which must not read
    # like a measured row.
    comparison = compare_graph_pool(
        recorded_pool(RECORDED_POOL_W1, "a printout"),
        reserves=reserved(qwen, int(spec.value("device.memory.capacity_bytes"))),
        predicts=entry_pool(1, assumed="copied from another platform's entry"),
    )
    rows = {line.split()[0]: line for line in comparison.table().splitlines()[2:5]}
    assert "declared" in rows["predicts()"]
    assert "analytical" in rows["reserves()"]
    assert "measured" in rows["recorded"]


def test_the_disagreement_is_inside_the_recorded_four_to_nineteen_times(spec, qwen):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    ratios = {}
    for tp_width in (1, 2):
        comparison = compare_graph_pool(
            recorded_pool(
                RECORDED_POOL_W1 if tp_width == 1 else RECORDED_POOL_W_GT1,
                "the measured line",
            ),
            reserves=reserved(qwen, total_bytes),
            predicts=entry_pool(tp_width),
        )
        ratios[tp_width] = comparison.disagreement
    assert ratios[1] == pytest.approx(4.4609, abs=5e-5)
    assert ratios[2] == pytest.approx(17.2221, abs=5e-5)


def test_the_two_graph_pool_readings_cannot_be_handed_in_the_wrong_way_round(
    spec, qwen
):
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    reserving = reserved(qwen, total_bytes)
    predicting = entry_pool(1)
    with pytest.raises(MemoryRefusal) as refusal:
        compare_graph_pool(
            recorded_pool(RECORDED_POOL_W1, "the measured line"),
            reserves=predicting,
            predicts=reserving,
        )
    assert "cudagraph_pool" in str(refusal.value)
    assert "it is the one that reserves" in str(refusal.value)


def test_the_graph_pool_cannot_be_folded_into_the_per_term_table(spec, qwen):
    # One row for the pool would pick one of the two numbers, and picking one
    # is what throws the finding away.
    total_bytes = int(spec.value("device.memory.capacity_bytes"))
    prediction = live_prediction(spec, qwen)
    pool = reserved(qwen, total_bytes)
    folded = Predicted(
        label="a prediction that folded the pool in",
        shape=prediction.shape,
        terms=prediction.terms
        + (
            Term(
                pool.name,
                pool.total,
                Method.DECLARED,
                "ATOM's own estimator",
                "the measured pool, which disagrees by 4-19x",
            ),
        ),
        at_shape=prediction.at_shape,
    )
    recording = agreeing_recording(prediction)
    with pytest.raises(MemoryRefusal) as refusal:
        compare(folded, recording)
    assert "compare_graph_pool" in str(refusal.value)


# --- what `from_readings` takes, and what it deliberately leaves -------------


def test_the_footprint_is_peak_torch_and_non_torch_and_nothing_else(spec, qwen):
    readings = live_readings(spec, qwen)
    prediction = Predicted.from_readings(
        readings, shape=Shape(tokens=WARMUP_TOKENS, phase="prefill"), at_shape=AT_SHAPE
    )
    assert [t.name for t in prediction.terms] == [
        "weights",
        "buffers",
        "load residue",
        "persistent",
        "activations",
        "driver and collective reserve",
    ]
    # `total` is the card and `free` is what is left of it, so neither is a
    # footprint term; `cudagraph_overhead` is one of two numbers.
    assert "capacity" not in {t.name for t in prediction.terms}
    assert "per-token x captured tokens" not in {t.name for t in prediction.terms}
    assert readings.spec_digest in prediction.label


# --- a term against the run it was built from --------------------------------

GIB = 1 << 30


def _server_log(path, *, peak, non_torch, start="08:31:41"):
    """The two lines `python -m atom.compass.memory.check` reads off a server."""
    path.write_text(
        f"[atom {start}] Model Runner0/1: warmup_model 40.35 seconds with 1 reqs "
        f"{WARMUP_TOKENS} tokens\n"
        f"[atom {start}] Memory budget: total_gpu=191.98GB, free=139.49GB, "
        f"peak_torch={peak / GIB:.2f}GB, non_torch={non_torch / GIB:.2f}GB, "
        "cudagraph_est=0.55GB\n"
    )
    return path


def _hand_in(tmp_path, qwen, record):
    """A hand-in sized off the suite's document, probed on the run `record`
    names, with its entry's readings, the activations included, measured on
    that run."""
    document = copy.deepcopy(DOCUMENT)
    document["provenance"] |= {"origin": "probed", "subject": {"record": record}}
    spec = MachineSpec.from_mapping(document)
    entry = measured(1, record=record, activation_bytes=WARMUP_TOKENS * 180_480)
    model = ModelTerms.from_tree(
        meta_tree(PARAMETERS * 2), activations=entry.activations(WARMUP_TOKENS)
    )
    readings = device_readings(
        spec,
        memory=entry,
        tp_width=1,
        model=model,
        cudagraph_overhead=reserved(qwen, int(spec.value(CAPACITY))),
    )
    path = tmp_path / "lp-engine.json"
    hand_in = check.record(readings, tokens=WARMUP_TOKENS, memory=entry)
    path.write_text(json.dumps({"memory": hand_in}))
    return path, readings


def _digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("factor", [1.0, 2.0])
def test_a_term_compared_with_its_own_source_run_is_not_passed(tmp_path, qwen, factor):
    _, readings = _hand_in(tmp_path, qwen, "sha256:none")
    source = _server_log(
        tmp_path / "source.log",
        peak=readings.peak_torch.total,
        non_torch=readings.non_torch.total * factor,
    )
    record = _digest(source)
    hand_in, readings = _hand_in(tmp_path, qwen, record)
    # The activations come off the entry, and the entry names its run.
    by_name = {t.name: t for t in readings.peak_torch.terms}
    assert by_name["activations"].method is Method.MEASURED
    assert by_name["activations"].records == (record,)

    comparison = check.parts(hand_in, source)[0]
    verdicts = {t.name: t.verdict for t in comparison.compared}
    assert verdicts["peak_torch"] is Verdict.NOT_DISCHARGED
    # Built from the run or not, a term outside its gate fails.
    expected = Verdict.FAIL if factor > 1 else Verdict.NOT_DISCHARGED
    assert verdicts["non_torch"] is expected
    named = f"this term was built from record {record}, the run it is compared with"
    why = {t.name: t.why for t in comparison.compared}
    assert named in why["peak_torch"]
    assert (named in why["non_torch"]) is (factor == 1.0)
    assert (
        f"* peak_torch: within the gate, and the gate is not discharged: {named}"
        in comparison.table()
    )


@pytest.mark.parametrize("factor, verdict", [(1.0, Verdict.PASS), (2.0, Verdict.FAIL)])
def test_another_run_of_the_same_configuration_is_judged_on_its_numbers(
    tmp_path, qwen, factor, verdict
):
    _, readings = _hand_in(tmp_path, qwen, "sha256:none")
    peak, non_torch = readings.peak_torch.total, readings.non_torch.total
    source = _server_log(tmp_path / "source.log", peak=peak, non_torch=non_torch)
    other = _server_log(
        tmp_path / "other.log",
        peak=peak,
        non_torch=non_torch * factor,
        start="16:32:07",
    )
    hand_in, _ = _hand_in(tmp_path, qwen, _digest(source))

    comparison = check.parts(hand_in, other)[0]
    verdicts = {t.name: t.verdict for t in comparison.compared}
    assert verdicts == {"peak_torch": Verdict.PASS, "non_torch": verdict}
    assert "built from record" not in comparison.table()
