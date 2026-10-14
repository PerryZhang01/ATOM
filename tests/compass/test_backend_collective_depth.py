# SPDX-License-Identifier: MIT
"""Which layer count a collective is charged on, shown on two configs at once.

A worker is described by two layer counts. `geometry.layers` counts the layers
that hold a cache of every past token, which is what a KV block is sized from;
`stage_layers` counts the layers the worker runs, which is what an all-reduce
runs on -- a layer keeping a bounded recurrent state takes part in the
collective and costs a block nothing. The published hybrid here is one paged
layer in four, so its two counts are 16 and 64.

One config cannot show this. On a dense stack every layer is paged, the two
counts are the same number, and a charge on either is the same seconds: a price
computed from the wrong count is right by coincidence and nothing distinguishes
it. So both configs are priced here side by side, and the pair is the assertion
-- equal on the dense one, four times apart on the hybrid.

The counts are read back out of the priced step rather than restated: the
collective term is `tokens x layers x a declared coefficient`, so dividing the
charge by the tokens and the coefficient recovers the count the pricing used.
A test that asserted the seconds instead would pass against either count with
the arithmetic adjusted, which is the thing being guarded against.

Nothing here is an accuracy check. The coefficient behind every duration below
was declared, not measured, and a ratio between two of them is a statement
about which count was multiplied and not about any hardware.
"""

import pathlib

import pytest

from atom.compass.backends import (
    BatchView,
    Coefficients,
    FakeModel,
    Parallelism,
    RequestShape,
    StepStubBackend,
    SyntheticStack,
    hf_config,
)
from atom.models.utils import get_pp_indices

CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
BLOCK_SIZE = 64
# The batch every price below is read at: four requests of one chunk each. Only
# the token total reaches the collective term, so this is the same 1024 tokens
# on both configs and the prices differ by the layer count alone.
CHUNK = 256
REQUESTS = 4
# The published stack, and the dial that makes its dense counterpart: the same
# depth with every layer paged, which is the config the two counts coincide on.
STACK_LAYERS = 64
PAGED_LAYERS = 16
# A width with a collective to charge. Which collective, and that the widths
# alone decide it, is the neighbouring file's; here it is the vehicle for the
# count.
SHARDED = Parallelism(tp_size=4)


@pytest.fixture(scope="module")
def published():
    return hf_config(CONFIG_JSON)


def batch():
    return BatchView(tuple(RequestShape(CHUNK, CHUNK, False) for _ in range(REQUESTS)))


def collective_seconds(backend):
    """What this backend charges for collectives on that batch, in total."""
    step = backend.price(batch())
    return sum(
        value for name, value, _ in step.rows() if name.startswith("collective.")
    )


def charged_layers(backend, coefficients=None):
    """The layer count the charge was made on, recovered from the charge.

    The count is not read off the backend, because the question is what the
    pricing multiplied and not what it was handed. A charge that is not a whole
    number of layers is not the term this reads it as, and says so rather than
    rounding into a plausible count.
    """
    c = Coefficients() if coefficients is None else coefficients
    per_layer = REQUESTS * CHUNK * c.collective_token_layer
    seconds = collective_seconds(backend)
    count = round(seconds / per_layer)
    assert seconds == pytest.approx(count * per_layer), (
        f"a collective of {seconds} s is not a whole number of layers at "
        f"{per_layer} s each"
    )
    return count


def substituted(model):
    """The same deployment priced on the paged count instead of the stack.

    Built through the backend's own constructor, so the price is the one the
    stub would produce if the charge were moved back to the pool's count --
    read out of the code rather than worked out in a comment.
    """
    return StepStubBackend(
        coefficients=model.coefficients,
        parallelism=model.parallelism,
        geometry=model.geometry,
        stack_layers=model.geometry.layers,
    )


# ── the named result ────────────────────────────────────────────────────────


class TestTheTwoCountsOnTwoConfigs:
    """The published hybrid and a dense stack, priced side by side."""

    def test_the_hybrid_declares_two_counts_and_charges_the_deeper_one(self, published):
        model = FakeModel(published, block_size=BLOCK_SIZE, parallelism=SHARDED)
        assert (model.geometry.layers, model.stage_layers) == (
            PAGED_LAYERS,
            STACK_LAYERS,
        )
        assert charged_layers(model.backend) == STACK_LAYERS, (
            "the collective was charged on this config's paged layers; an "
            "all-reduce runs on every layer of the stack, and on this stack "
            "that is four times as many"
        )

    def test_the_dense_stack_declares_one_count_and_charges_that(self):
        dense = FakeModel(
            SyntheticStack(num_hidden_layers=STACK_LAYERS),
            block_size=BLOCK_SIZE,
            parallelism=SHARDED,
        )
        assert dense.geometry.layers == dense.stage_layers == STACK_LAYERS
        assert charged_layers(dense.backend) == STACK_LAYERS

    def test_the_paged_count_substituted_back_is_invisible_on_one_and_4x_on_the_other(
        self, published
    ):
        """Why one config cannot settle this, as the two prices.

        On the dense stack the substitution changes nothing -- the two counts
        are one number there -- so a charge on the paged count is indistin-
        guishable from a charge on the stack, and a fit against it would carry
        the error forward as a coefficient. On the hybrid the same substitution
        divides the collective by exactly the ratio of the two counts.
        """
        hybrid = FakeModel(published, block_size=BLOCK_SIZE, parallelism=SHARDED)
        dense = FakeModel(
            SyntheticStack(num_hidden_layers=STACK_LAYERS),
            block_size=BLOCK_SIZE,
            parallelism=SHARDED,
        )
        assert collective_seconds(substituted(dense)) == pytest.approx(
            collective_seconds(dense.backend)
        )
        assert charged_layers(substituted(hybrid)) == PAGED_LAYERS
        assert collective_seconds(substituted(hybrid)) == pytest.approx(
            collective_seconds(hybrid.backend) * PAGED_LAYERS / STACK_LAYERS
        )

    def test_only_the_collective_moves_when_the_count_does(self, published):
        """The rest of the step is a function of the batch and nothing else."""
        model = FakeModel(published, block_size=BLOCK_SIZE, parallelism=SHARDED)
        on_stack = model.backend.price(batch())
        on_paged = substituted(model).price(batch())
        moved = {
            name
            for (name, stack, _), (_, paged, _) in zip(on_stack.rows(), on_paged.rows())
            if stack != paged
        }
        assert moved == {"collective.tp-all-reduce"}
        assert on_stack.seconds - on_paged.seconds == pytest.approx(
            REQUESTS
            * CHUNK
            * (STACK_LAYERS - PAGED_LAYERS)
            * Coefficients().collective_token_layer
        )


# ── a worker holds a span, not a stack ──────────────────────────────────────


class TestWhatAPipelineStageCharges:
    """The count is the worker's span, which is the stack only at one stage.

    The spans come from ATOM's own partitioner, so what is asserted is that the
    charge follows the engine's answer rather than a second one computed here.
    """

    def test_each_stage_charges_the_layers_it_runs(self, published):
        stages = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(tp_size=4, pp_size=4),
            layer_range=get_pp_indices(STACK_LAYERS, 0, 4),
        ).stages([get_pp_indices(STACK_LAYERS, rank, 4) for rank in range(4)])
        assert [stage.geometry.layers for stage in stages] == [4, 4, 4, 4]
        assert [charged_layers(stage.backend) for stage in stages] == [16, 16, 16, 16]

    def test_the_stages_charges_add_up_to_the_undivided_workers(self, published):
        """A stack split four ways still runs its 64 layers once per step."""
        whole = FakeModel(published, block_size=BLOCK_SIZE, parallelism=SHARDED)
        stages = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(tp_size=4, pp_size=4),
            layer_range=get_pp_indices(STACK_LAYERS, 0, 4),
        ).stages([get_pp_indices(STACK_LAYERS, rank, 4) for rank in range(4)])
        assert sum(
            collective_seconds(stage.backend) for stage in stages
        ) == pytest.approx(collective_seconds(whole.backend))


# ── what the record says ────────────────────────────────────────────────────


class TestWhatTheRecordSays:
    def test_the_record_names_the_count_the_charge_was_made_on(self, published):
        """Both counts, in the line a run record keeps.

        The seconds carry neither, and a reader who is told only that a
        collective was charged cannot tell a hybrid's 64 from its 16.
        """
        described = FakeModel(
            published, block_size=BLOCK_SIZE, parallelism=SHARDED
        ).describe()
        assert "tp-all-reduce on 64 layers, 16 paged" in described
        assert "16 paged of 64" in described
