# SPDX-License-Identifier: MIT
"""The stand-in model: the widths declared once, and what reads them.

The geometry and the price already exist and are already tested. What is
tested here is the join, and the reason a join is worth a type: a `KvGeometry`
is five integers and does not record the widths it was divided by, so a pool
sized for four ranks and a price charged as one rank's agree with each other
by accident or not at all. The first class below builds that mismatch to show
it is reachable, then shows the same declaration through `FakeModel` cannot
produce it.

Everything about a block count is asserted by handing the geometry to the code
that consumes it for real -- `page_pool` and `plan_pools` from
`atom.model_ops.attentions`, `BlockManager` from `atom.model_engine`, and
ATOM's own pipeline partitioner -- rather than by re-deriving the arithmetic
here, which would pass against a number the engine refuses.

The published Qwen3.8-27B config is vendored beside this file and is read
twice: once through this package's own JSON reader, and once through
`transformers`, so the reader is checked against the library rather than
against itself. The dialled stack is the other source the milestone needs --
shapes no released model has -- and it is read through the same code path, so
a test of it is a test of the reader and not of a second one.

Nothing here is an accuracy check. The coefficients behind every duration
below were declared, and the geometry is a double that omits the scale plane a
real backend carries. Every duration that carries a collective also depends on
which layer count the collective is charged per, which is a choice in the
pricing and not a property of this config; those expectations are keyed on
that count rather than written as one number, so a change to it moves them
rather than breaking them silently.
"""

import json
import pathlib

import pytest
from conftest import atom_config_double
from transformers import PretrainedConfig

from atom.compass.backends import (
    BatchView,
    Coefficients,
    FakeModel,
    HfConfig,
    KvGeometry,
    Parallelism,
    RequestShape,
    ShapeStubBackend,
    SyntheticStack,
    hf_config,
)
from atom.compass.backends import geometry as geometry_module
from atom.model_engine.block_manager import BlockManager
from atom.model_ops.attentions.pool_layout.sub_pool_spec import page_pool, plan_pools
from atom.models.utils import get_pp_indices

CONFIG_JSON = pathlib.Path(__file__).with_name("qwen3_5_27b_config.json")
BLOCK_SIZE = 64
MAX_NUM_SEQS = 512
# A budget, not a measurement: the device readings are another task's, and a
# fixed number is what makes these counts reproducible.
KV_BUDGET_BYTES = 64 << 30
# The batch every price below is read at: four requests of one chunk each,
# nothing cached, so the quadratic term is visible and the cross term is zero.
CHUNK = 256
REQUESTS = 4

# The published stack's two layer counts. The tensor-parallel all-reduce runs
# on every layer of the stack; the step price charges its collective per layer
# that holds a cache of every past token, which on this hybrid config is one
# layer in four. Every duration below that includes a collective therefore
# depends on which of the two counts is charged, so none of them is written
# down as a single literal: the tables are keyed on the charged count, the
# assertions read that count off the model, and a change to which count is
# charged moves them to the other row instead of reddening them with nothing
# in them to say the red was intended.
STACK_LAYERS = 64
PAGED_LAYERS = 16
# This batch with no collective charged at all. Independent of both counts.
UNCHARGED_STEP_SECONDS = 5.3724288e-4
# The same batch with the collective, by the layer count it is charged on.
STEP_SECONDS_BY_CHARGED_LAYERS = {
    PAGED_LAYERS: 5.5362688e-4,
    STACK_LAYERS: 6.0277888e-4,
}
# And as a ratio against the uncharged step: 1.030496x as charged today,
# 1.121986x if the charge moved to the depth an all-reduce actually runs on.
RATIO_BY_CHARGED_LAYERS = {
    PAGED_LAYERS: 1.0304964488314858,
    STACK_LAYERS: 1.1219857953259430,
}


@pytest.fixture(scope="module")
def published():
    """The vendored config, read by this package rather than by a library."""
    return hf_config(CONFIG_JSON)


def batch():
    return BatchView(tuple(RequestShape(CHUNK, CHUNK, False) for _ in range(REQUESTS)))


def blocks_for(geometry, budget=KV_BUDGET_BYTES):
    """The block count ATOM's own pool sizing gives this geometry."""
    plan = plan_pools([page_pool(geometry.bytes_per_block)], budget, MAX_NUM_SEQS)
    return plan.paged_entries


def block_manager_for(geometry, budget=KV_BUDGET_BYTES):
    """A real BlockManager sized from this geometry."""
    plan = plan_pools([page_pool(geometry.bytes_per_block)], budget, MAX_NUM_SEQS)
    config = atom_config_double(
        num_kvcache_blocks=plan.paged_entries,
        kv_cache_block_size=BLOCK_SIZE,
        max_num_seqs=MAX_NUM_SEQS,
    )
    config.pool_entries = dict(plan.entries)
    config.pool_entries_per_req = dict(plan.entries_per_req)
    return BlockManager(config)


def charged(step):
    return [name for name, _, _ in step.rows() if name.startswith("collective.")]


def charged_layers(model):
    """The layer count this model's collective was charged on, read back.

    The collective term is `tokens x layers x a coefficient`, so dividing the
    charge by the tokens and the coefficient recovers the count the pricing
    used. Recovered rather than restated: which count the collective is
    charged on is the pricing's choice and not this config's, so a change to
    it moves the expectations keyed on this to their other row instead of
    leaving them asserting the old one. A count that is neither of the two
    this config declares has no row here and is refused by name.
    """
    step = model.backend.estimate(batch())
    seconds = sum(
        value for name, value, _ in step.rows() if name.startswith("collective.")
    )
    per_layer = REQUESTS * CHUNK * model.coefficients.collective_token_layer
    count = round(seconds / per_layer)
    assert seconds == pytest.approx(count * per_layer), (
        f"a collective of {seconds} s is not a whole number of layers at "
        f"{per_layer} s each, so it is not the term this reads it as"
    )
    assert count in RATIO_BY_CHARGED_LAYERS, (
        f"the collective is charged on {count} layers, which is neither this "
        f"config's {PAGED_LAYERS} paged layers nor its {STACK_LAYERS}-layer "
        "stack; the expectations here have no row for that count"
    )
    return count


# ── the widths, declared once ───────────────────────────────────────────────


class TestOneDeclaration:
    """What the join is for, shown as the pair it makes unbuildable."""

    def test_the_halves_can_be_built_disagreeing_without_it(self, published):
        """The pool is four ranks wide and the price is one rank's.

        Neither half is wrong on its own and nothing can see the pair: the
        geometry carries no record of the widths it was divided by, so the
        backend has no way to check the number it was handed and does not
        try. This is the shape the type below exists to remove, asserted so
        the removal is a change in what is reachable rather than a claim.
        """
        sharded = KvGeometry.from_hf_config(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=4)
        )
        mismatched = ShapeStubBackend(geometry=sharded)
        assert blocks_for(sharded) == 4 * blocks_for(
            KvGeometry.from_hf_config(published, block_size=BLOCK_SIZE)
        )
        assert charged(mismatched.estimate(batch())) == []

    def test_the_model_hands_both_halves_the_same_widths(self, published):
        widths = Parallelism(tp_size=4)
        model = FakeModel(published, block_size=BLOCK_SIZE, parallelism=widths)
        assert model.backend.parallelism is widths
        assert model.backend.geometry is model.geometry
        assert charged(model.backend.estimate(batch())) == ["collective.tp-all-reduce"]

    def test_the_price_of_the_disagreement_is_the_collective(self, published):
        """The two prices differ by the term one of them does not know to add.

        Same batch, same coefficients, same sharded pool; the only difference
        is which object was told the width. What the difference *is* -- a term
        present or absent, not a term mis-sized -- is the assertion; how large
        it is depends on a declared coefficient and on which layer count the
        collective is charged per, so the ratio is looked up by that count
        rather than written down. Charged per paged layer, as it is today,
        this config gives 1.030496x; charged on the 64 layers an all-reduce
        runs on it gives 1.121986x. Both are in the table and neither is a
        measurement.
        """
        sharded = KvGeometry.from_hf_config(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=4)
        )
        model = FakeModel(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=4)
        )
        unpaired = ShapeStubBackend(geometry=sharded).estimate(batch()).seconds
        paired = model.backend.estimate(batch()).seconds
        count = charged_layers(model)
        assert paired - unpaired == pytest.approx(
            REQUESTS * CHUNK * count * Coefficients().collective_token_layer
        )
        assert paired / unpaired == pytest.approx(RATIO_BY_CHARGED_LAYERS[count])

    def test_the_other_row_of_the_tables_is_priced_and_not_asserted_arithmetic(self):
        """Where 1.121986x comes from: the same stub handed 64 layers.

        Only the collective term reads the layer count -- it is `tokens x
        layers x a coefficient` and the rest of the step depends on the batch
        alone -- so a stand-in whose layers are all paged prices this batch
        exactly as the published hybrid would if the charge moved to the depth
        an all-reduce runs on. The second row of both tables at the top of
        this file is that price, read out of the stub rather than worked out
        in a comment, so neither row is a number with nothing behind it.
        """
        dense = FakeModel(
            SyntheticStack(num_hidden_layers=STACK_LAYERS),
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(tp_size=4),
        )
        assert dense.geometry.layers == STACK_LAYERS
        # And the read-back above recovers 64 as readily as it recovers 16, so
        # the assertions keyed on it follow the charge to this row rather than
        # needing to be rewritten when it moves.
        assert charged_layers(dense) == STACK_LAYERS
        unpaired = ShapeStubBackend(geometry=dense.geometry).estimate(batch()).seconds
        paired = dense.backend.estimate(batch()).seconds
        assert unpaired == pytest.approx(UNCHARGED_STEP_SECONDS)
        assert paired == pytest.approx(STEP_SECONDS_BY_CHARGED_LAYERS[STACK_LAYERS])
        assert paired / unpaired == pytest.approx(RATIO_BY_CHARGED_LAYERS[STACK_LAYERS])


# ── the named result ────────────────────────────────────────────────────────


class TestTheWidthSweep:
    """One declaration per width, and both of the things that read it.

    The block count is ATOM's, from the real pool sizing; the duration is the
    stub's, for one fixed batch. The pair is what a simulated run holds, and
    at every width here it came from one object that was told the width once.

    The block counts are the model's own arithmetic and stand on their own.
    The durations at more than one rank carry a collective charged per layer
    holding a cache of every past token, so they are looked up by that count
    rather than stated flat -- see the two tables at the top of this file.
    """

    @pytest.mark.parametrize(
        "tp_size, kv_heads, blocks",
        [(1, 4, 16384), (2, 2, 32768), (4, 1, 65536), (8, 1, 65536)],
    )
    def test_the_pool_and_the_price_both_come_from_the_one_model(
        self, published, tp_size, kv_heads, blocks
    ):
        model = FakeModel(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=tp_size)
        )
        assert model.geometry.kv_heads == kv_heads
        assert blocks_for(model.geometry) == blocks
        step = model.backend.estimate(batch())
        assert charged(step) == ([] if tp_size == 1 else ["collective.tp-all-reduce"])
        assert step.seconds == pytest.approx(
            UNCHARGED_STEP_SECONDS
            if tp_size == 1
            else STEP_SECONDS_BY_CHARGED_LAYERS[charged_layers(model)]
        )

    def test_the_block_count_stops_moving_at_the_grouped_query_bound(self, published):
        """Four KV heads cannot be cut into eight ranks, so they replicate."""
        counts = {
            tp: blocks_for(
                FakeModel(
                    published,
                    block_size=BLOCK_SIZE,
                    parallelism=Parallelism(tp_size=tp),
                ).geometry
            )
            for tp in (1, 2, 4, 8)
        }
        assert counts[2] == 2 * counts[1]
        assert counts[4] == 4 * counts[1]
        assert counts[8] == counts[4]

    def test_a_real_block_manager_runs_on_the_count_this_model_produced(
        self, published, seq_factory
    ):
        model = FakeModel(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=8)
        )
        manager = block_manager_for(model.geometry)
        assert manager.kv.num_free == 65536
        sequence = seq_factory(list(range(3 * BLOCK_SIZE)), block_size=BLOCK_SIZE)
        manager.allocate(sequence)
        assert len(sequence.block_table) == 3
        assert manager.kv.num_free == 65536 - 3


# ── pipeline stages ─────────────────────────────────────────────────────────


class TestPipelineStages:
    def test_one_model_per_stage_over_the_spans_the_partitioner_gave(self, published):
        whole = FakeModel(published, block_size=BLOCK_SIZE)
        parallelism = Parallelism(pp_size=4)
        declared = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=parallelism,
            layer_range=get_pp_indices(64, 0, 4),
        )
        stages = declared.stages([get_pp_indices(64, rank, 4) for rank in range(4)])
        assert [stage.geometry.layers for stage in stages] == [4, 4, 4, 4]
        assert [blocks_for(stage.geometry) for stage in stages] == [
            4 * blocks_for(whole.geometry)
        ] * 4

    def test_fewer_spans_than_stages_is_refused(self, published):
        model = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(0, 16),
        )
        with pytest.raises(ValueError, match="3 layer ranges for 4"):
            model.stages([(0, 16), (16, 32), (32, 48)])

    def test_spans_that_tile_the_wrong_stack_are_refused(self, published):
        """The right number of well-formed spans over a quarter of the model.

        This is the partitioner handed the wrong depth. Each span satisfies
        the geometry's own bound on its own, and there are four of them for
        four stages, so neither the count check nor any per-span check
        refuses it -- and the four pools together hold a quarter of the KV.
        """
        model = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(0, 16),
        )
        with pytest.raises(ValueError, match="do not partition the 64-layer stack"):
            model.stages([get_pp_indices(16, rank, 4) for rank in range(4)])

    def test_spans_that_repeat_a_layer_are_refused(self, published):
        """Four stages all holding the first span is four spans, not a cover."""
        model = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(0, 16),
        )
        with pytest.raises(ValueError, match="do not partition"):
            model.stages([(0, 16)] * 4)

    def test_a_gap_between_two_spans_is_refused(self, published):
        model = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(0, 16),
        )
        with pytest.raises(ValueError, match="do not partition"):
            model.stages([(0, 16), (16, 32), (32, 47), (48, 64)])

    def test_the_spans_may_arrive_in_any_order(self, published):
        """The partition is checked sorted; the stages come back as given."""
        model = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(0, 16),
        )
        stages = model.stages([(48, 64), (0, 16), (32, 48), (16, 32)])
        assert [stage.layer_range for stage in stages] == [
            (48, 64),
            (0, 16),
            (32, 48),
            (16, 32),
        ]

    def test_a_declared_stage_that_does_not_say_which_one_is_refused(self, published):
        """The geometry's own refusal, reached through this constructor."""
        with pytest.raises(ValueError, match="no layer_range"):
            FakeModel(
                published, block_size=BLOCK_SIZE, parallelism=Parallelism(pp_size=4)
            )

    def test_each_stage_prices_a_step_as_well_as_holding_one(self, published):
        """A stage is a worker: it has a pool and it has a duration."""
        stages = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=2, tp_size=2),
            layer_range=(0, 32),
        ).stages([(0, 32), (32, 64)])
        for stage in stages:
            assert charged(stage.backend.estimate(batch())) == [
                "collective.tp-all-reduce"
            ]


# ── the two layer counts ────────────────────────────────────────────────────


class TestWhatAStageHolds:
    def test_the_paged_layers_are_a_subset_of_the_layers(self, published):
        """A recurrent layer runs and takes part; it just costs a block nothing.

        The published stack is one paged layer in four, so the two counts are
        16 and 64 on one worker. A collective runs on every layer, and the
        charge the shape stub makes is per paged layer -- so the count this
        exposes is not the one that pricing consumes, which is why both are
        readable rather than one implied by the other.
        """
        model = FakeModel(published, block_size=BLOCK_SIZE)
        assert (model.geometry.layers, model.stage_layers) == (
            PAGED_LAYERS,
            STACK_LAYERS,
        )

    def test_a_stage_reports_the_span_it_was_given(self, published):
        stage = FakeModel(
            published,
            block_size=BLOCK_SIZE,
            parallelism=Parallelism(pp_size=4),
            layer_range=(16, 32),
        )
        assert stage.stage_layers == 16
        assert stage.geometry.layers == 4

    def test_a_uniform_stack_has_one_count_for_both(self):
        model = FakeModel(SyntheticStack(num_hidden_layers=12), block_size=BLOCK_SIZE)
        assert (model.geometry.layers, model.stage_layers) == (12, 12)


# ── the dialled stack ───────────────────────────────────────────────────────


class TestSyntheticStack:
    """The second config source: shapes no released model has.

    Read through the same attributes as a published config, so these are
    tests of the reader rather than of a second path into the geometry.
    """

    def test_a_dialled_stack_is_read_by_the_same_code_as_a_published_one(self):
        dense = SyntheticStack(
            num_hidden_layers=32,
            num_key_value_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
            head_dim=128,
            dtype="float16",
        )
        direct = KvGeometry.from_hf_config(dense, block_size=BLOCK_SIZE)
        assert direct == FakeModel(dense, block_size=BLOCK_SIZE).geometry
        assert (direct.layers, direct.head_dim, direct.element_bytes) == (32, 128, 2)

    def test_a_uniform_stack_names_no_layer_kinds(self):
        assert SyntheticStack().layer_types is None
        assert SyntheticStack(full_attention_interval=4).layer_types[:4] == [
            "linear_attention",
            "linear_attention",
            "linear_attention",
            "full_attention",
        ]

    def test_the_hybrid_dial_pages_one_layer_in_the_interval(self):
        hybrid = SyntheticStack(num_hidden_layers=48, full_attention_interval=6)
        model = FakeModel(hybrid, block_size=BLOCK_SIZE)
        assert (model.geometry.layers, model.stage_layers) == (8, 48)

    def test_a_shape_no_released_model_has_still_sizes_a_real_pool(self):
        """A hundred layers on one KV head, at a head dimension nothing ships.

        The point of the dial is the engine's reaction to it, so the count
        below is `plan_pools`' and a real block manager runs on it.
        """
        extreme = SyntheticStack(
            num_hidden_layers=100,
            num_key_value_heads=1,
            num_attention_heads=64,
            head_dim=512,
            dtype="float8_e4m3fn",
        )
        geometry = FakeModel(extreme, block_size=BLOCK_SIZE).geometry
        assert geometry.bytes_per_block == 100 * 64 * 2 * 1 * 512 * 1
        assert block_manager_for(geometry).kv.num_free == blocks_for(geometry) > 0

    def test_dialling_the_kv_heads_moves_what_atom_sizes(self):
        counts = {
            heads: blocks_for(
                FakeModel(
                    SyntheticStack(num_key_value_heads=heads, num_attention_heads=32),
                    block_size=BLOCK_SIZE,
                ).geometry
            )
            for heads in (1, 2, 4, 8)
        }
        assert counts[1] == 8 * counts[8]
        assert counts[2] == 4 * counts[8]

    def test_an_unset_head_dim_is_derived_from_the_hidden_size(self):
        """The reader's one fallback, reached from the dial.

        A published config may state the head dimension or leave it out, and
        left out the reader divides `hidden_size` by the query head count.
        That shape is what a dialled fixture has to be able to produce, and it
        is the only position from which `hidden_size` decides anything.
        """
        narrow = FakeModel(
            SyntheticStack(head_dim=None, hidden_size=4096, num_attention_heads=32),
            block_size=BLOCK_SIZE,
        ).geometry
        wide = FakeModel(
            SyntheticStack(head_dim=None, hidden_size=8192, num_attention_heads=32),
            block_size=BLOCK_SIZE,
        ).geometry
        assert (narrow.head_dim, wide.head_dim) == (128, 256)
        assert wide.bytes_per_block == 2 * narrow.bytes_per_block

    def test_a_stated_head_dim_is_what_is_read(self):
        """Stated, `hidden_size` is carried and read by nobody -- as published."""
        stated = KvGeometry.from_hf_config(
            SyntheticStack(head_dim=64, hidden_size=4096), block_size=BLOCK_SIZE
        )
        assert stated.head_dim == 64
        assert stated == KvGeometry.from_hf_config(
            SyntheticStack(head_dim=64, hidden_size=999999), block_size=BLOCK_SIZE
        )

    def test_a_hidden_size_too_small_to_divide_is_refused_by_name(self):
        with pytest.raises(ValueError, match="head_dim is unset"):
            SyntheticStack(head_dim=None, hidden_size=16, num_attention_heads=32)

    def test_a_dial_that_is_not_a_number_is_refused_by_name(self):
        """`None` on a dial that has no unset meaning names the field."""
        with pytest.raises(TypeError, match="num_hidden_layers must be a whole"):
            SyntheticStack(num_hidden_layers=None)

    def test_the_dial_spells_its_layer_kinds_the_way_the_reader_reads_them(self):
        """The kinds come from the module that decides what they mean.

        The dial's promise is that a shape it produces is a shape the reader
        accepts, and that holds only while the two agree on how a kind is
        spelled. So the emitted kinds are checked against the reader's own
        sets rather than against a literal written out a second time here.
        """
        emitted = set(
            SyntheticStack(num_hidden_layers=8, full_attention_interval=2).layer_types
        )
        paged = geometry_module._PAGED_LAYER_KINDS
        uncached = geometry_module._UNCACHED_LAYER_KINDS
        assert emitted <= paged | uncached
        assert emitted & paged and emitted & uncached

    def test_a_stack_whose_heads_do_not_group_is_refused(self):
        with pytest.raises(ValueError, match="do not group"):
            SyntheticStack(num_attention_heads=30, num_key_value_heads=8)

    def test_an_interval_that_pages_no_layer_is_refused(self):
        with pytest.raises(ValueError, match="places none at all"):
            SyntheticStack(num_hidden_layers=8, full_attention_interval=9)

    def test_a_dial_below_one_is_refused(self):
        with pytest.raises(ValueError, match="num_hidden_layers must be at least 1"):
            SyntheticStack(num_hidden_layers=0)

    def test_a_dtype_the_geometry_cannot_size_is_refused_where_it_is_read(self):
        """The dial does not pre-empt the reader's own refusal."""
        with pytest.raises(ValueError, match="no element size known"):
            FakeModel(SyntheticStack(dtype="posit8"), block_size=BLOCK_SIZE)


# ── reading a published config ──────────────────────────────────────────────


class TestPublishedConfig:
    def test_the_reader_agrees_with_the_library_on_the_geometry(self, published):
        """Checked against `transformers`, not against a second derivation."""
        raw = json.loads(CONFIG_JSON.read_text())
        library = PretrainedConfig.from_dict(raw["text_config"])
        assert KvGeometry.from_hf_config(
            library, block_size=BLOCK_SIZE
        ) == KvGeometry.from_hf_config(published, block_size=BLOCK_SIZE)

    def test_the_nested_text_config_is_what_the_geometry_reads(self, published):
        """The published file is a multimodal config and the stack is inside it."""
        assert published.text_config.num_hidden_layers == 64
        assert FakeModel(published, block_size=BLOCK_SIZE).stage_layers == 64

    def test_a_key_that_is_absent_is_an_attribute_error_naming_it(self, published):
        with pytest.raises(AttributeError, match="no num_hidden_layers"):
            _ = published.num_hidden_layers

    def test_an_absent_optional_key_reads_as_none(self):
        """The geometry's optional reads have to come back `None`, not raise."""
        uniform = hf_config({"num_hidden_layers": 4})
        assert getattr(uniform, "layer_types", None) is None
        assert getattr(uniform, "full_attention_interval", None) is None

    def test_the_model_reads_its_config_from_a_path(self):
        model = FakeModel.from_json(CONFIG_JSON, block_size=BLOCK_SIZE)
        assert (model.geometry.layers, model.geometry.kv_heads) == (16, 4)

    def test_something_that_is_not_a_config_is_refused(self):
        with pytest.raises(TypeError, match="a mapping of names"):
            HfConfig(4)


# ── what the record says ────────────────────────────────────────────────────


class TestWhatTheRecordSays:
    def test_the_description_names_both_counts_and_the_block(self, published):
        described = FakeModel(published, block_size=BLOCK_SIZE).describe()
        assert "16 paged of 64" in described
        assert "4 KV heads x 256 at 2 B" in described
        assert "4194304 B/block" in described

    def test_a_single_replicated_head_is_not_described_as_heads(self, published):
        sharded = FakeModel(
            published, block_size=BLOCK_SIZE, parallelism=Parallelism(tp_size=4)
        )
        assert "1 KV head x 256" in sharded.describe()

    def test_the_description_carries_the_backends_disclaimer(self, published):
        """A reader of the record is told, without opening any source."""
        model = FakeModel(published, block_size=BLOCK_SIZE)
        assert model.backend.describe() in model.describe()
        assert "not an accuracy claim" in model.describe()

    def test_constant_pricing_reaches_the_model_as_a_coefficient_set(self, published):
        flat = Coefficients.constant(prefill_seconds=0.05, decode_seconds=0.002)
        model = FakeModel(published, block_size=BLOCK_SIZE, coefficients=flat)
        assert model.backend.estimate(batch()).seconds == 0.05
        assert "constant bring-up" in model.describe()

    def test_no_document_identifier_reaches_the_record(self, published):
        """The record says what the thing is, and cites nothing."""
        described = FakeModel(published, block_size=BLOCK_SIZE).describe()
        assert "D12" not in described
        assert "M1" not in described
