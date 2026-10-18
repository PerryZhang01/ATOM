# SPDX-License-Identifier: MIT
"""The invalidation matrix, the gate state in the artifact, and the ledger.

**The whole matrix is exercised, not one row, and the zeroes as hard as the
crosses.** `test_every_cell_of_the_matrix_decides_by_itself` walks
every cell: where the matrix says a row depends on an axis, moving that
axis refuses and the refusal names the cell; where it says the row does not,
moving the same axis loads clean. A matrix that has been tidied into a uniform rule
passes neither half, and one pair -- `price_list` surviving a model change
while `memory_readings` refuses one -- is also published and loaded through the
store, because a table that is right in the abstract and unwired in the store
is machinery with no caller.

Two things this file deliberately does not do.

**It moves no layout.** No column of the matrix is a parallel axis: the
layout is in the key, so another layout is another entry, never a stale one.

**It settles nothing left open.** Whether an aiter bump invalidates or only
warns is open (#168), and while it is open the default stands:
`test_an_aiter_bump_refuses_while_its_meaning_is_open` pins the default, and
it is the test that changes when that is settled.

Nothing here touches a driver, a device or a network. The conditions are
stated and never probed, which is also why a fixture can express a ROCm bump
on a host that has one ROCm.
"""

import json

import pytest

from atom.compass.artifacts import (
    MATRIX,
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    Conditions,
    Fingerprint,
    Gate,
    GateState,
    Key,
    Kind,
    OnMismatch,
    Provenance,
    Reading,
    Resolution,
    Row,
    Rule,
    SourceRoot,
    StaleArtifact,
    Topology,
    axes_of,
    differences,
    fingerprint,
    member_name,
    rows_for,
    verify,
)

ATOM_ROOT = SourceRoot(
    "atom",
    "/workspace/ATOM/atom",
    "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c",
    "tree",
    "git rev-parse HEAD^{tree}, the tree git archive ships",
    0,
)
AITER_ROOT = SourceRoot(
    "aiter",
    "/app/aiter-test/aiter",
    "v0.1.21.dev0-49-gf4e7c7509",
    "describe",
    "git describe --tags --always --dirty",
    0,
)
STANZA = Provenance(
    "compass calibrate phase-1b", "2026-09-22T11:00:00+00:00", (ATOM_ROOT, AITER_ROOT)
)
#: The gate whose dead form took every per-kernel breakdown at TP=2.
PRICE_KERNELS = Gate("PRICE_KERNELS", "off", "COMPASS_PRICE_KERNELS")
GATES = GateState.of(PRICE_KERNELS)

#: One rank per tensor-parallel rank, and no other axis.
TP2 = Topology(tp=2)
#: The layout the layout-keyed rows state, the one `TP2` publishes at.
LAYOUT = {
    "dp": 1,
    "pp": 1,
    "pcp": 1,
    "tp": 2,
    "enable_expert_parallel": False,
    "enable_dp_attention": False,
    "decode_context_parallel_size": 1,
}

BASE = Conditions.of(
    software_stack="rocm7.2.4 / aiter v0.1.21.dev0-49-gf4e7c7509 / rccl2.22.3",
    torch="2.10.0+rocm7.2.4",
    atom_src=Reading.of_source_root(ATOM_ROOT),
    model="Qwen/Qwen3-32B",
    device="MI308X-80CU",
    ptl="disabled",
    host_cpu="Intel(R) Xeon(R) Platinum 8480C",
    engine_config="cudagraph=piecewise,level=3",
)

#: A key per row of the matrix. The three `machine_spec` rows share one key,
#: because they are three facets of one artifact and that is the point of them.
KEY_OF: dict[Row, Key] = {
    Row.OP_GRAPH: Key.of(Kind.OP_GRAPH, structure="qwen3-moe-48L", shape="decode B=2"),
    Row.PRICE_LIST: Key.of(
        Kind.PRICE_LIST,
        model="Qwen/Qwen3-32B",
        machine_spec="sha256:mi308x",
        source_root=ATOM_ROOT.revision,
        **LAYOUT,
    ),
    Row.MEMORY_READINGS: Key.of(
        Kind.MEMORY_READINGS,
        model="Qwen/Qwen3-32B",
        machine_spec="sha256:mi308x",
        **LAYOUT,
        engine_config="cudagraph=piecewise,level=3",
    ),
    Row.MACHINE_SPEC_CAPACITY: Key.of(
        Kind.MACHINE_SPEC, device="MI308X-80CU", software_stack="rocm7.2.4", ptl="x"
    ),
    Row.STEP_GRID_BUNDLE: Key.of(
        Kind.STEP_GRID_BUNDLE,
        model="Qwen/Qwen3-32B",
        device="MI308X-80CU",
        engine_config="replay-mns48",
        grid="s2-sparse-margin0.25",
        source_root=ATOM_ROOT.revision,
    ),
}
KEY_OF[Row.MACHINE_SPEC_TOKENIZER_TERMS] = KEY_OF[Row.MACHINE_SPEC_CAPACITY]


def moved(conditions: Conditions, axis: Axis) -> Conditions:
    """The same conditions with one axis bumped, and its *kind* left alone.

    Moving the kind as well would be a different event -- a reading that
    cannot be compared rather than one that changed -- and the two have their
    own tests.
    """
    was = conditions.reading(axis)
    return conditions.with_reading(axis, Reading(was.kind, was.value + "+moved"))


def publish(
    store: ArtifactStore,
    key: Key,
    *,
    conditions: Conditions = BASE,
    gates: GateState = GATES,
    notes: str = "",
) -> object:
    """One entry, with every rank of `TP2` present."""
    return store.publish(
        key,
        provenance=STANZA,
        topology=TP2,
        conditions=conditions,
        gates=gates,
        members={
            member_name("rows", rank, "json"): b'{"measured": 1}'
            for rank in TP2.ranks()
        },
        notes=notes,
    )


# --- the matrix is a table ------------------------------------------------


def test_the_matrix_is_not_uniform():
    """The cells a tidying hand takes away, named one at a time.

    Each is a sentence the document argues for, and a matrix that lost them
    would still pass a test that only checked the crosses.
    """
    assert not MATRIX[Row.PRICE_LIST][Axis.MODEL].depends
    assert MATRIX[Row.PRICE_LIST][Axis.MODEL].note == "shape-parametric"
    assert MATRIX[Row.MEMORY_READINGS][Axis.MODEL].depends
    assert not MATRIX[Row.MACHINE_SPEC_TOKENIZER_TERMS][Axis.DEVICE].depends
    assert not MATRIX[Row.MACHINE_SPEC_CAPACITY][Axis.SOFTWARE_STACK].depends


@pytest.mark.parametrize("kind", [Kind.COVERAGE_HULL])
def test_a_kind_the_matrix_does_not_row_is_refused_by_name(kind):
    """The key table and the matrix do not hold the same artifacts.

    The tempting default is "depends on nothing", which loads clean forever,
    and the other is "depends on everything", which re-measures forever. Both
    answer a question the document did not.
    """
    with pytest.raises(ArtifactRefusal) as refused:
        rows_for(kind)
    assert refused.value.rule is Rule.INVALIDATED
    assert f"no dependency row for {kind}" in str(refused.value)
    assert "do not hold the same artifacts" in str(refused.value)


def test_a_shape_population_is_rowed_on_source_model_and_engine_config():
    """Its pricing backend is a key field, so no axis carries it."""
    assert rows_for(Kind.SHAPE_POPULATION) == (Row.SHAPE_POPULATION,)
    assert axes_of(Row.SHAPE_POPULATION) == (
        Axis.ATOM_SRC,
        Axis.MODEL,
        Axis.ENGINE_CONFIG,
    )


@pytest.mark.parametrize(
    "kind,row",
    [
        (Kind.STEP_MEASUREMENT, Row.STEP_MEASUREMENT),
        (Kind.STEP_LINEAR_LAW, Row.STEP_LINEAR_LAW),
        (Kind.HOST_LAWS, Row.HOST_LAWS),
    ],
)
def test_each_new_kind_is_checked_against_its_own_row(kind, row):
    assert rows_for(kind) == (row,)


def test_only_host_laws_depend_on_the_host_cpu():
    """Host laws time the host; step measurements and laws hold device time,
    so they keep the step grid bundle's axes, and the host CPU is not one."""
    assert [row for row in Row if MATRIX[row][Axis.HOST_CPU].depends] == [Row.HOST_LAWS]
    assert axes_of(Row.HOST_LAWS) == (
        Axis.ATOM_SRC,
        Axis.MODEL,
        Axis.HOST_CPU,
        Axis.ENGINE_CONFIG,
    )
    for row in (Row.STEP_MEASUREMENT, Row.STEP_LINEAR_LAW):
        assert axes_of(row) == axes_of(Row.STEP_GRID_BUNDLE)


def test_host_laws_published_on_one_host_cpu_refuse_another_naming_it(tmp_path):
    store = ArtifactStore(tmp_path)
    key = Key.of(
        Kind.HOST_LAWS,
        host_cpu=BASE.reading(Axis.HOST_CPU).value,
        atom_src="0f1e2d3c",
        model="sha256:191e0af2",
        engine_config="{}",
        machine_spec="sha256:5ec0",
    )
    publish(store, key)
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(key, conditions=moved(BASE, Axis.HOST_CPU), gates=GATES)
    assert refused.value.rule is Rule.INVALIDATED
    assert "`host_laws` x `host CPU`" in str(refused.value)


# --- every cell, the zeroes as hard as the crosses -------------------------


@pytest.mark.parametrize(
    "row,axis",
    [(row, axis) for row in Row for axis in Axis],
    ids=[f"{row.name}-{axis.name}" for row in Row for axis in Axis],
)
def test_every_cell_of_the_matrix_decides_by_itself(row, axis):
    """Every cell: a cross refuses and names itself, a zero loads clean.

    It is one test rather than one per row because the claim is about the table
    and not about any row: a rule that refused on every axis would pass every
    cross and fail every zero, and a rule that refused on none would do the
    reverse.
    """
    recorded = {row: fingerprint(row, BASE)}
    changed = moved(BASE, axis)
    if not MATRIX[row][axis].depends:
        assert verify(recorded, changed) == ()
        return
    with pytest.raises(ArtifactRefusal) as refused:
        verify(recorded, changed)
    assert refused.value.rule is Rule.INVALIDATED
    assert f"`{row}` x `{axis}`" in str(refused.value)
    assert BASE.reading(axis).value in str(refused.value)


def test_a_fingerprint_carries_only_the_cells_its_row_depends_on():
    """The recorded fingerprint is the row's, not every axis narrowed later.

    An entry that recorded every axis and filtered on read would be checked
    against whatever the reader believed the matrix said, which is a global
    fingerprint wearing a per-artifact name.
    """
    assert fingerprint(Row.PRICE_LIST, BASE).axes == axes_of(Row.PRICE_LIST)
    assert Axis.MODEL not in fingerprint(Row.PRICE_LIST, BASE).axes
    assert Axis.MODEL in fingerprint(Row.MEMORY_READINGS, BASE).axes
    assert fingerprint(Row.MACHINE_SPEC_CAPACITY, BASE).axes == (Axis.DEVICE,)


def test_no_axis_of_the_matrix_is_a_width():
    """No axis is a width: the layout is in the key, not in a fingerprint."""
    assert {axis.field for axis in Axis} == {
        "software_stack",
        "torch",
        "atom_src",
        "model",
        "device",
        "ptl",
        "host_cpu",
        "engine_config",
    }
    assert TP2.rank_count == TP2.tp


# --- one row survives what the other refuses, through the store ------------


def test_a_price_list_survives_a_model_change_and_memory_readings_refuse_one(tmp_path):
    """The pair that proves the table is a table and not a uniform rule."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    publish(store, KEY_OF[Row.MEMORY_READINGS])
    other = BASE.with_reading(Axis.MODEL, "deepseek-ai/DeepSeek-V3.1")

    survived = store.load(KEY_OF[Row.PRICE_LIST], conditions=other, gates=GATES)
    assert survived.key == KEY_OF[Row.PRICE_LIST]

    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.MEMORY_READINGS], conditions=other, gates=GATES)
    assert refused.value.rule is Rule.INVALIDATED
    assert "`memory_readings` x `model`" in str(refused.value)
    assert "deepseek-ai/DeepSeek-V3.1" in str(refused.value)


def test_a_machine_spec_is_its_rows_and_a_device_change_moves_the_capacity(tmp_path):
    """One artifact, a fingerprint per row, invalidated by different things.

    This is the row that would be destroyed by tidying `machine_spec` into one
    entry-wide fingerprint: the capacity is the silicon's, and the tokenizer
    terms are host CPU work that no device change touches.
    """
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[Row.MACHINE_SPEC_CAPACITY])
    assert set(entry.fingerprints) == set(rows_for(Kind.MACHINE_SPEC))

    with pytest.raises(ArtifactRefusal) as device:
        store.load(entry.key, conditions=moved(BASE, Axis.DEVICE), gates=GATES)
    assert "`machine_spec: capacity` x `device`" in str(device.value)
    assert "tokenizer terms" not in str(device.value)

    with pytest.raises(ArtifactRefusal) as tokenizer:
        store.load(entry.key, conditions=moved(BASE, Axis.MODEL), gates=GATES)
    assert "`machine_spec: tokenizer terms` x `model` (tokenizer)" in str(
        tokenizer.value
    )
    assert "capacity" not in str(tokenizer.value)

    assert (
        store.load(entry.key, conditions=moved(BASE, Axis.TORCH), gates=GATES).key
        == entry.key
    )


def test_an_aiter_bump_refuses_while_its_meaning_is_open(tmp_path):
    """The default, because what an aiter bump means is left open (#168).

    The rows that carry ROCm/AITER/RCCL refuse a bump; `machine_spec` does
    not, because the silicon did not move. If it is settled as "warn", this is
    the test that changes, and one cell of `MATRIX` with it.
    """
    store = ArtifactStore(tmp_path)
    bumped = BASE.with_reading(
        Axis.SOFTWARE_STACK,
        "rocm7.2.4 / aiter v0.1.20-103-g23f83724f / rccl2.22.3",
    )
    for row in (Row.PRICE_LIST, Row.MEMORY_READINGS):
        publish(store, KEY_OF[row])
        with pytest.raises(ArtifactRefusal) as refused:
            store.load(KEY_OF[row], conditions=bumped, gates=GATES)
        assert refused.value.rule is Rule.INVALIDATED
        assert f"`{row}` x `ROCm / AITER / RCCL`" in str(refused.value)

    spec = publish(store, KEY_OF[Row.MACHINE_SPEC_CAPACITY])
    assert store.load(spec.key, conditions=bumped, gates=GATES).key == spec.key


def test_a_refusal_names_every_cell_that_moved(tmp_path):
    """Two axes moved, two cells named -- never an aggregate without its parts."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.MEMORY_READINGS])
    drifted = moved(moved(BASE, Axis.DEVICE), Axis.ENGINE_CONFIG)
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.MEMORY_READINGS], conditions=drifted, gates=GATES)
    assert "`memory_readings` x `device`" in str(refused.value)
    assert "`memory_readings` x `engine cfg`" in str(refused.value)


# --- the explicit flag, and the comparison that cannot be made -------------


def test_the_explicit_flag_warns_and_still_names_the_cell(tmp_path):
    """A warning is allowed, and only under a flag the caller has to write."""
    store = ArtifactStore(tmp_path)
    published = publish(store, KEY_OF[Row.MEMORY_READINGS])
    with pytest.warns(StaleArtifact, match=r"`memory_readings` x `model`"):
        loaded = store.load(
            KEY_OF[Row.MEMORY_READINGS],
            conditions=moved(BASE, Axis.MODEL),
            gates=GATES,
            on_mismatch=OnMismatch.WARN,
        )
    assert loaded.digest == published.digest


@pytest.mark.parametrize("policy", list(OnMismatch))
def test_a_tree_and_a_commit_are_not_compared(tmp_path, policy):
    """`revision_kind` is a field to branch on, not to compare.

    The primary path records a git tree and the `.compass-commit` stamp path
    records a commit, so two entries published from byte-identical source
    carry different text in one field. Reporting that as a change is a claim
    nobody observed, and the flag has no answer to downgrade.
    """
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    stamped = BASE.with_reading(
        Axis.ATOM_SRC, Reading("commit", "065f34f06bc7de314b37ec85b549feea104aa57e")
    )
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(
            KEY_OF[Row.PRICE_LIST],
            conditions=stamped,
            gates=GATES,
            on_mismatch=policy,
        )
    assert refused.value.rule is Rule.NOT_COMPARABLE
    assert "recorded a tree and this run reads a commit" in str(refused.value)


def test_conditions_that_state_fewer_than_every_axis_are_refused():
    """An axis nobody stated is an axis nothing is certified against."""
    with pytest.raises(ArtifactRefusal) as refused:
        Conditions.of(torch="2.10.0", model="Qwen/Qwen3-32B")
    assert refused.value.rule is Rule.INVALIDATED
    assert "no atom_src, device, engine_config, host_cpu, ptl, software_stack" in str(
        refused.value
    )


def test_conditions_that_state_no_host_cpu_are_refused():
    """Required like every other axis, though only host laws depend on it."""
    with pytest.raises(ArtifactRefusal) as refused:
        Conditions.of(
            **{
                axis.field: BASE.reading(axis)
                for axis in Axis
                if axis is not Axis.HOST_CPU
            }
        )
    assert refused.value.rule is Rule.INVALIDATED
    assert "these conditions state no host_cpu" in str(refused.value)


#: Every axis but the host CPU, which no row depended on before it was one.
BUT_HOST_CPU = (
    "software_stack",
    "torch",
    "atom_src",
    "model",
    "device",
    "engine_config",
)
#: The cells each row recorded before the host-CPU axis, as entries already on
#: disk carry them.
BEFORE_HOST_CPU = {
    "op_graph": ("torch", "atom_src", "model", "engine_config"),
    "price_list": ("software_stack", "torch", "atom_src", "device"),
    "memory_readings": BUT_HOST_CPU,
    "machine_spec_capacity": ("device",),
    "machine_spec_tokenizer_terms": ("model",),
    "step_grid_bundle": BUT_HOST_CPU,
}


@pytest.mark.parametrize(
    "row",
    [Row.OP_GRAPH, Row.MACHINE_SPEC_CAPACITY],
    ids=lambda row: row.name,
)
def test_an_entry_published_before_the_host_cpu_axis_still_loads(tmp_path, row):
    """On any host CPU, since no row it records depends on one."""
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[row])
    path = entry.directory / "entry.json"
    document = json.loads(path.read_bytes())
    document["fingerprints"] = {
        field: {
            "row": field,
            "cells": {
                name: BASE.reading(Axis[name.upper()]).as_json()
                for name in BEFORE_HOST_CPU[field]
            },
        }
        for field in document["fingerprints"]
    }
    path.chmod(0o644)
    path.write_text(json.dumps(document))

    loaded = store.load(entry.key, conditions=moved(BASE, Axis.HOST_CPU), gates=GATES)

    assert set(loaded.fingerprints) == set(rows_for(entry.key.kind))


# --- every gate's state goes into the artifact -----------------------------


def test_a_gate_that_disagrees_with_the_flag_in_force_is_refused(tmp_path):
    """The recorded state is what is checked, never the flag re-read."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(
            KEY_OF[Row.PRICE_LIST],
            conditions=BASE,
            gates=GateState.of(Gate("PRICE_KERNELS", "on", "COMPASS_PRICE_KERNELS")),
        )
    assert refused.value.rule is Rule.GATE_STATE
    assert "`PRICE_KERNELS` was off when this entry was made and is on now" in str(
        refused.value
    )


def test_a_gate_reading_another_variable_is_another_gate(tmp_path):
    """The `PRICE_KERNELS` incident, as the thing an artifact can now refuse.

    The dead gate read `WORLD_SIZE`, which the engine never sets, so it was
    `off` in every worker -- the same state the live gate reports when it is
    genuinely off. Agreement on the state is exactly what made it invisible,
    so agreement on the state is not enough.
    """
    store = ArtifactStore(tmp_path)
    publish(
        store,
        KEY_OF[Row.PRICE_LIST],
        gates=GateState.of(Gate("PRICE_KERNELS", "off", "WORLD_SIZE")),
    )
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.PRICE_LIST], conditions=BASE, gates=GATES)
    assert refused.value.rule is Rule.GATE_STATE
    assert "read WORLD_SIZE" in str(refused.value)
    assert "reads COMPASS_PRICE_KERNELS" in str(refused.value)
    assert "both saying off" in str(refused.value)


def test_a_dead_gates_signature_survives_a_state_change_beside_it(tmp_path):
    """Both facts, when both moved -- the louder one does not hide the other.

    A reader told only "it was off and is on now" re-runs under the old flag.
    A reader told "and it read `WORLD_SIZE`" goes and looks at what that gate
    let through, which is the whole reason the source is recorded.
    """
    store = ArtifactStore(tmp_path)
    publish(
        store,
        KEY_OF[Row.PRICE_LIST],
        gates=GateState.of(Gate("PRICE_KERNELS", "off", "WORLD_SIZE")),
    )
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(
            KEY_OF[Row.PRICE_LIST],
            conditions=BASE,
            gates=GateState.of(Gate("PRICE_KERNELS", "on", "COMPASS_PRICE_KERNELS")),
        )
    assert refused.value.rule is Rule.GATE_STATE
    assert "was off when this entry was made and is on now" in str(refused.value)
    assert "read WORLD_SIZE when this entry was made" in str(refused.value)
    assert "reads COMPASS_PRICE_KERNELS now" in str(refused.value)
    assert "both saying" not in str(refused.value)


def test_a_gate_absent_from_one_side_is_named(tmp_path):
    """A gate that shaped the entry and is gone, and one that arrived since."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    with pytest.raises(ArtifactRefusal) as gone:
        store.load(KEY_OF[Row.PRICE_LIST], conditions=BASE, gates=GateState.of())
    assert "`PRICE_KERNELS` shaped this entry and is not in force now" in str(
        gone.value
    )

    publish(store, KEY_OF[Row.MEMORY_READINGS], gates=GateState.of())
    with pytest.raises(ArtifactRefusal) as arrived:
        store.load(KEY_OF[Row.MEMORY_READINGS], conditions=BASE, gates=GATES)
    assert "`PRICE_KERNELS` is in force now and did not shape this entry" in str(
        arrived.value
    )


def test_a_gate_that_cannot_say_what_it_read_is_refused():
    """A gate with no source is the dead gate before anyone has noticed."""
    with pytest.raises(ArtifactRefusal) as refused:
        Gate("PRICE_KERNELS", "off", "")
    assert refused.value.rule is Rule.GATE_STATE
    assert "resolved_from" in str(refused.value)


def test_the_explicit_flag_does_not_reach_the_gate_state(tmp_path):
    """The warning escape is the matrix's; a gate disagreement has none."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(
            KEY_OF[Row.PRICE_LIST],
            conditions=BASE,
            gates=GateState.of(),
            on_mismatch=OnMismatch.WARN,
        )
    assert refused.value.rule is Rule.GATE_STATE


# --- resolution names its answer -------------------------------------------


def test_a_step_records_which_artifact_answered_it(tmp_path):
    """By key and by digest, so "which artifact answered" has an answer."""
    store = ArtifactStore(tmp_path)
    published = publish(store, KEY_OF[Row.PRICE_LIST])
    step = Resolution("step 41")
    entry = store.answer(KEY_OF[Row.PRICE_LIST], step, conditions=BASE, gates=GATES)
    assert entry.digest == published.digest
    assert step.complete
    assert step.answers[0].key == KEY_OF[Row.PRICE_LIST]
    assert step.answers[0].digest == published.digest
    step.require_complete()


def test_a_refusal_says_which_key_missed_where_a_count_does_not(tmp_path):
    """`incomplete: 1/2` twice, and two different pieces of work.

    A missing `op_graph` means the structure was never traced; a refused
    `price_list` means it was traced and the prices are unavailable. The count
    cannot tell them apart and that cost a day once.
    """
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    publish(store, KEY_OF[Row.OP_GRAPH])

    no_graph = Resolution("step 41")
    store.answer(KEY_OF[Row.PRICE_LIST], no_graph, conditions=BASE, gates=GATES)
    store.answer(
        Key.of(Kind.OP_GRAPH, structure="never-traced", shape="decode B=2"),
        no_graph,
        conditions=BASE,
        gates=GATES,
    )

    no_price = Resolution("step 41")
    store.answer(KEY_OF[Row.OP_GRAPH], no_price, conditions=BASE, gates=GATES)
    store.answer(
        KEY_OF[Row.PRICE_LIST],
        no_price,
        conditions=moved(BASE, Axis.DEVICE),
        gates=GATES,
    )

    counts = [(len(step.answers), len(step.misses)) for step in (no_graph, no_price)]
    assert counts[0] == counts[1] == (1, 1)
    assert no_graph.report() != no_price.report()
    assert "op_graph(structure='never-traced', shape=" in no_graph.report()
    assert "no entry for" in no_graph.report()
    assert no_graph.misses[0].rule is Rule.RESOLUTION
    assert "price_list" in no_price.report()
    assert no_price.misses[0].rule is Rule.INVALIDATED
    assert "`price_list` x `device`" in no_price.report()


def test_a_step_names_every_key_that_missed_not_the_first(tmp_path):
    """The ledger collects so the refusal can name all of them at once."""
    store = ArtifactStore(tmp_path)
    step = Resolution("step 7")
    for row in (Row.PRICE_LIST, Row.MEMORY_READINGS):
        assert store.answer(KEY_OF[row], step, conditions=BASE, gates=GATES) is None
    assert not step.complete
    with pytest.raises(ArtifactRefusal) as refused:
        step.require_complete()
    assert refused.value.rule is Rule.RESOLUTION
    assert "price_list(" in str(refused.value)
    assert "memory_readings(" in str(refused.value)
    assert "0 of 2 artifacts" in str(refused.value)


def test_a_report_never_states_a_count_without_its_decomposition(tmp_path):
    """No count without the names behind it, where `incomplete: N/2570` was."""
    store = ArtifactStore(tmp_path)
    published = publish(store, KEY_OF[Row.PRICE_LIST])
    step = Resolution("step 41")
    store.answer(KEY_OF[Row.PRICE_LIST], step, conditions=BASE, gates=GATES)
    store.answer(KEY_OF[Row.MEMORY_READINGS], step, conditions=BASE, gates=GATES)
    lines = step.report().splitlines()
    assert lines[0] == "step 41: 1 of 2 answered"
    assert len(lines) == 3
    assert published.digest in step.report()


# --- what the entry states about itself ------------------------------------


def test_the_fingerprint_and_the_gate_state_are_in_the_entry(tmp_path):
    """Recorded by value, inside the document that is digested.

    Not beside it and not by reference: a fingerprint in a file the entry
    points at is a second statement of a fact, which is the shape of every
    incident this package exists for.
    """
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[Row.PRICE_LIST], notes="first campaign")
    document = json.loads((entry.directory / "entry.json").read_bytes())
    recorded = document["fingerprints"]["price_list"]["cells"]
    assert set(recorded) == {axis.field for axis in axes_of(Row.PRICE_LIST)}
    assert recorded["atom_src"] == {"kind": "tree", "value": ATOM_ROOT.revision}
    assert document["gates"] == [
        {
            "name": "PRICE_KERNELS",
            "state": "off",
            "resolved_from": "COMPASS_PRICE_KERNELS",
        }
    ]
    assert store.read(KEY_OF[Row.PRICE_LIST]).gate_state == GATES
    assert store.read(KEY_OF[Row.PRICE_LIST]).fingerprints == entry.fingerprints


def test_a_recorded_row_the_matrix_no_longer_names_is_refused(tmp_path):
    """A row nothing compares is a check that silently stopped.

    The cell-level version of this refuses eleven lines away in
    `Fingerprint.from_json`, and the row is the more likely one to move:
    `ROWS_OF` is one edit, as retiring `region_terms` was, and the schema
    version does not change when the matrix does.
    """
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[Row.MACHINE_SPEC_CAPACITY])
    document = json.loads((entry.directory / "entry.json").read_bytes())
    document["fingerprints"]["region_terms"] = fingerprint(
        Row.MEMORY_READINGS, BASE
    ).as_json()
    (entry.directory / "entry.json").chmod(0o644)
    (entry.directory / "entry.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(entry.key)
    assert refused.value.rule is Rule.INVALIDATED
    assert "records a fingerprint for `region_terms`" in str(refused.value)
    assert "silently stopped" in str(refused.value)


@pytest.mark.parametrize("given", [None, object(), [], {}, True])
def test_an_axis_handed_something_that_is_not_a_reading_is_refused(given):
    """`str()` is a fallback, and the two ends of it fail differently.

    An object records a heap address, which moves between processes and then
    refuses a change nobody made. `None` records the word `None`, which is
    indistinguishable from a device called that and certifies clean forever.
    """
    with pytest.raises(ArtifactRefusal) as refused:
        Reading.stated(given)
    assert refused.value.rule is Rule.INVALIDATED
    assert "is not a stated reading" in str(refused.value)

    with pytest.raises(ArtifactRefusal):
        BASE.with_reading(Axis.ENGINE_CONFIG, given)


def test_a_scalar_is_still_a_stated_reading():
    """The positive control, so the refusal above is not firing on everything."""
    assert Reading.stated("MI308X-80CU") == Reading("stated", "MI308X-80CU")
    assert Reading.stated(64).value == "64"
    assert Reading.stated(0.9).value == "0.9"


def test_an_incomparable_reading_does_not_swallow_the_cells_that_compared(tmp_path):
    """Refuse on the one that cannot be asked, and still say what else moved.

    Otherwise the caller re-takes the reading the entry's way, re-runs, and
    discovers a second cell that had already moved when the first refusal was
    written.
    """
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    both = moved(BASE, Axis.DEVICE).with_reading(
        Axis.ATOM_SRC, Reading("commit", "065f34f06bc7de314b37ec85b549feea104aa57e")
    )
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.PRICE_LIST], conditions=both, gates=GATES)
    assert refused.value.rule is Rule.NOT_COMPARABLE
    assert "recorded a tree and this run reads a commit" in str(refused.value)
    assert "`price_list` x `device`" in str(refused.value)


def test_a_publish_that_states_no_conditions_is_refused(tmp_path):
    """Required at publish, because a load is hours too late to find out."""
    store = ArtifactStore(tmp_path)
    with pytest.raises(TypeError):
        store.publish(
            KEY_OF[Row.PRICE_LIST],
            provenance=STANZA,
            topology=TP2,
            gates=GATES,
            members={member_name("rows", TP2.ranks()[0], "json"): b"{}"},
        )
    with pytest.raises(ArtifactRefusal) as refused:
        store.publish(
            KEY_OF[Row.PRICE_LIST],
            provenance=STANZA,
            topology=TP2,
            conditions="rocm7.2.4",
            gates=GATES,
            members={member_name("rows", TP2.ranks()[0], "json"): b"{}"},
        )
    assert refused.value.rule is Rule.INVALIDATED
    assert "is not a set of conditions" in str(refused.value)


# --- #169 part 1: a leak on the way to the rename --------------------------


def test_notes_that_are_not_text_are_refused_and_leave_nothing_behind(tmp_path):
    """Measured and filed as #169: `TypeError`, and litter.

    `notes` was never checked, so a non-string reached `json.dumps` and came
    back as a bare `TypeError` -- an unnamed exception where a named refusal
    belongs -- and the `except OSError` did not catch it, so the staging
    directory survived a publish that never happened. Both halves are pinned:
    the refusal is named, and the kind directory holds nothing afterwards.
    """
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactRefusal) as refused:
        publish(store, KEY_OF[Row.PRICE_LIST], notes=object())
    assert refused.value.rule is Rule.RESOLUTION
    assert "`notes` holds object, not text" in str(refused.value)
    assert not list(tmp_path.rglob("*")) or not [
        item for item in tmp_path.rglob(".*") if item.is_dir()
    ]
    publish(store, KEY_OF[Row.PRICE_LIST], notes="the publish that did happen")


def test_a_publish_that_fails_on_the_way_to_the_rename_leaves_nothing_behind(tmp_path):
    """#169's other half, reached directly because the first half hides it.

    Validating `notes` makes the measured `TypeError` unreachable through
    `publish`, so a test driven only through the public surface would leave
    the widened `except` as a claim nothing bites on. The staging directory is
    what is under test, so the test goes to where it is made.
    """
    store = ArtifactStore(tmp_path)
    destination = store.directory_for(KEY_OF[Row.PRICE_LIST])
    with pytest.raises(ArtifactRefusal) as refused:
        store._write(destination, {"notes": object()}, {})
    assert refused.value.rule is Rule.IMMUTABLE
    assert "TypeError" in str(refused.value)
    assert not destination.exists()
    assert list(destination.parent.iterdir()) == []


# --- refusals the package states, each reached by a caller ------------------


def test_a_recorded_cell_the_matrix_has_no_column_for_is_refused(tmp_path):
    """The cell-level companion of the row refusal above.

    Without it the read succeeds and the cell is dropped: the entry answers
    with a fingerprint narrower than the one it recorded, and the axis that
    was recorded is never compared again.
    """
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[Row.PRICE_LIST])
    document = json.loads((entry.directory / "entry.json").read_bytes())
    document["fingerprints"]["price_list"]["cells"]["compiler"] = Reading.stated(
        "hipcc 6.2"
    ).as_json()
    (entry.directory / "entry.json").chmod(0o644)
    (entry.directory / "entry.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(entry.key)
    assert refused.value.rule is Rule.INVALIDATED
    assert "records `compiler`, which the invalidation matrix has no column for" in str(
        refused.value
    )


def test_a_publish_that_omits_the_gates_is_a_type_error(tmp_path):
    """No default: an entry that states no gates cannot be checked on load."""
    store = ArtifactStore(tmp_path)
    with pytest.raises(TypeError):
        store.publish(
            KEY_OF[Row.PRICE_LIST],
            provenance=STANZA,
            topology=TP2,
            conditions=BASE,
            members={member_name("rows", rank, "json"): b"{}" for rank in TP2.ranks()},
        )
    assert not list(tmp_path.iterdir())


def test_a_publish_handed_something_that_is_not_a_gate_state_is_refused(tmp_path):
    """A named refusal at publish, not an `AttributeError` from the JSON writer."""
    store = ArtifactStore(tmp_path)
    with pytest.raises(ArtifactRefusal) as refused:
        publish(store, KEY_OF[Row.PRICE_LIST], gates="off")
    assert refused.value.rule is Rule.GATE_STATE
    assert "'off' is not a gate state" in str(refused.value)


def test_a_miss_where_the_gate_and_the_device_both_moved_is_filed_as_the_gate(
    tmp_path,
):
    """The gate is checked first, so the ledger names the gate as what declined.

    The control shows the device move refuses on its own; without it, a
    fingerprint that happened not to move would make the order unobservable.
    """
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    gate_on = GateState.of(Gate("PRICE_KERNELS", "on", "COMPASS_PRICE_KERNELS"))
    elsewhere = moved(BASE, Axis.DEVICE)

    device_only = Resolution("step 41")
    store.answer(KEY_OF[Row.PRICE_LIST], device_only, conditions=elsewhere, gates=GATES)
    assert [miss.rule for miss in device_only.misses] == [Rule.INVALIDATED]

    both = Resolution("step 41")
    store.answer(KEY_OF[Row.PRICE_LIST], both, conditions=elsewhere, gates=gate_on)
    assert [miss.rule for miss in both.misses] == [Rule.GATE_STATE]


def test_a_dead_gate_cannot_hide_behind_a_live_twin_of_its_name(tmp_path):
    """Two recorded rows for one gate name are refused rather than collapsed.

    Collapsed by name, the last row wins: an entry recording the dead
    `WORLD_SIZE` gate beside the live one would load clean under the live
    gate, and the dead one would never be compared.
    """
    store = ArtifactStore(tmp_path)
    entry = publish(store, KEY_OF[Row.PRICE_LIST])
    document = json.loads((entry.directory / "entry.json").read_bytes())
    document["gates"] = [
        Gate("PRICE_KERNELS", "off", "WORLD_SIZE").as_json(),
        PRICE_KERNELS.as_json(),
    ]
    (entry.directory / "entry.json").chmod(0o644)
    (entry.directory / "entry.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.PRICE_LIST], conditions=BASE, gates=GATES)
    assert refused.value.rule is Rule.GATE_STATE
    assert "two gates share a name" in str(refused.value)


def test_two_rows_over_the_same_axes_are_still_not_compared():
    """Two rows with identical columns, so only the row check can refuse.

    Compared cell by cell they would report no difference, which is an answer
    to a question that was never asked.
    """
    assert axes_of(Row.STEP_GRID_BUNDLE) == axes_of(Row.MEMORY_READINGS)
    with pytest.raises(ArtifactRefusal) as refused:
        differences(
            fingerprint(Row.STEP_GRID_BUNDLE, BASE),
            fingerprint(Row.MEMORY_READINGS, BASE),
        )
    assert refused.value.rule is Rule.NOT_COMPARABLE
    assert (
        "a `step grid bundle` fingerprint was compared with a `memory_readings` one"
        in str(refused.value)
    )


def test_one_row_over_different_axes_names_both_sides_axes():
    """A row recorded before the matrix gained a column: both axis lists are named."""
    now = fingerprint(Row.PRICE_LIST, BASE)
    with pytest.raises(ArtifactRefusal) as refused:
        differences(Fingerprint(Row.PRICE_LIST, now.cells[:-1]), now)
    assert (
        "`price_list` was fingerprinted over ROCm / AITER / RCCL, torch, ATOM src, "
        "device, but the matrix now rows it over ROCm / AITER / RCCL, torch, ATOM "
        "src, device, PTL"
    ) in str(refused.value)


def test_a_load_handed_something_that_is_not_a_gate_state_is_refused(tmp_path):
    """The same refusal at load, where the recorded gates are checked."""
    store = ArtifactStore(tmp_path)
    publish(store, KEY_OF[Row.PRICE_LIST])
    with pytest.raises(ArtifactRefusal) as refused:
        store.load(KEY_OF[Row.PRICE_LIST], conditions=BASE, gates="off")
    assert refused.value.rule is Rule.GATE_STATE
    assert "'off' is not a gate state" in str(refused.value)


def test_conditions_that_state_an_axis_the_matrix_has_no_column_for_are_refused():
    """All six stated plus one more: the extra axis is named, not dropped."""
    with pytest.raises(ArtifactRefusal) as refused:
        Conditions.of(
            **{axis.field: BASE.reading(axis) for axis in Axis}, compiler="hipcc 6.2"
        )
    assert refused.value.rule is Rule.INVALIDATED
    assert "these conditions state an unknown compiler" in str(refused.value)
