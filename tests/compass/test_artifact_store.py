# SPDX-License-Identifier: MIT
"""The artifact store: what names an entry, what made it, and what it refuses.

Every test here runs at **width two or more** wherever a width can matter. A
single rank cannot expose either of the two naming incidents as they happened: at
width one the failing code applied no suffix, so one writer and four writers
produced the same file and looked alike. The suffix here is unconditional --
that is this module's departure, and `test_the_bare_name_is_never_produced`
is what holds it -- but the four-ranks-one-file case still needs four ranks to
be visible at all, and every test that could pass at width one by accident is
run at width two as well.

Nothing here touches a driver, a device or a network:
`git` is reached only through an injected runner, so the source-root resolvers
are exercised without the CPU gate depending on the tree it was staged from.
"""

import json
import pathlib
import sys

import pytest

from atom.compass.artifacts import (
    KEY_FIELDS,
    ArtifactRefusal,
    ArtifactStore,
    Conditions,
    Gate,
    GateState,
    Key,
    Kind,
    Provenance,
    RankCoords,
    Reading,
    Rule,
    SourceRoot,
    Topology,
    git_described_root,
    git_tree_root,
    member_name,
    module_root,
    roots_for,
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


#: What every publish here states about the conditions it was made under and
#: the gates that shaped it. A publish requires both and this file is about
#: neither:
#: what they *do* is exercised in `test_artifact_invalidation.py`, and here
#: they are the fixture that lets a publish happen at all.
CONDITIONS = Conditions.of(
    software_stack="rocm7.2.4 / aiter v0.1.21.dev0-49-gf4e7c7509 / rccl2.22.3",
    torch="2.10.0+rocm7.2.4",
    atom_src=Reading.of_source_root(ATOM_ROOT),
    model="Qwen/Qwen3-32B",
    device="MI308X-80CU",
    ptl="disabled",
    host_cpu="Intel(R) Xeon(R) Platinum 8480C",
    engine_config="cudagraph=piecewise,level=3",
)
GATES = GateState.of(Gate("PRICE_KERNELS", "off", "COMPASS_PRICE_KERNELS"))


def stanza(produced_by: str = "compass calibrate phase-1b") -> Provenance:
    """A complete stanza: both executed source roots, ATOM's and aiter's."""
    return Provenance(produced_by, "2026-09-22T11:00:00+00:00", (ATOM_ROOT, AITER_ROOT))


#: The layout a key states when a test names only its TP width.
LAYOUT = {
    "dp": 1,
    "pp": 1,
    "pcp": 1,
    "tp": 1,
    "enable_expert_parallel": False,
    "enable_dp_attention": False,
    "decode_context_parallel_size": 1,
}


def price_key(tp: int = 2, **layout: object) -> Key:
    return Key.of(
        Kind.PRICE_LIST,
        model="Qwen/Qwen3-32B",
        machine_spec="sha256:mi308x",
        source_root=ATOM_ROOT.revision,
        **{**LAYOUT, "tp": tp, **layout},
    )


def graph_key(structure: str = "qwen3-moe-48L") -> Key:
    return Key.of(Kind.OP_GRAPH, structure=structure, shape="decode B=2")


def members_for(topology: Topology, stem: str, extension: str) -> dict[str, bytes]:
    """One file per rank, each naming the rank that wrote it."""
    return {
        member_name(stem, rank, extension): f"{stem}{rank.suffix}".encode()
        for rank in topology.ranks()
    }


# --- a key is a tuple, never a path -----------------------------------------


def test_every_kind_declares_its_key_fields():
    assert set(KEY_FIELDS) == set(Kind)


def test_a_price_list_asked_for_by_path_is_refused_by_name(tmp_path):
    """The three shapes of asking by path, each refused naming the triple."""
    with pytest.raises(ArtifactRefusal) as by_field:
        Key.of(Kind.PRICE_LIST, path="/data/prices.json")
    assert by_field.value.rule is Rule.KEY_IS_A_TUPLE
    assert "(model, dp, pp, pcp, tp, enable_expert_parallel," in str(by_field.value)
    assert "states no width and silently prices nothing" in str(by_field.value)

    with pytest.raises(ArtifactRefusal) as by_value:
        Key.of(
            Kind.PRICE_LIST,
            model="Qwen/Qwen3-32B",
            machine_spec="sha256:mi308x",
            source_root="/measurements/prices.json",
            **LAYOUT,
        )
    assert "which is a path" in str(by_value.value)
    assert "write the value itself there, not the path" in str(by_value.value)

    with pytest.raises(ArtifactRefusal) as by_store:
        ArtifactStore(tmp_path).read("/measurements/prices.json")
    assert "is not a key" in str(by_store.value)


def test_a_kind_named_by_string_is_refused_with_the_members_to_pass():
    with pytest.raises(ArtifactRefusal) as refused:
        Key.of(
            "price_list",
            model="Qwen/Qwen3-32B",
            **LAYOUT,
            source_root=ATOM_ROOT.revision,
        )
    assert refused.value.rule is Rule.KEY_IS_A_TUPLE
    assert "is not a Kind member" in str(refused.value)
    assert "Kind.PRICE_LIST" in str(refused.value)


def test_a_key_that_states_no_layout_is_refused():
    with pytest.raises(ArtifactRefusal) as refused:
        Key.of(Kind.PRICE_LIST, model="Qwen/Qwen3-32B", tp=2, source_root="t")
    assert "states no dp, pp, pcp, enable_expert_parallel" in str(refused.value)


def test_a_model_name_with_a_separator_is_not_a_path():
    """`Qwen/Qwen3-32B` is a value; the separator alone cannot be the test."""
    assert price_key().values["model"] == "Qwen/Qwen3-32B"
    for suspect in ("/abs/prices", "../prices", "prices.jsonl", pathlib.Path("p")):
        with pytest.raises(ArtifactRefusal):
            Key.of(Kind.OP_GRAPH, structure=suspect, shape="decode B=2")


def test_two_keys_never_share_a_directory(tmp_path):
    store = ArtifactStore(tmp_path)
    places = {store.directory_for(price_key(width)) for width in (1, 2, 4, 8)}
    assert len(places) == 4
    assert store.directory_for(price_key(2)).parent.name == "price_list"


@pytest.mark.parametrize(
    "kind,fields",
    [
        (
            Kind.STEP_MEASUREMENT,
            (
                "(model, device, engine_config, source_root, sampling, "
                "step_kind, view)"
            ),
        ),
        (
            Kind.STEP_LINEAR_LAW,
            "(model, machine_spec, engine_config, source_root, form)",
        ),
        (
            Kind.HOST_LAWS,
            "(host_cpu, atom_src, model, engine_config, machine_spec)",
        ),
    ],
)
def test_a_key_short_of_its_last_field_refuses_naming_the_tuple(kind, fields):
    *given, last = KEY_FIELDS[kind]
    with pytest.raises(ArtifactRefusal) as refused:
        Key.of(kind, **{name: "x" for name in given})
    assert refused.value.rule is Rule.KEY_IS_A_TUPLE
    assert f"this {kind} key states no {last}" in str(refused.value)
    assert f"a {kind} is keyed by {fields}" in str(refused.value)


# --- a key from the token a command line names -------------------------------


def publish_price(store: ArtifactStore, width: int) -> Key:
    store.publish(
        price_key(width),
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=Topology(tp=width),
        members=members_for(Topology(tp=width), "prices", "json"),
    )
    return price_key(width)


def test_a_key_resolves_from_the_token_it_prints(tmp_path):
    store = ArtifactStore(tmp_path)
    keys = [publish_price(store, width) for width in (1, 2)]
    for key in keys:
        assert store.key_from_token(Kind.PRICE_LIST, key.token()) == key


@pytest.mark.parametrize("asked", ["unknown", "other kind", "partial"])
def test_a_token_no_entry_of_the_kind_ends_in_refuses_naming_the_kind(tmp_path, asked):
    store = ArtifactStore(tmp_path)
    token = publish_price(store, 1).token()
    kind, given = {
        "unknown": (Kind.PRICE_LIST, "0123456789ab"),
        "other kind": (Kind.OP_GRAPH, token),
        "partial": (Kind.PRICE_LIST, token[:6]),
    }[asked]
    with pytest.raises(ArtifactRefusal) as refused:
        store.key_from_token(kind, given)
    assert refused.value.rule is Rule.RESOLUTION
    assert f"0 {kind} entries in {tmp_path} end in {given!r}" in str(refused.value)


def test_an_ambiguous_token_refuses_naming_the_kind_and_the_entries(tmp_path):
    store = ArtifactStore(tmp_path)
    key = publish_price(store, 1)
    place = store.directory_for(key)
    copy = place.with_name(f"copied-{key.token()}")
    copy.mkdir()
    for item in place.iterdir():
        (copy / item.name).write_bytes(item.read_bytes())
    with pytest.raises(ArtifactRefusal) as refused:
        store.key_from_token(Kind.PRICE_LIST, key.token())
    assert f"2 price_list entries in {tmp_path} end in" in str(refused.value)
    assert copy.name in str(refused.value) and place.name in str(refused.value)


def test_a_directory_renamed_onto_another_token_refuses(tmp_path):
    """The entry restates its key, and that key's token is not the one asked."""
    store = ArtifactStore(tmp_path)
    key = publish_price(store, 1)
    other = price_key(2).token()
    store.directory_for(key).rename(store.directory_for(key).with_name(other))
    with pytest.raises(ArtifactRefusal) as refused:
        store.key_from_token(Kind.PRICE_LIST, other)
    assert refused.value.rule is Rule.RESOLUTION
    assert f"restates {key}, whose token is {key.token()!r}" in str(refused.value)


# --- one naming function ----------------------------------------------------


def test_the_bare_name_is_never_produced():
    """No topology, at any width, names a file `steps.jsonl`."""
    produced = {
        member_name("steps", rank, "jsonl")
        for topology in (Topology(), Topology(tp=2), Topology(dp=2, tp=2))
        for rank in topology.ranks()
    }
    assert "steps.jsonl" not in produced
    assert len(produced) == 1 + 2 + 4
    assert "steps.dp0of1.pp0of1.pcp0of1.tp0of1.jsonl" in produced
    assert "steps.dp0of1.pp0of1.pcp0of1.tp0of2.jsonl" in produced


def test_four_ranks_do_not_resolve_to_one_name():
    """TP=2 x DP=2: the incident needed a coordinate that named one axis."""
    topology = Topology(dp=2, tp=2)
    names = {member_name("graph", rank, "json") for rank in topology.ranks()}
    assert len(names) == 4
    assert "graph.dp1of2.pp0of1.pcp0of1.tp0of2.json" in names
    with pytest.raises(ArtifactRefusal) as refused:
        RankCoords(Topology(tp=2), tp=1, dp=1)
    assert "outside this topology" in str(refused.value)


def test_a_coordinate_taken_in_another_topology_is_refused():
    with pytest.raises(ArtifactRefusal) as refused:
        member_name("graph", RankCoords(Topology(tp=2), tp=3), "json")
    assert "dp1.pp1.pcp1.tp2" in str(refused.value)


def test_the_topology_counts_ranks_without_spending_the_word_width():
    """`rank_count`, not `width`: at `-tp 2 -dp 2` the two are 2 and 4."""
    assert not hasattr(Topology(), "width")
    assert Topology(dp=2, tp=2).rank_count == 4
    assert Topology(dp=2, tp=2).widths == {"dp": 2, "pp": 1, "pcp": 1, "tp": 2}


# --- one entry, round-tripped at width one and at width two -----------------


@pytest.mark.parametrize(
    "topology, key",
    [
        (Topology(), price_key(1)),
        (Topology(tp=2), price_key(2)),
        (Topology(dp=2, tp=2), price_key(2, dp=2)),
    ],
)
def test_an_entry_round_trips_through_the_naming_function(tmp_path, topology, key):
    store = ArtifactStore(tmp_path)
    members = members_for(topology, "steps", "jsonl")
    published = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members,
        notes="first pass",
    )
    entry = store.read(key)
    assert entry.digest == published.digest
    assert entry.topology == topology
    assert len(entry.members) == topology.rank_count
    for rank in topology.ranks():
        assert (
            entry.read_member("steps", rank, "jsonl")
            == members[member_name("steps", rank, "jsonl")]
        )
    assert entry.provenance.root("atom").revision == ATOM_ROOT.revision
    assert entry.provenance.root("aiter").revision == "v0.1.21.dev0-49-gf4e7c7509"


def test_a_read_that_drops_the_rank_coordinates_refuses_by_name(tmp_path):
    """The `steps.jsonl` incident, with the stale neighbour actually present."""
    store = ArtifactStore(tmp_path)
    wide = Topology(tp=2)
    key = price_key(2)
    store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=wide,
        members=members_for(wide, "steps", "jsonl"),
    )
    entry = store.read(key)

    with pytest.raises(ArtifactRefusal) as refused:
        entry.read_member("steps", RankCoords(Topology()), "jsonl")
    assert refused.value.rule is Rule.ONE_NAMING_FUNCTION
    assert "written at dp1.pp1.pcp1.tp2" in str(refused.value)
    assert "asks at dp1.pp1.pcp1.tp1" in str(refused.value)

    stale = entry.directory / "steps.jsonl"
    stale.write_bytes(b"what a width-1 run left behind")
    with pytest.raises(ArtifactRefusal) as neighbour:
        store.read(key)
    assert "`steps.jsonl` was not written by the naming function" in str(
        neighbour.value
    )
    stale.unlink()

    from_width_one = member_name("steps", RankCoords(Topology()), "jsonl")
    (entry.directory / from_width_one).write_bytes(b"rank 0 of a width-1 run")
    with pytest.raises(ArtifactRefusal) as wrong_run:
        store.read(key)
    assert "was written at dp1.pp1.pcp1.tp1" in str(wrong_run.value)
    assert "published at dp1.pp1.pcp1.tp2" in str(wrong_run.value)


def test_one_rank_writing_for_four_is_refused_at_hand_off(tmp_path):
    """The survivor that looked complete at 807 operators, refused instead."""
    store = ArtifactStore(tmp_path)
    topology = Topology(dp=2, tp=2)
    only = RankCoords(topology)
    with pytest.raises(ArtifactRefusal) as refused:
        store.publish(
            graph_key("qwen3-moe-48L-tp2dp2"),
            provenance=stanza(),
            conditions=CONDITIONS,
            gates=GATES,
            topology=topology,
            members={member_name("graph", only, "json"): b"807 operators"},
        )
    assert refused.value.rule is Rule.EVERY_RANK_WRITES
    assert "covers 1 of 4 ranks" in str(refused.value)
    assert "graph.dp1of2.pp0of1.pcp0of1.tp1of2.json" in str(refused.value)


# --- a handed-off entry is immutable ----------------------------------------


def test_a_notes_only_rewrite_of_a_handed_off_entry_is_refused(tmp_path):
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    members = members_for(topology, "prices", "json")
    first = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members,
        notes="case (b) aggregate",
    )

    with pytest.raises(ArtifactRefusal) as refused:
        store.publish(
            key,
            provenance=stanza(),
            conditions=CONDITIONS,
            gates=GATES,
            topology=topology,
            members=dict(members),
            notes="case (b) aggregate, reuse note corrected",
        )
    assert refused.value.rule is Rule.IMMUTABLE
    assert first.digest in str(refused.value)
    assert "the same members" in str(refused.value)
    assert store.read(key).digest == first.digest
    assert store.read(key).notes == "case (b) aggregate"


def test_the_notes_are_inside_the_digest(tmp_path):
    """Two entries with identical members and different notes differ by digest."""
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    members = members_for(topology, "prices", "json")
    digests = set()
    for index, note in enumerate(("as measured", "as measured, note corrected")):
        digests.add(
            store.publish(
                graph_key(f"qwen3-moe-48L-v{index}"),
                provenance=stanza(),
                conditions=CONDITIONS,
                gates=GATES,
                topology=topology,
                members=members,
                notes=note,
            ).digest
        )
    assert len(digests) == 2


def test_a_member_changed_after_hand_off_is_refused(tmp_path):
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    entry = store.read(key)
    name = member_name("prices", topology.ranks()[0], "json")
    assert (entry.directory / name).stat().st_mode & 0o222 == 0
    assert (entry.directory / "entry.json").stat().st_mode & 0o222 == 0
    (entry.directory / name).chmod(0o644)
    (entry.directory / name).write_bytes(b"re-run in place")
    with pytest.raises(ArtifactRefusal) as refused:
        entry.read_member("prices", topology.ranks()[0], "json")
    assert refused.value.rule is Rule.IMMUTABLE
    assert entry.members[name] in str(refused.value)


def test_an_entry_moved_by_hand_is_found_out(tmp_path):
    """The key lives in the entry, so the path is a place and not the authority."""
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    mine, other = price_key(2), price_key(4)
    store.publish(
        mine,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    moved = store.directory_for(other)
    moved.parent.mkdir(parents=True, exist_ok=True)
    store.directory_for(mine).rename(moved)
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(other)
    assert refused.value.rule is Rule.RESOLUTION
    assert "tp=2" in str(refused.value)


def test_a_missing_entry_names_the_key_that_missed(tmp_path):
    with pytest.raises(ArtifactRefusal) as refused:
        ArtifactStore(tmp_path).read(price_key(8))
    assert "no entry for price_list(" in str(refused.value)
    assert "tp=8" in str(refused.value)


def publish_prices(store: ArtifactStore, key: Key, topology: Topology):
    return store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )


def test_each_layout_is_its_own_entry_and_a_miss_names_its_field(tmp_path):
    """TP2 and TP2 x DP2 publish apart; a read at a layout nobody published
    names the field in which the nearest entry differs."""
    store = ArtifactStore(tmp_path)
    tp2 = publish_prices(store, price_key(2), Topology(tp=2))
    dp2 = publish_prices(store, price_key(2, dp=2), Topology(dp=2, tp=2))
    assert tp2.directory != dp2.directory
    assert store.read(price_key(2, dp=2)).topology == Topology(dp=2, tp=2)

    with pytest.raises(ArtifactRefusal) as missed:
        store.read(price_key(2, enable_expert_parallel=True))
    assert "the nearest entry differs in enable_expert_parallel (False there)" in str(
        missed.value
    )


@pytest.mark.parametrize(
    "field, value",
    [("tp", True), ("tp", 0), ("tp", "2"), ("enable_dp_attention", 1)],
)
def test_a_layout_field_takes_only_its_own_type(field, value):
    """A width is a whole number of at least one and a switch a bool, so `True`
    and `1` can never name one layout under two directories."""
    with pytest.raises(ArtifactRefusal) as refused:
        price_key(**{field: value})
    assert f"`price_list.{field}` holds {value!r}" in str(refused.value)


@pytest.mark.parametrize(
    "damage, expected",
    [
        (lambda doc: doc.pop("provenance"), "KeyError"),
        (lambda doc: doc.update(kind="not_a_kind"), "ValueError"),
        (lambda doc: doc.update(topology={"tp": 2}), "does not state every axis"),
    ],
)
def test_a_hand_edited_entry_is_refused_by_name(tmp_path, damage, expected):
    """A tampered document is declined, not raised as a bare traceback.

    The module's argument for the directory convention is that the path is a
    place and never the authority. That is only true if a document this store
    does not recognise is refused by name; before this, three separate edits
    raised `KeyError`, `JSONDecodeError` and `ValueError` and none of them
    named an artifact.
    """
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    entry = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    document = json.loads((entry.directory / "entry.json").read_bytes())
    damage(document)
    (entry.directory / "entry.json").chmod(0o644)
    (entry.directory / "entry.json").write_text(json.dumps(document))
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(key)
    assert expected in str(refused.value)


def test_a_truncated_entry_is_refused_by_name(tmp_path):
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    entry = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    (entry.directory / "entry.json").chmod(0o644)
    (entry.directory / "entry.json").write_text("{not json")
    with pytest.raises(ArtifactRefusal) as refused:
        store.read(key)
    assert refused.value.rule is Rule.RESOLUTION
    assert "is not an entry document" in str(refused.value)


def test_an_empty_directory_in_the_way_is_not_silently_replaced(tmp_path):
    """`os.rename` replaces an empty directory; the occupancy check does not.

    Measured on ext4: `rename(2)` onto an existing *empty* directory succeeds.
    So the exclusivity cannot come from the rename, and this is the check it
    comes from instead — refused under immutability, naming what is in the way,
    rather than under resolution saying nothing is published there.
    """
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    store.directory_for(key).mkdir(parents=True)
    with pytest.raises(ArtifactRefusal) as refused:
        store.publish(
            key,
            provenance=stanza(),
            conditions=CONDITIONS,
            gates=GATES,
            topology=topology,
            members=members_for(topology, "prices", "json"),
        )
    assert refused.value.rule is Rule.IMMUTABLE
    assert "is in the way of" in str(refused.value)
    assert "nothing is published at" not in str(refused.value)


def test_an_abandoned_staging_directory_does_not_block_a_publish(tmp_path):
    """Staging is per-publisher, so a crashed one is not in anybody's way.

    The abandoned directory is named `.<entry>.publishing` on purpose: that is
    the name the pre-`mkdtemp` code derived from the key, so a revert has
    something to find and `shutil.rmtree` away. An earlier version of this test
    called it `.crashed`, which no version of the code ever touched -- it
    passed with the fix and passed reverted, and proved nothing.

    The directory must come out with its contents as they were, not merely
    still standing: a publish that empties it and leaves the husk has reached
    into another publisher's staging just the same, and `is_dir()` alone
    stays true when that happens.
    """
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    destination = store.directory_for(key)
    abandoned = destination.parent / f".{destination.name}.publishing"
    abandoned.mkdir(parents=True)
    half_built = abandoned / "prices.dp0of1.pp0of1.pcp0of1.tp0of2.json"
    half_built.write_bytes(b"half")
    entry = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    assert {item.name for item in entry.directory.iterdir()} == {
        "entry.json",
        *entry.members,
    }
    assert abandoned.is_dir()
    assert {item.name: item.read_bytes() for item in abandoned.iterdir()} == {
        half_built.name: b"half"
    }


# --- the physical form is a directory convention ----------------------------


def test_the_store_is_a_directory_convention_with_no_index(tmp_path):
    store = ArtifactStore(tmp_path)
    topology = Topology(tp=2)
    key = price_key(2)
    entry = store.publish(
        key,
        provenance=stanza(),
        conditions=CONDITIONS,
        gates=GATES,
        topology=topology,
        members=members_for(topology, "prices", "json"),
    )
    assert entry.directory == store.directory_for(key)
    assert {item.name for item in tmp_path.iterdir()} == {"price_list"}
    assert not [item for item in tmp_path.rglob("*index*")]
    assert not [item for item in tmp_path.rglob("*manifest*")]
    assert {item.name for item in entry.directory.iterdir()} == {
        "entry.json",
        *entry.members,
    }
    document = json.loads((entry.directory / "entry.json").read_bytes())
    assert document["key"] == key.values
    assert document["topology"] == {"dp": 1, "pp": 1, "pcp": 1, "tp": 2}


# --- provenance names every executed source root ----------------------------


def test_a_stanza_without_aiter_is_refused():
    with pytest.raises(ArtifactRefusal) as refused:
        Provenance("phase 1b", "2026-09-22T11:00:00+00:00", (ATOM_ROOT,))
    assert refused.value.rule is Rule.PROVENANCE
    assert "names no aiter source root" in str(refused.value)
    assert "Add a row for each missing root" in str(refused.value)


def test_a_stanza_without_an_offset_is_refused():
    with pytest.raises(ArtifactRefusal) as refused:
        Provenance("phase 1b", "2026-09-22T11:00:00", (ATOM_ROOT, AITER_ROOT))
    assert "states no offset" in str(refused.value)


def test_a_revision_says_which_object_it_is():
    """A tree and a commit in one field compare unequal for identical source."""
    with pytest.raises(ArtifactRefusal) as refused:
        SourceRoot("atom", "/t", "deadbeef", "sha", "somehow")
    assert "revision_kind" in str(refused.value)
    assert "tree, commit, describe" in str(refused.value)


def test_atoms_root_is_the_tree_git_archive_ships():
    calls = []

    def run(argv):
        calls.append(argv)
        if "rev-parse" in argv:
            return 0, "9091c1dc8" * 4, ""
        return 0, " M atom/compass/artifacts/store.py", ""

    root = git_tree_root("atom", "/workspace/ATOM/atom", run)
    assert calls[0][-1] == "HEAD^{tree}"
    assert root.revision_kind == "tree"
    assert root.dirty == 1
    assert "git archive ships" in root.method


def test_a_snapshot_with_no_git_falls_back_to_the_stamp(tmp_path):
    """The stamp answers with a commit, and cannot know whether it was dirty."""
    (tmp_path / ".compass-commit").write_text("92f1fdafe\n")
    nested = tmp_path / "atom" / "compass"
    nested.mkdir(parents=True)
    root = git_tree_root(
        "atom", str(nested), lambda argv: (128, "", "not a repository")
    )
    assert root.revision == "92f1fdafe"
    assert root.revision_kind == "commit"
    assert root.dirty is None
    assert ".compass-commit" in root.method


def test_a_tree_that_is_neither_is_refused(tmp_path):
    assert not any(
        (place / ".compass-commit").is_file() for place in (tmp_path, *tmp_path.parents)
    ), "an ancestor of the test directory carries a stamp; this box is not clean"
    with pytest.raises(ArtifactRefusal) as refused:
        git_tree_root("atom", str(tmp_path), lambda argv: (128, "", "not a repository"))
    assert "an rsync of a tree" in str(refused.value)


def test_aiters_version_is_gate_gpus_call():
    calls = []

    def run(argv):
        calls.append(argv)
        return 0, "v0.1.20-103-g23f83724f", ""

    root = git_described_root("aiter", "/app/aiter-test/aiter", run)
    assert calls[0][3:] == ["describe", "--tags", "--always", "--dirty"]
    assert root.revision == "v0.1.20-103-g23f83724f"
    assert root.revision_kind == "describe"
    assert root.dirty == 0


def test_an_unreadable_aiter_version_is_a_refusal_not_a_blank():
    with pytest.raises(ArtifactRefusal) as refused:
        git_described_root("aiter", "/app/aiter-test/aiter", lambda a: (128, "", "no"))
    assert "an unreadable version is a mismatch, not a match" in str(refused.value)


def test_a_source_root_is_located_without_importing_it():
    """Resolving aiter must not import aiter: that import hangs on a wedged node."""
    candidates = ["mailbox", "netrc", "imaplib", "colorsys", "wave", "shelve"]
    unimported = [name for name in candidates if name not in sys.modules]
    assert unimported, "every candidate was already imported; pick another"
    name = unimported[0]
    assert module_root(name)
    assert name not in sys.modules


def test_a_dotted_module_does_not_import_its_parents():
    """`find_spec` alone is not enough: it executes the parents of a dotted name.

    Measured before this was fixed: `find_spec("xml.dom.minidom")` leaves
    `xml` and `xml.dom` in `sys.modules`. Under `aiter.*` that import is the
    `rocminfo` shell-out that hangs on a wedged driver, and `roots_for`
    anticipates dotted names.
    """
    candidates = ["wsgiref.simple_server", "xmlrpc.client", "dbm.dumb", "html.parser"]
    dotted = next(
        (name for name in candidates if name.split(".")[0] not in sys.modules), None
    )
    assert dotted, "every candidate's package was already imported; pick another"
    root = module_root(dotted)
    assert pathlib.Path(root).is_dir()
    for depth in range(1, dotted.count(".") + 2):
        assert ".".join(dotted.split(".")[:depth]) not in sys.modules


def test_a_dotted_name_with_no_such_module_is_refused():
    with pytest.raises(ArtifactRefusal) as refused:
        module_root("json.no_such_submodule")
    assert "has no `no_such_submodule`" in str(refused.value)


def test_roots_for_gives_every_module_that_can_answer_its_own_row():
    """Rule 2's second half, through the function rather than beside it.

    The regions incident is the third row: a module resolving into a different
    tree shows up as a row that disagrees with ATOM's, instead of hiding
    behind it.
    """
    trees = {
        "/workspace/ATOM/atom": "aaaaaaaa",
        "/stale/atom/compass": "bbbbbbbb",
        "/app/aiter-test/aiter": "v0.1.21.dev0-49-gf4e7c7509",
    }
    seen = []

    def run(argv):
        seen.append(argv)
        root = argv[2]
        if "describe" in argv:
            return 0, trees[root], ""
        if "rev-parse" in argv:
            return 0, trees[root], ""
        return 0, "", ""

    roots = roots_for(
        {
            "atom": "atom",
            "atom.compass.regions": "atom.compass.regions",
            "aiter": "aiter",
        },
        run,
        locate={
            "atom": "/workspace/ATOM/atom",
            "atom.compass.regions": "/stale/atom/compass",
            "aiter": "/app/aiter-test/aiter",
        }.__getitem__,
    )
    by_name = {root.name: root for root in roots}
    assert set(by_name) == {"atom", "atom.compass.regions", "aiter"}
    assert by_name["atom"].revision_kind == "tree"
    assert by_name["aiter"].revision_kind == "describe"
    assert by_name["atom.compass.regions"].revision != by_name["atom"].revision
    assert ["describe" in argv for argv in seen].count(True) == 1
    Provenance("phase 1b", "2026-09-22T11:00:00+00:00", roots)


def test_two_source_roots_may_not_share_a_name():
    with pytest.raises(ArtifactRefusal) as refused:
        Provenance(
            "phase 1b",
            "2026-09-22T11:00:00+00:00",
            (ATOM_ROOT, ATOM_ROOT, AITER_ROOT),
        )
    assert "share a name" in str(refused.value)
