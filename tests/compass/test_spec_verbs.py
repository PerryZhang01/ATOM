# SPDX-License-Identifier: MIT
"""The verbs over the spec: combining fragments, checking one, reading one.

The fragments below are the ones the probes of each hardware probe class would emit --
a tokenizer and an interprocess measurement that need no device, a single-card
engine start, and a fabric measurement -- so a test that combines them combines
the same partial documents a real run would, and a test that breaks one breaks
exactly what it says it does.

The numbers are the measured ones wherever a measurement exists, because two of
these checks are judgements about size: the cross-rank spread accepted is the
one the hardware really shows, and the spread refused is the one a neighbour
really produced. Round numbers would make both tests vacuous.

The probe that needs no device has a section of its own. Its fragments are
built by the probe itself against a stand-in for the per-processor topology the
kernel publishes, so the core counts in them are read by the code a real run
reads them with rather than typed in beside the assertion that checks them.
"""

import copy
import importlib.util
import json
import pathlib

import pytest

from atom.compass import spec as spec_package
from atom.compass.spec import (
    Fragment,
    MachineSpec,
    SpecRefusal,
    across_ranks,
    explain,
    merge,
    tokenizer_fragment,
    tokenizer_sweep,
    validate,
)
from atom.compass.spec.fields import BY_PATH, Kind
from atom.compass.spec.probes import cpu_counts
from atom.compass.spec.rules import Rule
from atom.compass.spec.tokenizers import ENTRY_FIELDS

MACHINE = "mi355x-8gpu-2node"
#: A tokenizer entry's fields by name, to say which of its rates is a peak.
ENTRY_RATES = {field.path: field for field in ENTRY_FIELDS}
STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3", "torch": "2.10.0"}
MIB = 1024**2

TOKENIZER = {
    "id": "qwen3-151k-bpe",
    "backend": "fast",
    "vocab_size": 151936,
    "fingerprint": "sha256:" + "a" * 64,
    "applies_to": ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"],
    "encode_fixed_s": 3.0e-4,
    "encode_tokens_per_s": 2.0e6,
    "decode_fixed_s": 1.5e-4,
    "decode_tokens_per_s": 3.0e6,
    "derate": 0.85,
}

#: Probe class 0, no device: the tokenizer sweep and the interprocess round trips.
CLASS0 = {
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "http_intake": {"fixed_s": 2.0e-3, "per_byte_s": 2.0e-8},
    }
}

#: Probe class 1, one card: the device, its capacity, rates and stack.
CLASS1 = {
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {
            "bf16_flops": 2.5e15,
            "fp8_flops": 5.0e15,
            "derate": {"bf16": 0.70, "fp8": 0.70},
        },
        "software_pinned_to": dict(STACK),
    }
}

#: Probe classes 2 and 3: collectives in real groups, then the same across two nodes.
LINKS = {
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "domain_devices": 8,
            "link_bandwidth_bytes_per_s": 1.0e12,
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
        "router_relay_s": 1.5e-3,
    }
}


def fragment(
    source,
    body=None,
    *,
    machine=MACHINE,
    origin="probed",
    authored_by="a person",
    date="2026-09-18",
    **stanza,
):
    """One probe's output: what it measured, and where it was measured."""
    document = {
        "schema_version": 1,
        "name": machine,
        "provenance": dict(
            {"authored_by": authored_by, "date": date, "origin": origin}, **stanza
        ),
    }
    document.update(copy.deepcopy(body or {}))
    return Fragment.from_mapping(document, source)


def fragments(**edits):
    """The fragments a full spec is built from, with one body replaced."""
    bodies = {"class0": CLASS0, "class1": CLASS1, "links": LINKS}
    bodies.update(edits)
    return [fragment(source, body) for source, body in bodies.items()]


def merged(**edits):
    return merge(fragments(**edits))


def resolved(**edits):
    return MachineSpec.from_mapping(merged(**edits).document)


def without(body, *path):
    """A copy of a fragment body with one dotted field taken out of it."""
    fresh = copy.deepcopy(body)
    node = fresh
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return fresh


# --- merge: the conflict check, which is the point of the verb ----------------


def test_two_hosts_in_one_spec_are_refused_and_both_provenances_are_named():
    # Two tokenizers measured on two machines contradict nothing in their
    # shape: different identities, different entries, no field of one
    # overlapping a field of the other. Only the stanzas differ, so only the
    # stanzas can catch it.
    here = fragment("tokenizer-a", CLASS0, machine="node-18", authored_by="ana")
    elsewhere = fragment(
        "tokenizer-b",
        {"host": {"tokenizers": [dict(TOKENIZER, id="llama-128k-bpe")]}},
        machine="node-22",
        authored_by="bo",
        date="2026-09-19",
    )
    with pytest.raises(SpecRefusal) as refused:
        merge([here, elsewhere])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    for named in ("tokenizer-a", "tokenizer-b", "node-18", "node-22", "ana", "bo"):
        assert named in message, f"{named} is not named in: {message}"


def test_the_refusal_says_the_fragments_are_authored_for_different_machines():
    # `Fragment.machine` is the `name` field, which is the machine the spec is
    # being authored for. No field records where a probe ran, so the refusal
    # says what the fragments claim and not where their numbers came from.
    with pytest.raises(SpecRefusal) as refused:
        merge(
            [fragment("a", CLASS0, machine="node-18"), fragment("b", machine="node-22")]
        )
    message = str(refused.value)
    assert refused.value.what.endswith(
        "are authored for different machines, 'node-18' and 'node-22'"
    )
    assert "measured on" not in message


def test_the_same_two_tokenizers_merge_when_the_machine_agrees():
    # The control for
    # `test_two_hosts_in_one_spec_are_refused_and_both_provenances_are_named`:
    # the same two measurements, the same two authors and dates, differing only
    # in the machine they name.
    here = fragment("tokenizer-a", CLASS0, machine="node-18", authored_by="ana")
    also_here = fragment(
        "tokenizer-b",
        {"host": {"tokenizers": [dict(TOKENIZER, id="llama-128k-bpe")]}},
        machine="node-18",
        authored_by="bo",
        date="2026-09-19",
    )
    combination = merge([here, also_here])
    measured = [entry["id"] for entry in combination.document["host"]["tokenizers"]]
    assert measured == ["qwen3-151k-bpe", "llama-128k-bpe"]
    assert combination.document["name"] == "node-18"
    assert combination.document["provenance"]["fragments"] == [
        "tokenizer-a",
        "tokenizer-b",
    ]


def test_a_merged_document_merged_again_keeps_the_names_that_built_it():
    # Incremental authoring: merge what has been measured, save it, merge next
    # week's probe into the saved document. A merged document is a legal
    # fragment, so this works -- and the block whose whole job is to say where
    # the numbers came from must not lose the fragments of the first round.
    first = merge([fragment("class0", CLASS0), fragment("class1", CLASS1)])
    again = merge(
        [
            Fragment.from_mapping(first.document, "machine.yaml"),
            fragment("links", LINKS),
        ]
    )
    assert again.document["provenance"]["fragments"] == [
        "class0",
        "class1",
        "machine.yaml",
        "links",
    ]


def test_one_machine_named_twice_with_two_answers_is_refused():
    other_cpu = {
        "host": dict(CLASS0["host"], cpu={"cores_physical": 128, "cores_logical": 256})
    }
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", CLASS0), fragment("second", other_cpu)])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.cpu.cores_physical" in message
    assert "96" in message and "128" in message
    assert "first" in message and "second" in message


def test_two_fragments_that_agree_are_not_a_conflict():
    combination = merge([fragment("first", CLASS0), fragment("second", CLASS0)])
    assert combination.sources["host.cpu.cores_physical"] == ("first", "second")


def test_two_fragments_pinned_to_different_stacks_are_refused_as_a_stack_conflict():
    moved = copy.deepcopy(CLASS1)
    moved["device"]["software_pinned_to"]["rocm"] = "7.3.0"
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("class1", CLASS1), fragment("class1b", moved)])
    assert refused.value.rule is Rule.PINNED_STACK
    assert "fragments pinned to different stacks" in refused.value.remedy


def test_the_merged_document_reads_as_a_spec_and_echoes_back():
    spec = resolved()
    assert MachineSpec.from_mapping(spec.echo()).digest() == spec.digest()


def test_the_merged_provenance_names_every_fragment_and_its_origin():
    combination = merge(
        [
            fragment("class0", CLASS0),
            fragment("class1", CLASS1, origin="datasheet", date="2026-09-20"),
            fragment("links", LINKS, authored_by="another person"),
        ]
    )
    provenance = combination.document["provenance"]
    assert provenance["fragments"] == ["class0", "class1", "links"]
    assert provenance["origin"] == "mixed"
    assert provenance["date"] == "2026-09-20"
    assert provenance["authored_by"] == "a person, another person"


def test_one_origin_is_kept_when_every_fragment_used_it():
    assert merged().document["provenance"]["origin"] == "probed"


@pytest.mark.parametrize(
    "path",
    [
        "schema_version",
        "name",
        "provenance.authored_by",
        "provenance.date",
        "provenance.origin",
    ],
)
def test_a_fragment_that_does_not_state_where_it_came_from_is_refused(path):
    document = {
        "schema_version": 1,
        "name": MACHINE,
        "provenance": {
            "authored_by": "a person",
            "date": "2026-09-18",
            "origin": "probed",
        },
    }
    document.update(copy.deepcopy(CLASS0))
    *blocks, leaf = path.split(".")
    node = document
    for block in blocks:
        node = node[block]
    del node[leaf]
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(document, "nameless")
    assert refused.value.rule is Rule.ONE_MACHINE
    assert path in str(refused.value)


def test_merging_nothing_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        merge([])
    assert refused.value.rule is Rule.ONE_MACHINE


def test_a_fragment_is_read_against_the_same_closed_schema():
    with pytest.raises(SpecRefusal) as refused:
        fragment("deployment", {"device": {"tensor_parallel_size": 8}})
    assert refused.value.rule is Rule.SEPARATION
    assert "EngineArgs.tensor_parallel_size" in str(refused.value)


def test_a_fragment_that_is_not_a_mapping_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        Fragment.from_mapping(["schema_version", 1], "a list")
    assert refused.value.rule is Rule.SHAPE


# --- merge: the tokenizer table, keyed on the identity a person chose ---------


def test_one_id_over_two_files_is_refused_with_both_fingerprints():
    revised = {
        "host": {"tokenizers": [dict(TOKENIZER, fingerprint="sha256:" + "b" * 64)]}
    }
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", CLASS0), fragment("second", revised)])
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY
    what = refused.value.what
    assert what.startswith(f"'qwen3-151k-bpe' is 'sha256:{'a' * 64}' in 'first' (")
    assert f") and 'sha256:{'b' * 64}' in 'second' (" in what


def test_one_tokenizer_measured_on_both_backends_is_not_a_conflict():
    slow = {
        "host": {
            "tokenizers": [dict(TOKENIZER, backend="slow", encode_tokens_per_s=1.6e5)]
        }
    }
    combination = merge([fragment("fast", CLASS0), fragment("slow", slow)])
    backends = [
        entry["backend"] for entry in combination.document["host"]["tokenizers"]
    ]
    assert backends == ["fast", "slow"]


def test_the_same_entry_measured_twice_differently_is_refused():
    faster = {"host": {"tokenizers": [dict(TOKENIZER, encode_tokens_per_s=9.0e6)]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", CLASS0), fragment("second", faster)])
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.tokenizers[qwen3-151k-bpe fast]" in str(refused.value)


def test_a_tokenizer_entry_with_no_id_combines_with_nothing():
    nameless = {"host": {"tokenizers": [without(TOKENIZER, "id")]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("nameless", nameless)])
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY


# --- validate: every reason at once, and the refusal list --------------------


def test_a_complete_spec_validates_and_resolves():
    checked = validate(merged().document)
    assert checked.ok, [str(refusal) for refusal in checked.refusals]
    assert str(checked) == "ok: 0 refusal(s)"


def test_every_missing_field_is_reported_and_not_only_the_first():
    thin = without(CLASS1, "device", "arch")
    thin = without(thin, "device", "memory", "capacity_bytes")
    checked = validate(merge(fragments(class1=thin)).document)
    missing = {
        refusal.what.split("`")[1]
        for refusal in checked.refusals
        if "`" in refusal.what
    }
    assert {"device.arch", "device.memory.capacity_bytes"} <= missing
    assert not checked.ok


def test_a_spec_peak_with_no_derate_beside_it_is_refused():
    thin = without(CLASS1, "device", "compute", "derate", "bf16")
    checked = validate(merge(fragments(class1=thin)).document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]


def test_an_unknown_key_is_refused_where_it_sits():
    checked = validate(dict(merged().document, gpu_memory_utilization=0.9))
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SEPARATION]


def test_a_field_stated_twice_is_reported_beside_the_rest_rather_than_raised():
    # The reader refuses a document that states one field under two spellings.
    # The check walks the same document, so it reports that refusal with the
    # others and keeps walking: raised out of the walk instead, it would take
    # every other result of the run with it, the unknown key beside it included.
    document = dict(merged().document, gpu_memory_utilization=0.9)
    assert "cores_physical" in document["host"]["cpu"]
    document["host.cpu.cores_physical"] = 1
    checked = validate(document)
    rules = sorted(refusal.rule.name for refusal in checked.refusals)
    assert rules == ["SEPARATION", "SHAPE"]
    assert any("stated twice" in refusal.what for refusal in checked.refusals)


def test_one_mistyped_key_does_not_take_the_rest_of_the_document_with_it():
    # The refusal a mistyped key earns is about that key; the field it was
    # meant to state is reported missing beside it.
    document = merged().document
    memory = document["device"]["memory"]
    memory["capacity_byte"] = memory.pop("capacity_bytes")
    checked = validate(document)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.SEPARATION) == 1
    assert rules.count(Rule.SHAPE) == 1


def test_a_block_holding_a_scalar_does_not_take_the_rest_with_it_either():
    # The other way the walk refuses: a block written as a number. The fields
    # under it are gone, and everything beside it is still checked.
    document = merged().document
    document["device"]["memory"] = 288.0e9
    checked = validate(document)
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.DERATE) == 1


def test_a_value_of_the_wrong_shape_is_reported():
    document = merged().document
    document["host"]["cpu"]["cores_physical"] = True
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


def test_two_tokenizer_entries_claiming_one_architecture_are_refused():
    twice = {
        "host": dict(
            CLASS0["host"],
            tokenizers=[
                TOKENIZER,
                dict(TOKENIZER, id="other", fingerprint="sha256:" + "c" * 64),
            ],
        )
    }
    document = merge(fragments(class0=twice)).document
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.TOKENIZER_IDENTITY]


# --- explain: the basis of a number ------------------------------------------


def test_every_term_a_resolved_spec_holds_can_be_explained():
    # The exit criterion, and it has to assert what is in the basis rather than
    # that there is one: a path that is in `spec.values` cannot come back empty,
    # so reachability alone survives every wrong value this module could report.
    # Each row states the value the document holds, and carries a derate exactly
    # when the field it came from is a spec peak.
    spec = resolved()
    for path in sorted(spec.values):
        basis = explain(spec, path)
        assert basis.contributions, path
        assert basis.digest == spec.digest()
        field = BY_PATH[path]
        if field.kind is Kind.TOKENIZERS:
            assert {row.path.split("].")[0] + "]" for row in basis.contributions} == {
                f"{path}[{entry['id']} {entry['backend']}]"
                for entry in spec.values[path]
            }, path
        for row in basis.contributions:
            if field.kind is Kind.TOKENIZERS:
                rate = ENTRY_RATES[row.path.rsplit(".", 1)[-1]]
                assert (row.derate is not None) == rate.peak, row.path
                continue
            assert row.value == spec.values[path], path
            assert (row.derate is not None) == field.peak, path


def test_a_tokenizer_row_names_the_fragment_that_supplied_the_entry():
    # `merge` labels an entry with the raw backend string and `explain` rebuilds
    # the label from the parsed `Backend`; nothing else asserts that the two
    # still agree, and a row whose source is silently `()` looks identical to a
    # row explained without an origin.
    origin = merged()
    basis = explain(
        MachineSpec.from_mapping(origin.document), "host.tokenizers", origin=origin
    )
    assert basis.contributions
    for row in basis.contributions:
        assert row.supplied_by == ("class0",), row.path


def test_a_spec_peak_is_reported_with_its_derate_and_the_derated_value():
    basis = explain(resolved(), "device.compute.bf16_flops")
    (row,) = basis.contributions
    assert (row.value, row.derate) == (2.5e15, 0.70)
    assert row.effective == pytest.approx(1.75e15)
    assert "x 0.7" in str(row)


def test_a_number_that_is_not_a_spec_peak_carries_no_derate():
    (row,) = explain(resolved(), "device.memory.capacity_bytes").contributions
    assert row.derate is None and row.effective == row.value


def test_a_block_explains_to_every_field_under_it():
    paths = {row.path for row in explain(resolved(), "host.ipc").contributions}
    assert paths == {"host.ipc.zmq_roundtrip_s", "host.ipc.shm_broadcast_s"}


def test_a_basis_names_the_fragment_each_field_came_from():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "device.memory.capacity_bytes", origin=combination)
    supplied = {row.path: row.supplied_by for row in basis.contributions}
    assert supplied["device.memory.capacity_bytes"] == ("class1",)
    links = explain(spec, "interconnect.intra_node", origin=combination)
    supplied = {row.path: row.supplied_by for row in links.contributions}
    assert supplied["interconnect.intra_node.link_latency_s"] == ("links",)


def test_without_a_merge_a_basis_still_reads_but_names_no_fragment():
    basis = explain(resolved(), "device.memory.capacity_bytes")
    assert basis.contributions[0].supplied_by == ()


def test_tokenizer_rates_explain_per_entry_and_per_rate():
    basis = explain(resolved(), "host.tokenizers")
    paths = [row.path for row in basis.contributions]
    assert paths == [
        "host.tokenizers[qwen3-151k-bpe fast].encode_fixed_s",
        "host.tokenizers[qwen3-151k-bpe fast].encode_tokens_per_s",
        "host.tokenizers[qwen3-151k-bpe fast].decode_fixed_s",
        "host.tokenizers[qwen3-151k-bpe fast].decode_tokens_per_s",
    ]
    rates = {row.path.rsplit(".", 1)[-1]: row for row in basis.contributions}
    assert rates["encode_tokens_per_s"].derate == 0.85
    assert rates["encode_fixed_s"].derate is None


def test_a_term_the_schema_does_not_know_is_asked_for_again_by_path():
    # An unknown term and an optional field the document left out both come
    # back empty however they are spelled, so both are asked for again.
    assert "device.clock_ceiling" not in BY_PATH
    assert not BY_PATH["provenance.notes"].required
    for term in ("device.clock_ceiling", "provenance.notes"):
        with pytest.raises(SpecRefusal) as refused:
            explain(resolved(), term)
        assert refused.value.rule is Rule.ADDRESSING
        assert refused.value.what == f"`{term}` names no field this spec states"
        assert "ask again by the dotted path" in refused.value.remedy


def test_an_empty_tokenizer_table_explains_to_no_rows_rather_than_refusing():
    # The field is present and holds no entries, so nothing is absent and
    # nothing was measured. A refusal here would call a spec that was read
    # assembled, and would disagree with the accessor, which answers `()`.
    document = copy.deepcopy(merged().document)
    document["host"]["tokenizers"] = []
    spec = MachineSpec.from_mapping(document)
    assert spec.value("host.tokenizers") == ()
    basis = explain(spec, "host.tokenizers")
    assert basis.contributions == ()
    assert str(basis).splitlines() == [f"host.tokenizers, from spec {spec.digest()}"]


def test_a_block_explains_without_an_optional_field_the_document_left_out():
    spec = resolved()
    assert "provenance.notes" not in spec.values
    paths = {row.path for row in explain(spec, "provenance").contributions}
    assert "provenance.authored_by" in paths
    assert "provenance.notes" not in paths


def test_the_basis_prints_its_spec_and_one_line_per_field():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    printed = str(explain(spec, "host.ipc", origin=combination)).splitlines()
    assert printed[0].endswith(spec.digest())
    assert printed[1].strip().startswith("host.ipc.zmq_roundtrip_s = 5e-05")
    assert printed[1].endswith("[class0]  probed")


# --- one reading per rank ----------------------------------------------------


@pytest.mark.parametrize(
    "tp_width,reading,spread",
    [
        (1, 926 * MIB, 0),
        (2, 6906 * MIB, 0),
        (4, 7266 * MIB, 192 * MIB),
        (8, 10704 * MIB, 640 * MIB),
    ],
)
def test_the_spread_this_hardware_really_shows_is_not_an_error(
    tp_width, reading, spread
):
    across_ranks("non_torch", [reading] * (tp_width - 1) + [reading + spread])


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE


# --- what the package reaches -------------------------------------------------


def test_the_verbs_are_reachable_by_name():
    for name in ("merge", "validate", "explain", "across_ranks"):
        assert name in spec_package.__all__
        assert getattr(spec_package, name, None) is not None


# --- the probe that needs no device, and the counts it writes down -----------

#: A second tokenizer, so the two fragments below overlap in no entry either.
ELSEWHERES = dict(
    TOKENIZER,
    id="llama-128k-bpe",
    fingerprint="sha256:" + "d" * 64,
    applies_to=["LlamaForCausalLM"],
)


def published_topology(root, *, cores_physical, threads_per_core=2, packages=1):
    """A stand-in for the per-processor topology the kernel publishes."""
    processor = 0
    for package in range(packages):
        for core in range(cores_physical // packages):
            for _ in range(threads_per_core):
                topology = root / f"cpu{processor}" / "topology"
                topology.mkdir(parents=True)
                (topology / "core_id").write_text(f"{core}\n")
                (topology / "physical_package_id").write_text(f"{package}\n")
                processor += 1
    return str(root)


def test_the_probe_counts_cores_and_the_threads_over_them_apart(tmp_path):
    assert cpu_counts(published_topology(tmp_path, cores_physical=8)) == {
        "cores_physical": 8,
        "cores_logical": 16,
    }


def test_one_core_id_on_two_packages_is_two_cores(tmp_path):
    # `core_id` is numbered within its package, so counting core ids alone
    # would report half of a two-socket machine and the halving would look
    # exactly like a correct reading of a smaller host.
    assert cpu_counts(published_topology(tmp_path, cores_physical=96, packages=2)) == {
        "cores_physical": 96,
        "cores_logical": 192,
    }


def test_a_host_that_publishes_no_topology_is_refused_rather_than_halved(tmp_path):
    with pytest.raises(SpecRefusal) as refused:
        cpu_counts(str(tmp_path))
    assert refused.value.rule is Rule.MEASURED
    assert "lists no processor" in refused.value.what
    # No document is involved in this one: the host under the probe will not
    # say what it is. Naming a rule about a document's shape would describe
    # something that did not happen, to a reader who then looks for it.
    assert "shape the schema declares" not in str(refused.value)
    (tmp_path / "cpu0").mkdir()
    with pytest.raises(SpecRefusal) as unreadable:
        cpu_counts(str(tmp_path))
    assert "cpu0 publishes no topology" in unreadable.value.what


def test_a_topology_that_publishes_no_number_is_refused_rather_than_counted(tmp_path):
    # A file that holds nothing a number can be read out of is not a reading.
    # Counting it puts a core count into a spec that looks exactly like a
    # measurement, on the same path that refuses a file which is not there.
    processors = published_topology(tmp_path, cores_physical=1)
    (tmp_path / "cpu1" / "topology" / "core_id").write_text("\n")
    with pytest.raises(SpecRefusal) as refused:
        cpu_counts(processors)
    assert refused.value.rule is Rule.MEASURED
    assert "cpu1 publishes core_id" in refused.value.what


def test_a_fragment_carrying_no_tokenizer_is_refused(tmp_path):
    with pytest.raises(SpecRefusal) as refused:
        tokenizer_fragment(
            [],
            machine="node-18",
            authored_by="ana",
            date="2026-09-22",
            origin="probed",
            processors=published_topology(tmp_path, cores_physical=8),
        )
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY


def test_an_entry_that_is_not_a_tokenizer_entry_is_refused_here(tmp_path):
    # The entries are the caller's, and this is where they are read against
    # the table a tokenizer entry has. Left to `validate`, the same entry is
    # refused after a merge, in a document whose other fields came from
    # elsewhere and whose carrying fragment has to be worked back to.
    with pytest.raises(SpecRefusal) as refused:
        tokenizer_fragment(
            [{"not": "a tokenizer"}],
            machine="node-18",
            authored_by="ana",
            date="2026-09-22",
            origin="probed",
            processors=published_topology(tmp_path, cores_physical=8),
        )
    assert refused.value.rule is Rule.SEPARATION


def test_an_entry_without_the_derate_its_rates_oblige_is_refused_here(tmp_path):
    incomplete = {name: TOKENIZER[name] for name in TOKENIZER if name != "derate"}
    with pytest.raises(SpecRefusal) as refused:
        tokenizer_fragment(
            [incomplete],
            machine="node-18",
            authored_by="ana",
            date="2026-09-22",
            origin="probed",
            processors=published_topology(tmp_path, cores_physical=8),
        )
    assert refused.value.rule is Rule.DERATE


def test_the_pair_merged_cleanly_before_the_counts_were_emitted():
    # The control for the next test: what the tokenizer probe emitted before
    # it wrote the processor down -- the rates and the
    # stanza, and no field the node's own fragment also states. Nothing in
    # their shape says the two were measured on different processors, so the
    # merge has nothing to compare and combines them into one document.
    elsewhere = fragment(
        "tokenizer-laptop",
        {"host": {"tokenizers": [ELSEWHERES]}},
        machine="node-18",
        authored_by="ana",
    )
    combination = merge([elsewhere, fragment("class0", CLASS0, machine="node-18")])
    assert combination.document["host"]["cpu"]["cores_physical"] == 96
    assert [entry["id"] for entry in combination.document["host"]["tokenizers"]] == [
        "llama-128k-bpe",
        "qwen3-151k-bpe",
    ]


def test_the_probes_counts_refuse_the_pair_that_used_to_merge(tmp_path):
    # The same two fragments as the control above, with the tokenizer one
    # built by the probe on a host of 8 physical cores instead of by hand.
    # It now states a field the node's fragment also states, and one machine
    # cannot have both counts, so the merge refuses and prints both.
    laptop = tokenizer_fragment(
        [ELSEWHERES],
        machine="node-18",
        authored_by="ana",
        date="2026-09-22",
        origin="probed",
        source="tokenizer-laptop",
        processors=published_topology(tmp_path, cores_physical=8),
    )
    with pytest.raises(SpecRefusal) as refused:
        merge([laptop, fragment("class0", CLASS0, machine="node-18")])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "`host.cpu.cores_physical` is 8 in" in message
    assert "and 96 in" in message
    for named in ("tokenizer-laptop", "class0", "node-18", "ana"):
        assert named in message, f"{named} is not named in: {message}"


def test_the_same_pair_merges_when_the_counts_agree(tmp_path):
    # The other half of the test above,
    # `test_the_probes_counts_refuse_the_pair_that_used_to_merge`: the refusal
    # is about the two readings and not about the probe having spoken at all.
    # Run on the machine the spec is authored for, the same probe supplies the
    # same field and the pair merges.
    here = tokenizer_fragment(
        [ELSEWHERES],
        machine="node-18",
        authored_by="ana",
        date="2026-09-22",
        origin="probed",
        source="tokenizer-node",
        processors=published_topology(tmp_path, cores_physical=96, packages=2),
    )
    combination = merge([here, fragment("class0", CLASS0, machine="node-18")])
    assert combination.document["host"]["cpu"] == {
        "cores_physical": 96,
        "cores_logical": 192,
    }
    assert combination.sources["host.cpu.cores_physical"] == (
        "tokenizer-node",
        "class0",
    )


def test_the_fragment_states_where_it_is_authored_for_and_how(tmp_path):
    emitted = tokenizer_fragment(
        [TOKENIZER],
        machine="node-18",
        authored_by="ana",
        date="2026-09-22",
        origin="probed",
        subject={"model": "Qwen/Qwen3-8B"},
        processors=published_topology(tmp_path, cores_physical=8),
    )
    assert emitted.machine == "node-18"
    assert emitted.origin == "mixed"
    # The subject is what the sweep ran on; the core counts are of the host.
    stated = emitted.values["provenance.origins"]
    assert stated["host.tokenizers[qwen3-151k-bpe fast]"] == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }
    assert stated["host.cpu.cores_physical"] == {"origin": "probed"}
    assert "by ana on 2026-09-22" in emitted.stanza()


def test_how_the_rates_were_obtained_is_the_callers_claim_and_has_no_default(tmp_path):
    # The rates arrive as an argument and nothing here encodes or decodes
    # anything, so what the record claims about their source is the caller's to
    # state. `provenance.origin` is what a merge refusal quotes and what an
    # artifact carries as the numbers' source, and a default here would be this
    # function asserting a measurement it did not take, over data it received.
    processors = published_topology(tmp_path, cores_physical=8)
    stated = {
        "machine": "node-18",
        "authored_by": "ana",
        "date": "2026-09-22",
        "processors": processors,
    }
    carried = tokenizer_fragment([TOKENIZER], origin="assumed", **stated)
    origins = carried.values["provenance.origins"]
    assert origins["host.tokenizers[qwen3-151k-bpe fast]"] == {"origin": "assumed"}
    assert origins["host.cpu.cores_physical"] == {"origin": "probed"}
    with pytest.raises(TypeError):
        tokenizer_fragment([TOKENIZER], **stated)


def test_the_probe_is_reachable_by_name():
    assert "tokenizer_fragment" in spec_package.__all__
    assert spec_package.tokenizer_fragment is tokenizer_fragment


# --- the tokenizer sweep, and the node-18 spec built over it ------------------


class Clocked:
    """A tokenizer whose every call costs `fixed + tokens / rate` on its own
    clock, and three times the per-token cost at `slow_at` tokens."""

    def __init__(self, fixed, rate, slow_at=None):
        self.now, self.fixed, self.rate, self.slow_at = 0.0, fixed, rate, slow_at
        self.is_fast = True

    def clock(self):
        return self.now

    def _spend(self, tokens):
        scale = 3 if tokens == self.slow_at else 1
        self.now += self.fixed + scale * tokens / self.rate

    def encode(self, text):
        ids = [0] * len(text.split())
        self._spend(len(ids))
        return ids

    def decode(self, ids):
        self._spend(len(ids))
        return " ".join("w" for _ in ids)


SWEPT = tuple(4**k for k in range(7))
SWEEP_TEXT = "w " * SWEPT[-1]


def test_the_sweep_recovers_the_fixed_cost_and_the_rate_it_ran_over():
    tok = Clocked(3.0e-5, 2.0e6)
    rates = tokenizer_sweep(tok, SWEEP_TEXT, SWEPT, bound=0.01, clock=tok.clock)
    for side in ("encode", "decode"):
        assert rates[f"{side}_fixed_s"] == pytest.approx(3.0e-5)
        assert rates[f"{side}_tokens_per_s"] == pytest.approx(2.0e6)


def test_a_length_the_fit_does_not_hold_at_is_refused_by_name():
    tok = Clocked(3.0e-5, 2.0e6, slow_at=1024)
    with pytest.raises(SpecRefusal) as refused:
        tokenizer_sweep(tok, SWEEP_TEXT, SWEPT, bound=0.10, clock=tok.clock)
    assert refused.value.rule is Rule.MEASURED
    assert "encode at 1024 tokens" in str(refused.value)
    assert "bound of 10.0%" in str(refused.value)


def test_a_sweep_text_shorter_than_the_longest_length_is_refused():
    tok = Clocked(3.0e-5, 2.0e6)
    with pytest.raises(SpecRefusal, match="the sweep text is 2 tokens"):
        tokenizer_sweep(tok, "w w", SWEPT, bound=0.10, clock=tok.clock)


SPEC_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _node18_script():
    path = SPEC_ROOT / "scripts" / "compass" / "mi308x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi308x_spec", path)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


def _swept_on(script, tmp_path, cores_physical, **entry):
    tok = Clocked(3.0e-5, 2.0e6)
    rates = tokenizer_sweep(tok, SWEEP_TEXT, SWEPT, bound=0.01, clock=tok.clock)
    return tokenizer_fragment(
        [dict(TOKENIZER, **rates, derate=1.0, **entry)],
        machine=script.NAME,
        authored_by="ana",
        date="2026-10-11",
        origin="probed",
        source="tokenizer",
        processors=published_topology(
            tmp_path, cores_physical=cores_physical, packages=2
        ),
    )


def test_the_node_18_fragments_validate_and_explain_splits_measured_from_declared(
    tmp_path,
):
    script = _node18_script()
    merged = merge([_swept_on(script, tmp_path, 112), *script.device_fragments()])
    assert validate(merged.document).ok
    measured, declared = script.report(merged).split("assumed:")
    for field in ("device.memory.capacity_bytes", "encode_tokens_per_s"):
        assert field in measured and field not in declared
    for field in ("host.ipc.zmq_roundtrip_s", "host.ipc.shm_broadcast_s", "bf16_flops"):
        assert field in declared and field not in measured


def test_a_tokenizer_swept_off_node_18_is_refused_by_the_merge(tmp_path):
    script = _node18_script()
    with pytest.raises(SpecRefusal) as refused:
        merge([_swept_on(script, tmp_path, 8), *script.device_fragments()])
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "`host.cpu.cores_physical` is 8 in 'tokenizer'" in str(refused.value)
    assert "and 112 in 'device'" in str(refused.value)


def test_a_tokenizer_with_no_tokenizer_json_is_refused_before_the_merge(tmp_path):
    # `_fingerprint` gives None for a slow tokenizer, which has no such file.
    script = _node18_script()
    with pytest.raises(SpecRefusal, match="fingerprint"):
        _swept_on(script, tmp_path, 112, fingerprint=None)


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_checked_in_mi308x_specs_hold_their_start(monkeypatch, name):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    script = _node18_script()
    path = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    values = MachineSpec.from_mapping(json.loads(path.read_text())).values
    for fragment in script.device_fragments():
        for key, value in fragment.values.items():
            if not key.startswith("provenance."):
                assert values[key] == value, key


def _mi355x_spec(monkeypatch):
    """The MI355X script and its checked-in document, read."""
    monkeypatch.delenv("COMPASS_MACHINE", raising=False)
    path = SPEC_ROOT / "scripts" / "compass" / "mi355x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi355x_spec", path)
    script = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(script)
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{script.NAME}.json"
    return script, MachineSpec.from_mapping(json.loads(document.read_text()))


def test_the_checked_in_mi355x_spec_holds_its_start(monkeypatch):
    script, spec = _mi355x_spec(monkeypatch)
    for fragment in script.device_fragments():
        for key, value in fragment.values.items():
            if not key.startswith("provenance."):
                assert spec.values[key] == value, key


def test_the_mi355x_spec_explains_what_the_start_measured_apart_from_the_declared(
    monkeypatch,
):
    script, spec = _mi355x_spec(monkeypatch)
    origins = {}
    for block in ("host", "device", "interconnect"):
        for row in explain(spec, block).contributions:
            origins.setdefault(row.origin["origin"], set()).add(row.path)
    assert set(origins) == {"probed", "datasheet", "assumed"}
    assert "device.memory.capacity_bytes" in origins["probed"]
    assert origins["datasheet"] == {
        "device.memory.bandwidth_bytes_per_s",
        "device.compute.bf16_flops",
        "device.compute.fp8_flops",
        "device.compute.fp4_flops",
        "interconnect.intra_node.domain_devices",
        "interconnect.intra_node.link_bandwidth_bytes_per_s",
    }
    record = spec.origin("device.memory.capacity_bytes")["record"]
    assert record == f"sha256:{script.RECORD}"


def test_the_mi355x_spec_rebuilds_byte_for_byte_from_its_own_sweep(
    monkeypatch, tmp_path
):
    script, _ = _mi355x_spec(monkeypatch)
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{script.NAME}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    assert out.read_bytes() == document.read_bytes()


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_mi308x_specs_rebuild_from_their_own_sweeps(monkeypatch, tmp_path, name):
    # Rebuilt with the sweep the document carries, the script writes the
    # checked-in document's fields and origins; its key order is the script's.
    monkeypatch.setenv("COMPASS_MACHINE", name)
    script = _node18_script()
    document = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    assert json.loads(out.read_text()) == json.loads(document.read_text())
