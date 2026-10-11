# SPDX-License-Identifier: MIT
"""The three verbs over the spec: combining fragments, checking one, reading one.

The fragments below are the ones the probes of each hardware tier would emit --
a tokenizer and an interprocess measurement that need no device, a single-card
engine start, a multi-rank one, and a fabric measurement -- so a test that
combines them combines the same partial documents a real run would, and a test
that breaks one breaks exactly what it says it does.

The numbers are the measured ones wherever a measurement exists, because two of
these checks are judgements about size: the cross-rank spread accepted is the
one the hardware really shows, and the spread refused is the one a neighbour
really produced. Round numbers would make both tests vacuous.

The probe that needs no device has a section of its own. Its fragments are
built by the probe itself against a stand-in for the per-processor topology the
kernel publishes, so the core counts in them are read by the code a real run
reads them with rather than typed in beside the assertion that checks them.

The last two sections are the memory a probe that does start an engine reads
off each card, and the table of which probe supplies which runtime constant.
The readings are written as the term under test plus the rest of the card, so
a test says what it is about while the checks over them still see the four
numbers a rank really reports. The predictions they are judged
against are the spec's own measured reserves, which is what makes the accepted
readings and the refused one a judgement about size rather than about round
numbers.
"""

import copy
import importlib.util
import json
import pathlib
import sys

import pytest

from atom.compass import spec as spec_package
from atom.compass.spec import (
    QUANTITIES,
    DeviceMemory,
    Fragment,
    MachineSpec,
    Rule,
    SpecRefusal,
    StackMismatch,
    across_ranks,
    explain,
    merge,
    non_torch_across_ranks,
    probe_for,
    tokenizer_fragment,
    tokenizer_sweep,
    validate,
)
from atom.compass.spec import fields as schema_module
from atom.compass.spec import probes as probes_module
from atom.compass.spec.fields import BY_PATH, Field, Kind
from atom.compass.spec.probes import FILLED_BY, cpu_counts
from atom.compass.spec.tokenizers import ENTRY_FIELDS
from atom.compass.spec.validate import (
    ASKABLE_OF_A_DOCUMENT,
    CONDITIONS,
    PROBE_TABLES,
    WIDTH_TABLES,
)
from atom.compass.spec.validate import DERATES as DERATES_ASKED
from atom.compass.spec.validate import MISSING as MISSING_ASKED
from atom.compass.spec.validate import PROBES as PROBES_ASKED
from atom.compass.spec.validate import STACK as STACK_ASKED
from atom.compass.spec.validate import TRANSFERS as TRANSFERS_ASKED
from atom.compass.spec.validate import WIDTHS as WIDTHS_ASKED

MACHINE = "mi355x-8gpu-2node"
#: A tokenizer entry's fields by name, to say which of its rates is a peak.
ENTRY_RATES = {field.path: field for field in ENTRY_FIELDS}
STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3"}
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

#: Tier 0, no device: the tokenizer sweep and the interprocess round trips.
TIER0 = {
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    }
}

#: Tier 1, one card: the five readings and the single-width graph pool, plus
#: one entry no tier-1 probe produces. `allocator_retained_after_load_bytes` at
#: width 1 is the hole `FILLED_BY` records, so on a real machine it is measured
#: by hand. It is carried here because these fixtures exist to make a complete
#: document, and `provenance.method` is one claim over a whole fragment:
#: `probed` on this body is right about every entry but that one, which is a
#: limit of the fixture rather than a capability of the probe.
TIER1 = {
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {"bf16_flops": 2.5e15, "fp8_flops": 5.0e15, "derate": 0.70},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {1: 970.0e6},
            "allocator_retained_after_load_bytes": {1: 1.1e6},
            "persistent_forward_buffer_bytes": 124.0e6,
            "cudagraph_pool": {
                "w1_base_bytes": 95.5e6,
                "w1_bytes_per_captured_token": 0.318e6,
            },
        },
        "software_pinned_to": dict(STACK),
    }
}

#: Tier 2, N cards: one engine start per width, and the flat pool above one.
TIER2 = {
    "device": {
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {2: 7.2e9, 4: 7.6e9, 8: 11.2e9},
            "allocator_retained_after_load_bytes": {2: 2.17e9, 4: 2.17e9, 8: 2.17e9},
            "cudagraph_pool": {"w_gt1_flat_bytes": 109.0e6},
        }
    }
}

#: Tier 2 and 3: collectives in real groups, then the same across two nodes.
LINKS = {
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
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
    method="probed",
    authored_by="a person",
    date="2026-09-18",
    **stanza,
):
    """One probe's output: what it measured, and where it was measured."""
    document = {
        "schema_version": 1,
        "name": machine,
        "provenance": dict(
            {"authored_by": authored_by, "date": date, "method": method}, **stanza
        ),
    }
    document.update(copy.deepcopy(body or {}))
    return Fragment.from_mapping(document, source)


def fragments(**edits):
    """The five fragments a full spec is built from, with one body replaced."""
    bodies = {
        "tier0": TIER0,
        "tier1": TIER1,
        "tier2": TIER2,
        "links": LINKS,
    }
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
    here = fragment("tokenizer-a", TIER0, machine="node-18", authored_by="ana")
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
            [fragment("a", TIER0, machine="node-18"), fragment("b", machine="node-22")]
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
    here = fragment("tokenizer-a", TIER0, machine="node-18", authored_by="ana")
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
    first = merge([fragment("tier0", TIER0), fragment("tier1", TIER1)])
    again = merge(
        [
            Fragment.from_mapping(first.document, "machine.yaml"),
            fragment("tier2", TIER2),
            fragment("links", LINKS),
        ]
    )
    assert again.document["provenance"]["fragments"] == [
        "tier0",
        "tier1",
        "machine.yaml",
        "tier2",
        "links",
    ]


def test_one_machine_named_twice_with_two_answers_is_refused():
    other_cpu = {
        "host": dict(TIER0["host"], cpu={"cores_physical": 128, "cores_logical": 256})
    }
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", TIER0), fragment("second", other_cpu)])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.cpu.cores_physical" in message
    assert "96" in message and "128" in message
    assert "first" in message and "second" in message


def test_two_fragments_that_agree_are_not_a_conflict():
    combination = merge([fragment("first", TIER0), fragment("second", TIER0)])
    assert combination.sources["host.cpu.cores_physical"] == ("first", "second")


def test_a_width_table_is_combined_across_the_probes_that_measured_it():
    reserve = resolved().value(
        "device.runtime_constants.driver_and_collective_reserve_bytes"
    )
    assert sorted(reserve) == [1, 2, 4, 8]
    assert reserve[1] == 970.0e6 and reserve[8] == 11.2e9


def test_one_width_measured_twice_differently_is_refused():
    disagrees = copy.deepcopy(TIER2)
    disagrees["device"]["runtime_constants"]["driver_and_collective_reserve_bytes"][
        1
    ] = 980.0e6
    with pytest.raises(SpecRefusal) as refused:
        merged(tier2=disagrees)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "driver_and_collective_reserve_bytes[1]" in str(refused.value)


def test_two_fragments_pinned_to_different_stacks_are_refused_as_a_stack_conflict():
    moved = copy.deepcopy(TIER1)
    moved["device"]["software_pinned_to"]["rocm"] = "7.3.0"
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("tier1", TIER1), fragment("tier1b", moved)])
    assert refused.value.rule is Rule.PINNED_STACK
    assert "fragments pinned to different stacks" in refused.value.remedy


def test_the_merged_document_reads_as_a_spec_and_echoes_back():
    spec = resolved()
    assert MachineSpec.from_mapping(spec.echo()).digest() == spec.digest()


def test_the_merged_provenance_names_every_fragment_and_its_method():
    combination = merge(
        [
            fragment("tier0", TIER0),
            fragment("tier1", TIER1, method="datasheet", date="2026-09-20"),
            fragment("tier2", TIER2),
            fragment("links", LINKS, authored_by="another person"),
        ]
    )
    provenance = combination.document["provenance"]
    assert provenance["fragments"] == ["tier0", "tier1", "tier2", "links"]
    assert provenance["method"] == "mixed"
    assert provenance["date"] == "2026-09-20"
    assert provenance["authored_by"] == "a person, another person"


def test_one_method_is_kept_when_every_fragment_used_it():
    assert merged().document["provenance"]["method"] == "probed"


@pytest.mark.parametrize(
    "path",
    [
        "schema_version",
        "name",
        "provenance.authored_by",
        "provenance.date",
        "provenance.method",
    ],
)
def test_a_fragment_that_does_not_state_where_it_came_from_is_refused(path):
    document = {
        "schema_version": 1,
        "name": MACHINE,
        "provenance": {
            "authored_by": "a person",
            "date": "2026-09-18",
            "method": "probed",
        },
    }
    document.update(copy.deepcopy(TIER0))
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
        merge([fragment("first", TIER0), fragment("second", revised)])
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
    combination = merge([fragment("fast", TIER0), fragment("slow", slow)])
    backends = [
        entry["backend"] for entry in combination.document["host"]["tokenizers"]
    ]
    assert backends == ["fast", "slow"]


def test_the_same_entry_measured_twice_differently_is_refused():
    faster = {"host": {"tokenizers": [dict(TOKENIZER, encode_tokens_per_s=9.0e6)]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("first", TIER0), fragment("second", faster)])
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "host.tokenizers[qwen3-151k-bpe fast]" in str(refused.value)


def test_a_tokenizer_entry_with_no_id_combines_with_nothing():
    nameless = {"host": {"tokenizers": [without(TOKENIZER, "id")]}}
    with pytest.raises(SpecRefusal) as refused:
        merge([fragment("nameless", nameless)])
    assert refused.value.rule is Rule.TOKENIZER_IDENTITY


# --- validate: every reason at once, and the refusal list --------------------


def test_a_complete_spec_validates_and_resolves():
    checked = validate(merged(), tp_widths=(1, 2, 4, 8), observed_stack=STACK)
    assert checked.ok, [str(refusal) for refusal in checked.refusals]
    assert checked.spec.value("device.name") == "MI355X"
    assert checked.stack_differences == ()


def test_every_missing_field_is_reported_and_not_only_the_first():
    thin = without(TIER1, "device", "arch")
    thin = without(thin, "device", "memory", "capacity_bytes")
    checked = validate(merge(fragments(tier1=thin)).document)
    missing = {
        refusal.what.split("`")[1]
        for refusal in checked.refusals
        if "`" in refusal.what
    }
    assert {"device.arch", "device.memory.capacity_bytes"} <= missing
    assert not checked.ok and checked.spec is None


def test_a_missing_runtime_constant_has_no_default():
    thin = without(
        TIER1, "device", "runtime_constants", "persistent_forward_buffer_bytes"
    )
    checked = validate(merge(fragments(tier1=thin)).document)
    rules = {refusal.rule for refusal in checked.refusals}
    assert rules == {Rule.NO_DEFAULTS}
    assert "persistent_forward_buffer_bytes" in str(checked.refusals[0])


def test_a_spec_peak_with_no_derate_beside_it_is_refused():
    thin = without(TIER1, "device", "compute", "derate")
    checked = validate(merge(fragments(tier1=thin)).document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.DERATE]


def test_a_width_the_deployment_will_use_and_nobody_measured_is_refused():
    checked = validate(merged(), tp_widths=(1, 16))
    assert len(checked.refusals) == 2
    for refusal in checked.refusals:
        assert refusal.rule is Rule.NO_DEFAULTS
        assert "16" in refusal.what
    assert "[1, 2, 4, 8]" in checked.refusals[0].what


def test_the_widths_that_were_measured_are_not_refused():
    assert validate(merged(), tp_widths=(1, 2, 4, 8)).ok


def test_a_stack_that_has_moved_warns_and_names_both_versions():
    with pytest.warns(StackMismatch, match="7.2.4"):
        checked = validate(merged(), observed_stack=dict(STACK, rocm="7.3.0"))
    assert checked.ok
    assert checked.stack_differences == (("rocm", "7.2.4", "7.3.0"),)


def test_a_stack_that_has_moved_is_refused_under_the_strict_flag():
    with pytest.warns(StackMismatch):
        checked = validate(
            merged(), observed_stack=dict(STACK, rocm="7.3.0"), strict=True
        )
    assert not checked.ok
    assert checked.refusals[0].rule is Rule.PINNED_STACK
    assert "7.3.0" in checked.refusals[0].what


def test_a_stack_that_matches_is_silent():
    assert validate(merged(), observed_stack=STACK, strict=True).ok


def test_a_stack_difference_is_rendered_by_the_run_that_refused_nothing():
    # Without the strict flag nothing is refused, so the difference is the only
    # thing this check found. It is on the dataclass either way; the string is
    # where a reader of the summary looks for what the check did.
    with pytest.warns(StackMismatch):
        checked = validate(
            merged(),
            tp_widths=(1, 2, 4, 8),
            observed_stack=dict(STACK, rocm="7.3.0"),
        )
    assert checked.ok
    assert str(checked) == (
        "ok: 0 refusal(s)\n"
        "  stack moved: the constants were measured against rocm '7.2.4' "
        "(now '7.3.0')"
    )


def test_a_transfer_keeps_the_source_stack_out_of_this_machines_pin():
    # The constants came from another spec; the stack that spec was pinned to is
    # provenance about them, not a claim about this host's stack.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert combination.document["device"]["software_pinned_to"]["rocm"] == "7.2.4"
    checked = validate(combination)
    assert not checked.ok
    assert checked.refusals[0].rule is Rule.PINNED_STACK
    assert checked.refusals[0].what == (
        "'tier2' (machine 'mi355x-8gpu-2node', transferred-from:mi300x-8gpu, "
        "by a person on 2026-09-18) carried constants over from 'mi300x-8gpu', "
        "measured against rocm '7.0.2', into a spec pinned to rocm '7.2.4'"
    )


def test_a_transfer_that_names_no_stack_at_all_is_refused():
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", TIER2, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    checked = validate(combination)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.PINNED_STACK]
    assert "without saying which stack" in checked.refusals[0].what


def test_a_saved_transfer_merged_again_names_the_merge_that_dropped_its_pin():
    # The transfer is pinned to this stack; the merge that wrote the saved
    # document kept that pin out, so no author omitted it. The first-hand
    # transfer with no pin is the control, and its text does not change. Saved
    # twice, the provenance lists the pinned transfer and the first saved
    # document, which is itself a transfer with no pin; the remedy names both.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK)
    transfer = fragment("tier2", carried, method="transferred-from:mi300x-8gpu")
    saved = Fragment.from_mapping(merge([transfer]).document, "t.yaml")
    rest = fragments()[:2] + [fragment("links", LINKS)]
    (refused,) = validate(merge([saved] + rest)).refusals
    assert refused.rule is Rule.PINNED_STACK
    assert refused.what.startswith(
        "'t.yaml' (machine 'mi355x-8gpu-2node', transferred-from:mi300x-8gpu, "
        "by a person on 2026-09-18) carried constants over from 'mi300x-8gpu', "
        "and its provenance names the fragments an earlier merge built it from"
    )
    bare = fragment("tier2", TIER2, method="transferred-from:mi300x-8gpu")
    first_hand = validate(merge(fragments()[:2] + [bare, fragment("links", LINKS)]))
    assert first_hand.refusals[0].what == (
        "'tier2' (machine 'mi355x-8gpu-2node', transferred-from:mi300x-8gpu, "
        "by a person on 2026-09-18) carried constants over from 'mi300x-8gpu' "
        "without saying which stack they were measured against"
    )
    twice = Fragment.from_mapping(merge([saved]).document, "t2.yaml")
    (refused,) = validate(merge([twice] + rest)).refusals
    assert refused.remedy == (
        "merge the fragments it was built from ('tier2', 't.yaml') in its place, "
        "since a transfer states its source's stack pin there and nowhere else; "
        "any of them that is itself a saved document with no pin is refused the "
        "same way"
    )


def test_a_transfer_whose_provenance_lists_no_fragments_is_refused_first_hand():
    # `merge` never writes an empty list, so one names no earlier merge; whoever
    # wrote this transfer left its pin out, as in the first-hand case.
    method = "transferred-from:mi300x-8gpu"
    rest = fragments()[:2] + [fragment("links", LINKS)]
    empty = fragment("tier2", TIER2, method=method, fragments=[])
    bare = fragment("tier2", TIER2, method=method)
    (refused,) = validate(merge(rest + [empty])).refusals
    (control,) = validate(merge(rest + [bare])).refusals
    assert str(refused) == str(control)


def test_a_transfer_from_a_spec_pinned_to_this_stack_validates():
    carried = dict(copy.deepcopy(TIER2))
    carried["device"]["software_pinned_to"] = dict(STACK)
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert validate(combination).ok


def test_a_clear_check_says_which_conditions_it_could_not_ask():
    # Four of the six conditions are opt-in, and a check that is silent about
    # what it declined to ask is a clear with no content behind it.
    bare = validate(merged().document)
    assert bare.ok
    assert [condition.split(" -- ")[0] for condition in bare.not_asked] == [
        WIDTHS_ASKED,
        PROBES_ASKED,
        STACK_ASKED,
        TRANSFERS_ASKED,
    ]
    assert "`tp_widths=`" in bare.not_asked[0]
    assert bare.asked_in_part == ()
    asked = validate(
        merged(), tp_widths=(1, 2, 4, 8), observed_stack=STACK, strict=True
    )
    assert asked.ok and asked.not_asked == () and asked.asked_in_part == ()


def test_the_transfer_condition_names_itself_as_unaskable_of_a_document():
    # The same spec is refused as a `Merge` and clear as the document it makes,
    # because the source's pin is deliberately in no field of the document. A
    # caller holding only the document cannot ask it, so its check must say so.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    combination = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    assert not validate(combination).ok
    as_document = validate(combination.document)
    assert as_document.ok
    assert any(
        condition.startswith(TRANSFERS_ASKED) for condition in as_document.not_asked
    )
    assert "in no field" in str(as_document)


def test_a_saved_document_merged_again_does_not_claim_to_have_asked_the_transfer():
    # Validating the first `Merge` is refused. Its document keeps the transfer's
    # source pin out and says only `method: mixed`, so a `Merge` of the saved
    # document has no transfer to ask the condition of. A re-merged all-`probed`
    # document is the control.
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    transfer = fragment("tier2", carried, method="transferred-from:mi300x-8gpu")
    combination = merge(fragments()[:2] + [transfer, fragment("links", LINKS)])
    asked = {"tp_widths": (1, 2, 4, 8), "observed_stack": STACK}
    first = validate(combination, **asked)
    assert [refusal.rule for refusal in first.refusals] == [Rule.PINNED_STACK]
    bare = validate(combination.document, **asked)
    assert bare.ok and [c.split(" -- ")[0] for c in bare.not_asked] == [TRANSFERS_ASKED]
    saved = Fragment.from_mapping(combination.document, "machine.yaml")
    again = validate(merge([saved]), **asked)
    assert again.ok and again.asked_in_part == ()
    assert [c.split(" -- ")[0] for c in again.not_asked] == [TRANSFERS_ASKED]
    assert "method `mixed` in 'machine.yaml'" in again.not_asked[0]
    # A transfer stated beside the saved document is asked, and only in part.
    beside = validate(merge([saved, transfer]), **asked)
    assert [refusal.rule for refusal in beside.refusals] == [Rule.PINNED_STACK]
    assert [c.split(" -- ")[0] for c in beside.asked_in_part] == [TRANSFERS_ASKED]
    # With some stack pins resolved it was asked in part, and says so once.
    thin = copy.deepcopy(combination.document)
    del thin["device"]["software_pinned_to"]["rccl"]
    thin = Fragment.from_mapping(thin, "machine.yaml")
    partly = validate(merge([thin, transfer]), **asked)
    assert sum(c.startswith(TRANSFERS_ASKED) for c in partly.asked_in_part) == 1
    # With no stack pin resolved it was not asked at all, and says so once.
    unpinned = copy.deepcopy(combination.document)
    del unpinned["device"]["software_pinned_to"]
    unpinned = Fragment.from_mapping(unpinned, "machine.yaml")
    nowhere = validate(merge([unpinned, transfer]), **asked)
    assert sum(c.startswith(TRANSFERS_ASKED) for c in nowhere.not_asked) == 1
    assert nowhere.asked_in_part == ()
    control = Fragment.from_mapping(merged().document, "machine.yaml")
    assert validate(merge([control]), **asked).not_asked == ()


def test_a_desk_fix_refusal_does_not_hide_the_expensive_one():
    # The phase-one refusal here is a derate an author types in at a desk; the
    # width table behind it is present and schema-valid, and the width nobody
    # measured costs an eight-GPU reservation. Both are reported, and so is the
    # stack, because each question is asked of the fields that did resolve.
    thin = without(TIER1, "device", "memory", "derate")
    with pytest.warns(StackMismatch):
        checked = validate(
            merge(fragments(tier1=thin)).document,
            tp_widths=(1, 2, 4, 8, 16),
            observed_stack=dict(STACK, rocm="7.3.0"),
            strict=True,
        )
    rules = [refusal.rule for refusal in checked.refusals]
    assert checked.spec is None
    assert rules.count(Rule.DERATE) == 1
    assert rules.count(Rule.NO_DEFAULTS) == len(WIDTH_TABLES)
    assert rules.count(Rule.PINNED_STACK) == 1
    assert [condition.split(" -- ")[0] for condition in checked.not_asked] == [
        TRANSFERS_ASKED
    ]
    assert "16" in str(checked) and "7.3.0" in str(checked)


def test_the_stack_is_asked_of_the_pins_a_partial_document_carries():
    # A pin the document leaves out is reported once, as missing, by the schema
    # half. The stack question is still asked of the pins it does carry, so the
    # moved one is found, and the absent one is named as the part it could not
    # be asked of -- rather than the check itself declining the absent field and
    # taking every other result of the run down with it.
    document = merged().document
    del document["device"]["software_pinned_to"]["rccl"]
    with pytest.warns(StackMismatch):
        checked = validate(document, observed_stack=dict(STACK, rocm="7.3.0"))
    assert checked.spec is None
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]
    assert [component for component, _, _ in checked.stack_differences] == ["rocm"]
    (in_part,) = checked.asked_in_part
    assert in_part.startswith(STACK_ASKED)
    assert "`device.software_pinned_to.rccl`" in in_part


def test_a_question_asked_of_part_of_what_it_reads_is_not_called_unasked():
    # The width tables are what the width question reads. With one of them
    # gone the question is still asked of the other -- width 16 is still
    # refused -- so the table it could not be asked of is named apart from the
    # questions that went unasked. Filing it under those would report a refusal
    # this run did make as a question it did not, and would subtract it from
    # the count of conditions the run reached.
    absent = ("device", "runtime_constants", "allocator_retained_after_load_bytes")
    checked = validate(
        merge(
            fragments(tier1=without(TIER1, *absent), tier2=without(TIER2, *absent))
        ).document,
        tp_widths=(16,),
    )
    assert not checked.ok
    assert any("16" in refusal.what for refusal in checked.refusals)
    (in_part,) = checked.asked_in_part
    assert in_part.startswith(WIDTHS_ASKED)
    assert "allocator_retained_after_load_bytes" in in_part
    assert "asked of the rest" in in_part
    assert not any(
        condition.startswith(WIDTHS_ASKED) for condition in checked.not_asked
    )
    assert "asked in part" in str(checked)


def test_a_question_none_of_whose_fields_resolved_was_not_asked_at_all():
    # The other side of the same split: with both width tables gone there is
    # nothing the question could have been asked of, so it is unasked and the
    # count of what the run reached must not include it.
    thin1, thin2 = TIER1, TIER2
    for name in (
        "allocator_retained_after_load_bytes",
        "driver_and_collective_reserve_bytes",
    ):
        thin1 = without(thin1, "device", "runtime_constants", name)
        thin2 = without(thin2, "device", "runtime_constants", name)
    checked = validate(
        merge(fragments(tier1=thin1, tier2=thin2)).document, tp_widths=(16,)
    )
    (unasked,) = [
        condition
        for condition in checked.not_asked
        if condition.startswith(WIDTHS_ASKED)
    ]
    assert "could not be asked at all" in unasked
    assert checked.asked_in_part == ()


def test_a_document_that_is_not_a_mapping_is_refused():
    checked = validate("a filename, not a document")
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]
    # Nothing was read against the schema here, so the two conditions that are
    # otherwise asked of anything name themselves rather than being counted as
    # questions this run answered.
    assert [condition.split(" -- ")[0] for condition in checked.not_asked] == list(
        CONDITIONS
    )


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
    # The refusal a mistyped key earns is about that key. A check that stopped
    # there would leave every other field unchecked and every consistency
    # question with nothing resolved to be asked of -- the desk fix hiding the
    # eight-GPU one again, on a document edited after its merge and handed to
    # `validate` directly, since a merge refuses the unknown key before this runs.
    document = merged().document
    memory = document["device"]["memory"]
    memory["capacity_byte"] = memory.pop("capacity_bytes")
    with pytest.warns(StackMismatch):
        checked = validate(
            document,
            tp_widths=(1, 2, 4, 8, 16),
            observed_stack=dict(STACK, rocm="7.3.0"),
            strict=True,
        )
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.SEPARATION) == 1
    assert rules.count(Rule.SHAPE) == 1
    assert rules.count(Rule.NO_DEFAULTS) == len(WIDTH_TABLES)
    assert rules.count(Rule.PINNED_STACK) == 1
    assert [condition.split(" -- ")[0] for condition in checked.not_asked] == [
        TRANSFERS_ASKED
    ]


def test_a_block_holding_a_scalar_does_not_take_the_rest_with_it_either():
    # The other way the walk refuses: a block written as a number. The fields
    # under it are gone, and everything beside it is still checked.
    document = merged().document
    document["device"]["memory"] = 288.0e9
    checked = validate(document, tp_widths=(16,))
    rules = [refusal.rule for refusal in checked.refusals]
    assert rules.count(Rule.DERATE) == 1
    assert rules.count(Rule.NO_DEFAULTS) == len(WIDTH_TABLES)
    assert any("16" in refusal.what for refusal in checked.refusals)


def test_a_value_of_the_wrong_shape_is_reported():
    document = merged().document
    document["host"]["cpu"]["cores_physical"] = True
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.SHAPE]


def test_two_tokenizer_entries_claiming_one_architecture_are_refused():
    twice = {
        "host": dict(
            TIER0["host"],
            tokenizers=[
                TOKENIZER,
                dict(TOKENIZER, id="other", fingerprint="sha256:" + "c" * 64),
            ],
        )
    }
    document = merge(fragments(tier0=twice)).document
    checked = validate(document)
    assert [refusal.rule for refusal in checked.refusals] == [Rule.TOKENIZER_IDENTITY]


def test_raise_first_gives_the_spec_or_the_first_refusal():
    assert validate(merged()).raise_first().value("device.arch") == "gfx950"
    with pytest.raises(SpecRefusal):
        validate(without(merged().document, "device", "arch")).raise_first()


def refused_by_condition():
    """One subject per condition in the check set, each earning its refusal.

    Each subject comes with the rule and a piece of the text only that
    condition's refusal writes. A subject that is wrong in one respect is
    usually wrong in another too -- a document missing an entry is refused for
    the missing width as well as for the probe that could not fill it -- so a
    rule alone can be earned by a neighbouring condition while the one named
    here produces nothing.
    """
    thin = without(TIER1, "device", "arch")
    no_constant = without(
        TIER1, "device", "runtime_constants", "persistent_forward_buffer_bytes"
    )
    no_derate = without(TIER1, "device", "memory", "derate")
    no_probe = without(
        TIER1, "device", "runtime_constants", "allocator_retained_after_load_bytes"
    )
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK, rocm="7.0.2")
    transferred = merge(
        fragments()[:2]
        + [
            fragment("tier2", carried, method="transferred-from:mi300x-8gpu"),
            fragment("links", LINKS),
        ]
    )
    with pytest.warns(StackMismatch):
        moved = validate(
            merged(), observed_stack=dict(STACK, rocm="7.3.0"), strict=True
        )
    return {
        MISSING_ASKED: [
            (
                validate(merge(fragments(tier1=thin)).document),
                Rule.SHAPE,
                "`device.arch` is missing",
            ),
            (
                validate(merge(fragments(tier1=no_constant)).document),
                Rule.NO_DEFAULTS,
                "`device.runtime_constants.persistent_forward_buffer_bytes` is missing",
            ),
        ],
        DERATES_ASKED: [
            (
                validate(merge(fragments(tier1=no_derate)).document),
                Rule.DERATE,
                "`device.memory.derate` is missing, and its block states a spec peak",
            )
        ],
        WIDTHS_ASKED: [
            (
                validate(merged(), tp_widths=(16,)),
                Rule.NO_DEFAULTS,
                "was not measured at tensor-parallel width 16",
            )
        ],
        PROBES_ASKED: [
            (
                validate(merge(fragments(tier1=no_probe)).document, tp_widths=(1,)),
                Rule.NO_DEFAULTS,
                (
                    "no probe here fills `allocator_retained_after_load_bytes` "
                    "at tensor-parallel width 1"
                ),
            )
        ],
        STACK_ASKED: [
            (moved, Rule.PINNED_STACK, "this spec's constants were measured against")
        ],
        TRANSFERS_ASKED: [
            (
                validate(transferred),
                Rule.PINNED_STACK,
                "measured against rocm '7.0.2', into a spec pinned to rocm '7.2.4'",
            )
        ],
    }


def test_the_check_set_names_every_condition_a_spec_can_be_refused_by():
    # The check set is what a result's `not_asked` is measured against and what
    # the count of conditions the package says it reaches is a count of. A
    # condition earnable in the code and absent from the list is one no result
    # reports on; one listed and earnable by nothing is a count with no subject.
    assert set(refused_by_condition()) == set(CONDITIONS)
    # And the package's own statement of reach is drawn from the same list, so
    # it cannot name a condition no result reports on either.
    assert set(ASKABLE_OF_A_DOCUMENT) <= set(CONDITIONS)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_each_condition_in_the_check_set_is_earned_by_a_spec(condition):
    for checked, rule, written in refused_by_condition()[condition]:
        assert not checked.ok, condition
        # The rule and the text on one refusal: the rule alone is shared by
        # conditions, and the text alone could sit on a refusal of another rule.
        assert any(
            refusal.rule is rule and written in refusal.what
            for refusal in checked.refusals
        ), (condition, [str(refusal) for refusal in checked.refusals])


def test_a_document_reaches_what_the_package_says_a_document_reaches():
    # A document reaches less than a `Merge` does: a transfer's source pin is
    # in no field of a document however it was built. What the package states
    # it reaches is a value, and this holds a run's own record to that value
    # rather than to a sentence -- so a condition becoming askable of a
    # document, one ceasing to be, or one added to the check set moves a test
    # instead of going stale in prose.
    checked = validate(merged().document, tp_widths=(1, 2, 4, 8), observed_stack=STACK)
    assert checked.ok
    unasked = {condition.split(" -- ")[0] for condition in checked.not_asked}
    assert set(CONDITIONS) - unasked == set(ASKABLE_OF_A_DOCUMENT)
    assert unasked == {TRANSFERS_ASKED}
    assert len(CONDITIONS) - len(checked.not_asked) == len(ASKABLE_OF_A_DOCUMENT)


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
        if field.kind is Kind.WIDTH_TABLE:
            assert {row.path for row in basis.contributions} == {
                f"{path}[{width}]" for width in spec.values[path]
            }, path
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
            if field.kind is not Kind.WIDTH_TABLE:
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
        assert row.supplied_by == ("tier0",), row.path


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


def test_a_quantity_names_the_fields_it_is_built_out_of():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "kv_blocks", tp_width=8, origin=combination)
    paths = [row.path for row in basis.contributions]
    assert "device.memory.capacity_bytes" in paths
    assert "device.runtime_constants.driver_and_collective_reserve_bytes[8]" in paths
    assert len(paths) == len(set(paths))


def test_a_measured_activation_entry_is_part_of_the_kv_blocks_basis():
    entry = {
        "id": "m",
        "fingerprint": "sha256:f",
        "applies_to": ["A"],
        "bytes_per_token": {1: 180480},
    }
    document = copy.deepcopy(merged().document)
    document["device"]["activations"] = [entry]
    basis = explain(MachineSpec.from_mapping(document), "kv_blocks", tp_width=1)
    rows = {row.path: row.value for row in basis.contributions}
    assert list(rows["device.activations"]) == [entry]
    without = explain(resolved(), "kv_blocks", tp_width=1)
    assert "device.activations" not in [row.path for row in without.contributions]


def test_a_quantity_names_the_fragment_each_field_came_from():
    combination = merged()
    spec = MachineSpec.from_mapping(combination.document)
    basis = explain(spec, "kv_blocks", tp_width=8, origin=combination)
    supplied = {row.path: row.supplied_by for row in basis.contributions}
    assert supplied["device.memory.capacity_bytes"] == ("tier1",)
    assert supplied[
        "device.runtime_constants.driver_and_collective_reserve_bytes[8]"
    ] == ("tier2",)


def test_without_a_merge_a_basis_still_reads_but_names_no_fragment():
    basis = explain(resolved(), "device.memory.capacity_bytes")
    assert basis.contributions[0].supplied_by == ()


def test_a_width_keyed_constant_explains_at_every_measured_width():
    basis = explain(
        resolved(), "device.runtime_constants.driver_and_collective_reserve_bytes"
    )
    assert [row.path.rsplit("[", 1)[-1] for row in basis.contributions] == [
        "1]",
        "2]",
        "4]",
        "8]",
    ]


def test_explaining_at_a_width_nobody_measured_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        explain(resolved(), "kv_blocks", tp_width=16)
    assert refused.value.rule is Rule.NO_DEFAULTS


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


def test_every_quantity_is_built_out_of_fields_this_schema_really_has():
    for term, paths in QUANTITIES.items():
        for path in paths:
            assert path in BY_PATH, f"{term} names {path}, which is not a field"


def test_a_term_the_schema_does_not_know_is_asked_for_again_by_path():
    assert "device.clock_ceiling" not in BY_PATH
    with pytest.raises(SpecRefusal) as refused:
        explain(resolved(), "device.clock_ceiling")
    assert refused.value.rule is Rule.ADDRESSING
    assert refused.value.what.startswith(
        "`device.clock_ceiling` is not a field, a block of fields or a quantity"
    )
    assert "ask again by the whole dotted path" in refused.value.remedy
    assert "kv_blocks" in refused.value.remedy
    assert "declared by this schema" not in str(refused.value)


def test_an_optional_field_the_document_left_out_is_refused_as_absent():
    # The path is right, so a remedy that says to ask again would send the
    # reader round in a circle. The spec reads without `provenance.notes`
    # because the schema declares it optional, and `explain` says exactly what
    # the accessor says about the same field.
    spec = resolved()
    assert not BY_PATH["provenance.notes"].required
    assert "provenance.notes" not in spec.values
    with pytest.raises(SpecRefusal) as refused:
        explain(spec, "provenance.notes")
    with pytest.raises(SpecRefusal) as accessed:
        spec.value("provenance.notes")
    assert refused.value.rule is Rule.TOTALITY
    assert "is declared by this schema as optional" in refused.value.what
    assert "write it in the document" in refused.value.remedy
    assert "ask again" not in refused.value.remedy
    assert str(refused.value) == str(accessed.value)


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
    assert str(basis).splitlines() == [
        f"host.tokenizers, from spec {spec.digest()}",
        "  host.tokenizers = ()",
    ]


def test_an_empty_tokenizer_table_is_printed_as_empty_under_a_quantity():
    # Three rows from the other fields and none from the table. Without its own
    # line the table reads as a field the quantity does not use.
    document = copy.deepcopy(merged().document)
    document["host"]["tokenizers"] = []
    basis = explain(MachineSpec.from_mapping(document), "admission")
    assert len(basis.contributions) == 3
    printed = str(basis).splitlines()
    assert len(printed) == 5
    assert printed[-1] == "  host.tokenizers = ()"


def assembled_without(*paths):
    """The resolved spec assembled from parts, with the named fields left out."""
    whole = resolved()
    return MachineSpec(
        values={p: v for p, v in whole.values.items() if p not in paths},
        tokenizers=whole.tokenizers,
    )


def test_a_quantity_missing_one_field_is_refused_rather_than_explained_from_the_rest():
    # The other six fields still explain to twelve rows, which read as a whole
    # basis for a number that cannot be computed. The refusal is the accessor's.
    spec = assembled_without("device.memory.capacity_bytes")
    with pytest.raises(SpecRefusal) as refused:
        explain(spec, "kv_blocks")
    with pytest.raises(SpecRefusal) as accessed:
        spec.value("device.memory.capacity_bytes")
    assert refused.value.rule is Rule.TOTALITY
    assert str(refused.value) == str(accessed.value)


def test_every_missing_field_under_a_quantity_is_named_and_not_only_the_first():
    spec = assembled_without("host.admission_fixed_s", "host.ipc.shm_broadcast_s")
    with pytest.raises(SpecRefusal) as refused:
        explain(spec, "admission")
    assert refused.value.rule is Rule.TOTALITY
    assert refused.value.what.startswith(
        "`host.admission_fixed_s`, `host.ipc.shm_broadcast_s` are declared by "
        "this schema, and this spec carries no value for them"
    )
    assert "fragments that measure these fields" in refused.value.remedy


def test_a_block_an_assembled_spec_holds_nothing_under_names_every_field():
    # A spec built from parts can lack required fields too, and then the
    # absent field is required, so the remedy is in whatever assembled it.
    spec = assembled_without("host.ipc.zmq_roundtrip_s", "host.ipc.shm_broadcast_s")
    with pytest.raises(SpecRefusal) as refused:
        explain(spec, "host.ipc")
    assert refused.value.rule is Rule.TOTALITY
    assert "`host.ipc.zmq_roundtrip_s`, `host.ipc.shm_broadcast_s` are declared" in (
        refused.value.what
    )
    assert "merge the fragments" in refused.value.remedy
    assert "optional" not in refused.value.what


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
    assert printed[1].endswith("[tier0]")


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
    readings = [reading] * (tp_width - 1) + [reading + spread]
    measured = across_ranks("non_torch", tp_width, readings)
    assert measured.minimum == reading
    assert measured.spread == spread
    assert measured.relative == spread / reading


def test_the_reading_taken_is_the_smallest_whatever_order_the_ranks_came_in():
    high = across_ranks(
        "non_torch", 4, [7458 * MIB, 7266 * MIB, 7266 * MIB, 7266 * MIB]
    )
    assert high.minimum == 7266 * MIB


def test_a_neighbour_holding_fifty_times_the_rank_is_refused():
    # The case this exists for: a rank had reserved 2.94 GB and the card read
    # 152.01 GB, and six runs died on a negative memory budget before anything
    # compared the ranks.
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 4, [2.94e9, 2.94e9, 2.94e9, 152.01e9])
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "2940000000.0" in refused.value.what
    assert "152010000000.0" in refused.value.what


def test_the_spread_is_reported_and_not_folded_into_the_reading():
    measured = across_ranks("non_torch", 8, [10704 * MIB] * 7 + [11344 * MIB])
    assert "spread" in str(measured) and "6.0%" in str(measured)


def test_a_rank_that_was_not_read_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 8, [10704 * MIB] * 7)
    assert refused.value.rule is Rule.RANK_AGREEMENT
    assert "7 reading(s) for the 8 ranks" in refused.value.what


def test_a_reading_no_spec_could_hold_is_refused_where_the_ranks_are():
    with pytest.raises(SpecRefusal) as refused:
        across_ranks("non_torch", 2, [0, 970.0e6])
    assert refused.value.rule is Rule.SHAPE


def test_how_quiet_the_machine_has_to_be_is_the_callers_judgement():
    readings = [10704 * MIB] * 7 + [11344 * MIB]
    assert across_ranks("non_torch", 8, readings, limit=0.06).relative < 0.06
    with pytest.raises(SpecRefusal):
        across_ranks("non_torch", 8, readings, limit=0.01)


# --- what the package reaches -------------------------------------------------


def test_the_three_verbs_are_reachable_by_name():
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
            method="probed",
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
            method="probed",
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
            method="probed",
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
    combination = merge([elsewhere, fragment("tier0", TIER0, machine="node-18")])
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
        method="probed",
        source="tokenizer-laptop",
        processors=published_topology(tmp_path, cores_physical=8),
    )
    with pytest.raises(SpecRefusal) as refused:
        merge([laptop, fragment("tier0", TIER0, machine="node-18")])
    message = str(refused.value)
    assert refused.value.rule is Rule.ONE_MACHINE
    assert "`host.cpu.cores_physical` is 8 in" in message
    assert "and 96 in" in message
    for named in ("tokenizer-laptop", "tier0", "node-18", "ana"):
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
        method="probed",
        source="tokenizer-node",
        processors=published_topology(tmp_path, cores_physical=96, packages=2),
    )
    combination = merge([here, fragment("tier0", TIER0, machine="node-18")])
    assert combination.document["host"]["cpu"] == {
        "cores_physical": 96,
        "cores_logical": 192,
    }
    assert combination.sources["host.cpu.cores_physical"] == (
        "tokenizer-node",
        "tier0",
    )


def test_the_fragment_states_where_it_is_authored_for_and_how(tmp_path):
    emitted = tokenizer_fragment(
        [TOKENIZER],
        machine="node-18",
        authored_by="ana",
        date="2026-09-22",
        method="probed",
        processors=published_topology(tmp_path, cores_physical=8),
    )
    assert emitted.machine == "node-18"
    assert emitted.method == "probed" and emitted.transferred_from is None
    assert "by ana on 2026-09-22" in emitted.stanza()


def test_how_the_rates_were_obtained_is_the_callers_claim_and_has_no_default(tmp_path):
    # The rates arrive as an argument and nothing here encodes or decodes
    # anything, so what the record claims about their source is the caller's to
    # state. `provenance.method` is what a merge refusal quotes and what an
    # artifact carries as the numbers' source, and a default here would be this
    # function asserting a measurement it did not take, over data it received.
    processors = published_topology(tmp_path, cores_physical=8)
    stated = {
        "machine": "node-18",
        "authored_by": "ana",
        "date": "2026-09-22",
        "processors": processors,
    }
    carried = tokenizer_fragment(
        [TOKENIZER], method="transferred-from:mi300x-8gpu", **stated
    )
    assert carried.method == "transferred-from:mi300x-8gpu"
    assert carried.transferred_from == "mi300x-8gpu"
    with pytest.raises(TypeError):
        tokenizer_fragment([TOKENIZER], **stated)


def test_the_probe_is_reachable_by_name():
    assert "tokenizer_fragment" in spec_package.__all__
    assert spec_package.tokenizer_fragment is tokenizer_fragment


# --- the readings a probe takes off a card -----------------------------------

#: The card the readings below are taken on, and the two numbers a rank reads
#: beside the one the spec wants: what the torch allocator holds, and the
#: budget side of the clamp the engine sizes its cache by.
CAPACITY = 288.0e9
RESERVED = 2.94e9
KV_BUDGET = 92.1e9

#: The reserve the collective terms predict at each measured width, and what a
#: rank really read there: the honest readings sit just above their prediction,
#: and the spread is the one this hardware shows with nothing else on it.
PREDICTED = {1: 970.0e6, 2: 7.2e9, 4: 7.6e9, 8: 11.2e9}
MEASURED = {
    1: (926 * MIB, 0),
    2: (6906 * MIB, 0),
    4: (7266 * MIB, 192 * MIB),
    8: (10704 * MIB, 640 * MIB),
}


def reading(non_torch, *, total=CAPACITY, reserved=RESERVED, kv_budget=KV_BUDGET):
    """One rank's four readings, written as the term under test plus the rest."""
    return DeviceMemory(
        free_bytes=total - reserved - non_torch,
        total_bytes=total,
        reserved_bytes=reserved,
        kv_budget_bytes=kv_budget,
    )


def ranks_of(tp_width, non_torch, spread=0):
    """One engine start: every rank reading the same, and one reading higher."""
    return [reading(non_torch)] * (tp_width - 1) + [reading(non_torch + spread)]


def test_free_and_total_are_kept_apart_and_the_difference_is_derived():
    # The whole point of the four fields: the reading a spec wants is a
    # difference, and a probe that stored only the difference could not be
    # asked either of the questions below.
    one = reading(10704 * MIB)
    assert one.total_bytes == CAPACITY and one.reserved_bytes == RESERVED
    assert one.free_bytes == CAPACITY - RESERVED - 10704 * MIB
    assert one.non_torch == 10704 * MIB
    assert not one.free_was_binding
    assert "free of" in str(one) and "non_torch" in str(one)


@pytest.mark.parametrize("tp_width", sorted(PREDICTED))
def test_the_widest_honest_reading_at_every_width_is_accepted(tp_width):
    # Measured on a machine with nothing else on it, so a check that refused
    # any of these would be refusing the hardware for behaving as it behaves.
    lowest, spread = MEASURED[tp_width]
    measured = non_torch_across_ranks(
        tp_width, ranks_of(tp_width, lowest, spread), PREDICTED[tp_width]
    )
    assert measured.minimum == lowest
    assert measured.spread == spread


def test_the_spread_this_hardware_shows_is_reported_and_not_folded_away():
    # Eight ranks of one group, the widest honest spread this hardware shows,
    # kept as the smallest reading with the disagreement reported beside it
    # rather than reduced to one number.
    measured = non_torch_across_ranks(8, ranks_of(8, 10704 * MIB, 640 * MIB), 11.2e9)
    assert measured.minimum == 10704 * MIB
    assert measured.spread == 640 * MIB
    assert "spread" in str(measured) and "6.0%" in str(measured)


def test_a_card_every_rank_shares_with_one_neighbour_is_refused_by_name():
    # What the cross-rank check cannot see. Every rank reads the neighbour that
    # killed six engine starts, so they agree to the byte and the cross-rank
    # check is silent; the absolute one names the reading and the ceiling.
    crowded = ranks_of(8, 152.01e9)
    agreed = across_ranks("non_torch", 8, [rank.non_torch for rank in crowded])
    assert agreed.spread == 0
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(8, crowded, 11.2e9)
    assert refused.value.rule is Rule.DEVICE_WIDE
    assert "152010000000.0" in refused.value.what
    assert "22400000000.0" in refused.value.what
    assert "11200000000.0" in refused.value.what and "2x" in refused.value.what


def test_the_limit_is_a_multiple_of_what_the_width_predicts():
    # It is the prediction for the width that moves, not the reading: the same
    # number is a quiet card at width 8 and eleven times its prediction at 1.
    assert non_torch_across_ranks(8, ranks_of(8, 10704 * MIB), 11.2e9).minimum
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(1, ranks_of(1, 10704 * MIB), 970.0e6)
    assert refused.value.rule is Rule.DEVICE_WIDE


def test_how_crowded_the_card_may_be_is_the_callers_judgement():
    # 16.0e9 is 2.1 times what width 4 predicts: inside a limit of 2.5 and
    # outside the default of 2.0, so the judgement is the only thing moving.
    crowded = ranks_of(4, 16.0e9)
    assert non_torch_across_ranks(4, crowded, 7.6e9, limit=2.5).minimum == 16.0e9
    with pytest.raises(SpecRefusal):
        non_torch_across_ranks(4, crowded, 7.6e9)


def test_a_rank_whose_free_memory_was_binding_is_refused_and_named():
    # A cache sized by the gap a neighbour left is a property of the
    # neighbour, so the rank that read it is named while it is still in hand.
    binding = ranks_of(2, 200.0e9)
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(2, binding, 7.2e9)
    assert refused.value.rule is Rule.DEVICE_WIDE
    assert "rank 0" in refused.value.what
    assert "92100000000.0" in refused.value.what


def test_which_refusal_a_pair_of_readings_earns_is_not_decided_by_rank_order():
    # One rank read numbers no card produced; the other built its cache out of
    # the gap a neighbour left. Each fails a different one of the two per-rank
    # checks, so the pair earns both -- and the two name different things to
    # repair: a reading to take again, against a card to take it on. Every rank
    # is asked the first question before any is asked the second, so the pair
    # earns the same refusal whichever of them arrived first. Asked rank by
    # rank instead, whichever was listed first would decide the remedy, and a
    # run sent after the wrong one repairs it and is refused again.
    binding = reading(200.0e9)
    impossible = DeviceMemory(
        free_bytes=400.0e9,
        total_bytes=CAPACITY,
        reserved_bytes=-120.0e9,
        kv_budget_bytes=1.0,
    )
    assert binding.free_was_binding and not binding.impossible
    assert impossible.impossible and not impossible.free_was_binding
    for ranks, first_to_fail in (
        ([binding, impossible], "rank 1"),
        ([impossible, binding], "rank 0"),
    ):
        with pytest.raises(SpecRefusal) as refused:
            non_torch_across_ranks(2, ranks, 7.2e9)
        assert refused.value.rule is Rule.DEVICE_WIDE
        assert "which is not a reading of a card" in refused.value.what
        assert "set the cache size" not in refused.value.what
        assert "take them again and find out" in refused.value.remedy
        # Which check fired is the readings'; the rank it names is still the
        # first that failed that check, which is a question about where in the
        # sequence the reading came in and has no other answer.
        assert first_to_fail in refused.value.what


def test_a_reading_the_engine_would_floor_at_zero_is_refused_here():
    # The engine floors this term at zero. A probe does not: a floor turns an
    # impossible reading into a plausible one.
    impossible = reading(-1.0 * MIB, kv_budget=1.0)
    assert impossible.non_torch < 0
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(2, [impossible, impossible], 7.2e9)
    assert refused.value.rule is Rule.SHAPE


def test_a_prediction_that_is_not_a_quantity_is_refused():
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(8, ranks_of(8, 10704 * MIB), 0)
    assert refused.value.rule is Rule.SHAPE
    assert "the predicted reserve" in refused.value.what


def test_a_rank_that_was_not_read_is_refused_before_anything_is_kept():
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(8, ranks_of(4, 10704 * MIB), 11.2e9)
    assert refused.value.rule is Rule.RANK_AGREEMENT


@pytest.mark.parametrize(
    "overrides, named",
    [
        ({"total_bytes": 0.0}, "a card of 0.0 bytes"),
        ({"free_bytes": -1.0}, "-1.0 bytes free"),
        ({"free_bytes": 400.0e9}, "400000000000.0 bytes free of 288000000000.0"),
        ({"reserved_bytes": -120.0e9}, "-120000000000.0 bytes reserved"),
        (
            {"reserved_bytes": 400.0e9},
            "400000000000.0 bytes reserved of 288000000000.0",
        ),
    ],
)
def test_a_rank_whose_readings_cannot_be_one_card_is_refused_and_named(
    overrides, named
):
    # Keeping the four apart is what makes this askable at all: each of these
    # is a statement about one reading against another, and the difference the
    # spec wants has already thrown the comparison away.
    fields = {
        "free_bytes": CAPACITY - RESERVED - 10704 * MIB,
        "total_bytes": CAPACITY,
        "reserved_bytes": RESERVED,
        "kv_budget_bytes": 1.0,
    }
    odd = DeviceMemory(**dict(fields, **overrides))
    assert odd.impossible == named
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(2, [odd, odd], 7.2e9)
    assert refused.value.rule is Rule.DEVICE_WIDE
    assert "rank 0" in refused.value.what
    assert named in refused.value.what


def test_readings_no_card_could_produce_are_refused_though_their_difference_is_not():
    # 400e9 free on a 288e9 card with -120e9 reserved leaves a difference of
    # 8.0e9, which is a quiet width-2 card by every check over the difference
    # alone. The readings behind it are not a card.
    odd = DeviceMemory(
        free_bytes=400.0e9,
        total_bytes=CAPACITY,
        reserved_bytes=-120.0e9,
        kv_budget_bytes=1.0,
    )
    assert odd.non_torch == 8.0e9
    assert not odd.free_was_binding
    assert non_torch_across_ranks(2, [reading(8.0e9)] * 2, 7.2e9).minimum == 8.0e9
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(2, [odd, odd], 7.2e9)
    assert refused.value.rule is Rule.DEVICE_WIDE
    # These readings are two impossibilities, and the negative reserve would
    # be refused by the same rule on its own. What is named is the free memory
    # the card cannot hold, so a check that stopped asking that is seen here
    # rather than covered by its neighbour.
    assert "rank 0 reports 400000000000.0 bytes free of 288000000000.0" in (
        refused.value.what
    )
    assert "take them again and find out" in refused.value.remedy


def test_a_reading_just_inside_the_limit_is_accepted_and_is_what_the_spec_takes():
    # What the check lets through is the number a spec carries. Just inside the
    # limit at width 8 is nearly twice the prediction, and the whole of the
    # excess is memory this engine did not allocate.
    inside = non_torch_across_ranks(8, ranks_of(8, 22.39e9), 11.2e9)
    assert inside.minimum == 22.39e9
    assert inside.minimum - 11.2e9 > 11.1e9
    with pytest.raises(SpecRefusal):
        non_torch_across_ranks(8, ranks_of(8, 22.41e9), 11.2e9)


@pytest.mark.parametrize("tp_width", [0, -2])
def test_a_width_with_no_ranks_is_refused_rather_than_reduced(tp_width):
    # The width reaches this entry point from a caller, not from a document
    # the schema has already refused a non-positive key in.
    with pytest.raises(SpecRefusal) as refused:
        non_torch_across_ranks(tp_width, [], 7.2e9)
    assert refused.value.rule is Rule.RANK_AGREEMENT
    # The cross-rank reduction refuses a negative width by the same rule, for
    # having no reading per rank, and names the width too. The text below is
    # the width's own refusal, which is asked before any reading is counted.
    assert (
        f"tensor-parallel width {tp_width!r} has no ranks to read a card on"
        in refused.value.what
    )


def test_the_readings_are_reachable_by_name():
    for name in ("DeviceMemory", "non_torch_across_ranks", "ABSOLUTE_LIMIT"):
        assert name in spec_package.__all__
        assert getattr(spec_package, name, None) is not None


# --- which probe fills which constant, and the entry none fills ---------------


def test_the_probe_table_names_every_constant_the_schema_keys_by_width():
    # The table and the schema are two lists of the same constants. A width
    # table added to one and not the other is a term with no probe named for
    # it, or a probe named for a term the schema does not have.
    assert set(FILLED_BY) == {path.rsplit(".", 1)[-1] for path in WIDTH_TABLES}


@pytest.mark.parametrize("tp_width", [1, 2, 4, 8, 16])
def test_the_reserve_is_filled_at_every_width_by_one_probe_or_the_other(tp_width):
    filled = probe_for("driver_and_collective_reserve_bytes", tp_width)
    assert filled == ("device-memory" if tp_width == 1 else "device-runtime-constants")


@pytest.mark.parametrize("tp_width", [2, 4, 8, 16])
def test_the_allocators_retained_bytes_are_filled_above_one_card(tp_width):
    assert probe_for("allocator_retained_after_load_bytes", tp_width) == (
        "device-runtime-constants"
    )


def test_the_one_entry_no_probe_fills_is_refused_and_says_what_is_missing():
    # The hole. The single-card probe is specified to read the device's own
    # reserve and not this term, and the multi-rank probe is specified for the
    # widths above one, so this entry is measured by hand or it is not
    # measured. Naming either probe here would be inventing a capability.
    with pytest.raises(SpecRefusal) as refused:
        probe_for("allocator_retained_after_load_bytes", 1)
    assert refused.value.rule is Rule.NO_DEFAULTS
    assert "no probe here fills" in refused.value.what
    assert "allocator_retained_after_load_bytes" in refused.value.what
    assert "width 1" in refused.value.what
    assert "by hand" in refused.value.remedy
    for named in ("device-memory", "device-runtime-constants"):
        assert named in refused.value.remedy


def test_a_term_that_is_not_keyed_by_width_has_no_probe_to_ask_for():
    with pytest.raises(SpecRefusal) as refused:
        probe_for("persistent_forward_buffer_bytes", 1)
    assert refused.value.rule is Rule.NO_DEFAULTS
    assert "not one of the constants measured per" in refused.value.what


def test_a_missing_entry_no_probe_fills_is_two_refusals_and_not_one():
    # A width nobody measured is always refused; this one is refused twice,
    # because the remedy the first offers -- go and measure it -- names a probe
    # that does not exist. The two always arrive together: the probe question
    # is asked only where the entry is absent, which is when the first fires.
    thin = without(
        TIER1, "device", "runtime_constants", "allocator_retained_after_load_bytes"
    )
    checked = validate(merge(fragments(tier1=thin)).document, tp_widths=(1,))
    assert not checked.ok
    assert [refusal.rule for refusal in checked.refusals] == [
        Rule.NO_DEFAULTS,
        Rule.NO_DEFAULTS,
    ]
    measured, filled = (refusal.what for refusal in checked.refusals)
    assert "was not measured at tensor-parallel width 1" in measured
    assert "no probe here fills" in filled


def test_a_width_a_probe_would_fill_is_not_reported_as_a_hole_in_the_probes():
    # Width 16 is unmeasured and refused, and the remedy is one an author can
    # follow: the multi-rank probe starts an engine at any width above one.
    checked = validate(merged().document, tp_widths=(16,))
    assert not checked.ok
    assert all(
        "no probe here fills" not in refusal.what for refusal in checked.refusals
    )


def test_a_spec_that_carries_the_hand_measured_entry_is_not_refused_for_it():
    # The refusal is about the hole in the tools, not about how the number was
    # obtained. A spec that states it is a good spec at width 1.
    checked = validate(merged().document, tp_widths=(1,))
    assert checked.ok


def test_the_probe_question_is_asked_of_the_tables_it_says_it_reads(monkeypatch):
    # The condition registers `PROBE_TABLES` as what it reads, and the walk that
    # produces its refusals has to be over the same tables. Walking every width
    # table instead changes nothing today -- the reserve table is filled at
    # every width, so asking about it never refuses -- which is why the walk is
    # observed here by what it asks rather than by what it answers.
    asked = []

    def recording(constant, tp_width):
        asked.append((constant, tp_width))
        return probe_for(constant, tp_width)

    monkeypatch.setattr(
        importlib.import_module("atom.compass.spec.validate"), "probe_for", recording
    )
    checked = validate(merged().document, tp_widths=(16,))
    # Width 16 is missing from every width table, so a walk over any of them
    # would reach the probe; the width question says so once per table.
    unmeasured = [
        refusal
        for refusal in checked.refusals
        if "was not measured at tensor-parallel width 16" in refusal.what
    ]
    assert len(unmeasured) == len(WIDTH_TABLES) > len(PROBE_TABLES)
    assert asked, "the probe question asked nothing"
    assert asked == [(path.rsplit(".", 1)[-1], 16) for path in PROBE_TABLES]


@pytest.mark.parametrize("tp_width", [0, -3])
def test_a_width_below_one_card_is_refused_rather_than_handed_a_probe(tp_width):
    # Every width that was not exactly one fell to the multi-rank probe, zero
    # and the negatives with it. No engine starts on that many cards, so this
    # is a wrong question and not a term nobody has measured yet.
    for constant in sorted(FILLED_BY):
        with pytest.raises(SpecRefusal) as refused:
            probe_for(constant, tp_width)
        assert refused.value.rule is Rule.NO_DEFAULTS
        assert f"width {tp_width} is not a number of cards" in refused.value.what


def test_the_check_does_not_agree_a_probe_exists_for_a_width_below_one():
    # Reachable from the verb, which is where a width arrives from the caller
    # rather than from a document the schema has already refused it in.
    checked = validate(merged().document, tp_widths=(0,))
    assert not checked.ok
    assert any(
        "is not a number of cards" in refusal.what for refusal in checked.refusals
    )


def test_the_probe_question_reads_only_the_tables_a_probe_can_fall_short_on():
    # The reserve table is filled at every width by one probe or the other, so
    # the question can never say anything about it. Registering it as read
    # would make a run that could not ask the question and a run that asked it
    # and found nothing report the same record.
    assert set(PROBE_TABLES) <= set(WIDTH_TABLES)
    assert {path.rsplit(".", 1)[-1] for path in PROBE_TABLES} == {
        name for name, probes in FILLED_BY.items() if None in probes
    }


#: The one width table a probe falls short on, the retained bytes at width 1.
THE_HOLE = "device.runtime_constants.allocator_retained_after_load_bytes"


def reimported_validate(name):
    """`validate` imported again, the way a session imports it the first time.

    The probe tables are derived once, while the module is being imported, so a
    test about that derivation has to import the module rather than call
    something in it. This loads a second instance out of the same file under a
    name of its own, and does not register it, so the instance the rest of the
    suite is holding is the one it started with.
    """
    location = importlib.util.find_spec("atom.compass.spec.validate").origin
    loaded = importlib.util.spec_from_file_location(
        f"atom.compass.spec.{name}", location
    )
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    assert module not in sys.modules.values()
    return module


def test_a_width_table_no_probe_is_named_for_is_refused_and_not_an_import_error(
    monkeypatch,
):
    # A width-keyed constant written into the schema and not into the probe
    # table is the ordinary shape of a half-finished change, and the two lists
    # are held together by a test for exactly that reason. The derivation runs
    # while the module is being imported, so a table it could not answer for
    # would deny every caller of the package instead -- and that test would
    # fail at collection, as an import error naming the test session rather
    # than the term nobody entered.
    added = Field("device.runtime_constants.graph_replay_pool_bytes", Kind.WIDTH_TABLE)
    monkeypatch.setattr(schema_module, "SCHEMA", schema_module.SCHEMA + (added,))
    under_test = reimported_validate("validate_with_a_table_no_probe_is_named_for")
    assert added.path in under_test.WIDTH_TABLES
    # A table with no entry has no hole to report, so it is not a probe table,
    # and the table that has one still is.
    assert under_test.PROBE_TABLES == (THE_HOLE,)
    # The term is named where a caller asks about it, by the refusal this
    # package exists to give.
    with pytest.raises(SpecRefusal) as refused:
        probe_for("graph_replay_pool_bytes", 1)
    assert refused.value.rule is Rule.NO_DEFAULTS
    assert "`graph_replay_pool_bytes` is not one of the constants" in refused.value.what
    # And what the test holding the two lists together now sees is its own
    # comparison, failing on the term that is missing from the table.
    assert set(FILLED_BY) != {
        path.rsplit(".", 1)[-1] for path in under_test.WIDTH_TABLES
    }


def test_a_probe_given_the_width_that_has_none_empties_the_probe_tables(monkeypatch):
    # The property the derivation exists for, and the one a second list would
    # lose: give the hole a probe and the question has nothing left to ask
    # about, with nothing else to edit. The run still counts it as asked -- it
    # is not a question this run could not reach -- and it finds nothing to say.
    monkeypatch.setitem(
        FILLED_BY,
        "allocator_retained_after_load_bytes",
        (probes_module.SINGLE_CARD, probes_module.MULTI_RANK),
    )
    under_test = reimported_validate("validate_with_every_width_filled")
    # Emptied by the probe, and not empty before it.
    assert (PROBE_TABLES, under_test.PROBE_TABLES) == ((THE_HOLE,), ())
    checked = under_test.validate(merged().document, tp_widths=(16,))
    assert not checked.ok
    assert not any(
        condition.startswith(PROBES_ASKED) for condition in checked.not_asked
    )
    assert not any(
        condition.startswith(PROBES_ASKED) for condition in checked.asked_in_part
    )


def test_the_probe_question_says_it_could_not_be_asked_when_its_table_is_gone():
    # The other width table still resolves, so the width question was asked of
    # part of what it reads. The probe question was not asked at all, and the
    # count of what this run reached must not include it.
    absent = ("device", "runtime_constants", "allocator_retained_after_load_bytes")
    checked = validate(
        merge(
            fragments(tier1=without(TIER1, *absent), tier2=without(TIER2, *absent))
        ).document,
        tp_widths=(16,),
    )
    (unasked,) = [
        condition
        for condition in checked.not_asked
        if condition.startswith(PROBES_ASKED)
    ]
    assert "could not be asked at all" in unasked
    assert not any(
        condition.startswith(PROBES_ASKED) for condition in checked.asked_in_part
    )


def test_the_probe_table_is_reachable_by_name():
    assert "probe_for" in spec_package.__all__
    assert spec_package.probe_for is probe_for


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
        method="probed",
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
    assert validate(merged, tp_widths=(1,)).ok
    measured, declared = script.report(merged, {"tokenizer", "device"}).split(
        "declared:"
    )
    for field in (
        "device.memory.capacity_bytes",
        "driver_and_collective_reserve_bytes[1]",
        "allocator_retained_after_load_bytes[1]",
        "device.runtime_constants.persistent_forward_buffer_bytes",
        "device.activations",
        "encode_tokens_per_s",
    ):
        assert field in measured and field not in declared.split("kv_blocks")[0]
    for field in ("host.ipc.zmq_roundtrip_s", "host.admission_fixed_s", "w1_base"):
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


def test_the_checked_in_node_18_spec_validates_and_prices_its_activations():
    from atom.compass.run import _width_keys

    script = _node18_script()
    path = SPEC_ROOT / "scripts" / "compass" / "machines" / f"{script.NAME}.json"
    document = _width_keys(json.loads(path.read_text()))
    spec = validate(document, tp_widths=(1,)).raise_first()
    for fragment in script.device_fragments():
        for path, value in fragment.values.items():
            if not path.startswith("provenance.") and path != "name":
                assert spec.values[path] == value, path
    assert spec.activations_for(
        "Qwen3_5ForConditionalGeneration", script.CONFIG_JSON, 1
    ) == ("qwen3.8-27b", 180480.0)
