"""Where each field of a machine spec came from, and that it survives a save.

A spec is merged from fragments of different origins, written to a file and
read back, sometimes merged again with next week's probe. Each test here goes
through that save: what a merge knew and did not write down is what these
check for.
"""

import copy
import importlib.util
import json
import pathlib

import pytest
from test_spec_schema import DOCUMENT
from test_spec_verbs import LINKS, STACK, TIER0, TIER1, TIER2, fragment

from atom.compass.spec import MachineSpec, Rule, SpecRefusal, explain, merge
from atom.compass.spec.fields import SCHEMA, Kind
from atom.compass.spec.machine import entry_label

ROOT = pathlib.Path(__file__).resolve().parents[2]
CARD = {"model": "Qwen/Qwen3.8-27B", "width": 1, "source_root": "commit:983e42973"}


def saved(document):
    """A document as a file holds it: JSON, with width keys read back to ints."""

    def ints(node):
        if isinstance(node, list):
            return [ints(v) for v in node]
        if not isinstance(node, dict):
            return node
        return {int(k) if k.isdigit() else k: ints(v) for k, v in node.items()}

    return ints(json.loads(json.dumps(document)))


def labels(spec):
    """Every field the spec holds, as `origin` is asked for it."""
    for field in SCHEMA:
        path = field.path
        if path in ("name", "schema_version") or path.startswith("provenance."):
            continue
        if path not in spec.values:
            continue
        if field.kind is Kind.WIDTH_TABLE:
            yield from ((path, width) for width in spec.values[path])
        elif field.kind is Kind.TOKENIZERS:
            for e in spec.tokenizers.entries:
                yield entry_label(e.key.id, e.key.backend.value), None
        else:
            yield path, None


def four_origins():
    """Probed, transferred and assumed fragments of one machine."""
    carried = copy.deepcopy(TIER2)
    carried["device"]["software_pinned_to"] = dict(STACK)
    return [
        fragment("tier0", TIER0, subject={"model": "Qwen/Qwen3-8B"}),
        fragment("tier1", TIER1, subject=CARD),
        fragment("tier2", carried, origin="transferred-from:mi300x-8gpu"),
        fragment("links", LINKS, origin="assumed"),
    ]


def test_a_saved_spec_answers_origin_for_every_field():
    spec = MachineSpec.from_mapping(saved(merge(four_origins()).document))
    assert spec.value("provenance.origin") == "mixed"
    answered = {label: spec.origin(*label)["origin"] for label in labels(spec)}
    assert set(answered.values()) == {
        "probed",
        "transferred-from:mi300x-8gpu",
        "assumed",
    }
    reserve = "device.runtime_constants.driver_and_collective_reserve_bytes"
    assert spec.origin(reserve, 1) == {"origin": "probed", **CARD}
    assert spec.origin(reserve, 8) == {"origin": "transferred-from:mi300x-8gpu"}
    assert spec.origin("interconnect.intra_node.derate") == {"origin": "assumed"}
    assert spec.origin(entry_label("qwen3-151k-bpe", "fast")) == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


def test_a_field_two_fragments_agree_on_is_assumed_only_if_both_assumed_it():
    links = fragment("links", LINKS, origin="assumed")
    probed = fragment("probe", {"interconnect": {"router_relay_s": 1.5e-3}})
    document = saved(merge([*four_origins()[:3], probed, links]).document)
    spec = MachineSpec.from_mapping(document)
    assert spec.origin("interconnect.router_relay_s") == {"origin": "probed"}
    assert spec.origin("interconnect.inter_node.link_latency_s") == {
        "origin": "assumed"
    }


def test_a_saved_spec_merged_again_keeps_each_fields_origin():
    first = saved(merge(four_origins()).document)
    week_later = {"interconnect": {"inter_node": {"link_latency_s": 5.0e-6}}}
    again = merge(
        [
            fragment("machine.json", {k: v for k, v in first.items() if k != "name"}),
            fragment("probe", week_later, date="2026-09-25"),
        ]
    )
    spec = MachineSpec.from_mapping(saved(again.document))
    assert spec.origin("interconnect.inter_node.link_latency_s") == {"origin": "probed"}
    assert spec.origin("interconnect.intra_node.link_latency_s") == {
        "origin": "assumed"
    }
    reserve = "device.runtime_constants.driver_and_collective_reserve_bytes"
    assert spec.origin(reserve, 4) == {"origin": "transferred-from:mi300x-8gpu"}
    assert spec.origin(reserve, 1) == {"origin": "probed", **CARD}


def test_a_mixed_document_without_its_map_is_refused_by_name():
    document = saved(merge(four_origins()).document)
    del document["provenance"]["origins"]
    spec = MachineSpec.from_mapping(document)
    with pytest.raises(SpecRefusal) as refused:
        spec.origin("device.memory.capacity_bytes")
    assert refused.value.rule is Rule.TOTALITY
    assert "records no origin for `device.memory.capacity_bytes`" in str(refused.value)
    with pytest.raises(SpecRefusal, match="records no origin"):
        explain(spec, "kv_blocks", tp_width=1)
    with pytest.raises(SpecRefusal, match="records no origin"):
        merge(
            [fragment("machine.json", {"device": document["device"]}, origin="mixed")]
        )


def test_a_document_of_one_origin_answers_without_the_map():
    stanza = dict(DOCUMENT["provenance"], subject={"model": "Qwen/Qwen3-8B"})
    spec = MachineSpec.from_mapping(dict(DOCUMENT, provenance=stanza))
    assert "provenance.origins" not in spec.values
    assert spec.origin("host.admission_fixed_s") == {
        "origin": "probed",
        "model": "Qwen/Qwen3-8B",
    }


@pytest.mark.parametrize(
    "stated", ["guessed", "transferred-from:", "transferred-from:  ", "", 7]
)
def test_an_origin_nobody_defined_is_refused(stated):
    with pytest.raises(SpecRefusal, match="which is not an origin"):
        fragment("tier0", TIER0, origin=stated)


def test_one_field_is_never_mixed_and_only_a_probed_one_has_a_subject():
    document = saved(merge(four_origins()).document)
    origins = document["provenance"]["origins"]
    for entry, wanted in (
        ({"origin": "mixed"}, "which is not an origin"),
        ({"origin": "assumed", "model": "m"}, "a probed field"),
        ({"origin": "probed", "rack": "r7"}, "a mapping of some of"),
    ):
        broken = copy.deepcopy(document)
        broken["provenance"]["origins"] = dict(
            origins, **{"host.admission_fixed_s": entry}
        )
        with pytest.raises(SpecRefusal, match=wanted):
            MachineSpec.from_mapping(broken)
    with pytest.raises(SpecRefusal, match="state it on a probed document only"):
        fragment("links", LINKS, origin="assumed", subject={"model": "m"})


def test_the_old_key_is_refused_naming_the_new_one():
    document = saved(merge(four_origins()).document)
    document["provenance"]["method"] = document["provenance"].pop("origin")
    with pytest.raises(
        SpecRefusal, match="`provenance.method` is now `provenance.origin`"
    ):
        MachineSpec.from_mapping(document)


def test_origin_is_asked_at_a_measured_width_and_per_tokenizer_entry():
    spec = MachineSpec.from_mapping(saved(merge(four_origins()).document))
    reserve = "device.runtime_constants.driver_and_collective_reserve_bytes"
    with pytest.raises(SpecRefusal, match="none was given"):
        spec.origin(reserve)
    with pytest.raises(
        SpecRefusal, match="was not measured at tensor-parallel width 3"
    ):
        spec.origin(reserve, 3)
    for path in ("host.tokenizers", "host.tokenizers[nope fast]"):
        with pytest.raises(SpecRefusal, match="not one tokenizer entry"):
            spec.origin(path)
    with pytest.raises(SpecRefusal, match="describes the document"):
        spec.origin("provenance.date")
    with pytest.raises(SpecRefusal, match="not a field of this schema"):
        spec.origin("device.memory.colour")


def test_explain_names_each_contributions_origin():
    combination = merge(four_origins())
    spec = MachineSpec.from_mapping(saved(combination.document))
    rows = {
        row.path: row for row in explain(spec, "kv_blocks", tp_width=1).contributions
    }
    assert rows["device.memory.capacity_bytes"].origin == {"origin": "probed", **CARD}
    assert str(rows["device.memory.capacity_bytes"]).endswith(
        "probed on model Qwen/Qwen3.8-27B, width 1, source_root commit:983e42973"
    )
    stated = {row.path: row.origin for row in explain(spec, "provenance").contributions}
    assert set(stated.values()) == {None}


def _script(monkeypatch, name):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    path = ROOT / "scripts" / "compass" / "mi308x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi308x_spec", path)
    module = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(module)
    return module


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_checked_in_specs_say_where_each_field_came_from(monkeypatch, name):
    script = _script(monkeypatch, name)
    path = ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    spec = MachineSpec.from_mapping(saved(json.loads(path.read_text())))
    answered = {label: spec.origin(*label) for label in labels(spec)}
    runtime = [k for k in answered if k[0].startswith("device.runtime_constants.")]
    probed = runtime + [("device.activations", None)]
    for label in probed:
        if "cudagraph_pool" not in label[0]:
            assert answered[label] == {"origin": "probed", **script.SUBJECT}, label
    for label, found in answered.items():
        if label[0].endswith(".derate"):
            assert found == {"origin": "assumed"}, label
        elif found["origin"] != "probed":
            wanted = (
                "assumed"
                if name == script.SOURCE
                else "transferred-from:mi308x-hjbog18"
            )
            assert found == {"origin": wanted}, label
    assert script.SUBJECT["source_root"] == f"commit:{script.COMMIT}"
