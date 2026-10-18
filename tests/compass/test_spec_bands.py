# SPDX-License-Identifier: MIT
"""A probed number is published with the band its probe measured, and every
reader still reads the one value."""

import json
import math
import pathlib

import pytest
from test_pp_carriers import TRUTH, _samples, probe
from test_spec_origin import labels, saved, three_origins
from test_spec_verbs import TOKENIZER, fragment, published_topology

from atom.compass.artifacts import machine_spec_digest
from atom.compass.spec import (
    MachineSpec,
    SpecRefusal,
    merge,
    tokenizer_fragment,
)
from atom.compass.spec.bands import refuse_unbanded
from atom.compass.spec.machine import entry_label
from atom.compass.spec.rules import Rule

ROOT = pathlib.Path(__file__).resolve().parents[2]
ENTRY = entry_label("qwen3-151k-bpe", "fast")
RATES = (
    "encode_fixed_s",
    "encode_tokens_per_s",
    "decode_fixed_s",
    "decode_tokens_per_s",
)


def _banded(document, low=0.5, high=2.0):
    """`document` with every probed origin given a band around its value."""
    spec = MachineSpec.from_mapping(document)
    for label, found in document["provenance"]["origins"].items():
        if found["origin"] != "probed":
            continue
        if label == ENTRY:
            found["p10"] = {r: TOKENIZER[r] * low for r in RATES}
            found["p90"] = {r: TOKENIZER[r] * high for r in RATES}
        elif isinstance(spec.value(label), int | float):
            found["p10"], found["p90"] = (
                spec.value(label) * low,
                spec.value(label) * high,
            )
    return document


def test_a_probed_number_without_its_band_is_refused_by_name():
    spec = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    with pytest.raises(SpecRefusal) as refused:
        refuse_unbanded(spec)
    assert refused.value.rule is Rule.BAND
    said = refused.value.what
    for owed in ("host.ipc.zmq_roundtrip_s", "device.compute.bf16_flops", ENTRY):
        assert f"`{owed}`" in said
    # Read, not sampled: a count, a text, the capacity; a derate; an assumed link.
    for free in (
        "cores_physical",
        "device.name",
        "capacity_bytes",
        "derate",
        "interconnect",
    ):
        assert free not in said


def test_a_band_beside_each_probed_number_publishes_and_moves_no_value():
    plain = MachineSpec.from_mapping(saved(merge(three_origins()).document))
    banded = MachineSpec.from_mapping(_banded(saved(merge(three_origins()).document)))
    refuse_unbanded(banded)
    for label in labels(plain):
        if label != ENTRY:
            assert banded.value(label) == plain.value(label), label
    assert banded.tokenizers == plain.tokenizers
    assert machine_spec_digest(banded) == machine_spec_digest(plain)
    rate = banded.origin("host.ipc.zmq_roundtrip_s")
    assert (rate["p10"], rate["p90"]) == (2.5e-5, 1.0e-4)


def test_a_saved_band_survives_a_second_merge():
    first = _banded(saved(merge(three_origins()).document))
    body = {k: v for k, v in first.items() if k != "name"}
    again = merge([fragment("machine.json", body)])
    spec = MachineSpec.from_mapping(saved(again.document))
    refuse_unbanded(spec)
    assert spec.origin("device.compute.fp8_flops")["p90"] == 1.0e16


def test_a_tokenizer_entry_states_a_band_for_every_rate(tmp_path):
    def swept(band):
        return tokenizer_fragment(
            [TOKENIZER],
            machine="mi355x-8gpu-2node",
            authored_by="ana",
            date="2026-10-18",
            origin="probed",
            processors=published_topology(tmp_path, cores_physical=96),
            band=band,
        )

    whole = {"p10": {r: TOKENIZER[r] for r in RATES}}
    whole["p90"] = dict(whole["p10"])
    document = saved(merge([swept(whole), *three_origins()]).document)
    assert MachineSpec.from_mapping(document).origin(ENTRY)["p90"] == whole["p90"]
    short = {end: dict(list(rates.items())[:3]) for end, rates in whole.items()}
    document = _banded(document)
    document["provenance"]["origins"][ENTRY] |= short
    with pytest.raises(SpecRefusal) as refused:
        refuse_unbanded(MachineSpec.from_mapping(document))
    assert refused.value.what.startswith(f"`{ENTRY}` probed")


@pytest.mark.parametrize(
    "band",
    [
        {"p10": 2.0, "p90": 1.0},
        {"p10": 1.0},
        {"p10": True, "p90": 2.0},
        {"p10": "1", "p90": 2.0},
        {"p10": {"a": 1.0}, "p90": {"b": 2.0}},
    ],
)
def test_a_band_is_two_ends_the_low_one_first(band):
    document = saved(merge(three_origins()).document)
    document["provenance"]["origins"]["host.ipc.zmq_roundtrip_s"] |= band
    with pytest.raises(SpecRefusal, match="a band, a `p10` at most its `p90`"):
        MachineSpec.from_mapping(document)


def test_only_a_probed_field_states_a_band():
    document = saved(merge(three_origins()).document)
    assumed = document["provenance"]["origins"]["interconnect.router_relay_s"]
    assumed |= {"p10": 1e-3, "p90": 2e-3}
    with pytest.raises(SpecRefusal, match="only kind with a subject"):
        MachineSpec.from_mapping(document)


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39", "mi355x-gpu45"])
def test_every_checked_in_spec_publishes_its_probed_numbers_with_a_band(name):
    path = ROOT / "scripts" / "compass" / "machines" / f"{name}.json"
    refuse_unbanded(MachineSpec.from_mapping(json.loads(path.read_text())))


def test_the_pooled_probe_states_each_fields_band_over_its_records(tmp_path):
    """Two runs' wakes differ: the band runs from a tenth to nine tenths of the
    way between their own fits, and a field both fit alike has no width."""
    runs = []
    for k, level in enumerate((1e-4, 2e-4)):
        samples = _samples(TRUTH)
        for b in probe.BLOCKS:
            done = b + level * -math.expm1(-b / 0.5)
            stamps = {"send": 0.0, "done": done, "post": b, "end": done}
            samples.append(
                {"carrier": "gloo", "bytes": 8, "late": True, "block": b} | stamps
            )
        runs.append(tmp_path / f"run{k}.json")
        runs[-1].write_text(json.dumps({"samples": samples}) + "\n")
    probe.main(str(tmp_path / "pooled.json"), *map(str, runs))
    pooled = json.loads((tmp_path / "pooled.json").read_text())
    for carrier, fields in pooled["carriers"].items():
        stated = {k for k, v in fields.items() if not isinstance(v, list)}
        assert set(pooled["band"][carrier]) == stated
    one, two = (probe.fit(json.loads(r.read_text())["samples"]) for r in runs)
    one, two = one["gloo"]["wake_s"], two["gloo"]["wake_s"]
    assert one < two
    band = [one + 0.1 * (two - one), one + 0.9 * (two - one)]
    assert pooled["band"]["gloo"]["wake_s"] == pytest.approx(band)
    low, high = pooled["band"]["rccl"]["per_byte_s"]
    assert low == high == pytest.approx(pooled["carriers"]["rccl"]["per_byte_s"])
