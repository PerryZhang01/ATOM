# SPDX-License-Identifier: MIT
"""The PP carriers: fitted by the probe, stated by the MI308X spec, and
pricing a stage's input in the recurrence gate."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest
from test_pp_stage_input import MS, pp2_record, staged

from atom.compass.parity import recurrence
from atom.compass.runner.pp_send import CarrierModel, carrier_models
from atom.compass.spec import MachineSpec, SpecRefusal

TREE = Path(__file__).resolve().parents[2]
SPEC = TREE / "scripts/compass/machines/mi308x-hjbog39.json"
_probe = importlib.util.spec_from_file_location(
    "pp_carriers", TREE / "scripts/compass/pp_carriers.py"
)
probe = importlib.util.module_from_spec(_probe)
_probe.loader.exec_module(probe)

#: gloo free; each RCCL tensor of one 500-byte token rendezvous in 0.5 ms.
MODELS = {
    "gloo": CarrierModel(10**9, 0.0, 0.0, 0.0, 0.0),
    "rccl": CarrierModel(0, 0.0, 0.0, 1e-6, 0.0),
}


def _shift(steps, x):
    """`steps` copied, every host stamp `x` ns later."""
    steps = copy.deepcopy(steps)
    for s in steps:
        for k, v in s.items():
            if k.startswith("t_"):
                s[k] = [t + x for t in v] if isinstance(v, list) else v + x
    return steps


def _late(steps, x):
    """The last stage of `pp2_record` with its input arriving `x` ns after
    the head's device end: its first device wait, and everything after, later.
    It waits at its staging gate to its launch, so it posts each receive at
    its entry."""
    first = copy.deepcopy(steps[0])
    first["t_recv_ns"][1] += x
    for k in ("t_exit_ns", "d_busy_ns", "d_recv_ns"):
        first[k] += x
    late = [first, *_shift(steps[1:], x)]
    for s in late:
        s["t_gate_ns"][1] = s["t_launch_ns"][0]
    return late


def test_the_last_stage_waits_for_the_heads_tensors_on_the_carriers():
    head, last = pp2_record()
    _, sends = staged(head, (0, 2))
    late = _late(last, MS)
    priced = staged(late, (1, 2), sends, (MODELS, 500))[0]
    assert priced["pass"] and priced["total"] == pytest.approx(0, abs=1e-9)
    assert not staged(late, (1, 2), sends)[0]["pass"]


def test_the_last_stage_launches_when_the_heads_metadata_arrives():
    """gloo's metadata lands 9 ms after the head's 2 ms host send, at its
    11 ms device end, so the tensors land at 12 ms as with gloo free, and
    only the launch moves. A 10 ms launch from there leaves the first step
    host-bound, exiting at 22 ms where its device ends at 20."""
    head, last = pp2_record()
    _, sends = staged(head, (0, 2))
    late = _late(last, 2 * MS)
    late[0] |= {
        "t_pp_recv_ns": [MS, 11 * MS],
        "t_launch_ns": [MS, 21 * MS],
        "t_recv_ns": [21 * MS, 21 * MS],
        "d_recv_ns": 11 * MS,
        "d_busy_ns": 19 * MS,
    }
    slow = MODELS | {"gloo": CarrierModel(10**9, 0.0, 9e-3, 0.0, 0.0)}
    priced = staged(late, (1, 2), sends, (slow, 500))[0]
    assert priced["pass"] and priced["total"] == pytest.approx(0, abs=1e-9)
    assert not staged(late, (1, 2), sends, (MODELS, 500))[0]["pass"]


def test_the_script_prices_the_send_on_a_machine_specs_carriers(tmp_path):
    """`MODELS` written in a spec, each zero as 1e-15: the schema takes only
    positive times. The head's third and fourth forwards enter at the late
    stage's exits, which return their tokens, so the last stage's input
    arrives a further 1 ms later. The fourth enters 1 ms after them, once the
    third's tensors are received, which its device waits for."""
    head, last = pp2_record()
    head[2:] = _shift(head[2:], MS)
    head[3:] = _shift(head[3:], MS)
    last = _late(last, MS)
    last[2:] = _shift(last[2:], MS)
    last[3:] = _shift(last[3:], MS)
    for name, steps in (("dp0", head), ("pp1.dp0", last)):
        (tmp_path / f"{name}.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in steps)
        )
    document = json.loads(SPEC.read_text())
    tiny = 1e-15
    document["interconnect"]["intra_node"]["pp"] = {
        c: {
            "eager_threshold_bytes": m.eager_threshold_bytes,
            **{k: v or tiny for k, v in vars(m).items() if k.endswith("_s")},
        }
        for c, m in MODELS.items()
    }
    spec = tmp_path / "machine.json"
    spec.write_text(json.dumps(document))
    assert recurrence.main(str(tmp_path), str(spec), "500") == 0
    assert recurrence.main(str(tmp_path)) == 1


def test_a_carrier_may_have_no_eager_size_and_no_fewer():
    document = json.loads(SPEC.read_text())
    carrier_models(MachineSpec.from_mapping(document))  # the spec states both
    gloo = document["interconnect"]["intra_node"]["pp"]["gloo"]
    gloo["eager_threshold_bytes"] = 0
    MachineSpec.from_mapping(document)
    gloo["eager_threshold_bytes"] = -1
    with pytest.raises(SpecRefusal, match="eager_threshold_bytes"):
        MachineSpec.from_mapping(document)


def _sample(c, n, late, eager, o, lat, g, fixed):
    post = probe.LATE if late else -0.002
    if n <= eager:
        done = o + n * g
        end = max(post, done + lat)
    else:
        done = end = max(0.0, post) + fixed + n * g
    stamps = {"send": 0.0, "done": done, "post": post, "end": end}
    return {"carrier": c, "bytes": n, "late": late} | stamps


def test_the_probe_recovers_the_carriers_it_was_sampled_from():
    truth = {"rccl": (1024, 1e-5, 2e-6, 1e-11, 5e-5), "gloo": (0, 0, 0, 1e-9, 3e-5)}
    samples = [
        _sample(c, n, late, *truth[c])
        for c in probe.SIZES
        for n in probe.SIZES[c]
        for late in (False, True)
    ]
    fit = probe.fit(samples)
    eager, o, lat, g, fixed = truth["rccl"]
    assert fit["rccl"]["eager_threshold_bytes"] == eager
    got = [fit["rccl"][k] for k in ("overhead_s", "latency_s")]
    assert got == pytest.approx([o, lat], rel=1e-6)
    for c, (_, _, _, g, fixed) in truth.items():
        assert fit[c]["per_byte_s"] == pytest.approx(g, rel=1e-6)
        assert fit[c]["rendezvous_fixed_s"] == pytest.approx(fixed, rel=1e-6)
    assert fit["gloo"]["eager_threshold_bytes"] == 0
    assert fit["gloo"]["early"] == pytest.approx([0] * 5, abs=1e-12)
    # A 1% error at every size moves the fixed part 1%, not by the large sizes' error.
    for s in samples:
        if s["carrier"] == "rccl" and s["bytes"] > eager and not s["late"]:
            s["end"] *= 1 + (-1) ** s["bytes"].bit_length() * 0.01
    fixed = truth["rccl"][4]
    assert probe.fit(samples)["rccl"]["rendezvous_fixed_s"] == pytest.approx(
        fixed, rel=0.01
    )


def _samples(truth):
    return [
        _sample(c, n, late, *truth[c])
        for c in probe.SIZES
        for n in probe.SIZES[c]
        for late in (False, True)
    ]


TRUTH = {"rccl": (1024, 1e-5, 2e-6, 1e-11, 5e-5), "gloo": (0, 0, 0, 1e-9, 3e-5)}


def test_a_late_size_is_eager_only_if_its_sender_completes_before_the_post():
    """Each late receive posted 1 ms before `LATE`, as a device-timed RCCL
    send's is: rendezvous sizes that complete before `LATE` stay rendezvous."""
    samples = _samples(TRUTH)
    for s in samples:
        if s["late"]:
            rdv = s["bytes"] > TRUTH[s["carrier"]][0]
            for k in ("post", "end", "done") if rdv else ("post", "end"):
                s[k] -= 1e-3
    fit = probe.fit(samples)
    assert [fit[c]["eager_threshold_bytes"] for c in TRUTH] == [1024, 0]


def test_a_carrier_threshold_is_no_bool():
    document = json.loads(SPEC.read_text())
    document["interconnect"]["intra_node"]["pp"]["gloo"]["eager_threshold_bytes"] = True
    with pytest.raises(SpecRefusal, match="eager_threshold_bytes"):
        MachineSpec.from_mapping(document)


def test_the_probe_fits_how_late_a_blocked_gloo_sender_wakes():
    """A sender blocked `b` before the post completes `W (1 - exp(-b / rise))`
    past the rendezvous price; the late order's sender, blocked `LATE`, too. A
    block sample is no late-order sample: its receive lands 10 us after it."""
    wake = (1.5e-4, 0.05)
    samples = _samples(TRUTH)
    for s in samples:
        if s["carrier"] == "gloo" and s["late"]:
            s["done"] = s["end"] = s["end"] + wake[0] * -math.expm1(-probe.LATE / 0.05)
    _, _, _, g, fixed = TRUTH["gloo"]
    for b in probe.BLOCKS:
        done = b + fixed + 8 * g + wake[0] * -math.expm1(-b / wake[1])
        stamps = {"send": 0.0, "done": done, "post": b, "end": done + 1e-5}
        samples.append(
            {"carrier": "gloo", "bytes": 8, "late": True, "block": b} | stamps
        )
    gloo = probe.fit(samples)["gloo"]
    assert (gloo["wake_s"], gloo["wake_rise_s"]) == pytest.approx(wake, rel=1e-4)
    assert gloo["late"] == pytest.approx([0] * 5, abs=1e-9)
    assert gloo["rendezvous_fixed_s"] == pytest.approx(fixed, rel=1e-6)
    assert "wake_s" not in probe.fit(_samples(TRUTH))["gloo"]


def test_the_probe_pools_the_records_it_wrote(tmp_path):
    """Two runs' wakes differ; their pooled fit takes every block's median
    over both, and names each run's record by its digest."""
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
    both = [s for r in runs for s in json.loads(r.read_text())["samples"]]
    assert pooled["carriers"] == probe.fit(both)
    assert pooled["records"] == [
        hashlib.sha256(r.read_bytes()).hexdigest() for r in runs
    ]


def test_the_probe_prices_a_late_rendezvous_receive_with_its_threshold_moved():
    """Posted `LATE` after the send, a rendezvous RCCL receive ends its
    threshold's bytes sooner than one posted first, and the fit's late error
    on those samples is nil."""
    eager, _, _, g, _ = TRUTH["rccl"]
    samples = _samples(TRUTH)
    for s in samples:
        if s["carrier"] == "rccl" and s["late"] and s["bytes"] > eager:
            s["done"] = s["end"] = s["end"] - eager * g
    assert probe.fit(samples)["rccl"]["late"] == pytest.approx([0] * 5, abs=1e-15)
