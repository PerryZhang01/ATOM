# SPDX-License-Identifier: MIT
"""The PP carriers: fitted by the probe, stated by the MI308X spec, and
pricing a stage's input in the recurrence gate."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from test_pp_stage_input import MS, pp2_record

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


def _late(steps, x):
    """The last stage of `pp2_record` with its input arriving `x` ns after
    the head's device end: its first device wait, and everything after, later."""
    first, *rest = (copy.deepcopy(s) for s in steps)
    first["t_recv_ns"][1] += x
    for k in ("t_exit_ns", "d_busy_ns", "d_recv_ns"):
        first[k] += x
    for s in rest:
        for k, v in s.items():
            if k.startswith("t_"):
                s[k] = [t + x for t in v] if isinstance(v, list) else v + x
    return [first, *rest]


def test_the_last_stage_waits_for_the_heads_tensors_on_the_carriers():
    head, last = pp2_record()
    sends = recurrence.check(head, (0, 2))["sends"]
    late = _late(last, MS)
    priced = recurrence.check(late, (1, 2), sends, (MODELS, 500))
    assert priced["pass"] and priced["total"] == pytest.approx(0, abs=1e-9)
    assert not recurrence.check(late, (1, 2), sends)["pass"]


def test_the_script_prices_the_send_on_a_machine_specs_carriers(tmp_path):
    """`MODELS` written in a spec, each zero as 1e-15: the schema takes only
    positive times."""
    head, last = pp2_record()
    for name, steps in (("dp0", head), ("pp1.dp0", _late(last, MS))):
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
    for i, s in enumerate(samples):
        if s["carrier"] == "rccl" and s["bytes"] > eager and not s["late"]:
            s["end"] *= 1 + (-1) ** i * 0.01
    fixed = truth["rccl"][4]
    assert probe.fit(samples)["rccl"]["rendezvous_fixed_s"] == pytest.approx(
        fixed, rel=0.05
    )
