# SPDX-License-Identifier: MIT
"""`compass measure memory`'s entry, and a run reading it.

An engine start needs a GPU, so the ranks' files stand in for one here: each
is what `MemoryReadingModelRunner` writes as its rank starts, named by
`run.memory`'s naming, holding the key its rank read off the resolved config,
its engine and its readings. `measure_memory.publish` takes them from there,
under the stack `measure_memory.stack` reads, which these tests state.

The deployment is `test_run_step_grid`'s TP1 Qwen3.8-27B worker on the suite's
machine document.
"""

import copy
import hashlib
import json

import pytest
from test_artifact_invalidation import BASE, STANZA
from test_memory_readings import (
    CONFIG_JSON,
    DOCUMENT,
    META_8B,
    READINGS,
    STACK,
    meta_tree,
)
from test_run_step_grid import deployment
from test_vertical_slice import _run_file
from transformers import PretrainedConfig

from atom.compass import cli
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    machine_spec_digest,
    member_name,
)
from atom.compass.drive import measure_memory
from atom.compass.memory import ModelTerms, device_readings, reserves
from atom.compass.memory.readings import MemoryReadings
from atom.compass.run import _width_keys, memory
from atom.compass.spec import MachineSpec, SpecRefusal

MACHINE = MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))
#: One TP2 rank's readings beside `READINGS[2]`: a rank need not read as another.
RANK1 = dict(READINGS[2], persistent_forward_buffer_bytes=125_000_000)


def tp2(tmp_path):
    """A TP2 worker of the suite's deployment, with no entry in the store."""
    (tmp_path / "deployment").mkdir()
    return deployment(tmp_path / "deployment", tensor_parallel_size=2, tp_world_size=2)


def started(out, config, readings):
    """`out` as the start left it: the engine's log, and each rank's file in its
    readings directory; the log's digest, which the entry records."""
    (out / measure_memory.READINGS).mkdir(parents=True)
    (out / measure_memory.LOG).write_text("Memory budget: ...\n")
    for rank, held in enumerate(readings):
        coords = memory.coords(config, rank)
        name = member_name(memory.STEM, coords, memory.EXTENSION)
        held = {k: v for k, v in held.items() if k != "record"}
        written = {"key": memory.key_fields(config), "engine": {}, "readings": held}
        (out / measure_memory.READINGS / name).write_text(json.dumps(written))
    return "sha256:" + hashlib.sha256(b"Memory budget: ...\n").hexdigest()


@pytest.fixture
def stack(monkeypatch):
    """The stack the suite's machine document pins, as the one this process runs."""
    held = dict(STACK)
    monkeypatch.setattr(measure_memory, "stack", lambda: held)
    return held


def test_each_rank_reads_its_own_member_of_the_entry_measure_memory_publishes(
    tmp_path, stack
):
    r = tp2(tmp_path)
    record = started(tmp_path / "out", r.config, [READINGS[2], RANK1])

    measure_memory.publish(tmp_path / "out", MACHINE, tmp_path / "store", "c")

    for rank, held in enumerate([READINGS[2], RANK1]):
        read = memory.load(tmp_path / "store", r.config, MACHINE, rank)
        assert read.values == dict(held, record=record)
        assert read.term("persistent", "persistent_forward_buffer_bytes").records == (
            record,
        )
    copy_ = json.loads((tmp_path / "out" / "memory_readings.json").read_text())
    assert copy_["key"]["tp"] == 2 and copy_["key"]["machine_spec"] == (
        machine_spec_digest(MACHINE)
    )
    assert len(copy_["members"]) == 2


def test_an_entry_measured_under_another_stack_does_not_load(tmp_path, stack):
    r = tp2(tmp_path)
    started(tmp_path / "out", r.config, [READINGS[2]] * 2)
    stack["aiter"] = "9205e1059"

    measure_memory.publish(tmp_path / "out", MACHINE, tmp_path / "store", "c")

    with pytest.raises(ArtifactRefusal) as refused:
        memory.load(tmp_path / "store", r.config, MACHINE, 0)
    assert "ROCm / AITER / RCCL" in str(refused.value)


def test_ranks_whose_reserves_spread_past_the_limit_publish_nothing(tmp_path, stack):
    r = tp2(tmp_path)
    crowded = dict(READINGS[2], driver_and_collective_reserve_bytes=10_000_000_000)
    started(tmp_path / "out", r.config, [READINGS[2], crowded])

    with pytest.raises(SpecRefusal, match="spread"):
        measure_memory.publish(tmp_path / "out", MACHINE, tmp_path / "store", "c")
    assert not (tmp_path / "store").exists()


def test_ranks_that_read_two_configurations_publish_nothing(tmp_path, stack):
    r = tp2(tmp_path)
    started(tmp_path / "out", r.config, [READINGS[2]] * 2)
    one = next((tmp_path / "out" / measure_memory.READINGS).iterdir())
    held = json.loads(one.read_text())
    held["key"]["engine_config"] = "{}"
    one.write_text(json.dumps(held))

    with pytest.raises(ValueError, match="the ranks read 2 keys"):
        measure_memory.publish(tmp_path / "out", MACHINE, tmp_path / "store", "c")


def test_a_key_stating_another_layout_than_its_members_publishes_nothing(tmp_path):
    key = Key.of(
        Kind.MEMORY_READINGS,
        model="sha256:" + "a" * 64,
        machine_spec=machine_spec_digest(MACHINE),
        dp=1,
        pp=1,
        pcp=1,
        tp=2,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        decode_context_parallel_size=1,
        engine_config="{}",
    )
    topology = Topology(dp=2, tp=2)
    with pytest.raises(ArtifactRefusal) as refused:
        ArtifactStore(tmp_path).publish(
            key,
            provenance=STANZA,
            topology=topology,
            conditions=BASE,
            gates=GateState.of(),
            members={},
        )
    assert "states dp=1 and its members are written at dp2.pp1.pcp1.tp2" in str(
        refused.value
    )


@pytest.mark.parametrize(
    "flag", ["--load_dummy", "--load-dummy=random", "--load_d", "--load-dum"]
)
def test_dummy_weights_are_refused(tmp_path, flag, capsys):
    argv = ["measure", "memory", "--gpu", "0", "--machine", "m", "--store", "s"]
    with pytest.raises(SystemExit) as refused:
        cli.main([*argv, "--out", str(tmp_path), "--", "--model", "x", flag])
    assert "dummy weights are not yet shown to read as a real load" in str(
        refused.value
    )
    assert not any(tmp_path.iterdir())


# --- a run reading its configuration's entry ---------------------------------


def test_a_run_file_naming_no_store_refuses(tmp_path, monkeypatch):
    r = deployment(tmp_path)
    run = json.loads(_run_file(tmp_path).read_text())
    del run["store"]
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))

    with pytest.raises(ValueError, match="names no store"):
        compass_run.runner(r, META_8B)


def test_a_configuration_with_no_entry_refuses_naming_what_the_nearest_differs_in(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))
    r.config.max_num_seqs = 32

    with pytest.raises(ArtifactRefusal) as refused:
        compass_run.runner(r, META_8B)
    assert "no entry for memory_readings(" in str(refused.value)
    assert "the nearest entry differs in engine_config" in str(refused.value)


def test_the_run_takes_its_readings_off_the_entry(tmp_path, monkeypatch):
    r = deployment(tmp_path)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))

    compass_run.runner(r, META_8B)

    readings = r.compass_readings
    assert (
        readings.non_torch.total == READINGS[1]["driver_and_collective_reserve_bytes"]
    )
    peak = {t.name: t for t in readings.peak_torch.terms}
    assert peak["load residue"].nbytes == 1_100_000
    assert peak["persistent"].nbytes == 124_000_000
    assert peak["activations"].nbytes == READINGS[1]["activation_bytes"]
    assert peak["activations"].source.startswith("memory_readings ")


# --- the named result: an entry gives what the spec gave ---------------------

#: Qwen3.8-27B at TP1 on node 18's MI308X, as one start read it
#: (`scripts/compass/mi308x_spec.py`), as a rank's member.
MI308X_READINGS = {
    "capacity_bytes": 206_141_652_992,
    "driver_and_collective_reserve_bytes": 1_249_902_592,
    "allocator_retained_after_load_bytes": 14_924_832,
    "persistent_forward_buffer_bytes": 315_351_040,
    "activation_bytes": 2_956_984_320,
    "warmup_tokens": 16_384,
    "cudagraph_pool_bytes": None,
    "record": "sha256:f4951d84d5fd2b76c50c18c8234a1586fbeb5f0ee89897dc54f0c44f6bb1eb87",
}
#: The five readings the base read off `mi308x-hjbog18.json` for the same
#: model and width, the meta build holding the start's parameter bytes.
FROM_THE_SPEC = {
    "total": 206_141_652_992,
    "peak_torch": 58_034_271_744,
    "non_torch": 1_249_902_592,
    "cudagraph_overhead": 591_396_864,
    "free": 146_857_478_656,
}


def test_the_mi308x_entry_gives_the_readings_its_spec_gave():
    path = CONFIG_JSON.parents[2] / "scripts/compass/machines/mi308x-hjbog18.json"
    spec = MachineSpec.from_mapping(_width_keys(json.loads(path.read_text())))
    raw = json.loads(CONFIG_JSON.read_text())
    entry = MemoryReadings.from_member(MI308X_READINGS, "memory_readings mi308x")
    model = ModelTerms.from_config(
        PretrainedConfig.from_dict(raw["text_config"]),
        activations=entry.activations(16_384),
        config_json=raw,
        tree=meta_tree(54_713_457_120),
    )
    readings = device_readings(
        spec,
        memory=entry,
        tp_width=1,
        model=model,
        cudagraph_overhead=reserves(activations=model.activations),
    )
    assert {k: v.total for k, v in readings.as_dict().items()} == FROM_THE_SPEC
    assert readings.declared == ()
