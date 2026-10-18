# SPDX-License-Identifier: MIT
"""`compass measure memory`'s entry, and a run reading it.

An engine start needs a GPU, so the ranks' files stand in for one here: each
is what `MemoryReadingModelRunner` writes as its rank starts, named by
`run.memory`'s naming, holding the key its rank read off the resolved config,
its engine and its readings. `measure_memory.publish` takes them from there,
under the stack run.json's identity states, which these tests state as
the suite's machine document pins it.

The deployment is `test_run_step_grid`'s TP1 Qwen3.8-27B worker on the suite's
machine document.
"""

import contextlib
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_artifact_invalidation import BASE, STANZA
from test_memory_readings import (
    CONFIG_JSON,
    DOCUMENT,
    META_8B,
    READINGS,
    STACK,
    meta_tree,
    publish_readings,
)
from test_run_step_grid import deployment
from test_vertical_slice import _run_file

from atom.compass import cli
from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    Row,
    Topology,
    machine_spec_digest,
    member_name,
)
from atom.compass.drive import measure_memory
from atom.compass.memory import ModelTerms, device_readings, reserves
from atom.compass.memory.readings import MemoryReadings
from atom.compass.run import memory
from atom.compass.spec import MachineSpec, SpecRefusal

MACHINE = MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))
#: One TP2 rank's readings beside `READINGS[2]`: a rank need not read as another.
RANK1 = dict(READINGS[2], persistent_forward_buffer_bytes=125_000_000)
#: GPU 5 as rocm-smi names it on the suite's machine document's device.
MI355X = {"index": 5, "name": "MI355X", "arch": "gfx950"}


def tp2(tmp_path):
    """A TP2 worker of the suite's deployment, with no entry in the store."""
    (tmp_path / "deployment").mkdir()
    return deployment(tmp_path / "deployment", tensor_parallel_size=2, tp_world_size=2)


def started(out, config, readings):
    """`out` as the start left it: the engine's log, and each rank's file in its
    readings directory; the log's digest, which the entry records."""
    (out / measure_memory.READINGS).mkdir(parents=True)
    (out / measure_memory.LOG).write_text("Memory budget: ...\n")
    ranks_wrote(out, config, readings)
    return "sha256:" + hashlib.sha256(b"Memory budget: ...\n").hexdigest()


def ranks_wrote(out, config, readings):
    """Each rank's file in `out`'s readings directory, as the runner writes it."""
    for rank, held in enumerate(readings):
        coords = memory.coords(config, rank)
        name = member_name(memory.STEM, coords, memory.EXTENSION)
        held = {k: v for k, v in held.items() if k != "record"}
        written = {"key": memory.key_fields(config), "engine": {}, "readings": held}
        (out / measure_memory.READINGS / name).write_text(json.dumps(written))


def test_each_rank_reads_its_own_member_of_the_entry_measure_memory_publishes(tmp_path):
    r = tp2(tmp_path)
    record = started(tmp_path / "out", r.config, [READINGS[2], RANK1])

    measure_memory.publish(tmp_path / "out", MACHINE, tmp_path / "store", "c", STACK)

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


def a_start(tmp_path, monkeypatch, stack):
    """The suite's TP2 worker and `main`'s arguments for a start on GPU 5 under
    `stack`, its engine and identity stood in for: the ranks write their files
    as the engine starts, and run.json names `stack`."""
    r, out = tp2(tmp_path), tmp_path / "out"

    class Engine:
        pid = 0

        def __init__(self, *args, **kwargs):
            (tmp_path / "hip").write_text(kwargs["env"]["HIP_VISIBLE_DEVICES"])
            ranks_wrote(out, r.config, [READINGS[2]] * 2)

        def wait(self, timeout=None):
            return 0

    def gone(*args):
        raise ProcessLookupError

    monkeypatch.setattr(
        measure_memory, "quiet", lambda gpus, out: contextlib.nullcontext()
    )
    started_by = SimpleNamespace(
        Popen=Engine, STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired
    )
    monkeypatch.setattr(measure_memory, "subprocess", started_by)
    monkeypatch.setattr(measure_memory.os, "killpg", gone)
    monkeypatch.setattr(measure_memory, "_device", lambda gpus: [MI355X])
    monkeypatch.setattr(measure_memory, "_ptl", lambda gpus: "disabled")
    # rocm-smi GPU 5 as HIP device 2.
    monkeypatch.setattr(measure_memory, "hip_visible", {(5,): "2"}.get)
    monkeypatch.setattr(measure_memory, "identity", lambda *a: {"stack": stack})
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(DOCUMENT))
    flags = ["--gpu", "5", "--machine", str(machine), "--store", str(tmp_path / "s")]
    return r, [*flags, "--out", str(out), "--", "--model", r.config.model]


def test_the_engine_starts_on_the_hip_device_of_its_rocm_smi_gpu(tmp_path, monkeypatch):
    _, argv = a_start(tmp_path, monkeypatch, STACK)
    measure_memory.main(argv)
    assert (tmp_path / "hip").read_text() == "2"


def test_the_command_publishes_under_the_stack_the_spec_pins(tmp_path, monkeypatch):
    r, argv = a_start(tmp_path, monkeypatch, STACK)

    measure_memory.main(argv)

    key = Key.of(
        Kind.MEMORY_READINGS,
        machine_spec=machine_spec_digest(MACHINE),
        **memory.key_fields(r.config),
    )
    cells = dict(
        ArtifactStore(tmp_path / "s").read(key).fingerprints[Row.MEMORY_READINGS].cells
    )
    assert (
        cells[Axis.SOFTWARE_STACK].value == "rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3"
    )
    assert json.loads((tmp_path / "out" / "run.json").read_text()) == {"stack": STACK}


@pytest.mark.parametrize(
    "gpus, cards",
    [
        ("5", [dict(MI355X, name="MI308X", arch="gfx942")]),
        ("4,5", [dict(MI355X, index=4), dict(MI355X, name="MI308X", arch="gfx942")]),
    ],
)
def test_a_start_on_another_device_than_the_spec_states_refuses_naming_both(
    tmp_path, monkeypatch, gpus, cards
):
    monkeypatch.setattr(measure_memory, "_device", lambda gpus: cards)
    machine = tmp_path / "machine.json"
    machine.write_text(json.dumps(DOCUMENT))
    flags = ["--gpu", gpus, "--machine", str(machine), "--store", str(tmp_path / "s")]

    with pytest.raises(ValueError) as refused:
        measure_memory.main(
            [*flags, "--out", str(tmp_path / "out"), "--", "--model", "m"]
        )

    assert str(refused.value) == (
        f"GPU 5 is MI308X gfx942, and {machine} states MI355X gfx950; "
        "name the machine spec of the device the start runs on"
    )
    assert not (tmp_path / "out").exists() and not (tmp_path / "s").exists()


def test_a_start_under_another_stack_than_the_spec_pins_publishes_nothing(
    tmp_path, monkeypatch
):
    other = dict(STACK, aiter="9205e1059", torch="2.9.1")
    _, argv = a_start(tmp_path, monkeypatch, other)

    with pytest.raises(ValueError) as refused:
        measure_memory.main(argv)

    assert str(refused.value) == (
        "the start ran another stack than the machine spec mi355x-8gpu-2node "
        "pins: aiter '9205e1059' in the start and 'f4e7c7509' in the spec; "
        "torch '2.9.1' in the start and '2.10.0' in the spec"
    )
    assert not (tmp_path / "s").exists()


def test_the_engine_session_starts_the_command_beside_before_the_engine(
    tmp_path, monkeypatch
):
    r, argv = a_start(tmp_path, monkeypatch, STACK)
    envs = []
    engine = measure_memory.subprocess.Popen

    def recorded(command, **kwargs):
        envs.append(kwargs["env"])
        return engine(command, **kwargs)

    monkeypatch.setattr(measure_memory.subprocess, "Popen", recorded)
    cut = argv.index("--")
    measure_memory.main([*argv[:cut], "--beside", "lmcache server -p 1", *argv[cut:]])
    assert envs[0][measure_memory.BESIDE_ENV] == "lmcache server -p 1"

    class Started(Exception):
        pass

    def beside(command, **kwargs):
        raise Started(command, kwargs)

    monkeypatch.setenv(measure_memory.BESIDE_ENV, envs[0][measure_memory.BESIDE_ENV])
    monkeypatch.setattr(measure_memory.subprocess, "Popen", beside)
    with pytest.raises(Started) as started:
        measure_memory.start(["--model", r.config.model])
    # No session, group or env of its own: the engine's killpg stops it.
    assert started.value.args == (["lmcache", "server", "-p", "1"], {})


def starts_autotuning(monkeypatch, r, out, autotuned):
    """Engine starts that write each rank's file with the next count of kernels
    inductor autotuned in `autotuned`, `non_torch` 12 MiB high when it is not
    0; the environment each start was given."""
    left, envs = list(autotuned), []

    class Engine:
        pid = 0

        def __init__(self, *args, env, **kwargs):
            envs.append(env)
            tuned = left.pop(0)
            high = 12_582_912 if tuned else 0
            reserve = READINGS[2]["driver_and_collective_reserve_bytes"] + high
            read = dict(READINGS[2], driver_and_collective_reserve_bytes=reserve)
            ranks_wrote(out, r.config, [read] * 2)
            for rank in (out / measure_memory.READINGS).iterdir():
                held = json.loads(rank.read_text())
                rank.write_text(json.dumps(held | {"autotuned_kernels": tuned}))

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(measure_memory.subprocess, "Popen", Engine)
    return envs


def test_a_start_that_autotuned_publishes_the_next_starts_readings(
    tmp_path, monkeypatch
):
    r, argv = a_start(tmp_path, monkeypatch, STACK)
    monkeypatch.setenv("ATOM_DP_MASTER_PORT", "29500")
    envs = starts_autotuning(monkeypatch, r, tmp_path / "out", [6, 0])

    measure_memory.main(argv)

    read = memory.load(tmp_path / "s", r.config, MACHINE, 0).values
    assert read["driver_and_collective_reserve_bytes"] == (
        READINGS[2]["driver_and_collective_reserve_bytes"]
    )
    assert (tmp_path / "out.cold" / measure_memory.LOG).exists()
    # A port an earlier start used would load its compiled graphs, which
    # autotune again on their first load.
    assert [env["ATOM_DP_MASTER_PORT"] != "29500" for env in envs] == [True, True]


def test_a_second_start_that_autotunes_publishes_nothing(tmp_path, monkeypatch):
    r, argv = a_start(tmp_path, monkeypatch, STACK)
    starts_autotuning(monkeypatch, r, tmp_path / "out", [6, 6])

    with pytest.raises(ValueError) as refused:
        measure_memory.main(argv)

    assert str(refused.value) == (
        "readings.dp0of1.pp0of1.pcp0of1.tp0of2.json, "
        "readings.dp0of1.pp0of1.pcp0of1.tp1of2.json autotuned kernels again on a "
        "second start; non_torch on a cold inductor cache reads high, so nothing is "
        "published"
    )
    assert not (tmp_path / "s").exists()


def test_the_reading_runner_warms_up_once_as_a_serving_start_does(monkeypatch):
    # A second warmup's peak counts what the first left allocated, and the
    # KV pool a serving start sizes reads the first's.
    warmups = []

    class ModelRunner:
        def forward(self, batch):
            return batch

        def _maybe_warmup(self):
            warmups.append(self.forward(SimpleNamespace(total_tokens_num=16_384)))

    engine = types.ModuleType("atom.model_engine.model_runner")
    engine.ModelRunner = ModelRunner
    monkeypatch.setitem(sys.modules, engine.__name__, engine)
    path = Path(measure_memory.__file__).with_name("memory_runner.py")
    spec = importlib.util.spec_from_file_location("reading_runner", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runner = module.MemoryReadingModelRunner()
    runner._read = {}

    runner._maybe_warmup()

    assert len(warmups) == 1
    assert runner._read == {"warmup_tokens": 16_384}


def test_readings_taken_without_the_offload_connector_refuse_a_run_that_offloads(
    tmp_path,
):
    r = deployment(tmp_path)
    assert "kv_offload" not in memory.key_fields(r.config)["engine_config"]
    r.config.kv_transfer_config = {"kv_connector": "lmcache_mp", "kv_role": "offload"}

    with pytest.raises(ArtifactRefusal) as refused:
        memory.load(tmp_path / "store", r.config, MACHINE, 0)
    assert '"kv_offload": ["lmcache_mp"]' in str(refused.value)
    assert "the nearest entry differs in engine_config" in str(refused.value)

    publish_readings(tmp_path / "store", r.config, readings=READINGS[2])
    assert memory.load(tmp_path / "store", r.config, MACHINE, 0).values == READINGS[2]


def test_a_pd_connector_reads_the_readings_measured_without_one(tmp_path):
    r = deployment(tmp_path)
    measured = memory.load(tmp_path / "store", r.config, MACHINE, 0).values
    r.config.kv_transfer_config = {"kv_connector": "moriio", "kv_role": "kv_producer"}
    assert memory.load(tmp_path / "store", r.config, MACHINE, 0).values == measured


def test_ranks_whose_reserves_spread_past_the_limit_publish_nothing(tmp_path):
    r = tp2(tmp_path)
    crowded = dict(READINGS[2], driver_and_collective_reserve_bytes=10_000_000_000)
    started(tmp_path / "out", r.config, [READINGS[2], crowded])

    with pytest.raises(SpecRefusal, match="spread"):
        measure_memory.publish(
            tmp_path / "out", MACHINE, tmp_path / "store", "c", STACK
        )
    assert not (tmp_path / "store").exists()


def test_ranks_that_read_two_configurations_publish_nothing(tmp_path):
    r = tp2(tmp_path)
    started(tmp_path / "out", r.config, [READINGS[2]] * 2)
    one = next((tmp_path / "out" / measure_memory.READINGS).iterdir())
    held = json.loads(one.read_text())
    held["key"]["engine_config"] = "{}"
    one.write_text(json.dumps(held))

    with pytest.raises(ValueError, match="the ranks read 2 keys"):
        measure_memory.publish(
            tmp_path / "out", MACHINE, tmp_path / "store", "c", STACK
        )


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
    spec = MachineSpec.from_mapping(json.loads(path.read_text()))
    entry = MemoryReadings.from_member(MI308X_READINGS, "memory_readings mi308x")
    model = ModelTerms.from_tree(
        meta_tree(54_713_457_120), activations=entry.activations(16_384)
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


#: Prints the FX graph cache's hash of a weight's metadata, as a real runner's
#: process keys it; with `first`, a module reaching `torch` sits ahead of
#: torch's own in `sys.modules`, as an API server's imports put one.
_WEIGHT_KEY = """
import sys, types
if sys.argv[1] == "first":
    sys.modules["a_first"] = first = types.ModuleType("a_first")
import torch
from torch._inductor.codecache import FxGraphCachePickler
from torch._inductor.codecache import extract_tensor_metadata_for_cache_key
from atom.compass.parity import key_memory_formats_by_name

if sys.argv[1] == "first":
    first.torch = torch
key_memory_formats_by_name()
gm = torch.fx.GraphModule(torch.nn.Module(), torch.fx.Graph())
meta = extract_tensor_metadata_for_cache_key(torch.empty(4))
print(FxGraphCachePickler(gm).get_hash(meta))
"""


def test_a_weights_compile_key_is_the_same_whatever_the_import_order():
    root = CONFIG_JSON.parents[2]
    keys = {
        subprocess.run(
            [sys.executable, "-c", _WEIGHT_KEY, order],
            cwd=root,
            env=os.environ | {"PYTHONPATH": str(root)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for order in ("first", "torch")
    }
    assert len(keys) == 1, keys
