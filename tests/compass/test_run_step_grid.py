# SPDX-License-Identifier: MIT
"""The step grid selected by a run file's ``step_grid`` key: its bundle published to
and loaded from an `ArtifactStore` by `run.runner`, checked against the
deployment's config, and named in the summary `_write_summary` writes.

The deployment is the Qwen3.5-27B config at TP1 on the suite's machine
document, replaying graphs on `test_backend_step_grid`'s ladder; the bundle is
that file's replay grid and a unit residual, published under the readings
`step_grid.conditions` takes of the same deployment unless a test moves one.
The kinds `fit` publishes and `measure steps` measures are published and
loaded under those readings too.
"""

import copy
import json
import math
import re
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_artifact_invalidation import STANZA, moved
from test_backend_step_grid import (
    B37,
    C_NODES,
    LADDER,
    REPLAY,
    REPLAY_B,
    bundle,
    decode,
    fitted,
    graphed,
    linear,
    point,
)
from test_dp_slice import _deployment
from test_memory_readings import CONFIG_JSON, DOCUMENT, META_8B, publish_readings
from test_refused_step_continues import drive
from test_runner_control_refusals import Worker
from test_vertical_slice import _run_file
from transformers import PretrainedConfig

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    KEY_FIELDS,
    MATRIX,
    ArtifactRefusal,
    ArtifactStore,
    Axis,
    GateState,
    Key,
    Kind,
    Row,
    Topology,
    axes_of,
    machine_spec_digest,
    member_name,
    rows_for,
)
from atom.compass.backends import StepGridBackend
from atom.compass.backends.resolver import CostRefused
from atom.compass.backends.step_grid_fit import VALUE
from atom.compass.backends.step_roofline import StepFallbackBackend
from atom.compass.run import step_grid
from atom.compass.spec import MachineSpec
from atom.utils import clock
from atom.utils.clock import LPRuntime

KEY = {
    "model": "Qwen/Qwen3.5-27B",
    "device": "MI355X",
    "parallel": "tp1",
    "engine_config": "replay-mns48",
    "grid": "s2-sparse-margin0.25",
    "source_root": "tree-67c176f0",
}
BOUNDS = {"max_num_seqs": 48, "max_num_batched_tokens": 8192, "max_model_len": 2048}
BUNDLE = {
    "grid": {**bundle("replay", REPLAY), "value": VALUE},
    "residual": fitted({"unit": linear()}, {"unit": 1.0}),
    "bounds": BOUNDS,
}


def deployment(tmp_path, **config):
    """A graphed worker's runner, carrying the config fields `run.runner` reads,
    with its memory readings published in the store at ``tmp_path / "store"``."""
    (tmp_path / "config.json").write_text(CONFIG_JSON.read_text())
    hf = PretrainedConfig.from_dict(json.loads(CONFIG_JSON.read_text())["text_config"])
    r = graphed(None)
    r._piecewise_cg_active = lambda: False
    r.rank = 0
    r.config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=hf,
        tensor_parallel_size=1,
        tp_world_size=1,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_expert_parallel=False,
        enable_dp_attention=False,
        parallel_config=SimpleNamespace(
            data_parallel_size=1, data_parallel_rank=0, pipeline_parallel_rank=0
        ),
        **BOUNDS,
        gpu_memory_utilization=0.9,
        torch_dtype=torch.bfloat16,
        enforce_eager=False,
        enable_tbo=False,
        capture_sizes=list(LADDER),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(LADDER)),
        kv_cache_block_size=16,
        kv_cache_dtype="bf16",
        # ATOM's `QuantizationConfig` resolves aiter dtypes, which needs a driver.
        quant_config=SimpleNamespace(compute_hash=lambda: "unquantized-bf16"),
        kv_transfer_config={},
    )
    vars(r.config).update(config)
    publish_readings(tmp_path / "store", r.config)
    return r


def readings(r):
    return step_grid.conditions(r.config, MachineSpec.from_mapping(DOCUMENT))


def publish(tmp_path, r, member=BUNDLE, conditions=None, **key):
    """The bundle handed off under `r`'s own readings, unless others are given."""
    return ArtifactStore(tmp_path / "store").publish(
        Key.of(Kind.STEP_GRID_BUNDLE, **{**KEY, **key}),
        provenance=STANZA,
        topology=Topology(),
        conditions=conditions or readings(r),
        gates=GateState.of(),
        members={member_name(*step_grid.MEMBER): json.dumps(member).encode()},
    )


def run_file(tmp_path, monkeypatch, **keys):
    """The suite's run file with ``step_grid`` in place of ``step_stub``."""
    run = json.loads(_run_file(tmp_path).read_text())
    del run["step_stub"]
    run["step_grid"] = {"store": str(tmp_path / "store"), **KEY}
    run.update(keys)
    (tmp_path / "run.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))
    return run


def summary(monkeypatch, run, r):
    """The engine's and the frontend's hand-ins after the finish, and the summary."""
    authority = compass_run._RecordingAuthority(run)
    authority.started = authority.finished = time.monotonic()
    table = compass_run.channel_table(run)
    engine = LPRuntime(compass_run.ENGINE, table, None)
    engine.now = math.inf
    monkeypatch.setattr(clock, "_installed", engine)
    with compass_run.engine_loop(SimpleNamespace(runner_mgr=Worker(r))):
        pass
    monkeypatch.setattr(compass_run, "_authority", authority)
    frontend = LPRuntime(compass_run.FRONTEND, table, None)
    frontend.now = math.inf
    frontend.loop = SimpleNamespace(executor=SimpleNamespace(refusals=[]))
    monkeypatch.setattr(clock, "_installed", frontend)
    assert compass_run.frontend_done(SimpleNamespace(close=lambda: None))
    return json.loads((Path(run["out_dir"]) / compass_run.SUMMARY_FILE).read_text())


def test_a_run_file_bundle_prices_the_runs_steps_and_the_summary_names_it(
    monkeypatch, tmp_path
):
    r = deployment(tmp_path)
    entry = publish(tmp_path, r)
    run = run_file(tmp_path, monkeypatch)

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend, StepGridBackend)
    # Rung 48's cell, as `test_backend_step_grid` reads it from the same grid.
    assert r.forward(decode(B37)).predicted_s == 8897.5e-6
    out = summary(monkeypatch, run, r)
    assert out["step_grid"] == {
        "engine": {
            "key": KEY,
            "digest": entry.digest,
            "describe": (
                f"step grid, bundle {entry.key}; replay decode grid, "
                "timing scope forward; recipe step-grid-residual v2, members "
                "decode: unit; prefill_output: unit; prefill_no_output: unit"
            ),
        }
    }
    assert out["schedule"]["predicted_seconds"] == 8897.5e-6
    assert out["coverage_report"] is False


def test_a_run_ended_by_abort_still_names_its_bundle(monkeypatch, tmp_path):
    """The grid has no prefill point, so under ``on_refusal: abort`` the first
    step ends the run, and the summary written at that early end names the
    bundle."""
    monkeypatch.setattr(torch.distributed, "all_reduce", None)
    r = deployment(tmp_path)
    entry = publish(tmp_path, r)
    run = run_file(
        tmp_path, monkeypatch, on_refusal="abort", clock_endpoint=f"inproc:{tmp_path}"
    )
    for module, name in [(compass_run, "_authority"), (clock, "_installed")]:
        monkeypatch.setattr(module, name, None)
    with compass_run.frontend(_deployment()):
        pass
    compass_run.runner(r, META_8B)
    engine = LPRuntime(compass_run.ENGINE, compass_run.channel_table(run), None)
    monkeypatch.setattr(clock, "_installed", engine)

    with (
        pytest.raises(CostRefused),
        compass_run.engine_loop(SimpleNamespace(runner_mgr=Worker(r))),
    ):
        drive(r)

    path = Path(run["out_dir"]) / compass_run.SUMMARY_FILE
    deadline = time.monotonic() + 10
    while not path.exists():
        assert time.monotonic() < deadline, "no summary.json in 10 wall seconds"
        time.sleep(0.05)
    out = json.loads(path.read_text())
    assert out["step_grid"]["engine"]["digest"] == entry.digest
    [[reason, _]] = out["schedule"]["refusals"]["reasons"]
    assert reason.startswith("ended:engine raised CostRefused(")


def test_a_step_roofline_still_prices_below_step_grid(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    run_file(tmp_path, monkeypatch, step_roofline=True)

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend, StepFallbackBackend)
    assert isinstance(r.compass_backend.first, StepGridBackend)


def test_step_grid_is_one_of_the_backend_keys(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    run_file(tmp_path, monkeypatch, step_stub={})

    with pytest.raises(ValueError, match="this one has step_stub and step_grid"):
        compass_run.runner(r, META_8B)


@pytest.mark.parametrize(
    "fields,named",
    [
        ({"grid": "/data/grids/s2.json"}, "`step_grid_bundle.grid` holds"),
        ({"path": "bundles/qwen"}, "has no key field 'path'"),
        ({"source_root": None}, "this step_grid_bundle key states no source_root"),
    ],
    ids=["path", "unknown", "missing"],
)
def test_the_key_is_its_fields_never_a_path(monkeypatch, tmp_path, fields, named):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    key = {k: v for k, v in {**KEY, **fields}.items() if v is not None}
    run_file(tmp_path, monkeypatch, step_grid={"store": str(tmp_path / "store"), **key})

    with pytest.raises(ArtifactRefusal, match=re.escape(named)) as refused:
        compass_run.runner(r, META_8B)
    assert "(model, device, parallel, engine_config, grid, source_root)" in str(
        refused.value
    )


def test_a_key_with_no_bundle_refuses_naming_it(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    publish(tmp_path, r, grid="s2-dense")
    run_file(tmp_path, monkeypatch)

    with pytest.raises(ArtifactRefusal, match="no entry for step_grid_bundle\\("):
        compass_run.runner(r, META_8B)


@pytest.mark.parametrize(
    "axis",
    [a for a in axes_of(Row.STEP_GRID_BUNDLE) if a is not Axis.ENGINE_CONFIG],
    ids=lambda a: a.field,
)
def test_a_bundle_read_under_other_conditions_refuses_naming_the_cell(
    monkeypatch, tmp_path, axis
):
    """Each axis of its row but the engine config, whose reading is the keyed
    conditions."""
    r = deployment(tmp_path)
    publish(tmp_path, r, conditions=moved(readings(r), axis))
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ArtifactRefusal, match=re.escape(f"`step grid bundle` x `{axis}`")
    ):
        compass_run.runner(r, META_8B)


def test_a_bundle_measured_on_another_host_cpu_loads(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    publish(tmp_path, r, conditions=moved(readings(r), Axis.HOST_CPU))
    run_file(tmp_path, monkeypatch)

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend, StepGridBackend)


def test_a_bundle_is_checked_against_the_specs_torch_not_the_simulators(
    monkeypatch, tmp_path
):
    r = deployment(tmp_path)
    publish(tmp_path, r)
    run_file(tmp_path, monkeypatch)
    monkeypatch.setattr(torch, "__version__", "0.0.0+the-simulating-host")

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend, StepGridBackend)


def test_the_runs_readings_take_the_stack_from_the_spec_and_the_host_cpu_as_host_key(
    tmp_path,
):
    r = deployment(tmp_path)
    document = copy.deepcopy(DOCUMENT)
    document["device"]["software_pinned_to"]["torch"] = "2.9.1+target"
    read = step_grid.conditions(r.config, MachineSpec.from_mapping(document))

    assert read.reading(Axis.TORCH).value == "2.9.1+target"
    stack = document["device"]["software_pinned_to"]
    assert read.reading(Axis.SOFTWARE_STACK).value == (
        f"rocm {stack['rocm']} / aiter {stack['aiter']} / rccl {stack['rccl']}"
    )
    host = compass_run.host_key(r.config)["host_cpu"]
    assert read.reading(Axis.HOST_CPU).value == host


#: The suite's machine document's digest, as a law fitted on it is keyed.
SPEC_DIGEST = machine_spec_digest(MachineSpec.from_mapping(DOCUMENT))


def host_laws_key(r) -> dict:
    """A `host_laws` key from the values `run.host_key` reads and the spec's
    digest."""
    host = compass_run.host_key(r.config)
    return {
        "host_cpu": host["host_cpu"],
        "atom_src": host["atom"],
        "model": host["model"],
        "engine_config": json.dumps(host["engine"], sort_keys=True),
        "machine_spec": SPEC_DIGEST,
    }


#: The fields a step measurement and a step linear law share with a bundle.
MEASURED = {k: KEY[k] for k in ("model", "parallel", "engine_config", "source_root")}
#: A key per kind `fit` publishes and `measure steps` measures; a `host_laws`
#: key is the deployment's `host_laws_key`.
NEW_KINDS = {
    Kind.STEP_MEASUREMENT: {
        **MEASURED,
        "device": KEY["device"],
        "sampling": "uniform-s7",
        "step_kind": "decode",
        "view": "sha256:5e1f",
    },
    Kind.STEP_LINEAR_LAW: {
        **MEASURED,
        "machine_spec": SPEC_DIGEST,
        "form": "fixed+tokens+seqs",
    },
    Kind.HOST_LAWS: None,
}


@pytest.mark.parametrize("kind", list(NEW_KINDS), ids=str)
def test_a_new_kind_loads_under_the_runs_readings_and_refuses_on_its_rows_cells(
    tmp_path, kind
):
    """Published and loaded through the store under `step_grid.conditions`;
    each axis moved refuses where the kind's row depends on it, and loads
    where it does not."""
    r = deployment(tmp_path)
    here = readings(r)
    store = ArtifactStore(tmp_path / "store")
    key = Key.of(kind, **(NEW_KINDS[kind] or host_laws_key(r)))
    assert tuple(key.values) == KEY_FIELDS[kind]
    store.publish(
        key,
        provenance=STANZA,
        topology=Topology(),
        conditions=here,
        gates=GateState.of(),
        members={member_name("laws", Topology().ranks()[0], "json"): b"{}"},
    )
    [row] = rows_for(kind)

    assert store.load(key, conditions=readings(r), gates=GateState.of()).key == key
    for axis in Axis:
        there = moved(here, axis)
        if not MATRIX[row][axis].depends:
            store.load(key, conditions=there, gates=GateState.of())
            continue
        with pytest.raises(ArtifactRefusal, match=re.escape(f"`{row}` x `{axis}`")):
            store.load(key, conditions=there, gates=GateState.of())


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_each_mi308x_spec_states_its_torch_and_the_runs_readings_take_it(
    tmp_path, name
):
    path = Path(__file__).parents[2] / "scripts/compass/machines" / f"{name}.json"
    spec = MachineSpec.from_mapping(
        compass_run._width_keys(json.loads(path.read_text()))
    )
    pinned = "device.software_pinned_to.torch"

    assert spec.origin(pinned)["origin"] == "probed"
    read = step_grid.conditions(deployment(tmp_path).config, spec)
    assert read.reading(Axis.TORCH).value == spec.value(pinned)


@pytest.mark.parametrize(
    "config,named",
    [
        (
            {"kv_cache_block_size": 32},
            "kv_cache_block_size 16 in the bundle and 32 here",
        ),
        (
            {"kv_cache_dtype": "fp8"},
            "kv_cache_dtype 'bf16' in the bundle and 'fp8' here",
        ),
        (
            {"quant_config": SimpleNamespace(compute_hash=lambda: "fp8-e4m3")},
            "quantization 'unquantized-bf16' in the bundle and 'fp8-e4m3' here",
        ),
        (
            {"tensor_parallel_size": 2},
            "tensor_parallel_size 1 in the bundle and 2 here",
        ),
        (
            {"capture_sizes": LADDER[:-1]},
            f"ladder {LADDER} in the bundle and {LADDER[:-1]} here",
        ),
        ({"enforce_eager": True}, "graph mode 'replay' in the bundle and 'eager' here"),
    ],
    ids=["block-size", "kv-dtype", "quantization", "tp", "ladder", "graph-mode"],
)
def test_a_deployment_off_the_bundles_keyed_conditions_refuses_naming_them(
    monkeypatch, tmp_path, config, named
):
    publish(tmp_path, deployment(tmp_path))
    run_file(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="measured under other conditions") as refused:
        compass_run.runner(deployment(tmp_path, **config), META_8B)
    assert named in str(refused.value)


def test_a_capture_size_past_max_num_seqs_is_not_on_the_ladder(monkeypatch, tmp_path):
    publish(tmp_path, deployment(tmp_path))
    run_file(tmp_path, monkeypatch)
    r = deployment(tmp_path, capture_sizes=LADDER + [64])

    compass_run.runner(r, META_8B)

    assert isinstance(r.compass_backend, StepGridBackend)


def test_a_timing_scope_narrower_than_the_forward_refuses(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    narrow = copy.deepcopy(BUNDLE)
    narrow["grid"]["scope"] = "run_model+logits"
    publish(tmp_path, r, member=narrow)
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match="timing scope 'run_model\\+logits' in the bundle and 'forward' here",
    ):
        compass_run.runner(r, META_8B)


@pytest.mark.parametrize("value", [None, "busy window"])
def test_a_bundle_stating_no_value_or_another_than_the_bench_window_refuses(
    monkeypatch, tmp_path, value
):
    r = deployment(tmp_path)
    other = copy.deepcopy(BUNDLE)
    del other["grid"]["value"]
    if value:
        other["grid"]["value"] = value
    publish(tmp_path, r, member=other)
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match=re.escape(f"value {value!r} in the bundle and 'bench window' here"),
    ):
        compass_run.runner(r, META_8B)


def test_a_data_parallel_run_refuses(monkeypatch, tmp_path):
    publish(tmp_path, deployment(tmp_path))
    run_file(tmp_path, monkeypatch)
    dp2 = SimpleNamespace(data_parallel_size=2, data_parallel_rank=0)

    with pytest.raises(ValueError, match="this run has data_parallel_size 2"):
        compass_run.runner(deployment(tmp_path, parallel_config=dp2), META_8B)


def test_an_unmeasured_rung_between_measured_ones_refuses_listing_it(
    monkeypatch, tmp_path
):
    """Rung 32 has B nodes and no point; 16 and 48 were measured around it."""
    sparse = copy.deepcopy(BUNDLE)
    rung16 = [point(16, b, c, time=1000) for b in (9, 16) for c in C_NODES]
    sparse["grid"] |= bundle("replay", [p for p in REPLAY if p["at"][0] == 48] + rung16)
    sparse["grid"]["decode"]["nodes"]["B"] = {"16": [9, 16], **REPLAY_B}
    sparse["grid"]["decode"]["nodes"]["c"]["16"] = C_NODES
    r = deployment(tmp_path)
    publish(tmp_path, r, member=sparse)
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "measured the decode rungs [16, 48], and not the ladder's rungs [32]"
        ),
    ):
        compass_run.runner(r, META_8B)


@pytest.mark.parametrize(
    "bound,value",
    [("max_num_seqs", 64), ("max_num_batched_tokens", 16384), ("max_model_len", 4096)],
)
def test_a_bound_past_the_one_the_grid_was_planned_for_refuses(
    monkeypatch, tmp_path, bound, value
):
    publish(tmp_path, deployment(tmp_path))
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=f"this run's {bound} {value} is above the {BOUNDS[bound]} "
    ):
        compass_run.runner(deployment(tmp_path, **{bound: value}), META_8B)


@pytest.mark.parametrize(
    "edit,named",
    [
        (lambda m: m.pop("bounds"), "KeyError: 'bounds'"),
        (
            lambda m: m["residual"]["kinds"]["decode"]["members"]["unit"].update(
                transform="log"
            ),
            "the residual's decode member unit does not load",
        ),
        # The layout before prefill was read on (B, T, c) and decode's c nodes
        # were kept per rung.
        (
            lambda m: m["grid"]["prefill"].update(
                nodes={"B": [1], "N_Q": [8], "c": [0]}
            ),
            "KeyError: 'T'",
        ),
        (
            lambda m: m["grid"]["decode"]["nodes"].update(c=C_NODES),
            "the decode c nodes are one list per rung, not a list",
        ),
    ],
    ids=["bounds", "member", "prefill-n-q", "decode-c-list"],
)
def test_a_bundle_that_does_not_load_names_itself(monkeypatch, tmp_path, edit, named):
    broken = copy.deepcopy(BUNDLE)
    edit(broken)
    r = deployment(tmp_path)
    entry = publish(tmp_path, r, member=broken)
    run_file(tmp_path, monkeypatch)

    with pytest.raises(
        ValueError, match=re.escape(f"{entry.key} does not load")
    ) as refused:
        compass_run.runner(r, META_8B)
    assert named in str(refused.value)
