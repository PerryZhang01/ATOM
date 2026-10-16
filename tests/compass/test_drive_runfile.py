# SPDX-License-Identifier: MIT
"""The run file `drive.runfile` builds from a command's flags.

Each backend's entry is published through `ArtifactStore` as `fit` publishes
one, params and report, under `test_run_step_grid`'s readings; the run file is
read back through `run.spec` and installed by `run.runner` on that file's
deployment, whose bundle the step grid tests use.
"""

import argparse
import dataclasses
import hashlib
import json

import pytest
from test_artifact_invalidation import STANZA
from test_backend_step_linear import law
from test_memory_readings import DOCUMENT, META_8B
from test_run_step_grid import (
    KEY,
    NEW_KINDS,
    SPEC_DIGEST,
    deployment,
    host_laws_key,
    publish,
    readings,
)

from atom.compass import run as compass_run
from atom.compass.artifacts import (
    ArtifactRefusal,
    ArtifactStore,
    GateState,
    Key,
    Kind,
    Topology,
    member_name,
)
from atom.compass.backends import StepGridBackend, StepLinearBackend, StepStubBackend
from atom.compass.backends.step_roofline import StepFallbackBackend
from atom.compass.backends.step_stub import Coefficients
from atom.compass.drive import runfile
from atom.compass.fit.publish import RANK
from atom.compass.run.step_grid import keyed
from atom.compass.spec import SpecRefusal

LAWS = {"decode": {"W_pre": [1e-4, 1e-8, 1e-7]}, "prefill": {"E": [2e-4, 0, 0]}}
REPORT = {"provenance": [{"source": "record", "run": "r1", "traces": ["trace-a"]}]}


def published(tmp_path, r, kind, key, params, report=REPORT):
    """`params` and `report` handed off at `key` as `fit` hands them off."""
    return ArtifactStore(tmp_path / "store").publish(
        Key.of(kind, **key),
        provenance=STANZA,
        topology=Topology(),
        conditions=readings(r),
        gates=GateState.of(),
        members={
            member_name(stem, RANK, "json"): json.dumps(body).encode()
            for stem, body in {"params": params, "report": report}.items()
        },
    )


def host_laws(tmp_path, r):
    key = {**compass_run.host_key(r.config), "machine_spec": SPEC_DIGEST}
    return published(
        tmp_path, r, Kind.HOST_LAWS, host_laws_key(r), {"key": key, "laws": LAWS}
    )


def step_linear(tmp_path, r):
    engine = json.dumps(keyed(r.config), sort_keys=True)
    key = {**NEW_KINDS[Kind.STEP_LINEAR_LAW], "engine_config": engine}
    return published(tmp_path, r, Kind.STEP_LINEAR_LAW, key, law())


def build(tmp_path, *flags, atom=(), machine=DOCUMENT):
    """The run file of `flags`, with the suite's machine and the store."""
    (tmp_path / "machine.json").write_text(json.dumps(machine))
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)
    common = ["--machine", str(tmp_path / "machine.json")]
    args = parser.parse_args([*common, "--store", str(tmp_path / "store"), *flags])
    return runfile.build(args, ["--model", str(tmp_path), *atom])


def install(tmp_path, monkeypatch, run, r):
    """`run` as the server hands it to `r`'s worker, through `run.spec`."""
    (tmp_path / "run_file.json").write_text(json.dumps(run))
    monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run_file.json"))
    monkeypatch.setenv(compass_run.NOW_ENV, str(tmp_path / "now.bin"))
    table = compass_run.channel_table(compass_run.spec())
    compass_run._PublishedClock(compass_run.ENGINE, table, None)
    compass_run.runner(r, META_8B)


def test_a_step_grid_and_host_laws_by_token_install_both(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    grid, host = publish(tmp_path, r), host_laws(tmp_path, r)
    run = build(
        tmp_path, "--step-grid", grid.key.token(), "--host-laws", host.key.token()
    )

    install(tmp_path, monkeypatch, run, r)

    assert isinstance(r.compass_backend, StepGridBackend)
    assert r.compass_host.laws == LAWS
    store = str((tmp_path / "store").resolve())
    assert run["step_grid"] == {"store": store, **KEY} and run["store"] == store
    provenance = {**STANZA.as_json(), "inputs": None}
    assert run["backend"] == {"digest": grid.digest, "provenance": provenance}


def test_a_step_linear_law_carries_its_engine_config_and_fit_provenance(
    monkeypatch, tmp_path
):
    r = deployment(tmp_path, enable_dp_attention=False, enable_expert_parallel=False)
    entry = step_linear(tmp_path, r)
    run = build(tmp_path, "--step-linear", entry.key.token())

    install(tmp_path, monkeypatch, run, r)

    assert isinstance(r.compass_backend, StepLinearBackend)
    assert run["step_linear"] == {**law(), "engine_config": keyed(r.config)}
    assert run["backend"]["digest"] == entry.digest
    assert run["backend"]["provenance"]["inputs"] == REPORT["provenance"]
    assert "host" not in run and r.compass_host.laws == {}


def test_a_law_fitted_under_another_engine_config_refuses_naming_each_field(
    monkeypatch, tmp_path
):
    fitted = {"enable_dp_attention": False, "enable_expert_parallel": False}
    run = build(
        tmp_path,
        "--step-linear",
        step_linear(tmp_path, deployment(tmp_path, **fitted)).key.token(),
    )
    here = deployment(tmp_path, **fitted, kv_cache_block_size=32, kv_cache_dtype="fp8")

    with pytest.raises(ValueError, match="fitted under another engine config") as e:
        install(tmp_path, monkeypatch, run, here)
    assert "kv_cache_block_size 16 in the law and 32 here" in str(e.value)
    assert "kv_cache_dtype 'bf16' in the law and 'fp8' here" in str(e.value)


def test_a_step_stub_file_is_stated_with_its_digest(monkeypatch, tmp_path):
    stub = dataclasses.asdict(Coefficients())
    (tmp_path / "stub.json").write_text(json.dumps(stub))
    run = build(tmp_path, "--step-stub", str(tmp_path / "stub.json"))

    install(tmp_path, monkeypatch, run, r := deployment(tmp_path))

    assert isinstance(r.compass_backend, StepStubBackend)
    stated = json.dumps(stub, sort_keys=True).encode()
    digest = "sha256:" + hashlib.sha256(stated).hexdigest()
    assert run["backend"] == {"digest": digest, "provenance": None}


MACHINE_AND_STORE = ["--machine", "m", "--store", "s"]


@pytest.mark.parametrize(
    "flags,named",
    [
        (
            MACHINE_AND_STORE,
            "one of the arguments --step-stub --step-linear --step-grid",
        ),
        ([*MACHINE_AND_STORE, "--step-grid", "a", "--step-linear", "b"], "not allowed"),
        (
            ["--step-grid", "a"],
            "the following arguments are required: --machine, --store",
        ),
    ],
    ids=["no-backend", "two-backends", "no-machine-or-store"],
)
def test_the_flags_refuse_without_one_backend_a_machine_and_a_store(
    capsys, flags, named
):
    parser = argparse.ArgumentParser()
    runfile.add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(flags)
    assert named in capsys.readouterr().err


@pytest.mark.parametrize("given", ["unknown", "another-kind"])
def test_a_token_no_entry_of_the_flags_kind_ends_in_refuses_naming_it(tmp_path, given):
    r = deployment(tmp_path)
    token = {
        "unknown": "0123456789ab",
        "another-kind": host_laws(tmp_path, r).key.token(),
    }[given]

    with pytest.raises(
        ArtifactRefusal, match=f"0 step_grid_bundle entries in .* end in '{token}'"
    ):
        build(tmp_path, "--step-grid", token)


@pytest.mark.parametrize(
    "atom,width",
    [
        ((), 1),
        (("-dp", "2"), 2),
        (("-tp", "2", "-dp", "2", "--enable-dp-attention"), 4),
    ],
)
def test_the_fixed_keys_and_the_dp_width_from_atoms_arguments(tmp_path, atom, width):
    stub = tmp_path / "stub.json"
    stub.write_text(json.dumps(dataclasses.asdict(Coefficients())))

    run = build(tmp_path, "--step-stub", str(stub), "--on-refusal", "abort", atom=atom)

    assert run["data_parallel_size"] == width
    assert {k: run[k] for k in runfile.FIXED} == runfile.FIXED
    assert run["on_refusal"] == "abort" and run["step_roofline"] is False
    assert run["client_runs"] is None


def test_a_machine_file_that_is_no_spec_refuses_before_any_entry_is_read(tmp_path):
    with pytest.raises(SpecRefusal):
        build(tmp_path, "--step-grid", "0123456789ab", machine={"device": {}})


def test_the_step_roofline_reads_the_models_parameter_count(monkeypatch, tmp_path):
    r = deployment(tmp_path)
    grid = publish(tmp_path, r).key.token()
    with pytest.raises(
        ValueError, match=f"its {runfile.INDEX} has no metadata.total_parameters"
    ):
        build(tmp_path, "--step-grid", grid, "--step-roofline")
    index = {"metadata": {"total_parameters": 27_000_000_000, "total_size": 1}}
    (tmp_path / runfile.INDEX).write_text(json.dumps(index))

    run = build(tmp_path, "--step-grid", grid, "--step-roofline")
    install(tmp_path, monkeypatch, run, r)

    assert run["parameter_count"] == 27_000_000_000
    assert isinstance(r.compass_backend, StepFallbackBackend)
