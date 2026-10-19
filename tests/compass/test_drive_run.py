# SPDX-License-Identifier: MIT
"""`compass run`: one simulated instance's server, its coverage check before
the server starts, and what run.json records for grading once it stops.

Steps are ATOM's `Scheduler`'s, recorded as `test_step_grid_plan.record`
records them and laid out as `record` writes a record directory; a population
is published from one by `trace_steps.publish`. Bundles are planned and
measured from those steps and published as `test_step_grid_plan.loaded`
publishes them, and ATOM's arguments resolve that test's deployment. The
server is `test_drive_trace_steps`'s planted one, which also copies a run
summary into its out_dir; the summary after a refused step is the one
`test_refused_step_continues.py` produces.
"""

import dataclasses
import json
import sys

import pytest
import test_drive_trace_steps
import torch
from test_artifact_store import AITER_ROOT, ATOM_ROOT
from test_dp_step_max import runner
from test_drive_runfile import build
from test_drive_trace_steps import IDENTITY
from test_memory_readings import ASSUMED, DOCUMENT, READINGS
from test_refused_step_continues import REFUSED, drive, finish, no_eager_rung
from test_refused_step_continues import run_file as refusing_run_file
from test_run_graph_pool import _installed
from test_run_step_grid import KEY, deployment
from test_step_grid_plan import (
    ENGINE,
    LADDER,
    MIXED,
    SHORT,
    loaded,
    located,
    measured,
    plan,
    record,
    refusing,
)

from atom.compass import run as compass_run
from atom.compass.artifacts import Key, Kind
from atom.compass.backends.step_stub import Coefficients
from atom.compass.clock import RefusalTally
from atom.compass.drive import run as drive_run
from atom.compass.drive import serve, trace_steps

#: `test_drive_trace_steps`'s planted server, which first copies the summary
#: `PLANTED_SUMMARY` names as client run 0's into the out_dir of the run file it
#: is handed.
PLANTED = (
    "import json, os, shutil, sys\n"
    'run = json.load(open(sys.argv[sys.argv.index("--compass-run") + 1]))\n'
    'first = os.path.join(run["out_dir"], "client-0")\n'
    "os.makedirs(first)\n"
    'shutil.copy(os.environ["PLANTED_SUMMARY"], first)\n'
    + test_drive_trace_steps.PLANTED
)


def recorded(tmp_path, monkeypatch, lengths, **drive):
    """A record directory, as `record` lays one out, of prompts of `lengths`;
    `drive` goes to `record`."""
    directory = tmp_path / ("record-" + "-".join(map(str, lengths)))
    directory.mkdir()
    record(tmp_path, monkeypatch, *lengths, **drive).rename(directory / serve.STEPS)
    (directory / serve.RUN_JSON).write_text(json.dumps(IDENTITY))
    return directory


def bundle(tmp_path, monkeypatch, member, **config):
    """`member`'s token and loaded backend, with ATOM's arguments resolving the
    deployment it was published under, changed by `config`."""
    backend = loaded(tmp_path, member)
    bounds = {k: ENGINE[k] for k in ("max_num_seqs", "max_num_batched_tokens")}
    r = deployment(tmp_path, **{**bounds, **config}, capture_sizes=LADDER)
    monkeypatch.setattr(drive_run, "_config", lambda atom_args: r.config)
    return Key.of(Kind.STEP_GRID_BUNDLE, **KEY).token(), backend


def stub(tmp_path):
    path = tmp_path / "stub.json"
    path.write_text(json.dumps(dataclasses.asdict(Coefficients())))
    return str(path)


def flags(tmp_path, *backend, atom=()):
    """`compass run`'s arguments, the suite's machine and the store with
    `backend`'s flags, out to ``out``, serving the model in `tmp_path`."""
    (tmp_path / "machine.json").write_text(json.dumps(DOCUMENT))
    common = ["--machine", str(tmp_path / "machine.json")]
    common += ["--store", str(tmp_path / "store"), "--out", str(tmp_path / "out")]
    return [*backend, *common, "--", "--model", str(tmp_path), *atom]


def refused_summary(directory, monkeypatch):
    """The run summary written after a refused step, in `directory`."""
    directory.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, "all_reduce", None)
        run = refusing_run_file(directory, patch)
        r = runner(no_eager_rung())
        drive(r)
        finish(patch, directory, run, r)
    return directory / compass_run.SUMMARY_FILE


def plant(tmp_path, monkeypatch, steps, summary) -> None:
    """Serve from the planted server, which writes the record `steps` and
    client run 0's summary `summary`."""
    (tmp_path / "planted.py").write_text(PLANTED)
    monkeypatch.setattr(serve, "SERVER", (sys.executable, str(tmp_path / "planted.py")))
    monkeypatch.setattr(serve, "identity", lambda *args: IDENTITY)
    monkeypatch.setenv("PLANTED_RECORD", str(steps))
    monkeypatch.setenv("PLANTED_SUMMARY", str(summary))


def unstarted(*args, **kwargs):
    raise AssertionError("a server started")


def test_steps_the_grid_cannot_answer_start_the_run_and_are_counted_by_name(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    points = located(tmp_path, planned)
    member = measured(planned, points, drop=("decode", points["decode"][0]))
    token, backend = bundle(tmp_path, monkeypatch, member)
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", str(directory)))

    lines = [json.loads(x) for x in (directory / "steps/dp0.jsonl").open()]
    named = [(f"{directory} dp0 step {line['step']}", line) for line in lines]
    declined = refusing(backend, named)
    assert 0 < len(declined) < len(named)
    coverage = json.loads((tmp_path / "out" / serve.RUN_JSON).read_text())["coverage"]
    assert [d["step"] for d in coverage["declined"]] == declined
    assert coverage["fallback"] == len(declined)
    assert coverage["covered"] == len(named) - len(declined)


def test_a_bundle_measured_under_another_engine_config_refuses_before_any_server(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED)
    planned = plan(directory / serve.STEPS)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member, kv_cache_dtype="fp8")
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="kv_cache_dtype 'bf16' in the bundle"):
        drive_run.main(
            flags(tmp_path, "--step-grid", token, "--coverage", str(directory))
        )


def test_an_input_recorded_at_another_token_budget_refuses_naming_that_field(
    tmp_path, monkeypatch
):
    directory = recorded(tmp_path, monkeypatch, MIXED, budget=2048)
    # Planned for this run's 8192 tokens, the bundle covers the input's steps.
    target = {"max_num_batched_tokens": 8192, "max_model_len": 16384}
    planned = plan(directory / serve.STEPS, engine=target)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member, max_num_batched_tokens=8192)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError) as refused:
        drive_run.main(
            flags(tmp_path, "--step-grid", token, "--coverage", str(directory))
        )

    assert str(refused.value) == (
        f"--coverage: {directory} dp0 ran under another engine than ATOM's "
        "arguments resolve: max_num_batched_tokens 2048 there, 8192 here"
    )


def test_a_population_on_another_ladder_and_tp_width_refuses_naming_both(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trace_steps, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT)
    )
    served = recorded(tmp_path, monkeypatch, SHORT, ladder=[1, 2, 4])
    # A TP-2 rank's engine file, which this suite's runner does not write.
    engine = served / serve.STEPS / "dp0.engine.json"
    stated = json.loads(engine.read_text())
    engine.write_text(json.dumps(stated | {"tensor_parallel_size": 2}))
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    population = trace_steps.publish(served, run, "compass trace steps").token()
    # Planned from the population's steps, the bundle covers them.
    planned = plan(served / serve.STEPS)
    member = measured(planned, located(tmp_path, planned))
    token, _ = bundle(tmp_path, monkeypatch, member)
    monkeypatch.setattr(serve, "serve", unstarted)

    with pytest.raises(ValueError, match="arguments resolve: ladder") as refused:
        drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", population))

    assert str(refused.value).endswith(
        f"ladder [1, 2, 4] there, {LADDER} here; tensor_parallel_size 2 there, 1 here"
    )


@pytest.mark.parametrize("backend", ["--step-stub", "--step-linear"])
def test_coverage_with_a_backend_that_has_no_check_refuses_naming_it(tmp_path, backend):
    argv = flags(tmp_path, backend, "k", "--coverage", str(tmp_path))

    with pytest.raises(ValueError, match=f"^--coverage: {backend} has no coverage"):
        drive_run.main(argv)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "role, side", [("kv_producer", "prefill"), ("kv_consumer", "decode")]
)
def test_a_prefill_or_a_decode_instance_refuses_before_its_server_starts(
    tmp_path, role, side
):
    kv = json.dumps({"kv_connector": "compass", "kv_role": role})
    argv = flags(tmp_path, "--step-stub", stub(tmp_path))

    with pytest.raises(ValueError, match=f"makes this a {side} instance"):
        drive_run.main([*argv, "--kv-transfer-config", kv])
    assert not (tmp_path / "out").exists()


def test_run_json_records_the_summarys_refusals_the_roofline_and_the_backend(
    tmp_path, monkeypatch
):
    summary = refused_summary(tmp_path / "authority", monkeypatch)
    directory = recorded(tmp_path, monkeypatch, MIXED)
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    drive_run.main(flags(tmp_path, "--step-stub", stub(tmp_path), "--step-roofline"))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    handed = json.loads((out / serve.RUN_FILE).read_text())
    refusals = json.loads(summary.read_text())["schedule"]["refusals"]
    assert written["refusals"] == refusals
    assert [reason for reason, _ in refusals["reasons"]] == [REFUSED]
    assert written["run_file"] == handed and handed["out_dir"] == str(out)
    assert written["step_roofline"] is handed["step_roofline"] is True
    assert written["skip"] == [] and written["coverage"] is None
    provenance = handed["backend"] | {"memory_readings_assumed": []}
    assert written["provenance"] == provenance
    assert written["host_key"] == IDENTITY["host_key"]


def test_a_covered_run_starts_and_records_its_check_over_a_record_and_a_population(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        trace_steps, "roots_for", lambda modules: (ATOM_ROOT, AITER_ROOT)
    )
    directory = recorded(tmp_path, monkeypatch, MIXED)
    served = recorded(tmp_path, monkeypatch, SHORT)
    run = build(tmp_path, "--step-stub", stub(tmp_path))
    population = trace_steps.publish(served, run, "compass trace steps").token()
    planned = plan(directory / serve.STEPS, served / serve.STEPS)
    token, _ = bundle(
        tmp_path, monkeypatch, measured(planned, located(tmp_path, planned))
    )
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    plant(tmp_path, monkeypatch, directory / serve.STEPS, summary)

    inputs = [str(directory), population]
    drive_run.main(flags(tmp_path, "--step-grid", token, "--coverage", *inputs))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    coverage = written["coverage"]
    assert coverage["inputs"] == inputs
    assert coverage["covered"] == coverage["steps"] == 15
    assert coverage["declined"] == [] and written["step_roofline"] is False
    handed = json.loads((out / serve.RUN_FILE).read_text())
    provenance = handed["backend"] | {"memory_readings_assumed": []}
    assert written["provenance"] == provenance
    assert handed["step_grid"]["grid"] == KEY["grid"]


#: The planted server, which also hands in the engine's memory as
#: `PLANTED_HAND_IN` holds it.
HANDING_IN = (
    "import json, os, shutil, sys\n"
    'run = json.load(open(sys.argv[sys.argv.index("--compass-run") + 1]))\n'
    'shutil.copy(os.environ["PLANTED_HAND_IN"], run["out_dir"])\n'
) + PLANTED


def test_run_json_names_the_assumed_memory_readings_its_engine_read(
    tmp_path, monkeypatch
):
    # The engine's hand-in, as a worker installed from an assumed entry writes it.
    worker = tmp_path / "worker"
    worker.mkdir()
    readings = dict(READINGS[1], assumed=ASSUMED)
    memory = _installed(monkeypatch, worker, eager=True, readings=readings)
    monkeypatch.delenv(compass_run.ENV)
    run = {"out_dir": str(worker)}
    compass_run._hand_in(run, "engine", [], memory=memory.compass_memory)
    (tmp_path / "authority").mkdir()
    summary = tmp_path / "authority" / compass_run.SUMMARY_FILE
    summary.write_text(json.dumps({"schedule": {"refusals": RefusalTally().record()}}))
    steps = recorded(tmp_path, monkeypatch, SHORT) / serve.STEPS
    plant(tmp_path, monkeypatch, steps, summary)
    (tmp_path / "planted.py").write_text(HANDING_IN)
    hand_in = worker / compass_run.LP_FILE.format("engine")
    monkeypatch.setenv("PLANTED_HAND_IN", str(hand_in))

    drive_run.main(flags(tmp_path, "--step-stub", stub(tmp_path)))

    out = tmp_path / "out"
    written = json.loads((out / serve.RUN_JSON).read_text())
    handed = json.loads((out / serve.RUN_FILE).read_text())
    provenance = handed["backend"] | {"memory_readings_assumed": [ASSUMED]}
    assert written["provenance"] == provenance
