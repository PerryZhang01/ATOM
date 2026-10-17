# SPDX-License-Identifier: MIT
"""A ``-pcp 2`` or ``-dcp 2`` engine in a run: it reads the memory readings of
its own layout, and a step linear law or a step grid bundle, which states no
context-parallel width, refuses it by name.

The deployment is `test_run_step_grid`'s Qwen3.8-27B worker on the suite's
machine document.
"""

import json

import pytest
from test_backend_step_linear import law
from test_memory_readings import META_8B
from test_run_step_grid import deployment, publish, run_file
from test_vertical_slice import _run_file

from atom.compass import run as compass_run
from atom.compass.artifacts import ArtifactRefusal

PCP2 = {"prefill_context_parallel_size": 2}
DCP2 = {
    "tensor_parallel_size": 2,
    "tp_world_size": 2,
    "decode_context_parallel_size": 2,
}


def test_a_pcp2_engine_with_only_the_width_one_entry_refuses_naming_pcp(
    tmp_path, monkeypatch
):
    r = deployment(tmp_path)
    vars(r.config).update(PCP2)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))

    with pytest.raises(ArtifactRefusal) as refused:
        compass_run.runner(r, META_8B)
    print(f"\n{refused.value}")
    assert "no entry for memory_readings(" in str(refused.value)
    assert "the nearest entry differs in pcp" in str(refused.value)


@pytest.mark.parametrize("width", [PCP2, DCP2], ids=["pcp2", "dcp2"])
def test_a_context_parallel_engine_starts_on_its_own_layouts_entry(
    tmp_path, monkeypatch, width
):
    r = deployment(tmp_path, **width)
    monkeypatch.setenv(compass_run.ENV, str(_run_file(tmp_path)))

    compass_run.runner(r, META_8B)

    assert r.compass_readings.non_torch.total == 970_000_000


# A DCP width divides the TP width, which a step linear law already refuses.
@pytest.mark.parametrize(
    "backend, width",
    [
        (compass_run.STEP_LINEAR, PCP2),
        (compass_run.STEP_GRID, PCP2),
        (compass_run.STEP_GRID, DCP2),
    ],
    ids=["step_linear-pcp2", "step_grid-pcp2", "step_grid-dcp2"],
)
def test_a_measured_backend_refuses_a_context_parallel_engine_by_name(
    tmp_path, monkeypatch, width, backend
):
    r = deployment(tmp_path, **width)
    if backend == compass_run.STEP_GRID:
        publish(tmp_path, r)
        run_file(tmp_path, monkeypatch)
    else:
        run = json.loads(_run_file(tmp_path).read_text())
        del run["step_stub"]
        (tmp_path / "run.json").write_text(json.dumps(run | {backend: law()}))
        monkeypatch.setenv(compass_run.ENV, str(tmp_path / "run.json"))

    with pytest.raises(ValueError) as refused:
        compass_run.runner(r, META_8B)
    print(f"\n{refused.value}")
    name, value = next((n, v) for n, v in width.items() if "context" in n)
    assert f"run file's {backend} is measured at context-parallel width 1" in str(
        refused.value
    )
    assert f"{name} {value}" in str(refused.value)
