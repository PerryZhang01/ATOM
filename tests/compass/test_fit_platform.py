# SPDX-License-Identifier: MIT
"""The platform `compass fit`'s records ran on, against the machine spec its
`--machine` names: run.json states the device and the stack as the spec does,
so a law fitted from a record of that platform loads under a run's readings of
the spec, and records from another device, stack or torch refuse at the fit.

The records are `test_fit_steps.py`'s. The first test's run.json is what
`drive.identity` writes in compass-next:nightly-20261013 on hjbog-srdc-18's
MI308X, from a planted `rocm-smi` listing, ROCm version file and RCCL and torch
versions, and aiter's commit planted as that image's, which no planted checkout
can have; its law loads under `test_run_step_grid.py`'s deployment.
"""

import json
import os
import re
from pathlib import Path

import pytest
import torch
from test_fit_samples import edit
from test_fit_steps import (  # noqa: F401
    clients,
    fit,
    law_key,
    planted_roots,
    record,
)
from test_memory_readings import DOCUMENT, STACK
from test_run_step_grid import deployment

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
    machine_spec_key,
)
from atom.compass.drive import identity as drive
from atom.compass.run import step_grid
from atom.compass.spec import MachineSpec

MACHINES = Path(__file__).parents[2] / "scripts/compass/machines"
#: rocm-smi's listing of GPU 0 in compass-next:nightly-20261013 on hjbog-srdc-18.
CARD = {
    "Card Series": "AMD Instinct MI308X",
    "Card Model": "0x74a2",
    "Card Vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
    "Card SKU": "M3080202",
    "Subsystem ID": "0x74a2",
    "Device Rev": "0x00",
    "Node ID": "2",
    "GUID": "63420",
    "GFX Version": "gfx942",
    "PCI Bus": "0000:0A:00.0",
}


def spec(name: str) -> tuple[Path, MachineSpec]:
    path = MACHINES / f"{name}.json"
    document = json.loads(path.read_text())
    return path, MachineSpec.from_mapping(document)


def compass_next(tmp_path, monkeypatch):
    """What identity reads in compass-next on hjbog-srdc-18, planted."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "cards.json").write_text(json.dumps({"card0": CARD}))
    (bindir / "rocm-smi").write_text(f"#!/bin/sh\ncat {bindir / 'cards.json'}\n")
    (bindir / "rocm-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    (tmp_path / "rocm" / ".info").mkdir(parents=True)
    (tmp_path / "rocm" / ".info" / "version").write_text("7.2.4\n")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    monkeypatch.setattr(torch.cuda.nccl, "version", lambda: (2, 27, 7))
    monkeypatch.setattr(torch, "__version__", "2.10.0+rocm7.2.4.git3d3aa833")
    monkeypatch.setattr(drive, "_commit", lambda root: "9205e1059")
    monkeypatch.setattr(drive, "PCI", str(tmp_path / "pci"))


def test_a_law_fitted_from_identity_s_record_loads_under_the_spec_it_ran_on(
    tmp_path, monkeypatch
):
    compass_next(tmp_path, monkeypatch)
    r = deployment(tmp_path)
    a = record(tmp_path / "a", 19, key=compass_run.host_key(r.config))
    run = drive.identity(a / "steps/dp0.engine.json", r.config.model, ["c"], (0,))
    (a / "run.json").write_text(json.dumps(run))
    path, n39 = spec("mi308x-hjbog39")
    store, out = tmp_path / "store", tmp_path / "out"
    flags = [a, *clients(a), "--machine", path, "--store", store, "--out", out]
    cli.main(["fit", "steps", "--backend", "step-linear", *map(str, flags)])
    copy = json.loads((out / "step_linear_law.json").read_text())
    law = Key.of(Kind.STEP_LINEAR_LAW, **copy["key"])

    readings = step_grid.conditions(r.config, n39)
    loaded = ArtifactStore(store).load(law, conditions=readings, gates=GateState.of())
    assert loaded.key == law
    n18 = step_grid.conditions(r.config, spec("mi308x-hjbog18")[1])
    with pytest.raises(ArtifactRefusal, match="`ROCm / AITER / RCCL`") as refused:
        ArtifactStore(store).load(law, conditions=n18, gates=GateState.of())
    assert "aiter 9205e1059" in str(refused.value)
    assert "aiter f4e7c7509" in str(refused.value)


@pytest.mark.parametrize("command", ["fit steps --backend step-linear", "fit host"])
@pytest.mark.parametrize(
    "change, named",
    [
        (
            lambda r: r["device"][0].update(name="MI300X"),
            "device 'MI300X gfx950' in the records and 'MI355X gfx950'",
        ),
        (
            lambda r: r["stack"].update(aiter="9205e1059"),
            (
                "software_stack 'rocm 7.2.4 / aiter 9205e1059 / rccl 2.22.3' in the "
                "records and 'rocm 7.2.4 / aiter f4e7c7509 / rccl 2.22.3'"
            ),
        ),
        (
            lambda r: r["stack"].update(torch="2.9.0"),
            "torch '2.9.0' in the records and '2.10.0'",
        ),
    ],
    ids=["device", "aiter", "torch"],
)
def test_records_from_another_platform_than_the_spec_refuse_naming_it(
    tmp_path, command, change, named
):
    a = record(tmp_path / "a", 19)
    edit(a / "run.json", change)

    in_spec = f"{named} in {tmp_path / 'spec.json'}"
    with pytest.raises(SystemExit, match=re.escape(in_spec)):
        fit(tmp_path, command, a, *clients(a))
    assert not (tmp_path / "store").exists()


def test_the_readings_are_the_spec_s_whatever_order_run_json_lists_the_stack(
    tmp_path,
):
    a = record(tmp_path / "a", 19, stack=dict(reversed(STACK.items())))
    fit(tmp_path, "fit steps --backend step-linear", a, *clients(a))

    entry = ArtifactStore(tmp_path / "store").read(law_key())
    cells = dict(entry.fingerprints[Row.STEP_LINEAR_LAW].cells)
    stated = machine_spec_key(MachineSpec.from_mapping(DOCUMENT)).values
    assert cells[Axis.SOFTWARE_STACK].value == stated["software_stack"]
    assert cells[Axis.DEVICE].value == stated["device"]
