# SPDX-License-Identifier: MIT
"""The HTTP intake probe: what it stamps on a server it runs, what its fit
recovers, and the machine specs that state its terms."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TREE = Path(__file__).resolve().parents[2]
PROBE = TREE / "scripts/compass/http_intake.py"
_loaded = importlib.util.spec_from_file_location("http_intake", PROBE)
probe = importlib.util.module_from_spec(_loaded)
_loaded.loader.exec_module(probe)

#: A stand-in for ATOM's API server: reads one request's body twice, as the
#: real handler does, renders its template, and prints its arguments.
SERVER = {
    "__init__.py": "",
    "entrypoints/__init__.py": "",
    "entrypoints/openai/__init__.py": "",
    "entrypoints/openai/chat_encoders.py": (
        "def apply_chat_template(*a, **k):\n    return 'prompt'\n"
    ),
    "entrypoints/openai/api_server.py": """
import asyncio, sys
from starlette.requests import Request
from .chat_encoders import apply_chat_template

async def main():
    async def receive():
        return {"type": "http.request", "body": b"x" * 1234}

    request = Request({"type": "http"}, receive)
    assert await request.body() == await request.body()
    apply_chat_template(None, None, [])

if __name__ == "__main__":
    print(sys.argv[1:])
    asyncio.run(main())
""",
}


def test_serve_stamps_a_body_read_off_the_socket_once_and_each_template(tmp_path):
    for name, text in SERVER.items():
        (tmp_path / "atom" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "atom" / name).write_text(text)
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [sys.executable, PROBE, "serve", stamps, "--model", "m"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
        capture_output=True,
        text=True,
        check=True,
    )
    assert done.stdout == "['--model', 'm']\n"
    rows = [line.split() for line in stamps.read_text().splitlines()]
    assert [(r[2], r[4]) for r in rows] == [("body", "1234"), ("tmpl", "6")]
    assert all(int(r[0]) >= int(r[3]) for r in rows)


def test_a_cell_with_http_intake_runs_its_server_under_the_probe(tmp_path):
    # The script is executed as a program, as a user runs it, with a python3
    # stand-in that records the server's arguments and exits before it is ready.
    bin_, args = tmp_path / "bin", tmp_path / "args"
    bin_.mkdir()
    (bin_ / "python3").write_text(
        "#!/bin/sh\n"
        f'case "$1" in *http_intake.py) printf "%s\\n" "$@" >{args}; exit 1;; esac\n'
        f'exec {sys.executable} "$@"\n'
    )
    (bin_ / "harness-python").write_text(
        '#!/bin/sh\nexec echo "${PYTHONPATH%%:*}/compass_harness/x.py"\n'
    )
    for f in bin_.iterdir():
        f.chmod(0o755)
    traces, run_file = tmp_path / "traces.jsonl", tmp_path / "run.json"
    traces.write_text('{"id": 0}\n')
    run_file.write_text("{}")
    stamps = tmp_path / "stamps"
    done = subprocess.run(
        [TREE / "scripts/compass/cctraces_sim.sh", run_file, tmp_path / "cell"],
        cwd="/",
        env=dict(
            os.environ,
            PATH=f"{bin_}:{os.environ['PATH']}",
            MODEL=str(tmp_path),
            TRACES=str(traces),
            HARNESS_PYTHON=str(bin_ / "harness-python"),
            HTTP_INTAKE=str(stamps),
        ),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert "the deployment exited before it started" in done.stderr, done.stderr
    probe_path, *rest = args.read_text().splitlines()
    assert Path(probe_path).resolve() == PROBE
    assert rest[:4] == ["serve", str(stamps), "--model", str(tmp_path)]


def _burst(fixed, per_byte, sizes, stall=0.0, t0=10**12):
    """Stamp lines for a loop that takes `sizes` in from `t0` one request at a
    time, each template starting once its own request is taken in, the largest
    request's intake `stall` seconds longer, with a template before the burst and
    a later turn after it."""
    lines = [f"{t0 - 10**9} 1 tmpl {t0 - 10**9} 9"]
    done = 0
    for i, size in enumerate(sizes, 1):
        done += size + (size == max(sizes)) * stall / per_byte
        start = t0 + round((i * fixed + done * per_byte) * 1e9)
        lines += [f"{start - 1000} 1 body {start - 2000} {size}"]
        lines += [f"{start + 500} 1 tmpl {start} {size}"]
    end = t0 + 10**11
    lines += [f"{end} 1 body {end - 10} 7", f"{end + 20} 1 tmpl {end + 10} 7"]
    return "\n".join(lines) + "\n"


def test_the_fit_recovers_the_terms_of_a_loop_that_takes_a_burst_one_at_a_time(
    tmp_path,
):
    sizes = np.random.default_rng(0).integers(100, 200_000, 64).tolist()
    stamps = tmp_path / "stamps"
    stamps.write_text(_burst(2.1e-3, 1.8e-8, sizes))
    got = probe.fit(str(stamps), 64)
    assert got["fixed_s"] == pytest.approx(2.1e-3, rel=1e-3)
    assert got["per_byte_s"] == pytest.approx(1.8e-8, rel=1e-3)
    assert got["rms_s"] < 1e-8
    assert (got["requests"], got["bytes"]) == (64, sum(sizes))
    with pytest.raises(SystemExit, match="65 bodies and 65 templates, not 66"):
        probe.fit(str(stamps), 66)


def test_a_stall_on_the_loop_moves_the_fixed_term_and_not_the_per_byte_one(
    tmp_path,
):
    # 50 ms of other work on the loop while it takes in the largest request:
    # spread over the 63 gaps, so the burst still spans what it measured.
    sizes = np.random.default_rng(0).integers(100, 200_000, 64).tolist()
    stamps = tmp_path / "stamps"
    stamps.write_text(_burst(2.1e-3, 1.8e-8, sizes, stall=0.05))
    got = probe.fit(str(stamps), 64)
    print(f"\n  {got}")
    assert got["per_byte_s"] == pytest.approx(1.8e-8, rel=0.05)
    span = 63 * 2.1e-3 + 0.05 + sum(sizes[1:]) * 1.8e-8
    assert 63 * got["fixed_s"] + sum(sizes[1:]) * got["per_byte_s"] == pytest.approx(
        span
    )


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39", "mi355x-gpu45"])
def test_every_committed_spec_states_a_probed_intake_from_a_named_record(name):
    from atom.compass.spec import MachineSpec

    document = json.loads((TREE / f"scripts/compass/machines/{name}.json").read_text())
    spec = MachineSpec.from_mapping(document)
    for term in ("fixed_s", "per_byte_s"):
        origin = spec.origin(f"host.http_intake.{term}")
        assert origin["origin"] == "probed"
        assert origin["record"].startswith("sha256:")


@pytest.mark.parametrize("name", ["mi308x-hjbog18", "mi308x-hjbog39"])
def test_the_mi308x_specs_rebuild_byte_for_byte_from_their_own_sweeps(
    monkeypatch, tmp_path, name
):
    monkeypatch.setenv("COMPASS_MACHINE", name)
    path = TREE / "scripts/compass/mi308x_spec.py"
    loaded = importlib.util.spec_from_file_location("mi308x_spec", path)
    script = importlib.util.module_from_spec(loaded)
    loaded.loader.exec_module(script)
    document = TREE / f"scripts/compass/machines/{name}.json"
    out = tmp_path / "rebuilt.json"
    assert script.main(["--sweep-from", str(document), "--out", str(out)]) == 0
    assert out.read_bytes() == document.read_bytes()
