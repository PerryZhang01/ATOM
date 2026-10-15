# SPDX-License-Identifier: MIT
"""The server driver's lifecycle, against a planted server command that records
its arguments, starts a child and answers ``/health``; the GPU checks run with
`test_drive_gpu.py`'s planted ``rocm-smi`` and ``rocminfo``.

``test_a_simulated_server_starts_answers_and_stops`` starts ATOM's API server
with ``--compass-run``, skipping without a driver and without
``ATOM_COMPASS_SLICE_MODEL``, as ``test_cctraces_sim_smoke.py`` does.
"""

import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from test_drive_gpu import FREE, LISTING, _plant

from atom.compass.drive import serve
from atom.compass.drive.gpu import GpuRefusal
from atom.compass.run import ENV

TREE = Path(__file__).resolve().parents[2]
PLANTED = """\
import http.server, json, os, subprocess, sys, time
d = os.environ["PLANTED_DIR"]
when, _, rc = os.environ.get("PLANTED_EXIT", "never:0").partition(":")
argv, keep = sys.argv[1:], ("HIP_VISIBLE_DEVICES", "PYTHONPATH", "ATOM_COMPASS_RUN")
keep += ("ATOM_COMPASS_PARITY_RECORD",)
with open(d + "/argv.json", "w") as f:
    json.dump({"argv": argv, "env": {k: os.environ.get(k) for k in keep}}, f)
if os.environ.get("PLANTED_ENGINE", "1") == "1":
    record = os.environ["ATOM_COMPASS_PARITY_RECORD"]
    os.makedirs(record)
    open(record + "/dp0.engine.json", "w").close()
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
with open(d + "/child", "w") as f:
    f.write(str(child.pid))
print("planted server up", flush=True)
if when == "early":
    sys.exit(int(rc))
time.sleep(float(os.environ.get("PLANTED_DELAY", "0")))

class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        open(d + "/answered", "w").close()
        self.send_response(200)
        self.end_headers()

port = int(argv[argv.index("--server-port") + 1])
server = http.server.HTTPServer(("127.0.0.1", port), Health)
server.handle_request()
if when == "late":
    sys.exit(int(rc))
server.serve_forever()
"""
#: Starts `serve` in a process of its own: the planted server, if any, the
#: directory, the run file's mapping, then ATOM's arguments.
DRIVE = """\
import json, sys
from atom.compass.drive import serve
if sys.argv[1]:
    serve.SERVER = (sys.executable, sys.argv[1])
    serve.identity = lambda *args: {"identity": "planted"}
serve.serve(sys.argv[4:], sys.argv[2], run=json.loads(sys.argv[3]))
"""
ARGS = ["--model", "m", "-tp", "2", "--kv-transfer-config", '{"kv_role": "x"}']
ARGS += ["--disagg-constrained"]


@pytest.fixture
def planted(tmp_path, monkeypatch):
    """The planted server, and an `identity` that records what it is called
    with and whether the server is still running then."""
    (tmp_path / "planted.py").write_text(PLANTED)
    monkeypatch.setattr(serve, "SERVER", (sys.executable, str(tmp_path / "planted.py")))
    monkeypatch.setenv("PLANTED_DIR", str(tmp_path))

    def identity(*args):
        out = Path(args[0]).parents[1]
        pid = json.loads((out / serve.READY_FILE).read_text())["pid"]
        return {"args": [str(a) for a in args], "server running": _running(pid)}

    monkeypatch.setattr(serve, "identity", identity)
    return tmp_path


def _running(pid) -> bool:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return False
    return stat.rsplit(")", 1)[1].split()[0] != "Z"


def _seen(planted: Path) -> dict:
    return json.loads((planted / "argv.json").read_text())


def _ready(out: Path, driver, polls: int, poll_s: float) -> dict:
    """The ready file, once the driver in `driver` has written it."""
    for _ in range(polls):
        if (out / serve.READY_FILE).exists():
            return json.loads((out / serve.READY_FILE).read_text())
        assert driver.poll() is None, (out / serve.LOG_FILE).read_text()[-4000:]
        time.sleep(poll_s)
    pytest.fail(f"no {serve.READY_FILE} after {polls * poll_s} s")


def test_atom_arguments_pass_verbatim_and_the_driver_assigns_the_rest(
    planted, monkeypatch
):
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    monkeypatch.setenv(ENV, "a run file the environment names")
    out = planted / "out"
    serve.serve(ARGS, out, run={"bound_s": 1.0})
    seen, run_file = _seen(planted), out / serve.RUN_FILE
    argv = seen["argv"]
    assert argv[: 2 + len(ARGS)] == [*serve.KEEP_ALIVE, *ARGS]
    assigned = dict(zip(argv[2 + len(ARGS) :: 2], argv[3 + len(ARGS) :: 2]))
    assert assigned.keys() == {"--host", "--server-port", "--port", "--compass-run"}
    assert assigned["--compass-run"] == str(run_file)
    run = json.loads(run_file.read_text())
    clock_port = run["clock_endpoint"].rsplit(":", 1)[1]
    assert len({assigned["--server-port"], assigned["--port"], clock_port}) == 3
    assert run == {
        "bound_s": 1.0,
        "out_dir": str(out),
        "clock_endpoint": f"tcp://127.0.0.1:{clock_port}",
    }
    assert seen["env"]["ATOM_COMPASS_RUN"] is None
    assert seen["env"]["ATOM_COMPASS_PARITY_RECORD"] == str(out / serve.STEPS)
    assert seen["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(serve.TREE)
    ready = json.loads((out / serve.READY_FILE).read_text())
    assert ready.pop("pid") > 0
    assert ready == {
        "url": f"http://127.0.0.1:{assigned['--server-port']}",
        "clock_endpoint": run["clock_endpoint"],
        "run_file": str(run_file),
    }
    assert "planted server up" in (out / serve.LOG_FILE).read_text()
    engine = out / serve.STEPS / "dp0.engine.json"
    assert json.loads((out / serve.RUN_JSON).read_text()) == {
        "args": [str(engine), "m", str(sys.argv), "()"],
        "server running": False,
    }


def test_a_real_server_runs_on_its_gpus_between_the_gpu_checks(planted, monkeypatch):
    out = _plant(planted, monkeypatch, FREE)
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    serve.serve(ARGS, out, gpus=(6, 7))
    seen = _seen(planted)
    assert seen["env"]["HIP_VISIBLE_DEVICES"] == "6,7"
    assert "--compass-run" not in seen["argv"]
    assert (out / "gpu_before.txt").exists() and (out / "gpu_after.txt").exists()
    assert json.loads((out / serve.READY_FILE).read_text()).keys() == {"url", "pid"}
    assert json.loads((out / serve.RUN_JSON).read_text())["args"][3] == "(6, 7)"


def test_a_process_on_a_named_gpu_refuses_before_the_server_starts(
    planted, monkeypatch
):
    out = _plant(planted, monkeypatch, LISTING)
    with pytest.raises(GpuRefusal, match="GPU 7 has PID 4242 on it before"):
        serve.serve(ARGS, out, gpus=(7,))
    assert not (planted / "argv.json").exists()


@pytest.mark.parametrize(
    "flag",
    [
        ["--host", "0.0.0.0"],
        ["--server_port=8000"],
        ["--port", "8006"],
        ["--compass-run", "f.json"],
        ["--compass_clock_endpoint", "tcp://h:1"],
        ["--server-p", "8000"],
        ["--torch-profiler-dir", "/tmp/p"],
    ],
)
def test_a_flag_the_driver_owns_refuses_before_the_server_starts(planted, flag):
    with pytest.raises(serve.ServeError, match=f"^{re.escape(flag[0])} "):
        serve.serve([*ARGS, *flag], planted / "out", run={})
    assert not (planted / "argv.json").exists()


@pytest.mark.parametrize(
    "args, kind, used, refusal",
    [
        (ARGS, {"gpus": (7,), "run": {}}, False, "exactly one"),
        (ARGS, {}, False, "exactly one"),
        (ARGS[2:], {"run": {}}, False, "no --model"),
        (ARGS, {"run": {}}, True, "earlier run's files"),
    ],
)
def test_a_server_is_refused_before_it_starts(planted, args, kind, used, refusal):
    out = planted / "out"
    if used:
        out.mkdir()
        (out / serve.RUN_JSON).write_text("{}")
    with pytest.raises(serve.ServeError, match=refusal):
        serve.serve(args, out, **kind)
    assert not (planted / "argv.json").exists()


def test_a_server_that_served_no_step_writes_no_run_json(planted, monkeypatch):
    monkeypatch.setenv("PLANTED_EXIT", "late:0")
    monkeypatch.setenv("PLANTED_ENGINE", "0")
    with pytest.raises(serve.ServeError, match="^the server served no step"):
        serve.serve(ARGS, planted / "out", run={})
    assert not (planted / "out" / serve.RUN_JSON).exists()


@pytest.mark.parametrize(
    "exit_, failure",
    [
        ("early:3", "^the server exited 3 before /health answered"),
        ("late:5", "^the server exited 5;"),
        ("late:0", None),
    ],
)
def test_a_server_that_exits_takes_its_process_group_with_it(
    planted, monkeypatch, exit_, failure
):
    monkeypatch.setenv("PLANTED_EXIT", exit_)
    with pytest.raises(serve.ServeError, match=failure) if failure else nullcontext():
        serve.serve(ARGS, planted / "out", run={})
    assert (planted / "out" / serve.READY_FILE).exists() == exit_.startswith("late")
    assert not _running((planted / "child").read_text())


@pytest.mark.parametrize("sig", [signal.SIGTERM, signal.SIGINT])
def test_the_ready_file_follows_the_first_answer_and_a_stop_leaves_nothing(
    planted, sig
):
    """The named result: a server that answers ``/health`` after 2 s."""
    out = planted / "out"
    env = dict(os.environ, PYTHONPATH=str(TREE), PLANTED_DELAY="2")
    driver = subprocess.Popen(
        [
            sys.executable,
            "-c",
            DRIVE,
            str(planted / "planted.py"),
            str(out),
            "{}",
            *ARGS,
        ],
        env=env,
    )
    server = _ready(out, driver, 600, 0.05)["pid"]
    assert (planted / "answered").exists()
    child = int((planted / "child").read_text())
    assert _running(server) and _running(child)
    driver.send_signal(sig)
    assert driver.wait(timeout=60) == 0
    assert not _running(server) and not _running(child)


def test_what_ignores_sigterm_is_killed_after_the_grace(monkeypatch):
    monkeypatch.setattr(serve, "STOP_GRACE_S", 1)
    ignore = "import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    stubborn = subprocess.Popen(
        [sys.executable, "-c", ignore + "print(flush=True)\ntime.sleep(600)"],
        stdout=subprocess.PIPE,
        start_new_session=True,
    )
    stubborn.stdout.readline()
    serve._stop(stubborn.pid)
    assert stubborn.wait(timeout=5) == -signal.SIGKILL


MODEL = os.environ.get("ATOM_COMPASS_SLICE_MODEL")


@pytest.mark.skipif(
    not (torch.cuda.is_available() and MODEL),
    reason="needs a driver and ATOM_COMPASS_SLICE_MODEL",
)
def test_a_simulated_server_starts_answers_and_stops(tmp_path):
    from test_cctraces_sim_smoke import _template

    run = json.loads(_template(tmp_path / "template.json").read_text())
    # The template adds the model's architecture, which may be there already.
    tokenizer = run["machine"]["host"]["tokenizers"][0]
    tokenizer["applies_to"] = list(dict.fromkeys(tokenizer["applies_to"]))
    out = tmp_path / "out"
    atom_args = ["--model", MODEL, "--enforce-eager", "--max-model-len", "2048"]
    driver = subprocess.Popen(
        [sys.executable, "-c", DRIVE, "", str(out), json.dumps(run), *atom_args],
        env=dict(os.environ, PYTHONPATH=str(TREE)),
        stderr=subprocess.PIPE,
        text=True,
    )
    ready = _ready(out, driver, 1800, 0.5)
    conn = http.client.HTTPConnection(ready["url"].removeprefix("http://"))
    conn.request("GET", "/health")
    assert conn.getresponse().status == 200
    assert json.loads(Path(ready["run_file"]).read_text())["out_dir"] == str(out)
    driver.send_signal(signal.SIGTERM)
    # No client sent a request, so no step wrote the engine file run.json needs.
    assert "the server served no step" in driver.communicate(timeout=300)[1]
    assert driver.returncode == 1 and not _running(ready["pid"])
