# SPDX-License-Identifier: MIT
"""Start one ATOM instance's API server, real or simulated, and serve until the
command is stopped or the server exits.

A server handed a run file is simulated, and co-hosts its clock authority. One
without a run file is real, on the GPUs the command names, which `gpu.quiet`
checks before the server starts and after it stops. ATOM's arguments pass
verbatim; the driver assigns the host, its ports and the run file, and
refuses them, and the clock endpoint flag, among ATOM's arguments. Each start
takes a free `ATOM_DP_MASTER_PORT`: the port keys ATOM's compiled-graph cache,
and the first start that loads that cache autotunes every graph's kernels
again, whose configs stay loaded and read in `non_torch`.

The server runs this driver's tree in its own process group, so its engine
processes stop with it, and writes ``<out>/server.log`` and its step record in
``<out>/steps``. Once its ``/health`` first answers, ``<out>/server.json``
names its URL and process id and, for a simulated server, its clock endpoint
and run file. SIGINT or SIGTERM to the command, or the server's exit, stops the
whole group: SIGTERM first, SIGKILL for what is left after `STOP_GRACE_S`.
Then ``<out>/run.json`` gets the run's identity, read off the engine file its
first step wrote. No command starts the client.
"""

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path

import atom
from atom.compass import parity
from atom.compass.drive.gpu import hip_visible, quiet
from atom.compass.drive.identity import identity
from atom.compass.run import CLOCK_ENV, ENV, _write_whole

SERVER = (sys.executable, "-m", "atom.entrypoints.openai.api_server")
HOST = "127.0.0.1"
#: Flags ATOM's arguments may not carry, in either spelling ATOM's parser
#: accepts or any abbreviation argparse expands: the driver assigns the host,
#: its ports and the run file, and a simulated server co-hosts its clock.
OWNED = (
    "--host",
    "--server-port",
    "--port",
    "--compass-run",
    "--compass-clock-endpoint",
    "--data-parallel-master-port",
)
#: ATOM's profiler: only a profiled pass may time ops, and no command runs one yet.
PROFILER = "--torch-profiler-dir"
#: As `cctraces_sim.sh` sets it, so a pooling client's idle connection is not
#: closed under it; before ATOM's arguments, so one there wins.
KEEP_ALIVE = ("--timeout-keep-alive", "1000000")
READY_FILE, LOG_FILE, RUN_JSON, STEPS = "server.json", "server.log", "run.json", "steps"
RUN_FILE = "run_file.json"
POLL_S = 0.5
#: Seconds the group has to leave after SIGTERM. ATOM's server waits
#: ATOM_SHUTDOWN_TIMEOUT_S for its engines, and they as long for their workers.
STOP_GRACE_S = 60
TREE = Path(atom.__file__).resolve().parents[1]


class ServeError(ValueError):
    """ATOM's arguments carry a flag the driver owns, or the server failed; a
    `ValueError`, so the console script exits with its message."""


def serve(atom_args, out, gpus=(), run=None, extra_env=None) -> None:
    """Serve one instance from ATOM's arguments `atom_args`, writing to the
    directory `out`, until SIGINT or SIGTERM or the server's exit.

    A real server names its GPUs in `gpus`; a simulated one is handed `run`,
    the run file's mapping, whose ``out_dir`` and ``clock_endpoint`` are set
    here. `extra_env` maps variables the server's environment holds over the
    command's. Raises `ServeError` when the server exits before ``/health``
    answers, or on its own with a non-zero status, and `GpuRefusal` from the
    GPU checks. Once the server has stopped, writes run.json from the engine
    file, and raises `ServeError` when no step wrote one.
    """
    _refuse(atom_args)
    model = _model(atom_args)
    if (run is None) == (not gpus):
        raise ServeError(
            "a server is real, on the GPUs it names, or simulated, from a run "
            f"file, exactly one; got gpus={gpus} and a run file: {run is not None}"
        )
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ServeError(f"{out} holds an earlier run's files; one run per directory")
    out.mkdir(parents=True, exist_ok=True)
    server_port, engine_port, clock_port, dp_port = _free_ports(4)
    argv = [*SERVER, *KEEP_ALIVE, *atom_args, "--host", HOST]
    argv += ["--server-port", str(server_port), "--port", str(engine_port)]
    ready = {"url": f"http://{HOST}:{server_port}"}
    # Nor can the environment hand the server a run file or a clock.
    env = {**os.environ, **(extra_env or {})}
    env = {k: v for k, v in env.items() if k not in (ENV, CLOCK_ENV)}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TREE), env.get("PYTHONPATH")])
    )
    env[parity.ENV] = str(out / STEPS)
    env["ATOM_DP_MASTER_PORT"] = str(dp_port)
    if run is None:
        env["HIP_VISIBLE_DEVICES"] = hip_visible(gpus)
    else:
        endpoint, run_file = f"tcp://{HOST}:{clock_port}", out / RUN_FILE
        run = {**run, "out_dir": str(out), "clock_endpoint": endpoint}
        run_file.write_text(json.dumps(run))
        argv += ["--compass-run", str(run_file)]
        ready |= {"clock_endpoint": endpoint, "run_file": str(run_file)}
    with _stop_signals() as stopped, quiet(gpus, out) if gpus else nullcontext():
        with (out / LOG_FILE).open("w") as log:
            server = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        try:
            while not (stopped or _answers(server_port)):
                if server.poll() is not None:
                    raise ServeError(
                        f"the server exited {server.returncode} before /health "
                        f"answered; see {out / LOG_FILE}"
                    )
                time.sleep(POLL_S)
            if not stopped:
                ready["pid"] = server.pid
                _write_whole(out / READY_FILE, json.dumps(ready))
            while not stopped and server.poll() is None:
                time.sleep(POLL_S)
        finally:
            _stop(server.pid)
            server.wait()
    if not stopped and server.returncode:
        raise ServeError(f"the server exited {server.returncode}; see {out / LOG_FILE}")
    # Any rank's: every DP rank resolves one engine configuration.
    engines = sorted((out / STEPS).glob("dp*.engine.json"))
    if not engines:
        raise ServeError(
            "the server served no step, so no engine file states the engine it "
            f"resolved, and {out / RUN_JSON} is not written"
        )
    _write_whole(
        out / RUN_JSON, json.dumps(identity(engines[0], model, sys.argv, gpus))
    )


def _model(atom_args) -> str:
    """The model ATOM's arguments name, whose config.json the host key digests."""
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--model")
    model = parser.parse_known_args(atom_args)[0].model
    if not model:
        raise ServeError(
            "ATOM's arguments name no --model; run.json digests its config"
        )
    return model


def _refuse(atom_args) -> None:
    for arg in atom_args:
        flag = arg.split("=", 1)[0].replace("_", "-")
        if len(flag) <= 2 or not flag.startswith("--"):
            continue
        if any(f.startswith(flag) for f in OWNED):
            raise ServeError(f"{arg} is among ATOM's arguments; the driver owns it")
        if PROFILER.startswith(flag):
            raise ServeError(f"{arg} profiles the server, and only a profiled pass may")


def _free_ports(n: int) -> list[int]:
    """`n` free ports, bound together so no two are the same."""
    socks = [socket.socket() for _ in range(n)]
    try:
        for s in socks:
            s.bind((HOST, 0))
        return [s.getsockname()[1] for s in socks]
    finally:
        for s in socks:
            s.close()


def _answers(port: int) -> bool:
    """Whether the server's ``/health`` answers 200."""
    conn = http.client.HTTPConnection(HOST, port, timeout=5)
    try:
        conn.request("GET", "/health")
        return conn.getresponse().status == 200
    except (OSError, http.client.HTTPException):
        return False
    finally:
        conn.close()


@contextmanager
def _stop_signals():
    """A list SIGINT and SIGTERM append to instead of ending the command, so
    the server is stopped before the command goes on."""
    got = []
    old = {
        s: signal.signal(s, lambda signum, frame: got.append(signum))
        for s in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        yield got
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


def _stop(group: int) -> None:
    """SIGTERM the process group `group`; SIGKILL what is left after
    `STOP_GRACE_S`."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return
        for _ in range(int(STOP_GRACE_S / POLL_S)):
            if not _live(group):
                return
            time.sleep(POLL_S)


def _live(group: int) -> bool:
    """Whether a process of `group` has not exited; a zombie has."""
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            state, _, pgrp = stat.read_text().rsplit(")", 1)[1].split()[:3]
        except OSError:
            continue
        if int(pgrp) == group and state != "Z":
            return True
    return False
