# SPDX-License-Identifier: MIT
"""Start one ATOM instance's real API server, and serve until the command is
stopped or the server exits.

The server runs on the GPUs the command names, which `gpu.quiet` checks
before it starts and after it stops. ATOM's arguments pass verbatim; the
driver assigns the host and both ports, and refuses them, and the Compass
flags that would make the server a simulation, among ATOM's arguments.

The server runs this driver's tree in its own process group, so its engine
processes stop with it, and writes ``<out>/server.log`` and its step record in
``<out>/steps``. Once its ``/health`` first answers, ``<out>/server.json``
names its URL and process id. SIGINT or SIGTERM to the command, or the
server's exit, stops the whole group: SIGTERM first, SIGKILL for what is left
after `STOP_GRACE_S`. Then ``<out>/run.json`` gets the run's identity, read
off the engine file its first step wrote. No command starts the client.
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
from contextlib import contextmanager
from pathlib import Path

import atom
from atom.compass import parity
from atom.compass.drive.gpu import quiet
from atom.compass.drive.identity import identity
from atom.compass.run import CLOCK_ENV, ENV, _write_whole

SERVER = (sys.executable, "-m", "atom.entrypoints.openai.api_server")
HOST = "127.0.0.1"
#: Flags ATOM's arguments may not carry, in either spelling ATOM's parser
#: accepts or any abbreviation argparse expands: the driver assigns the host and
#: both ports, and the server it starts is real, with no run file or clock.
OWNED = (
    "--host",
    "--server-port",
    "--port",
    "--compass-run",
    "--compass-clock-endpoint",
)
#: ATOM's profiler: only a profiled pass may time ops, and no command runs one yet.
PROFILER = "--torch-profiler-dir"
#: As `cctraces_sim.sh` sets it, so a pooling client's idle connection is not
#: closed under it; before ATOM's arguments, so one there wins.
KEEP_ALIVE = ("--timeout-keep-alive", "1000000")
READY_FILE, LOG_FILE, RUN_JSON, STEPS = "server.json", "server.log", "run.json", "steps"
POLL_S = 0.5
#: Seconds the group has to leave after SIGTERM. ATOM's server waits
#: ATOM_SHUTDOWN_TIMEOUT_S for its engines, and they as long for their workers.
STOP_GRACE_S = 60
TREE = Path(atom.__file__).resolve().parents[1]


class ServeError(ValueError):
    """ATOM's arguments carry a flag the driver owns, or the server failed; a
    `ValueError`, so the console script exits with its message."""


def serve(atom_args, out, gpus) -> None:
    """Serve one instance from ATOM's arguments `atom_args` on the GPUs `gpus`,
    writing to the directory `out`, until SIGINT or SIGTERM or the server's
    exit.

    Raises `ServeError` when the server exits before ``/health`` answers, or on
    its own with a non-zero status, and `GpuRefusal` from the GPU checks. Once
    the server has stopped, writes run.json from the engine file, and raises
    `ServeError` when no step wrote one.
    """
    _refuse(atom_args)
    model = _model(atom_args)
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise ServeError(f"{out} holds an earlier run's files; one run per directory")
    out.mkdir(parents=True, exist_ok=True)
    server_port, engine_port = _free_ports(2)
    argv = [*SERVER, *KEEP_ALIVE, *atom_args, "--host", HOST]
    argv += ["--server-port", str(server_port), "--port", str(engine_port)]
    ready = {"url": f"http://{HOST}:{server_port}"}
    # Nor can the environment hand the server a run file or a clock.
    env = {k: v for k, v in os.environ.items() if k not in (ENV, CLOCK_ENV)}
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(TREE), env.get("PYTHONPATH")])
    )
    env[parity.ENV] = str(out / STEPS)
    env["HIP_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    with _stop_signals() as stopped, quiet(gpus, out):
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
