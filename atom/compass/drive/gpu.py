# SPDX-License-Identifier: MIT
"""The checks a GPU command makes before it starts its server or engine and
after it stops: the node passes `scripts/compass/preflight.sh`, and no process
is on a GPU the run names.

A command names every GPU it uses by `--gpu`; there is no default device. Both
checks keep `rocm-smi`'s process listing in the run's directory, as
`gpu_before.txt` and `gpu_after.txt`, and read it as `cctraces_sim.sh` does: a
process is on a GPU when its `PID <n> is using <k> DRM device(s):` entry lists
that GPU's index, or when KFD charges it memory of that GPU
(`/sys/class/kfd/kfd/proc/<n>/vram_<gpu_id>`), which also counts a process that
holds memory there without having opened the GPU's DRM device.

The listing's PIDs are the kernel's, which a process in a container cannot
match against its own, so this run's processes are told apart by when the
checks run rather than by PID: before the run none has started, and after it
every one has exited, so a process listed on a named GPU at either check is not
this run's.
"""

import argparse
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

PREFLIGHT = Path(__file__).resolve().parents[3] / "scripts/compass/preflight.sh"
#: preflight.sh bounds its own rocminfo at 25 s; this bounds the rest of it.
PREFLIGHT_TIMEOUT_S = 300
SMI_TIMEOUT_S = 60
_ON = re.compile(r"PID (\d+) is using \d+ DRM device\(s\):\n([\d ]*)")
_PID = re.compile(r"PID (\d+) is using")
KFD = Path("/sys/class/kfd/kfd")


class GpuRefusal(Exception):
    """A GPU the run did not have to itself, or a node preflight.sh failed on."""


def add_gpu_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gpu",
        required=True,
        type=_gpus,
        help="every GPU the run uses, comma-separated, as rocm-smi indexes them; "
        "there is no default device",
    )


def _gpus(text: str) -> tuple[int, ...]:
    gpus = tuple(int(p) for p in text.split(",") if p.isdigit())
    if len(gpus) != len(text.split(",")) or len(set(gpus)) != len(gpus):
        raise argparse.ArgumentTypeError(
            f"{text!r} is not a comma-separated list of distinct GPU indices"
        )
    return gpus


def smi(*args: str) -> str:
    """What `rocm-smi <args>` prints, or a refusal when it fails."""
    done = subprocess.run(
        ["rocm-smi", *args],
        capture_output=True,
        text=True,
        timeout=SMI_TIMEOUT_S,
        check=False,
    )
    if done.returncode:
        raise GpuRefusal(
            f"rocm-smi {' '.join(args)} exited {done.returncode}: {done.stderr.strip()}"
        )
    return done.stdout


@contextmanager
def quiet(gpus: tuple[int, ...], out: Path):
    """Refuse to start the run unless the node passes preflight.sh and no
    process is on a GPU in `gpus`; at the run's stop, however it ended, check
    the same and fail the run, naming each process found.

    Enter it before the run touches a GPU, and leave it once the run's
    processes have exited.
    """
    if held := _check(gpus, out, "before"):
        raise GpuRefusal(f"{held} before the run, so it was not started")
    try:
        yield
    finally:
        if held := _check(gpus, out, "after"):
            raise GpuRefusal(f"{held} after the run, so it fails")


def _check(gpus: tuple[int, ...], out: Path, when: str) -> str:
    """Run preflight.sh and save the process listing; name what is on `gpus`."""
    log = out / f"preflight_{when}.txt"
    with log.open("w") as f:
        # A file, not a pipe: a rocminfo wedged past preflight's own timeout
        # cannot be killed and would hold a pipe open for ever.
        flight = subprocess.Popen(
            ["bash", str(PREFLIGHT)], stdout=f, stderr=subprocess.STDOUT
        )
        try:
            rc = flight.wait(PREFLIGHT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            flight.kill()
            rc = f"no exit in {PREFLIGHT_TIMEOUT_S} s"
    if rc != 0:
        raise GpuRefusal(f"preflight.sh failed {when} the run ({rc}); see {log}")
    listing = smi("--showpids", "--showpidgpus")
    (out / f"gpu_{when}.txt").write_text(listing)
    on = {pid: drm.split() for pid, drm in _ON.findall(listing)}
    return ", ".join(
        f"GPU {gpu} has PID {pid} on it"
        for pid in _PID.findall(listing)
        for gpu in gpus
        if str(gpu) in on.get(pid, ()) or _vram(pid, gpu)
    )


def _vram(pid: str, gpu: int) -> int:
    """Bytes of GPU `gpu`'s memory KFD charges to `pid`; 0 once it has exited.
    GPU indices follow KFD's GPU nodes in order, as rocm-smi's and HIP's do."""
    try:
        nodes = sorted((KFD / "topology/nodes").iterdir(), key=lambda n: int(n.name))
        ids = [i for n in nodes if (i := (n / "gpu_id").read_text().strip()) != "0"]
        return int((KFD / "proc" / pid / f"vram_{ids[gpu]}").read_text())
    except FileNotFoundError:
        return 0
