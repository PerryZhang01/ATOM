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

The listing's PIDs are the kernel's. The calling process, which may itself hold
a named GPU's memory (importing LMCache opens a HIP context), is left out by
the kernel PID `/proc/self/sched` prints, which differs from `os.getpid()` in a
container. The run's own processes are told apart by when the checks run: before
the run none has started, and after it every one has exited, so any other
process listed on a named GPU at either check is not this run's.

`--gpu` and the listing use rocm-smi's GPU indexes; HIP and KFD's memory number
GPUs by KFD node, the one `rocm-smi --showhw` lists for each (`_nodes`).
"""

import argparse
import re
import subprocess
from contextlib import contextmanager
from functools import cache, partial
from pathlib import Path

PREFLIGHT = Path(__file__).resolve().parents[3] / "scripts/compass/preflight.sh"
#: preflight.sh bounds its own rocminfo at 25 s; this bounds the rest of it.
PREFLIGHT_TIMEOUT_S = 300
SMI_TIMEOUT_S = 60
_ON = re.compile(r"PID (\d+) is using \d+ DRM device\(s\):\n([\d ]*)")
#: A `rocm-smi --showhw` row: the GPU's index, then its KFD node.
_NODE = re.compile(r"^(\d+) +(\d+) ", re.MULTILINE)
_PID = re.compile(r"PID (\d+) is using")
KFD = Path("/sys/class/kfd/kfd")
SCHED = Path("/proc/self/sched")


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


def _nodes(gpus: tuple[int, ...]) -> dict[int, int]:
    """Each rocm-smi GPU's KFD node; refuses a GPU of `gpus` it lacks, or a shared node."""
    node = {int(g): int(n) for g, n in _NODE.findall(smi("--showhw"))}
    if missing := [gpu for gpu in gpus if gpu not in node]:
        raise GpuRefusal(f"rocm-smi --showhw lists no GPU {missing}")
    if len(set(node.values())) != len(node):
        raise GpuRefusal(f"rocm-smi --showhw puts two GPUs on one KFD node: {node}")
    return node


def hip_visible(gpus: tuple[int, ...]) -> str:
    """`HIP_VISIBLE_DEVICES` for the rocm-smi GPUs `gpus`: each one's HIP
    ordinal is its KFD node's rank among the nodes `rocm-smi --showhw` lists."""
    node = _nodes(gpus)
    order = sorted(node.values())
    return ",".join(str(order.index(node[gpu])) for gpu in gpus)


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
    nodes = cache(partial(_nodes, gpus))
    me = re.search(r"\((\d+), #threads", SCHED.read_text()).group(1)
    return ", ".join(
        f"GPU {gpu} has PID {pid} on it"
        for pid in _PID.findall(listing)
        if pid != me
        for gpu in gpus
        if str(gpu) in on.get(pid, ()) or _vram(pid, nodes, gpu)
    )


def _vram(pid: str, nodes, gpu: int) -> int:
    """Bytes of GPU `gpu`'s memory KFD charges to `pid`; 0 once it has exited.
    KFD names the GPU by its node's `gpu_id`; `nodes()` gives that node.
    The topology is read only for a process KFD lists: a container without the
    GPU devices may not read it."""
    proc = KFD / "proc" / pid
    if not proc.is_dir():
        return 0
    try:
        gpu_id = (KFD / f"topology/nodes/{nodes()[gpu]}/gpu_id").read_text().strip()
        return int((proc / f"vram_{gpu_id}").read_text())
    except FileNotFoundError:
        return 0
