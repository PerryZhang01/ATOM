# SPDX-License-Identifier: MIT
"""Probe one GPU host's attainable peaks and print them as a spec fragment.

    HIP_VISIBLE_DEVICES=a,b python3 scripts/compass/probe_device.py SPEC OUT

SPEC is the machine file the peaks belong to. Card 0 runs the one-card probes,
cards 0 and 1 the link. Every sample is device time between two events queued
behind a spin kernel, so no host time is in it.

- Compute: per format, the median of `REPS` GEMMs ``x @ w.T`` at each of
  `SHAPES`, BF16 through `torch.matmul` and FP8 through `torch._scaled_mm`;
  the best shape's rate over SPEC's ``*_flops`` is the format's derate. Each
  GEMM follows a spin, so the rate is a burst's: with PTL enabled a sustained
  GEMM slows within a second or two.
- Bandwidth: a `copy_` of half of each of `SIZES`, the bytes read plus
  written, each after a write of `FLUSH` bytes so it starts from HBM; the
  median rate per size is ``memory.bandwidth_curve`` and its largest point over
  SPEC's bandwidth is ``memory.derate``.
- Kernel floor: `GRAPH` empty kernels captured in one graph, replayed `REPS`
  times; ``kernel_floor_s`` is the median per kernel, with its p10 and p90.
- Link: one kernel per card passes a counter `ROUNDS` times through the other
  card's memory; half a round trip is ``link_latency_s``, with its p10 and p90
  over `REPS` runs. A `LINK_BYTES` copy from card 0 to card 1 over SPEC's link
  bandwidth is ``intra_node.derate``.

The run is refused unless every card is idle (under `IDLE_BYTES` of VRAM used,
0% busy), before and after, and reads SPEC's ``device.ptl`` from
``ptl/ptl_enable`` under its sysfs device (``disabled`` where it has none). OUT
is the record: the cards, every sample and this file's text; the fragment
printed names its sha256.
"""

import datetime
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

import triton
import triton.language as tl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atom.compass.spec import Fragment, MachineSpec, SpecRefusal  # noqa: E402
from atom.compass.spec.rules import Rule  # noqa: E402

SHAPES = (4096, 8192, 16384)
SIZES = tuple(2**k for k in range(12, 33))  # 4 KiB to 4 GiB moved
FLUSH = 2**29  # twice the MI300 series' 256 MiB last-level cache
GRAPH, ROUNDS, LINK_BYTES = 1000, 1000, 2**30
REPS = 20
SPIN = 10**7  # device cycles a sample waits behind, well past its host calls
IDLE_BYTES = 2**30
DRM = "/sys/class/drm"
#: Each format's derate, by the spec-peak number it is taken over.
FORMATS = {"bf16": "bf16_flops", "fp8": "fp8_flops"}
#: The fields read with a band, and each one's sample list in the record.
BANDED = {
    "device.kernel_floor_s": "floor",
    "interconnect.intra_node.link_latency_s": "one_hop",
}


@triton.jit
def _pingpong(mine, theirs, rounds, serve: tl.constexpr):
    """Write `theirs`, on the other card, then wait to see `mine` written
    back, `rounds` times; the side that does not serve answers each round."""
    for i in range(1, rounds + 1):
        if serve:
            tl.atomic_xchg(theirs, i, sem="release", scope="sys")
        seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
        while seen < i:
            seen = tl.atomic_add(mine, 0, sem="acquire", scope="sys")
        if not serve:
            tl.atomic_xchg(theirs, i, sem="release", scope="sys")


def cards(bdfs: list[str], drm: str = DRM) -> list[dict]:
    """Each card's sysfs reading by PCI address: VRAM used, busy and PTL."""
    found = {}
    for card in Path(drm).glob("card[0-9]*"):
        device = card / "device"
        if "-" not in card.name and device.exists():
            found[device.resolve().name] = device
    out = []
    for bdf in bdfs:
        device = found[bdf]
        ptl = device / "ptl" / "ptl_enable"
        out.append(
            {
                "bdf": bdf,
                "vram_used": int((device / "mem_info_vram_used").read_text()),
                "busy": int((device / "gpu_busy_percent").read_text()),
                "ptl": ptl.read_text().strip() if ptl.exists() else "disabled",
            }
        )
    return out


def refuse_unfit(read: list[dict], ptl: str) -> None:
    """Refuse cards that hold other work or disagree with the spec's PTL."""
    wrong = [
        f"{c['bdf']} (ptl {c['ptl']}, {c['vram_used']} B used, {c['busy']}% busy)"
        for c in read
        if c["ptl"] != ptl or c["vram_used"] > IDLE_BYTES or c["busy"]
    ]
    if wrong:
        raise SpecRefusal(
            Rule.MEASURED,
            f"cards {', '.join(wrong)} are not idle cards with PTL {ptl}",
            "pick idle cards whose ptl_enable is the spec's `device.ptl`; a peak "
            "read beside a neighbour or under another PTL state is not this "
            "machine's",
        )


def _timed(torch, fn, before=None, reps: int = REPS) -> list[float]:
    """Device seconds of `reps` calls of `fn`, each behind a spin and `before`."""
    out = []
    for _ in range(reps):
        torch.cuda._sleep(SPIN)
        if before:
            before()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        fn()
        end.record()
        end.synchronize()
        out.append(start.elapsed_time(end) / 1e3)
    return out


def probe(torch) -> dict:
    """Every sample, on visible cards 0 and 1."""
    torch.cuda.set_device(0)
    fp8 = torch.float8_e4m3fn
    if torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx942"):
        fp8 = torch.float8_e4m3fnuz
    one = torch.ones((), device="cuda")
    gemm = {"bf16": {}, "fp8": {}}
    for n in SHAPES:
        x, w = (torch.randn(n, n, device="cuda", dtype=torch.bfloat16) for _ in "xw")
        gemm["bf16"][n] = _timed(torch, lambda: torch.matmul(x, w.T))
        x, w = x.to(fp8), w.to(fp8)
        gemm["fp8"][n] = _timed(
            torch, lambda: torch._scaled_mm(x, w.T, one, one, out_dtype=torch.bfloat16)
        )
        del x, w
        time.sleep(1)  # let a PTL card recover before the next burst
    flush = torch.empty(FLUSH, dtype=torch.uint8, device="cuda")
    copy = {}
    for size in SIZES:
        src = torch.empty(size // 2, dtype=torch.uint8, device="cuda")
        dst = torch.empty_like(src)
        copy[size] = _timed(torch, lambda: dst.copy_(src), flush.zero_)
        del src, dst
    del flush
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(GRAPH):
            torch.cuda._sleep(0)
    floor = [t / GRAPH for t in _timed(torch, graph.replay)]

    mine = torch.zeros(1, dtype=torch.int32, device="cuda:0")
    theirs = torch.zeros(1, dtype=torch.int32, device="cuda:1")
    near = torch.empty(LINK_BYTES, dtype=torch.uint8, device="cuda:0")
    far = torch.empty_like(near, device="cuda:1")
    far.copy_(near)  # also maps each card's memory into the other
    near.copy_(far)
    link = [LINK_BYTES / t for t in _timed(torch, lambda: far.copy_(near))]
    del near, far
    trips = []
    for _ in range(REPS + 1):  # the first compiles both kernels
        mine.zero_()
        theirs.zero_()
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        with torch.cuda.device(1):
            _pingpong[(1,)](theirs, mine, ROUNDS, serve=False, num_warps=1)

        def serve():
            _pingpong[(1,)](mine, theirs, ROUNDS, serve=True, num_warps=1)

        trips += _timed(torch, serve, reps=1)
        torch.cuda.synchronize(1)
    return {
        "gemm": gemm,
        "copy": copy,
        "floor": floor,
        "link": link,
        "one_hop": [t / ROUNDS / 2 for t in trips[1:]],
    }


def readings(record: dict) -> dict:
    """The attainable values a record's samples give: each format's best
    median rate, the median copy rate per size, the link's median rate, and
    each banded field's median, p10 and p90."""
    med = statistics.median

    def band(samples):
        cut = statistics.quantiles(samples, n=10, method="inclusive")
        return [med(samples), cut[0], cut[8]]

    found = {
        f: max(2 * int(n) ** 3 / med(t) for n, t in record["gemm"][f].items())
        for f in FORMATS
    }
    found["bandwidth_curve"] = [
        [int(n), int(n) / med(t)] for n, t in record["copy"].items()
    ]
    found["link"] = med(record["link"])
    return found | {path: band(record[key]) for path, key in BANDED.items()}


def fragment(
    name: str, found: dict, peaks: dict, author: str, date: str, subject: dict
) -> dict:
    """The fragment `readings` make on machine `name`, whose spec-peak numbers
    `peaks` holds by path, every field probed on `subject`."""
    curve = found["bandwidth_curve"]
    values = {
        **{
            f"device.compute.derate.{f}": found[f] / peaks[f"device.compute.{peak}"]
            for f, peak in FORMATS.items()
        },
        "device.memory.bandwidth_curve": curve,
        "device.memory.derate": curve[-1][1]
        / peaks["device.memory.bandwidth_bytes_per_s"],
        "interconnect.intra_node.derate": found["link"]
        / peaks["interconnect.intra_node.link_bandwidth_bytes_per_s"],
        **{path: found[path][0] for path in BANDED},
    }
    origins = {path: {"origin": "probed", **subject} for path in values}
    for path in BANDED:
        origins[path] |= {"p10": found[path][1], "p90": found[path][2]}
    document = {
        "schema_version": 1,
        "name": name,
        "provenance": {
            "authored_by": author,
            "date": date,
            "origin": "mixed",
            "origins": origins,
            "notes": "probe_device.py: GEMM, copy, graph and link samples",
        },
    }
    for path, value in values.items():
        *blocks, leaf = path.split(".")
        node = document
        for block in blocks:
            node = node.setdefault(block, {})
        node[leaf] = value
    return document


def main(argv: list[str] | None = None) -> int:
    import torch

    from atom.compass.artifacts.provenance import git_tree_root

    spec_path, out = (Path(a) for a in (argv or sys.argv[1:]))
    spec = MachineSpec.from_mapping(json.loads(spec_path.read_text()))
    ptl = spec.values.get("device.ptl", "disabled")
    props = [torch.cuda.get_device_properties(i) for i in (0, 1)]
    bdfs = [
        f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
        for p in props
    ]
    record = {"cards": cards(bdfs), "arch": props[0].gcnArchName}
    refuse_unfit(record["cards"], ptl)
    record |= probe(torch)
    torch.cuda.empty_cache()
    record["after"] = cards(bdfs)
    record |= {"torch": torch.__version__, "source": Path(__file__).read_text()}
    out.write_text(json.dumps(record) + "\n")
    refuse_unfit(record["after"], ptl)
    root = git_tree_root("atom", str(ROOT))
    subject = {"source_root": f"{root.revision_kind}:{root.revision}"}
    subject["record"] = f"sha256:{hashlib.sha256(out.read_bytes()).hexdigest()}"
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    document = fragment(
        spec.value("name"),
        readings(record),
        spec.values,
        "probe_device.py",
        today,
        subject,
    )
    Fragment.from_mapping(document, "probe")  # refuses a derate above 1
    print(json.dumps(document, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
