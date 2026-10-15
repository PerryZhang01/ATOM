# SPDX-License-Identifier: MIT
"""The mi355-gpu-45 MI355X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi355x_spec.py [--model M] [--out PATH]

Run on mi355-gpu-45, in a container that has the model's files; the fragment
and sweep functions are `mi308x_spec.py`'s. The device fragment is the readings
ATOM's own `ModelRunner` took at one DeepSeek-V4-Pro engine start at the
nightly DPA-TP8 prefill side's arguments (`FLAGS`; expert parallelism and MTP
off). Every rank of it runs attention at TP 1 inside 8-wide DP groups, so the
reserve and the load residue are keyed by the layout 1x8 and the activations
entry by width 1. The device peaks are AMD's MI355X datasheet; every other
field is assumed. Writes the merged document to `--out` when `validate`
passes at 1x8, then prints `explain`'s rows by origin and the basis of
`kv_blocks`.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mi308x_spec as base
from mi308x_spec import fragment, swept

from atom.compass.spec import Fragment, merge, validate

NAME = "mi355x-gpu45"
# The shared fragment and sweep functions name the machine they describe.
base.NAME = NAME
MODEL = "/models/DeepSeek-V4-Pro"
LAYOUT = "1x8"
FLAGS = "-tp 8 --enable-dp-attention --enable-tbo"

#: What ATOM's `ModelRunner` read at the start, at commit COMMIT (image
#: nightly-20261013): `non_torch`, the minimum over the 8 ranks; from that
#: rank, `allocated` and the unique parameter and buffer bytes right after the
#: model loaded, then the allocator's current and peak after warmup.
DATE, COMMIT = "2026-10-16", "0000000"
READ = {
    "total": 0,
    "non_torch": 0,
    "loaded": 0,
    "parameters": 0,
    "buffers": 0,
    "current": 0,
    "peak": 0,
    "warmup_tokens": 0,
}
#: The sha256 of the start's server log.
RECORD = "0" * 64
SUBJECT = {"model": MODEL, "width": 8, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"
#: The sha256 of the model's config.json, which the run fingerprints.
CONFIG_JSON = "sha256:5fe4568daee51c208cb8a79538eaeda090ae011ade1dee2c386aa95f569c810e"

DEVICE = {
    "host": {"cpu": {"cores_physical": 128, "cores_logical": 256}},
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {"capacity_bytes": READ["total"]},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {LAYOUT: READ["non_torch"]},
            "allocator_retained_after_load_bytes": {
                LAYOUT: READ["loaded"] - READ["parameters"] - READ["buffers"]
            },
            "persistent_forward_buffer_bytes": READ["current"] - READ["loaded"],
        },
        "activations": [
            {
                "id": "deepseek-v4-pro",
                "applies_to": ["DeepseekV4ForCausalLM"],
                "fingerprint": CONFIG_JSON,
                "bytes_per_token": {
                    1: (READ["peak"] - READ["current"]) / READ["warmup_tokens"]
                },
            }
        ],
        "software_pinned_to": {"rocm": "7.2.4", "aiter": "9205e1059", "rccl": "2.27.7"},
    },
}

#: AMD Instinct MI355X GPU and platform datasheets: dense peaks, one Infinity
#: Fabric link, and the eight GPUs one fabric connects.
DATASHEET = {
    "device": {
        "memory": {"bandwidth_bytes_per_s": 8.0e12},
        "compute": {
            "bf16_flops": 2.5166e15,
            "fp8_flops": 5.0332e15,
            "fp4_flops": 1.00664e16,
        },
    },
    "interconnect": {
        "intra_node": {"domain_devices": 8, "link_bandwidth_bytes_per_s": 1.536e11}
    },
}

#: No probe fills these: each derate is the haircut this spec's author chose,
#: the inter-node link is one 400 Gb/s NIC's line rate until an RDMA pair is
#: measured, and the rest are the schema's example values.
DECLARED = {
    "host": {
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    },
    "device": {
        "memory": {"derate": 0.85},
        "compute": {"derate": 0.70},
        "runtime_constants": {
            "cudagraph_pool": {
                "w1_base_bytes": 95.5e6,
                "w1_bytes_per_captured_token": 0.318e6,
                "w_gt1_flat_bytes": 109.0e6,
            }
        },
    },
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
        "router_relay_s": 1.5e-3,
    },
}


def device_fragments() -> list[Fragment]:
    """The start's readings, the datasheet's peaks and the assumed rest."""
    notes = f"one DeepSeek-V4-Pro {FLAGS} start at {COMMIT}"
    return [
        fragment("device", DEVICE, "probed", notes, SUBJECT, DATE),
        fragment("datasheet", DATASHEET, "datasheet", "MI355X datasheet", date=DATE),
        fragment("assumed", DECLARED, "assumed", "no probe fills these", date=DATE),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default=MODEL)
    parser.add_argument(
        "--out",
        type=Path,
        default=base.ROOT / "scripts/compass/machines" / f"{NAME}.json",
    )
    args = parser.parse_args(argv)
    merged = merge([swept(args.model), *device_fragments()])
    verdict = validate(merged, tp_widths=(LAYOUT,))
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged, tp_width=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
