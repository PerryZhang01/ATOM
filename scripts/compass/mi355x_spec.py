# SPDX-License-Identifier: MIT
"""The mi355-gpu-45 MI355X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi355x_spec.py [--out PATH]

The device fragment is one DeepSeek-V4-Pro start at the nightly DPA-TP8 prefill
side's arguments (`FLAGS`): the reserve and load residue sit at layout 1x8, the
activations at width 1. The peaks are the datasheet's; the rest is assumed.
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
base.NAME = NAME  # the shared fragment and sweep functions name this machine
MODEL = "/models/DeepSeek-V4-Pro"
LAYOUT = "1x8"
FLAGS = "-tp 8 --enable-dp-attention --enable-tbo"

#: What `ModelRunner` read at the second of two starts (warm compile cache): the
#: minimum `non_torch` over the 8 ranks; that rank's bytes after the model loaded
#: and the allocator's current and peak after warmup.
DATE, COMMIT = "2026-10-16", "409bedc98"
READ = {
    "total": 309220868096,
    "non_torch": 13291749376,
    "loaded": 136926286848,
    "parameters": 132113063884,
    "buffers": 2039971840,
    "current": 139247200256,
    "peak": 148339363840,
    "warmup_tokens": 16384,
}
#: The sha256 of the start's server log.
RECORD = "02618dade3d07bcaac510da11cf8a279a1fe3ef2fce72d41e88a76f2df7c32c6"
RESIDUE = READ["loaded"] - READ["parameters"] - READ["buffers"]
PER_TOKEN = (READ["peak"] - READ["current"]) / READ["warmup_tokens"]
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
            "allocator_retained_after_load_bytes": {LAYOUT: RESIDUE},
            "persistent_forward_buffer_bytes": READ["current"] - READ["loaded"],
        },
        "activations": [
            {
                "id": "deepseek-v4-pro",
                "applies_to": ["DeepseekV4ForCausalLM"],
                "fingerprint": CONFIG_JSON,
                "bytes_per_token": {1: PER_TOKEN},
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
    out = Path(__file__).resolve().parent / "machines" / f"{NAME}.json"
    parser.add_argument("--out", type=Path, default=out)
    args = parser.parse_args(argv)
    merged = merge([swept(MODEL), *device_fragments()])
    verdict = validate(merged, tp_widths=(LAYOUT,))
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged, tp_width=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
