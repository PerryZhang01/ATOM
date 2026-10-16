# SPDX-License-Identifier: MIT
"""The mi355-gpu-45 MI355X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi355x_spec.py [--sweep-from DOC] [--out PATH]

The device fragment is one DeepSeek-V4-Pro start at the nightly DPA-TP8 prefill
side's arguments (`FLAGS`): the reserve and load residue sit at layout 1x8, the
activations at width 1. The peaks are the datasheet's; the rest is assumed.
`--sweep-from` takes the tokenizer sweep from a document this script wrote.
"""

import argparse
import copy
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

#: AMD Instinct MI355X GPU and platform datasheets: dense peaks, the eight GPUs
#: one fabric connects, and one Infinity Fabric link's one-way rate, half its
#: 153.6 GB/s bidirectional, as a KV transfer streams one way.
DATASHEET = {
    "device": {
        "memory": {"bandwidth_bytes_per_s": 8.0e12},
        "compute": {
            "bf16_flops": 2.5166e15,
            "fp8_flops": 5.0332e15,
            "fp4_flops": 1.00663e16,
        },
    },
    "interconnect": {
        "intra_node": {"domain_devices": 8, "link_bandwidth_bytes_per_s": 7.68e10}
    },
}

#: No probe fills these: mi308x_spec's assumed values less what the datasheet
#: states. Its inter-node link is one 400 Gb/s NIC's line rate here too.
DECLARED = copy.deepcopy(base.DECLARED)
for block, groups in DATASHEET.items():
    for group, values in groups.items():
        for key in values:
            DECLARED[block][group].pop(key, None)


def saved(path: Path) -> Fragment:
    """The tokenizer sweep a written document carries, in place of a new one."""
    document = json.loads(path.read_text())
    stanza = document["provenance"]
    subject = dict(stanza["origins"]["host.tokenizers[deepseek-v4-pro fast]"])
    stanza = {"authored_by": stanza["authored_by"], "date": stanza["date"]}
    stanza |= {"origin": subject.pop("origin"), "subject": subject}
    host = {key: document["host"][key] for key in ("cpu", "tokenizers")}
    body = {"schema_version": 1, "name": NAME, "provenance": stanza, "host": host}
    return Fragment.from_mapping(body, "tokenizer")


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
    parser.add_argument("--sweep-from", type=Path, help="a document to reuse")
    args = parser.parse_args(argv)
    sweep = saved(args.sweep_from) if args.sweep_from else swept(MODEL)
    merged = merge([sweep, *device_fragments()])
    verdict = validate(merged, tp_widths=(LAYOUT,))
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(base.report(merged, tp_width=None))
    return 0


if __name__ == "__main__":
    sys.exit(main())
