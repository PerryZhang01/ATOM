# SPDX-License-Identifier: MIT
"""The node-18 MI308X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi308x_spec.py [--model M] [--out PATH]

Run on hjbog-srdc-18, in a container that has the model's files: the tokenizer
fragment carries this host's core counts, and the merge refuses it if they are
not the ones the device fragment recorded there. The device fragment is the
readings ATOM's own `ModelRunner` took at one engine start (Qwen3.8-27B, TP1,
CUDA graphs and prefix caching on, `--max-model-len 262144`); every field no
probe fills is in the declared fragment. Writes the merged document to `--out`
when `validate` passes at width 1, then prints `explain`'s rows for the host,
device and interconnect blocks, split by whether a measured fragment supplied
them, and the basis of `kv_blocks`.
"""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from atom.compass.spec import (
    Fragment,
    MachineSpec,
    explain,
    merge,
    tokenizer_fragment,
    tokenizer_sweep,
    validate,
)

NAME = "mi308x-hjbog18"
AUTHOR = "compass agent for jgong5"
DATE = "2026-10-11"
#: Token counts swept, one to past the 204,288-token p90 prompt of cc-traces.
LENGTHS = tuple(4**k for k in range(10))
#: The worst relative residual a tokenizer fit may leave at any swept length.
#: Encode on node 18 costs more per token past 64k tokens than below it, so a
#: fixed-plus-linear law holds it only to about 20% at the longest length; the
#: bound leaves room for that and still refuses a sweep of the wrong shape.
BOUND = 0.25

#: What ATOM's `ModelRunner` read at one engine start on card 3 of node 18, at
#: commit 983e42973 (the stage-2 arguments, max_num_batched_tokens 16384):
#: `allocated` and the unique parameter and buffer bytes right after the model
#: loaded, then the four figures `_read_device_memory` takes after warmup.
READ = {
    "total": 206141652992,
    "non_torch": 1249902592,
    "loaded": 54761936384,
    "parameters": 54713457120,
    "buffers": 33554432,
    "current": 55077287424,
    "peak": 58034271744,
    "warmup_tokens": 16384,
}
CONFIG_JSON = "sha256:191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"

DEVICE = {
    "host": {"cpu": {"cores_physical": 112, "cores_logical": 224}},
    "device": {
        "name": "MI308X",
        "arch": "gfx942",
        "count_per_node": 8,
        "memory": {"capacity_bytes": READ["total"]},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {1: READ["non_torch"]},
            "allocator_retained_after_load_bytes": {
                1: READ["loaded"] - READ["parameters"] - READ["buffers"]
            },
            "persistent_forward_buffer_bytes": READ["current"] - READ["loaded"],
        },
        "activations": [
            {
                "id": "qwen3.8-27b",
                "applies_to": ["Qwen3_5ForConditionalGeneration"],
                "fingerprint": CONFIG_JSON,
                "bytes_per_token": {
                    1: (READ["peak"] - READ["current"]) / READ["warmup_tokens"]
                },
            }
        ],
        "software_pinned_to": {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.27.7"},
    },
}

#: No probe here fills these. The peaks are MI300X's datasheet scaled by
#: 80/304 compute units; the rest are the schema's example values.
DECLARED = {
    "host": {
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    },
    "device": {
        "memory": {"bandwidth_bytes_per_s": 5.3e12, "derate": 0.85},
        "compute": {"bf16_flops": 3.44e14, "fp8_flops": 6.88e14, "derate": 0.70},
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
            "link_bandwidth_bytes_per_s": 6.4e10,
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


def fragment(source: str, body: dict, method: str, notes: str) -> Fragment:
    stanza = {"authored_by": AUTHOR, "date": DATE, "method": method, "notes": notes}
    document = {"schema_version": 1, "name": NAME, "provenance": stanza, **body}
    return Fragment.from_mapping(document, source)


def device_fragments() -> list[Fragment]:
    """The fragment one engine start measured, and the declared one."""
    return [
        fragment("device", DEVICE, "probed", "one Qwen3.8-27B TP1 start at 983e42973"),
        fragment("declared", DECLARED, "assumed", "no probe fills these"),
    ]


def swept(model: str) -> Fragment:
    """The tokenizer the server loads, swept over this tree's own source."""
    from transformers import PretrainedConfig

    from atom.compass.run import _fingerprint
    from atom.model_engine.llm_engine import _load_tokenizer

    tok = _load_tokenizer(model)
    text = "\n".join(p.read_text() for p in sorted((ROOT / "atom").rglob("*.py")))
    entry = {
        "id": Path(model).name.lower(),
        "backend": "fast" if tok.is_fast else "slow",
        "vocab_size": len(tok),
        "fingerprint": _fingerprint(tok),
        "applies_to": PretrainedConfig.get_config_dict(model)[0]["architectures"],
        **tokenizer_sweep(tok, text, LENGTHS, bound=BOUND, clock=time.perf_counter),
        # Measured on the host it describes, so nothing is taken off it.
        "derate": 1.0,
    }
    return tokenizer_fragment(
        [entry],
        machine=NAME,
        authored_by=AUTHOR,
        date=datetime.datetime.now(datetime.UTC).date().isoformat(),
        method="probed",
        source="tokenizer",
    )


def report(merged, measured: set[str]) -> str:
    """`explain`'s rows, measured then declared, and the basis of `kv_blocks`."""
    spec = MachineSpec.from_mapping(merged.document)
    rows = [
        row
        for block in ("host", "device", "interconnect")
        for row in explain(spec, block, tp_width=1, origin=merged).contributions
    ]
    split = {True: [], False: []}
    for row in rows:
        split[bool(measured & set(row.supplied_by))].append(f"  {row}")
    return "\n".join(
        ["measured on node 18:", *split[True], "declared:", *split[False]]
        + [str(explain(spec, "kv_blocks", tp_width=1, origin=merged))]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "scripts/compass/machines" / f"{NAME}.json"
    )
    args = parser.parse_args(argv)
    merged = merge([swept(args.model), *device_fragments()])
    verdict = validate(merged, tp_widths=(1,))
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(report(merged, {"tokenizer", "device"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
