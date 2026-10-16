# SPDX-License-Identifier: MIT
"""An MI308X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi308x_spec.py [--model M] [--out PATH]

Run on hjbog-srdc-18, or on hjbog-srdc-39 with COMPASS_MACHINE=mi308x-hjbog39,
in a container that has the model's files: the tokenizer
fragment carries this host's core counts, and the merge refuses it if they are
not the ones the device fragment recorded there. The device fragment is the
readings ATOM's own `ModelRunner` took at one engine start (Qwen3.8-27B, TP1,
CUDA graphs and prefix caching on, `--max-model-len 262144`); every field no
probe fills is assumed, on either host. On hjbog-srdc-18 two Qwen3-30B-A3B
starts add the width-keyed terms at width 8 and at the DP-attention layout 1x8.
One fragment per start or origin. Writes the merged document to `--out` when
`validate` passes at every width and layout it holds, then prints
`explain`'s rows for the host, device and interconnect blocks, grouped by
origin, and the basis of `kv_blocks`.
"""

import argparse
import datetime
import json
import os
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

NAME = os.environ.get("COMPASS_MACHINE", "mi308x-hjbog18")
AUTHOR = "compass agent for jgong5"
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
NODE18 = {
    "total": 206141652992,
    "non_torch": 1249902592,
    "loaded": 54761936384,
    "parameters": 54713457120,
    "buffers": 33554432,
    "current": 55077287424,
    "peak": 58034271744,
    "warmup_tokens": 16384,
}
#: The same readings on card 5 of hjbog-srdc-39 at commit ce625ddd9.
HJBOG39 = dict(
    NODE18,
    non_torch=1277165568,
    loaded=54761861632,
    current=55145206784,
    peak=58102190592,
)
#: The software stack each host's readings were taken under, torch included.
STACK = {
    "rocm": "7.2.4",
    "aiter": "f4e7c7509",
    "rccl": "2.27.7",
    "torch": "2.10.0+rocm7.2.4.git3d3aa833",
}
DATE, COMMIT, READ, PIN = {
    "mi308x-hjbog18": ("2026-10-11", "983e42973", NODE18, STACK),
    "mi308x-hjbog39": (
        "2026-10-14",
        "ce625ddd9",
        HJBOG39,
        dict(STACK, aiter="9205e1059"),
    ),
}[NAME]
#: Each start's record: the sha256 of its server log, as
#: `python -m atom.compass.memory.check` digests it.
RECORD = {
    "mi308x-hjbog18": "f4951d84d5fd2b76c50c18c8234a1586fbeb5f0ee89897dc54f0c44f6bb1eb87",
    "mi308x-hjbog39": "e7c309168326a052460eb20e1cec1f0744415b69161e80856069339158600c52",
}[NAME]
#: What the start ran: the model, the width, the tree and the run's record.
SUBJECT = {"model": "Qwen/Qwen3.8-27B", "width": 1, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"
CONFIG_JSON = "sha256:191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab"

#: Two Qwen3-30B-A3B starts on node 18 at commit 7370fbe47 (image
#: nightly-20261013), by the key the run reads them at: the flags, `non_torch`
#: (minimum over the 8 ranks) and the log's sha256. Both read the same load
#: residue, `allocated` after load less unique parameter and buffer bytes.
STARTS = {
    "1x8": (
        "-tp 8 --enable-dp-attention",
        8189378560,
        "72a76cc8907cf77ae9c22583583b5d6ac5a6e2acb94d396e1df08b4ddd001d55",
    ),
    8: (
        "-tp 8",
        9485418496,
        "21991eab509e154ac6da9a2c3d769603da9a9ed492e52a0a9ec18d2f4e18caa2",
    ),
}
STARTS = STARTS if NAME == "mi308x-hjbog18" else {}
RESIDUE = 2165604480

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
        "software_pinned_to": PIN,
    },
}

#: No probe here fills these. The peaks are MI300X's datasheet scaled by
#: 80/304 compute units, each derate is the haircut this spec's author chose on
#: its peak, and the rest are the schema's example values.
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
            "domain_devices": 8,
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


def fragment(
    source: str, body: dict, origin: str, notes: str, subject=None, date=DATE
) -> Fragment:
    stanza = {"authored_by": AUTHOR, "date": date, "origin": origin, "notes": notes}
    if subject:
        stanza["subject"] = subject
    document = {"schema_version": 1, "name": NAME, "provenance": stanza, **body}
    return Fragment.from_mapping(document, source)


def device_fragments() -> list[Fragment]:
    """The fragments each engine start probed, and the assumed rest."""
    notes = f"one Qwen3.8-27B TP1 start at {COMMIT}"
    starts = []
    for key, (flags, reserve, record) in STARTS.items():
        constants = {
            "driver_and_collective_reserve_bytes": {key: reserve},
            "allocator_retained_after_load_bytes": {key: RESIDUE},
        }
        model, root = "Qwen/Qwen3-30B-A3B", "commit:7370fbe47"
        subject = {"model": model, "width": 8, "source_root": root}
        subject["record"] = f"sha256:{record}"
        # aiter on gfx942 has no MoE kernel for this model's 8-wide expert shard.
        how = f"one {model} {flags} --enable-expert-parallel --all2all-backend none"
        body = {"device": {"runtime_constants": constants}}
        notes_of = f"{how} start at {root}"
        starts.append(
            fragment(f"start {key}", body, "probed", notes_of, subject, "2026-10-14")
        )
    return [
        fragment("device", DEVICE, "probed", notes, SUBJECT),
        fragment("assumed", DECLARED, "assumed", "no probe fills these"),
        *starts,
    ]


def swept(model: str) -> Fragment:
    """The tokenizer the server loads, swept over this tree's own source."""
    from transformers import PretrainedConfig

    from atom.compass.artifacts.provenance import git_tree_root
    from atom.compass.run import _fingerprint
    from atom.model_engine.llm_engine import _load_tokenizer

    tok = _load_tokenizer(model)
    root = git_tree_root("atom", str(ROOT))
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
        origin="probed",
        subject={
            "model": model,
            "source_root": f"{root.revision_kind}:{root.revision}",
        },
        source="tokenizer",
    )


def report(merged, tp_width=1) -> str:
    """`explain`'s rows grouped by origin, and the basis of `kv_blocks`."""
    spec = MachineSpec.from_mapping(merged.document)
    groups: dict[str, list[str]] = {}
    for block in ("host", "device", "interconnect"):
        for row in explain(spec, block, tp_width=tp_width, origin=merged).contributions:
            groups.setdefault(row.origin["origin"], []).append(f"  {row}")
    lines = [line for origin, rows in groups.items() for line in (f"{origin}:", *rows)]
    return "\n".join(
        lines + [str(explain(spec, "kv_blocks", tp_width=tp_width, origin=merged))]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "scripts/compass/machines" / f"{NAME}.json"
    )
    args = parser.parse_args(argv)
    merged = merge([swept(args.model), *device_fragments()])
    verdict = validate(merged, tp_widths=(1, *STARTS))
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(report(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
