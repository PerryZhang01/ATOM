# SPDX-License-Identifier: MIT
"""An MI308X machine spec, from a tokenizer swept here and one real start.

    python3 scripts/compass/mi308x_spec.py [--model M] [--sweep-from DOC] [--out PATH]

Run on hjbog-srdc-18, or on hjbog-srdc-39 with COMPASS_MACHINE=mi308x-hjbog39,
in a container that has the model's files: the tokenizer
fragment carries this host's core counts, and the merge refuses it if they are
not the ones the device fragment recorded there. The device fragment is what
ATOM's own `ModelRunner` read at one engine start (Qwen3.8-27B, TP1, CUDA
graphs and prefix caching on, `--max-model-len 262144`): the device, its
capacity and the stack; every field no probe fills is assumed, on either host.
On hjbog-srdc-39 one real burst adds the HTTP intake terms (`INTAKE`). An
engine start's memory readings are `compass measure memory`'s, not the spec's.
One fragment per probe or origin. Writes the merged document to `--out` when
`validate` passes, then prints `explain`'s rows for the host, device and
interconnect blocks, grouped by origin.
`--sweep-from` takes the tokenizer sweep from a document this script wrote.
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

#: The total ATOM's `ModelRunner` read at one engine start on card 7 of node 18,
#: at commit 0f045543d, and on card 5 of hjbog-srdc-39 at commit ce625ddd9.
TOTAL = 206141652992
#: The software stack both hosts' readings were taken under, torch included:
#: compass-next:nightly-20261013's.
STACK = {
    "rocm": "7.2.4",
    "aiter": "9205e1059",
    "rccl": "2.27.7",
    "torch": "2.10.0+rocm7.2.4.git3d3aa833",
}
DATE, COMMIT = {
    "mi308x-hjbog18": ("2026-10-17", "0f045543d"),
    "mi308x-hjbog39": ("2026-10-14", "ce625ddd9"),
}[NAME]
#: Every card's `ptl_enable` on each host, as sysfs read on 2026-10-18.
PTL = {"mi308x-hjbog18": "enabled", "mi308x-hjbog39": "disabled"}[NAME]
#: Each start's record: the sha256 of its server log, as
#: `python -m atom.compass.memory.check` digests it.
RECORD = {
    "mi308x-hjbog18": "213874c331d63384299d21d68bdffd1cda1aecf72ba4a9eab80beb63a7243bd7",
    "mi308x-hjbog39": "e7c309168326a052460eb20e1cec1f0744415b69161e80856069339158600c52",
}[NAME]
#: What the start ran: the model, the width, the tree and the run's record.
SUBJECT = {"model": "Qwen/Qwen3.8-27B", "width": 1, "source_root": f"commit:{COMMIT}"}
SUBJECT["record"] = f"sha256:{RECORD}"

#: The API server's event loop taking in one HTTP request, probed on
#: hjbog-srdc-39 at commit 2d1d223be in one real c=256 first-turn cc-traces
#: burst on Qwen3.8-27B TP1: the least-squares fit of the i-th request's chat
#: template start on the loop to i and the body bytes of the requests before it.
#: The record is the probe's stamp file, the run's server log beside it.
INTAKE = {
    "mi308x-hjbog39": (
        "2026-10-16",
        "2d1d223be",
        {"fixed_s": 0.0021464994614327293, "per_byte_s": 1.869416019632594e-08},
        "40da90fe439e06efa8aeb8d8b3d5a17a955c507afb6f9aa18401cbbc957739de",
    ),
}.get(NAME)

#: A pipeline stage's send, per carrier, fitted on hjbog-srdc-39 GPUs 4 and 5
#: by `scripts/compass/pp_carriers.py` at commit 93e01fbb4, pooled over six of
#: its probe records. The record is that fit, its records' digests and source.
PP = {
    "mi308x-hjbog39": (
        "2026-10-18",
        "93e01fbb4",
        {
            "rccl": {
                "eager_threshold_bytes": 33554432,
                "latency_s": 1.5106052160263062e-06,
                "overhead_s": 3.796907745993579e-05,
                "per_byte_s": 2.0165586891245436e-11,
                "rendezvous_fixed_s": 5.63815019316789e-05,
            },
            "gloo": {
                "eager_threshold_bytes": 0,
                "latency_s": 1.0642048437148333e-05,
                "overhead_s": 1.3869054915274048e-05,
                "per_byte_s": 1.207313759980259e-10,
                "rendezvous_fixed_s": 2.428104060241613e-05,
                "wake_s": 0.00018016894767673692,
                "wake_rise_s": 0.5825204944577904,
            },
        },
        "53ef27fd3bca30cd1a58c637debab9ee5b14712acb572bfd0eabeb6807a9eaa0",
    ),
}.get(NAME)

DEVICE = {
    "host": {"cpu": {"cores_physical": 112, "cores_logical": 224}},
    "device": {
        "name": "MI308X",
        "arch": "gfx942",
        "count_per_node": 8,
        "memory": {"capacity_bytes": TOTAL},
        "software_pinned_to": STACK,
    },
}

#: No probe here fills these. The peaks are MI300X's datasheet scaled by
#: 80/304 compute units, each derate is the haircut this spec's author chose on
#: its peak, the bandwidth curve ends at the derated bandwidth, and the rest are
#: the schema's example values.
DECLARED = {
    "host": {
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
    },
    "device": {
        "memory": {
            "bandwidth_bytes_per_s": 5.3e12,
            "bandwidth_curve": [[4096, 1.1e11], [4294967296, 4.505e12]],
            "derate": 0.85,
        },
        "compute": {
            "bf16_flops": 3.44e14,
            "fp8_flops": 6.88e14,
            "derate": {"bf16": 0.70, "fp8": 0.70},
        },
        "kernel_floor_s": 2.0e-6,
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
    """The fragments the engine start and any other probe filled, and the
    assumed rest."""
    notes = f"one Qwen3.8-27B TP1 start at {COMMIT}"
    ptl = {"device": {"ptl": PTL}}
    probes = [fragment("ptl", ptl, "probed", "ptl_enable", date="2026-10-18")]
    if INTAKE:
        date, root, terms, record = INTAKE
        subject = {"model": "Qwen/Qwen3.8-27B", "source_root": f"commit:{root}"}
        subject["record"] = f"sha256:{record}"
        how = f"one real c=256 first-turn cc-traces burst at {root}"
        body = {"host": {"http_intake": terms}}
        probes.append(fragment("intake", body, "probed", how, subject, date))
    if PP:
        date, root, carriers, record = PP
        subject = {"source_root": f"commit:{root}", "record": f"sha256:{record}"}
        body = {"interconnect": {"intra_node": {"pp": carriers}}}
        how = f"pp_carriers.py on two GPUs at {root}"
        probes.append(fragment("pp", body, "probed", how, subject, date))
    return [
        fragment("device", DEVICE, "probed", notes, SUBJECT),
        fragment("assumed", DECLARED, "assumed", "no probe fills these"),
        *probes,
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


def saved(path: Path) -> Fragment:
    """The tokenizer fragment a written document carries, in place of a new
    sweep: its core counts and tokenizers, each with the origin it records."""
    document = json.loads(path.read_text())
    stanza = document["provenance"]
    origins = {
        label: found
        for label, found in stanza["origins"].items()
        if label.startswith(("host.cpu.", "host.tokenizers["))
    }
    stanza = {"authored_by": stanza["authored_by"], "date": stanza["date"]}
    stanza |= {"origin": "mixed", "origins": origins}
    host = {key: document["host"][key] for key in ("cpu", "tokenizers")}
    body = {"schema_version": 1, "name": NAME, "provenance": stanza, "host": host}
    return Fragment.from_mapping(body, "tokenizer")


def report(merged) -> str:
    """`explain`'s rows grouped by origin."""
    spec = MachineSpec.from_mapping(merged.document)
    groups: dict[str, list[str]] = {}
    for block in ("host", "device", "interconnect"):
        for row in explain(spec, block, origin=merged).contributions:
            groups.setdefault(row.origin["origin"], []).append(f"  {row}")
    lines = [line for origin, rows in groups.items() for line in (f"{origin}:", *rows)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "scripts/compass/machines" / f"{NAME}.json"
    )
    parser.add_argument("--sweep-from", type=Path, help="a document to reuse")
    args = parser.parse_args(argv)
    sweep = saved(args.sweep_from) if args.sweep_from else swept(args.model)
    merged = merge([sweep, *device_fragments()])
    verdict = validate(merged.document)
    print(verdict)
    if not verdict.ok:
        return 1
    args.out.write_text(json.dumps(merged.document, indent=2) + "\n")
    print(report(merged))
    return 0


if __name__ == "__main__":
    sys.exit(main())
