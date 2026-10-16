# SPDX-License-Identifier: MIT
"""A client run's export, written from an aiperf run's raw records.

    python -m atom.compass.parity.client_run ARTIFACTS

ARTIFACTS is the artifact directory of an aiperf run at ``--export-level raw``;
its ``client_run.json`` gets the export `served` reads. It names the requests
of aiperf's profiling phase only, and states under ``warmup`` how many its
warmup phase sent, so a reader can see the warmup ran: a step holding only
warmup requests belongs to no client run, and one holding a warmup and a
profiling request belongs to the profiling request's. Each request is named by
the ``id`` of the first data chunk the server sent it, the id the step record
names it by; its trace is its ``conversation_id`` up to any ``::``, which
aiperf's trace loader appends to name the conversations it derives from one
trace; and its request parameters are its payload less the prompt, the model,
the output length and the stream's framing. A profiling request that got no
response id, failing before its first chunk, names nothing and is counted.
"""

import argparse
import json
from pathlib import Path

RAW = "profile_export_raw.jsonl"
NAME = "client_run.json"
#: aiperf's phases, as a raw record's `benchmark_phase` states them.
WARMUP, PROFILING = "warmup", "profiling"
#: Payload fields that are not request parameters.
NOT_SAMPLING = frozenset(
    {
        "messages",
        "prompt",
        "tools",
        "model",
        "max_tokens",
        "max_completion_tokens",
        "stream",
        "stream_options",
    }
)


def response_id(responses) -> str | None:
    """The `id` of the first data chunk: an SSE message's `data`, or a body."""
    for r in responses or ():
        texts = [p.get("value") for p in r.get("packets", ()) if p["name"] == "data"]
        texts += [r["text"]] if "text" in r else []
        if texts:
            try:
                return json.loads(texts[0]).get("id")
            except (TypeError, ValueError, AttributeError):
                return None
    return None


def write(artifacts) -> tuple[Path, int, int]:
    """Write `artifacts`' export; its path, how many warmup requests it counts,
    and how many profiling requests got no id."""
    artifacts, requests, warmup, unnamed = Path(artifacts), {}, 0, 0
    for line in (artifacts / RAW).read_text().splitlines():
        record = json.loads(line)
        meta = record["metadata"]
        if meta["benchmark_phase"] == WARMUP:
            warmup += 1
            continue
        if meta["benchmark_phase"] != PROFILING:
            raise ValueError(
                f"request {meta.get('x_request_id')} is in phase "
                f"{meta['benchmark_phase']!r}, neither {WARMUP} nor {PROFILING}"
            )
        rid = response_id(record.get("responses"))
        if rid is None:
            unnamed += 1
            continue
        if rid in requests:
            raise ValueError(f"two requests got response id {rid}")
        if not meta.get("conversation_id"):
            raise ValueError(
                f"request {meta.get('x_request_id')} has no conversation_id, "
                "so no trace"
            )
        requests[rid] = {
            "trace": meta["conversation_id"].split("::", 1)[0],
            "sampling": {
                k: v for k, v in record["payload"].items() if k not in NOT_SAMPLING
            },
        }
    path = artifacts / NAME
    path.write_text(
        json.dumps({"requests": requests, "warmup": warmup}, sort_keys=True)
    )
    return path, warmup, unnamed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.parity.client_run",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("artifacts", help="an aiperf run's artifact directory")
    path, warmup, unnamed = write(parser.parse_args().artifacts)
    print(
        f"client_run: wrote {path}; {warmup} warmup requests; {unnamed} profiling "
        "requests got no response id"
    )
