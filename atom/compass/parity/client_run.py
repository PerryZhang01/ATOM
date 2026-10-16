# SPDX-License-Identifier: MIT
"""A client run's export, written from an aiperf run's raw records.

    python -m atom.compass.parity.client_run ARTIFACTS

ARTIFACTS is the artifact directory of an aiperf run at ``--export-level raw``;
its ``client_run.json`` gets the export `served` reads. Each request is named
by the ``id`` of the first data chunk the server sent it, the id the step
record names it by; its trace is its ``conversation_id`` up to any ``::``,
which aiperf's trace loader appends to name the conversations it derives from
one trace; and its request parameters are its payload less the prompt, the
model, the output length and the stream's framing. A request that got no
response id, failing before its first chunk, names nothing and is counted.
"""

import argparse
import json
from pathlib import Path

RAW = "profile_export_raw.jsonl"
NAME = "client_run.json"
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


def write(artifacts) -> tuple[Path, int]:
    """Write `artifacts`' export; its path, and how many requests got no id."""
    artifacts, requests, unnamed = Path(artifacts), {}, 0
    for line in (artifacts / RAW).read_text().splitlines():
        record = json.loads(line)
        rid, meta = response_id(record.get("responses")), record["metadata"]
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
    path.write_text(json.dumps({"requests": requests}, sort_keys=True))
    return path, unnamed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.parity.client_run",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("artifacts", help="an aiperf run's artifact directory")
    path, unnamed = write(parser.parse_args().artifacts)
    print(f"client_run: wrote {path}; {unnamed} requests got no response id")
