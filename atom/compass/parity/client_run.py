# SPDX-License-Identifier: MIT
"""A client run's export, written from an aiperf run's raw records.

    python -m atom.compass.parity.client_run ARTIFACTS

ARTIFACTS is the artifact directory of an aiperf run at ``--export-level raw``;
its ``client_run.json`` gets the export `served` reads. ``requests`` names the
requests of aiperf's profiling phase and ``warmup`` those of its warmup phase,
so a reader can see the warmup ran and which steps it took. `served` reads
``requests`` alone: a step holding only warmup requests belongs to no client
run, and one holding a warmup and a profiling request belongs to the profiling
request's. Each request is named by the ``id`` of the first data chunk the
server sent it, the id the step record names it by; its trace is its
``conversation_id`` up to any ``::``, which aiperf's trace loader appends to
name the conversations it derives from one trace; and its request parameters
are its payload less the prompt, the model, the output length and the stream's
framing. A request that got no response id, failing before its first chunk,
names nothing and is counted. ``traces`` names every trace aiperf was given:
the ``id`` of each weka trace its input config's dataset holds, a file or each
``*.json`` of a directory, as aiperf's weka trace loader reads them. Which of
them a run's finished requests came from moves with its schedule, since aiperf
records nothing of a request in flight at its cutoff.
"""

import argparse
import json
from pathlib import Path

RAW = "profile_export_raw.jsonl"
#: aiperf's export, whose `input_config` names the dataset aiperf was given.
AIPERF = "profile_export_aiperf.json"
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


def given(artifacts) -> list:
    """The traces of the one dataset aiperf was given, as its export names it."""
    config = json.loads((Path(artifacts) / AIPERF).read_text())["input_config"]
    (d,) = config["datasets"]
    if d.get("format") != "weka_trace" or not d.get("path"):
        raise ValueError(f"aiperf's dataset {d['name']} is not weka traces at a path")
    path = Path(d["path"])
    files = path.glob("*.json") if path.is_dir() else [path]
    return sorted({json.loads(f.read_text())["id"] for f in files})


def write(artifacts) -> tuple[Path, dict, int]:
    """Write `artifacts`' export; its path, the export, and how many requests
    got no response id."""
    artifacts, named, unnamed = Path(artifacts), {PROFILING: {}, WARMUP: {}}, 0
    for line in (artifacts / RAW).read_text().splitlines():
        record = json.loads(line)
        meta = record["metadata"]
        if meta["benchmark_phase"] not in named:
            raise ValueError(
                f"request {meta.get('x_request_id')} is in phase "
                f"{meta['benchmark_phase']!r}, neither {WARMUP} nor {PROFILING}"
            )
        rid = response_id(record.get("responses"))
        if rid is None:
            unnamed += 1
            continue
        if rid in named[PROFILING] or rid in named[WARMUP]:
            raise ValueError(f"two requests got response id {rid}")
        if not meta.get("conversation_id"):
            raise ValueError(
                f"request {meta.get('x_request_id')} has no conversation_id, "
                "so no trace"
            )
        named[meta["benchmark_phase"]][rid] = {
            "trace": meta["conversation_id"].split("::", 1)[0],
            "sampling": {
                k: v for k, v in record["payload"].items() if k not in NOT_SAMPLING
            },
        }
    export = {
        "requests": named[PROFILING],
        "warmup": named[WARMUP],
        "traces": given(artifacts),
    }
    path = artifacts / NAME
    path.write_text(json.dumps(export, sort_keys=True))
    return path, export, unnamed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m atom.compass.parity.client_run",
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument("artifacts", help="an aiperf run's artifact directory")
    path, export, unnamed = write(parser.parse_args().artifacts)
    print(
        f"client_run: wrote {path}: {len(export['requests'])} profiling and "
        f"{len(export['warmup'])} warmup requests; {unnamed} got no response id"
    )
