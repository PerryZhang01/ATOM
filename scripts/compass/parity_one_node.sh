#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The scheduling-parity test on one node: one workload (atom/compass/parity/
# traffic.py) at `-tp 2 --enable-dp-attention` on the real engine, recording
# each step, then on the simulated runner replaying the real steps' times
# (atom/compass/parity/replay.py), and the two records compared per DP rank.
#
# Usage: parity_one_node.sh OUT_DIR
#   MODEL            model directory with its weights (required)
#   REQUESTS         requests in the workload (default 64)
#   GAP_S            seconds between sends, each also waiting for the server to
#                    admit the one before; 0 sends them back to back (default 0.05)
#   MACHINE          machine spec (default scripts/compass/machines/mi308x-hjbog18.json)
#   SERVER_ARGS      more API server flags, on both runs (default --enforce-eager
#                    --max-model-len 8192 --no-enable_prefix_caching)
#   WAIT_S           seconds a server has to start, and to stop (default 1800)
# The real run uses HIP_VISIBLE_DEVICES as set. OUT_DIR gets real/ and sim/,
# each with its server log and step record (sim/ also has the run file, step
# table and summary), and compare.json; the last lines are the real run's
# admission, send to response headers, and the compare per DP rank.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
OUT=${1:?usage: parity_one_node.sh OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 8192 --no-enable_prefix_caching}
WAIT_S=${WAIT_S:-1800}
mkdir -p "$OUT/real" "$OUT/sim"
OUT=$(realpath "$OUT")
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }

server=
trap 'kill -KILL -- ${server:+-$server} 2>/dev/null || true' EXIT
# serve SIDE [FLAGS...]: an API server recording to OUT/SIDE/record, in its own
# process group; returns once it listens.
serve() {
    local side=$1 deadline=$((SECONDS + WAIT_S))
    shift
    PORT=$(free_port)
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    ATOM_DP_MASTER_PORT=$(free_port) ATOM_COMPASS_PARITY_RECORD=$OUT/$side/record setsid \
        python3 -m atom.entrypoints.openai.api_server --model "$MODEL" --host 127.0.0.1 \
        --server-port "$PORT" -tp 2 --enable-dp-attention $SERVER_ARGS "$@" \
        >"$OUT/$side/server.log" 2>&1 &
    server=$!
    until grep -qs "Uvicorn running" "$OUT/$side/server.log"; do
        kill -0 $server 2>/dev/null || { echo "parity_one_node: the $side server exited" >&2; exit 1; }
        ((SECONDS < deadline)) || { echo "parity_one_node: no $side server after $WAIT_S s" >&2; exit 1; }
        sleep 1
    done
}
# leave SIDE: wait up to WAIT_S for the server to exit.
leave() {
    local deadline=$((SECONDS + WAIT_S))
    while kill -0 $server 2>/dev/null; do
        ((SECONDS < deadline)) || { echo "parity_one_node: the $1 server is still up" >&2; exit 1; }
        sleep 1
    done
}

serve real
python3 -m atom.compass.parity.traffic "$PORT" "$MODEL" "$OUT/real/record" \
    "${REQUESTS:-64}" "${GAP_S:-0.05}" >"$OUT/real/traffic.log" 2>&1
# The record's last lines are written as the workers exit.
kill -TERM -- -$server
leave real

RUN=$OUT/sim/run.json
python3 - "$RUN" "$OUT" "${MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}" \
    "$MODEL" "$(free_port)" <<'EOF'
import json, sys
from pathlib import Path

run, out, machine, model, port = sys.argv[1:]
machine = json.loads(Path(machine).read_text())
# ponytail: prices the model's tokenizer as the spec's first entry measured it;
# only arrival latency reads it. Measure the model's own to drop this.
arch = json.loads((Path(model) / "config.json").read_text())["architectures"][0]
machine["host"]["tokenizers"][0]["applies_to"].append(arch)
Path(run).write_text(json.dumps({
    "clock_endpoint": f"tcp://127.0.0.1:{port}", "bound_s": 1e7,
    "admission_path": "serving", "ipc_s": 2.0**-14, "stream_s": 2.0**-12,
    "replay": f"{out}/real/record", "machine": machine,
    "data_parallel_size": 2, "out_dir": f"{out}/sim",
}))
EOF
serve sim --compass-run "$RUN"
ATOM_COMPASS_RUN=$RUN python3 -m atom.compass.parity.traffic "$PORT" "$MODEL" \
    "$OUT/real/record" >"$OUT/sim/traffic.log" 2>&1
leave sim
wait $server

python3 -m atom.compass.parity "$OUT/real/record" "$OUT/sim/record" >"$OUT/compare.json"
python3 - "$OUT/compare.json" "$OUT/real/record/arrivals.json" <<'EOF'
import json, statistics, sys

report, arrivals = (json.loads(open(path).read()) for path in sys.argv[1:])
ms = [(a - s) / 1e6 for s, a in zip(arrivals["sent_ns"], arrivals["admitted_ns"])]
print(f"parity_one_node: real admission ms min={min(ms):.2f} "
      f"median={statistics.median(ms):.2f} max={max(ms):.2f}")
for rank, n in report["steps"].items():
    first = report["first_divergence"][rank]
    print(f"parity_one_node: dp{rank} compared={n['compared']} "
          f"identical={n['identical']} first_divergence={first and first['step']}")
EOF
