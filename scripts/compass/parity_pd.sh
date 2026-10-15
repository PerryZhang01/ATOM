#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The router's rank parity at 1P1D: one workload (atom/compass/parity/
# traffic.py) through `atomesh launch --pd-disaggregation --dp-aware --policy
# dp_sticky`, the agentic cell's router, first to a real prefill and decode
# server, each at `-tp 2 --enable-dp-attention` with its KV over Mooncake's TCP
# transport and its steps recorded, then to pd_sim.sh's simulated pair
# replaying those steps (the run file's `replay`). Each request's DP rank on
# each side is read from both runs' records and compared.
#
# Usage: parity_pd.sh OUT_DIR
#   MODEL            model directory with its weights (required)
#   REQUESTS GAP_S   as parity_one_node.sh (defaults 16 and 0.05)
#   PREFILL_GPUS DECODE_GPUS
#                    the real servers' devices (defaults 0,1 and 2,3); the
#                    simulated run sees PREFILL_GPUS and allocates nothing
#   ATOMESH          the router binary (default atomesh), on both runs
#   MACHINE          machine spec (default scripts/compass/machines/mi308x-hjbog18.json)
#   SERVER_ARGS      more API server flags, on all four servers (default
#                    --enforce-eager --max-model-len 8192 --no-enable_prefix_caching)
#   WAIT_S           seconds a server has to start, and to stop (default 1800)
# OUT_DIR gets real/ and sim/, each with its logs and record/{prefill,decode},
# and ranks.json; the last line counts the requests whose ranks differ.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
OUT=${1:?usage: parity_pd.sh OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 8192 --no-enable_prefix_caching}
ATOMESH=${ATOMESH:-atomesh}
WAIT_S=${WAIT_S:-1800}
mkdir -p "$OUT/real/record" "$OUT/sim"
OUT=$(realpath "$OUT")
free_port() { python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'; }
until_up() { # what url pid
    local deadline=$((SECONDS + WAIT_S))
    until curl -sf --max-time 10 "$2" >/dev/null; do
        kill -0 "$3" 2>/dev/null || { echo "parity_pd: $1 exited" >&2; exit 1; }
        ((SECONDS < deadline)) || { echo "parity_pd: no $1 after $WAIT_S s" >&2; exit 1; }
        sleep 1
    done
}
pids=()
trap 'kill -KILL -- "${pids[@]/#/-}" 2>/dev/null || true' EXIT
ROUTER_PORT=$(free_port)
# The router's flags as pd_sim.sh passes them under BENCHMARK_KIND=aiperf_agentic.
router() { # prefill_url decode_url
    setsid "$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" \
        --pd-disaggregation --prefill "$1" --decode "$2" --policy dp_sticky --dp-aware \
        --backend atom --disable-health-check --disable-circuit-breaker --disable-retries \
        --prometheus-port "$(free_port)" --log-level info >"$OUT/real/router.log" 2>&1 &
    pids+=($!)
}

# serve SIDE GPUS KV_ROLE: a real server recording to OUT/real/record/SIDE.
serve() {
    local port
    port=$(free_port)
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    HIP_VISIBLE_DEVICES=$2 ATOM_COMPASS_PARITY_RECORD=$OUT/real/record/$1 setsid \
        python3 -m atom.entrypoints.openai_server --model "$MODEL" --host 127.0.0.1 \
        --server-port "$port" -tp 2 --enable-dp-attention \
        --data-parallel-master-port "$(free_port)" $SERVER_ARGS --kv-transfer-config \
        "{\"kv_role\":\"$3\",\"kv_connector\":\"mooncake\",\"protocol\":\"tcp\",\"proxy_ip\":\"127.0.0.1\",\"handshake_port\":$(free_port)}" \
        >"$OUT/real/$1.log" 2>&1 &
    pids+=($!)
    until_up "$1 server" "http://127.0.0.1:$port/health" $!
    url=http://127.0.0.1:$port
}
serve prefill "${PREFILL_GPUS:-0,1}" kv_producer
prefill_url=$url
serve decode "${DECODE_GPUS:-2,3}" kv_consumer
router "$prefill_url" "$url"
until_up router "http://127.0.0.1:$ROUTER_PORT/v1/models" "${pids[-1]}"
python3 -m atom.compass.parity.traffic "$ROUTER_PORT" "$MODEL" "$OUT/real/record" \
    "${REQUESTS:-16}" "${GAP_S:-0.05}" >"$OUT/real/traffic.log" 2>&1
# The records' last lines are written as the workers exit.
for pid in "${pids[@]}"; do kill -TERM -- "-$pid" 2>/dev/null || true; done
for pid in "${pids[@]}"; do
    deadline=$((SECONDS + WAIT_S))
    while kill -0 "$pid" 2>/dev/null; do
        ((SECONDS < deadline)) || { echo "parity_pd: a real server is still up" >&2; exit 1; }
        sleep 1
    done
done
pids=()
# Each side's record is a whole record, as the replay reads it.
for side in prefill decode; do cp "$OUT/real/record/arrivals.json" "$OUT/real/record/$side/"; done

RUN=$OUT/sim/run.json
CLOCK_PORT=$(free_port)
python3 - "$RUN" "$OUT" "${MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}" \
    "$MODEL" "$CLOCK_PORT" <<'EOF'
import json, sys
from pathlib import Path

run, out, machine, model, port = sys.argv[1:]
machine = json.loads(Path(machine).read_text())
# ponytail: prices the model's tokenizer as the spec's first entry measured it,
# as parity_one_node.sh does; only arrival latency reads it.
arch = json.loads((Path(model) / "config.json").read_text())["architectures"][0]
machine["host"]["tokenizers"][0]["applies_to"].append(arch)
# ponytail: a DP rank's device constants read as TP1's, as test_dp_slice.py's
# machine reads them, with TP1's origin; measure the 1x2 layout to drop this.
origins = machine["provenance"]["origins"]
for name in ("driver_and_collective_reserve_bytes", "allocator_retained_after_load_bytes"):
    width = machine["device"]["runtime_constants"][name]
    width.setdefault("1x2", width["1"])
    field = f"device.runtime_constants.{name}"
    origins.setdefault(f"{field}[1x2]", origins[f"{field}[1]"])
Path(run).write_text(json.dumps({
    "clock_endpoint": f"tcp://127.0.0.1:{port}", "bound_s": 1e7,
    "admission_path": "serving", "ipc_s": 2.0**-14, "stream_s": 2.0**-12,
    "router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "kv_link": "intra_node",
    "replay": f"{out}/real/record", "machine": machine,
    "data_parallel_size": 2, "out_dir": f"{out}/sim",
}))
EOF
ports=("CLOCK_PORT=$CLOCK_PORT")
for name in PREFILL_PORT DECODE_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT \
    PREFILL_DP_PORT DECODE_DP_PORT; do
    ports+=("$name=$(free_port)")
done
# shellcheck disable=SC2086 # SERVER_ARGS is a word list
env "${ports[@]}" ROUTER_PORT="$ROUTER_PORT" MODEL="$MODEL" ATOMESH="$ATOMESH" \
    BENCHMARK_KIND=aiperf_agentic HIP_VISIBLE_DEVICES="${PREFILL_GPUS:-0,1}" \
    SERVER_ARGS="-tp 2 --enable-dp-attention $SERVER_ARGS" WAIT_S="$WAIT_S" \
    ATOM_COMPASS_PARITY_RECORD="$OUT/sim/record" setsid \
    bash "$ROOT/scripts/compass/pd_sim.sh" "$RUN" >"$OUT/sim/pd_sim.out" 2>&1 &
pids+=($!)
until grep -qs "pd_sim: ready" "$OUT/sim/pd_sim.out"; do
    kill -0 "${pids[-1]}" 2>/dev/null || { echo "parity_pd: pd_sim.sh exited" >&2; exit 1; }
    sleep 1
done
ATOM_COMPASS_RUN=$RUN python3 -m atom.compass.parity.traffic "$ROUTER_PORT" "$MODEL" \
    "$OUT/real/record" >"$OUT/sim/traffic.log" 2>&1
wait "${pids[-1]}"
pids=()

python3 - "$OUT" <<'EOF'
import json, sys
from pathlib import Path

from atom.compass.parity import _request_ranks, read


def ranks_of(record: Path) -> dict:
    """Each request's [prefill rank, decode rank], by its prefill key. Decode
    digests no prompt, so a decode request is the prefill request whose prompt
    is one token shorter than its first context; the workload's prompts all
    differ in length."""
    prefill = read(record / "prefill")
    prompt = {}  # key -> prompt length, its longest prefill window's context
    for steps in prefill.values():
        for s in steps:
            for key, num, context in s["batch"]:
                if num > 1:
                    prompt[key] = max(prompt.get(key, 0), context)
    by_length = {n: key for key, n in prompt.items()}
    if len(by_length) != len(prompt):
        raise ValueError("two prompts have one length, so decode cannot name them")
    ranks = {key: [rank, None] for key, rank in _request_ranks("prefill", prefill).items()}
    first = {}  # (decode rank, request id) -> its first context
    for rank, steps in read(record / "decode").items():
        for s in steps:
            for rid, _, context in s["batch"]:
                first.setdefault((rank, rid), context)
    for (rank, rid), context in first.items():
        ranks[by_length[context - 1]][1] = rank
    return ranks


out = Path(sys.argv[1])
ranks = {run: ranks_of(out / run / "record") for run in ("real", "sim")}
differ = sorted(k for k in ranks["real"] if ranks["real"][k] != ranks["sim"].get(k))
(out / "ranks.json").write_text(json.dumps(ranks | {"differ": differ}, indent=1))
print(f"parity_pd: {len(ranks['real'])} requests, (prefill, decode) ranks differ on {len(differ)}")
EOF
