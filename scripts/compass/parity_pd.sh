#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The router's parity at xPyD: one workload (atom/compass/parity/traffic.py)
# through `atomesh launch --pd-disaggregation --dp-aware --policy dp_sticky`,
# the agentic cell's router, first to real prefill and decode servers, each at
# `-tp <its DP width> --enable-dp-attention` with its KV over Mooncake's TCP
# transport and its steps recorded, then to pd_sim.sh's simulated servers
# replaying those steps (the run file's `replay`), its router listing each
# side's workers in the order the real one registered them, read from the
# real router's debug log (`worker_order`). Each request's server and
# DP rank on each side is read from both runs' records and compared.
#
# Usage: parity_pd.sh OUT_DIR
#   MODEL            model directory with its weights (required)
#   REQUESTS GAP_S   requests, and wall seconds between sends (defaults 16 and
#                    0.01); no request waits for the one before it to be
#                    admitted, so prefills overlap and the router chooses a
#                    prefill rank
#   PREFILL_DP DECODE_DP
#                    each side's DP width (default 2); width 1 serves at -tp 1
#   PREFILL_GPUS DECODE_GPUS
#                    the real servers' devices, a comma list per server and a
#                    space between servers (defaults 0,1 and 2,3: 1P1D); the
#                    simulated run sees the first prefill server's and
#                    allocates nothing
#   ATOMESH          the router binary (default atomesh), on both runs
#   MACHINE          machine spec (default scripts/compass/machines/mi308x-hjbog18.json)
#   STORE            artifact store holding the simulated configuration's memory
#                    readings, taken by `compass measure memory` (required)
#   SERVER_ARGS      more API server flags, on every server (default
#                    --enforce-eager --max-model-len 8192 --no-enable_prefix_caching)
#   WAIT_S           seconds a server has to start, and to stop (default 1800)
# OUT_DIR gets real/ and sim/, each with its logs and a record directory per
# server, prefill/ and decode/, or prefill<k>/ and decode<k>/ above one server
# of a side, and ranks.json; the last line counts the requests whose server or
# rank differs.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
OUT=${1:?usage: parity_pd.sh OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
STORE=${STORE:?STORE names the store holding the memory readings}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 8192 --no-enable_prefix_caching}
ATOMESH=${ATOMESH:-atomesh}
WAIT_S=${WAIT_S:-1800}
PREFILL_DP=${PREFILL_DP:-2} DECODE_DP=${DECODE_DP:-2}
# A side's DP width as its server flags.
dp_args() { if (($1 > 1)); then echo "-tp $1 --enable-dp-attention"; else echo "-tp 1"; fi; }
mkdir -p "$OUT/real/record" "$OUT/sim"
OUT=$(realpath "$OUT")
pids=()
trap 'kill -KILL -- "${pids[@]/#/-}" 2>/dev/null || true' EXIT
ROUTER_PORT=$(compass_free_port)
read -ra prefill_gpus <<<"${PREFILL_GPUS:-0,1}"
read -ra decode_gpus <<<"${DECODE_GPUS:-2,3}"

# serve SIDE K N GPUS KV_ROLE DP: server K of N of a side, recording to
# OUT/real/record/SIDE, or SIDE<K> above one, as pd_sim.sh names its servers.
urls=()
serve() {
    local port name=$1
    (($3 == 1)) || name+=$2
    port=$(compass_free_port)
    # shellcheck disable=SC2086 # the server flags are word lists
    HIP_VISIBLE_DEVICES=$4 ATOM_COMPASS_PARITY_RECORD=$OUT/real/record/$name setsid \
        python3 -m atom.entrypoints.openai_server --model "$MODEL" --host 127.0.0.1 \
        --server-port "$port" $(dp_args "$6") \
        --data-parallel-master-port "$(compass_free_port)" $SERVER_ARGS --kv-transfer-config \
        "{\"kv_role\":\"$5\",\"kv_connector\":\"mooncake\",\"protocol\":\"tcp\",\"proxy_ip\":\"127.0.0.1\",\"handshake_port\":$(compass_free_port)}" \
        >"$OUT/real/$name.log" 2>&1 &
    pids+=($!)
    compass_wait_up "$name server" $! "http://127.0.0.1:$port/health"
    urls+=("--$1" "http://127.0.0.1:$port")
}
for k in "${!prefill_gpus[@]}"; do
    serve prefill "$k" "${#prefill_gpus[@]}" "${prefill_gpus[k]}" kv_producer "$PREFILL_DP"
done
for k in "${!decode_gpus[@]}"; do
    serve decode "$k" "${#decode_gpus[@]}" "${decode_gpus[k]}" kv_consumer "$DECODE_DP"
done
# The router's flags as pd_sim.sh passes them under BENCHMARK_KIND=aiperf_agentic.
setsid "$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" \
    --pd-disaggregation "${urls[@]}" --policy dp_sticky --dp-aware \
    --backend atom --disable-health-check --disable-circuit-breaker --disable-retries \
    --prometheus-port "$(compass_free_port)" --log-level debug >"$OUT/real/router.log" 2>&1 &
pids+=($!)
compass_wait_up router "${pids[-1]}" "http://127.0.0.1:$ROUTER_PORT/v1/models"
python3 -c 'import sys; from pathlib import Path; from atom.compass.parity.traffic import bodies, real
port, model, record, n, gap_s = sys.argv[1:]
real(int(port), bodies(model, int(n)), float(gap_s), Path(record), wait=False)' \
    "$ROUTER_PORT" "$MODEL" "$OUT/real/record" "${REQUESTS:-16}" "${GAP_S:-0.01}" \
    >"$OUT/real/traffic.log" 2>&1
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
# Each server's record is a whole record, as the replay reads it.
for record in "$OUT"/real/record/*/; do cp "$OUT/real/record/arrivals.json" "$record"; done

RUN=$OUT/sim/run.json
CLOCK_PORT=$(compass_free_port)
python3 - "$RUN" "$OUT" "${MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}" \
    "$MODEL" "$STORE" "$CLOCK_PORT" "${#prefill_gpus[@]}" "${#decode_gpus[@]}" \
    "$PREFILL_DP" "$DECODE_DP" "${urls[@]}" <<'EOF'
import json, re, sys
from pathlib import Path

run, out, machine, model, store, port, prefills, decodes, dp_p, dp_d, *urls = sys.argv[1:]
machine = json.loads(Path(machine).read_text())
# ponytail: prices the model's tokenizer as the spec's first entry measured it,
# as parity_one_node.sh does; only arrival latency reads it.
arch = json.loads((Path(model) / "config.json").read_text())["architectures"][0]
machine["host"]["tokenizers"][0]["applies_to"].append(arch)
# Each side's (instance, DP rank) pairs in the order the real router logged
# registering them.
deployment, count = {}, {"--prefill": 0, "--decode": 0}
for flag, url in zip(urls[::2], urls[1::2]):
    deployment[url] = flag[2].upper(), count[flag]
    count[flag] += 1
order = {}
log = Path(f"{out}/real/router.log").read_text()
for url, rank in re.findall(r"Registered worker (\S+)@(\d+) ", log):
    side, k = deployment[url]
    order.setdefault(side, []).append([k, int(rank)])
Path(run).write_text(json.dumps({
    "clock_endpoint": f"tcp://127.0.0.1:{port}", "bound_s": 1e7,
    "admission_path": "serving", "ipc_s": 2.0**-14, "stream_s": 2.0**-12,
    "router_s": 2.0**-12, "kv_write_req_s": 2.0**-10, "kv_link": "intra_node",
    "replay": f"{out}/real/record", "machine": machine, "store": store,
    "prefill_data_parallel_size": int(dp_p), "decode_data_parallel_size": int(dp_d),
    "out_dir": f"{out}/sim",
    "prefill_instances": int(prefills), "decode_instances": int(decodes),
    "worker_order": order,
}))
EOF
ports=("CLOCK_PORT=$CLOCK_PORT")
for name in PREFILL_PORT DECODE_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT \
    PREFILL_DP_PORT DECODE_DP_PORT; do
    ports+=("$name=$(compass_free_port)")
done
# shellcheck disable=SC2086 # SERVER_ARGS is a word list
env "${ports[@]}" ROUTER_PORT="$ROUTER_PORT" MODEL="$MODEL" ATOMESH="$ATOMESH" \
    BENCHMARK_KIND=aiperf_agentic HIP_VISIBLE_DEVICES="${prefill_gpus[0]}" \
    SERVER_ARGS="$SERVER_ARGS" PREFILL_ARGS="$(dp_args "$PREFILL_DP")" \
    DECODE_ARGS="$(dp_args "$DECODE_DP")" WAIT_S="$WAIT_S" \
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
from collections import Counter
from pathlib import Path

from atom.compass.parity import _request_ranks, read


def ranks_of(record: Path) -> dict:
    """Each request's [[prefill server, rank], [decode server, rank]], by its
    prefill key, a server named by its record directory. Decode digests no
    prompt, so a decode request is the prefill request whose prompt is one
    token shorter than its first context; the workload's prompts all differ in
    length."""
    prompt, ranks = {}, {}  # key -> prompt length, its longest prefill window's context
    for server in sorted(record.glob("prefill*")):
        prefill = read(server)
        for steps in prefill.values():
            for s in steps:
                for key, num, context in s["batch"]:
                    if num > 1:
                        prompt[key] = max(prompt.get(key, 0), context)
        for key, rank in _request_ranks(server.name, prefill).items():
            ranks[key] = [[server.name, rank], None]
    by_length = {n: key for key, n in prompt.items()}
    if len(by_length) != len(prompt):
        raise ValueError("two prompts have one length, so decode cannot name them")
    first = {}  # (decode server, rank, request id) -> its first context
    for server in sorted(record.glob("decode*")):
        for rank, steps in read(server).items():
            for s in steps:
                for rid, _, context in s["batch"]:
                    first.setdefault((server.name, rank, rid), context)
    for (server, rank, rid), context in first.items():
        ranks[by_length[context - 1]][1] = [server, rank]
    if missing := [key for key, (_, d) in ranks.items() if d is None]:
        raise ValueError(f"{record}: {len(missing)} requests have no decode rank")
    return ranks


out = Path(sys.argv[1])
ranks = {run: ranks_of(out / run / "record") for run in ("real", "sim")}
differ = sorted(k for k in ranks["real"] if ranks["real"][k] != ranks["sim"].get(k))
(out / "ranks.json").write_text(json.dumps(ranks | {"differ": differ}, indent=1))
for side in (0, 1):
    placed = Counter(f"{s}:{r}" for s, r in (v[side] for v in ranks["real"].values()))
    print(f"parity_pd: the real run's requests by server:rank, {dict(sorted(placed.items()))}")
print(f"parity_pd: {len(ranks['real'])} requests, server or rank differs on {len(differ)}")
EOF
