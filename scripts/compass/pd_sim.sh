#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A simulated prefill-decode run, deployed as
# .github/scripts/atomesh/pd_server_atom.sh deploys the real one: RUN_FILE's
# prefill_instances prefill and decode_instances decode API servers (1 each when
# absent), each on the simulated KV connector in Mooncake's place, and
# `atomesh launch --pd-disaggregation` between them. The clock authority runs
# on its own, started before the servers, and every LP reaches it at
# tcp://127.0.0.1:$CLOCK_PORT.
#
# Usage: pd_sim.sh RUN_FILE
#   MODEL               model directory (required)
#   CLOCK_PORT PREFILL_PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT
#   PROMETHEUS_PORT     the ports it uses, all on 127.0.0.1; prefill server k
#                       serves on PREFILL_PORT+k, decode server k on
#                       DECODE_PORT+k; with p prefill and d decode servers at DP
#                       widths m and n the write requests take p*d*m*n ports from
#                       KV_WRITE_REQ_PORT
#   PREFILL_DP_PORT DECODE_DP_PORT
#                       each side's --data-parallel-master-port, plus 16k for
#                       server k; above DP width one its ranks derive their
#                       sockets from it
#   DECODE_EXEC         a command prefix that runs the decode server elsewhere
#                       on this node, e.g. `docker exec <container>`; on a
#                       failure, stopping it there is the caller's
#   SERVER_ARGS         more API server flags (default: --enforce-eager
#                       --max-model-len 2048)
#   PREFILL_ARGS DECODE_ARGS
#                       more API server flags for that side only, after
#                       SERVER_ARGS, e.g. its DP width
#   PREFILL_ENV DECODE_ENV
#                       NAME=VALUE words set in that side's servers only, e.g.
#                       `ATOM_NUMA_BIND=1 GPU_MAX_HW_QUEUES=5`
#   ATOMESH            the router binary (default: atomesh)
#   ATOM_COMPASS_PARITY_RECORD
#                       when set, each server records its steps
#                       (atom/compass/parity) under it, in prefill/ and decode/,
#                       or prefill<k>/ and decode<k>/ above one server of a side
#   BENCHMARK_KIND ROUTER_POLICY ATOM_PD_RANK_MAPPING_POLICY
#                       as `start_router` in .github/scripts/atomesh/pd_server_atom.sh
#                       reads them (defaults random, random, none)
#   SEED                the run's seed, which the router LP's policies draw
#                       from (default 20260707, aiperf's in cctraces_sim.sh)
#
# The router is launched as `start_router` launches the real cell's. With
# --enable-dp-attention in either side's flags, under
# BENCHMARK_KIND=aiperf_agentic or a rank mapping policy other than none, that
# is --dp-aware: the router chooses among DP ranks on load that responses
# move, which makes it LP `router` of the run, and RUN_FILE gains `router_in_s`
# unless it has one already, and `router`, the binary that runs it, refused unless built from this tree's
# atom/mesh (`router_binary` in atom/compass/run). Several servers of a side
# make it LP `router` too, as it chooses among them. As LP `router` it lists
# each side's workers in RUN_FILE's `worker_order`, else sorted. Otherwise
# the router has one worker per side, and ATOM's own DP dispatch picks the rank.
#
# It runs its own tree, as the gates do. Logs go beside RUN_FILE. It prints
# `pd_sim: ready` once the router answers; start the traffic LP then. It exits
# once every server has, 0 when each exited 0.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
RUN_FILE=$(realpath "${1:?usage: pd_sim.sh RUN_FILE}")
MODEL=${MODEL:?MODEL names the model directory}
OUT=$(dirname "$RUN_FILE")
CLOCK_PORT=${CLOCK_PORT:-29400}
PREFILL_PORT=${PREFILL_PORT:-8010}
DECODE_PORT=${DECODE_PORT:-8020}
ROUTER_PORT=${ROUTER_PORT:-8000}
KV_WRITE_REQ_PORT=${KV_WRITE_REQ_PORT:-29401}
PROMETHEUS_PORT=${PROMETHEUS_PORT:-29100}
PREFILL_DP_PORT=${PREFILL_DP_PORT:-29500}
DECODE_DP_PORT=${DECODE_DP_PORT:-29700}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 2048}
ATOMESH=${ATOMESH:-atomesh}
WAIT_S=${WAIT_S:-600}
CLOCK=tcp://127.0.0.1:$CLOCK_PORT
KV_WRITE_REQ=tcp://127.0.0.1:$KV_WRITE_REQ_PORT

trap 'kill $(jobs -p) 2>/dev/null || true' EXIT

# The decode side may run in another container, so its environment travels
# on its command line.
passed=()
for name in PYTHONPATH PYTHONHASHSEED AITER_LOG_LEVEL HIP_VISIBLE_DEVICES; do
    [[ -n "${!name:-}" ]] && passed+=("$name=${!name}")
done

read -r PREFILLS DECODES < <(python3 -c 'import json, sys
run = json.load(open(sys.argv[1]))
print(run.get("prefill_instances", 1), run.get("decode_instances", 1))' "$RUN_FILE")

server() { # role port kv_role dp_port instance count [exec]
    # Above one server of a side each is named, and names itself, by its index.
    local name=$1 kv="\"kv_role\":\"$3\",\"kv_connector\":\"compass\",\"compass_kv_write_req\":\"$KV_WRITE_REQ\""
    (($6 > 1)) && name=$1$5 kv+=",\"compass_instance\":$5"
    # uvicorn's keep-alive timeout runs on the LP clock, the router drops idle
    # connections on the wall clock: which comes first, and with it whether the
    # timeout takes a grant, would differ between runs. Only the router closes.
    local extra=${DECODE_ARGS:-} vars=${DECODE_ENV:-}
    [[ $1 == prefill ]] && extra=${PREFILL_ARGS:-} vars=${PREFILL_ENV:-}
    # shellcheck disable=SC2086 # DECODE_EXEC, the side's variables and the server flags are word lists
    ${7:-} env "${passed[@]}" $vars ${ATOM_COMPASS_PARITY_RECORD:+"ATOM_COMPASS_PARITY_RECORD=$ATOM_COMPASS_PARITY_RECORD/$name"} python3 -m atom.entrypoints.openai_server \
        --model "$MODEL" --host 127.0.0.1 --server-port "$2" --timeout-keep-alive 1000000 \
        --data-parallel-master-port "$4" \
        $SERVER_ARGS $extra \
        --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
        --kv-transfer-config "{$kv}" \
        >"$OUT/$name.log" 2>&1 &
}

# The real cell's router arguments, chosen as `start_router` chooses them. With
# the health check, circuit breaker and retries off, no wall-clock failure
# detector can resend or drop a request.
policy=${ROUTER_POLICY:-random}
mapping=${ATOM_PD_RANK_MAPPING_POLICY:-none}
mapping_args=()
dp_aware=()
if [[ " $SERVER_ARGS ${PREFILL_ARGS:-} ${DECODE_ARGS:-} " == *" --enable-dp-attention "* ]]; then
    [[ $mapping == none ]] || mapping_args=(--atom-pd-rank-mapping-policy "$mapping")
    if [[ ${BENCHMARK_KIND:-random} == aiperf_agentic ]]; then
        [[ $policy == cache_aware ]] || policy=dp_sticky
        dp_aware=(--dp-aware)
    elif ((${#mapping_args[@]})); then
        dp_aware=(--dp-aware)
    fi
fi
router=(--policy "$policy")
if [[ $policy == cache_aware ]]; then
    router+=(--prefill-policy cache_aware --decode-policy cache_aware --cache-threshold 0.8
        --balance-abs-threshold "${ROUTER_BALANCE_ABS_THRESHOLD:-20}"
        --balance-rel-threshold 2.0 --eviction-interval 300)
    mapping_args=(--atom-pd-rank-mapping-policy "$mapping")
fi
router+=("${mapping_args[@]}" "${dp_aware[@]}")
# Each server's role, port, kv_role, DP port, index and count, and, above one
# server of a side, its URL's frontend LP for the router.
specs=() workers=()
for ((k = 0; k < PREFILLS; k++)); do
    specs+=("prefill $((PREFILL_PORT + k)) kv_producer $((PREFILL_DP_PORT + 16 * k)) $k $PREFILLS")
    ((PREFILLS == 1)) || workers+=("\"http://127.0.0.1:$((PREFILL_PORT + k))\":\"frontend-P$k\"")
done
for ((k = 0; k < DECODES; k++)); do
    specs+=("decode $((DECODE_PORT + k)) kv_consumer $((DECODE_DP_PORT + 16 * k)) $k $DECODES")
    ((DECODES == 1)) || workers+=("\"http://127.0.0.1:$((DECODE_PORT + k))\":\"frontend-D$k\"")
done
router_env=()
if ((${#dp_aware[@]} || PREFILLS > 1 || DECODES > 1)); then
    # The hop into the router takes the run's ipc_s, its declared floor for one
    # hop between two local processes; the rest of each path's lookahead stays
    # on the hop out.
    # Its lookaheads, then the run file's worker_order as (URL, DP rank) pairs.
    lp=$(python3 - "$RUN_FILE" "$ATOMESH" "$PREFILL_PORT" "$DECODE_PORT" <<'EOF'
import json, sys
from pathlib import Path
from atom.compass.run import ROUTER, channel_table, router_binary, worker_order

path = Path(sys.argv[1])
run = json.loads(path.read_text())
try:
    run["router"] = router_binary(sys.argv[2])
    order = worker_order(run)
except Exception as e:
    sys.exit(f"pd_sim: {e}")
run.setdefault("router_in_s", run["ipc_s"])
path.write_text(json.dumps(run))
table = channel_table(run)
print(json.dumps({c.name: table.lookahead(c.name) for c in table.channels_from(ROUTER)}))
port = {"P": int(sys.argv[3]), "D": int(sys.argv[4])}
print(json.dumps([[f"http://127.0.0.1:{port[s] + k}", r] for s in order for k, r in order[s]]))
EOF
)
    router_env=(ATOM_COMPASS_CLOCK_ENDPOINT="$CLOCK" ATOM_COMPASS_SEED="${SEED:-20260707}"
        ATOM_COMPASS_ROUTER_LOOKAHEADS="${lp%%$'\n'*}" ATOM_COMPASS_ROUTER_ORDER="${lp#*$'\n'}"
        ATOM_COMPASS_ROUTER_WORKERS="{$(IFS=,; echo "${workers[*]}")}")
fi

python3 -m atom.compass.run --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
    >"$OUT/authority.log" 2>&1 &
authority=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$CLOCK_PORT") 2>/dev/null; do
    kill -0 $authority 2>/dev/null || { echo "pd_sim: the authority exited" >&2; exit 1; }
    sleep 0.2
done

servers=() urls=()
for spec in "${specs[@]}"; do
    read -r role port kv_role dp_port k n <<<"$spec"
    exec_prefix=()
    [[ $role == decode && -n ${DECODE_EXEC:-} ]] && exec_prefix=("$DECODE_EXEC")
    server "$role" "$port" "$kv_role" "$dp_port" "$k" "$n" "${exec_prefix[@]}"
    servers+=($!)
    urls+=("--$role" "http://127.0.0.1:$port")
done
for i in "${!specs[@]}"; do
    read -r role port _ <<<"${specs[$i]}"
    compass_wait_up "$role" "${servers[$i]}" "http://127.0.0.1:$port/health"
done

env "${router_env[@]}" "$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" --pd-disaggregation \
    "${urls[@]}" \
    "${router[@]}" --backend atom --disable-health-check --disable-circuit-breaker \
    --disable-retries --prometheus-port "$PROMETHEUS_PORT" --log-level info \
    >"$OUT/router.log" 2>&1 &
compass_wait_up router $! "http://127.0.0.1:$ROUTER_PORT/v1/models"
echo "pd_sim: ready"

rc=0
for pid in "${servers[@]}"; do
    wait "$pid" || rc=$?
done
# The authority leaves at the finish, once it has written the step table and
# the run summary.
((rc != 0)) || wait $authority || rc=$?
exit $rc
