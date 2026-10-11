#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# A simulated 1P1D run, deployed as .github/scripts/atomesh/pd_server_atom.sh
# deploys the real one: a prefill and a decode API server, each on the
# simulated KV connector in Mooncake's place, and `atomesh launch
# --pd-disaggregation` between them. The clock authority runs on its own,
# started before the servers, and every LP reaches it at
# tcp://127.0.0.1:$CLOCK_PORT.
#
# Usage: pd_sim.sh RUN_FILE
#   MODEL               model directory (required)
#   CLOCK_PORT PREFILL_PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT
#   PROMETHEUS_PORT     the ports it uses, all on 127.0.0.1
#   DECODE_EXEC         a command prefix that runs the decode server elsewhere
#                       on this node, e.g. `docker exec <container>`; on a
#                       failure, stopping it there is the caller's
#   SERVER_ARGS         more API server flags (default: --enforce-eager
#                       --max-model-len 2048)
#   ATOMESH             the router binary (default: atomesh)
#
# It runs its own tree, as the gates do. Logs go beside RUN_FILE. It prints
# `pd_sim: ready` once the router answers; start the traffic LP then. It exits
# once both servers have, 0 when both exited 0.
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

server() { # role port kv_role
    # uvicorn's keep-alive timeout runs on the LP clock, the router drops idle
    # connections on the wall clock: which comes first, and with it whether the
    # timeout takes a grant, would differ between runs. Only the router closes.
    # shellcheck disable=SC2086 # DECODE_EXEC and SERVER_ARGS are word lists
    ${4:-} env "${passed[@]}" python3 -m atom.entrypoints.openai_server \
        --model "$MODEL" --host 127.0.0.1 --server-port "$2" --timeout-keep-alive 1000000 \
        $SERVER_ARGS \
        --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
        --kv-transfer-config "{\"kv_role\":\"$3\",\"kv_connector\":\"compass\",\"compass_kv_write_req\":\"$KV_WRITE_REQ\"}" \
        >"$OUT/$1.log" 2>&1 &
}

wait_for() { # what pid url
    local deadline=$((SECONDS + WAIT_S))
    until curl -sf --max-time 10 "$3" >/dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "pd_sim: $1 exited before it answered" >&2; exit 1; }
        ((SECONDS < deadline)) || { echo "pd_sim: no $1 after $WAIT_S s" >&2; exit 1; }
        sleep 1
    done
}

python3 -m atom.compass.run --compass-run "$RUN_FILE" --compass-clock-endpoint "$CLOCK" \
    >"$OUT/authority.log" 2>&1 &
authority=$!
until (exec 3<>"/dev/tcp/127.0.0.1/$CLOCK_PORT") 2>/dev/null; do
    kill -0 $authority 2>/dev/null || { echo "pd_sim: the authority exited" >&2; exit 1; }
    sleep 0.2
done

server prefill "$PREFILL_PORT" kv_producer
prefill=$!
server decode "$DECODE_PORT" kv_consumer "${DECODE_EXEC:-}"
decode=$!
wait_for prefill $prefill "http://127.0.0.1:$PREFILL_PORT/health"
wait_for decode $decode "http://127.0.0.1:$DECODE_PORT/health"

# One worker per role, so the policy has nothing to choose. With the health
# check, circuit breaker and retries off, no wall-clock failure detector can
# resend or drop a request.
"$ATOMESH" launch --host 127.0.0.1 --port "$ROUTER_PORT" --pd-disaggregation \
    --prefill "http://127.0.0.1:$PREFILL_PORT" --decode "http://127.0.0.1:$DECODE_PORT" \
    --policy random --backend atom --disable-health-check --disable-circuit-breaker \
    --disable-retries --prometheus-port "$PROMETHEUS_PORT" --log-level info \
    >"$OUT/router.log" 2>&1 &
wait_for router $! "http://127.0.0.1:$ROUTER_PORT/v1/models"
echo "pd_sim: ready"

rc=0
wait $prefill || rc=$?
wait $decode || rc=$?
# The authority leaves at the finish, once it has written the step table and
# the run summary.
((rc != 0)) || wait $authority || rc=$?
exit $rc
