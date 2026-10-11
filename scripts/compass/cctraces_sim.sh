#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# One cell of cc-traces through agentx-harness on a simulated TP1 or 1P1D run:
# ATOM with --compass-run, driven by `aiperf profile` with the AgentX
# scenario and compass-harness, until the scenario's duration has passed on the
# simulated clock. With REAL=1, the same cell on a real TP1 server: ATOM on the
# GPU, aiperf as shipped, the duration on the wall clock.
#
# Usage: cctraces_sim.sh RUN_FILE OUT_DIR
#   RUN_FILE        a run file; its out_dir and clock_endpoint are replaced
#   MODEL           model directory: config.json and the tokenizer (required;
#                   with REAL=1, the weights too)
#   TRACES          a cc-traces traces.jsonl (required)
#   HARNESS_PYTHON  the Python that has agentx-harness and compass-harness
#                   installed; with REAL=1, agentx-harness without
#                   compass-harness (required)
#   REAL            1 for a real cell (default 0)
#   RECORD          0 turns the step record off (default 1)
#   FIRST_SESSION   skip the first FIRST_SESSION traces (default 0)
#   SESSIONS        replay SESSIONS traces (default 2)
#   CONCURRENCY     aiperf --concurrency (default 1)
#   DURATION_S      aiperf --benchmark-duration, seconds on the cell's clock
#                   (default 1800)
#   SEED            aiperf --random-seed (default 20260707)
#   SERVER_ARGS     more API server flags (default: --enforce-eager
#                   --max-model-len 262144)
#   DECODE_EXEC, ATOMESH  as pd_sim.sh takes them, for a 1P1D cell
#
# A run file with `router_s` and `kv_write_req_s` makes the cell 1P1D: the
# deployment is pd_sim.sh's, and aiperf drives its atomesh router.
#
# It runs its own tree, as the gates do, compass-harness included. OUT_DIR gets
# run.json (RUN_FILE plus a `workload` entry hashing the traces, the dataset
# and the aiperf arguments), the server and aiperf logs, aiperf's artifacts,
# and the step table and run summary. The last line is the cell's result, with
# the KV transfer count and mean simulated seconds for a 1P1D cell. It exits
# non-zero when aiperf or the deployment did, or when aiperf's benchmark id is
# not the one recorded.
#
# A TP1 cell also writes its step record to steps/. A real cell writes no step
# table or summary; it saves rocm-smi's process listing before and after it in
# gpu_before.txt and gpu_after.txt, and its result names the processes on its
# GPU (the first in HIP_VISIBLE_DEVICES, default 0), with their VRAM bytes.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
export PYTHONPATH=$ROOT:$ROOT/compass_harness
compass_require_tree "$ROOT" || exit $?
RUN_FILE=$(realpath "${1:?usage: cctraces_sim.sh RUN_FILE OUT_DIR}")
OUT=${2:?usage: cctraces_sim.sh RUN_FILE OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
TRACES=${TRACES:?TRACES names a cc-traces traces.jsonl}
HARNESS_PYTHON=${HARNESS_PYTHON:?HARNESS_PYTHON names the Python with agentx-harness}
SESSIONS=${SESSIONS:-2}
REAL=${REAL:-0}
SERVER_ARGS=${SERVER_ARGS:---enforce-eager --max-model-len 262144}
WAIT_S=${WAIT_S:-600}
# compass-harness pins aiperf's benchmark id: uuid4 is UUID(int=0).
BENCHMARK_ID=000000000000

mkdir -p "$OUT/traces"
OUT=$(realpath "$OUT")
# Importing compass_harness also checks the aiperf it would run against.
harness=$(cd / && "$HARNESS_PYTHON" -c 'import compass_harness; print(compass_harness.__file__)')
[[ $harness == "$ROOT"/* ]] || { echo "cctraces_sim: compass_harness resolved to $harness" >&2; exit 92; }
if ((REAL)); then
    # Its plugin paces aiperf on a simulated clock, which a real cell has not got.
    plugins=$(cd / && "$HARNESS_PYTHON" -c 'from importlib.metadata import entry_points as e
print(*(p.value for p in e(group="aiperf.plugins")))')
    [[ $plugins != *compass_harness* ]] || { echo "cctraces_sim: REAL=1 and $HARNESS_PYTHON has compass-harness installed" >&2; exit 92; }
fi

awk -v d="$OUT/traces" -v first="${FIRST_SESSION:-0}" -v n="$SESSIONS" '
    NR > first {f = sprintf("%s/%05d.json", d, NR - first - 1); print > f; close(f)}
    NR >= first + n {exit}' "$TRACES"

# Everything but paths and the URL, so it can be hashed into the run record.
args=(--endpoint-type chat --custom-dataset-type weka_trace
    --concurrency "${CONCURRENCY:-1}" --benchmark-duration "${DURATION_S:-1800}"
    --random-seed "${SEED:-20260707}" --scenario inferencex-agentx-mvp --unsafe-override
    --streaming --use-server-token-count --extra-inputs ignore_eos:true
    --no-server-metrics --no-gpu-telemetry --ui none)

read -r MODE CLOCK_PORT PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT < <(
python3 - "$RUN_FILE" "$OUT" "$TRACES" "$BENCHMARK_ID" "${args[@]}" <<'EOF'
import hashlib, json, socket, sys
from pathlib import Path

run_file, out, traces, benchmark_id, *args = sys.argv[1:]
out = Path(out)
# Held together, so no two are the same port.
socks = [socket.socket() for _ in range(6)]
for s in socks:
    s.bind(("127.0.0.1", 0))
ports = [s.getsockname()[1] for s in socks]
for s in socks:
    s.close()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 24):
            h.update(chunk)
    return h.hexdigest()


run = json.loads(Path(run_file).read_text())
run["out_dir"] = str(out)
run["clock_endpoint"] = f"tcp://127.0.0.1:{ports[0]}"
run["workload"] = {
    "dataset_sha256": sha256(traces),
    "traces_sha256": [sha256(p) for p in sorted((out / "traces").iterdir())],
    "aiperf_args": args,
    "benchmark_id": benchmark_id,
}
(out / "run.json").write_text(json.dumps(run))
print("1p1d" if "kv_write_req_s" in run else "tp1", *ports)
EOF
)

trap 'kill -- $(jobs -p) ${server:+-$server} 2>/dev/null || true' EXIT
gpu=${HIP_VISIBLE_DEVICES:-0}
gpu=${gpu%%,*}
gpu_processes() { rocm-smi --showpids --showpidgpus >"$OUT/gpu_$1.txt"; }
if ((REAL)); then
    [[ $MODE == tp1 ]] || { echo "cctraces_sim: a real cell is TP1, and the run file is 1P1D" >&2; exit 2; }
    gpu_processes before
fi
if [[ $MODE == 1p1d ]]; then
    # pd_sim.sh takes the authority's port from its own variable; the run file
    # names the same one for the traffic LP.
    export MODEL SERVER_ARGS WAIT_S CLOCK_PORT DECODE_PORT ROUTER_PORT KV_WRITE_REQ_PORT PROMETHEUS_PORT
    PREFILL_PORT=$PORT bash "$ROOT/scripts/compass/pd_sim.sh" "$OUT/run.json" >"$OUT/pd_sim.log" 2>&1 &
    ready=("pd_sim: ready" "$OUT/pd_sim.log")
    PORT=$ROUTER_PORT
else
    compass=(--compass-run "$OUT/run.json")
    ((REAL)) && compass=()
    record=$OUT/steps
    ((${RECORD:-1})) || record=
    # The keep-alive timeout as in pd_sim.sh, aiperf's pool in the router's place.
    # Its own process group, so a real server's workers stop with it.
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    ATOM_COMPASS_PARITY_RECORD=$record setsid python3 -m atom.entrypoints.openai.api_server \
        --model "$MODEL" --host 127.0.0.1 --server-port "$PORT" --timeout-keep-alive 1000000 \
        $SERVER_ARGS "${compass[@]}" >"$OUT/server.log" 2>&1 &
    # Not /health: before the traffic LP joins the run, the frontend answers nothing.
    ready=("Uvicorn running" "$OUT/server.log")
fi
server=$!
deadline=$((SECONDS + WAIT_S))
until grep -q "${ready[@]}"; do
    kill -0 $server 2>/dev/null || { echo "cctraces_sim: the deployment exited before it started" >&2; exit 1; }
    ((SECONDS < deadline)) || { echo "cctraces_sim: no deployment after $WAIT_S s" >&2; exit 1; }
    sleep 1
done

rc=0
aiperf=(-m aiperf)
# Without compass-harness, the benchmark id is pinned here as it pins it, so
# both modes send the same prompts.
((REAL)) && aiperf=(-c 'import uuid, aiperf.cli_runner as r
r.uuid4 = lambda: uuid.UUID(int=0)
from aiperf.cli import app
app()')
(cd "$OUT" && ATOM_COMPASS_RUN=$OUT/run.json AIPERF_DATASET_MMAP_BASE_PATH=$OUT \
    "$HARNESS_PYTHON" "${aiperf[@]}" profile --url "127.0.0.1:$PORT" --model "$MODEL" \
    --tokenizer "$MODEL" --input-file "$OUT/traces" --artifact-dir "$OUT/artifacts" \
    "${args[@]}") >"$OUT/aiperf.log" 2>&1 || rc=$?
# The deployment leaves at the finish, once the step table is written. A real
# server serves until it is stopped, so one that has already left failed.
((REAL)) && { kill -TERM -- -$server 2>/dev/null || rc=$((rc ? rc : 1)); }
deadline=$((SECONDS + WAIT_S))
while kill -0 $server 2>/dev/null && ((SECONDS < deadline)); do sleep 1; done
kill -0 $server 2>/dev/null && { echo "cctraces_sim: the deployment is still up $WAIT_S s after aiperf" >&2; exit 1; }
wait $server || rc=$((rc || REAL ? rc : $?))
((REAL)) && { kill -KILL -- -$server 2>/dev/null || true; gpu_processes after; }

python3 - "$OUT" "$BENCHMARK_ID" "$rc" "$REAL" "$gpu" <<'EOF'
import json, re, sys
from pathlib import Path

out, benchmark_id, rc = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
export = json.loads((out / "artifacts/profile_export_aiperf.json").read_text())
avg = lambda k: export[k]["avg"]
if export["benchmark_id"] != benchmark_id:
    sys.exit(f"cctraces_sim: benchmark id {export['benchmark_id']}, recorded {benchmark_id}")
if sys.argv[4] == "1":
    def others(when):
        """pid:VRAM bytes of each KFD process on the cell's GPU."""
        text = (out / f"gpu_{when}.txt").read_text()
        vram = dict(re.findall(r"^(\d+) *\t[^\t]*\t[^\t]*\t(\d+)", text, re.M))
        on = re.findall(r"PID (\d+) is using \d+ DRM device\(s\):\n([\d ]*)", text)
        return ",".join(f"{p}:{vram.get(p, '?')}" for p, d in on if sys.argv[5] in d.split())

    before, after = others("before"), others("after")
    print(
        f"cctraces_sim: rc={rc} requests={avg('request_count'):.0f} "
        f"errors={len(export['error_summary'])} "
        f"cache_read_pct={avg('overall_usage_prompt_cache_read_pct'):.2f} "
        f"theoretical_hit_pct={avg('theoretical_prefix_cache_hit'):.2f} "
        f"ttft_ms={avg('time_to_first_token'):.2f} itl_ms={avg('inter_token_latency'):.3f} "
        f"request_throughput={avg('request_throughput'):.4f} "
        f"contaminated={bool(before or after)} gpu_before={before} gpu_after={after}"
    )
    sys.exit(rc)
summary = json.loads((out / "summary.json").read_text())
# Prefill logs one line per KV transfer, with its simulated seconds.
prefill = out / "prefill.log"
kv = ""
if prefill.exists():
    t = [float(s) for s in re.findall(r"KV transfer .*, (\S+) simulated s", prefill.read_text())]
    kv = f" kv_transfers={len(t)} kv_transfer_mean_s={sum(t) / len(t) if t else 0:.6g}"
print(
    f"cctraces_sim: rc={rc} requests={avg('request_count'):.0f} "
    f"errors={len(export['error_summary'])} "
    f"simulated_s={summary['cost']['simulated_seconds']:.2f} wall_s={summary['cost']['wall_seconds']:.0f} "
    f"cache_read_pct={avg('overall_usage_prompt_cache_read_pct'):.2f} "
    f"theoretical_hit_pct={avg('theoretical_prefix_cache_hit'):.2f} "
    f"ttft_ms={avg('time_to_first_token'):.2f} itl_ms={avg('inter_token_latency'):.3f} "
    f"request_throughput={avg('request_throughput'):.4f} "
    f"refusals={summary['schedule']['refusals']['count']} "
    f"coverage_report={summary['coverage_report']}"
    + kv
)
sys.exit(rc)
EOF
