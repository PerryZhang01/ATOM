#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The scheduling-parity test on one node: one workload (atom/compass/parity/
# traffic.py) at one configuration on the real engine, recording
# each step, then on the simulated runner replaying the real steps' times
# (atom/compass/parity/replay.py), and the two records compared per DP rank.
#
# Usage: parity_one_node.sh OUT_DIR
#   CONFIG           tp2: -tp 2 --enable-dp-attention (default). tp2-plain: tp2
#                    without --enable-dp-attention, so one DP rank. dsv4-dpa8:
#                    DeepSeek-V4-Pro's nightly DP-attention prefill server of
#                    recipes/mesh/DeepSeek-V4.md, aggregated, MTP and expert
#                    parallelism off: -tp 8 --enable-dp-attention --enable-tbo with
#                    the nightly's flags and environment; it needs MACHINE.
#                    Under DP-attention the DP width is the -tp width.
#                    rapidserve: -tp 1 --enable-rapidserve, eager at compile
#                    level 0; each engine records, and is compared, in its own
#                    prefill/ and decode/.
#   MODEL            model directory with its weights (required)
#   REQUESTS         requests in the workload (default 64)
#   GAP_S            seconds between sends, each also waiting for the server to
#                    admit the one before; 0 sends them back to back (default
#                    0.05, or 0.25 under ALONE=1)
#   ALONE            1: request 0 runs alone to its end before the rest are sent,
#                    and GAP_S counts from its end, so later requests join
#                    running decode batches. Default 0: all sent from the start.
#   LONG_TOKENS      output tokens request 0 asks for, to keep one request (and
#                    under LMCACHE its server session) alive for long (default
#                    0: its usual count)
#   MACHINE          machine spec (tp2's and tp2-plain's default scripts/compass/machines/mi308x-hjbog18.json)
#   STORE            artifact store holding the simulated configuration's memory
#                    readings, taken by `compass measure memory`. Unset, the
#                    script takes them first into OUT_DIR/store: `compass measure
#                    memory` on the real run's GPUs, flags and environment,
#                    under LMCACHE with an lmcache server of its own beside it.
#   SERVER_ARGS      more API server flags, on both runs (default CONFIG's; tp2's are
#                    --enforce-eager --max-model-len 8192 --no-enable_prefix_caching)
#   WAIT_S           seconds a server has to start, and to stop (default 1800)
#   LMCACHE          1: both runs offload KV to an lmcache server under
#                    ATOM_KV_OFFLOAD=lmcache_mp, loading and saving from 256
#                    tokens; the real run to a real `lmcache server` started
#                    beside it with LMCACHE_FLAGS, the simulated one to LP lmcache
#                    with the same flags, its request loop on the declared laws
#                    below and its copies on MACHINE's interconnect.host_link,
#                    which it then needs. Every request starts with PREFIX_WORDS
#                    shared words (default 600). Default 0.
#   FAMILIES         prefix families: above 1, request k shares PREFIX_WORDS
#                    only with the requests of family k % FAMILIES (default 1)
#   LMCACHE_FLAGS    the server's flags besides --host and --port (default the
#                    recipes' --chunk-size 256 --null-block-id -1
#                    --separate-object-groups --supported-transfer-mode
#                    lmcache_driven --l1-size-gb 8 --eviction-policy LRU)
#   IMAGES           images in each request, which is then a chat completion
#                    (default 0); the simulated run prices each at IMAGE_S
#                    seconds (default 0.01)
# The real run uses HIP_VISIBLE_DEVICES as set, and the measurement those GPUs
# (unset, the first -tp of them). OUT_DIR gets memory/ when it measures, real/
# and sim/, each with its server log and step record (sim/ also has the run
# file, step table and summary), and compare.json; the last lines are the real run's
# admission, send to response headers, and the compare per DP rank. Under
# LMCACHE they also give the requests whose offload loads and saves differ, and
# each run's expired-session sweeps and eviction ticks, from its server's log. A
# simulated run that needs a step past the real record ends at the replay's
# refusal, which the last lines name; the records are compared all the same.
set -euo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
compass_env "$ROOT"
compass_require_tree "$ROOT" || exit $?
OUT=${1:?usage: parity_one_node.sh OUT_DIR}
MODEL=${MODEL:?MODEL names the model directory}
DPA=--enable-dp-attention SIDES=.
case ${CONFIG:=tp2} in
tp2 | tp2-plain)
    TP=2 ARGS="--enforce-eager --max-model-len 8192 --no-enable_prefix_caching"
    MACHINE=${MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}
    [[ $CONFIG == tp2 ]] || DPA=
    ;;
dsv4-dpa8)
    TP=8 ARGS="--trust-remote-code --no-enable_prefix_caching --kv_cache_dtype fp8
        --block-size 16 --gpu-memory-utilization 0.85 --max-num-seqs 512 --enable-tbo"
    MACHINE=${MACHINE:?CONFIG=dsv4-dpa8 needs MACHINE, the MI355X machine spec}
    export AITER_BF16_FP8_MOE_BOUND=0 ATOM_MOE_GU_ITLV=1 GPU_MAX_HW_QUEUES=5 ATOM_NUMA_BIND=1
    ;;
rapidserve)
    TP=1 DPA= ARGS="--enforce-eager --level 0 --max-model-len 8192 --no-enable_prefix_caching"
    ARGS+=" --enable-rapidserve" SIDES="prefill decode"
    MACHINE=${MACHINE:-$ROOT/scripts/compass/machines/mi308x-hjbog18.json}
    ;;
*) echo "parity_one_node: CONFIG is tp2, tp2-plain, dsv4-dpa8 or rapidserve, not $CONFIG" >&2; exit 2 ;;
esac
SERVER_ARGS=${SERVER_ARGS:-$ARGS}
WAIT_S=${WAIT_S:-1800}
ALONE=${ALONE:-0} GAP=0.05
[[ $ALONE == 1 ]] && GAP=0.25
mkdir -p "$OUT/real" "$OUT/sim"
OUT=$(realpath "$OUT")
LMCACHE=${LMCACHE:-0} PREFIX=0 LM_FLAGS=
if [[ $LMCACHE == 1 ]]; then
    PREFIX=${PREFIX_WORDS:-600} LM_PORT=$(compass_free_port)
    LM_FLAGS="--host 127.0.0.1 --port $LM_PORT ${LMCACHE_FLAGS:---chunk-size 256
        --null-block-id -1 --separate-object-groups --supported-transfer-mode
        lmcache_driven --l1-size-gb 8 --eviction-policy LRU}"
    export ATOM_KV_OFFLOAD=lmcache_mp OFFLOAD_MIN_LOAD_TOKENS=256 OFFLOAD_MIN_SAVE_TOKENS=256
    export ATOM_KV_OFFLOAD_EXTRA_CONFIG="{\"lmcache.mp.host\": \"tcp://127.0.0.1\",
        \"lmcache.mp.port\": $LM_PORT}"
fi
if [[ -z ${STORE:-} ]]; then
    STORE=$OUT/store
    # measure memory takes rocm-smi's indexes of the real run's HIP devices.
    # A RapidServe engine reads a plain engine's readings at its flags: their key
    # states no RapidServe, and ATOM refuses the measuring runner under it.
    gpus=$(compass_gpus smi "${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((TP - 1)))}")
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    ATOM_DP_MASTER_PORT=$(compass_free_port) python3 -c \
        "from atom.compass.cli import main; main()" measure memory \
        --gpu "$gpus" --machine "$MACHINE" \
        ${LM_FLAGS:+--beside "lmcache server $LM_FLAGS"} \
        --store "$STORE" --out "$OUT/memory" -- --model "$MODEL" -tp "$TP" \
        $DPA ${SERVER_ARGS//--enable-rapidserve/}
fi

server= lmcache=
trap 'kill -KILL -- ${server:+-$server} ${lmcache:+-$lmcache} 2>/dev/null || true' EXIT
# serve SIDE [FLAGS...]: an API server recording to OUT/SIDE/record, in its own
# process group; returns once it listens.
serve() {
    local side=$1 deadline=$((SECONDS + WAIT_S))
    shift
    PORT=$(compass_free_port)
    # shellcheck disable=SC2086 # SERVER_ARGS is a word list
    ATOM_DP_MASTER_PORT=$(compass_free_port) ATOM_COMPASS_PARITY_RECORD=$OUT/$side/record setsid \
        python3 -m atom.entrypoints.openai.api_server --model "$MODEL" --host 127.0.0.1 \
        --server-port "$PORT" --port "$(compass_free_port)" -tp "$TP" $DPA \
        $SERVER_ARGS "$@" \
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

if [[ $LMCACHE == 1 ]]; then
    # shellcheck disable=SC2086 # LM_FLAGS is a word list
    setsid lmcache server $LM_FLAGS >"$OUT/real/lmcache.log" 2>&1 &
    lmcache=$!
fi
serve real
python3 -m atom.compass.parity.traffic "$PORT" "$MODEL" "$OUT/real/record" \
    "${REQUESTS:-64}" "${GAP_S:-$GAP}" "$PREFIX" "${IMAGES:-0}" "$ALONE" "${LONG_TOKENS:-0}" \
    "${FAMILIES:-1}" \
    >"$OUT/real/traffic.log" 2>&1
# The record's last lines are written as the workers exit.
kill -TERM -- -$server
leave real
if [[ -n $lmcache ]]; then
    kill -KILL -- -$lmcache
    { wait $lmcache; } 2>/dev/null || true  # LP lmcache binds its port next
fi
# Each engine's record is a whole record, as the replay reads it.
for side in $SIDES; do
    [[ $side == . ]] && continue
    mkdir -p "$OUT/real/record/$side"
    cp "$OUT/real/record/arrivals.json" "$OUT/real/record/$side"
done

RUN=$OUT/sim/run.json
python3 - "$RUN" "$OUT" "$MACHINE" "$MODEL" "$STORE" "$(compass_free_port)" "${DPA:+$TP}" \
    "$LM_FLAGS" "${IMAGE_S:-0.01}" "$TP" "$SIDES" <<'EOF'
import json, re, sys
from pathlib import Path

run, out, machine, model, store, port, width, lmcache, image_s, tp, sides = sys.argv[1:]
machine = json.loads(Path(machine).read_text())
# ponytail: a spec that measured no tokenizer for the model prices it as its
# first entry; only arrival latency reads it. Measure the model's own to drop this.
arch = json.loads((Path(model) / "config.json").read_text())["architectures"][0]
tokenizers = machine["host"]["tokenizers"]
if not any(arch in entry["applies_to"] for entry in tokenizers):
    tokenizers[0]["applies_to"].append(arch)
spec = {
    "clock_endpoint": f"tcp://127.0.0.1:{port}", "bound_s": 1e7,
    "admission_path": "serving", "ipc_s": 2.0**-14, "stream_s": 2.0**-12,
    "replay": f"{out}/real/record", "step_stub": {}, "machine": machine, "store": store,
    "data_parallel_size": int(width or 1), "out_dir": f"{out}/sim",
    "image_preprocess_s": float(image_s),
}
if sides != ".":
    spec["rapidserve"] = True
if lmcache:
    # Declared, not measured: a LOOKUP 2^-13 s plus 2^-17 s a prompt chunk,
    # any other request 2^-14 s.
    laws = {"lookup_s": [2.0**-13, 2.0**-17], "request_s": 2.0**-14}
    # Writers per DP rank: one under DP-attention, else every TP rank, which
    # a model whose ranks share one copy refuses by name.
    writers = 1 if width else int(tp)
    spec["lmcache"] = {"server": lmcache.split(), "writers": writers, **laws}
    # A rank's recurrent-state image, as the real run logged it; the simulated
    # engine refuses a run file that names another.
    image = re.findall(r"image_bytes=(\d+)", Path(f"{out}/real/server.log").read_text())
    spec["lmcache"]["state_bytes"] = int(image[0]) if image else 0
Path(run).write_text(json.dumps(spec))
EOF
serve sim --compass-run "$RUN"
ATOM_COMPASS_RUN=$RUN python3 -m atom.compass.parity.traffic "$PORT" "$MODEL" \
    "$OUT/real/record" >"$OUT/sim/traffic.log" 2>&1 || sim_rc=$?
leave sim
wait $server || sim_rc=${sim_rc:-$?}
refused=$(grep -ohm1 "replay: .* has [0-9]* steps, and this is step [0-9]*" "$OUT/sim/server.log" || true)
if [[ -n $refused ]]; then
    echo "parity_one_node: the simulated run ended at the $refused"
elif [[ -n ${sim_rc:-} ]]; then
    echo "parity_one_node: the simulated run failed ($sim_rc); see $OUT/sim" >&2
    exit 1
fi

# compare.json, or for each RapidServe engine compare-<engine>.json.
for side in $SIDES; do
    to=$OUT/compare.json
    [[ $side == . ]] || to=$OUT/compare-$side.json
    python3 -m atom.compass.parity "$OUT/real/record/$side" "$OUT/sim/record/$side" >"$to"
done
python3 - "$OUT" <<'EOF'
import json, statistics, sys
from pathlib import Path
from atom.compass.parity import read

out = sys.argv[1]
arrivals = json.loads(Path(f"{out}/real/record/arrivals.json").read_text())
ms = [(a - s) / 1e6 for s, a in zip(arrivals["sent_ns"], arrivals["admitted_ns"])]
print(f"parity_one_node: real admission ms min={min(ms):.2f} "
      f"median={statistics.median(ms):.2f} max={max(ms):.2f}")
for path in sorted(Path(out).glob("compare*.json")):
    report, engine = json.loads(path.read_text()), path.stem[len("compare-") :]
    for rank, n in report["steps"].items():
        first = report["first_divergence"][rank]
        label = f"{engine} " if engine else ""
        print(f"parity_one_node: {label}dp{rank} compared={n['compared']} "
              f"identical={n['identical']} first_divergence={first and first['step']}")
decisions = []  # per record: request key -> its offload loads and saves, in order
for side in ("real", "sim"):
    mine = {}
    for steps in read(f"{out}/{side}/record").values():
        for step in steps:
            for key, *decision in step.get("kv", ()):
                mine.setdefault(key, []).append(decision)
    decisions.append(mine)
if any(decisions):
    real, sim = decisions
    differ = sorted(k for k in real.keys() | sim.keys() if real.get(k) != sim.get(k))
    print(f"parity_one_node: offload decisions of {len(real)} requests real, "
          f"{len(sim)} simulated; {len(differ)} differ: {differ}")
EOF
if [[ $LMCACHE == 1 ]]; then
    grep -H -e "expired sessions" -e "triggering eviction" "$OUT/real/lmcache.log" \
        "$OUT/sim/server.log" || true
fi
