#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# The real cells the step linear law and the host laws are fitted from: cc-traces 256
# to 392 on a real TP1 server through `REAL=1 cctraces_sim.sh`, at each
# concurrency in CELLS, one cell at a time, since a cell run beside another
# moves the host time the host laws measure. Each cell is stopped STEPS_S
# seconds after its first recorded step: at concurrency 64 and above aiperf's
# warmup alone outlasts the benchmark duration.
#
# Usage: calib_capture.sh RUN_FILE OUT_DIR
#   RUN_FILE, MODEL, TRACES, HARNESS_PYTHON  as cctraces_sim.sh takes them
#   CELLS        concurrencies, in run order (default: 1 4 16 64 256)
#   STEPS_S      wall seconds recorded per cell (default 1800)
#   SERVER_ARGS  API server flags (default: --max-model-len 262144, so decode
#                runs on captured graphs)
#   HIP_VISIBLE_DEVICES  the GPUs as HIP numbers them (default 0); a cell refuses
#                to start while `rocm-smi` lists a process on any of them
#
# OUT_DIR/c<C>/ is the cell's cctraces_sim.sh directory, its step record in
# steps/; OUT_DIR/c<C>.log holds the tree's commit, the host load (`uptime`),
# the GPUs' processes, VRAM and use at the cell's start and end, and the cell's
# exit status, aiperf's: on SIGINT aiperf writes its export, or, stopped in its
# warmup, none and exits 1. The cell's run.json gains a `compass record` identity.
# The capture exits 1 naming each cell left without artifacts/client_run.json.
#   compass fit steps --backend step-linear OUT_DIR/c<C>... --client-run OUT_DIR/c<C> \
#       OUT_DIR/c<C>/artifacts/client_run.json... --machine SPEC --store S --out D
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
RUN_FILE=$(realpath "${1:?usage: calib_capture.sh RUN_FILE OUT_DIR}")
OUT=${2:?usage: calib_capture.sh RUN_FILE OUT_DIR}
mkdir -p "$OUT" && OUT=$(realpath "$OUT")
export REAL=1 RECORD=1 FIRST_SESSION=256 SESSIONS=137 WAIT_S=${WAIT_S:-1800}
export SERVER_ARGS=${SERVER_ARGS:---max-model-len 262144}
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
unset ATOM_COMPASS_RUN
# The same GPUs as rocm-smi indexes them, which its listings use.
SMI_GPUS=$(compass_gpus smi "$HIP_VISIBLE_DEVICES") || exit 2
# The GPUs `rocm-smi` lists a process on.
busy() { rocm-smi --showpids --showpidgpus | awk '/is using [1-9]/ {getline; print}' | tr ' ' '\n'; }
state() {
    echo "$1 $(date -u +%FT%TZ) uptime:$(uptime)"
    { rocm-smi --showpids --showpidgpus; rocm-smi --showmeminfo vram --showuse; } | sed 's/^/  /'
}

for c in ${CELLS:-1 4 16 64 256}; do
    o=$OUT/c$c log=$OUT/c$c.log
    [[ -e $o ]] && { echo "calib_capture: $o exists" >&2; exit 4; }
    { cat "$ROOT/.compass-commit" 2>/dev/null || git -C "$ROOT" rev-parse HEAD
      echo "CONCURRENCY=$c GPU=$HIP_VISIBLE_DEVICES ROCM_SMI_GPU=$SMI_GPUS STEPS_S=${STEPS_S:-1800} SERVER_ARGS=$SERVER_ARGS"
      state start; } >"$log"
    gpu=$(busy | grep -xFm1 -f <(tr , '\n' <<<"$SMI_GPUS")) && { echo "calib_capture: rocm-smi GPU $gpu is in use" | tee -a "$log" >&2; exit 3; }
    CONCURRENCY=$c timeout 7200 bash "$ROOT/scripts/compass/cctraces_sim.sh" "$RUN_FILE" "$o" >"$o.out" 2>&1 &
    sim=$!
    until [[ -s $o/steps/dp0.jsonl ]] || ! kill -0 $sim 2>/dev/null; do sleep 10; done
    if kill -0 $sim 2>/dev/null; then
        echo "first step $(date -u +%FT%TZ)" >>"$log"
        # Wakes each 10 s, so a cell that ends on its own is not held to STEPS_S.
        for ((t = 0; t < ${STEPS_S:-1800}; t += 10)); do kill -0 $sim 2>/dev/null && sleep 10; done
        # aiperf runs as a child of cctraces_sim.sh, which runs under timeout.
        pkill -INT -P "$(pgrep -P $sim)" -f "^aiperf system_controller" && echo "stopped aiperf $(date -u +%FT%TZ)" >>"$log"
    fi
    wait $sim
    echo "CELL_RC=$?" >>"$log"
    # The identity `compass record` writes, so `compass fit` reads the cell.
    (cd / && PYTHONPATH=$ROOT python3 -c 'import json, sys; from pathlib import Path
from atom.compass.drive.identity import identity
o, model, gpus, *argv = sys.argv[1:]; run = Path(o, "run.json")
ids = identity(Path(o, "steps/dp0.engine.json"), model, argv, tuple(map(int, gpus.split(","))))
run.write_text(json.dumps(json.loads(run.read_text()) | ids))' "$o" "${MODEL-}" "$SMI_GPUS" \
        "$ROOT/scripts/compass/cctraces_sim.sh" "$RUN_FILE" "$o") 2>>"$log" || echo "no identity" >>"$log"
    state end >>"$log"
    [[ -s $o/artifacts/client_run.json ]] || unexported+=" c$c"
done
[[ -z ${unexported-} ]] || { echo "calib_capture: no client-run export:$unexported" >&2; exit 1; }
