#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# GPU pre-flight: is this node usable, and which devices are free? Run BEFORE
# and AFTER every GPU task -- three of the last five prior pilot attempts were
# lost or degraded by another tenant arriving mid-run.
#
#   0. D-state census    <- `timeout` cannot kill a probe already wedged
#   1. rocminfo          <- hangs? node is WEDGED. Do not use it.
#   2. rocm-smi --showuse     compute busy?
#   3. rocm-smi --showmemuse  VRAM held?   <- the one that gets missed
#
# Check 3 is not optional: a node can read 0% utilisation and 99% VRAM (a loaded,
# idle model), and `non_torch` is a device-wide reading, so a neighbour's
# allocation is indistinguishable from ours.
#
# This is not only for GPU tasks. aiter's arch discovery shells out to rocminfo
# when aiter is imported, and so does importing any ATOM module that imports
# aiter at load time (atom.model_engine.model_runner, atom.model_ops.linear) -- so on
# a wedged node a "CPU-only, no hardware" task that reaches the model layer hangs
# too, and looks like a bug in the task. MEASURED 2026-09-23 on node 18: a bare
# `import atom`, atom.config, atom.compass, atom.model_engine.llm_engine and
# atom.entrypoints.openai_server run no rocminfo.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"
RC=0

# --- 0. D-state census -------------------------------------------------------
# What this counts is whatever this PID namespace can see, which in a container
# is not the node. MEASURED 2026-09-20T08:45Z on hjbog-srdc-39, the same second
# from both sides: the host had 10840 processes of which 2236 were in D state;
# the jgong5_vllm container on it saw 3171 and 103. `docker inspect
# -f '{{.HostConfig.PidMode}}'` is empty, so the namespace is private and there
# is no /host/proc bind-mount to look through. The container census sees 4.6% of
# the D-state processes that are actually there.
#
# The threshold of 20 was calibrated on host-wide readings -- 0 on a healthy
# node, 2183 on the wedged one. Against a container-local count, a number under
# it means only "nothing is wedged in here". It does not mean the node is clear,
# so this check reports PARTIAL rather than a clear: the question was not
# answered, and an unanswered question must not read as a "no".
#
# So the census reports its own scope, and the scope is measured rather than
# assumed: a readable host procfs is used when one is mounted, and otherwise the
# container is detected positively rather than inferred from the count.
#
# Counting is done over /proc directly rather than with `ps`, so an alternative
# procfs can be counted the same way, and so the check has one less dependency
# on a node where a wedge makes the process tools themselves unreliable.
#
# Each entry is read by the shell rather than handed to awk as a glob. The
# container's awk is mawk 1.3.4, and mawk treats an input file it cannot open as
# FATAL: a process that exits between glob expansion and read -- routine on a
# shared box -- takes the whole census down with rc=2 and NO output at all, END
# never runs, CENSUS comes back empty and this check reports UNMEASURED. In
# gate_gpu.sh that is `ABORT: pre-flight failed before the run` and a booked GPU
# slot spent on a vanished PID. Measured rare -- 0 aborts in 60 consecutive real
# censuses over a 3179-process namespace -- and cheap to remove, so it is
# removed: `read` is a builtin, forks nothing, reads only the first line, and a
# vanished entry is skipped rather than fatal.
printf '=== 0. D-state census ===\n'
PROCFS=/proc
SCOPE=host
for c in /host/proc /hostfs/proc /rootfs/proc; do
    [ -r "$c/1/stat" ] && { PROCFS=$c; SCOPE="host (via $c)"; break; }
done
if [ "$SCOPE" = host ] &&
    { [ -e /.dockerenv ] || grep -qaE '(docker|containerd|kubepods|lxc)' /proc/1/cgroup 2>/dev/null; }; then
    SCOPE=container-local
fi
CENSUS=$(
    for f in "$PROCFS"/[0-9]*/stat; do
        IFS= read -r line <"$f" 2>/dev/null || continue
        printf '%s\n' "$line"
    done |
        awk '{ sub(/^[0-9]+ \(.*\) /, ""); t++; if ($1 == "D") d++ }
             END { printf "%d %d", t + 0, d + 0 }'
)
TOTAL=${CENSUS% *}
DSTATE=${CENSUS#* }
if [ -z "$CENSUS" ] || [ "${TOTAL:-0}" -eq 0 ]; then
    printf 'UNMEASURED: %s yielded no process entries, so the census did not run.\n' "$PROCFS" >&2
    printf '            Checks 1-3 below still apply; this one has no reading.\n' >&2
    RC=1
else
    printf 'scope: %s\n' "$SCOPE"
    printf 'processes visible: %s, of them uninterruptible-sleep: %s\n' "$TOTAL" "$DSTATE"
    if [ "$DSTATE" -gt 20 ]; then
        printf 'WEDGED: a driver wedge leaves thousands of unkillable probes behind.\n' >&2
        printf '        They survive `docker stop` and every signal; clearing it needs a host\n' >&2
        printf '        reset, which is the owner/admin call. Use another node.\n' >&2
        case "$SCOPE" in container-local)
            printf '        This count is container-local, so the node-wide figure is larger --\n' >&2
            printf '        measured 103 in-container against 2236 on the host.\n' >&2 ;;
        esac
        RC=1
    elif [ "$SCOPE" = container-local ]; then
        printf 'PARTIAL: nothing is wedged inside this container. The node was NOT\n'
        printf '         examined -- this PID namespace is private and no host procfs is\n'
        printf '         mounted, so a wedge outside the container is invisible here and\n'
        printf '         a low number is not a clear node. Check 1 is what still sees it:\n'
        printf '         rocminfo goes through the driver, so it hangs on a wedged host\n'
        printf '         whatever this census says.\n'
    fi
fi

printf '\n=== 1. rocminfo (25 s) ===\n'
if timeout 25 rocminfo >/dev/null 2>&1; then
    printf 'rocminfo: OK\n'
else
    printf 'WEDGED or absent: rocminfo did not return in 25 s (exit %s).\n' "$?" >&2
    printf '        `timeout` cannot kill it if it is already in D state -- it stays.\n' >&2
    RC=1
fi

USE=$(rocm-smi --showuse 2>&1)
MEM=$(rocm-smi --showmemuse 2>&1)
printf '\n=== 2. compute ===\n%s\n' "$(printf '%s\n' "$USE" | grep -E '^GPU\[')"
printf '\n=== 3. VRAM ===\n%s\n' "$(printf '%s\n' "$MEM" | grep -E '^GPU\[.*VRAM%')"

# Name what is bookable. Deliberately not part of RC: our own run trips any
# threshold, and a neighbour's allocation reads identically to ours, so this
# reports and the human books. An empty list is the useful answer.
printf '\n=== bookable ===\n'
FREE=
for i in $(printf '%s\n' "$USE" | grep -oE '^GPU\[[0-9]+\]' | grep -oE '[0-9]+' | sort -un); do
    u=$(printf '%s\n' "$USE" | grep -E "^GPU\[$i\].*use" | grep -oE '[0-9]+$' | tail -1)
    m=$(printf '%s\n' "$MEM" | grep -E "^GPU\[$i\].*VRAM%" | grep -oE '[0-9]+$' | tail -1)
    if [ "${u:-100}" -le 5 ] && [ "${m:-100}" -le 5 ]; then FREE="$FREE $i"; fi
done
if [ -n "$FREE" ]; then
    printf 'idle and empty:%s\n' "$FREE"
    # Every free device, comma-joined. This printed ONE device until 2026-09-20:
    # `printf ... | cut -d' ' -f1-2` took fields of the whole line, so
    # "export HIP_VISIBLE_DEVICES=0 1 2" was cut back to "...=0" -- the line
    # above reported three devices free and the line you copy handed you one.
    # FREE is rocm-smi's indexes; HIP_VISIBLE_DEVICES takes the same GPUs' HIP ordinals.
    hip=$(compass_gpus hip "$(printf '%s' "${FREE# }" | tr ' ' ',')") &&
        printf 'export HIP_VISIBLE_DEVICES=%s\n' "$hip"
else
    printf 'none idle and empty -- every device is busy or holds VRAM.\n'
fi

printf '\nPREFLIGHT_RC=%s\n' "$RC"
exit "$RC"
