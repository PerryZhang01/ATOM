#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Per-task test gate: the test files that run without a GPU driver.
# Green is the bar: this subset really is green.
#
# Excluded, and why neither is a judgement call:
#   tests/plugin/          needs sglang + vllm -- in neither image
#   cpu_gate_exclude.txt   files that reach the driver, most at *collection*
#                          time via rocminfo, the rest on a driver call after
#                          collecting. That file names which is which and
#                          which is generated; README.md's row counts both.
#
# What this tier cannot see is not left to memory. gpu_gate_triggers.txt lists
# the source paths whose only coverage is in the excluded set; a diff touching
# one of them fails this gate with an instruction to run gate_gpu.sh. Before
# that, the rule was a sentence in a design document and nothing read it.
#
# Run in the CPU container. Regenerate the exclusion list with
# regen_cpu_gate_exclude.sh; never hand-edit the GENERATED section.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

# GATE_CPU_RC is printed exactly once, on every path out of this script. It is
# defined here, above the first exit, and not further down beside the pytest
# result, because three drafts got the arithmetic of "exactly once" wrong in
# three different ways:
#   - an early draft printed it after pytest and again after the blind-spot
#     check, so a run that failed the second emitted `GATE_CPU_RC=0` followed by
#     `GATE_CPU_RC=98`, and anything grepping for the first answer got the wrong
#     one;
#   - a second draft exited before pytest on an unanswerable GPU question, so
#     that run printed it zero times -- equally unusable to a caller;
#   - a third draft -- this one, until the review that produced this comment --
#     defined finish() *below* seven FATAL exits. A tree that could not be
#     resolved, an unreadable exclusion list, or a stale --ignore= line all left
#     the script with no verdict line at all, which reads to a caller exactly
#     like a run that has not finished yet. A gate that cannot say it failed has
#     not failed safely.
# The rule this encodes: the verdict is the script's only output contract, so no
# path may skip it, including the ones that give up before measuring anything.
#
# The verdict line also carries its reason, because it is the only line every
# pipe keeps. `gate_cpu.sh | tail -6` hands the caller tail's exit status, not
# this script's, and nothing a script does can change that: the status of a
# pipeline belongs to the caller's shell. What survives is the text, and the
# three common pipes keep different parts of it -- `2>/dev/null | tail -6` drops
# every stderr line, so a bare `GATE_CPU_RC=98` there sits under pytest's green
# summary with nothing to say why it is not a pass. Stated on the verdict line,
# the reason survives any pipe that keeps the verdict at all.
#
# Refusing to run when stdout is a pipe was weighed and rejected. It cannot tell
# `| tail` from the callers that keep the status: under `docker exec`, which is
# how the gate runs on the CPU node, stdout is a FIFO too, as it is under any
# harness that captures output. Every normal run would need an override, and an
# override set by habit lets `| tail` through with it.
finish() {
    if [ "$1" -eq 0 ]; then
        printf 'GATE_CPU_RC=0 PASSED\n'
    else
        printf 'GATE_CPU_RC=%s NOT PASSED -- %s\n' "$1" "$2"
    fi
    exit "$1"
}
printf 'verdict: the last line of stdout; if this run is piped, $? is the pipe'"'"'s\n'

# pytest's -r is store-last-wins, so a caller's -rE silently replaces the -rf
# below. This gate's own verdict is pytest's exit code and survives that, but
# its sibling gate_gpu.sh builds its verdict out of those report lines and does
# not -- `gate_gpu.sh -rE` on d78f3bbd3 reported all five baseline failures "no
# longer failing" and exited 0 while they had failed. The two gates are read as
# one contract, so they refuse the same flag rather than differing on which
# arguments are safe; and the -rf list is the only place this script names which
# tests failed, which is the whole of its value to the person reading it.
for arg in "$@"; do
    case "$arg" in
    -r | -r*)
        printf 'REFUSED: %s sets pytest -r, which this gate owns. It selects the\n' "$arg" >&2
        printf '  report lines that name the failing tests, and -r is\n' >&2
        printf '  store-last-wins, so your flag would replace them silently.\n' >&2
        finish 95 "refused a -r argument; nothing was run"
        ;;
    esac
done

INTEGRATION=${COMPASS_INTEGRATION_REF:-feature/atomcompass_new}

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

EXCLUDE=$ROOT/scripts/compass/cpu_gate_exclude.txt
TRIGGERS=$ROOT/scripts/compass/gpu_gate_triggers.txt
[ -r "$EXCLUDE" ] || { printf 'FATAL: cannot read %s\n' "$EXCLUDE" >&2; finish 94 "exclusion list unreadable; nothing was run"; }
[ -r "$TRIGGERS" ] || { printf 'FATAL: cannot read %s\n' "$TRIGGERS" >&2; finish 94 "trigger list unreadable; nothing was run"; }

# Parse defensively: strip a trailing CR so a list that has been through a
# Windows editor does not turn into --ignore=tests/foo.py^M, which pytest
# accepts and silently matches nothing; drop inline comments; and check the
# file exists, because a renamed test would otherwise be excluded forever by a
# stale line nobody can see failing.
NEX=0
IGN=(--ignore=tests/plugin)
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}
    line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}
    line=${line%"${line##*[![:space:]]}"}
    [ -n "$line" ] || continue
    [ -e "$ROOT/$line" ] || {
        printf 'FATAL: %s lists %s, which does not exist.\n' "$EXCLUDE" "$line" >&2
        printf '       Re-run regen_cpu_gate_exclude.sh.\n' >&2
        finish 94 "exclusion list names a missing file; nothing was run"
    }
    IGN+=(--ignore="$line")
    NEX=$((NEX + 1))
done <"$EXCLUDE"
printf 'gate:   %s files excluded + tests/plugin\n' "$NEX"

# --- does this change need the GPU tier? -------------------------------------
# Answered here, enforced after pytest, so a run yields both answers rather than
# trading one for the other. That includes the case where the question cannot be
# answered at all: it is recorded, not exited on, so the suite still reports.
GPU_NEEDED=
CHANGED=
SRC=
GPU_SRC_UNKNOWN=
if [ -n "${COMPASS_CHANGED_FILES:-}" ]; then
    [ -r "$COMPASS_CHANGED_FILES" ] ||
        { printf 'FATAL: COMPASS_CHANGED_FILES=%s unreadable\n' "$COMPASS_CHANGED_FILES" >&2; finish 94 "COMPASS_CHANGED_FILES unreadable; nothing was run"; }
    CHANGED=$(cat "$COMPASS_CHANGED_FILES")
    SRC="COMPASS_CHANGED_FILES"
elif git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1 &&
     REF=$(compass_resolve_ref "$ROOT" "$INTEGRATION") &&
     BASE=$(git -C "$ROOT" merge-base HEAD "$REF" 2>/dev/null); then
    CHANGED=$(git -C "$ROOT" diff --name-only "$BASE")
    SRC="diff vs $REF (${BASE:0:9})"
    # Announce the fallback the way snapshot.sh does. Without this the resolved
    # name appears only inside the `gpu:` source string -- reported, not
    # announced, so a caller who set nothing sees a prefix it never asked for.
    [ "$REF" = "$INTEGRATION" ] ||
        printf 'ref:    %s is not a ref here; resolved it as %s\n' "$INTEGRATION" "$REF"
    compass_ref_drift "$ROOT" "$REF"
elif [ -r "$ROOT/.compass-changed" ]; then
    # A snapshot built by snapshot.sh, which is how this gate normally runs.
    # The stamp was written from the same rev-parse that selected the archived
    # tree, so it describes this tree and not whichever one the archiver
    # happened to be sitting in.
    #
    # The drift line above does not reach here, and this is the path the gate
    # normally runs on. .compass-changed records the file list and
    # .compass-commit the sha; neither records the *base*, so a stamp built on
    # a stale local branch cannot be told from one built on the remote, and the
    # announcement lives only in the staging log while this is the log that
    # gets pasted. Closing it means recording the base in a stamp, which
    # changes the stamp format and is not #102.
    CHANGED=$(cat "$ROOT/.compass-changed")
    SRC=".compass-changed stamp"
else
    # No supplied list, no usable diff, no stamp: the question cannot be
    # answered. It is recorded as unanswered, with which of the three ways in
    # failed, and the run below exits 98 rather than passing -- reporting "GPU
    # not required" here would be a guess that reads as a clear. Only a tree with
    # no .git is a staging omission; a git checkout reaches here because the
    # integration ref does not resolve, or shares no commit with HEAD.
    if ! git -C "$ROOT" rev-parse --git-dir >/dev/null 2>&1; then
        GPU_SRC_UNKNOWN="this tree was never stamped (no .git, no .compass-changed); a staging omission, see snapshot.sh"
    elif REF=$(compass_resolve_ref "$ROOT" "$INTEGRATION"); then
        GPU_SRC_UNKNOWN="a git checkout with no .compass-changed, and $REF shares no commit with HEAD"
    else
        GPU_SRC_UNKNOWN="a git checkout with no .compass-changed, and $INTEGRATION resolves to no commit here"
    fi
fi

if [ -z "$GPU_SRC_UNKNOWN" ]; then
    while IFS= read -r pat; do
        pat=${pat%$'\r'}
        case "$pat" in ''|'#'*) continue ;; esac
        while IFS= read -r f; do
            [ -n "$f" ] || continue
            case "$pat" in
            */) case "$f" in "$pat"*) GPU_NEEDED="$GPU_NEEDED$f -> $pat"$'\n' ;; esac ;;
            *)  [ "$f" = "$pat" ] && GPU_NEEDED="$GPU_NEEDED$f -> $pat"$'\n' ;;
            esac
        done <<<"$CHANGED"
    done <"$TRIGGERS"
fi

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'gpu:    UNKNOWN -- %s\n' "$GPU_SRC_UNKNOWN"
elif [ -n "$GPU_NEEDED" ]; then
    printf 'gpu:    REQUIRED (%s)\n' "$SRC"
    printf '%s' "$GPU_NEEDED" | sed 's/^/          /'
else
    printf 'gpu:    not required (%s)\n' "$SRC"
fi
printf '\n'

# Identify the tree before running, not after: an attestation that this diff
# already passed the GPU tier is per-tree, so it cannot be honoured on a tree we
# cannot name.
HEAD_SHA=$(compass_head_sha "$ROOT") || HEAD_SHA=UNKNOWN

# The compass_harness tests skip by name when compass-harness, or the aiperf it
# adapts, does not import, and the CPU container has neither. So this gate
# installs the tree's own compass_harness into a run-local directory every run,
# beside the aiperf wheel COMPASS_AGENTX_HARNESS names, and refuses to run when
# the tests' own skip condition still holds.
if [ -r "$ROOT/compass_harness/pyproject.toml" ]; then
    HARNESS=$(mktemp -d) || finish 96 "mktemp failed; nothing was run"
    trap 'rm -rf "$HARNESS"' EXIT
    # A copy, because setuptools writes build/ and *.egg-info into the source.
    cp -r "$ROOT/compass_harness" "$HARNESS/src" || finish 96 "cannot copy compass_harness; nothing was run"
    python -m pip install -q --no-deps --no-build-isolation --no-index --target "$HARNESS/site" \
        "$HARNESS/src" ${COMPASS_AGENTX_HARNESS:+"$COMPASS_AGENTX_HARNESS"} >&2 ||
        finish 96 "pip could not install compass_harness${COMPASS_AGENTX_HARNESS:+ and $COMPASS_AGENTX_HARNESS}; nothing was run"
    export PYTHONPATH=$PYTHONPATH:$HARNESS/site
    WHY=$(cd / && python -c 'from importlib import metadata; metadata.version("compass-harness")
import compass_harness, aiperf; print(aiperf.__file__, compass_harness.__file__)' 2>&1) || {
        WHY=${WHY##*$'\n'}
        finish 96 "the compass_harness tests would skip ($WHY); set COMPASS_AGENTX_HARNESS to an agentx-harness 56a0cf70 wheel, see scripts/compass/README.md; nothing was run"
    }
    printf 'harness: %s\n' "$WHY"
fi

# pytest's own status, captured directly. Piping it into `tail` discards the
# exit code and a failing suite reports success.
python -m pytest tests/ "${IGN[@]}" -q --no-header -p no:cacheprovider "$@" -rf
RC=$?
printf '\npytest: rc=%s\n' "$RC"

if [ "$RC" -ne 0 ]; then
    # Keep these nine lines last in this block, and keep the class name and
    # scripts/compass/README.md inside the final five. A caller who pipes loses
    # $? but not the text: `2>&1 | tail -6` keeps only the last five of them,
    # and `2>/dev/null | tail -6` drops them all but keeps pytest's own FAILED
    # line. Either way the reader still ends up holding the test's identity.
    # Appending to this block -- or inserting after the class name -- breaks
    # that, and nothing here will fail if it does.
    printf 'CPU tier of the test gate FAILED. The baseline is measured, not read:\n' >&2
    printf 'run this script on the integration head this branch forked from and\n' >&2
    printf 'compare -- see scripts/compass/README.md.\n' >&2
    printf 'Before you read it as your diff, check the FAILED line above. One test in\n' >&2
    printf 'ATOM'"'"'s own suite -- TestTheRegionIsNotCopiedPerChunk in\n' >&2
    printf 'tests/entrypoints/test_stream_marker_properties.py -- asserts a wall-clock\n' >&2
    printf 'timing property and fails intermittently on a loaded box. The\n' >&2
    printf 'scripts/compass/README.md section names it, what it was measured to do, and\n' >&2
    printf 'to run gates one at a time. It is not a Compass defect and is not excluded.\n' >&2
    finish "$RC" "pytest failed; the FAILED lines above name the tests"
fi

if [ -n "$GPU_SRC_UNKNOWN" ]; then
    printf 'The CPU tier is green, but the test gate is not passed: %s.\n' "$GPU_SRC_UNKNOWN" >&2
    printf 'So this run cannot tell whether the diff needs the GPU tier, and an\n' >&2
    printf 'unanswered question is not a "no". It is not a finding about the tree.\n' >&2
    printf '  Fix by any one of: stage with scripts/compass/snapshot.sh, which stamps\n' >&2
    printf '  .compass-changed; set COMPASS_CHANGED_FILES=<file listing the diff>; or\n' >&2
    printf '  run in a tree where %s resolves.\n' "$INTEGRATION" >&2
    finish 98 "unknown whether this diff needs the GPU tier: $GPU_SRC_UNKNOWN"
fi

# A green CPU tier is not a passed gate when the diff lands in the blind spot:
# the GPU tier has to have run on this same tree, and to have said so.
if [ -n "$GPU_NEEDED" ]; then
    DONE=${COMPASS_GPU_GATE_DONE:-}
    if [ -z "$DONE" ]; then
        printf 'this diff touches the CPU tier'"'"'s blind spot; the GPU tier has not run.\n' >&2
        printf '  Run gate_gpu.sh, then re-run with COMPASS_GPU_GATE_DONE=<that tree'"'"'s HEAD>\n' >&2
        finish 98 "this diff needs the GPU tier, which has not run; run gate_gpu.sh"
    fi
    # An attestation names a tree. Without a commit we cannot check it names
    # THIS tree, and accepting it unchecked would make the whole rule
    # bypassable by running in a snapshot -- which is where the gate normally
    # runs. Refuse instead: stamp the snapshot with snapshot.sh.
    if [ "$HEAD_SHA" = UNKNOWN ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s cannot be checked: this tree has no commit.\n' "$DONE" >&2
        printf '  No .git and no .compass-commit stamp, so the attestation could name\n' >&2
        printf '  any tree. Rebuild the snapshot with scripts/compass/snapshot.sh.\n' >&2
        finish 98 "COMPASS_GPU_GATE_DONE cannot be checked; this tree has no commit"
    fi
    if [ "$DONE" != "$HEAD_SHA" ]; then
        printf 'COMPASS_GPU_GATE_DONE=%s but this tree is %s.\n' "$DONE" "$HEAD_SHA" >&2
        printf '  The GPU tier passed on a different tree. Re-run it here.\n' >&2
        finish 98 "COMPASS_GPU_GATE_DONE names a different tree; run gate_gpu.sh here"
    fi
    printf 'gpu:    satisfied by COMPASS_GPU_GATE_DONE=%s\n' "${DONE:0:9}"
fi
finish 0
