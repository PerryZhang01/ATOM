#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Per-wave test gate: the driver-dependent superset of what gate_cpu.sh runs.
# Judged as a DELTA, not as green -- the baseline carries failures. A report
# that says "green" has not read it.
#
# The delta is judged on the failing node-IDS, not only on their count:
# gpu_gate_known_failures.txt beside this script names them verbatim. "5 failed"
# could never be checked against "the SAME 5 failed"; a list can.
#
# Also run per-task by any change gate_cpu.sh flags against
# gpu_gate_triggers.txt. That check is mechanical now; this comment is no longer
# the only place the rule lives.
#
# Run in the GPU container. preflight.sh is invoked here, before and after the
# run -- it used to be named in a comment and called by nobody.
#
# Five ways a green verdict was previously reachable on an unmeasured run, all
# of them measured, all of them now refusals:
#   - a summary carrying `N errors` was parsed for passed/failed only, so a run
#     with 12 collection errors read as PASS (reviewer, replayed against this
#     script);
#   - pytest's own exit status was captured and never consulted, so rc=2
#     (interrupted) or rc=4 (usage error) read as a measurement;
#   - a caller's `-rE` overrode this script's `-rf` -- pytest's `-r` is
#     store-last-wins -- so NO `FAILED ` lines were printed, the by-name check
#     compared against an empty observed set, and the gate reported all five
#     baseline failures as "no longer failing" and exited 0 while they had in
#     fact failed (EXECUTED on d78f3bbd3, run 2, gate_gpu_run2_rE.log);
#   - `GONE_IDS` only printed a note, so "the by-name check found nothing" and
#     "the failures were fixed" were one reading;
#   - the pass count was a floor, so the new tests every task adds loosened it
#     by exactly enough to absorb a silently skipped file.
#
# One way is NOT a refusal, and this header claimed otherwise until 2026-09-20
# ("EVERY question this gate cannot answer is a refusal"). A toolchain differing
# from the baseline's -- including an AITER version that reads UNKNOWN, which the
# toolchain comparison below explicitly calls a mismatch -- warns on stderr and
# can still end
# GATE_GPU_RC=0. Most known failures are AITER kernels, so on an
# AITER bump the delta compares two different things and exits 0 anyway. Making
# it a refusal is a behaviour change with its own cost -- a `git describe` that
# cannot read aiter's checkout would then block every run on a node where the
# stack is fine -- so the gap is stated here rather than closed unilaterally.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

# GATE_GPU_RC is printed exactly once, on stdout, on every path -- including the
# FATAL ones, which is why finish() is defined before the first exit rather than
# after pytest. Previously a compound regression (a count change AND a new
# node-id) printed it twice with no statement of which was the verdict, four
# FATAL paths printed it not at all, and the failure verdicts went to stderr
# while the success verdict went to stdout -- so `gate_gpu.sh 2>/dev/null | grep
# GATE_GPU_RC` was silent on failure and indistinguishable from "never ran".
#
# The verdict line carries its reason, for the reasons given above finish() in
# gate_cpu.sh: it is the only line every pipe keeps.
finish() {
    if [ "$1" -eq 0 ]; then
        printf 'GATE_GPU_RC=0 PASSED\n'
    else
        printf 'GATE_GPU_RC=%s NOT PASSED -- %s\n' "$1" "$2"
    fi
    exit "$1"
}
printf 'verdict: the last line of stdout; if this run is piped, $? is the pipe'"'"'s\n'

# The baseline is a pair AND the tree and toolchain it was measured on. A bare
# pair cannot distinguish a regression from drift: after a torch bump, 4728/7
# has no reading. Changing any of these seven constants is a reviewed edit.
#
# Measured 2026-10-13 on hjbog-srdc-20 in xiaobizh_n20_next (image
# rocm/atom-dev vllm-v0.31.0-nightly_20261011), HIP_VISIBLE_DEVICES=3, against a
# git archive of pure upstream 2479251d3, the ATOM that image ships: 10213
# passed, 119 failed, 0 errors, 485 skipped, 3 xfailed in 1026.7 s. A run on
# GPU 5 on 2026-10-12 failed the same 119 node-ids; its two extra skips were
# the timing class in tests/entrypoints/test_stream_marker_properties.py.
# Upstream has no tests/compass/, so every Compass test is surplus.
BASE_PASSED=10213
BASE_FAILED=119
BASE_COMMIT=2479251d3
BASE_TORCH=2.10.0+rocm7.2.4.git3d3aa833
# BASE_ROCM is compared against torch.version.hip below, so it carries the HIP
# runtime version, not the ROCm release. ROCm release was 7.2.4.
BASE_ROCM=7.2.53211
# AITER has no __version__; this is `git describe --tags --always` in the source
# checkout aiter imports from. Most known failures are AITER kernels or the
# gfx950-only formats they test, so AITER is the likeliest single cause of a
# change that moves them -- a bump used to pass the drift check in silence
# because only torch and HIP were compared.
BASE_AITER=v0.1.24-180-g4eecc6509
# How many of BASE_PASSED came from tests/compass/ at BASE_COMMIT. The pass count
# is checked for EQUALITY against BASE_PASSED adjusted by what tests/compass/
# contributes on this tree, so a surplus is accounted for rather than absorbed.
# Every Compass task adds tests there; a task that adds tests anywhere else makes
# this an explicit mismatch, which is the point.
#
# Both sides are PASS counts. This one was a pass count all along; the tree side
# used to be a COLLECTED count, so the first Compass test to skip or xfail on a
# GPU host would have made a legitimately green tree read `unaccounted -1` and
# blamed tests outside tests/compass/. compass_compass_pass_count in _lib.sh now
# counts passes on both sides, so the two are the same quantity by construction
# rather than while a condition happens to hold.
BASE_COMPASS_TESTS=0

# The baseline's failures by name. The count above and the list here are two
# statements of one fact, so a disagreement between them is a defect in this
# script rather than a test result -- refuse instead of picking one.
KNOWN_FILE=$(dirname -- "${BASH_SOURCE[0]}")/gpu_gate_known_failures.txt
known_ids() { grep -vE '^[[:space:]]*(#|$)' "$KNOWN_FILE"; }
[ -r "$KNOWN_FILE" ] || {
    printf 'FATAL: %s is missing; the baseline names no failures.\n' "$KNOWN_FILE" >&2
    finish 93 "the known-failures list is missing; nothing was run"
}
KNOWN_N=$(known_ids | grep -c .)
[ "$KNOWN_N" -eq "$BASE_FAILED" ] || {
    printf 'FATAL: BASE_FAILED=%s but %s names %s node-id(s).\n' \
        "$BASE_FAILED" "$KNOWN_FILE" "$KNOWN_N" >&2
    finish 93 "BASE_FAILED disagrees with the known-failures list; nothing was run"
}

# pytest's -r is store-last-wins, so a caller's -rE silently replaces the -rfE
# below and the by-name check loses its input. (EXECUTED: `gate_gpu.sh -rE` on
# d78f3bbd3 reported all five baseline failures "no longer failing" and exited
# GATE_GPU_RC=0 while they had failed.)
#
# What makes the verdict safe is the ORDERING at the pytest call: -rfE is placed
# after "$@", so it wins whatever the caller passed. The refusal below is UX,
# not the fix -- it tells the caller why their flag is being ignored. It is also
# incomplete: argparse accepts combined short options, and `-qrE` matches
# neither arm here. That is tolerable exactly because the ordering, not this
# loop, is what holds.
for arg in "$@"; do
    case "$arg" in
    -r | -r*)
        printf 'REFUSED: %s sets pytest -r, which this gate owns. Its verdict is\n' "$arg" >&2
        printf '  built from the FAILED and ERROR report lines, and -r is\n' >&2
        printf '  store-last-wins, so your flag would silently empty the observed\n' >&2
        printf '  failure set and the by-name check would compare nothing.\n' >&2
        finish 95 "refused a -r argument; nothing was run"
        ;;
    esac
done

ROOT=$(compass_tree_root) || finish $? "the tree to measure could not be resolved; nothing was run"
cd "$ROOT" || finish 90 "cannot cd into the tree; nothing was run"
compass_env "$ROOT"
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}

PRE=$(dirname -- "${BASH_SOURCE[0]}")/preflight.sh
printf '===== pre-flight (before) =====\n'
"$PRE" || { printf 'ABORT: pre-flight failed before the run.\n' >&2; finish 91 "pre-flight failed before the run; nothing was run"; }

compass_require_tree "$ROOT" || finish $? "atom does not import from this tree; nothing was run"
compass_describe "$ROOT"

TORCH=$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo UNKNOWN)
ROCM=$(python -c 'import torch; print(torch.version.hip)' 2>/dev/null || echo UNKNOWN)
# aiter's checkout is located without importing aiter: its import shells out to
# rocminfo, which hangs uninterruptibly on a wedged driver -- the box where the
# version is most worth recording. module_root resolves a top-level name with
# importlib.util.find_spec, which executes nothing, and its import chain loads
# neither aiter nor torch.
AITER_DIR=$(python -c 'from atom.compass.artifacts.provenance import module_root; print(module_root("aiter"))' 2>/dev/null)
if [ -n "$AITER_DIR" ]; then
    AITER=$(git -C "$AITER_DIR" describe --tags --always --dirty 2>/dev/null || echo UNKNOWN)
else
    AITER=UNKNOWN
fi
printf 'torch:  %s\nhip:    %s\naiter:  %s\n' "$TORCH" "$ROCM" "$AITER"
printf 'baseline: %s passed / %s failed / 0 errors at %s\n' \
    "$BASE_PASSED" "$BASE_FAILED" "$BASE_COMMIT"
printf '          torch %s, hip %s, aiter %s\n' "$BASE_TORCH" "$BASE_ROCM" "$BASE_AITER"
# UNKNOWN is a mismatch, not a match: an unreadable version is exactly the case
# where the pair below cannot be trusted.
if [ "$TORCH" != "$BASE_TORCH" ] || [ "$ROCM" != "$BASE_ROCM" ] || [ "$AITER" != "$BASE_AITER" ]; then
    printf 'WARNING: toolchain differs from the baseline'"'"'s. The known failures are\n' >&2
    printf '         mostly AITER kernel results that depend on exactly this\n' >&2
    printf '         stack; the pair below is not comparable until it is\n' >&2
    printf '         re-measured and the constants and %s updated.\n' "$(basename "$KNOWN_FILE")" >&2
fi

# The surplus this tree is allowed, derived from this tree rather than declared.
# An absent tests/compass/ counts zero and the gate still produces a verdict --
# on such a tree that is BASE_PASSED + 0 - BASE_COMPASS_TESTS, which is what a
# tree without this phase's tests measures. See compass_compass_pass_count in _lib.sh
# for why that is a count and not a refusal.
COMPASS_N=$(compass_compass_pass_count "$ROOT") || finish $? "the tests/compass pass count has no source; the suite was not run"
[ -d "$ROOT/tests/compass" ] ||
    printf 'tests/compass: absent on this tree, so it contributes no tests.\n'
EXPECT_PASSED=$((BASE_PASSED + COMPASS_N - BASE_COMPASS_TESTS))
printf 'expected: %s passed = %s baseline + (%s - %s) tests/compass\n' \
    "$EXPECT_PASSED" "$BASE_PASSED" "$COMPASS_N" "$BASE_COMPASS_TESTS"
printf '\n'

OUT=$(mktemp)
# -rfE, after "$@" so it wins even if a -r form slips past the refusal above.
python -m pytest tests/ --ignore=tests/plugin -q --no-header -p no:cacheprovider "$@" -rfE \
    2>&1 | tee "$OUT"
PYRC=${PIPESTATUS[0]}
SUMMARY=$(tail -20 "$OUT" | grep -E '[0-9]+ (passed|failed|error)' | tail -1)
PASSED=$(printf '%s' "$SUMMARY" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' || echo 0)
FAILED=$(printf '%s' "$SUMMARY" | grep -oE '[0-9]+ failed' | grep -oE '[0-9]+' || echo 0)
ERRORS=$(printf '%s' "$SUMMARY" | grep -oE '[0-9]+ errors?' | grep -oE '[0-9]+' || echo 0)
printf '\nGATE_GPU passed=%s failed=%s errors=%s pytest_rc=%s (baseline %s/%s/0, expected %s passed)\n' \
    "$PASSED" "$FAILED" "$ERRORS" "$PYRC" "$BASE_PASSED" "$BASE_FAILED" "$EXPECT_PASSED"

# A summary line that parses to 0/0/0, or no summary line at all, means the run
# did not get far enough to produce one -- a collection error, an OOM, a killed
# process. That is not a pass count below baseline; it is no measurement at all,
# and it must not be read through the delta rule below.
if [ -z "$SUMMARY" ] || { [ "$PASSED" -eq 0 ] && [ "$FAILED" -eq 0 ] && [ "$ERRORS" -eq 0 ]; }; then
    printf 'NO MEASUREMENT -- no usable pytest summary line found.\n' >&2
    tail -20 "$OUT" >&2
    rm -f "$OUT"
    finish 1 "no usable pytest summary line; the run is not a measurement"
fi

# pytest's own verdict, which the delta rule cannot replace: 0 = all passed,
# 1 = tests failed. Anything else (2 interrupted, 3 internal error, 4 usage
# error, 5 nothing collected, 128+n a signal) means the summary above describes
# a run that did not complete, whatever it says.
case "$PYRC" in
0 | 1) ;;
*)
    printf 'pytest rc=%s -- not "all passed" and not "tests failed", so the run\n' "$PYRC" >&2
    printf '  did not complete and the counts above are not a measurement.\n' >&2
    rm -f "$OUT"
    finish 1 "pytest rc=$PYRC; the run did not complete"
    ;;
esac

# Each finding goes to stderr in full. The first one's one-line form, and how
# many more there were, travel on the verdict line. The checks run with the
# node-id comparisons ahead of the counts they explain, so a new failure is
# named there by its node-id rather than as a changed count.
# A finding passed without a one-line form is named by its own first line, so
# a call that omits it still ends on a verdict rather than on `set -u`.
VERDICT=0
WHY=
fail() { printf '%s\n' "$1" >&2; VERDICT=$((VERDICT + 1)); [ -n "$WHY" ] || WHY=${2:-${1%%$'\n'*}}; }

# Errors are not failures and were never parsed: a run reading "4800 passed, 5
# failed, 12 errors" produced GATE_GPU_RC=0 without one word about the twelve.
[ "$ERRORS" -eq 0 ] ||
    fail "$(printf '%s collection/teardown error(s). The baseline has none, and an
  error is not a failure -- the tests behind it did not run at all, so the
  pass count below is over a smaller suite than the baseline'"'"'s.' "$ERRORS")" \
    "$ERRORS collection/teardown error(s)"

# The names, not only the counts: one baseline failure fixed and one new one
# introduced reads as 5/5.
OBS_IDS=$(mktemp)
ERR_IDS=$(mktemp)
KNOWN_IDS=$(mktemp)
grep '^FAILED ' "$OUT" | sed -e 's/^FAILED //' -e 's/ - .*$//' | sort -u >"$OBS_IDS"
grep '^ERROR ' "$OUT" | sed -e 's/^ERROR //' -e 's/ - .*$//' | sort -u >"$ERR_IDS"
known_ids | sort -u >"$KNOWN_IDS"
OBS_N=$(grep -c . "$OBS_IDS")
ERR_N=$(grep -c . "$ERR_IDS")

# The observed set has to be as big as the summary says, or the by-name check
# below is comparing against an accident. This is the check that makes the
# whole name comparison honest: an empty observed set is otherwise
# indistinguishable from "nothing failed".
[ "$OBS_N" -eq "$FAILED" ] ||
    fail "$(printf 'the summary says %s failed but %s FAILED line(s) were printed.
  The by-name comparison has no set to compare, so it is not evidence either
  way. (pytest -r is store-last-wins: a caller flag, a truncated log or a
  plugin can all empty it.)' "$FAILED" "$OBS_N")"
[ "$ERR_N" -eq "$ERRORS" ] ||
    fail "$(printf 'the summary says %s error(s) but %s ERROR line(s) were printed.' \
        "$ERRORS" "$ERR_N")"

NEW_IDS=$(comm -23 "$OBS_IDS" "$KNOWN_IDS")
GONE_IDS=$(comm -13 "$OBS_IDS" "$KNOWN_IDS")
if [ -n "$NEW_IDS" ]; then
    fail "$(printf 'failure(s) the baseline does not name:\n%s' \
        "$(printf '%s\n' "$NEW_IDS" | sed 's/^/  /')")" \
        "$(printf '%s\n' "$NEW_IDS" | grep -c .) failure(s) the baseline does not name, first ${NEW_IDS%%$'\n'*}"
fi
# Not good news by default, and the gate cannot tell the three causes apart:
# the failure was fixed; the test stopped running; or the observed set could not
# be read. Only the first is a pass, so this refuses and asks for the reviewed
# edit the known-failures file already requires.
if [ -n "$GONE_IDS" ]; then
    fail "$(printf 'baseline failure(s) not observed failing:\n%s\n  Confirm which of the three this is -- fixed, no longer collected, or an
  observed set that could not be read -- then drop them from %s.
  That edit is reviewed; it is not something this gate may assume.' \
        "$(printf '%s\n' "$GONE_IDS" | sed 's/^/  /')" "$(basename "$KNOWN_FILE")")" \
        "$(printf '%s\n' "$GONE_IDS" | grep -c .) baseline failure(s) not observed failing, first ${GONE_IDS%%$'\n'*}"
fi
# grep -c always prints a number, so an emptiness test here would be dead.
if [ "$ERR_N" -gt 0 ]; then
    printf 'erroring node-id(s):\n' >&2
    sed 's/^/  /' "$ERR_IDS" >&2
fi
rm -f "$OBS_IDS" "$ERR_IDS" "$KNOWN_IDS"

[ "$FAILED" -eq "$BASE_FAILED" ] ||
    fail "$(printf '%s failed, baseline %s.' "$FAILED" "$BASE_FAILED")"

# EQUALITY, not a floor. A floor is loosened by exactly the new tests every task
# adds, so a task that adds 30 tests and silently skips a 20-test file still
# clears it, and the only trace was a `note:`. The surplus is derived from this
# tree rather than declared by the caller, and the mismatch text below states it
# as its parts, since a single total cannot show which side moved.
[ "$PASSED" -eq "$EXPECT_PASSED" ] ||
    fail "$(printf '%s passed, expected %s. The difference decomposes as:
    baseline        %s   at %s
    tests/compass   %s now, %s at the baseline (delta %s)
    expected        %s
    observed        %s
    unaccounted     %s
  A surplus is not slack: it absorbs a file that stopped being collected. Both
  tests/compass figures are pass counts, so a skip or an xfail there is already
  accounted for; if this tree adds tests outside tests/compass/, account for
  them here.' \
        "$PASSED" "$EXPECT_PASSED" "$BASE_PASSED" "$BASE_COMMIT" \
        "$COMPASS_N" "$BASE_COMPASS_TESTS" "$((COMPASS_N - BASE_COMPASS_TESTS))" \
        "$EXPECT_PASSED" "$PASSED" "$((PASSED - EXPECT_PASSED))")" \
    "$PASSED passed, expected $EXPECT_PASSED"

rm -f "$OUT"

printf '\n===== pre-flight (after) =====\n'
"$PRE" || printf 'WARNING: pre-flight failed AFTER the run -- the node changed under it,\n         and the numbers above may not be attributable.\n' >&2

[ "$VERDICT" -le 1 ] || WHY="${WHY%.}; $((VERDICT - 1)) more finding(s) on stderr"
[ "$VERDICT" -eq 0 ] || finish 1 "$WHY"

# An attestation names a tree. On a tree with no commit, gate_cpu.sh refuses
# COMPASS_GPU_GATE_DONE=UNKNOWN -- so say that here rather than print an
# instruction that is guaranteed to fail when the caller follows it.
HEAD_SHA=$(compass_head_sha "$ROOT") || HEAD_SHA=UNKNOWN
if [ "$HEAD_SHA" = UNKNOWN ]; then
    printf 'satisfied for this tree, but it cannot be attested: no .git and no\n'
    printf '  .compass-commit stamp, so the attestation could name any tree and\n'
    printf '  gate_cpu.sh will refuse it. Rebuild with scripts/compass/snapshot.sh.\n'
else
    printf 'satisfied for this tree. gate_cpu.sh will accept:\n'
    printf '  export COMPASS_GPU_GATE_DONE=%s\n' "$HEAD_SHA"
fi
finish 0
