#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
#
# Regenerate gpu_gate_triggers.txt -- the source paths that no *running*
# CPU-gate test names. Run after any change to cpu_gate_exclude.txt; the two
# files are derived from the same tree and are wrong separately.
#
# THE RULE, in one sentence: an atom module named by an excluded test is a blind
# spot unless a CPU-gate test that actually runs names it too.
#
# Three decisions make that sentence operational. Decisions 1 and 3 were each
# forced by a measured counter-example, named below with its commit so the next
# reader does not have to rediscover it; decision 2 removed no path at 236abfd9a
# and is kept as a forward guard, which is also said below rather than left
# to be inferred from this sentence.
#
#   1. Imports are read at ANY indentation, not just column 0. The previous
#      version anchored on `^`, and at fada7424e 190 `import atom.*` lines in
#      non-plugin test files were indented -- function-local and guarded imports
#      were the dominant idiom. It was reading roughly a third of the import lines
#      it claimed to read.
#
#   2. A test whose file is not collected covers nothing, so coverage is
#      credited only from a CPU-gate file that `pytest --collect-only` shows
#      collecting at least one test. MEASURED at 236abfd9a in xiaobizh_n18_cpu,
#      2026-09-20: this probe removed NO path from that tree. Run whole with it,
#      and whole without it -- crediting every CPU-gate file, collecting or not
#      -- the result is 30 triggers both ways and the difference is empty. It
#      withheld 15 coverage paths; exactly one of them,
#      atom/model_ops/v4_kernels/state_writes.py, is even a candidate, and that
#      one sits under the candidate SUBTREE entry atom/model_ops/v4_kernels/,
#      so the collapse at the bottom of this script absorbs it whether its
#      coverage was credited or not. It is kept as a forward guard on trees that
#      one does not represent, NOT because anything here depends on it, and it
#      is not free: a full `pytest --collect-only` over the CPU gate, plus a
#      refusal path (exit 97). Decision 3, not this one, carries both of the
#      measured counter-examples.
#
#   3. The two sides are deliberately ASYMMETRIC, and the asymmetry is what
#      carries both counter-examples. Candidates come from any import in an
#      excluded test. Coverage is subtracted only for a MODULE-LEVEL import in a
#      CPU-gate file that collects at least one test -- the only imports this
#      tree can prove execute, since collection imports the module. An indented
#      import in a CPU-gate file is not credited, because text cannot tell
#      whether its enclosing test ran. Two paths turn on that, both measured at
#      236abfd9a in xiaobizh_n18_cpu:
#        - atom/model_ops/topK.py. Its only excluded-side reference is
#          `test_moe_dp_token_capacity.py:39`, indented under
#          `skipif(not torch.cuda.is_available())` and skipped in the CPU
#          container. Read coverage at any indentation and the set goes 30 -> 29,
#          losing exactly this path.
#        - atom/model_engine/model_runner.py. `tests/test_mla_index_cache.py`
#          imports ModelRunner at LINES 99-100, indented four spaces inside a
#          test function -- NOT at module level, whatever earlier drafts of this
#          comment said. The indentation is what leaves it uncredited; what the
#          file collects does not enter into it. Read coverage at any
#          indentation AND credit non-collecting files and the set goes 30 -> 27,
#          losing this path, atom/model_ops/attentions/aiter_mla.py and topK.py.
#
# Where this is still wrong, in both directions, so neither is a surprise:
#
#   - Toward FIRING: an indented import in a CPU-gate test that does run is not
#     credited, so its module can appear here although the CPU gate exercises it.
#     Coverage subtraction is also exact-string, so a candidate SUBTREE entry is
#     never cancelled by coverage of the files under it, and a candidate file is
#     never cancelled by coverage of its package.
#   - Toward SILENCE: imports are read as text, not resolved as a graph. A module
#     reached only transitively -- imported by a module a test imports -- is
#     invisible to both sides. A path absent from the output is NOT a claim that
#     the CPU gate covers it.
#
# So the output is neither a floor nor a ceiling. Of the two mistakes, this file
# prefers firing: running the GPU gate when in doubt is never the wrong one.
# Decision 3 is that preference, written down.
set -uo pipefail
. "$(dirname -- "${BASH_SOURCE[0]}")/_lib.sh"

ROOT=$(compass_tree_root) || exit $?
cd "$ROOT" || exit 90
compass_env "$ROOT"

EXCLUDE=$ROOT/scripts/compass/cpu_gate_exclude.txt
OUT=$ROOT/scripts/compass/gpu_gate_triggers.txt
[ -e "$EXCLUDE" ] || { printf 'REFUSED: %s missing\n' "$EXCLUDE" >&2; exit 94; }

D=$(mktemp -d)
trap 'rm -rf "$D"' EXIT

# --- the two test-file sets --------------------------------------------------
grep '^tests/' "$EXCLUDE" | sort >"$D/ex"
find tests -name 'test_*.py' -not -path 'tests/plugin/*' | sort >"$D/all"
comm -23 "$D/all" "$D/ex" >"$D/cpu"

N_EX=$(grep -c . "$D/ex")
N_CPU=$(grep -c . "$D/cpu")
[ "$N_EX" -gt 0 ] && [ "$N_CPU" -gt 0 ] || {
    printf 'REFUSED: ex=%s cpu=%s -- one side is empty, %s left untouched.\n' \
        "$N_EX" "$N_CPU" "$OUT" >&2
    exit 96
}

# --- which CPU-gate files actually collect a test ----------------------------
# Decision 2. Measured, not assumed: a file that collects nothing here is a file
# whose module-level imports we will not credit. Collection is also the weakest
# probe that answers the question, which is why it is collection and not a run --
# a module-level import executes at collection time whether its tests pass, fail
# or skip.
IGN=(--ignore=tests/plugin)
while IFS= read -r line || [ -n "$line" ]; do
    line=${line%$'\r'}; line=${line%%#*}
    line=${line#"${line%%[![:space:]]*}"}; line=${line%"${line##*[![:space:]]}"}
    [ -n "$line" ] || continue
    IGN+=(--ignore="$line")
done <"$EXCLUDE"

COLLECT=$(python -m pytest tests/ "${IGN[@]}" --collect-only -q -p no:cacheprovider 2>&1)
N_COLLECTED=$(printf '%s' "$COLLECT" | grep -oE '[0-9]+ tests? collected' | grep -oE '^[0-9]+')
[ -n "$N_COLLECTED" ] && [ "$N_COLLECTED" -gt 0 ] || {
    printf 'REFUSED: pytest --collect-only reported no test count.\n' >&2
    printf '         Without it, "the file collects nothing" cannot be told apart\n' >&2
    printf '         from "the probe did not run", and the two have opposite\n' >&2
    printf '         meanings here. %s left untouched. Last lines:\n' "$OUT" >&2
    printf '%s\n' "$COLLECT" | tail -5 >&2
    exit 97
}
printf '%s\n' "$COLLECT" | grep -oE '^tests/[^ ]*\.py::' | sed 's/::$//' | sort -u >"$D/live"
N_LIVE=$(grep -c . "$D/live")
N_DEAD=$((N_CPU - N_LIVE))

# --- import extraction -------------------------------------------------------
# `from a.b import c, d` yields a.b.c and a.b.d, not the bare package a.b. The
# previous version produced a.b, and `from atom.model_ops import eplb` therefore
# resolved to the whole atom/model_ops/ subtree, which then swallowed
# every sibling entry by the subtree rule at the bottom of this script.
refs() { # $1 = the import-keyword prefix regex. The difference between the two
         # call sites is entirely here, so get it wrong and the asymmetry that
         # decision 3 rests on silently disappears: an anchor written as
         # '^[[:space:]]*' is not module-level, it matches every indented import
         # there is, and the two sides collapse into one.
    local kw=$1
    xargs -r grep -hoE "${kw}[[:space:]]+atom\.[A-Za-z0-9_.]+([[:space:]]+import[[:space:]]+[A-Za-z0-9_,() ]+)?" 2>/dev/null |
        sed -E 's/^[[:space:]]+//' |
        awk '
        /^import[[:space:]]/ { print "mod\t" $2; next }
        /^from[[:space:]]/ {
            base = $2
            i = index($0, " import ")
            if (i == 0) { print "mod\t" base; next }
            syms = substr($0, i + 8)
            gsub(/[()]/, " ", syms)
            n = split(syms, a, /[, \t]+/)
            got = 0
            skip = 0
            for (j = 1; j <= n; j++) {
                if (a[j] == "") continue
                if (a[j] == "as") { skip = 1; continue }   # the next token is an alias
                if (skip) { skip = 0; continue }
                print "sym\t" base "." a[j]; got = 1
            }
            if (!got) print "mod\t" base
        }' | sort -u
}

# Resolve a dotted reference to a path in this tree. A `mod` reference names a
# module, so a package becomes the subtree. A `sym` reference names something
# inside its parent, so when the symbol is not itself a module the answer is the
# parent's own file -- atom/model_ops/__init__.py, never atom/model_ops/.
resolve() {
    local kind dotted p parent
    while IFS=$'\t' read -r kind dotted; do
        [ -n "$dotted" ] || continue
        p=${dotted//.//}
        if [ -f "$p.py" ]; then printf '%s.py\n' "$p"; continue; fi
        if [ -d "$p" ]; then printf '%s/\n' "$p"; continue; fi
        if [ "$kind" = sym ]; then
            parent=${p%/*}
            if [ -f "$parent.py" ]; then printf '%s.py\n' "$parent"; continue; fi
            if [ -f "$parent/__init__.py" ]; then printf '%s/__init__.py\n' "$parent"; continue; fi
        fi
        printf '%s\t%s\n' "$kind" "$dotted" >>"$D/unres"
    done | sort -u
}

: >"$D/unres"
refs "^[[:space:]]*(from|import)" <"$D/ex" | resolve >"$D/cand"          # candidates: any indentation
comm -12 "$D/cpu" "$D/live" >"$D/cpulive"      # CPU-gate files that collect a test
refs '^(from|import)' <"$D/cpulive" | resolve >"$D/covered" # coverage: module level, live files only

# An `atom.*` reference that resolves to nothing is not a warning to scroll past.
# It means either this parser mis-read a line, or a test imports a module that
# does not exist -- and the first of those silently shrinks both sets. Both are
# defects; neither is a result, so this refuses and leaves the output file alone.
N_UNRES=$(sort -u "$D/unres" | grep -c .)
[ "$N_UNRES" -eq 0 ] || {
    printf 'REFUSED: %s atom.* reference(s) resolve to no path in this tree:\n' "$N_UNRES" >&2
    sort -u "$D/unres" | sed 's/^/  /' >&2
    printf '         Either the import parser in refs() mis-read the line, or the\n' >&2
    printf '         test imports something this tree does not have. Until it is\n' >&2
    printf '         known which, the derived sets are incomplete by an unknown\n' >&2
    printf '         amount. %s left untouched.\n' "$OUT" >&2
    exit 98
}

comm -23 "$D/cand" "$D/covered" >"$D/final0"

# Drop anything already covered by a subtree entry above it.
: >"$D/final"
while read -r p; do
    covered=
    while read -r q; do
        case "$q" in */) [ "$p" != "$q" ] && case "$p" in "$q"*) covered=1 ;; esac ;; esac
    done <"$D/final0"
    [ -n "$covered" ] || printf '%s\n' "$p" >>"$D/final"
done <"$D/final0"

N=$(grep -c . "$D/final")
[ "$N" -gt 0 ] || {
    printf 'REFUSED: no trigger paths derived from %s excluded files. That is not\n' "$N_EX" >&2
    printf '         a clean bill of health, it is a broken derivation. %s\n' "$OUT" >&2
    printf '         left untouched.\n' >&2
    exit 97
}

SHA=$(compass_head_sha "$ROOT") || SHA=UNKNOWN
N_ALL=$(grep -c . "$D/all")
N_PLUGIN=$(find tests -name 'test_*.py' -path 'tests/plugin/*' | grep -c .)
N_CAND=$(grep -c . "$D/cand")
N_COV=$(grep -c . "$D/covered")
# The one count in the header that is about the tree's imports rather than its
# files. It was a hand-typed 190 inside a "do not hand-edit, including the
# counts" header until 2026-09-20; it is now counted from the tree.
N_IND=$(xargs -r grep -hcE '^[[:space:]]+(from|import)[[:space:]]+atom\.' <"$D/all" 2>/dev/null |
    awk '{ s += $1 } END { printf "%d", s + 0 }')

# The header is generated, counts and all. Every count in its `Measured at`
# block, and the indented-import figure, comes from this run over this tree; the
# template's other figures and line citations name the commit they were measured
# at. The previous file carried its counts by hand, and by hand they went stale:
# a header shipped crediting its counts to a commit at which tests/compass/ did
# not yet exist.
TMP=$OUT.tmp
{
    cat <<'HDR'
# Source paths whose only test coverage is in cpu_gate_exclude.txt.
#
# GENERATED by regen_gpu_gate_triggers.sh -- do not hand-edit, including the
# counts below. Re-run it after any change to cpu_gate_exclude.txt or to the
# imports in tests/. The two files are derived from the same tree and are wrong
# separately.
#
# A change touching the CPU gate's blind spot must also run the GPU gate. This
# file is that blind spot, stated as paths a script can match, so the rule is a
# gate rather than a sentence somebody remembers.
#
# THE RULE: an atom module named by an excluded test is a blind spot unless a
# CPU-gate test that actually runs names it too. Imports are read at any
# indentation on the excluded side -- @@IND@@ of this tree's `import atom.*`
# lines are indented -- but coverage is credited only for a module-level import
# in a CPU-gate file that collects at least one test, because those are the only
# imports that provably execute. At 236abfd9a the MODULE-LEVEL half carried both
# measured counter-examples: test_moe_dp_token_capacity.py:39 imports topK
# inside a test marked skipif(not torch.cuda.is_available()), and
# tests/test_mla_index_cache.py:99-100 imports ModelRunner indented inside a
# test function rather than at module level. Crediting either would delete a
# real trigger -- topK.py and atom/model_engine/model_runner.py respectively.
# The COLLECTS-A-TEST half removed no path at 236abfd9a: 30 triggers with it,
# 30 without, difference empty. It is kept as a forward guard. The full
# derivation, with that 2x2, is in the script's header comment.
#
# This list errs in BOTH directions, which is why it is not described as a floor:
#   - toward firing: an indented import in a CPU-gate test that does run is not
#     credited, so its module can appear here even though the CPU gate reaches it,
#     and coverage subtraction is exact-string, so a candidate subtree entry is
#     never cancelled by coverage of the files under it;
#   - toward silence: imports are read as text, so a module reached only
#     transitively is invisible. A path absent from this list is NOT a claim that
#     the CPU gate covers it.
# Of the two mistakes, this list prefers firing: running the GPU gate when in
# doubt is never the wrong one.
#
# A trailing / matches the whole subtree.
HDR
    printf '#\n# Measured at %s over this tree:\n' "$SHA"
    printf '#   tests/       %4s files  = %s in tests/plugin/ + %s excluded + %s CPU gate\n' \
        "$((N_ALL + N_PLUGIN))" "$N_PLUGIN" "$N_EX" "$N_CPU"
    printf '#   CPU gate     %4s files collect >=1 test, %s collect none\n' "$N_LIVE" "$N_DEAD"
    printf '#   imports      %4s indented `import atom.*` lines in non-plugin test files\n' "$N_IND"
    printf '#   collected  @@N@@ tests by --collect-only. A run of the same set reports\n'
    printf '#              the same @@M@@ outcomes. Its "skipped" total is LARGER than any\n'
    printf '#              skip count here, because a module-level skip in one of the %s\n' "$N_DEAD"
    printf '#              files that collect nothing is counted as skipped without ever\n'
    printf '#              becoming a collected test. This file states the set; gate_cpu.sh\n'
    printf '#              prints the run, and is the only source for its outcome counts.\n'
    printf '#   candidates   %4s paths named by an excluded test, any indentation\n' "$N_CAND"
    printf '#   covered      %4s paths named at module level by a live CPU-gate file\n' "$N_COV"
    printf '#   triggers     %4s paths below\n' "$N"
    cat "$D/final"
} >"$TMP"
mv -f "$TMP" "$OUT"

# The collected-test count is measured AGAIN, after the rewrite, because this
# file is part of the suite it is measured against: tests/compass/
# test_cpu_gate_exclude.py parametrises test_every_trigger_path_still_exists
# over the paths below, one case each. So regenerating moves the CPU gate's
# pass count by exactly the change in the number of entries -- 13 -> 30 at
# f3f584ea4 -- and a count taken before the rewrite describes the previous
# list, not this one. That is not a quirk to work around; it is why the number
# is re-taken rather than reused.
COLLECT2=$(python -m pytest tests/ "${IGN[@]}" --collect-only -q -p no:cacheprovider 2>&1)
N_COLLECTED2=$(printf '%s' "$COLLECT2" | grep -oE '[0-9]+ tests? collected' | grep -oE '^[0-9]+')
[ -n "$N_COLLECTED2" ] && [ "$N_COLLECTED2" -gt 0 ] || {
    printf 'REFUSED: %s was rewritten, but the re-measure of the collected count\n' "$OUT" >&2
    printf '         failed, so the header cannot state the suite this list now\n' >&2
    printf '         produces. Re-run; the file on disk is the new list with an\n' >&2
    printf '         unfilled @@N@@/@@M@@ placeholder.\n' >&2
    exit 97
}
sed -i "s/@@N@@/$(printf '%5s' "$N_COLLECTED2")/; s/@@M@@/$N_COLLECTED2/; s/@@IND@@/$N_IND/" "$OUT"
printf 'wrote %s: %s trigger paths (%s cpu-gate files, %s live, %s excluded) at %s\nCPU gate now collects %s tests, was %s before this rewrite\n' \
    "$OUT" "$N" "$N_CPU" "$N_LIVE" "$N_EX" "$SHA" "$N_COLLECTED2" "$N_COLLECTED"
