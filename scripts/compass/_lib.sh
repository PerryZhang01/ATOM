# SPDX-License-Identifier: MIT
# Shared helpers for the Compass gate and pre-flight scripts. Source, don't run.

# The checkout that contains $PWD, or nothing. A checkout is recognised by the
# file this comment is in, so the probe cannot disagree with the thing it is
# probing for.
compass_enclosing_tree() {
    local d=$PWD
    while [ -n "$d" ]; do
        [ -f "$d/scripts/compass/_lib.sh" ] && { printf '%s' "$d"; return 0; }
        d=${d%/*}
    done
    return 1
}

# The tree this invocation exercises: the repo root that owns *this copy* of the
# script. Derived from $BASH_SOURCE, so the copy in a worktree points at that
# worktree and cannot be made to point anywhere else.
#
# That settles which tree the script acts on. It leaves open the question it
# looks like it answers -- whether that is the tree the caller meant -- and the
# answer is no whenever the script is invoked by path from inside a different
# checkout.
#
# EXECUTED 2026-09-20: from /workspace/llm_infer_deploy_study/perf_modeling/ATOM
# (HEAD 83daf636d, feature/atomcompass_new), running the p0 worktree's copy
# `bash .../compass-worktrees/p0/scripts/compass/snapshot.sh <outdir>` wrote
# compass-d78f3bbd3.tar stamped `commit: d78f3bbd3...`, exit 0, no warning. The
# stamp was truthful about the tree it archived and silent about the tree the
# caller was standing in -- a wrong answer carrying the full confidence of a
# right one. Every caller of this function has the same exposure; for the
# gates the consequence is worse than for snapshot.sh, since the result of a
# pytest run would be attributed to a commit it did not come from.
#
# The fix is NOT to resolve from $PWD instead: that lets a script act on a tree
# that is not its own, which is the failure this project has already paid for
# twice. It is to refuse when the two disagree -- a result attributed to the
# wrong tree is worse than no result. Being run by path from outside any
# checkout is still allowed -- there the caller has named the tree and nothing
# contradicts it, and that is exactly how snapshot.sh's own extract
# instructions say to run a gate.
compass_tree_root() {
    local here src cwd
    here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd) || return 90
    src=$(cd -- "$here/../.." && pwd) || return 90
    cwd=$(compass_enclosing_tree) || { printf '%s' "$src"; return 0; }
    if [ "$(cd -- "$cwd" && pwd -P)" != "$(cd -- "$src" && pwd -P)" ]; then
        printf 'FATAL: this script belongs to %s\n' "$src" >&2
        printf '       but you are standing in    %s\n' "$cwd" >&2
        printf '       Two different checkouts. This script acts on its own tree and\n' >&2
        printf '       not on yours, so it would measure, archive or rewrite a tree\n' >&2
        printf '       you are not looking at and name it only in a path you did\n' >&2
        printf '       not read.\n' >&2
        printf '       Run %s/scripts/compass/%s instead.\n' "$cwd" "${0##*/}" >&2
        return 99
    fi
    printf '%s' "$src"
}

# The integration ref, resolved against this tree: the bare name first, then the
# same name qualified by each configured remote. A linked worktree and a fresh
# clone both carry fork/feature/atomcompass_new and no local branch of that
# name, so the bare default resolves nowhere and the comparison the caller
# wanted never happens. Prints the ref that resolved -- the caller compares that
# against what it asked for to report the fallback. When nothing resolves it
# prints git's own message for the bare form instead, so a refusal can quote git
# rather than paraphrase it into a claim about the history. That message cannot
# travel in a global: the caller reads this through a command substitution,
# which is a subshell, and an assignment made in there never reaches the caller.
compass_resolve_ref() {
    local root=$1 ref=$2 remote err
    err=$(git -C "$root" rev-parse --verify "$ref^{commit}" 2>&1 >/dev/null) &&
        { printf '%s' "$ref"; return 0; }
    # `git remote` prints alphabetically, which is an order and not a
    # preference: with two remotes both carrying the ref at different commits
    # the first by name wins, and that picks a base and hence a diff, not just a
    # name. The chosen remote is printed on the ref: and base: lines, which is
    # what makes the choice checkable rather than silent.
    for remote in $(git -C "$root" remote); do
        git -C "$root" rev-parse --verify --quiet "$remote/$ref^{commit}" >/dev/null 2>&1 &&
            { printf '%s' "$remote/$ref"; return 0; }
    done
    printf '%s' "$err"
    return 1
}

# Whether the ref that resolved is a local branch that has drifted from the
# remote branch of the same name. This reports; it does not redirect. The
# resolution order above is untouched and the base stays the ref that resolved.
#
# Bare-name-first is right when the local branch is what the caller means, and
# nothing told that apart from a local branch that is simply old. MEASURED
# 2026-09-22, in a worktree of /workspace/ATOM checked out at 83ef2a094 --
# which *is* fork/feature/atomcompass_new: with nothing set, the local
# feature/atomcompass_new at 1b473e5af won and .compass-changed listed the 5
# files of the landing in between; with COMPASS_INTEGRATION_REF naming the
# remote, 0 files. Both runs exited 0 and printed no ref: line, so the output
# of the wrong one was indistinguishable from the output of the right one.
#
# The counterpart is found by scanning the remotes for the same name rather
# than by reading branch.<name>.merge: that branch has no upstream configured
# (measured, same tree, `rev-parse feature/atomcompass_new@{upstream}` exits
# 128), and the same-name scan is exactly what the resolver above would have
# chosen had the local branch been absent -- so the comparison is against the
# ref that was passed over, which is the one the caller might have wanted.
#
# What that rule costs. A local branch whose configured upstream is a
# *differently* named remote branch is not compared against what it tracks.
# The class is exactly: branches for which `git config --get
# branch.<name>.merge` names something other than refs/heads/<name>. To
# enumerate it, walk `git for-each-ref --format='%(refname:short)' refs/heads`
# and compare that config value against each branch's own name. A member with
# no refs/remotes/*/<name> gets no counterpart, so this function returns 0 and
# prints nothing; a member that has one gets a line about a ref it does not
# track. Both halves are deliberate: the counterpart has to be the ref
# compass_resolve_ref would have picked, so announcing against
# branch.<name>.merge would name a ref the resolver would never use. The
# silent half is the case a future reader most needs, because it is where the
# tool goes back to being quiet.
#
# How many such branches exist is deliberately not recorded here. Membership
# turns over by ordinary development in minutes: `git push -u` creates the
# same-name remote ref *and* rewrites branch.<name>.merge to it, so publishing
# a branch moves it out of the class and into the level-and-silent early return
# below, while a landing on the integration branch changes the arithmetic of
# the ones that stay. A count in a comment is read without the timestamp that
# would make it checkable, and a reading taken here was already wrong within
# the hour. A dated census belongs in a change record, where it is met as a
# record; run the enumeration above for the reading that is true now.
#
# Announced, not refused, and the choice is not close. The caller named a ref
# that exists here and got that ref; nothing was guessed and no fallback was
# taken, so there is no declined answer to give. Refusing would also change
# what exit 92 means -- today it says "this ref names nothing here", which
# other agents key on and #82 was careful not to move -- and it would refuse
# the legitimate case of a deliberately older base, with no way back except an
# environment variable the caller would have to already suspect.
#
# Behind is the only direction that can corrupt the base: merge-base ignores
# commits the local branch has and HEAD does not, so a local branch that is
# *ahead* leaves the base where it was. It is still named, because when HEAD
# descends from those unlanded commits the base moves forward with them and the
# changed set shrinks -- and a changed set that is too small is how the GPU
# blind-spot question gets answered "no" without being asked.
#
# "diverged from (N ahead, M behind)" is also what *unrelated* histories print:
# rev-list --left-right --count returns the whole of each side when there is no
# merge-base, so an orphan counterpart reads as diverged. The counts are true
# either way and this line only reports them; the refusal that has to tell the
# two apart is snapshot.sh's merge-base one, which says "shares no commit with
# HEAD" on its first line and "these two histories are unrelated" on its
# second, and never says "diverged". That vocabulary is the refusal's own text
# on stderr, not a gloss on it, and tests/compass/test_snapshot_ref.py pins the
# word. So the distinction already exists one file over, in the place that has
# to make it, under a test -- which is the reason not to state it a second time
# here. It is also not split here because the orphan fixture is the only test
# that reaches this branch of the ladder, so a separate wording would leave the
# genuine diverged case with no test at all.
compass_ref_drift() {
    local root=$1 ref=$2 remote up= counts ahead behind how lsha rsha
    git -C "$root" show-ref --verify --quiet "refs/heads/$ref" || return 0
    for remote in $(git -C "$root" remote); do
        git -C "$root" show-ref --verify --quiet "refs/remotes/$remote/$ref" &&
            { up=$remote/$ref; break; }
    done
    [ -n "$up" ] || return 0
    counts=$(git -C "$root" rev-list --left-right --count "$ref...$up") || return 0
    ahead=${counts%%[!0-9]*}
    behind=${counts##*[!0-9]}
    if [ "$ahead" -eq 0 ] && [ "$behind" -eq 0 ]; then return 0; fi
    if [ "$behind" -eq 0 ]; then
        how="$ahead commit(s) ahead of"
    elif [ "$ahead" -eq 0 ]; then
        how="$behind commit(s) behind"
    else
        how="diverged from ($ahead ahead, $behind behind)"
    fi
    lsha=$(git -C "$root" rev-parse "$ref")
    rsha=$(git -C "$root" rev-parse "$up")
    printf 'ref:    %s (%s) is %s %s (%s)\n' \
        "$ref" "${lsha:0:9}" "$how" "$up" "${rsha:0:9}"
    printf '        the base is the local branch; COMPASS_INTEGRATION_REF=%s uses the remote\n' "$up"
}

# PYTHONPATH is set to the tree and nothing else. Inherited entries are not
# merged: an inherited entry ahead of ours is exactly how a take2 script silently
# ran another branch's engine, and an entry behind ours can still satisfy an
# import our tree does not have.
compass_env() {
    export PYTHONPATH=$1
    export AITER_LOG_LEVEL=${AITER_LOG_LEVEL:-WARNING}
}

# Assert that `atom` resolves under the tree being exercised, before trusting
# any result -- never after. The failure this catches has cost days: a prior run
# resolved `atom` to a non-git snapshot of a different branch, 72 files
# divergent, and failed silently wherever both trees defined the symbol.
#
# The probe runs from / on purpose. Every caller cd's to the tree first, and
# Python puts the script's directory -- here, the cwd -- at the head of
# sys.path, so a probe run in place resolves `atom` via cwd and passes whatever
# PYTHONPATH says. That is the one thing this function exists to check.
compass_require_tree() {
    local root=$1 resolved
    resolved=$(cd / && python -c 'import atom; print(atom.__file__)' 2>&1) || {
        printf 'FATAL: import atom failed:\n%s\n' "$resolved" >&2
        return 91
    }
    case "$resolved" in
    "$root"/*)
        printf 'tree:   %s\natom:   %s\n' "$root" "$resolved"
        ;;
    *)
        printf 'FATAL: atom resolved to %s\n       expected under %s\n       PYTHONPATH=%s\n' \
            "$resolved" "$root" "${PYTHONPATH:-<unset>}" >&2
        return 92
        ;;
    esac
}

# The commit this tree is, or UNKNOWN. Tried in order: git, then the
# .compass-commit stamp that snapshot.sh writes.
#
# A `git archive` snapshot has no .git, which is the point of using one -- but
# it also meant every gate run in a container reported `commit: UNKNOWN`, so
# results could not be attributed to a tree. The stamp is written by the same
# command that builds the archive, from the same rev-parse, so it cannot name a
# different commit than the one archived.
compass_head_sha() {
    local root=$1 sha
    sha=$(git -C "$root" rev-parse HEAD 2>/dev/null) && { printf '%s' "$sha"; return 0; }
    if [ -r "$root/.compass-commit" ]; then
        read -r sha <"$root/.compass-commit"
        case "$sha" in [0-9a-f]*) printf '%s' "$sha"; return 0 ;; esac
    fi
    printf 'UNKNOWN'
    return 1
}

# Name the commit in every artifact, so a result can never be attributed to a
# tree it did not come from. A dirty tree is reported, not refused -- gates run
# on uncommitted work by definition.
compass_describe() {
    local root=$1 sha src=stamp dirty
    git -C "$root" rev-parse HEAD >/dev/null 2>&1 && src=git
    if ! sha=$(compass_head_sha "$root"); then
        printf 'commit: UNKNOWN -- no .git and no .compass-commit stamp\n'
        return 0
    fi
    dirty=$(git -C "$root" status --porcelain 2>/dev/null | wc -l)
    printf 'commit: %s (%s)%s\n' "${sha:0:9}" "$src" \
        "$([ "$dirty" -gt 0 ] && echo " +${dirty} uncommitted")"
}

# How many tests under tests/compass/ pass on this tree. The GPU gate adds this
# to its baseline, so a tree that carries Compass tests is judged by an equality
# rather than by "no worse than".
#
# An ABSENT tests/compass/ is a count of zero, not a refusal. A tree without this
# phase's tests -- as the integration branch was until 4c16792d9 -- has none,
# and a gate that produces no verdict there cannot be used to show that a branch
# is gate-neutral. REPRODUCED 2026-09-21 on 4da2f3a2d and on the branch of PR #9:
# `pytest tests/compass --collect-only` exits 4 with `file or directory not
# found`, the count came back empty, and the gate refused with GATE_GPU_RC=93
# before running a single test. The two cases are told apart by asking the
# filesystem, not by parsing pytest's error text.
#
# A PASS count, not a collected count. The figure it is compared against is a
# pass count, and the two diverge the moment tests/compass/ holds a skip or an
# xfail -- which would read as a missing pass and fail a legitimately green tree.
# Counting passes costs one extra pytest invocation over a CPU-only directory
# and removes the condition entirely instead of documenting it. What it assumes
# instead is narrower: that these tests give the same result alone as inside the
# superset run.
#
# Anything else -- a failure, an error, an unparsable summary -- is a refusal.
# The surplus would have no source, and a delta judged against a surplus that
# has no source is not a measurement.
compass_compass_pass_count() {
    local root=$1 out rc n
    if [ ! -d "$root/tests/compass" ]; then
        printf '0'
        return 0
    fi
    out=$(cd "$root" && python -m pytest tests/compass -q --no-header -p no:cacheprovider 2>&1)
    rc=$?
    # rc 5 is pytest's "no tests collected": a directory that exists and holds
    # nothing runnable is a real zero, not an unreadable answer.
    if [ "$rc" -eq 5 ]; then
        printf '0'
        return 0
    fi
    n=$(printf '%s' "$out" | grep -oE '[0-9]+ passed' | tail -1 | grep -oE '^[0-9]+')
    if [ "$rc" -ne 0 ] || [ -z "$n" ]; then
        printf 'FATAL: could not count the tests that pass under %s/tests/compass.\n' "$root" >&2
        printf '       The pass surplus this tree is allowed would have no source, so\n' >&2
        printf '       the delta cannot be judged. pytest rc=%s, last lines:\n' "$rc" >&2
        printf '%s\n' "$out" | tail -5 >&2
        return 93
    fi
    printf '%s' "$n"
}

# A free TCP port on 127.0.0.1.
compass_free_port() {
    python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# compass_wait_up WHAT PID URL: return once URL answers; exit 1 if PID exits
# first or WAIT_S seconds pass.
compass_wait_up() {
    local deadline=$((SECONDS + WAIT_S)) me
    me=$(basename -- "$0" .sh)
    until curl -sf --max-time 10 "$3" >/dev/null; do
        kill -0 "$2" 2>/dev/null || { echo "$me: $1 exited before it answered" >&2; exit 1; }
        ((SECONDS < deadline)) || { echo "$me: no $1 after $WAIT_S s" >&2; exit 1; }
        sleep 1
    done
}

# compass_gpus hip|smi LIST: the comma-separated GPUs LIST names, as HIP's
# ordinals (`hip`, LIST in rocm-smi's indexes) or as rocm-smi's indexes (`smi`,
# LIST in HIP's ordinals), mapped as `atom/compass/drive/gpu.py::hip_visible`
# maps them. A GPU it cannot map fails.
compass_gpus() {
    rocm-smi --showhw | awk '/^[0-9]+ +[0-9]+ /' | sort -k2,2n | awk -v to="$1" -v list="$2" '
        NR > 1 && $2 == last { dup = 1 }
        { last = $2; hip[$1] = NR - 1; smi[NR - 1] = $1 }
        END {
            k = split(list, want, ",")
            for (i = 1; i <= k; i++) {
                if (dup || (to == "hip" ? !(want[i] in hip) : !(want[i] in smi))) exit 1
                out = out (i > 1 ? "," : "") (to == "hip" ? hip[want[i]] : smi[want[i]])
            }
            print out
        }' || { echo "compass_gpus: rocm-smi --showhw does not map every GPU of $2 to its own KFD node" >&2; return 1; }
}
