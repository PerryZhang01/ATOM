---
repo: jgong5/ATOM
integration_branch: feature/atomcompass_new
gate_task: scripts/compass/gate_cpu.sh
gate_wave: scripts/compass/gate_gpu.sh
design_entry: atom/compass/design/README.md
never_touch: ROCm/ATOM
new_tests_dir: tests/compass
import_check: python3 -c "import atom; print(atom.__file__)"
---
**This environment.** Each owner runs from their own environment and keeps a
git-ignored `.claude/agent-team.local.md` in the main worktree
(`$(git rev-parse --path-format=absolute --git-common-dir)/../.claude/`, not
the task worktree) saying how it meets the requirements below: where git,
`gh` and `python3` run, path mappings, file ownership, the CPU gate container
and how to stage into it, GPU node access. Read it before anything else. If
it is missing, stop and ask the owner to write it from this file. Scratch and
working logs go in `agent_scratch/`, beside the repository.

## Several owners

Several owners drive agents on this repository at once, each through their
own `gh` login. Ownership is the assignee:

- **Scope.** A `run` acts only on issues and PRs assigned to its own `gh`
  login: claim, review, develop and land all stay inside that set. Every
  other open issue and PR belongs to someone else; `status` may report
  them, nothing else touches them. One exception: in a stack that holds
  another owner's PR, linking, unstacking and base PATCHes may name it,
  since they land and push nothing.
- **Assignee on creation.** Every issue and PR gets an assignee when it is
  created: an issue its creator's login, a PR the assignee of the issue it
  delivers (its creator's login when it delivers none). With `gh issue
  create`, pass `--assignee @me`. Over REST, `@me` is not accepted: read
  the login with `gh api user --jq .login` and pass `-f
  "assignees[]=<login>"` to `POST repos/<repo>/issues`; `POST
  repos/<repo>/pulls` takes no assignee, so follow it with `POST
  repos/<repo>/issues/<pr>/assignees`. An issue or PR with no assignee is
  in no run's scope: `status` lists it, and an owner assigns it.
- **Reassignment** replaces the assignee, never adds one, and moves an
  issue and its open PRs together, so nothing is in two runs' scope. It takes effect at the next pass: the old
  owner's run stops touching them, the new owner's run takes them over
  (recreating the worktree from the PR branch). A push rejected during the overlap is
  fetched and merged, never forced.
- **`need human`** is removed by the assignee of the issue or PR that
  carries it, except an escalation
  about a design decision (a D-numbered decision, a design document, a new
  hook point in ATOM): only @jgong5 removes that one.

## Environment requirements

- Git, `gh`, `python3`, `gate_wave` and the `agent-team` scripts run where
  the local file says, with the project's toolchain; a stray host Python
  that happens to work tells you nothing.
- A `gh` login with write access to `repo`, recent enough that `gh pr view`
  accepts `closingIssuesReferences` (`pr_state.py` asks for it on every PR;
  2.45.0 rejects it), and the `gh stack` extension.
- Files a git command writes are owned by the user who edits them; if git
  runs as another user (root in a container), chown the worktree after every
  git command that writes files (worktree add, fetch into a branch, merge,
  checkout) and check nothing is left.
- **`gate_task` runs in a CPU environment with no GPU driver**, on a
  `scripts/compass/snapshot.sh` tarball of the head
  (`scripts/compass/README.md`), and is judged against the same run on the
  base. Not where a driver is present: kernel tests in the CPU gate run on
  the device there and fail at the base too. New tests in `new_tests_dir`
  are CPU-only: they pass there with no driver.
- **Python resolution.** Run `python3` and `pytest` from the tree's root, or
  with `PYTHONPATH` set to it, and run `import_check` before trusting a
  result: an environment with another ATOM checkout resolves `atom` there
  from anywhere else. A container that runs a tree mounts the worktree
  root's parent, not the tree alone; mounting one tree makes `pytest` fail
  with a path error.
- **The local integration branch.** The Compass scripts diff against the
  local `feature/atomcompass_new` when it exists, before
  `fork/feature/atomcompass_new`, so a stale local branch gives them a wrong
  base unless `COMPASS_INTEGRATION_REF` names the ref. Other owners land
  without touching your clone, so fetch and fast-forward the local branch,
  or set `COMPASS_INTEGRATION_REF=fork/feature/atomcompass_new`, before
  staging any gate.

## Project rules

- **Few ATOM hook points.** Hook Compass into ATOM at the fewest points (a
  factory, a constructor, or a bootstrap patch applied only when Compass is
  on), never with per-use-site wrappers in ATOM's engine or frontend files.
  Compass-off behaviour stays byte-identical. A task touching ATOM engine
  code compares the hook options under Decided in its Dev record before
  building.
- **Production reachability.** Code a PR adds or changes outside a `tests/`
  directory must run from an entry point under some production
  configuration. For `atom/` the entry points are ATOM's serving and
  offline engine startup, Compass on or off; elsewhere, the invocation a
  README, this file or CI documents. Code only tests reach, including code
  a test installs by monkeypatch or assignment (as `wrap_encode` /
  `wrap_decode` in `atom/utils/compass_loop.py` were), is a violation. One
  exception: a building-block PR may carry code its own head does not reach
  if it is stacked under a PR that wires that code into an entry point's
  path. It is sent to review only once the wiring PR is approved, and the
  wiring PR lands right after it, once a delta review covers its rebased
  head.
  A reviewer traces each added or changed non-test symbol at the head to a
  caller outside `tests/` and on up to an entry point; a violation is a
  required finding and blocks APPROVE until the PR wires or deletes the
  code.
- **Labels.** An issue or PR gets its `module: compass-*` and `dev-process`
  labels from the `compass-labels` skill
  (`.claude/skills/compass-labels/SKILL.md`): run
  `compass_labels.py pr <N> --apply` when it is created, and again whenever a
  PR round adds files the opening round did not touch.
- **Inline comments.** `pr_state.py` and the issue-comments endpoint do not
  show inline comments (`pulls/<n>/comments`), and owners review with them.
  Read both endpoints before acting on a PR, and answer every inline
  comment in the next round, whoever posted it. A search over a PR that
  found nothing counts only if it covered both. On REST, merge state is
  `.merged` / `.merged_at`, not `.mergedAt`.
- **`gh` and the fork's parent.** Commands that look up the fork's parent
  (`pr create`, `issue create`, `pr ready`, `stack`) can fail with a ROCm
  SAML error. Never follow the authorization link it prints: use REST on
  `repo`, and prefix `gh stack` with `GH_REPO=<repo>`. Pass file bodies as
  `-F body=@<file>`; `-f` posts the literal path and still succeeds.
  `land.py` cannot land a draft PR: open PRs ready for review, or clear
  draft with the GraphQL `markPullRequestReadyForReview` mutation.
- **Re-basing a stack.** A PR in a `gh stack` refuses a base PATCH with 422.
  Run `gh stack unstack`, PATCH each base, then `gh stack link --base
  <integration_branch>` the whole chain again.
- **Gating a stacked PR.** Set `COMPASS_INTEGRATION_REF` to the parent PR's
  head, so `.compass-changed` lists only this PR's diff. Otherwise a
  GPU-trigger path changed lower in the stack makes `gate_cpu.sh` exit 98
  for every PR above it; only the PR that changes that path needs
  `gate_gpu.sh`.
- **Timing noise.** `scripts/compass/README.md` ("A red CPU gate that may
  not be your diff") covers one timing class in
  `tests/entrypoints/test_stream_marker_properties.py`;
  `TestNoSizeAtWhichACallStopsBeingOne::test_a_long_call_costs_what_it_is_and_not_its_square`
  in the same file also times wall clock and is read the same way.
- **Bound every gate.** Run gates under `timeout`. `gate_cpu.sh` prints
  `GATE_CPU_RC=` on every exit, pass or fail; a run that shows a pytest
  summary and no such line hung.
