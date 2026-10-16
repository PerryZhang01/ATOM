# SPDX-License-Identifier: MIT
"""The per-process bootstrap of a simulated run: which LP a process is, and its clock.

A run is on when `ENV` names a run file, a JSON object every process of the
deployment reads; spawned processes inherit it. The API server's
``--compass-run`` sets it. With it unset, every function here returns at once
and ATOM runs as it always does.

One engine LP at one stage, one frontend LP, and the traffic LP the harness
drives, over `single_engine_table`. At data-parallel width above one, each DP
rank's engine process is a member ``dp<rank>`` of the one engine LP and owns
its rank's channels. Each pipeline stage's engine process is an engine LP of
its own, ``engine-pp<stage>``, and each stage's forward waits for the stage
before it on ``pp_data`` (`atom.compass.runner.pp_send`). The Clock Authority is co-hosted in the frontend's
process and served at the run file's ``clock_endpoint``, which every LP and
member, the frontend's included, connects to.

- `frontend(config)` wraps the `CoreManager` construction in `LLMEngine`: it
  selects the simulated runner, serves the authority and installs the
  frontend LP's runtime; leaving it, after READY from every engine, starts
  the run. It refuses an `LLMEngine` the API server is not building and the
  Atomesh entrypoints, and makes a sync utility command on the frontend's
  clock owner refuse. On a worker node of a multi-node DP group it only
  selects the simulated runner.
- `engine(config)` wraps the engine's construction in its own process the
  same way, and maps every DP rank to the engine LP for the process-group
  check.
- `runner(runner, tree)` installs the cost backend and the device readings on a
  `CompassModelRunner` where it would build its model, in the worker process.
- `tokenizer(tokenizer, config)` charges the served tokenizer's ``encode`` and
  ``decode`` to the frontend's clock, at the rates the machine spec measured.
- `http_intake()` is what `atom.utils.compass_loop.HttpChannel` charges the
  frontend's event loop for each request it takes in.
- `engine_done(engine)` and `frontend_done(engine)` close each side after the
  finish, each handing in its LP's refusals; the frontend's, when it co-hosts
  the authority, then writes the step table and the run summary. A SIGINT or
  SIGTERM that stopped the frontend's event loop
  (`atom.utils.compass_loop.CompassEventLoop.stopped`) is a stop: the
  frontend's runtime first ends every LP's run through the authority, so each
  engine leaves its loop at its ``+inf`` grant and hands in, and a client
  still bound gets ``+inf`` too, each grant releasing every message sent its
  LP; `frontend_done` hands in the signal, which the summary counts as an
  ``ended:`` reason, and the server leaves once the summary is written from
  every hand-in. A server whose ``client_runs`` is ``null`` serves until it is
  stopped: a stop with no client run open is its run's end, summarised as the
  finish would be.
- `engine_loop(engine)` wraps the engine's loop: under ``step_grid`` it asks the
  workers for the bundle before it, `engine_done` after it, and if either
  raises, the engine hands in the exception as an ``ended:`` reason, with the
  bundle; `engine_done` raises for a loop that left before the finish. The
  engine ignores SIGTERM until it has handed in. A
  hand-in before the finish is an early end: the authority's host writes the
  step table and the run summary at once, each LP's clock as it stands, from
  what has been handed in.

Run file keys: ``clock_endpoint``, ``bound_s`` (finite), ``admission_path``,
``ipc_s``, ``stream_s``, one of ``step_stub`` (a `Coefficients` mapping,
priced by `StepStubBackend`), ``step_linear`` (a `StepLinearLaw` mapping,
priced by `StepLinearBackend`; its ``machine_spec`` digest and, when stated,
its ``engine_config``, `step_grid.keyed` as it was fitted, refuse a run under
any other), ``step_grid`` (an artifact store's root and a
bundle's key, priced by `StepGridBackend`; see `step_grid`) and ``replay`` (a
real run's parity record directory, or for a prefill-decode run one holding
each side's as ``prefill/`` and ``decode/``: each step ends when the real one
did, by `atom.compass.parity.replay`),
``step_roofline`` (optional, ``true`` or ``false``: with ``true`` a
`StepRooflineBackend` prices the steps the backend refuses, through
`StepFallbackBackend`),
``on_refusal`` (optional: ``mark``, the default, prices a step no cost source
answers at 0 s, hands in its ``cost:`` reason and continues; ``abort`` ends
the run there, which still writes the summary),
``client_runs`` (optional: how many client runs the server serves, one traffic
LP after another, 1 when absent and ``null`` for no limit; each ends when no
essential work is left, and the last one's end finishes the run),
``host`` (optional: ``laws``, per forward kind and host segment, coefficients
on ``(1, tokens, seqs)``, refused where absent; ``key``, the `host_key` they were
fitted under and the ``machine_spec`` digest of the spec of the platform they
were fitted on, which a run with laws must match; ``skip``, segments zeroed for
a what-if run, which the graders refuse),
``machine`` (a machine spec mapping), ``parameter_count`` (read with
``step_roofline``), ``store`` (the artifact store's root, where `memory` reads
the configuration's memory readings) and ``backend`` (the backend's ``digest``
and its fit ``provenance``, which nothing here reads), both written by
`atom.compass.drive.runfile`, and ``out_dir``,
where the step table, the run summary and each server LP's hand-in go, one
file written as the LP leaves its loop. It holds the LP's refusals: an
engine's are the worker's refused commands and the engine's refused clock
calls, a frontend's its executor's refused jobs and its refused clock calls.
An engine's also holds what its worker's steps cost, for the summary's
predicted seconds and their shares by method (`backends.cost.GROUPS`).
It holds the memory the worker predicts too, for
``python -m atom.compass.memory.check``, and under ``step_grid`` the bundle's
key, digest and ``describe()`` line, which the summary carries. The authority
removes an earlier run's files there as it starts. At the end of client run
``k`` every server LP hands in to ``client-<k>/`` there, and the authority's
host writes the step table and that client run's summary, its steps taken as
each hand-in less the one at client run ``k - 1``'s end; the finish writes the
last client run's beside the run summary. An optional
``data_parallel_size``, 1 when absent, is the deployment's DP width, which
under DP-attention is its TP size times its DP size, and an optional
``pipeline_parallel_size``, 1 when absent, its pipeline width, above which the
machine states each stage send carrier under ``interconnect.intra_node.pp``; the traffic LP reads the
channel table from the run file alone. Each DP rank's engine process hands in
its own refusals, as ``<lp>.<member>``. A key `RENAMED_KEYS` lists refuses,
naming its name now.

A run file that also has ``router_s`` and ``kv_write_req_s`` describes a
prefill-decode run over `prefill_decode_table`, and ``kv_link`` (``intra_node``
or ``inter_node``) names the link its KV transfers are priced on. An optional
``router_in_s`` makes the router the ``router`` LP, with that lookahead on each
hop into it; it prices no work, so it is no server LP and hands nothing in.
Its ``router`` entry, `router_binary`'s record of the binary that ran it, is
per-run and goes to the summary as it is. A deployment
whose ``kv_connector`` is ``compass`` is its prefill (``kv_producer``) or
decode side: `engine(config)` makes it the ``engine-P`` or ``engine-D`` LP and
binds the connector's transfer model. The write-request address comes with the
deployment's own ``kv_transfer_config``. Above DP width one each DP rank is a
member of its side's engine LP, and ATOM's DP dispatch on each side picks the
rank a request runs on. Optional ``prefill_instances`` and ``decode_instances``,
1 when absent, run that many deployments of a side, each naming its index in
its ``kv_transfer_config`` as ``compass_instance``; above one, its LPs carry
the index (``engine-P1``), and the router is the ``router`` LP, which chooses
among them. `frontend(config)` refuses such a
deployment unless the authority is standalone: a prefill-decode run spans two
API servers that must share one.

The API server's ``--compass-clock-endpoint`` sets `CLOCK_ENV`: the authority
is not co-hosted but served by `authority`, its own process started before the
engines (``python -m atom.compass.run``), and every LP connects to that
endpoint. Its frontend is ``frontend-P`` or ``frontend-D`` by ``kv_role``. The
standalone authority writes the step table and the run summary once every
server LP has handed in its refusals, which the out_dir both API servers share
carries.

A multi-node DP group runs on a standalone authority too, named on every node.
The coordinator's API server, global DP rank 0's, is the one frontend LP; a
worker node's runs no frontend LP and serves no authority, and each of its
engine processes joins the engine LP over TCP as the member of its global rank.
Hand-ins go to the out_dir, which every node must see at one path: the
authority writes `AUTHORITY_FILE` there before it serves, and a node that does
not find it naming its endpoint's port is refused.
"""

import ast
import contextlib
import functools
import hashlib
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np

import atom
from atom.compass.artifacts import provenance
from atom.compass.clock import (
    TAR,
    ClockAbort,
    ClockAuthority,
    LpId,
    RefusalTally,
    RunSummary,
    prefill_decode_table,
    single_engine_table,
)
from atom.compass.clock_transport import connect, serve
from atom.compass.detect.determinism import StepTable
from atom.compass.runner import COMPASS_RUNNER_QUALNAME
from atom.utils import clock

ENV = "ATOM_COMPASS_RUN"
CLOCK_ENV = "ATOM_COMPASS_CLOCK_ENDPOINT"
#: The file an engine LP's clock is mapped from, for its model runner workers.
NOW_ENV = "ATOM_COMPASS_NOW"
FRONTEND, ENGINE, TRAFFIC = LpId("frontend"), LpId("engine"), LpId("traffic")
ROUTER = LpId("router")
#: Run-file keys for how many deployments of each prefill-decode side run.
INSTANCES = {"P": "prefill_instances", "D": "decode_instances"}
#: The kv_transfer_config key a deployment names its index among them by.
INSTANCE_KEY = "compass_instance"
ATOM_RUNNER = "atom.model_engine.model_runner.ModelRunner"
#: Keys that differ between two runs of one configuration, kept out of its name.
#: The store is a place: the memory readings a run reads from it are keyed by
#: the configuration and the machine spec, which the name holds.
PER_RUN = ("clock_endpoint", "out_dir", "client_runs", "router", "store")
LP_FILE, STEP_TABLE_FILE, SUMMARY_FILE = (
    "lp-{}.json",  # by LP name, or <lp>.<member> for a member
    "step_table.txt",
    "summary.json",
)
#: Each client run's hand-ins and summary, by its index.
CLIENT_DIR = "client-{}"
#: The endpoint a standalone authority serves at, written to its out_dir.
AUTHORITY_FILE = "authority_endpoint.txt"
#: Wall seconds the summary's writer waits for the server LPs' refusals.
HAND_IN_WAIT_S = 300.0
#: The format of the run summary; raise it when a field is added, changes
#: meaning or goes.
SUMMARY_SCHEMA_VERSION = 5
#: Compass source the host-law key leaves out, by path under the package: all
#: of it (None), or the top-level names (def, class or assigned) only a simulated
#: run reaches, with the comments above each and the module docstring.
COMPASS_ONLY = {
    "compass": None,
    "utils/compass_loop.py": None,
    "utils/clock.py": {
        *("DIAG_S", "job", "current_job_time", "install", "Straggler", "_Wakeup"),
        *("LPRuntime", "UnsentRelayItem", "WrappedSocket", "WrappedPoller"),
        *("RelayQueue", "channel_of", "_of_kind", "socket_channels"),
        *("REPLY_WAIT_S", "_WaitingReplyQueue"),
        *("COLLECTIVES", "_on_host", "settle_before_collectives"),
    },
    "utils/zmq_shim.py": {"_Socket", "_ChannelSocket", "_Context"},
}
#: The run-file key of each backend a flag names (`atom.compass.drive.runfile`).
STEP_STUB, STEP_LINEAR, STEP_GRID = "step_stub", "step_linear", "step_grid"
#: Run-file keys that were renamed, to their names now.
RENAMED_KEYS = {
    "coefficients": STEP_STUB,
    "law": STEP_LINEAR,
    "step_fit": STEP_GRID,
    "roofline_floor_s": "step_roofline",
}


def refuse_renamed_keys(run: dict, where) -> None:
    """Refuse a run file that holds a renamed key, naming the key's name now."""
    if old := [k for k in RENAMED_KEYS if k in run]:
        named = ", ".join(f"{k} (now {RENAMED_KEYS[k]})" for k in old)
        raise ValueError(f"{where} holds run-file keys that were renamed: {named}")


_authority: "_RecordingAuthority | None" = None
#: The co-hosted authority's early-end writer.
_early_end: threading.Thread | None = None
#: This engine process's member name in the engine LP, None at DP width one.
_member: str | None = None


def spec() -> dict | None:
    """The run file's mapping, or None on a real run."""
    path = os.environ.get(ENV)
    if not path:
        return None
    run = json.loads(Path(path).read_text())
    bound = run.get("bound_s")
    if not isinstance(bound, (int, float)) or not math.isfinite(bound):
        raise ValueError(
            f"{path}: bound_s is {bound!r}; a run needs a finite simulated-time "
            "bound, or a timer nobody declared daemon keeps it alive forever"
        )
    if run.get("on_refusal", "mark") not in ("mark", "abort"):
        raise ValueError(
            f"{path}: on_refusal is {run['on_refusal']!r}; a step no cost source "
            "answers is marked and the run continues (mark), or ends it (abort)"
        )
    count = run.get("client_runs", 1)
    if count is not None and (type(count) is not int or count < 1):
        raise ValueError(
            f"{path}: client_runs is {count!r}; a server serves a positive number "
            "of client runs, or null for no limit"
        )
    _check_host(path, run.get("host", {}))
    run["clock_endpoint"] = os.environ.get(CLOCK_ENV, run.get("clock_endpoint"))
    return run


def _check_host(path, host: dict) -> None:
    """Refuse a ``host`` entry the runner would read as zero host time."""
    from atom.compass.runner.overrides import FORWARD_KINDS, HOST_SEGMENTS, MTP_SEGMENT

    laws, segments = host.get("laws", {}), (*HOST_SEGMENTS, MTP_SEGMENT)
    if (
        set(host) - {"laws", "skip", "key"}
        or set(host.get("skip", ())) - set(segments)
        or set(laws) - set(FORWARD_KINDS)
        or {s for v in laws.values() for s in v} - set(segments)
        or {len(c) for v in laws.values() for c in v.values()} - {3}
    ):
        raise ValueError(
            f"{path}: host is {host!r}; it takes skip, a list of {segments}, "
            f"laws, mapping {FORWARD_KINDS} to segments to three coefficients, "
            "and key, the host key the laws were fitted under"
        )


def _dp(run: dict) -> int:
    return run.get("data_parallel_size", 1)


def _pp(run: dict) -> int:
    return run.get("pipeline_parallel_size", 1)


def channel_table(run: dict):
    common = {k: run[k] for k in ("admission_path", "ipc_s", "stream_s")}
    if "kv_write_req_s" in run:
        return prefill_decode_table(
            **common,
            router_s=run["router_s"],
            kv_write_req_s=run["kv_write_req_s"],
            dp=_dp(run),
            router_in_s=run.get("router_in_s"),
            prefill=run.get(INSTANCES["P"], 1),
            decode=run.get(INSTANCES["D"], 1),
        )
    links = ()
    if _pp(run) > 1:
        from atom.compass.runner.pp_send import carrier_models, lookaheads

        links = lookaheads(carrier_models(_machine(run)))
    return single_engine_table(**common, dp=_dp(run), pp=_pp(run), pp_links=links)


def _machine(run: dict):
    from atom.compass.spec import MachineSpec

    return MachineSpec.from_mapping(_width_keys(run["machine"]))


def _members(table, dp: int) -> dict | None:
    """At DP width above one, each engine LP's members: rank ``dp<r>`` owns the
    channels whose ``#`` tag names it at that LP's end, ``#dp<r>`` or, on a
    channel between two engine LPs, ``#dp<sender rank>.dp<receiver rank>``."""
    if dp == 1:
        return None

    def rank(c, lp):
        tags = c.name.rpartition("#")[2].split(".")
        return tags[0] if c.source == lp else tags[-1]

    return {
        lp: {
            f"dp{r}": [
                c.name
                for c in table.channels_into(lp) + table.channels_from(lp)
                if rank(c, lp) == f"dp{r}"
            ]
            for r in range(dp)
        }
        for lp in table.registry.ids()
        if lp.name.startswith(ENGINE.name)
    }


def _pd_side(config) -> str | None:
    """``P`` or ``D`` for a deployment on the simulated KV connector, else None."""
    kv = config.kv_transfer_config
    if not kv:
        return None
    from atom.kv_transfer.disaggregation.factory import KVConnectorFactory

    if KVConnectorFactory.canonical_name(kv.get("kv_connector", "moriio")) != "compass":
        return None
    return "P" if kv.get("kv_role", "kv_producer") == "kv_producer" else "D"


def _pd_name(config, run: dict) -> str | None:
    """`_pd_side`, followed above one deployment of that side by the index
    ``compass_instance`` in its ``kv_transfer_config`` names."""
    side = _pd_side(config)
    if side is None:
        return None
    n, k = run.get(INSTANCES[side], 1), config.kv_transfer_config.get(INSTANCE_KEY)
    if n == 1 and k is None:
        return side
    if type(k) is not int or not 0 <= k < n:
        raise ValueError(
            f"kv_transfer_config's {INSTANCE_KEY} is {k!r} and the run file "
            f"declares {n} as {INSTANCES[side]}; each deployment names its own "
            "index, from 0"
        )
    return side if n == 1 else f"{side}{k}"


def _bind_transfer(config, run: dict) -> None:
    """Price the simulated connector's transfers from the machine spec and the
    KV block this deployment's workers hold."""
    from atom.compass.backends import KvGeometry, Parallelism
    from atom.compass.kv import TRANSFER_KEY, Scope, TransferModel
    from atom.compass.spec import MachineSpec

    config.kv_transfer_config[TRANSFER_KEY] = TransferModel.from_spec(
        MachineSpec.from_mapping(_width_keys(run["machine"])),
        KvGeometry.from_hf_config(
            config.hf_config,
            block_size=config.kv_cache_block_size,
            parallelism=Parallelism(tp_size=config.tensor_parallel_size),
            kv_dtype=config.kv_cache_dtype,
        ),
        Scope(run["kv_link"]),
    )


def configuration(run: dict) -> dict:
    """The run file without its per-run keys."""
    return {k: v for k, v in run.items() if k not in PER_RUN}


def router_binary(path: str, mesh: str | None = None) -> dict:
    """The router binary `path` names, resolved on PATH, as the run summary
    records it: where it is, and the commit and git status it was built at.

    Refuses, naming the binary, one built from an uncommitted tree or at no
    commit, and one whose router source differs from `mesh` (this tree's
    ``atom/mesh`` by default), the working tree included. A stamped snapshot
    has no history to diff, so there only its own commit matches."""
    exe = shutil.which(path) or path
    text = subprocess.run(
        [exe, "--version-verbose"], capture_output=True, text=True, check=True
    ).stdout
    built = dict(re.findall(r"^\s*Git (Commit|Status): *(\S*)", text, re.MULTILINE))
    commit, status = built.get("Commit", ""), built.get("Status", "")
    named = f"router binary {exe} (commit {commit or '?'}, status {status or '?'})"
    if status != "clean":
        raise ValueError(f"{named} was built from an uncommitted tree; rebuild it")
    if not re.fullmatch(r"[0-9a-f]{7,40}", commit):
        raise ValueError(f"{named} names no commit; build it in a git checkout")
    mesh = mesh or os.path.join(provenance.module_root("atom"), "mesh")
    tree = provenance.git_tree_root("atom", mesh)
    if tree.revision_kind == "commit":
        if not tree.revision.startswith(commit):
            raise ValueError(
                f"{named} is not this snapshot's commit {tree.revision}; "
                "a snapshot has no history to compare the router source across"
            )
    else:
        diff = subprocess.run(
            ["git", "-C", mesh, "diff", "--quiet", commit, "--", "."],
            capture_output=True,
            text=True,
            check=False,
        )
        if diff.returncode:
            why = diff.stderr.strip() or "it differs from the tree's"
            raise ValueError(f"{named}: router source at {mesh}: {why}")
    return {"path": exe, "commit": commit, "status": status}


def _servers(run: dict) -> list[str]:
    """The run's server LPs by name: every LP but the traffic and router LPs."""
    return [
        lp.name
        for lp in channel_table(run).registry.ids()
        if lp not in (TRAFFIC, ROUTER)
    ]


def _hand_ins(run: dict) -> list[str]:
    """The name each server LP hands its refusals in under: ``<lp>.<member>`` for
    each member of an LP declared with members, else the LP's name."""
    members = _members(channel_table(run), _dp(run)) or {}
    return [
        f"{lp}.{m}" if m else lp
        for lp in _servers(run)
        for m in members.get(LpId(lp), [None])
    ]


class _RecordingAuthority(ClockAuthority):
    """The authority, recording the step table: a row per reply and a row per
    message each reply releases, from the receiver's clock before that grant to
    the message's arrival; and the wall seconds of the run, from its start
    to the end of each client run in `ends` and to the finish, which sets `done`.
    A co-hosting frontend starts the run as it starts its own; otherwise it
    starts once every server LP, and every member of one declared with members,
    has asked once. Each client run that ends before the last has its summary
    written once every server LP has handed in. A frontend's TAR to ``+inf``
    before the finish is a stop: it registers that frontend's send log,
    finishes the run there and sets `stopped`. Every LP but the client is
    granted ``+inf``, and the client too while a client run is open, from its
    bind to its end; outside one it holds no connection. Each such grant
    releases every message registered for its receiver. A stop with none open,
    on a server whose ``client_runs`` is ``null``, sets `stopped_whole`: the
    run ended as its run file says, as at the finish."""

    def __init__(self, run: dict) -> None:
        # The tree that runs, read before the run so an unidentifiable one is
        # refused before it costs anything.
        self.source_root = provenance.git_tree_root(
            "atom", provenance.module_root("atom")
        )
        self.steps = StepTable(json.dumps(configuration(run), sort_keys=True))
        self.run = run
        self.started = self.finished = None
        self.ends = []  # the wall time each client run ended
        self.stopped = self.stopped_whole = self.client_run_open = False
        self.done = threading.Event()
        self._joining = set(_hand_ins(run))
        self._clock_before = {}  # LP -> its clock before its latest grant
        # A rerun into the same out_dir starts from none of the last run's files.
        out = Path(run["out_dir"])
        for name in (
            LP_FILE,
            STEP_TABLE_FILE,
            SUMMARY_FILE,
        ):
            for f in out.glob(name.format("*")):
                f.unlink()
        for d in out.glob(CLIENT_DIR.format("*")):
            shutil.rmtree(d)
        table = channel_table(run)
        count = run.get("client_runs", 1)
        super().__init__(
            table,
            timeline=self,
            members=_members(table, _dp(run)),
            bound_s=run["bound_s"],
            client=TRAFFIC,
            client_runs=math.inf if count is None else count,
        )

    def record(self, lp, time_from, time_to, kind, recovered) -> None:
        self._clock_before[lp] = time_from
        self.steps.record(lp, time_from, time_to, kind, detail="recovered" * recovered)

    def on_bind(self, lp):
        self.client_run_open |= lp == TRAFFIC
        return super().on_bind(lp)

    def on_request(self, lp, kind, t, log, t_daemon=math.inf, member=None) -> list:
        self._joining.discard(f"{lp.name}.{member}" if member else lp.name)
        if not self._joining and self.started is None:
            self.started = time.monotonic()
        stop = kind == TAR and t == math.inf and lp.name.startswith(FRONTEND.name)
        if stop and self.final_clocks is None:
            for ch, seq, arrival in log:
                self._register(lp, ch, seq, arrival, member)
            self.stopped = True
            self.stopped_whole = (
                self._client_runs == math.inf and not self.client_run_open
            )
            replies = [
                (to, g, self._release_all(released))
                for to, g, released in self._address(self._finish())
                if to != TRAFFIC or self.client_run_open
            ]
        else:
            replies = super().on_request(lp, kind, t, log, t_daemon, member)
        for to, _, released, *_ in replies:
            # A member's reply is addressed (lp, member); its channels name its rank.
            lp = to[0] if isinstance(to, tuple) else to
            for ch, msgs in released.items():
                for seq, arrival in msgs:
                    # From the receiver's clock as the grant found it.
                    clock = self._clock_before[lp]
                    self.steps.record(lp, clock, arrival, "release", ch, seq)
        while len(self.ends) < len(self.ended):
            self.ends.append(time.monotonic())
            self.client_run_open = False
            if self.final_clocks is None:  # an end before the last
                out = Path(self.run["out_dir"])
                _write_whole(out / STEP_TABLE_FILE, self.steps.text())
                threading.Thread(
                    target=_write_client_run,
                    args=(self.run, self, len(self.ends) - 1),
                    name="compass client run summary",
                    daemon=True,
                ).start()
        if self.final_clocks is not None and self.finished is None:
            self.finished = time.monotonic()
            self.done.set()
        return replies

    def _release_all(self, into: dict) -> dict:
        """Every message registered on the channels of `into` and not yet
        released, now released: by channel, its ``(seq, arrival)`` pairs."""
        released = {ch: [*self._undelivered[ch].items()] for ch in into}
        for ch in into:
            self._undelivered[ch].clear()
        return released


class _PublishedClock(clock.LPRuntime):
    """An LP runtime whose clock is also written to the file `NOW_ENV` names."""

    def __init__(self, *args) -> None:
        self.slot = np.memmap(os.environ[NOW_ENV], np.float64, "w+", shape=(1,))
        super().__init__(*args)

    @property
    def now(self) -> float:
        return self._now

    @now.setter
    def now(self, t: float) -> None:
        self._now = self.slot[0] = t


class _StoppingClock(clock.LPRuntime):
    """An LP runtime that, when its run ends before its ``+inf`` grant, at a
    stop, first ends every LP's run through the authority with a TAR to
    ``+inf``, carrying its send log, so each engine leaves its loop and hands
    in before the server closes the engines. A run a refusal ended has nothing
    left to end."""

    def end_run(self) -> None:
        if self.now != math.inf:
            with contextlib.suppress(ClockAbort):
                self._ca_send(TAR, math.inf, math.inf)
                while self.conn.recv()[0] != math.inf:  # up to its own +inf grant
                    pass
        super().end_run()


def _runtime(lp: LpId, run: dict, member=None, cls=clock.LPRuntime) -> clock.LPRuntime:
    rt = cls(lp, channel_table(run), connect(lp, run["clock_endpoint"], member))
    clock.install(rt)
    return rt


@contextlib.contextmanager
def _start_on_leaving(rt: clock.LPRuntime, authority=None):
    yield
    rt.start_run()
    if authority is not None:
        authority.started = time.monotonic()


def frontend(config):
    """Around the `CoreManager` construction; a no-op context on a real run.

    A real run with the parity record's variable set runs ATOM's runner
    through the recording one, and refuses any other runner, which would
    record nothing. With it set, real or simulated, each request's sequence
    carries its response id (`parity.carry_response_ids`).
    """
    global _authority, _early_end
    from atom.compass import parity

    if os.environ.get(parity.ENV):
        parity.carry_response_ids()
    run = spec()
    if run is None:
        if os.environ.get(parity.ENV):
            if config.runner_qualname != ATOM_RUNNER:
                raise ValueError(
                    f"{parity.ENV} is set and runner {config.runner_qualname!r} "
                    f"is named; only {ATOM_RUNNER} has a recording runner"
                )
            config.runner_qualname = parity.RUNNER
        return contextlib.nullcontext()
    _refuse_unserved()
    pc = config.parallel_config
    if config.enable_rapidserve:
        raise ValueError("a simulated run here is an engine LP without RapidServe")
    if config.pipeline_parallel_size != _pp(run):
        raise ValueError(
            f"the deployment runs {config.pipeline_parallel_size} pipeline stages "
            f"and the run file's pipeline_parallel_size is {_pp(run)}; the "
            "channel table every LP reads would declare the wrong stages"
        )
    if config.pipeline_parallel_size > 1 and config.tensor_parallel_size > 1:
        from atom.utils import envs

        if envs.ATOM_PP_SEND_ALLGATHER:
            raise ValueError(
                "a pipeline stage at TP above 1 sends its shard of each tensor and "
                "the next stage all-gathers it, which nothing prices; set "
                "ATOM_PP_SEND_ALLGATHER=0"
            )
    if config.runner_qualname not in (ATOM_RUNNER, COMPASS_RUNNER_QUALNAME):
        raise ValueError(
            f"runner {config.runner_qualname!r} is named, and a simulated run "
            f"prices its steps with {COMPASS_RUNNER_QUALNAME}"
        )
    side, standalone = _pd_side(config), CLOCK_ENV in os.environ
    if side is not None and not standalone:
        raise ValueError(
            "this API server would co-host the clock authority, and a "
            "prefill-decode run spans two API servers that must share one; "
            "start it on its own and name it with --compass-clock-endpoint"
        )
    # Read before `CoreManager` makes every TP rank a DP rank under DP-attention.
    dp = pc.data_parallel_size * (
        config.tensor_parallel_size if config.enable_dp_attention else 1
    )
    if dp != _dp(run):
        raise ValueError(
            f"the deployment runs {dp} data-parallel ranks and the run file's "
            f"data_parallel_size is {_dp(run)}; the channel table every LP "
            "reads would declare the wrong ranks"
        )
    if pc.data_parallel_size > 1 and pc.is_multinode_dp:
        _refuse_unshared(run, standalone)
    if dp > 1 and config.fake_eplb:
        raise ValueError(
            "--fake-eplb may start fewer engines than the data-parallel ranks "
            "the channel table declares, and a rank that never joins holds the run"
        )
    config.runner_qualname = COMPASS_RUNNER_QUALNAME
    if pc.data_parallel_rank > 0:
        # A worker node of a multi-node group: engines only, each a member.
        return contextlib.nullcontext()
    _refuse_sync_utility_commands()
    name = _pd_name(config, run)
    lp = FRONTEND if name is None else LpId(f"{FRONTEND.name}-{name}")
    authority = None
    if not standalone:
        _authority = authority = _RecordingAuthority(run)
        serve(_authority, run["clock_endpoint"])
        # Not a daemon: the API server's main thread may die of the early end
        # before this thread has written the summary.
        _early_end = threading.Thread(
            target=_write_on_early_end, args=(run, authority), name="compass early end"
        )
        _early_end.start()
    rt = _runtime(lp, run, cls=_StoppingClock)
    rt.client_run_ended = lambda k: _frontend_hand_in(run, rt, k)
    return _start_on_leaving(rt, authority)


def _refuse_unshared(run: dict, standalone: bool) -> None:
    """Refuse a node of a multi-node data-parallel group that would not reach
    the run's one authority, or whose hand-ins would miss the authority's
    out_dir: that directory holds no `AUTHORITY_FILE` naming the port this
    node's clock endpoint does."""
    if not standalone:
        raise ValueError(
            "this node runs part of a multi-node data-parallel group, whose nodes "
            "share one clock authority; start it on its own and name it with "
            "--compass-clock-endpoint on every node"
        )
    mark = Path(run["out_dir"]) / AUTHORITY_FILE
    port = run["clock_endpoint"].rpartition(":")[2]
    if not mark.is_file() or mark.read_text().rpartition(":")[2] != port:
        raise ValueError(
            f"{mark} names no authority at {run['clock_endpoint']}: every node of "
            "a multi-node data-parallel group hands in to the authority's out_dir, "
            "so each must see it at the same path; start the authority first"
        )


def _refuse_unserved() -> None:
    """Refuse the frontends no client of a simulated run reaches: the Atomesh
    standalone service, and an `LLMEngine` that the API server's ``main`` is
    not building, such as an offline one."""
    from atom.utils import envs

    if envs.USE_ATOMESH_ENTRYPOINTS:
        raise ValueError(
            "USE_ATOMESH_ENTRYPOINTS=1 serves through the Atomesh standalone "
            "service, whose worker thread sends each request, and only the "
            "frontend's clock owner may; serve a simulated run with the API server"
        )
    root = Path(atom.__file__).parent
    on_stack = {
        (Path(f.f_code.co_filename), f.f_code.co_name)
        for f, _ in traceback.walk_stack(sys._getframe())
    }
    building = (root / "model_engine" / "llm_engine.py", "__init__") in on_stack
    served = (root / "entrypoints" / "openai" / "api_server.py", "main") in on_stack
    if building and not served:
        raise ValueError(
            "an offline LLMEngine has no client to bind the traffic LP, so no "
            "grant passes its first request; serve a simulated run with the API "
            "server's --compass-run"
        )


def _refuse_sync_utility_commands() -> None:
    """Make `CoreManager.broadcast_utility_command_sync` refuse a call on the
    frontend's clock owner, which waits on wall time for replies that reach it
    only at that owner's next clock call."""
    from atom.model_engine.engine_core_mgr import CoreManager

    sync = CoreManager.broadcast_utility_command_sync

    @functools.wraps(sync)
    def refusing(self, cmd, *args, **kwargs):
        rt = clock.installed()
        if rt is not None and threading.current_thread() is rt.owner:
            why = f"sync utility command {cmd!r} on the clock owner"
            rt.refusals.append(f"clock:{why}")
            raise RuntimeError(
                f"{rt.me}: {why}, which would wait on wall time for replies "
                "released only at its next clock call"
            )
        return sync(self, cmd, *args, **kwargs)

    CoreManager.broadcast_utility_command_sync = refusing


def engine(config):
    """Around the engine's construction in its process; a no-op context on a real run.

    At DP width above one, the process joins the engine LP as the member of its
    DP rank, and its host collectives first join the group rounds it owes. A
    pipeline stage is the engine LP ``engine-pp<stage>``.
    """
    global _member
    run = spec()
    if run is None:
        return contextlib.nullcontext()
    from atom.utils.distributed import utils

    lp, side = ENGINE, _pd_side(config)
    pc = config.parallel_config
    if getattr(config, "pipeline_parallel_size", 1) > 1:
        lp = LpId(f"{ENGINE.name}-pp{pc.pipeline_parallel_rank}")
    if side is not None:
        lp = LpId(f"{ENGINE.name}-{_pd_name(config, run)}")
        _bind_transfer(config, run)
    utils.LP_OF_RANK = dict.fromkeys(range(pc.data_parallel_size), lp)
    _member = f"dp{pc.data_parallel_rank}" if pc.data_parallel_size > 1 else None
    name = lp.name if _member is None else f"{lp.name}.{_member}"
    os.environ[NOW_ENV] = str(Path(run["out_dir"]) / f"now-{name}.bin")
    clock.settle_before_collectives()
    return _named(_start_on_leaving(_runtime(lp, run, _member, _PublishedClock)))


@contextlib.contextmanager
def _named(inner):
    """`inner`, with `NOW_ENV` set for the processes started in it."""
    try:
        with inner:
            yield
    finally:
        del os.environ[NOW_ENV]


def runner(model_runner, tree) -> None:
    """Install the cost backend and the device readings from the run file.

    `tree` is the model built on the meta device, which the weights are read
    off."""
    run = spec()
    if run is None:
        return
    from atom.compass.artifacts import machine_spec_digest
    from atom.compass.backends.geometry import KvGeometry, Parallelism
    from atom.compass.backends.step_linear import StepLinearBackend, StepLinearLaw
    from atom.compass.backends.step_roofline import (
        StepFallbackBackend,
        StepRooflineBackend,
    )
    from atom.compass.backends.step_stub import Coefficients, StepStubBackend
    from atom.compass.memory import ModelTerms, device_readings, predicts
    from atom.compass.memory.check import record
    from atom.compass.runner.overrides import (
        HostDeviceRecurrence,
        install_cost_backend,
        install_device_readings,
    )
    from atom.compass.spec import MachineSpec

    config = model_runner.config
    tp = config.tensor_parallel_size
    refuse_renamed_keys(run, "the run file")
    backends = (STEP_STUB, STEP_LINEAR, STEP_GRID, "replay")
    keys = [k for k in backends if k in run]
    if len(keys) != 1:
        raise ValueError(
            f"a run file prices its steps from one of {', '.join(backends[:-1])} "
            f"and replay, and this one has {' and '.join(keys) or 'none'}"
        )
    replay = None
    if "replay" in run:
        from atom.compass.parity.replay import ReplayBackend

        dp_rank = config.parallel_config.data_parallel_rank
        if max(run.get(k, 1) for k in INSTANCES.values()) > 1:
            raise ValueError(
                "replay reads one record per side, prefill/ and decode/, and this "
                "run has several deployments of a side"
            )
        side = {"P": "prefill", "D": "decode"}.get(_pd_side(config), "")
        backend = replay = ReplayBackend(Path(run["replay"]) / side, dp_rank)
    if STEP_LINEAR in run:
        if tp != 1:
            raise ValueError(
                "a step linear law is fitted on one rank of tensor-parallel width "
                f"1, and this run has tensor_parallel_size {tp}"
            )
        # Under DP-attention every rank runs at TP 1, and experts shard over the group.
        dpa = config.enable_dp_attention
        dp = config.parallel_config.data_parallel_size if dpa else 1
        ep = dp if config.enable_expert_parallel else 1
        law = run[STEP_LINEAR]
        backend = StepLinearBackend(StepLinearLaw(law), (dp, ep))
        if "engine_config" in law:
            from atom.compass.run.step_grid import keyed

            fitted = law["engine_config"]
            differ = [
                f"{name} {fitted.get(name)!r} in the law and {value!r} here"
                for name, value in keyed(config).items()
                if fitted.get(name) != value
            ]
            if differ:
                raise ValueError(
                    "the step linear law was fitted under another engine "
                    "config: " + "; ".join(differ)
                )
    elif STEP_STUB in run:
        # A pipeline stage is priced for its own span of the stack, the span
        # ATOM's `make_layers` builds it from.
        hf, pp = config.hf_config, getattr(config, "pipeline_parallel_size", 1)
        depth, share = hf.num_hidden_layers, None
        start, end = 0, depth
        if pp > 1:
            from atom.compass.backends.geometry import paged_layers
            from atom.models.utils import get_pp_indices

            pp_rank = config.parallel_config.pipeline_parallel_rank
            start, end = get_pp_indices(depth, pp_rank, pp)
            text = getattr(hf, "text_config", hf)
            paged = paged_layers(text, 0, depth)
            if not paged:
                raise ValueError(
                    "a pipeline stage prices its history terms on its share of the "
                    "paged layers, and this stack has no layer that holds a cache "
                    "of every past token"
                )
            share = ((end - start) / depth, paged_layers(text, start, end) / paged)
        backend = StepStubBackend(
            Coefficients(**run[STEP_STUB]),
            Parallelism(tp_size=tp, pp_size=pp),
            stack_layers=end - start,
            stage_share=share,
        )
    machine = MachineSpec.from_mapping(_width_keys(run["machine"]))
    platform = machine_spec_digest(machine)
    fitted_on = run.get(STEP_LINEAR, {}).get("machine_spec")
    if STEP_LINEAR in run and fitted_on != platform:
        raise ValueError(
            f"the step linear law was fitted on machine spec {fitted_on} and this "
            f"run's is {platform}; a law measured on one platform is no evidence "
            "about another"
        )
    bundle = None
    if STEP_GRID in run:
        from atom.compass.run.step_grid import load

        backend, bundle = load(run[STEP_GRID], config, machine)
    if "store" not in run:
        raise ValueError(
            "the run file names no store, and a run reads its configuration's "
            "memory readings from one; compass measure memory takes them"
        )
    from atom.compass.run import memory

    measured = memory.load(run["store"], config, machine, model_runner.rank)
    model = ModelTerms.from_config(
        config.hf_config,
        activations=measured.activations(config.max_num_batched_tokens),
        config_json=_config_json(config),
        tree=tree,
    )
    fallback = run.get("step_roofline", False)
    if not isinstance(fallback, bool):
        raise TypeError(
            f"the run file's step_roofline is {fallback!r}; it is true, which "
            "prices the steps the backend refuses by the step roofline, or false"
        )
    if fallback:
        kv = KvGeometry.from_hf_config(
            config.hf_config,
            block_size=config.kv_cache_block_size,
            parallelism=Parallelism(tp_size=tp),
            kv_dtype=config.kv_cache_dtype,
        )
        roofline = StepRooflineBackend(
            machine,
            parameters=run["parameter_count"] // tp,
            weight_bytes=model.weights.nbytes,
            kv_token_bytes=kv.layers * kv.bytes_per_token_per_layer,
        )
        backend = StepFallbackBackend(backend, roofline)
    install_cost_backend(model_runner, backend)
    if NOW_ENV in os.environ:
        slot = np.memmap(os.environ[NOW_ENV], np.float64, "r", shape=(1,))
        host = run.get("host", {})
        ours = host.get("laws") and {**host_key(config), "machine_spec": platform}
        theirs = host.get("key") or {}
        if ours and theirs != ours:
            differ = [k for k in ours if theirs.get(k) != ours[k]]
            raise ValueError(
                f"the run file's host laws were fitted under another {differ}; "
                f"this run's key is {ours}, and theirs is {theirs}"
            )
        pp = getattr(config, "pipeline_parallel_size", 1)
        stage = (config.parallel_config.pipeline_parallel_rank, pp) if pp > 1 else None
        model_runner.compass_host = replay or HostDeviceRecurrence(
            host.get("laws", {}), host.get("skip", ()), lambda: float(slot[0]), stage
        )
        if replay is not None:
            replay.now = lambda: float(slot[0])
    model_runner.compass_abort = run.get("on_refusal") == "abort"
    reserved, captured = _graph_pool(model_runner, machine, model)
    # The readings' label: under DP-attention the layout, with the EP width
    # when the experts shard; one rank of it is a TP1 engine.
    dp = config.parallel_config.data_parallel_size
    layout = tp
    if dp > 1 and config.enable_dp_attention:
        ep = f"x{tp * dp}" if config.enable_expert_parallel else ""
        layout = f"{tp}x{dp}{ep}"
    readings = device_readings(
        machine,
        memory=measured,
        tp_width=layout,
        model=model,
        cudagraph_overhead=reserved,
    )
    install_device_readings(model_runner, readings)
    pool = (
        None
        if captured is None
        else predicts(machine, tp_width=tp, captured_tokens=captured)
    )
    model_runner.compass_memory = record(
        readings, tokens=config.max_num_batched_tokens, pool=pool
    )
    if bundle is not None:
        model_runner.compass_step_grid = {**bundle, "describe": backend.describe()}


def _graph_pool(model_runner, machine, model):
    """What ATOM's `_estimate_cudagraph_overhead` reserves for this deployment:
    nothing under ``enforce_eager``, else the branch the runner's own
    `_piecewise_cg_active` picks, over the same config fields it reads. With it,
    the tokens `capture_cudagraph` captures, None under ``enforce_eager``. The
    piecewise reservation caps only what it reserves; the capture takes the
    same bounded ladder in both modes."""
    from atom.compass.memory import (
        PiecewiseCapture,
        capture_token_shapes,
        piecewise_per_token_bytes,
        reserves,
    )

    config = model_runner.config
    if config.enforce_eager:
        return reserves(enforce_eager=True), None
    # At one token per sequence a batch size is its token count, so the
    # capture loop's schedulable bound applies to both.
    bound = min(config.max_num_seqs, config.max_num_batched_tokens)
    captured = sum(
        capture_token_shapes(config.capture_sizes, max_num_batched_tokens=bound)
    )
    if not model_runner._piecewise_cg_active():
        return reserves(activations=model.activations), captured
    hf = config.hf_config
    sizes = config.compilation_config.cudagraph_capture_sizes or [config.max_num_seqs]
    capacity = machine.value("device.memory.capacity_bytes")
    capture = PiecewiseCapture(
        per_token_bytes=piecewise_per_token_bytes(
            hidden_size=int(hf.hidden_size),
            layers=int(hf.num_hidden_layers),
            dtype_bytes=config.torch_dtype.itemsize,
            dp_size=config.parallel_config.data_parallel_size,
        ),
        token_shapes=capture_token_shapes(
            sizes, max_num_batched_tokens=config.max_num_batched_tokens
        ),
        budget_bytes=int(config.gpu_memory_utilization * capacity),
    )
    return reserves(piecewise=capture), captured


def tokenizer(tok, config) -> None:
    """Wrap `tok`'s ``encode`` and ``decode`` with the machine spec's entry for
    the model's architecture and the backend that loaded; the entry warns when
    the loaded ``tokenizer.json`` is not the one it was measured on."""
    run = spec()
    if run is None:
        return
    from atom.compass.spec import Backend, MachineSpec
    from atom.utils.compass_loop import wrap_decode, wrap_encode

    entry = MachineSpec.from_mapping(_width_keys(run["machine"])).tokenizer_for(
        config.hf_config.architectures[0],
        Backend.FAST if tok.is_fast else Backend.SLOW,
        _fingerprint(tok),
    )
    tok.encode = wrap_encode(tok.encode, entry)
    tok.decode = wrap_decode(tok.decode, entry)


def http_intake() -> tuple[float, float] | None:
    """``(fixed_s, per_byte_s)``, the frontend loop's time to take in one HTTP
    request, from the run's machine spec; None on a real run, a run file with no
    machine spec, or a spec that states neither. A spec that states one of them
    is refused naming the other."""
    run = spec()
    if run is None or "machine" not in run:
        return None
    machine = _machine(run)
    paths = ("host.http_intake.fixed_s", "host.http_intake.per_byte_s")
    if not any(p in machine.values for p in paths):
        return None
    return tuple(machine.value(p) for p in paths)


def _fingerprint(tok) -> str | None:
    """``sha256:<hex>`` of the ``tokenizer.json`` a fast tokenizer was loaded
    from, resolved from its ``name_or_path`` as loading did; None without one."""
    name = getattr(tok, "name_or_path", None)
    if not (tok.is_fast and name):
        return None
    return _file_digest(name, "tokenizer.json")


def host_cpu() -> str:
    """This host's CPU model, as /proc/cpuinfo names it, else as
    `platform.processor` does."""
    cpu = re.search(
        r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE
    )
    return cpu.group(1) if cpu else platform.processor()


def host_key(config) -> dict:
    """What host laws are keyed by: this host's CPU, ATOM's source less
    `COMPASS_ONLY`, the served config.json and the engine config a step grid
    bundle is keyed by."""
    from atom.compass.run.step_grid import keyed

    return host_key_of(config.model, keyed(config))


def host_key_of(model: str, engine: dict) -> dict:
    """`host_key` for the model `model` names and the engine config `engine`
    that `keyed` gave, for a process that holds no resolved config."""
    root = Path(atom.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root)
        skip = next(
            (v for k, v in COMPASS_ONLY.items() if relative.is_relative_to(k)), ()
        )
        if skip is None:
            continue
        lines = path.read_bytes().splitlines(keepends=True)
        tree = ast.parse(b"".join(lines)) if skip else None
        doc = tree.body[0] if tree and ast.get_docstring(tree) is not None else None
        body = tree.body if tree else []
        for prev, node in reversed(list(zip([None, *body], body))):
            named = (node, *getattr(node, "targets", ()))
            names = {getattr(n, "name", getattr(n, "id", None)) for n in named}
            if node is doc or names & skip:
                del lines[prev.end_lineno if prev else 0 : node.end_lineno]
        digest.update(f"{relative}\0".encode() + b"".join(lines))
    return {
        "host_cpu": host_cpu(),
        "atom": digest.hexdigest(),
        "model": _file_digest(model, "config.json"),
        "engine": engine,
    }


def _config_json(config) -> dict:
    """The model's config.json as loading reads it, with `config.hf_overrides`
    set over its text-level keys, as `Config` sets them over `hf_config`, so
    an overridden field counts as stated."""
    from transformers import PretrainedConfig

    file = PretrainedConfig.get_config_dict(config.model)[0]
    file.get("text_config", file).update(getattr(config, "hf_overrides", None) or {})
    return file


def _file_digest(name: str, filename: str) -> str | None:
    """``sha256:<hex>`` of a model's `filename`, resolved from `name` as loading
    does; None when it is not there."""
    from transformers.utils import cached_file

    try:
        path = cached_file(name, filename, local_files_only=True)
    except OSError:
        return None
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _width_keys(node):
    """JSON writes a width table's integer keys as strings; read them back."""
    if isinstance(node, list):
        return [_width_keys(v) for v in node]
    if not isinstance(node, dict):
        return node
    return {int(k) if k.isdigit() else k: _width_keys(v) for k, v in node.items()}


def _write_whole(path: Path, text: str) -> None:
    """Write `path` whole or not at all, so a reader that polls for it never
    reads it part-written."""
    part = path.with_name(path.name + ".part")
    part.write_text(text)
    part.replace(path)


def _hand_in(
    run: dict, lp: str, reasons: list, client_run: int | None = None, **sections
) -> None:
    """Write `lp`'s refusals, and its other sections, for the summary's writer:
    at the end of client run `client_run`, or None for the finish."""
    out = Path(run["out_dir"])
    if client_run is not None:
        out = out / CLIENT_DIR.format(client_run)
        out.mkdir(exist_ok=True)
    _write_whole(
        out / LP_FILE.format(lp), json.dumps({"refusals": reasons, **sections})
    )


def _workers_say(engine_core) -> dict:
    """The workers' command refusals, what their steps cost and their memory."""
    workers = engine_core.runner_mgr
    return {
        "refusals": list(workers.call_func("refused_commands", wait_out=True)),
        "priced": dict(workers.call_func("priced_steps", wait_out=True)),
        "memory": workers.call_func("predicted_memory", wait_out=True),
    }


def engine_done(engine_core, **sections) -> None:
    """After the engine's loop: leave the clock and hand in."""
    clock.close()
    run = spec()
    if run is not None:
        _engine_hand_in(run, engine_core, sections)


def _engine_hand_in(run: dict, engine_core, sections: dict, client_run=None) -> None:
    """Hand in what the worker's steps cost and the memory it predicts, with
    `sections`, its command refusals and the engine's refused clock calls."""
    said = _workers_say(engine_core)
    rt = clock.installed()
    lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
    reasons = said.pop("refusals") + rt.refusals
    _hand_in(run, lp, reasons, client_run, **said, **sections)


@contextlib.contextmanager
def engine_loop(engine_core):
    """Around the engine's loop: `engine_done` once it leaves, which raises if
    that was before the finish. If either raises in a run, hand in the engine's
    refused clock calls and the exception, and what the workers say in 10 wall
    seconds unless one died. Under ``step_grid`` the workers are asked for the bundle
    before the loop, so a run that ends early still hands it in.

    In a run the engine ignores SIGTERM until it has handed in: ATOM's handler
    would stop the workers inside a clock call, before they say what their
    steps cost. A stop reaches the engine as the ``+inf`` grant instead.

    A pipeline stage's runner calls go through `pp_send.install`'s sends."""
    run, sections = spec(), {}
    if run is not None and getattr(engine_core, "pp_size", 1) > 1:
        from atom.compass.runner import pp_send

        pp_send.install(engine_core, _machine(run))
    if run is not None and STEP_GRID in run:
        workers = engine_core.runner_mgr
        sections["step_grid"] = workers.call_func("step_grid_bundle", wait_out=True)
    if run is not None:
        clock.installed().client_run_ended = lambda k: _engine_hand_in(
            run, engine_core, sections, k
        )
        before = signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        yield
        engine_done(engine_core, **sections)
    except BaseException as e:
        if run is not None:
            box = []
            if not isinstance(e, SystemExit):
                ask = threading.Thread(
                    target=lambda: box.append(_workers_say(engine_core)), daemon=True
                )
                ask.start()
                ask.join(10.0)
            said = box[0] if box else {}
            rt = clock.installed()
            lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
            reasons = said.pop("refusals", []) + rt.refusals
            _hand_in(
                run, lp, reasons + [f"ended:{lp} raised {e!r}"], **said, **sections
            )
        raise
    finally:
        if run is not None and before is not None:  # None: not set from Python
            signal.signal(signal.SIGTERM, before)


def _until_end(run: dict, a: "_RecordingAuthority") -> bool:
    """Wait for the finish, or for an LP to hand in before it; True for the
    latter, an early end. Gives up, False, once this process's main thread ends."""
    out = Path(run["out_dir"])
    while not a.done.wait(0.1):
        # Every LP hands in after the finish sets `done`, so read it second.
        if any(out.glob(LP_FILE.format("*"))) and not a.done.is_set():
            return True
        if not threading.main_thread().is_alive():
            return False
    return False


def _write_on_early_end(run: dict, a: "_RecordingAuthority") -> None:
    if _until_end(run, a):
        _write_summary(run, a, ended=True)


def frontend_done(llm_engine) -> bool:
    """Whether the server left at a simulated run's finish, or at a stop: a
    signal that ended its event loop's run. At either, stop the engines and
    hand in the frontend's refusals, at a stop with the signal that ended it.
    With the authority co-hosted, then write the step table and the run summary;
    at a stop, after the early end's writer, which has written its own if an LP
    handed in before the stop, so the summary left counts every hand-in."""
    rt = clock.installed()
    stop = None if rt is None else getattr(rt.loop, "stopped", None)
    if rt is None or (rt.now != math.inf and stop is None):
        return False
    llm_engine.close()
    run = spec()
    _frontend_hand_in(run, rt, stopped=stop)
    if _authority is not None:
        if stop is not None:
            _early_end.join()
        early = stop is not None and not _authority.stopped_whole
        _write_summary(run, _authority, ended=early)
    return True


def _frontend_hand_in(run: dict, rt, client_run=None, stopped=None) -> None:
    """Hand in the frontend's executor's refused jobs and its refused clock
    calls, and the signal that stopped it, if one did."""
    reasons = rt.loop.executor.refusals + rt.refusals
    sections = {} if stopped is None else {"stopped": stopped}
    _hand_in(run, rt.me.name, reasons, client_run, **sections)


def _read_hand_ins(run: dict, where: Path, wait: bool) -> dict:
    """Every server LP's hand-in in `where`, by name; with `wait`, once all are."""
    files = {n: where / LP_FILE.format(n) for n in _hand_ins(run)}
    deadline = time.monotonic() + HAND_IN_WAIT_S
    while wait and (missing := [f.name for f in files.values() if not f.exists()]):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no {', '.join(missing)} in {where} after {HAND_IN_WAIT_S} wall "
                "seconds, so the run summary would miss those LPs' refusals"
            )
        time.sleep(0.1)
    return {n: json.loads(f.read_text()) for n, f in files.items() if f.exists()}


def _write_summary(run: dict, a: _RecordingAuthority, ended: bool = False) -> None:
    """Write the step table, then the run summary once every server LP has
    handed in its refusals, or after an early end, from those handed in so far;
    after the finish, unless a stop brought it, the last client run's summary
    too."""
    out = Path(run["out_dir"])
    _write_whole(out / STEP_TABLE_FILE, a.steps.text())
    hand_ins = _read_hand_ins(run, out, not ended)
    end = a.finished or time.monotonic()
    summary = _summary(run, a, a, end - (a.started or end), hand_ins, {})
    _write_whole(out / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))
    if not ended and a.ended and not a.stopped:
        _write_client_run(run, a, len(a.ended) - 1, hand_ins)


def _write_client_run(run: dict, a: _RecordingAuthority, k: int, hand_ins=None):
    """Write client run `k`'s summary from the hand-ins at its end, those in its
    directory unless given, less those at client run ``k - 1``'s end."""
    out = Path(run["out_dir"])
    here = out / CLIENT_DIR.format(k)
    if hand_ins is None:
        hand_ins = _read_hand_ins(run, here, True)
    since = _read_hand_ins(run, out / CLIENT_DIR.format(k - 1), False) if k else {}
    record = a.ended[k]
    summary = _summary(run, a, record, a.ends[k] - a.started, hand_ins, since)
    summary["client_run"] = {"index": k, "start": record.start, "end": record.end}
    here.mkdir(exist_ok=True)
    _write_whole(here / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))


def _summary(run: dict, a, record, wall: float, hand_ins: dict, since: dict) -> dict:
    """The summary of `record`, the authority or one client run, over the steps
    the hand-ins count beyond those in `since`."""
    from atom.compass.backends.cost import GROUPS

    reasons = Counter(r for h in hand_ins.values() for r in h["refusals"])
    # A frontend's stop ends the run early, unless it left the run whole.
    if not a.stopped_whole:
        stops = [(n, h["stopped"]) for n, h in hand_ins.items() if "stopped" in h]
        reasons.update(f"ended:{n} stopped by {sig}" for n, sig in stops)
    reasons -= Counter(r for h in since.values() for r in h["refusals"])
    reasons = sorted(reasons.elements())
    priced = [h["priced"] for h in hand_ins.values() if "priced" in h]
    before = [h["priced"] for h in since.values() if "priced" in h]

    def total(key):
        return sum(p.get(key, 0) for p in priced) - sum(p.get(key, 0) for p in before)

    seconds = total("seconds")
    tally = RefusalTally.of(reasons, total("steps"))
    summary = RunSummary.of(record, wall, refusals=tally).as_record()
    schedule = summary["schedule"]
    schedule["predicted_seconds"] = seconds
    # How the seconds were obtained: reported, never gated.
    schedule["seconds_by_method"] = {
        g: {
            "steps": total(f"{g}_steps"),
            "fraction_of_predicted_seconds": (
                total(f"{g}_seconds") / seconds if seconds else 0.0
            ),
        }
        for g in GROUPS
    }
    # Each assumed treatment the steps were priced under, with its step count.
    treatments = sum((Counter(p.get("treatments", {})) for p in priced), Counter())
    treatments -= sum((Counter(p.get("treatments", {})) for p in before), Counter())
    schedule["treatments"] = dict(sorted(treatments.items()))
    # An engine whose workers died, or that never handed in, priced steps no
    # one counted: they are absent, not 0.
    if any(
        "priced" not in hand_ins.get(n, {})
        for n in _hand_ins(run)
        if n.startswith("engine")
    ):
        schedule.update(predicted_seconds=None, seconds_by_method=None, treatments=None)
        schedule["refusals"].update(steps=None, fraction_of_steps=None)
    summary["configuration"] = configuration(run)
    summary["schema_version"] = SUMMARY_SCHEMA_VERSION
    summary["source_root"] = a.source_root.as_json()
    if "router" in run:
        summary["router"] = run["router"]
    # Any refusal, of any source, leaves part of the run unsimulated; a run
    # that ended early handed in why.
    summary["coverage_report"] = bool(reasons)
    if grids := {n: h["step_grid"] for n, h in hand_ins.items() if "step_grid" in h}:
        summary["step_grid"] = grids
    return summary


def authority(endpoint: str) -> None:
    """Name `endpoint` in the out_dir's `AUTHORITY_FILE`, then serve the run
    file's authority there until the finish or an early
    end, then write its step table, and the run summary once the server LPs
    have handed in."""
    run = spec()
    a = _RecordingAuthority(run)
    _write_whole(Path(run["out_dir"]) / AUTHORITY_FILE, endpoint)
    serve(a, endpoint)
    _write_summary(run, a, _until_end(run, a))
