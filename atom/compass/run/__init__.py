# SPDX-License-Identifier: MIT
"""The per-process bootstrap of a simulated run: which LP a process is, and its clock.

A run is on when `ENV` names a run file, a JSON object every process of the
deployment reads; spawned processes inherit it. The API server's
``--compass-run`` sets it. With it unset, every function here returns at once
and ATOM runs as it always does.

One engine LP at one stage, one frontend LP, and the traffic LP the harness
drives, over `single_engine_table`. At data-parallel width above one, each DP
rank's engine process is a member ``dp<rank>`` of the one engine LP and owns
its rank's channels. The Clock Authority is co-hosted in the frontend's
process and served at the run file's ``clock_endpoint``, which every LP and
member, the frontend's included, connects to.

- `frontend(config)` wraps the `CoreManager` construction in `LLMEngine`: it
  selects the simulated runner, serves the authority and installs the
  frontend LP's runtime; leaving it, after READY from every engine, starts
  the run.
- `engine(config)` wraps the engine's construction in its own process the
  same way, and maps every DP rank to the engine LP for the process-group
  check.
- `runner(runner)` installs the cost backend and the device readings on a
  `CompassModelRunner` where it would build its model, in the worker process.
- `tokenizer(tokenizer, config)` charges the served tokenizer's ``encode`` and
  ``decode`` to the frontend's clock, at the rates the machine spec measured.
- `engine_done(engine)` and `frontend_done(engine)` close each side after the
  finish, each handing in its LP's refusals; the frontend's, when it co-hosts
  the authority, then writes the step table and the run summary.
- `engine_loop(engine)` wraps the engine's loop: under ``step_fit`` it asks the
  workers for the bundle before it, `engine_done` after it, and if either
  raises, the engine hands in the exception as an ``ended:`` reason, with the
  bundle; `engine_done` raises for a loop that left before the finish. A
  hand-in before the finish is an early end: the authority's host writes the
  step table and the run summary at once, each LP's clock as it stands, from
  what has been handed in.

Run file keys: ``clock_endpoint``, ``bound_s`` (finite), ``admission_path``,
``ipc_s``, ``stream_s``, one of ``coefficients`` (a `Coefficients` mapping,
priced by `ShapeStubBackend`), ``law`` (a `CoarseLaw` mapping, priced by
`CoarseBackend`), ``step_fit`` (an artifact store's root and a bundle's key,
priced by `StepFitBackend`; see `step_fit`) and ``replay`` (a real run's parity
record directory: each step ends when the real one did, by
`atom.compass.parity.replay`),
``roofline_floor_s`` (optional: a tier-0 `Roofline` with this host floor prices
the steps the backend refuses),
``on_refusal`` (optional: ``mark``, the default, prices a step no cost source
answers at 0 s, hands in its ``cost:`` reason and continues; ``abort`` ends
the run there, which still writes the summary),
``host`` (optional: ``laws``, per forward kind and host segment, coefficients
on ``(1, tokens, seqs)``, zero where absent; ``key``, the `host_key` they were
fitted under, which a run with laws must match; ``skip``, segments zeroed for
a what-if run, which the graders refuse),
``machine`` (a machine spec mapping), ``parameter_count``, and ``out_dir``,
where the step table, the run summary and each server LP's hand-in go, one
file written as the LP leaves its loop. It holds the LP's refusals: an
engine's are the worker's refused commands and the engine's refused clock
calls, a frontend's its executor's refused jobs and its refused clock calls.
An engine's also holds what its worker's steps cost, for the summary's
predicted seconds and its analytical and extrapolated shares.
It holds the memory the worker predicts too, for
``python -m atom.compass.memory.check``, and under ``step_fit`` the bundle's
key, digest and ``describe()`` line, which the summary carries. The authority
removes an earlier run's files there as it starts. An optional
``data_parallel_size``, 1 when absent, is the deployment's DP width, which
under DP-attention is its TP size times its DP size; the traffic LP reads the
channel table from the run file alone. Each DP rank's engine process hands in
its own refusals, as ``<lp>.<member>``.

A run file that also has ``router_s`` and ``kv_write_req_s`` describes a
prefill-decode run over `prefill_decode_table`, and ``kv_link`` (``intra_node``
or ``inter_node``) names the link its KV transfers are priced on. A deployment
whose ``kv_connector`` is ``compass`` is its prefill (``kv_producer``) or
decode side: `engine(config)` makes it the ``engine-P`` or ``engine-D`` LP and
binds the connector's transfer model. The write-request address comes with the
deployment's own ``kv_transfer_config``. `frontend(config)` refuses such a
deployment unless the authority is standalone: a prefill-decode run spans two
API servers that must share one.

The API server's ``--compass-clock-endpoint`` sets `CLOCK_ENV`: the authority
is not co-hosted but served by `authority`, its own process started before the
engines (``python -m atom.compass.run``), and every LP connects to that
endpoint. Its frontend is ``frontend-P`` or ``frontend-D`` by ``kv_role``. The
standalone authority writes the step table and the run summary once every
server LP has handed in its refusals, which the out_dir both API servers share
carries.
"""

import ast
import contextlib
import hashlib
import json
import math
import os
import platform
import re
import threading
import time
from pathlib import Path

import numpy as np

import atom
from atom.compass.artifacts import provenance
from atom.compass.clock import (
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
ATOM_RUNNER = "atom.model_engine.model_runner.ModelRunner"
#: Keys that differ between two runs of one configuration, kept out of its name.
PER_RUN = ("clock_endpoint", "out_dir")
LP_FILE, STEP_TABLE_FILE, SUMMARY_FILE = (
    "lp-{}.json",  # by LP name, or <lp>.<member> for a member
    "step_table.txt",
    "summary.json",
)
#: Wall seconds the summary's writer waits for the server LPs' refusals.
HAND_IN_WAIT_S = 300.0
#: The format of the run summary; raise it when a field changes meaning or goes.
SUMMARY_SCHEMA_VERSION = 1
#: Compass source the host-law key leaves out, by path under the package: all
#: of it (None), or the top-level names only an installed LP runtime reaches.
COMPASS_ONLY = {
    "compass": None,
    "utils/compass_loop.py": None,
    "utils/clock.py": {"LPRuntime", "WrappedSocket", "WrappedPoller", "RelayQueue"},
    "utils/zmq_shim.py": {"_Socket", "_ChannelSocket", "_Context"},
}

_authority: "_RecordingAuthority | None" = None
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
    _check_host(path, run.get("host", {}))
    run["clock_endpoint"] = os.environ.get(CLOCK_ENV, run.get("clock_endpoint"))
    return run


def _check_host(path, host: dict) -> None:
    """Refuse a ``host`` entry the runner would read as zero host time."""
    from atom.compass.runner.overrides import FORWARD_KINDS, HOST_SEGMENTS

    laws = host.get("laws", {})
    if (
        set(host) - {"laws", "skip", "key"}
        or set(host.get("skip", ())) - set(HOST_SEGMENTS)
        or set(laws) - set(FORWARD_KINDS)
        or {s for v in laws.values() for s in v} - set(HOST_SEGMENTS)
        or {len(c) for v in laws.values() for c in v.values()} - {3}
    ):
        raise ValueError(
            f"{path}: host is {host!r}; it takes skip, a list of {HOST_SEGMENTS}, "
            f"laws, mapping {FORWARD_KINDS} to segments to three coefficients, "
            "and key, the host key the laws were fitted under"
        )


def _dp(run: dict) -> int:
    return run.get("data_parallel_size", 1)


def channel_table(run: dict):
    common = {k: run[k] for k in ("admission_path", "ipc_s", "stream_s")}
    if "kv_write_req_s" in run:
        if _dp(run) != 1:
            raise ValueError(
                f"a prefill-decode run declares data_parallel_size {_dp(run)}; "
                "its channel table has the channels of DP rank 0 only"
            )
        return prefill_decode_table(
            **common, router_s=run["router_s"], kv_write_req_s=run["kv_write_req_s"]
        )
    return single_engine_table(**common, dp=_dp(run))


def _members(table, dp: int) -> dict | None:
    """At DP width above one, the engine LP's members: each rank owns its ``#dp<rank>`` channels."""
    if dp == 1:
        return None
    names = [c.name for c in table.channels_into(ENGINE) + table.channels_from(ENGINE)]
    return {
        ENGINE: {
            f"dp{r}": [n for n in names if n.endswith(f"#dp{r}")] for r in range(dp)
        }
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


def _servers(run: dict) -> list[str]:
    """The run's server LPs by name: every LP but the traffic LP."""
    return [lp.name for lp in channel_table(run).registry.ids() if lp != TRAFFIC]


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
    message each reply releases; and the wall seconds of the run, from its start
    to the finish, which sets `done`. A co-hosting frontend starts the run as it
    starts its own; otherwise it starts once every server LP has asked once."""

    def __init__(self, run: dict) -> None:
        # The tree that runs, read before the run so an unidentifiable one is
        # refused before it costs anything.
        self.source_root = provenance.git_tree_root(
            "atom", provenance.module_root("atom")
        )
        self.steps = StepTable(json.dumps(configuration(run), sort_keys=True))
        self.started = self.finished = None
        self.done = threading.Event()
        self._joining = set(_servers(run))
        # A rerun into the same out_dir starts from none of the last run's files.
        out = Path(run["out_dir"])
        for name in (
            LP_FILE,
            STEP_TABLE_FILE,
            SUMMARY_FILE,
        ):
            for f in out.glob(name.format("*")):
                f.unlink()
        table = channel_table(run)
        super().__init__(
            table,
            timeline=self,
            members=_members(table, _dp(run)),
            bound_s=run["bound_s"],
        )

    def record(self, lp, time_from, time_to, kind, recovered) -> None:
        self.steps.record(lp, time_from, time_to, kind, detail="recovered" * recovered)

    def on_request(self, *args, **kwargs) -> list:
        self._joining.discard(args[0].name)
        if not self._joining and self.started is None:
            self.started = time.monotonic()
        replies = super().on_request(*args, **kwargs)
        for to, _, released in replies:
            # A member's reply is addressed (lp, member); its channels name its rank.
            lp = to[0] if isinstance(to, tuple) else to
            for ch, msgs in released.items():
                for seq, arrival in msgs:
                    self.steps.record(lp, arrival, arrival, "release", ch, seq)
        if self.final_clocks is not None and self.finished is None:
            self.finished = time.monotonic()
            self.done.set()
        return replies


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
    record nothing.
    """
    global _authority
    run = spec()
    if run is None:
        from atom.compass import parity

        if os.environ.get(parity.ENV):
            if config.runner_qualname != ATOM_RUNNER:
                raise ValueError(
                    f"{parity.ENV} is set and runner {config.runner_qualname!r} "
                    f"is named; only {ATOM_RUNNER} has a recording runner"
                )
            config.runner_qualname = parity.RUNNER
        return contextlib.nullcontext()
    pc = config.parallel_config
    widths = (pc.data_parallel_size, config.pipeline_parallel_size)
    if widths[1] != 1 or config.enable_rapidserve:
        raise ValueError(
            f"a simulated run here is one engine LP at pipeline width 1 without "
            f"RapidServe, got (dp, pp)={widths}, rapidserve={config.enable_rapidserve}"
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
    if dp > 1 and config.fake_eplb:
        raise ValueError(
            "--fake-eplb may start fewer engines than the data-parallel ranks "
            "the channel table declares, and a rank that never joins holds the run"
        )
    config.runner_qualname = COMPASS_RUNNER_QUALNAME
    lp = FRONTEND if side is None else LpId(f"{FRONTEND.name}-{side}")
    if standalone:
        return _start_on_leaving(_runtime(lp, run))
    _authority = _RecordingAuthority(run)
    serve(_authority, run["clock_endpoint"])
    # Not a daemon: the API server's main thread may die of the early end
    # before this thread has written the summary.
    threading.Thread(
        target=_write_on_early_end, args=(run, _authority), name="compass early end"
    ).start()
    return _start_on_leaving(_runtime(lp, run), _authority)


def engine(config):
    """Around the engine's construction in its process; a no-op context on a real run.

    At DP width above one, the process joins the engine LP as the member of its
    DP rank.
    """
    global _member
    run = spec()
    if run is None:
        return contextlib.nullcontext()
    from atom.utils.distributed import utils

    lp, side = ENGINE, _pd_side(config)
    if side is not None:
        lp = LpId(f"{ENGINE.name}-{side}")
        _bind_transfer(config, run)
    pc = config.parallel_config
    utils.LP_OF_RANK = dict.fromkeys(range(pc.data_parallel_size), lp)
    _member = f"dp{pc.data_parallel_rank}" if pc.data_parallel_size > 1 else None
    name = lp.name if _member is None else f"{lp.name}.{_member}"
    os.environ[NOW_ENV] = str(Path(run["out_dir"]) / f"now-{name}.bin")
    return _named(_start_on_leaving(_runtime(lp, run, _member, _PublishedClock)))


@contextlib.contextmanager
def _named(inner):
    """`inner`, with `NOW_ENV` set for the processes started in it."""
    try:
        with inner:
            yield
    finally:
        del os.environ[NOW_ENV]


def runner(model_runner) -> None:
    """Install the cost backend and the device readings from the run file."""
    run = spec()
    if run is None:
        return
    from transformers import PretrainedConfig

    from atom.compass.backends.analytic import Roofline, Tiered
    from atom.compass.backends.coarse import CoarseBackend, CoarseLaw
    from atom.compass.backends.geometry import KvGeometry, Parallelism
    from atom.compass.backends.shape import Coefficients, ShapeStubBackend
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
    keys = [k for k in ("coefficients", "law", "step_fit", "replay") if k in run]
    if len(keys) != 1:
        raise ValueError(
            "a run file prices its steps from one of coefficients, law, "
            f"step_fit and replay, and this one has {' and '.join(keys) or 'none'}"
        )
    replay = None
    if "replay" in run:
        from atom.compass.parity.replay import ReplayBackend

        dp_rank = config.parallel_config.data_parallel_rank
        backend = replay = ReplayBackend(run["replay"], dp_rank)
    if "law" in run:
        if tp != 1:
            raise ValueError(
                f"a law is fitted on one rank of tensor-parallel width 1, and this "
                f"run has tensor_parallel_size {tp}"
            )
        backend = CoarseBackend(CoarseLaw(run["law"]))
    elif "coefficients" in run:
        backend = ShapeStubBackend(
            Coefficients(**run["coefficients"]),
            Parallelism(tp_size=tp),
            stack_layers=config.hf_config.num_hidden_layers,
        )
    machine = MachineSpec.from_mapping(_width_keys(run["machine"]))
    bundle = None
    if "step_fit" in run:
        from atom.compass.run.step_fit import load

        backend, bundle = load(run["step_fit"], config, machine)
    config_json = PretrainedConfig.get_config_dict(config.model)[0]
    model = ModelTerms.from_declared_config(
        config.hf_config,
        parameter_count=run["parameter_count"],
        tp_size=tp,
        warmup_tokens=config.max_num_batched_tokens,
        config_json=config_json,
        measured_activations=machine.activations_for(
            (config_json.get("architectures") or [None])[0],
            _file_digest(config.model, "config.json"),
            tp,
        ),
    )
    if "roofline_floor_s" in run:
        kv = KvGeometry.from_hf_config(
            config.hf_config,
            block_size=config.kv_cache_block_size,
            parallelism=Parallelism(tp_size=tp),
            kv_dtype=config.kv_cache_dtype,
        )
        roofline = Roofline(
            machine,
            parameters=run["parameter_count"] // tp,
            weight_bytes=model.weights.nbytes,
            kv_token_bytes=kv.layers * kv.bytes_per_token_per_layer,
            host_floor_s=run["roofline_floor_s"],
        )
        backend = Tiered(backend, roofline)
    install_cost_backend(model_runner, backend)
    if NOW_ENV in os.environ:
        slot = np.memmap(os.environ[NOW_ENV], np.float64, "r", shape=(1,))
        host = run.get("host", {})
        ours, theirs = host.get("laws") and host_key(config), host.get("key") or {}
        if ours and theirs != ours:
            differ = [k for k in ours if theirs.get(k) != ours[k]]
            raise ValueError(
                f"the run file's host laws were fitted under another {differ}; "
                f"this run's key is {ours}"
            )
        model_runner.compass_host = replay or HostDeviceRecurrence(
            host.get("laws", {}), host.get("skip", ()), lambda: float(slot[0])
        )
        if replay is not None:
            replay.now = lambda: float(slot[0])
    model_runner.compass_abort = run.get("on_refusal") == "abort"
    reserved, captured = _graph_pool(model_runner, machine, model)
    readings = device_readings(
        machine, tp_width=tp, model=model, cudagraph_overhead=reserved
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
        model_runner.compass_step_fit = {**bundle, "describe": backend.describe()}


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
        return reserves(activation_bytes=model.activations.nbytes), captured
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


def _fingerprint(tok) -> str | None:
    """``sha256:<hex>`` of the ``tokenizer.json`` a fast tokenizer was loaded
    from, resolved from its ``name_or_path`` as loading did; None without one."""
    name = getattr(tok, "name_or_path", None)
    if not (tok.is_fast and name):
        return None
    return _file_digest(name, "tokenizer.json")


def host_key(config) -> dict:
    """What host laws are keyed by: this host's CPU, ATOM's source less
    `COMPASS_ONLY`, the served config.json and the engine config a step fit
    bundle is keyed by."""
    from atom.compass.run.step_fit import keyed

    cpu = re.search(
        r"^model name\s*:\s*(.+)$", Path("/proc/cpuinfo").read_text(), re.MULTILINE
    )
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
        for node in reversed(ast.parse(b"".join(lines)).body if skip else []):
            if getattr(node, "name", None) in skip:
                del lines[node.lineno - 1 : node.end_lineno]
        digest.update(f"{relative}\0".encode() + b"".join(lines))
    return {
        "host_cpu": cpu.group(1) if cpu else platform.processor(),
        "atom": digest.hexdigest(),
        "model": _file_digest(config.model, "config.json"),
        "engine": keyed(config),
    }


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


def _hand_in(run: dict, lp: str, reasons: list, **sections) -> None:
    """Write `lp`'s refusals, and its other sections, for the summary's writer."""
    out = Path(run["out_dir"]) / LP_FILE.format(lp)
    _write_whole(out, json.dumps({"refusals": reasons, **sections}))


def _workers_say(engine_core) -> dict:
    """The workers' command refusals, what their steps cost and their memory."""
    workers = engine_core.runner_mgr
    return {
        "refusals": list(workers.call_func("refused_commands", wait_out=True)),
        "priced": dict(workers.call_func("priced_steps", wait_out=True)),
        "memory": workers.call_func("predicted_memory", wait_out=True),
    }


def engine_done(engine_core, **sections) -> None:
    """After the engine's loop: leave the clock and hand in what the worker's
    steps cost and the memory it predicts, with `sections`, its command
    refusals and the engine's refused clock calls."""
    clock.close()
    run = spec()
    if run is not None:
        said = _workers_say(engine_core)
        rt = clock.installed()
        lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
        _hand_in(
            run,
            lp,
            said.pop("refusals") + rt.refusals,
            **said,
            **sections,
        )


@contextlib.contextmanager
def engine_loop(engine_core):
    """Around the engine's loop: `engine_done` once it leaves, which raises if
    that was before the finish. If either raises in a run, hand in the engine's
    refused clock calls and the exception, and what the workers say in 10 wall
    seconds unless one died. Under ``step_fit`` the workers are asked for the bundle
    before the loop, so a run that ends early still hands it in."""
    run, sections = spec(), {}
    if run is not None and "step_fit" in run:
        workers = engine_core.runner_mgr
        sections["step_fit"] = workers.call_func("step_fit_bundle", wait_out=True)
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
    """Whether the stopped server was a simulated run's finish. If it was, stop
    the engines, hand in the frontend's refusals and, with the authority
    co-hosted, write the step table and the run summary."""
    rt = clock.installed()
    if rt is None or rt.now != math.inf:
        return False
    llm_engine.close()
    run = spec()
    _hand_in(run, rt.me.name, rt.loop.executor.refusals + rt.refusals)
    if _authority is not None:
        _write_summary(run, _authority)
    return True


def _write_summary(run: dict, a: _RecordingAuthority, ended: bool = False) -> None:
    """Write the step table, then the run summary once every server LP has
    handed in its refusals, or after an early end, from those handed in so far."""
    out = Path(run["out_dir"])
    _write_whole(out / STEP_TABLE_FILE, a.steps.text())
    files = {n: out / LP_FILE.format(n) for n in _hand_ins(run)}
    deadline = time.monotonic() + HAND_IN_WAIT_S
    while not ended and (missing := [f.name for f in files.values() if not f.exists()]):
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no {', '.join(missing)} in {out} after {HAND_IN_WAIT_S} wall "
                "seconds, so the run summary would miss those LPs' refusals"
            )
        time.sleep(0.1)
    hand_ins = {n: json.loads(f.read_text()) for n, f in files.items() if f.exists()}
    reasons = [r for h in hand_ins.values() for r in h["refusals"]]
    priced = [h["priced"] for h in hand_ins.values() if "priced" in h]

    def total(key):
        return sum(p.get(key, 0) for p in priced)

    seconds = total("seconds")
    tally = RefusalTally.of(reasons, total("steps"))
    end = a.finished or time.monotonic()
    summary = RunSummary.of(a, end - (a.started or end), refusals=tally).as_record()
    schedule = summary["schedule"]
    schedule["predicted_seconds"] = seconds
    # Answers no measurement backs: diagnostics, never refusals.
    for species in ("analytical", "extrapolated"):
        schedule[species] = {
            "steps": total(f"{species}_steps"),
            "fraction_of_predicted_seconds": (
                total(f"{species}_seconds") / seconds if seconds else 0.0
            ),
        }
    # An engine whose workers died, or that never handed in, priced steps no
    # one counted: they are absent, not 0.
    if any(
        "priced" not in hand_ins.get(n, {}) for n in files if n.startswith("engine")
    ):
        schedule.update(predicted_seconds=None, analytical=None, extrapolated=None)
        schedule["refusals"].update(steps=None, fraction_of_steps=None)
    summary["configuration"] = configuration(run)
    summary["schema_version"] = SUMMARY_SCHEMA_VERSION
    summary["source_root"] = a.source_root.as_json()
    # Any refusal, of any source, leaves part of the run unsimulated; a run
    # that ended early handed in why.
    summary["coverage_report"] = bool(reasons)
    if fits := {n: h["step_fit"] for n, h in hand_ins.items() if "step_fit" in h}:
        summary["step_fit"] = fits
    _write_whole(out / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))


def authority(endpoint: str) -> None:
    """Serve the run file's authority at `endpoint` until the finish or an early
    end, then write its step table, and the run summary once the server LPs
    have handed in."""
    run = spec()
    a = _RecordingAuthority(run)
    serve(a, endpoint)
    _write_summary(run, a, _until_end(run, a))
