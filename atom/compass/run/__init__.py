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

Run file keys: ``clock_endpoint``, ``bound_s`` (finite), ``admission_path``,
``ipc_s``, ``stream_s``, ``coefficients`` (a `Coefficients` mapping, priced by
`ShapeStubBackend`) or ``law`` (a `CoarseLaw` mapping, priced by `CoarseBackend`),
``machine`` (a machine spec mapping), ``parameter_count``, and ``out_dir``,
where the step table, the run summary and each server LP's refusals go: an
engine's are the worker's refused commands and the engine's refused clock
calls, a frontend's its executor's refused jobs and its refused clock calls.
Each engine's worker writes the memory it predicts there too, for
``python -m atom.compass.memory.check``. The authority removes an earlier
run's files there as it starts. An optional ``data_parallel_size``, 1 when
absent, is the deployment's DP width, which under DP-attention is its TP size
times its DP size; the traffic LP reads the channel table from the run file
alone. Each DP rank's engine process hands in its own refusals, as
``<lp>.<member>``.

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

import contextlib
import hashlib
import json
import math
import os
import threading
import time
from pathlib import Path

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
FRONTEND, ENGINE, TRAFFIC = LpId("frontend"), LpId("engine"), LpId("traffic")
ATOM_RUNNER = "atom.model_engine.model_runner.ModelRunner"
#: Keys that differ between two runs of one configuration, kept out of its name.
PER_RUN = ("clock_endpoint", "out_dir")
COMMANDS_FILE, MEMORY_FILE, STEP_TABLE_FILE, SUMMARY_FILE = (
    "commands-{}.json",  # by LP name, or <lp>.<member> for a member
    "memory-{}.json",  # by DP rank, after P or D on a prefill-decode side
    "step_table.txt",
    "summary.json",
)
#: Wall seconds the summary's writer waits for the server LPs' refusals.
HAND_IN_WAIT_S = 300.0

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
    run["clock_endpoint"] = os.environ.get(CLOCK_ENV, run.get("clock_endpoint"))
    return run


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
        self.steps = StepTable(json.dumps(configuration(run), sort_keys=True))
        self.started = self.finished = None
        self.done = threading.Event()
        self._joining = set(_servers(run))
        # A rerun into the same out_dir starts from none of the last run's files.
        out = Path(run["out_dir"])
        for name in (COMMANDS_FILE, MEMORY_FILE, STEP_TABLE_FILE, SUMMARY_FILE):
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


def _runtime(lp: LpId, run: dict, member: str | None = None) -> clock.LPRuntime:
    rt = clock.LPRuntime(
        lp, channel_table(run), connect(lp, run["clock_endpoint"], member)
    )
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
    return _start_on_leaving(_runtime(lp, run, _member))


def runner(model_runner) -> None:
    """Install the cost backend and the device readings from the run file."""
    run = spec()
    if run is None:
        return
    from transformers import PretrainedConfig

    from atom.compass.backends.coarse import CoarseBackend, CoarseLaw
    from atom.compass.backends.geometry import Parallelism
    from atom.compass.backends.shape import Coefficients, ShapeStubBackend
    from atom.compass.memory import ModelTerms, device_readings, predicts
    from atom.compass.memory.check import record
    from atom.compass.runner.overrides import (
        install_cost_backend,
        install_device_readings,
    )
    from atom.compass.spec import MachineSpec

    config = model_runner.config
    tp = config.tensor_parallel_size
    keys = [k for k in ("coefficients", "law") if k in run]
    if len(keys) != 1:
        raise ValueError(
            "a run file prices its steps from one of coefficients and law, and "
            f"this one has {' and '.join(keys) or 'neither'}"
        )
    if "law" in run:
        if tp != 1:
            raise ValueError(
                f"a law is fitted on one rank of tensor-parallel width 1, and this "
                f"run has tensor_parallel_size {tp}"
            )
        backend = CoarseBackend(CoarseLaw(run["law"]))
    else:
        backend = ShapeStubBackend(
            Coefficients(**run["coefficients"]),
            Parallelism(tp_size=tp),
            stack_layers=config.hf_config.num_hidden_layers,
        )
    install_cost_backend(model_runner, backend)
    machine = MachineSpec.from_mapping(_width_keys(run["machine"]))
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
    name = f"dp{config.parallel_config.data_parallel_rank}"
    if side := _pd_side(config):
        name = f"{side}.{name}"
    _write_whole(
        Path(run["out_dir"]) / MEMORY_FILE.format(name),
        json.dumps(record(readings, tokens=config.max_num_batched_tokens, pool=pool)),
    )


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


def _hand_in(run: dict, lp: str, reasons: list) -> None:
    """Write `lp`'s refusals for the summary's writer."""
    _write_whole(Path(run["out_dir"]) / COMMANDS_FILE.format(lp), json.dumps(reasons))


def engine_done(engine_core) -> None:
    """After the engine's loop: leave the clock and hand in the worker's command
    refusals and the engine's refused clock calls."""
    clock.close()
    run = spec()
    if run is not None:
        refused = engine_core.runner_mgr.call_func("refused_commands", wait_out=True)
        rt = clock.installed()
        lp = rt.me.name if _member is None else f"{rt.me.name}.{_member}"
        _hand_in(run, lp, list(refused) + rt.refusals)


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
        out = Path(run["out_dir"])
        _write_whole(out / STEP_TABLE_FILE, _authority.steps.text())
        _write_summary(run, _authority)
    return True


def _write_summary(run: dict, a: _RecordingAuthority) -> None:
    """Write the run summary once every server LP has handed in its refusals."""
    out = Path(run["out_dir"])
    files = [out / COMMANDS_FILE.format(n) for n in _hand_ins(run)]
    deadline = time.monotonic() + HAND_IN_WAIT_S
    while missing := [f.name for f in files if not f.exists()]:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"no {', '.join(missing)} in {out} after {HAND_IN_WAIT_S} wall "
                "seconds, so the run summary would miss those LPs' refusals"
            )
        time.sleep(0.1)
    reasons = [r for f in files for r in json.loads(f.read_text())]
    steps = sum(r.lp.startswith(ENGINE.name) and r.event == "TAR" for r in a.steps.rows)
    summary = RunSummary.of(
        a, a.finished - a.started, refusals=RefusalTally.of(reasons, steps)
    ).as_record()
    summary["configuration"] = configuration(run)
    # A refused command leaves ATOM's half of it applied, so what ran after it
    # was not the deployment simulated.
    summary["coverage_report"] = any(r.startswith("command:") for r in reasons)
    _write_whole(out / SUMMARY_FILE, json.dumps(summary, sort_keys=True, indent=1))


def authority(endpoint: str) -> None:
    """Serve the run file's authority at `endpoint` until the finish, then write
    its step table, and the run summary once the server LPs have handed in."""
    run = spec()
    a = _RecordingAuthority(run)
    serve(a, endpoint)
    a.done.wait()
    _write_whole(Path(run["out_dir"]) / STEP_TABLE_FILE, a.steps.text())
    _write_summary(run, a)
