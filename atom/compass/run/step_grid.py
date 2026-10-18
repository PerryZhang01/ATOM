# SPDX-License-Identifier: MIT
"""The step grid in a run: its bundle loaded from the artifact store by key, and
checked against the resolved ATOM config before the first step.

The run file's ``step_grid`` is the store's root and the bundle's key fields:

    {"store": <root directory>, "model": ..., "device": ...,
     "engine_config": ..., "grid": ..., "source_root": ...}

A bundle prices a TP group's whole step, so a run reads one member at any
width, ``params.dp0of1.pp0of1.pcp0of1.tp0of1.json``, the one `compass fit
steps` publishes beside its report:

    {"grid": <a StepGrid mapping>, "residual": <a Residual mapping>,
     "bounds": {"max_num_seqs": n, "max_num_batched_tokens": n,
                "max_model_len": n}}

The grid states its timing scope (`scope`) and what its points hold
(`value`); `step_grid_fit.bundle`, which writes a bundle, also writes the grid
definition and the residual's sources, which loading does not read.

Its publisher states the readings `conditions` takes. The engine-config reading
is `keyed`, as JSON: the ladder the runner replays (none under
``enforce_eager``), the KV block size and dtype, the quantization, the TP and
DP widths, DP attention, expert parallelism, TBO, the performance
environment and, under speculation, its method and draft width. Loading refuses, naming what differs:

- a run at data-parallel width above one, since every bundle is measured at one;
- a keyed condition, the grid's graph mode, a timing scope other than the
  whole ``ModelRunner.forward``, or a value other than ``measure steps``'
  bench window (`step_grid_fit.VALUE`), none included;
- the gates, or any reading, through `ArtifactStore.load`;
- a rung of the ladder between the grid's lowest and highest measured rungs
  that was not measured;
- a configured bound above the one the grid was planned for.

A decode corner past the KV pool the grid records is extrapolated at lookup,
and any other unmeasured corner refuses there, so neither needs a check here;
the run composes the step roofline below the backend for the steps it refuses.
The residual's recipe reads no chunk size, so none is keyed.

Grading a run reads its bundle with `read` instead, checking nothing against
the grading process: the run checked it against its own as it started, and its
summary records the digest of the bundle it priced with.
"""

import json
from collections.abc import Mapping
from hashlib import sha256
from pathlib import Path

from atom.benchmarks.results.metadata import performance_env
from atom.compass.artifacts import (
    ArtifactStore,
    Axis,
    Conditions,
    GateState,
    Key,
    Kind,
    Reading,
    Row,
    Topology,
    git_tree_root,
    machine_spec_key,
    member_name,
    module_root,
)
from atom.compass.artifacts.store import ENTRY_FILE
from atom.compass.backends.step_grid import StepGrid, StepGridBackend
from atom.compass.backends.step_grid_fit import VALUE
from atom.compass.backends.step_grid_plan import BOUNDS, SCOPE, keyed_of  # noqa: F401
from atom.compass.backends.step_grid_residual import Residual
from atom.compass.runner.overrides import schedulable_ladder

#: The bundle's member a run reads: a group's step has a single writer.
MEMBER = ("params", Topology().ranks()[0], "json")


def keyed(config) -> dict:
    """The conditions a bundle is measured under, as the resolved config holds them,
    and the performance environment of this process less the client's and the
    device selection, and under speculation its method and draft width."""
    spec = getattr(config, "speculative_config", None)
    drafts = {"speculative": [spec.method, spec.num_speculative_tokens]} if spec else {}
    return drafts | {
        "ladder": [] if config.enforce_eager else schedulable_ladder(config),
        "kv_cache_block_size": config.kv_cache_block_size,
        "kv_cache_dtype": config.kv_cache_dtype,
        "quantization": config.quant_config.compute_hash(),
        "tensor_parallel_size": config.tensor_parallel_size,
        "data_parallel_size": config.parallel_config.data_parallel_size,
        "enable_dp_attention": config.enable_dp_attention,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_tbo": config.enable_tbo,
        "enable_tbo_decode": config.enable_tbo_decode,
        "env": {
            k: v
            for k, v in sorted(performance_env().items())
            if not (k.startswith("AIPERF_") or k.endswith("_VISIBLE_DEVICES"))
        },
    }


def conditions(config, machine) -> Conditions:
    """This run's readings: `readings` of this process's ATOM tree, the served
    config.json and `keyed`."""
    from atom.compass.run import _file_digest

    return readings(
        machine,
        atom_src=Reading.of_source_root(git_tree_root("atom", module_root("atom"))),
        model=_file_digest(config.model, "config.json"),
        engine_config=json.dumps(keyed(config), sort_keys=True),
    )


def readings(machine, **inputs) -> Conditions:
    """The readings a run on `machine` takes: the stack, torch included, and the
    device the spec states, and this host's CPU; `inputs` state the ATOM tree,
    the model and the engine config."""
    from atom.compass.run import host_cpu

    return Conditions.of(
        **machine_spec_key(machine).values,
        torch=machine.value("device.software_pinned_to.torch"),
        host_cpu=host_cpu(),
        **inputs,
    )


def load(spec: Mapping, config, machine) -> tuple[StepGridBackend, dict]:
    """The backend the bundle at the run file's key prices with, and the
    bundle's key and digest for the run summary; or a refusal naming what
    differs."""
    dp = config.parallel_config.data_parallel_size
    if dp != 1:
        raise ValueError(
            "a step grid bundle is measured at data-parallel width 1, and this "
            f"run has data_parallel_size {dp}"
        )
    fields = dict(spec)
    store = ArtifactStore(fields.pop("store"))
    key = Key.of(Kind.STEP_GRID_BUNDLE, **fields)
    entry = store.read(key)
    try:
        member = json.loads(entry.read_member(*MEMBER))
        grid, residual = StepGrid(member["grid"]), Residual(member["residual"])
        bounds = {name: member["bounds"][name] for name in BOUNDS}
        row = dict(entry.fingerprints[Row.STEP_GRID_BUNDLE].cells)
        recorded = json.loads(row[Axis.ENGINE_CONFIG].value)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{key} does not load: {type(error).__name__}: {error}"
        ) from error
    ours = keyed(config)
    pairs = {
        "graph mode": (grid.graph_mode, "eager" if config.enforce_eager else "replay"),
        "timing scope": (grid.scope, SCOPE),
        "value": (member["grid"].get("value"), VALUE),
        **{name: (recorded.get(name), value) for name, value in ours.items()},
    }
    differ = [
        f"{name} {theirs!r} in the bundle and {here!r} here"
        for name, (theirs, here) in pairs.items()
        if theirs != here
    ]
    if differ:
        raise ValueError(
            f"{key} was measured under other conditions: " + "; ".join(differ)
        )
    store.load(key, conditions=conditions(config, machine), gates=GateState.of())
    if grid.graph_mode == "replay":
        measured = sorted(rung for rung, cells in grid.decode.items() if cells.points)
        unmeasured = [
            r
            for r in ours["ladder"]
            if measured and measured[0] < r < measured[-1] and r not in measured
        ]
        if unmeasured:
            raise ValueError(
                f"{key} measured the decode rungs {measured}, and not the "
                f"ladder's rungs {unmeasured} between them"
            )
    over = [
        f"{name} {getattr(config, name)} is above the {bounds[name]}"
        for name in BOUNDS
        if getattr(config, name) > bounds[name]
    ]
    if over:
        raise ValueError(
            f"this run's {'; '.join(over)} the grid of {key} was planned for"
        )
    return StepGridBackend(grid, residual, bundle=str(key)), {
        "key": key.values,
        "digest": entry.digest,
    }


def read(spec: Mapping, digest: str) -> StepGridBackend:
    """The backend of the bundle a run priced with: the entry in the store the
    run file names whose key is the run file's and whose digest is `digest`,
    the one the run summary records, and its member read back at the digest
    the entry states. The key is compared as the run file states it, so a
    bundle keyed by another tree's key fields reads. Refused, naming the key,
    when the store holds no such entry or its member has changed."""
    fields = dict(spec)
    folder = Path(fields.pop("store")) / Kind.STEP_GRID_BUNDLE.value
    named = f"{Kind.STEP_GRID_BUNDLE} {json.dumps(fields, sort_keys=True)}"
    found = {}
    for path in folder.glob(f"*/{ENTRY_FILE}"):
        raw = path.read_bytes()
        if (document := json.loads(raw))["key"] == fields:
            found[f"sha256:{sha256(raw).hexdigest()}"] = path.parent, document
    if digest not in found:
        raise ValueError(
            f"{folder} holds {len(found)} entries keyed {named}, digesting "
            f"{sorted(found)}; the run priced with the one digesting {digest}"
        )
    directory, document = found[digest]
    path = directory / member_name(*MEMBER)
    payload = path.read_bytes()
    if f"sha256:{sha256(payload).hexdigest()}" != document["members"].get(path.name):
        raise ValueError(f"{path} is not the member the entry of {named} digests")
    member = json.loads(payload)
    grid, residual = StepGrid(member["grid"]), Residual(member["residual"])
    return StepGridBackend(grid, residual, bundle=named)
