# SPDX-License-Identifier: MIT
"""The memory readings in a run: the rank's member of the configuration's
`memory_readings` entry, loaded from the artifact store by key.

The run file's ``store`` is the artifact store's root. The key is the
machine spec's digest and `key_fields`, read off the resolved ATOM config: the
served config.json's digest, the parallel layout, and as sorted JSON the
engine config the readings move with, `step_grid.keyed`'s fields and the
scheduler bounds. `compass measure memory`
publishes one member per rank, ``readings.<rank>.json``, holding
`ENTRY_FIELDS`; each rank reads its own. A configuration with no entry
refuses, naming its key and the fields the nearest entry differs in, and so
does an entry `ArtifactStore.load` refuses under `conditions`.
"""

import json

from atom.compass.artifacts import (
    ArtifactStore,
    Conditions,
    GateState,
    Key,
    Kind,
    RankCoords,
    Reading,
    Topology,
    git_tree_root,
    machine_spec_digest,
    machine_spec_key,
    module_root,
)
from atom.compass.memory.readings import MemoryReadings

STEM, EXTENSION = "readings", "json"


def topology(config) -> Topology:
    """The widths of the deployment `config` resolved to."""
    return Topology(
        dp=config.parallel_config.data_parallel_size,
        pp=config.pipeline_parallel_size,
        pcp=config.prefill_context_parallel_size,
        tp=config.tensor_parallel_size,
    )


def coords(config, rank: int) -> RankCoords:
    """Where the runner of rank `rank` of its engine sits: its DP rank, its
    stage, and in the stage PCP-major, as ATOM's rank arithmetic orders them."""
    pc, width = config.parallel_config, config.tp_world_size
    return RankCoords(
        topology(config),
        dp=pc.data_parallel_rank,
        pp=pc.pipeline_parallel_rank,
        pcp=rank // width,
        tp=rank % width,
    )


def key_fields(config) -> dict:
    """The key of `config`'s readings, less the machine spec's digest."""
    from atom.compass.run import _file_digest
    from atom.compass.run.step_grid import BOUNDS, keyed

    engine = keyed(config) | {name: getattr(config, name) for name in BOUNDS}
    return {
        "model": _file_digest(config.model, "config.json"),
        **topology(config).widths,
        "enable_expert_parallel": config.enable_expert_parallel,
        "enable_dp_attention": config.enable_dp_attention,
        "decode_context_parallel_size": config.decode_context_parallel_size,
        "engine_config": json.dumps(engine, sort_keys=True),
    }


def conditions(fields: dict, **platform) -> Conditions:
    """The readings an entry keyed by `fields` is published or loaded under:
    `platform`'s device, software stack and torch, this process's ATOM tree
    and host CPU, and the key's model and engine config."""
    from atom.compass.run import host_cpu

    return Conditions.of(
        **platform,
        atom_src=Reading.of_source_root(git_tree_root("atom", module_root("atom"))),
        model=fields["model"],
        host_cpu=host_cpu(),
        engine_config=fields["engine_config"],
    )


def load(store: str, config, machine, rank: int) -> MemoryReadings:
    """Rank `rank`'s readings of `config` on the platform `machine` states, or
    a refusal naming what is missing or differs."""
    fields = key_fields(config)
    key = Key.of(
        Kind.MEMORY_READINGS, machine_spec=machine_spec_digest(machine), **fields
    )
    platform = machine_spec_key(machine).values
    platform["torch"] = machine.value("device.software_pinned_to.torch")
    entry = ArtifactStore(store).load(
        key, conditions=conditions(fields, **platform), gates=GateState.of()
    )
    member = entry.read_member(STEM, coords(config, rank), EXTENSION)
    return MemoryReadings.from_member(json.loads(member), f"{key.kind} {key.token()}")
