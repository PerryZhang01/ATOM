# SPDX-License-Identifier: MIT
"""ATOM's own runner, reading its rank's memory as it starts: the runner
`compass measure memory` starts the engine with.

Kept apart from the command because importing `ModelRunner` needs a driver.
"""

import json
import os
from pathlib import Path

import torch

from atom.compass.artifacts import member_name
from atom.compass.drive.measure_memory import OUT_ENV
from atom.compass.run import memory
from atom.compass.run.step_grid import BOUNDS, keyed
from atom.model_engine.model_runner import ModelRunner


def _allocated(which: str) -> int:
    return torch.cuda.memory_stats()[f"allocated_bytes.all.{which}"]


def _storage_bytes(tensors) -> int:
    """The bytes of the storages `tensors` hold, each storage once."""
    held = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in tensors
        if t.is_cuda
    }
    return sum(held.values())


class MemoryReadingModelRunner(ModelRunner):
    """`ModelRunner`, unchanged but for a second warmup, with its rank's memory
    read through its start into ``readings.<rank>.json`` in the directory
    `OUT_ENV` names: allocated bytes once the weights load, less their
    storages; allocated bytes where the KV budget reads the device, less
    those; that warmup's peak above them over its tokens; the budget's
    `non_torch` and total; and the reserved bytes graph capture adds."""

    def _build_and_load_model(self, model_class):
        super()._build_and_load_model(model_class)
        model = self.model
        self._read = {
            "loaded": _allocated("current"),
            "weights": _storage_bytes([*model.parameters(), *model.buffers()]),
        }

    def _maybe_warmup(self):
        # On a cold compile cache the compiler's own tensors can set the first
        # warmup's peak, so the readings take a second, warm one's.
        super()._maybe_warmup()
        tokens, forward = [], self.forward

        def counted(batch, *args, **kwargs):
            tokens.append(batch.total_tokens_num)
            return forward(batch, *args, **kwargs)

        self.forward = counted
        try:
            super()._maybe_warmup()
        finally:
            del self.forward
        self._read["warmup_tokens"] = sum(tokens)

    def _read_device_memory(self):
        readings = super()._read_device_memory()
        self._read |= {
            "total": readings.total,
            "non_torch": readings.non_torch,
            "current": _allocated("current"),
            "peak": _allocated("peak"),
        }
        return readings

    def get_num_blocks(self):
        reply = super().get_num_blocks()
        self._read["num_kvcache_blocks"] = reply["num_kvcache_blocks"]
        self._write(None)
        return reply

    def capture_cudagraph(self):
        reply = super().capture_cudagraph()
        self._write(reply[2])
        return reply

    def _write(self, pool) -> None:
        read, config = self._read, self.config
        engine = keyed(config) | {name: getattr(config, name) for name in BOUNDS}
        engine["num_kvcache_blocks"] = read["num_kvcache_blocks"]
        readings = {
            "capacity_bytes": read["total"],
            "driver_and_collective_reserve_bytes": read["non_torch"],
            "allocator_retained_after_load_bytes": read["loaded"] - read["weights"],
            "persistent_forward_buffer_bytes": read["current"] - read["loaded"],
            "activation_bytes": read["peak"] - read["current"],
            "warmup_tokens": read["warmup_tokens"],
            "cudagraph_pool_bytes": pool,
        }
        name = member_name(
            memory.STEM, memory.coords(config, self.rank), memory.EXTENSION
        )
        Path(os.environ[OUT_ENV], name).write_text(
            json.dumps(
                {
                    "key": memory.key_fields(config),
                    "engine": engine,
                    "readings": readings,
                }
            )
        )
