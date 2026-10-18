# SPDX-License-Identifier: MIT
"""What a capture declares in place of a device, so ATOM builds and steps on none.

Every substitution is process-global, and the `torch.cuda` ones must precede
aiter's import (its Triton configs read device properties at import), so a
capture owns its process.
"""

import contextlib
import os
import pathlib
import types

import torch

#: The devices a capture can declare, by arch: compute units, capability and
#: memory. ATOM and aiter branch on the arch and size grids by the CU count.
#: Nothing here reads a device.
DEVICES = {
    "gfx942": (80, (9, 4), 192 << 30),  # MI308X
    "gfx950": (256, (9, 5), 288 << 30),  # MI355X
}


def _props(arch):
    cus, (major, minor), memory = DEVICES[arch]
    return types.SimpleNamespace(
        multi_processor_count=cus,
        gcnArchName=arch,
        name=arch,
        major=major,
        minor=minor,
        total_memory=memory,
        warp_size=64,
        max_threads_per_multi_processor=2048,
        L2_cache_size=8 << 20,
        regs_per_multiprocessor=65536,
        shared_memory_per_block=65536,
    )


def on_wait(waiter, waited):
    """Told of each wait one stream makes on another; the recorder replaces it."""


class Stream:
    """A stream with an identity and no device."""

    created = 0

    def __init__(self, *args, **kwargs):
        self.stream_id = self.cuda_stream = Stream.created
        Stream.created += 1

    def wait_stream(self, other):
        on_wait(self, other)

    def wait_event(self, event):
        on_wait(self, event.stream)

    def record_event(self, event=None):
        event = event or Event()
        event.record(self)
        return event

    def synchronize(self):
        return None

    def query(self):
        return True


class Event:
    """An event remembers the stream it was recorded on, so a wait names it."""

    def __init__(self, *args, **kwargs):
        self.stream = None

    def record(self, stream=None):
        self.stream = stream or current[0]

    def wait(self, stream=None):
        on_wait(stream or current[0], self.stream)

    def synchronize(self):
        return None

    def query(self):
        return True

    def elapsed_time(self, other):
        raise RuntimeError("a capture runs no kernel and has no elapsed time")


main_stream = Stream()
current = [main_stream]
#: True while a stand-in collective runs: its functional collective is ATOM's
#: work, and the copy back into ATOM's tensor is the stand-in's own.
standing_in = [False]


@contextlib.contextmanager
def _stand_in():
    standing_in[0] = True
    try:
        yield
    finally:
        standing_in[0] = False


@contextlib.contextmanager
def _stream(stream):
    previous, current[0] = current[0], stream or current[0]
    try:
        yield
    finally:
        current[0] = previous


def install(tmpdir: str, arch: str) -> dict:
    """Declare the device of `arch`; say what."""
    if arch not in DEVICES:
        raise ValueError(f"no declared device for {arch}; declared: {sorted(DEVICES)}")
    cuda = {
        # True: on False ATOM takes paths no deployment runs. The fake mode is
        # told otherwise by `driverless_mode`.
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "_lazy_init": lambda *a, **k: None,
        "get_rng_state": lambda *a, **k: torch.zeros(16, dtype=torch.uint8),
        "set_rng_state": lambda *a, **k: None,
        "get_device_properties": lambda *a, **k: _props(arch),
        "current_device": lambda: 0,
        "get_device_capability": lambda *a, **k: DEVICES[arch][1],
        "Stream": Stream,
        "Event": Event,
        "current_stream": lambda *a, **k: current[0],
        "default_stream": lambda *a, **k: main_stream,
        "stream": _stream,
        "set_device": lambda *a, **k: None,
        "synchronize": lambda *a, **k: None,
        "memory_stats": lambda *a, **k: {
            "allocated_bytes.all.current": 0,
            "allocated_bytes.all.peak": 0,
            "reserved_bytes.all.current": 0,
        },
        "memory_allocated": lambda *a, **k: 0,
        "reset_peak_memory_stats": lambda *a, **k: None,
        "empty_cache": lambda *a, **k: None,
        "is_current_stream_capturing": lambda: False,
    }
    for name, value in cuda.items():
        setattr(torch.cuda, name, value)
    # ATOM's H2D publication reads the current stream's id off `torch._C`.
    torch._C._cuda_getCurrentStream = lambda *a: (current[0].stream_id, 0, 1)
    # No checkpoint is read, so a parameter's fill is arbitrary either way, and
    # the in-place RNG fills ask the device for a generator.
    torch.Tensor.uniform_ = torch.Tensor.normal_ = lambda self, *a, **k: self

    from torch._subclasses import fake_tensor

    # A device-tagged fake primes a real CUDA context; nothing here runs backward.
    fake_tensor.init_gpu_context = lambda device: None
    flags = {
        # The default custom vocab gather needs the device communicator.
        "ATOM_USE_CUSTOM_ALL_GATHER": "0",
        # The packed H2D transport reads `.numpy()` and device pointers off one
        # staging block; direct copies each buffer with its own `copy_`.
        "ATOM_H2D_BACKEND": "direct",
        # aiter's CU count, which it would otherwise ask rocminfo for.
        "CU_NUM": str(DEVICES[arch][0]),
    }
    os.environ.update(flags)
    _declare_arch(tmpdir, arch)
    return {"cuda": sorted(cuda), "arch": arch, "flags": flags}


def driverless_mode(shape_env):
    """A fake mode told the device is absent, while ATOM is told it is there.

    `avoid_device_init` keeps `torch.tensor` on the host, makes the device guard
    a no-op and skips constant propagation across a device move, each of which
    would otherwise reach a driver.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    class _DriverlessFakeTensorMode(FakeTensorMode):
        avoid_device_init = property(lambda self: True)

    return _DriverlessFakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def _declare_arch(tmpdir: str, arch: str) -> None:
    """Answer aiter's `rocminfo` shell-out and Triton's target query with `arch`."""
    import triton
    from triton.backends.compiler import GPUTarget

    rocminfo = pathlib.Path(tmpdir) / "declared-bin" / "rocminfo"
    rocminfo.parent.mkdir(parents=True, exist_ok=True)
    rocminfo.write_text(
        f"#!/bin/sh\necho 'Agent 1\n  Name: {arch}\n  Device Type: GPU\n"
        f"  Compute Unit: {DEVICES[arch][0]}\n  ASIC Revision: 0'\n"
    )
    rocminfo.chmod(0o755)
    os.environ["PATH"] = f"{rocminfo.parent}:{os.environ.get('PATH', '')}"
    os.environ["GPU_ARCHS"] = arch

    class _DeclaredTarget:
        def get_current_target(self):
            return GPUTarget("hip", arch, 64)

        def __getattr__(self, name):
            raise RuntimeError(f"a device-free capture has no Triton driver: {name}")

    triton.runtime.driver.set_active(_DeclaredTarget())


def group(tp: int, dp: int) -> dict:
    """A TP x DP process group in one process, its transport declined.

    Every shard is the one a `tp`-way deployment holds and every collective ATOM
    issues is dispatched; aiter's `all_reduce_` answers from its fake. The
    legacy `c10d` collectives ATOM reaches have no fake kernel, so they go
    through their functional forms, which give the same shapes, and copy the
    result back under `standing_in`, since the legacy ones write in place.
    """
    import torch.distributed as dist
    from torch.distributed import _functional_collectives as functional
    from torch.testing._internal.distributed.fake_pg import FakeStore

    world = tp * dp
    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE=str(world))
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(map(str, range(world)))
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=world)

    from aiter.dist import parallel_state

    group, new_group = parallel_state.init_model_parallel_group, dist.new_group

    def no_transport(*args, **kwargs):
        kwargs.update(
            use_device_communicator=False, use_message_queue_broadcaster=False
        )
        return group(*args, **kwargs)

    def fake_backend(ranks=None, **kwargs):
        kwargs.pop("pg_options", None)
        return new_group(ranks, **{**kwargs, "backend": "fake"})

    parallel_state.init_model_parallel_group, dist.new_group = (
        no_transport,
        fake_backend,
    )
    try:
        parallel_state.init_distributed_environment(
            world_size=tp, rank=0, backend="fake", local_rank=0, data_parallel_size=dp
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, data_parallel_size=dp, backend="fake"
        )
    finally:
        parallel_state.init_model_parallel_group, dist.new_group = group, new_group

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        with _stand_in():
            gathered = functional.all_gather_tensor(input, 0, group or dist.group.WORLD)
            output.copy_(gathered.reshape(output.shape))

    def broadcast(tensor, src=0, group=None, async_op=False):
        with _stand_in():
            tensor.copy_(functional.broadcast(tensor, src, group or dist.group.WORLD))

    def all_to_all_single(output, input, *args, group=None, async_op=False):
        with _stand_in():
            sent = functional.all_to_all_single(input, None, None, group)
            output.copy_(sent)

    dist.all_gather_into_tensor, dist.broadcast = all_gather_into_tensor, broadcast
    dist.all_to_all_single = all_to_all_single
    dist.barrier = lambda *args, **kwargs: None  # no shape, nobody to wait for
    return {
        "world_size": parallel_state.get_tp_group().world_size,
        "dp": parallel_state.get_dp_group().world_size,
        "declined": ["device_communicator", "message_queue_broadcaster", "barrier"],
        "functional": ["all_gather_into_tensor", "broadcast", "all_to_all_single"],
    }
