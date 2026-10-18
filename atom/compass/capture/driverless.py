# SPDX-License-Identifier: MIT
"""What a capture declares in place of a device, so ATOM builds and steps on none.

Every substitution is process-global, and the `torch.cuda` ones must be in
place before aiter is imported (its Triton configs read device properties at
import), so a capture owns its process.
"""

import contextlib
import os
import pathlib

import torch

# The declared device: an MI308X. Nothing here reads a device.
ARCH = "gfx942"
CU_COUNT = 80
CAPABILITY = (9, 4)
TOTAL_MEMORY_BYTES = 192 << 30


class Stream:
    """A stream with an identity and no device; each wait is passed to `on_wait`."""

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
        raise RuntimeError("a capture runs no kernel and has no elapsed time to report")


def on_wait(waiter, waited):
    """Replaced by the recorder for the forward it traces."""


main_stream = Stream()
current = [main_stream]


@contextlib.contextmanager
def _stream(stream):
    previous, current[0] = current[0], stream or current[0]
    try:
        yield
    finally:
        current[0] = previous


class _Props:
    multi_processor_count = CU_COUNT
    gcnArchName = name = ARCH
    major, minor = CAPABILITY
    total_memory = TOTAL_MEMORY_BYTES
    warp_size = 64
    max_threads_per_multi_processor = 2048
    L2_cache_size = 8 << 20
    regs_per_multiprocessor = 65536
    shared_memory_per_block = 65536


def install(tp: int, tmpdir: str) -> dict:
    """Declare the device, the arch and a peerless group of width `tp`.

    Returns what was declared, for the artifact's record.
    """
    cuda = {
        # True: ATOM branches on the device throughout and on False takes paths
        # no deployment runs. `driverless_mode` tells the fake mode otherwise.
        "is_available": lambda: True,
        "device_count": lambda: 1,
        "_lazy_init": lambda *a, **k: None,
        "get_rng_state": lambda *a, **k: torch.zeros(16, dtype=torch.uint8),
        "set_rng_state": lambda *a, **k: None,
        "get_device_properties": lambda *a, **k: _Props(),
        "current_device": lambda: 0,
        "get_device_capability": lambda *a, **k: CAPABILITY,
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
        "is_current_stream_capturing": lambda: False,
    }
    for name, value in cuda.items():
        setattr(torch.cuda, name, value)
    # ATOM's H2D publication reads the current stream's id off `torch._C`.
    torch._C._cuda_getCurrentStream = lambda *a: (current[0].stream_id, 0, 1)

    # No checkpoint is read, so a parameter's fill is arbitrary either way, and
    # the in-place RNG fills ask the device for a generator.
    for name in ("uniform_", "normal_"):
        setattr(torch.Tensor, name, lambda self, *a, **k: self)

    from torch._subclasses import fake_tensor

    # A device-tagged fake primes a real CUDA context; nothing here runs backward.
    fake_tensor.init_gpu_context = lambda device: None

    flags = {
        # The default custom vocab gather needs the device communicator, which
        # opens a rendezvous with absent ranks; the non-custom one gathers the
        # same shapes.
        "ATOM_USE_CUSTOM_ALL_GATHER": "0",
        # The packed H2D transport reads `.numpy()` and device pointers off one
        # staging block; direct copies each buffer with its own `copy_`.
        "ATOM_H2D_BACKEND": "direct",
    }
    os.environ.update(flags)
    return {
        "cuda": sorted(cuda),
        "arch": _declare_arch(tmpdir),
        "flags": flags,
        "group": _build_group(tp),
    }


def driverless_mode(shape_env):
    """A fake mode told the device is absent, while ATOM is told it is there.

    `avoid_device_init` keeps `torch.tensor` on the host, swaps the device guard
    for a no-op and skips constant propagation across a device move, each of
    which would otherwise reach a driver.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    class _DriverlessFakeTensorMode(FakeTensorMode):
        @property
        def avoid_device_init(self):
            return True

    return _DriverlessFakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True)


def _declare_arch(tmpdir: str) -> str:
    """Answer aiter's `rocminfo` shell-out and Triton's target query with `ARCH`."""
    import triton
    from triton.backends.compiler import GPUTarget

    binpath = pathlib.Path(tmpdir) / "declared-bin"
    binpath.mkdir(parents=True, exist_ok=True)
    rocminfo = binpath / "rocminfo"
    rocminfo.write_text(f'#!/bin/sh\necho "  Name:                    {ARCH}"\n')
    rocminfo.chmod(0o755)
    os.environ["PATH"] = f"{binpath}:{os.environ.get('PATH', '')}"
    os.environ["GPU_ARCHS"] = ARCH

    class _DeclaredTarget:
        def get_current_target(self):
            return GPUTarget("hip", ARCH, 64)

        def __getattr__(self, name):
            raise RuntimeError(
                f"a device-free capture has no Triton driver; read {name}"
            )

    triton.runtime.driver.set_active(_DeclaredTarget())
    return ARCH


def _build_group(tp: int) -> dict:
    """A process group of width `tp` in one process, its transport declined.

    The width is real, so every shard is the one a `tp`-way deployment holds and
    every collective ATOM issues is dispatched; aiter's `all_reduce_` answers
    from its fake implementation. The legacy `c10d` collectives have no fake
    kernel, so the two ATOM reaches go through their functional forms, which
    give the same shapes.
    """
    import torch.distributed as dist
    from torch.distributed import _functional_collectives as functional
    from torch.testing._internal.distributed.fake_pg import FakeStore

    os.environ.update(RANK="0", LOCAL_RANK="0", WORLD_SIZE=str(tp))
    os.environ["HIP_VISIBLE_DEVICES"] = ",".join(map(str, range(tp)))
    dist.init_process_group(backend="fake", store=FakeStore(), rank=0, world_size=tp)

    from aiter.dist import parallel_state

    original_group, original_new_group = (
        parallel_state.init_model_parallel_group,
        dist.new_group,
    )

    def decline_transport(*args, **kwargs):
        kwargs.update(
            use_device_communicator=False, use_message_queue_broadcaster=False
        )
        return original_group(*args, **kwargs)

    def decline_backend(ranks=None, **kwargs):
        kwargs["backend"] = "fake"
        kwargs.pop("pg_options", None)
        return original_new_group(ranks, **kwargs)

    parallel_state.init_model_parallel_group = decline_transport
    dist.new_group = decline_backend
    try:
        parallel_state.init_distributed_environment(
            world_size=tp, rank=0, backend="fake", local_rank=0
        )
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp, backend="fake"
        )
    finally:
        parallel_state.init_model_parallel_group = original_group
        dist.new_group = original_new_group

    def all_gather_into_tensor(output, input, group=None, async_op=False):
        gathered = functional.all_gather_tensor(input, 0, group or dist.group.WORLD)
        output.copy_(gathered.reshape(output.shape))

    def broadcast(tensor, src=0, group=None, async_op=False):
        tensor.copy_(functional.broadcast(tensor, src, group or dist.group.WORLD))

    dist.all_gather_into_tensor = all_gather_into_tensor
    dist.broadcast = broadcast
    # No shape and no bytes; there is nobody to wait for.
    dist.barrier = lambda *args, **kwargs: None
    return {
        "world_size": parallel_state.get_tp_group().world_size,
        "declined": ["device_communicator", "message_queue_broadcaster", "barrier"],
        "functional": ["all_gather_into_tensor", "broadcast"],
    }
