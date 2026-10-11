# SPDX-License-Identifier: MIT
"""The runner subclass, and the construction that owns no device memory.

These run without a driver, which is the reason the behaviour lives in
`atom.compass.runner.overrides` rather than beside the subclass: importing
`atom.model_engine.model_runner` runs aiter's architecture probe, which shells
out to `rocminfo` and raises where there is no GPU. So the bodies are exercised
directly and the binding to ATOM's class is read from the source instead.

The two facts worth naming, because both were surprises:

* The base class warms the model from inside its own `__init__`, and warmup
  drives a forward. A runner with no weights therefore cannot construct unless
  it skips warmup -- the forward that warmup would drive is its own. That chain
  is asserted over ATOM's source below, so this stops being true loudly rather
  than quietly.
* Allocation is checked by counting dispatched operators, not by reading a CUDA
  allocator, so the check is meaningful on a machine with no CUDA allocator to
  read. A control asserts the counter sees an allocation when there is one;
  without it a broken counter and a clean runner look identical.

The import scan walks the package, so a root that does not resolve yields
nothing and the parametrisation passes on zero cases. The non-empty case is
asserted on its own, because the neighbouring test that reads a path under the
same root is what reddens this file today -- an accident of who its sibling is,
not a statement about this scan.
"""

import ast
import dataclasses
import importlib
import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch
from test_runner_rpc_surface import _methods
from torch.utils._python_dispatch import TorchDispatchMode

from atom.compass.runner import COMPASS_RUNNER_QUALNAME
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    UnbuiltModel,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / "atom" / "compass" / "runner"
ENGINE = REPO / "atom" / "model_engine"
ATOM_RUNNER = ENGINE / "model_runner.py"

# The `atom` modules outside this package and `atom.compass.memory` that a
# module of this package imports at module scope, keyed by its path in the
# package; every other module, none.
ENGINE_READS = {
    "model_runner.py": {"atom.model_engine.model_runner"},
    "overrides.py": {"atom.compass.backends.base"},
    "projection.py": {
        "atom.compass.backends.shape",
        "atom.model_engine.sequence",
        "atom.utils.forward_context",
    },
}

# The methods that own memory or run a step, and so are the ones replaced.
OVERRIDDEN = {
    "_build_and_load_model",
    "_maybe_warmup",
    "_read_device_memory",
    "_estimate_cudagraph_overhead",
    "get_num_blocks",
    "allocate_kv_cache",
    "capture_cudagraph",
    "forward",
}


def _classes(path):
    tree = ast.parse(path.read_text())
    for n in ast.walk(tree):
        n.file = path.name
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _self_calls(node):
    """Names of `self.x(...)` calls anywhere inside a function definition."""
    return {
        n.func.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "self"
    }


class Recorder(TorchDispatchMode):
    """Every operator torch dispatches while this is entered."""

    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


class Runner(NonAllocatingRunner):
    """The overrides over a base that supplies only what they read."""

    def __init__(self):
        self.config = SimpleNamespace(
            num_kvcache_blocks=None,
            disagg_is_decode=False,
            speculative_config=None,
            eos_token_id=-1,
            stop_token_ids=[],
            pipeline_parallel_size=1,
            kv_transfer_config={},
        )


class SomeModel:
    """Stands in for the model class the engine hands the runner."""


# --- what is replaced, and how that differs from the runner already in tree ---


def test_the_replaced_methods_are_the_ones_that_own_memory_or_run_a_step():
    assert _methods(_classes(PACKAGE / "overrides.py")["NonAllocatingRunner"]) == (
        OVERRIDDEN
    )


def test_every_replaced_method_exists_on_the_class_being_replaced():
    """A rename upstream turns an override into a new method, silently."""
    assert OVERRIDDEN <= _methods(_classes(ATOM_RUNNER)["ModelRunner"])


def test_the_difference_from_the_in_tree_non_allocating_runner_is_five_methods():
    """`RapidServeModelRunner` is the working non-allocating runner in the tree.

    It overrides two things this one does not. `__init__`: it has to bind a
    method before the base runs, and this class has nothing to bind, so leaving
    `__init__` alone is what keeps every read lazy. `_kv_budget_extra_reserve`:
    it holds bytes back because a second process shares its GPU, and a runner
    whose readings describe a card with one tenant on it has nobody to hold
    anything back from -- the base's zero is the right answer here, not an
    override that was forgotten.

    This one overrides three things it does not. `capture_cudagraph`:
    RapidServe allocates no weights of its own but imports real ones over CUDA
    IPC, so it has a model to trace and keeps ATOM's capture. This runner has
    none, and ATOM's capture zeroes device buffers and opens a graph pool
    before it finds that out. `_read_device_memory` and
    `_estimate_cudagraph_overhead`: the two calls through which
    `get_num_blocks` reaches the device, which RapidServe leaves alone because
    it runs on the card it is sizing and this runner does not.

    Its `_init_weight_params_on_meta` is not in the difference because it is not
    an override -- it is a helper the base does not have.
    """
    base = _methods(_classes(ATOM_RUNNER)["ModelRunner"])
    template = _methods(_classes(ATOM_RUNNER)["RapidServeModelRunner"]) & base
    assert template - OVERRIDDEN == {"__init__", "_kv_budget_extra_reserve"}
    assert OVERRIDDEN - template == {
        "capture_cudagraph",
        "_read_device_memory",
        "_estimate_cudagraph_overhead",
    }


# --- construction allocates nothing -----------------------------------------


def test_construction_dispatches_no_operator_at_all():
    """The measured result: zero, against a base that reads a checkpoint."""
    runner = Runner()
    with Recorder() as recorded:
        runner._build_and_load_model(SomeModel)
        runner._maybe_warmup()
        runner.allocate_kv_cache(2048)
    assert recorded.ops == []


def test_the_recorder_sees_an_allocation_when_there_is_one():
    """Otherwise a dead recorder reads exactly like a clean runner."""
    with Recorder() as recorded:
        torch.empty(4)
    assert recorded.ops


def test_the_model_that_replaces_the_weights_owns_nothing():
    runner = Runner()
    runner._build_and_load_model(SomeModel)
    assert isinstance(runner.model, UnbuiltModel)
    assert list(runner.model.parameters()) == []
    assert list(runner.model.buffers()) == []


def test_calling_that_model_refuses_and_names_the_class_it_stands_for():
    with pytest.raises(RunnerRefusal, match="SomeModel"):
        UnbuiltModel(SomeModel)(torch.zeros(1))


# --- what stays resident after construction, and which names hold it ---------

# A value that builds or holds one of the ring's buffers, as it reads in the
# source. `self.forward_vars` is one because the ring is built out of it.
BUFFER_TERMS = ("CpuGpuBuffer", "torch.empty", "self.forward_vars")


def _method_def(node, name):
    """The `def name` in class definition *node*."""
    return next(
        n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name
    )


# Every name through which an attribute can be written without an assignment.
_WRITERS = ("setattr", "__setattr__", "__dict__", "vars")


def _on_self(node):
    return isinstance(node, ast.Attribute) and ast.unparse(node.value) == "self"


def _self_assigned(node, unreadable=()):
    """Every binding of `self.x` in *node*, as (name, the source of its value).

    Pairs, not a mapping keyed by name. A name can be assigned more than once --
    `forward_vars` is bound to the dict of buffers and later rebound to a slot
    of the ring it already holds -- and a mapping keeps only the last binding
    walked, which here is the rebind. The rebind names no buffer, so keying by
    name dropped `forward_vars` out of the holder set entirely. Keeping the
    pairs is what lets a name count as a holder when *any* of its bindings is.

    Read in every spelling that states the name: a target of `=`, including
    one inside a tuple or list, of an annotated or augmented `=`, and
    `setattr(self, "x", ...)`. Any other write -- a `for` or `with` target, and
    every other mention of `setattr`, `__setattr__`, `__dict__` or `vars`,
    however it is reached -- binds something this cannot read. Each of those
    must be listed in *unreadable* by its source text, or this refuses, so a
    spelling it cannot read fails here rather than leaving the set smaller
    than the class.
    """
    bound, read = set(), set()
    for n in ast.walk(node):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for target in n.targets if isinstance(n, ast.Assign) else [n.target]:
                for t in ast.walk(target):
                    if _on_self(t) and isinstance(t.ctx, ast.Store) and n.value:
                        bound.add((t.attr, ast.unparse(n.value)))
                    read.add(id(t))
        elif (
            isinstance(n, ast.Call)
            and ast.unparse(n.func) == "setattr"
            and [ast.unparse(a) for a in n.args[:1]] == ["self"]
            and len(n.args) == 3
            and isinstance(n.args[1], ast.Constant)
            and isinstance(n.args[1].value, str)
        ):
            bound.add((n.args[1].value, ast.unparse(n.args[2])))
            read.add(id(n.func))
    calls = {id(n.func): n for n in ast.walk(node) if isinstance(n, ast.Call)}
    unread = [
        (n.lineno, ast.unparse(calls.get(id(n), n)))
        for n in ast.walk(node)
        if id(n) not in read
        and (
            (_on_self(n) and isinstance(n.ctx, ast.Store))
            or getattr(n, "id", getattr(n, "attr", None)) in _WRITERS
        )
    ]
    assert sorted(text for _, text in unread) == sorted(unreadable), (
        f"{node.name} binds attributes on self in a form not read here: "
        + "; ".join(f"{node.file}:{line}: {text}" for line, text in sorted(unread))
        + f" -- listed as expected: {sorted(unreadable)}"
    )
    return bound


def test_exactly_two_attributes_of_the_runner_hold_the_ring():
    """Construction leaves the base's forward-vars ring resident, and named
    attributes of the runner hold it. Both holders are the base's, so a
    rename, or a third one bound anywhere in the class, fails here.

    The whole `ModelRunner` body is read, not the two methods that build the
    ring, because a holder bound in `__init__` is just as much a holder and an
    earlier draft of this test could not see one.

    Two nearby bindings are deliberately not in it. `self.forward_vars` is
    assigned twice: `_advance_forward_vars` rebinds the name to a slot of the
    ring it already holds, which is a rotation and not a fourth holder. And
    `self.tokenID_processor.input_ids` is a fourth *name* reaching a ring
    buffer, one attribute deeper -- true, and outside a claim about attributes
    on the runner.
    """
    # The three `setattr`s bind whatever the attention builders return --
    # the KV cache and the per-request state, by names the source never states.
    assigned = _self_assigned(
        _classes(ATOM_RUNNER)["ModelRunner"],
        unreadable=["setattr(self, name, value)"] * 3,
    )
    holders = {n for n, v in assigned if any(t in v for t in BUFFER_TERMS)}
    assert holders == {"forward_vars", "_fv_ring"}


def test_what_that_ring_costs_is_the_batch_budget_by_the_hidden_size():
    """The shape of the residue, read off the allocation the docstring names.

    Its dominant term, so a runner that allocates no weights still holds
    device memory that grows with the batch budget and the model's hidden
    size. The bytes need a device to measure and are not asserted; what is
    checkable here is which two numbers they are a product of.
    """
    allocate = _method_def(
        _classes(ATOM_RUNNER)["ModelRunner"], "allocate_forward_vars"
    )
    built = next(
        n.value
        for n in ast.walk(allocate)
        if isinstance(n, ast.Assign)
        and any(getattr(t, "attr", None) == "forward_vars" for t in n.targets)
    )
    outputs = dict(zip([k.value for k in built.keys], built.values))["outputs"]
    assert ast.unparse(outputs.func) == "torch.empty"
    assert ast.unparse(outputs.args[0]) == "self.max_num_batched_tokens"
    assert ast.unparse(outputs.args[-1]) == "hidden_size"


def test_the_overrides_bind_no_attribute_that_could_hold_a_tensor():
    """The half of the claim that is this package's own: it adds none.

    `model` registers no parameter and no buffer, `_token_stream` is the
    deferral bookkeeping `forward` builds on first use, and `kv_pool_sizing`
    is a `SizedKVPool` -- a count, a name-to-count table and the readings --
    built in a package whose whole import closure `test_kv_budget.py` holds
    free of any tensor library. `capture_sizes` and `capture_sizes_np` are the
    base's own ladder, rebound to a list of ints and a host int32 array as
    ATOM's capture rebinds them. Anything else appearing here is a tensor this
    class put on a device, which is the thing it exists not to do.
    """
    overrides = _classes(PACKAGE / "overrides.py")["NonAllocatingRunner"]
    bound = {n for n, _ in _self_assigned(overrides)}
    assert bound == {
        "model",
        "_token_stream",
        "kv_pool_sizing",
        "capture_sizes",
        "capture_sizes_np",
    }


# --- warmup drives a forward, which is why it is skipped ---------------------


def test_the_base_class_warms_the_model_from_init_and_warmup_runs_a_forward():
    """The chain that decides whether a weightless runner can construct."""
    runner = _classes(ATOM_RUNNER)["ModelRunner"]
    bodies = {n.name: n for n in runner.body if isinstance(n, ast.FunctionDef)}
    assert "_maybe_warmup" in _self_calls(bodies["__init__"])
    assert "warmup_model" in _self_calls(bodies["_maybe_warmup"])
    assert "forward" in _self_calls(bodies["warmup_model"])


def test_skipping_warmup_is_what_lets_construction_finish():
    """`forward` refuses, so a warmup that ran one would raise out of `__init__`."""
    runner = Runner()
    assert runner._maybe_warmup() is None
    with pytest.raises(RunnerRefusal, match="produces output"):
        runner.forward(object())


# --- the two answers that are refused, and the one that is arithmetic --------


def test_sizing_the_kv_pool_refuses_rather_than_inventing_a_block_count():
    """It sizes a pool now, but only from readings somebody installed.

    Without them the arithmetic would run against nothing, so the refusal
    stayed and only its reason changed -- and it names the call that supplies
    them rather than the absence.
    """
    with pytest.raises(RunnerRefusal, match="install_device_readings"):
        Runner().get_num_blocks()


def test_allocating_the_kv_cache_records_the_count_and_no_bytes():
    runner = Runner()
    assert runner.allocate_kv_cache(2048) is True
    assert runner.config.num_kvcache_blocks == 2048


# --- the seam: reachable without changing ATOM -------------------------------


def test_the_qualname_names_a_class_this_package_defines():
    module, _, name = COMPASS_RUNNER_QUALNAME.rpartition(".")
    path = REPO.joinpath(*module.split(".")).with_suffix(".py")
    assert name in _classes(path)


def test_selecting_a_runner_is_a_config_field_that_already_exists():
    from atom.config import Config

    field = {f.name: f for f in dataclasses.fields(Config)}["runner_qualname"]
    assert field.default == "atom.model_engine.model_runner.ModelRunner"


def test_the_worker_process_instantiates_whatever_that_field_names():
    assert "config.runner_qualname" in (ENGINE / "engine_core.py").read_text()
    assert (
        "resolve_obj_by_qualname(runner_qualname)"
        in (ENGINE / "async_proc.py").read_text()
    )


def _import_time_imports(source):
    """The imports module *source* takes at module scope.

    That approximates what a module imports when it is imported, which is the
    property the guard below states, and neither of the two obvious predicates
    states even the approximation. Top-level statements alone miss an import
    nested in a module-scope `try:`/`except ImportError:`, in a module-scope
    `if`, or in a class body, all of which run at import; walking every node
    counts one inside a function body, which does not. So this walks and prunes
    at `def`/`lambda`, and the six forms below are checked against what the
    interpreter actually runs, rather than against the rule of thumb stated
    here.

    Module scope is the approximation, and it is loose in both directions. It
    over-catches a module-scope branch the interpreter never takes --
    `if TYPE_CHECKING:`, which is how someone annotating a signature with an
    engine type will write it, and `if False:` and `if __name__ ==
    "__main__":` with it -- and there top-level statements alone are what
    agrees with the interpreter. It under-catches a module-scope call of a
    function defined in the same module, whose body does import at import
    time, and there walking every node is what agrees. Neither is a defect a
    stricter predicate could remove: whether a call runs is not decidable from
    the source. The trade is taken deliberately, because the import that would
    strand this package is the one written at module scope.

    Both halves are load-bearing. `overrides.forward` takes its single engine
    import at call time, on a worker that has imported the engine already, so
    counting it would forbid the reply this package exists to build. And the
    `try:`/`except ImportError:` form is the one case where this assertion is
    the only guard there is: collecting the package without a driver would not
    fail on it, because the `except` swallows the failure. The under-catch is
    blind in the same place: the same `except` inside a function body hides
    that import from driverless collection too.
    """
    imported = set()
    stack = list(ast.parse(source).body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Import):
            imported |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add("." * node.level + (node.module or ""))
        elif not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            stack.extend(ast.iter_child_nodes(node))
    return imported


# One sample per form an import can take, each importing a probe module of its
# own. Which of them run at import is not asserted here: the test writes the
# sample out, imports it, and reads the answer off `sys.modules`.
IMPORT_FORMS = {
    "module_scope": "import {probe}\n",
    "module_scope_try": (
        "try:\n    import {probe}\nexcept ImportError:\n    {probe} = None\n"
    ),
    "module_scope_if": "if True:\n    import {probe}\n",
    "class_body": "class C:\n    import {probe}\n",
    "method_body": "class C:\n    def m(self):\n        import {probe}\n",
    "function_body": "def f():\n    import {probe}\n",
}


@pytest.mark.parametrize("form", sorted(IMPORT_FORMS))
def test_the_guard_reads_the_imports_that_run_at_import(form, tmp_path, monkeypatch):
    """The predicate is checked against the interpreter, both ways.

    The probe module is named after the form and imported nowhere else, so it
    reaches `sys.modules` only if importing the sample ran that import -- which
    is the property the guard is trying to state, rather than a rule of thumb
    about where imports are allowed to sit.
    """
    probe, sample = f"probe_{form}", f"sample_{form}"
    (tmp_path / f"{probe}.py").write_text("")
    source = IMPORT_FORMS[form].format(probe=probe)
    (tmp_path / f"{sample}.py").write_text(source)
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, probe, raising=False)
    monkeypatch.delitem(sys.modules, sample, raising=False)
    importlib.import_module(sample)
    assert (probe in _import_time_imports(source)) == (probe in sys.modules)
    sys.modules.pop(sample, None)
    sys.modules.pop(probe, None)


def _runner_modules():
    # rglob, so a module added under the package is covered the day it lands.
    return sorted(PACKAGE.rglob("*.py"))


def test_the_package_was_found():
    assert _runner_modules(), f"no modules under {PACKAGE}"


def test_the_guard_finds_nothing_when_the_root_moves(monkeypatch, tmp_path):
    """The control for the guard above, which otherwise only proves it is alive.

    A guard that has never been seen failing is a liveness check: it passes
    today because the package is where it always was. Pointed at a root that
    does not resolve it must come back empty -- and the sibling module one level
    out is there so a derivation that widened past its own root would be caught
    here instead of quietly keeping the parametrisation non-empty.
    """
    (tmp_path / "sibling.py").write_text("")
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path / "moved")
    assert not _runner_modules()


def test_the_walk_returns_every_module_under_the_root(monkeypatch, tmp_path):
    """Non-empty says the walk found something; this, every module of a tree built here.

    The tree is built here, so the expected set does not move when the package
    gains a module. A walk that is not recursive misses `sub/b.py`, and any
    narrowing or slice that drops a module of the built tree fails the same way.
    """
    for rel in ("__init__.py", "a.py", "sub/b.py"):
        (tmp_path / rel).parent.mkdir(exist_ok=True)
        (tmp_path / rel).write_text("")
    monkeypatch.setitem(globals(), "PACKAGE", tmp_path)
    modules = {str(p.relative_to(tmp_path)) for p in _runner_modules()}
    assert modules == {"__init__.py", "a.py", "sub/b.py"}


@pytest.mark.parametrize(
    "path",
    _runner_modules(),
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_only_the_binding_module_reaches_the_engine(path):
    """Everything else stays runnable where the engine cannot be imported.

    `atom.compass.memory` joins the exemption because `overrides` now imports
    it at module scope for the readings the KV budget runs against, and
    `test_kv_budget.py` asserts that package's whole import closure -- not one
    level of its import statements -- reaches no tensor library and no engine.
    The exemption is exactly the two packages whose closure something asserts;
    widening it to `atom.compass` would exempt packages nothing has checked.

    The modules that read named modules outside the exemption are listed in
    `ENGINE_READS` by the exact set. `model_runner` binds ATOM's runner.
    `overrides` reads the cost backend's base class, from a package whose own
    test keeps it free of the engine. `projection` reads a scheduled batch, a
    sequence's kind and the dispatch rule; none of those reaches
    `atom.model_engine.model_runner`, and
    `test_every_other_module_imports_here` runs the import to measure it.
    """
    # A relative import is resolved against this module's package first, so
    # one that climbs out of the package is read as the module it names.
    package = path.relative_to(REPO).parent.parts
    imported = {
        (
            ".".join([*package[: len(package) + 1 - level], name[level:]]).strip(".")
            if (level := len(name) - len(name.lstrip(".")))
            else name
        )
        for name in _import_time_imports(path.read_text())
    }
    engine = {m for m in imported if m.split(".")[0] == "atom"} - {
        m
        for m in imported
        if m.startswith(("atom.compass.runner", "atom.compass.memory"))
    }
    assert engine == ENGINE_READS.get(str(path.relative_to(PACKAGE)), set())


def test_every_module_the_walk_returns_is_a_case():
    """The cases are compared to the walk, not only the walk to its tree.

    A slice or filter where the walk is handed to the parametrisation drops
    cases while the walk itself still returns every module.
    """
    (mark,) = test_only_the_binding_module_reaches_the_engine.pytestmark
    assert mark.args[1] == _runner_modules()


@pytest.mark.parametrize(
    "path",
    [p for p in _runner_modules() if p != PACKAGE / "model_runner.py"],
    ids=lambda p: str(p.relative_to(PACKAGE)),
)
def test_every_other_module_imports_here(path):
    """The measurement the scan above can only approximate.

    A source scan cannot see a driver reached through an import of an import,
    and the scan's own predicate is blind to a module-scope `try:`/`except
    ImportError:` in a module it does not read. Running the import is blind to
    neither. On the tier that has no driver this is the whole property; on a
    machine that has one it degrades to a syntax and resolution check, which is
    why the scan above is kept as well.
    """
    module = ".".join(path.relative_to(REPO).with_suffix("").parts)
    assert importlib.import_module(module.removesuffix(".__init__")) is not None
