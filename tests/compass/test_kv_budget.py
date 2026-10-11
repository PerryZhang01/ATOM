# SPDX-License-Identifier: MIT
"""The seam ATOM's KV budget is sized through, and the record it produces.

Everything here runs without a driver. The block count itself cannot: sizing a
pool means executing `ModelRunner.get_num_blocks`, which means importing
`atom.model_engine.model_runner`, which runs aiter's architecture probe and
raises where there is no GPU. That import is not stubbed around. The tree
already carries an aiter stub for this exact problem (`tests/aiter_stub.py`),
and it covers three module names -- measured 2026-09-22 in `xiaobizh_n18_cpu`,
`atom/model_engine/model_runner.py` imports a fourth at module scope,
`aiter.dist.parallel_state`, and the import fails there. Widening that stub is
the thing `tests/conftest.py`'s own header records as having silently stopped
four test modules from running, so the binding lives in
`test_kv_budget_engine.py` and runs where there is a driver.

What this tier holds is the half that decides whether the binding is right
rather than whether it runs: that ATOM's budget method reaches the device
through one call and through no other, that the four figures are handed back
under the names they were taken by, that four named proxies for that
arithmetic are not written down in this package, that the refusals it carries
read fields ATOM declares, and that the count carries what sized it.

The spec document and the model config are imported from `test_memory_readings`
rather than copied. The numbers here are the numbers there; a second MI355X
written out in this file would let the two drift and still pass.
"""

import ast
import copy
import dataclasses
import json
import os
import pathlib
import subprocess
import sys
from types import SimpleNamespace

import pytest
from test_memory_readings import (  # one spec document and one model, not two
    CONFIG_JSON,
    DOCUMENT,
    EXPECTED,
    readings_at,
)
from transformers import PretrainedConfig

import atom.compass.memory as memory_package
from atom.compass.backends.geometry import KvGeometry, Parallelism
from atom.compass.memory import MemoryRefusal, SizedKVPool, reserves
from atom.compass.memory.readings import DeviceReadings
from atom.compass.runner.overrides import (
    NonAllocatingRunner,
    RunnerRefusal,
    install_device_readings,
)
from atom.compass.spec import MachineSpec
from atom.model_ops.attentions.sub_pool_spec import page_pool, plan_pools
from atom.models.utils import get_pp_indices

# Located through the package as the suite imported it, never by walking up
# from this file: if `atom` resolves from another root -- an installed copy, a
# PYTHONPATH ahead of the tree, a staged snapshot -- a path-derived root would
# be read here while every other test imports the other one, and pass.
COMPASS = pathlib.Path(memory_package.__file__).parent.parent
REPO = COMPASS.parent.parent
ATOM_RUNNER = REPO / "atom" / "model_engine" / "model_runner.py"
BLOCK_SIZE = 64


@pytest.fixture(scope="module")
def qwen():
    return PretrainedConfig.from_dict(
        json.loads(CONFIG_JSON.read_text())["text_config"]
    )


@pytest.fixture(scope="module")
def spec():
    return MachineSpec.from_mapping(copy.deepcopy(DOCUMENT))


def _classes(path):
    tree = ast.parse(path.read_text())
    return {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}


def _method(class_node, name):
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{class_node.name} has no {name}")


def _device_reads(node):
    """Every `torch.cuda.X` this function definition names, at any depth."""
    return {
        n.attr
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute)
        and isinstance(n.value, ast.Attribute)
        and isinstance(n.value.value, ast.Name)
        and n.value.value.id == "torch"
        and n.value.attr == "cuda"
    }


# --- the seam: one call down, and nothing beside it --------------------------


def test_the_budget_method_names_no_device_read_of_its_own():
    """What makes "substitute the readings, reuse the arithmetic" possible.

    A fifth read added to that method upstream lands here as a failure rather
    than as a block count that quietly came off the host's own card.
    """
    runner = _classes(ATOM_RUNNER)["ModelRunner"]
    assert _device_reads(_method(runner, "get_num_blocks")) == set()


def test_the_extracted_seam_holds_the_reads_it_took_over():
    """And the control: the walker above finds reads where there are some.

    Without it, a walker that matched nothing would report the same clean
    result for a method full of device reads as for one with none.
    """
    runner = _classes(ATOM_RUNNER)["ModelRunner"]
    assert _device_reads(_method(runner, "_read_device_memory")) == {
        "mem_get_info",
        "memory_stats",
        "memory_reserved",
    }
    assert _device_reads(_method(runner, "_estimate_cudagraph_overhead")) == {
        "mem_get_info",
        "memory_stats",
    }


def test_the_four_figures_are_handed_back_under_the_names_they_were_taken_by():
    """Four same-typed integers, so a swap is invisible everywhere but here.

    The field names are read off ATOM's own named tuple, so a rename upstream
    fails here instead of inside a worker; and each keyword is checked against
    the reading it is given, so `free` and `total` trading places -- which
    nothing downstream could detect, both being plausible -- is a failure
    rather than a pool sized against the wrong box.
    """
    fields = [
        node.target.id
        for node in _classes(ATOM_RUNNER)["DeviceMemoryReadings"].body
        if isinstance(node, ast.AnnAssign)
    ]
    assert fields == ["free", "total", "peak_torch", "non_torch"]

    overrides = _classes(COMPASS / "runner" / "overrides.py")["NonAllocatingRunner"]
    calls = [
        node
        for node in ast.walk(_method(overrides, "_read_device_memory"))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "DeviceMemoryReadings"
    ]
    assert len(calls) == 1
    assert not calls[0].args, "positional, so the order is invisible to a reader"
    assert {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords} == {
        "free": "readings.free.total",
        "total": "readings.total.total",
        "peak_torch": "readings.peak_torch.total",
        "non_torch": "readings.non_torch.total",
    }


def _safety_margin_coefficients():
    """The float literals in ATOM's own `safety_margin = ...` line."""
    method = _method(_classes(ATOM_RUNNER)["ModelRunner"], "get_num_blocks")
    return {
        n.value
        for node in ast.walk(method)
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["safety_margin"]
        for n in ast.walk(node.value)
        if isinstance(n, ast.Constant) and isinstance(n.value, float)
    }


def test_the_margin_walker_finds_the_margin_where_there_is_one():
    """The control: a walker that found nothing would clear every module."""
    assert _safety_margin_coefficients() == {0.02}


def test_no_budget_arithmetic_is_written_anywhere_in_this_package():
    """Four proxies for the claim, checked over the package rather than said.

    What is checked: no module carries `plan_pools` or
    `_kv_budget_extra_reserve` -- ATOM's override point for a reserve inside
    the budget -- as any string in its syntax tree (a name, attribute,
    parameter, keyword, import or string constant, so a call through the
    module, `setattr`, `getattr` and `__dict__` spellings count), none carries
    `gpu_memory_utilization` the same way except `spec/rules.py`, as the
    string-constant key of a dict display it holds to word a refusal, and
    inside a `budget_bytes=` argument, the utilisation budget ATOM's piecewise
    graph-pool estimate caps its buckets against, and
    none writes the coefficient of ATOM's safety margin -- read off ATOM's
    own line, so it follows ATOM -- or its complement as a literal. What is
    not: the `min(budget, free)` clamp, which has no name to find, a margin
    spelled some other way (`2 / 100`), a name built at runtime (a
    concatenation or an f-string) or held inside a longer string (source text
    handed to `exec`), what `spec/rules.py` does with such a key, and a key
    read out of its dict by another module (`sorted(DEPLOYMENT_OWNED)[3]`).

    Read as syntax trees rather than as text: the words appear in docstrings
    all over this package, so a grep would pass for as long as somebody kept
    writing about the formula while copying it. A docstring holds the name
    inside prose, never as the whole string, so it does not match.
    """
    method = _method(_classes(ATOM_RUNNER)["ModelRunner"], "get_num_blocks")
    assert "_kv_budget_extra_reserve" in {
        n.attr for n in ast.walk(method) if isinstance(n, ast.Attribute)
    }, (
        "ATOM's get_num_blocks no longer reads _kv_budget_extra_reserve; "
        "the refusal below would pin a dead name"
    )
    margins = {round(c, 12) for m in _safety_margin_coefficients() for c in (m, 1 - m)}
    for module in sorted(COMPASS.rglob("*.py")):
        tree = ast.parse(module.read_text())
        keys = {
            id(k)
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            for k in n.keys
            if isinstance(k, ast.Constant) and module == COMPASS / "spec" / "rules.py"
        } | {
            id(a)
            for n in ast.walk(tree)
            if isinstance(n, ast.keyword) and n.arg == "budget_bytes"
            for a in ast.walk(n.value)
        }
        named = [
            (id(n), s)
            for n in ast.walk(tree)
            for _, value in ast.iter_fields(n)
            for s in (value if isinstance(value, list) else [value])
            if isinstance(s, str)
        ]
        strings = {s for _, s in named}
        assert "plan_pools" not in strings, f"{module} names ATOM's plan_pools"
        assert (
            "_kv_budget_extra_reserve" not in strings
        ), f"{module} names ATOM's budget reserve override point"
        assert "gpu_memory_utilization" not in {
            s for n, s in named if n not in keys
        }, (
            f"{module} names gpu_memory_utilization outside rules.py's string "
            "dict keys and a budget_bytes argument"
        )
        literals = {
            round(n.value, 12)
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and type(n.value) is float
        }
        assert not literals & margins, (
            f"{module}: {sorted(literals & margins)} is ATOM's safety-margin "
            "coefficient (or its complement) written as a literal"
        )


# --- the count carries what it was sized from --------------------------------


def _sizing(spec, qwen, tp_width, blocks=100_000):
    return SizedKVPool(
        num_kvcache_blocks=blocks,
        entries={"kv": blocks},
        readings=readings_at(spec, qwen, tp_width),
    )


def test_the_count_says_how_much_of_the_footprint_was_a_coefficient(spec, qwen, capsys):
    """`Basis.DECLARED` on each term, read back as bytes by the block count.

    Both figures are derived from `test_memory_readings`' per-term table rather
    than restated here, so a term that moves there moves here too instead of
    this becoming a second place the same bytes are written down.
    """
    sizing = _sizing(spec, qwen, 1)
    declared = {
        "peak_torch.weights",
        "peak_torch.buffers",
        "peak_torch.activations",
        "cudagraph_overhead.per-token x captured tokens",
    }
    assert set(sizing.declared_terms) == declared
    expected = EXPECTED[1]
    assert sizing.declared_bytes == sum(
        expected[name.partition(".")[0]][name.partition(".")[2]] for name in declared
    )
    assert sizing.subtracted_bytes == sum(
        sum(expected[name].values())
        for name in ("peak_torch", "non_torch", "cudagraph_overhead")
    )
    with capsys.disabled():
        print()
        print(sizing.declared_line())


def test_free_is_not_counted_as_footprint_because_it_is_made_of_the_others(spec, qwen):
    """`free` is `total - peak_torch - non_torch`, so counting it as part of
    the footprint would count two of the three subtracted readings twice."""
    sizing = _sizing(spec, qwen, 1)
    assert "free" not in {name.partition(".")[0] for name in sizing.declared_terms}
    assert sizing.subtracted_bytes < sizing.readings.total.total


def test_the_table_is_the_count_and_its_readings_and_never_the_count_alone(spec, qwen):
    table = _sizing(spec, qwen, 2).table()
    assert "100,000 paged blocks" in table
    for term in ("weights", "buffers", "load residue", "persistent", "activations"):
        assert term in table, term
    assert "declared:" in table


def test_a_count_of_zero_is_a_run_that_did_not_start_and_refuses_to_be_recorded(
    spec, qwen
):
    with pytest.raises(MemoryRefusal, match="is not a pool"):
        _sizing(spec, qwen, 1, blocks=0)


def test_a_footprint_with_no_declared_term_says_so_rather_than_saying_nothing(
    spec, qwen
):
    """Otherwise an absent line and a line nobody printed read the same."""
    readings = readings_at(spec, qwen, 1)
    obtained = DeviceReadings(
        total=readings.total,
        peak_torch=readings.non_torch,
        non_torch=readings.non_torch,
        cudagraph_overhead=readings.non_torch,
        free=readings.free,
        tp_width=1,
        spec_digest=readings.spec_digest,
    )
    sizing = SizedKVPool(num_kvcache_blocks=7, entries={}, readings=obtained)
    assert sizing.declared_terms == ()
    assert sizing.declared_fraction == 0.0
    assert "none of the subtracted footprint" in sizing.declared_line()


# --- the two paths that are refused rather than sized ------------------------


def _runner_stub(**config):
    config.setdefault("disagg_is_decode", False)
    return SimpleNamespace(config=SimpleNamespace(**config))


def test_a_runner_with_no_readings_refuses_and_names_what_would_supply_them():
    with pytest.raises(RunnerRefusal, match="install_device_readings"):
        NonAllocatingRunner.get_num_blocks(_runner_stub())


def test_the_disagg_decode_process_is_refused_rather_than_handed_a_pool():
    """ATOM's own runner answers zero there. Substituting readings into the
    base method reaches neither that short-circuit nor the four safety margins
    the same class holds back on the prefill side."""
    with pytest.raises(RunnerRefusal, match="owns no device memory"):
        NonAllocatingRunner.get_num_blocks(_runner_stub(disagg_is_decode=True))


def test_that_refusal_comes_before_the_readings_are_even_looked_for():
    """Otherwise a disagg decode process with readings installed would size a
    pool, and one without them would report the wrong reason for not."""
    stub = _runner_stub(disagg_is_decode=True)
    stub.compass_readings = "not readings, and not reached"
    with pytest.raises(RunnerRefusal, match="owns no device memory"):
        NonAllocatingRunner.get_num_blocks(stub)


def _fields_read_with_no_default():
    """Every name the overrides read through `_config_field`, off the source."""
    tree = ast.parse((COMPASS / "runner" / "overrides.py").read_text())
    return sorted(
        {
            node.args[1].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and getattr(node.func, "id", None) == "_config_field"
        }
    )


def test_exactly_these_config_fields_are_read_with_no_default():
    """The control: the collector below is not pinning an empty list.

    It collects only reads made through `_config_field`; a field read any other
    way is not seen here.
    """
    assert _fields_read_with_no_default() == [
        "disagg_is_decode",
        "enforce_eager",
        "eos_token_id",
        "kv_transfer_config",
        "model",
        "pipeline_parallel_size",
        "speculative_config",
        "stop_token_ids",
        "trust_remote_code",
    ]


@pytest.mark.parametrize("name", _fields_read_with_no_default())
def test_each_config_field_read_with_no_default_is_one_atom_declares(name):
    """The stubs that reach these reads either supply the fields themselves or
    omit them to be refused, so none of them would notice ATOM renaming one.
    This reads ATOM's config class instead."""
    from atom.config import Config

    assert name in {f.name for f in dataclasses.fields(Config)}


@pytest.mark.parametrize(
    "method",
    [
        NonAllocatingRunner.get_num_blocks,
        NonAllocatingRunner._estimate_cudagraph_overhead,
    ],
)
def test_a_config_without_the_field_a_refusal_reads_is_itself_refused(
    spec, qwen, method
):
    """A running engine on a renamed field refuses naming it, rather than
    reading the field as unset and sizing a pool the refusal was for."""
    runner = SimpleNamespace(config=SimpleNamespace())
    install_device_readings(runner, readings_at(spec, qwen, 1))
    with pytest.raises(RunnerRefusal, match="ATOM's config has no field"):
        method(runner)


def test_installing_something_that_is_not_the_readings_is_refused(spec, qwen):
    runner = SimpleNamespace()
    with pytest.raises(RunnerRefusal, match="is not them"):
        install_device_readings(runner, {"free": 1, "total": 2})
    install_device_readings(runner, readings_at(spec, qwen, 1))
    assert runner.compass_readings.tp_width == 1


def _eager_runner(spec, qwen, *, configured, built):
    runner = SimpleNamespace(config=SimpleNamespace(enforce_eager=configured))
    readings = readings_at(spec, qwen, 1)
    if built:
        readings = dataclasses.replace(
            readings, cudagraph_overhead=reserves(enforce_eager=True)
        )
    install_device_readings(runner, readings)
    return runner


@pytest.mark.parametrize("configured,built", [(True, False), (False, True)])
def test_a_graph_reading_built_for_the_other_deployment_is_refused(
    spec, qwen, configured, built
):
    """The one thing the substitution cannot check for itself, checked.

    ATOM returns zero here under `enforce_eager` and this returns whatever was
    installed, so correctness would otherwise rest on whoever built the reading
    having passed the same flag the runner is configured with -- and a reading
    built for a capturing deployment, installed on a runner told not to
    capture, holds back bytes ATOM never would and moves the block count in
    silence. The reading says which deployment it was built for, and a
    disagreement is declined naming both.
    """
    runner = _eager_runner(spec, qwen, configured=configured, built=built)
    with pytest.raises(RunnerRefusal, match="enforce_eager"):
        NonAllocatingRunner._estimate_cudagraph_overhead(runner)


@pytest.mark.parametrize("agreed", [True, False])
def test_a_graph_reading_built_for_this_deployment_is_answered(spec, qwen, agreed):
    """The control: the guard above declines a mismatch and nothing else."""
    runner = _eager_runner(spec, qwen, configured=agreed, built=agreed)
    overhead = NonAllocatingRunner._estimate_cudagraph_overhead(runner)
    assert (overhead == 0) is agreed


# --- the pipeline minimum, which is inert on an even split and not otherwise -


def test_the_pipeline_minimum_is_inert_on_an_even_split_and_not_otherwise(
    qwen, monkeypatch
):
    """The disposition, measured over ATOM's own partitioner and `plan_pools`.

    `get_num_blocks` reduces the block count to the minimum across pipeline
    stages, and that reduction is guarded by `torch.distributed.is_initialized`
    -- so with no process group it does not run at all and needs no stub, and
    with one it runs ATOM's own code unchanged either way.

    Whether it is *inert* is a different question and the answer is not
    "always". Every stage computes the same readings here, because
    `device_readings` takes no pipeline rank. But each stage sizes its pool
    from the layers it holds -- `_get_total_num_layers` takes a
    `get_pp_indices` slice under pipeline parallelism -- so the entry size
    differs whenever the split is uneven, and the minimum is then what decides
    the count.

    The spans come from `get_pp_indices` rather than being written out here,
    so the test partitions the stack the way the runner does: it hands the
    remainder to the middle partitions, which is not the split a reader would
    guess. On this model -- 64 layers, every fourth one paged -- two and four
    stages divide evenly and three, five and six do not.

    `VLLM_PP_LAYER_PARTITION` overrides the partitioner, so it is cleared:
    otherwise this reads a layout from the environment and calls it ATOM's.
    """
    monkeypatch.delenv("VLLM_PP_LAYER_PARTITION", raising=False)
    budget = 200_000_000_000
    layers = int(qwen.num_hidden_layers)

    def distinct_counts(pp_size):
        counts = set()
        for rank in range(pp_size):
            geometry = KvGeometry.from_hf_config(
                qwen,
                block_size=BLOCK_SIZE,
                parallelism=Parallelism(pp_size=pp_size),
                layer_range=get_pp_indices(layers, rank, pp_size),
            )
            plan = plan_pools([page_pool(geometry.bytes_per_block)], budget, 256)
            counts.add(plan.entries["kv"])
        return len(counts)

    assert {pp: distinct_counts(pp) for pp in (2, 3, 4, 5, 6)} == {
        2: 1,
        3: 2,
        4: 1,
        5: 2,
        6: 2,
    }


# --- the whole import closure, which a one-level guard cannot see -----------


def _pulls_a_tensor_library(module: str) -> bool:
    """Import one module in a fresh interpreter and report what came with it."""
    probe = (
        f"import {module}, sys; "
        "print('torch' in sys.modules or 'transformers' in sys.modules)"
    )
    done = subprocess.run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(REPO)),
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    return done.stdout.strip() == "True"


def test_importing_the_memory_package_pulls_no_tensor_library_at_all():
    """The whole import closure, not one level of it.

    `test_memory_readings.py::test_the_package_imports_no_device` reads each
    module's own import statements, so a memory module importing an
    `atom.compass` module that itself imports torch would pass it. This is the
    same claim made over whatever actually arrives, and it matters because the
    runner override imports this package at module scope, and a worker that
    dies on an import dies before it can say anything a reader would see.
    """
    assert not _pulls_a_tensor_library("atom.compass.memory")


def test_the_probe_reports_true_when_a_tensor_library_really_does_arrive():
    """Without this, a probe that always printed False would read as a clean
    closure for every module in the tree."""
    assert _pulls_a_tensor_library("torch")


def test_the_runner_override_still_imports_no_engine_at_module_scope():
    """It imports the memory package at module scope, which is new, and the
    engine only inside the method that needs it -- so a worker on a machine
    with no driver still gets as far as a refusal it can read."""
    tree = ast.parse((COMPASS / "runner" / "overrides.py").read_text())
    top = {
        alias.name
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    } | {
        node.module
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not any(name.startswith("atom.model_engine") for name in top), top
    assert "atom.compass.memory" in top
