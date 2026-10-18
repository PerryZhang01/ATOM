# SPDX-License-Identifier: MIT
"""The step roofline, one roofline per module summed, and the step-level
fallback to it: a step the law cannot price is priced by the roofline.

A module costs `max(t0, F/peak, M/BW(M))` plus the collective it joins, or
the collective inside that max when it is fused. The answer is declared when
an input a module read for its winning bound is stated, naming each such
input, and carries what the law declined. Steps reach the backends through
the runner's own `dummy_execution`, and the backends reach the runner through
`run.runner` from a run file's `step_linear`, `step_roofline` and `padding`
keys.
"""

import copy

import pytest
import torch
from test_backend_step_linear import IN_HULL, _installed, backend, graphed, law, view
from test_memory_readings import DOCUMENT

from atom.compass.backends import (
    BatchView,
    CostRefused,
    Granularity,
    Method,
    ProvenanceMix,
    RequestShape,
    StepLinearBackend,
)
from atom.compass.backends.step_roofline import (
    Module,
    StepFallbackBackend,
    StepRooflineBackend,
)
from atom.compass.backends.work import attention, bounds, elementwise, gemm
from atom.compass.run.modules import walk
from atom.compass.spec import MachineSpec

ROOFLINE = copy.deepcopy(DOCUMENT)
ROOFLINE["device"]["kernel_floor_s"] = 2e-6
ROOFLINE["device"]["memory"]["bandwidth_curve"] = [[4096, 1.1e11], [2**32, 6.8e12]]
MACHINE = MachineSpec.from_mapping(ROOFLINE)
#: A stack of one kind, so the roofline answers in one term.
STACK = [Module(f"layers.{i}.proj", "gemm", (4096, 4096)) for i in range(8)]
#: One TP2 dense layer: a row-parallel GEMM that reduces, a norm that fuses
#: the all-reduce, and attention.
LAYER = [
    Module("o_proj", "gemm", (4096, 2048), ranks=2),
    Module("norm", "norm", (4096,), ranks=2, fused=True),
    Module("attn", "attention", (16, 4, 128)),
]


def roofline():
    return StepRooflineBackend(MACHINE, STACK)


def fallback():
    return StepFallbackBackend(backend(), roofline())


def _module(work, fused=False):
    """One module's price by hand from its bounds."""
    b = bounds(work, MACHINE)
    kernel = max(b["floor"], b["compute"], b["memory"])
    return max(kernel, b["comm"]) if fused else kernel + b["comm"]


def test_a_step_is_one_roofline_per_module_summed_with_its_collective():
    rows = [(1, 900, True), (1, 300, True)]
    cost = StepRooflineBackend(MACHINE, LAYER).price(view(*rows))
    by_name = {t.name: t.seconds for t in cost.terms}
    # The all-reduce each rank joins: 2 (N-1)/N of 2 rows of 4096 bf16.
    sent = 2 * 4096 * 2
    comm = 2e-6 + sent / (1.0e12 * 0.80)

    assert by_name["roofline.gemm"] == _module(gemm(2, 4096, 2048)) + comm
    assert by_name["roofline.norm"] == max(_module(elementwise(2 * 4096)), comm)
    assert by_name["roofline.attention"] == _module(
        attention([1, 1], [900, 300], 16, 128, 4)
    )
    assert cost.seconds == sum(by_name.values())
    assert roofline().granularity is Granularity.STEP


def test_modules_of_one_shape_are_counted_once_and_priced_by_their_count():
    (term,) = roofline().price(view((8192, 8192, False))).terms

    assert term.seconds == pytest.approx(8 * _module(gemm(8192, 4096, 4096)))


@pytest.mark.parametrize("padded,tokens", [(True, 8), (False, 3)])
def test_a_graph_step_prices_the_rows_it_pads_to_unless_padding_is_none(padded, tokens):
    rows = [(1, 900, True)] * 3
    backend = StepRooflineBackend(MACHINE, LAYER, padded=padded)
    by_name = {t.name: t.seconds for t in backend.price(view(*rows, rung=8)).terms}
    comm = 2e-6 + 2 * tokens * 4096 / (1.0e12 * 0.80)

    assert by_name["roofline.gemm"] == _module(gemm(tokens, 4096, 2048)) + comm
    assert by_name["roofline.attention"] == _module(
        attention([1] * 3, [900] * 3, 16, 128, 4)
    )


@pytest.mark.parametrize(
    "rows,produces,sampled",
    [
        ([(512, 512, False), (8, 1000, False), (1, 40, True)], True, 3),
        ([(2, 40, True), (2, 90, True)], None, 4),
        ([(512, 512, False)], False, 0),
    ],
    ids=["mixed", "two-token-decode", "middle-chunk"],
)
def test_the_lm_head_runs_on_each_prefill_rows_last_token_and_each_decode_token(
    rows, produces, sampled
):
    head = STACK[:1] + [Module("lm_head", "lm_head", (1000, 64), ranks=2)]
    view = BatchView(tuple(RequestShape(*r) for r in rows), produces_output=produces)
    terms = StepRooflineBackend(MACHINE, head).price(view).terms
    gathered = sampled * 1000 * 2 * 2
    comm = 2e-6 + gathered / 2 / (1.0e12 * 0.80)
    priced = [_module(gemm(sampled, 1000, 64)) + comm] if sampled else []

    assert [t.seconds for t in terms if t.name == "roofline.lm_head"] == priced


@pytest.mark.parametrize("bf16", [2.5e15, 1e9], ids=["memory-bound", "compute-bound"])
def test_linear_attention_and_act_on_a_padded_two_token_decode_step_by_hand(bf16):
    document = copy.deepcopy(ROOFLINE)
    document["device"]["kernel_floor_s"] = 1e-12
    document["device"]["memory"]["bandwidth_curve"] = [[1, 1e12], [2**40, 1e12]]
    document["device"]["compute"]["bf16_flops"] = bf16
    modules = [
        Module("gdn", "linear_attention", (4, 2, 8, 16)),
        Module("act", "act", (64,)),
    ]
    # A fresh prefill row, and two two-token decode rows in a rung-4 graph.
    step = view((5, 5, False), (2, 50, True), (2, 90, True), rung=4)
    backend = StepRooflineBackend(MachineSpec.from_mapping(document), modules)
    by_name = {t.name: t.seconds for t in backend.price(step).terms}
    # 9 real tokens read q and k (2 heads of 8), v (4 heads of 16), gate and
    # beta (4 heads) and write o (4 heads of 16); the 3 rows write their 4x8x16
    # state and the 2 with history read it; two products a token and head.
    moved = 9 * (2 * 2 * 8 + 2 * 4 * 16 + 2 * 4) * 2 + (3 + 2) * 4 * 8 * 16 * 2
    flops = 9 * 4 * 2 * (2 * 8 * 16)
    # 9 real tokens and 2 pad rows of 2: two 64-wide inputs read, one written.
    act = (9 + 2 * 2) * 64 * 3 * 2

    assert by_name["roofline.linear_attention"] == pytest.approx(
        max(moved / 1e12, flops / (bf16 * 0.70))
    )
    assert by_name["roofline.act"] == pytest.approx(act / 1e12)


def test_an_unclassified_module_refuses_every_step_by_name():
    tree = torch.nn.Sequential(
        torch.nn.Linear(16, 32, dtype=torch.bfloat16, device="meta"),
        torch.nn.Sequential(torch.nn.Dropout(), torch.nn.Dropout()),
    )
    modules = walk(tree)

    assert [(m.kind, m.dims) for m in modules] == [
        ("gemm", (32, 16)),
        ("unclassified", ()),
    ]
    with pytest.raises(CostRefused, match=r"unclassified modules: Dropout at 1\.0\)$"):
        StepRooflineBackend(MACHINE, modules).price(view((1, 9, True)))


def test_a_spec_field_a_module_reads_and_the_spec_lacks_refuses_naming_it():
    document = copy.deepcopy(ROOFLINE)
    del document["device"]["kernel_floor_s"]
    lacking = StepRooflineBackend(MachineSpec.from_mapping(document), STACK)

    with pytest.raises(CostRefused, match="gemm reads device.kernel_floor_s, which"):
        lacking.price(view((1, 9, True)))


def test_a_collective_wider_than_the_scale_up_domain_refuses_naming_it():
    wide = StepRooflineBackend(MACHINE, [Module("o_proj", "gemm", (64, 8), ranks=16)])

    with pytest.raises(CostRefused, match="over 16 ranks leaves the scale-up domain"):
        wide.price(view((1, 9, True)))


def _spec(origin, origins=None):
    document = copy.deepcopy(ROOFLINE)
    document["provenance"]["origin"] = origin
    if origins is not None:
        document["provenance"]["origins"] = origins
    return MachineSpec.from_mapping(document)


#: The fields a memory-bound decode step's GEMMs read for their bound.
MEMORY = ("device.memory.bandwidth_curve",)
#: And a compute-bound prefill's.
COMPUTE = ("device.compute.bf16_flops", "device.compute.derate.bf16")


@pytest.mark.parametrize(
    "batch,reads",
    [(view((1, 900, True)), MEMORY), (view((8192, 8192, False)), COMPUTE)],
    ids=["memory", "compute"],
)
@pytest.mark.parametrize(
    "spec,named",
    [
        (_spec("datasheet"), ()),
        (_spec("probed"), ()),
        (_spec("assumed"), MEMORY + COMPUTE),
        *[
            (
                _spec(
                    "mixed",
                    {p: {"origin": "datasheet"} for p in MEMORY + COMPUTE}
                    | {"device.kernel_floor_s": {"origin": "datasheet"}}
                    | {path: {"origin": "assumed"}},
                ),
                (path,),
            )
            for path in MEMORY + COMPUTE
        ],
    ],
    ids=["datasheet", "probed", "assumed"] + [f"{p}-assumed" for p in MEMORY + COMPUTE],
)
def test_a_term_is_declared_iff_a_field_its_winning_bound_reads_is_stated(
    spec, named, batch, reads
):
    (term,) = StepRooflineBackend(spec, STACK).price(batch).terms
    note = term.provenance.note
    stated = [p for p in reads if p in named]

    assert term.provenance.method is (Method.DECLARED if stated else Method.ANALYTICAL)
    assert [p for p in MEMORY + COMPUTE if p in note] == stated


@pytest.mark.parametrize("batch", IN_HULL, ids=range(len(IN_HULL)))
def test_a_step_the_law_prices_is_the_laws_answer(batch):
    assert fallback().price(batch) == backend().price(batch)
    assert fallback().granularity is Granularity.STEP


@pytest.mark.parametrize(
    "batch",
    [view((1, 10, True), rung=4), view((8, 8, False), (1, 10, True))],
    ids=["no-rung", "mixed"],
)
def test_a_step_the_law_refuses_is_priced_by_the_roofline_naming_what_declined(batch):
    with pytest.raises(CostRefused) as refused:
        backend().price(batch)
    cost = fallback().price(batch)
    mix = ProvenanceMix()
    mix.record(cost)

    assert cost.seconds == roofline().price(batch).seconds
    assert cost.refusals == refused.value.declined
    assert mix.seconds_by_method() == {(Method.DECLARED, False): cost.seconds}


def test_a_fallback_across_granularities_is_refused():
    class OpLevel(StepLinearBackend):
        @property
        def granularity(self):
            return Granularity.OP

    with pytest.raises(ValueError, match="inside one granularity"):
        StepFallbackBackend(OpLevel(backend().law), roofline())


def test_a_runner_whose_law_lacks_its_rung_prices_the_step_by_the_roofline():
    priced = graphed(fallback()).dummy_execution().predicted_s

    assert priced == roofline().price(view((1, 1, True), rung=1)).seconds


@pytest.mark.parametrize("padding,padded", [({}, True), ({"padding": "none"}, False)])
def test_a_run_file_step_roofline_puts_the_roofline_under_the_law(
    monkeypatch, tmp_path, padding, padded
):
    installed = _installed(
        monkeypatch, tmp_path, step_linear=law(), step_roofline=True, **padding
    )

    assert isinstance(installed, StepFallbackBackend)
    assert isinstance(installed.first, StepLinearBackend)
    assert installed.roofline.padded is padded


def test_a_run_file_padding_other_than_none_is_refused(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="padding is 'executed!'; it is none"):
        _installed(
            monkeypatch,
            tmp_path,
            step_linear=law(),
            step_roofline=True,
            padding="executed!",
        )


def test_a_run_file_step_roofline_false_leaves_the_law_alone(monkeypatch, tmp_path):
    installed = _installed(
        monkeypatch, tmp_path, step_linear=law(), step_roofline=False
    )

    assert isinstance(installed, StepLinearBackend)


@pytest.mark.parametrize("value", [0.004, 1, "true", None])
def test_a_run_file_step_roofline_that_is_not_a_bool_is_refused(
    monkeypatch, tmp_path, value
):
    with pytest.raises(TypeError, match=f"step_roofline is {value!r}"):
        _installed(monkeypatch, tmp_path, step_linear=law(), step_roofline=value)


def test_a_run_file_roofline_floor_is_refused_naming_step_roofline(
    monkeypatch, tmp_path
):
    with pytest.raises(
        ValueError, match=r"renamed: roofline_floor_s \(now step_roofline\)"
    ):
        _installed(monkeypatch, tmp_path, step_linear=law(), roofline_floor_s=0.004)
