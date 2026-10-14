# SPDX-License-Identifier: MIT
"""The step grid's lookup: a batch's equivalent homogeneous shape, read in a grid of
measured whole forwards, answered only inside a cell whose every corner was
measured.

Decode steps come from ATOM's `Scheduler` and are priced through the runner's
own `forward`, `_group_step_seconds` and `projection.batch_view`, on a ladder
where ATOM's `ForwardMode.decide` picks the rung. Prefill chunks come from the
same scheduler, chunking one prompt into pure middle chunks and a final one.

Decode times are `f(B, c)` microseconds, plus `100 * rung` in a replay bundle.
A cell's interpolation reproduces any `a + x.B + y.c + z.B.c` exactly, so the
expected values below are `f` evaluated by hand. Each backend's residual is a
factor of 1 over a hull no step here leaves, so a price is the lookup's.
"""

import copy
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import MockConfig
from test_backend_step_roofline import roofline
from test_dp_step_max import BLOCK, runner

from atom.compass.backends import (
    BatchView,
    CostRefused,
    Granularity,
    RequestShape,
    Species,
    StepGrid,
    StepGridBackend,
)
from atom.compass.backends.step_grid import equivalent_shape
from atom.compass.backends.step_grid_residual import V1, Residual
from atom.compass.backends.step_roofline import StepFallbackBackend
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

LADDER = [1, 2, 4, 8, 16, 32, 48]
C_NODES = [512, 1024]
REPLAY_B = {"32": [17, 32], "48": [33, 48]}
EAGER_B = [32, 40, 48]
PREFILL_NODES = {"B": [1, 2], "T": [8, 64], "c": [0, 64, 128, 192]}


def f(b, c):
    return 1000 + 30 * b + c + b * c / 16


def point(*at, time):
    return {"at": list(at), "time": time, "statistic": "median", "repeats": 20}


REPLAY = [
    point(int(r), b, c, time=f(b, c) + 100 * int(r))
    for r, nodes in REPLAY_B.items()
    for b in nodes
    for c in C_NODES
]
EAGER = [point(b, c, time=f(b, c)) for b in EAGER_B for c in C_NODES]


def bundle(mode, points, output=(), no_output=(), unit="us"):
    return {
        "graph_mode": mode,
        "scope": "forward",
        "unit": unit,
        "prefill": {
            "nodes": PREFILL_NODES,
            "output": list(output),
            "no_output": list(no_output),
        },
        "decode": {
            "nodes": (
                {"B": REPLAY_B, "c": dict.fromkeys(REPLAY_B, C_NODES)}
                if mode == "replay"
                else {"B": EAGER_B, "c": C_NODES}
            ),
            "points": list(points),
        },
    }


def linear(intercept=1.0, transform="identity", **coefficients):
    return {
        "form": "linear",
        "transform": transform,
        "intercept": intercept,
        "coefficients": {**dict.fromkeys(V1.names, 0.0), **coefficients},
    }


def fitted(members, selected, hull=None):
    """A residual mapping under this code's recipe; by default its hull is a box."""
    box = [
        [s * (i == j) for j in range(len(V1.names))] + [-1e6 * (s > 0)]
        for i in range(len(V1.names))
        for s in (1, -1)
    ]
    return {
        "recipe": {"name": V1.name, "version": V1.version, "digest": V1.digest},
        "hull": hull or {"facets": box},
        "members": members,
        "selected": selected,
    }


UNIT = Residual(fitted({"unit": linear()}, {"unit": 1.0}))


def replay(points=REPLAY, residual=UNIT):
    return StepGridBackend(StepGrid(bundle("replay", points)), residual, "replay")


def eager(points=EAGER, residual=UNIT, **prefill):
    grid = StepGrid(bundle("eager", points, **prefill))
    return StepGridBackend(grid, residual, "eager")


def view(*rows, rung=None, produces_output=None):
    return BatchView(
        tuple(RequestShape(*r) for r in rows),
        capture_rung=rung,
        produces_output=produces_output,
    )


def scheduler_of(prompts):
    """ATOM's scheduler holding one prompt of each length."""
    built = Scheduler(
        MockConfig(
            num_kvcache_blocks=2048,
            kv_cache_block_size=BLOCK,
            max_model_len=1024,
            max_num_seqs=64,
            max_num_batched_tokens=32768,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    for n in prompts:
        built.add(
            Sequence(list(range(5, 5 + n)), BLOCK, sampling_params=SamplingParams())
        )
    return built


def decode(prompts):
    """The first decode batch ATOM's scheduler builds after one prompt of each length."""
    scheduler = scheduler_of(prompts)
    while True:
        batch, seqs = scheduler.schedule()
        if batch.total_seqs_num_decode:
            return batch
        scheduler.postprocess(
            list(seqs.values()),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(7,) for _ in batch.req_ids],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )


# 37 decode rows whose cached context averages 600: each row caches its prompt.
B37 = [500] * 18 + [700] * 18 + [600]


def graphed(backend, ladder=LADDER):
    r = runner(backend)
    r.capture_sizes_np, r.enforce_eager = np.asarray(ladder, dtype=np.int32), False
    return r


def test_a_decode_step_is_priced_from_its_rungs_cell_or_at_its_width():
    batch = decode(B37)

    # decide replays rung 48: f(37, 600) = 1000 + 1110 + 600 + 1387.5 = 4097.5,
    # read between rung 48's B nodes 33 and 48, plus 100 * 48.
    assert graphed(replay()).forward(batch).predicted_s == 8897.5e-6
    # The eager bundle ignores the rung and reads B = 37 between 32 and 40.
    assert graphed(eager()).forward(batch).predicted_s == 4097.5e-6


@pytest.mark.parametrize(
    "make,points,gone,named",
    [
        (replay, REPLAY, [48, 48, 1024], "decode rung 48 cell"),
        (eager, EAGER, [40, 1024], "decode cell"),
    ],
    ids=["replay", "eager"],
)
def test_with_one_corner_removed_the_step_refuses_naming_it(make, points, gone, named):
    r = graphed(make([p for p in points if p["at"] != gone]))

    assert r.forward(decode(B37)).predicted_s == 0.0
    b, c = gone[-2:]
    assert r.refused_commands() == (
        (
            f"cost:step-grid: the {named} holding B=37, c=600 has no measurement "
            f"at B={b}, c={c}"
        ),
    )


def test_a_hole_refuses_only_the_steps_that_need_it():
    holed = replay([p for p in REPLAY if p["at"] != [48, 48, 1024]])
    rows = [(1, 601, True)] * 20

    # Rung 32 is complete: f(20, 600) + 3200 = 1000 + 600 + 600 + 750 + 3200.
    assert holed.price(view(*rows, rung=32)).seconds == 6150e-6
    with pytest.raises(CostRefused, match="no measurement at B=48, c=1024"):
        holed.price(view(*[(1, 601, True)] * 37, rung=48))


def test_a_hole_under_a_roofline_floor_is_priced_by_the_roofline():
    holed = replay([p for p in REPLAY if p["at"] != [48, 48, 1024]])
    r = graphed(StepFallbackBackend(holed, roofline()))
    rows = [(1, 501, True)] * 18 + [(1, 701, True)] * 18 + [(1, 601, True)]

    assert r.forward(decode(B37)).predicted_s == (
        roofline().price(view(*rows, rung=48)).seconds
    )
    assert r.refused_commands() == ()
    assert r.priced_steps()["analytical_steps"] == 1
    reason = (
        "the decode rung 48 cell holding B=37, c=600 has no measurement at "
        "B=48, c=1024"
    )
    assert r._priced.reasons() == {("step-grid", reason): 1}


def test_a_decode_step_wider_than_the_widest_rung_refuses_under_a_replay_bundle():
    r = graphed(replay(), ladder=LADDER[:-1])

    assert r.forward(decode(B37)).predicted_s == 0.0
    assert r.refused_commands() == (
        (
            "cost:step-grid: a decode step of 37 rows replays no graph, and a "
            "replay bundle has no eager decode"
        ),
    )


def chunks(backend):
    """One 200-token prompt at a 64-token budget, each prefill step priced by the runner."""
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=64,
            kv_cache_block_size=BLOCK,
            max_model_len=256,
            parallel_config=SimpleNamespace(data_parallel_rank=0),
        )
    )
    scheduler.add(
        Sequence(list(range(5, 205)), BLOCK, sampling_params=SamplingParams())
    )
    r, steps = runner(backend), []
    while not steps or not steps[-1][0]:
        batch, seqs = scheduler.schedule()
        reply = r.forward(batch)
        steps.append((batch.produces_output(), reply.predicted_s))
        scheduler.postprocess(list(seqs.values()), reply, batch=batch)
    return steps, r.refused_commands()


# The three middle chunks, and the final chunk's shape, in the no-output grid.
MIDDLE = [point(1, 64, c, time=1) for c in (0, 64, 128)] + [point(1, 8, 192, time=1)]


def test_a_middle_chunk_is_priced_from_the_no_output_grid_and_a_final_one_never_is():
    steps, refused = chunks(eager(no_output=MIDDLE, unit="ms"))

    assert steps == [(False, 1e-3)] * 3 + [(True, 0.0)]
    assert refused == (
        (
            "cost:step-grid: the prefill output cell holding B=1, T=8, c=192 "
            "has no measurement at B=1, T=8, c=192"
        ),
    )
    final = [point(1, 8, 192, time=3)]
    steps, refused = chunks(eager(output=final, no_output=MIDDLE, unit="ms"))
    assert steps == [(False, 1e-3)] * 3 + [(True, 3e-3)]
    assert refused == ()


def test_a_prefill_step_is_read_at_its_token_count():
    """Prompts of 20 and 10 tokens open in one step: B = 2 and T = 30, and work
    210 + 55 = 265, 132.5 a row, so c = (132.5 - 15 * 16 / 2) / 15 = 5/6."""
    grid = [
        point(b, t, c, time=1000 + t) for b in (1, 2) for t in (8, 64) for c in (0, 64)
    ]
    batch, _ = scheduler_of([20, 10]).schedule()
    r = runner(eager(output=grid))

    # 1000 + T, read between the T nodes 8 and 64.
    assert r.forward(batch).predicted_s == 1030e-6
    assert r.refused_commands() == ()


def test_the_equivalent_shape_keeps_the_token_count_and_the_attention_work():
    # Rows (N_Q, c) of (3, 0) and (5, 10): 8 tokens over 2 rows is N_Q = 4.
    # Work 3*0 + 3*4/2 + 5*10 + 5*6/2 = 71, 35.5 a row; 4c + 4*5/2 = 35.5.
    rows = (RequestShape(3, 3, False), RequestShape(5, 15, False))
    shape = equivalent_shape(rows)

    assert shape == (2, 4, Fraction(51, 8))
    assert all(type(x) is Fraction for x in shape)
    # A decode row is N_Q = 1, so c is the mean cached context.
    decoding = (RequestShape(1, 501, True), RequestShape(1, 701, True))
    assert equivalent_shape(decoding) == (2, 1, 600)


ON_NODE = [(1, 513, True)] * 32
RAGGED = [(1, 257, True)] * 16 + [(1, 769, True)] * 16
INSIDE = [(1, 601, True)] * 32


# f(32, 512) = 1000 + 960 + 512 + 1024; f(32, 600) = 1000 + 960 + 600 + 1200.
@pytest.mark.parametrize(
    "rows,species,seconds",
    [
        (ON_NODE, Species.MEASURED, 3496e-6),
        (RAGGED, Species.INTERPOLATED, 3496e-6),
        (INSIDE, Species.INTERPOLATED, 3760e-6),
    ],
    ids=["homogeneous-on-a-node", "ragged-on-a-node", "inside-a-cell"],
)
def test_only_a_homogeneous_batch_on_a_node_is_a_measured_lookup(
    rows, species, seconds
):
    cost = eager().price(view(*rows, rung=32))

    (term,) = cost.terms
    assert term.name == "forward"
    assert term.provenance.species is Species.FITTED
    assert term.seconds == seconds
    assert term.provenance.detail.startswith(f"lookup {term.seconds!r} s, {species};")
    assert eager().price(view(*rows, rung=32)) == cost


@pytest.mark.parametrize(
    "batch,reason",
    [
        (view(*[(1, 2001, True)] * 32), "c=2000 is outside the decode grid's"),
        (view(*[(1, 601, True)] * 20), "B=20 is outside the decode grid's"),
        (view((8, 8, False), (1, 10, True)), "1 prefill and 1 decode rows"),
        (view((2, 10, True)), "a decode row computes 2 tokens"),
        (view((64.0, 64, False), produces_output=True), "count 64.0 is not an"),
        (view(*ON_NODE, rung=48.0), "the count 48.0 is not an int"),
        (view((64, 64, False)), "does not say whether it produces output"),
    ],
    ids=["past-c", "below-B", "mixed", "speculative", "float", "rung", "output"],
)
def test_a_step_the_grid_cannot_answer_is_refused_by_name(batch, reason):
    with pytest.raises(CostRefused, match=reason):
        eager().price(batch)


def test_a_replay_bundle_refuses_a_rung_it_has_no_grid_for():
    with pytest.raises(CostRefused, match="no decode rung 16"):
        replay().price(view(*[(1, 601, True)] * 10, rung=16))
    with pytest.raises(CostRefused, match="replays no graph"):
        replay().price(view(*INSIDE))


def test_a_negative_count_never_reaches_the_backend():
    with pytest.raises(ValueError, match="at least one token"):
        view((-1, 3, False))


def test_it_is_one_tier_a_line():
    assert replay().granularity is Granularity.STEP
    assert replay().describe() == (
        "step grid, bundle replay; replay decode grid, timing scope forward; "
        "recipe step-grid-residual v1, members unit"
    )


def _edited(edit, mode="eager"):
    mapping = copy.deepcopy(bundle(mode, REPLAY if mode == "replay" else EAGER))
    edit(mapping)
    return mapping


@pytest.mark.parametrize(
    "mapping,match",
    [
        (_edited(lambda m: m.update(graph_mode="lazy")), "neither replay nor eager"),
        (_edited(lambda m: m.update(unit="min")), "unit 'min' is not one of"),
        (
            _edited(
                lambda m: m["prefill"].update(nodes={**PREFILL_NODES, "B": [2, 1]})
            ),
            "prefill output grid's B nodes \\[2, 1\\] are not strictly ascending",
        ),
        (
            _edited(lambda m: m["decode"]["points"].append(point(36, 512, time=1))),
            "point at \\[36, 512\\] is not on the grid's B/c nodes",
        ),
        (
            _edited(lambda m: m["decode"]["points"].append(point(32, 512, time=1))),
            "measures \\[32, 512\\] twice",
        ),
        (
            _edited(lambda m: m["decode"]["points"][0].update(time=0)),
            "takes 0.0, which is not a positive duration",
        ),
        (
            _edited(lambda m: m["decode"]["points"][0].update(time=float("nan"))),
            "takes nan, which is not a positive duration",
        ),
        (
            _edited(lambda m: m["decode"]["points"][0].update(repeats=0)),
            "records no statistic and repeat count",
        ),
        (
            _edited(lambda m: m["decode"]["points"][0].update(statistic="")),
            "records no statistic and repeat count",
        ),
        (
            _edited(
                lambda m: m["decode"]["points"].append(point(16, 9, 512, time=1)),
                mode="replay",
            ),
            "is on rung 16, which has no B nodes",
        ),
        (
            _edited(lambda m: m["decode"]["nodes"].update(B=EAGER_B), mode="replay"),
            "under graph mode replay the decode B nodes are one list per rung, not a list",
        ),
        (
            _edited(lambda m: m["decode"]["nodes"].update(B=REPLAY_B)),
            "under graph mode eager the decode B nodes are one list, not a dict",
        ),
        (
            _edited(lambda m: m["decode"]["nodes"].update(c=C_NODES), mode="replay"),
            "under graph mode replay the decode c nodes are one list per rung, not a list",
        ),
    ],
    ids=[
        "mode",
        "unit",
        "descending",
        "off-node",
        "twice",
        "zero",
        "nan",
        "repeats",
        "statistic",
        "rung",
        "replay-b-list",
        "eager-b-per-rung",
        "replay-c-list",
    ],
)
def test_a_bundle_that_cannot_be_read_fails_loading(mapping, match):
    with pytest.raises(ValueError, match=match):
        StepGrid(mapping)
