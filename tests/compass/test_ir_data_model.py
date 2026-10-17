# SPDX-License-Identifier: MIT
"""`atom.compass.ir`: the cost graph's data model."""

import ast
from pathlib import Path

import pytest

from atom.compass.ir import (
    Applicability,
    ContextRef,
    Graph,
    JoinPolicy,
    NodeKind,
    Op,
    Par,
    Region,
    Repeat,
    Seq,
    SymDim,
)
from atom.compass.ir.shapes import as_dim, as_shape

IR_PACKAGE = Path(__file__).resolve().parents[2] / "atom" / "compass" / "ir"
CAPTURE = "qwen3-27b/decode"


def _op(name="aiter::gemm_a16w16", **kwargs):
    kwargs.setdefault("kind", NodeKind.CAPTURED)
    kwargs.setdefault("in_shapes", ((4, 8),))
    kwargs.setdefault("out_shapes", ((4, 16),))
    return Op(name=name, **kwargs)


class _Unrenderable:
    """An object with no rendering of its own, as most objects have none."""


class _Compound:
    def __str__(self):
        return "2*s26 + 1"


class _Tensor:
    shape = (4, 8)
    dtype = "bf16"


class _Everywhere(Applicability):
    def describe(self):
        return "every step"


# --- repeats -----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        _op(),
        Seq((_op("embed"), _op("norm"))),
        Repeat(_op(), 3, "inner"),
        Par((_op("routed"), _op("shared")), JoinPolicy.MAX),
    ],
    ids=["op", "seq", "repeat", "par"],
)
def test_a_repeat_body_is_any_region(body):
    assert Repeat(body, 20, "period").body is body


def test_the_nested_non_contiguous_form_is_expressible():
    # Three of one layer class then one of another, that period repeated.
    period = Seq((Repeat(_op("linear_attention"), 3, "sub"), _op("full_attention")))
    model = Seq(
        (
            _op("embed"),
            _op("layer_0_dense"),
            Repeat(period, 20, "period", step=4),
            _op("lm_head"),
        )
    )
    assert model.bound_indices == frozenset({"period", "sub"})


@pytest.mark.parametrize("body", [[_op()], "gemm", None])
def test_a_repeat_body_that_is_not_a_region_is_refused(body):
    with pytest.raises(TypeError, match="repeat body is a region"):
        Repeat(body, 2, "layer")


def test_a_repeat_cannot_be_built_without_its_index():
    with pytest.raises(TypeError):
        Repeat(_op(), 20)


@pytest.mark.parametrize("index", ["", "a.b", 0, None])
def test_an_index_that_is_not_a_plain_name_is_refused(index):
    with pytest.raises(ValueError, match="binds a plain name"):
        Repeat(_op(), 2, index)


def test_an_index_that_never_varies_is_refused():
    with pytest.raises(ValueError, match="step 0"):
        Repeat(_op(), 2, "layer", step=0)


@pytest.mark.parametrize("count", [1, 0, -3])
def test_a_repeat_of_fewer_than_two_is_refused(count):
    with pytest.raises(ValueError, match="at least twice"):
        Repeat(_op(), count, "layer")


def test_a_name_bound_far_below_is_still_refused_from_outside():
    # Through a parallel region and two sequences: composition is bottom-up.
    deep = Par((Repeat(_op("a"), 2, "layer"), _op("b")), JoinPolicy.MAX)
    with pytest.raises(ValueError, match="'layer' is already bound"):
        Repeat(Seq((Seq((deep,)), _op("c"))), 4, "layer")


def test_sibling_scopes_may_reuse_an_index_name():
    left, right = Repeat(_op("a"), 2, "i"), Repeat(_op("b"), 3, "i")
    assert Seq((left, right)).bound_indices == frozenset({"i"})


def test_each_instance_binds_its_own_index_value():
    assert Repeat(_op(), 4, "layer", step=4).index_values() == (0, 4, 8, 12)
    assert Repeat(_op(), 3, "layer", start=1).index_values() == (1, 2, 3)


# --- context keys and free indices -------------------------------------------


def test_a_body_reaches_its_instance_through_the_context_key():
    ref = ContextRef("model.layers.{layer}.experts.{expert}")
    assert ref.index_names == ("layer", "expert")
    assert ref.bind(layer=7, expert=2).key == "model.layers.7.experts.2"


@pytest.mark.parametrize(
    "values", [{}, {"layer": 1}, {"layer": 1, "expert": 2, "x": 3}]
)
def test_a_binding_that_does_not_match_the_indices_is_refused(values):
    with pytest.raises(KeyError, match=r"in terms of \['layer', 'expert'\]"):
        ContextRef("model.layers.{layer}.experts.{expert}").bind(**values)


@pytest.mark.parametrize("key", ["", "  ", "layers.{0}", "layers.{}", "layers.{x.y}"])
def test_a_key_no_repeat_index_could_fill_is_refused(key):
    with pytest.raises(ValueError, match="plain names"):
        ContextRef(key)


def test_a_context_key_that_does_not_parse_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="not a usable context key"):
        ContextRef("model.layers.{")


def test_a_repeat_resolves_the_index_its_body_names():
    body = _op("attn", context_ref=ContextRef("model.layers.{layer}.self_attn"))
    assert body.free_indices == frozenset({"layer"})
    repeat = Repeat(body, 64, "layer")
    assert repeat.free_indices == frozenset()
    assert Graph(_Everywhere(), repeat).region is repeat


def test_a_graph_refuses_an_index_nothing_binds():
    stray = _op("attn", context_ref=ContextRef("model.layers.{layer}.self_attn"))
    with pytest.raises(ValueError, match=r"nothing binds \['layer'\]"):
        Graph(_Everywhere(), Seq((_op("embed"), stray)))


def test_two_nodes_differing_only_in_where_they_read_state_are_different_nodes():
    first = _op("unified_attention", context_ref=ContextRef("layers.0"))
    second = _op("unified_attention", context_ref=ContextRef("layers.1"))
    assert first != second
    assert _op("unified_attention") != first


# --- overlap and sequence ----------------------------------------------------


def test_overlap_needs_two_branches():
    with pytest.raises(TypeError, match="at least 2 regions"):
        Par((_op(),), JoinPolicy.MAX)


@pytest.mark.parametrize("join", ["sum", None, 0])
def test_a_parallel_region_states_how_its_branches_combine(join):
    with pytest.raises(ValueError, match="JoinPolicy"):
        Par((_op("a"), _op("b")), join)


def test_a_sequence_with_nothing_in_it_is_refused():
    with pytest.raises(TypeError, match="at least 1 regions"):
        Seq(())


# --- a size that is not known yet --------------------------------------------


def test_a_live_symbolic_size_is_not_rendered_on_anybody_s_behalf():
    with pytest.raises(TypeError, match="SymDim.of"):
        as_dim(_Compound())
    assert as_dim(SymDim.of(_Compound(), CAPTURE)) == SymDim("2*s26 + 1", CAPTURE)


def test_a_node_holding_a_symbolic_size_is_hashable_and_comparable():
    first = _op(in_shapes=((SymDim.of(_Compound(), CAPTURE), 4096),))
    second = _op(in_shapes=((SymDim("2*s26 + 1", CAPTURE), 4096),))
    third = _op(in_shapes=((SymDim("s99", CAPTURE), 4096),))
    assert first == second != third
    assert len({first, second, third}) == 2


def test_two_captures_that_spell_a_symbol_the_same_are_not_the_same_size():
    here, there = SymDim("s26", "capture-a"), SymDim("s26", "capture-b")
    assert here != there
    assert _op(in_shapes=((here, 8),)) != _op(in_shapes=((there, 8),))


@pytest.mark.parametrize(
    "text, scope", [("", CAPTURE), ("   ", CAPTURE), ("s52\ns53", CAPTURE), ("s26", "")]
)
def test_a_symbolic_size_needs_a_rendering_and_a_scope(text, scope):
    with pytest.raises(ValueError, match="one-line rendering"):
        SymDim(text, scope)


@pytest.mark.parametrize("bad", ["4096", " 17 ", "-3", "1.5", "inf"])
def test_a_size_that_renders_as_a_number_is_a_known_size(bad):
    # Held as a symbol it would compare unequal to the int it is.
    with pytest.raises(ValueError, match="belongs in the shape as an int"):
        SymDim(bad, CAPTURE)


def test_an_object_with_no_rendering_of_its_own_is_refused():
    with pytest.raises(ValueError, match="no rendering of its own"):
        SymDim.of(_Unrenderable(), CAPTURE)


def test_a_width_from_another_library_is_converted_not_rendered():
    numpy = pytest.importorskip("numpy")
    with pytest.raises(TypeError, match="SymDim.of"):
        as_dim(numpy.int64(4096))
    with pytest.raises(ValueError, match="belongs in the shape as an int"):
        SymDim.of(numpy.int64(4096), CAPTURE)


def test_a_concrete_dimension_is_bounded():
    assert as_shape([0, 8]) == (0, 8)
    with pytest.raises(ValueError, match="cannot be negative"):
        as_shape([-1])


@pytest.mark.parametrize(
    "bad", [True, 1.5, "8", None, b"8", _Unrenderable(), _Tensor()]
)
def test_a_dimension_that_only_arrives_by_mistake_is_refused(bad):
    with pytest.raises(TypeError, match="a dimension is a concrete int"):
        as_shape([bad])


def test_shapes_are_one_per_operand():
    op = _op(in_shapes=[[4, 8], [8, 16]], out_shapes=[[4, 16]])
    assert op.in_shapes == ((4, 8), (8, 16))


@pytest.mark.parametrize("shapes", [{(4, 8): "x"}, {"a": 1}, "48"])
def test_a_mapping_or_str_is_not_a_shape(shapes):
    with pytest.raises(TypeError, match="expected a sequence"):
        _op(in_shapes=shapes)
    with pytest.raises(TypeError, match="expected a sequence"):
        as_shape(shapes)


def test_a_live_symbolic_value_reaches_no_field_of_a_node():
    # Against the real types under a live ShapeEnv: hashing a SymInt raises, a
    # SymBool or SymFloat would specialise; an isinstance check installs no guard.
    torch = pytest.importorskip("torch")
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.symbolic_shapes import ShapeEnv

    shape_env = ShapeEnv()
    dense = torch.empty(17, 4096)
    torch._dynamo.mark_dynamic(dense, 0)
    with FakeTensorMode(shape_env=shape_env, allow_non_fake_inputs=True) as mode:
        fake = mode.from_tensor(dense, static_shapes=False)
        tokens = fake.shape[0]
        assert type(tokens).__name__ == "SymInt"
        with pytest.raises(TypeError, match="unhashable"):
            hash(tokens)

        causal = tokens > 100
        scale = tokens * 1.5
        unbacked = shape_env.create_unbacked_symint()
        assert [type(v).__name__ for v in (causal, scale)] == ["SymBool", "SymFloat"]

        for label, value in (
            ("SymInt", tokens),
            ("SymBool", causal),
            ("SymFloat", scale),
            ("unbacked SymInt", unbacked),
            ("nested in a tuple", (tokens, 4)),
        ):
            before = len(shape_env.guards)
            with pytest.raises(TypeError, match="An attribute is a number"):
                _op(attrs=(("whatever", value),))
            assert len(shape_env.guards) == before, label

        before = len(shape_env.guards)
        with pytest.raises(TypeError, match="SymDim.of"):
            _op(in_shapes=((tokens, 4096),))
        rendered = SymDim.of(tokens, "one-capture")
        op = _op(in_shapes=((rendered, 4096),), out_shapes=((rendered, 11008),))
        assert len(shape_env.guards) == before

    assert rendered.text == str(tokens)
    assert hash(op) == hash(op)


# --- attributes --------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [4, 1.5, "bf16", True, None, b"raw", NodeKind.CAPTURED, (4, ("a", 1.5), None)],
    ids=lambda v: type(v).__name__,
)
def test_an_attribute_holds_what_the_operator_is(value):
    assert _op(attrs={"a": value}).attrs == (("a", value),)


@pytest.mark.parametrize(
    "value",
    [[1, 2], {"a": 1}, {1, 2}, _Unrenderable(), (1, [2]), _Tensor()],
    ids=["list", "dict", "set", "object", "list nested in a tuple", "tensor"],
)
def test_an_attribute_that_is_not_what_the_operator_is_is_refused(value):
    with pytest.raises(TypeError, match="An attribute is a number"):
        _op(attrs={"a": value})


def test_attributes_do_not_depend_on_the_order_they_were_recorded_in():
    forwards = _op(attrs={"dtype": "bf16", "transpose_b": True})
    backwards = _op(attrs=(("transpose_b", True), ("dtype", "bf16")))
    assert forwards == backwards
    assert forwards.attrs == (("dtype", "bf16"), ("transpose_b", True))


def test_a_name_given_twice_is_refused_rather_than_silently_resolved():
    with pytest.raises(ValueError, match="given twice"):
        _op(attrs=(("dtype", "bf16"), ("dtype", "fp8")))


@pytest.mark.parametrize("attrs", [["ab"], [3], [("dtype",)], [("a", 1, 2)], [(1, 2)]])
def test_an_item_that_is_not_a_name_value_pair_is_refused(attrs):
    with pytest.raises(TypeError, match=r"\(name, value\) pairs"):
        _op(attrs=attrs)


# --- a graph states where it is valid ----------------------------------------


def test_a_statement_that_holds_everywhere_cannot_be_conjured():
    with pytest.raises(TypeError, match="abstract"):
        Applicability()


@pytest.mark.parametrize("applicability", [None, "prefill", True])
def test_a_graph_without_a_validity_statement_is_refused(applicability):
    with pytest.raises(TypeError, match="states where its record is valid"):
        Graph(applicability, _op())


def test_a_graph_is_not_a_region():
    graph = Graph(_Everywhere(), _op())
    assert not isinstance(graph, Region)
    with pytest.raises(TypeError, match="regions"):
        Seq((graph,))


# --- the package reaches nothing ---------------------------------------------


def test_the_package_imports_only_the_standard_library():
    # A graph must be buildable on a machine with no device runtime.
    allowed = {"abc", "collections", "dataclasses", "enum", "string", "typing"}
    modules = sorted(IR_PACKAGE.glob("*.py"))
    assert modules, f"no modules under {IR_PACKAGE}"
    for module in modules:
        roots = set()
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots.add("." if node.level == 1 else (node.module or "").split(".")[0])
        assert (
            roots - {"."} <= allowed
        ), f"{module.name} imports {sorted(roots - allowed)}"
