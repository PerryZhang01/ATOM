# SPDX-License-Identifier: MIT
"""The machine spec: what it may carry, and what it refuses to answer.

The document below is a complete, valid spec and every test starts from a copy
of it, so a test that constructs a violation constructs exactly one and the
refusal it gets can only be about that.

Two of the checks read ATOM's own sources rather than the spec package. The
separation rule says the spec carries nothing ATOM could configure, and that is
a claim about ATOM's argument surface, not about a list somebody wrote once. So
the surface is parsed out of the engine's arguments dataclass and compared both
ways: every knob the refusal message names has to be a real one, and no field of
the spec's machine sections may collide with it. Parsed rather than imported --
reading the source needs no driver, and this tier has none.

Three sections are about the echo rather than about what a spec may hold, and
they are one claim: the echo equals the document that was read, because that is
the whole of what a run artifact can be held to. A field written under two
spellings would leave one of two values out of it, and which one is decided by
the order the keys happen to iterate in, so both orders are read. A document
from a later schema has to be refused for its version rather than for the
fields that version added, or the echo missing is the reader's fault and the
message says it is the author's. And a checked value the document still owns
can move after the artifact carrying its digest was written.

One section is about which rule a refusal names rather than about its words.
The rules are indexed by the action they ask for, so a member's sentence has to
be true of every refusal that carries it. The sites that decline a document are
driven rather than listed, and the set driven is checked against the package's
own source; each is then shown to ask either for an edit this reader goes on to
read, or for a different reader altogether. A document written to a later
schema is the second kind, and it is the only one.

The last section is about the refusals themselves rather than about what they
refuse. A refusal is read by a person, so a message that is accurate about what
could not be done and wrong about why costs that person the time it was meant to
save; an assertion loose enough to hold of a wrong message does not protect
them. Each of the three ways a dotted path fails to resolve is checked to say
its own sentence and to deny the others', so swapping any two fails these tests.
"""

import ast
import copy
import pathlib
import sys
import warnings

import pytest

from atom.compass import spec as spec_package
from atom.compass.spec import (
    DECLARED,
    DEPLOYMENT_OWNED,
    SCHEMA,
    SCHEMA_VERSION,
    Backend,
    FingerprintMismatch,
    Fragment,
    Kind,
    MachineSpec,
    Rule,
    SpecRefusal,
    StackMismatch,
    explain,
    validate,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
PACKAGE = REPO / "atom" / "compass" / "spec"
ENGINE_ARGS = REPO / "atom" / "model_engine" / "arg_utils.py"

#: The stack the constants below were measured against.
STACK = {"rocm": "7.2.4", "aiter": "f4e7c7509", "rccl": "2.22.3"}

TOKENIZER = {
    "id": "qwen3-151k-bpe",
    "backend": "fast",
    "vocab_size": 151936,
    "fingerprint": "sha256:" + "a" * 64,
    "applies_to": ["Qwen3ForCausalLM", "Qwen3MoeForCausalLM"],
    "encode_fixed_s": 3.0e-4,
    "encode_tokens_per_s": 2.0e6,
    "decode_fixed_s": 1.5e-4,
    "decode_tokens_per_s": 3.0e6,
    "derate": 0.85,
}

DOCUMENT = {
    "schema_version": 1,
    "name": "mi355x-8gpu-2node",
    "provenance": {
        "authored_by": "a person",
        "date": "2026-09-18",
        "method": "probed",
    },
    "host": {
        "cpu": {"cores_physical": 96, "cores_logical": 192},
        "tokenizers": [TOKENIZER],
        "ipc": {"zmq_roundtrip_s": 5.0e-5, "shm_broadcast_s": 2.0e-5},
        "admission_fixed_s": 9.0e-3,
    },
    "device": {
        "name": "MI355X",
        "arch": "gfx950",
        "count_per_node": 8,
        "memory": {
            "capacity_bytes": 288.0e9,
            "bandwidth_bytes_per_s": 8.0e12,
            "derate": 0.85,
        },
        "compute": {"bf16_flops": 2.5e15, "fp8_flops": 5.0e15, "derate": 0.70},
        "runtime_constants": {
            "driver_and_collective_reserve_bytes": {
                1: 970.0e6,
                2: 7.2e9,
                4: 7.6e9,
                8: 11.2e9,
            },
            "allocator_retained_after_load_bytes": {
                1: 1.1e6,
                2: 2.17e9,
                4: 2.17e9,
                8: 2.17e9,
            },
            "persistent_forward_buffer_bytes": 124.0e6,
            "cudagraph_pool": {
                "w1_base_bytes": 95.5e6,
                "w1_bytes_per_captured_token": 0.318e6,
                "w_gt1_flat_bytes": 109.0e6,
            },
        },
        "software_pinned_to": dict(STACK),
    },
    "interconnect": {
        "intra_node": {
            "topology": "fully_connected",
            "link_bandwidth_bytes_per_s": 1.0e12,
            "link_latency_s": 2.0e-6,
            "derate": 0.80,
        },
        "inter_node": {
            "link_bandwidth_bytes_per_s": 5.0e10,
            "link_latency_s": 5.0e-6,
            "derate": 0.80,
        },
        "router_relay_s": 1.5e-3,
    },
}


def document(**edits):
    """A fresh copy of the reference spec, with one block replaced per edit."""
    fresh = copy.deepcopy(DOCUMENT)
    for section, value in edits.items():
        fresh[section] = value
    return fresh


def read(**edits):
    return MachineSpec.from_mapping(document(**edits))


def drop(mapping, *path):
    """The document with one dotted field removed, for a missing-term test."""
    node = mapping
    for key in path[:-1]:
        node = node[key]
    del node[path[-1]]
    return mapping


# --- the document goes in and comes back out ---------------------------------


def test_a_spec_round_trips():
    assert read().echo() == DOCUMENT


def test_the_echo_carries_every_field_that_was_read():
    # Rule: a number whose spec cannot be recovered from the run artifact is
    # unattributable, so the echo is built from the field table and is total by
    # construction rather than being whatever the reader kept.
    machine = read()
    echoed = machine.echo()
    for path in machine.values:
        node = echoed
        for key in path.split("."):
            assert key in node, path
            node = node[key]


def test_the_digest_moves_when_any_number_does():
    before = read().digest()
    shifted = document()
    shifted["device"]["memory"]["capacity_bytes"] = 287.0e9
    assert MachineSpec.from_mapping(shifted).digest() != before
    assert read().digest() == before


# --- one field, however the document spells it -------------------------------
#
# A dotted key resolves to the nested path, deliberately: a probe fragment
# writes them. So one field has two spellings and a document can carry both.
# Reading such a document by keeping one of the two keeps whichever the mapping
# yields last, which is decided by the order the keys were written in, and the
# value that loses is then absent from the echo a run artifact records and
# takes its digest over. The pair is refused by name instead, and both orders
# are read here, because order-dependence is the defect and one order alone
# does not test it.

#: A field stated nested by the reference document, for a dotted twin to meet.
TWIN = "host.cpu.cores_physical"

#: What the reference document states there, for a twin that agrees with it.
AGREED = DOCUMENT["host"]["cpu"]["cores_physical"]


def both_spellings(dotted_first, value=1):
    """The reference document stating one field twice, in one of the two orders."""
    nested = document()
    dotted = {TWIN: value}
    return {**dotted, **nested} if dotted_first else {**nested, **dotted}


def refusal_from(written):
    """The refusal reading a document earns, for a test to read in full."""
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(written)
    return refusal.value


def test_the_two_orders_differ_only_in_the_order():
    # What makes the pair below a control rather than two tests: one mapping,
    # written the other way round, so nothing but the iteration order changed.
    assert both_spellings(True) == both_spellings(False)
    assert list(both_spellings(True)) != list(both_spellings(False))


@pytest.mark.parametrize("dotted_first", [True, False])
def test_a_field_stated_twice_is_refused_by_name(dotted_first):
    refusal = refusal_from(both_spellings(dotted_first))
    assert refusal.rule is Rule.SHAPE
    assert TWIN in refusal.what
    assert "stated twice" in refusal.what
    # Not the closed-schema refusal: both spellings are the schema's own field,
    # and the reader is not being sent to the field table to look for it.
    assert "not a field of this schema" not in refusal.what


def test_both_orders_give_the_same_refusal():
    # Order decided which value survived, so nothing in the message may depend
    # on which spelling the mapping yielded first -- and a message that named
    # the value it kept would.
    dotted_first = refusal_from(both_spellings(True))
    nested_first = refusal_from(both_spellings(False))
    assert dotted_first.rule is nested_first.rule
    assert str(dotted_first) == str(nested_first)
    for value in ("96", "1"):
        assert value not in str(dotted_first)


@pytest.mark.parametrize("dotted_first", [True, False])
def test_a_field_stated_twice_is_refused_even_where_the_two_agree(dotted_first):
    # The rule is the shape, not the two values, and the narrower rule looks
    # equivalent: `path in found and found[path] != value` passes every other
    # test in this file. It would let this document through. Both orders are
    # read here too, because a rule that compared the values could still turn
    # on which spelling landed first, and one order alone would not see it.
    refusal = refusal_from(both_spellings(dotted_first, AGREED))
    assert refusal.rule is Rule.SHAPE
    assert TWIN in refusal.what
    assert "stated twice" in refusal.what


def test_resolving_the_collision_would_leave_the_flat_key_out_of_the_echo():
    # Why refusing is better than resolving, and the whole of what this test
    # holds: it reads the same document with the flat key resolved away, which
    # is what any resolution yields, because the echo is rebuilt from the field
    # table and carries the nested spelling only. The echo is then missing a
    # key the document states, whichever value the two keys agreed on. The
    # refusal itself is held by the test above, not by this one.
    written = both_spellings(False, AGREED)
    resolved = {key: value for key, value in written.items() if key != TWIN}
    echoed = MachineSpec.from_mapping(resolved).echo()
    assert TWIN in written
    assert TWIN not in echoed
    assert set(written) - set(echoed) == {TWIN}


def test_the_collision_is_about_the_path_and_not_about_the_top_level():
    # Two spellings meeting one block down, neither of them the whole path, so
    # what is compared is what each resolves to.
    written = document()
    written["host"]["cpu.cores_logical"] = 1
    refusal = refusal_from(written)
    assert refusal.rule is Rule.SHAPE
    assert "host.cpu.cores_logical" in refusal.what


def test_a_dotted_key_on_its_own_is_a_supported_spelling():
    # The fix is not "refuse dotted keys". Written flat, every field reads, and
    # the echo is the nested document -- which is also why carrying both
    # spellings is one field twice rather than two fields.
    written = document()
    cpu = written["host"].pop("cpu")
    flat = {f"host.cpu.{name}": value for name, value in cpu.items()}
    machine = MachineSpec.from_mapping({**written, **flat})
    assert machine.value(TWIN) == 96
    assert machine.echo() == DOCUMENT
    assert machine.digest() == read().digest()


# --- the version is the one thing an old reader can say ----------------------


def test_a_later_document_is_told_this_reader_is_old():
    # A v2 document carries what v2 added; the schema is closed, so a version
    # checked after the fields lets the closed schema speak first and tells the
    # author their new field is illegitimate, in the one case where this reader
    # is what is out of date.
    later = document()
    later["schema_version"] = 2
    later["device"]["power_cap_watts"] = 700
    refusal = refusal_from(later)
    assert refusal.rule is Rule.VERSION
    assert "schema_version 2" in refusal.what
    assert f"understands version {SCHEMA_VERSION}" in refusal.what
    assert "power_cap_watts" not in refusal.what
    assert "not a field of this schema" not in refusal.what


def test_the_version_is_read_before_the_fields_it_governs():
    # Not only an added field: anything a later schema did differently reaches
    # the field table first, and each of those refusals would be about the
    # document when the thing to fix is the reader.
    later = drop(document(), "device", "memory", "capacity_bytes")
    later["schema_version"] = 2
    refusal = refusal_from(later)
    assert refusal.rule is Rule.VERSION
    assert "schema_version 2" in refusal.what
    assert "capacity_bytes" not in refusal.what


def test_a_document_that_states_no_version_is_told_it_is_missing():
    # The gate declines to speak for a document that names no version; the
    # field table reports it with everything else the schema requires.
    written = document()
    del written["schema_version"]
    refusal = refusal_from(written)
    assert "schema_version` is missing" in refusal.what


def test_a_version_that_is_not_one_is_refused_as_a_shape():
    # The boundary between the two rules. A version this reader does not know
    # is about the reader; a version that is not a version is about the
    # document, and the author has a line to correct.
    refusal = refusal_from(document(schema_version="1"))
    assert refusal.rule is Rule.SHAPE
    assert "a positive whole number" in refusal.what


# --- the rule a refusal names is the action it asks for ----------------------


def written(path, value):
    """The reference document with one dotted field set, for a one-edit break."""
    fresh = document()
    node = fresh
    *blocks, leaf = path.split(".")
    for block in blocks:
        node = node[block]
    node[leaf] = value
    return fresh


#: One edit each, and every one of them a document this reader declines as a
#: shape. Driven rather than listed, because what is claimed about them is the
#: action they ask for, and only running them shows that.
BREAKAGES = {
    "a block holding a scalar": lambda: written("device.memory", 7),
    "a required field is missing": lambda: drop(
        document(), "device", "memory", "capacity_bytes"
    ),
    "a spec that is not a mapping at all": lambda: [1, 2],
    "a tokenizer entry field is missing": lambda: drop(
        document(), "host", "tokenizers", 0, "vocab_size"
    ),
    "a tokenizer entry that is not a mapping": lambda: written(
        "host.tokenizers", ["not an entry"]
    ),
    "an activation entry missing a field": lambda: written(
        "device.activations", [{"id": "m", "applies_to": ["A"]}]
    ),
    "two activation entries for one model": lambda: written(
        "device.activations",
        [
            {
                "id": i,
                "fingerprint": "f",
                "applies_to": ["A"],
                "bytes_per_token": {1: 1},
            }
            for i in ("m", "n")
        ],
    ),
    "a value the schema cannot hold": lambda: written(
        "device.memory.capacity_bytes", "lots"
    ),
    "one field written under two spellings": lambda: both_spellings(True),
}

#: The same rule named by the verbs other than the reader, each driven through
#: the verb that declines. Not part of the edit-then-read claim below, which is
#: about what this reader goes on to read; here only so the partition covers
#: every site the package has rather than the reader's share of them.
ELSEWHERE = {
    "a fragment that is not a mapping": lambda: Fragment.from_mapping([1, 2], "p"),
    "a subject to check that is not a mapping": lambda: validate([1, 2]).raise_first(),
}


def _spec_modules():
    # The one walk over the package: every check below that reads its source
    # takes its modules from here, so no two of them can disagree about which
    # modules exist. Recursive, so a module added under a subpackage is read
    # the day it lands.
    return sorted(PACKAGE.rglob("*.py"))


def _named(path):
    # A site's file, relative to the package, so `sub/machine.py` is not
    # `machine.py`. A module loaded from outside the package walked here
    # refuses rather than being compared under a name that happens to match.
    return pathlib.Path(path).resolve().relative_to(PACKAGE).as_posix()


def _site_of(declines):
    """The file and line of the `SpecRefusal(...)` that `declines()` raises.

    Taken where the refusal is built rather than from its traceback. The walk
    yields its refusals and the reader over it raises the first, so the
    traceback of a refusal the walk found names the reader's `raise` and not
    the site that declined.
    """
    built = {}
    init = SpecRefusal.__init__

    def recording(refusal, *args):
        caller = sys._getframe(1)
        built[id(refusal)] = (
            _named(caller.f_code.co_filename),
            caller.f_lineno,
        )
        init(refusal, *args)

    SpecRefusal.__init__ = recording
    try:
        with pytest.raises(SpecRefusal) as raised:
            declines()
    finally:
        SpecRefusal.__init__ = init
    return built[id(raised.value)]


def _is_refusal(raised):
    """Whether a raised expression constructs a `SpecRefusal`."""
    callee = raised.func if isinstance(raised, ast.Call) else None
    return getattr(callee, "id", None) == "SpecRefusal"


def _refusal_sites():
    """Every `SpecRefusal(...)` built in the package, mapped to the rule it names.

    A site is where a refusal is built, whether it is raised there, yielded to
    a walk's consumer or collected into a result, because all three decline.

    One site picks its rule from a local name rather than naming it inline, so
    a name is resolved through the assignments in the function it sits in. The
    binding map is per-name and last-write-wins, so a second binding of the
    same local in one function misattributes one of the two sites; the check
    below then reddens for a site that is on the wrong side of the partition
    rather than for the rule it actually names.

    A site whose rule will not resolve is kept, carrying whatever it named,
    rather than dropped -- a dropped site would be a hole in the coverage check
    below, and holes of exactly that kind are why the check exists.

    The module body is a scope too, so a refusal built outside any function is
    a site. Scopes are read innermost first and the module body last, and a
    site keeps the rule the first scope holding it resolved.
    """
    found = {}
    for source in _spec_modules():
        module = ast.parse(source.read_text())
        functions = [
            scope
            for scope in ast.walk(module)
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        for scope in [*reversed(functions), module]:
            bound = {
                target.id: ast.unparse(node.value)
                for node in ast.walk(scope)
                if isinstance(node, ast.Assign)
                for target in node.targets
                if isinstance(target, ast.Name)
            }
            for node in ast.walk(scope):
                if _is_refusal(node):
                    named = ast.unparse(node.args[0])
                    found.setdefault(
                        (_named(source), node.lineno), bound.get(named, named)
                    )
    return found


def _mentions_the_walk_cannot_follow():
    """Every mention of `SpecRefusal` in the package that is not calling it.

    The walk above sees a refusal only where `SpecRefusal(...)` is called by
    that bare name. Catching the class, testing against it and annotating with
    it build nothing. Any other mention -- an alias, an `import ... as`, a
    subclass, a `partial`, a qualified `rules.SpecRefusal` -- is a way to build
    a refusal the walk never sees, and a refusal built that way and thrown by
    name is invisible to every check here. A class reached without its name,
    as `type(refusal)(...)`, is past what reading the source can see.
    """
    found = []
    for source in _spec_modules():
        lines = source.read_text().splitlines()
        module = ast.parse("\n".join(lines))
        harmless = set()
        for node in ast.walk(module):
            if _is_refusal(node):
                harmless.add(id(node.func))
            if isinstance(node, ast.ExceptHandler):
                cleared = [node.type]
            elif isinstance(node, (ast.arg, ast.AnnAssign)):
                cleared = [node.annotation]
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                cleared = [node.returns]
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) in ("isinstance", "issubclass")
                and len(node.args) == 2
            ):
                cleared = [node.args[1]]
            else:
                continue
            for subtree in filter(None, cleared):
                harmless.update(id(inner) for inner in ast.walk(subtree))
        for node in ast.walk(module):
            if isinstance(node, ast.alias):
                mentioned = node.name == "SpecRefusal" and node.asname is not None
            elif isinstance(node, (ast.Name, ast.Attribute)):
                spelled = node.id if isinstance(node, ast.Name) else node.attr
                mentioned = spelled == "SpecRefusal" and id(node) not in harmless
            else:
                continue
            if mentioned:
                found.append(
                    f"{_named(source)}:{node.lineno}: {lines[node.lineno - 1].strip()}"
                )
    return found


def _built_where_thrown():
    """Every `raise` or `yield` in the package that builds what it throws.

    A `raise` of a refusal already built -- the reader raising the first one
    its walk yielded, or a result handing back its first -- builds nothing
    there, so it is not a site.
    """
    found = set()
    for source in _spec_modules():
        for node in ast.walk(ast.parse(source.read_text())):
            if isinstance(node, ast.Raise):
                thrown = node.exc
            elif isinstance(node, ast.Yield):
                thrown = node.value
            else:
                continue
            if isinstance(thrown, ast.Call):
                found.add((_named(source), thrown.lineno))
    return found


def _sites_naming(member):
    return {site for site, rule in _refusal_sites().items() if f"Rule.{member}" in rule}


def test_every_refusal_site_in_the_package_resolves_to_a_rule():
    # The resolver has to read all of them. One it could not read would leave
    # the two checks below green over a smaller package than the one that
    # exists, which is the shape of defect they are here to catch.
    unread = {
        site: rule
        for site, rule in _refusal_sites().items()
        if not rule.startswith("Rule.")
    }
    assert unread == {}
    # A site the walk never reaches -- in a scope it does not descend, or
    # building through a name that is not `SpecRefusal` -- is absent rather
    # than unread, and absence is invisible to a check that reads only what
    # the walk found. A refusal built through another name and thrown by name
    # is absent from both sets below, so the class is never named any other
    # way; and everything the package throws as it builds it is held against
    # what the resolver recognised.
    loose = _mentions_the_walk_cannot_follow()
    assert loose == [], (
        "SpecRefusal is reached other than by calling it by name, so a refusal "
        "built through it is invisible to the site walk. Call SpecRefusal(...) "
        "there, or teach _is_refusal the new spelling: " + "; ".join(loose)
    )
    thrown = _built_where_thrown()
    assert thrown
    assert thrown <= set(_refusal_sites())


def test_every_rule_is_named_by_at_least_one_site():
    # The checks on a rule are written over the sites that name it, so a rule
    # no site names passes every one of them by having nothing to check. A
    # member is kept only if some refusal the package builds names it, whether
    # inline or through the local it picks its rule from.
    named = {
        node.attr
        for rule in _refusal_sites().values()
        for node in ast.walk(ast.parse(rule, mode="eval"))
        if isinstance(node, ast.Attribute) and getattr(node.value, "id", None) == "Rule"
    }
    assert named
    assert [member.name for member in Rule if member.name not in named] == []


def test_every_site_that_declines_a_document_is_driven_here():
    # A table written by hand that quietly covers six of seven sites passes for
    # a reason nobody stated, so the set is read out of the package's source
    # instead of counted. A site added later is red here until it is placed on
    # one side of the partition or the other.
    driven = {
        _site_of(lambda make=make: MachineSpec.from_mapping(make()))
        for make in BREAKAGES.values()
    } | {_site_of(declines) for declines in ELSEWHERE.values()}
    assert driven == _sites_naming("SHAPE")


def test_explain_declines_an_unknown_term_and_an_absent_one_at_two_sites():
    # Neither is a document to correct, so neither is a shape. A term the
    # schema does not know is re-asked by path; a declared field this spec
    # does not state would come back empty under any spelling, so it is the
    # absent-field refusal the accessor gives, and not a second copy of it.
    unknown = _site_of(lambda: explain(read(), "no_such_term"))
    absent = _site_of(lambda: explain(read(), "provenance.notes"))
    assert unknown[0] == "explain.py" and unknown in _sites_naming("ADDRESSING")
    assert absent == _site_of(lambda: read().value("provenance.notes"))
    assert absent in _sites_naming("TOTALITY")
    assert not {unknown, absent} & _sites_naming("SHAPE")


def test_the_version_is_the_one_site_that_is_not_about_the_document():
    site = _site_of(lambda: MachineSpec.from_mapping(document(schema_version=2)))
    assert _sites_naming("VERSION") == {site}
    assert site not in _sites_naming("SHAPE")


@pytest.mark.parametrize("breakage", sorted(BREAKAGES))
def test_a_shape_refusal_asks_for_an_edit_this_reader_then_reads(breakage):
    # SHAPE's action, and it is one action: correct the document and read it
    # again. Each breakage is one edit from the reference document, so the
    # reference document is the correction, and this reader still reads it.
    refusal = refusal_from(BREAKAGES[breakage]())
    assert refusal.rule is Rule.SHAPE, breakage
    assert MachineSpec.from_mapping(document()).echo() == DOCUMENT


def test_the_version_refusal_asks_for_a_newer_reader_and_says_so():
    # What this asserts is the member and the action. A count of members would
    # pass the next time somebody adds one.
    refusal = refusal_from(document(schema_version=2))
    assert refusal.rule is Rule.VERSION
    assert refusal.rule is not Rule.SHAPE
    assert "reader" in refusal.rule.value
    assert "read it with a Compass that knows that version" in refusal.remedy
    assert Rule.VERSION.value in str(refusal)


def test_no_edit_a_shape_refusal_asks_for_reaches_a_later_document():
    # Why the action differs, measured rather than argued. Every breakage above
    # earns its own refusal on a document this reader knows the version of; on
    # a v2 document each earns the version refusal instead, byte for byte the
    # same one, because nothing past the version was read. So there is no edit
    # here for the author to make, which is what puts this use outside SHAPE.
    plain = refusal_from(document(schema_version=2))
    unreachable = [
        name for name, make in BREAKAGES.items() if not isinstance(make(), dict)
    ]
    assert unreachable == ["a spec that is not a mapping at all"]
    for name, make in BREAKAGES.items():
        if name in unreachable:
            continue
        later = make()
        later["schema_version"] = 2
        refusal = refusal_from(later)
        assert refusal.rule is Rule.VERSION, name
        assert str(refusal) == str(plain), name


# --- a checked value belongs to the spec, not to the document ----------------


def test_the_spec_does_not_move_when_the_document_does():
    # The digest is written into a run artifact, so it has to be a property of
    # what was read. A spec holding the document's own containers would move
    # afterwards, under the artifact that already named it.
    written = document()
    written["provenance"]["fragments"] = ["probe-cpu"]
    machine = MachineSpec.from_mapping(written)
    before = machine.digest()
    echoed = machine.echo()
    constants = written["device"]["runtime_constants"]
    entries = written["host"]["tokenizers"]
    written["provenance"]["fragments"].append("probe-device")
    constants["driver_and_collective_reserve_bytes"][16] = 1.0
    entries.append(dict(TOKENIZER, id="another"))
    entries[0]["encode_tokens_per_s"] = 1.0
    entries[0]["applies_to"].append("LlamaForCausalLM")
    assert machine.digest() == before
    assert machine.echo() == echoed


def test_no_container_the_document_owns_reaches_the_spec():
    # A list of names copies with `tuple` and a width table with `dict`; a
    # tokenizer entry holds mappings and lists of its own, so the same
    # guarantee there is a walk rather than a constructor.
    written = document()
    written["provenance"]["fragments"] = ["probe-cpu"]
    machine = MachineSpec.from_mapping(written)
    constants = written["device"]["runtime_constants"]
    entries = written["host"]["tokenizers"]
    table_path = "device.runtime_constants.driver_and_collective_reserve_bytes"
    held = machine.value("host.tokenizers")
    assert (
        machine.value("provenance.fragments") is not written["provenance"]["fragments"]
    )
    assert (
        machine.value(table_path)
        is not constants["driver_and_collective_reserve_bytes"]
    )
    assert held is not entries
    assert held[0] is not entries[0]
    assert held[0]["applies_to"] is not entries[0]["applies_to"]


# --- the separation rule -----------------------------------------------------


def _engine_argument_names():
    """The engine's configurable surface, read out of its arguments dataclass."""
    tree = ast.parse(ENGINE_ARGS.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "EngineArgs":
            return frozenset(
                statement.target.id
                for statement in node.body
                if isinstance(statement, ast.AnnAssign)
                and isinstance(statement.target, ast.Name)
            )
    return frozenset()


ENGINE_SURFACE = _engine_argument_names()


def test_the_engine_argument_surface_was_found():
    assert len(ENGINE_SURFACE) > 20


def test_the_deployment_knobs_are_the_nine_the_list_is_pinned_to_hold():
    # Two tests below are parametrised over this list, so emptying it makes them
    # collect nothing and skip. The list may legitimately shrink, even to
    # nothing: the closed schema refuses these keys without it, and it only
    # adds where ATOM configures each. So the count is pinned rather than
    # asserted non-empty, and every change to it is an edit made here too.
    assert len(DEPLOYMENT_OWNED) == 9, sorted(DEPLOYMENT_OWNED)


@pytest.mark.parametrize("knob", sorted(DEPLOYMENT_OWNED))
def test_every_knob_the_refusal_names_is_one_the_engine_really_has(knob):
    assert knob in ENGINE_SURFACE, (
        f"{knob} is named as ATOM's, and is not on ATOM's argument surface; "
        "the message would send a reader somewhere that does not exist"
    )


@pytest.mark.parametrize(
    "field", [f for f in SCHEMA if f.path.split(".")[0] != "provenance"]
)
def test_no_field_describing_the_machine_is_one_the_engine_configures(field):
    # Only the sections that describe the machine. `provenance` describes how
    # the document was authored, which is not something a deployment can set,
    # and the rule is about the machine.
    assert field.path.rsplit(".", 1)[-1] not in ENGINE_SURFACE, field.path


@pytest.mark.parametrize("knob", sorted(DEPLOYMENT_OWNED))
def test_a_deployment_knob_is_refused_and_told_where_it_lives(knob):
    edited = document()
    edited["device"][knob] = 4
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.SEPARATION
    assert knob in refusal.value.what
    assert DEPLOYMENT_OWNED[knob] in refusal.value.remedy


def test_the_closed_schema_refuses_a_key_no_rule_names():
    # The mechanism is the closed schema, not the list: a knob nobody has
    # thought of is refused the same way, just without the forwarding address.
    edited = document()
    edited["device"]["some_future_engine_knob"] = 1
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.SEPARATION
    assert "some_future_engine_knob" in refusal.value.what


def test_the_reader_raises_the_first_refusal_the_walk_meets():
    # Two unknown keys in one block, one before its fields and one after, so a
    # reader that raised any later refusal names the other key. The first key
    # sorts last, so a walk in sorted order names the other key as well.
    edited = document()
    edited["device"] = {"zeta_knob": 1, **edited["device"], "alpha_knob": 2}
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert "`device.zeta_knob`" in refusal.value.what


# --- no defaults for the runtime constants -----------------------------------


def test_a_missing_runtime_constant_refuses_and_names_itself():
    edited = drop(
        document(), "device", "runtime_constants", "persistent_forward_buffer_bytes"
    )
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.NO_DEFAULTS
    assert "persistent_forward_buffer_bytes" in refusal.value.what


def test_an_unmeasured_width_refuses_and_names_the_measured_ones():
    with pytest.raises(SpecRefusal) as refusal:
        read().runtime_constant("driver_and_collective_reserve_bytes", tp_width=3)
    assert refusal.value.rule is Rule.NO_DEFAULTS
    assert "width 3" in refusal.value.what
    assert "[1, 2, 4, 8]" in refusal.value.what


def test_a_width_keyed_constant_will_not_answer_without_a_width():
    with pytest.raises(SpecRefusal) as refusal:
        read().runtime_constant("driver_and_collective_reserve_bytes")
    assert refusal.value.rule is Rule.NO_DEFAULTS
    # The rule alone does not say this: the sibling branch above carries it too,
    # and "not measured at width None" would satisfy an assertion that stopped
    # at the rule while saying something the accessor never found out.
    assert "none was given" in refusal.value.what


def test_a_measured_width_answers():
    machine = read()
    assert machine.runtime_constant("driver_and_collective_reserve_bytes", 2) == 7.2e9
    assert machine.runtime_constant("persistent_forward_buffer_bytes") == 124.0e6
    assert machine.runtime_constant("cudagraph_pool.w_gt1_flat_bytes") == 109.0e6


# --- a derate wherever a spec peak appears -----------------------------------


def test_no_derate_is_declared_by_hand():
    # Each one is derived from a field marked as a spec peak, so a peak number
    # added later cannot arrive without one.
    assert not [f for f in DECLARED if f.kind is Kind.DERATE]
    derated = {f.block for f in SCHEMA if f.kind is Kind.DERATE}
    assert derated == {f.block for f in SCHEMA if f.peak}
    assert len(derated) == 4


@pytest.mark.parametrize("block", ["memory", "compute"])
def test_a_spec_peak_without_its_derate_is_refused(block):
    edited = drop(document(), "device", block, "derate")
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.DERATE
    assert f"device.{block}.derate" in refusal.value.what


@pytest.mark.parametrize("value", [1.2, 0.0, -0.5])
def test_a_derate_may_shrink_a_peak_and_never_grow_it(value):
    edited = document()
    edited["device"]["compute"]["derate"] = value
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.DERATE
    # The document states this derate, so the rule's other message -- the one
    # for a derate left out -- would be false here and the rule cannot tell.
    assert "(0, 1]" in refusal.value.what


def test_a_tokenizer_entry_owes_a_derate_too():
    entry = dict(TOKENIZER)
    del entry["derate"]
    edited = document()
    edited["host"]["tokenizers"] = [entry]
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.DERATE
    assert "host.tokenizers[0].derate` is missing" in refusal.value.what


# --- the stack pin is checked ------------------------------------------------


def test_a_matching_stack_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read().check_stack(dict(STACK)) == ()


def test_a_stack_mismatch_warns_and_names_both_versions():
    running = dict(STACK, rocm="7.3.0")
    with pytest.warns(StackMismatch) as caught:
        differences = read().check_stack(running)
    assert differences == (("rocm", "7.2.4", "7.3.0"),)
    message = str(caught[0].message)
    assert "7.2.4" in message and "7.3.0" in message


def test_a_stack_component_nobody_reported_is_a_mismatch_not_a_pass():
    with pytest.warns(StackMismatch):
        differences = read().check_stack({"rocm": "7.2.4", "aiter": "f4e7c7509"})
    assert differences == (("rccl", "2.22.3", None),)


def test_a_missing_stack_pin_is_refused():
    edited = drop(document(), "device", "software_pinned_to", "rccl")
    with pytest.raises(SpecRefusal):
        MachineSpec.from_mapping(edited)


# --- the tokenizer is keyed by the tokenizer ---------------------------------


def test_two_models_sharing_one_tokenizer_resolve_to_one_entry():
    # Two architectures, one measured entry: four rates and a derate stored
    # once. Keyed by model the same spec would hold one copy per
    # architecture -- two here, more in a real family -- with nothing in the
    # document ever comparing them, so they would drift silently.
    machine = read()
    qwen = machine.tokenizer_for("Qwen3ForCausalLM", Backend.FAST)
    qwen_moe = machine.tokenizer_for("Qwen3MoeForCausalLM", Backend.FAST)
    assert qwen is qwen_moe
    assert len(machine.tokenizers.entries) == 1
    copies_if_keyed_by_model = sum(
        len(entry.key.applies_to) for entry in machine.tokenizers.entries
    )
    assert copies_if_keyed_by_model == 2
    assert qwen.encode_tokens_per_s == 2.0e6


def test_an_unmeasured_architecture_is_refused_not_defaulted():
    with pytest.raises(SpecRefusal) as refusal:
        read().tokenizer_for("LlamaForCausalLM", Backend.FAST)
    assert refusal.value.rule is Rule.TOKENIZER_IDENTITY
    assert "LlamaForCausalLM" in refusal.value.what
    assert "qwen3-151k-bpe" in refusal.value.what


def test_the_other_backend_is_a_different_measurement():
    # The two implementations run an order of magnitude apart on the same file,
    # so an entry measured on one says nothing about the other.
    with pytest.raises(SpecRefusal) as refusal:
        read().tokenizer_for("Qwen3ForCausalLM", Backend.SLOW)
    assert refusal.value.rule is Rule.TOKENIZER_IDENTITY
    # Both backends appear in this message -- the one asked for and the one the
    # entry was measured on -- so naming the word alone would hold with the two
    # of them the wrong way round, which is the claim being made.
    assert "on the slow backend" in refusal.value.what
    assert "qwen3-151k-bpe (fast)" in refusal.value.what


def test_a_fingerprint_mismatch_warns_and_names_both():
    loaded = "sha256:" + "b" * 64
    with pytest.warns(FingerprintMismatch) as caught:
        entry = read().tokenizer_for("Qwen3ForCausalLM", Backend.FAST, loaded)
    assert entry.key.id == "qwen3-151k-bpe"
    message = str(caught[0].message)
    assert TOKENIZER["fingerprint"] in message and loaded in message


def test_the_measured_fingerprint_is_silent():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        read().tokenizer_for("Qwen3ForCausalLM", Backend.FAST, TOKENIZER["fingerprint"])


def test_two_entries_claiming_one_architecture_are_refused():
    other = dict(TOKENIZER, id="qwen3-151k-bpe-again")
    edited = document()
    edited["host"]["tokenizers"] = [TOKENIZER, other]
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.TOKENIZER_IDENTITY
    assert "qwen3-151k-bpe-again" in refusal.value.what


def test_an_unknown_backend_names_the_two_that_exist():
    edited = document()
    edited["host"]["tokenizers"] = [dict(TOKENIZER, backend="rust")]
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.TOKENIZER_IDENTITY
    assert "fast" in refusal.value.remedy and "slow" in refusal.value.remedy


def test_a_tokenizer_entry_is_closed_like_the_rest_of_the_schema():
    edited = document()
    edited["host"]["tokenizers"] = [dict(TOKENIZER, max_model_len=4096)]
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.SEPARATION


# --- shapes ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, value",
    [
        (("host", "cpu", "cores_physical"), True),
        (("host", "cpu", "cores_physical"), 0),
        (("host", "admission_fixed_s"), -1.0),
        (("device", "name"), ""),
        (("device", "arch"), 950),
        (("schema_version",), 2),
    ],
)
def test_a_value_the_schema_cannot_hold_is_refused(path, value):
    edited = document()
    node = edited
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(SpecRefusal):
        MachineSpec.from_mapping(edited)


def test_a_width_table_is_keyed_by_a_width():
    edited = document()
    edited["device"]["runtime_constants"]["driver_and_collective_reserve_bytes"] = {
        "one": 1.0
    }
    with pytest.raises(SpecRefusal):
        MachineSpec.from_mapping(edited)


def test_a_block_written_as_a_scalar_is_refused():
    edited = document()
    edited["device"]["memory"] = 288.0e9
    with pytest.raises(SpecRefusal) as refusal:
        MachineSpec.from_mapping(edited)
    assert refusal.value.rule is Rule.SHAPE


def test_a_spec_is_a_mapping():
    with pytest.raises(SpecRefusal):
        MachineSpec.from_mapping([("schema_version", 1)])


def test_asking_for_a_field_that_is_not_one_is_refused():
    refusal = refused(read(), "device.tensor_parallel_size")
    assert refusal.rule is Rule.SEPARATION
    assert DEPLOYMENT_OWNED["tensor_parallel_size"] in refusal.remedy


# --- the three ways a path fails to resolve ----------------------------------
#
# A dotted path that does not resolve has three causes and they want three
# different actions. The schema is closed, so a key with no row in the field
# table is a typo or a deployment knob and the reader belongs at the table. A
# spec resolves every field it was required to state, so a declared field can
# only be missing from one assembled from parts -- and that reader, sent to the
# table, finds the field sitting in it and stops. A block is in the table and
# holds no value of its own, so that reader wanted one segment more. One
# message cannot be true of all three, so there are three, and the tests below
# are written to fail if any two are swapped: each names its own sentence and
# denies the others'.

#: A key with no row in the field table, and not a knob the engine owns either,
#: so the refusal is the plain closed-schema one and not the forwarding address.
UNDECLARED = "device.clock_mhz"

#: A declared field for a spec to be missing. Any would do; this is the one a
#: reader of an inter-node transfer cost asks for, where the wrong message was
#: first read off a spec that had been assembled without the whole block.
ABSENT = "interconnect.inter_node.link_latency_s"

#: A block that holds fields directly, so the refusal has something to list.
BLOCK = "device.memory"

DECLARED_PATHS = frozenset(field.path for field in SCHEMA)

#: Every dotted prefix of a declared path, which is what a block is. Derived
#: here rather than imported so the two derivations can disagree out loud.
BLOCK_PATHS = frozenset(
    path.rsplit(".", index + 1)[0]
    for path in DECLARED_PATHS
    for index in range(path.count("."))
)


def without(path):
    """The reference spec assembled without one field, as a fragment would be."""
    whole = read()
    return MachineSpec(
        values={p: v for p, v in whole.values.items() if p != path},
        tokenizers=whole.tokenizers,
    )


def refused(machine, path):
    """The refusal `value` raises for a path, for a test to read in full."""
    with pytest.raises(SpecRefusal) as refusal:
        machine.value(path)
    return refusal.value


def test_the_three_paths_under_test_are_what_they_claim_to_be():
    # Each case below asserts something about the schema, so the schema is what
    # decides which case is right, not the three names chosen here.
    assert UNDECLARED not in DECLARED_PATHS and UNDECLARED not in BLOCK_PATHS
    assert UNDECLARED.rsplit(".", 1)[-1] not in DEPLOYMENT_OWNED
    assert ABSENT in DECLARED_PATHS
    assert BLOCK in BLOCK_PATHS and BLOCK not in DECLARED_PATHS


def test_a_key_with_no_row_in_the_table_is_refused_as_not_a_field():
    refusal = refused(read(), UNDECLARED)
    assert refusal.rule is Rule.SEPARATION
    assert UNDECLARED in refusal.what
    assert "is not a field of this schema" in refusal.what
    assert "the schema is closed" in refusal.remedy


def test_a_declared_field_a_spec_lacks_is_never_called_undeclared():
    # Every field in the table, so the message's claim is checked against the
    # table rather than against one hand-picked path -- and so a field added
    # later is covered the day it lands. A loop rather than a parametrize: the
    # table is one branch on `required`, so per-path node ids would name 39
    # cases with two outcomes between them. The path travels in the assertion
    # message instead, which is where a failure needs it.
    for path in sorted(DECLARED_PATHS):
        refusal = refused(without(path), path)
        assert refusal.rule is Rule.TOTALITY, path
        assert path in refusal.what, path
        assert "is declared by this schema" in refusal.what, path
        assert "not a field of this schema" not in refusal.what, path
        assert "the schema is closed" not in refusal.remedy, path


def test_a_block_is_refused_as_a_block_and_names_what_it_groups():
    # The reader who is one segment short. Nothing is wrong with the spec or
    # with the path, so the remedy is the list rather than a direction to go
    # and look -- and "is not a field of this schema" is false twice over,
    # because the block is in the schema and is not something ATOM configures.
    refusal = refused(read(), BLOCK)
    assert refusal.rule is Rule.ADDRESSING
    assert BLOCK in refusal.what
    assert "is a block of this schema, not one of its fields" in refusal.what
    assert "not a field of this schema" not in refusal.what
    assert "the schema is closed" not in refusal.remedy
    for held in ("capacity_bytes", "bandwidth_bytes_per_s", "derate"):
        assert held in refusal.remedy
    # And every other block in the table, including the ones that hold only
    # further blocks, which are the ones a reader is most likely to type.
    for block in sorted(BLOCK_PATHS):
        refusal = refused(read(), block)
        assert refusal.rule is Rule.ADDRESSING, block
        assert "is a block of this schema" in refusal.what, block
        assert "not a field of this schema" not in refusal.what, block


def test_the_three_refusals_are_not_interchangeable():
    # Swap any two messages and this fails; match a substring they share and it
    # would not.
    unknown = refused(read(), UNDECLARED)
    absent = refused(without(ABSENT), ABSENT)
    block = refused(read(), BLOCK)
    assert len({unknown.rule, absent.rule, block.rule}) == 3
    assert "is not a field of this schema" not in str(absent) + str(block)
    assert "carries no value for it" not in str(unknown) + str(block)
    assert "is a block of this schema" not in str(unknown) + str(absent)


def test_a_required_field_says_the_spec_was_assembled_rather_than_read():
    # The remedy a reader can act on: the document path cannot produce this, so
    # the fix is in whatever built the object, not in the document.
    refusal = refused(without(ABSENT), ABSENT)
    assert "from_mapping" in refusal.remedy
    assert "optional" not in refusal.what


def test_an_optional_field_the_document_omits_is_refused_as_optional():
    # Not a hand-built object at all. The reference document states no
    # `provenance.notes`, the reader leaves it out because the field is
    # optional, and a spec read straight from a document then declines a path
    # that "is not a field of this schema" would be false of.
    optional = [field.path for field in SCHEMA if not field.required]
    assert "provenance.notes" in optional
    refusal = refused(read(), "provenance.notes")
    assert refusal.rule is Rule.TOTALITY
    assert "as optional" in refusal.what
    assert "not a field of this schema" not in refusal.what


def test_the_other_accessors_decline_on_a_fragment_rather_than_raise():
    # `runtime_constant` and `check_stack` read the same map, and a KeyError
    # out of a fragment is a refusal nobody can act on.
    name = "persistent_forward_buffer_bytes"
    with pytest.raises(SpecRefusal) as constant:
        without(f"device.runtime_constants.{name}").runtime_constant(name)
    assert constant.value.rule is Rule.TOTALITY
    with pytest.raises(SpecRefusal) as pinned:
        without("device.software_pinned_to.rccl").check_stack(dict(STACK))
    assert pinned.value.rule is Rule.TOTALITY
    # And a block reaches `runtime_constant` too: `cudagraph_pool` is the one
    # name under that section a reader can plausibly ask for without its leaf.
    with pytest.raises(SpecRefusal) as block:
        read().runtime_constant("cudagraph_pool")
    assert block.value.rule is Rule.ADDRESSING


def test_a_width_table_a_fragment_lacks_is_declined_rather_than_raised():
    # A width-keyed constant is read past the point where a single-valued one
    # returns, so it is declined at its own read.
    name = "driver_and_collective_reserve_bytes"
    with pytest.raises(SpecRefusal) as refusal:
        without(f"device.runtime_constants.{name}").runtime_constant(name, 2)
    assert refusal.value.rule is Rule.TOTALITY


# --- what the package reaches ------------------------------------------------


def test_the_package_was_found():
    assert _spec_modules(), f"no modules under {PACKAGE}"


@pytest.mark.parametrize("module", _spec_modules(), ids=_named)
def test_the_package_imports_only_the_standard_library_it_names(module):
    # An allowlist of what the package actually imports. The claim kept is that
    # a spec can be authored and checked on any machine: no device runtime, no
    # engine, and no document parser either -- turning a file into a mapping is
    # the caller's, which keeps a dependency the engine does not declare out of
    # the path that reads a spec. `pathlib` is here for the probe that reads the
    # processor topology the kernel publishes: reading a path is not a device
    # runtime, and nothing on the path that checks a document touches it.
    allowed = {
        "collections",
        "dataclasses",
        "enum",
        "hashlib",
        "json",
        "pathlib",
        "typing",
        "warnings",
    }
    tree = ast.parse(module.read_text())
    roots = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level:
            roots.append((node.module or "").split(".")[0])
    strays = sorted({root for root in roots if root not in allowed})
    assert not strays, f"{_named(module)} imports {strays}; allowed: {sorted(allowed)}"


def test_everything_the_package_exports_is_reachable_by_name():
    # An empty list would let the loop pass having checked nothing, with no
    # count moving. The package exists to re-export its modules' names and this
    # file imports them from it, so exporting none is never a legitimate state.
    assert spec_package.__all__, "atom.compass.spec declares no exports"
    for name in spec_package.__all__:
        assert getattr(spec_package, name, None) is not None, name
