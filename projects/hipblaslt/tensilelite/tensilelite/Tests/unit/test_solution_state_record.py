# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The C++ record builder has to stay on the Python builder.

solutionFromState and Solution.FromOriginalState are maintained together.
A field added on the Python side has to show up here, or this test fails.
Updating the checked-in record then fails SolutionRecord in tensilelite-tests
until the C++ builder emits the same lines. There is no Python binding: the
two tests share the files in test_data/solution_records and nothing else.

Each fixture is a derived state <name>.msgpack, the map FromOriginalState and
solutionFromState read, with the files built from it:

  <name>.record.txt          one line per field of the library record, and the
                             type sequence of each predicate tree
  <name>.predicates.yaml     the three predicate trees with their values, in
                             the {type, index, value} form state() gives them
  <name>.predicates.msgpack  the same trees for the C++ test, which has no
                             YAML parser

SolutionRecord in tensilelite-tests builds the record from the msgpack, writes
it with MessagePackWriteSolutionData and compares the predicate trees node by
node with <name>.predicates.msgpack, so a threshold, an operand or a feature
the C++ builder drops fails there. The fixtures are found by listing the
directory, so adding one needs no change to either test.

The gfx950-plain state is the custom.config block of a generated kernel. The
others come from derive_solution_records.py, which runs a real library logic
file or benchmark config through the Python derivation; see that module for the
fixtures that edit a field the checked-in sources do not reach. The fixtures
together have to cover the cases listed in
test_fixtures_cover_the_predicate_families, so a swapped operand or an unread
feature cannot hide behind a symmetric kernel.
"""

import contextlib
import io
import json
import copy
from pathlib import Path

import msgpack
import pytest
import yaml

from tensilelite.Common import state
from tensilelite.Contractions import Solution

pytestmark = pytest.mark.unit


def records_dir():
    return Path(__file__).resolve().parent / "test_data" / "solution_records"


def fixture_names():
    """Every derived state in the directory: <name>.msgpack, not <name>.predicates.msgpack."""
    return sorted(
        path.name[: -len(".msgpack")]
        for path in records_dir().glob("*.msgpack")
        if not path.name.endswith(".predicates.msgpack")
    )


FIXTURES = fixture_names()


def _tags(predicate):
    found = []

    def walk(node):
        if isinstance(node, dict) and "type" in node:
            found.append(node["type"])
            value = node.get("value")
            if isinstance(value, list):
                for item in value:
                    walk(item)
            elif isinstance(value, dict):
                walk(value)

    walk(state(predicate))
    return found


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def _built(derived):
    """Solution.RecordFromOriginalState, the part of FromOriginalState the library write reads."""
    with contextlib.redirect_stdout(io.StringIO()):
        return Solution.RecordFromOriginalState(copy.deepcopy(derived))


def _linear_model(model):
    # The defaults the library loader gives a key the record leaves out.
    return {
        "slope": float(model.get("slope", 1.0)),
        "intercept": float(model.get("intercept", 0.0)),
        "max": float(model.get("max", 1000.0)),
    }


def library_record(derived):
    """The library record FromOriginalState builds from one derived state, as the library reads it."""
    solution = _built(derived)
    # FromOriginalState leaves these unset without SolutionNameMin,
    # KernelNameMin or SolutionIndex. ContractionSolution has no unset value,
    # and applyNaming and FixSolutionIndices set them before a library write.
    name = "" if solution.name is None else solution.name
    kernelName = getattr(solution, "kernelName", "")
    index = 0 if solution.index is None else solution.index
    return {
        "solutionName": name,
        "kernelName": kernelName,
        "sizeMapping": state(solution.sizeMapping),
        "internalArgsSupport": state(solution.internalArgsSupport),
        "problemType": state(solution.problemType),
        "problemPredicate": _tags(solution.problemPredicate),
        "taskPredicate": _tags(solution.taskPredicate),
        "hardwarePredicate": _tags(solution.hardwarePredicate),
        "customKernel": state(solution.customKernel),
        "debugKernel": solution.debugKernel,
        "libraryLogicIndex": solution.libraryLogicIndex,
        "index": index,
        "ideals": {int(key): float(value) for key, value in solution.ideals.items()},
        "linearModel": _linear_model(solution.linearModel),
    }


def predicate_payload(derived):
    """Predicate arguments, not only their types. A threshold change fails this."""
    solution = _built(derived)
    return {
        "problem": state(solution.problemPredicate),
        "task": state(solution.taskPredicate),
        "hardware": state(solution.hardwarePredicate),
    }


def record_lines(record):
    lines = []

    def emit(prefix, value):
        if isinstance(value, dict) and value:
            for key, item in value.items():
                emit(f"{prefix}.{key}" if prefix else str(key), item)
        else:
            lines.append(f"{prefix}={_json(value)}")

    for key, value in record.items():
        emit(key, value)
    return lines


def _load_derived(name):
    packed = (records_dir() / f"{name}.msgpack").read_bytes()
    return msgpack.unpackb(packed, strict_map_key=False)


@pytest.mark.parametrize("name", FIXTURES)
def test_python_builder_matches_checked_in_record(name):
    derived = _load_derived(name)
    expected = (records_dir() / f"{name}.record.txt").read_text().splitlines()
    assert record_lines(library_record(derived)) == expected


@pytest.mark.parametrize("name", FIXTURES)
def test_predicate_arguments_match_checked_in_payload(name):
    derived = _load_derived(name)
    path = records_dir() / f"{name}.predicates.yaml"
    with path.open() as handle:
        expected = yaml.safe_load(handle)
    assert predicate_payload(derived) == expected


@pytest.mark.parametrize("name", FIXTURES)
def test_predicate_msgpack_matches_yaml(name):
    """The C++ test reads the msgpack copy of the predicate trees. It has to be the YAML."""
    with (records_dir() / f"{name}.predicates.yaml").open() as handle:
        expected = yaml.safe_load(handle)
    packed = (records_dir() / f"{name}.predicates.msgpack").read_bytes()
    assert msgpack.unpackb(packed, strict_map_key=False) == expected


def _predicate_types(node):
    found = []
    if isinstance(node, dict):
        if "type" in node:
            found.append(node["type"])
        for item in node.values():
            found.extend(_predicate_types(item))
    elif isinstance(node, list):
        for item in node:
            found.extend(_predicate_types(item))
    return found


def test_fixtures_cover_the_predicate_families():
    """A fixture set of symmetric kernels cannot tell A from B or M from N.

    Each case below is one way a C++ builder can be wrong and still match a
    square, default-feature kernel. A fixture that stops covering a case fails
    here, not silently in the C++ comparison.
    """
    states = {name: _load_derived(name) for name in FIXTURES}
    payloads = {name: predicate_payload(state) for name, state in states.items()}

    def problem(state):
        return state["ProblemType"]

    def some(test):
        return [name for name, state in states.items() if test(state)]

    assert some(lambda s: s["MacroTile0"] != s["MacroTile1"]), "no non-square tile"
    assert some(lambda s: s["MacroTile0"] >= 128 or s["MacroTile1"] >= 128), "no large tile"
    assert some(
        lambda s: s["GlobalReadVectorWidthA"] != s["GlobalReadVectorWidthB"]
    ), "GlobalReadVectorWidthA is always B"
    assert len({s["DepthU"] for s in states.values()}) >= 3, "DepthU barely varies"
    assert some(lambda s: problem(s)["DataTypeA"] != problem(s)["DataTypeB"]), "A type is B type"
    assert len({tuple(s["ISA"]) for s in states.values()}) >= 4, "too few architectures"

    assert some(lambda s: problem(s).get("MXBlockA", 0) > 0), "no MX kernel"
    assert some(lambda s: s.get("MXScaleFormat", "NoSwizzle") != "NoSwizzle"), "no MX scale format"
    assert some(
        lambda s: problem(s).get("MXBlockA", 0) > 0
        and problem(s).get("MXBlockFreeA", 1) != problem(s).get("MXBlockFreeB", 1)
    ), "MXBlockFree is always 1 or the same for A and B"
    assert some(
        lambda s: len(problem(s).get("ScaleBlockSizesA", [])) > 1
        and problem(s).get("Int4EncodingA") == "UnsignedBias8"
    ), "no int4 kernel with several group sizes and UnsignedBias8"
    assert some(lambda s: problem(s).get("StochasticRounding")), "no stochastic rounding"
    assert some(lambda s: s.get("TileProcessingStrategy") == "StreamK"), "no StreamK kernel"
    assert some(lambda s: s.get("ReuseAcrossPersistent")), "no ReuseAcrossPersistent kernel"
    assert some(
        lambda s: s["InternalSupportParams"].get("PersistentLoopArgsVersion") == 2
    ), "no PersistentLoopArgsVersion 2 kernel"
    assert some(
        lambda s: s["InternalSupportParams"].get("SupportOccupancyProbe")
    ), "no occupancy probe"

    types = {name: set(_predicate_types(payload)) for name, payload in payloads.items()}
    assert any("Or" in kinds for kinds in types.values()), "no Or predicate"
    assert any("ScaleBlockSizeA" in kinds for kinds in types.values())
    assert any("GlobalSplitUCheckMinK" in kinds for kinds in types.values())


def test_record_holds_every_solution_state_key():
    """A field added to Solution.StateKeys has to reach the record and the C++ comparison."""
    record = library_record(_load_derived("gfx950-plain"))
    renamed = {"name": "solutionName"}
    assert {renamed.get(key, key) for key in Solution.StateKeys} == set(record)


@pytest.mark.parametrize(
    "name, edit",
    [
        ("gfx1250-streamk-cluster-rap-probe", lambda s: s.update(ClusterDim=[1, 3])),
        ("gfx1151-int4", lambda s: s["ProblemType"].update(Int4EncodingA="Unsigned")),
    ],
    ids=["cluster-1x3", "int4-unsigned"],
)
def test_builds_records_solution_from_state_rejects(name, edit):
    """SolutionState.hpp lists the states solutionFromState rejects and this builder accepts."""
    derived = copy.deepcopy(_load_derived(name))
    edit(derived)
    library_record(derived)


def test_mx_block_free_is_recorded_with_mx_predicates():
    derived = copy.deepcopy(_load_derived("gfx950-plain"))
    derived["ProblemType"]["MXBlockA"] = 16
    derived["ProblemType"]["MXBlockFreeA"] = 7
    record = library_record(derived)
    assert record["problemType"]["mxBlockFreeA"] == 7
    assert "MXBlockA" in record["problemPredicate"]
    assert "MXBlockFreeA" in record["problemPredicate"]


def test_occupancy_probe_support_is_recorded():
    derived = copy.deepcopy(_load_derived("gfx950-plain"))
    derived["InternalSupportParams"]["SupportOccupancyProbe"] = True
    record = library_record(derived)
    assert record["internalArgsSupport"]["occupancyProbe"] is True


@pytest.mark.parametrize("declared", (True, False))
def test_custom_kernel_uniform_summation_order_is_recorded(declared):
    derived = copy.deepcopy(_load_derived("gfx950-plain"))
    derived["CustomKernel"]["uniformSummationOrder"] = declared
    record = library_record(derived)
    assert record["customKernel"]["uniformSummationOrder"] is declared
    assert f"customKernel.uniformSummationOrder={_json(declared)}" in record_lines(record)


def test_custom_kernel_uniform_summation_order_defaults_to_false():
    derived = copy.deepcopy(_load_derived("gfx950-plain"))
    assert "uniformSummationOrder" not in derived["CustomKernel"]
    record = library_record(derived)
    assert record["customKernel"]["uniformSummationOrder"] is False


def _pack(value):
    return msgpack.packb(value, use_bin_type=True)


def write_fixture(name):
    """The files built from <name>.msgpack. Each C++ comparison reads them back."""
    derived = _load_derived(name)
    directory = records_dir()
    lines = record_lines(library_record(derived))
    (directory / f"{name}.record.txt").write_text("\n".join(lines) + "\n")
    payload = predicate_payload(derived)
    (directory / f"{name}.predicates.yaml").write_text(yaml.safe_dump(payload, sort_keys=False))
    (directory / f"{name}.predicates.msgpack").write_bytes(_pack(payload))
    return len(lines)


if __name__ == "__main__":
    # Regenerates the checked-in files after a deliberate Python change, or after
    # derive_solution_records.py wrote a new <name>.msgpack. The C++ test then
    # fails until solutionFromState builds the new record and predicate trees.
    for name in fixture_names():
        print(name, write_fixture(name))
