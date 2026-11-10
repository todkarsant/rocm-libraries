# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for TDMStore: the store paths of each level, solution validation, the kernel-name
token, the LDS layout, the store modifiers and the emitted TDM block: its C load and the cache
hints of C, its fp16 and narrow store vectors, and the waits around it after StinkyTofu."""

import re
from collections import defaultdict
from copy import deepcopy

import pytest
from rocisa.enum import CacheScope, NonVolatile, TemporalHint

from tensilelite.Common.DataType import DataType
from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Components.TDMStore import STINKY_KEEP, TDM_STORE_PATHS, tdmStoreBlocks, tdmStoreLdsBytes, \
  tdmStoreModifiers, tdmStoreRejectReason
from tensilelite.SolutionStructs.Naming import getKernelNameMin

from emit_test_helpers import assert_every_vector_preload_barriers, emit_asm
from toolchain_test_helpers import copy_isa_info_map

pytestmark = pytest.mark.unit

_LEVELS = [1, 2, 3]

# The ProblemType keys an override of _state may name; every other override goes to the solution.
_PROBLEM_TYPE_KEYS = {
  "DestDataType", "ComputeDataType", "HighPrecisionAccumulate", "FusedGemmA2A", "UseE", "Gradient",
  "UseGateResidual", "GroupedGemm", "UseBias", "UseScaleAlphaVec", "UseScaleAB", "UseScaleCD",
  "ActivationType", "OutputAmaxD",
}


def _state(**overrides):
  """The keys tdmStoreRejectReason and the ExecutionPolicy helpers it calls read, for a TDMStore=2
  solution inside the supported configurations, with `overrides` applied."""
  problemType = {
    "DestDataType": DataType("b"),
    "ComputeDataType": DataType("s"),
    "HighPrecisionAccumulate": True,
    "FusedGemmA2A": False,
    "UseE": False,
    "Gradient": False,
    "UseGateResidual": False,
    "UseScaleCD": False,
    "OutputAmaxD": False,
    "GroupedGemm": False,
    "UseBias": 0,
    "UseScaleAlphaVec": 0,
  }
  state = {
    "TDMStore": 2,
    "ISA": (12, 5, 0),
    "BufferStore": True,
    "TileProcessingStrategy": "None",
    "WorkAssignment": "StaticGrid",
    "StreamKAtomic": 0,
    "ClusterDim": [1, 1],
    "PrefetchAcrossPersistent": 0,
    "ReuseAcrossPersistent": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "AdaptiveGemmGSUA": 0,
    "UseSubtileImpl": False,
    "StoreRemapVectorWidth": 0,
    "CompactLoopStore": False,
    "MacroTile1": 256,
    "NumThreads": 128,
    "WavefrontSize": 32,
    "ProblemType": problemType,
  }
  for key, value in overrides.items():
    (problemType if key in _PROBLEM_TYPE_KEYS else state)[key] = value
  return state


@pytest.mark.parametrize("dest", ["b", "h"])
def test_accepts_bf16_and_fp16(dest):
  assert tdmStoreRejectReason(_state(DestDataType=DataType(dest))) is None


def test_accepts_non_atomic_streamk3():
  assert tdmStoreRejectReason(_state(TileProcessingStrategy="StreamK", GlobalSplitU=0)) is None


def test_accepts_multiple_buffer_dynamic_gsu():
  assert tdmStoreRejectReason(_state(GlobalSplitU=-1, UseBias=1, UseScaleAlphaVec=1)) is None


@pytest.mark.parametrize("overrides", [
  {"AdaptiveGemm": 1},
  {"StoreRemapVectorWidth": -1},
  {"UseBias": 1},
  {"UseScaleAlphaVec": 1},
  {"UseScaleAB": "Scalar"},
  {"UseScaleAB": "Vector"},
  {"ActivationType": "relu"},
  {"ActivationType": "hipblaslt_all"},
  {"UseBias": 1, "UseScaleAlphaVec": 1, "ActivationType": "hipblaslt_all",
   "TileProcessingStrategy": "StreamK", "GlobalSplitU": 0},
])
def test_accepts_supported_options(overrides):
  """AdaptiveGemm, the automatic StoreRemapVectorWidth (-1, which Solution resolves and checks), the
  epilogue vectors along M, ScaleAB and the activations."""
  assert tdmStoreRejectReason(_state(**overrides)) is None


def test_disabled_is_not_validated():
  assert tdmStoreRejectReason(_state(TDMStore=0, ISA=(9, 5, 0), GlobalSplitU=4)) is None


_HPA = "TDMStore requires HighPrecisionAccumulate with a single-precision ComputeDataType"
_GSU_DYNAMIC = "TDMStore supports GlobalSplitU=-1 only with MultipleBuffer and AdaptiveGemmGSUA=0"
_PERSISTENT = "TDMStore supports persistent kernels only as non-atomic StreamK=3"
_ACROSS_TILES = \
  "TDMStore does not support PrefetchAcrossPersistent, ReuseAcrossPersistent or StreamK spatial clusters"
_PROBLEM = "TDMStore does not support E, Gradient, gate residual, ScaleCD, AmaxD or GroupedGemm"
_ALONG_M = "TDMStore supports bias and ScaleAlphaVec only along M"
_STREAMK = {"TileProcessingStrategy": "StreamK", "GlobalSplitU": 0}


@pytest.mark.parametrize("overrides, reason", [
  ({"ISA": (9, 5, 0)}, "TDMStore requires gfx1250"),
  ({"DestDataType": DataType("s")}, "TDMStore supports only bf16 and fp16 DestDataType"),
  ({"HighPrecisionAccumulate": False}, _HPA),
  ({"ComputeDataType": DataType("h")}, _HPA),
  ({"BufferStore": False}, "TDMStore requires BufferStore"),
  ({"GlobalSplitU": 2}, "TDMStore requires GlobalSplitU=1 or -1"),
  ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "SingleBuffer"}, _GSU_DYNAMIC),
  ({"GlobalSplitU": -1, "GlobalSplitUAlgorithm": "MultipleBufferSingleKernel"}, _GSU_DYNAMIC),
  ({"GlobalSplitU": -1, "AdaptiveGemmGSUA": 1}, _GSU_DYNAMIC),
  ({"TileProcessingStrategy": "DataParallel", "GlobalSplitU": 0}, _PERSISTENT),
  (dict(_STREAMK, WorkAssignment="DynamicWorkQueue"), _PERSISTENT),
  (dict(_STREAMK, StreamKAtomic=1), _PERSISTENT),
  (dict(_STREAMK, PrefetchAcrossPersistent=1), _ACROSS_TILES),
  (dict(_STREAMK, ReuseAcrossPersistent=1), _ACROSS_TILES),
  (dict(_STREAMK, ClusterDim=[2, 1]), _ACROSS_TILES),
  ({"UseSubtileImpl": True}, "TDMStore does not support UseSubtileImpl"),
  ({"StoreRemapVectorWidth": 4}, "TDMStore does not support StoreRemapVectorWidth"),
  ({"CompactLoopStore": True}, "TDMStore does not support CompactLoopStore"),
  ({"FusedGemmA2A": True}, "TDMStore does not support FusedGemmA2A"),
  ({"UseE": True}, _PROBLEM),
  ({"Gradient": True}, _PROBLEM),
  ({"UseGateResidual": True}, _PROBLEM),
  ({"UseScaleCD": True}, _PROBLEM),
  ({"OutputAmaxD": True}, _PROBLEM),
  ({"GroupedGemm": True}, _PROBLEM),
  ({"UseBias": 2}, _ALONG_M),
  ({"UseBias": 3}, _ALONG_M),
  ({"UseScaleAlphaVec": 2}, _ALONG_M),
  ({"NumThreads": 96}, "TDMStore requires MacroTile1 to be a multiple of the number of waves"),
])
def test_rejects_unsupported(overrides, reason):
  assert tdmStoreRejectReason(_state(**overrides)) == reason


def test_levels_match_the_valid_values_and_add_one_path_each():
  assert list(TDM_STORE_PATHS) == validParameters["TDMStore"]
  for level in _LEVELS:
    assert TDM_STORE_PATHS[level][:-1] == TDM_STORE_PATHS[level - 1]


def _blocks(level, edge=True, atomic=False, accumulation=None, vectorWidths=(8, 1), bpe=2):
  vectorWidths = list(vectorWidths)
  elements = ["elements%u" % vw for vw in vectorWidths]
  return tdmStoreBlocks({"TDMStore": level, "_GlobalAccumulation": accumulation}, edge, atomic, bpe,
                        vectorWidths, elements)


# The store blocks generateBetaModules gets with StoreVectorWidth 8: AdaptiveGemm=0 keeps the
# widest and the narrowest, AdaptiveGemm=1 every width.
_VECTOR_WIDTHS = {0: (8, 1), 1: (8, 4, 2, 1)}


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_blocks_of_each_level(level, adaptive):
  """EdgeM takes the slot of the narrowest path; EdgeN leaves one edge block; NonEdge drops the edge checks."""
  vectorWidths = list(_VECTOR_WIDTHS[adaptive])
  expected = {
    0: (vectorWidths, ["elements%u" % vw for vw in vectorWidths], None, True),
    1: ([8, 1], ["elements8", "elements8"], 1, True),
    2: ([8], ["elements8"], 8, True),
    3: ([8], ["elements8"], 8, False),
  }[level]
  assert _blocks(level, vectorWidths=vectorWidths) == expected


@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_blocks_without_edge(level):
  """Only level 3 stores the unmasked path of a store without edge blocks through the TDM block."""
  assert _blocks(level, edge=False, vectorWidths=(8,)) == ([8], ["elements8"], 8 if level == 3 else None, False)


def test_level_1_needs_a_path_narrower_than_the_store_vector_width():
  assert _blocks(1, vectorWidths=(1,)) == ([1], ["elements1"], None, True)
  assert _blocks(2, vectorWidths=(1,)) == ([1], ["elements1"], 1, True)


@pytest.mark.parametrize("accumulation, atomic", [
  ("MultipleBuffer", False),
  ("MultipleBufferSingleKernel", False),
  ("SingleBuffer", True),
  (None, True),
])
def test_partial_and_atomic_stores_keep_buffer_store(accumulation, atomic):
  assert _blocks(3, atomic=atomic, accumulation=accumulation) == \
    ([8, 1], ["elements8", "elements1"], None, True)


@pytest.mark.parametrize("level", _LEVELS)
def test_four_byte_elements_keep_buffer_store(level):
  """fp32 elements, such as GSU>1 partials at the compute type, never go through the TDM block."""
  assert _blocks(level, bpe=4) == ([8, 1], ["elements8", "elements1"], None, True)
  assert _blocks(level, edge=False, vectorWidths=(8,), bpe=4) == ([8], ["elements8"], None, False)


def test_streamk_final_store_uses_the_tdm_block():
  assert _blocks(2, accumulation="PartialsBuffer") == ([8], ["elements8"], 8, True)


@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_lds_bytes_of_the_staged_tile(level):
  state = {"TDMStore": level, "MacroTile0": 256, "MacroTile1": 128, "ProblemType": {"DestDataType": DataType("h")}}
  assert tdmStoreLdsBytes(state) == (256 * 128 * 2 if level else 0)


def test_store_modifiers_follow_the_hints_of_d():
  """tensor_store_from_lds takes NonTemporalD, TemporalHintD and NonVolatileD as addStore decodes them.

  TH_WB and TH_LU share one value, so th is compared by value: whether the bound enum makes them
  one Python object depends on the rocisa backend.
  """
  caps = {"HasTHModifier": True}
  kernel = {"NonTemporalD": 2, "TemporalHintD": TemporalHint.TH_WB.value, "NonVolatileD": NonVolatile.NV.value}
  mods = tdmStoreModifiers(caps, kernel)
  assert (mods.scope, mods.th.value, mods.nv, mods.isStore) == \
    (CacheScope.SCOPE_DEV, TemporalHint.TH_WB.value, NonVolatile.NV, True)
  assert tdmStoreModifiers(caps, {"NonTemporalD": 0, "TemporalHintD": 0, "NonVolatileD": 0}) is None
  assert tdmStoreModifiers({"HasTHModifier": False}, kernel) is None


def _kernel(tdmStore):
  """The smallest kernel dict that exercises Naming._getName."""
  kernel = deepcopy(defaultSolution)
  kernel["ProblemType"] = {
    "OperationIdentifier": "Cijk_Alik_Bljk",
    "DataType": 0,
    "DestDataType": 0,
    "ComputeDataType": 0,
    "GroupedGemm": False,
    "UseBeta": False,
    "UseBias": 0,
  }
  kernel["MacroTile0"] = 256
  kernel["MacroTile1"] = 256
  kernel["DepthU"] = 64
  kernel["MatrixInstM"] = 16
  kernel["MatrixInstN"] = 16
  kernel["MatrixInstB"] = 1
  kernel["MatrixInstruction"] = [16, 16, 32, 1]
  kernel["MIWaveTile"] = [8, 8]
  kernel["WorkGroup"] = [32, 4, 1]
  kernel["ISA"] = (12, 5, 0)
  kernel["TDMStore"] = tdmStore
  return kernel


@pytest.mark.parametrize("level", _LEVELS)
def test_kernel_name_token_only_when_enabled(level):
  """TDMStore=0 keeps every existing kernel name; each level adds its TDMS token."""
  off = getKernelNameMin(_kernel(0), False)
  on = getKernelNameMin(_kernel(level), False)
  assert "TDMS%u" % level not in off
  assert on.replace("_TDMS%u" % level, "", 1) == off


def _params(gfx1250_iim, mi=None, **overrides):
  """A bf16 TN gfx1250 kernel with beta: by default MT256x256 over 4 waves, StoreVectorWidth 8."""
  from tensilelite.Common.Architectures import gfxToIsa
  from tensilelite.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters

  isa = gfxToIsa("gfx1250")
  mi = mi or [16, 16, 32, 1, 1, 8, 8, 2, 2]
  problemType = {
    "OperationType": "GEMM", "DataType": "B", "MacDataTypeA": "B", "MacDataTypeB": "B",
    "DestDataType": "B", "ComputeDataType": "s", "HighPrecisionAccumulate": True,
    "TransposeA": True, "TransposeB": False, "UseBeta": True, "Batched": True,
    "MXBlockA": 0, "MXBlockB": 0,
  }
  problemType.update(overrides.pop("ProblemType", {}))
  params = {
    "ProblemType": problemType, "ISA": isa, "MatrixInstruction": mi,
    "WorkGroup": [16, 16, 1], "WavefrontSize": 32, "DepthU": 64,
    "KernelLanguage": "Assembly", "PrefetchGlobalRead": 2, "PrefetchLocalRead": 1,
    "ScheduleIterAlg": 4, "StaggerU": 0, "GlobalSplitU": 1, "InnerUnroll": 1,
    "TransposeLDS": -1, "LdsPadA": -1, "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1, "LdsBlockSizePerPadB": -1, "1LDSBuffer": 0,
    "VectorWidthA": -1, "VectorWidthB": -1, "StoreVectorWidth": -1,
    "GlobalReadVectorWidthA": -1, "GlobalReadVectorWidthB": -1,
    "LocalReadVectorWidth": -1, "SourceSwap": False, "ExpandPointerSwap": False,
    "GlobalSplitUAlgorithm": "MultipleBuffer", "TDMInst": 3, "LDSTrInst": False,
    "StreamK": 0, "StreamKForceDPOnly": 0, "PrefetchAcrossPersistent": 0,
    "UseSubtileImpl": False, "StoreRemapVectorWidth": 0,
    "DirectToVgprA": False, "DirectToVgprB": False,
    "DirectToVgprSparseMetadata": False, "WorkGroupMapping": 1,
    "TDMFuse": 0, "TDMSplit": False, "InitCIterWmma": 0,
  }
  params.update(overrides)
  params.update(matrixInstructionToMIParameters(
    mi, isa, params["WavefrontSize"], problemType, params["WorkGroup"], gfx1250_iim))
  return params


def _emit_asm(gfx1250_iim, assembler, mi=None, **overrides):
  """(solution, assembly text or None) of the kernel of _params. CPU-only; no GPU is touched."""
  return emit_asm(gfx1250_iim, assembler, _params(gfx1250_iim, mi, **overrides))


# The store blocks of a GlobalSplitU=1 kernel are labelled by write path: GSU1 stores D directly,
# MB copies the GSU>1 partials to the workspace and OptNLL_MB is the store of the optimized NLL.
_LABEL = re.compile(r"^label_GW_B(\d)_FD0_VW(\d+)_(\w+?)_(Then|NonEdge):")
_STORE = re.compile(r"^\s*buffer_store\S* ")
_STORE_D = re.compile(r"^\s*buffer_store\S* [^\n]*s\[sgprSrdD:")


def _stores(asm):
  """{(path, beta): Then vector widths, NonEdge label, slots of the TDM stores, D buffer_store count,
  buffer_store count}."""
  paths = defaultdict(lambda: {"then": [], "nonEdge": False, "tdm": [], "buffer": 0, "stores": 0})
  block = None
  for line in asm.splitlines():
    m = _LABEL.match(line)
    if m:
      block = paths[(m.group(3), m.group(1))]
      slot = int(m.group(2))
      if m.group(4) == "Then":
        block["then"].append(slot)
      else:
        block["nonEdge"] = True
    elif block is not None and "tensor_store_from_lds " in line:
      block["tdm"].append(slot)
    elif block is not None and _STORE.match(line):
      block["stores"] += 1
      block["buffer"] += bool(_STORE_D.match(line))
  return paths


def _tdmBlock(asm, beta, slot):
  """The GSU1 TDM block of `beta`, the Then block of vector width `slot`, up to its tensor_store_from_lds."""
  then = re.search(r"^label_GW_B%s_FD0_VW%u_GSU1_Then:" % (beta, slot), asm, re.M)
  assert then, "no GSU1 Then block of vector width %u for beta %s" % (slot, beta)
  return asm[then.end():asm.index("tensor_store_from_lds ", then.end())]


def _code(line):
  """The instruction of an assembly line, without its comment."""
  return line.partition("//")[0].strip()


# Level 0 has no TDM block, so it is emitted with the default TDMInst only.
_LEVEL_TDM_INST = [(0, 3)] + [(level, tdmInst) for level in _LEVELS for tdmInst in (0, 3)]


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("level, tdmInst", _LEVEL_TDM_INST)
def test_emitted_paths_of_each_level(gp_gfx1250, gfx1250_iim, assembler, capsys, level, tdmInst, adaptive,
                                     sourceSwap):
  """Each beta of the direct store keeps the paths below its level and stores the others through one
  TDM block, which reads C from the staged tile; the GSU>1 copy of the partials keeps buffer_store.
  The block drains tensor_load_to_lds before it reuses the main-loop LDS only when A and B are TDM loads."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=level, TDMInst=tdmInst, AdaptiveGemm=adaptive,
                       SourceSwap=sourceSwap)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  cascade = list(_VECTOR_WIDTHS[adaptive])
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == {0: cascade, 1: [8, 1], 2: [8], 3: [8]}[level]
    assert direct["tdm"] == {0: [], 1: [1], 2: [8], 3: [8]}[level]
    assert direct["nonEdge"] == (level < 3) and (direct["buffer"] > 0) == (level < 3)
  partials = stores[("MB", "0")]
  assert sorted(partials["then"], reverse=True) == cascade and partials["nonEdge"] and not partials["tdm"]
  if level:
    block = _tdmBlock(asm, "1", {1: 1, 2: 8, 3: 8}[level])
    assert not re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:", block, re.M)
    loads = list(re.finditer(r"^\s*tensor_load_to_lds ", block, re.M))
    assert len(loads) == 1 and re.search(r"^\s*ds_load\S* ", block[loads[0].end():], re.M)
    drain = re.search(r"^\s*s_wait_tensorcnt[^\n]*drain tensor_load_to_lds", asm, re.M)
    assert bool(drain) == (tdmInst == 3)


@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_optnll_store_goes_through_the_tdm_block_at_level_3(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """The OptNLL store is an unmasked full-tile store of D, so only level 3 sends it through the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=level)
  assert sol.get("Valid"), capsys.readouterr().out
  optNLL = _stores(asm)[("OptNLL_MB", "0")]
  assert optNLL["then"] == [8]
  assert optNLL["tdm"] == ([8] if level == 3 else [])
  assert (optNLL["buffer"] > 0) == (level < 3)


@pytest.mark.parametrize("adaptive", [0, 1])
@pytest.mark.parametrize("level", [0] + _LEVELS)
def test_sourceswap_store_vector_width_2(gp_gfx1250, gfx1250_iim, assembler, capsys, level, adaptive):
  """With SourceSwap the store vector width is VectorWidthA: MIWaveTile0=2 gives 2, whose only
  narrower path is VW1, so level 1 sends the VW1 path through the TDM block."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, mi=[16, 16, 32, 1, 1, 2, 8, 4, 1], TDMStore=level,
                       AdaptiveGemm=adaptive, SourceSwap=True)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 2
  stores = _stores(asm)
  for beta in "01":
    direct = stores[("GSU1", beta)]
    assert sorted(direct["then"], reverse=True) == {0: [2, 1], 1: [2, 1], 2: [2], 3: [2]}[level]
    assert direct["tdm"] == {0: [], 1: [1], 2: [2], 3: [2]}[level]


_FP16 = {"DataType": "H", "MacDataTypeA": "H", "MacDataTypeB": "H", "DestDataType": "H"}


def test_fp16_c_is_read_back_from_the_staged_tile(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """fp16 C comes back from the staged tile in 16-byte vectors at StoreVectorWidth 8."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=2, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 8
  block = _tdmBlock(asm, "1", 8)
  load = re.search(r"^\s*tensor_load_to_lds ", block, re.M)
  assert load and re.search(r"^\s*ds_load_b128 ", block[load.end():], re.M)


@pytest.mark.parametrize("level", [1, 2, 3])
def test_fp16_with_beta_at_store_vector_width_1(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """At StoreVectorWidth 1 level 1 has no narrower path and keeps buffer_store. The TDM block of
  levels 2 and 3 reads each C element back into its own VGPR: an even one with ds_load_u16, an odd
  one with ds_load_u16_d16_hi into the high half, where readInput puts it."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=level, StoreVectorWidth=1, ProblemType=_FP16)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 1
  if level == 1:
    assert "tensor_store_from_lds " not in asm
    return
  loads = re.findall(r"^\s*(ds_load_\w+) v\S+, .*// load C element \((\d+), (\d+)\)", _tdmBlock(asm, "1", 1), re.M)
  assert {mnemonic for mnemonic, _, _ in loads} == {"ds_load_u16", "ds_load_u16_d16_hi"}
  for mnemonic, coord0, coord1 in loads:
    assert (mnemonic == "ds_load_u16_d16_hi") == (int(coord0) % 2 == 1), (mnemonic, coord0, coord1)


@pytest.mark.parametrize("level", _LEVELS)
def test_store_vector_width_4(gp_gfx1250, gfx1250_iim, assembler, capsys, level):
  """At StoreVectorWidth 4 level 1 sends the VW1 path through the TDM block, which stages 8-byte vectors."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=level, StoreVectorWidth=4)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreVectorWidth"] == 4
  slot = {1: 1, 2: 4, 3: 4}[level]
  stores = _stores(asm)
  for beta in "01":
    assert stores[("GSU1", beta)]["tdm"] == [slot]
    assert re.search(r"^\s*ds_store_b64 ", _tdmBlock(asm, beta, slot), re.M)


def test_dynamic_gsu_with_epilogue_vectors_at_level_3(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """GlobalSplitU=-1 with bias and ScaleAlphaVec: the direct and the OptNLL store go through the TDM
  block, and the copy of the GSU>1 partials keeps buffer_store."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=3, GlobalSplitU=-1,
                       ProblemType={"UseBias": 1, "UseScaleAlphaVec": 1})
  assert sol.get("Valid"), capsys.readouterr().out
  stores = _stores(asm)
  for beta in "01":
    assert stores[("GSU1", beta)]["tdm"] == [8]
  assert stores[("OptNLL_MB", "0")]["tdm"] == [8]
  partials = stores[("MB", "0")]
  assert not partials["tdm"] and partials["stores"] > 0


def test_automatic_store_remap_vector_width(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """StoreRemapVectorWidth=-1 resolves to 0 with SourceSwap, which TDMStore takes; Solution rejects a
  nonzero resolved width."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=2, SourceSwap=True, StoreRemapVectorWidth=-1)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["StoreRemapVectorWidth"] == 0
  assert "tensor_store_from_lds " in asm
  capsys.readouterr()
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=2, SourceSwap=False, StoreRemapVectorWidth=-1,
                       VectorWidthA=1)
  assert not sol.get("Valid") and asm is None
  assert "TDMStore does not support StoreRemapVectorWidth" in capsys.readouterr().out


def _cacheTokens(line):
  """The th: and scope: tokens of an instruction line, as {name: value}."""
  return dict(re.findall(r"\b(th|scope):(\S+)", _code(line)))


def test_c_load_takes_the_hints_of_c(gp_gfx1250, gfx1250_iim, assembler, capsys):
  """The tensor_load_to_lds of C carries the th: and scope: of the buffer_load of C on the other paths."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=2, TemporalHintC=3, NonTemporalC=2)
  assert sol.get("Valid"), capsys.readouterr().out
  load = re.search(r"^\s*tensor_load_to_lds [^\n]*", _tdmBlock(asm, "1", 8), re.M)
  bufferLoad = re.search(r"^\s*buffer_load\S* [^\n]*s\[sgprSrdC:[^\n]*", asm, re.M)
  assert load and bufferLoad
  expected = _cacheTokens(bufferLoad.group(0))
  assert "th" in expected
  assert _cacheTokens(load.group(0)) == expected


def test_requires_tdm_in_the_assembler(gp_gfx1250, gfx1250_iim, assembler, capsys):
  from tensilelite.Common.Architectures import gfxToIsa

  noTdm = copy_isa_info_map(gfx1250_iim)
  noTdm[gfxToIsa("gfx1250")].asmCaps["HasTDM"] = False
  sol, asm = _emit_asm(noTdm, assembler, TDMStore=2, TDMInst=0)
  assert not sol.get("Valid") and asm is None
  assert "TDMStore requires asmCaps HasTDM" in capsys.readouterr().out


def _nextLine(lines, start, prefixes):
  """The index of the first line after `start` whose instruction starts with one of `prefixes`, or None."""
  return next((i for i in range(start + 1, len(lines)) if _code(lines[i]).startswith(prefixes)), None)


@pytest.mark.parametrize("streamk", [0, 3])
@pytest.mark.parametrize("level", [2, 3])
def test_tdm_waits_survive_stinkytofu(gp_gfx1250, gfx1250_iim, assembler, capsys, level, streamk):
  """In the emitted assembly the drain of tensor_load_to_lds still comes before the next barrier, and
  every tensor_store_from_lds is waited for before the next barrier or the end of the kernel."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=level, TDMInst=3, StreamK=streamk)
  assert sol.get("Valid"), capsys.readouterr().out
  lines = asm.splitlines()
  # Only the StinkyTofu pipeline writes STINKY_TOTAL_INST_BYTES; the kept staging wait shows its marker.
  assert "STINKY_TOTAL_INST_BYTES" in asm
  assert any(_code(line).startswith("s_wait_dscnt") and STINKY_KEEP in line for line in lines)
  drains = [i for i, line in enumerate(lines)
            if _code(line).startswith("s_wait_tensorcnt") and "drain tensor_load_to_lds" in line]
  assert drains
  for drain in drains:
    barrier = _nextLine(lines, drain, ("s_barrier",))
    assert barrier is not None and any("main-loop LDS" in line for line in lines[barrier:barrier + 3]), lines[drain]
  stores = [i for i, line in enumerate(lines) if _code(line).startswith("tensor_store_from_lds")]
  assert stores
  for store in stores:
    end = _nextLine(lines, store, ("s_barrier", "s_endpgm"))
    assert end is not None
    assert any(_code(line).startswith("s_wait_tensorcnt") for line in lines[store + 1:end]), lines[store]


@pytest.mark.parametrize("streamk", [0, 3])
def test_vector_epilogue_goes_after_the_staged_tile(gp_gfx1250, gfx1250_iim, assembler, capsys, streamk):
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=2, StreamK=streamk,
                       ProblemType={"UseBias": 1, "UseScaleAlphaVec": 1})
  assert sol.get("Valid"), capsys.readouterr().out
  assert asm
  staged = sol["MacroTile0"] * sol["MacroTile1"] * 2
  assert tdmStoreLdsBytes(sol) == staged
  assert sol["LdsOffsetBias"] == (staged + 15) // 16 * 16
  assert sol["LdsOffsetBiasNonGSU"] == sol["LdsOffsetBias"]
  assert sol["LdsOffsetBiasGSU"] == sol["LdsOffsetBias"]
  assert sol["LdsNumBytes"] >= sol["LdsOffsetBias"] + 256 * 4 * 2
  # The moved vectors can still overlap the main-loop LDS, so every preload barriers before its stores.
  assert "main-loop LDS is free for the epilogue vectors" in asm
  assert_every_vector_preload_barriers(asm)


def test_solution_rejects_with_the_reason(gp_gfx1250, gfx1250_iim, assembler, capsys):
  sol, asm = _emit_asm(gfx1250_iim, assembler, TDMStore=1, GlobalSplitU=2)
  assert not sol.get("Valid") and asm is None
  assert "TDMStore requires GlobalSplitU=1 or -1" in capsys.readouterr().out


@pytest.mark.parametrize("sourceSwap", [False, True])
@pytest.mark.parametrize("level", _LEVELS)
def test_local_split_u_stages_each_element_after_the_lsu_offset(gp_gfx1250, gfx1250_iim, assembler, capsys, level,
                                                                sourceSwap):
  """With LocalSplitU=2, shaped like the LSU=2 kernels of the gfx1250 bf16 TN logic, each thread stages
  the elements it owns after the LDS reduction where the LSU global write would store them: the LSU
  offset goes into coord0 (coord1 with SourceSwap) after the reduction, the TDM setup reads that VGPR
  with nothing in the block writing it first, and the staging stores address the tile from the setup."""
  sol, asm = _emit_asm(gfx1250_iim, assembler, [16, 16, 32, 1, 1, 4, 4, 1, 2], TDMStore=level,
                       SourceSwap=sourceSwap, WorkGroup=[16, 4, 2], DepthU=128)
  assert sol.get("Valid"), capsys.readouterr().out
  assert sol["LocalSplitU"] == 2 and sol["NumThreads"] == 128 and sol["StoreVectorWidth"] == 4
  operands = lambda line: [op.strip() for op in _code(line).partition(" ")[2].split(",")]
  lines = asm.splitlines()
  dim = 1 if sourceSwap else 0
  offsets = [i for i, line in enumerate(lines) if "coord%u += LSU offset%u" % (dim, dim) in line]
  assert len(offsets) == 1 and "valid offset coord%u is zero." % (1 - dim) in asm
  lsuCoord = operands(lines[offsets[0]])[0]
  assert max(i for i, line in enumerate(lines) if "post-lsu local write" in line) < offsets[0]
  slot = 1 if level == 1 else 4
  stores = _stores(asm)
  for beta in "01":
    assert stores[("GSU1", beta)]["tdm"] == [slot]
    then = re.search(r"^label_GW_B%s_FD0_VW%u_GSU1_Then:" % (beta, slot), asm, re.M)
    assert asm.count("\n", 0, then.start()) > offsets[0]
    block = _tdmBlock(asm, beta, slot).splitlines()
    setup = next(i for i, line in enumerate(block) if "tile column = coord1 - wg1*MT1" in line)
    plus0 = next(i for i in range(setup, len(block)) if block[i].rstrip().endswith("+ coord0"))
    coords = {1: operands(block[setup])[1], 0: operands(block[plus0])[2]}
    assert coords[dim] == lsuCoord, (block[setup], block[plus0])
    assert not [line for line in block[:setup] if operands(line)[0] == lsuCoord
                and not _code(line).startswith(("s_", "v_cmp", "buffer_store", "ds_store", "tensor_"))]
    staging = [line for line in block if "stage D element" in line]
    assert staging and all(operands(line)[0] == operands(block[setup])[0] for line in staging)
