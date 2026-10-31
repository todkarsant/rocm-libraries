# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# f16 accumulation kernels emitted in process, one size group at a time (the
# harness emits the first group of an entry, so every group is written out alone):
# - f16acc_streamk_gfx1250.yaml: every solution of every group builds; every
#   StreamK kernel has an f16 fixup that adds the partials in f32 and rounds the
#   sum to f16 once; a StaticGrid group also has the tree fixup; a linear fixup
#   adds the workgroup's own partial before the earlier pieces' partials under
#   StaticGrid, after them under DynamicWorkQueue, and a Hybrid kernel has both.
# - f16acc_brotate_mnf4_gfx1250.yaml: the tail loop steps the interleaved MN-major
#   fp4 B by the main loop's K step; its 16x16x64 group takes two tail steps.
################################################################################

import os
import re

import pytest
import yaml

import config_harness

pytestmark = pytest.mark.unit

_GFX12 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "common", "gemm", "gfx12")


def _groups(config):
    with open(os.path.join(_GFX12, config)) as f:
        entries = yaml.safe_load(f)["BenchmarkProblems"]
    return [(config, p, g) for p, entry in enumerate(entries) for g in range(1, len(entry))]


def _fork(group, name):
    return [v for item in group.get("ForkParameters", []) for v in item.get(name, [])]


def _emit(config, p, g, tmp_path, arch):
    with open(os.path.join(_GFX12, config)) as f:
        doc = yaml.safe_load(f)
    entry = doc["BenchmarkProblems"][p]
    path = tmp_path / ("%s_%d_%d.yaml" % (config[:-5], p, g))
    path.write_text(yaml.safe_dump({**{k: v for k, v in doc.items() if k != "BenchmarkProblems"},
                                    "BenchmarkProblems": [[entry[0], entry[g]]]}))
    results = config_harness.emit_kernels_from_config(str(path), limit=None, arch=arch, canonical=False)
    assert results and not [base for base, src, err in results if err or not src], (config, p, g)
    return entry[g], [src for _, src, _ in results]


def _lines(src):
    return [l for l in (l.split("//")[0].strip() for l in src.splitlines()) if l]


def _own_partial_order(src):
    # A linear fixup works block by block; each block adds the workgroup's own f16 partial (v_fma_mix_f32 from
    # v[vgprValuC..]) before or after its loop over the earlier pieces' partials.
    lines = _lines(src)
    starts = [i for i, l in enumerate(lines) if re.fullmatch(r"label_SK_F16_AddPartial(?!End)\w*:", l)]
    ends = [i for i, l in enumerate(lines) if re.fullmatch(r"label_SK_F16_AddPartialEnd\w*:", l)]
    own = [i for i, l in enumerate(lines) if re.match(r"v_fma_mix_f32 \S+, v\[vgprValuC", l)]
    assert own and starts and len(starts) == len(ends)
    assert not [i for i in own for s, e in zip(starts, ends) if s < i < e], "own partial inside a loop"
    first, last = any(i < starts[0] for i in own), any(i > ends[-1] for i in own)
    return "first" if first and not last else "last" if last and not first else "mixed"


_STREAMK = _groups("f16acc_streamk_gfx1250.yaml")


@pytest.mark.parametrize("config, p, g", _STREAMK, ids=["entry%d_group%d" % (p, g) for _, p, g in _STREAMK])
def test_f16acc_streamk_groups_emit_their_f16_fixups(tmp_path, config, p, g):
    group, sources = _emit(config, p, g, tmp_path, "gfx1250")
    fixups = [s for s in sources if "label_SK_F16_AddPartial:" in s]
    if set(_fork(group, "TileProcessingStrategy")) == {"StreamK"}:
        assert len(fixups) == len(sources)
    else:
        assert 0 < len(fixups) < len(sources)  # persistent DataParallel kernels have no fixup
    assert all("round the sum to f16" in s for s in fixups)
    assignments = set(_fork(group, "WorkAssignment"))
    tree = [s for s in fixups if "label_SK_F16_StoreSums:" in s]
    assert len(tree) < len(fixups) and (tree or "StaticGrid" not in assignments)
    # A Hybrid kernel holds a static and a dynamic fixup, so both orders.
    want = {order for assignment, order in (("StaticGrid", "first"), ("DynamicWorkQueue", "last"), ("Hybrid", "mixed"))
            if assignment in assignments}
    assert {_own_partial_order(s) for s in fixups if s not in tree} == want


_LR_B = re.compile(r"ds_load_tr4_b64 v\[vgprValuB_\S+\], v\[vgprLocalReadAddrB\+0\](?: offset:(\d+))?")
_INC_B = re.compile(r"v_add_co_u32 v\[vgprLocalReadAddrB\+0\], \S+, (s\d+), v\[vgprLocalReadAddrB\+0\]")


def _b_k_steps(src):
    # Main-loop sub-iteration k reads B at the tail loop's offsets plus k K steps.
    lines = _lines(src)
    main = lines[lines.index("label_LoopBeginL:"):lines.index("label_LoopEndL:")]
    tail = lines[lines.index("label_TailLoopBeginL:"):lines.index("label_TailLoopEndL:")]
    first, every = ({int(m.group(1) or 0) for m in map(_LR_B.fullmatch, rows) if m} for rows in (tail, main))
    step = min(every - first) - min(first)
    assert every == {o + k * step for o in first for k in range(len(every) // len(first))}, (first, every)
    movs, incs = {}, []
    for l in tail:
        m = re.fullmatch(r"s_mov_b32 (s\d+), (\d+)", l)
        if m:
            movs[m.group(1)] = int(m.group(2))
        m = _INC_B.fullmatch(l)
        if m:
            incs.append(movs[m.group(1)])
    return step, incs


_MNF4 = _groups("f16acc_brotate_mnf4_gfx1250.yaml")


@pytest.mark.parametrize("config, p, g", _MNF4, ids=["entry%d_group%d" % (p, g) for _, p, g in _MNF4])
def test_f16acc_brotate_mnf4_tail_loop_steps_b_by_the_main_loop_k_step(tmp_path, config, p, g):
    for src in _emit(config, p, g, tmp_path, "gfx1250")[1]:
        step, incs = _b_k_steps(src)
        assert incs and set(incs) == {step}, (config, p, g, step, incs)
