"""Mainen & Sejnowski 1996 layer-4 spiny stellate (j7), single AP.

Reuses fem_neuron's already-set-up cells dir
(``../fem_neuron/comparisons/ms_j7/cells/``) which contains the M&S
HOC + compiled NMODL channels (na/kv/km/ca/cad/kca). NEURON loads the
HOC directly — no shim, no codegen.

Records per-segment ``i_membrane_`` across all sections, then drives:
  - LSA (closed-form) for V_e at radial probes
  - FEM (3D Poisson in ECS only, mesh built via fem_neuron's branched
    mesher with per_section_tags=True, ECS submesh extracted)

NEURON's HOC path resolution latches cwd at import time, so we
``os.chdir(CELLS_DIR)`` BEFORE ``from neuron import h``.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


# Self-contained cells dir, fetched from ModelDB 2488 on first run. The
# M&S 1996 archive (demofig1.hoc + cells/j7.hoc + na/kv/km/ca/cad/kca
# mechanisms) is downloaded + compiled by fem_lfp.modeldb, so this
# scenario no longer depends on a fem_neuron checkout for its cell data
# (fem_neuron is still used for the branched mesher — a code dependency).
CELLS_DIR = Path(__file__).resolve().parent / "cells"

# Probes radially out from the cell, in the cell's HOC frame (j7's
# soma sits near (0, 0, 0); cell extends ~±200 µm in xy and ±400 µm
# in z). Log-spaced 20 → 800 µm so we span near-membrane → far-field.
PROBES_UM = np.array(
    [(float(r), 0.0, 0.0) for r in np.geomspace(20.0, 800.0, 12)],
    dtype=np.float64,
)

# Stim — match the M&S figure 1B protocol amplitude (0.07 nA), but
# truncate at ~25 ms after first AP to keep wall time bounded.
STIM = dict(t_inj_ms=5.0, dur_ms=20.0, amp_nA=0.5)
T_STOP_MS = 30.0
DT_MS = 0.025
CELSIUS = 37.0  # M&S body temperature; HH kinetics ~20× slower at 6.3°C


# Mesh sizing (passed to fem_neuron's branched mesher).
MESH = dict(
    ecs_pad_um=1000.0,
    h_membrane_um=1.5,
    h_outer_um=200.0,
    grade_distance_um=300.0,
)

# Body-fitted mesh sizing (preferred path — single watertight surface
# from AMS, then TetGen volumetric mesh of cell + ECS box). pad=4000
# matches the cylinder convergence study; h_outer=80 ditto.
# tetgen_quality=1.4 is the default radius-edge ratio; lower = better
# tet quality but more cells.
MESH_BODY_FITTED = dict(
    ecs_pad_um=4000.0,
    # h_outer is a global TetGen volume bound. Body-fitted has no
    # graded sizing — tets near the cell are sized by AMS's alpha-wrap
    # face mesh (~1-2 µm) while bulk tets are bounded by ``-a (h³/6)``.
    # On j7 with pad=4000 (box ≈ 8400 µm/side, vol ≈ 5.9e11 µm³),
    # h_outer=80 produces 14.5M bulk tets. Pushing to 400 µm gives
    # ~55k bulk tets + ~150k near-cell — manageable.
    h_outer_um=400.0,
    tetgen_quality=1.8,
)


def run_neuron():
    """Load j7 + biophysics, run, return NeuronRun.

    ``demofig1.hoc`` builds the j7 cell with the M&S 1996 channel
    densities (Na/Kv/Km/Ca/KCa). It also sets ``celsius=37`` and
    declares hoc-side stim/recording — we override stim after load.

    Downloads ModelDB 2488 + compiles its mechanisms on first run.
    """
    from fem_lfp.modeldb import ensure_cell
    ensure_cell(2488, CELLS_DIR, inner="cells", mod_subdir="")

    cwd_before = os.getcwd()
    os.chdir(CELLS_DIR)
    try:
        from neuron import h
        h.load_file("stdrun.hoc")
        h.load_file("demofig1.hoc")
        # demofig1 defines load_3dcell + a procs that build a chosen cell.
        # j7 is the layer-4 spiny stellate.
        h.load_3dcell("cells/j7.hoc")

        # demofig1 already configures Na/Kv/Km/Ca/KCa on soma/axon/dendrites
        # at M&S densities. The axon (iseg/hill/myelin/node) is created
        # via topology only — no explicit pt3dadd. Call define_shape to
        # auto-generate pt3d coords from L/diam/topology so the FEM
        # mesher and LSA both have geometric coordinates to work with.
        h.define_shape()
        sections = list(h.allsec())
        soma = None
        for sec in sections:
            if sec.name() == "soma" or sec.name().endswith(".soma"):
                soma = sec
                break
        if soma is None:
            raise RuntimeError(f"no `soma` section after load_3dcell; "
                               f"have {[s.name() for s in sections][:8]}...")

        stim = h.IClamp(soma(0.5))
        stim.delay = STIM["t_inj_ms"]
        stim.dur = STIM["dur_ms"]
        stim.amp = STIM["amp_nA"]

        from fem_lfp.neuron_sim import setup_imem_recording, finalize_run
        handles = setup_imem_recording(sections)
        t_vec = h.Vector().record(h._ref_t)
        rec_v = {
            "soma(0.5)": h.Vector().record(soma(0.5)._ref_v),
        }

        h.dt = DT_MS
        h.celsius = CELSIUS
        h.finitialize(-70.0)
        h.continuerun(T_STOP_MS)

        # Keep refs alive — NEURON GC's locals across return.
        _keepalive = (sections, stim, t_vec, rec_v, handles)

        # Record the per-section info we need to build a SectionPath
        # list for the mesher. We capture this BEFORE leaving the cwd.
        section_info = []
        for sec in sections:
            n_pt3d = int(h.n3d(sec=sec))
            if n_pt3d < 2:
                continue
            pts = np.array(
                [[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)]
                 for i in range(n_pt3d)],
                dtype=np.float64,
            )
            diams = np.array(
                [h.diam3d(i, sec=sec) for i in range(n_pt3d)],
                dtype=np.float64,
            )
            section_info.append({
                "name": sec.name(),
                "points_um": pts,
                "diameters_um": diams,
                "nseg": int(sec.nseg),
            })

        nrun = finalize_run(handles, t_vec, rec_v)
    finally:
        os.chdir(cwd_before)

    return nrun, section_info
