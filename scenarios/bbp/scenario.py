"""Hay et al. 2011 L5 pyramidal cell (ModelDB 139653, BBP-style biophysics).

Cell template + biophysics + mods all live in
``./cells/L5bPCmodelsEH/`` (downloaded from ModelDB by hand). Run
``nrnivmodl`` once in ``cells/L5bPCmodelsEH/mod/`` to produce
``mod/arm64/special`` + ``libnrnmech.dylib``.

The cell is a real L5 pyramidal reconstruction (cell1.asc — Neurolucida)
with the Hay 2011 channel set: NaTa_t, NaTs2_t, Nap_Et2, K_Pst, K_Tst,
SKv3_1, SK_E2, Ca_HVA, Ca_LVAst, CaDynamics_E2, Ih, Im. Standard BAC
firing or step-current protocols are implemented in
``simulationcode/`` but we drive the cell with a simple soma IClamp
to keep the test focused on the LFP forward problem.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np


HAY_ROOT = Path(__file__).parent / "cells" / "L5bPCmodelsEH"

# Probes radially out from the cell's HOC frame origin. Hay's cell1
# is centered roughly at (0, 0, 0) with apical dendrite extending up
# in +y. ~800 µm outermost matches the cylinder & M&S studies.
PROBES_UM = np.array(
    [(float(r), 0.0, 0.0) for r in np.geomspace(20.0, 800.0, 12)],
    dtype=np.float64,
)

# Match the M&S protocol shape (5 ms baseline, 20 ms stim, capture ~2 APs).
STIM = dict(t_inj_ms=5.0, dur_ms=20.0, amp_nA=1.0)
T_STOP_MS = 30.0
DT_MS = 0.025
CELSIUS = 34.0   # Hay 2011 uses 34°C

# fem_neuron's branched-mesh defaults.
MESH = dict(
    ecs_pad_um=1000.0,
    h_membrane_um=1.5,
    h_outer_um=200.0,
    grade_distance_um=300.0,
)

# Body-fitted (preferred) — bigger box for cleaner far-field.
# h_outer=400 keeps the bulk tet count bounded (M&S empirically: ~1M
# bulk cells at this size; smaller h_outer balloons to 10M+).
MESH_BODY_FITTED = dict(
    ecs_pad_um=4000.0,
    h_outer_um=400.0,
    tetgen_quality=1.8,
)


# Module-level keepalive so the L5PCtemplate instance survives across
# the run_neuron() return — when the local `cell` ref drops, NEURON
# garbage-collects the template's sections, and ``h.allsec()`` returns
# nothing in the SWC export step.
_KEEPALIVE: list = []


def run_neuron():
    """Load Hay et al. L5 PC, run, return NeuronRun + section_info.

    Downloads ModelDB 139653 and compiles its NMODL mechanisms on first
    run (idempotent) so the scenario is clone-and-run.
    """
    from fem_lfp.modeldb import ensure_cell
    ensure_cell(139653, HAY_ROOT, inner="L5bPCmodelsEH", mod_subdir="mod")

    # Resolve the compiled mechanism library (arch dir name is platform
    # specific: arm64 / x86_64 / .libs).
    libs = (list((HAY_ROOT / "mod").glob("*/libnrnmech.dylib"))
            + list((HAY_ROOT / "mod").glob("*/libnrnmech.so"))
            + list((HAY_ROOT / "mod").glob("*/.libs/libnrnmech.so")))
    if not libs:
        raise RuntimeError(f"no compiled libnrnmech under {HAY_ROOT / 'mod'}")
    mech_lib = libs[0]

    cwd_before = os.getcwd()
    os.chdir(HAY_ROOT)
    try:
        from neuron import h
        h.load_file("stdlib.hoc")
        h.load_file("stdrun.hoc")
        h.load_file("import3d.hoc")
        # Load the .mod libraries we just compiled.
        h.nrn_load_dll(str(mech_lib))
        # Load biophysics + template.
        h.load_file("models/L5PCbiophys3.hoc")
        h.load_file("models/L5PCtemplate.hoc")
        # Instantiate. The template's __init__ runs Import3d_Neurolucida3
        # on the .asc file, sets up biophys, and trims the axon.
        cell = h.L5PCtemplate("morphologies/cell1.asc")

        h.define_shape()
        sections = list(h.allsec())
        print(f"[hay-L5PC] {len(sections)} sections after template init")

        # Find the soma section (cell.soma[0])
        soma = None
        for sec in sections:
            if "soma" in sec.name():
                soma = sec
                break
        if soma is None:
            raise RuntimeError("no soma section after template init")

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
        h.finitialize(-80.0)
        h.continuerun(T_STOP_MS)

        # Keep the template instance alive past run_neuron() — Python
        # would otherwise GC ``cell``, NEURON would drop the
        # template's sections, and the body-fitted SWC export would
        # see an empty ``h.allsec()``.
        _KEEPALIVE.append(cell)
        _keepalive = (cell, sections, stim, t_vec, rec_v, handles)

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
