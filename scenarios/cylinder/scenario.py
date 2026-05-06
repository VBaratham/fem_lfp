"""HH cylinder, IClamp at z=0.

A 200 µm × 5 µm cable with HH channels, current step at the middle. The
spike sweeps along the cable and we record V_m at the soma + V_e at a
log-spaced set of radial probes from just outside the membrane out to a
millimeter, so the FEM-vs-LSA comparison spans both the near-field
(probe geometry / finite tissue boundaries matter) and the far-field
(LSA's homogeneous-infinite-medium assumption is fine).
"""
from __future__ import annotations

import numpy as np


# ----------------------------- geometry ---------------------------- #

L_UM = 200.0
DIAM_UM = 5.0
NSEG = 41

# Probes radially out from the cable axis at z=0, log-spaced from
# just-outside-the-membrane (3.5 µm = radius + 1 µm) to 800 µm.
PROBES_UM = np.array(
    [(float(r), 0.0, 0.0) for r in np.geomspace(3.5, 800.0, 14)],
    dtype=np.float64,
)

# Mesh sizing — graded ECS, matching the values that gave clean
# FEM-vs-LSA at near probes in the original cylinder run (25k DOFs,
# ~100s wall time). The Stage-1 ramp covers the probe shell;
# Stage 2 (set per-pad in the sweep script) only kicks in when
# ``ecs_pad_um`` is larger than the probe shell, to keep DOF count
# tractable in the bulk.
MESH = dict(
    L_um=L_UM,
    radius_um=DIAM_UM / 2.0,
    ecs_pad_um=1500.0,           # default: small box for fast iteration.
                                  # Sweep script overrides up to ≥5× outer
                                  # probe to study Dirichlet wall bias.
    h_membrane_um=0.8,
    h_outer_um=80.0,
    grade_distance_um=400.0,     # original M3.5-validated probe-shell
                                  # value; keeps mesh ~25k DOFs / 150k
                                  # cells so per-step solve stays
                                  # ~300 ms. Stage-2 grading (sweep
                                  # script) extends the box without
                                  # exploding DOF count.
)

# Stim
STIM = dict(t_inj_ms=2.0, dur_ms=0.5, amp_nA=0.6)

# Run
T_STOP_MS = 8.0
DT_MS = 0.025

# Title for plots
TITLE = "HH cylinder, 0.6 nA × 0.5 ms IClamp at center"


def run_neuron():
    """Run the NEURON sim, return a NeuronRun."""
    import os
    # NEURON latches cwd at import; do it here in a clean cwd.
    os.environ.setdefault("NEURON_HOME", "")
    from neuron import h
    h.load_file("stdrun.hoc")

    from fem_lfp.neuron_sim import (
        setup_imem_recording, finalize_run,
    )

    sec = h.Section(name="cyl")
    sec.L = L_UM
    sec.diam = DIAM_UM
    sec.nseg = NSEG
    sec.Ra = 100.0
    sec.cm = 1.0
    sec.insert("hh")
    sec.insert("pas")
    for seg in sec:
        seg.pas.g = 1.0 / 30000.0
        seg.pas.e = -70.0

    half_L = float(sec.L) / 2.0
    diam = float(sec.diam)
    h.pt3dclear(sec=sec)
    h.pt3dadd(0.0, 0.0, -half_L, diam, sec=sec)
    h.pt3dadd(0.0, 0.0, +half_L, diam, sec=sec)

    stim = h.IClamp(sec(0.5))
    stim.delay = STIM["t_inj_ms"]
    stim.dur = STIM["dur_ms"]
    stim.amp = STIM["amp_nA"]

    handles = setup_imem_recording([sec])
    t_vec = h.Vector().record(h._ref_t)
    rec_v = {
        "soma(0.5)": h.Vector().record(sec(0.5)._ref_v),
        "soma(0.0)": h.Vector().record(sec(0.0)._ref_v),
    }

    h.dt = DT_MS
    h.celsius = 6.3   # canonical HH temperature
    h.finitialize(-65.0)
    h.continuerun(T_STOP_MS)

    # Keep `stim` alive — NEURON GC's locals across return.
    _keepalive = (sec, stim, t_vec, rec_v, handles)
    return finalize_run(handles, t_vec, rec_v)
