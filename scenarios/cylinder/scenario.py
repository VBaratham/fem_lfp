"""HH cylinder, IClamp at z=0.

A 200 µm × 5 µm cable with HH channels, current step at the middle. The
spike sweeps along the cable and we record V_m at the soma + V_e at a
log-spaced set of radial probes from just outside the membrane out to a
millimeter, so the FEM-vs-LSA comparison spans both the near-field
(probe geometry / finite tissue boundaries matter) and the far-field
(LSA's homogeneous-infinite-medium assumption is fine).

This scenario just *builds and drives the cell*; the extracellular
forward model (mesh + FEM + LSA) is handled by
:class:`fem_lfp.ExtracellularModel` in the driver script.
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

# ECS mesh sizing (passed straight to ExtracellularModel as overrides).
# Graded: fine at the membrane, coarser toward the Dirichlet wall. The
# sweep script overrides ecs_pad_um / h_far_um to study wall bias.
MESH = dict(
    ecs_pad_um=1500.0,
    h_membrane_um=0.8,
    h_outer_um=80.0,
    grade_distance_um=400.0,
)

# Stim
STIM = dict(t_inj_ms=2.0, dur_ms=0.5, amp_nA=0.6)

# Run
T_STOP_MS = 8.0
DT_MS = 0.025

# Title for plots
TITLE = "HH cylinder, 0.6 nA × 0.5 ms IClamp at center"


# Module-level keepalive: NEURON garbage-collects sections/objects whose
# only Python references are function locals once build_cell() returns.
_KEEPALIVE: list = []


def build_cell():
    """Build the HH cable + stimulus. Returns the section list.

    Does NOT record or run — the caller arms recording (via
    ExtracellularModel) *before* finitialize, then runs, then solves.
    """
    import os
    # NEURON latches cwd at import; do it here in a clean cwd.
    os.environ.setdefault("NEURON_HOME", "")
    from neuron import h
    h.load_file("stdrun.hoc")

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

    _KEEPALIVE.extend([sec, stim])
    return [sec]


def run():
    """Initialize and integrate the cable equation."""
    from neuron import h
    h.dt = DT_MS
    h.celsius = 6.3   # canonical HH temperature
    h.finitialize(-65.0)
    h.continuerun(T_STOP_MS)
