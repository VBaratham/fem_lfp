"""Line-source approximation (Holt & Koch 1999) for V_e.

Each NEURON segment is treated as a uniform line current in an infinite
homogeneous medium. The closed-form integral of the 1/r monopole kernel
along the line gives V_e at a probe point; we sum across segments.

This is the LFP-literature reference solution that the FEM-direct V_e is
meant to improve on — close to membrane and near probes/boundaries the
infinite-homogeneous-medium assumption breaks.

Pure-numpy; no NEURON or FEM imports here. Lifted from fem_neuron's
``_validation/line_source.py`` with the NEURON-specific recording helpers
trimmed out (those live in `neuron_sim.py`).
"""
from __future__ import annotations

import numpy as np


def line_source_v_e(
    probe_xyz_um: np.ndarray,
    p1_um: np.ndarray,
    p2_um: np.ndarray,
    imem_nA: np.ndarray,
    sigma_S_per_m: float = 0.3,
    r_min_um: float = 1.0,
) -> np.ndarray:
    """Closed-form line-source V_e at probe positions.

    Parameters
    ----------
    probe_xyz_um : (P, 3)
        Probe positions in µm.
    p1_um, p2_um : (S, 3)
        Per-segment endpoint coordinates in µm.
    imem_nA : (S, T)
        Per-segment transmembrane current in nA, outward-positive
        (NEURON's `i_membrane_` convention).
    sigma_S_per_m : float, default 0.3
        Extracellular conductivity (LFP-standard value).
    r_min_um : float, default 1.0
        Floor on perpendicular distance to a segment (µm), to avoid
        log singularity when a probe sits on the line.

    Returns
    -------
    V_e : (P, T) array, in volts.
    """
    p1 = np.asarray(p1_um, dtype=np.float64) * 1e-6
    p2 = np.asarray(p2_um, dtype=np.float64) * 1e-6
    probes = np.asarray(probe_xyz_um, dtype=np.float64) * 1e-6
    I = np.asarray(imem_nA, dtype=np.float64) * 1e-9

    L_seg = np.linalg.norm(p2 - p1, axis=1)
    L_safe = np.where(L_seg > 0, L_seg, 1.0)
    d_hat = (p2 - p1) / L_safe[:, None]

    diff = probes[:, None, :] - p1[None, :, :]
    h = np.einsum("psi,si->ps", diff, d_hat)
    r2 = np.sum(diff * diff, axis=-1) - h * h
    r2 = np.maximum(r2, (r_min_um * 1e-6) ** 2)

    Lh = L_seg[None, :] - h
    num = Lh + np.sqrt(Lh * Lh + r2)
    den = -h + np.sqrt(h * h + r2)
    den = np.maximum(den, 1e-30)
    F = np.log(num / den)

    coeff = F / (4.0 * np.pi * sigma_S_per_m * L_safe[None, :])
    return coeff @ I
