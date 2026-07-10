"""Small polyline-geometry helpers shared by the NEURON and FEM sides.

Both the segment-endpoint capture (``neuron_sim``) and the facet-to-segment
binning (``fem``) walk a section's pt3d polyline by arc length. These two
functions are the single implementation of that; they are unit-agnostic
(pass all coordinates in the same unit — µm or m).
"""
from __future__ import annotations

import numpy as np


def point_at_arc_fraction(points: np.ndarray, frac: float) -> np.ndarray:
    """3D point at normalized arc-length fraction ``frac`` (0..1) along the
    polyline ``points`` (shape ``(n, 3)``).

    ``frac`` is clamped to [0, 1]; a degenerate (zero-length) polyline
    returns its first point.
    """
    seg_lens = np.linalg.norm(np.diff(points, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    L = arc[-1]
    if L <= 0:
        return points[0].copy()
    target = float(np.clip(frac, 0.0, 1.0)) * L
    j = int(np.searchsorted(arc, target))
    if j <= 0:
        return points[0].copy()
    if j >= points.shape[0]:
        return points[-1].copy()
    a = float(np.clip((target - arc[j - 1]) / max(seg_lens[j - 1], 1e-30),
                      0.0, 1.0))
    return (1.0 - a) * points[j - 1] + a * points[j]


def arc_fraction_of_projection(point: np.ndarray, polyline: np.ndarray) -> float:
    """Normalized arc length (0=proximal, 1=distal) of ``point``'s
    closest-point projection onto ``polyline`` (shape ``(n, 3)``).

    Returns the closest endpoint's fraction if the point projects outside
    every segment; 0.0 for a degenerate polyline.
    """
    if polyline.shape[0] < 2:
        return 0.0
    seg_lens = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total <= 0:
        return 0.0
    best_arc = 0.0
    best_d2 = float("inf")
    for j in range(polyline.shape[0] - 1):
        a, b = polyline[j], polyline[j + 1]
        ab = b - a
        ap = point - a
        L2 = float(ab @ ab)
        if L2 < 1e-30:
            t = 0.0
            closest = a
        else:
            t = float(np.clip((ap @ ab) / L2, 0.0, 1.0))
            closest = a + t * ab
        d2 = float(np.sum((point - closest) ** 2))
        if d2 < best_d2:
            best_d2 = d2
            best_arc = (cum[j] + t * seg_lens[j]) / total
    return best_arc
