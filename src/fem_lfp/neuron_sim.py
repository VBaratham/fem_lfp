"""NEURON helpers for the hybrid pipeline.

We need three pieces of data from a NEURON run:

    1. per-segment transmembrane current i_mem(t)   (drives the FEM RHS)
    2. per-segment endpoint coordinates              (for LSA + for matching
                                                      FEM membrane facets to
                                                      segments)
    3. V_m at chosen recording sites                 (for the overlay plot)

NEURON's `cvode.use_fast_imem(1)` exposes `seg._ref_i_membrane_`, which is
**per-segment total current in nA** (NOT mA/cm²). No area multiplication
needed — NEURON has already done that integration. (See
fem_neuron/_validation/line_source.py for the bug history on this.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NeuronRun:
    """Bundle of everything the FEM and LSA postprocessors need."""
    t_ms: np.ndarray              # (T,)
    p1_um: np.ndarray             # (S, 3)
    p2_um: np.ndarray             # (S, 3)
    imem_nA: np.ndarray           # (S, T)  outward-positive
    rec_v_mV: dict[str, np.ndarray]   # site label → (T,)


def _segment_endpoints_um(sec) -> tuple[np.ndarray, np.ndarray]:
    """Return per-segment (p1, p2) endpoint coords in µm for a Section.

    Walks the section's pt3d polyline by arc length. Requires ≥2 pt3d
    points — call h.pt3dadd before recording.
    """
    from neuron import h

    n_pt3d = int(h.n3d(sec=sec))
    if n_pt3d < 2:
        raise ValueError(
            f"Section {sec.name()} needs ≥2 pt3d points; "
            f"call h.pt3dadd(...) on it first."
        )
    pts = np.array(
        [[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)]
         for i in range(n_pt3d)],
        dtype=np.float64,
    )
    deltas = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(deltas)])
    L = arc[-1] if arc[-1] > 0 else 1.0
    arc_norm = arc / L

    nseg = int(sec.nseg)
    p1 = np.zeros((nseg, 3))
    p2 = np.zeros((nseg, 3))
    for k in range(nseg):
        p1[k] = _interp_pt(arc_norm, pts, k / nseg)
        p2[k] = _interp_pt(arc_norm, pts, (k + 1) / nseg)
    return p1, p2


def _interp_pt(arc_norm: np.ndarray, pts: np.ndarray, x: float) -> np.ndarray:
    if x <= arc_norm[0]:
        return pts[0]
    if x >= arc_norm[-1]:
        return pts[-1]
    j = int(np.searchsorted(arc_norm, x))
    a = (x - arc_norm[j - 1]) / (arc_norm[j] - arc_norm[j - 1])
    return (1 - a) * pts[j - 1] + a * pts[j]


@dataclass
class _ImemHandles:
    sections: list
    p1_um: np.ndarray
    p2_um: np.ndarray
    vec_handles: list   # h.Vector per segment, recording i_membrane_ in nA


def setup_imem_recording(sections) -> _ImemHandles:
    """Set up per-segment i_membrane recording. Call BEFORE finitialize.

    Uses fast_imem so the recorded value is per-segment current in nA
    directly.
    """
    from neuron import h

    cvode = h.CVode()
    cvode.use_fast_imem(1)

    p1_all, p2_all, vec_handles = [], [], []
    for sec in sections:
        p1, p2 = _segment_endpoints_um(sec)
        p1_all.append(p1)
        p2_all.append(p2)
        for seg in sec:
            v = h.Vector()
            v.record(seg._ref_i_membrane_)
            vec_handles.append(v)

    return _ImemHandles(
        sections=list(sections),
        p1_um=np.vstack(p1_all),
        p2_um=np.vstack(p2_all),
        vec_handles=vec_handles,
    )


def export_swc_from_neuron(
    sections,
    out_path,
    *,
    type_overrides: dict[str, int] | None = None,
    min_point_spacing_um: float = 0.1,
) -> None:
    """Export a NEURON cell to SWC format for Alpha_Mesh_Swc.

    SWC rows: ``id type x y z radius parent_id``.
    Type codes: 1=soma, 2=axon, 3=basal dend, 4=apical dend (default
    here is 3 for any non-soma section unless ``type_overrides``
    matches by section-name substring).

    Walks each section's pt3d polyline, emitting one SWC row per
    pt3d point. Section connections (NEURON's ``Section.parent`` /
    ``parentseg``) drive the parent_id column — the first pt3d of a
    child section gets the LAST pt3d id of its parent as its parent.

    Note: SWC has one radius per node, so per-pt3d diameter variation
    inside NEURON's pt3dadd is preserved as-is. AMS's surface fits a
    spline through these radii.
    """
    from neuron import h
    from pathlib import Path

    overrides = type_overrides or {}
    out_path = Path(out_path)

    rows: list[tuple[int, int, float, float, float, float, int]] = []
    sec_to_first_id: dict[object, int] = {}
    sec_to_last_id: dict[object, int] = {}
    next_id = 1

    for sec in sections:
        n_pt3d = int(h.n3d(sec=sec))
        if n_pt3d < 2:
            continue
        # Determine SWC type: soma=1, axon=2, dend=3, apical=4 by name.
        name = sec.name().lower()
        sw_type = 3
        if "soma" in name:
            sw_type = 1
        elif "axon" in name or "iseg" in name or "hill" in name or "myelin" in name or "node" in name:
            sw_type = 2
        elif "apical" in name or name.startswith(("a", "apic")):
            sw_type = 4
        for k, v in overrides.items():
            if k in name:
                sw_type = v

        # Parent id — last pt3d id of parent section.
        parent_id_for_first = -1
        try:
            parent_sec = sec.parentseg().sec if sec.parentseg() is not None else None
        except Exception:
            parent_sec = None
        if parent_sec is not None and parent_sec in sec_to_last_id:
            parent_id_for_first = sec_to_last_id[parent_sec]

        # If child section, the parent's last pt3d coords are
        # typically equal to this section's first pt3d (NEURON
        # connect convention). Skip our first pt3d in that case to
        # avoid a zero-arc parent-child edge that AMS's spline
        # fitter rejects.
        parent_last_xyz: tuple[float, float, float] | None = None
        if parent_id_for_first > 0:
            for r in rows:
                if r[0] == parent_id_for_first:
                    parent_last_xyz = (r[2], r[3], r[4])
                    break

        first_emitted = False
        last_xyz: tuple[float, float, float] | None = None
        for i in range(n_pt3d):
            x = float(h.x3d(i, sec=sec))
            y = float(h.y3d(i, sec=sec))
            z = float(h.z3d(i, sec=sec))
            d = float(h.diam3d(i, sec=sec))
            r = max(d / 2.0, 0.05)   # AMS chokes on radius=0
            # Skip child section's first pt3d if it coincides with
            # parent's last (zero-arc edge → AMS rejects).
            if (not first_emitted and parent_last_xyz is not None and
                    abs(x - parent_last_xyz[0]) < 1e-6 and
                    abs(y - parent_last_xyz[1]) < 1e-6 and
                    abs(z - parent_last_xyz[2]) < 1e-6):
                continue
            # Dedupe within section: skip points closer than the
            # minimum spacing to the previous emitted point.
            if last_xyz is not None and i != n_pt3d - 1:
                dx = x - last_xyz[0]; dy = y - last_xyz[1]; dz = z - last_xyz[2]
                if dx * dx + dy * dy + dz * dz < min_point_spacing_um ** 2:
                    continue
            if not first_emitted:
                p = parent_id_for_first
                first_emitted = True
                sec_to_first_id[sec] = next_id
            else:
                p = next_id - 1
            rows.append((next_id, sw_type, x, y, z, r, p))
            sec_to_last_id[sec] = next_id
            last_xyz = (x, y, z)
            next_id += 1

    if not rows:
        raise RuntimeError("no pt3d in any section; can't export SWC")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# id type x y z radius parent\n")
        for r in rows:
            f.write(f"{r[0]} {r[1]} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f} "
                    f"{r[5]:.4f} {r[6]}\n")
    print(f"[swc export] {len(rows)} pt3d points → {out_path}")


def per_segment_polylines_um(
    points_um: np.ndarray,
    diameters_um: np.ndarray,
    nseg: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Slice a section's pt3d polyline into ``nseg`` equal-arc-length pieces.

    Returns a list of length ``nseg``, each entry ``(pts2, diams2)`` —
    the simplified 2-point polyline (first and last pt3d) of that
    NEURON segment plus its corresponding diameters at those points,
    linearly interpolated along arc length.

    Used by the M&S / BBP scenarios to give the FEM mesher one
    primitive per NEURON segment instead of one per section, so each
    NEURON segment owns its own unique facet tag — no
    polyline-simplification dropping segments at the per-section bin
    level.
    """
    if nseg <= 0:
        raise ValueError(f"nseg must be positive, got {nseg}")
    pts = np.asarray(points_um, dtype=np.float64)
    diams = np.asarray(diameters_um, dtype=np.float64)
    if pts.shape[0] < 2:
        raise ValueError("need at least 2 pt3d points")
    if pts.shape[0] != diams.shape[0]:
        raise ValueError(
            f"pts {pts.shape} and diams {diams.shape} length mismatch"
        )
    # cumulative arc length along the full polyline
    seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lens)])
    L = arc[-1]
    if L <= 0:
        # degenerate (zero-length) section: return nseg copies of the
        # single point with epsilon offset so the mesher doesn't choke
        # on zero-length primitives later.
        pts_dup = np.array([pts[0], pts[0] + 1e-3], dtype=np.float64)
        diams_dup = np.array([diams[0], diams[0]], dtype=np.float64)
        return [(pts_dup.copy(), diams_dup.copy()) for _ in range(nseg)]

    def interp(target_arc: float) -> tuple[np.ndarray, float]:
        # find the segment index in the polyline where target_arc falls
        idx = int(np.searchsorted(arc, target_arc))
        if idx <= 0:
            return pts[0], float(diams[0])
        if idx >= pts.shape[0]:
            return pts[-1], float(diams[-1])
        a = (target_arc - arc[idx - 1]) / max(seg_lens[idx - 1], 1e-30)
        a = float(np.clip(a, 0.0, 1.0))
        p = (1.0 - a) * pts[idx - 1] + a * pts[idx]
        d = (1.0 - a) * diams[idx - 1] + a * diams[idx]
        return p, float(d)

    out = []
    for k in range(nseg):
        a0 = (k / nseg) * L
        a1 = ((k + 1) / nseg) * L
        p0, d0 = interp(a0)
        p1, d1 = interp(a1)
        if np.allclose(p0, p1):
            # zero-length sub-segment (all NEURON pt3d at one location)
            p1 = p0 + np.array([1e-3, 0.0, 0.0])
        out.append(
            (np.vstack([p0, p1]), np.array([d0, d1], dtype=np.float64))
        )
    return out


def finalize_run(
    handles: _ImemHandles,
    t_vec,
    rec_v: dict[str, "h.Vector"],   # noqa: F821
) -> NeuronRun:
    """Drain Vectors into a NeuronRun."""
    t_ms = np.asarray(t_vec.to_python(), dtype=np.float64)
    n_t = len(t_ms)
    n_seg = len(handles.vec_handles)
    imem_nA = np.zeros((n_seg, n_t), dtype=np.float64)
    for k, v in enumerate(handles.vec_handles):
        py = np.asarray(v.to_python())
        imem_nA[k, :len(py)] = py
    rec_v_mV = {
        label: np.asarray(vec.to_python(), dtype=np.float64)
        for label, vec in rec_v.items()
    }
    return NeuronRun(
        t_ms=t_ms,
        p1_um=handles.p1_um,
        p2_um=handles.p2_um,
        imem_nA=imem_nA,
        rec_v_mV=rec_v_mV,
    )
