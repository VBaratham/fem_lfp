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

import logging
from dataclasses import dataclass

import numpy as np

from ._geom import point_at_arc_fraction

logger = logging.getLogger(__name__)


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
    nseg = int(sec.nseg)
    p1 = np.zeros((nseg, 3))
    p2 = np.zeros((nseg, 3))
    for k in range(nseg):
        p1[k] = point_at_arc_fraction(pts, k / nseg)
        p2[k] = point_at_arc_fraction(pts, (k + 1) / nseg)
    return p1, p2


@dataclass
class SectionGeometry:
    """Per-section pt3d geometry the mesher + segmentation need.

    ``points_um`` / ``diameters_um`` are the section's full pt3d polyline
    (NOT simplified); ``nseg`` matches NEURON's compartment count. The
    façade captures one of these per section, in the SAME order the
    per-segment ``imem`` currents are recorded, so segment indices line up.
    """
    name: str
    points_um: np.ndarray       # (npts, 3)
    diameters_um: np.ndarray    # (npts,)
    nseg: int

    @classmethod
    def from_dict(cls, d: dict) -> "SectionGeometry":
        return cls(
            name=str(d["name"]),
            points_um=np.asarray(d["points_um"], dtype=np.float64),
            diameters_um=np.asarray(d["diameters_um"], dtype=np.float64),
            nseg=int(d["nseg"]),
        )


def capture_section_geometry(sections) -> list[SectionGeometry]:
    """Snapshot every section's pt3d polyline, diameters and nseg.

    Call after the cell (and its 3D shape) exist — if a section has no
    3D points, run ``h.define_shape()`` first. Raises a clear error
    naming the offending section rather than silently dropping it, so
    the captured geometry stays aligned 1:1 with the recorded currents.
    """
    from neuron import h

    geoms: list[SectionGeometry] = []
    for sec in sections:
        n_pt3d = int(h.n3d(sec=sec))
        if n_pt3d < 2:
            raise ValueError(
                f"Section {sec.name()} has {n_pt3d} 3D point(s); need >=2. "
                f"Call h.define_shape() after building the cell so every "
                f"section has a pt3d polyline."
            )
        pts = np.array(
            [[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)]
             for i in range(n_pt3d)],
            dtype=np.float64,
        )
        diams = np.array(
            [h.diam3d(i, sec=sec) for i in range(n_pt3d)], dtype=np.float64,
        )
        geoms.append(SectionGeometry(
            name=sec.name(), points_um=pts, diameters_um=diams,
            nseg=int(sec.nseg),
        ))
    return geoms


@dataclass
class _ImemHandles:
    sections: list      # kept alive so NEURON doesn't GC the cell mid-run
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
    logger.info(f"[swc export] {len(rows)} pt3d points → {out_path}")


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
