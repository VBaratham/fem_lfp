"""Public interface: turn a NEURON simulation into extracellular potential.

The rest of the package (meshers, the ECS Poisson solver, per-segment
segmentations, the line-source approximation) is machinery. This module is
the one thing a user should need to import. It owns the whole pipeline:

    1. arm per-segment transmembrane-current recording (before the run),
    2. after the run, build an extracellular mesh around the cell,
    3. solve the 3D Poisson problem in the extracellular space, and
    4. sample the potential at the requested probe locations,

and it hides the choices that require FEM/ECP background (which mesher,
how to tag the membrane, how to match FEM facets to NEURON compartments,
what boundary conditions to impose) behind sensible defaults.

Minimal use::

    import numpy as np
    from fem_lfp import ExtracellularModel

    # ... build your NEURON cell, set nseg / biophysics / stimulus ...

    probes_um = np.array([[r, 0.0, 0.0] for r in (20, 50, 100, 400)])
    model = ExtracellularModel(h.allsec(), probes_um)   # BEFORE finitialize
    h.finitialize(-65); h.continuerun(30)
    result = model.solve()                              # V_e at the probes
    result.plot("lfp.png")

The constructor arms the current recording, so build it after your cell
exists but before ``h.finitialize()``. Everything after that is automatic;
override any default (mesher, conductivity, mesh sizing) via keyword.
"""
from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lsa import line_source_v_e
from .neuron_sim import (
    SectionGeometry,
    capture_section_geometry,
    finalize_run,
    setup_imem_recording,
)

# Valid values for the ``mesh`` argument, plus a one-line description used
# in error messages so a user who mistypes learns what the options mean.
MESHERS = {
    "cylinder": "single straight cable, axis along z (fast, self-contained)",
    "branched": "arbitrary morphology via fem_neuron's OCC-fuse mesher",
    "body_fitted": "arbitrary morphology via AlphaMeshSwc + TetGen (cleanest)",
}

# Per-mesher default sizing (micrometers). Chosen to reproduce the tuned
# values in the bundled scenarios; override any of them with keyword args.
_MESH_DEFAULTS = {
    "cylinder": dict(
        ecs_pad_um=1500.0, h_membrane_um=0.8, h_outer_um=80.0,
        grade_distance_um=400.0,
    ),
    "branched": dict(
        ecs_pad_um=1000.0, h_membrane_um=1.5, h_outer_um=200.0,
        grade_distance_um=300.0,
    ),
    "body_fitted": dict(
        ecs_pad_um=4000.0, h_outer_um=400.0, tetgen_quality=1.8,
    ),
}


def _as_probe_array(probes_um) -> np.ndarray:
    """Coerce a probe spec into a contiguous (P, 3) float array."""
    pts = np.asarray(probes_um, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts[None, :]
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"probes_um must be shape (P, 3) or (3,), got {pts.shape}"
        )
    return np.ascontiguousarray(pts)


def _looks_like_z_cylinder(g: SectionGeometry) -> bool:
    """True if a lone section is an origin-centered, z-aligned cable.

    That is the only geometry the ``cylinder`` mesher reproduces (it builds
    a z-axis cylinder at the origin), so ``auto`` only picks it when the
    section actually matches; anything else falls back to ``branched``.
    """
    pts = g.points_um
    ext = pts.max(axis=0) - pts.min(axis=0)
    r_max = float(np.max(g.diameters_um)) / 2.0
    xy_flat = ext[0] <= 2.0 * r_max + 1e-6 and ext[1] <= 2.0 * r_max + 1e-6
    z_len = ext[2]
    centered = abs(pts[:, 2].min() + pts[:, 2].max()) <= 0.05 * max(z_len, 1e-9)
    on_axis = (np.abs(pts[:, 0]).max() <= 2.0 * r_max + 1e-6
               and np.abs(pts[:, 1]).max() <= 2.0 * r_max + 1e-6)
    return bool(xy_flat and centered and on_axis and z_len > 0)


@dataclass
class ExtracellularResult:
    """Result of :meth:`ExtracellularModel.solve`.

    Potentials are in microvolts, shaped ``(n_time, n_probe)``. ``v_e_fem``
    is the FEM solution; ``v_e_lsa`` is the line-source approximation over
    the same currents (``None`` if not requested), handy as a reference.
    """
    t_ms: np.ndarray                       # (T,)
    probes_um: np.ndarray                  # (P, 3)
    v_e_fem_uV: np.ndarray | None          # (T, P)
    v_e_lsa_uV: np.ndarray | None          # (T, P)
    imem_nA: np.ndarray                    # (S, T) outward-positive
    p1_um: np.ndarray                      # (S, 3) segment proximal ends
    p2_um: np.ndarray                      # (S, 3) segment distal ends
    v_m_mV: dict[str, np.ndarray]          # label -> (T,)
    mesh: str                              # mesher actually used
    mesh_path: str | None                  # written .xdmf, if any
    timings_s: dict[str, float] = field(default_factory=dict)

    def probe_radii_um(self) -> np.ndarray:
        """Radial (xy) distance of each probe from the z axis, in µm."""
        return np.linalg.norm(self.probes_um[:, :2], axis=1)

    def save(self, path: str | Path) -> Path:
        """Write everything to a single ``.npz`` for later analysis/replot."""
        path = Path(path)
        np.savez(
            path,
            t_ms=self.t_ms,
            probes_um=self.probes_um,
            v_e_fem_uV=(self.v_e_fem_uV if self.v_e_fem_uV is not None
                        else np.array([])),
            v_e_lsa_uV=(self.v_e_lsa_uV if self.v_e_lsa_uV is not None
                        else np.array([])),
            imem_nA=self.imem_nA,
            p1_um=self.p1_um, p2_um=self.p2_um,
            mesh=self.mesh,
            **{f"vm_{k}": v for k, v in self.v_m_mV.items()},
        )
        return path

    def plot(self, path: str | Path, *, title: str | None = None) -> Path:
        """Save the standard V_m / V_e(t) / V_e(r) overlay figure."""
        from .plotting import overlay_fem_vs_lsa
        return overlay_fem_vs_lsa(self, path, title=title)

    @classmethod
    def load(cls, path: str | Path) -> "ExtracellularResult":
        """Reconstruct a result saved by :meth:`save` (e.g. to re-plot)."""
        z = np.load(Path(path))

        def _or_none(key):
            a = z[key]
            return a if a.size else None

        v_m = {k[3:]: z[k] for k in z.files if k.startswith("vm_")}
        return cls(
            t_ms=z["t_ms"], probes_um=z["probes_um"],
            v_e_fem_uV=_or_none("v_e_fem_uV"),
            v_e_lsa_uV=_or_none("v_e_lsa_uV"),
            imem_nA=z["imem_nA"], p1_um=z["p1_um"], p2_um=z["p2_um"],
            v_m_mV=v_m, mesh=str(z["mesh"]), mesh_path=None,
        )


class ExtracellularModel:
    """Extracellular-potential forward model for a NEURON cell.

    Parameters
    ----------
    sections
        The cell's NEURON sections (e.g. ``h.allsec()`` or an explicit
        list). Order is preserved and must match across the run.
    probes_um
        Recording-site coordinates, shape ``(P, 3)`` (or ``(3,)`` for a
        single site), in micrometers, in the cell's own coordinate frame.
    mesh
        Which extracellular mesh to build: ``"auto"`` (default), or one of
        ``"cylinder"``, ``"branched"``, ``"body_fitted"`` (see
        :data:`MESHERS`). ``"auto"`` uses ``cylinder`` for a single
        origin-centered z-cable and ``branched`` otherwise.
    sigma
        Extracellular conductivity in S/m (default 0.3, the LFP-standard
        isotropic value).
    vm_record
        Optional ``{label: segment}`` map of membrane-potential traces to
        keep for plotting. Defaults to the soma midpoint if a soma section
        is found, else the first section's midpoint.
    work_dir
        Where to write mesh files (default: a fresh temp directory).
    record
        If True (default) arm current recording immediately — do this
        before ``h.finitialize()``. Pass False to configure now and call
        :meth:`record` yourself later.
    **mesh_kwargs
        Any extra mesher knob (``ecs_pad_um``, ``h_membrane_um``,
        ``h_outer_um``, ``grade_distance_um``, ``tetgen_quality``, ...),
        overriding the per-mesher defaults.
    """

    def __init__(
        self,
        sections,
        probes_um,
        *,
        mesh: str = "auto",
        sigma: float = 0.3,
        vm_record: dict | None = None,
        work_dir: str | Path | None = None,
        record: bool = True,
        **mesh_kwargs,
    ) -> None:
        self.sections = list(sections)
        if not self.sections:
            raise ValueError("no sections given")
        self.probes_um = _as_probe_array(probes_um)
        if mesh not in ("auto", *MESHERS):
            raise ValueError(
                f"unknown mesh={mesh!r}; choose 'auto' or one of "
                + ", ".join(f"{k!r}" for k in MESHERS)
            )
        self.mesh_request = mesh
        self.sigma = float(sigma)
        self.mesh_kwargs = dict(mesh_kwargs)
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self._geoms = capture_section_geometry(self.sections)

        # Filled by record() / solve().
        self._handles = None
        self._t_vec = None
        self._rec_v: dict = {}
        self._neuron_run = None
        self._vm_record = vm_record

        if record:
            self.record(vm_record)

    # -- alternate constructor: reuse an already-finished NEURON run ------ #
    @classmethod
    def from_run(
        cls,
        neuron_run,
        section_geometries,
        probes_um,
        *,
        sections=None,
        mesh: str = "auto",
        sigma: float = 0.3,
        work_dir: str | Path | None = None,
        **mesh_kwargs,
    ) -> "ExtracellularModel":
        """Build a model around data captured from a run you drove yourself.

        For code that already records ``imem`` and section geometry (e.g.
        the bundled scenarios). ``neuron_run`` is a
        :class:`~fem_lfp.neuron_sim.NeuronRun`; ``section_geometries`` a
        list of :class:`~fem_lfp.neuron_sim.SectionGeometry` (or the plain
        dicts the scenarios collect). Pass ``sections`` (live NEURON
        sections) if you want the ``body_fitted`` mesher, which needs to
        export an SWC.
        """
        self = cls.__new__(cls)
        self.sections = list(sections) if sections is not None else []
        self.probes_um = _as_probe_array(probes_um)
        if mesh not in ("auto", *MESHERS):
            raise ValueError(f"unknown mesh={mesh!r}")
        self.mesh_request = mesh
        self.sigma = float(sigma)
        self.mesh_kwargs = dict(mesh_kwargs)
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self._geoms = [
            g if isinstance(g, SectionGeometry) else SectionGeometry.from_dict(g)
            for g in section_geometries
        ]
        self._handles = None
        self._t_vec = None
        self._rec_v = dict(getattr(neuron_run, "rec_v_mV", {}))
        self._neuron_run = neuron_run
        self._vm_record = None
        return self

    # -------------------------------------------------------------------- #
    def record(self, vm_record: dict | None = None) -> "ExtracellularModel":
        """Arm per-segment current (and V_m) recording. Call before the run.

        Idempotent-ish: calling it a second time re-arms fresh vectors.
        """
        from neuron import h

        self._handles = setup_imem_recording(self.sections)
        self._t_vec = h.Vector().record(h._ref_t)
        if vm_record is None:
            vm_record = self._vm_record
        if vm_record is None:
            vm_record = self._default_vm_sites()
        self._rec_v = {
            label: h.Vector().record(seg._ref_v)
            for label, seg in vm_record.items()
        }
        return self

    def _default_vm_sites(self) -> dict:
        soma = None
        for sec in self.sections:
            name = sec.name().lower()
            if name == "soma" or name.endswith(".soma") or "soma" in name:
                soma = sec
                break
        sec = soma if soma is not None else self.sections[0]
        return {f"{sec.name()}(0.5)": sec(0.5)}

    @property
    def n_seg(self) -> int:
        return sum(g.nseg for g in self._geoms)

    def resolve_mesh(self) -> str:
        """The concrete mesher that will be used (resolving ``auto``)."""
        if self.mesh_request != "auto":
            return self.mesh_request
        if len(self._geoms) == 1 and _looks_like_z_cylinder(self._geoms[0]):
            return "cylinder"
        return "branched"

    # -------------------------------------------------------------------- #
    def solve(
        self,
        *,
        compute_lsa: bool = True,
        progress: bool = True,
    ) -> ExtracellularResult:
        """Build the mesh, run the FEM solve, return potentials at probes.

        Call after ``h.continuerun(...)``. Set ``compute_lsa=False`` to skip
        the line-source reference solution.
        """
        from .fem import run_fem_lfp

        nrun = self._finish_run()
        timings: dict[str, float] = {}

        t0 = time.time()
        mesh, facet_tags, seg, outer_tag, membrane_tag, mesh_path, mesher = (
            self._build_mesh()
        )
        timings["mesh"] = time.time() - t0

        t0 = time.time()
        v_e_fem_uV = run_fem_lfp(
            mesh, facet_tags, seg,
            imem_nA=nrun.imem_nA,
            probe_xyz_um=self.probes_um,
            sigma_S_per_m=self.sigma,
            outer_tag=outer_tag,
            membrane_tag=membrane_tag,
            scale_mesh_to_meters=True,
            progress=progress,
        )
        timings["fem"] = time.time() - t0

        # NaN at a probe means it fell outside the extracellular mesh —
        # either inside the cell (which was carved out) or beyond the box.
        # This is the most common surprise, so flag it explicitly.
        nan_probes = np.where(np.isnan(v_e_fem_uV).any(axis=0))[0]
        if nan_probes.size:
            import warnings
            bad = ", ".join(
                f"{tuple(np.round(self.probes_um[i], 1))}" for i in nan_probes
            )
            warnings.warn(
                f"{nan_probes.size} probe(s) returned NaN (outside the "
                f"extracellular mesh — inside the cell or beyond the "
                f"ecs_pad_um box): {bad}. Move them into the ECS or grow "
                f"ecs_pad_um.",
                stacklevel=2,
            )

        v_e_lsa_uV = None
        if compute_lsa:
            t0 = time.time()
            v_e_lsa_uV = (
                line_source_v_e(
                    self.probes_um, nrun.p1_um, nrun.p2_um, nrun.imem_nA,
                    sigma_S_per_m=self.sigma,
                ).T * 1e6
            )
            timings["lsa"] = time.time() - t0

        return ExtracellularResult(
            t_ms=nrun.t_ms,
            probes_um=self.probes_um,
            v_e_fem_uV=v_e_fem_uV,
            v_e_lsa_uV=v_e_lsa_uV,
            imem_nA=nrun.imem_nA,
            p1_um=nrun.p1_um, p2_um=nrun.p2_um,
            v_m_mV=dict(nrun.rec_v_mV),
            mesh=mesher,
            mesh_path=str(mesh_path) if mesh_path is not None else None,
            timings_s=timings,
        )

    def line_source(self) -> ExtracellularResult:
        """Cheap path: line-source approximation only, no mesh, no FEM.

        Useful as a fast sanity check or when you only want the analytical
        reference. ``result.v_e_fem_uV`` is ``None``.
        """
        nrun = self._finish_run()
        v_e_lsa_uV = (
            line_source_v_e(
                self.probes_um, nrun.p1_um, nrun.p2_um, nrun.imem_nA,
                sigma_S_per_m=self.sigma,
            ).T * 1e6
        )
        return ExtracellularResult(
            t_ms=nrun.t_ms, probes_um=self.probes_um,
            v_e_fem_uV=None, v_e_lsa_uV=v_e_lsa_uV,
            imem_nA=nrun.imem_nA, p1_um=nrun.p1_um, p2_um=nrun.p2_um,
            v_m_mV=dict(nrun.rec_v_mV), mesh="none", mesh_path=None,
        )

    # -------------------------------------------------------------------- #
    def _finish_run(self):
        if self._neuron_run is not None:
            return self._neuron_run
        if self._handles is None:
            raise RuntimeError(
                "recording was never armed — build the model (or call "
                ".record()) BEFORE h.finitialize(), then run, then solve()."
            )
        self._neuron_run = finalize_run(self._handles, self._t_vec, self._rec_v)
        if not np.any(self._neuron_run.imem_nA):
            raise RuntimeError(
                "recorded transmembrane current is all zero — was the "
                "model built after h.finitialize(), or did the cell not "
                "spike? Arm recording before the run."
            )
        return self._neuron_run

    def _mesh_stem(self) -> Path:
        if self.work_dir is None:
            self.work_dir = Path(tempfile.mkdtemp(prefix="fem_lfp_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)
        return self.work_dir / "_mesh"

    def _build_mesh(self):
        """Dispatch to the chosen mesher, return everything the solver needs.

        Returns ``(mesh, facet_tags, seg, outer_tag, membrane_tag,
        mesh_path, mesher_name)``.
        """
        mesher = self.resolve_mesh()
        kw = {**_MESH_DEFAULTS[mesher], **self.mesh_kwargs}
        stem = self._mesh_stem()

        if mesher == "cylinder":
            return (*self._build_cylinder(kw, stem), mesher)
        if mesher == "branched":
            return (*self._build_branched(kw, stem), mesher)
        if mesher == "body_fitted":
            return (*self._build_body_fitted(kw, stem), mesher)
        raise AssertionError(mesher)  # unreachable

    def _build_cylinder(self, kw, stem):
        from .fem import CableSegmentation
        from .mesh_cylinder import (
            CylinderEcsSpec, build_cylinder_ecs_mesh, TAG_MEMBRANE, TAG_OUTER,
        )
        g = self._geoms[0]
        L_um = float(g.points_um[:, 2].max() - g.points_um[:, 2].min())
        radius_um = float(np.max(g.diameters_um)) / 2.0
        spec = CylinderEcsSpec(L_um=L_um, radius_um=radius_um, **kw)
        mesh, _ct, ft = build_cylinder_ecs_mesh(spec, stem)
        seg = CableSegmentation(n_seg=g.nseg, L_um=L_um, axis="z")
        return mesh, ft, seg, TAG_OUTER, TAG_MEMBRANE, stem.with_suffix(".xdmf")

    def _build_branched(self, kw, stem):
        from .fem import BranchedSegmentation
        try:
            from .mesh_branched import (
                SectionPath, build_branched_ecs,
                TAG_MEMBRANE_BASE, TAG_OUTER,
            )
        except ImportError as e:
            raise ImportError(self._fem_neuron_hint("branched")) from e
        sections = [
            SectionPath(points_um=g.points_um, diameters_um=g.diameters_um)
            for g in self._geoms
        ]
        section_nseg = [g.nseg for g in self._geoms]
        bex = build_branched_ecs(
            sections=sections, section_nseg=section_nseg, out_stem=stem, **kw,
        )
        seg = BranchedSegmentation(
            section_tags=[TAG_MEMBRANE_BASE + i for i in range(len(self._geoms))],
            section_polylines_um=bex.section_polylines_um,
            section_nseg=section_nseg,
        )
        return (bex.mesh, bex.facet_tags, seg, TAG_OUTER, None,
                stem.with_suffix(".xdmf"))

    def _build_body_fitted(self, kw, stem):
        from .fem import BranchedSegmentation
        try:
            from .mesh_body_fitted import (
                build_body_fitted_ecs, TAG_MEMBRANE_BASE, TAG_OUTER,
            )
            from .neuron_sim import export_swc_from_neuron
        except ImportError as e:
            raise ImportError(self._fem_neuron_hint("body_fitted")) from e
        if not self.sections:
            raise RuntimeError(
                "the 'body_fitted' mesher needs live NEURON sections to "
                "export an SWC; build the model with sections (not "
                "from_run without sections=...)."
            )
        swc_path = stem.with_name(stem.name + ".swc")
        export_swc_from_neuron(self.sections, swc_path)
        section_nseg = [g.nseg for g in self._geoms]
        anchor = self._soma_center_um()
        bex = build_body_fitted_ecs(
            swc_path=swc_path,
            section_polylines_um=[g.points_um for g in self._geoms],
            section_diameters_um=[g.diameters_um for g in self._geoms],
            section_nseg=section_nseg,
            out_stem=stem,
            ics_anchor_um=anchor,
            **kw,
        )
        seg = BranchedSegmentation(
            section_tags=[TAG_MEMBRANE_BASE + i for i in range(len(self._geoms))],
            section_polylines_um=bex.section_polylines_um,
            section_nseg=section_nseg,
        )
        return (bex.mesh, bex.facet_tags, seg, TAG_OUTER, None,
                stem.with_suffix(".xdmf"))

    def _soma_center_um(self):
        for g in self._geoms:
            if "soma" in g.name.lower():
                return g.points_um.mean(axis=0)
        return None

    @staticmethod
    def _fem_neuron_hint(mesher: str) -> str:
        return (
            f"the {mesher!r} mesher needs the sibling 'fem_neuron' package "
            f"(and, for body_fitted, a patched Alpha_Mesh_Swc). Place "
            f"fem_neuron next to fem_lfp or set FEM_LFP_FEM_NEURON_SRC. For "
            f"a self-contained single-cable model use mesh='cylinder'."
        )
