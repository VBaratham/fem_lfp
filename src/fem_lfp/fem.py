"""ECS-only Poisson FEM solver.

Solves
    ∇·(σ ∇φ) = 0      in Ω_e
    σ ∂φ/∂n_e = i_mem  on Γ_m   (n_e outward from ECS, i.e. into cell;
                                  i_mem outward-positive ⇒ current INTO ECS)
    φ = 0              on Γ_outer

with i_mem(x,t) supplied as per-NEURON-segment total currents in nA.
The membrane is partitioned into per-segment patches; in each patch the
current density is uniform at I_k(t) / A_k.

Bilinear form is constant in time → LU-factor once via PETSc, reuse per
timestep. Per-step work is RHS assembly + back-substitution + probe
evaluation.

Internally we work in SI (meters, A, V, S/m). The mesh is scaled from
micrometers to meters once at construction time.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx
import dolfinx.fem.petsc as fem_petsc
import ufl
from dolfinx import fem, geometry

from ._geom import arc_fraction_of_projection, point_at_arc_fraction

logger = logging.getLogger(__name__)

SEG_TAG_OFFSET = 1000  # membrane facet tags become SEG_TAG_OFFSET + seg_idx


@dataclass
class CableSegmentation:
    """Maps each membrane facet of a straight-cylinder mesh to a NEURON
    segment index by binning along the cable axis.

    For now we only support a cylinder oriented along z, with segments
    spanning [-L/2, +L/2]. The first and last bins absorb the end caps.
    """
    n_seg: int
    L_um: float
    axis: str = "z"  # "x", "y", or "z"

    def assign(self, centroids_m: np.ndarray) -> np.ndarray:
        """Return per-facet segment index, given facet centroid coordinates
        in meters."""
        ax = "xyz".index(self.axis)
        L_m = self.L_um * 1e-6
        z = centroids_m[:, ax]
        # bin along [-L/2, +L/2] into n_seg uniform bins
        norm = (z + L_m / 2.0) / L_m  # 0..1
        idx = np.floor(norm * self.n_seg).astype(np.int64)
        return np.clip(idx, 0, self.n_seg - 1)


@dataclass
class BranchedSegmentation:
    """Maps membrane facets of a multi-section cell to per-segment indices.

    NEURON divides each section into ``nseg`` compartments tiling the
    section's pt3d polyline by arc length. Compartment k owns the arc
    interval ``[k/nseg, (k+1)/nseg]`` of normalized arc.

    For each membrane facet of a section: project the facet centroid
    onto the section's FULL pt3d polyline (closest-point projection),
    compute its normalized arc length, bin into the corresponding
    ``floor(frac * nseg)`` compartment.

    This *arc-length tiling* approach (rather than nearest-segment-
    center distance) ensures every NEURON compartment owns the
    facets whose arc-projection falls in its arc interval —
    eliminating the empty-bin failure mode of the nearest-center
    method on sections where the surface mass is unevenly
    distributed along the polyline.
    """
    section_tags: list[int]                  # facet tag for each section
    section_polylines_um: list[np.ndarray]   # (npts, 3) per section, FULL pt3d
    section_nseg: list[int]                  # nseg per section

    @property
    def n_seg_total(self) -> int:
        return sum(self.section_nseg)

    @property
    def section_seg_offsets(self) -> list[int]:
        offs = [0]
        for n in self.section_nseg:
            offs.append(offs[-1] + n)
        return offs

    def segment_centers_m(self, sec_idx: int) -> np.ndarray:
        """3D centers of section ``sec_idx``'s NEURON segments, in meters."""
        poly_um = self.section_polylines_um[sec_idx]
        poly_m = poly_um * 1e-6
        nseg = self.section_nseg[sec_idx]
        return np.stack([
            point_at_arc_fraction(poly_m, (k + 0.5) / nseg)
            for k in range(nseg)
        ])

    def assign(self, sec_idx: int, centroids_m: np.ndarray) -> np.ndarray:
        """Return per-facet GLOBAL segment indices for one section's facets."""
        nseg = self.section_nseg[sec_idx]
        seg_offset = self.section_seg_offsets[sec_idx]
        if nseg == 1:
            return np.full(centroids_m.shape[0], seg_offset, dtype=np.int64)
        poly_um = self.section_polylines_um[sec_idx]
        poly_m = poly_um * 1e-6
        out = np.zeros(centroids_m.shape[0], dtype=np.int64)
        for k in range(centroids_m.shape[0]):
            arc = arc_fraction_of_projection(centroids_m[k], poly_m)
            local = int(min(int(arc * nseg), nseg - 1))
            out[k] = seg_offset + local
        return out


class EcsPoissonSolver:
    """Time-stepping Poisson solver in the ECS, driven by per-segment
    transmembrane currents.

    Geometry inputs are dolfinx objects (mesh, facet_tags) — typically
    produced by ``mesh_cylinder.build_cylinder_ecs_mesh`` (single cable)
    or ``mesh_branched.build_branched_ecs`` (full cell).

    Two segmentation modes are supported via ``seg``:
      - ``CableSegmentation``: single-section cable, classify facets
        by axial position (z bin). ``membrane_tag`` is the tag of the
        whole membrane.
      - ``BranchedSegmentation``: multi-section cell, classify facets
        per-section by arc length along the polyline. The membrane
        is given as the list of per-section tags via
        ``seg.section_tags``; ``membrane_tag`` is ignored.
    """

    def __init__(
        self,
        mesh: dolfinx.mesh.Mesh,
        facet_tags: dolfinx.mesh.MeshTags,
        seg,
        sigma_S_per_m: float = 0.3,
        outer_tag: int = 2,
        membrane_tag: int = 3,
        scale_mesh_to_meters: bool = True,
    ) -> None:
        # Scale mesh to meters so all FEM operations are in SI. Caller
        # may pass an already-scaled mesh (scale_mesh_to_meters=False).
        # This mutates mesh.geometry.x in place, so guard against scaling
        # the same mesh object twice (e.g. two solvers on one mesh) —
        # a double scale would silently shrink the geometry by 1e-12.
        if scale_mesh_to_meters and not getattr(mesh, "_fem_lfp_scaled_m", False):
            mesh.geometry.x[:] *= 1e-6
            try:
                mesh._fem_lfp_scaled_m = True
            except (AttributeError, TypeError):
                pass  # can't tag this mesh object; caller must avoid reuse

        self.mesh = mesh
        self.sigma = sigma_S_per_m
        if isinstance(seg, CableSegmentation):
            self.n_seg = seg.n_seg
        elif isinstance(seg, BranchedSegmentation):
            self.n_seg = seg.n_seg_total
        else:
            raise TypeError(
                f"seg must be CableSegmentation or BranchedSegmentation, "
                f"got {type(seg).__name__}"
            )

        # ----- per-segment facet tags -----
        new_indices, new_values = self._partition_membrane_facets(
            facet_tags, membrane_tag, outer_tag, seg
        )
        per_seg_ft = dolfinx.mesh.meshtags(
            mesh, mesh.topology.dim - 1, new_indices, new_values
        )
        self.facet_tags = per_seg_ft
        n_seg = self.n_seg

        # ----- function space + Dirichlet BC -----
        V = fem.functionspace(mesh, ("Lagrange", 1))
        self.V = V
        self.phi = fem.Function(V)

        outer_facets = per_seg_ft.find(outer_tag)
        outer_dofs = fem.locate_dofs_topological(
            V, mesh.topology.dim - 1, outer_facets
        )
        zero = fem.Function(V)
        zero.x.array[:] = 0.0
        self.bc = fem.dirichletbc(zero, outer_dofs)

        # ----- bilinear form -----
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = self.sigma * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
        self.a = fem.form(a)
        self.A = fem_petsc.assemble_matrix(self.a, bcs=[self.bc])
        self.A.assemble()

        # ----- per-segment: current Constants, area, RHS form -----
        ds = ufl.Measure("ds", domain=mesh, subdomain_data=per_seg_ft)
        self.I_k = [fem.Constant(mesh, PETSc.ScalarType(0.0)) for _ in range(n_seg)]
        self.A_k = np.zeros(n_seg, dtype=np.float64)
        for k in range(n_seg):
            tag = SEG_TAG_OFFSET + k
            area_form = fem.form(
                fem.Constant(mesh, PETSc.ScalarType(1.0)) * ds(tag)
            )
            self.A_k[k] = mesh.comm.allreduce(
                fem.assemble_scalar(area_form), op=MPI.SUM
            )

        # Segments a mesher left with no facets (A_k == 0) have their
        # current redirected to the nearest non-empty neighbour so it
        # isn't silently dropped — see _compute_redirect.
        self._redirect = self._compute_redirect(seg)

        L_terms = []
        for k in range(n_seg):
            tag = SEG_TAG_OFFSET + k
            A_k = self.A_k[k]
            if A_k <= 0:
                continue   # empty bin — its current goes via redirect
            # i_mem density on this patch is I_k / A_k.
            # Outward-positive i_mem means current flows from cell into ECS.
            # j_e = -σ∇φ; flux INTO the ECS at the boundary is j·n_in =
            # -j·n_e = σ∇φ·n_e = i_mem. After integration by parts the
            # boundary term enters the linear form with the same sign as
            # i_mem, no negation.
            L_terms.append(
                (self.I_k[k] / fem.Constant(mesh, PETSc.ScalarType(A_k)))
                * v * ds(tag)
            )
        if not L_terms:
            raise RuntimeError("no segments with non-zero membrane area")
        self.L = fem.form(sum(L_terms))
        self.b = fem_petsc.create_vector(self.L)

        # ----- KSP: LU once, reused across timesteps -----
        ksp = PETSc.KSP().create(mesh.comm)
        ksp.setOperators(self.A)
        ksp.setType("preonly")
        pc = ksp.getPC()
        pc.setType("lu")
        pc.setFactorSolverType("mumps")
        ksp.setFromOptions()
        self.ksp = ksp

        # Geometry is static across timesteps, so build the probe
        # bounding-box tree once here instead of on every probe() call
        # (which was rebuilding it once per timestep).
        self._bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)

    # -------------------------------------------------------------- #
    def _compute_redirect(self, seg) -> dict[int, int]:
        """Map each empty-bin segment (``A_k == 0``) to the nearest
        non-empty NEURON segment by 3D distance.

        Empty bins arise when a mesher merges nearby surfaces (e.g. AMS's
        alpha-wrap fusing M&S j7's proximal axon hillock into the soma),
        leaving some NEURON segments with no facets. Redirecting the
        current cell-wide to the nearest non-empty center — rather than
        within the same section — matches the merged geometry and
        minimizes dipole rotation. Requires ``self.A_k`` to be filled.
        """
        n_seg = self.n_seg
        redirect: dict[int, int] = {}
        if isinstance(seg, BranchedSegmentation):
            offsets = seg.section_seg_offsets
            centers = np.zeros((n_seg, 3), dtype=np.float64)
            for sec_i in range(len(seg.section_nseg)):
                lo, hi = offsets[sec_i], offsets[sec_i + 1]
                centers[lo:hi] = seg.segment_centers_m(sec_i)   # meters
            empties = np.where(self.A_k == 0)[0]
            non_empties = np.where(self.A_k > 0)[0]
            if non_empties.size:
                non_empty_centers = centers[non_empties]
                for k in empties:
                    d2 = ((non_empty_centers - centers[k]) ** 2).sum(axis=1)
                    redirect[int(k)] = int(non_empties[d2.argmin()])
        elif isinstance(seg, CableSegmentation):
            non_empties = np.where(self.A_k > 0)[0]
            if non_empties.size:
                for k in range(n_seg):
                    if self.A_k[k] == 0:
                        redirect[k] = int(
                            non_empties[np.abs(non_empties - k).argmin()]
                        )
        if redirect:
            logger.info(
                "redirecting %d empty-bin segment(s) cell-wide to nearest "
                "non-empty by 3D distance", len(redirect)
            )
        return redirect

    # -------------------------------------------------------------- #
    def _partition_membrane_facets(
        self,
        facet_tags: dolfinx.mesh.MeshTags,
        membrane_tag: int,
        outer_tag: int,
        seg,
    ):
        mesh = self.mesh
        fdim = mesh.topology.dim - 1

        # Centroids of each tagged facet (mesh is now in meters).
        all_idx = facet_tags.indices
        mid_all = dolfinx.mesh.compute_midpoints(mesh, fdim, all_idx)

        # Outer subset is the same in both modes.
        outer_mask = facet_tags.values == outer_tag
        outer_idx = all_idx[outer_mask]

        new_indices_parts: list[np.ndarray] = [outer_idx]
        new_values_parts: list[np.ndarray] = [
            np.full(outer_idx.shape, outer_tag, dtype=np.int32)
        ]

        if isinstance(seg, CableSegmentation):
            mem_mask = facet_tags.values == membrane_tag
            mem_idx = all_idx[mem_mask]
            mem_centroids = mid_all[mem_mask]
            seg_idx = seg.assign(mem_centroids)
            new_values_mem = (SEG_TAG_OFFSET + seg_idx).astype(np.int32)
            new_indices_parts.append(mem_idx)
            new_values_parts.append(new_values_mem)

        elif isinstance(seg, BranchedSegmentation):
            for i, sec_tag in enumerate(seg.section_tags):
                sec_mask = facet_tags.values == sec_tag
                sec_facets = all_idx[sec_mask]
                if sec_facets.size == 0:
                    continue
                sec_centroids = mid_all[sec_mask]
                global_seg = seg.assign(i, sec_centroids)
                new_values_parts.append(
                    (SEG_TAG_OFFSET + global_seg).astype(np.int32)
                )
                new_indices_parts.append(sec_facets)

        else:
            raise TypeError(seg)

        new_indices = np.concatenate(new_indices_parts).astype(np.int32)
        new_values = np.concatenate(new_values_parts)
        order = np.argsort(new_indices)
        return new_indices[order], new_values[order]

    # -------------------------------------------------------------- #
    def step(self, imem_nA: np.ndarray) -> None:
        """Solve for φ given per-segment currents at one timestep."""
        if imem_nA.shape[0] != self.n_seg:
            raise ValueError(
                f"imem_nA length {imem_nA.shape[0]} != n_seg {self.n_seg}"
            )
        # Redirect empty-bin currents to their nearest non-empty
        # neighbour by 3D segment-center distance. Preserves per-cell
        # total current; minimizes dipole rotation by keeping each
        # current as close as possible to its NEURON-prescribed
        # location.
        if self._redirect:
            i_eff = imem_nA.copy()
            for src, dst in self._redirect.items():
                i_eff[dst] += imem_nA[src]
                i_eff[src] = 0.0
        else:
            i_eff = imem_nA
        for k in range(self.n_seg):
            self.I_k[k].value = float(i_eff[k]) * 1e-9   # nA → A

        with self.b.localForm() as loc:
            loc.set(0.0)
        fem_petsc.assemble_vector(self.b, self.L)
        # Lift Dirichlet BC into RHS.
        fem_petsc.apply_lifting(self.b, [self.a], bcs=[[self.bc]])
        self.b.ghostUpdate(
            addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
        )
        fem_petsc.set_bc(self.b, [self.bc])

        self.ksp.solve(self.b, self.phi.x.petsc_vec)
        self.phi.x.scatter_forward()

    def probe(self, probe_xyz_um: np.ndarray) -> np.ndarray:
        """Sample φ at probe points (input µm). Returns φ in volts (P,).

        Uses the dolfinx bb_tree machinery; probes outside the mesh return
        NaN.
        """
        pts = np.asarray(probe_xyz_um, dtype=np.float64) * 1e-6
        # eval expects (n, 3)
        if pts.ndim == 1:
            pts = pts[None, :]

        # Reuse the bb_tree built at construction (geometry is static).
        cand = geometry.compute_collisions_points(self._bb_tree, pts)
        cells = geometry.compute_colliding_cells(self.mesh, cand, pts)

        out = np.full(pts.shape[0], np.nan, dtype=np.float64)
        eval_pts, eval_cells, eval_dst = [], [], []
        for i in range(pts.shape[0]):
            cell_arr = cells.links(i)
            if len(cell_arr) == 0:
                continue
            eval_pts.append(pts[i])
            eval_cells.append(cell_arr[0])
            eval_dst.append(i)
        if eval_pts:
            vals = self.phi.eval(np.array(eval_pts), np.array(eval_cells))
            vals = np.asarray(vals).reshape(-1)
            for j, idx in enumerate(eval_dst):
                out[idx] = float(vals[j])
        return out


def run_fem_lfp(
    mesh,
    facet_tags,
    seg,
    imem_nA: np.ndarray,         # (S, T)
    probe_xyz_um: np.ndarray,    # (P, 3)
    sigma_S_per_m: float = 0.3,
    outer_tag: int = 2,
    membrane_tag: int = 3,
    scale_mesh_to_meters: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """Convenience driver: build solver, step through time, return V_e at
    probes.

    ``imem_nA`` is (S, T); the returned V_e is in microvolts, shape
    (T, P) = (timesteps, probes).
    """
    solver = EcsPoissonSolver(
        mesh, facet_tags, seg,
        sigma_S_per_m=sigma_S_per_m,
        outer_tag=outer_tag, membrane_tag=membrane_tag,
        scale_mesh_to_meters=scale_mesh_to_meters,
    )
    n_t = imem_nA.shape[1]
    n_p = probe_xyz_um.shape[0]
    out = np.zeros((n_t, n_p), dtype=np.float64)

    for ti in range(n_t):
        solver.step(imem_nA[:, ti])
        out[ti] = solver.probe(probe_xyz_um) * 1e6   # V → µV
        if progress and (ti < 3 or ti % max(1, n_t // 20) == 0):
            logger.info("  fem step %d/%d", ti + 1, n_t)
    return out
