"""ECS-only mesh for a single cylindrical cell.

Single-domain mesh (no ICS). The cell is cut out of the bounding box,
leaving the ECS as the only volume. The cylinder's outer surface (lateral
+ end caps) is tagged as ``MEMBRANE``; the bounding box exterior is
tagged as ``OUTER`` for Dirichlet far-field.

Per-segment partitioning of the membrane is done post-mesh by the FEM
solver (it bins membrane facets by z-coordinate of their centroid into
the same nseg-along-z bins NEURON uses). Keeps gmsh geometry simple.

Mesh sizing is graded: small near the membrane (h_membrane_um), growing
to h_outer_um at the bounding-box wall over `grade_distance_um`. This
lets us use mm-scale boxes for far-field probes without exploding DOFs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
from mpi4py import MPI

import dolfinx
import gmsh

logger = logging.getLogger(__name__)

TAG_ECS = 1
TAG_OUTER = 2
TAG_MEMBRANE = 3


@dataclass(frozen=True)
class CylinderEcsSpec:
    """Two-stage graded mesh for an ECS cylinder.

    Stage 1 (probe shell): h ramps from ``h_membrane_um`` at the membrane
    to ``h_outer_um`` over distance ``grade_distance_um`` — should cover
    all probe positions.

    Stage 2 (bulk past probes, OPTIONAL): if ``h_far_um`` is set, h ramps
    further from ``h_outer_um`` to ``h_far_um`` over the next
    ``far_grade_distance_um`` of distance from the membrane. Beyond that
    the bulk is uniform at ``h_far_um``. This keeps DOF count tractable
    when ``ecs_pad_um`` is large (mm scale).

    The two stages are combined with a gmsh Min field so we take the
    finer of the two sizes everywhere — Stage 1 dominates near the
    membrane, Stage 2 dominates in the bulk.
    """
    L_um: float                  # cylinder length along z (cell extent)
    radius_um: float             # cylinder radius
    ecs_pad_um: float            # box half-extent beyond cylinder, all sides
    h_membrane_um: float = 1.5
    h_outer_um: float = 30.0
    grade_distance_um: float | None = None  # None → ecs_pad_um
    h_far_um: float | None = None         # None → no second-stage grading
    far_grade_distance_um: float | None = None  # None → ecs_pad_um

    @property
    def box_x_half(self) -> float:
        return self.radius_um + self.ecs_pad_um

    @property
    def box_z_half(self) -> float:
        return self.L_um / 2.0 + self.ecs_pad_um


def build_cylinder_ecs_mesh(
    spec: CylinderEcsSpec,
    out_stem: Path | str,
):
    """Build the ECS-only mesh and return (mesh, cell_tags, facet_tags).

    Also writes <out_stem>.xdmf for inspection.

    Cell tags: 1 = ECS (only volume).
    Facet tags: 2 = OUTER, 3 = MEMBRANE (= cylinder surface).

    Lengths are in µm everywhere (we do not scale to meters; the FEM
    solver carries the conversion in σ and i_mem).
    """
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    Bx = spec.box_x_half
    Bz = spec.box_z_half
    R = spec.radius_um
    L_half = spec.L_um / 2.0
    grade_d = (
        spec.grade_distance_um if spec.grade_distance_um is not None
        else spec.ecs_pad_um
    )

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    try:
        gmsh.model.add("cylinder_ecs")

        box = gmsh.model.occ.addBox(-Bx, -Bx, -Bz, 2 * Bx, 2 * Bx, 2 * Bz)
        cyl = gmsh.model.occ.addCylinder(0.0, 0.0, -L_half, 0.0, 0.0, spec.L_um, R)

        # Subtract cylinder from box → ECS volume only. removeTool=True
        # so the cylinder solid is gone afterward; we only keep its
        # surface as part of the ECS boundary.
        cut_out, _ = gmsh.model.occ.cut(
            [(3, box)], [(3, cyl)], removeObject=True, removeTool=True,
        )
        gmsh.model.occ.synchronize()

        ecs_vols = [tag for dim, tag in cut_out if dim == 3]
        if len(ecs_vols) != 1:
            raise RuntimeError(
                f"expected 1 ECS volume after cut, got {len(ecs_vols)}: {cut_out!r}"
            )
        vol_ecs = ecs_vols[0]

        # All boundary surfaces of the ECS volume.
        ecs_surfs = [s for _, s in gmsh.model.getBoundary(
            [(3, vol_ecs)], oriented=False)]
        # Discriminate outer-box surfaces from the cylinder surface using
        # the centroid of each surface: outer-box surfaces sit on the
        # box wall (|x|=Bx, |y|=Bx, or |z|=Bz); the cylinder surface(s)
        # are the leftover.
        mem_surfs, outer_surfs = [], []
        tol = 1e-6 * max(Bx, Bz)
        for s in ecs_surfs:
            x, y, z = gmsh.model.occ.getCenterOfMass(2, s)
            on_box = (
                abs(abs(x) - Bx) < tol
                or abs(abs(y) - Bx) < tol
                or abs(abs(z) - Bz) < tol
            )
            (outer_surfs if on_box else mem_surfs).append(s)

        if not mem_surfs:
            raise RuntimeError("no membrane surfaces detected on ECS volume")

        gmsh.model.addPhysicalGroup(3, [vol_ecs], TAG_ECS)
        gmsh.model.addPhysicalGroup(2, outer_surfs, TAG_OUTER)
        gmsh.model.addPhysicalGroup(2, mem_surfs, TAG_MEMBRANE)

        f_dist = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(f_dist, "SurfacesList", mem_surfs)

        # Stage 1: membrane → h_outer over grade_distance.
        f_th = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(f_th, "InField", f_dist)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMin", spec.h_membrane_um)
        gmsh.model.mesh.field.setNumber(f_th, "SizeMax", spec.h_outer_um)
        gmsh.model.mesh.field.setNumber(f_th, "DistMin", 0.0)
        gmsh.model.mesh.field.setNumber(f_th, "DistMax", grade_d)
        gmsh.model.mesh.field.setNumber(f_th, "StopAtDistMax", 1)

        # Stage 2 (optional): h_outer → h_far over the bulk past the
        # probe shell. Combined with Min so Stage 1 dominates inside
        # the probe shell (Stage 1 gives the finer h there); past the
        # probe shell Stage 2 takes over and lets h get large in the
        # bulk before reaching the wall.
        active_field = f_th
        if spec.h_far_um is not None:
            far_grade = (
                spec.far_grade_distance_um
                if spec.far_grade_distance_um is not None
                else spec.ecs_pad_um
            )
            f_th2 = gmsh.model.mesh.field.add("Threshold")
            gmsh.model.mesh.field.setNumber(f_th2, "InField", f_dist)
            gmsh.model.mesh.field.setNumber(f_th2, "SizeMin", spec.h_outer_um)
            gmsh.model.mesh.field.setNumber(f_th2, "SizeMax", spec.h_far_um)
            gmsh.model.mesh.field.setNumber(f_th2, "DistMin", grade_d)
            gmsh.model.mesh.field.setNumber(f_th2, "DistMax", grade_d + far_grade)
            gmsh.model.mesh.field.setNumber(f_th2, "StopAtDistMax", 1)
            f_min = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [f_th, f_th2])
            active_field = f_min

        gmsh.model.mesh.field.setAsBackgroundMesh(active_field)

        gmsh.model.mesh.generate(3)

        with tempfile.TemporaryDirectory() as tmp:
            msh_path = Path(tmp) / "ecs.msh"
            gmsh.write(str(msh_path))
            mesh, ct, ft = dolfinx.io.gmshio.read_from_msh(
                str(msh_path), MPI.COMM_WORLD, gdim=3,
            )

        ct.name = "ct"
        ft.name = "ft"
        xdmf = out_stem.with_suffix(".xdmf")
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf, "w") as f:
            f.write_mesh(mesh)
            f.write_meshtags(ct, mesh.geometry)
            f.write_meshtags(ft, mesh.geometry)

        n_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        n_nodes = mesh.geometry.x.shape[0]
        n_mem = int((ft.values == TAG_MEMBRANE).sum())
        n_outer = int((ft.values == TAG_OUTER).sum())
        far_str = (
            f" → {spec.h_far_um}µm" if spec.h_far_um is not None else ""
        )
        logger.info(
            f"[cylinder_ecs] L={spec.L_um}µm r={spec.radius_um}µm "
            f"pad={spec.ecs_pad_um}µm  "
            f"h={spec.h_membrane_um}→{spec.h_outer_um}{far_str} µm  →  "
            f"nodes={n_nodes} cells={n_cells} "
            f"facets(mem={n_mem}, outer={n_outer})"
        )
        return mesh, ct, ft
    finally:
        try:
            gmsh.finalize()
        except Exception:
            pass
