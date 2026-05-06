"""ECS-only mesh for a branched cell, by reusing fem_neuron's branched
mesher and extracting the ECS sub-domain.

fem_neuron builds an EMI-style mesh with both ICS and ECS volumes, sharing
the membrane interface. For our LFP forward problem we only solve in the
ECS, so we extract that volume as a submesh and carry the per-section
membrane facet tags + outer-wall tags forward.

The user's preference (2026-05-04): when we move to M&S we reuse
fem_neuron's existing mesh pipeline rather than re-implementing the
branch fusion, OCC repair flags, polygonal-prism workaround, etc. This
module is the thin adapter that does that.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.mesh as dmesh


# Add fem_neuron to sys.path on import. The user keeps the two projects
# as siblings under ~/claude; if their layout differs they can override
# via FEM_LFP_FEM_NEURON_SRC.
def _ensure_fem_neuron_on_path() -> None:
    import os
    explicit = os.environ.get("FEM_LFP_FEM_NEURON_SRC")
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        here = Path(__file__).resolve()
        candidates = [
            here.parents[3] / "fem_neuron" / "src",   # default sibling layout
        ]
    for c in candidates:
        if (c / "fem_neuron" / "__init__.py").is_file():
            if str(c) not in sys.path:
                sys.path.insert(0, str(c))
            return
    raise ImportError(
        "Couldn't find fem_neuron source on disk. Set "
        "FEM_LFP_FEM_NEURON_SRC=/path/to/fem_neuron/src or place "
        "fem_neuron next to fem_lfp."
    )


_ensure_fem_neuron_on_path()


from fem_neuron.mesh.branched import (   # noqa: E402
    SectionPath,
    BranchedMeshSpec,
    cylinder_branched_to_xdmf,
    TAG_ICS, TAG_ECS, TAG_OUTER, TAG_MEMBRANE, TAG_MEMBRANE_BASE,
)


@dataclass
class BranchedEcs:
    """ECS submesh with carried-over facet tags + per-section polylines.

    Attributes
    ----------
    mesh : the ECS submesh (cells = parent ECS cells only).
    facet_tags : MeshTags on the submesh — TAG_OUTER on the bounding-box
        wall, TAG_MEMBRANE_BASE + i on section-i's membrane patch.
        Per-segment-within-section partitioning is done downstream by
        ``BranchedSegmentation``.
    section_polylines_um : list of per-section pt3d polylines
        (each shape ``(npts, 3)``), in MICROMETERS — same units as the
        ``SectionPath`` inputs. Used for arc-length binning of facets
        into segments.
    section_nseg : nseg per section (matching NEURON's split).
    """
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    section_polylines_um: list[np.ndarray]
    section_nseg: list[int]


def build_branched_ecs(
    sections: list[SectionPath],
    section_nseg: list[int],
    out_stem: Path | str,
    *,
    ecs_pad_um: float = 200.0,
    h_membrane_um: float = 1.5,
    h_outer_um: float = 30.0,
    grade_distance_um: float | None = None,
    primitive_shape: str = "polygonal_prism",
    cross_section_n: int = 12,
    simplify_pt3d_polyline: bool = True,
) -> BranchedEcs:
    """Build an ECS-only mesh for the given branched cell.

    1. Calls fem_neuron's branched mesher with ``per_section_tags=True``
       so the membrane is split per section.
    2. Reads the resulting XDMF.
    3. Extracts the ECS sub-domain as a dolfinx submesh.
    4. Transfers the facet tags (membrane + outer) from parent → submesh.
    5. Returns ``BranchedEcs`` ready for the FEM solver.

    The membrane facets of the parent mesh are *interior* facets (between
    ICS and ECS cells); after extracting the ECS submesh they become
    *boundary* facets of that submesh.
    """
    out_stem = Path(out_stem)

    # Save FULL polylines (before simplification) so the BranchedEcs
    # returns them for the segmentation step. The mesher itself runs
    # on simplified polylines (so OCC fuse doesn't choke on j7-class
    # reconstructions with 1500+ pt3d points). Within-section binning
    # in BranchedSegmentation uses the FULL polylines to compute
    # NEURON-segment 3D centers — accurate even when the FEM cell is
    # piecewise straight.
    full_polylines = [s.points_um.copy() for s in sections]

    if simplify_pt3d_polyline:
        sections = [
            SectionPath(
                points_um=np.vstack([s.points_um[0], s.points_um[-1]]),
                diameters_um=np.array(
                    [s.diameters_um[0], s.diameters_um[-1]],
                    dtype=np.float64,
                ),
            )
            for s in sections
        ]

    from fem_neuron.config import config as _fc
    saved = (_fc.mesh.primitive_shape, _fc.mesh.cross_section_n)
    _fc.mesh.primitive_shape = primitive_shape
    _fc.mesh.cross_section_n = cross_section_n
    try:
        spec = BranchedMeshSpec(
            sections=sections,
            ecs_pad_um=ecs_pad_um,
            h_membrane_um=h_membrane_um,
            h_outer_um=h_outer_um,
            grade_distance_um=grade_distance_um,
            per_section_tags=True,
        )
        cylinder_branched_to_xdmf(spec, out_stem)
    finally:
        _fc.mesh.primitive_shape, _fc.mesh.cross_section_n = saved

    # Read parent mesh + tags. fem_neuron's branched mesher writes the
    # mesh under name "mesh" (not the dolfinx default "Grid").
    xdmf = out_stem.with_suffix(".xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf, "r") as f:
        parent = f.read_mesh(name="mesh")
        parent.topology.create_connectivity(
            parent.topology.dim - 1, parent.topology.dim
        )
        ct_parent = f.read_meshtags(parent, "ct")
        ft_parent = f.read_meshtags(parent, "ft")

    # Extract ECS cells.
    tdim = parent.topology.dim
    fdim = tdim - 1
    ecs_cell_idx = ct_parent.indices[ct_parent.values == TAG_ECS]
    if ecs_cell_idx.size == 0:
        raise RuntimeError("No cells with TAG_ECS in parent mesh.")

    sub, sub_to_parent_cells, *_ = dmesh.create_submesh(
        parent, tdim, ecs_cell_idx,
    )
    sub.topology.create_connectivity(fdim, tdim)

    # Build a sub-side facet meshtag by walking the parent's tagged
    # facets and locating each one by VERTEX coordinates on the
    # submesh. This is robust against the parent → sub facet index
    # mapping that dolfinx's submesh API exposes only obliquely.
    parent_tagged = ft_parent.indices
    parent_tagged_values = ft_parent.values
    parent.topology.create_connectivity(fdim, 0)
    p_f_to_v = parent.topology.connectivity(fdim, 0)
    parent_geom = parent.geometry
    parent_dofmap = parent_geom.dofmap

    sub.topology.create_connectivity(fdim, 0)
    s_f_to_v = sub.topology.connectivity(fdim, 0)
    sub_geom_x = sub.geometry.x

    # Optimization: only the submesh's BOUNDARY facets can match a
    # parent membrane/outer facet (membrane was an interior facet of
    # the parent → boundary of the ECS submesh; outer was already
    # boundary). Restricting the key→idx dict to boundary facets cuts
    # the work from ~600k all-facets entries to ~50k boundary
    # entries.
    sub_boundary_facets = dolfinx.mesh.exterior_facet_indices(sub.topology)

    def _facet_key_from_v(coords3: np.ndarray) -> tuple:
        # facet's 3 vertex coords → lexicographically-sorted tuple,
        # rounded to nanometers (1e-9 of mesh unit).
        coords_q = np.round(coords3, 9)
        order = np.lexsort(coords_q.T[::-1])
        return tuple(coords_q[order].flatten())

    sub_key_to_idx: dict[tuple, int] = {}
    for sf in sub_boundary_facets:
        verts = s_f_to_v.links(sf)
        sub_key_to_idx[_facet_key_from_v(sub_geom_x[verts])] = int(sf)

    sub_indices: list[int] = []
    sub_values: list[int] = []
    parent_geom_x = parent_geom.x
    for pf, val in zip(parent_tagged, parent_tagged_values):
        verts = p_f_to_v.links(pf)
        sf = sub_key_to_idx.get(_facet_key_from_v(parent_geom_x[verts]))
        if sf is None:
            continue
        sub_indices.append(sf)
        sub_values.append(int(val))

    if not sub_indices:
        raise RuntimeError(
            "No tagged facets transferred from parent to submesh — likely a "
            "vertex-coordinate rounding mismatch. Try widening the rounding."
        )

    sub_idx_arr = np.array(sub_indices, dtype=np.int32)
    sub_val_arr = np.array(sub_values, dtype=np.int32)
    order = np.argsort(sub_idx_arr)
    sub_ft = dolfinx.mesh.meshtags(
        sub, fdim, sub_idx_arr[order], sub_val_arr[order],
    )
    sub_ft.name = "ft"

    return BranchedEcs(
        mesh=sub,
        facet_tags=sub_ft,
        section_polylines_um=full_polylines,
        section_nseg=list(section_nseg),
    )
