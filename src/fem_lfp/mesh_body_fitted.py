"""ECS-only mesh for a branched cell, body-fitted via AlphaMeshSwc + TetGen.

Wraps fem_neuron's ``mesh.body_fitted.cylinder_body_fitted_to_xdmf``,
extracts the ECS submesh, and post-classifies every membrane facet
into a per-section + per-NEURON-segment tag using the cell's pt3d
polylines.

Why body-fitted instead of branched: the branched mesher (OCC fuse +
fragment) loses NEURON segments inside curvy multi-nseg sections —
the simplified per-section straight cylinder can't accommodate all
nseg bins, and 7/199 segments end up with A_k=0 (see
project_ms_simplification_limit.md). Body-fitted produces a single
watertight surface that traces the actual SWC frustum union, so every
NEURON segment along a curvy section has surface area near it.

Fem_neuron's body_fitted produces TAG_MEMBRANE (= 4) as one bulk
tag — it doesn't split per section. That classification is on us:
for each membrane facet on the submesh, find the closest NEURON
section's pt3d polyline, then within that section the closest
NEURON-segment center.

License note: AMS is GPL-3.0. fem_neuron treats it as an external
optional CLI; we follow the same convention. Set
``FEM_NEURON_AMS_ROOT`` if AMS isn't at one of the default sibling
paths.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mpi4py import MPI

import dolfinx
import dolfinx.mesh as dmesh

from .mesh_branched import _ensure_fem_neuron_on_path

_ensure_fem_neuron_on_path()


def _ensure_patched_ams() -> Path:
    """Verify a patched Alpha_Mesh_Swc clone is reachable.

    fem_lfp does not vendor AMS (GPL-3.0). The user clones AMS
    themselves and applies the two patches in
    ``third_party/ams_patches/``. This function reads the resulting
    AMS root path from ``FEM_NEURON_AMS_ROOT`` (or falls back to
    standard sibling paths via fem_neuron's ``_resolve_ams_root``)
    and verifies our patches are present. It does NOT modify
    ``FEM_NEURON_AMS_ROOT`` — fem_neuron's body_fitted already
    consumes that env var directly.

    Why patches are needed: see ``third_party/ams_patches/README.md``.
    """
    import os
    from fem_neuron.mesh.body_fitted import _resolve_ams_root
    explicit = os.environ.get("FEM_NEURON_AMS_ROOT")
    try:
        ams_root = _resolve_ams_root()
    except RuntimeError:
        raise RuntimeError(
            "Couldn't locate Alpha_Mesh_Swc. Clone it from "
            "https://github.com/AlexMcSD/Alpha_Mesh_Swc, apply "
            "fem_lfp's patches with "
            "third_party/ams_patches/apply.sh, then set "
            "FEM_NEURON_AMS_ROOT to point at the clone.\n"
            f"(explicit FEM_NEURON_AMS_ROOT={explicit!r}, "
            f"none of the default sibling paths exist either.)"
        )
    # Sanity-check: look for the marker we put in patch 0001.
    marker_path = ams_root / "src" / "mesh_processing.py"
    if marker_path.is_file():
        marker_text = marker_path.read_text()
        if "fem_lfp patch" not in marker_text:
            raise RuntimeError(
                f"AMS clone at {ams_root} does not have fem_lfp's "
                f"patches applied. Run "
                f"`bash third_party/ams_patches/apply.sh {ams_root}` "
                f"to apply them. See "
                f"third_party/ams_patches/README.md for why."
            )
    return ams_root


from fem_neuron.mesh.body_fitted import (   # noqa: E402
    BodyFittedMeshSpec,
    cylinder_body_fitted_to_xdmf,
)
from fem_neuron.mesh.cylinder import (   # noqa: E402
    TAG_ECS, TAG_OUTER, TAG_MEMBRANE,
)


# We keep the same per-segment tag offset as the branched path so the
# downstream solver code doesn't need to care which mesher made the
# submesh.
TAG_MEMBRANE_BASE = 100


@dataclass
class BodyFittedEcs:
    """ECS submesh produced by AMS + TetGen, with per-section facet tags.

    Same shape as ``mesh_branched.BranchedEcs`` so the FEM driver
    code is mesher-agnostic.
    """
    mesh: dolfinx.mesh.Mesh
    facet_tags: dolfinx.mesh.MeshTags
    section_polylines_um: list[np.ndarray]
    section_nseg: list[int]


def _patched_run_ams_factory(min_faces):
    """Build a replacement for fem_neuron's _run_ams that passes --min_faces.

    fem_neuron's default lets AMS auto-pick min_faces (~29k on j7),
    which collapses the axon hillock + thin-axon detail. Bumping
    min_faces to e.g. 100k keeps enough surface area for nseg>1
    sections to populate all their bins.
    """
    def _run(swc, out_dir, alpha_fraction):
        import subprocess
        from fem_neuron.mesh.body_fitted import _resolve_ams_root
        swc_p = Path(swc).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        ams_root = _resolve_ams_root()
        cmd = ["python", "mesh_swc.py", str(swc_p),
               "--output_dir", str(out_dir)]
        if alpha_fraction is not None:
            cmd += ["--alpha", str(alpha_fraction)]
        if min_faces is not None:
            cmd += ["--min_faces", str(int(min_faces))]
        print(f"[body_fitted] AMS: {' '.join(cmd)} (cwd={ams_root})")
        res = subprocess.run(cmd, cwd=str(ams_root),
                             capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"AlphaMeshSwc failed (exit {res.returncode}):\n"
                f"--- stdout ---\n{res.stdout}\n"
                f"--- stderr ---\n{res.stderr}"
            )
        expected = out_dir / (swc_p.stem + ".ply")
        if not expected.exists():
            raise RuntimeError(
                f"AMS finished but didn't produce {expected}. "
                f"stdout tail:\n{res.stdout[-1000:]}"
            )
        return expected
    return _run


def build_body_fitted_ecs(
    swc_path: Path | str,
    section_polylines_um: list[np.ndarray],
    section_nseg: list[int],
    out_stem: Path | str,
    *,
    section_diameters_um: list[np.ndarray] | None = None,
    ecs_pad_um: float = 4000.0,
    h_outer_um: float = 200.0,
    alpha_fraction: float | None = None,
    tetgen_quality: float = 1.4,
    ams_min_faces: int | None = 100000,
    ics_anchor_um: np.ndarray | None = None,
) -> BodyFittedEcs:
    """Build a body-fitted ECS-only mesh and tag membrane facets per
    NEURON section.

    Parameters mirror fem_neuron's ``BodyFittedMeshSpec`` plus the
    per-section polylines we need for downstream segment binning.
    ``section_polylines_um[i]`` is the FULL pt3d polyline of NEURON
    section i — used by ``BranchedSegmentation`` to place each
    segment center at its physical position on the curve, then the
    classifier here assigns each membrane facet to the closest
    section.
    """
    out_stem = Path(out_stem)
    _ensure_patched_ams()

    spec = BodyFittedMeshSpec(
        swc_path=Path(swc_path),
        ecs_pad_um=ecs_pad_um,
        h_outer_um=h_outer_um,
        alpha_fraction=alpha_fraction,
        tetgen_quality=tetgen_quality,
    )

    # Cache the AMS+TetGen output. fem_neuron's body_fitted re-runs AMS
    # + TetGen each call (~5 min for j7). We hash the geometry+sizing
    # spec + SWC content; if the resulting XDMF is on disk and matches
    # the hash, skip the rebuild. ~/.cache/fem_lfp_meshes/<hash>.xdmf
    import hashlib
    swc_bytes = Path(swc_path).read_bytes()
    h = hashlib.sha256()
    h.update(swc_bytes)
    h.update(repr((
        float(ecs_pad_um), float(h_outer_um),
        alpha_fraction, float(tetgen_quality),
        ams_min_faces,
    )).encode())
    cache_key = h.hexdigest()[:16]
    cache_dir = Path.home() / ".cache" / "fem_lfp_meshes"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_xdmf = cache_dir / f"{cache_key}.xdmf"
    cache_h5 = cache_dir / f"{cache_key}.h5"
    target_xdmf = out_stem.with_suffix(".xdmf")
    target_h5 = out_stem.parent / f"{out_stem.name}.h5"
    if cache_xdmf.is_file() and cache_h5.is_file():
        import shutil
        print(f"[body_fitted cache HIT] {cache_xdmf} → {target_xdmf}")
        shutil.copyfile(cache_xdmf, target_xdmf)
        shutil.copyfile(cache_h5, target_h5)
    else:
        # Inject our --min_faces patch into fem_neuron's body_fitted
        # only for this call. Also override _interior_point_in_cell
        # when the user supplies an explicit anchor: fem_neuron's
        # default heuristic (face[0] centroid + small inward offset,
        # fallback to mesh COM) fails on cells with branchy
        # non-convex morphology (Hay 2011 L5 PC's 1.4 mm apical
        # dendrite — COM lands outside the cell, regions swap, 0
        # membrane facets).
        import fem_neuron.mesh.body_fitted as _bf_mod
        _orig_run_ams = _bf_mod._run_ams
        _bf_mod._run_ams = _patched_run_ams_factory(ams_min_faces)
        if ics_anchor_um is not None:
            _orig_interior = _bf_mod._interior_point_in_cell
            anchor = np.asarray(ics_anchor_um, dtype=np.float64)
            _bf_mod._interior_point_in_cell = lambda cell_mesh: anchor.copy()
        try:
            cylinder_body_fitted_to_xdmf(spec, out_stem)
        finally:
            _bf_mod._run_ams = _orig_run_ams
            if ics_anchor_um is not None:
                _bf_mod._interior_point_in_cell = _orig_interior
        import shutil
        shutil.copyfile(target_xdmf, cache_xdmf)
        shutil.copyfile(target_h5, cache_h5)
        print(f"[body_fitted cache STORE] {target_xdmf} → {cache_xdmf}")

    xdmf = out_stem.with_suffix(".xdmf")
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, xdmf, "r") as f:
        parent = f.read_mesh(name="mesh")
        parent.topology.create_connectivity(
            parent.topology.dim - 1, parent.topology.dim
        )
        ct_parent = f.read_meshtags(parent, "ct")
        ft_parent = f.read_meshtags(parent, "ft")

    tdim = parent.topology.dim
    fdim = tdim - 1
    ecs_cells = ct_parent.indices[ct_parent.values == TAG_ECS]
    if ecs_cells.size == 0:
        raise RuntimeError("No TAG_ECS cells in parent mesh.")

    sub, sub_to_parent_cells, *_ = dmesh.create_submesh(parent, tdim, ecs_cells)
    sub.topology.create_connectivity(fdim, tdim)
    sub.topology.create_connectivity(fdim, 0)
    parent.topology.create_connectivity(fdim, 0)

    # ----- transfer outer + bulk-membrane facet tags from parent → sub -----
    sub_bndry = dmesh.exterior_facet_indices(sub.topology)
    s_f_to_v = sub.topology.connectivity(fdim, 0)
    p_f_to_v = parent.topology.connectivity(fdim, 0)
    sub_x = sub.geometry.x
    parent_x = parent.geometry.x

    def _facet_key(coords3: np.ndarray) -> tuple:
        cq = np.round(coords3, 9)
        order = np.lexsort(cq.T[::-1])
        return tuple(cq[order].flatten())

    sub_key_to_idx: dict[tuple, int] = {}
    for sf in sub_bndry:
        verts = s_f_to_v.links(sf)
        sub_key_to_idx[_facet_key(sub_x[verts])] = int(sf)

    # parent's ft has TAG_MEMBRANE for the cell surface and TAG_OUTER for
    # the box. We want to keep TAG_OUTER on the submesh as-is, and
    # RECLASSIFY TAG_MEMBRANE facets per-section below.
    sub_outer_idx: list[int] = []
    parent_membrane_subfacets: list[int] = []   # sub-facet indices
    for pf, val in zip(ft_parent.indices, ft_parent.values):
        verts = p_f_to_v.links(pf)
        sf = sub_key_to_idx.get(_facet_key(parent_x[verts]))
        if sf is None:
            continue
        if int(val) == TAG_OUTER:
            sub_outer_idx.append(sf)
        elif int(val) == TAG_MEMBRANE:
            parent_membrane_subfacets.append(sf)

    if not parent_membrane_subfacets:
        raise RuntimeError("body-fitted mesh has no TAG_MEMBRANE facets")

    # ----- per-section classification of the membrane facets -----
    # Centroids of just the membrane sub-facets, in mesh units (µm).
    mem_idx_arr = np.array(parent_membrane_subfacets, dtype=np.int32)
    mem_centroids_um = dolfinx.mesh.compute_midpoints(
        sub, fdim, mem_idx_arr,
    )

    polylines_um = [np.asarray(p, dtype=np.float64) for p in section_polylines_um]
    if section_diameters_um is not None:
        diameters = [np.asarray(d, dtype=np.float64) for d in section_diameters_um]
    else:
        # Fall back to a uniform 1 µm radius for every polyline point —
        # equivalent to the old polyline-only classifier.
        diameters = [np.full(p.shape[0], 2.0) for p in polylines_um]
    sec_assigned = _classify_facets_to_sections(
        mem_centroids_um, polylines_um, diameters,
    )

    sub_indices: list[int] = sub_outer_idx[:]
    sub_values: list[int] = [TAG_OUTER] * len(sub_outer_idx)
    for k, sf in enumerate(parent_membrane_subfacets):
        sub_indices.append(int(sf))
        sub_values.append(TAG_MEMBRANE_BASE + int(sec_assigned[k]))

    sub_idx_arr = np.array(sub_indices, dtype=np.int32)
    sub_val_arr = np.array(sub_values, dtype=np.int32)
    order = np.argsort(sub_idx_arr)
    sub_ft = dolfinx.mesh.meshtags(
        sub, fdim, sub_idx_arr[order], sub_val_arr[order],
    )
    sub_ft.name = "ft"

    n_outer = len(sub_outer_idx)
    n_mem = len(parent_membrane_subfacets)
    print(f"[body_fitted_ecs] submesh: {sub.topology.index_map(tdim).size_local}"
          f" cells; outer={n_outer}, membrane={n_mem} facets across "
          f"{len(polylines_um)} sections")

    return BodyFittedEcs(
        mesh=sub,
        facet_tags=sub_ft,
        section_polylines_um=polylines_um,
        section_nseg=list(section_nseg),
    )


def _classify_facets_to_sections(
    centroids_um: np.ndarray,
    polylines_um: list[np.ndarray],
    diameters_um: list[np.ndarray],   # accepted for API compat; unused for now
) -> np.ndarray:
    """Assign each facet centroid to the closest section's polyline.

    We tried a radius-aware metric (``|distance − r_section|``) to fix
    the soma-vs-hillock attribution problem, but it produced
    drastically WORSE results — many membrane facets ended up
    assigned to thin distal sections whose local radius coincidentally
    matched the alpha-wrap-vs-polyline distance better than the
    physically-correct fat section. Reverted.

    For now: closest-polyline classifier. The hillock-attribution
    issue manifests as ~3 empty bins on j7, handled downstream by
    ``EcsPoissonSolver``'s area-weighted in-section redistribution.
    """
    out = np.zeros(centroids_um.shape[0], dtype=np.int64)
    F = centroids_um.shape[0]
    chunk = 2000
    best_d2 = None
    for sec_i, poly in enumerate(polylines_um):
        if poly.shape[0] < 2:
            continue
        segs_a = poly[:-1]
        segs_b = poly[1:]
        ab = segs_b - segs_a
        L2 = (ab * ab).sum(axis=1)
        L2_safe = np.where(L2 > 1e-30, L2, 1.0)

        sec_min = np.full(F, np.inf, dtype=np.float64)
        for c0 in range(0, F, chunk):
            c1 = min(F, c0 + chunk)
            ap = centroids_um[c0:c1, None, :] - segs_a[None, :, :]
            t = (ap * ab[None, :, :]).sum(axis=2) / L2_safe[None, :]
            t = np.clip(t, 0.0, 1.0)
            closest = segs_a[None, :, :] + t[..., None] * ab[None, :, :]
            d2 = ((centroids_um[c0:c1, None, :] - closest) ** 2).sum(axis=2)
            sec_min[c0:c1] = d2.min(axis=1)

        if best_d2 is None:
            best_d2 = sec_min.copy()
            out.fill(sec_i)
        else:
            better = sec_min < best_d2
            best_d2 = np.where(better, sec_min, best_d2)
            out = np.where(better, sec_i, out)
    return out
