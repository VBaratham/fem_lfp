"""Diagnostic: drive a SINGLE segment with constant current on the M&S
mesh, check FEM vs LSA at probes.

If LSA and FEM agree at probes far from the active segment, the
solver/mesh/units are correct, and the M&S divergence is per-segment
(BranchedSegmentation mapping issue). If they disagree by a constant
factor, the problem is global (units, scaling).
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "ms_j7"))
logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    import scenario as sc

    # Load NEURON's section info (cached)
    nrun, section_info = sc.run_neuron()
    print(f"loaded {len(section_info)} sections, {nrun.imem_nA.shape[0]} segments")

    from fem_lfp.mesh_branched import (
        SectionPath, build_branched_ecs, TAG_OUTER, TAG_MEMBRANE_BASE,
    )
    from fem_lfp.fem import BranchedSegmentation, EcsPoissonSolver

    sections = []
    section_nseg = []
    section_tags = []
    for i, info in enumerate(section_info):
        sections.append(SectionPath(
            points_um=info["points_um"],
            diameters_um=info["diameters_um"],
        ))
        section_nseg.append(info["nseg"])
        section_tags.append(TAG_MEMBRANE_BASE + i)

    print("[FEM] building / loading mesh ...")
    bex = build_branched_ecs(
        sections=sections,
        section_nseg=section_nseg,
        out_stem=ROOT / "scenarios" / "ms_j7" / "_mesh",
        **sc.MESH,
    )
    seg = BranchedSegmentation(
        section_tags=section_tags,
        section_polylines_um=bex.section_polylines_um,
        section_nseg=section_nseg,
    )
    n_seg_total = seg.n_seg_total
    print(f"[FEM] BranchedSegmentation has {n_seg_total} global segments")

    solver = EcsPoissonSolver(
        bex.mesh, bex.facet_tags, seg,
        sigma_S_per_m=0.3, outer_tag=TAG_OUTER, membrane_tag=99999,
        scale_mesh_to_meters=True,
    )
    print(f"solver: A_k stats — sum={solver.A_k.sum():.6e} m², "
          f"min={solver.A_k.min():.3e}, max={solver.A_k.max():.3e}")
    print(f"        ({solver.A_k.sum()*1e12:.1f} µm² total membrane)")

    # Test 2: drive ALL segments uniformly at +1 nA each → far-field should
    # be LSA's ΣI/(4πσr) with ΣI = n_seg_total nA. If FEM matches, sup
    # works; if FEM blows up, the multi-source path has a bug.
    print("\n=== Test A: ALL segs at +1 nA ===")
    imem_all = np.ones(n_seg_total)
    solver.step(imem_all)
    probe_r_um = np.geomspace(20.0, 1500.0, 14)
    probes = np.column_stack([probe_r_um, np.zeros_like(probe_r_um), np.zeros_like(probe_r_um)])
    fem_uV = solver.probe(probes) * 1e6
    from fem_lfp.lsa import line_source_v_e
    v_e_lsa = line_source_v_e(probes, nrun.p1_um, nrun.p2_um, imem_all[:, None].astype(np.float64), sigma_S_per_m=0.3)
    lsa_uV = v_e_lsa[:, 0] * 1e6
    print(f"  total ΣI = {imem_all.sum():.1f} nA")
    print(f"  {'r (µm)':>8}  {'LSA µV':>10}  {'FEM µV':>10}  {'FEM/LSA':>8}")
    for k, r in enumerate(probe_r_um):
        ratio = fem_uV[k] / lsa_uV[k] if abs(lsa_uV[k]) > 1e-6 else float("nan")
        print(f"  {r:8.1f}  {lsa_uV[k]:10.3e}  {fem_uV[k]:10.3e}  {ratio:8.3f}")

    # Test B: feed the actual saved imem at AP-rising t=8.25 ms
    print("\n=== Test B: actual i_mem at t=8.25 ms ===")
    z = np.load(ROOT / "scenarios" / "ms_j7" / "trace.npz")
    t = z["t_ms"]
    ti = int(np.argmin(np.abs(t - 8.25)))
    imem_t = z["imem_nA"][:, ti]
    solver.step(imem_t)
    fem_uV = solver.probe(probes) * 1e6
    v_e_lsa = line_source_v_e(probes, nrun.p1_um, nrun.p2_um, imem_t[:, None].astype(np.float64), sigma_S_per_m=0.3)
    lsa_uV = v_e_lsa[:, 0] * 1e6
    print(f"  total ΣI = {imem_t.sum():.4f} nA  (sum |i_k| = {np.abs(imem_t).sum():.2f} nA)")
    print(f"  {'r (µm)':>8}  {'LSA µV':>10}  {'FEM µV':>10}  {'FEM/LSA':>8}")
    for k, r in enumerate(probe_r_um):
        ratio = fem_uV[k] / lsa_uV[k] if abs(lsa_uV[k]) > 1e-6 else float("nan")
        print(f"  {r:8.1f}  {lsa_uV[k]:10.3e}  {fem_uV[k]:10.3e}  {ratio:8.3f}")

    # OLD Test 1 (single source) for comparison
    print("\n=== Test C: single segment 0 at +1 nA ===")
    target_seg = 0
    imem = np.zeros(n_seg_total)
    imem[target_seg] = 1.0  # 1 nA
    solver.step(imem)

    # Sample at probes radially around soma at log-spaced r
    probe_r_um = np.geomspace(20.0, 1500.0, 14)
    probes = np.column_stack([probe_r_um, np.zeros_like(probe_r_um), np.zeros_like(probe_r_um)])
    fem_uV = solver.probe(probes) * 1e6   # V → µV

    # LSA from segment endpoints + 1 nA at target
    p1 = nrun.p1_um
    p2 = nrun.p2_um
    seg_xyz = 0.5 * (p1[target_seg] + p2[target_seg])
    print(f"\nTarget seg {target_seg} at {seg_xyz} (length "
          f"{np.linalg.norm(p2[target_seg]-p1[target_seg]):.2f} µm)")

    from fem_lfp.lsa import line_source_v_e
    imem_T = imem[:, None].astype(np.float64)   # (S, 1)
    v_e_lsa = line_source_v_e(probes, p1, p2, imem_T, sigma_S_per_m=0.3)
    lsa_uV = v_e_lsa[:, 0] * 1e6

    print(f"\n  {'r (µm)':>8}  {'dist to seg':>12}  {'LSA µV':>10}  {'FEM µV':>10}  {'FEM/LSA':>8}")
    for k, r in enumerate(probe_r_um):
        d = np.linalg.norm(probes[k] - seg_xyz)
        ratio = fem_uV[k] / lsa_uV[k] if abs(lsa_uV[k]) > 1e-6 else float("nan")
        print(f"  {r:8.1f}  {d:12.1f}  {lsa_uV[k]:10.3e}  {fem_uV[k]:10.3e}  {ratio:8.3f}")


if __name__ == "__main__":
    main()
