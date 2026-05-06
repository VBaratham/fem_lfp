"""Mainen & Sejnowski j7: NEURON dynamics + LSA + ECS-FEM, overlay plot.

Mirrors ``cylinder_compare.py`` but for a real reconstructed morphology.
Mesh + ECS-submesh extraction goes through ``fem_lfp.mesh_branched``,
which wraps fem_neuron's branched mesher.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "scenarios" / "ms_j7"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCEN_DIR))


def _wall(s: float) -> str:
    if s >= 1.0:
        return f"{s:.2f}s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f}ms"
    return f"{s * 1e6:.2f}µs"


def run_full(body_fitted: bool = False) -> None:
    import scenario as sc

    # 1) NEURON
    print("[NEURON] loading j7 + running ...")
    t0 = time.time()
    nrun, section_info = sc.run_neuron()
    t_neuron = time.time() - t0
    print(f"[NEURON] {len(section_info)} sections, "
          f"{nrun.imem_nA.shape[0]} segs × {nrun.t_ms.size} steps, "
          f"V_m peak {nrun.rec_v_mV['soma(0.5)'].max():.1f} mV "
          f"in {_wall(t_neuron)}")

    # 2) LSA
    from fem_lfp.lsa import line_source_v_e
    t0 = time.time()
    v_e_lsa = line_source_v_e(
        sc.PROBES_UM, nrun.p1_um, nrun.p2_um, nrun.imem_nA,
        sigma_S_per_m=0.3,
    )
    t_lsa = time.time() - t0
    print(f"[LSA]    {_wall(t_lsa)}")

    # 3) FEM
    section_polylines = [info["points_um"] for info in section_info]
    section_diameters = [info["diameters_um"] for info in section_info]
    section_nseg = [info["nseg"] for info in section_info]
    section_tags = [None] * len(section_info)   # filled after mesh

    if body_fitted:
        from fem_lfp.mesh_body_fitted import (
            build_body_fitted_ecs, TAG_OUTER, TAG_MEMBRANE_BASE,
        )
        # Export NEURON's loaded sections as SWC for AMS.
        from fem_lfp.neuron_sim import export_swc_from_neuron
        # Need NEURON's section list — re-load (cheap; cached HOC).
        from neuron import h
        neuron_sections = list(h.allsec())
        swc_path = SCEN_DIR / "_mesh.swc"
        export_swc_from_neuron(neuron_sections, swc_path)

        section_tags = [TAG_MEMBRANE_BASE + i for i in range(len(section_info))]
        print(f"[FEM]    body-fitted: AMS + TetGen ...")
        t0 = time.time()
        bex = build_body_fitted_ecs(
            swc_path=swc_path,
            section_polylines_um=section_polylines,
            section_diameters_um=section_diameters,
            section_nseg=section_nseg,
            out_stem=SCEN_DIR / "_mesh",
            **sc.MESH_BODY_FITTED,
        )
        t_mesh = time.time() - t0
    else:
        from fem_lfp.mesh_branched import (
            SectionPath, build_branched_ecs, TAG_OUTER, TAG_MEMBRANE_BASE,
        )
        sections = [
            SectionPath(points_um=info["points_um"], diameters_um=info["diameters_um"])
            for info in section_info
        ]
        section_tags = [TAG_MEMBRANE_BASE + i for i in range(len(section_info))]
        print(f"[FEM]    branched (OCC fuse): {len(sections)} sections ...")
        t0 = time.time()
        bex = build_branched_ecs(
            sections=sections,
            section_nseg=section_nseg,
            out_stem=SCEN_DIR / "_mesh",
            **sc.MESH,
        )
        t_mesh = time.time() - t0

    print(f"[FEM]    mesh in {_wall(t_mesh)}; submesh has "
          f"{bex.mesh.topology.index_map(bex.mesh.topology.dim).size_local} cells")

    from fem_lfp.fem import BranchedSegmentation, run_fem_lfp
    seg = BranchedSegmentation(
        section_tags=section_tags,
        section_polylines_um=bex.section_polylines_um,
        section_nseg=section_nseg,
    )

    t0 = time.time()
    v_e_fem_uV = run_fem_lfp(
        bex.mesh, bex.facet_tags, seg,
        imem_nA=nrun.imem_nA,
        probe_xyz_um=sc.PROBES_UM,
        sigma_S_per_m=0.3,
        outer_tag=TAG_OUTER,
        scale_mesh_to_meters=True,
    )
    t_fem = time.time() - t0
    print(f"[FEM]    {nrun.t_ms.size} timesteps in {_wall(t_fem)} "
          f"({_wall(t_fem / nrun.t_ms.size)}/step)")

    np.savez(
        SCEN_DIR / "trace.npz",
        t_ms=nrun.t_ms,
        v_m_mV=nrun.rec_v_mV["soma(0.5)"],
        imem_nA=nrun.imem_nA,
        p1_um=nrun.p1_um, p2_um=nrun.p2_um,
        probes_um=sc.PROBES_UM,
        v_e_lsa_uV=(v_e_lsa.T * 1e6),
        v_e_fem_uV=v_e_fem_uV,
        wall_neuron=t_neuron, wall_lsa=t_lsa,
        wall_mesh=t_mesh, wall_fem=t_fem,
    )
    print(f"saved {SCEN_DIR / 'trace.npz'}")
    plot_overlay()


def plot_overlay() -> None:
    npz = np.load(SCEN_DIR / "trace.npz")
    t = npz["t_ms"]
    v_m = npz["v_m_mV"]
    probes = npz["probes_um"]
    v_e_lsa = npz["v_e_lsa_uV"]
    v_e_fem = npz["v_e_fem_uV"]

    radii = np.linalg.norm(probes[:, :2], axis=1)
    order = np.argsort(radii)
    near_i = int(order[0])
    far_i = int(order[-1])
    mid_i = int(order[len(order) // 2])

    snap_idx = int(np.argmax(np.abs(v_e_lsa[:, near_i])))
    t_snap = float(t[snap_idx])

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.5, 2.5, 2.5], hspace=0.35, wspace=0.3)
    ax_vm = fig.add_subplot(gs[0, :])
    ax_near = fig.add_subplot(gs[1, 0], sharex=ax_vm)
    ax_mid = fig.add_subplot(gs[1, 1], sharex=ax_vm)
    ax_far = fig.add_subplot(gs[1, 2], sharex=ax_vm)
    ax_radial = fig.add_subplot(gs[2, :])

    ax_vm.plot(t, v_m, color="C0", lw=1.5, label="V_m at soma(0.5)")
    ax_vm.set_ylabel("V_m (mV)")
    ax_vm.set_xlabel("t (ms)")
    ax_vm.legend(fontsize=9)
    ax_vm.grid(alpha=0.3)
    ax_vm.axvline(t_snap, color="C2", lw=0.8, ls=":")

    for ax, idx in zip([ax_near, ax_mid, ax_far], [near_i, mid_i, far_i]):
        ax.plot(t, v_e_lsa[:, idx], color="C0", lw=1.4, label="LSA")
        ax.plot(t, v_e_fem[:, idx], color="C3", lw=1.4, ls="--", label="FEM")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(f"V_e (µV)  r={radii[idx]:.0f} µm", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(t_snap, color="C2", lw=0.8, ls=":")

    r_sorted = radii[order]
    ax_radial.plot(r_sorted, v_e_lsa[snap_idx, order], "o-",
                   color="C0", lw=1.5, label="LSA")
    ax_radial.plot(r_sorted, v_e_fem[snap_idx, order], "s--",
                   color="C3", lw=1.5, label="FEM")
    ax_radial.set_xscale("log")
    ax_radial.set_xlabel("radial distance (µm)")
    ax_radial.set_ylabel(f"V_e at t={t_snap:.2f} ms (µV)")
    ax_radial.legend(fontsize=9)
    ax_radial.grid(alpha=0.3, which="both")

    plt.suptitle("M&S j7 — FEM vs LSA", y=0.995, fontsize=12)
    out = SCEN_DIR / "overlay.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    print(f"\nV_e summary at t={t_snap:.2f} ms:")
    print(f"  {'r (µm)':>8}  {'LSA (µV)':>10}  {'FEM (µV)':>10}  {'FEM/LSA':>8}")
    for idx in order:
        l = v_e_lsa[snap_idx, idx]
        f = v_e_fem[snap_idx, idx]
        ratio = f / l if abs(l) > 1e-3 else float("nan")
        print(f"  {radii[idx]:8.1f}  {l:10.3f}  {f:10.3f}  {ratio:8.3f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true")
    p.add_argument(
        "--body-fitted", action="store_true",
        help="use AMS + TetGen body-fitted mesh (single watertight cell "
             "surface) instead of fem_neuron's branched OCC-fuse mesher",
    )
    args = p.parse_args()
    if args.replot:
        plot_overlay()
    else:
        run_full(body_fitted=args.body_fitted)


if __name__ == "__main__":
    main()
