"""Compare FEM vs LSA on the SAME simplified geometry.

The default LSA uses NEURON's full pt3d for p1/p2 of each segment.
The FEM mesh uses simplified per-section polylines (first→last pt3d).
For curvy sections, this geometry mismatch produces an apparent
"FEM/LSA gap" that's really a simplification artifact, not a physics
difference.

This script recomputes LSA using simplified-section geometry — each
segment k of nseg in section i is placed along the chord between the
section's first and last pt3d at fraction (k+0.5)/nseg, with chord
length = section's full arc length divided by nseg. Then compares to
FEM at the same probes.

If FEM ≈ simplified LSA, the gap is geometric (we lose curvy detail
in FEM but LSA still sees it). If FEM still disagrees, there's a
real FEM-vs-LSA-on-same-geometry gap (boundary effects, etc.).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scenarios" / "ms_j7"))


def main() -> None:
    import scenario as sc
    from fem_lfp.lsa import line_source_v_e

    z = np.load(ROOT / "scenarios" / "ms_j7" / "trace.npz")
    t = z["t_ms"]
    imem = z["imem_nA"]
    probes = z["probes_um"]

    # Re-extract section_info to know nseg per section
    print("loading NEURON section info ...")
    nrun, section_info = sc.run_neuron()
    p1_full = nrun.p1_um
    p2_full = nrun.p2_um

    # Build SIMPLIFIED per-segment endpoints: each section is a straight
    # chord first→last; segment k of nseg spans arc fraction k/nseg →
    # (k+1)/nseg along the chord.
    p1_simp = np.zeros_like(p1_full)
    p2_simp = np.zeros_like(p2_full)
    seg_offset = 0
    for info in section_info:
        nseg = info["nseg"]
        first = info["points_um"][0]
        last = info["points_um"][-1]
        for k in range(nseg):
            f0 = k / nseg
            f1 = (k + 1) / nseg
            p1_simp[seg_offset + k] = (1 - f0) * first + f0 * last
            p2_simp[seg_offset + k] = (1 - f1) * first + f1 * last
        seg_offset += nseg

    # LSA returns (P, T); transpose to (T, P) to match v_e_fem.
    v_e_lsa_full = line_source_v_e(probes, p1_full, p2_full, imem).T * 1e6
    v_e_lsa_simp = line_source_v_e(probes, p1_simp, p2_simp, imem).T * 1e6
    v_e_fem = z["v_e_fem_uV"]

    radii = np.linalg.norm(probes[:, :2], axis=1)
    order = np.argsort(radii)

    # Snapshot at AP rising
    snap_idx = int(np.argmin(np.abs(t - 8.25)))
    print(f"\nSnapshot at t={t[snap_idx]:.2f} ms:")
    print(f"  {'r (µm)':>8}  {'LSA full':>10}  {'LSA simp':>10}  "
          f"{'FEM':>10}  {'FEM/L_simp':>10}  {'FEM/L_full':>10}")
    for i in order:
        if np.isnan(v_e_fem[snap_idx, i]):
            continue
        l_full = v_e_lsa_full[snap_idx, i]
        l_simp = v_e_lsa_simp[snap_idx, i]
        f = v_e_fem[snap_idx, i]
        rs = f / l_simp if abs(l_simp) > 1e-3 else float('nan')
        rf = f / l_full if abs(l_full) > 1e-3 else float('nan')
        print(f"  {radii[i]:8.1f}  {l_full:10.2f}  {l_simp:10.2f}  "
              f"{f:10.2f}  {rs:10.3f}  {rf:10.3f}")

    # Plot V_e(r) at the snapshot for full vs simp LSA + FEM
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    r_sorted = radii[order]
    finite_mask = ~np.isnan(v_e_fem[snap_idx, order])
    ax.plot(r_sorted, v_e_lsa_full[snap_idx, order],
            "o-", color="C0", lw=1.5, label="LSA (full pt3d)")
    ax.plot(r_sorted, v_e_lsa_simp[snap_idx, order],
            "^-.", color="C2", lw=1.5, label="LSA (simplified geometry)")
    ax.plot(r_sorted[finite_mask], v_e_fem[snap_idx, order][finite_mask],
            "s--", color="C3", lw=1.5, label="FEM (simplified mesh)")
    ax.set_xscale("log")
    ax.set_xlabel("radial distance (µm)")
    ax.set_ylabel(f"V_e at t={t[snap_idx]:.2f} ms (µV)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Snapshot: full-LSA vs simp-LSA vs FEM")

    ax = axes[1]
    near_i = int(order[0])
    ax.plot(t, v_e_lsa_full[:, near_i], color="C0", lw=1.4, label="LSA (full)")
    ax.plot(t, v_e_lsa_simp[:, near_i], color="C2", lw=1.4, ls="-.",
            label="LSA (simp)")
    ax.plot(t, v_e_fem[:, near_i], color="C3", lw=1.4, ls="--", label="FEM")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel(f"V_e (µV) at r={radii[near_i]:.0f} µm")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"V_e(t) at near probe r={radii[near_i]:.0f} µm")

    plt.tight_layout()
    out = ROOT / "scenarios" / "ms_j7" / "lsa_simp_compare.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
