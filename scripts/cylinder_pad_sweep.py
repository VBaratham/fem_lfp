"""Sweep ecs_pad_um and plot FEM-vs-LSA convergence.

Runs the cylinder scenario at several outer-box sizes, then overlays the
V_e(r) snapshot curves at the same time point. As ecs_pad_um grows the
FEM curve should approach LSA in the far field (probes far enough from
the wall stop seeing the Dirichlet pull-down).

Usage::

    python scripts/cylinder_pad_sweep.py
    python scripts/cylinder_pad_sweep.py --replot      # just remake the plot
    python scripts/cylinder_pad_sweep.py --pads 2000 4000 8000

Outputs scenarios/cylinder/pad_sweep.png and the per-pad trace_padXXXX.npz
files (so the per-pad runs are also viewable individually via
``python scripts/cylinder_compare.py --replot --ecs-pad-um <X>``).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "scenarios" / "cylinder"
RUNNER = ROOT / "scripts" / "cylinder_compare.py"


def run_one(pad: float) -> Path:
    npz = SCEN_DIR / f"trace_pad{int(pad)}.npz"
    if npz.is_file():
        print(f"[pad {pad:.0f}] already done → {npz.name}, skipping run")
        return npz
    # For larger pads, enable Stage-2 grading so the bulk past the
    # probe shell coarsens further toward the wall. The probe shell
    # (set by scenario.MESH['grade_distance_um']=1000 µm) covers all
    # probes; Stage 2 only handles the bulk past probes. h_far is
    # capped at 400 µm — coarser than that and the Dirichlet wall's
    # near field is too poorly resolved to give a clean comparison.
    cmd = [
        sys.executable, "-u", str(RUNNER),
        "--ecs-pad-um", str(pad),
    ]
    if pad > 2000:
        # Stage-2 ramps gently from h_outer=80 over far_grade=2000 µm,
        # so at the outermost probe (r=800) the local mesh is still
        # ~100 µm. h_far is the asymptotic bulk size — picked per-pad
        # so the bulk DOF count stays bounded:
        #   pad=4000: bulk volume small, h_far=200 µm (~3k bulk tets)
        #   pad=8000: bulk volume ~26× larger, h_far=300 µm (~140k tets)
        h_far = 200.0 if pad <= 4000 else 300.0
        cmd += ["--h-far-um", str(h_far),
                "--far-grade-distance-um", "2000"]
    print(f"[pad {pad:.0f}] running: {' '.join(cmd)}")
    subprocess.check_call(cmd)
    return npz


def plot_sweep(pads):
    fig, (ax_r, ax_t) = plt.subplots(2, 1, figsize=(10, 9))

    # Pick snapshot time using the smallest-pad run's near-probe LSA
    # peak — same snap t_index for all pads since LSA is identical.
    smallest = SCEN_DIR / f"trace_pad{int(min(pads))}.npz"
    z = np.load(smallest)
    radii = np.linalg.norm(z["probes_um"][:, :2], axis=1)
    order = np.argsort(radii)
    near_i = int(order[0])
    far_i = int(order[-1])
    snap_idx = int(np.argmax(z["v_e_lsa_uV"][:, near_i]))
    t_snap = float(z["t_ms"][snap_idx])
    r_sorted = radii[order]

    # LSA reference (same for all pads).
    ax_r.plot(r_sorted, z["v_e_lsa_uV"][snap_idx, order], "o-",
              color="k", lw=2.0, label="LSA (reference)")

    cmap = plt.get_cmap("viridis")
    for i, pad in enumerate(sorted(pads)):
        npz = SCEN_DIR / f"trace_pad{int(pad)}.npz"
        if not npz.is_file():
            print(f"  missing {npz}, skip")
            continue
        d = np.load(npz)
        c = cmap(0.15 + 0.7 * i / max(1, len(pads) - 1))
        ax_r.plot(r_sorted, d["v_e_fem_uV"][snap_idx, order], "s--",
                  color=c, lw=1.4, alpha=0.9,
                  label=f"FEM ecs_pad={int(pad)} µm")
        # Far-probe time trace per pad.
        ax_t.plot(d["t_ms"], d["v_e_fem_uV"][:, far_i],
                  color=c, lw=1.4, label=f"FEM pad={int(pad)} µm")

    ax_t.plot(z["t_ms"], z["v_e_lsa_uV"][:, far_i],
              color="k", lw=2.0, label="LSA")

    ax_r.set_xscale("log")
    ax_r.set_xlabel("radial distance from cable axis (µm)")
    ax_r.set_ylabel(f"V_e at t = {t_snap:.2f} ms (µV)")
    ax_r.set_title("V_e(r) snapshot — convergence to LSA as box grows")
    ax_r.grid(alpha=0.3, which="both")
    ax_r.legend(loc="best", fontsize=9)

    ax_t.set_xlabel("t (ms)")
    ax_t.set_ylabel(f"V_e (µV) at outermost probe (r={radii[far_i]:.0f} µm)")
    ax_t.set_title("V_e(t) at far probe — same convergence in time")
    ax_t.grid(alpha=0.3)
    ax_t.legend(loc="best", fontsize=9)

    out = SCEN_DIR / "pad_sweep.png"
    plt.tight_layout()
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Quick numeric summary.
    print("\nFEM/LSA ratio at snap t per probe & pad:")
    print(f"  {'r (µm)':>8}", end="")
    for pad in sorted(pads):
        print(f"  {'pad' + str(int(pad)):>10}", end="")
    print()
    for idx in order:
        print(f"  {radii[idx]:8.1f}", end="")
        lsa_val = z["v_e_lsa_uV"][snap_idx, idx]
        for pad in sorted(pads):
            npz = SCEN_DIR / f"trace_pad{int(pad)}.npz"
            if not npz.is_file():
                print(f"  {'-':>10}", end=""); continue
            d = np.load(npz)
            fem_val = d["v_e_fem_uV"][snap_idx, idx]
            ratio = fem_val / lsa_val if abs(lsa_val) > 1e-3 else float("nan")
            print(f"  {ratio:10.3f}", end="")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pads", type=float, nargs="+",
                   default=[1500.0, 4000.0, 8000.0])
    p.add_argument("--replot", action="store_true",
                   help="skip simulations; just rebuild the sweep plot")
    args = p.parse_args()

    if not args.replot:
        for pad in args.pads:
            run_one(pad)
    plot_sweep(args.pads)


if __name__ == "__main__":
    main()
