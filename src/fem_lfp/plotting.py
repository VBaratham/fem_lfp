"""Standard FEM-vs-LSA overlay figure for an :class:`ExtracellularResult`.

One figure, reused by every scenario: the soma V_m on top, V_e(t) at a
near / mid / far probe, and a V_e-vs-radius snapshot at the moment the
near-probe signal peaks. Kept deliberately generic so any result (cylinder,
branched, body-fitted) plots the same way.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402


def overlay_fem_vs_lsa(
    result,
    out_path: str | Path,
    *,
    title: str | None = None,
    dpi: int = 140,
) -> Path:
    """Render and save the overlay figure; return the output path."""
    out_path = Path(out_path)
    t = result.t_ms
    probes = result.probes_um
    v_fem = result.v_e_fem_uV
    v_lsa = result.v_e_lsa_uV

    radii = result.probe_radii_um()
    order = np.argsort(radii)
    near_i, mid_i, far_i = (
        int(order[0]), int(order[len(order) // 2]), int(order[-1]),
    )

    # Snapshot time = when |V_e| at the near probe peaks (prefer LSA, which
    # is always present; fall back to FEM).
    ref = v_lsa if v_lsa is not None else v_fem
    snap_idx = int(np.argmax(np.abs(ref[:, near_i])))
    t_snap = float(t[snap_idx])

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[2.5, 2.5, 2.5],
                          hspace=0.35, wspace=0.3)
    ax_vm = fig.add_subplot(gs[0, :])
    ax_near = fig.add_subplot(gs[1, 0], sharex=ax_vm)
    ax_mid = fig.add_subplot(gs[1, 1], sharex=ax_vm)
    ax_far = fig.add_subplot(gs[1, 2], sharex=ax_vm)
    ax_radial = fig.add_subplot(gs[2, :])

    # --- V_m ---
    if result.v_m_mV:
        for label, trace in result.v_m_mV.items():
            ax_vm.plot(t, trace, lw=1.5, label=f"V_m {label}")
        ax_vm.legend(fontsize=9)
    ax_vm.set_ylabel("V_m (mV)")
    ax_vm.set_xlabel("t (ms)")
    ax_vm.grid(alpha=0.3)
    ax_vm.axvline(t_snap, color="C2", lw=0.8, ls=":")

    # --- V_e(t) at near / mid / far ---
    for ax, idx in zip((ax_near, ax_mid, ax_far), (near_i, mid_i, far_i)):
        if v_lsa is not None:
            ax.plot(t, v_lsa[:, idx], color="C0", lw=1.4, label="LSA")
        if v_fem is not None:
            ax.plot(t, v_fem[:, idx], color="C3", lw=1.4, ls="--", label="FEM")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(f"V_e (µV)  r={radii[idx]:.0f} µm", fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(t_snap, color="C2", lw=0.8, ls=":")

    # --- V_e(r) snapshot ---
    r_sorted = radii[order]
    if v_lsa is not None:
        ax_radial.plot(r_sorted, v_lsa[snap_idx, order], "o-",
                       color="C0", lw=1.5, label="LSA")
    if v_fem is not None:
        ax_radial.plot(r_sorted, v_fem[snap_idx, order], "s--",
                       color="C3", lw=1.5, label="FEM")
    ax_radial.set_xscale("log")
    ax_radial.set_xlabel("radial distance from z axis (µm)")
    ax_radial.set_ylabel(f"V_e at t={t_snap:.2f} ms (µV)")
    ax_radial.legend(fontsize=9)
    ax_radial.grid(alpha=0.3, which="both")

    if title is None:
        title = f"Extracellular potential — FEM vs LSA ({result.mesher} mesh)"
    plt.suptitle(title, y=0.995, fontsize=12)
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # Console summary — same table the scenarios used to print by hand.
    if v_fem is not None and v_lsa is not None:
        print(f"\nV_e summary at t={t_snap:.2f} ms:")
        print(f"  {'r (µm)':>8}  {'LSA (µV)':>10}  {'FEM (µV)':>10}  "
              f"{'FEM/LSA':>8}")
        for idx in order:
            lsa_val = v_lsa[snap_idx, idx]
            fem_val = v_fem[snap_idx, idx]
            ratio = fem_val / lsa_val if abs(lsa_val) > 1e-3 else float("nan")
            print(f"  {radii[idx]:8.1f}  {lsa_val:10.3f}  {fem_val:10.3f}  "
                  f"{ratio:8.3f}")
    print(f"saved {out_path}")
    return out_path
