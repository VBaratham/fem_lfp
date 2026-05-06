"""Side-by-side V_e(r) summary across the three test cells.

Each cell's snapshot V_e(r) at a representative time point,
overlayed: LSA reference (solid blue) vs body-fitted FEM (dashed red).

Reads from the canonical trace.npz for each scenario.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent

CELLS = [
    ("Cylinder (HH cable, pad=8000)",
     ROOT / "scenarios" / "cylinder" / "trace_pad8000.npz"),
    ("M&S j7 (body-fitted, cell-wide redirect)",
     ROOT / "scenarios" / "ms_j7" / "trace.npz"),
    ("Hay 2011 L5 PC (body-fitted)",
     ROOT / "scenarios" / "bbp" / "trace.npz"),
]


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (title, npz_path) in zip(axes, CELLS):
        if not npz_path.is_file():
            ax.text(0.5, 0.5, f"missing\n{npz_path.name}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=10)
            continue
        z = np.load(npz_path)
        t = z["t_ms"]
        probes = z["probes_um"]
        radii = np.linalg.norm(probes[:, :2], axis=1)
        order = np.argsort(radii)
        v_e_lsa = z["v_e_lsa_uV"]
        v_e_fem = z["v_e_fem_uV"]
        # Snapshot time = peak |LSA| at the near probe
        near_i = int(order[0])
        snap_idx = int(np.argmax(np.abs(v_e_lsa[:, near_i])))
        t_snap = float(t[snap_idx])
        r_sorted = radii[order]
        finite = ~np.isnan(v_e_fem[snap_idx, order])
        ax.plot(r_sorted, v_e_lsa[snap_idx, order], "o-",
                color="C0", lw=1.5, label="LSA")
        ax.plot(r_sorted[finite], v_e_fem[snap_idx, order][finite],
                "s--", color="C3", lw=1.5, label="FEM")
        ax.set_xscale("log")
        ax.set_xlabel("radial distance (µm)")
        ax.set_ylabel(f"V_e at t={t_snap:.2f} ms (µV)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9, loc="best")
        ax.axhline(0, color="gray", lw=0.5)

    plt.suptitle("fem_lfp: LSA vs ECS-only FEM across three test cells",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = ROOT / "three_cell_summary.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
