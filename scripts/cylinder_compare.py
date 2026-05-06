"""Run the HH cylinder, compare FEM-LFP vs LSA at log-spaced radial probes.

Outputs
-------
scenarios/cylinder/trace.npz   — t, V_m, i_mem, V_e_lsa, V_e_fem, probes
scenarios/cylinder/overlay.png — V_m (top), V_e(t) at near + far probes,
                                 V_e(r) snapshot at peak.

Usage
-----
    python scripts/cylinder_compare.py            # full run
    python scripts/cylinder_compare.py --replot   # just regen plot from npz
    python scripts/cylinder_compare.py --neuron-only  # skip FEM (debug)
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
SCEN_DIR = ROOT / "scenarios" / "cylinder"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCEN_DIR))


def _wall(s: float) -> str:
    if s >= 1.0:
        return f"{s:.2f}s"
    if s >= 1e-3:
        return f"{s * 1e3:.2f}ms"
    return f"{s * 1e6:.2f}µs"


def run_full(
    ecs_pad_um: float | None = None,
    h_far_um: float | None = None,
    far_grade_distance_um: float | None = None,
    label: str | None = None,
) -> None:
    import scenario as sc

    mesh_kwargs = dict(sc.MESH)
    if ecs_pad_um is not None:
        mesh_kwargs["ecs_pad_um"] = float(ecs_pad_um)
    if h_far_um is not None:
        mesh_kwargs["h_far_um"] = float(h_far_um)
    if far_grade_distance_um is not None:
        mesh_kwargs["far_grade_distance_um"] = float(far_grade_distance_um)
    pad = mesh_kwargs["ecs_pad_um"]
    suffix = label if label is not None else f"pad{int(pad)}"
    out_npz = SCEN_DIR / f"trace_{suffix}.npz"
    out_mesh = SCEN_DIR / f"_mesh_{suffix}"

    # 1) NEURON
    t0 = time.time()
    print(f"[NEURON] running cable equation ... (label={suffix})")
    nrun = sc.run_neuron()
    t_neuron = time.time() - t0
    print(f"[NEURON] done in {_wall(t_neuron)}; "
          f"{nrun.imem_nA.shape[0]} segments × {nrun.t_ms.size} timesteps")

    # 2) LSA
    from fem_lfp.lsa import line_source_v_e
    t0 = time.time()
    v_e_lsa = line_source_v_e(
        sc.PROBES_UM, nrun.p1_um, nrun.p2_um, nrun.imem_nA,
        sigma_S_per_m=0.3,
    )  # (P, T) volts
    t_lsa = time.time() - t0
    print(f"[LSA]    done in {_wall(t_lsa)}")

    # 3) FEM
    print(f"[FEM]    building mesh (ecs_pad={pad:.0f} µm) ...")
    t0 = time.time()
    from fem_lfp.mesh_cylinder import (
        CylinderEcsSpec, build_cylinder_ecs_mesh,
        TAG_OUTER, TAG_MEMBRANE,
    )
    spec = CylinderEcsSpec(**mesh_kwargs)
    mesh, ct, ft = build_cylinder_ecs_mesh(spec, out_mesh)
    t_mesh = time.time() - t0
    print(f"[FEM]    mesh in {_wall(t_mesh)}; solving ...")

    from fem_lfp.fem import CableSegmentation, run_fem_lfp
    seg = CableSegmentation(n_seg=sc.NSEG, L_um=sc.L_UM, axis="z")
    t0 = time.time()
    v_e_fem_uV = run_fem_lfp(   # (T, P) µV
        mesh, ft, seg,
        imem_nA=nrun.imem_nA,
        probe_xyz_um=sc.PROBES_UM,
        sigma_S_per_m=0.3,
        outer_tag=TAG_OUTER, membrane_tag=TAG_MEMBRANE,
    )
    t_fem = time.time() - t0
    print(f"[FEM]    {nrun.t_ms.size} timesteps in {_wall(t_fem)} "
          f"({_wall(t_fem / nrun.t_ms.size)}/step)")

    # 4) save
    np.savez(
        out_npz,
        t_ms=nrun.t_ms,
        v_m_mV=nrun.rec_v_mV["soma(0.5)"],
        imem_nA=nrun.imem_nA,
        p1_um=nrun.p1_um, p2_um=nrun.p2_um,
        probes_um=sc.PROBES_UM,
        v_e_lsa_uV=(v_e_lsa.T * 1e6),     # (T, P) µV
        v_e_fem_uV=v_e_fem_uV,             # (T, P) µV
        ecs_pad_um=float(pad),
        wall_neuron=t_neuron, wall_lsa=t_lsa,
        wall_mesh=t_mesh, wall_fem=t_fem,
    )
    print(f"saved {out_npz}")
    plot_overlay(suffix=suffix)


def run_neuron_only() -> None:
    import scenario as sc
    from fem_lfp.lsa import line_source_v_e
    nrun = sc.run_neuron()
    v_e_lsa = line_source_v_e(
        sc.PROBES_UM, nrun.p1_um, nrun.p2_um, nrun.imem_nA,
    )
    print(f"V_m peak: {nrun.rec_v_mV['soma(0.5)'].max():.2f} mV")
    print(f"V_e(LSA) at r=PROBES_UM[0]: peak {v_e_lsa[0].max()*1e6:.1f} µV")


def plot_overlay(suffix: str = "pad4000") -> None:
    npz_path = SCEN_DIR / f"trace_{suffix}.npz"
    if not npz_path.is_file():
        # fall back to legacy single-trace name
        legacy = SCEN_DIR / "trace.npz"
        if legacy.is_file():
            npz_path = legacy
        else:
            raise FileNotFoundError(f"no trace npz at {npz_path} or {legacy}")
    npz = np.load(npz_path)
    t = npz["t_ms"]
    v_m = npz["v_m_mV"]
    imem = npz["imem_nA"]
    probes = npz["probes_um"]
    v_e_lsa = npz["v_e_lsa_uV"]   # (T, P)
    v_e_fem = npz["v_e_fem_uV"]   # (T, P)

    radii = np.linalg.norm(probes[:, :2], axis=1)
    order = np.argsort(radii)
    near_i = int(order[0])
    far_i = int(order[-1])
    mid_i = int(order[len(order) // 2])

    # Snapshot time = peak of V_e_LSA at the near probe (Na sink phase
    # of the AP makes V_e most positive close to the membrane).
    t_snap = float(t[int(np.argmax(v_e_lsa[:, near_i]))])
    snap_idx = int(np.argmin(np.abs(t - t_snap)))

    fig = plt.figure(figsize=(11, 11.5))
    gs = fig.add_gridspec(
        4, 3, height_ratios=[1, 2.5, 2.5, 2.5], hspace=0.35, wspace=0.30,
    )
    ax_imem = fig.add_subplot(gs[0, :])
    ax_vm = fig.add_subplot(gs[1, :], sharex=ax_imem)
    ax_near = fig.add_subplot(gs[2, 0], sharex=ax_imem)
    ax_mid = fig.add_subplot(gs[2, 1], sharex=ax_imem)
    ax_far = fig.add_subplot(gs[2, 2], sharex=ax_imem)
    ax_radial = fig.add_subplot(gs[3, :])

    # Total membrane current (sum across segments) — sanity / time
    # context.
    i_total = imem.sum(axis=0)
    ax_imem.plot(t, i_total, color="k", lw=1.0)
    ax_imem.set_ylabel("Σ i_mem\n(nA)", fontsize=9)
    ax_imem.tick_params(labelbottom=False)
    ax_imem.grid(alpha=0.3)
    ax_imem.axvline(t_snap, color="C2", lw=0.8, ls=":")

    # V_m
    ax_vm.plot(t, v_m, color="C0", lw=1.5, label="V_m at soma(0.5)")
    ax_vm.set_ylabel("V_m (mV)")
    ax_vm.set_xlabel("t (ms)")
    ax_vm.legend(loc="best", fontsize=9)
    ax_vm.grid(alpha=0.3)
    ax_vm.axvline(t_snap, color="C2", lw=0.8, ls=":")

    # V_e(t) at near / mid / far
    for ax, idx in zip([ax_near, ax_mid, ax_far], [near_i, mid_i, far_i]):
        r = radii[idx]
        ax.plot(t, v_e_lsa[:, idx], color="C0", lw=1.4, label="LSA")
        ax.plot(t, v_e_fem[:, idx], color="C3", lw=1.4, ls="--", label="FEM")
        ax.set_xlabel("t (ms)")
        ax.set_ylabel(f"V_e (µV)  r={r:.1f} µm", fontsize=9)
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
        ax.axvline(t_snap, color="C2", lw=0.8, ls=":")

    # V_e(r) snapshot
    r_sorted = radii[order]
    ax_radial.plot(r_sorted, v_e_lsa[snap_idx, order], "o-",
                   color="C0", lw=1.5, label="LSA")
    ax_radial.plot(r_sorted, v_e_fem[snap_idx, order], "s--",
                   color="C3", lw=1.5, label="FEM")
    ax_radial.set_xscale("log")
    ax_radial.set_xlabel("radial distance from cable axis (µm)")
    ax_radial.set_ylabel(f"V_e at t = {t_snap:g} ms (µV)")
    ax_radial.legend(loc="best", fontsize=9)
    ax_radial.grid(alpha=0.3, which="both")

    out = SCEN_DIR / f"overlay_{suffix}.png"
    plt.suptitle(
        f"HH cylinder — FEM vs LSA  (snap t={t_snap:.2f} ms, {suffix})",
        y=0.995, fontsize=12,
    )
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")

    # Quick summary numbers.
    print(f"\nV_e summary at snapshot t={t_snap:.2f} ms:")
    print(f"  {'r (µm)':>8}  {'LSA (µV)':>10}  {'FEM (µV)':>10}  "
          f"{'Δ (µV)':>10}  {'FEM/LSA':>8}")
    for idx in order:
        l = v_e_lsa[snap_idx, idx]
        f = v_e_fem[snap_idx, idx]
        ratio = f / l if abs(l) > 1e-3 else float("nan")
        print(f"  {radii[idx]:8.1f}  {l:10.2f}  {f:10.2f}  "
              f"{f-l:+10.2f}  {ratio:8.2f}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true",
                   help="just regenerate overlay png from existing trace npz")
    p.add_argument("--neuron-only", action="store_true",
                   help="run NEURON + LSA only, skip FEM")
    p.add_argument("--ecs-pad-um", type=float, default=None,
                   help="override scenario.MESH['ecs_pad_um']; trace + plot "
                        "are saved with a pad-tagged suffix")
    p.add_argument("--h-far-um", type=float, default=None,
                   help="override scenario.MESH['h_far_um'] (Stage-2 bulk "
                        "size); enables two-stage grading")
    p.add_argument("--far-grade-distance-um", type=float, default=None,
                   help="override scenario.MESH['far_grade_distance_um']")
    p.add_argument("--label", type=str, default=None,
                   help="custom label for output filenames "
                        "(default: padXXXX from ecs_pad_um)")
    args = p.parse_args()
    if args.replot:
        if args.label is not None:
            plot_overlay(suffix=args.label)
        elif args.ecs_pad_um is not None:
            plot_overlay(suffix=f"pad{int(args.ecs_pad_um)}")
        else:
            plot_overlay()
    elif args.neuron_only:
        run_neuron_only()
    else:
        run_full(
            ecs_pad_um=args.ecs_pad_um,
            h_far_um=args.h_far_um,
            far_grade_distance_um=args.far_grade_distance_um,
            label=args.label,
        )


if __name__ == "__main__":
    main()
