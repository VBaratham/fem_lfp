"""Hay et al. 2011 BBP-style L5 PC: NEURON + LSA + ECS-FEM overlay.

Same pipeline as ms_j7_compare.py — only the scenario import + output
dir differ. ``--body-fitted`` uses AMS + TetGen instead of fem_neuron's
branched OCC mesher. The body-fitted mesher auto-anchors TetGen's
interior point at the soma center (from the captured geometry), which
the plain COM heuristic gets wrong on this cell's long apical dendrite.

    python scripts/bbp_compare.py                 # branched OCC mesh
    python scripts/bbp_compare.py --body-fitted   # AMS + TetGen mesh
    python scripts/bbp_compare.py --replot        # regen plot from npz
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "scenarios" / "bbp"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCEN_DIR))


def run_full(body_fitted: bool = False) -> None:
    import scenario as sc
    from fem_lfp import ExtracellularModel

    print("[NEURON] loading Hay L5 PC + running ...")
    t0 = time.time()
    nrun, section_info = sc.run_neuron()
    print(f"[NEURON] {len(section_info)} sections, "
          f"{nrun.imem_nA.shape[0]} segs × {nrun.t_ms.size} steps, "
          f"V_m peak {nrun.rec_v_mV['soma(0.5)'].max():.1f} mV "
          f"in {time.time() - t0:.1f}s")

    mesh = "body_fitted" if body_fitted else "branched"
    mesh_kwargs = sc.MESH_BODY_FITTED if body_fitted else sc.MESH

    sections = None
    if body_fitted:
        from neuron import h
        sections = list(h.allsec())

    model = ExtracellularModel.from_run(
        nrun, section_info, sc.PROBES_UM,
        sections=sections, mesh=mesh, sigma=0.3,
        work_dir=SCEN_DIR, **mesh_kwargs,
    )
    print(f"[FEM]    mesh={mesh} ...")
    result = model.solve()
    print("[timings] " + ", ".join(
        f"{k}={v:.2f}s" for k, v in result.timings_s.items()))

    result.save(SCEN_DIR / "trace.npz")
    print(f"saved {SCEN_DIR / 'trace.npz'}")
    result.plot(SCEN_DIR / "overlay.png",
                title=f"Hay 2011 L5 PC — FEM vs LSA ({mesh})")


def replot() -> None:
    from fem_lfp import ExtracellularResult
    result = ExtracellularResult.load(SCEN_DIR / "trace.npz")
    result.plot(SCEN_DIR / "overlay.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true")
    p.add_argument("--body-fitted", action="store_true",
                   help="use AMS + TetGen body-fitted mesh")
    args = p.parse_args()
    if args.replot:
        replot()
    else:
        run_full(body_fitted=args.body_fitted)


if __name__ == "__main__":
    main()
