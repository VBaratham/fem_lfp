"""Run the HH cylinder, compare FEM-LFP vs LSA at log-spaced radial probes.

Demonstrates the public interface: build a NEURON cell, wrap it in an
``ExtracellularModel``, run, solve. Everything mesh/FEM-related is the
model's job.

Outputs
-------
scenarios/cylinder/trace_<suffix>.npz   — saved ExtracellularResult
scenarios/cylinder/overlay_<suffix>.png — V_m + V_e(t) + V_e(r) overlay

Usage
-----
    python scripts/cylinder_compare.py            # full run
    python scripts/cylinder_compare.py --replot   # regen plot from npz
    python scripts/cylinder_compare.py --neuron-only  # LSA only, skip FEM
    python scripts/cylinder_compare.py --ecs-pad-um 4000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCEN_DIR = ROOT / "scenarios" / "cylinder"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SCEN_DIR))


def run_full(
    ecs_pad_um: float | None = None,
    h_far_um: float | None = None,
    far_grade_distance_um: float | None = None,
    label: str | None = None,
) -> None:
    import scenario as sc
    from fem_lfp import ExtracellularModel

    overrides = dict(sc.MESH)
    if ecs_pad_um is not None:
        overrides["ecs_pad_um"] = float(ecs_pad_um)
    if h_far_um is not None:
        overrides["h_far_um"] = float(h_far_um)
    if far_grade_distance_um is not None:
        overrides["far_grade_distance_um"] = float(far_grade_distance_um)
    suffix = label if label is not None else f"pad{int(overrides['ecs_pad_um'])}"

    # Build the cell, then arm recording BEFORE the run by constructing
    # the model. mesh="cylinder" because this is a single z-aligned cable.
    sections = sc.build_cell()
    model = ExtracellularModel(
        sections, sc.PROBES_UM,
        mesh="cylinder", sigma=0.3, work_dir=SCEN_DIR, **overrides,
    )
    print(f"[NEURON] running cable equation ... (label={suffix})")
    sc.run()

    result = model.solve()
    print("[timings] " + ", ".join(
        f"{k}={v:.2f}s" for k, v in result.timings_s.items()))

    out_npz = SCEN_DIR / f"trace_{suffix}.npz"
    result.save(out_npz)
    print(f"saved {out_npz}")
    result.plot(SCEN_DIR / f"overlay_{suffix}.png",
                title=f"{sc.TITLE}  ({suffix})")


def run_neuron_only() -> None:
    import scenario as sc
    from fem_lfp import ExtracellularModel

    sections = sc.build_cell()
    model = ExtracellularModel(sections, sc.PROBES_UM, mesh="cylinder")
    sc.run()
    result = model.line_source()
    v_m_peak = max(v.max() for v in result.v_m_mV.values())
    print(f"V_m peak: {v_m_peak:.2f} mV")
    print(f"V_e(LSA) near probe peak: "
          f"{result.v_e_lsa_uV[:, 0].max():.1f} µV")


def replot(suffix: str) -> None:
    from fem_lfp import ExtracellularResult
    npz = SCEN_DIR / f"trace_{suffix}.npz"
    if not npz.is_file():
        legacy = SCEN_DIR / "trace.npz"
        npz = legacy if legacy.is_file() else npz
    result = ExtracellularResult.load(npz)
    result.plot(SCEN_DIR / f"overlay_{suffix}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replot", action="store_true",
                   help="just regenerate overlay png from existing trace npz")
    p.add_argument("--neuron-only", action="store_true",
                   help="run NEURON + LSA only, skip FEM")
    p.add_argument("--ecs-pad-um", type=float, default=None)
    p.add_argument("--h-far-um", type=float, default=None)
    p.add_argument("--far-grade-distance-um", type=float, default=None)
    p.add_argument("--label", type=str, default=None,
                   help="custom output-filename label (default: padXXXX)")
    args = p.parse_args()

    if args.replot:
        suffix = (args.label if args.label is not None
                  else f"pad{int(args.ecs_pad_um)}" if args.ecs_pad_um
                  else "pad1500")
        replot(suffix)
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
