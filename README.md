# fem_lfp

Hybrid LFP simulator. NEURON solves the 1D cable equation for intracellular
dynamics (V_m, gating, per-segment transmembrane current). A 3D FEM Poisson
solve in the **extracellular space only** then computes V_e from those
membrane currents — used as Neumann boundary data on the cell surface.

The aim is to compare against the line-source approximation (LSA) in the
near and far field, where LSA's infinite-homogeneous-medium assumption
breaks down (probe geometry, anisotropic σ, finite tissue boundaries).

This is a middle ground between
[fem_neuron](../fem_neuron) (full self-consistent EMI / KNP-EMI inside +
outside) and the standard LSA postprocessor (analytical line integral in
infinite homogeneous medium). Cheaper than EMI; more geometrically faithful
than LSA.

## Math

ECS only. The transmembrane current i_mem(x,t) (computed by NEURON's
cable equation) enters as a Neumann boundary on the cell surface Γ_m.

```
∇·(σ ∇φ_e) = 0        in Ω_e
σ ∂φ_e/∂n_e = i_mem    on Γ_m    (n_e outward from ECS = into the cell;
                                   i_mem outward-positive ⇒ current
                                   flowing INTO the ECS at the membrane)
φ_e = 0                on Γ_outer (bounding box, Dirichlet far field)
```

Weak form, per timestep:

```
∫ σ ∇φ · ∇v dx = Σ_k (I_k(t) / A_k) ∫_{Γ_m,k} v ds
```

where Γ_m,k is the patch of cell surface owned by NEURON segment k, A_k
its area, and I_k(t) the per-segment transmembrane current in nA. The
bilinear form is time-independent — LU-factor once, refactor only when
geometry changes; per-step cost is RHS reassembly + back-substitution.

## Install

```bash
conda activate fem_neuron-env   # reuse the fem_neuron environment
pip install -e .
```

## Run

```bash
# 1. Single-cylinder HH cable: clean FEM-vs-LSA demo
python scripts/cylinder_compare.py
python scripts/cylinder_pad_sweep.py    # box-size convergence study

# 2. Mainen & Sejnowski j7 reconstruction (uses fem_neuron's branched
#    mesh pipeline; reuses fem_neuron/comparisons/ms_j7/cells/).
python scripts/ms_j7_compare.py
python scripts/ms_j7_compare.py --body-fitted   # AMS+TetGen, cleaner

# 3. Hay 2011 BBP-style L5 PC (download from ModelDB 139653 once,
#    nrnivmodl in cells/L5bPCmodelsEH/mod first).
python scripts/bbp_compare.py --body-fitted
```

## Status

**Cylinder (200 µm × 5 µm HH cable):** clean LSA-vs-FEM agreement.
At near probes (r ≤ 100 µm) FEM matches LSA to ≤ 1-2 %; at far probes
the FEM/LSA ratio is dominated by Dirichlet wall pull-down, which
converges out as ``ecs_pad_um`` grows (1500 → 4000 → 8000 µm sweep
shows the ratio at r=800 µm rising 0.36 → 0.79 → 0.82). See
``scenarios/cylinder/pad_sweep.png``.

**M&S j7 (93 sections, 199 segments, 3-AP train):** body-fitted mesh
(AMS+TetGen) + cell-wide redirect of empty-bin currents gives a
clean qualitative match — sign at all probes, mid-field inversion
gone, far-field at r=800 within 60% of LSA. Near-field FEM/LSA at
r=20 = 1.55× — a residual offset from 4 NEURON segments (3 in axon
hillock, 1 small dendrite) that get redistributed cell-wide because
AMS's alpha-wrap merged the proximal hillock surface into the soma.
The redirected currents land at the soma's segment closest to the
original NEURON position (~5 µm displacement); residual amp
amplification is the signature of that displacement.

**Hay 2011 BBP-style L5 PC (196 sections, 642 segments, 2-AP burst):**
Best LSA-vs-FEM match in the suite — **0 empty bins** (Hay's
trimmed axon doesn't have the M&S thin-hillock geometry).
FEM/LSA tracks cleanly at all 12 probes: 1.75× at r=20 µm,
0.88× at r=77 µm, 1.20× at r=800 µm. Same waveform shape, same
sign across the entire 30 ms time window. The systematic 1.5–2×
offset at near probes is the actual LSA-vs-FEM physics gap —
FEM applies currents on the cell surface, LSA on a 1D line at the
center, and probes near the cell see the geometric difference.

## Layout

```
src/fem_lfp/
  lsa.py             # closed-form line-source approximation (numpy-only)
  neuron_sim.py      # NEURON helpers: i_membrane capture, pt3d slicing,
                     # SWC export
  mesh_cylinder.py   # ECS-only mesh for a single cylindrical cell
  mesh_branched.py   # ECS-only mesh for branched cells via fem_neuron's
                     # OCC-fuse branched mesher + ECS submesh extraction
  mesh_body_fitted.py# ECS-only mesh via fem_neuron's body_fitted (AMS PLY
                     # + TetGen) — single watertight surface, no OCC fuse
                     # quality issues; mesh-cached under ~/.cache/fem_lfp_meshes/
  fem.py             # ECS Poisson solver: bilinear-form-once, RHS-per-step,
                     # CableSegmentation + BranchedSegmentation; cell-wide
                     # redirect of empty-bin currents to nearest non-empty
                     # 3D segment center
scenarios/
  cylinder/scenario.py
  ms_j7/scenario.py
  bbp/scenario.py    # Hay et al. 2011 L5 PC, ModelDB 139653
scripts/
  cylinder_compare.py
  cylinder_pad_sweep.py
  ms_j7_compare.py
  ms_j7_diagnostic.py        # single-source / uniform-source / actual-i_mem
  ms_j7_lsa_simplified.py    # LSA full vs LSA simplified vs FEM
  bbp_compare.py
third_party/
  Alpha_Mesh_Swc_patched/    # local AMS clone with two patches:
                             # (1) skip TetGen self-intersection probe
                             #     (was hanging 25+ min on j7 alpha-wrap)
                             # (2) honor user --min_faces (upstream
                             #     silently overrode it with dfaces/2)
```
