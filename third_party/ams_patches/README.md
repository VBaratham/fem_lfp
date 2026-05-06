# Alpha_Mesh_Swc patches

The body-fitted mesh path (`fem_lfp.mesh_body_fitted`) calls
[Alpha_Mesh_Swc][ams] (AMS) under the hood to produce a watertight
surface from a NEURON-derived SWC. AMS is GPL-3.0 — we depend on it
as an external CLI rather than vendoring it, in keeping with how
[fem_neuron][fn] already wraps it.

Two upstream bugs in AMS make body-fitted meshing unusable on dense
reconstructions like Mainen & Sejnowski's j7 cell. The patches in
this directory fix them.

## What the patches do

**`0001-skip-tetgen-watertight-probe.patch`** — `is_watertight()` calls
TetGen with `-dBENF` to probe self-intersections on the alpha-wrap
PLY. On M&S j7 (~1.7M-face alpha-wrap), that probe runs 25+ min and
frequently hangs. Skip the TetGen probe and trust `trimesh.is_watertight`
alone. TetGen volume meshing downstream catches any residual
self-intersections by failing the `.poly` tetrahedralization.

**`0002-honor-min-faces.patch`** — `_simplify_mesh()` had identical
`min_faces = int(dfaces/2)` in both branches of an `if/else`,
silently ignoring the `--min_faces` flag. The patched else branch
now honors the user value.

## Applying

Clone AMS yourself, then run `apply.sh`:

```bash
git clone https://github.com/AlexMcSD/Alpha_Mesh_Swc ~/Alpha_Mesh_Swc
bash third_party/ams_patches/apply.sh ~/Alpha_Mesh_Swc
export FEM_NEURON_AMS_ROOT=~/Alpha_Mesh_Swc
```

`fem_neuron`'s body_fitted reads `FEM_NEURON_AMS_ROOT`; our
`mesh_body_fitted` defers to that env var, falls back to a few default
sibling paths (`~/Alpha_Mesh_Swc`, `~/code/Alpha_Mesh_Swc`,
`~/claude/Alpha_Mesh_Swc`) otherwise.

[ams]: https://github.com/AlexMcSD/Alpha_Mesh_Swc
[fn]: ../../README.md
