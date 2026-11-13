# 2D ablator solver — scoped kickoff

Primer for a chat dedicated to exploring a 2D version of Martin's heat-shield
solver. Paste or point at this instead of loading the whole `CLAUDE.md`.
Written 2026-09-24.

---

## Don't re-derive these

**The 1D solver.** Martin's ~3,200-line F77 charring-ablator code, ported to
Python in `LAUMeltFlow/heat_python/` and validated: case aw1 bit-exact (~1e-8 K),
case aw2_tc21 to 0.064 K max across all 23 output columns. aw2_tc22 ships
byte-identical to tc21, so only one independent aw2 case exists.

**Geometry as shipped.** Slab 50 mm (`CASE_L0 = 5.0D-2`). Uniform mesh,
`dx = l0/n` (`domain.py:62-66`) — stretched grid exists in the Fortran but is
used by no example and was never ported. aw1 = 100 cells, aw2 = 500 cells.
`CASE_M = 0` in all three examples, so `dA = 1` and `dV = dx` everywhere and the
cylindrical/spherical generality is never exercised.

**Physics.** 3 species from `solid.dat` (sp1 light volatiles, sp2 the main resin
that leaves the char, sp3 inert carbon fiber — the 5 in GNN checkpoints is a
padded channel layout, not physics). The solver advances ENTHALPY and recovers T
by Newton inversion, so there is no `cp` on the LHS. Gas is Darcy flow solved
tridiagonally. The aero BC carries B-prime, blowing correction and wall enthalpy.

**The surrogate.** `HeatConductanceMPGNN`, bounded conductance update with a
discrete maximum principle, anchored conduction rate, real Arrhenius kinetics
under `no_grad`. Best model 18.2 K held-out on aw2. ~10x faster than the Python
solver, ~2x vs compiled Fortran.

---

## What "2D" means here

**Depth x arc length along the surface.** The second axis runs along the
vehicle's skin, the direction you would travel walking across the outside of the
heat shield. The domain becomes a cross-section slice: 50 mm deep by some lateral
extent. NOT depth-by-time — the existing space-time rollout plots are 1D space
plus time and are already what a 1D transient solver produces.

The physics this buys is **lateral conduction**. The forcing goes from `q(t)`,
one number per timestep at one face, to `q(s,t)`, a flux that varies along the
surface. On a capsule the stagnation point takes peak heating and it falls off
toward the shoulder, so the hot spot bleeds sideways into cooler material. A
stack of independent 1D columns cannot represent that.

**Two other defensible second axes.** (a) Radial, for an axisymmetric capsule:
domain is (depth, radius) and axisymmetry gives the full 3D body free. Note
`CASE_M = 1, 2` already reinterpret the depth axis as a radius, but that is still
1D. (b) Across a tile joint: depth by lateral distance spanning a seam. Narrower,
but it is the case where 1D is unambiguously wrong, so the 2D-vs-1D comparison is
sharp.

**Free verification case.** Make `q(s,t)` uniform in `s` and the 2D solution must
reproduce the 1D answer column-for-column. This is the closest available
substitute for the role the Fortran played in phase 1.

---

## What actually has to change in the code

Reusable as-is, because it is all pointwise per cell: `materials.py`,
`pyrolysis.py`, `aero.py`, most of `gas.py`'s property lookups, `io_files.py`,
`case.py`. The chemistry and the property tables do not care about dimensionality.

Must be rewritten:

- **`domain.py`** — a line of cells becomes a 2D mesh. Structured quads or
  unstructured triangles (see decisions below).
- **The conduction stencil in `solver.py`** — "left face minus right face"
  becomes a sum over all faces of a cell, each with its own area and outward
  normal. The `dA = 1`, `dV = dx` shortcut disappears.
- **The gas solve** — `TRIDIG` no longer applies. Darcy in 2D is a sparse linear
  system. This is the largest single piece of work.
- **The aero BC** — applies along a boundary CURVE with a local normal and a
  spatially varying incident flux, instead of one number at one face.
- **Cost** — 500 cells becomes 1e4 or more, and the timestep is still CFL-bound.

---

## Decisions to make early

1. **Structured quads or unstructured triangles?** Structured is far less work,
   keeps a lattice, and still gives you real lateral conduction. Unstructured is
   what real TPS geometry needs AND is the first case in this whole program where
   a GNN is not substitutable by a CNN (see the architecture honesty block in
   `CLAUDE.md`). Pick deliberately — this choice decides whether the 2D work
   strengthens the architecture argument or is neutral to it.
2. **Does the gas have to be 2D at all?** A staged path exists: 2D conduction and
   pyrolysis, with gas still solved 1D per column. That defers the sparse solve
   and still captures lateral heat spreading. Much cheaper first milestone.
3. **Extend `heat_python/` or write fresh?** Extending keeps the validated
   property/chemistry code. Writing fresh risks silently diverging from the
   reference physics.
4. **How is it verified with no Fortran ground truth?** Uniform-forcing reduction
   to 1D, a manufactured solution, and a global energy balance. Say plainly that
   this is weaker than phase 1, where a trusted reference existed.

---

## Why this is worth doing

The 1D solver is cheap. Fortran runs aw2 in about 3 minutes, which is why the
honest speedup number is only ~2x and why the surrogate's value proposition is
thin. 2D is where FEA cost becomes real and a surrogate starts to pay. It is also
where the graph formulation stops being stylistic. Production ablation codes are
multi-dimensional — the Amar 2016 CHAR paper already in `heat_gnn_refs.bib` is
one.

## Honest risks

- **This is a solver-writing project, not an ML project.** Writing and verifying
  a 2D pyrolysis solver is a months-scale job, not a weeks-scale one.
- **No reference solution exists.** Phase 1's credibility rested entirely on
  bit-exact agreement with Martin's Fortran. Nothing plays that role in 2D.
- **Scope collision.** EE699 (the IPM motor GNN) and the thesis writing are both
  live. Three fronts is probably one too many.

---

## Running the existing 1D code

```bash
# Fortran reference (from WSL Ubuntu):
wsl bash -c "cd /mnt/c/Users/glgo230/Projects/LAUMeltFlow/heat_2026-04-11_1837 && make heat"
wsl bash -c "cd /mnt/c/.../heat_2026-04-11_1837/examples/aw1 && ../../heat"
wsl bash -c "cd /mnt/c/.../heat_2026-04-11_1837/examples/aw2_tc21 && ../../heat gas"

# Python port (from Windows shell, repo root):
python -m heat_python.solver                          # aw1, gas off
python -m heat_python.solver --case-dir .../aw2_tc21 --gas
python -m heat_python.validate --fortran <f>/con.out --python <p>/python_out/con.out
```

**`lgas` gotcha.** `lgas` mirrors the Fortran `-g` flag and is NOT
`case.gas_energy` (both cases set that True). Default is gas ON only if
`use_aero_bcn`, i.e. aw2 yes, aw1 no. The reference outputs were generated aw1
without `-g` and aw2 with gas. Getting this wrong produces a fake 470 K
"conduction drift."

**Note on paths.** Give full paths starting `LAUMeltFlow/...`, and flag anything
outside the repo with its absolute path.
