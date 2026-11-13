# GNN surrogates for physics solvers — summary

Grey Goodwin, University of Kentucky. Current as of November 2026. Written as a
standing summary of the work in this repository. Numbers here supersede anything
in older drafts or in superseded blocks of `CLAUDE.md`.

## The question

Physics solvers used in design loops are expensive, so each design iteration is
slow. Neural surrogates promise to replace them, and on easy problems they do.
The question this work asks is sharper: **how much of the known physics has to be
built into a surrogate before it can handle a genuinely stiff, coupled problem?**
The answer turns out to be "most of it," and the useful result is understanding
exactly which parts cannot be learned from data.

## The program

Each step established one thing, and the heat shield is the synthesis.

| Study | Established |
|---|---|
| Bead chain (MPGNN) | A message-passing network trained on one chain length generalizes to others. The weight sharing is what buys it. |
| 1D Sod shock tube | On a fixed two-cell stencil an MLP suffices; the graph structure earns nothing. *Largely group work, not mine.* |
| 2D Sod shock tube | Once the stencil widens, the MPGNN beats the MLP. *Largely group work, not mine.* |
| **1D heat shield** | **The main contribution. Below.** |
| 2D heat shield | In progress. |

## The heat-shield result

The reference solver is a charring-ablator code: heat conduction, Arrhenius
pyrolysis of the resin, and Darcy flow of pyrolysis gas through the porous char,
all coupled, with an aerothermal surface boundary condition.

Two cases, and the gap between them is the finding.

- **Gas off (conduction and pyrolysis only).** A mostly-learned per-cell
  message-passing GNN reproduces a held-out 60 s heating scenario to **7.6 K**
  mean error and tracks the char front to a fraction of a millimeter.
- **Gas and aerothermal coupling on.** The same approach **diverges**, to 141,000 K
  for the direct model. More data, wider stencils, noise injection, multi-step
  training, ensembling and adaptive time-stepping were each tried and each failed.
- **Structuring the physics in fixes it.** Anchoring the conduction rate to its
  measured value, computing the solver's exact Arrhenius kinetics inside the
  forward pass, and constraining the update to a bounded convex combination that
  satisfies a discrete maximum principle gives **25.3 K**, stable by construction.
- **Relaxing the anchor and focusing training on the transient gives 18.2 K.**
  Letting the conduction rate learn around its physical anchor while upweighting
  the loss on the sharp flux transient reaches the headline result. Neither move
  works alone: the loss weighting has no authority until the anchor is relaxed.

So a stiff coupled problem needs its known physics built in, where the easier case
did not. That contrast, measured on the same solver and the same mesh, is the
contribution.

## What "physics-informed" means here

The physics is in the **architecture**, not in a loss residual. This is a
physics-informed graph neural network, not a PINN in the Raissi sense: there is no
PDE residual term, and training is supervised on solver trajectories. The lineage
is MeshGraphNets. Conservation and boundedness hold by construction rather than by
penalty, which is why the surrogate cannot blow up at any timestep.

## Speed

About **2x** faster than the compiled reference solver and about **10x** faster
than an equivalent Python implementation, for a 120 s trajectory. The honest
caveat is scale: the mesh is 502 cells, so a GPU barely helps and the run is
launch-bound. Batching many design cases at once is the real lever and is untested.

## Limits, stated plainly

The mesh is uniform, so a constrained convolutional network would do the same job
here; the graph formulation is chosen for where this is going, not for what it
buys today. Only 10 of 110,299 allocated parameters receive gradient and 5 move
off their initialization. The surrogate runs about 42 K hot in the near-surface
band at peak heating. Gas mass conservation is not enforced.

## Current work

Extending to a **2D axisymmetric capsule**, using Dr. Alexandre Martin's KITTEN
Core solver on the KRUPS/KREPE-2 re-entry geometry: a 539 s trajectory, 40 x 90
mesh, 25.4 mm TACOT shell, with charring, Darcy flow, radiation, blowing and
thermochemical surface recession. Two things make it the real test. The surface
recedes, so the mesh moves and the domain shrinks during a rollout, which the 1D
problem never did. And 2D is where solver cost becomes large enough that a
surrogate is worth having.

## Manuscripts

In `docs/heat_TPS/`, with status in that folder's `README.md`.

- `Goodwin_report` — finished and sent to Dr. Martin, the reference solver's author.
- `Goodwin_TPS` — journal manuscript, results current, reorganization pending.
- `Goodwin_chapter` — thesis chapter, early.
- `Goodwin_equations` — all 38 governing equations and which ones the surrogate
  computes, replaces, or never touches.

## Repository map

`heat_python/` the 1D solver and surrogate, with the model and training pipeline
in `heat_python/gnn/`. `docs/heat_TPS/` the manuscripts and figures. `whip/` the
bead-chain MPGNN and the symposium paper. `meltflow_gnn/` the Sod flux models.
`CLAUDE.md` the running project record, which is long and contains superseded
blocks; this file and `docs/heat_TPS/README.md` are the current account.
