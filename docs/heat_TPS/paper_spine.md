# Paper writing spine — answer the questions, then assemble

How to use this: answer each question in fragments/bullets, in your own words.
Don't write polished sentences yet. When every question has an answer, you have
the whole paper; a second pass turns the fragments into prose. The `(you have:)`
notes point to the result or number so you're never staring at a blank.

---

## Abstract (write this LAST, ~150 words)
One question: *what did you do and what did you find?*
- What is the problem in one sentence? (TPS design needs many expensive solver runs)
- What did you build? (validated Python port of a Fortran heat-shield solver + a GNN surrogate)
- What worked? *(you have: gas-off surrogate, 7.6 K rollout; port matches Fortran to machine precision / 0.064 K)*
- What didn't, stated honestly? *(you have: gas case = flux-form removes the catastrophic divergence but no variant is physically faithful; open problem)*
- Why does anyone care? (one line: surrogate method for a coupled multi-physics engineering solver)

## 1. Introduction
Question: *why this problem, why now, and what exactly did you contribute?*
- What is a TPS and what does designing one require? (material + thickness, keep structure cool, minimize mass)
- What is the governing physics, in one sentence? (conduction + pyrolysis + gas percolation + surface energy balance)
- Why is a surrogate the right response? (high-fidelity/coupled solvers cost a lot; design loops call them many times) *(honestly scoped: our 1D solver is cheap, it's a testbed for the method)*
- Why a GNN specifically? (the discretized solver IS a graph: cells = nodes, neighbors = edges; weight-sharing = mesh-native)
- State your two contributions in two sentences. *(you have: (1) faithful validated port; (2) working gas-off surrogate + diagnosed gas-case failure with a flux-form that removes the divergence but not the unfaithfulness)*

## 2. Reference Solver
Question: *what does the code you ported actually compute?*
- What are the 4 physics pieces? (conduction w/ T- and char-dependent props; pyrolysis Arrhenius; Darcy gas flow; aerothermal surface BC with B' blowing)
- What's the discretization? (1D finite volume, N cells through the thickness)
- What are the inputs and outputs? (heat-flux history in; thermocouple temps + densities out — the `con.out` columns)
- Which cases do you use and how do they differ? *(you have: aw1 = gas-off conduction+pyrolysis; aw2 = full gas + aerothermal square-pulse forcing)*

## 3. Methodology
Question: *what did you build, and how did you make sure it's trustworthy?*
### 3a. The port + verification
- How is the port structured? (modular Python, one module per Fortran block: case/domain/materials/pyrolysis/gas/aero/solver)
- How did you verify it? (column-by-column diff of every `con.out` output vs Fortran: max/mean/relative error per column)
- What was the result? *(you have: aw1 machine precision; aw2 all 23 columns to 0.064 K)*
### 3b. Training-data generation
- How do you turn the solver into training data? (run it with full-field recording -> trajectory .npz -> graph pairs)
- What is the super-step / gap? (record every K solver steps; one GNN step = K solver steps)
- How do you sweep conditions? (vary the forcing: peak flux + cutoff time; 8 vs 24 trajectories)
### 3c. GNN architecture
- What are the nodes, edges, features? (nodes = cells w/ [T, rho, rho_i, porosity]; edges = neighbor differences + spacing; K=1)
- What does the model predict? (per-cell next-state delta of the independent DOF [T, rho_i0, rho_i1])
- What is the flux-form variant and why? *(you have: predict antisymmetric face fluxes + a local source, so a uniform field gives exactly zero change by construction -> kills the quiescent-cell bias)*
- How is it trained for stable rollout? (single-step + noise injection; then multi-step BPTT for the gas-off case)

## 4. Results
Question: *what actually happened, with numbers and honesty?*
### 4a. Port validation
- The per-case validation table. *(you have: aw1 machine precision; aw2 0.064 K over 23 columns)*
### 4b. Gas-off (aw1) surrogate — the success
- How accurate over a full rollout? *(you have: 7.6 K mean, tracks the pyrolysis front to a fraction of a mm)*
- What made it work? (multi-step training; K=1 matches the conduction stencil)
- What failed along the way and taught you something? *(you have: volume/state-space sampling diverged -> the char manifold can't be sampled independently)*
### 4c. Gas (aw2) — the diagnosed failure
- What happens with the naive direct model? *(you have: diverges to ~25,000 K on held-out; quiescent-cell bias)*
- What does the flux-form fix, and what does it NOT fix? *(you have: removes the catastrophic divergence by construction; does NOT make it faithful)*
- The seed study: what did 8 seeds x 2 datasets show? *(you have: n8 mean 850 K, 3/8 blow up; n24 mean 419 K, 0/8 blow up in the MEAN)*
- Why is even the bounded model still wrong? *(you have: mean flatters — dominated by cool bulk; n8 best undershoots the surface; n24 has a surface runaway + false char front to the back face; T and char are decoupled output channels)*
- Which levers failed? *(you have: ensembling diverges; BPTT overfits thin data; surface loss-weighting destabilizes)*
- What is the honest verdict? (gas case = open problem; gas-off = the working surrogate)
### 4d. Wall time
- Solver vs port vs surrogate timing, with honest caveats. *(you have: Fortran ~3 min, Python port ~15 min, GNN ~4x faster on CPU at this toy scale; CUDA slower; not a 1000x story yet)*

## 5. Discussion
Question: *what do the results mean, and where does the method break?*
- What did the port establish? (a trusted data generator — a surrogate is only as good as its data)
- What did the gas-off surrogate establish? (a per-cell weight-shared GNN is an accurate, mesh-native surrogate for the coupled conduction+pyrolysis rollout)
- Why is the gas case the edge of the method? (fast square aerothermal forcing -> stiff surface transient + sharp cutoff -> closed-loop rollout instability; NOT the gas transport itself, which is held from truth)
- What's the honest limitation? (single mesh resolution; gas case unsolved; low mean error can mask an unphysical field)

## 6. Future Work
Question: *what would you try next, given what failed?*
- The gas case is closed-loop stability, not data quantity -> what levers are left? (stability-aware loss penalizing rollout error growth; rate-prediction + integrate so dt is explicit; time-step-aware adaptive cadence)
- Broaden: mesh-resolution generalization (train across resolutions, exploit weight-sharing)
- Measure speedup properly on a design-relevant sweep
- Physics extensions: radiation, surface recession, 2D/3D

## 7. Conclusion (write SECOND-to-last, ~4 sentences)
Question: *what should the reader remember?*
- We ported and validated the solver.
- We built a working gas-off surrogate (number).
- The gas case is a diagnosed open problem, and why.
- The next step.

---

### Grey's own editing rules (apply on the prose pass)
- Cut hype and throat-clearing.
- Every claim is either cited (abstract-verified) or shown by your own results; else cut/soften.
- Separate claims from facts.
- No em dashes; minimize hyphens to source terms.
- Define every term the first time (you're writing for a mixed ML/engineering reader).
