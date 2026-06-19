# Heat-shield GNN surrogate — working notes for the white paper

Draft material I'm collecting for the paper. The prose is rough but the numbers
are final unless I've flagged them. Headings follow the structure I want the
paper to have.

## Methodology

### Reference solver and Python port

The reference is Martin's 1D finite-volume heat-shield code, written in Fortran
(`1Dheat.f` and a handful of `.INC` includes). It does heat conduction with
pyrolysis, optional gas transport through the char (Darcy plus advection), and an
aerothermal surface boundary. I rebuilt it in Python one module at a time, and I
matched the Fortran closely wherever the answer depended on it: the grid and
geometry weights, the mixture properties (k, cp, enthalpy), the closed-form
Arrhenius pyrolysis update, the gas solve, and the aerothermal boundary with its
B′ tables, blowing correction, and wall enthalpy. The thermocouple sampling was
the fiddly part; it follows the reference `SAMPLE_PROBES` and
`INTERP_FACE_FROM_CELL`, clamping to the face value at the boundaries and
reconstructing the face elsewhere.

Three port details I want in the methods section, since each one cost me time and
each maps to a specific spot in the Fortran:
- Thermocouple positions are given as depth from the right wall, so `XTC = L0 − depth`.
- The face density that feeds the conductivity at a face is the pre-pyrolysis
  density, not the updated one. That matches the order the Fortran does things in.
- Output is written before the physics step, not after, which is the time-offset
  convention the Fortran uses.

### Verification approach

I verify column by column against the Fortran `con.out` from a run with the same
configuration, reporting max-absolute, mean-absolute, and max-relative error for
each output column. That per-column table is what goes in the Results section for
each case.

### Training-data generation

Once the port is trusted, the Python solver generates the training data. Each run
records the full per-cell field over time (`T`, `ρ`, `ρᵢ`, and `pg`/`mdotf` for
gas runs), ghost cells included so the boundary forcing lives in the data. The
nice property is that one recorded trajectory can feed any downstream target or
super-step later without paying for another solver run. The first dataset just
varies the boundary forcing, a sweep of wall-temperature histories, on the fast
gas-off `aw1` case. Mesh-resolution and material sweeps come after.

### GNN architecture

The surrogate is a per-cell message-passing network. Every finite-volume cell is
a node that carries `[T, ρ, ρᵢ, porosity]`, and neighboring cells are joined by
edges whose attributes are the interface spacing and the relative state of the
neighbor. The model predicts how each cell's state changes over a step, and we
roll that forward in time. Because the weights are shared across every cell, the
same trained model runs on a mesh of any length, which is the whole reason
resolution generalization is even on the table. One message-passing pass (K=1) is
enough, since that matches the width of the conduction stencil.

## Results

### Validation, aw1 (conduction plus pyrolysis, gas off): bit-exact

Every one of the 15 output columns matches the Fortran `con.out` to about 10⁻⁸,
which is machine precision, and that includes all the interior thermocouples. The
conduction and pyrolysis port is exact, full stop.

### Validation, aw2_tc21 (full gas plus aerothermal): to 0.064 K

With the gas physics turned on, against the same-config Fortran `con.out`:

| Quantity | Agreement |
|---|---|
| All temperature columns | max **0.064 K**, mean ~4×10⁻⁴ K |
| Hot aerothermal surface | **0.06 K** |
| Densities | ~10⁻³ to 10⁻⁹ |
| Time, cold back face | bit-exact |

There's a verification cautionary tale here that I think is worth a sentence in
the paper. For a while it looked like the interior was drifting and the back face
sat about 37.7 K too high. Neither was real. Both came from how the
thermocouples were being sampled, and once that matched `SAMPLE_PROBES` they
disappeared. The lesson is that comparing two fields through a sampling layer can
invent a discrepancy that isn't in the physics.

One thing still open: as it shipped, `aw2_tc22` is byte-for-byte identical to
`aw2_tc21`. Same `heat.case`, same `tc_locations.dat`, same flux file, and the
`CASE_NAME` field even still reads `'aw2_tc21'`. So it doesn't actually add an
independent validation case. I need to check with Martin whether tc22 was meant
to have a different configuration before I treat 2.1 and 2.2 as two cases.

### Wall time

The Fortran does `aw2_tc21` in about 3 minutes; my unoptimized NumPy version
takes about 15.6, so roughly 5× slower. That's fine, and expected. The speed
argument in the paper isn't the port anyway, it's the surrogate.

### Surrogate, aw1 rollout

Trained on the solver trajectories, the surrogate gets rolled forward on its own
predictions over a held-out `aw1` heating scenario (gas off) for the full 60
seconds. It holds up:
- mean rollout error of about **7.6 K**, which is half a percent to one percent
  of the 298 to 1500 K field;
- it tracks the char and pyrolysis front to within a fraction of a millimeter.

The thing that made this work was training on multi-step rollouts instead of
single steps. Single-step models look fine for one prediction but accumulate
error and wander off once you let them run. That's the same failure mode the
bead paper warned about, and I'm seeing it here, so it belongs in both Results
and Discussion.

### Mesh-resolution generalization (still in progress)

This is the headline claim: train at one mesh resolution and run at others
without retraining, which the weight sharing should make possible. Right now
zero-shot transfer doesn't work, but I've pinned down why, and it's two separate
things.

1. **The time step is tangled up with resolution.** I was recording every fixed
   number of solver steps, but the solver's step size scales as dx², so 1/N², so
   the time between snapshots quietly changes with resolution. The model learned
   one fixed Δt and then over- or under-steps on any other mesh. Switching to
   recording at a fixed physical Δt (0.11513 s) cut the fine-mesh error by about
   4×. Necessary, but not enough on its own.
2. **It only ever saw one cell size.** Even with the time step fixed, coarser and
   finer meshes still fail, because the model trained at a single dx. The edge
   feature does carry dx, but the model treated it as a constant rather than
   something that varies.

So the conclusion for this section is that weight sharing gives you the
architecture for free, but you still have to train across a range of resolutions
for the model to actually learn how the update depends on dx. Building that
multi-resolution dataset and retraining is the next experiment.

## Discussion (things to develop)

- Comparing fields through a sampling layer can look like physical drift when it
  isn't. The aw2 offset story is a clean example.
- Single-step surrogates drift on rollout. Multi-step training is what makes the
  surrogate usable, and I can put numbers on it: end-of-run error went from
  hundreds of K down to tens.
- State-space (volume) sampling worked for the earlier bead and Sod cases but
  diverges here. The ablator couples temperature and char state tightly, and
  sampling them independently puts the model in states the physics never visits,
  so trajectory plus multi-step training stays ahead. I think that's a result in
  its own right, not just a footnote.
- Resolution generalization isn't automatic. You get it only after handling both
  the time-step and the cell-size dependence.

## Future work

Get the surrogate working on the gas plus aerothermal (`aw2`) case, finish and
quantify the mesh-resolution generalization, and measure the actual design-loop
speedup against the solver. Further out is extending to 2D or 3D, though there's
no reference solver for that yet, so it would mean building one first.
