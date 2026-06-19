# LAUMeltFlow project context

## Response style
- Default to brief responses (1-2 paragraphs unless I ask for depth).
- No long preambles, recap sections, or "let me check first" narration.
- Skip caveats unless they actually matter for the decision.
- For tutorials and conceptual questions, lead with the answer, expand only if asked.
- Don't repeat back what I just said before answering.

## Code style
- Match the style of nearby existing code.
- No unnecessary error handling or defensive programming around things that can't actually fail.
- Comments only when the "why" is non-obvious. Default to no comments.

## Workflow
- For long-running compute (training, data gen): use background tasks.
- For file reads: prefer offset/limit and Grep over reading whole files.
- Don't re-read files I've already shown you in the conversation.

---

# Current active work: Python port of Martin's heat-shield code → GNN surrogate

**The job (phase 1):** port `heat_2026-04-11_1837/` (Martin's Fortran 1D heat-shield pyrolysis solver, ~3,200 lines of F77) to Python; validate against the Fortran outputs. **The job (phase 2, now active):** train a GNN surrogate on data the validated Python solver generates (the white-paper goal).

## ⇨ HANDOFF (state as of 2026-06-17 — read this first)

**⇨ RESUME HERE:** Deliverable is now a **small IEEE conference paper** (skeleton built: `docs/heat_TPS/heat_gnn_ieee.tex` + `docs/heat_TPS/heat_gnn_refs.bib`). Grey edits prose; I run experiments. **aw2 (gas) surrogate is DONE as a research thread — it's a diagnosed failure case: works in-sample but DIVERGES held-out (3587 K). LOCKED IN as a paper "future work / diagnosed limitation" result — see the AW2 GAS SURROGATE section below.** Solver correctness re-verified (4-angle adversarial workflow — t=60 s spike is physical, the flux turn-off; aw2 bit-validated to 0.064 K). **Likely next work:** finish the IEEE paper (fill body prose, 3 TODO citations, pick figures), and/or the older still-open **mesh-resolution generalization** thread (multi-res training run never done). Don't keep grinding aw2.

**Unrelated aside:** `interactive_curve_tracer.html` in the repo root is courseware Dr. Lau sent without context (an EE461g BJT/MOSFET curve-tracer sim). NOT part of this project — ignore it.

### ⇨ AW2 GAS SURROGATE — diagnosed FAILURE CASE; does NOT generalize held-out (2026-06-18). Recommend LOCK IN as future work.
Extended the surrogate from aw1 (gas-off) to **aw2 (gas + aerothermal)**. Pipeline now fully gas/material-general. **DEFINITIVE RESULT: aw2 does not generalize. In-sample (train=test trajectory) the best recipe is bounded ~625 K, but that was OPTIMISTIC — the true HELD-OUT test (train on 8 swept forcings, roll out on the unseen stock 1.5 MW/60 s forcing) DIVERGES at 3587 K.** Early rollout is excellent even held-out (30 K @ t=1.7 s) so the model learns local dynamics fine; it's long-horizon generalization across forcings that fails. aw1 works (7.6 K); aw2 is a rigorously-diagnosed open problem. **Recommendation: lock in as a paper "diagnosed failure / future work" result; don't keep grinding.**

**The arc (this is paper-grade methodology):**
1. **Naive recipes diverge.** Multi-traj forcing sweep (8 traj, `make_aw2_dataset.py`): held-out **4132 K, diverges** (the single-traj 506 K was *in-sample*, misleading). K=2 wider stencil: **5159 K, worse** (overfits — same as aw1). Ruled out: data starvation (more traj didn't fix), gas-state prediction (`--no-pred-gas` holding pg/mdotf from truth didn't help).
2. **Diagnosis CORRECTED (Grey caught my error): it's TEMPORAL, not spatial.** Per *cell*, aw2 is GENTLER than aw1 (max 111 vs 221 K/cell — aw2's 500-cell mesh spreads the gradient). The real difference is the **per-step surface jump**: aw2 surface leaps **734 K/super-step** vs aw1's **41 K** (18×), because the square-pulse aero flux heats the surface 298→1534 K in ~1 s. A fixed coarse super-step can't track that. (`viz_why.py` → `figs/why_aw1_aw2.png` shows spatial=not-cause, temporal=cause, consequence.)
3. **Finer cadence confirms the diagnosis** (intervention test). Recording every 200 steps (Δt 0.0184 s, 10× finer): early rollout drops to **40 K** (aw1 regime!) — but 6515 steps accumulate to 2885 K and it detonates at the t=60 flux-off.
4. **Adaptive cadence is the fix that works** (best). Δt-aware GNN: **add Δt as a node feature**, train on mixed-cadence pairs (`graph.build_adaptive_dataset`, gaps=(1,10)), roll out with a schedule = fine Δt within 2 s of each forcing event (t=0, t=60), coarse on the plateau (`eval_adaptive.py`). 945 steps, **784 K, BOUNDED** (plateaus ~1000 K, never blows up) vs coarse/fine→9000 K. `figs/aw2_progress.png` shows it. Multi-step adaptive (`train_adaptive_rollout.py`) gave 925 K — slightly *worse* than single-step+noise (robustness/horizon tradeoff, like aw1's M=8).
5. **Residual = bulk-conduction plateau drift, traced to a quiescent-cell BIAS.** Adaptive *solved the surface* (at t=59 s the hot surface became the BEST region); the error moved to the cold/mid bulk. Verified the cause: the model predicts **+1.5 K/step for a cold untouched cell** that should be exactly 0 — a per-step bias that accumulates over the ~310 coarse plateau steps. Added a **stay-put regularizer** (`train_gnn --stayput`, penalizes cold cells toward zero delta): cut the bias 1.5→0.35 K/step. Weight sweep: **0.2 is the sweet spot (in-sample 784→625 K, bounded)**; 0.5/1.0 over-regularize and diverge in the cooldown.
6. **HELD-OUT TEST = the definitive result (and the key correction).** Built a fine-cadence 8-traj adaptive sweep (`make_aw2_adaptive_dataset.py`, dt feature, 20832 pairs), trained adaptive+stayput-0.2, rolled out on the UNSEEN stock forcing. **Held-out 3587 K, diverges** — vs the 625 K in-sample. So the in-sample wins were largely the model fitting its own rollout trajectory; aw2 does NOT generalize across forcings. Naive coarse held-out was 4132 K, so all the machinery (adaptive+stayput+multi-traj) only nudged held-out 4132→3587 K. Early rollout stays great held-out (30 K), confirming the failure is long-horizon generalization, not local dynamics.

**⇨ STATUS: LOCK IN. aw2 is a diagnosed, well-supported failure case for the paper, not a hole to fill.** We ruled out data starvation, gas-prediction, stencil width; verified the temporal-stiffness + quiescent-bias mechanisms; showed adaptive cadence + stay-put recover in-sample but not held-out. Continuing is deep diminishing returns.
**The ONE remaining genuinely-different swing (Future Work, NOT a deadline fix):** a **flux-conservative GNN** (predict edge fluxes → take divergence → a uniform field gives *exactly* zero delta by construction, eliminating the quiescent bias at the root instead of regularizing it; this is the project's "Sod flux model" form). Real new architecture; uncertain payoff given held-out divergence persists with every state-delta variant. Other deferred ideas: rate-prediction (dstate/dt + integrate), 3-tier cadence, far more trajectories (>>8).

**Solver re-verified (the t=60 spike is NOT a bug).** Grey worried the t=60 s surface spike signaled a solver error. A 4-angle adversarial workflow (forcing-timing, Fortran re-validation, physical-invariants, code-audit) found **zero concerns**: the flux file is a square pulse (ON 0.1 s, OFF 60 s); every per-step surface jump >10 K falls inside the two forcing windows (heating at t≈0, cooling at t=60), nothing elsewhere; aw2 still matches Fortran to 0.064 K (23 cols); density strictly non-increasing, no NaN, pg within 0.78% ambient; dt-clipping lands exactly on t=60 with no overshoot. **The aw2 surrogate failure is the model's, not bad data.**

**Speedup (§4.4) measured (`measure_speedup.py`):** GNN rollout ~**4× faster than the Python solver** on CPU for aw1 60 s (0.47 s vs 1.9 s). Honest caveats: CUDA is *slower* at this toy scale (kernel-launch overhead on 100 cells); ~break-even vs compiled Fortran. Real speedup levers (bigger super-step, larger meshes, amortization) not exercised. Write it plainly — not a "1000×" story yet.

**Deliverable is now an IEEE conference paper (scope changed mid-session).** Prof wants "something to share with Dr. Martin," then "small IEEE conference paper." Built: `docs/heat_TPS/heat_gnn_ieee.tex` (IEEEtran two-col skeleton, all section heads, validation table, Eq.1, Index Terms, `\cite{}` anchors) + `docs/heat_TPS/heat_gnn_refs.bib` (MeshGraphNets + Sanchez-Gonzalez filled; surrogate-motivation, TPS-surrogate, own-symposium = **TODO**). Also `docs/heat_TPS/heat_gnn_whitepaper.{md,tex,pdf}` (earlier prose draft — mine for sentences). `docs/heat_TPS/whitepaper_outline.md` updated with real numbers + DONE/[future] tags. Grey edits prose in markdown; settled sections drop into the .tex. **Grey's editing style (apply it): cut hype/throat-clearing, ground every term in the Fortran source (flag coinages), separate claims from facts, NO em dashes, minimize hyphens to source terms.** He's a Master's student, NOT a domain expert — for conceptual Qs see memory `response-style-explanations` (define every term plainly, build from ground up).

**New viz this session (all in heat_python/, → figs/):** `viz_mesh.py` (mesh cells 3 resolutions), `viz_graph.py` (node/edge/ghost overlay, annotated, mm axis), `viz_front_on_nodes.py` (char front sweeping FIXED nodes — colored strips), `viz_fcn.py` (GNN-of-MLPs architecture + per-stage equations → `figs/fcn.png`), `viz_why.py` (aw1-works/aw2-fails, temporal-not-spatial), `viz_aw2_progress.py` (error-vs-time across aw2 recipes), `viz_rollout.py` (now takes `--case-dir`/derives n for aw2). `docs/heat_TPS/equations.html` = MathJax governing-equations explainer (every var defined). Memory added: `response-style-explanations.md`.

**Port: DONE & validated.** aw1 bit-exact; aw2_tc21(=tc22) to 0.06 K vs same-config Fortran. The famous "interior drift" was a thermocouple-sampling-method mismatch + a gas-flag mistake (lgas = Fortran `-g`, NOT `case.gas_energy`) — no real numerical drift. Details below.

**GNN surrogate: WORKING.** Best model `heat_python/models/heat_mpgnn_rollout.pt` — a per-cell message-passing GNN (`HeatMPGNN`, adapted from `whip/gnn/bead_mpgnn.py`), predicts next-state deltas, rolled out in time. **7.6 K mean rollout error** vs the solver on a held-out aw1 heating scenario (~0.5–1% of the field); tracks the char/pyrolysis front to a fraction of a mm. Trained: `make_dataset.py` (24-trajectory forcing sweep) → `train_gnn.py --noise` (single-step + noise injection) → `train_rollout.py` (multi-step M=4 BPTT). Only aw1 (gas-off, single mesh) so far.

**Key research finding (paper-worthy):** chased the residual error. Model tuning (M=8, K=2) regressed; the symposium paper's "volume/state-space sampling" fix was tried (`make_volume_dataset.py`) but **diverges** — diagnosed to the ablator's coupled (temperature, char-state) manifold, which independent sampling violates (hot-but-virgin cells pyrolyze explosively). v3 (T-correlated char) cut it 15× but still diverged (1675 K). So trajectory+multi-step (7.6 K) remains best; volume sampling is a "here's why it's hard" methodology result, not a better model. Full detail in the GNN section below.

**OPEN DECISION (pick a direction):** (1) **broaden** — mesh-resolution generalization (vary NBRN — the white-paper headline; exploits weight-sharing), gas cases (aw2), speedup measurement [recommended]; (2) **consolidate** — write up working surrogate + sampling-methodology finding; (3) **one more volume shot** — volume-pretrain → multi-step-finetune, or 2-species-aware char sampling.

### ⇨ MESH-RESOLUTION GENERALIZATION — STARTED (2026-06-03 cont.). Zero-shot FAILS; diagnosed two mechanisms.
Took the existing `heat_mpgnn_rollout.pt` (trained at NBRN=100) and rolled it out at other resolutions on the held-out forcing (added `--nbrn` to `eval_rollout`, `nbrn=` override to `solver.run`). **Zero-shot transfer does not work**, and we diagnosed *why* (this is the paper's resolution-generalization methodology section):
- **Mechanism 1 — CFL time-step coupling.** Dataset recorded every fixed *number of solver steps*; `dt_nom = 0.5·max_dx²·ρ·cp/k/cfl ∝ 1/N²`, so snapshot Δt drifts with resolution (N=40→0.72s, N=100→0.115s, N=200→0.029s). Model learned a fixed-Δt delta, so it grossly over/under-steps at other N. **Fix:** added `record_dt` (fixed *physical* snapshot interval) to `solver.run` + `--record-dt` to `eval_rollout`; training Δt = **0.11513 s**. Re-running the sweep at fixed Δt cut N=200 ~4× (19,117→4,681 K) — necessary but **not sufficient**.
- **Mechanism 2 — single-resolution training can't learn the Δx-dependence.** Even at fixed Δt, N=40 (500 K) and N=200 (4,681 K, still diverges) fail because the model only ever saw dx=0.5 mm; the edge feature carries Δx but the model learned it as a *constant*, not a variable. (Likely compounded by K=1 reach: one message pass = one cell, but a fixed Δt on a fine mesh should spread heat across more cells.)

Sweep results (mean rollout |err| K): **naive(fixed-steps)** 40→225, 100→7.6, 200→19117 (→38k). **fixed-Δt** 40→500, 100→7.4 ✓, 200→4681 (→9.7k). Rollout npzs: `data/rollout_aw1{,_fixeddt}_n{40,100,200}.npz`.

**CONCLUSION / NEXT:** weight-sharing gives the architecture but you must **train across multiple resolutions** for the model to learn the operator's Δx-scaling. Next concrete step: build a multi-resolution dataset (sweep NBRN∈{40..200}, fixed physical Δt=0.115s) → retrain → re-run this sweep; expect the outer columns to go from "diverges" to "transfers." New viz: `viz_mesh.py` (mesh cells at 3 resolutions → `figs/mesh.png`), `viz_graph.py` (node/edge/ghost graph overlay, annotated → `figs/graph.png`).

## Where things stand

### Build environment
- Fortran compiles cleanly in **WSL Ubuntu** with `gfortran` 13.3 (`apt install gfortran make`).
- `make heat` in `heat_2026-04-11_1837/` produces a working executable.
- Tested on examples `aw1` (simple heat+pyrolysis, ~30s wall) and `aw2_tc21` (full aero, ~3 min wall).
- `CASE.INC` was missing from the original tarball; Martin sent it. Now in place.

### Python port (`heat_python/`)

Module-by-module status:

| Module | Replaces | Status |
|---|---|---|
| `case.py` | CASE.INC + READ_CASE | ✅ Done. Parses heat.case namelist. Tested against all 3 examples. |
| `domain.py` | DOMA.INC + grid setup | ✅ Done. Supports Cartesian/cylindrical/spherical. Vectorized. |
| `materials.py` | READ_KAPPA, READ_CP, CALC_K_MIX, CALC_CP_MIX, CALC_HS_MIX, etc. | ✅ Done. Verified `hs_mix(T_ref, tau) = 0` (energy origin alignment). |
| `pyrolysis.py` | DECOMP.INC, READ_RHO, CALC_TAU, DECOMP, CALC_POROSITY, species update | ✅ Done. Vectorized closed-form Arrhenius. |
| `bcs.py` | BC blocks in PROGRAM HEAT | ✅ Done. Aerothermal BC branch (B', blowing correction, wall enthalpy) wired + exercised by aw2. |
| `io_files.py` | READ_TEMP, READ_TC_LOCATIONS | ✅ Done. Handles Fortran D-notation. |
| `solver.py` | Main time loop body | ✅ Phase-4 wired: gas solve, gas-energy coupling, advection, aero BC, per-step pamb, pg/mdotf/rhogsto in State. Property helpers now **vectorized** (bit-identical to old scalar loops — verified). Thermocouple sampling rewritten to match Fortran SAMPLE_PROBES (see below). |
| `main.py` | CLI | ⬜ Stub only. Not needed — `python -m heat_python.solver [--gas]` works. |
| `gas.py` | READ_GAS, CALC_GCP/GMU/RGAS/HGAS/UGAS/RHOGP/CVG, SOLVE_GAS, TRIDIG | ✅ Done. Property lookups + explicit (sealed-back/open-front) and tridiag Darcy branches. Exercised by aw2. |
| `aero.py` | READ_AERO, READ_BPRIME_HW, CALC_AERO_*, CALC_BLOWCORR, CALC_WALL_HW | ✅ Done. Aero edge table + 25×74 B' tables + blowing/wall-enthalpy. Feeds bcs.py aero branch. |
| `radiation.py` | 1Drad.f, multi-band P1 | ⬜ Stub. Not needed for aw1/aw2 (surface re-radiation lives in the aero BC, not the P1 solver). |
| `validate.py` | comparison utility | ✅ Done. Column-by-column diff (max_abs, mean_abs, max_rel). |
| `compare_thermocouple.py` | (new) compares con.out vs reference `thermocouple.txt` | ✅ Done. Interpolates Python probes onto the reference time grid. |

### Validation status on aw1 — ✅ BIT-EXACT

`python -m heat_python.solver --case-dir .../aw1` (gas off — see lgas note). **All 15 columns match the Fortran `examples/aw1/con.out` to ~10⁻⁸** (machine precision), including every interior thermocouple. The conduction + pyrolysis port is exact.

**The "interior conduction drift" was never real** — it was two output/config artifacts, now both resolved:
1. Thermocouple sampling-method mismatch (the original ~227 K) — fixed by matching Fortran `SAMPLE_PROBES`.
2. Gas physics wrongly auto-enabled on the aw1 Python run (today's extra → 470 K). The aw1 Fortran reference was generated **without** `-g`; running Python with gas off makes it bit-exact.

### Validation status on aw2_tc21 (phase 4 — gas + aero)

Run with `python -m heat_python.solver --case-dir .../aw2_tc21 --gas`. **1.3 M steps, ~15.6 min wall** (Fortran ~3 min — Python is ~5× slower, as expected; the speedup story is the GNN surrogate, not the port). Compared **apples-to-apples against the same-config Fortran `examples/aw2_tc21/con.out`** (NOT `reference/2.1/thermocouple.txt`, which was generated with init_temp ≈ 300 K, not our 298.15 — a config-vintage mismatch that shows as a constant ~1.85 K offset).

Pre-sampling-fix result: time / cold-back-face / all densities bit-exact; interior thermocouples within a few K; **hot aero surface validates well (~6–8 K)**. The one large residual was a **constant +37.7 K at the back thermocouple** — diagnosed as a **thermocouple-sampling bug, not physics**: the two probes sitting exactly on the boundaries (x=0, x=L0) were sampled as the nearest cell instead of the wall face.

**Sampling fix (landed):** `setup_thermocouples`/`sample_probes` rewritten to mirror Fortran `SAMPLE_PROBES` + `INTERP_FACE_FROM_CELL` — boundary probes clamp to the face value, interior probes use face-reconstruction interpolation. Verified line-by-line against the Fortran.

**✅ aw2_tc21 VALIDATED** (re-run with the sampling fix, vs the same-config Fortran `con.out`). **All 23 columns pass:** temperatures match to **max 0.064 K** (mean ~4×10⁻⁴ K), densities to ~10⁻³–10⁻⁹. The +37.7 K back gap is gone (now 3×10⁻⁵); the hot aero surface is **0.06 K**. So the apparent "drift"/"surface offset" on aw2 was **almost entirely the sampling mismatch, not FP conduction drift** — the phase-4 port (gas + Darcy + aero BC + B′) is faithful to ~machine precision. Re-validate command:
```
python -m heat_python.validate --fortran heat_2026-04-11_1837/examples/aw2_tc21/con.out \
    --python heat_2026-04-11_1837/examples/aw2_tc21/python_out/con.out
```
(Shape note: Fortran writes 1001 rows, Python 1000 — a harmless off-by-one in output cadence; compares first 1000.)

(Resolved: aw1's con.out was NOT stale — a fresh Fortran run was byte-identical. The aw1 mismatch was the gas-flag issue above, not the reference.)

### Bug fixes that landed during validation
- TC positions: Martin's file gives **depth from right wall**; `XTC = L0 - depth`.
- Time-offset: Fortran writes BEFORE the physics step; Python loop restructured to match.
- `rho_face` for k_face uses pre-pyrolysis `state.rho`, not `rho_new`.
- D-notation in `tc_locations.dat`.
- **Thermocouple sampling** now matches Fortran `SAMPLE_PROBES` (face-clamp + face-reconstruction).

### lgas (`-g`) handling — IMPORTANT
`lgas` mirrors the Fortran `-g` CLI flag; it is **NOT** `case.gas_energy` (which is only a sub-option active *within* gas mode — both aw1 and aw2 set it True). The case file does not encode the `-g` choice, so `run(lgas=None)` now defaults: **gas ON only if `use_aero_bcn`** (aw2), **OFF otherwise** (aw1). Override with `--gas` / `--no-gas`. The reference con.out files were generated: **aw1 without `-g`, aw2 with `gas`** — so the defaults now match each reference. (An earlier `lgas = gas_energy or use_aero_bcn` heuristic wrongly enabled gas on aw1 → the 470 K mirage.)

### aw2_tc22 — ✅ validated, but it's a DUPLICATE of aw2_tc21
Ran `--gas` vs a fresh same-config Fortran con.out: **all 23 columns pass, identical numbers to tc21** (temps to 0.064 K). Reason: `aw2_tc22/heat.case`, `tc_locations.dat`, and the time-flux file (`time_flux_tc21.dat`!) are **byte-identical** to tc21, and the con.out files are identical (`cmp` confirms). The `CASE_NAME` in tc22's heat.case even still says `'aw2_tc21'` — looks like a copy that was never edited. Only `reference/2.2` (Martin's reference thermocouple traces) differs from `reference/2.1`. **So tc22 adds no independent physics coverage as shipped.** → **Ask Martin** whether tc22 should have a distinct config (different flux history / TC depths), or whether 2.1 vs 2.2 are just two measurements of the same condition.

## GNN data-generation pipeline (started — port is done)

Two-stage, target-agnostic, super-step-tunable. Mirrors meltflow_gnn conventions (np.savez, node/edge graphs).
- **Stage 1 — `heat_python/data_gen.py`** + solver `record_every`/`write_con` params: runs a case with full-field recording, saves a raw **trajectory** `.npz` (T, rho, rho_i, pg/mdotf, all *with ghost cells* so the boundary forcing is in the data). One trajectory supports any downstream target/gap without re-running the solver.
  `python -m heat_python.data_gen --case-dir .../aw1 --no-gas --record-every 5 --out heat_python/data/aw1_traj.npz`
- **Stage 2 — `heat_python/graph.py`**: trajectory → GNN dataset `.npz`. Nodes = all m=n+2 cells (incl. 2 ghosts carrying forcing); edges = 1D path (bidirectional), edge_attr = interface spacing. Node features `[T, rho, rho_i…, porosity]`; target = interior next-state (+ `target_delta`); normalization stats (std clamped for the inert/constant species). `--gap` sets the super-step.
  `python -m heat_python.graph --traj .../aw1_traj.npz --gap 1 --out .../aw1_pairs_gap1.npz`

**Stage 3 — GNN model + training (first end-to-end run done):**
- `heat_python/gnn_model.py` — `HeatMPGNN`, adapted from `whip/gnn/bead_mpgnn.py` (NOT the Sod flux model — bead has the per-NODE delta output we want). Node feats = absolute `[T,rho,rho_i…,porosity]`; edge feats = relative neighbor `[dT,drho,drho_i…,dx]` (gradient-driven, like bead's relative pos/vel); per-cell delta output; K=1 (conduction stencil). Has `step_mesh()` for full-mesh rollout. Weight-shared → any mesh length.
- `heat_python/graph.to_neighbor_samples()` — full-mesh dataset → bead-style per-cell (self/left/right/target) samples.
- `heat_python/train_gnn.py` — trains on the forcing dataset, predicts independent DOF `[T, rho_i0, rho_i1]` (rho/porosity/inert derive), split train/val **by trajectory**. First run: train MSE 0.54→0.05, val 0.33→0.01 (normalized); CUDA available. `models/heat_mpgnn.pt`.
- **Smoke test only** — 40 epochs, default hyperparams, single aw1 mesh, gas-off.
- `dataset`: `heat_python/data/aw1_forcing_dataset.npz` (24 trajectories, 12.5k pairs, forcing sweep on gas-off aw1).

**Stage 4 — ROLLOUT test done (`heat_python/eval_rollout.py`). Result: drifts (error accumulation).**
Rolls the GNN forward on its own predictions over a held-out heating scenario (ghost forcing re-imposed each step; rho/porosity derived from predicted species), vs the solver. Over 522 steps / 60 s:
- t=6 s (10%): mean 3.8 K / max 59 K
- t=30 s (50%): mean 43 K / max 405 K
- t=60 s (end): **mean 230 K / max 1076 K**
One-step prediction is good but small errors **compound** → large drift by the end. Expected for a one-step-loss model (cf. bead paper's rollout-error note). **Not yet a usable surrogate.** This IS the white-paper "Discussion: failure modes" content.

**Stage 5 — noise-injection training DONE (`train_gnn.py --noise`). Tamed the blow-up.**
Adds noise eps to predicted input cols, propagates to relative edge feats (edge=nbr-self), and adjusts the delta target (delta-eps) so the model learns to map a slightly-WRONG state back to the TRUE next state (contractive/self-correcting). `--noise 0.03`, 60 epochs → `models/heat_mpgnn_noise.pt`. Rollout before→after:
- end mean 230→**106 K**, end **max 1076→259 K** (runaway gone), overall mean 75→**55 K**.
- mid (t=30s) got slightly worse 43→57 K (the accuracy/stability trade-off).
Figure `heat_python/figs/noise/`: GNN field now matches solver, hot-wall artifact gone, error smooth+bounded. **Stable but not yet accurate enough** (~100 K end error at the hot wall).

**Stage 6 — multi-step (rollout) training DONE (`train_rollout.py`). Now a usable surrogate.**
Unrolls M=4 steps on full-mesh sequences, BPTT, loss on the rollout; warm-started from the noise model. Re-imposes ghost forcing from truth each step; derives rho/porosity differentiably. `models/heat_mpgnn_rollout.pt`. Rollout progression (overall mean | end mean | end max):
- no-noise: 75 | 230 | 1076 K
- noise 0.03: 55 | 106 | 259 K
- **multi-step M=4: 7.6 | 18.8 | 112 K**  ← ~0.5–1% of the 298–1500 K field.
Figure `heat_python/figs/multistep/`: GNN field ≈ solver; error dark everywhere except a faint hot-wall band at late time. **This is a working surrogate for aw1-style (gas-off) conduction+pyrolysis.**

**Stage 7 — chased the residual; reproduced BOTH symposium-paper findings (great methodology result).**
Best working surrogate remains the **multi-step M=4 model (`heat_mpgnn_rollout.pt`, 7.6K mean)**. The chase:
- **Model tuning doesn't help (it's a data problem):** M=8 unroll regressed (7.6→22.5K); K=2 wider stencil regressed (7.6→20.4K, lower train loss but worse rollout — overfitting the thin trajectory data). Confirms paper finding #1 + the "match K to the physical stencil" lesson (conduction is K=1).
- **Volume/state-space sampler built** (`make_volume_dataset.py`): tile smooth windows, run real solver K_SUPER=10 steps, emit center+neighbors → single-step samples. Coverage 100% of the T-vs-density plane vs trajectory's 44%.
- **But volume models DIVERGED on rollout** (v1 mean 8000K, v2 25000K) with perfectly fine training loss — reproducing the paper's "Cartesian diverges / polar works" finding. **Diagnosis:** the divergence is NOT from spatial gradients (cutting slope 45→10 left target ΔT ~212K, basically unchanged) — it's from sampling **(T, char-state) INDEPENDENTLY**, which creates off-manifold hot-but-virgin cells that pyrolyze explosively (200K+ targets). The ablator manifold strongly couples thermal+chemical state; independent sampling violates it. (This is *why* the heat shield is harder to volume-sample than the bead — and is itself white-paper methodology content.)

**Volume-sampling iterations (all single-step trained):**
- v1 (broad): diverged, mean 8000K (target dT std 227K).
- v2 (gentle gradients): diverged worse, 25000K — proved gradients were NOT the cause (target dT stayed 212K).
- v3 (**T-correlated char**, cap virginity by temperature): target dT 212→**37K**, rollout 25000→**1675K**. Big improvement, diagnosis confirmed — but **still diverges** (early 78K @ t=6s, climbs to thousands). Volume+single-step has not beaten trajectory+multi-step.

**Read:** the ablator manifold is genuinely hard to volume-sample (T + 2-species char coupling; my 1-param `u*w_cap` collapses a 2D char structure). Pure volume + single-step keeps diverging. **Best surrogate is still trajectory + multi-step M=4 (7.6K).**

**→ NEXT — options (decide):**
1. **Consolidate** (recommended). Working surrogate at 7.6K + a clean methodology result (reproduced + diagnosed the paper's volume-sampling failure mode; ablator manifold harder than bead). Coherent writeup.
2. **One more shot at volume:** volume *pretrain* → multi-step *finetune* on trajectory data (combine coverage + rollout optimization); or 2-species-aware char sampling; or real-state seeding. Diminishing-returns risk.
3. **Broaden** (more valuable than squeezing aw1): mesh-resolution generalization (vary NBRN — the white-paper headline claim); gas cases (aw2); speedup measurement (GNN vs solver wall-time).

Rollout npzs per recipe in `heat_python/data/`: `rollout_aw1{,_noise,_ms,_ms8,_k2,_volume,_volume_v2,_volume_v3}.npz`.

**Design decisions still open (deferred until we look at more data / start training):**
- **Target: state vs flux.** Data supports both. State model is the speedup play (drop the solver, roll out the GNN) and is the white-paper main event; flux model (à la Sod MPGNN) is a conservative baseline/comparison. Build state first.
- **Super-step size** (`--gap`): bigger = more speedup but stiffer/harder to learn + rollout stability. Start gap=1, sweep up.
- **Sampling / dataset scope:** currently one trajectory (aw1). Need many trajectories across varied conditions (flux histories, mesh resolution `NBRN`, material params) — generate more and concatenate. This is the trajectory-vs-state-space-sampling decision from the white-paper outline.
- Note: independent per-cell DOF is `[T, rho_i]` (rho and porosity derive from rho_i) — relevant for choosing what the model predicts.

## What's next (pick up here) — TOP PRIORITY: fix GNN rollout drift

1. **Fix rollout drift (the active task).** See the "Stage 4" block above. Start with **noise-injection / push-forward training** in `train_gnn.py` (add noise to inputs so the model corrects its own errors), then **multi-step rollout loss**. Re-measure with `python -m heat_python.eval_rollout` after each change — rollout error (not 1-step val loss) is the metric. Target: keep end-of-trajectory mean error to ~single/low-double-digit K.
2. Then: sweep super-step `--gap`, try K=2, more trajectories, then move to gas cases (aw2) and mesh-resolution generalization.

**Lower priority / deferred (port side — essentially done):**
- Save validation tables + wall-times to `docs/` for the paper (aw1 bit-exact; aw2_tc21=tc22 to 0.06 K; Fortran ~3 min vs Python ~15.6 min). `validate.py` output is the table.
- Ask Martin about aw2_tc22 (byte-identical to tc21 — only one independent aw2 case exists).
- Optional: `radiation.py` (laser/`-r` only); `main.py` polish.

**STATUS: Port DONE (aw1 bit-exact; aw2 to 0.06 K). GNN pipeline DONE end-to-end (data → train → rollout). Current blocker: the trained surrogate drifts on rollout (error accumulation) — fixing that is the next real work.**

### GNN pipeline files (all in heat_python/)
**aw1 (gas-off) core:** `data_gen.py` (solver→trajectory .npz) · `make_dataset.py` (aw1 forcing-sweep multi-traj) · `graph.py` (build_dataset, to_neighbor_samples, **build_adaptive_dataset** [mixed-cadence + dt feature], gas-aware node_features) · `gnn_model.py` (HeatMPGNN) · `train_gnn.py` (one-step + `--noise`; **select_out_cols** by name, `--no-pred-gas`) · `train_rollout.py` (multi-step BPTT; now column-agnostic `mesh_step`+`_Cols`, single-traj support, `--case-dir`, `--no-pred-gas`) · `eval_rollout.py` (aw1 rollout vs solver; `--nbrn`/`--record-dt`).
**aw2 (gas) additions (2026-06-17):** `make_aw2_dataset.py` (flux-sweep gas dataset) · `eval_rollout_traj.py` (generic trajectory rollout, aw1+aw2) · `eval_adaptive.py` (adaptive-cadence rollout) · `train_adaptive_rollout.py` (dt-aware multi-step) · `measure_speedup.py`. `solver.run` now takes `nbrn=`, `record_dt=`.
**Key aw2 models:** `heat_mpgnn_aw2_adaptive_noise.pt` (BEST aw2, 784 K, dt-aware single-step+noise) · `..._adaptive_rollout.pt` (multi-step, 925 K) · `..._aw2sweep_rollout.pt` (8-traj coarse, diverges). **Key data:** `aw2_traj.npz` (full-res default-forcing), `aw2_traj_coarse.npz` (stride-10), `aw2_adaptive.npz` (dt-feature dataset), `aw2_forcing_dataset.npz` (8-traj sweep). Data in `heat_python/data/`, models in `heat_python/models/`.
Viz: `viz_rollout.py` (space-time + profile gif from a rollout npz) · `viz_compare.py` (thermocouple traces + error-vs-time progression + **char_front_compare** [solver vs GNN pyrolysis-front overlay; needs rho_gt/rho_pred which `eval_rollout` now saves] → `figs/compare/`) · `viz_model.py` (architecture one-pager) · `viz_all.py` (regenerates ALL rollout viz into one folder-per-recipe layout). Figures in `heat_python/figs/`: shared `model_onepager.png`, `char_front.png`, `compare/`; one `rollout_NN_<recipe>/` folder per recipe (01_onestep, 02_noise, 03_multistep_m4_BEST, 04_multistep_m8, 05_k2, 06/07/08_volume_v1/v2/v3_diverged), each with `rollout_spacetime.png` + `rollout_profile.gif`. Rerun `python -m heat_python.viz_all` after new rollouts.

White-paper outline + abstract drafted in [docs/heat_TPS/whitepaper_outline.md](docs/heat_TPS/whitepaper_outline.md).

### Handy artifacts from this session
- `examples/aw2_tc21/con_prev_16h17.bak` — backup of the Fortran con.out (byte-identical to a fresh run; it's trustworthy current-config output).
- `heat_python/aw2_full.log`, `aw2_full2.log` — Python run logs. `examples/aw2_tc21/fortran_aw2.log` — Fortran run log.
- To re-run Fortran aw2: `wsl bash -c "cd .../examples/aw2_tc21 && ../../heat gas"`.

## White paper (the eventual writeup)

Once the port is validated and a TPS-surrogate GNN is trained on data it generates, the project culminates in a white paper. This is the longer-term destination for the work — keep its structure in mind when making implementation choices.

**Working narrative.** TPS / heat-shield design needs many evaluations of an expensive coupled physics solver (heat conduction + pyrolysis + gas flow + radiation + surface chemistry). Train a GNN surrogate on data from a trusted reference solver (Martin's Fortran, now ported to Python) so design loops run orders of magnitude faster. Builds on the LAUMeltFlow GNN-surrogate program: bead chain (variable-length MPGNN works), 1D Sod flux (MLP suffices for fixed-topology 2-cell stencil), 2D Sod flux (MPGNN beats MLP when the stencil widens). Heat-shield is the synthesis — variable-resolution 1D mesh, multi-physics per cell, time-varying boundary inputs.

**Suggested sections.**

1. **Introduction.** TPS surrogates motivation, design-loop bottleneck, gap this fills.
2. **Reference solver.** Physics, discretization, and what Martin's code computes. Lean on [docs/heat_TPS/heat_shield_summary.md](docs/heat_TPS/heat_shield_summary.md).
3. **Methodology.** Python port + verification approach (column-level diffs against Fortran's `con.out`, etc.). Training-data generation strategy (state-space sampling vs trajectory sampling — same lesson the meltflow work taught). GNN architecture (per-cell nodes with `[T, ρ, ρᵢ, porosity]`, edge features for interface attributes, K=1 or K=2 message passing).
4. **Results.** Validation on the three reference cases (`aw1`, `aw2_tc21`, `aw2_tc22`). Thermocouple-trace MAE, pyrolysis-front depth, energy-balance residual. Wall-time speedup vs the Fortran. Mesh-resolution generalization (train at one `NBRN`, run at others).
5. **Discussion.** Limitations, extrapolation behavior, failure modes (cf. the bead paper's note about rollout error accumulation).
6. **Future work.** Multi-step (rollout) training, 2D/3D mesh extension, integration into the lab's design workflow.

**What to capture as you go** (so the paper doesn't require excavation later):

- Per-case validation tables: keep Fortran-vs-Python MAE + max-rel for every output column. The `validate.py` output is exactly this — just save it per case.
- Wall-time measurements: log both Fortran and Python run times for each case (the Fortran needed 3 min for `aw2_tc21`).
- Decisions and reasons: each time you choose a port detail (e.g. "rho_face uses pre-pyrolysis rho"), note the Fortran line you matched it to. Useful when reviewers ask.
- Plots: the bead/Sod work has a precedent for paired error-map + 1D-slice figures ([meltflow_gnn/outputs/sod2d_three_way.png](meltflow_gnn/outputs/sod2d_three_way.png)). Same template should work for `aw1` / `aw2_*`.

## Files / paths to know

- Fortran source: [heat_2026-04-11_1837/](heat_2026-04-11_1837/)
- Python port: [heat_python/](heat_python/)
- Per-case data: [heat_2026-04-11_1837/examples/aw1/](heat_2026-04-11_1837/examples/aw1/), `aw2_tc21/`, `aw2_tc22/`
- Reference Fortran outputs: `aw1/con.out` (we generated), `aw2_*/reference/2.x/*.txt` (Martin shipped)
- Python outputs: `aw1/python_out/con.out` (overwritten on each Python run)
- Summary doc: [docs/heat_TPS/heat_shield_summary.md](docs/heat_TPS/heat_shield_summary.md)

## Build / run cheatsheet

```bash
# Fortran (from WSL Ubuntu):
wsl bash -c "cd /mnt/c/Users/glgo230/Projects/LAUMeltFlow/heat_2026-04-11_1837 && make heat"
wsl bash -c "cd /mnt/c/Users/glgo230/Projects/LAUMeltFlow/heat_2026-04-11_1837/examples/aw1 && ../../heat"

# Python (from Windows shell):
python -m heat_python.solver                # runs aw1 by default
python -m heat_python.validate --fortran heat_2026-04-11_1837/examples/aw1/con.out --python heat_2026-04-11_1837/examples/aw1/python_out/con.out
```

## Other context (separate threads, less active)

- Bead MPGNN + meltflow GNN work is in [whip/gnn/](whip/gnn/) and [meltflow_gnn/](meltflow_gnn/). Param sweep done (h=32 sweet spot). 1D and 2D Sod MPGNN ports done. Symposium paper already submitted.
- Draft reply email to Dr. Lau about the bead/meltflow results is in conversation history but not yet sent.
- Reports update planned: `reports/gnn_analysis/gnn_report.tex` needs MPGNN section added; not done.
