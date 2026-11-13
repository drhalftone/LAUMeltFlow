# LAUMeltFlow project context

## Response style
**BE MUCH BRIEFER THAN FEELS NATURAL. Default target: 1-4 sentences.** A few
short paragraphs only when I ask for depth or the answer genuinely needs them.
Efficiency, not omission: keep every load-bearing fact, cut everything else.

- Answer first, in the first sentence. No preamble, no restating my question, no
  "let me check" narration, no recap of what you just did.
- One point per reply. If there are three things, give the one that changes what I
  do and offer the rest in a half-sentence.
- Don't re-explain context I already have (I wrote the handoff, I know the numbers).
- Don't list what you read, how you found it, or what you considered and skipped.
- No summary/verdict section restating the body. No closing offer unless it's a
  real decision I need to make.
- Caveats only when they change the decision. Say the caveat, not a paragraph on it.
- Conceptual/tutorial questions: lead with the answer, expand only if asked. See
  memory `response-style-explanations` for the ground-up style when I do ask.
- Code: change it and say what changed in a line. Don't quote the diff back.

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

## ⇨⇨⇨⇨⇨ HANDOFF (2026-09, STUDY + DOCS SESSION — read this first; adds measured facts that CORRECT claims below)

**Mode this session was "Grey is learning the material," not "ship code."** No model training, no commits. Three documents were built or restructured, several claims below were corrected by measurement, and some intro-level Python teaching happened.

### ⇨ MEASURED FACTS THAT CORRECT OLDER CLAIMS IN THIS FILE

- **Parameter count: 10 learnable, not "about three."** Ran a real backward pass on `models/heat_cond_aw2_fine_n24_relax.pt`. **110,299 allocated, exactly 10 receive gradient** (2 of 54 tensors: `cond_logit` 5 + `pyro_coef` 5). 52 tensors get nothing. Trained values: `cond_logit = [0.713, -9.211, -9.061, 0.032, 0.061]`, `pyro_coef = [-0.0919, 0, 0, 0.312, 3.9e-05]`. **Only 5 move off their init.** The other 110,289 params (node_encoder, geom_encoder, cond_mlp, source_mlp) are **allocated but never called** under `--uniform-cond` — they are structurally bypassed, not merely unimportant. Papers still say "about three gas-coupling parameters"; the defensible claim is "10 of 110,299 allocated, 5 of which move."
- **There is no middle ground at ~50k params.** The 110k is unreachable code, not a network doing partial work. Either gate the MLP construction on `uniform_cond` (bookkeeping, result unchanged) or drop `--uniform-cond` so `cond_mlp` predicts a per-face D from local state (a retrain, untested, the genuinely better science).
- **`pg` conductance fell 0.3 -> 0.032 in training**, a factor of ten, while D (temperature) moved only 0.375 -> 0.336. **The gas channels have no physical reading.** D had a measured anchor; `pg` and `mdotf` were guesses standing in for Darcy flow. §5 of `Goodwin_TPS.tex` should NOT lump all three as "the calibrated conductances" — split the anchored D claim from the numerical gas couplings. `mdotf`'s `pyro_coef` is 3.9e-05, i.e. numerically dead; an ablation to see whether the channel is load-bearing at all is untried.
- **ONLY `Goodwin_TPS.tex` has the wrong energy equation (line ~112), not both papers.** `Goodwin_report.tex` never states a governing equation, so it is fine. The TPS form `rho*cp*dT/dt` is the conventional textbook one but does not describe this solver: there is no `cp` on the LHS (the solver advances enthalpy and recovers T by Newton inversion) and the stored gas energy `phi*rho_g*u_g` is missing entirely.
- **The solver has 3 species, not 5.** The 5 in the checkpoint is the GNN's padded channel layout. From `examples/aw2_tc21/solid.dat`: sp1 (300 -> 0, A=1.2e4, E/R=8556, Tmin=333) light volatiles, sp2 (900 -> 600, A=4.48e9, E/R=20444, Tmin=556) main resin that leaves the char, sp3 (1600 -> 1600, A=0) the inert carbon fiber. Columns are rho_v, rho_c, A, psi, E/R, Gamma, Tmin.
- **Slab is 50 mm** (`CASE_L0 = 5.0D-2`), aw1 = 100 cells, aw2 = 500 cells. `CASE_M = 0` in all three example cases, so `dA = 1` and `dV = dx` everywhere — the cylindrical/spherical generality is never exercised, and the surrogate cannot transfer to curved geometry.
- **Measured reaction-band width:** the 777-920 K window where the main resin goes from "survives the 60 s pulse" to "gone in one second" occupies **1.2 to 2.5 mm** of the slab, widening as the front travels inward, and vanishes by t=96 s when the surface cools below it. At final state the outer ~12 mm is charred and the back three quarters of the slab is untouched (back face ends at 312 K).

### ⇨ ARCHITECTURE HONESTY CHECK (2026-09-24 — answer a reviewer WILL ask; nothing built, conversation only)

**Q: could a CNN have done this instead of a GNN? A: yes, on every project in this program so far.** Grey asked directly and asked not to be flattered. The findings:

- **The mesh is UNIFORM, so edge lengths are CONSTANT.** `domain.py:62-66` sets `dx = l0/n` for every cell and the comment says Martin's stretched-grid option (F != 1) is used by no example case and was never ported. **So "we needed a GNN because edge lengths vary" is FALSE** and anyone who opens `domain.py` will catch it. This is already logged elsewhere in this file as a failure mechanism: the mesh-resolution section says the model learned dx "as a constant, not a variable," which is exactly what a constant feature does.
- **A K=1 MPGNN on a uniform path IS a 3-tap 1D convolution.** Same weight sharing, same locality, same mesh-length independence, ghosts = padding. The bounded conductance update `u_i += sum c_ij (u_j - u_i)` with `c_ij >= 0, sum <= 1` expands on a uniform mesh to the kernel `[c, 1-2c, c]`, i.e. a convolution with its weights reparameterized into the stable region.
- **Every physics-informed ingredient is architecture-agnostic.** Bounded update = a constrained kernel. Anchored D = a constant in that kernel. Real Arrhenius under `no_grad` = pointwise, ignores neighbors entirely. Sub-stepping, BPTT rollout training and ghost-cell forcing = procedure and padding. A constrained 1D CNN gets the same 18.2 K. **Physics-informing makes the architecture matter LESS, not more** — with 10 live params out of 110,299, GNN-vs-CNN is arguing about the packaging of a 3-point stencil.
- **Same verdict for the whip.** `whip/gnn/bead_mpgnn.py` edge feats `[dpos, dvel, neighbor_mass, rest_len]` are all differences between adjacent array entries, and a difference operator is a conv kernel `[-1, 1]`. The chain's GEOMETRY is irregular and deforming but its CONNECTIVITY is a uniform 1D lattice in index space, which is all a CNN requires. **A CNN would be FASTER there** (cuDNN conv vs memory-bound gather/scatter). The 1D Sod result (an MLP sufficed on the fixed 2-cell stencil) is the same finding. The concept that settles all cases: **a CNN needs regular CONNECTIVITY, not regular GEOMETRY.**
- **What would break a CNN and was never done:** self-contact (bead 3 hits bead 40 = a dynamic non-local edge), branching topology, changing connectivity, or a genuinely unstructured mesh.

**⇨ THE DEFENSIBLE CLAIM (use this wording, do not overclaim):** "On this mesh a CNN is equivalent and we would expect the same result. We used a graph formulation because the goal is extending to unstructured meshes, where the CNN has no analogue." The graph costs nothing in 1D and is the honest path forward. **The EE699 motor is the FIRST problem in this entire program where a CNN has no substitute** (unstructured triangles, no lattice to convolve over, and per-face conductance `k_face*A_face/d` genuinely varies per edge because area, distance and `mu(|B|)` all differ). That is the strongest architecture justification Grey has and the first time it is honest.

**⇨ TODO this creates:** `figs/architecture/fcn_alt.png` compares the FCN alternative, NOT the CNN. **A CNN is the stronger alternative baseline and the one a reviewer will actually raise.** Consider a `viz_cnn_alt.py` companion, or at minimum a sentence in the thesis/journal architecture section conceding the equivalence on a uniform mesh. **Trap to avoid in EE699:** the Newton residual loss and the fixed B-from-A operator are ALSO architecture-agnostic, so justify the GNN there by the unstructured mesh, never by the physics-informing.

### ⇨ DOCUMENTS BUILT OR RESTRUCTURED

- **`docs/heat_TPS/Goodwin_equations.tex` / `.pdf` — 17 pp, REORDERED at Grey's request.** New order: (1) **Which Equations the Surrogate Uses** [NEW], (2) **How a Heat Shield Protects** [NEW], (3) Nomenclature, (4) Symbols that are reused, (5) the "This is a working reference" intro, (6) Equations 1-38. The equations moved to the back so the reference tables come first.
  - **"Which Equations the Surrogate Uses"** places all 38 in three groups. **Computed in the forward pass unchanged (4):** Eqs 15-18, all chemistry. **Replaced by the learned bounded update (18):** Eqs 2-14 plus 27, 28, 30, 31, 32. **Upstream only (16):** Eqs 1, 19-26, 29, 33-38 — they shape the training data or arrive as ghost-cell forcing. Closing line: four survive intact, eighteen collapse into one bounded stencil with ten learned parameters, sixteen never appear. Names the gas equations (30-32, mass conservation NOT enforced) as the weakest substitution.
  - **"How a Heat Shield Protects"** is the four-mechanism section, each cross-referenced: char insulates (Eq 23 blended by tau, Eq 20), char re-radiates (`eps*sigma*T_w^4` in Eq 34, emissivity Eq 26), reaction absorbs energy (formation enthalpy Eq 24 appearing as E_chem in Eq 8), pyrolysis gas both cools and blocks (advection Eq 7 via Darcy 27/28, plus the blowing correction Eq 35 which reduces incoming flux rather than removing absorbed heat). Ends with an explicit refusal to rank them without a source.
- **THESIS OUTLINE — `C:\Users\glgo230\OneDrive - University of Kentucky\Thesis\outline.pdf` (OUTSIDE the repo; source .tex is in the session scratchpad and will NOT persist — regenerate it if you need to edit).** 3 pp. Grey's five chapters as agreed with his professor: (1) FEA/FVM/GNN methods and motivation, (2) Martin's code and the problem, (3) the surrogate [professor called it "PINN"], (4) results, (5) conclusion. Carries the problem framing, the audience note ("must be explainable in layman's terms"), what material already exists per chapter, and the known corrections to carry. **Grey asked that nothing else go in that Thesis folder.**
- **`x/learn_self.py`** — intro-level Python teaching file on `self` and classes. Five lesson steps plus a six-task quiz with PASS/FAIL self-checking. `SHOW_LESSON` and `RUN_TASK` switches at the top select one step or task at a time (0 = all, -1 = skip). Task 6 is a hand-written version of the `W = (rho - rho_c)/span` virgin-fraction computation from `gnn_flux_model.py:274`. Grey completed tasks 2, 3, 4, 6a.

### ⇨ FRAMING DECISIONS MADE

- **PIGNN, yes. "PINN" unqualified, no.** Grey's professor wrote "Chapter 3: PINN" meaning physics-informed NN broadly. That is fine and **physics-informed graph neural network is accurate**, because the physics is in the architecture. But avoid the bare word PINN, which in that subfield names the Raissi method and makes a reader look for a PDE residual in the loss that does not exist. One plain sentence saying "informed through the architecture rather than the loss, supervised on solver trajectories" earns the label. This sharpens the Cao contrast rather than hedging it.
- **How to answer "what problem are you solving?"** Lead with the question, not the parameters and not the architecture: *how much of the known physics do you have to build into a surrogate before it can handle a stiff coupled problem?* Then the arc: gas-off works at 7.6 K, gas-on diverges to 100,000 K, so the question becomes what the network has to be told, and the answer is that conduction and chemistry cannot be learned from data in that regime. Do NOT open with "we're using a GNN" — it invites evaluation of the tool instead of the result. Keep the divergence mechanism (square aero pulse, coupled T/char manifold) ready for the follow-up.
- **The four protection mechanisms are distinct and should not be collapsed.** Grey caught a sloppy sentence that credited the gas alone for blocking heat. Char insulates AND re-radiates; the reaction absorbs; the gas both carries heat out and thickens the boundary layer. Do not rank them without a source.

### ⇨ NEW FIGURES AND SCRIPTS (all uncommitted)

- **`heat_python/viz_arrhenius.py` -> `figs/method/arrhenius.png`** — "Pyrolysis is a switch, not a dial." Reaction timescale `1/k` versus temperature with the 60 s aero pulse drawn across it. Reads the real kinetics from `solid.dat`. The 143 K band between "lasts the whole flight" and "gone in one second" is the headline. A first version plotting the bare `exp(-E/RT)` tail was rejected by Grey as conveying nothing, which was correct.
- **`heat_python/viz_fcn_alt.py` -> `figs/architecture/fcn_alt.png`** — the fully-connected alternative drawn to scale: flatten 502x8 into 4,016 inputs, two hidden layers, 1.74M params at width 256. Notes: locked to one mesh size, no locality, no weight sharing.
- **`heat_python/viz_gnn_alt.py` -> `figs/architecture/gnn_alt.png`** — the companion, same layout. The three-cell stencil, the bounded update, 5 learned conductance + 5 learned source, and the 110,289 dead params shown in grey. These two are a matched pair and belong together in Chapter 1 or 3.

### ⇨ STANDING PREFERENCES CONFIRMED THIS SESSION

- **Give full file paths starting `LAUMeltFlow/...`** Flag anything outside the repo with its absolute path, and say explicitly when a file is in the session scratchpad and will not persist. (Saved to memory as `file-paths-full-from-repo-root`.)
- **Brief by default.** Grey pushed back on drift more than once. One-at-a-time when walking equations or symbols.
- **LaTeX and math syntax do not render in the chat.** Use plain-text math.
- **Nothing committed. Still.**

### ⇨ UNRELATED THREADS THAT CAME UP (do not confuse with the TPS work)

- **EE699 final individual project (UPDATED 2026-09-21 — now concrete).** A **physics-informed GNN surrogate for 2D magnetostatic FEA of a spoke-type IPM motor** (16 poles / 18 slots, Formula E, deep saturation), with Dr. Ionel's SPARK Lab supplying the **Ansys Maxwell** model (FEMM is no longer the plan). GNN predicts A_z per node; B per element, flux linkages and torque follow by exact physics. Geometry FIXED, so the mesh is fixed — deliberately dodges the remeshing failure from the ablator work. Baseline to beat is Silva et al. ECCE 2025 (GPR, 12 FEA runs, 2 outputs, fails under strong saturation). **Full context lives in memory: `project_ee699_ipm_gnn.md`.** Two verified IEEE references: **Taran, Ionel, Dorrell, IEEE Trans. Magnetics 2018, 54(11), DOI 10.1109/TMAG.2018.2856858** (Ionel's own surrogate-assisted 3-D FEA optimization — citing it is high leverage) and **Cheng et al., IEEE Trans. Transportation Electrification 2024, 10(4), 8413-8431, DOI 10.1109/TTE.2024.3366417** (a recent review). Nothing built yet; separate from the TPS work.
- **NREL ADVISOR 2003 on MATLAB.** Coursework, not this project. Root cause of all its errors was that **R2024b has no Simulink installed** (R2023b does). `simset` was never removed — it is a P-file at `toolbox/simulink/core/general/simset.p`. All ADVISOR source edits were reverted to the original zip. One deliberate edit remains: six `xlabel('Time (s)')` calls added to `gui/ResultsFig.m`. Also, `restoredefaultpath` + `matlabrc` fixes a broken MATLAB path, and `savepath` must NOT be run (it writes to Program Files, needs admin, and broke the ACL on `pathdef.m` once already); save to `Documents\MATLAB\pathdef.m` instead.

### ⇨ WHERE THE WORK LEFT OFF

Grey is walking the **four protection mechanisms** one at a time, briefly, with equation references. Mechanism 1 (char insulates) and 2 (char re-radiates, Eqs 34 and 26) are done. **Next: mechanism 3, the reaction absorbing energy, Eqs 8 and 24.** Then mechanism 4, the gas.

Still pending from earlier: fix the energy equation in `Goodwin_TPS.tex`; expand its citations to Cao density; the 9-file surrogate code walkthrough (`heat_python/gnn/CODE_MAP.html`); and the open question of whether the journal paper keeps IEEEtran.


---

## ⇨⇨⇨⇨ DELIVERABLES DIRECTION (set 2026-07-31 — this work now feeds THREE outputs; keep ONE canonical source of results/numbers/figures and specialize)

Grey is taking this work to **three deliverables**, all from the same science (don't fork into diverging copies — the older `heat_gnn_ieee`/`heat_gnn_whitepaper` siblings already drifted):
1. **Thesis chapter (superset)** — everything, INCLUDING negatives (flux-form seed study, the temporal-weight null) + the relax experiment. Depth/dead-ends are a feature here.
2. **Journal paper, organized like Cao_2026 (TL-PINN)** — the publishable subset restructured to their template. Maps cleanly: their forward+inverse split ↔ ours (forward = predict field; "inverse" = recovering conduction rate D from data, i.e. the relax/learnable-D model); their sensitivity/robustness results ↔ ours (n8 vs n24, seed study, temporal weighting, relax, speedup); discussion hook = architecture-vs-loss physics-informing (we applied Cao's transient-focus idea). **Recommended to do FIRST** (forces the cleanest full version; the other two derive cheaply). `Goodwin_TPS.tex` is the seed.
3. **Memo to Dr. Martin (Fortran author)** — physics-forward, validation-heavy (bit-exact aw1 / 0.064 K aw2), surrogate framed as "trained on your validated code," minimal ML jargon, short. Build from `docs/heat_TPS/martin_progress_memo.md`.

**⇨ RELAX MODEL (built 2026-07-31, this session) — `heat_python/models/heat_cond_aw2_fine_n24_relax.pt`.** Applied Cao's "focus training on the sharp transient" idea via a new `train_rollout.py --temporal-weight` flag (upweights cells with large per-super-step |ΔT|). Temporal weighting ALONE did nothing (bias +52.7 K, unchanged — the bias lives in the FIXED conduction, which the loss can't reach since the net only controls the ~3 gas-coupling params). But **relaxing the fixed conduction to learnable-around-anchor** (new: `--uniform-cond` now inits `cond_logit` from `--fixed-cond` when both given; `HeatConductanceMPGNN` `has_fixed_cond = fixed_cond and not uniform_cond` so the learnable path wins) **+ temporal weight** → **18.2 K held-out mean (from 25.3), hot-band bias +41.7 K (from +52.8), t=60s cutoff error 33 K (from 45)**, still bounded, char front still tracked. Learned D settled 0.375→0.336 (fine-tuned the anchor, didn't collapse). Train cmd = the n24 cmd + `--uniform-cond --temporal-weight 8`. Figs: `figs/aw2_conductance/aw2_relax_3panel.png` + `.gif`. **Framing: still physics-informed** — chemistry (real Arrhenius) is fixed/non-learnable, the max-principle update is structural, and D is anchored (not free). The learnable coupling = `pyro_coef` for 3 non-char channels: c_T, c_pg, c_mdotf. **NOT a PINN** (physics is in the architecture, not a loss residual; supervised on solver data; lineage = MeshGraphNets, not Raissi). **DECIDED 2026-08-07: relax IS THE HEADLINE** (Grey's call). "Transfer learning" framing: we don't do Cao's staged layer-freezing — we do temporal loss weighting + relaxing the conduction anchor, framed as "the effect Cao reaches with transfer learning, applied as a loss weight not staged freezing." Honest arc for the paper: fixed-D works+stable (25 K, the foundation) → relax + transient-focus training = headline (18 K); the key mechanism = temporal weighting has NO authority until the conduction is relaxed.

**⇨ PAPER STATE (updated 2026-08-07, LATER session — supersedes the block just below; pushed `8ef2d70` still STALE vs working tree).**
- **`Goodwin_report.tex`/`.pdf` (deliverable #3 — RENAMED from `Goodwin_Martin.tex`; old Goodwin_Martin.* build artifacts deleted). DONE, ready to email Dr. Martin.** Reworked this session from a surrogate-forward summary into an **easy-terms narrative arc**: §3 GNN Surrogate now tells the journey — A GNN on the mesh (finite-volume-mesh-IS-a-graph framing) → gas-off aw1 works (7.6 K) → coupled aw2 breaks/diverges (why: square-pulse aero transient) → Building the physics in (fixed-D, 25 K) → Borrowing a transfer-learning trick (relax + transient-focus, 18 K) → Speed. Title = **"A Physics-Informed Graph Neural Network Surrogate for the 1D Heat-Shield Solver"**. §4 = "Training Data and Port Validation" (4.1 Choosing the training data [real trajectories NOT random states → stay on the coupled T/char manifold; forcing sweep; held-out], 4.2 Validation = 1 para + `tab:val`). Added: the **bounded-update equation** (conduction update `u_i ← u_i + Σ c_ij(u_j−u_i)`, `c_ij≥0, Σ≤1` → discrete max principle) with all symbols defined (T, ρ, ρ_i, φ, p_g, ṁ, D); char-front definition (98% of virgin = a *chosen reference threshold*; the β=(ρ_v−ρ)/(ρ_v−ρ_c) "degree of char" formalism + citation kept in back pocket, NOT used — see CHAR paper below). **Bibliography** added (7 refs, `unsrt` = citation order: lu2024/queipo2005/pan2023 intro, meshgraphnets/sanchez2020learning/brandstetter2022 §3.1, cao2026 §3.5). 6 pp, bibtex clean. **NEW figures** (in `docs/heat_TPS/figs/`, gen from `heat_python/`): **`gnn_arch.pdf`** (`viz_report_arch.py` — mesh→graph + 3-channel update [fixed conduction/fixed Arrhenius/learned gas coupling] + transfer-learning strip + color key; the "why it stays stable" beat was removed per Grey); **`aw1_rollout_spacetime_signed.pdf`** + **`aw2_relax_3panel_signed.pdf`** (`viz_rollout.py` NEW **signed error mode**: RdBu_r diverging, `TwoSlopeNorm` centered 0, robust-clipped colorbar [`_nice_ceil(99.5 pct)`] with ▽ extend arrow for the lone t≈0 wall-shock outlier pixel [aw2 −112 is 1 pixel]; solver-front-only on solver panel, both fronts on GNN panel, GNN front on error panel; even colorbar ticks via `_even_ticks`/`_signed_ticks`); **`aw2_contrast_3map_relax.pdf`** (`viz_contrast.py` rewritten to **2 panels** [solver + purely-learned-diverges, dropped the 3rd physics-informed panel]; uses `tight_layout()` NOT `bbox_inches="tight"` so per-panel aspect = the rollout figs' 1.19; included at `0.667\textwidth` so panels render same physical size). UNCOMMITTED.
- **`Goodwin_TPS.tex` (journal paper, deliverable #2, THE canonical one):** relax = headline (prior session). THIS session: **PORT FULLY STRIPPED** (Grey: "nothing about the python port / validated port / bit-exact / faithful data — 'collected training data from a reliable solver' is all that's needed") — abstract, intro, methodology (`Training-Data Generation` now "The reference solver generates the training data"), discussion, conclusion, and Wall Time all scrubbed of port/verify/machine-precision/0.064; **speedup reframed to ~2× vs the reference solver ONLY** (dropped the ~10× vs Python — it needs the port introduced to mean anything; flagged to Grey). 7 pp, compiles clean. **NEXT (NOT started, Grey directed): reorganize like `Cao_2026` (TL-PINN) + heavy citations.** Cao layout mapped → ours: their **forward/inverse/sensitivity/robustness** split ↔ (forward = predict the field [aw1 + fixed-D aw2 25 K]; **INVERSE = recover the conduction rate D from data = the RELAX model 18 K = our headline**; sensitivity = n8-vs-n24 + seeds + temporal-weight; robustness = seed spread). Cao §-order: Intro (dense lit review) / Iverson's physical model + eqs / TL-PINN method / Results(forward,inverse,sensitivity,robustness) / Discussion / Conclusion. Cao is EXTREMELY heavily cited (~100 refs, 2–4 per sentence) — need ~30–40 REAL VERIFIED refs (Grey's rule: verify before cite). Cao PDF at **`docs/heat_TPS/References/Cao_2026.pdf`**. OPEN Qs for Grey: (1) reorg first then citations? (2) keep IEEEtran or switch to a journal class (elsarticle, Cao = Elsevier CACIE)? **UNCOMMITTED** — pushed `8ef2d70` is the OLD fixed-D + full-port version.
- Thesis chapter (#1): not started.
- **Amar et al. 2016 "Overview of the CHarring Ablator Response (CHAR) Code"** (AIAA 2016-3385, doi 10.2514/6.2016-3385; NTRS **20160005889** — the previously-noted 20150021133 was WRONG, it resolves to a different CHAR paper). NOW IN BIB as `amar2016char`. — gives the canonical **degree-of-char β=(ρ_v−ρ_s)/(ρ_v−ρ_c)** (Eq. 26) and confirms material ≈ **TACOT (ρ_v=280, ρ_c=220 → our rhov_bulk=280)** + the per-species Arrhenius form matches our `pyrolysis.py`. Download: https://ntrs.nasa.gov/citations/20150021133 . Suggested bibtex key `amar2016char`. Kept in back pocket for the char-front threshold justification (thesis/journal), deliberately NOT cited in the memo.

**⇨ COMMITTED + PUSHED (2026-07-31):** `8ef2d70` "Add Goodwin_TPS conference paper" — ONLY the 6 paper files (Goodwin_TPS.tex/.pdf, heat_gnn_refs.bib, figs graph/aw1_rollout_spacetime/why_aw1_aw2/flux_aw2_compare/aw2_contrast_3map). Pushed to origin/main (drhalftone/LAUMeltFlow). **This is the OLD fixed-D + full-port version — the working-tree Goodwin_TPS is now ahead (relax headline + port trimmed) and NOT yet committed.** Everything else (models, scripts, the Symposium→whip/symposium reorg, the relax figs, the Martin report) is STILL UNCOMMITTED. Added ref: `cao2026` (UK Civil Eng, TL-PINN) cited in §4.3 + Discussion + methodology. New viz scripts (uncommitted): `viz_cond_arch.py`, `viz_approach_compare.py` (PINN-vs-ours explainer), `viz_contrast.py` (the 3-model contrast; `--phys` arg picks the model), `viz_rollout.py` now draws char fronts. New figs (uncommitted): `aw2_contrast_3map_relax.png`, `aw2_relax_3panel.png`/`.gif`.

---

## ⇨⇨⇨ HANDOFF (state as of 2026-07-20 — the FINAL n24 model + a real SPEEDUP result on top of the 2026-07-14 block below)

**THIS SESSION: trained the FINAL n24 aw2 model + measured a genuine speedup. aw2 now "works AND saves compute."**

**⇨ FINAL MODEL — `heat_python/models/heat_cond_aw2_fine_n24.pt`.** Same physics-informed conductance recipe as phys3, but trained on **24 forcings** (`aw2_fine_dataset_n24.npz`) instead of 8. Held-out result is **statistically identical to phys3** (25.3 K mean, 8.7 K at t=120 s, max 109 K, bounded, nonlinear) — the point being it does NOT degrade going n8→n24, so the surrogate is **not overfit to a lucky small training set**. This upgrades the paper claim from "works on one held-out case" to "generalizes across a swept forcing range." **This is the headline aw2 model now** (use it over phys3). Exact train cmd: `train_rollout.py --data aw2_fine_dataset_n24.npz --init "" --case-dir .../aw2_tc21 --conductance --physical-source --real-kinetics --fixed-cond 0.375,0,0,0.3,0.3 --substeps 8 --rollout-steps 2 --stride 16 --loss-weight 8 --epochs 24 --batch-size 128 --lr 3e-4`. Rollout npz: `data/rollout_cond_aw2_fine_n24.npz`. Figure: `figs/aw2_conductance/aw2_physics_n24_3panel.png` (GNN map ≈ solver map, char fronts ~1 mm apart).

**⇨ SPEEDUP MEASURED (real apples-to-apples, this session) — aw2 IS faster, ~10×.** For the 120 s held-out case: **GNN 92 s (GPU) / 100 s (CPU)** vs **Python solver 925 s (measured, 15.4 min)** vs **Fortran ~180 s (documented)**. So ≈**10× over the Python solver it was trained on, ≈2× over compiled Fortran**, on CPU or a single GPU. The ~25× step-count reduction wins even with 8 sub-steps/super-step (52k total model evals): each GNN eval is far cheaper than the ~200 solver steps it replaces. This BEATS the aw1 speedup (aw1 was ~4× vs Python, break-even vs Fortran). **Honest caveats to carry into the paper:** (1) GPU barely helps at this scale (92 vs 100 s) — the 502-cell graph is small so 52k sequential kernel launches are launch-bound; the real GPU payoff is *batching many design cases at once*, untested. (2) The solver-timing run reproduced the stored held-out field only to 65 K max (I guessed t_on slightly wrong), so it's NOT the byte-exact held-out forcing — but the timing is valid anyway because the solver's step count is set by dt(~9.2e-5 s)·t_final(120 s)≈1.3M steps, i.e. forcing-INDEPENDENT. Timing scripts (scratchpad, not committed): `time_n24.py` (GNN rollout), `time_solver_holdout.py` (solver). To nail a byte-exact number for the paper, re-find the held-out t_on (peak=1.4e6, t_off=70, t_final=120; only t_on unknown) and re-run `time_solver_holdout.py`.

**⇨ PAPER FRAMING (settled this session — how to write aw2):** frame it as a finding about *how much physics structure a surrogate needs*, NOT "our GNN cracked aw2." Spine: *a stiff coupled multiphysics case defeats a purely-learned GNN, so we structure the reference solver's own physics in and let the net learn only the residual gas coupling; that makes it work AND run ~10× faster.* aw1 (gas-off) and aw2 (gas+aero) become two points on a spectrum: aw1 a mostly-learned GNN suffices (7.6 K); aw2 needs the physics structured in. Contributions change from "port / aw1 surrogate / aw2 DIAGNOSED FAILURE" to "port / aw1 surrogate / **a physics-informed surrogate for the coupled case + the finding that structure is required + a real speedup.**" Caveats that MUST be in the text (Grey's rule = separate claims from facts): (a) physics-informed NOT pure-learned — GNN learns ~3 gas-coupling params, conduction rate + kinetics are anchored to the solver; (b) the **hot-corner overshoot** — hot-band (T>800K) signed bias is **+53 K**, i.e. the surrogate runs slightly HOT right at the aero surface near t≈55-60 s (the neon-green sliver in the 3-panel fig); (c) keep speedup honest (GPU near-parity at this scale; batching is the bigger untested lever). NEW study figure this session: `figs/architecture/model_conductance_arch.png` (gen `heat_python/viz_cond_arch.py`) — the physics-informed architecture one-pager (3 channels: fixed conduction / fixed Arrhenius / learned gas coupling → sum → 8 substeps; "why it cannot blow up" max-principle box; fixed-vs-learned box; NOT in the paper, a study/explainer aid).

**⇨ NEXT STEPS (nothing started without Grey's go):** (1) write the physics-informed aw2 result into `Goodwin_TPS.tex` — rewrite the aw2 section from "diagnosed failure" to the success-with-a-lesson framing above, update abstract + contributions, add n24 held-out table + speedup + the 3-panel figure; Grey does this sentence-by-sentence in his voice. (2) optional: nail the byte-exact solver timing (find held-out t_on). (3) optional: relax `--fixed-cond` to learnable-around-anchor. (4) commit when Grey says.

**⇨ STILL NOTHING IS COMMITTED — Grey said commit LATER (holds through this session too).** New uncommitted this session: `heat_cond_aw2_fine_n24.pt`, `rollout_cond_aw2_fine_n24.npz`, `figs/aw2_conductance/aw2_physics_n24_3panel.png`, `figs/architecture/model_conductance_arch.png`, `heat_python/viz_cond_arch.py`, plus `n24_train.log`/`solver_holdout_time.log` — on top of everything uncommitted from prior sessions.

---

## ⇨⇨ HANDOFF (state as of 2026-07-14 — supersedes the "aw2 = diagnosed failure" verdict everywhere below; the 2026-07-20 block above adds the final model + speedup)

**BREAKTHROUGH THIS SESSION: aw2 now has a WORKING surrogate (~19 K held-out mean, bounded, nonlinear).** This supersedes every "aw2 does not generalize / diagnosed open failure" statement in the older handoff blocks below. The win came from a new **physics-informed conductance-form GNN**, not the old flux/direct models.

**The architecture** — `heat_python/gnn_flux_model.py`, class **`HeatConductanceMPGNN`** (built this session):
- **Conductance-form flux with a discrete MAXIMUM PRINCIPLE.** Each update is a bounded convex combination of a cell and its neighbors (per-face conductance capped at 1/degree = 0.5), so the field can *never* blow up — unconditionally stable at any timestep. Kills the checkerboard/anti-diffusion divergence the free flux model had. Still conserves + zero-baseline.
- **FIXED physical conductance** (`--fixed-cond 0.375,0,0,0.3,0.3` for T,rhoi0,rhoi1,pg,mdotf). D=0.375 = measured diffusion number (~3/super-step) ÷ substeps(8). The diffusion rate is *anchored*, not learned, because a short BPTT window is **accumulation-blind** and always collapses to under-diffusion.
- **Internal SUB-STEPPING** (`--substeps 8`): one data super-step = 8 bounded convex sub-steps (reach), forcing held constant across them. Wired in both `train_rollout.mesh_step`-caller loop and `eval_rollout_traj` loop; stored in ckpt.
- **REAL per-species Arrhenius pyrolysis source** (`--physical-source --real-kinetics`): the char channels decompose using the solver's EXACT DECOMP kinetics (A/E/psi/T_min from the material file, mirrors `pyrolysis.step_species`), computed **under `torch.no_grad()`** (the W^(1-psi) term NaNs through BPTT otherwise, and it has no learnable params anyway). The net learns only the T/gas coupling to the total decomposition rate (`pyro_coef`, ~3 params).
- Trained via `train_rollout.py` (BPTT M=2, `--stride 16` to subsample the overlapping fine-cadence windows, `--loss-weight 8` on hot cells).

**⇨ THE BUG that caused this ENTIRE session's struggle** (many hours of "frozen front / under-diffusion" symptoms): the conductance flux uses **node_std-normalized** state differences, but the delta was denormalized by **y_std** (the tiny per-step delta std, ~a few K) instead of node_std (~248 K). That throttled effective diffusion **~30×** — the thermal front could never penetrate no matter what D, loss-weight, or reach was used. Fix = one block in `train_rollout.py` (`if args.conductance: stats[5] = node_std_out`). Took it from **178 K (frozen 6-cell boundary layer) → 19 K (correct ~100-cell layer)**. Several earlier "diagnoses" this session (frozen front, accumulation-blind loss, uniform-cond collapse) were *symptoms* of this one scaling bug.

**⇨ FRAMING for the paper (important):** this is a **PHYSICS-INFORMED** surrogate — conduction and pyrolysis are anchored to known physics; the GNN learns the gas coupling + provides the stable differentiable rollout. A **pure learned GNN DIVERGES on aw2** (direct per-cell model → 141,000 K; free flux-form → checkerboard to ±10,000 K, both confirmed this session). So the real, honest contribution is: *a stiff coupled multiphysics case (gas + aero) needs its known physics structured into the surrogate; then the GNN handles the rest and it works (19 K)* — a cleaner story than pretending an unstructured GNN cracks aw2. (aw1, gas-off, still works as a more-learned model at 7.6 K.)

**⇨ KEY FILES/MODELS (this session):**
- `heat_python/models/heat_cond_aw2_fine_phys2.pt` — the confirmed working 19 K model (fixed cond + *crude* structured source; char-front onset ~800 K a bit high).
- `heat_python/models/heat_cond_aw2_fine_phys3.pt` — **real-kinetics version, CONFIRMED WORKING: ~25 K overall mean (8.7 K at t=120 s), char front now tracks the solver to ~1.4 mm** (was ~7 mm off with the crude gate). Fixed char onset to the real ~500/700 K. Figure: `figs/aw2_conductance/aw2_physics_realkinetics_3panel.png` (char fronts nearly coincident). First real-kinetics attempt NaN'd (W^-2 gradient through BPTT); fixed by computing the kinetics under `torch.no_grad()` (they have no learnable params). This is the best/most-faithful aw2 model — use it as the headline unless the n24 run beats it.
- Data: `heat_python/data/aw2_fine_dataset.npz` (n8, fine cadence record-every 200) + `aw2_fine_dataset_n24.npz` (n24, for the final run). Held-out: `aw2_holdout_fine.npz` (fine-cadence, peak 1.4 MW t_off 70 s). Rollout npzs: `rollout_cond_aw2_fine_*.npz`.
- Figure: `heat_python/figs/aw2_conductance/aw2_physics_rollout_3panel.png` (the working result — GNN map ≈ solver map). Gen with `viz_rollout_anim.py --with-solver --still`.
- NEW script: `heat_python/make_aw2_fine.py` (parallel fine-cadence dataset builder, 28-core).
- New `train_rollout.py` flags: `--conductance --uniform-cond --physical-source --fixed-cond --real-kinetics --substeps --stride --loss-weight`. Same conductance flags in `train_gnn.py`/`eval_rollout_traj.py`. All model config (substeps, fixed_cond, kinetics, etc.) is saved in the ckpt so eval auto-reconstructs.

**⇨ PORT RE-VALIDATED + A SOLVER BUG FIXED (earlier this session):** re-ran aw1 (bit-exact, ~5e-8 K) and aw2 (0.064 K) FRESH from current code. Found + fixed a real regression: `solver.py` con.out generation **crashed** (`record_every=None` → `% None`) and truncated the output file first — introduced by the `record_dt` feature, hidden because the GNN pipeline always passes `record_every` and **there are no tests**. Fixed at `solver.py:~547` (guard the snapshot computation on `rec`). The documented `python -m heat_python.solver` command works again.

**⇨ NOTHING IS COMMITTED — Grey said commit LATER (this session).** Everything is working-tree: the entire `gnn_flux_model.py` conductance/pyro architecture, `train_rollout.py` + `train_gnn.py` + `eval_rollout_traj.py` flag wiring, `make_aw2_fine.py`, the `solver.py` bug fix, `figs/aw2_conductance/`, plus all the older uncommitted paper/fig/seed-study work from prior sessions.

**⇨ NEXT STEPS:** (1) re-eval `phys3` (real kinetics) — confirm char front now matches the solver (onset was the only residual; T field already 19 K). (2) Train the **final model on `aw2_fine_dataset_n24.npz`** (24 forcings → generalization) + regenerate the clean 3-panel figure. (3) Optional: now the scaling bug is fixed, try RELAXING `--fixed-cond` to a learnable conductance (around the physical anchor) to make it a more genuinely-learned GNN. (4) Write the physics-informed aw2 result into `Goodwin_TPS.tex` (replaces the "diagnosed failure / future work" aw2 section). (5) THEN commit when Grey says.

---

## ⇨ HANDOFF (state as of 2026-07-02 — prior-session paper-writing context; aw2 verdict below is SUPERSEDED by the block above)

**📍 DOCUMENT MAP — which file is what (read this to navigate):**
- **`CLAUDE.md` (this file)** — THE authoritative handoff + project context, auto-loaded every session. **Start here.**
- **`docs/heat_TPS/Goodwin_TPS.tex` / `.pdf`** — ⭐ **THE CANONICAL PAPER, actively being written/edited.** IEEEtran, 6 pp, compiles clean. Grey is editing it sentence-by-sentence in his own voice. This is where the active work is.
- **`docs/heat_TPS/paper_spine.md`** — writing worksheet (one-question-per-section scaffold with `(you have: …)` hints). Grey's fill-in guide for drafting.
- **`docs/heat_TPS/heat_gnn_ieee.tex` / `.pdf`** and **`heat_gnn_whitepaper.md`** — OLDER siblings that `Goodwin_TPS` descends from. Kept in sync earlier this session but `Goodwin_TPS` is now the one that matters. (Eventually retire/merge these.)
- **`docs/heat_TPS/gnn_model_lecture_notes.md`** — perceptron→GNN tutorial notes (Grey's study reference).
- **`docs/heat_TPS/{whitepaper_outline,heat_shield_summary,martin_progress_memo}.md`** — supporting reference docs.
- **`ONBOARDING.md` — STALE (April 2026, the whip/Symposium subproject). IGNORE for the heat-shield work.**
- Active project is the **heat-shield GNN** (this handoff). whip/meltflow/Symposium/electrostatic dirs are older/separate.

**⇨ RESUME HERE — WE ARE WRITING THE PAPER (`Goodwin_TPS.tex`).** The science is done and diagnosed; current work is turning it into a polished conference paper. Grey (Master's student, first mostly-solo white paper) goes **sentence by sentence**: he pastes a sentence, wants the current version + 1-3 rewrite options, picks one. He also had me do a full style pass (see prefs below). As of 2026-07-08 the **abstract + intro + Reference-Solver section are thoroughly edited and cited**, a **whole-paper semicolon/colon/run-on cleanup is done**, contributions are framed as **three** (port / aw1 surrogate / aw2 diagnosed failure), and Eq. 1 (the charring-ablator energy equation) now has a "where …" symbol glossary and is attributed to the reference solver (NO citation — we pulled the unverified `mazzaracchio2010` cite off it because I hadn't read that paper's full text). **PICK UP:** continue refining sentence-by-sentence wherever Grey points, or offer to draft/shape a section in his voice (he calls this "mode B"). **NOTHING IS COMMITTED — Grey said hold off; everything is working-tree.** **RECOMPILE after each batch of edits** (`pdflatex; bibtex; pdflatex; pdflatex` in `docs/heat_TPS/`) — Grey checks the PDF.

**⇨ TEACHING / EXPLANATION ASIDES (Grey is learning the material as he writes).** He asks conceptual questions (what is a thermocouple, PDE refresher, why the model fails, FCN↔GNN). Answer plainly, ground-up, define every term (see memory `response-style-explanations`). Two study figures were built this session, both in `figs/architecture/` (study aids, NOT in the paper): `model_onepager.png`/`fcn.png` (existing) and **`model_6steps.png` (NEW, gen: `heat_python/viz_6steps.py`)** — the 6-step forward pass (encode cell / encode edges / concat / update / readout / Euler step) with a **worked example using REAL trained weights** from the aw1 direct model (`heat_mpgnn_rollout.pt`): a real normalized cell state through the actual 64×6 encoder first layer (3 of 64 units shown). It's the aw1 DIRECT model (matches the 6-step diagram). The aw2 flux-form now has its OWN diagram: **`model_6steps_flux.png` (gen: `heat_python/viz_6steps_flux.py`)** — the flux-form steps (encode cells / encode face geom / directed half-flux ψ / antisymmetric face flux Φ=ψ(s,n)−ψ(n,s) / local source / divergence+step), with a REAL-weights worked example from `heat_flux_aw2.pt` that numerically demonstrates the two by-construction guarantees: conservation (Φ(nbr→hot) = −Φ(hot→nbr), sum 0) and zero baseline (identical cells → Φ = 0 exactly). Both study figures are gitignored-free working-tree, not in the paper.

**⇨ GREY'S EDITING PREFERENCES (apply these to every paper edit):**
- **No prose semicolons; no colons that join two independent clauses.** Split into sentences. Colons that INTRODUCE a genuine list are OK (he kept two).
- **Split run-ons.** One idea per sentence.
- **Plainer wording, less jargon.** Define a specialist term only if a smart reader outside the subfield would stall (he accepted a 5-word gloss on B-prime; left Darcy/blowing/wall-enthalpy bare).
- **Lists → First/Second/Third prose** (matches his contributions paragraph), not semicolon lists.
- **Every claim is either cited (abstract-verified) or shown by our results.** No invented precision — he caught "hundreds of times" (unsupported) → changed to "many times over" (Pan's actual words). Verify a cite before leaning on it.
- **No em dashes. Minimize hyphens to source terms.** Honest scoping (don't imply our cheap 1D solver is the "expensive" one).
- Papers use `heat_gnn_refs.bib`. All 8 keys verified real EXCEPT `own_symposium` = **TODO placeholder stub** (Grey's bead/Sod symposium paper — fill it, then `\cite{own_symposium}` in the contributions/flux-form spot).

**⇨ AW2 VERDICT — DIAGNOSED OPEN PROBLEM (corrected AGAIN 2026-06-30; supersedes the "more data makes it reliable" framing above/below).** The conservative **flux-form** kills the *direct model's* catastrophic 25,434 K divergence — that structural win HOLDS (`heat_python/gnn_flux_model.py`, `HeatFluxMPGNN`: antisymmetric face flux + local source → conservation + zero baseline BY CONSTRUCTION). **BUT no flux variant is physically faithful, and the char-front metric (NEW) is the cleaner, more honest story than the temperature mean:**
- The temperature MEAN flatters (it's dominated by the easy cool interior). n24's low ~420 K mean HIDES a near-surface runaway to ~3100 K + a char front that falsely reaches the back face.
- **Char-front tracking (over the whole rollout, vs solver front at ~31 mm):** n8 best 0.8 mm (seed6), n24 best 1.5 mm (seed7), but **only 2/8 (n8) and 3/8 (n24) seeds track the front to within a few mm; 5/8 either way drive the front to ~0 mm (false full-char to the back face).** So a faithful char front is the EXCEPTION at both dataset sizes.
- **⇒ "More data makes it reliable" was too generous.** Correct framing: more data removes the catastrophic MEAN blow-ups (n8 3/8 diverge → n24 0/8) but does NOT fix the char front or make the field faithful. **aw2 = diagnosed open problem; the gas-OFF aw1 case (7.6 K) is the surrogate that actually works.** Also NEW failure mode found: T and char species are **independent output channels with no enforced coupling**, so the model can advance the char front into cells it fails to heat (physically impossible). Both papers now carry the char-front table + this framing.

**⇨ THE 2026-06-23 SEED STUDY (this is the corrected, definitive aw2 result — supersedes the earlier 3-run "more data hurt" claim, which was small-sample noise):**
- **Single-step flux is INIT-SENSITIVE; the famous 120 K was a LUCKY seed, not typical.** Training is unseeded, so rollout stability varies run to run. Added `train_gnn.py --seed`. Ran **8 seeds each** on 8-traj vs 24-traj (`seed_sweep.sh`), held-out mean |err|:
  - **8-traj: mean 850 K, range 155–2312 K, 3/8 diverge (>1000 K).** Heavy-tailed, unreliable. The 120 K (`heat_flux_aw2.pt`) is the extreme good end.
  - **24-traj (n24): mean 419 K, range 203–543 K, 0/8 diverge.** ~half the mean, ~6× tighter, no blow-ups.
- **⇒ MORE DATA REMOVES BLOW-UPS BUT DOES NOT MAKE IT FAITHFUL (see AW2 VERDICT above — this bullet's original "more data makes it reliable" claim was itself later corrected by the char-front metric).** The 8-seed sweep DID reverse the earlier 3-run "more data hurt" artifact (those 3 n8 runs landed on the good tail 120/291/412; the 3 n24 runs landed high 1771/440/836). So on the TEMPERATURE MEAN, n24 is tighter + no mean-blow-ups. But the char-front metric then showed n24 is still physically wrong (5/8 false-char). So: n24 is the least-bad on the mean, but NOT a faithful/working model. Don't headline it as a success.
- **Ensembling HURT.** Averaging the 3 n8 models' per-step predictions (`eval_ensemble_traj.py`, NEW) diverged to **840 K** — closed-loop averaging drifts off-manifold. Rollout ≠ one-shot prediction.
- **BPTT overfits** the thin 8-traj sweep (M=2→701, M=4→757 K; models `heat_flux_aw2_ms{,2}.pt`). **Retrying BPTT on n24 is the top untried lever** (n24 single-step is the reliable set, so BPTT may now help instead of overfit).
- **Loss-weighting the surface BACKFIRES.** Added `train_gnn.py --loss-weight` (upweight large-ΔT cells to fix surface bias). `bias_sweep.sh` (3 recipes × 3 seeds on n24): λ=8 diverged 2/3, λ=8+2×capacity diverged 3/3. Confirms bias-vs-stability tension. Also: the surface bias is NOT one-directional — n24 models OVERSHOOT the hot band (+~400 K signed), only the lucky n8 model undershot. So there's no uniform bias to fix.
- **Loss-weighting scan** used `bias_sweep.sh`; the char-front metric came from re-training all 16 seeds via `seed_stills.sh` (NEW) and computing per-seed front error.
- **Takeaway:** the residual is closed-loop cooldown-tail stability + the physically-unfaithful field. **Ensembling / BPTT-on-thin-data / loss-weighting all fail.** Untried future work: stability-aware loss (penalize rollout error growth), rate-prediction + integrate (Δt explicit), BPTT on n24, adaptive cadence on a fair held-out, and enforcing the T↔char coupling (the decoupled-channel bug).

**⇨ CHAR-FRONT VIZ + FILES (NEW 2026-06-30):**
- `heat_python/viz_rollout_anim.py` now: `--with-solver` (3-panel: profile + Fortran-solver map + GNN map), `--still` (one completed-rollout PNG, no gif), `--filmstrip`, and a **solver-pinned color scale** (`vmax = solver max`; GNN divergence renders as neon-green off-scale `set_over`, colorbar `extend="max"`). This fixed a real bug: auto-scaling to `max(solver,GNN)` made the SAME solver panel look different across gifs because a diverging GNN stretched the bar. Now the solver panel is identical everywhere and green = diverged.
- `heat_python/eval_rollout_traj.py` now saves `rho_pred` and prints a hot-band (T>800K) signed-bias metric.
- `heat_python/seed_stills.sh` (NEW) = train all 16 seeds → completed 3-panel stills. Output: `figs/aw2_diverged/n8/seed0..7.png` and `figs/aw2_diverged/n24/seed0..7.png`. `figs/best/aw2/` = the bounded n8 best-of-8 (`aw2_flux_rollout{,_3panel}.gif` + frames + `aw2_flux_vs_direct.png`).
- Best single models by char front: **n8 seed6** (front 0.8 mm, meanT 167 K) and **n24 seed7** (front 1.5 mm, meanT 276 K). `heat_flux_aw2.pt` = the lucky n8 120 K seed (best case shown in papers WITH the "best of 8, 3/8 diverge" caveat).

**⇨ NEXT STEP — FINISH THE PAPER (`Goodwin_TPS.tex`).** Science + diagnosis are done; both papers carry the 8-seed table + char-front table + all negatives + the decoupled-channel failure mode. Active work is **writing/polishing `Goodwin_TPS.tex` sentence-by-sentence in Grey's voice** (abstract + intro done + full style pass done; continue through any sentences Grey flags, or draft sections in "mode B"). **Then commit** (Grey said hold off this session — EVERYTHING below is uncommitted working-tree: paper edits across Goodwin_TPS/heat_gnn_ieee/whitepaper, `paper_spine.md`, the 16 seed stills + figs, viz + eval + train code changes, `seed_stills.sh`/`bias_sweep.sh`). Deferred research options if Grey returns to experiments: retry BPTT on n24, stability-aware loss, or broaden to mesh-resolution generalization (still open — see below).

**Prior thread (still open):** mesh-resolution generalization (multi-res training run never done). aw2 was previously logged as a diagnosed-failure "limitation" result; the flux-form now supersedes that as the active fix.

**Unrelated aside:** `interactive_curve_tracer.html` in the repo root is courseware Dr. Lau sent without context (an EE461g BJT/MOSFET curve-tracer sim). NOT part of this project — ignore it. (Now gitignored.)

### ⇨ AW2 GAS SURROGATE — diagnosed FAILURE CASE (2026-06-18). **SUPERSEDED 2026-06-19 by the flux-form fix (see RESUME HERE): bounded ~104 K held-out. Text below is the pre-fix diagnosis, kept for context.**
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
  `python -m heat_python.gnn.data_gen --case-dir .../aw1 --no-gas --record-every 5 --out heat_python/data/aw1_traj.npz`
- **Stage 2 — `heat_python/graph.py`**: trajectory → GNN dataset `.npz`. Nodes = all m=n+2 cells (incl. 2 ghosts carrying forcing); edges = 1D path (bidirectional), edge_attr = interface spacing. Node features `[T, rho, rho_i…, porosity]`; target = interior next-state (+ `target_delta`); normalization stats (std clamped for the inert/constant species). `--gap` sets the super-step.
  `python -m heat_python.gnn.graph --traj .../aw1_traj.npz --gap 1 --out .../aw1_pairs_gap1.npz`

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

1. **Fix rollout drift (the active task).** See the "Stage 4" block above. Start with **noise-injection / push-forward training** in `train_gnn.py` (add noise to inputs so the model corrects its own errors), then **multi-step rollout loss**. Re-measure with `python -m heat_python.gnn.archive.eval_rollout` after each change — rollout error (not 1-step val loss) is the metric. Target: keep end-of-trajectory mean error to ~single/low-double-digit K.
2. Then: sweep super-step `--gap`, try K=2, more trajectories, then move to gas cases (aw2) and mesh-resolution generalization.

**Lower priority / deferred (port side — essentially done):**
- Save validation tables + wall-times to `docs/` for the paper (aw1 bit-exact; aw2_tc21=tc22 to 0.06 K; Fortran ~3 min vs Python ~15.6 min). `validate.py` output is the table.
- Ask Martin about aw2_tc22 (byte-identical to tc21 — only one independent aw2 case exists).
- Optional: `radiation.py` (laser/`-r` only); `main.py` polish.

**STATUS: Port DONE (aw1 bit-exact; aw2 to 0.06 K). GNN pipeline DONE end-to-end (data → train → rollout). Current blocker: the trained surrogate drifts on rollout (error accumulation) — fixing that is the next real work.**

### GNN pipeline files

**⇨ REORGANIZED 2026-08-19: all 18 model/pipeline scripts now live in `heat_python/gnn/`** (the solver port stays flat in `heat_python/`). Moved: `gnn_model, gnn_flux_model, graph, train_gnn, train_rollout, train_adaptive_rollout, eval_rollout, eval_rollout_traj, eval_adaptive, eval_ensemble_traj, data_gen, make_dataset, make_aw2_dataset, make_aw2_adaptive_dataset, make_aw2_fine, make_aw2_holdout_traj, make_volume_dataset, measure_speedup`. **Commands gained a segment: `python -m heat_python.gnn.<module>`** (solver-side commands `heat_python.solver` / `.validate` are UNCHANGED). Inside `gnn/`, imports of solver-port modules use `..` (e.g. `from ..solver import run`); imports within `gnn/` stay `.`. `.sh` sweep scripts and the commands in this file are updated. Checkpoints are unaffected (they store `state_dict`, not pickled classes). `viz_*.py` deliberately left flat. All 18 verified importable + `--help` smoke-tested after the move.

**⇨ SUPERSEDED SCRIPTS ARCHIVED (same session) to `heat_python/gnn/archive/`** (8 files, run as `python -m heat_python.gnn.archive.<module>`): `make_volume_dataset` (volume sampling, DIVERGED), `eval_ensemble_traj` (ensembling, DIVERGED 840 K), `train_adaptive_rollout` + `eval_adaptive` + `make_aw2_adaptive_dataset` (adaptive-cadence line, superseded), `make_aw2_dataset` (coarse n8, superseded by `make_aw2_fine`), `eval_rollout` (aw1-only, superseded by `eval_rollout_traj`), `measure_speedup` (early aw1 ~4x timing, superseded by the n24 runs). **Do NOT delete these — the volume-sampling and ensembling failures are cited results in `Goodwin_TPS.tex`.** `archive/README.md` documents each one. Imports there go three levels up to the solver port (`from ...solver import run`) and two levels up to the live gnn modules (`from ..graph import ...`). **LIVE pipeline is the 10 files remaining in `gnn/`:** `data_gen → make_aw2_fine → graph → train_gnn → train_rollout → eval_rollout_traj`, plus `gnn_model`/`gnn_flux_model` (the networks), `make_dataset` (aw1), `make_aw2_holdout_traj`. **Gotcha: `train_gnn.py` looks superseded by `train_rollout.py` but is NOT — `train_rollout` imports `select_out_cols` from it.**

Filenames below are as listed; prefix with `gnn/` for the moved ones, `gnn/archive/` for the superseded ones.
**aw1 (gas-off) core:** `data_gen.py` (solver→trajectory .npz) · `make_dataset.py` (aw1 forcing-sweep multi-traj) · `graph.py` (build_dataset, to_neighbor_samples, **build_adaptive_dataset** [mixed-cadence + dt feature], gas-aware node_features) · `gnn_model.py` (HeatMPGNN) · `train_gnn.py` (one-step + `--noise`; **select_out_cols** by name, `--no-pred-gas`) · `train_rollout.py` (multi-step BPTT; now column-agnostic `mesh_step`+`_Cols`, single-traj support, `--case-dir`, `--no-pred-gas`) · `eval_rollout.py` (aw1 rollout vs solver; `--nbrn`/`--record-dt`).
**aw2 (gas) additions (2026-06-17):** `make_aw2_dataset.py` (flux-sweep gas dataset) · `eval_rollout_traj.py` (generic trajectory rollout, aw1+aw2) · `eval_adaptive.py` (adaptive-cadence rollout) · `train_adaptive_rollout.py` (dt-aware multi-step) · `measure_speedup.py`. `solver.run` now takes `nbrn=`, `record_dt=`.
**Key aw2 models:** `heat_flux_aw2_n24.pt` (**HEADLINE aw2**, flux-form, 24-forcing seed 0, ~478 K, reproducible) · `heat_flux_aw2.pt` (8-traj flux, the lucky 120 K seed) · `heat_flux_aw2_{r2,r3,n24_r2,n24_r3}.pt` (seed-study artifacts) · `heat_flux_aw2_ms{,2}.pt` (BPTT, overfit/worse) · `heat_mpgnn_aw2_adaptive_noise.pt` (old direct-model adaptive, 784 K). **Key data:** `aw2_holdout_traj.npz` (the held-out forcing all aw2 rollouts test on) · `aw2_forcing_dataset.npz` (8-traj) · `aw2_forcing_dataset_n24.npz` (24-traj, reliable set) · `aw2_traj{,_coarse}.npz`, `aw2_adaptive.npz`. Data in `heat_python/data/`, models in `heat_python/models/`.
**Seed/sweep drivers (NEW 2026-06-23):** `seed_sweep.sh` (8-seed held-out spread, 8-traj vs n24) · `bias_sweep.sh` (loss-weight × capacity × seed) · `eval_ensemble_traj.py` (multi-model per-step averaging). `train_gnn.py` now has `--seed` and `--loss-weight`. `eval_rollout_traj.py` now saves `rho_pred` and prints a hot-band (T>800K) signed-bias metric.
Viz: `viz_rollout.py` (space-time + profile gif) · `viz_rollout_anim.py` (**NEW** animated rollout: profile-with-trail + filling space-time map + char fronts; `--with-solver` adds a 3rd solver-truth panel; `--filmstrip` dumps frames) · `viz_compare.py` (thermocouple + error-vs-time + char_front_compare → `figs/compare/`) · `viz_model.py` · `viz_all.py` (regenerates all sweep rollouts).
**figs/ layout (reorganized 2026-06-23):** `best/aw1/` (headline aw1 = direct multi-step M4) · `best/aw2/` (headline aw2 = flux-form n24: `aw2_flux_rollout{,_3panel}.gif` + frames + `aw2_flux_vs_direct.png`) · `architecture/` (fcn, model_onepager, flux_model_onepager) · `method/` (mesh, graph, front_on_nodes, why_aw1_aw2, aw2_progress) · `rollouts/` (the 19 numbered sweep folders `NN_<recipe>/` + per-case `aw1_default/aw2_default/aw2_adaptive/`, each with `spacetime.png` + `profile.gif`) · `compare/`. `viz_all.py` writes to `figs/rollouts/`. Rerun after new rollouts.

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
