# LAUMeltFlow — Onboarding for Claude Code

> ⚠️ **STALE / WRONG PROJECT (as of 2026-06-19).** This onboarding describes the **whip / ECE Symposium** subproject (April 2026) and its "read first" pointers send you there. The **current active work is the heat-shield GNN surrogate** — see **`CLAUDE.md`** (the authoritative, auto-loaded handoff) for current state and which docs to read. Use this file only for background on the older whip/Symposium thread.

This is a multi-subproject research repo for **physics-informed machine learning surrogates**. The user is Grey Goodwin (UK ECE PhD researcher, advisor: Daniel L. Lau). Work centers on replacing expensive per-timestep FEA/physics computations with learned models — primarily GNN/MPGNN architectures.

## Repo Layout

| Subproject | Purpose |
|---|---|
| `whip/` | **Active.** 1D bead-chain FEA whip simulation + GNN surrogates. The Symposium 2026 paper is built from this work. |
| `whip/gnn/` | Variable-length MPGNN surrogate. Trained model, evaluation, benchmarking. |
| `whip/qt/` | Real-time Qt GUI for the whip simulation. |
| `Symposium/` | ECE Symposium 2026 paper + poster (LaTeX sources, figures, presentation notes, PowerPoint poster). |
| `meltflow_gnn/` | Earlier work: 1D compressible Euler with GNN/MLP surrogates on fixed-topology path graphs. Referenced in the Symposium paper as prior work. |
| `meltflow/` | Original MeltFlow CFD/heat solver code. |
| `electrostatic_unet/` | Separate project: U-Net + GNN hybrids for 2D electrostatic problems. |
| `docs/`, `reports/`, `models/`, `scripts/`, `matlab/` | Supporting documentation, archived reports, saved checkpoints, helper scripts. |

## Current Active Work (as of 2026-04-21)

**ECE Symposium 2026 poster session is tomorrow.** The paper is `Symposium/final_paper_update.tex` (6 pages, IEEE conference format). The poster is `Symposium/ECE_Symposium_Poster_2026.pptx` (48"×36", 3 columns, UK blue/gold). Both compile cleanly.

Key files:
- `Symposium/final_paper_update.tex` — paper source
- `Symposium/presentation_notes.md` — comprehensive Q&A notes for the poster session
- `Symposium/build_poster.py` — generates the poster .pptx programmatically (python-pptx)
- `whip/gnn/bead_mpgnn.py` — BeadMPGNN architecture (80K params, 1 message-passing round)
- `whip/gnn/outputs/mpgnn_best.pt` — trained checkpoint
- `whip/gnn/generate_volume_data_v2.py` — training data generator (polar sampling, variable chain lengths 8–32)

## Key Technical Context

**The paper's two findings:**
1. **Sampling strategy matters as much as architecture.** Uniform Cartesian sampling produces models that look accurate but diverge during rollout. Polar sampling near rest length (±5%) produces stable rollouts. This was discovered empirically.
2. **Variable-length generalization via message passing.** A true MPGNN with separate node/edge encoders, ghost masking, and weight sharing runs on chains of any length from one trained model. Earlier work in the lab (meltflow_gnn) showed that fixed-topology 1D paths don't need MPGNN — simple MLP concatenation suffices. The MPGNN matters specifically when chain length varies.

**Honest caveat the user knows about:** the GNN is currently **slower** than the reference numpy solver (see `whip/gnn/time_speedup.py` for the benchmark). PyTorch framework overhead dominates at small N. The value proposition is variable-mesh generalization, not raw speed. Don't claim speedup.

## Environment / Conventions

- **Windows 11**, PowerShell + Bash both available. Default to Bash for POSIX scripts.
- **MiKTeX** for LaTeX (will print warnings about update checks — ignore).
- **Python** with `torch`, `numpy`, `matplotlib`, `python-pptx`, `PIL`.
- The reference solver lives in `whip/simulation.py` (imported by GNN code via relative path hack `sys.path.insert`).
- Training data files (`whip/gnn/volume_data*.npz`, 50–70 MB each) are **not committed** and not gitignored either. Regenerate with `generate_volume_data_v2.py`.
- LaTeX build artifacts (.aux, .log, .out) clutter `Symposium/` but are not currently gitignored.

## Recent Git Activity

Most recent commit (`7fc959f`): added `compare_models.py` and `time_speedup.py` benchmarking scripts; bumped matplotlib font sizes across evaluation and pipeline figures for poster-readable output. Before that, `abeec17` introduced the BeadMPGNN generalization work itself.

The `Symposium/` folder, `Bead_mpgnn_convrsation.md`, and `whip/whip_FEA_GNN_animation.mp4` are **untracked** as of this writing — the user chose to not commit them yet.

## User Preferences

- Wants concise responses; doesn't want narration of internal deliberation.
- Prefers honesty over hype — explicitly asked to **not** put a fabricated speedup number on the poster after benchmarks showed the GNN is slower.
- Working in VSCode (Claude Code extension); will often open files in the IDE.
- Email: grey.goodwin@uky.edu.

## What to Read First

If you're picking up this repo cold:
1. `Symposium/final_paper_update.tex` — the most current, polished description of the whip MPGNN work.
2. `Symposium/presentation_notes.md` — Q&A-style explanations of every component.
3. `whip/gnn/bead_mpgnn.py` — the actual model architecture.
4. `whip/gnn/bead_eval_variable.py` — how the model is evaluated and rolled out.

## Likely Follow-up Work

- **Rollout training** (multi-step loss) to reduce drift over long rollouts — explicitly listed as future work in the paper.
- **K > 1 message passing** with multi-hop training data.
- **2D/3D mesh extension** — generalize the architecture to triangular or tetrahedral FEA meshes.
- **Speed optimization** — torch.compile, batched rollouts, or larger N where the GNN's O(N) per-step compute would actually beat the numpy reference.
- **SHAKE correction layer** or constraint-aware loss to tighten rod-length errors below the current ~1e−3 range.
