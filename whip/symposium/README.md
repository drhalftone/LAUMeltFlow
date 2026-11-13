# ECE Symposium 2026 — publication materials

Paper and poster for the bead-chain MPGNN work. This directory holds **only** the
LaTeX sources, figures, and build outputs. The model, training code, and result
figures live in [`../gnn/`](../gnn/).

Everything at the top level is part of the final paper or poster. Superseded
drafts, unused figures, and other people's templates are in [`archive/`](archive/).

## Final versions

| Deliverable | Source | Output |
|---|---|---|
| Paper | `final_paper_update.tex` | `final_paper_update.pdf` |
| Poster (LaTeX) | `whip_poster_v2.tex` | `whip_poster_v2.pdf` |
| Poster (PowerPoint) | `ECE_Symposium_Poster_2026.pptx` | `ECE_Symp_Poster_GLG_26.pdf` |

`beamerthemegemini.sty` is the poster theme. `build_poster.py` builds the
PowerPoint route, `generate_mlp_only.py` builds the two architecture diagrams.
`presentation_notes.md` is the talk script.

## Where the science lives

| Artifact | Location |
|---|---|
| MPGNN model | `../gnn/bead_mpgnn.py` |
| MLP baseline | `../gnn/bead_model.py` |
| Training | `../gnn/bead_train_mpgnn.py`, `../gnn/bead_train.py` |
| Trained weights | `../gnn/outputs/mpgnn_best.pt`, `bead_best.pt` |
| Hidden-size sweep (h=32 sweet spot) | `../gnn/outputs/sweep/h{004..128}/` |
| Data generation | `../generate_data.py` → `../gnn/volume_data*.npz` |

## Figure inventory

| Figure | Paper | Poster | Origin |
|---|:--:|:--:|---|
| `architecture_mlp_only.png` | ✅ | | `generate_mlp_only.py` (local) |
| `architecture_mpgnn_only.png` | ✅ | ✅ | `generate_mlp_only.py` (local) |
| `pipeline_visualization.png` | ✅ | ✅ | `../gnn/visualize_pipeline.py` — **local copy differs** from `../gnn/pipeline_visualization.png` |
| `frame_initial.png` | ✅ | | no generator in repo (see below) |
| `frame_falling.png` | ✅ | ✅ | no generator in repo (see below) |
| `frame_whip.png` | ✅ | | no generator in repo (see below) |
| `mpgnn_training.png` | ✅ | | `../gnn/bead_visualize.py` — **local copy differs** from `../gnn/outputs/` |
| `mpgnn_variable_eval.png` | ✅ | | `../gnn/bead_eval_variable.py` — identical to `../gnn/outputs/` |
| `mpgnn_loss_only.png` | | ✅ | poster-only variant |
| `mpgnn_pos_error_only.png` | | ✅ | poster-only variant |

Two cautions before regenerating anything:

- `mpgnn_training.png` and `pipeline_visualization.png` are **not** byte-identical
  to their `../gnn/` counterparts. The copies here are what the submitted PDFs were
  built from, so treat these as authoritative and do not blindly overwrite them
  from `../gnn/`.
- The three `frame_*.png` stills have no generating script anywhere in the repo
  (searched `whip/` and `whip/gnn/`). They appear to be hand-captured from the
  simulation animation, so treat them as source assets, not reproducible output.

## archive/

Nothing here is referenced by the final paper or poster.

| Contents | Notes |
|---|---|
| `whip_paper.tex/.pdf` | first paper draft, original title |
| `whip_paper_Xmpgnn.tex/.pdf` | retitled draft; `final_paper_update` differs only by merging the two author blocks into one |
| `Symposium_2026_Paper.pdf`, `ECE_Symp_26_GLG_GNN.pdf`, `ECE_symp_GLG_v2.pdf` | intermediate compiled drafts, no matching `.tex` |
| `whip_poster.tex` | first poster draft (never produced a PDF) |
| `architecture_comparison.png` + `generate_architecture_fig.py` | superseded by the two `architecture_*_only.png` splits |
| `mpgnn_math_reference.*` + `generate_math_reference.py` | standalone equation sheet, never used in either document |
| `mihir_example.tex`, `ECE_Symposium_Poster_Spring2025.pdf`, `OwenWilkerson_spring2019_UKY_Grad_research_poster.pptx` | templates from other people, kept for reference |

The archived `.tex` files reference figures by bare filename, so they will not
compile from inside `archive/`. To rebuild one, copy it back to the top level
first. No figure is exclusive to a draft, so nothing was stranded by the move.

## Housekeeping notes

LaTeX intermediates (`.aux`, `.out`, `.log`, `.nav`, `.snm`, `.toc`) are present on
disk but gitignored. Removed during the 2026-07-29 cleanup: a byte-identical
duplicate of the Owen Wilkerson template pptx, and two failed poster builds
(`grey_poster_fail.pdf`, `grey_latest_fail2pdf.pdf`).
