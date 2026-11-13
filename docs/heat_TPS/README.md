# Manuscripts — status

Which documents here are finished and which are drafts. Read this before quoting
a number out of any of them.

| Document | Status | What it is |
|---|---|---|
| `Goodwin_report.tex` / `.pdf` | **FINISHED, SENT** | Progress report for Dr. Alexandre Martin, author of the reference solver. Plain-language narrative of the whole arc, minimal ML jargon. Sent to him. The only completed document here. |
| `Goodwin_TPS.tex` / `.pdf` | **WORKING COPY** | Journal manuscript, the canonical technical write-up. Results are current and the bibliography has been built out to 29 entries. The planned reorganization to match the Cao 2026 structure is not done. |
| `Goodwin_chapter.tex` / `.pdf` | **WORKING COPY** | Thesis chapter. Earliest stage of the three. |
| `Goodwin_equations.tex` / `.pdf` | Reference, complete as a reference | All 38 governing equations with a section mapping which the surrogate computes, replaces, or never touches. Not a manuscript. |
| `heat_gnn_whitepaper.*` | **SUPERSEDED** | The first prose draft, June 2026, plain `article` class. Where the writing started. Kept for history. Do not cite. |
| `heat_gnn_ieee.*` | **SUPERSEDED** | The same content recast into `IEEEtran` when the deliverable became a conference paper. `Goodwin_TPS` is its direct descendant. Kept for history. Do not cite. |
| `heat_gnn_refs.bib` | **LIVE** | Shared bibliography, 29 entries, used by all current documents. Not superseded. One placeholder remains: the key `own`, for the ECE Symposium paper. |
| `2d_ablator_kickoff.md` | Current | Scope, open decisions and risks for the 2D axisymmetric work. |

## Current results, as of November 2026

Use these numbers, not any you find in an older draft or in a superseded block of
`CLAUDE.md`.

- Gas-off case (aw1): mostly-learned GNN, **7.6 K** mean held-out rollout error.
- Coupled gas and aerothermal case (aw2): a purely learned GNN **diverges**.
- Physics-informed version with a fixed conduction anchor: **25.3 K**, bounded.
- Relaxing that anchor plus transient-focused training: **18.2 K**, the headline.
- Speed: about **2x** the reference solver, about 10x the Python port.
- Learned parameters: **10 receive gradient of 110,299 allocated, and 5 move**.

## Notes

- LaTeX intermediates (`.aux`, `.bbl`, `.blg`, `.log`, `.out`) are gitignored. Rebuild
  with `pdflatex; bibtex; pdflatex; pdflatex`.
- All four current documents share `heat_gnn_refs.bib` and `figs/`.
- `References/` and `FEvFV/` hold third-party PDFs for reading only. Not redistributed.
