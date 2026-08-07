# Figures

Every figure here is **generated output** — none of it is hand-drawn, and all of it can be
rebuilt from the scripts in `src/`. They are committed anyway because the notes in
`docs/notes/` reference them, and GitHub renders them inline.

Each script writes into this directory by default. Override with the `FIG_OUT` environment
variable (single-figure scripts) or `FIG_OUT_DIR` (multi-figure scripts).

## The v1 → v2 rebuild (July 22)

Why the first generator was replaced. These are the evidence behind `docs/notes/JULY22.md`.

| Figure | What it shows | Regenerate with |
|---|---|---|
| `the_problem_with_v1.png` | v1's ceiling is U-shaped — task difficulty moved with distance, so the recovery threshold was scored against a bar that moved with it | `python src/plot_v1_problem.py` |
| `embedding_2d_v1_v2.png` | 2-D projection of v1 vs v2 embeddings. v1's families are far too separated; v2's overlap the way real ones do | `python src/plot_embedding_2d.py` |
| `realism_v1_v2_real.png` | v1 vs v2 vs **real** ESM-C embeddings, same data volume and same projection. The direct "which one looks like reality" comparison | `python src/make_realism_fig.py` |
| `recovery_v1_v2.png` | Recovery-threshold curves before and after the fix — v2's ceiling is flat, so `r*(d)` finally measures one thing | `python src/plot_recovery_compare.py` |

## Validation (July 22)

Does v2 reproduce real geometry, and does it predict real recovery out-of-sample?

| Figure | What it shows | Regenerate with |
|---|---|---|
| `validation_geometry.png` | v2's measured geometry against real embeddings across every dataset | `python src/make_validation_figs.py` |
| `validation_recovery.png` | v2's predicted recovery behaviour against the real taxonomy ladder | `python src/make_validation_figs.py` |
| `validation_data.json` | the numbers behind both panels | `python src/compute_validation.py` |

## EC / functional label space (Aug 4)

Referenced by `docs/notes/AUG4.md`.

| Figure | What it shows | Regenerate with |
|---|---|---|
| `ec_embedding_2d.png` | proteins coloured by EC group rather than by Pfam family — the functional label axis | `python src/make_ec_figs.py --outdir docs/figures …` |
| `ec_shift_angles.png` | distribution of angles between shift vectors, against the empirical null | `python src/make_ec_figs.py --outdir docs/figures …` |

`make_ec_figs.py` needs the embedding cache and the EC annotations; see
`slurm/run_ec_pilot2.slurm` for the full invocation.

## Removed

Five `taxonomy_axis*.png` sketches were deleted in the reorganisation. They were exploratory
views of the taxonomy-axis idea, superseded by the EC analysis, referenced by no document, and
two of them had no surviving generating script. The three that do can be rebuilt from
`src/make_taxonomy_axis_fig.py`, `src/make_taxonomy_axis_compare.py` and
`src/make_taxonomy_axis_noise.py`.
