# Result — recovery threshold vs. distance (fixed synthetic generator)

_Run July 14, 2026. Job 46964648. Outputs in `/scratch/lmk04992/synth_distance_sweep/`._

## Headline

**The recovery threshold rises monotonically with distance from the training data**,
with a sharp knee between distance 0.5 and 0.7. This is the quantity the thesis set
out to measure, and the fixed synthetic generator now produces it cleanly — the old
generator structurally could not (single oracle + nested source; see
`HOW_TO_SEE_THE_DIP.md`).

`r*(d)` = the smallest OOD (target) fraction in the adaptation pool at which the
adapted model realigns to within 90% of the target-only ceiling:

| distance d | pre-adapt baseline F1 | recovery threshold r*(d) |
|---|---|---|
| 0.00 | 1.00 | 0.00 |
| 0.30 | 1.00 | 0.00 |
| 0.40 | 0.98 | 0.05 |
| 0.50 | 0.51 | 0.10 |
| 0.60 | 0.02 | 0.30 |
| 0.70 | 0.00 | 0.75 |
| 0.85 | 0.00 | 1.00 |
| 1.00 | 0.00 | 1.00 |

Figures: `threshold_vs_distance.png` (the r*(d) curve), `heatmap_final_f1.png` (the
recovery frontier over the full distance × OOD-fraction grid), `curves_by_distance.png`
(per-distance adaptation trajectories). Data: `sweep_results.csv` (per distance × r ×
seed), `threshold_vs_distance.csv` (aggregated).

## How to read it

- **Below d ≈ 0.3** the target still falls in the same decision regions as the source,
  so the source model is already near-perfect and essentially no target data is needed.
- **The knee (d ≈ 0.5–0.7)** is where the target manifold crosses into other families'
  source basins. The pre-adaptation baseline collapses (0.98 → 0.51 → 0.02), and the
  fraction of target data required to overcome the now-wrong source boundary climbs
  steeply: r* goes 0.05 → 0.10 → 0.30 → 0.75.
- **Beyond d ≈ 0.85** the source-optimal boundary is maximally wrong and the source data
  in the pool actively fights realignment; only a near-pure-target pool (r* = 1.0)
  recovers.
- **At r = 1.0 (pure-target pool) every distance recovers to F1 ≈ 1.0** — confirming the
  gap is negative transfer from the source composition of the pool, not a capacity limit.

## Why this is a valid fix (mechanism)

The generator now (`src/generate_synthetic_precomputed.py`):
1. **Labels by family, not by a global oracle** — so `P(y|x)` is no longer forced to be
   globally consistent between source and target.
2. **Displaces the target manifold instead of tightening the source** — each target
   family slides a fraction `d` toward another family's source region (a fixed
   derangement), so the source-trained boundary becomes genuinely wrong. `d` is the
   continuous distance knob.
3. **Builds families on a low-rank latent manifold** (latent_dim 32 → 1280) to mimic
   ESM's low effective rank. (Note: the projection does *not* set difficulty — an
   injective map preserves Bayes error; the achievable ceiling is set by the
   separation-to-spread ratio, `~ 1 − Q(min_centroid_gap / 2·within_sigma)`. See the
   ceiling note below.)

This is **concept / conditional shift, not covariate shift**: the target keeps each
family's label while its cluster moves, so `P(y|x)` changes and the Bayes-optimal
boundary itself relocates. In the Ben-David bound `ε_T ≤ ε_S + ½·d_HΔH + λ`, distance
drives up the `λ` term (best achievable joint error) — the rigorous reason large-`d`
adaptation is hard and source data *fights* realignment. It is the non-monotonic-OOD
regime of **De Silva et al. 2023 (ICML)**, distinct from sample-wise double descent.

Output matches `precompute_real_embeddings.py`, so synthetic and real feed the *same*
downstream pipeline — "software can do either data gen (synthetic vs real)."

## Caveats / next

- Ceilings are 1.0 here (clean synthetic families) — not a bug or undersampling, but the
  separation-to-spread ratio: at `centroid_spread=3` / `within_sigma=1` the 16 families
  sit ~4× their own radius apart → Bayes error ≈ 0. **Confirmed robust:** re-running the
  full sweep at `within_sigma=4` (ceiling ≈ 0.71–0.94, matching the real 0.94) and `=6`
  (≈ 0.44–0.72) preserves the `r*(d)` law — `r*` still rises 0 → 0.3 → 0.75 → off-grid,
  and recovery becomes realistically *incomplete* at large `d` (even `r=1` falls short of
  the ceiling). Outputs in `/scratch/lmk04992/synth_distance_sweep_sigma{4.0,6.0}/`
  (jobs 47087421, 47087436). With a realistic ceiling the ceiling itself moves with `d`,
  so keep measuring recovery relative to the per-`d` ceiling (as here).
- 3 seeds; the r*(d) knee is sharp enough to be clear at 3, but more seeds tighten it.
- The **real-data validation arm** (ESM-C on SwissProt→TrEMBL and, ideally, a few
  discrete real "distances") is the natural companion figure — see `NEXT_STEPS.md`.
