# Thesis update — recovery threshold under distribution shift

_Draft for [advisor]. — Liam, July 14, 2026_

## Where things stand

Since we last spoke, the big result is that the **real-data pipeline reproduces the
negative-transfer dip that the synthetic pipeline structurally could not** — and it
shows a recovery threshold around **r ≈ 0.5** (the fraction of out-of-distribution
target data the adapting model needs before it realigns instead of degrading).

Recap of *why* the synthetic setup never dipped: it used one global oracle labeler
and made the source a lower-variance subset of the same GMM as the target, so a
source-trained model was already approximately correct on the target — adaptation
could only help. Swapping in **real ESM-2 embeddings** of real proteins under a
genuine biological shift removed that guarantee.

**June result (UniProt, ESM-2 650M, Bacteria → Archaea, 16 Pfam families, 3 seeds):**

| target OOD frac r | dip depth (mean ± std) | seeds recovered | final target F1 |
|---|---|---|---|
| 0.00 | 0.087 ± 0.018 | 0/3 | 0.836 |
| 0.10 | **0.189 ± 0.071** | 0/3 | 0.810 |
| 0.25 | 0.107 ± 0.030 | 1/3 | 0.880 |
| 0.50 | 0.113 ± 0.040 | 2/3 | 0.904 |
| 1.00 | 0.053 ± 0.013 | 3/3 | 0.942 |

Reading it: the r = 0.10 dip clearly exceeds the in-distribution noise floor (~0.09),
so a *little* OOD data genuinely hurts first — real negative transfer. The cleaner
signal is *recovery*: seeds-recovered climbs 0 → 0 → 1/3 → 2/3 → 3/3 and final F1
rises monotonically 0.81 → 0.94 as r increases, with **pool size held constant**, so
this is about source/target *composition*, not simply "more data."

Honest caveats: only the r = 0.10 dip is cleanly above the noise floor at 3 seeds;
the intermediate dips sit near the noise band. The recovery trend is the more robust
evidence than dip-depth per se.

## New result: recovery threshold vs. distance

I rebuilt the synthetic data generator to fix the two defects that made the dip
impossible (one global oracle labeling both domains; the target being a tighter
*subset* of the source rather than a displaced distribution). The fixed generator
labels by family and **slides the target manifold a tunable distance `d` toward
another family's source region**, so distance is now a continuous knob — which lets
me measure the thing we actually care about: **how the recovery threshold changes
with distance from the training data.**

Sweeping distance `d` × OOD fraction `r` (3 seeds), the recovery threshold `r*(d)`
— the minimum fraction of target data needed to realign to within 90% of the
target-only ceiling — rises monotonically with a sharp knee at d ≈ 0.5–0.7:

| distance d | pre-adapt baseline | recovery threshold r*(d) |
|---|---|---|
| 0.0–0.3 | ~1.00 | 0.00 |
| 0.40 | 0.98 | 0.05 |
| 0.50 | 0.51 | 0.10 |
| 0.60 | 0.02 | 0.30 |
| 0.70 | 0.00 | 0.75 |
| 0.85–1.0 | 0.00 | 1.00 |

Two figures tell the story: a clean sigmoidal `r*(d)` curve, and a heatmap of final
target F1 over the full grid showing a sharp "recovery frontier" — the diagonal
boundary between realignment and getting stuck in negative transfer. At a pure-target
pool (r = 1.0) every distance recovers to F1 ≈ 1.0, so the gap is negative transfer
from the *source composition* of the adaptation pool, not a capacity limit. Details
in `RESULTS_distance_sweep.md`.

## What I'm setting up next (real-data validation)

Two extensions, both now coded and ready to submit on Sapelo2:

1. **A second, independent shift axis — SwissProt → "distance from SwissProt."**
   Rather than a taxonomic shift, source = curated SwissProt proteins and target = a
   population at a distance (starting with unreviewed/TrEMBL, same Pfam families, taxon
   held fixed so curation status is the only axis). This tests whether the dip and the
   ~0.5 threshold are specific to Bacteria→Archaea or a general phenomenon. MGnify
   metagenomic proteins are the more extreme distance and are the follow-up (they need
   GCP BigQuery access set up on the cluster).

2. **ESM-2 → ESM-C embeddings.** ESM-C (EvolutionaryScale, late 2024) is the newer PLM.
   I've wired it in alongside ESM-2 and will embed the *same* sequences with both, so we
   get a clean model-only comparison: does a stronger PLM change the geometry of
   negative transfer, or does the threshold persist? The downstream pipeline is
   dimension-agnostic, so nothing else changes.

## Questions for you

- Is the **SwissProt → TrEMBL curation** shift a satisfying "distance," or would you
  rather I prioritize getting **MGnify metagenomic** data (needs credentials) as the
  headline shift?
- For the final figures, should I push seeds from 3 to ~10 to tighten the intervals,
  and refine the r grid around 0.3–0.6 to localize the threshold?
- Any preference on ESM-C size (300M vs 600M) for the comparison?

Happy to walk through the curves whenever works for you.
