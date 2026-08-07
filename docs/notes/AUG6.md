# Aug 6 — five questions from the professor, answered on a dataset 11x larger

Follow-up to `AUG4.md`. Five questions arrived by email about the EC dashboard.
Question 3 turned out to be the key: the dataset really was subset, and rebuilding
it EC-first took us from 5 taxonomic groups to 15 — which is what every other
question was actually blocked on.

**Two claims from AUG4 do not survive, and one intermediate claim I made during
this run also did not survive its own better-powered rerun.** Details below.

## 0. What was built

| script | what it does |
|---|---|
| `src/fetch_ec_swissprot.py` | EC-first UniProt pull, resumable, keeps full lineage |
| `src/build_ec_dataset.py` | assigns taxonomic groups from lineage, writes the (meta, ec, fasta) triple |
| `src/ec_permutations.py` | all 8 conditioned shift types, gap definitions, additivity, EC hierarchy |
| `src/beta_diagnosis.py` | beta injection experiment + 12-predictor damage battery |
| `src/realism_scorecard.py` | distribution/neighbourhood statistics, real vs synthetic |

Jobs: 47289231 (ladder follow-up), 47289277 (A100 embedding, 231k seqs in ~2 h),
47289440 (full analysis), 47290799 (matched-null injection + class-matched scorecard).
Data in `/scratch/lmk04992/ec_swissprot`, outputs in `.../analysis/`.

## 1. The dataset was capped by 16 Pfam families, not by requiring Pfam

| query | proteins |
|---|---|
| Swiss-Prot reviewed | 575,503 |
| … with an EC number | 280,036 |
| … and a Pfam xref | 277,716 ← requiring Pfam costs **0.8%** |
| … and in our 16 chosen families | 36,981 ← **the actual constraint** |
| … and a single unambiguous EC at L3 | 20,285 |

Rebuilt EC-first: **231,285 proteins, 17 taxonomic groups (15 used; `other_bacteria`
and `other_eukaryota` are remainders, not clades), 270 EC groups, 3,282 Pfams.**
Groups are assigned at phylum/class level from the NCBI lineage, so `euryarchaeota`
and `crenarchaeota` are separate and the five proteobacterial classes are separate.

**15 groups → 210 ordered pairs.** This is what mattered: with 5 groups the
domain-level permutation null had only 120 arrangements and nothing could reach
significance. Nine of the twelve damage predictors are now significant.

## 2. The gap normaliser, stated precisely

`gap = mean over all pairwise Euclidean distances between EC-group CENTROIDS,
computed in the source domain only, over EC groups with >= min_n proteins there.`

131 centroids → 8,515 pairwise distances (was 18 → 153). Conversion factors:
median 0.95x, p10 0.58x, closest pair 0.24x, pooled over all groups 1.09x,
Pfam centroids 1.29x, **random protein pair 1.57x**.

**Separation ratio gap / within-EC sigma = 1.11.** EC groups sit about one
within-group sigma apart. Every `|v|/gap` should be read against that — report it
alongside.

## 3. beta: the sign is right, the definition has a scale bug

`B` = top directions of the SVD of source-domain EC centroids holding 90% of
between-EC variance (**22 of 960** on the rebuild, 8 on the ladder).
`beta` = fraction of the shift's squared length in `B`.

The controlled injection depends entirely on how the **off-B** component is drawn:

| control | in-B drawn | off-B drawn | verdict |
|---|---|---|---|
| isotropic | uniform in B | uniform in R^960 | beta damages |
| on-manifold | uniform in B | data covariance | no clear effect |
| **matched** | **data covariance** | **data covariance** | **beta protects** |

Only `matched` is a fair test — otherwise the contrast is "typical vs atypical
direction", not "in B vs out of B". Under `matched`, at 117 EC classes, retained F1
rises monotonically with beta: **0.116 → 0.489** at 2 gaps, **0.520 → 0.923** at
1 gap. Differential mode gives the same (0.113 → 0.460).

This reproduces the real-data sign causally: over 210 pairs
`Spearman(beta_shared, retained) = +0.280, p = 0.006`.

**Leading explanation:** `B` is built as the directions where EC centroids are most
*spread out*, i.e. the high-variance directions. The shift has a fixed **Euclidean**
length, so the same `|v|/gap` is proportionally a small move along a high-variance
direction and an enormous one along a low-variance direction. beta is confounded
with the variance of the direction travelled. Supporting evidence: off-subspace
displacement predicts damage nearly 2x as strongly (rho −0.595) as in-subspace
displacement (−0.367).

**Fix to try next:** define magnitude and beta in **whitened** units (normalise each
direction by the source domain's sd along it). Test = does whitened beta recover a
negative correlation with retained F1. Contained change to `measure_ec_geometry.py`.

> **Superseded:** an earlier run of this same experiment on the 5-domain data, using
> the *isotropic* control, said beta causally damages function. That was reported and
> is wrong. It did not survive either the matched control or 117 classes.

## 4. Damage predictors, n = 210 pairs from 15 groups

| predictor | rho | p |
|---|---|---|
| **‖v − v̄‖ / gap** (absolute differential motion) | **−0.784** | <0.001 |
| ‖P_B(v − v̄)‖ / gap | −0.773 | <0.001 |
| **probe logit spread** | **−0.671** | <0.001 |
| **Procrustes disparity** | **−0.634** | <0.001 |
| ‖P_B⊥ v‖ / gap | −0.595 | <0.001 |
| `|v|/gap` (current) | −0.592 | <0.001 |
| ‖P_B v‖ / gap | −0.367 | <0.001 |
| beta_shared (current) | +0.280 | 0.006 |
| beta per cell (current) | +0.239 | 0.023 |
| gap ratio target/source | −0.094 | 0.260 |
| **alpha (current)** | **+0.022** | **0.871** |
| 1 − shared_frac (differential *fraction*) | −0.008 | 0.961 |

Three lessons:
* **alpha does not predict damage at all.** Keep it as a realism knob; stop
  describing it as related to cost.
* **Absolutes work, fractions do not.** Absolute differential motion is the best
  predictor; the differential *fraction* is worthless. beta is a fraction too.
* `probe logit spread` = `sd_c(w_c · v/sigma) / sd(source logits)` — the first-order
  damage to the actual probe, derived rather than guessed. Third-best but the
  interpretable one.

## 5. alpha was never comparable across shift types — the cross-move null

AUG4 compared `TAX|EC` alpha (+0.59) against `FAM|EC` alpha (+0.08) and concluded
scaffold change is unstructured. **Those are not the same statistic.** `TAX|EC`
fixes one move (bacteria→plants) and varies the context — a *within-move* number.
`FAM|EC` pooled cosines across *different* family pairs — a *cross-move* number.

Both are now computed for all 8 conditioned shifts, plus the cross-move null.
15 groups, min_n=30, EC level 3:

| shift | moves | vectors | alpha within | alpha cross | excess |
|---|---|---|---|---|---|
| TAX\|EC | 105 | 3,867 | +0.696 | +0.054 | +0.642 |
| FAM\|EC,DOM | 979 | 3,006 | +0.597 | +0.008 | +0.589 |
| TAX\|FAM | 83 | 2,900 | +0.619 | +0.072 | +0.547 |
| EC\|DOM,FAM | 25 | 75 | +0.527 | +0.001 | +0.526 |
| **EC\|DOM** | **4,000** | **18,924** | **+0.531** | **+0.008** | **+0.523** |
| TAX\|EC,FAM | 71 | 2,716 | +0.577 | +0.059 | +0.517 |
| FAM\|EC | 12 | 24 | +0.244 | +0.008 | +0.235 (underpowered) |
| EC\|FAM | 10 | 21 | −0.070 | +0.033 | −0.104 (underpowered, ignore) |

**Functional shift is strongly structured**: `EC|DOM` excess +0.523 over 4,000
distinct EC→EC moves and 50,400 cosines. At **EC level 4** (exact reaction) it is
*higher* — alpha +0.641, excess +0.635 — so it is not an artifact of coarse binning.
The e1→e2 displacement is close to the same vector in every taxonomic group, which
is exactly the generator's assumption.

> **Superseded:** mid-run I concluded from the 5-domain data that scaffold change is
> unstructured (excess +0.09, from 8 moves / 36 vectors). With 3,282 Pfam families it
> rests on 979 moves and gives **+0.589**. All three label axes are structured
> (excess +0.52 to +0.64) against cross-move nulls of ~0.

`--max_pairs 4000` caps the EC|DOM row; other rows are uncapped.

## 6. Additivity, and a caution for the thesis

| grid | cells | rows alone | cols alone | additive | interaction |
|---|---|---|---|---|---|
| group x EC | 966 | 0.455 | 0.429 | 0.787 | 0.213 |
| group x Pfam | 1,835 | 0.278 | 0.744 | 0.872 | 0.128 |
| **Pfam x EC** | 1,018 | **0.956** | 0.319 | **0.994** | **0.006** |

79% of the group x EC grid is additive — the generator's structural assumption is
broadly right and misses about a fifth. The last row is the striking one: **Pfam
alone explains 95.6% of the Pfam x EC grid and the grid is 99.4% additive.**
Homology essentially fixes position; function adds very little on top. **Caution: an
EC probe may largely be reading homology.**

## 7. Realism scorecard — three metrics to add

v2 at calibrated knobs (d=1.0, alpha=0.59, beta=0.42), **regenerated with 131 classes
to match the real class count** (without that control several statistics are
meaningless).

Matching well: hubness 0.98, alpha 0.89, additivity 0.90, shared fraction 0.92,
`|v|/gap` 1.24, kNN consistency 1.30, effective rank 1.34.

Add these, in order:
1. **Procrustes disparity of the shift** — 12x off (real 0.322, v2 0.026) **and** the
   4th-best damage predictor (rho −0.634). Both badly mismatched and strongly
   predictive. alpha/beta/magnitude are structurally blind to it.
2. **Intrinsic dimension / spectrum** — 8.6x off (real 1.77, v2 15.17); dims-to-90%
   0.22x (55 vs 12). v2 is too full locally and too compact globally.
3. **Class-covariance heterogeneity** — 3.2x off (real 0.284, v2 0.920). Cheap fix:
   per-class random rotation of the covariance instead of one shared.

**Dropped from the recommendation list:** hubness looked 4.1x off against a 16-class
synthetic; with class count matched it is **0.98**. The whole discrepancy was the
class-count mismatch. kNN consistency shrank 1.79 → 1.30 the same way.

## 8. ESM-C does not encode the EC hierarchy — now solid

Spearman(embedding centroid distance, EC-tree distance) is **≈ 0 in all 15 groups**
(−0.05 to +0.34, no consistent sign; −0.040 over 131 classes in gammaproteobacteria).

Mid-run I attributed the flat ladder result to 102 EC groups drawn from 16 Pfams and
said the rebuild would be the test. It ran — 270 EC groups from 3,282 Pfams — and the
answer did not change. **Do not build EC-tree structure into the oracle.** Worth a
line in the thesis on its own.

## Next steps

1. **Whitened beta and magnitude** (§3). The one change that could make beta mean
   what its name says. Test: does whitened beta go negative against retained F1.
2. **Procrustes term in the generator** (§7) so the synthetic shift rearranges the
   class configuration instead of translating it rigidly.
3. **Per-class covariance rotation** and a lower intrinsic dimension (§7).
4. Report `|v|/gap` alongside the separation ratio everywhere (§2).
5. Revisit the AUG4 "archaea is off the taxonomy axis" result — with euryarchaeota
   and crenarchaeota separate and 15 groups, it is now testable.
6. Re-check the AUG4 oracle calibration targets (purity 69.3% / promiscuity 100% /
   coverage 2.98) against the rebuild — they were measured on the 16-family subset.
