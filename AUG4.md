# Aug 4 — the shift measured in a FUNCTIONAL label space (EC number)

Everything before today conditioned on Pfam **family**. Families are defined by
homology, so "family" and "sequence similarity" are nearly the same axis, and a
family-conditioned shift cannot separate two very different claims:

* the embedding **moved** when we went bacteria → plants, versus
* the embedding **stopped encoding function** when we went bacteria → plants.

EC number labels the same proteins by the reaction catalysed instead. The same EC
occurs in bacteria and in plants, and usually in more than one Pfam. That gives a
second, independent label axis and makes β measurable on real data for the first
time.

## 0. What was built

`src/fetch_ec_annotations.py` pulled EC numbers from UniProt for all 36,981
accessions already in `scratch/taxonomy_ladder/raw/metadata.tsv`. **No re-embedding
was needed** — the existing ESM-C cache is reused, so the whole analysis is CPU-only.

* 20,285 proteins keep a single unambiguous EC at level 3 (sub-subclass)
* 102 distinct EC groups across 16 Pfams and 5 domains
* 14 EC groups are present in **all five** domains; 32 in bacteria and plants both

Three shift types are now measurable in the same units:

| name | hold fixed | move |
|---|---|---|
| `TAX\|EC` | EC (function) | domain A → B |
| `TAX\|FAM` | Pfam (homology) | domain A → B — the pre-existing measurement |
| `FAM\|EC` | EC | Pfam f1 → f2 — same function, different scaffold |

Jobs: `run_ec_pilot.slurm` (47270164), `run_ec_pilot2.slurm` (47270413),
`run_oracle_pilot.slurm` (47270758). Outputs in `/scratch/lmk04992/ec_analysis/`.

## 1. The taxonomy shift does not care which label you hold fixed

| bacteria → | \|v\|/EC-gap (EC) | (Pfam) | α (EC) | α (Pfam) | cos(mean_EC, mean_Pfam) |
|---|---|---|---|---|---|
| archaea | 1.18 | 1.18 | +0.59 | +0.63 | **+0.99** |
| fungi   | 1.37 | 1.36 | +0.72 | +0.63 | **+0.99** |
| metazoa | 1.11 | 1.10 | +0.52 | +0.47 | **+0.99** |
| plants  | 1.14 | 1.14 | +0.59 | +0.61 | **+1.00** |

Magnitudes agree to two decimals and the mean shift vectors are essentially the
same vector. The taxonomic shift is a property of the domain, not an artifact of
the homology label — so the earlier family-based calibration was measuring
something real.

**Caveat, stated because it is easy to over-read:** both means are dominated by the
global domain-mean difference, so cos ≈ 1 is close to guaranteed. The sharper test
is whether the *residual* (after removing the shared translation) is better
explained by EC or by Pfam. It is **underpowered**: only 16–18 (EC, Pfam) cells
survive `min_n`, spread over ~12 groups, so raw η² ≈ 0.75–0.91 is near-saturated.
Against a label-permuted null only archaea separates (EC η² 0.911 vs null
0.751±0.076, p<0.001); fungi/metazoa/plants give p = 0.08–0.97. **We cannot
currently say whether function or homology better explains the residual.**

## 2. JULY22 open item 1 is resolved — α ≈ 0.5–0.7 is real

Open item 1 guessed the measured α was inflated by centroid-estimation noise and
that the true α was nearer 0.9–1.0. Split-half reliability says otherwise:
recomputing each shift from disjoint halves gives cos(v₁, v₂) = **0.97–0.99**.

Centroid noise is independent between classes, so it contributes nothing to the
numerator of a pairwise cosine while inflating both magnitudes — it biases α
**downward**, the opposite of what the open item assumed, and by ~1–3% here.
Corrected α moves 0.594 → 0.611. **α ≈ 0.6 stands.** The generator should use
α ≈ 0.6, not 0.5.

## 3. Same function, different scaffold is a completely different regime

`FAM|EC`, pooled over all five domains (44 cells):

* mean pairwise cosine **+0.08** (vs +0.59 for taxonomy), shared fraction 0.13
* magnitude 1.03 EC-gaps — *comparable to the taxonomic shift*
* sign-free alignment with the taxonomy axis 0.235, against a null of 0.204

So swapping scaffold moves a protein about as far as crossing a taxonomic domain,
but in an **idiosyncratic direction with no shared component**. Taxonomy is the
α→1 regime; scaffold change is the α→0 regime. Both exist in real data and they
are close to the two extremes the generator already models.

*(The signed cosine between the FAM|EC mean vector and the taxonomy axis is
**not** interpretable — f1→f2 is alphabetical, so its sign is a convention.
Use the sign-free cos² numbers.)*

## 4. There is no single taxonomy axis — archaea is off it

Sign-free alignment (mean cos²) with the bacteria→plants axis, null = 0.204:

| shift | alignment |
|---|---|
| bacteria → plants | 0.661 |
| bacteria → fungi | 0.577 |
| bacteria → metazoa | 0.519 |
| **bacteria → archaea** | **0.057** ← below the null |
| FAM\|EC | 0.235 ≈ null |

And in signed terms the three eukaryote shifts agree with each other at +0.81 to
+0.92 while each agrees with the archaeal shift at only +0.20 to +0.37.

**The shift geometry recapitulates the tree of life**: bacteria→archaea is a
prokaryote–prokaryote move and is nearly orthogonal to the shared
"becoming-eukaryotic" direction. The generator has one shared direction with one
α; real data has at least two distinct directions. This is the clearest concrete
upgrade available to v2.

## 5. β does NOT predict damage — this contradicts the JULY22 story

JULY22 concluded β is the damage knob and that "distance alone does not determine
cost", fitting β ≈ 0.15 to the eukaryote rungs and β ≈ 0.02–0.05 to archaea.
Testing that directly with a linear probe over **all 20 ordered domain pairs**
(`src/ec_allpairs.py`):

| predictor | Spearman vs retained EC F1 |
|---|---|
| β (per-cell) | **+0.32** ← wrong sign |
| β (of the shared translation, the generator's own definition) | **+0.38** ← wrong sign |
| \|v\|/gap (raw distance) | −0.53 |
| α | −0.19 |

partial ρ(β_shared, retained | magnitude) = **+0.19** — β adds nothing once
distance is controlled. partial ρ(magnitude, retained | β_shared) = −0.44.

**But neither survives a proper significance test.** The 20 pairs come from only 5
domains and are not independent; β and damage both look mostly like properties of
the *target* domain. Permuting whole domains rather than pairs:
|ρ| = 0.53, p = **0.122** for magnitude; |ρ| = 0.38, p = **0.217** for β_shared.

Honest bottom line: **with five domains we cannot establish that either β or
distance predicts functional damage.** What we can say is that the evidence for β
specifically is now *negative* — it has the wrong sign in every operationalization
— so the JULY22 β-fit should not be treated as validated. The β=0.15 fit
reproduced four zero-shot numbers, but four points can be fit by many things.

Two further facts from the probe worth keeping:

* **EC transfers better than Pfam** in every domain pair (e.g. bacteria→plants
  retained 0.721 vs 0.685). Taxonomic shift damages the homology readout more than
  the function readout.
* **The cost is strongly asymmetric**: archaea→bacteria retains 0.508 but
  bacteria→archaea retains 0.670; plants→fungi retains 0.946 but fungi→plants
  0.763. Direction of travel matters, which a symmetric distance cannot express.

## 6. The oracle label space does not yet look like EC

Real targets, measured (they replace the guessed purity 50–70% / promiscuity
40–60% / coverage ~10):

| quantity | real (EC vs Pfam) |
|---|---|
| within-family purity | **69.3%** |
| family promiscuity | **100%** |
| coverage | **2.98** families per class |

Two findings from `src/oracle_label_space.py`:

**(a) The `scale` knob is inert.** `RandomOracleNN` applies `LayerNorm` after every
hidden layer, so multiplying the input by a constant cannot change the argmax.
Scales 0.25 → 4.0 produced byte-identical label spaces. Any past tuning over input
scale was a no-op.

**(b) The oracle's labels were nearly independent of family, and that was the bug.**
It reads the full 960-d embedding, whose variance is mostly *nuisance* by
construction — so a random network slices along directions carrying no family
information. Real function is not independent of homology. Adding `signal_weight`,
which tilts the oracle's input toward the family-carrying signal subspace, lifts
purity from 33% to **67%** (real: 69.3%).

**But purity and coverage cannot be satisfied at once:**

| config | purity | coverage |
|---|---|---|
| 20 classes, 512-256, w=0.95 | **66.8%** ✓ | 8.32 ✗ |
| 200 classes, 256-128, w=0.99 | 39.3% ✗ | **4.72** (closest) |

Real data wants purity 69% *and* coverage 2.98 simultaneously. Those imply ~19 ECs
per Pfam of which one holds 69% of the members — a **heavy-tailed** within-family
label distribution. The oracle's argmax produces roughly balanced classes within a
family, so it can only buy purity by having few classes. **Fix: give the oracle a
Zipf-like label distribution *within* each family, not just the Zipf family-size
distribution it already has.**

## 7. What the generator gets right

Encouragingly, with the shift knobs set to the measured values (d=1.0, α=0.59,
β=0.42) the shift measured *in the oracle's own label space* lands inside the real
range on every statistic:

| statistic | synthetic | real range (4 domains) |
|---|---|---|
| α | +0.73 | +0.52 … +0.72 |
| \|v\|/gap | 1.17 | 1.11 … 1.37 |
| β | 0.38 | 0.32 … 0.62 |

And the cosine *spread* matches too: real sd 0.27 vs v2 sd 0.23 — v2 is not
over-tidy. So **the v2 shift model is well calibrated; it is the label space that
is not.**

## Next steps, in priority order

1. **More domains.** Everything statistical here is limited by having 5. Adding
   more taxonomic groups (or biomes) turns n=20 non-independent pairs into a real
   test of whether distance, β, or something else predicts damage.
2. **Two taxonomy axes in the generator** (§4) — a prokaryote/eukaryote direction
   distinct from the within-group one.
3. **Heavy-tailed within-family oracle labels** (§6) — the one change that could
   make purity and coverage agree with EC at the same time.
4. **Re-fit β, or drop it.** §5 says the current fit is not validated. Either find
   an operationalization of "function-damaging" that does predict damage, or let
   distance + direction carry the model.
5. Set **α = 0.6** (§2), not 0.5.

## Files

New: `src/{fetch_ec_annotations, measure_ec_geometry, measure_ec_damage,
ec_allpairs, oracle_label_space, make_ec_figs}.py`,
`run_{ec_pilot, ec_pilot2, oracle_pilot}.slurm`.
Data: `scratch/taxonomy_ladder/raw/ec_annotations.tsv` (36,981 rows).
Figures: `ec_embedding_2d.png`, `ec_shift_angles.png`.
