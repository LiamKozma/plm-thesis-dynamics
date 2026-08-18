# Aug 18 — the homology-subspace question, and what a fixed displacement actually measures

Written 2026-08-18. Seven batch jobs submitted 2026-08-17 evening, all complete,
plus three reruns after finding bugs in my own analysis. Read with
[`AUG7.md`](AUG7.md) (the EC ladder and the six estimator defects) and
[`BRIEF_ec_recovery_threshold.md`](BRIEF_ec_recovery_threshold.md) (the landmines,
all of which earned their place again today).

The headline is not the one I expected to write. The question "if we shift data out
of the homology subspace, what happens to function prediction?" has an answer, and
the answer is that **the question as posed cannot separate the subspace from the
metric**. Once both magnitude measures are in the model, which subspace the
displacement lies in adds +0.06 R². That is a more useful finding than a
confirmation would have been, and it makes the whitened-coordinates decision (T3.1)
much easier.

---

## 0. What was run

| Job | Arm | Elapsed |
|---|---|---|
| 47525589 | `r*` under subspace-restricted shift, matched Euclidean length, 43 cells | 54 min |
| 47525590 | same, matched within-class SD, 22 cells | 28 min |
| 47525591 | same, whitened coordinates, 22 cells | 28 min |
| 47525588 | observational twin: 210 real shifts decomposed into H / B / outside | 16 min |
| 47525587 | Tier 1, all eight items | 6 min |
| 47525593 → 47525594 | ESM-2 650M on all 15 groups (59,284 seqs), then ladder + archaea subspace | 2h53 + 11h26 |
| 47525602 | BLAST, 15 databases × 15 query sets, 225 hits files | 1h17 |
| 47542235 | T1.6 rerun with matched source volume and self-hits dropped | 12 min |

New code: `src/subspace_rstar.py`, `src/shift_decomposition.py`,
`src/tier1_backlog.py`, `src/subspace_rstar_analyse.py`, `src/pident_allpairs.py`,
`src/clades.py`.

---

## 1. The design, and the one thing it got right

Source and target are disjoint EC-stratified halves of **one** taxonomic group, so
the only shift present is the injected one. The displacement is a rigid translation
of the whole target cloud, so it is an isometry and the target-only problem is
unchanged — landmines 8 and 9. That prediction is now checked rather than assumed:

**the ceiling across all 43 cells of the Euclidean arm spans 0.9752 to 0.9804**,
against zero-shot scores in the same cells ranging from 0.977 down to 0.211. The
bar does not move. This is the first shift experiment in the project where that is
demonstrable rather than argued, and it is worth keeping as the template.

Magnitude 0 doubles as the within-group null (see §3, T1.4): retention 0.9966.

## 2. `r*` cannot answer this question, and the reason is instructive

`r*_budget` is 0.0 in 40 of 43 cells at P = 200 and 36 of 43 at P = 500, including
cells where zero-shot retention has collapsed to 0.216. The budget-relative bar is
0.9 × the score of a model trained from scratch on P target proteins, and for a
101-class problem that model reaches only 0.312 at P = 200 and 0.532 at P = 500. A
badly damaged source model still clears 0.9 × 0.312.

So the well-posed threshold from AUG7 §5 is well-posed and **uninformative here**:
it asks whether the source model beats a very weak target-only model, and at these
budgets it always does. `r*` against the full ceiling discriminates a little
(0.0 when undamaged, 0.5–1.0 at moderate damage, censored at severe damage) but is
censored in 24 of 43 cells.

**Retention is the readable outcome, again.** That is now three independent arms —
the real ladder, the all-pairs scan, and this injected experiment — where `r*` is
the fragile quantity and zero-shot-over-ceiling is the stable one.

## 3. The subspace result: the ordering is the metric

### The two normalisations disagree, and each is confounded the other way

At matched **Euclidean** length (1.0 EC gap), EC retention:

| condition | dim | within-class SD | EC retained |
|---|---|---|---|
| B (function) | 26 | 2.6 / 5.0 | 0.986 / 0.836 |
| H (homology) | 50 | 3.4 / 4.3 | 0.858 / 0.932 |
| H − B | 50 | 14.9 / 14.2 | **0.359 / 0.451** |
| B − H | 26 | 42.0 / 46.9 | 0.216 / 0.564 |
| outside both | 884 | 36.4 / 38.2 | 0.435 / 0.474 |
| random, matched dim 26 | 26 | 15.3 / 28.7 | 0.517 / 0.784 |
| random, matched dim 50 | 50 | 18.6 / 17.3 | 0.826 / 0.834 |

At matched **within-class SD**, the ordering inverts:

| condition | 3 SD | 8 SD | 20 SD | Euclidean cost of 20 SD |
|---|---|---|---|---|
| H (homology) | 0.917 | **0.088** | **0.009** | 5.94 gaps |
| B (function) | 0.982 | 0.467 | 0.018 | 7.74 gaps |
| H − B | 0.990 | 0.921 | 0.143 | 1.34 gaps |
| random, matched dim 50 | 0.991 | 0.987 | 0.761 | 1.07 gaps |
| outside both | 0.994 | 0.992 | 0.908 | 0.55 gaps |
| B − H | 0.998 | 0.993 | 0.915 | 0.48 gaps |

Neither table can be read as "subspace X is special", because in the first the
apparent winner is the condition with the most SD per unit length and in the second
it is the condition with the most gaps per unit SD.

### Pooling the arms settles it

Pooling all three arms (84 cells, budget 200) and regressing ranked retention:

| model | R² (raw coords) | R² (whitened) |
|---|---|---|
| within-class SD alone | 0.509 | 0.794 |
| Euclidean length in gaps alone | 0.764 | 0.775 |
| **both magnitude measures** | **0.851** | **0.854** |
| condition label alone | 0.031 | 0.113 |
| both + condition label | 0.908 | 0.919 |

**The condition label adds +0.058 R² on top of the two magnitude measures** (partial
F = 5.65 on 6 and 54 df, so not literally nothing, but small). The mechanism is one
number per condition:

| condition | within-class SD per EC gap, raw | whitened |
|---|---|---|
| B (function) | **3.4** | 47.4 |
| H (homology) | **3.7** | 30.0 |
| H − B | 14.7 | 28.3 |
| random dim 50 | 18.2 | 26.8 |
| random dim 26 | 19.7 | 26.5 |
| outside both | 37.0 | 25.6 |
| B − H | **43.6** | 26.8 |

So: **the homology and function subspaces are the high-variance directions of the
embedding.** A gap-sized step along them is a 3.5-SD intervention; the same step
outside them is a 37–44-SD intervention. They look robust per unit of Euclidean
length for that reason and no other. In whitened coordinates, where every direction
carries the same spread by construction, the spread column flattens to 25–47 and
nothing discriminates: maximum damage across all 22 whitened cells is retention
0.980.

### The sentence that survives

A displacement damages EC prediction in proportion to how far it moves relative to
the within-class spread along the direction it moves. The homology subspace is not
functionally privileged; it is where the wide directions live. Any claim of the form
"shifting along homology breaks function" is a claim about the anisotropy of ESM-C,
restated.

This supersedes the reading offered on 12–13 August, where H − B looked worst at
matched Euclidean length in gammaproteobacteria. The four extra groups run today
(§5) show that ordering is not even stable across groups.

## 4. The observational twin points somewhere else entirely

No injection: split each of the 210 real taxonomic shifts into its components and
correlate with linear-probe retention (group-permutation null throughout, 25
predictors, ~1.2 false positives expected at p < 0.05).

| predictor | ρ | partial given magnitude | p |
|---|---|---|---|
| **differential motion / gap** | **−0.819** | **−0.761** | <0.0005 |
| shared component outside both subspaces (excess over null) | −0.577 | −0.610 | <0.0005 |
| shared component in H (excess over null) | +0.556 | +0.599 | <0.0005 |
| shared displacement magnitude / gap | −0.472 | — | <0.0005 |
| shared component in B (excess over null) | +0.335 | +0.495 | 0.002 |

The dominant predictor of real function damage is the **family-specific
differential** motion, not the shared translation — and differential motion is
precisely the component the injected experiment cannot apply, because a
family-specific displacement is not an isometry and would move the ceiling
(landmine 8). The two experiments are therefore not in conflict; they measure
different halves of a real shift, and the injected half is the less important one.

Note also the sign: more of the shared shift lying *outside* both subspaces predicts
*worse* retention, and more lying *inside* H predicts *better*. In raw coordinates a
typical data-covariance direction already puts ~72% of itself inside B, so
"outside" means "atypical direction", and atypical directions damage probes. That is
the same confound that produced the β sign error in AUG4, arriving by a new route.

**Where this leaves the generator:** the knob that matters for function damage is
one the v2 generator has, `alpha` (shared vs differential), and the thing it should
be anchored against is differential motion in whitened units. Neither β nor the
subspace decomposition is carrying its weight in raw coordinates.

## 5. T2.1 — the two-encoder replication, and the archaea it was missing

ESM-2 650M on all 15 usable groups, 59,284 sequences. Retention, gammaproteobacteria
source, linear probe:

| target | ESM-C | ESM-2 | Δ | clade |
|---|---|---|---|---|
| betaproteobacteria | 0.932 | 0.971 | +0.039 | bacteria |
| spirochaetes | 0.872 | 0.951 | +0.079 | bacteria |
| bacteroidetes | 0.844 | 0.952 | +0.109 | bacteria |
| alphaproteobacteria | 0.788 | 0.922 | +0.134 | bacteria |
| epsilonproteobacteria | 0.721 | 0.928 | +0.207 | bacteria |
| cyanobacteria | 0.709 | 0.900 | +0.192 | bacteria |
| firmicutes | 0.572 | 0.817 | +0.245 | bacteria |
| actinobacteria | 0.524 | 0.845 | +0.321 | bacteria |
| insecta | 0.405 | 0.427 | +0.022 | eukaryota |
| euryarchaeota | 0.362 | 0.576 | +0.214 | archaea |
| crenarchaeota | 0.332 | 0.466 | +0.134 | archaea |
| ascomycota | 0.285 | 0.434 | +0.149 | eukaryota |
| streptophyta | 0.180 | 0.393 | +0.213 | eukaryota |
| vertebrata | 0.170 | 0.256 | +0.086 | eukaryota |

* **The domain separation replicates, with a wider gap.** ESM-C: min bacterial 0.524
  vs max non-bacterial 0.405 (+0.119). ESM-2: 0.817 vs 0.576 (+0.242).
* **The ordering is nearly identical**: Spearman 0.960 over the 14 ladder targets,
  0.914 over all 209 ordered pairs.
* ESM-2 retains more everywhere. **Do not report that as ESM-2 being better**: the
  ESM-2 arm is a 5,000-per-group subsample, so its class sets and data volumes
  differ. The ordering agreement is the claim; the level difference is confounded.

Subspace displacement on the four groups the first replication missed
(crenarchaeota, euryarchaeota, epsilonproteobacteria, ascomycota), raw coordinates,
EC retained at 2 gaps: **`outside_both` is the most damaging condition in all four
groups and both encoders** (0.015–0.116), with H − B second (0.28–0.57) and B and H
nearly harmless (0.68–0.99). In gammaproteobacteria the worst condition was H − B.
The raw-coordinate ordering is not stable across groups, exactly as §3 predicts if
it is tracking spread rather than subspace identity.

## 6. Tier 1, item by item

### T1.4 — retention finally has a zero point, and two bacterial targets sit inside it

Retention with **no taxonomic shift at all**, splitting one group against itself:

| group | random | by organism | by genus | by order |
|---|---|---|---|---|
| gammaproteobacteria | 0.998 ± 0.016 | 1.005 ± 0.016 | 0.990 ± 0.015 | 0.972 ± 0.034 |

Full range over 48 partitions: **0.902 to 1.039**. The MLP arm agrees (0.9966).

Consequence: bacteroidetes (1.001) and spirochaetes (0.983) are **not
distinguishable from no shift**, and the write-up must say so. The six archaeal and
eukaryotic targets at 0.221–0.537 are far outside the band. The "8 against 6"
framing survives as a statement about the band, not about a gap in a list.

### T1.5 — the domain split is robust, the within-bacteria ranking is not

Retention per target across 9 comparable arms:

| target | range | spread |
|---|---|---|
| bacteroidetes | 0.583 – 1.001 | **0.417** |
| crenarchaeota | 0.133 – 0.501 | 0.368 |
| insecta | 0.298 – 0.664 | 0.366 |
| streptophyta | 0.216 – 0.570 | 0.353 |
| spirochaetes | 0.669 – 1.007 | 0.338 |
| … | | |
| betaproteobacteria | 0.942 – 0.996 | 0.053 |

**The 8-against-6 separation holds in all 9 testable arms**, gap 0.153 to 0.238. The
ranking *within* bacteria does not: bacteroidetes moves 0.42 between arms. Present
the split; do not present the ranking without this error bar.

### T1.1 — coverage and geometry are both real, neither explains the other away

Pfam coverage is a strong predictor of retention (cov_same_ec ρ = +0.848,
cov_any +0.838, n = 209), which is the competing explanation the backlog was worried
about. It does not win:

| predictor | ρ | partial given proxy-A distance |
|---|---|---|
| cov_same_ec | +0.848 | **+0.694** |
| cov_any | +0.838 | +0.670 |
| geo_diff_abs | −0.793 | −0.616 |
| proxy_a_dist | −0.785 | — |
| med_homologues | +0.748 | +0.584 |
| geo_procrustes | −0.641 | −0.576 |
| pident_median | +0.793 | **+0.214** |

Coverage survives partialling on the geometry and the geometry survives partialling
on coverage. They are partly independent accounts. The archaeal control behaves as
predicted: coverage 0.79 but retention 0.33 / 0.36, well below the coverage trend.

Note the demotion of **sequence identity**: ρ = +0.793 marginally, +0.214 once
embedding distance is partialled out. Identity is largely a proxy for embedding
distance, not an independent axis.

### T1.2 — the n = 14 correlations shrink but survive

| predictor | ρ at n = 14 | ρ at n = 209 | mean within-source ρ |
|---|---|---|---|
| geo_diff_abs | −0.912 | −0.793 | −0.739 |
| geo_diff_inB_abs | −0.925 | −0.789 | −0.694 |
| proxy_a_dist | −0.811 | −0.785 | −0.724 |
| geo_procrustes | −0.925 | −0.641 | −0.669 |
| geo_logit_spread | −0.930 | −0.630 | −0.766 |
| geo_gap_ratio | −0.543 | **−0.065** | −0.264 |

Every predictor is weaker off the gamma row, one (`gap_ratio`) collapses to nothing
(p = 0.39), and the within-source analysis corroborates the survivors. The n = 14
numbers were optimistic by roughly 0.1–0.3 in ρ but were not artifacts. **Report the
n = 209 column.**

For `r_star_budget` the same predictors run +0.51 at best (was +0.82 at n = 14),
which is the same story as §2.

### T1.7 — the loss is uniform across enzyme chemistry (a clean null)

Class-level retention against the source-target prevalence difference for that EC
top-level class: **ρ = −0.048 over 1,028 class × target cells.** Prevalence shift
does not explain which chemistry survives. That strengthens the representational
account over label-marginal shift.

There is a gradient, and it is not prevalence-driven:

| EC top class | n | mean retention |
|---|---|---|
| 1 oxidoreductases | 193 | 0.509 |
| 3 hydrolases | 230 | 0.513 |
| 2 transferases | 253 | 0.568 |
| 4 lyases | 115 | 0.573 |
| 5 isomerases | 97 | 0.584 |
| 6 ligases | 86 | 0.666 |
| 7 translocases | 54 | **0.785** |

Oxidoreductases and hydrolases transfer worst, translocases best. This is the
biologically interesting sentence the thesis could not previously write.

### T1.8 — length is a null, length *distribution* is not

`median_length_target` ρ = −0.085 (p = 0.53) and `median_length_ratio` +0.066
(p = 0.42): clean nulls, as designed. But `length_ks` (−0.759, partial −0.577) and
`ec_composition_jsd` (−0.745, partial −0.524) both survive. Those are distributional
distances, so they are more distance measures rather than a mechanism.

The dissociation holds and should be quoted: **euryarchaeota (310 aa) and
crenarchaeota (312 aa) are the two shortest groups in the corpus**, shorter than
gammaproteobacteria (327 aa), yet retain 0.36 and 0.33. Length cannot be the
mechanism.

### T1.3 — the dip is not a second estimator, and the test cannot settle it

Mean normalised dip falls monotonically with r in every arm (P = 500: 0.350, 0.333,
0.279, 0.171, 0.114, 0.061, 0.027, 0.016), so the axis is real. Agreement with
`r*_budget` is negative (Kendall tau −0.21 to −0.65 in 7 of 8 arms).

**Do not report that as a finding.** I checked for censoring — there is none, 0 of 14
targets fail to enter the band — but the reason is worse: `r*_budget` is 0.0 for 10
of 14 targets, so there is almost no variance to correlate against, and the per-
target noise band is tight for easy targets and wide for hard ones, which
manufactures the sign. The honest statement is that this test cannot answer the
question, and answering it needs a band definition that is not itself
target-dependent.

## 7. T2.2 and T2.3 — the BLAST arms, which are the uncomfortable ones

BLAST ran out of every source group, not just gammaproteobacteria: 15 databases,
225 hits files, 1h17 on 32 cores. It was budgeted at a day.

### The identity correlation survives at n = 209

| predictor | ρ (n = 209) | partial given proxy-A | ρ (gamma only, n = 14) |
|---|---|---|---|
| pident_mean | +0.781 | +0.483 | +0.793 |
| pident_median | +0.752 | +0.488 | +0.807 |
| frac_below_40 | −0.762 | −0.490 | −0.815 |
| pident_p10 | +0.702 | +0.462 | +0.820 |
| frac_nohit | −0.521 | −0.404 | −0.859 |

All at group-permutation p < 0.0005. **The most quotable axis in the project is now
supported on 209 pairs instead of 14.**

### …but the ~40% boundary is a prokaryote phenomenon

Asked relative to each source's own clade (near targets = same clade), median
identity separates near from far in **11 of 15 sources**: all 9 bacterial and both
archaeal sources, gaps +0.2 to +5.4 points. **All four eukaryotic sources fail**
(gaps −0.1 to −4.2): a plant's EC-annotated enzymes are not more similar to a fungus's
than to a bacterium's.

Pooled, within-clade pairs run median 41.4% and outside-clade 37.3%, with ranges
that overlap (within min 36.7, outside max 41.5). The README's "36.7% to 42.0%" is
the **tightest row in the corpus**, not a typical one, and it must be labelled as
the gammaproteobacteria row.

### T2.3 — matched BLAST against the embedding: BLAST wins 14 of 15

The comparison the backlog called the single most consequential unrun experiment.
Same query proteins, same shared-EC label set, same macro-F1 denominator, self-hits
dropped, and — the part that was missing before — **the same source data**: the
probe trains on exactly the protein set the BLAST database contains, because the
database is the whole source group and capping the probe at 6,000 hands BLAST a
10× data advantage.

| target | embedding | BLAST | median pident |
|---|---|---|---|
| betaproteobacteria | 0.772 | **0.805** | 61.2 |
| alphaproteobacteria | 0.686 | **0.758** | 50.2 |
| epsilonproteobacteria | 0.521 | **0.677** | 44.5 |
| actinobacteria | 0.495 | **0.682** | 42.9 |
| firmicutes | 0.534 | **0.662** | 45.1 |
| spirochaetes | 0.552 | **0.656** | 41.9 |
| bacteroidetes | 0.526 | **0.628** | 43.3 |
| cyanobacteria | 0.535 | **0.604** | 45.8 |
| euryarchaeota | 0.258 | **0.431** | 36.7 |
| crenarchaeota | 0.169 | **0.356** | 35.5 |
| streptophyta | 0.183 | **0.337** | 35.7 |
| ascomycota | 0.218 | **0.311** | 34.8 |
| insecta | 0.231 | **0.277** | 34.7 |
| vertebrata | 0.155 | **0.275** | 34.3 |
| *gammaproteobacteria (within-group control)* | **0.915** | 0.890 | 99.4 |

Matching the source volume roughly halved BLAST's margin — actinobacteria went from
0.839 vs 0.518 to 0.682 vs 0.495 — but did not reverse it anywhere except the
within-group control.

**Three things keep this from being fatal, and all three must be said, not just the
convenient one.**

1. The embedding comparator is a **linear probe**, deliberately the weakest readout
   in the project. The MLP arm's zero-shot numbers are the fair comparator and have
   not been put on the same footing yet. That run is the obvious next job.
2. AUG7 §8 already established the defence: **inside a single Pfam**, where homology
   is constant by construction and a lookup is impossible, ESM-C separates EC at
   0.95–1.00 and transfers at 0.75–0.83 zero-shot against a label-shuffle floor of
   0.19–0.29, with r* = 0.05–0.10. Homology search cannot do that, because there is
   no homology signal left to use.
3. This is the expected regime. In ProteInfer's own clustered split BLAST beats the
   CNN; in CARE, BLASTp beats CLEAN at 30–50% identity. EC is where homology
   baselines are hardest to beat.

The honest framing for the chapter is therefore: **at ladder level the embedding
does not beat homology search, and the thesis should stop implying otherwise; the
embedding's contribution is within-family functional resolution, which is where it
beats homology search outright.** Per-protein identity supports this — pooled
ρ(identity, correct) = +0.319 but mean within-group ρ = +0.150, so roughly half the
apparent identity effect is between-group composition.

## 8. Three bugs of mine, recorded because two produced wrong conclusions

1. **`GROUPS` is a bash built-in.** It holds the caller's group-id array; bash
   silently discards an assignment to it, and `$GROUPS` then expands to a numeric
   GID. The first BLAST job "completed" in 20 seconds having BLASTed nothing.
2. **Hard-coded clade membership, twice.** I wrote "the 8 bacterial targets" as a
   literal set. It included gammaproteobacteria (the ladder's source, never its
   target, but a target for all 14 other sources) and omitted epsilonproteobacteria.
   First it made the 8-against-6 separation read `False` in all 12 arms; then it made
   the identity boundary look like a gammaproteobacteria-only artifact. Both
   conclusions were wrong and both were about to be reported.
   Fix: `src/clades.py` is now the only definition, and callers ask
   `targets_by_clade(source)` rather than writing a set.
3. **An O(n²) membership filter** inside a list comprehension recomputed the
   class-count sets once per protein; the first smoke test hung on it.

Landmine 2 of the brief said match your class counts; the new one to add is **match
your clade definitions, and never inline them.**

## 9. What to decide, and what to run next

### For the meeting — decisions, not experiments

* **T3.1, whitened coordinates: the case is now much stronger than on 13 August.**
  In raw coordinates §3 shows the subspace decomposition is measuring anisotropy,
  §4 shows the same for the shift decomposition's sign, and both β diagnoses reduce
  to the same thing. Either the generator moves to whitened coordinates and the July
  v2 sweeps become historical, or the document states plainly that the generator
  models whitened embeddings while every comparison is against raw ones.
* **T3.2, β:** drop it. It is now failing for a reason we can name rather than an
  unexplained sign error.
* **T3.5, EC-like oracle labels:** still the empty cell, still blocked on the
  purity/coverage conflict, and §7 raises the stakes — a synthetic arm with
  Pfam-like labels cannot speak to the within-family resolution that is the
  embedding's actual contribution.

### Ready to run

1. **The MLP zero-shot against BLAST on identical footing** (§7 item 1). This is the
   number the chapter now turns on and it is a few hours.
2. **Within-Pfam r* at `--min_n 8`** to widen AUG7 §8 beyond three Pfams with
   K = 3–5. That result is now load-bearing and rests on a thin basis (landmine 3).
3. **Differential-motion injection in whitened coordinates.** §4 says the component
   that matters is the one the isometry rule forbids injecting. In whitened
   coordinates with an explicit variance correction it may be injectable with the
   ceiling held fixed — and if it is, that is the shift experiment worth having.
4. **A target-independent dip band** so T1.3 can be asked properly.

Nothing in this note needed a GPU except §5.

---

## 10. The three follow-up runs, same day

Submitted after §1–§9 was written, all complete: `47542345` MLP against BLAST,
`47542346` the within-Pfam arm widened, `47542347` the synthetic sweep with its
estimator repaired.

### T3.4 — the estimator repair changes the synthetic result, in both directions

Three arms on identical universes: a July reproduction, the repair, and an α sweep
under the repair.

| d | zero-shot | ceiling | r* July (single pass) | r* repaired |
|---|---|---|---|---|
| 0.00 | 0.922 | 0.923 | 0.0 | 0.0 |
| 0.25 | 0.859 | 0.912 | 0.0 | 0.0 |
| 0.50 | 0.610 | 0.903 | 0.75 | **0.5** |
| 0.75 | 0.268 | 0.881 | 0.75 | **0.5** |
| 1.00 | 0.054 | 0.888 | **never** | **0.75** |
| 1.25 | 0.015 | 0.861 | **never** | **1.0** |
| 1.50 | 0.006 | 0.875 | **never** | **0.75** |
| 2.00 | 0.001 | 0.832 | **never** | **0.75** |

**The "never recovers beyond d = 0.75" points were artifacts of the single-pass
adaptation.** With multiple passes and early stopping on a held-out slice of the
pool, the model recovers at every distance tested. `r*` is also lower everywhere it
was already defined.

The shape of the law changes with it. As reported, `r*(d)` rose and then diverged;
repaired, it rises and then **plateaus at 0.75–1.0**. The statement that survives is
"past about one unit of distance you need roughly three-quarters of your budget from
the target, and no more than that", not "past 0.75 it is hopeless". The July curve
was describing its own step budget.

**The α result survives the repair and gets considerably stronger.** This is the
central synthetic claim — that `r*` depends on the *type* of shift, not its size:

| α | 0.0 (pure concept) | 0.25 | 0.5 | 0.75 | 1.0 (pure covariate) |
|---|---|---|---|---|---|
| r* at d = 0.5 | 0.75 | 0.5 | 0.5 | 0.3 | **0.05** |
| r* at d = 1.0 | 1.0 | 0.75 | 0.75 | 0.75 | **0.1** |

At d = 0.5 the spread is 0.05 against 0.75, a factor of **15**. The README currently
reports this as "30% against 75%" from the unrepaired sweep, a factor of 2.5. The
repaired numbers make the claim much sharper, and the ceiling stays between 0.897
and 0.911 across the whole α row, so the bar is not moving underneath it.

### The within-Pfam defence, widened from 3 Pfams to 6

`--min_n 8` admits three more families (two further ones, `PF01261` and `PF01266`,
were found but could not afford even a 100-protein budget on the target side).

| Pfam | K | ceiling | zero-shot | shuffled ceiling | shuffled zero-shot | r* (P = 200) |
|---|---|---|---|---|---|---|
| `PF00005` ABC transporter | 5 | 0.956 | 0.817 | 0.170 | 0.156 | 0.05 |
| `PF04055` radical SAM | 5 | 0.926 | 0.722 | 0.185 | 0.175 | 0.20 |
| `PF00291` PLP-dependent | 4 | 0.810 | **0.486** | 0.215 | 0.214 | **1.0** |
| `PF00561` α/β hydrolase | 4 | 0.934 | 0.731 | 0.247 | 0.236 | 0.10 |
| `PF00701` DHDPS | 3 | 0.945 | 0.662 | 0.328 | 0.271 | 0.05 |
| `PF00266` aminotransferase V | 3 | 0.973 | 0.826 | 0.317 | 0.292 | 0.10 |

**Every one of the six transfers well above its own label-shuffle floor**, so the
core claim holds on twice the basis: with homology constant by construction, ESM-C
carries real functional signal that no Pfam→EC lookup could supply.

Two honest corrections to AUG7 §8. The zero-shot range widens *downward*, 0.49–0.83
rather than 0.75–0.83, because `PF00291` is much weaker than any of the original
three. And the individual numbers all drift down a little (`PF00005` 0.833 → 0.817,
`PF04055` 0.812 → 0.722, `PF00701` 0.747 → 0.662) because `--min_n 8` admits rarer
classes and makes each problem harder. **"The threshold is small" now holds on 5 of
6 rather than 3 of 3** — `PF00291` needs the entire budget.

### T2.3 revisited — the MLP does not rescue the comparison

The defence offered in §7 was that the comparator was a linear probe. It was, and
the MLP is better — but not nearly enough.

| target | median identity | MLP ± sd | probe | BLAST | margin | BLAST no-hit |
|---|---|---|---|---|---|---|
| vertebrata | 33.9 | 0.393 ± 0.013 | 0.396 | 0.364 | **+0.029** | 0.423 |
| ascomycota | 34.0 | 0.435 ± 0.018 | 0.447 | 0.532 | −0.097 | 0.411 |
| insecta | 34.0 | 0.519 ± 0.025 | 0.512 | 0.513 | **+0.006** | 0.387 |
| crenarchaeota | 34.6 | 0.495 ± 0.012 | 0.440 | 0.770 | **−0.275** | 0.161 |
| streptophyta | 35.4 | 0.422 ± 0.018 | 0.382 | 0.514 | −0.093 | 0.319 |
| euryarchaeota | 36.2 | 0.596 ± 0.007 | 0.543 | 0.738 | −0.142 | 0.198 |
| spirochaetes | 40.6 | 0.908 ± 0.019 | 0.935 | 0.995 | −0.087 | 0.006 |
| actinobacteria | 42.3 | 0.854 ± 0.015 | 0.819 | 0.938 | −0.084 | 0.046 |
| bacteroidetes | 42.6 | 0.928 ± 0.008 | 0.907 | 0.965 | −0.038 | 0.025 |
| epsilonproteobacteria | 45.2 | 0.911 ± 0.017 | 0.863 | 0.980 | −0.069 | 0.015 |
| firmicutes | 45.4 | 0.872 ± 0.018 | 0.819 | 0.968 | −0.096 | 0.036 |
| cyanobacteria | 46.3 | 0.849 ± 0.010 | 0.831 | 0.909 | −0.060 | 0.052 |
| alphaproteobacteria | 50.3 | 0.957 ± 0.011 | 0.929 | 0.988 | −0.031 | 0.015 |
| betaproteobacteria | 60.9 | 0.988 ± 0.001 | 0.984 | 0.993 | −0.005 | 0.011 |
| **mean** | | **0.723** | 0.701 | **0.798** | | |

* The MLP beats the probe on 12 of 14, by **+0.022 on the mean**. Real, and small.
* **BLAST still wins on 12 of 14.** The two exceptions are vertebrata (+0.029, about
  2 seed standard deviations) and insecta (+0.006, well inside noise). Only
  vertebrata is arguably a win at all.
* **There is no identity-dependent crossover.** Spearman between median identity and
  the MLP-minus-BLAST margin is **+0.191** (n = 14) — no relationship. The tempting
  story that the embedding takes over in the twilight zone is not supported:
  crenarchaeota sits at 34.6% identity and is the single worst loss, −0.275.
* BLAST wins on vertebrata *despite having no hit at all for 42% of the queries*,
  which score as wrong. That is how large its advantage is where it has hits.
* The hybrid — best hit where there is one, model prediction where there is not —
  is the best system, but only just: mean 0.811, ahead of the better of the two by
  **+0.011** on average and +0.078 at most, winning outright on 9 of 14.

**So the §7 framing stands and the escape route is closed.** At ladder level the
representation does not beat homology search, and a stronger readout does not change
that. What survives is the within-Pfam result above, on six families rather than
three: where homology is held constant, the embedding carries function that homology
search cannot reach. That is the claim the chapter should make, and it is narrower
than the one the document currently implies.
