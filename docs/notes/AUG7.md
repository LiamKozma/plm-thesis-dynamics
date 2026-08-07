# Aug 7 — r* for EC (function) labels on real data, and three ways the estimator lied

Answers [`BRIEF_ec_recovery_threshold.md`](BRIEF_ec_recovery_threshold.md). That
brief left four design questions open (§4) and asked for the empty cell in its
table to be filled: a recovery threshold measured against **EC labels on real
data**, and a statement of how it depends on a **distance a biologist or an ML
person would actually ask for**.

**The result, stated once, up front.** On a fixed budget of 500 EC labels:

> **Every bacterial target needs none of that budget spent on its own proteins —
> a gammaproteobacteria-trained model beats what 500 of the target's own proteins
> can train from scratch. Every archaeal and eukaryotic target needs roughly a
> third to a half of it. Fourteen targets out of fourteen, split exactly on the
> domain of life.**
>
> The boundary sits between **36.7% and 42.0%** median sequence identity to the
> nearest training protein — straddling the ~40% threshold the enzyme-function
> literature independently gives for reliable EC-level-3 annotation transfer.
> Nothing was tuned to land there.

**But the more transferable finding is that the estimator inherited from the
synthetic sweep does not survive contact with real embeddings, and that five
separate versions of it would each have produced a confident wrong answer.** All
five are documented in §3, because each is exactly the kind of thing §5 of the
brief exists to prevent. Everything in §5 is from the repaired estimator.

Two of the brief's own open questions turned out to be **answered by the data
rather than by a decision** (§2), and one result — §8 — is a direct threat to the
thesis's framing that is now quantified rather than suspected.

---

## 0. What was built, and what is running

| script | what it does |
|---|---|
| `src/ec_recovery_threshold.py` | the r* ladder itself: EC labels, real ESM-C, gamma → 14 groups |
| `src/ec_rstar_allpairs.py` | the same question over all 210 ordered pairs with a linear probe |
| `src/ec_distance_metrics.py` | label-free distances (MMD, energy, Fréchet, proxy-A) + taxonomic rank from NCBI lineage |
| `src/ec_seq_identity.py` | % identity to the nearest source protein via BLAST+, and the BLAST-nearest-neighbour EC baseline |
| `src/ec_homology_confound.py` | Pfam→EC lookup baseline, EC recovery *within* one Pfam, label-shuffle floor |
| `src/ec_rstar_regress.py` | joins every r* to every distance; group-level permutation null |

Batch scripts: `slurm/ec_rstar_{pilot,ladder,allpairs,report}.slurm`,
`slurm/ec_{distances,seq_identity,homology_confound}.slurm`.

Jobs submitted (all writing to `/scratch/lmk04992/ec_rstar/`):

| job | id | what | state at write-up |
|---|---|---|---|
| `ec_distances` | 47292604 | label-free distances, 210 pairs | **complete** |
| `ec_seqid` | 47292605 | BLAST identity + BLAST-NN EC baseline, 15 groups | **complete** |
| `ec_homology` | 47292610 | homology-confound arm | **complete** |
| `ec_rstar_fast` | 47292734 | reduced 14-target ladder, P = 500 | **complete** — §5 |
| `ec_rstar_ladder` | 47292627 | the four full r* arms | running (1/14 targets of arm 1) |
| `ec_rstar_allpairs` | 47292628 | 210-pair linear-probe r*, rerun | running |
| `ec_rstar_report` | 47292629 | join + regression, `afterany` on the two above | queued |

Jobs 47292608/09/11 were the first attempt; they were cancelled and resubmitted
as 47292627/28/29 after §3(d). Their output is kept, not deleted.

`afterany`, not `afterok`, on purpose: a partial join is useful, a missing one is
not, and `ec_rstar_regress.py` skips inputs that are absent.

---

## 1. The four design questions, decided

### Q4 first — which estimator — because it constrains everything else

**Decision: the MLP adaptation loop, matching `run_distance_sweep_v2.py`, is the
primary estimator; the linear probe runs as a separate wide scan and its r* is
reported in its own column and never pooled with the other.**

The whole point of filling this cell is that the number should be comparable to
the synthetic `r*(d)` curve, and a logistic regression is not the same estimator
as a warm-started MLP. But the linear probe is the only thing cheap enough to run
over all 210 pairs, and 14 points cannot survive a permutation null over 15
groups (landmine 4). So both, in separate columns, with the 14 pairs they share
used as a check on whether they at least *rank* targets the same way.

§3 explains why "matching `run_distance_sweep_v2`" turned out to require
deliberately breaking one of its choices.

### Q1 — which experiment

**Decision: (a), the real EC ladder from one source, as the brief recommends —
plus (b) as the wide scan, and the (c)/Q3 novelty axis constructed rather than
found (see §2).**

Source is `gammaproteobacteria` (64,791 proteins, the largest group); targets are
the 14 other usable groups. Four arms, kept separate because their numbers are
not interchangeable:

| arm | label set | why |
|---|---|---|
| `_pair` | shared by that pair, min_n 15 | most data per pair; **K varies 30…112**, so K is a confound carried into the regression as a covariate |
| `_matched` | shared by **all 15 groups**, min_n 10 | identical K-way problem for every target, so r* is comparable by construction (landmine 2) |
| `_novelty` | `_pair`, with ν of the classes deleted from the source side | the constructed functional-novelty axis of §2 |
| `_sizematched` | `_pair`, source model given as many training proteins as the target has | removes gammaproteobacteria's 40× data advantage; the arm actually comparable to the synthetic sweep |

### Q2 — which distance

**Decision: all four families, in this priority order.** The metric has to be
computable *before* any target protein is labelled, or it cannot tell anyone
their budget in advance.

1. **% identity to the nearest source-group protein** (BLAST+). The number
   biologists already have a feel for, and the one with a calibrated literature
   threshold to test against — EC level 3 is reported to transfer reliably above
   ~40% identity (Tian & Skolnick 2003; Addou et al. 2009), 30% on the more
   optimistic reading (Todd et al. 2001). If r* turns sharply near there, that is
   the quotable sentence.
2. **Taxonomic rank of the split**, free from the NCBI lineage already in
   `raw/ec_swissprot_raw.tsv`. Ordinal, so it is the framing rather than the
   regressor.
3. **Label-free distribution distances**: MMD (RBF, median heuristic), energy
   distance, Fréchet, and **proxy-A distance** — the last because it is literally
   the quantity in the Ben-David et al. (2010) adaptation bound, so it connects
   the measurement to theory rather than to a correlation.
4. **The internal geometry** already computed for all 210 pairs in
   `analysis/beta_diagnosis.json` — joined in, not recomputed.

### Q3 — the two conflated distances

**Decision: report them separately, and construct the second one, because it does
not occur naturally in this dataset.** See §2 — this stopped being a judgement
call once the counts were in.

---

## 2. Two of the open questions were answered by the data, not by a decision

### There is almost no natural functional novelty to measure

The brief worried that "same ECs, new organisms" and "new ECs absent from the
source" were being conflated and must not average together silently. Counting
first:

| target | EC classes with ≥15 proteins in the target and **0** in gammaproteobacteria |
|---|---|
| actinobacteria, alphaproteobacteria, bacteroidetes, betaproteobacteria, crenarchaeota, cyanobacteria, epsilonproteobacteria, insecta, spirochaetes | **0** |
| firmicutes | 1 (64 proteins) |
| ascomycota | 3 (49) |
| euryarchaeota | 3 (98) |
| streptophyta | 4 (99) |
| vertebrata | 5 (214) |

**Gammaproteobacteria contains essentially every EC class at level 3 that any
other group has.** So in this dataset the taxonomic ladder is nearly pure
covariate shift, and the functional-novelty axis cannot be *found* — the largest
available case is 5 classes and 214 proteins.

That is worth stating on its own: it means every r* here measures "same enzymes,
new organisms", and it means the λ term in the Ben-David bound (the error of the
best hypothesis that works on both domains) is small by construction. It also
means a paper claiming to study functional novelty via a taxonomic split on
Swiss-Prot is not doing so.

So novelty is **constructed**: `--holdout_frac ν` deletes ν of the shared classes
from the source training data and from the source half of the pool, leaving them
reachable only through the target half. The label space stays the union so the
model can still emit them. ν = 0, 0.25, 0.5. Matched, well-defined, and not
confounded with group size — which the natural version would have been, since the
only way to get novel classes is to pick a small source group.

### The class-count-matched label set is small, and that is a real constraint

Intersecting across all 15 groups:

| min_n | classes shared by every group | proteins on that set in the smallest group (insecta) |
|---|---|---|
| 5 | 26 | 596 |
| 10 | **20** | 788 |
| 15 | 16 | 500 |

The 20 classes at min_n = 10 are core metabolism — `1.1.1, 1.2.1, 2.1.1, 2.3.1,
2.4.2, 2.5.1, 2.7.1, 2.7.4, 2.7.7, 2.8.1, 3.1.1, 3.1.3, 3.6.1, 3.6.5, 4.1.1,
4.2.1, 5.3.1, 6.1.1, 6.3.4, 6.3.5`. This is a biased and probably easy subset,
which is exactly why it is a *control arm* rather than the primary: it buys
comparability across targets at the cost of representativeness. Both arms are
reported; if they disagree, neither gets quoted alone.

---

## 3. Five ways the estimator lied, and what each would have cost

This is the part worth reading. The r* definition was inherited from
`run_distance_sweep_v2.py` and looks innocuous. On real ESM-C embeddings with
~90–110 EC classes, three separate things in it are wrong, and each produces a
confident number that means something other than what it appears to.

### (a) The ceiling is not automatically a bar

First working run, `gamma → euryarchaeota`: **zero-shot 0.240 against a ceiling of
0.219.** The source-trained model beat the target-trained one outright, so r* = 0
trivially.

Cause: the groups differ in size by 40× (gammaproteobacteria 64,791, insecta
1,970). "Train a model on the target" is not a stiff bar when the target has a
fortieth of the data. The synthetic sweep never hits this because it draws
`n_source` samples for both sides.

Fix: report **three** ceilings next to every r* — the full-reservoir one, a
**size-matched** one where the target model gets exactly as many training
proteins as the source model did, and a **budget** one trained on only P target
proteins. And run a whole arm (`_sizematched`) with the source model's data
advantage removed. Landmine 9 says always report the ceiling; the real lesson is
that on real data one ceiling is not enough to say what the bar even is.

### (b) One pass over the pool measures the step count, not the pool

`run_distance_sweep_v2.adapt` takes exactly one pass over the pool. Measured at a
pool of 4,000 all-target proteins on `gamma → vertebrata`:

| | macro-F1 on the target test set |
|---|---|
| one warm-started pass over 4,000 target proteins | **0.656** |
| from scratch, 20 epochs, on **the same 4,000 proteins** | **0.919** |

A 0.26 gap on identical data. An r* defined on the one-pass number would have
said "vertebrata needs more target labels than the budget allows" when what it
actually measured is that one pass of Adam is not enough. That is a wrong
conclusion of precisely the shape §5 of the brief is about, and it would have
been invisible — the number is plausible and the curve is monotone.

### (c) …but a fixed number of passes is wrong at the other end

Setting `adapt_epochs = 20` fixed vertebrata and broke betaproteobacteria: on a
500-protein pool, 20 passes **overfit**, and the adapted model finished at 0.848
having started at 0.939. Fixed-1 underfits large pools; fixed-20 overfits small
ones. The pool size varies by an order of magnitude across this sweep, so no
constant is right.

Fix: **every model in the sweep — the source model, all three ceilings, and every
adapted model — is now trained by the same rule.** Hold out a stratified 20% of
its own training data, train up to 30 epochs, keep the checkpoint with the best
validation macro-F1. Uniform across conditions, and it is what someone spending a
real annotation budget would do. Note the validation slice comes from the *pool*,
so at r = 0 it is entirely source data — not a leak, but the honest constraint
that with no target labels you cannot do target model selection either.

The one-pass number is kept as its own column (`f1_1pass`, `r_star_1pass`)
because it is the synthetic-comparable one and because the first-pass trajectory
is where the negative-transfer dip lives. It is never pooled with the other.

### (d) r* against the full ceiling is not a well-posed question at a small budget

The first 210-pair linear-probe run came back **100% censored at a budget of 200
and 80% censored at 500** — that is, in 167 of 209 pairs the adapted model never
reached 90% of the ceiling at *any* mixing fraction, including r = 1.0.

That is not a property of the shift. The ceiling was a model trained on up to
6,000 target proteins; asking whether 500 labels can reach 90% of that is asking
a budget question, and the answer is no regardless of how close the two groups
are. r* defined this way is dominated by P and barely reads the shift at all.

Fix: define the primary threshold against the **budget-matched ceiling** —
a from-scratch model trained on P *target* proteins:

> **r\*_budget** = the smallest fraction of a budget of P labels that must be
> target-native to match spending the *whole* budget on target data.

r = 1.0 reaches that bar by construction, so it is never censored for want of
budget, and it isolates the composition question from the budget question. r*
against the full ceiling is still reported — it is the honest absolute number,
and how often it is censored is itself a finding — but it is not the headline.

### (e) A fifth, smaller one: r* must allow doing nothing

With the repaired estimator, `gamma → betaproteobacteria` at P = 500 has zero-shot
at **99.8% of the ceiling** and still scored r* = 1.0, because fine-tuning on 500
proteins damages an already-excellent model at every r on the grid.

Nobody ships a model worse than the one they started with. `r_star_noadapt` takes
the better of {adapted, un-adapted} at each r, which restores monotonicity and
makes r* mean "how much target data before the best available model clears the
bar". Both columns are reported.

---

## 4. What each arm is, in one place

| quantity | definition |
|---|---|
| pool | **fixed size P**; fraction r of it target-labelled, 1−r source-labelled. Total labels constant, only composition changes. P ∈ {200, 500, 1000}. |
| r grid | 0, .05, .1, .2, .3, .5, .75, 1.0 |
| ceiling | from-scratch target model, early-stopped, on the whole target training reservoir |
| bar | 0.9 × ceiling |
| r* | smallest r on the grid whose mean final macro-F1 reaches the bar; `nan` = never |
| scaler | fit on **source train only** — fitting it on the target is unsupervised domain adaptation in disguise and would silently shrink r* |
| seeds | 42, 43, 44 |
| test | held-out target, stratified, disjoint from the reservoir |

Sweeping P at all is deliberate: Ben-David et al. (2010) give an optimal
source/target mixing weight with a phase transition at an *absolute* target
sample count, which predicts r* should scale roughly as 1/P. Three budgets is
enough to see whether it does.

---

## 5. r* on real EC labels — the first numbers

**These are from the reduced arm (job 47292734, `ladder_fast/`), not the full
ladder.** It runs all 14 targets but with 2 seeds instead of 3, one budget
(P = 500) instead of four, 8,000 source / 4,000 target training proteins instead
of 20,000 / 10,000, and 20 adaptation epochs instead of 30. It was submitted
alongside the full ladder specifically so that this note would have measured
numbers in it rather than a promise. The full ladder (47292627) supersedes it and
was still running at write-up; **if the two disagree, the full ladder wins.**

Sorted by zero-shot ÷ ceiling. `barBud` = 0.9 × a from-scratch model on 500
target proteins; `r*bud` is the smallest target fraction of a 500-label budget
that reaches it.

| target | K | ceiling | zero-shot | 0shot÷ceil | barBud | **r\*bud** | r\* (abs) |
|---|---|---|---|---|---|---|---|
| betaproteobacteria | 100 | 0.948 | 0.932 | 0.983 | 0.544 | **0.0** | 0.2 |
| spirochaetes | 45 | 0.916 | 0.825 | 0.900 | 0.685 | **0.0** | 0.5 |
| bacteroidetes | 40 | 0.926 | 0.808 | 0.873 | 0.741 | **0.0** | 0.5 |
| alphaproteobacteria | 105 | 0.903 | 0.775 | 0.858 | 0.441 | **0.0** | nan |
| cyanobacteria | 74 | 0.960 | 0.792 | 0.825 | 0.627 | **0.0** | nan |
| epsilonproteobacteria | 67 | 0.963 | 0.754 | 0.783 | 0.688 | **0.0** | 0.75 |
| firmicutes | 112 | 0.863 | 0.674 | 0.781 | 0.370 | **0.0** | nan |
| actinobacteria | 102 | 0.891 | 0.532 | 0.596 | 0.411 | **0.0** | nan |
| insecta | 30 | 0.873 | 0.385 | 0.441 | 0.701 | **0.5** | 0.75 |
| euryarchaeota | 70 | 0.937 | 0.393 | 0.419 | 0.434 | **0.5** | nan |
| crenarchaeota | 40 | 0.959 | 0.357 | 0.372 | 0.755 | **0.5** | 0.75 |
| streptophyta | 88 | 0.821 | 0.218 | 0.266 | 0.348 | **0.5** | nan |
| ascomycota | 78 | 0.876 | 0.219 | 0.249 | 0.386 | **0.5** | nan |
| vertebrata | 89 | 0.816 | 0.145 | 0.177 | 0.247 | **0.5** | nan |

### r* takes two values, and the split is exactly the domain of life

**Every one of the eight bacterial targets has r\*bud = 0. Every one of the six
archaeal and eukaryotic targets has r\*bud = 0.5. Fourteen out of fourteen, no
exceptions.**

The two halves mean different things, and both are worth stating plainly:

* **Bacteria → bacteria: you need no target labels at all.** r*bud = 0 because
  the zero-shot gammaproteobacterial model *by itself* already beats a model
  trained from scratch on 500 of the target's own proteins — 0.932 vs 0.605 for
  betaproteobacteria, and still 0.532 vs 0.456 for actinobacteria, the worst
  case. Spending any of a 500-label budget on target data is worse than spending
  none of it.
* **Out of bacteria: about half the budget must be target-native.** For all six,
  the curve crosses the bar between r = 0.3 and r = 0.5.

### But the grid cannot resolve the second number, and the honest reading is 0.3–0.5

The uniform 0.5 is partly grid granularity. Looking at the curves, five of the
six non-bacterial targets fall *just* short at r = 0.3:

| target | barBud | F1 at r=0.3 | F1 at r=0.5 |
|---|---|---|---|
| insecta | 0.701 | 0.681 | 0.777 |
| euryarchaeota | 0.434 | 0.416 | 0.496 |
| ascomycota | 0.386 | 0.371 | 0.450 |
| streptophyta | 0.348 | 0.322 | 0.390 |
| vertebrata | 0.247 | 0.220 | 0.294 |
| crenarchaeota | 0.755 | 0.630 | 0.784 |

So the true threshold is somewhere in 0.3–0.5 for all six and the eight-point
grid puts it at 0.5. It should be quoted as **"roughly a third to a half"**, not
as 0.5. The curves are also visibly noisy at 2 seeds — vertebrata goes 0.241 at
r = 0.2 then *down* to 0.220 at r = 0.3 — which is another reason to wait for the
3-seed full ladder before hardening the number.

### The absolute r* is censored for 8 of 14, as §3(d) predicted

The `r*` column is `nan` wherever 500 labels cannot reach 90% of what the target's
whole reservoir buys. It is censored exactly for the targets with many classes
and a hard task, which is a statement about the budget, not the shift. This is
the column that would have been reported as the headline under the original
estimator, and it would have said "most of the tree of life is unrecoverable"
when what it measures is "500 is fewer than 4,000".


## 6. The distances themselves — jobs 47292604 and 47292605, both complete

### Sequence identity to the nearest gammaproteobacterial protein

1,500 query proteins per group, blastp against all 64,791 gammaproteobacterial
proteins, best non-self hit. `blastNN F1` is the macro-F1 of transferring that
best hit's EC label — the homology baseline that has to be beaten.

| target | median % id | frac < 30% | no hit | **blastNN EC F1** |
|---|---|---|---|---|
| *gammaproteobacteria (self control)* | *99.4* | *0.003* | *0* | *0.974* |
| betaproteobacteria | 61.2 | 0.006 | 1 | 0.942 |
| alphaproteobacteria | 50.2 | 0.019 | 1 | 0.896 |
| cyanobacteria | 45.5 | 0.039 | 11 | 0.790 |
| firmicutes | 45.0 | 0.057 | 5 | 0.738 |
| epsilonproteobacteria | 44.5 | 0.051 | 1 | 0.928 |
| bacteroidetes | 43.1 | 0.053 | 1 | 0.883 |
| actinobacteria | 42.8 | 0.064 | 5 | 0.747 |
| spirochaetes | 42.0 | 0.072 | 1 | 0.919 |
| euryarchaeota | 36.7 | 0.175 | 11 | 0.520 |
| crenarchaeota | 35.5 | 0.196 | 11 | 0.561 |
| streptophyta | 35.5 | 0.208 | 41 | 0.381 |
| ascomycota | 34.6 | 0.235 | 56 | 0.383 |
| insecta | 34.5 | 0.224 | 64 | 0.327 |
| vertebrata | 34.3 | 0.240 | 85 | 0.306 |

Three things worth noting.

* **The range is narrow and the structure is coarse.** Bacteria sit at 42–61%
  identity, archaea and eukaryotes all pile up at 34–37%. Median identity does
  *not* separate a plant from a vertebrate from an archaeon — they are all "about
  35%". So identity is an excellent axis for the bacteria→bacteria half of the
  ladder and a nearly degenerate one across the rest of it. Any claim of the form
  "r* is a function of identity" has to survive that.
* **The self-control is 99.4%, not ~40%.** Dropping self-hits was not enough:
  Swiss-Prot's gammaproteobacteria contain large numbers of near-identical
  proteins from different strains. The within-group row is therefore inflated by
  strain-level redundancy and should not be read as "what a typical same-group
  protein looks like".
* **Everything sits in or just above Rost's twilight zone.** Median identity is
  above the 30% line for every group, but 17–24% of eukaryotic and archaeal
  queries are below it, and the canonical EC-level-3 threshold is ~40%
  (Tian & Skolnick 2003; Addou et al. 2009). Half the ladder is below that.

### Label-free distances, gammaproteobacteria → each target

| target | MMD (RBF) | energy | proxy-A | shared lineage nodes | LCA rank |
|---|---|---|---|---|---|
| crenarchaeota | **0.4705** | 0.384 | 1.998 | 1 | — |
| ascomycota | 0.2779 | 0.242 | 1.951 | 1 | — |
| vertebrata | 0.2687 | 0.206 | 1.976 | 1 | — |
| insecta | 0.2557 | 0.200 | 1.962 | 1 | — |
| streptophyta | 0.2535 | 0.206 | 1.964 | 1 | — |
| euryarchaeota | 0.2531 | 0.190 | 1.962 | 1 | — |
| epsilonproteobacteria | 0.2325 | 0.157 | 1.916 | 3 | kingdom |
| spirochaetes | 0.2095 | 0.148 | 1.917 | 3 | kingdom |
| actinobacteria | 0.1499 | 0.098 | 1.889 | 2 | domain |
| firmicutes | 0.1325 | 0.085 | 1.860 | 2 | domain |
| bacteroidetes | 0.1016 | 0.062 | 1.831 | 3 | kingdom |
| cyanobacteria | 0.0687 | 0.051 | 1.778 | 2 | domain |
| betaproteobacteria | 0.0524 | 0.035 | 1.362 | 4 | phylum |
| alphaproteobacteria | 0.0454 | 0.026 | 1.600 | 4 | phylum |

* **MMD and energy distance agree closely and span 10×**, so unlike identity they
  do separate the far half of the ladder. They are the metrics with room to
  predict anything.
* **Crenarchaeota is the most distant group in embedding space — further than
  vertebrates.** This is the AUG4 §4 observation ("archaea is off the taxonomy
  axis") reappearing in a label-free statistic on the rebuilt dataset, and it is
  the clearest case where embedding distance and taxonomic intuition disagree.
* **Proxy-A distance is nearly saturated.** 12 of 14 pairs exceed 1.83 out of a
  maximum of 2, i.e. a linear discriminator separates source from target almost
  perfectly. It is the metric with the best theoretical claim — it is the
  estimator in the Ben-David bound — and on protein embeddings it has almost no
  dynamic range. Worth reporting as a negative result about the metric.
* **Fréchet distance came out uninformative** (all values 0.0–0.2 at the printed
  precision). With 3,000 samples in 960 dimensions the covariance is badly
  rank-deficient and the matrix square root is not trustworthy. Not used.
* **Taxonomic rank is coarse and NCBI-inconsistent**: bacteroidetes and
  epsilonproteobacteria share a "kingdom"-ranked node with gammaproteobacteria
  while actinobacteria and cyanobacteria share only a "domain"-ranked one, which
  does not match their MMD ordering. `n_shared_lineage` (4 = same phylum, down to
  1 = different domain of life) is the usable ordinal; the rank *names* are not.

## 7. Which distance predicts r*

With r*bud taking only two values, this reduces to: which distances separate the
same 8 from the same 6? **Four independent metric families all separate them
perfectly, and none can be distinguished from the others on this evidence.**

| metric | bacterial targets (r\*bud = 0) | non-bacterial (r\*bud = 0.5) | gap |
|---|---|---|---|
| shared lineage nodes | 2, 2, 2, 3, 3, 3, 4, 4 | 1, 1, 1, 1, 1, 1 | clean |
| median % identity | 42.0 – 61.2 | 34.3 – 36.7 | **36.7 → 42.0** |
| MMD (RBF) | 0.045 – 0.233 | 0.253 – 0.470 | 0.233 → 0.253 |
| energy distance | 0.026 – 0.157 | 0.190 – 0.384 | 0.157 → 0.190 |

The identity row is the one worth quoting, because it has an external
calibration. **The r\* boundary falls between 36.7% and 42.0% median identity to
the nearest training protein — straddling the ~40% threshold that the enzyme-
function literature independently gives for reliable EC-level-3 transfer**
(Tian & Skolnick 2003; Addou et al. 2009; Todd et al. 2001 put it at 30%). We did
not tune anything to land there. That is the sentence this experiment exists to
produce, and it is the form in which it should reach an abstract.

Two honest limits on that claim:

1. **With a binary outcome, there is no power to rank the predictors.** Taxonomic
   rank is free, sequence identity costs a BLAST job, MMD costs a matrix
   multiply; all three give the same answer here. Nothing in this data says the
   expensive one is worth it. A finer r* grid, or more source groups, would be
   needed to separate them.
2. **The correlation-with-`retained` table has far more resolution**, and it was
   computed over all 210 pairs (see below). It is the continuous version of the
   same question and does discriminate.

### The continuous version, over 210 pairs

Against `retained` = zero-shot ÷ ceiling, on the 210-pair linear-probe scan, with
the group-level permutation null. Abbreviated; the full 24-predictor table is in
`rstar_vs_distance_P*.txt`.

*(These come from the **first** 210-pair scan, `rstar_allpairs_run1_superseded`,
which capped training at 6,000 rather than 4,000. `retained` is unaffected by the
budget-ceiling fix that prompted the rerun, but the ceiling and zero-shot values
themselves shift slightly with the cap, so 47292629 will regenerate this table.
p-values are omitted deliberately: the permutation null for the 14-pair rows was
wrong in the version that produced them — see §11.1 — and is fixed but not yet
re-run.)*

| predictor | ρ | n | note |
|---|---|---|---|
| `pident_p10` | **+0.877** | 14 | 10th-percentile identity — best of all, but only 14 pairs |
| `frac_below_30` | −0.864 | 14 | fraction of queries in the twilight zone |
| `pident_median` | +0.837 | 14 | |
| `proxy_a_dist` | **−0.796** | 210 | best label-free metric despite being saturated |
| `diff_abs` = ‖v−v̄‖/gap | −0.787 | 210 | **AUG6's champion, reproduced to 0.003** |
| `feat_wasserstein` | −0.759 | 210 | |
| `n_shared_lineage` | +0.687 | 210 | free, from the lineage |
| `energy_dist` | −0.656 | 210 | |
| `mmd_rbf` | −0.636 | 210 | |
| `procrustes` | −0.605 | 210 | |
| `|v|/gap` | −0.578 | 210 | |
| `beta_shared` | +0.297 | 210 | **wrong sign, reproducing AUG6** |
| `alpha` | +0.051 | 210 | **no predictive power, reproducing AUG6** |

The bottom three rows are the check that the join is correct: AUG6 measured
`diff_abs` −0.784, `beta_shared` +0.280 and `alpha` +0.022 on this dataset by a
different route, and they come back −0.787, +0.297 and +0.051 here.

**Sequence identity beats every embedding metric**, and the best variant is the
10th percentile rather than the median — i.e. what predicts damage is how bad the
*worst-matched* tenth of your proteins are, not the typical one. But it rests on
14 points from a single source group and must be treated as suggestive until
BLAST is run for all 210 pairs (§11.1).

---

*(filled from 47292629)*

---

## 8. The homology confound is real and large

The brief's §2 flagged that the Pfam × EC grid is 99.4% additive with Pfam alone
explaining 95.6%, and that "an EC probe may largely be reading homology". That is
now measured directly, with no embedding involved at all: build a Pfam → EC
majority-vote lookup on gammaproteobacteria only, and apply it to each target.

| target | n | coverage | accuracy on covered | accuracy overall |
|---|---|---|---|---|
| betaproteobacteria | 14,415 | 0.993 | 0.944 | **0.938** |
| spirochaetes | 2,165 | 0.984 | 0.945 | **0.929** |
| epsilonproteobacteria | 4,247 | 0.982 | 0.945 | **0.928** |
| alphaproteobacteria | 19,286 | 0.983 | 0.931 | **0.915** |
| bacteroidetes | 2,048 | 0.956 | 0.930 | 0.889 |
| firmicutes | 35,850 | 0.930 | 0.915 | 0.851 |
| cyanobacteria | 7,443 | 0.919 | 0.915 | 0.841 |
| actinobacteria | 13,477 | 0.911 | 0.876 | 0.798 |
| crenarchaeota | 2,605 | 0.723 | 0.883 | 0.638 |
| euryarchaeota | 6,998 | 0.726 | 0.844 | 0.613 |
| streptophyta | 13,960 | 0.610 | 0.801 | 0.488 |
| ascomycota | 11,081 | 0.614 | 0.794 | 0.487 |
| insecta | 1,970 | 0.609 | 0.671 | 0.409 |
| vertebrata | 16,256 | 0.479 | 0.754 | **0.361** |

**86.8% of the 1,406 source Pfams map to exactly one EC class at level 3.** A
lookup table with no learning in it gets 0.36–0.94 accuracy across the whole
ladder.

Read the two columns separately, because they say different things:

* **Accuracy on covered proteins barely moves** — 0.94 for a sibling bacterial
  class, 0.75 for vertebrates. Where the Pfam is known, the Pfam→EC map transfers
  across the whole tree of life almost intact.
* **Coverage is what collapses** — 0.99 → 0.48. The taxonomic shift's cost is
  almost entirely *"I have never seen this family"*, not *"this family means
  something different here"*.

That is a specific and useful reframing of what the ladder is measuring, and it
is the number every r* in §5 has to be read against. Two arms address it
directly: EC recovery **within a single Pfam** (homology held constant by
construction, so anything left cannot be a lookup) with a label-shuffle floor
underneath it, and a **BLAST-nearest-neighbour** EC transfer baseline computed
from the same hits used for the identity distances. Both are in 47292610 /
47292605.

This is not an isolated worry. In `ProteInfer`'s own clustered split BLAST beats
the CNN (Fmax 0.950 vs 0.914); in `DeepGOPlus` the DIAMOND term alone beats the
CNN alone on MFO and BPO; in the CARE benchmark BLASTp beats CLEAN at 30–50%
identity and on the promiscuous split by 24 points. EC is the regime where
homology baselines are hardest to beat, so this arm is not optional.

### …but the embedding is *not* only a homology lookup — job 47292610, complete

Holding homology genuinely constant: restrict to one Pfam at a time, so every
protein in the problem is a homolog of every other, and run the identical r*
machinery on the gammaproteobacteria → everything-else split inside it. Three
Pfams had enough EC diversity on both sides to qualify (a fourth, `PF00561`, had
only 248 target proteins and could not afford the 200-protein budget):

| Pfam | K | ceiling | zero-shot | r* | **shuffled ceiling** | **shuffled zero-shot** |
|---|---|---|---|---|---|---|
| `PF00005` ABC transporter | 5 | 0.951 | 0.833 | 0.10 | 0.202 | 0.189 |
| `PF04055` radical SAM | 4 | 0.979 | 0.812 | 0.05 | 0.234 | 0.200 |
| `PF00701` DHDPS | 3 | 0.996 | 0.747 | 0.05 | 0.285 | 0.287 |

**Inside a single family, ESM-C separates EC classes at 0.95–1.00 and transfers
across the taxonomic split at 0.75–0.83 zero-shot, against a label-shuffle floor
of 0.19–0.29.** Homology is constant by construction here, so none of that can be
a Pfam→EC lookup.

So both things are true, and the thesis needs to say both: the *ladder-level*
number in §5 is heavily contaminated by homology — a lookup table gets 0.36–0.94
— but the embedding does carry real within-family functional signal, and the
recovery threshold for it is small (r* = 0.05–0.10). The shuffled rows are the
control that makes this readable; note their own r* values are meaningless, since
90% of a floor-level ceiling is trivially reachable.

Caveat, and it is landmine 3: three Pfams with K = 3–5 is a thin basis. The
qualifying set is small because a Pfam needs several EC classes each with ≥15
proteins on *both* sides of the split. `--min_n 8` would admit more Pfams at the
cost of noisier cells; that run has not been done.

---

## 9. Where this sits in the literature

Worth recording because it changes what is worth claiming.

* **r* has a theoretical parent.** Ben-David, Blitzer, Crammer, Kulesza, Pereira &
  Vaughan, *Machine Learning* 79:151–175 (2010) derive an optimal source/target
  mixing weight α\* with a phase transition at m_T ≥ d/A², where A is the
  H∆H-divergence. r* is the empirical instantiation of that, with a fixed pool
  instead of a loss weight. **The measurement should be positioned as testing a
  2010 theorem in a biological domain, not as a new idea.** The 1/P scaling in §4
  is the concrete prediction.
* **The label-free-distance → *label budget* mapping does appear to be open.**
  Existing transferability metrics (OTDD, LEEP, LogME, H-score, proxy-A) are all
  correlated against an *accuracy drop*, never against how much target labelling
  a shift costs — and none of them has been applied to proteins at all.
* **No mainline EC predictor splits by taxonomy.** DEEPre, ECPred, DeepEC, CLEAN,
  ProteInfer, EnzBert, CARE and EC-Bench all split by sequence identity or by
  time. The taxonomic split is the gap.
* **The exposure to be honest about**: ESM-C is the *best* taxonomic classifier of
  any pLM tested (Hallee et al., bioRxiv 2025.10.07.681002 — ESM-C 600M is top at
  class, order, family, genus and species). We are measuring taxonomic transfer
  with the embedding most saturated in taxonomic signal. Whether r* is a property
  of the biology or of the embedding is not something this design can currently
  separate; repeating one arm with ESM-2 would settle it and has not been done.
* **EC level 3 is the defensible level** and there is a citation for it: roughly
  83% of level-4 errors in ECPICK are correct through level 3, and EC4 encodes
  substrate specificity rather than reaction chemistry.

---

## 10. Status at write-up, and how to resume

All data lives under `/scratch/lmk04992/ec_rstar/`. Everything below was
submitted with `sbatch` and writes incrementally, so a killed session loses
nothing.

| stage | job | state | output |
|---|---|---|---|
| label-free distances, 210 pairs | 47292604 | **complete** | `distances_allpairs.json` |
| BLAST identity + BLAST-NN EC baseline | 47292605 | **complete** | `seq_identity.json`, `seqid/` |
| homology confound (Pfam lookup, within-Pfam, shuffle) | 47292610 | **complete** | `homology/` |
| reduced 14-target ladder, P = 500 | 47292734 | **complete** — this is §5 | `ladder_fast/` |
| r* ladder, four full arms | 47292627 | **running**, 1/14 targets of arm 1 done | `ladder/rstar_summary_*.csv` |
| 210-pair linear-probe r*, rerun | 47292628 | **running** | `rstar_allpairs.json` |
| join + regression | 47292629 | **queued**, `afterany` on the two above | `rstar_vs_distance_P*.json` |

The full ladder was at roughly six minutes per target when this was written, so
arm 1 (`_pair`, 14 targets, four budgets, three seeds) needs about 90 minutes and
all four arms several hours. It writes `rstar_summary_pair.csv` incrementally
after every target, so partial results are readable at any point. **Nothing in §5
depends on it** — §5 is the reduced arm, which completed.

Three arms have not produced any output yet and will only exist once 47292627
gets past `_pair`: `_matched` (class-count-matched label set), `_novelty`
(constructed functional novelty, ν ∈ {0, 0.25, 0.5}) and `_sizematched` (equal
training volume on both sides). Check which `rstar_summary_*.csv` files exist
before relying on them.

Superseded first attempts are kept, not deleted:
`ladder_run1_superseded/`, `rstar_allpairs_run1_superseded.json`. They are the
evidence for §3(d) — the run where 100% of pairs at budget 200 were censored.

The `ec_seqid` job ran before the BLAST-nearest-neighbour baseline was added to
the parse step, so the baseline column was filled by re-running just the parse:

```bash
python src/ec_seq_identity.py parse --outdir /scratch/lmk04992/ec_rstar/seqid \
    --ec /scratch/lmk04992/ec_swissprot/data/ec_annotations.tsv --ec_level 3 \
    --out /scratch/lmk04992/ec_rstar/seq_identity.json
```

**To resume from scratch**, from the repo root on the cluster
(`/work/ah2lab/LiamK/tidythesis`):

```bash
LAD=$(sbatch --parsable slurm/ec_rstar_ladder.slurm)
ALL=$(sbatch --parsable slurm/ec_rstar_allpairs.slurm)
sbatch --parsable --dependency=afterany:$LAD:$ALL slurm/ec_rstar_report.slurm
# independent of the above, and already complete:
#   sbatch slurm/ec_distances.slurm
#   sbatch slurm/ec_seq_identity.slurm
#   sbatch slurm/ec_homology_confound.slurm
```

**To re-run only the join** once the ladder finishes, without re-running anything
expensive:

```bash
python src/ec_rstar_regress.py \
    --rstar_mlp    /scratch/lmk04992/ec_rstar/ladder/rstar_summary_pair.csv \
    --rstar_lr     /scratch/lmk04992/ec_rstar/rstar_allpairs.json \
    --distances    /scratch/lmk04992/ec_rstar/distances_allpairs.json \
    --seq_identity /scratch/lmk04992/ec_rstar/seq_identity.json \
    --geometry     /scratch/lmk04992/ec_swissprot/analysis/beta_diagnosis.json \
    --budget 500 --n_perm 5000 \
    --out /scratch/lmk04992/ec_rstar/rstar_vs_distance_P500.json
```

---

## 11. Caveats

Ordered by how much they could change a conclusion.

1. **Sequence identity exists for 14 pairs, not 210.** BLAST was run out of
   gammaproteobacteria only, so every `pident_*` correlation rests on 14 points
   from 14 groups. The group-level permutation null on 14 single-source rows is
   weak, and the first version of it was *wrong* — permuting both members of the
   pair produced lookup keys that do not exist, silently rejected nearly every
   draw, and returned p ≈ 0 for predictors that had not been tested. It now holds
   the source fixed and permutes targets, reports `nan` when the null cannot be
   built, and uses `(cnt+1)/(n+1)` so p is never exactly zero. **Any
   identity-vs-r\* claim should be treated as suggestive until BLAST is run for
   all 210 pairs.**
2. **Median identity barely separates the far half of the ladder** (§6): archaea,
   plants, fungi, insects and vertebrates all sit at 34–37%. Identity is the most
   quotable axis and the least discriminating one over the range that matters.
   MMD and energy distance span 10× over the same pairs.
3. **The EC probe is substantially reading homology.** A Pfam→EC lookup with no
   learning gets 0.36–0.94 (§7). The within-Pfam arm shows the embedding also
   carries genuine within-family signal, but that arm rests on **three Pfams with
   K = 3–5**, which is exactly the kind of thin row landmine 3 warns about.
4. **ESM-C is the best taxonomic classifier of any pLM tested.** We are measuring
   taxonomic transfer with the embedding most saturated in taxonomic signal, and
   nothing here separates "the biology differs" from "the head is unlearning a
   taxonomic nuisance direction". Repeating one arm with ESM-2 would settle it and
   has not been done.
5. **r\* is a grid value on eight points**, so its resolution is coarse and ties
   are common. Correlations against it are correspondingly blunt; `retained`
   (zero-shot ÷ ceiling) is continuous and is reported alongside for that reason.
6. **Proxy-A distance is saturated and Fréchet is unstable** at 3,000 samples in
   960 dimensions (§6). Neither should be quoted as a calibrated distance.
7. **Class count K varies 30–112 across the per-pair arm.** It is reported per
   row and partialled out in the regression, and the `_matched` arm exists to
   check it, but the matched arm's 20 classes are core-metabolism ECs and easier
   than a random 20.
8. **The functional-novelty axis is constructed, not observed** (§2), so it
   measures what happens when classes are withheld — not what happens in a real
   metagenome. Swiss-Prot simply does not contain the natural version of this
   experiment at EC level 3.
9. **Multiple comparisons**: 24 predictors are tested against each response, so
   about 1.2 false positives at p < 0.05 are expected per table. Stated in the
   output of every table.
10. **The `_matched`, `_novelty` and `_sizematched` arms run after `_pair`** in
    the same job. If the job is killed part-way the earlier arms are still on
    disk, but the later ones may be missing entirely — check which
    `rstar_summary_*.csv` files exist before drawing on them.
