# Research brief: how much target-domain labelling does an adapting classifier need?

You are being asked to propose experiments. Everything below is a measurement that
has already been made, stated in domain-neutral terms. **Do not ask what the items
are.** The substantive content is entirely in the geometry, the estimator and the
statistics, and the domain adds nothing you would need. Treat every dataset as an
abstract labelled point cloud.

---

## 1. The setting

A **frozen encoder** `φ` maps items to vectors in **R^960**. It was pretrained on an
unsupervised objective and has never seen any of the labels below. It is not
fine-tuned in any experiment here; every method operates on its outputs.

There are **N ≈ 231,000 items**, partitioned into **15 environments** `E₁ … E₁₅`
with sizes from **1,970 to 64,791**. Environments are natural populations, not
random splits; they differ in distribution in ways nobody chose.

Every item carries **two categorical labels**:

| | symbol | cardinality | role |
|---|---|---|---|
| coarse label | `G` | 3,282 values | a grouping that is *not* the prediction target |
| fine label | `Y` | 270 values | **the prediction target** |

`Y` is what we want to predict from `φ(x)`. `G` is available at training time but is
not the goal, and it is the central nuisance: see §3.

A **cheap pairwise similarity** `s(x, x′) ∈ [0,100]` also exists. It is computed
directly from the raw items, not from `φ`, is expensive to compute at scale but
tractable, and is the standard non-learned tool in this area. It supports a
retrieval baseline: label a query by the `Y` of its most similar training item.

## 2. The quantity of interest

Fix a **labelling budget** `P` — the total number of labels you can afford. Build an
adaptation pool of exactly `P` items, of which a fraction `r` come from the target
environment and `(1 − r)` from the source. Total labels are held constant; only
composition changes.

- **zero-shot**: source-trained model scored on target test data, `r = 0`
- **ceiling**: a model trained from scratch on target data only
- **retention**: `zero-shot / ceiling`
- **r\***: the smallest `r` on the grid `{0, .05, .1, .2, .3, .5, .75, 1}` at which the
  adapted model reaches `0.9 × ceiling`

The classifier is a small MLP (960 → 512 → 256 → K), Adam, warm-started from the
source model, trained to convergence with early stopping on a held-out slice of the
pool. Scores are macro-F1. A logistic-regression probe is used where breadth matters
more than comparability; the two are never pooled.

**`r*` is the headline quantity and it is fragile.** It is a threshold on a curve,
read off an 8-point grid, scored against a bar that itself depends on `P`. Retention
turned out far more stable and is now reported alongside it. Several results below
exist because `r*` misbehaved.

## 3. What is established, with numbers

### 3.1 The two label sets are badly entangled

- The `G × Y` centroid grid is **99.4% additive**, and **`G` alone explains 95.6%**
  of the variance in the `Y`-centroid positions. Knowing the coarse label almost
  determines where an item sits.
- **This is the central threat: a classifier for `Y` may largely be reading `G`.**
- Held constant by construction — restricting to a single `G` value, so every item in
  the problem shares the coarse label — the encoder still separates `Y` at 0.81–0.97
  and transfers across an environment shift at **0.49–0.83 zero-shot**, against a
  label-shuffle floor of **0.16–0.29**, on 6 such groups. So real fine-label signal
  exists that no coarse-label lookup could supply.
- But at the whole-dataset level, the **retrieval baseline on `s` beats the encoder
  MLP on 12 of 14 target environments** (mean 0.798 against 0.723), matched on
  items, label set, scoring, and source data volume. The encoder's advantage is
  local, not global.

### 3.2 The encoder is extremely anisotropic, and this contaminates everything

- Nominal dimension 960, **effective rank 3–11**.
- Define two subspaces from the top principal directions of each label's centroid
  cloud, at 90% of between-class variance: `B` (for `Y`) has **26 dimensions**, `H`
  (for `G`) has **50**. Mean cosine of principal angles between them: **0.928**.
- A random direction drawn from the *data covariance* already lands **~72% inside
  `B`**. Naive "is the shift inside the signal subspace" statistics are therefore
  measuring typicality, not label structure. This produced one wrong published-in-
  the-writeup conclusion before it was caught.

### 3.3 A controlled displacement experiment, and its negative result

Take one environment, split it into two disjoint halves stratified by `Y`, and
displace one half by a fixed vector drawn from the data covariance and projected
into a chosen subspace. Because the displacement is a **rigid translation**, it is an
isometry, so the target-only problem is unchanged — verified: **the ceiling holds
between 0.9752 and 0.9804 across all 43 conditions** while zero-shot falls to 0.211.

Two normalisations were run and they **disagree**:

- matched **Euclidean** length → `H \ B` looks most damaging
- matched **within-class σ** → `H` looks most damaging, and `B \ H` becomes harmless

Pooling all conditions and regressing ranked retention:

| model | R² (raw) | R² (whitened) |
|---|---|---|
| within-class σ of the step alone | 0.509 | 0.794 |
| Euclidean length alone | 0.764 | 0.775 |
| **both magnitude measures** | **0.851** | **0.854** |
| subspace identity alone | 0.031 | 0.113 |
| both + subspace identity | 0.908 | 0.919 |

**Subspace identity adds +0.058.** The mechanism is a single column — within-class σ
traversed per unit of centroid gap:

| direction | σ per gap (raw) | (whitened) |
|---|---|---|
| `B` | **3.4** | 47.4 |
| `H` | **3.7** | 30.0 |
| `H \ B` | 14.7 | 28.3 |
| random, matched dim | 18–20 | 26–27 |
| outside both | 37.0 | 25.6 |
| `B \ H` | **43.6** | 26.8 |

So the label subspaces are simply the **wide** directions. In whitened coordinates
every distinction vanishes: maximum damage across all whitened conditions is
retention 0.980. **"Shifting along the coarse-label subspace breaks fine-label
prediction" is a statement about anisotropy, restated.**

### 3.4 …but the observational version points elsewhere

Decomposing the 210 *naturally occurring* ordered environment-pair shifts instead of
injecting one, the dominant predictor of damage is not the shared translation at all:

| predictor | ρ vs retention | partial, given magnitude |
|---|---|---|
| **class-specific (differential) motion / gap** | **−0.819** | **−0.761** |
| shared component outside both subspaces | −0.577 | −0.610 |
| shared displacement magnitude | −0.472 | — |

**Differential motion — the class-conditional part of the shift — is what predicts
damage, and it is exactly the component a rigid translation cannot inject**, because
a class-specific displacement is not an isometry and moves the ceiling. This is the
sharpest open tension in the project.

### 3.5 Retention has a two-regime structure, with a null band

- Splitting one environment against *itself* (no shift), retention reads **0.902 to
  1.039** over 48 partitions. That is the null band.
- Across real pairs, 8 target environments retain **0.707–1.001** and 6 retain
  **0.221–0.537**, with nothing in between. The separation holds in all 9 estimator
  variants tried, gap 0.153–0.238. Two of the 8 sit *inside* the null band.
- Replicating with a **second, independent pretrained encoder** (different
  architecture, different dimension): the separation reproduces with a *wider* gap,
  and the environment ordering agrees at **Spearman 0.960** (14 targets) and
  **0.914** (209 pairs).

### 3.6 Label-free adaptation splits the environments a *different* way

Using **no target labels at all** — fit a transform or recalibrate statistics on
unlabelled target data, then re-score:

| environment class | best label-free gain |
|---|---|
| 6 "near" environments | +0.006 to +0.095 |
| 2 "mid" environments (retain badly, 0.30–0.35) | **+0.104 and +0.236** |
| 3 "far" environments (retain badly, 0.16–0.27) | **−0.016, −0.020, −0.040** |

Methods: covariance alignment (CORAL), batch-norm recalibration, entropy
minimisation (TENT) all behave similarly; **optimal-transport alignment is
catastrophic everywhere** (drops to 0.02–0.40).

**The environments that retain badly and the environments that are cheaply fixable
are not the same set.** Two failure modes exist and only one has a free remedy.
Nothing yet explains the difference. *This is currently the most promising lead.*

(11 environments rather than the 14 of §3.5: three were too small to supply
disjoint unlabelled and test sets at the sizes this protocol needs. The label-free
ceiling here is also trained on only 1,000 target items, so on the easiest
environments the source model beats it outright — the absolute gains above are
sound, any "fraction of the achievable gap recovered" figure is not.)

### 3.7 The controlled simulator

A generator produces synthetic point clouds where the shift is set by hand. Latent
space splits into a **signal subspace** carrying class identity and a **nuisance
subspace** carrying most of the variance under a power-law spectrum. Knobs:

- `d` — displacement magnitude
- `α` — fraction of squared displacement that is a **shared** translation; `α = 1` is
  pure covariate shift, `α = 0` is pure class-specific (concept) shift
- within-class spread inflation — the only knob permitted to move the ceiling

Results, after repairing two estimator defects (single-pass adaptation; a ceiling
trained on 4× the pool's data):

- **`r*` depends far more on the *type* of shift than its size.** At fixed `d = 0.5`:
  `r* = 0.05` at `α = 1` against **`r* = 0.75` at `α = 0`** — a factor of 15.
- `r*(d)` rises and then **plateaus at 0.75–1.0**; it does not diverge. The earlier
  "never recovers" region was an artifact of the unrepaired estimator.

**The simulator models only one label level.** It has one centroid per coarse class.
It cannot currently represent the `G`/`Y` two-level structure that §3.1 says is the
whole problem. Making it do so is the main open construction task, and the geometry
to target is measured: `dim B = 26` nested in `dim H = 50`, principal-angle cosine
0.928, `Y`-centroid gap ≈ 1.11 within-`Y` σ, the 99.4%-additive grid, and a
heavy-tailed `Y` distribution within each `G` (empirically: 69% purity with 2.98
distinct `Y` values per `G` on average).

## 4. Methodological constraints that have already cost us

Proposals that violate these will be rejected, because each of these produced a wrong
conclusion at least once:

1. **Match your control.** Comparing an in-subspace direction drawn *within* the
   subspace against an out-of-subspace direction drawn *isotropically* compares
   typicality, not subspace membership. Draw both from the data covariance.
2. **Match class counts** before comparing anything across datasets.
3. **Permute groups, not pairs.** The 210 ordered pairs come from 15 environments
   and are not independent. Every p-value must come from a group-level permutation.
4. **Report `n` and expected false positives.** A 25-predictor battery expects ~1.2
   hits at p < 0.05. Rows with fewer than ~20 units are not evidence.
5. **Never let the ceiling move with the treatment.** `r*` is scored against a bar;
   a bar that moves with the knob makes `r*` uninterpretable. Only isometries are
   safe: translation and rotation preserve the target problem, independent random
   per-class displacement inflates gaps as `√(g² + 2m²)`, and interpolation toward a
   permutation collapses them.
6. **A fixed Euclidean step is not a fixed intervention** in an anisotropic space.
   Report any displacement in both metrics or the result restates the metric.

## 5. Compute and cost

- The full simulator sweep runs in **11 minutes**. Simulator experiments are
  effectively free; propose them liberally.
- Encoder outputs for all items are precomputed and cached. Anything that reuses them
  is minutes to a couple of hours on 16 CPUs.
- Re-encoding a fresh corpus needs a GPU: ~20 min per 20,000 items.
- The pairwise-similarity computation over all environment pairs takes ~1.5 hours.

## 6. What I want from you

Propose **experiments**, not conclusions. For each proposal give:

1. **The question**, in one sentence, phrased so it can come back negative.
2. **The measurement** — what is computed, on which data, with which estimator.
3. **What would undermine the current account**, explicitly, not only what would
   support it. Proposals that can only confirm are not useful.
4. **Cost**, using §5.
5. **Which of §4's constraints it is most at risk from**, and how it is controlled.

Prioritise, in roughly this order:

- **(a) The two failure modes in §3.6.** Why are some badly-retaining environments
  cheaply fixable with no labels and others not? A diagnostic computable *before*
  labelling anything would be the most useful single result available, because it
  tells you which of two remedies to reach for.
- **(b) The differential-motion tension in §3.4.** Class-specific motion predicts
  damage but cannot be injected without moving the ceiling. Is there a construction
  that varies it while provably preserving the target problem? If not, what is the
  best observational design?
- **(c) A two-level simulator (§3.7).** What is the right generative construction for
  a fine label nested inside a coarse one, hitting the measured targets? What would
  falsify it as a model of the real data?
- **(d) Estimator design.** Is there a better-posed quantity than `r*` — one that is
  not a threshold on a grid scored against a moving bar — that answers the same
  practical question ("how much target labelling do I need")?
- **(e) Anything in §3 you think is wrong.** Several of these results overturned
  earlier ones. Say which you would bet against and what measurement would settle it.

Where a proposal has a known precedent in the domain-adaptation, transfer-learning,
concept-erasure, or distribution-shift literature, say so and cite it — a result that
is already established elsewhere should be reframed as confirmation rather than
discovery. Be specific about what is genuinely novel here and what is not.
