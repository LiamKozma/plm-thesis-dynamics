# How to see the negative-transfer dip (and the recovery threshold)

This note explains **why your synthetic pipeline never produced the
characteristic "performance dips before it recovers" curve**, and exactly what
to change to make it appear. It then describes the real-data test we are
standing up (precomputed ESM-2 embeddings of real protein sequences) and the
sweep that measures the *recovery threshold* — the fraction of out-of-distribution
(OOD) target data required to overcome negative transfer and trigger geometric
realignment to the target manifold.

---

## 1. Why the synthetic setup can't dip

The dip is **negative transfer**: a source-pretrained model, when first exposed
to target data, gets *worse* before it gets better. For that to happen, the
source-optimal decision boundary has to be actively *wrong* on the target
region — wrong enough that the first gradient steps degrade test accuracy.

Your synthetic generator (`src/generate_simulation.py`) makes this structurally
almost impossible, for two compounding reasons:

1. **A single global labeler.** Labels come from one frozen `RandomOracleNN`
   that maps `1280-D → n_classes`. The *same* oracle labels both source and
   target. So `P(y | x)` is **globally identical** everywhere in space — this is
   *pure covariate shift* with a perfectly consistent labeling function.

2. **Source and target share one GMM universe.** `generate_dispersion_gmm`
   builds the family centroids once, then the only thing `shift` changes is the
   sampling *dispersion*:
   `sigma_source = base_sigma / max(1, shift)` (tight) vs
   `sigma_target = base_sigma` (broad). Source is just a **lower-variance subset
   of the same manifold** the target is drawn from. (See
   `generate_simulation.py:164` and `:174`.)

Put those together: the target is a *superset region* of the source, labeled by
the *same smooth function*. A source-trained MLP is already approximately
correct on the target — there is no boundary that is right for source and wrong
for target. So adaptation only ever *adds* information. **Monotonic
improvement, no dip.** You observed exactly the behavior the construction
guarantees.

> Increasing `shift` makes the source *tighter*, not *displaced*. You can crank
> it arbitrarily and still never induce negative transfer, because tightening a
> distribution around the same centroids under the same labeler does not create
> a conflicting optimum.

---

## 2. What actually produces a dip

Negative transfer needs **at least one** of these, and real biological data
tends to supply all three:

| Ingredient | Synthetic (current) | What creates the dip |
|---|---|---|
| Label function `P(y\|x)` | one global oracle, identical S/T | locally **inconsistent** between source and target populations |
| Covariate support `P(x)` | target ⊇ source (nested) | source and target occupy **different, partially disjoint** regions |
| Manifold geometry | isotropic GMM blobs | real ESM-2 embeddings: anisotropic, clustered, low effective rank |

The cleanest way to get all three with real data is a **shared label space
across two genuinely different biological populations**:

- **Labels `y` = protein family (Pfam).** The label *set* is shared between
  source and target — that's what makes "transfer" meaningful.
- **Shift axis = a biological grouping.** Source and target are the *same
  families* but drawn from **different taxa** (UniProt: e.g. Bacteria → Archaea)
  or **different biomes** (MGnify: e.g. human gut → marine). Because the
  same family folds/decorates differently across taxa or environments, the
  source-optimal boundary is genuinely misplaced for the target → the model
  must *unlearn* before it realigns → **dip, then recovery**.

This is the design we're implementing.

---

## 3. The recovery-threshold experiment

"Recovery threshold" = the **fraction of OOD target data in the adaptation
pool** at which the curve stops being monotonic-up and starts showing the
dip-then-recover U-shape (and, at higher fractions, full realignment).

We operationalize the shift as a single knob, `--target_ood_frac` (call it
`r ∈ [0, 1]`), in `src/precompute_real_embeddings.py`:

- **Source train set:** 100% source-group sequences (one taxon/biome).
- **Adaptation pool:** a mixture — fraction `r` from the **target** group, the
  rest from the source group.
- **Test set:** 100% **target** group — the manifold we are trying to realign to.

Then sweep `r` and read each run's adaptation curve from
`adapt.py`'s `*_batch_log.csv` (`test_f1` / `test_ce` vs `samples_seen`):

```
r = 0.0   →  no shift in the pool        → monotonic, no dip (sanity check / matches old result)
r = 0.1   →  ...
r = 0.25  →  expect the dip to emerge here-ish
r = 0.5   →  deep dip, then recovery
r = 1.0   →  pure-target pool, fastest realignment
```

The **threshold** is the smallest `r` at which `min(test_f1)` over the
adaptation trajectory drops below the `batch 0` (pre-adaptation) `test_f1` by
more than noise — i.e. the first `r` that shows real negative transfer. Plot
`test_f1` vs `samples_seen` as one line per `r`; the dip's depth and the
samples-to-recover are your headline numbers.

### Wiring it into the existing sweep

`shifts` in the YAML is already your per-run shift axis. For the real-data test,
each `shift` value is reinterpreted as a `target_ood_frac`, and
`precompute_real_embeddings.py` emits one `source/pool/test` `.npy` set per
value (named with `Shf` like the synthetic pipeline). `main_precomputed.nf`
then runs `TRAIN_SOURCE` + `TEST_ADAPTATION` on the precomputed files **without
calling the data generator** — see `README` section below and the run steps in
your chat.

---

## 4. Concrete change list (synthetic → dip-capable)

1. **Stop labeling with the global oracle.** Use real family labels so
   `P(y|x)` can differ across populations. *(Done by the real-data path.)*
2. **Displace the source instead of tightening it.** Source and target must be
   different *groups* (taxa/biomes), not the same blobs at different variance.
   *(Done: group-based split, not sigma-based.)*
3. **Make the pool a source/target *mixture* controlled by `target_ood_frac`,**
   and keep the test set pure-target. *(Done: `--target_ood_frac`.)*
4. **Sweep the mixture fraction**, not the GMM sigma, to find the threshold.
   *(Set `shifts:` in the config to the `r` grid.)*
5. **Keep everything else identical** — same `model.py`, same `train.py` /
   `adapt.py`, same 1280-D width — so any dip you now see is attributable to the
   data, not the machinery.

If, after all this, you *still* see no dip, that itself is a real result: it
would mean the ESM-2 family manifold is consistent enough across your chosen
groups that a source-trained classifier transfers without conflict — and you'd
then increase the *biological distance* of the shift (more distant taxa, more
dissimilar biomes, or more divergent families) rather than touch the model.
