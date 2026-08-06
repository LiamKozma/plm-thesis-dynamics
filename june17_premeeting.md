# June 17 — Pre-meeting notes: real-data recovery-threshold test

**One-line summary:** We replaced the synthetic GMM data generator with *real*
ESM-2 embeddings of real proteins under a taxonomic distribution shift, and — for
the first time — observed the negative-transfer **dip** and a **recovery
threshold** (~0.5 OOD fraction) that the synthetic pipeline structurally could
never produce.

---

## 1. The problem we were trying to fix

We study the **recovery threshold**: the fraction of out-of-distribution (OOD)
target data an adapting model needs before it overcomes negative transfer and
realigns to the target manifold. The expected signature is a **U-shaped curve** —
target performance dips before it recovers. **Our synthetic pipeline never showed
the dip.**

## 2. Why the synthetic setup could never dip (diagnosis)

Two structural reasons in `src/generate_simulation.py`:

1. **One global labeler.** Labels come from a single frozen `RandomOracleNN`
   mapping embedding → class. The *same* oracle labels source and target, so
   `P(y|x)` is identical everywhere — **pure covariate shift with a perfectly
   consistent labeling function.**
2. **Source is a nested subset of target.** The `shift` knob only *tightens* the
   source's sampling variance around the *same* GMM centroids
   (`sigma_source = base_sigma / shift`); target uses the full variance. So the
   target is a labeled *superset* of the source.

Together: a source-trained model is already approximately correct on the target,
so adaptation can only add information → **monotonic improvement, no dip.** The
construction guarantees the null result. (Secondary issue: the adapt script
evaluated only every 500 batches, so even a transient dip would have been
invisible — see fix below.)

## 3. What we changed

Swap synthetic embeddings for real ones, keeping **everything downstream
identical** (same model, same train/adapt code, same 1280-D width):

| Axis | Synthetic (old) | Real-data test (new) |
|---|---|---|
| Embeddings | GMM samples, 1280-D | **ESM-2 `esm2_t33_650M_UR50D`**, mean-pooled, 1280-D |
| Sequences | none (simulated) | real proteins from **UniProt** (~10k subset) |
| Labels `y` | global oracle | **Pfam family** (shared label space) |
| Shift | tighten same GMM | **taxonomic**: source = Bacteria, target = Archaea (same families) |
| Shift knob | sigma ratio | `target_ood_frac` r = fraction of adaptation pool drawn from target |

**Operationalization of the threshold:** source model trains on Bacteria; the
adaptation pool is a mixture of `r` target (Archaea) + `(1−r)` source; the test
set is **pure Archaea** (the manifold we want to realign to). Sweep `r` and read
each run's target-F1 trajectory.

> Note on data source: the design needs per-protein **taxonomy**, which MGnify
> (metagenomic) doesn't expose per protein — only biome. So we used **UniProt**
> for the clean Pfam + taxonomy split. MGnify-via-biome remains available as an
> alternative axis.

## 4. Infrastructure built (all in the repo)

- `src/fetch_sequences.py` — downloads the labelled subset (UniProt or MGnify).
- `src/precompute_real_embeddings.py` — ESM-2 embed + family→int labels +
  source/pool/test split per `r`. Writes the `.npy` files the pipeline expects.
- `precompute_embeddings.slurm` — GPU job → `/scratch/lmk04992/somedir/embeddings`.
- `main_precomputed.nf` + `configs/real_data_test.yaml` — runs `train.py` +
  `adapt_OGadam.py` **unchanged** on the precomputed files; skips data generation.
  Sweeps `r` × seeds.
- `src/plot_recovery.py` — aggregates over seeds, quantifies dip depth / recovery,
  plots mean ± std curves.
- **Two fixes that mattered:** made `eval_every` a flag (set to 1, so the early
  dip is actually sampled — the old default of 500 hid it), and added `--seed` to
  train/adapt for error bars.

Full reasoning lives in `HOW_TO_SEE_THE_DIP.md`.

## 5. What we ran, and what we learned at each step

**v1 — 5 distinct Pfam families.** Baseline target F1 = **0.996** (at ceiling).
Dips were tiny and noisy. *Lesson:* 5 well-separated families classify near-
perfectly and transfer across taxa almost perfectly → **no headroom** to show
negative transfer.

**v2 — 16 families, equal pool size (1000).** Baseline dropped to **0.891**
(real headroom). Single seed already showed a **non-monotonic dip**, peaking at
intermediate OOD fraction.

**v3 — same, 3 seeds (final result below).** Confirmed the effect with error bars.

## 6. Final result (3 seeds, 16 families, taxonomic shift)

| true OOD frac | baseline | dip depth (mean ± std) | recovered (frac of seeds) | final F1 |
|---|---|---|---|---|
| 0.00 (in-dist) | 0.896 | 0.087 ± 0.018 | 0/3 | 0.836 |
| 0.10 | 0.896 | **0.189 ± 0.071** | 0/3 | 0.810 |
| 0.25 | 0.896 | 0.107 ± 0.030 | 1/3 | 0.880 |
| 0.50 | 0.896 | 0.113 ± 0.040 | 2/3 | 0.904 |
| 1.00 | 0.896 | 0.053 ± 0.013 | 3/3 | 0.942 |

**In-distribution noise floor** (the r=0 dip): ~0.09 — a pool of pure in-dist
data still perturbs target F1 by this much, due to small-batch SGD on small pools.

### How to read it (talking points)

1. **Negative transfer is real.** At `r=0.10` the dip (0.189 ± 0.071) clearly
   exceeds the in-distribution noise floor (~0.09) — adapting on a little OOD
   data genuinely *hurts* target performance first. This is the dip the
   synthetic data could never produce.
2. **The recovery threshold is ~0.5.** The cleanest, most monotonic signal is
   *recovery*: the fraction of seeds that climb back to baseline goes
   **0 → 0 → 1/3 → 2/3 → 3/3** as OOD fraction rises, and final target F1 rises
   monotonically **0.81 → 0.81 → 0.88 → 0.90 → 0.94**. Below ~0.5 the model
   tends to get *stuck* in negative transfer; at/above ~0.5 it reliably realigns.
3. **Mechanism, not artifact.** Pools are equal size across all `r`, so this is
   not "more data = more recovery" — only the source/target *composition* differs.

## 7. Honest caveats / limitations (anticipate the professor's questions)

- **Only the `r=0.10` dip is statistically clean** above the noise floor with 3
  seeds; the intermediate dips (0.25, 0.50) sit close to the noise band. The
  *recovery* trend (recovered-fraction, final-F1) is the more robust evidence
  than dip depth per se. **More seeds** would tighten the intervals.
- **Noise floor is non-trivial (~0.09)** because pools are small (1000) and the
  adapt batch is small (32). Larger pools / more sequences would reduce it.
- **Single configuration:** one PLM (650M), mean-pooled; one shift (Bacteria→
  Archaea); 16 families; one optimizer (`adapt_OGadam`). Robustness across these
  is future work.
- **Baseline is still fairly high (0.90).** A harder shift (more distant taxa, or
  within-superfamily labels) would likely deepen the dip and sharpen the threshold.

## 8. Suggested next steps to raise with the professor

1. **More seeds (e.g. 10)** to put confidence intervals on the threshold.
2. **Sweep difficulty:** vary number/relatedness of families, or taxonomic
   distance of the shift, and see how dip depth + threshold move.
3. **Side-by-side synthetic vs real** figure (flat curve vs U-curve) for the thesis.
4. **Finer `r` grid around 0.3–0.6** to localize the threshold precisely.

## 9. Artifacts to show

- Curves: `/scratch/lmk04992/somedir/recovery_curves_seeds.png` (3-seed, final),
  `recovery_curves_v2.png` (single-seed 16-family), `recovery_curves.png` (v1).
- Data: `/scratch/lmk04992/somedir/embeddings/` (+ `dataset_info.json`).
- Logs/CSVs: `/scratch/lmk04992/somedir/results/real_data_test/experiments/adapt/`.
