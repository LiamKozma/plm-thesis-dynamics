# Identifying the Recovery Threshold for Protein Language Models under Data Distribution Shift

> M.S. Statistics thesis, University of Georgia. How much out-of-distribution data must an
> adapting classifier see before it overcomes negative transfer — and how much of that number
> is a property of the data rather than of the estimator that measured it.

A model trained on one region of protein space and asked to predict on another loses accuracy.
Some of that loss comes back if you fine-tune on labelled target data, and the question here is
how much of that data it takes. Call the answer `r*`: the smallest target fraction of a fixed
labelling budget at which an adapted model reaches 90% of the ceiling a target-trained model
reaches.

Most of the work in this repository turned out to be about making `r*` mean something. Two of
the three synthetic generators here produce a number that looks like a recovery threshold and
is not one, and the estimator that survived simulation broke six further ways on real
embeddings. Both failures are documented below with the measurements that exposed them,
because they are the part of this project I would most want a reader to check.

## What it found

**Synthetic arm.** `r*` depends on the *type* of shift, not its size. At one shift distance, a
pure covariate shift — families translating together along a shared direction — recovers on 30%
out-of-distribution adaptation data, while a pure concept shift of equal magnitude needs 75%.
Distance alone does not determine cost.

**Real arm.** Over 231,285 Swiss-Prot proteins in ESM-C embeddings, `r*` itself is fragile: it
is a threshold on a curve, scored against a budget-dependent bar, read off an eight-point grid,
and it tells three different stories at three annotation budgets. The stable quantity is
zero-shot retention. Eight bacterial targets keep 0.707–1.001 of their ceiling; six archaeal and
eukaryotic targets keep 0.221–0.537. Nothing falls in the gap. That boundary sits between 36.7%
and 42.0% median sequence identity to the nearest training protein, straddling the ~40% line the
enzyme-function literature gives for reliable EC-level-3 annotation transfer. Nothing was tuned
to land there.

**The lab notes are the real narrative**, in date order:
[`JULY22`](docs/notes/JULY22.md) rebuilds the generator ·
[`AUG4`](docs/notes/AUG4.md) adds the EC (function) label axis ·
[`AUG6`](docs/notes/AUG6.md) re-runs everything on a 231k-protein dataset and overturns two
earlier conclusions ·
[`AUG7`](docs/notes/AUG7.md) measures `r*` on real EC labels and enumerates six ways the
estimator lied.

---

## Table of Contents

1. [The three generators, and why there are three](#the-three-generators-and-why-there-are-three)
2. [The real-data arm](#the-real-data-arm)
3. [Repository structure](#repository-structure)
4. [Environment setup](#environment-setup)
5. [Running the current work](#running-the-current-work)
6. [The legacy Nextflow pipeline](#the-legacy-nextflow-pipeline)
7. [Citation](#citation)

---

## The three generators, and why there are three

`src/` contains three generations of synthetic data generator. Only the third is current, and
the first two are kept because the notes cite them and because what went wrong in each is the
argument for the design of the next.

### v0 — frozen oracle over a GMM (`src/generate_simulation.py`)

Embeddings are drawn from a Gaussian mixture; labels come from a `RandomOracleNN`, a
randomly-initialised network that is never trained, applied by `argmax` over its logits. Shift
is injected by tightening the source dispersion, `sigma_source = base_sigma / max(1, shift)`,
while the target keeps the full spread.

**Why it cannot show negative transfer.** One global oracle labels both source and target, so
`P(y|x)` is identical everywhere — pure covariate shift with a perfectly consistent labelling
function. The source is a lower-variance subset of the same manifold, so no decision boundary is
right for the source and wrong for the target. Adaptation can only add information, and target
performance rises monotonically. Turning the shift knob up makes the source *tighter*, never
displaced, so no setting of it produces a dip. This is structural, not a tuning problem.

### v1 — derangement interpolation (`src/generate_synthetic_precomputed.py`)

Labels become the family index directly, and target centroids interpolate toward a derangement
of the source centroids, `M_f(d) = (1-d)·C_f + d·C_π(f)`. This does produce a dip. It also
produces an artifact that invalidates the number the dip was measured to obtain.

**The U-shaped ceiling.** The interpolation has no variance correction, so the centroid
configuration contracts by `sqrt((1-d)² + d²)`. At `d=1` the centroid set is the source set
permuted, which is isometric to it, so the ceiling returns to its starting value; at `d≈0.5`
every family sits at a midpoint, classes overlap, and the ceiling craters. Bayes accuracy
computed from the centroid geometry alone predicts the measured ceilings at `r = 0.993` (σ=4)
and `0.997` (σ=6), which is how I know the U is geometry rather than training noise. Because
`r*` was scored against `0.9 × ceiling(d)`, the bar dipped exactly where the ceiling dipped:
shift distance and target task difficulty were confounded, and `r*(d)` could not separate them.

**A second problem: the realism was fake.** Real ESM and ESM-C embeddings have a mean
inter-family gap of roughly one within-family sigma and still classify at macro-F1 0.90. An
isotropic mixture at that separation ratio caps at 0.18 accuracy. v1 reached 0.93 only by
inflating separation about sixfold, which papered over the missing anisotropy rather than
reproducing it.

### v2 — calibrated shift geometry (`src/generate_synthetic_v2.py`)

The latent space splits into a **signal subspace** where family identity lives and a
**nuisance subspace** carrying most of the variance under a power-law spectrum and no label
information. That one change reproduces "gap ≈ sigma yet F1 ≈ 0.90" and the low effective rank
of real embeddings at the same time.

Shift decomposes into four knobs, each anchored to something measurable on real data:

| Knob | Meaning | Anchor |
|---|---|---|
| `d` | displacement magnitude, in units of the real bacteria→plants rung | `d_unit_gaps = 1.20` mean gaps, measured |
| `alpha` | fraction of squared displacement that is a shared direction, so `E[cos(shift_i, shift_j)] = alpha` | the +0.41 to +0.71 measured across the real ladder |
| `beta` | fraction of the *shared* translation lying in the signal subspace | archaea and plants shift by nearly the same distance (1.16 vs 1.20 gaps) yet differ in zero-shot F1 (0.85 vs 0.61), so distance cannot be the whole story |
| `target_sigma_inflate` | the only knob that lowers the ceiling | set to reproduce the real ladder's mild decline |

`alpha = 1` is pure covariate shift, `alpha = 0` pure concept shift.

#### The ceiling-invariance rule, and three ways I got it wrong

The rule that makes `r*(d)` interpretable: **any family-specific displacement must be an exact
isometry of the centroid configuration.** If it is not, target task difficulty moves with the
distance knob and the bar moves again. The shared component is a rigid translation, which
preserves every pairwise centroid distance exactly. The family-specific component rotates the
centroid configuration inside the signal subspace, `M = C @ R(θ)ᵀ`, with `θ` solved by bisection
to hit the requested mean displacement. A rotation is an isometry at every angle, so families
move relative to the source-trained boundary with no drift in difficulty.

Three designs failed before that one, and each failed in a way worth recording:

| Attempt | Mechanism | Measured failure |
|---|---|---|
| v1 | interpolate toward a derangement of the same centroids | configuration contracts; ceiling craters at `d≈0.5` and returns at `d=1`; 21-point swing |
| v2a | displace each family by an independent random vector | in high dimensions random vectors are near-orthogonal, so gaps *inflate*, `g → sqrt(g² + 2m²)` (+56% at `d=1`); ceiling pinned at 1.00 for every `d>0` |
| v2b | interpolate toward a fresh centroid draw with sqrt weights | preserves the distribution of configurations but not the realisation; with 16 families the fresh draw was more separable and the ceiling drifted 0.90 → 0.95 |
| **v2c (kept)** | rotate the configuration in the signal subspace | mean gap 10.15 and min gap 3.53 hold to all printed digits across every `alpha` and `d` |

A related trap, also measured: `beta` must apply to the shared translation only. Centroids have
no nuisance component to begin with, so displacing each family by a *different* nuisance vector
makes the nuisance subspace label-informative and sends the ceiling to 1.00.

**On the phrase "flat ceiling."** The isometry argument fixes the geometry, but the shipped
default `target_sigma_inflate = 0.15` lowers the ceiling with distance on purpose, so the
measured ceiling declines monotonically (about 0.850 to 0.707 over `d = 0 → 2`) rather than
staying flat. The point is that it now moves under one deliberate knob instead of swinging as a
side effect of the distance axis. Some source comments still say "flat"; read them as
"invariant under the shift geometry."

#### Calibration

`src/calibrate_v2.py` scores candidate geometries against a five-statistic target box measured
off real embeddings by `src/measure_real_geometry.py`:

```
mean_gap / within_sigma     1.0  – 1.3
min_gap  / within_sigma     0.28 – 0.38
effective rank              3    – 11
per-family sigma spread     1.7  – 2.1
in-domain macro-F1 at d=0   0.90 – 0.94
```

The search is an exhaustive 36-cell grid over `sigma_signal × nuisance_ratio ×
spectrum_exponent`, scored by relative distance outside the box.

> **Known inconsistency, not yet resolved.** `slurm/run_distance_sweep_v2.slurm` pins
> `sigma_signal 1.5, nuisance_ratio 3.0, spectrum_exponent 2.0`, which is *not* a point in the
> grid above; the generator's own argparse defaults are `1.15 / 3.3 / 1.6`, which are.
> `slurm/structure_sweep.slurm` notes the same conflict. Any run of `run_distance_sweep_v2.py`
> that does not pin these explicitly uses a different universe from the published sweep. Pick
> one setting and re-label or re-run before the manuscript.

#### What the rebuilt generator showed

`r*` rises with distance and depends on the type of shift, across `d = 0 → 1`:

| `alpha` | `r*` across `d = 0 → 1` |
|---|---|
| 1.0 (pure covariate) | 0, 0, 0.3, 0.3, 0.2 |
| 0.7 | 0, 0.05, 0.75, 0.75, 1.0 |
| 0.5 | 0, 0.1, 0.75, 1.0, off-grid |
| 0.0 (pure concept) | 0, 0.2, 0.75, off-grid, off-grid |

A displaced but functionally intact sample needs few novel proteins; a function-reorganising one
needs many.

As an out-of-sample check, `beta = 0.15` reproduces three real eukaryote rungs it was not fitted
on — synthetic zero-shot 0.628 against fungi 0.619, metazoa 0.632, plants 0.610 — while archaea
needs `beta ≈ 0.02–0.05`, meaning the bacteria→archaea shift is close to orthogonal to function.

---

## The real-data arm

Swiss-Prot proteins carrying an EC number, embedded with ESM-C `esmc_300m` (960-D) and grouped
by NCBI lineage. The embedding cache holds **231,285 proteins**; after keeping only those with a
single unambiguous EC at level 3 and dropping the two remainder groups, the analysis set is
**216,592 proteins across 15 taxonomic groups, 269 EC classes and 3,237 Pfam families**, with
group sizes from 1,970 (insecta) to 64,791 (gammaproteobacteria). Source is gammaproteobacteria;
the 14 other groups are targets.

Two estimators run over it, reported in separate columns and never pooled: a warm-started MLP
adaptation loop on the 14-target ladder, and a cheaper logistic probe over all 210 ordered pairs.

**The estimator did not survive contact with real data.** `docs/notes/AUG7.md` §3 enumerates six
failures, each of which would have produced a confident wrong answer on its own. The two that
generalise beyond this project:

- **The ceiling is not automatically a bar.** Groups differ in size by 40×, so a target-trained
  "ceiling" model can be *worse* than the source model — the first run had zero-shot 0.240
  against a ceiling of 0.219, making `r* = 0` trivially. The fix reports three ceilings beside
  every `r*`: full-reservoir, size-matched, and budget.
- **`r*` against a full ceiling is not well posed at a small budget.** At a budget of 200 labels
  every pair was censored — never reaching the bar at any `r` — and at 500 about four in five
  were. Redefining the bar as budget-relative, the smallest fraction of a budget of P labels
  that must be target-native to match spending the whole budget on target data, cut censoring at
  P=500 to 2 pairs in 209, because `r = 1.0` reaches that bar by construction.

**Controls.** Permutation nulls permute taxonomic *groups*, never pairs, since 210 pairs come
from 15 groups and are not independent. A Pfam→EC majority-vote lookup with no embedding at all
reaches macro-F1 0.40–0.85, which bounds how much of the result is homology rather than
function. Within a single Pfam, where homology is constant by construction, ESM-C still
separates EC classes at 0.95–1.00 and transfers at 0.75–0.83 zero-shot against a label-shuffle
floor of 0.19–0.29 — though only three Pfam families had enough EC classes on both sides to
qualify, so that is a thin basis. The number of predictors tested is printed next to every
correlation table, and censored values are reported both ways.

**Honest limits**, stated at more length in `AUG7.md` §11: ESM-C is the strongest taxonomic
classifier of the pLMs tested, so this design cannot separate whether `r*` measures biology or
the embedding; the `_novelty` and `_sizematched` ladder arms produced no output, so the
synthetic and real halves are adjacent rather than joined; and the BLAST nearest-neighbour
baseline reports `nan` in the shipped run despite being the comparator the identity analysis is
framed against.

---

## Repository structure

```
tidythesis/
├── src/                          # All Python. Flat on purpose — see note below.
│   └── oracle_search/            # v0 oracle calibration (legacy)
├── slurm/                        # Every SLURM batch script; the current entry points
├── configs/                      # YAML configs for the legacy Nextflow pipeline
├── docs/
│   ├── notes/                    # Dated lab notes, newest last — start here
│   └── figures/                  # Generated figures + the command to rebuild each one
├── archive/                      # Superseded analysis and the five legacy experiments
├── main.nf                       # Legacy Nextflow DAG (drives v0)
├── main_precomputed.nf           # Legacy variant consuming a precomputed embedding cache
├── nextflow.config               # Executor profiles
├── requirements.txt              # Pinned Python environment (see the ESM-C caveat below)
├── Dockerfile                    # Containerised PyTorch + CUDA environment
└── verify_setup.sh               # Environment sanity check
```

**Why `src/` is flat.** The scripts import one another by bare module name
(`from measure_ec_geometry import …`) and the SLURM jobs invoke them as `src/<name>.py`.
Subpackages would break both, so they stay in one directory and are grouped here instead:

| Group | Files |
|---|---|
| **Synthetic generators** | `generate_synthetic_v2.py` (current), `generate_synthetic_precomputed.py` (v1), `generate_simulation.py` (v0) |
| **Calibration & realism** | `calibrate_v2.py`, `measure_real_geometry.py`, `realism_scorecard.py`, `compute_validation.py` |
| **Synthetic sweeps** | `run_distance_sweep_v2.py`, `run_distance_sweep.py` |
| **Real data** | `fetch_ec_swissprot.py`, `fetch_ec_annotations.py`, `build_ec_dataset.py`, `precompute_real_embeddings.py`, `embed_esm2_ec.py` |
| **EC recovery threshold** | `ec_recovery_threshold.py`, `ec_rstar_allpairs.py`, `ec_rstar_regress.py` |
| **Controls** | `ec_permutations.py`, `ec_homology_confound.py`, `ec_seq_identity.py`, `ec_distance_metrics.py` |
| **Geometry** | `measure_ec_geometry.py`, `measure_ec_damage.py`, `ec_allpairs.py`, `beta_diagnosis.py`, `subspace_experiment.py`, `whitened_geometry.py` |
| **Model core** | `model.py`, `train.py`, `adapt_OGadam.py`, `adapt_adamw.py`, `metrics.py` |
| **Plotting** | `plot_*.py`, `make_*.py`, `replot_*.py`, `export_*.py` |

Several analysis scripts carry hardcoded `/scratch/lmk04992/...` input paths as module
constants with no CLI override. They will need editing to run anywhere else.

---

## Environment setup

Python 3.11+ and PyTorch with CUDA. **Two conda environments are needed, not one.** `fair-esm`
(ESM-2) and the EvolutionaryScale SDK (ESM-C) both claim the import name `esm`, so they cannot
share an environment and `requirements.txt` cannot express both. `requirements.txt` covers the
analysis stack and ESM-2; ESM-C needs its own environment with the SDK installed.

```bash
conda create -n plm_dynamics python=3.11 -y
conda activate plm_dynamics
pip install -r requirements.txt
bash verify_setup.sh
```

Container alternative, built on `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime`:

```bash
docker build -t plm-thesis:1.0 .
docker run --gpus all -it -v "$PWD":/app plm-thesis:1.0 bash
```

On Sapelo2 the SLURM scripts load `Miniforge3` and activate prebuilt environments under
`/work/ah2lab/LiamK/conda_envs/` (`plm_dynamics` for analysis, `plm_esmc` for ESM-C embedding).
`slurm/ec_seq_identity.slurm` also loads the `BLAST+` module. `mmseqs`, `diamond` and `foldseek`
are not available on that cluster.

> **Reproducibility caveat.** Several scripts seed RNGs from Python's `hash()` of a string,
> which is salted per process unless `PYTHONHASHSEED` is set, and none of the SLURM scripts sets
> it. Re-running those draws different splits. Set `PYTHONHASHSEED=0` if you need exact
> reproduction. Separately, six SLURM scripts use `set -uo pipefail` without `-e` and swallow
> per-item errors, so a job can exit 0 and print `ALLDONE` with every invocation having failed —
> check the log body, not the exit status.

---

## Running the current work

Everything current runs as a standalone SLURM script from the repository root. Submit with
`sbatch slurm/<name>.slurm`; chain stages with `--dependency=afterok:<jobid>` rather than
waiting interactively.

**Synthetic arm**

```bash
python src/calibrate_v2.py                  # 36-cell geometry grid against the real target box
sbatch slurm/run_distance_sweep_v2.slurm    # 3 arms: distance, alpha, beta (~1,000 adapt runs)
sbatch slurm/structure_sweep.slurm          # 18 generator configurations × 5 d × 8 r × 3 seeds
```

**Real arm**

```bash
sbatch slurm/run_ec_swissprot_embed.slurm   # ESM-C embedding of the Swiss-Prot EC corpus
sbatch slurm/ec_rstar_ladder.slurm          # 14-target ladder, 4 budgets, 3 seeds, MLP estimator
sbatch slurm/ec_rstar_allpairs.slurm        # 210 ordered pairs, logistic probe, 16 workers
```

**Controls and distances**

```bash
sbatch slurm/ec_distances.slurm             # MMD, energy, proxy-A, feature-Wasserstein
sbatch slurm/ec_seq_identity.slurm          # BLAST identity to the nearest source protein
sbatch slurm/ec_homology_confound.slurm     # Pfam→EC lookup baseline + within-Pfam recovery
```

Figures regenerate from `src/`; see [`docs/figures/README.md`](docs/figures/README.md) for the
command behind each one.

---

## The legacy Nextflow pipeline

`main.nf` is a four-process Nextflow DAG (`GEN_SOURCE`, `TRAIN_SOURCE`, `GEN_TARGET`,
`TEST_ADAPTATION`) that fans a combinatorial sweep across SLURM, with CPU data generation and
A100 training. It is kept because it ran the original thesis experiments and because the
`liam_sapelo2` profile is a working example of Nextflow on a cluster with a job-count limit.

**It is not the current pipeline, and it should not be read as one.** Three reasons:

1. It calls `src/generate_simulation.py`, the v0 oracle generator, whose structural defect is
   described above. None of the v2 or EC work runs through it.
2. It passes the same file as both `--ref_x` and `--source_x`, so the Wasserstein distance it
   reports is a comparison of the source set with itself. Older text describing that distance as
   the x-axis for the recovery threshold is wrong twice over: feature-Wasserstein was separately
   rejected as an x-axis because it is non-monotone in shift distance and blind to label shift.
3. It drops `--seed` and `--eval_every` when calling the training and adaptation scripts, so
   raising `num_seeds` varies only the generated data, and the default 500-batch evaluation
   interval is too coarse to resolve the dip on realistic pool sizes.

The five original experiment launchers and their configs are in `archive/legacy_experiments/`.
To run the DAG anyway:

```bash
nextflow run main.nf -profile standard -params-file configs/template_master.yaml -resume
```

Only the `liam_sapelo2` profile is complete; the generic `slurm` profile has its queue and
`beforeScript` commented out and submits jobs with no partition.

---

## Citation

> Liam Kozma. *Identifying the Recovery Threshold for Protein Language Models under Data
> Distribution Shift.* Master of Science in Statistics Thesis, University of Georgia, 2026.
> Advisor: Dr. Adrienne Hoarfrost.

*Compute provided by the University of Georgia GACRC Sapelo2 cluster.*
