# Brief — measuring the recovery threshold for EC (function) prediction

**Written 2026-08-06 as a handoff.** Everything below is established fact plus open
questions. Nothing here has been decided; the point of the next session is to decide
it, run it, and report.

Read this together with [`AUG6.md`](AUG6.md) (most recent results) and
[`AUG4.md`](AUG4.md) (where the EC label axis came from).

---

## 1. The gap, stated precisely

The thesis's goal is that the model should **predict EC number** (function). Every
recovery-threshold number produced so far is **family (Pfam) recovery**.

| | Family labels | **EC labels** |
|---|---|---|
| Synthetic | ✅ `r*(d)` — the v2 distance sweep | ❌ oracle stands in, never swept |
| Real | ✅ `r*(d)` — the 5-domain taxonomy ladder | ❌ **nothing** |

Why the hole exists, mechanically: `src/precompute_real_embeddings.py` builds the
`.npy` splits the adaptation loop consumes, and it hardcodes the label as Pfam family:

```python
fam2id = {f: i for i, f in enumerate(fam_list)}
y_all  = np.array([fam2id[f] for f in families], dtype=np.int64)
```

EC labels never reach the adaptation loop. The EC analyses
(`src/measure_ec_damage.py`, `src/ec_allpairs.py`) do use EC labels, but they only
compute **zero-shot** and **ceiling** — they never mix an adaptation pool, so they
never produce `r*`.

**Filling that cell is the next step.** All inputs already exist; see §3.

---

## 2. What is established, and how confidently

Cite these rather than re-deriving them. Statistics are over 210 ordered group pairs
from 15 taxonomic groups unless stated.

### Solid

- **Functional shift is highly structured.** Moving EC e1 → e2 gives nearly the same
  displacement vector in every taxonomic group: α = **+0.531** against a cross-move
  null of **+0.008**, over 4,000 distinct EC→EC moves and 50,400 cosines. At EC
  level 4 (exact reaction) it is *higher* (+0.641), so it is not a binning artifact.
- **All three label axes are structured.** Taxonomy, function and scaffold all give
  within-move-minus-cross-move excesses of **+0.52 to +0.64**.
- **`gap ÷ within-EC σ = 1.11`** (131 EC centroids, 8,515 pairwise distances). EC
  groups sit about one within-group width apart. Every `|v|/gap` should be read
  against this.
- **ESM-C does not encode the EC tree.** Spearman(embedding distance, EC-tree
  distance) ≈ 0 in all 15 groups (−0.05 to +0.34, no consistent sign). Tested at 270
  EC classes from 3,282 Pfams, so this is not a small-sample artifact. **Do not build
  EC-tree structure into the oracle.**
- **Best damage predictors** (Spearman vs retained EC F1, all p < 0.001):
  `‖v−v̄‖/gap` −0.784 · `‖P_B(v−v̄)‖/gap` −0.773 · probe logit spread −0.671 ·
  Procrustes disparity −0.634 · `|v|/gap` −0.592.
- **α does not predict damage at all**: +0.022, p = 0.87. It remains a good realism
  knob; it is not a cost measure.
- **β has the wrong sign** (+0.280, p = 0.006) and this is *causally reproducible*
  under a matched control — see the landmine in §5.

### Important and under-explored

- **The Pfam × EC centroid grid is 99.4% additive, and Pfam alone explains 95.6%.**
  Homology almost entirely determines where a protein sits; function adds very little
  on top. **This is a direct threat to the thesis's framing: an EC probe may largely
  be reading homology.** Worth a dedicated experiment — e.g. measure EC recovery
  *within* a single Pfam, where homology is held constant.

---

## 3. What exists to build on

### Data — `/scratch/lmk04992/ec_swissprot/`

| Path | What | Size |
|---|---|---|
| `data/metadata.tsv` | `id, family(Pfam), group(taxon)` — 231,285 rows | |
| `data/ec_annotations.tsv` | `id, ec, ec_full` — EC level 3 | |
| `data/seqs.fasta` | sequences, same row order | |
| `emb_cache_esmc.npy` | **(231285, 960) float32** — row *i* ↔ line *i+1* of `metadata.tsv` | 768 MB |
| `raw/ec_swissprot_raw.tsv` | + full NCBI `lineage_ids`, `organism_id`, `length` | |
| `raw/ec_swissprot.fasta` | all sequences (pre-filter) | |
| `analysis/*.json` | the AUG6 results | |

231,285 proteins · 17 taxonomic groups (**15 usable** — drop `other_bacteria` and
`other_eukaryota`, they are remainders not clades) · 270 EC classes at level 3 ·
3,282 Pfams. Group sizes run 64,791 (`gammaproteobacteria`) down to 1,970 (`insecta`).

Older, smaller: `/scratch/lmk04992/taxonomy_ladder/` — 36,981 proteins, 5 domains,
family labels. This is what the existing *real* family `r*` curve was measured on.

### Environments

```bash
module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate /work/ah2lab/LiamK/conda_envs/plm_dynamics   # numpy/sklearn/scipy/torch — CPU analysis
conda activate /work/ah2lab/LiamK/conda_envs/plm_esmc       # ESM-C SDK — GPU embedding only
```

`BLAST+/2.13.0-gompi-2022a` is available as a module (not installed in either env).
Biopython is in `plm_dynamics`. `mmseqs`, `diamond`, `foldseek` are **not** available.

### Relevant code

| Script | Does |
|---|---|
| `src/measure_ec_damage.py` | `run_pair()` — linear probe, EC labels, zero-shot + ceiling for one group pair. **The natural place to add pool mixing.** |
| `src/beta_diagnosis.py` | 12-predictor battery over all 210 pairs, with a group-level permutation null. Reuse `part_b`'s structure for regressing `r*`. |
| `src/ec_permutations.py` | all 8 conditioned shift types, gap definitions, additivity, EC hierarchy |
| `src/precompute_real_embeddings.py` | builds `.npy` splits — **needs an EC-labelled variant** |
| `src/adapt_OGadam.py` | the MLP adaptation loop the synthetic `r*` is defined against |
| `src/run_distance_sweep_v2.py` | how `r*` is computed on the synthetic side; mirror its definition |

Batch scripts live in `slurm/`. Submit from the repo root: `sbatch slurm/<name>.slurm`.

---

## 4. The open questions — decide these first

### Q1 · Which experiment first?

**(a) Real EC ladder, one source.** Fix `gammaproteobacteria` (64,791 proteins) as
source, sweep to all 14 other groups × r ∈ {0, .05, .1, .2, .3, .5, .75, 1.0}. ~112
runs. Mirrors the original bacteria→X ladder with 14 rungs instead of 4, and is
directly comparable to the existing synthetic curve. *This is the recommendation
unless analysis says otherwise.*

**(b) All 210 pairs, linear probe.** Many more points to regress distance against and
cheap per pair — but a linear probe is not the same estimator as the MLP adaptation
loop, so the resulting `r*` is not comparable to the synthetic curve. Possibly worth
running *as well*, as a cheap wide scan.

**(c) Split the novelty axis first.** See Q3.

**(d) Fix the generator to emit EC-like labels first.** Keeps synthetic and real
aligned, but the oracle's purity/coverage conflict (AUG4 §6) is unsolved and could
stall.

### Q2 · Which "distance"?

The metric should be **computable before any labelling**, because the practical
payoff is telling a scientist their annotation budget in advance.

| Metric | Cost | Why it matters |
|---|---|---|
| `|v|/gap`, `‖v−v̄‖/gap`, Procrustes, logit spread | **free** — already computed for 210 pairs | best current predictors of damage |
| **% identity to nearest training protein** | BLAST+ job, hours | the most familiar number in biology; the "twilight zone" below ~30% is exactly this regime. Most likely to reach the abstract. |
| **taxonomic rank of the split** | free — lineage is in `raw/ec_swissprot_raw.tsv` | gives a quotable rule of thumb: "different phylum → label X%" |
| **MMD / Wasserstein** source vs target embeddings | cheap | fully label-free; the strongest form of "know your budget in advance" |

At minimum do the free ones. Sequence identity is the one that makes the result
speak to biologists rather than only to ML people.

### Q3 · Two different "distances" are currently conflated

1. **Same ECs, new organisms** — taxonomic shift. All that has been measured.
2. **New ECs absent from the source** — functional novelty. Never measured.

A real metagenome has both. `r*` may not even be well-defined for (2): you cannot
recover a class you have never seen, only discover it. Decide whether to restrict to
the shared-EC set (clean, comparable, what `run_pair` already does) or to measure both
and report them separately. **Do not let them average together silently.**

### Q4 · Which estimator defines `r*`?

The synthetic `r*` uses the MLP adaptation loop + "90% of ceiling". The EC analyses
use `sklearn` logistic regression. Mixing makes numbers non-comparable. Pick one,
state it, and if both are run, report them as separate columns.

---

## 5. Landmines — these already cost us once each

1. **Match your control condition.** The AUG4 conclusion "β causally damages function"
   was an artifact of drawing the off-subspace direction *isotropically* while the
   in-subspace direction was drawn *within B*. That contrast is "typical vs atypical
   direction", not "in B vs out of B". With both drawn from the data covariance, β is
   **protective**, monotonically. Any injection-style experiment must use a matched
   control.
2. **Match class counts when comparing synthetic to real.** The "hubness is 4× off"
   finding evaporated (ratio 0.98) once the synthetic set had 131 classes instead of
   16. kNN label consistency shrank the same way.
3. **Underpowered rows look like findings.** "Scaffold change is unstructured" came
   from 8 moves / 36 vectors on the 16-family dataset; with 3,282 Pfams it is 979
   moves and the opposite conclusion. Report `n` next to every statistic and distrust
   rows with < ~20 moves.
4. **Permute groups, not pairs.** The 210 pairs come from 15 groups and are not
   independent. `beta_diagnosis.py:perm_p` does this correctly — copy it.
5. **Multiple comparisons.** 12 predictors were tested; ~0.6 false positives at
   p < 0.05 are expected. Say so.
6. **`--max_pairs 4000` caps the EC|DOM row**; other rows are uncapped.
7. **Labels are assigned in source coordinates and carried through a shift**, never
   recomputed on shifted points. A plant enzyme doing 1.1.1 is still doing 1.1.1.
8. **Only isometries preserve target difficulty.** Translation and rotation are safe;
   independent random displacement inflates gaps (`√(g²+2m²)`), interpolating toward a
   swap collapses them, and interpolating toward a *fresh draw* preserves only the
   distribution and drifted the ceiling 0.90 → 0.95. See `generate_synthetic_v2.py`'s
   docstring.
9. **Always report the ceiling next to `r*`.** `r*` is scored against it; a moving
   ceiling silently changes the question. This was v1's fatal bug.

---

## 6. Working on the cluster

- Slurm. `batch` = CPU; `gpu_p` = A100, 7-day limit; `gpu_30d_p` = 30-day.
- An A100 embedded 231k sequences with ESM-C 300m in **~2 h**. The 210-pair analysis
  ran in **~16 min** on 16 CPUs.
- Chain stages with `sbatch --dependency=afterok:<jobid>` rather than waiting
  interactively — submitted jobs survive if the SSH session dies.
- **The SSH connection needs an interactive Duo login to establish.** Once the
  ControlMaster is up, `ssh sapelo2 '<cmd>'` is passwordless, but it will not survive
  indefinitely. Anything long-running must be `sbatch`-ed with output written to
  `/scratch`, so a later session can pick up the results regardless.
- Keep BLAS single-threaded (`OMP_NUM_THREADS=1`) when parallelising per-pair work
  across processes, otherwise they oversubscribe.

---

## 7. What to deliver

Write `docs/notes/<MONTHDAY>.md` in the style of `AUG4.md` / `AUG6.md`: what was
built, what was measured, what held, what did not, honest caveats, next steps.
**Do not overwrite `AUG6.md`** — new file.

**If the experiments do not finish, still write the report.** State plainly:
- which stages completed and what they showed
- which are still running (job IDs) or never started, and why
- where partial output lives on `/scratch`
- the exact command to resume

A truthful partial report is the deliverable; a missing report is not acceptable
merely because a sweep was still queued.

Then commit and push to `github.com/LiamKozma/plm-thesis-dynamics`.

> **Hard requirement:** commits are authored by Liam Kozma and must contain **no AI
> attribution of any kind** — no `Co-Authored-By` trailer, no "generated with", no
> assistant or vendor names, and no assistant-hosted share links, in commit messages
> or in any committed file. This has been enforced across the existing history; keep
> it that way.
>
> Local editor/tooling state is excluded via `.git/info/exclude` rather than
> `.gitignore`, because `.gitignore` is itself committed and would advertise the tool.
> Keep that arrangement, and check new files with a grep before committing.
