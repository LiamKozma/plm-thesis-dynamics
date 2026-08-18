# Task: turn an outside research agent's proposals into experiments worth running

Another model was given `docs/notes/ABSTRACT_RESEARCH_BRIEF.md` — this project stated
in domain-neutral terms, with the domain deliberately stripped — and asked to propose
experiments. Its output is what you are evaluating.

**Your job is not to implement its proposals.** It reasoned without access to any of
the data and without knowing the domain, so a good fraction of what it returns will be
already done, already answered, impossible under constraints it could not see, or
based on an assumption that is false here. Your job is to find the ones that are not,
translate those into this project, and make them runnable.

**Be adversarial before you are constructive.** A proposal that survives you is worth
more than five that you merely tidied up.

---

## 1. Read these first

In order. Do not start evaluating until you have.

1. `docs/notes/AUG18.md` — the most recent lab note. The current state of every result.
2. The **Open work** page of `docs/notes/PIPELINE_dashboard.v2.html`, especially the
   roadmap block at the top (`id="ow-roadmap"`) — what is blocked on what.
3. `docs/notes/BRIEF_ec_recovery_threshold.md` §5 — the landmines. Every one of them
   cost a wrong conclusion at least once.
4. `docs/notes/ABSTRACT_RESEARCH_BRIEF.md` — what the other model was actually told,
   so you can see what it did and did not know.

---

## 2. The de-abstraction key

The other model was writing about abstract point clouds. Map its vocabulary back:

| it says | it means here |
|---|---|
| frozen encoder `φ`, R^960 | ESM-C `esmc_300m` embeddings of protein sequences |
| items | proteins (231,285; 216,592 after dropping two remainder categories) |
| environments `E₁…E₁₅` | taxonomic groups — 9 bacterial, 2 archaeal, 4 eukaryotic |
| coarse label `G` (3,282 values) | Pfam family |
| fine label `Y` (270 values) | EC number at level 3 — **the prediction target** |
| cheap pairwise similarity `s` | percent identity from BLAST+ |
| second pretrained encoder | ESM-2 650M |
| "the simulator" | `src/generate_synthetic_v2.py` + `run_distance_sweep_v2.py` |

If a proposal only makes sense under the abstraction — if translating it produces
something biologically incoherent — say so plainly. That is a real finding about the
proposal, not a translation failure on your part.

---

## 3. Reject on sight

**Already answered.** Check these before anything else; the other model was told most
but not all of it, and it may propose them anyway:

- coverage vs geometry as competing explanations of retention (both survive partialling
  on each other; ρ = +0.848 for coverage, geometry holds at −0.37 to −0.62)
- extending the n=14 correlations to n=209 (done; they shrink but hold, `gap_ratio` dies)
- a within-group null for retention (done; the band is 0.902–1.039)
- per-class-of-enzyme retention against prevalence (done; clean null, ρ = −0.048)
- length or composition as the mechanism (done; absolute length is a null, distributional
  distance is just another distance)
- whether the subspace result is encoder-specific (done; ESM-2, ordering agrees at 0.960)
- percent identity at all 210 pairs (done; ρ = +0.752, and the ~40% boundary is a
  prokaryote phenomenon that fails in all four eukaryotic sources)
- BLAST against the embedding (done matched, with both a probe and the MLP; **and it is
  out of scope regardless** — a lookup table has no adaptation curve, so no r\*)

**Structurally impossible.** Any proposal that displaces classes independently while
claiming to hold the target problem fixed. That is not an isometry: independent
per-class displacement of magnitude *m* inflates the centroid gap as `√(g² + 2m²)`, so
the ceiling moves, so r\* becomes unreadable. This is the single most common way to
propose something that cannot work. **If a proposal needs differential motion — and the
best ones will, since that is the strongest real predictor at ρ = −0.819 — it must say
explicitly how it keeps target difficulty fixed, or it must be reframed as
observational.**

**Underpowered by construction.** Anything whose unit of analysis has n < ~20. We have
15 groups, not 210 independent ones. A proposal resting on 2 archaea vs 3 eukaryotes is
an anecdote unless it finds a continuous statistic underneath the split.

---

## 4. Judge what survives

For each remaining proposal, write:

1. **What it would establish**, in one sentence, and **what result would undermine the
   current account.** If it can only confirm, it is not worth running. Say so.
2. **Is it actually new?** Cross-check against the experiment catalogue on the dashboard
   and against `AUG18.md`. Near-duplicates of finished work are the most common failure.
3. **Which landmine is it most exposed to**, and is that controlled? The recurring four:
   matched controls drawn from the data covariance rather than isotropically;
   group-level permutation nulls, never pair-level; class counts matched before
   cross-dataset comparison; the ceiling held fixed under the treatment.
4. **Cost**, concretely. Simulator sweeps are ~11 minutes. Anything reusing the cached
   embeddings is minutes to a couple of hours on 16 CPUs. Re-embedding needs a GPU at
   ~20 min per 20,000 sequences. BLAST over all pairs is ~1.5 h on 32 cores.
5. **Your confidence it returns something**, and whether a null would still be worth
   having. Several of the most valuable runs here returned nulls.

Then **rank them**, and be willing to rank most of them last.

---

## 5. What we most want

In descending order. A proposal that lands on one of these beats a cleverer one that
does not.

- **(a) Telling the two failure modes apart before labelling.** Some groups transfer
  badly and repair for free with unlabelled data (archaea, +0.104 and +0.236); others
  transfer badly and are made worse by every method (all three eukaryotic groups,
  −0.016 to −0.040). Nothing explains the difference. A label-free diagnostic that
  predicts which regime you are in would be the most useful single result available,
  because the two call for opposite actions. This is item **T1.9**.
- **(b) A two-level generator.** Fine labels nested inside coarse ones, hitting the
  measured targets: 26 function dimensions inside 50 family ones at principal-angle
  cosine 0.928, centroids 1.11 within-group widths apart, a 99.4% additive grid, 69%
  purity at 2.98 distinct functions per family. Plus: what would falsify it.
- **(c) Differential motion without moving the ceiling** — see §3.
- **(d) A better-posed quantity than r\*.** It is a threshold on an 8-point grid scored
  against a budget-dependent bar, and it has misbehaved in three separate arms while
  retention stayed stable. If the other model proposes a replacement estimator that
  answers the same practical question, take it seriously.

---

## 6. Deliverable

**Do not** produce a summary of the other model's output. Produce:

1. A **ranked shortlist**, ideally 3 to 6, each with §4's five fields.
2. A **rejected list**, one line each, with the reason. Short. This is the part that
   stops the same proposal being reconsidered next month.
3. For the top 2 or 3: **a working batch script** in `slurm/`, following the existing
   ones (`slurm/tier1_backlog.slurm` is a good template — module load, conda activate
   `/work/ah2lab/LiamK/conda_envs/plm_dynamics`, `PYTHONHASHSEED=0`, write to
   `/scratch`, incremental output so a killed job still leaves something usable).
   Smoke-test at small settings before submitting the real thing.
4. A short note in `docs/notes/` in the style of `AUG18.md` recording what was proposed,
   what was kept, what was rejected and why — including the nulls.

---

## 7. Practical

- **Cluster.** `ssh sapelo2 '<cmd>'` works once the ControlMaster is up; check with
  `ssh -O check sapelo2`. If it is down the user must run `./connect-sapelo2.sh`
  themselves — it needs an interactive Duo push and you cannot do it for them.
- **Repo.** `/work/ah2lab/LiamK/tidythesis`, mounted locally at
  `~/Documents/sapelo2/work/tidythesis`. Submit from the repo root: `sbatch slurm/<x>.slurm`.
- **Data.** `/scratch/lmk04992/ec_swissprot/` — `emb_cache_esmc.npy` (231285 × 960,
  row *i* ↔ line *i+1* of `data/metadata.tsv`), `data/ec_annotations.tsv`,
  `raw/ec_swissprot_raw.tsv` (adds lineage, organism, length, full multi-Pfam list).
  Results from this week: `/scratch/lmk04992/{subspace_rstar,shift_decomp,tier1,
  ec_zero_label,ec_rstar,synth_v2_repaired}/`.
- **The other project.** `/work/ah2lab/LiamK/threshold_lowering/` holds the twelve
  adaptation methods and is **not** under version control. Do not edit it destructively;
  there is a `.bak_18aug` beside the one file already modified.
- **Clades.** Use `src/clades.py` and `targets_by_clade(source)`. Never write a literal
  set of group names — doing that twice produced two wrong conclusions in one day.
- **Commits.** Authored by Liam Kozma. **No AI attribution of any kind** — no
  `Co-Authored-By`, no "generated with", no assistant or vendor names, in messages or
  in any committed file. Grep before committing; this has been enforced across the
  whole history.
- **Long jobs.** Always `sbatch`, never interactive — the SSH session may not outlive
  the run. Write output to `/scratch` so a later session can pick it up.

---

## 8. If the other model returned little of value

Say so directly rather than manufacturing a shortlist. Then propose your own, against
§5, using the same five fields. That is a perfectly good outcome and more useful than
a polished list of things not worth running.
