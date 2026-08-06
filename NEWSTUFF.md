# NEWSTUFF — running work log

A chronological log of thesis work. Newest entries go at the **top** of the log
section. Each entry: date, what we set goal was, what we did, what came out, and
where the artifacts live. Add freely as the week goes.

**Project in one line:** measure the *recovery threshold* — how much out-of-distribution
(OOD) target data an adapting model needs to overcome negative transfer — and how that
threshold changes with **distance** from the training data.

**Key locations**
- Repo (sapelo2): `/work/ah2lab/LiamK/tidythesis` (mounted locally at `sapelo2_files/work/tidythesis`)
- Scratch outputs: `/scratch/lmk04992/`
- Conda env: `/work/ah2lab/LiamK/conda_envs/plm_dynamics` (ESM-2 + torch); ESM-C will use `plm_esmc` (not built yet)
- Background docs: `june17_premeeting.md`, `HOW_TO_SEE_THE_DIP.md`, `NEXT_STEPS.md`, `RESULTS_distance_sweep.md`

---

## Log

### 2026-07-15 — Real-data validation of the identifiability finding + estimation follow-up

**Other session ran the real-data test** (`threshold_lowering/REPORT_REAL.md`; outputs in
`/scratch/lmk04992/threshold_lowering/real/`). 5 real shifts (SwissProt→TrEMBL, Bacteria→{archaea,
fungi,metazoa,plants}), ESM-C 960-D, zero-label alignment.
- **Identifiability prediction CONFIRMED — by correctly predicting a FAILURE.** Real Pfam clusters in
  ESM-C space have silhouette 0.02–0.05 (between synthetic σ=4 and σ=6 = the unidentifiable regime).
  Prediction: OT must fail. Measured: **OT is the WORST method on all 5 shifts** (−0.21 to −0.34 F1).
  This REFUTES the task's premise that families are "distinct clusters" — they overlap heavily.
- **Method ranking INVERTS vs synthetic: BN recalibration WINS** (best everywhere, up to +0.13);
  genuine zero-label recovery on 2/5 (archaea, swissprot) — archaea r* 0.1→0 (removes ~100 labels).
- **NEW real-data insight — a SECOND obstacle: ESTIMATION.** n≈d (1000 unlabeled, 960 dims): CORAL
  must fit a 960×960 covariance from 1000 samples (under-determined → hurts); BN fits 2×960 diagonal
  (well-determined → wins); OT worst. Lesson: use lowest-capacity alignment matching the shift.
- Real shifts confirmed genuinely COVARIATE (λ̂ 0.11–0.19 low, d̂_A 0.89–1.85 large); r* low (0–0.1).
- Rigor: pre-registered silhouette prediction, pca_shrink + k-sweep control (OT failure is transport
  not projection), leak guard (dropped pool∩test rows), held-out. Found data issues: it's 5 shifts
  not 6, pool∩test overlap ≤2.6% (deduped), severe test class imbalance (plants/fungi noisy).
- HONEST GAP: only the FAILURE half of identifiability is shown on real data (all 5 shifts cluster at
  low silhouette); the positive half (OT working) rests on synthetic evidence only.

**My overnight follow-up (job 47009884): the ESTIMATION-obstacle sweep.** Tests the report's new
claim directly — does label-free CORAL/OT recover as unlabeled n grows past d=960? Uses the cached
ESM-C embeddings (no re-embed); `src/estimation_sweep.py` reuses methods.py verbatim; sweeps
n∈{250..6000} on Bacteria→{archaea,fungi}, 3 seeds. **RESULT (job done, plot
`estimation/estimation_sweep.png`):** the two obstacles cleanly SEPARATE.
- **CORAL rises with n** (archaea 0.767→0.804; fungi 0.480→0.515), saturating ~n≈2500–4000 →
  estimation obstacle CONFIRMED (better covariance estimate with more samples). But the gain is
  MODEST (~+0.04) and doesn't change the ranking: on archaea CORAL crosses baseline (0.766) only
  slightly; on fungi it never reaches baseline (0.626) even at n=6000.
- **OT flat-broken at every n** (~0.55 archaea / ~0.32 fungi) → pure identifiability failure,
  n-independent.
- **BN flat-best, n-independent** (~0.848 / ~0.678) → wins by capacity-matching (2×960 diagonal
  params well-determined from few samples), not sample size.
- **Conclusion: identifiability DOMINATES; estimation is real but secondary.** Even with 6× more
  unlabeled data than dimensions, low-capacity BN beats full-covariance alignment on real ESM-C data
  because the family clusters are unidentifiable. Cleanly completes the two-obstacle decomposition on
  real data. Output: `/scratch/lmk04992/threshold_lowering/estimation/`.

### 2026-07-15 — Threshold-lowering study RESULTS (autonomous agent, audited)

The threshold-lowering agent (Fable→Opus) completed the study in `work/threshold_lowering/`
(own `REPORT.md` + `LOG.md`; figures/CSVs in `/scratch/lmk04992/threshold_lowering/analysis/`).
A second agent independently AUDITED it (re-derived key numbers from raw thresholds.csv, held-out
seeds, controls) — findings hold. Answer to "can we need less new data?": **yes, and how depends on
shift type.**

- **OT feature alignment (+BN), `ot_bn`/`align_ot`** = most robust. On covariate shift reaches
  **zero-label recovery** (r* 0.148→0.000 at D=0.6). Best single concept-shift win =
  **`conflict_prune` (source pruning), ~6× fewer labels** (0.854→0.144) when clusters are distinct.
- **Zero-label test-time adaptation (BN-recalib, TENT) does ~nothing** in either regime — only fixes
  translation/scale (already free); can't touch rotation or concept shift. Confirms the floor.
- **CORE FINDING — identifiability:** "covariate shift is fixable with unlabeled data" only holds if
  the shift is IDENTIFIABLE from P(x). CORAL (covariance matching) can't identify a rotation and is
  WORSE than doing nothing; OT fixes rotation perfectly when clusters are distinct, collapses when
  they blur (an analytic oracle recovers it throughout → info present but not estimable). An
  unidentifiable covariate shift is as label-hungry as concept shift despite no info-theoretic floor.
  Novel, sharp, publishable.
- Frozen backbone/head-only HURTS covariate (can't undo a rotation of its own input); LP-FT helps
  concept. Theory: concept loads on Ben-David λ (0.24→1.06), covariate on divergence d_A (0.40→1.41);
  matches Blitzer β* direction.
- **Caveats:** agent found & fixed 2 of its own bugs mid-run (a wrong-budget ceiling that made all
  methods look identical — pre-fix runs were invalid); **real-data validation NOT run** (synthetic
  only); headroom-vs-identifiability tension is intrinsic to the generator. Open next step: test the
  identifiability prediction on the real ESM-C embeddings.

### 2026-07-14 (~midnight) — Launched the threshold-lowering agent OVERNIGHT

Kicked off a long-running automated experiment sweep on the master prompt in
`/work/ah2lab/LiamK/threshold_lowering/`. Task: implement the covariate-shift regime + the 5
interventions (test-time/BN adaptation, instance reweighting, active sampling, feature alignment,
LP-FT/frozen), compare each to the naive-finetune baseline, and produce overlaid r*(D) curves for
concept AND covariate shift + a REPORT.md. It works in its own dir (outputs to
`/scratch/lmk04992/threshold_lowering/`), uses plm_dynamics (no ESM-C), keeps its own LOG.md, and
was told strict guardrails (only its own dir + jobs, don't touch tidythesis). Check its LOG.md /
REPORT.md in the morning.

### 2026-07-14 (night) — BUG: ESM-C silent CPU fallback (cu130 vs node driver); fixed

The taxonomy-ladder embed job (46980032) ran 1h36m and only reached 5,700/36,981 (~60/min vs the
~2,000/min the SwissProt ESM-C job hit). Root cause: **`pip install esm` pulled torch 2.13.0+cu130
(CUDA 13.0), but GPU node b8-2 has NVIDIA driver 12.8** → `torch.cuda.is_available()` False → SILENT
CPU fallback → ~30x slower. The SwissProt job only worked because it landed on b7-4 (newer driver).
The cluster's A100 nodes have **mixed driver versions**, so cu130 is a landmine.

Fixes: (1) cancelled the doomed job + its 4 dependent sweeps; (2) repinned torch to **2.6.0+cu124**
in `plm_esmc` (cu124 runs on driver ≥12.4 = every A100 node); (3) updated `setup_esmc_env.sh` to
install esm first then force cu124 torch last; (4) added a **guard** to `precompute_real_embeddings.py`
so ESM-C now FAILS LOUDLY if it would run on CPU instead of silently crawling. Re-running the ladder
after verifying the torch fix. (Note: the earlier SwissProt ESM-C results are unaffected — they ran
correctly on GPU.)

### 2026-07-14 (night) — Research direction: LOWERING the recovery threshold

Idea (Liam): can we engineer a method so a PLM + a scientist's new-source data needs LESS new data
to keep predicting? = design an adaptation method whose threshold-vs-distance curve sits BELOW naïve
fine-tuning. Feasible because the threshold isn't fundamental — in our pool the SOURCE data fights
realignment, and fine-tuning distorts pretrained features (Kumar et al. ICLR 2022). Hard floor:
Ben-David λ — genuine *concept* shift (P(y|x), = our synthetic knob) needs SOME target labels; can't
beat that without labels. Whether labels can be avoided depends on covariate (P(x)) vs concept
(P(y|x)) shift — which we can MEASURE. That distinction is itself a result.

Interventions to test, each = a new threshold-vs-distance curve in the existing pipeline (ranked):
1. **Test-time adaptation on UNLABELED target** — our MLP has BatchNorm; update BN stats / entropy-min
   (TENT). Zero labels. Cheapest; directly matches "scientist drops in new data." TRY FIRST.
2. **Reweight/prune the pool** — downweight source examples conflicting with target region (importance
   weighting / density ratio). Attacks the "source fights recovery" mechanism directly.
3. **Active sampling of target** — label most-informative target points (uncertainty near shifted
   boundary) vs random. Provable label-complexity reduction; biggest practical win.
4. **Feature alignment (CORAL / OT)** — align target onto source manifold. Sharp test of covariate-vs-
   concept: should work on a covariate-shift synthetic, FAIL on our concept-shift one.
5. **LP-FT / LoRA / frozen backbone** — fewer DOF, less feature distortion, lower threshold.
Deliverable: which interventions move r*(d) down, by how much, vs distance AND shift type; and does
empirical r*(d) match classical optimal-mixing theory (Blitzer β*). Prototype #1 after the ladder.

### 2026-07-14 (night) — Taxonomic ladder RESULTS

Real threshold-vs-distance (source=Bacteria; distance = 1 − pre-adapt baseline F1; r-grid
{0,0.1,0.25,0.5,1.0}, 3 seeds). Unified figure vs synthetic:
`/scratch/lmk04992/taxonomy_ladder/threshold_vs_distance_unified.png`.

| target | distance (1−base) | ceiling | recovery threshold r* |
|---|---|---|---|
| Archaea | 0.18 | 0.91 | 0.25 |
| Fungi | 0.38 | 0.86 | 0.10 |
| Metazoa | 0.36 | 0.90 | 0.25 |
| Plants | 0.37 | 0.84 | 0.25 |

Reads:
- Real taxa **order sensibly by distance**: Bacteria→Archaea (both prokaryotes) is the CLOSEST
  (deg 0.18); the three eukaryotic groups cluster farther (deg ~0.36–0.38). So really ~2 distinct
  real distances, not 4.
- **All real shifts are MILD**: max degradation ~0.38, and ≤25% target data recovers even
  Bacteria→Plants. None reach the synthetic curve's steep knee (which is at degradation >0.9,
  i.e. near-total source failure). The threshold "explosion" is a SEVERE-shift regime real
  taxonomy on conserved families doesn't reach.
- Directionally consistent with the synthetic law but points sit somewhat ABOVE the curve and are
  noisy (coarse 5-pt r-grid, 3 seeds; archaea anomaly = higher r* at lower distance, likely grid
  noise / its higher 0.91 ceiling raising the 90% bar).
- INTERPRETATION tying to the covariate-vs-concept axis: real taxonomic shift is largely
  **covariate** (families conserved, embeddings relocate) → stays benign/recoverable → low
  threshold; the synthetic **concept** shift is where thresholds explode. Real lands in the benign
  regime BECAUSE it's covariate-ish. Directly motivates the threshold-lowering directions (covariate
  shift is alignable → little target data needed).
- To strengthen: finer r-grid + more seeds; and a HARDER real shift (beyond kingdom-level, or
  within-superfamily label confusion) to push real data into the steep regime.

### 2026-07-14 (night) — Launched real taxonomic distance ladder + novelty check

**Experiment launched.** Real threshold-vs-distance: source = Bacteria, targets at increasing
taxonomic distance = Archaea, Fungi, Metazoa, Plants (non-overlapping clades), 16 shared Pfam
families. Fetched **36,981 seqs** (bacteria 8000 / archaea 7802 / fungi 7107 / metazoa 7012 /
plants 7060), all families present. Pipeline chained on SLURM:
`taxonomy_embed.slurm` (46980032, ESM-C, embeds once via new `--emb_cache`, splits per target)
→ 4 × `run_real_sweep.slurm` (46980033-36, afterok) → then `src/plot_threshold_vs_realdistance.py`.
Unified figure plots recovery threshold vs **distance = (1 − pre-adapt baseline F1)** =
classifier degradation (the right ruler; Wasserstein is non-monotone here), synthetic curve + real
taxa on one axis. Data → `/scratch/lmk04992/taxonomy_ladder/{archaea,fungi,metazoa,plants}`.

**Novelty check (literature agent, ~30 verified sources).** Verdict: the exact framing —
*a recovery threshold (target-data fraction to overcome negative transfer) that grows with a measured
source→target distance, with a U-shaped adaptation curve, on protein language models* — was NOT
found in any single paper. Component ideas all have prior art; the **coupling + operationalization on
PLMs is the novel contribution**. Anchor citations to position against:
- Negative transfer defined: Wang et al. CVPR 2019 (1811.09751); survey Zhang et al. 2023 (2009.00909).
- Divergence→target-error bounds: Ben-David et al. 2010 (A-distance/HΔH); optimal source/target MIX
  vs divergence+sample-size: Blitzer et al. NeurIPS 2007 (β* — closest "how much target vs distance").
- Phase transitions (run the OTHER direction — similarity threshold rises with data): Dhifallah & Lu,
  Entropy 2021; Tahir et al. 2024 (2410.08194); Karakida & Akaho ICLR 2022.
- "Adaptation distorts pretrained features / underperforms OOD": Kumar et al. ICLR 2022 (2202.10054).
- Closest PLM benchmarks (lack the fraction-sweep + threshold): CoPeP continual-pretraining bench;
  Hu et al. JBHI 2024 (transferability metric on protein reps). [several PLM refs UNVERIFIED — check].
- Terminology: "recovery threshold" is unclaimed in this sense (only used in community-detection);
  coinage appears available. **TODO: verify the OpenReview CPT/SFT paper (guUUlHPXRw)** — closest
  potential prior art (U-shape tied to domain similarity), couldn't confirm the quote (bot-walled).

**Citation verification (peer session relay, full-text fetched where noted):**
- CONFIRMED: Ben-David, Blitzer, Crammer, Kulesza, Pereira, Wortman, *A Theory of Learning from
  Different Domains*, Machine Learning 79:151–175, **2010** (journal version; precursors 2006/07).
  Bound: target_err ≤ source_err + ½·d_{HΔH} + λ.
- CONFIRMED: Blitzer et al., *Learning Bounds for Domain Adaptation*, NeurIPS **2007** — optimal
  source/target mixing weight β = f(divergence, sample sizes, hypothesis complexity). Closest prior
  art to "fraction of target data vs divergence."
- CONFIRMED (biological analog, strong): Provost, Yang, Carstens, *Impacts of fine-tuning,
  phylogenetic distance, and sample size on big-data bioacoustics*, PLOS ONE 2022 (DOI
  10.1371/journal.pone.0278522). Accuracy drops with phylogenetic distance (1%→43%; R²=0.30,
  p<3.5e-11); fine-tuning / adding target species RECOVERS accuracy. Directly parallels our
  distance→difficulty + target-data-recovers story, in a non-protein domain — cite as precedent.
- CONFIRMED: Vural & Karaca, arXiv 2507.22632 (2025) — semi-supervised DA; target-loss weight
  α = O(√M_t) ties target-label count to divergence compensation.
- PARTIAL: Wang & Mao, *f-Divergence Principled Domain Adaptation*, NeurIPS 2024 (arXiv 2402.01887)
  — f-DD target-error + sample-complexity bounds (multi-snippet, no direct page fetch).
- CORRECTED: Redko, Habrard, Sebban, *Theoretical Analysis of DA with Optimal Transport*,
  ECML-PKDD 2017 — cite the **HAL record hal-01613564** (arXiv:1610.04420 NOT confirmed). Wasserstein
  DA bound. UNVERIFIED arXiv IDs (real papers, confirm numbers before citing): WDGRL (Shen et al.
  AAAI 2018), Courty JDOT (NeurIPS 2017), Wang & Mao f-DD (NeurIPS 2024).
- **Best single analog to cite: Provost et al. PLOS ONE 2022** — distance→accuracy-loss + data-recovers
  in bioacoustics; "phenomenon exists, just not on PLMs and not as a threshold law."

### 2026-07-14 (late) — What is distance `d`? (and why it isn't Wasserstein)

`d` in the synthetic sweep is a **dimensionless interpolation fraction**, NOT a metric distance:
`center_target(f) = C_f + d·(C_perm(f) − C_f)` — the fraction of the way from family f's own source
centroid toward a *different* family's centroid (fixed derangement). d=0 no shift; d=0.5 midpoint
(on the source decision boundary); d=1 target sits on the confuser family's source location.

**Checked whether feature-wise Wasserstein(source,target) could be the x-axis. It can't, cleanly:**

| d | 0.0 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.85 | 1.0 |
|---|---|---|---|---|---|---|---|---|
| W | 0.08 | 0.53 | 0.62 | 0.66 | 0.62 | 0.53 | 0.31 | 0.09 |

Wasserstein is **non-monotonic** — peaks at d=0.5, returns to ~0 at d=1.0. Reason: at d=1.0 the
target reuses the SAME cluster locations (permuted labels), so P(x) is ~unchanged; feature-Wasserstein
measures P(x) and is blind to the label shift. Our knob is a **P(y|x) / concept shift**, not a
covariate shift — which is also why pure covariate displacement never produced the dip (families drift
into empty space, still classified correctly). Real SwissProt→TrEMBL W = 0.0014 (ESM-C) / 0.008
(ESM-2) — tiny, and a different scale from synthetic; not directly comparable.

**Takeaway:** the right monotone, P(y|x)-aware, synthetic↔real-comparable distance is
**classifier degradation** = (1 − pre-adaptation baseline F1) [synthetic: 1.00→0.98→0.51→0.02→0→0],
not feature-Wasserstein. Worth raising with the professor. TODO if we pursue: re-plot threshold vs
(1 − baseline F1) for both synthetic and real, and/or add a joint (x,y) distance.

### 2026-07-14 (evening) — Standing up the real-data (ESM-C) arm

Goal: do as much as possible today; get the real-data pipeline fully set up so large
experiments can run later. Working through it end-to-end on sapelo2.

- **ESM-C conda env built.** `setup_esmc_env.slurm` (job 46972331, batch node) created
  `/work/ah2lab/LiamK/conda_envs/plm_esmc` with EvolutionaryScale `esm-3.2.1.post1`. Loaded
  `esmc_300m` (333M params) and cached weights to `/scratch/lmk04992/hf_cache`. Separate env
  because EvolutionaryScale `esm` collides with fair-esm (both `import esm`).
- **Caveat found + resolved:** `pip install esm` pulled **torch 2.13.0+cu130** (overriding the
  cu121 pin). GPU smoke test (job 46972412) confirmed `cuda avail: True` on A100-SXM4-80GB and
  `embed_esmc()` produced shape (3, 960) — cu130 works on the cluster driver, no reinstall needed.
- **Real sequences fetched + cached.** SwissProt→TrEMBL, 16 Bacterial Pfam families:
  **20,969 seqs (8,169 swissprot / 12,800 trembl)**, all families present → num_classes=16.
  Cached at `/scratch/lmk04992/swissprot_esmc/raw` (embed jobs reuse it via a fetch-skip guard).
- **Hardened scripts:** persistent `HF_HOME`/`TORCH_HOME` caches on scratch; fetch-skip guard;
  `setup_esmc_env.slurm` wrapper (keeps heavy install off the login node); `embed_smoke.slurm`
  (GPU smoke test for the ESM-C path).

- **Full real-data pipeline launched + chained (SLURM dependencies).**
  - ESM-C embed job `46973912` → recovery sweep `46973939` (afterok) → `/scratch/lmk04992/swissprot_esmc/`
  - ESM-2 embed job `46973913` → recovery sweep `46973940` (afterok) → `/scratch/lmk04992/swissprot_esm2/`
  - Each sweep = `run_real_sweep.slurm`: TRAIN_SOURCE + TEST_ADAPTATION over r∈{0,0.1,0.25,0.5,1.0}
    × seeds {42,43,44} on the precomputed embeddings (reuses train.py/adapt_OGadam.py), then
    plot_recovery.py → `recovery_curves.png` + `recovery_summary.txt`. Both embed jobs read the
    SAME sequences, so it's a clean ESM-2 vs ESM-C comparison. Self-driving; awaiting completion.

**Note on scope:** this real-data run is a *single-distance* recovery curve (SwissProt→TrEMBL) on
real ESM-C embeddings — it validates that the dip/threshold phenomenon appears on real data with the
newer PLM, and gives the ESM-2-vs-ESM-C comparison. The *threshold-vs-distance* curve on real data
would need multiple real distances (e.g. a taxonomic-distance ladder) — that's the next big
experiment, now that all the infrastructure is set up and validated.

**RESULTS (both sweeps COMPLETED, 3 seeds).** final_f1 by OOD fraction r:

| r | ESM-C final F1 | ESM-C recovered | ESM-2 final F1 | ESM-2 recovered |
|---|---|---|---|---|
| 0.00 | 0.879 | 0/3 | 0.853 | 0/3 |
| 0.10 | 0.896 | 2/3 | 0.903 | 3/3 |
| 0.25 | 0.893 | 2/3 | 0.902 | 3/3 |
| 0.50 | 0.929 | 3/3 | 0.911 | 3/3 |
| 1.00 | 0.935 | 3/3 | 0.938 | 3/3 |

- **Baselines (source model on target, batch 0): ESM-C 0.892 > ESM-2 0.863.** The newer PLM's
  source model transfers to TrEMBL better out-of-the-box — a clean model-comparison finding.
- **Recovery threshold ≈ 0.10 for both** (majority of seeds return to baseline). Both recover to
  ~0.94 at r=1.0; r=0 stays below baseline (negative transfer at zero OOD).
- **Honest limitation:** the SwissProt→TrEMBL curation shift is MILD — dip depths (~0.01–0.03) are
  not separated from the in-dist noise floor (~0.04). To get a deeper dip / sharper threshold we
  need a harder shift (taxonomic ladder or MGnify). This is consistent with the synthetic result:
  small distance → small threshold.
- Figures: `/scratch/lmk04992/swissprot_esmc/sweep/recovery_curves.png`,
  `/scratch/lmk04992/swissprot_esm2/sweep/recovery_curves.png`. Summaries: `recovery_summary.txt`
  in each sweep dir.

**All of today's job chain COMPLETED:** setup_esmc (46972331) → esmc_smoke (46972412) →
precompute_esmc (46973912) + precompute_esm2 (46973913) → real_sweep esmc (46973939) +
real_sweep esm2 (46973940). No failures.

### 2026-07-14 (later) — Smoke tests + a useful negative result on synthetic ceilings

**Smoke test: real-data fetch.** Validated the new `--source curation` path against the live
UniProt API (job-free, on login node). `src/fetch_sequences.py --source curation --taxid 2`
returns clean `swissprot` (reviewed) and `trembl` (unreviewed) buckets, correctly tagged, distinct
accessions. The only untested external code path in the real-data arm now works. Output:
`/scratch/lmk04992/curation_smoke/`.

**Calibration: is the synthetic ceiling a knob?** Yes. Crowding the latent manifold
(latent_dim 32→8, tighter spacing) drops the target-only ceiling from 1.0 to ~0.73. Confirmed the
ceiling is tunable.

**Negative result: crowding backfires for THIS experiment.** Ran a full crowded-manifold sweep
(job `46972312`, latent_dim 8, within_sigma 2.0, spread 2.8, n_source 6000). The ceiling came out
**U-shaped in distance** (0.79 → 0.49 at d=0.5 → 0.77), because sliding a target family halfway
toward its confuser lands it in an intrinsically ambiguous midpoint region that even a target-only
model can't separate. So the distance knob contaminates the ceiling, "recover to 90% of ceiling"
becomes a moving target, and r*=nan beyond d=0.5. **Conclusion:** the clean-ceiling run (latent 32,
ceiling ≡ 1.0 at every distance) is the *correct controlled design* — displacement changes only the
source-model wrongness and pool composition. Realistic headroom should come from the **real-data
(ESM-C) arm**, not from degrading the synthetic control. Result kept for the record at
`/scratch/lmk04992/synth_distance_sweep_hardceil/` (`run_distance_sweep_hardceil.slurm`).

**Next decision (open).** Build `plm_esmc` env and run the real-data companion:
SwissProt→TrEMBL (runnable now) vs MGnify (needs GCP creds). Awaiting go-ahead.

### 2026-07-14 — Fixed the synthetic generator + ran recovery-threshold-vs-distance

**Goal.** Reconcile old notes with the June-17 state; the clarified aim is to see how the
recovery threshold changes with distance from the training data, and to fix the old
(dip-incapable) synthetic data generation.

**Environment check.** Confirmed jobs can run on sapelo2 via the open SSH ControlMaster
socket (submit node `ss-sub2`, user `lmk04992`); `gpu_p` A100 partition available.

**Fixed the data generation.** The old `src/generate_simulation.py` could never show the
dip for two structural reasons: (1) one frozen oracle labeled both source and target →
globally consistent labels; (2) the shift knob only *tightened* the source variance around
the same centroids → source was a nested subset of the target. New generator
`src/generate_synthetic_precomputed.py`:
- labels by **family** (no oracle) → labeling function can differ across domains;
- **displaces** each target family a fraction `distance` toward another family's source
  region (fixed derangement) → source-optimal boundary becomes genuinely wrong;
- builds families on a **low-rank latent manifold** (latent_dim 32 → 1280) for realistic
  headroom (raw 1280-D Gaussians are trivially separable);
- emits the SAME `.npy` interface as `precompute_real_embeddings.py`, so synthetic and real
  data now feed one downstream pipeline.

**Built the sweep.** `src/run_distance_sweep.py` + `run_distance_sweep.slurm` — trains a
source model, adapts on the pool, and records the target-F1 trajectory across the full
(distance × OOD-fraction × seed) grid. "Recovery" is measured vs the **target-only ceiling**
(not vs the pre-adaptation start, which is trivially satisfied at large distance).

**Ran it.** Job `46964648`, ~3 min on one A100. Grid: distances {0,0.3,0.4,0.5,0.6,0.7,0.85,1.0}
× OOD frac {0,0.05,0.1,0.2,0.3,0.5,0.75,1.0} × seeds {42,43,44}.

**Result — recovery threshold r*(d) rises monotonically with distance:**

| distance d | pre-adapt baseline F1 | recovery threshold r* |
|---|---|---|
| 0.0–0.3 | ~1.00 | 0.00 |
| 0.40 | 0.98 | 0.05 |
| 0.50 | 0.51 | 0.10 |
| 0.60 | 0.02 | 0.30 |
| 0.70 | 0.00 | 0.75 |
| 0.85–1.0 | 0.00 | 1.00 |

Flat/near-zero while the target stays in the source's decision regions; steep knee at
d ≈ 0.5–0.7; saturates (needs pure-target pool) beyond d ≈ 0.85. At r=1.0 every distance
recovers to F1 ≈ 1.0 → the gap is negative transfer from the source composition of the
pool, not a capacity limit.

**Artifacts.**
- Figures + data: `/scratch/lmk04992/synth_distance_sweep/` (`threshold_vs_distance.png`,
  `heatmap_final_f1.png`, `curves_by_distance.png`, `sweep_results.csv`, `threshold_vs_distance.csv`)
- Writeup: `RESULTS_distance_sweep.md`
- Professor update (text draft): `professor_update.md`

**Also staged (not yet run).** Real-data validation arm — ESM-C embeddings on a
SwissProt→distance shift. Code ready: ESM-C support in `src/precompute_real_embeddings.py`
(auto-dispatch on model name), `--source curation` (SwissProt reviewed → TrEMBL unreviewed)
in `src/fetch_sequences.py`, `setup_esmc_env.sh`, `precompute_swissprot_esmc.slurm`,
`precompute_swissprot_esm2.slurm`, `configs/swissprot_shift.yaml`. Open decision:
SwissProt→TrEMBL (runnable now) vs MGnify metagenomic (needs GCP credentials).

**Open next steps.**
- Build `plm_esmc` env + launch the SwissProt ESM-C run (real-data companion figure).
- Non-trivial ceilings: raise `within_sigma` so ceilings drop below 1, test robustness of the law.
- More seeds around the knee (d = 0.5–0.7).

<!-- Add new dated entries ABOVE this line, newest first. -->
