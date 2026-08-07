# Next steps — decoded from the handwritten notes (July 14)

This reconciles the terse notes ("magnify… precompute swissprot… esm-c instead of
esm2…") with the code that already exists, and gives the exact commands to run.
Context on the *previous* milestone is in `june17_premeeting.md` (the dip + the
~0.5 recovery threshold, on UniProt ESM-2, Bacteria→Archaea).

## What the notes translate to

| Note (as written) | Meaning | Status |
|---|---|---|
| "software can do either data gen (synthetic vs real)" | one framework, synthetic **or** real | done: `main.nf` (synthetic) + `main_precomputed.nf` (real) |
| "esm-c … whichever is newer, instead of esm2" | swap ESM-2 → **ESM-C** (EvolutionaryScale, Dec 2024) | **built this pass** — `--esm_model esmc_300m` |
| "precompute swissprot", "source looks like swissprot, target is a distance from swissprot" | new shift axis: **SwissProt (curated) = source → distance = target** | **built this pass** — `--source curation` |
| "magnify" (= MGnify) | extreme metagenomic distance as the target | available (`--source mgnify`), **needs GCP BigQuery creds** |
| "redownload swissprot, do esm-2 embeddings" | ESM-2 baseline on the same SwissProt data | **built this pass** — `precompute_swissprot_esm2.slurm` |

## The design (this pass)

Shift axis = **curation status**, holding taxon fixed (Bacteria, taxid 2), so the
only thing that changes between source and target is how the protein was annotated:

- **source = SwissProt** (`reviewed:true`) — curated, the model trains here
- **target = "distance from SwissProt"** (`reviewed:false`, TrEMBL) — computationally
  annotated; same Pfam families, different population statistics
- test set = pure target; sweep `r` = fraction of the adaptation pool that is target

This is fetchable **today** via the UniProt REST API (no GCP). MGnify metagenomic
proteins are the *more extreme* distance and remain the follow-up once BigQuery
credentials are sorted.

Embeddings swap ESM-2 (1280-D) → **ESM-C esmc_300m (960-D)**. Downstream is
dim-agnostic (`train.py` infers `input_dim` from the `.npy` width, `model.py` is
dynamically sized), so **nothing downstream changed** — only the embed step.

## Run order (from a sapelo2 login node, in `work/tidythesis`)

```bash
# 0. ONE TIME: create the ESM-C conda env (separate because EvolutionaryScale's
#    `esm` and fair-esm collide on the import name).
bash setup_esmc_env.sh

# 1. Fetch SwissProt+TrEMBL and embed with ESM-C (fetches the shared raw sequences).
sbatch slurm/precompute_swissprot_esmc.slurm

# 2. AFTER (1) finishes: ESM-2 baseline on the SAME sequences (reuses (1)'s raw fetch).
sbatch slurm/precompute_swissprot_esm2.slurm

# 3. CHECK num_classes actually returned, then edit configs/swissprot_shift.yaml if needed:
cat /scratch/lmk04992/swissprot_esmc/embeddings/dataset_info.json   # -> num_classes

# 4. Run the recovery-threshold sweep (ESM-C):
nextflow -log nextflow_swissprot_esmc.log run main_precomputed.nf \
    -profile sapelo2 -params-file configs/swissprot_shift.yaml -work-dir work_swissprot_esmc

# 5. ESM-2 baseline sweep: copy the config, point precomputed_dir at
#    /scratch/lmk04992/swissprot_esm2/embeddings, dataset: 'swissprot_shift_esm2', rerun (4).

# 6. Plot (both), same as June 17:
python src/plot_recovery.py   # (point it at each results/.../adapt dir)
```

## What this buys the thesis

1. A **second, independent shift axis** (curation, not taxonomy) — tests whether the
   dip + ~0.5 threshold generalize beyond Bacteria→Archaea.
2. An **ESM-2 vs ESM-C** comparison on identical sequences — does a newer/better PLM
   change the geometry of negative transfer?
3. Groundwork for the **MGnify** extreme-distance run (the deepest dip we'd expect).

## Open items / decisions for the professor

- Is the **curation (SwissProt→TrEMBL)** axis an acceptable "distance", or do we want
  to wait for **MGnify metagenomic** (needs GCP creds set up on sapelo2)?
- ESM-C model size: `esmc_300m` (960-D, cheap) vs `esmc_600m` (1152-D) — start small.
- More seeds (June 17 used 3; the intervals were wide). Bump to ~10 for the final figure?
