#!/usr/bin/env python
"""
Turn real protein sequences into the EXACT .npy artifacts the existing pipeline
(src/train.py, src/adapt*.py via main_precomputed.nf) consumes -- but with real
ESM-2 embeddings instead of synthetic GMM samples.

Steps
-----
1. Read seqs.fasta + metadata.tsv (id, family, group) from fetch_sequences.py.
2. Embed every sequence with ESM-2 (esm2_t33_650M_UR50D), mean-pooled over
   residues -> (N, 1280) float32. 1280 matches the synthetic `dim`, so the
   downstream model is byte-for-byte unchanged.
3. family -> contiguous integer label y (shared label space across groups).
4. For each --ood_frac r, build the source/target/pool/test split that
   operationalizes the recovery-threshold experiment (see HOW_TO_SEE_THE_DIP.md):
       source_X/source_y : SOURCE group only           (baseline training set)
       pool_X/pool_y     : r fraction TARGET, (1-r) SOURCE   (adaptation pool)
       test_X/test_y     : TARGET group only, held out  (the manifold to realign to)
   ref_x for the Wasserstein metric is just source_X (the pipeline passes it).

Output filenames (per r) match what main_precomputed.nf globs:
    source_X_Shf{r}.npy  source_y_Shf{r}.npy
    pool_X_Shf{r}.npy    pool_y_Shf{r}.npy
    test_X_Shf{r}.npy    test_y_Shf{r}.npy
plus dataset_info.json (num_classes, family map, counts).
"""
import argparse
import json
import os

import numpy as np
import torch


def read_inputs(fasta, meta):
    seqs = {}
    cur = None
    with open(fasta) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                cur = line[1:].split()[0]
                seqs[cur] = ""
            elif cur is not None:
                seqs[cur] += line
    ids, families, groups = [], [], []
    with open(meta) as f:
        header = f.readline()  # id\tfamily\tgroup
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] not in seqs:
                continue
            ids.append(parts[0]); families.append(parts[1]); groups.append(parts[2])
    return ids, [seqs[i] for i in ids], families, groups


def embed(sequences, ids, model_name, max_len, batch_size, device):
    """Dispatch on model family. Both backends return (N, D) mean-pooled float32.

    ESM-2  (fair-esm, `import esm` -> esm.pretrained.*): D = 1280 for the 650M model.
    ESM-C  (EvolutionaryScale SDK, `from esm.models.esmc import ESMC`): D = 960 for
           esmc_300m, 1152 for esmc_600m. NOTE the two `esm` packages COLLIDE on the
           import name, so ESM-C must run in its own conda env (see the esmc SLURM
           script). Downstream is dim-agnostic (train.py infers input_dim), so the
           only thing that changes when you switch model is D.
    """
    if model_name.startswith("esmc") or model_name.startswith("esm3"):
        return embed_esmc(sequences, ids, model_name, max_len, batch_size, device)
    return embed_esm2(sequences, ids, model_name, max_len, batch_size, device)


@torch.no_grad()
def embed_esm2(sequences, ids, model_name, max_len, batch_size, device):
    import esm
    print(f"Loading ESM-2 {model_name} (fair-esm) ...", flush=True)
    model, alphabet = getattr(esm.pretrained, model_name)()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    repr_layer = model.num_layers  # last layer (33 for the 650M model)

    # truncate over-long sequences (ESM-2 positional limit ~1022 residues)
    data = [(i, s[:max_len]) for i, s in zip(ids, sequences)]
    out = np.zeros((len(data), model.embed_dim), dtype=np.float32)

    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        _, _, toks = bc(batch)
        toks = toks.to(device)
        reps = model(toks, repr_layers=[repr_layer])["representations"][repr_layer]
        # mean-pool over real residues only (drop BOS, EOS, padding)
        for j, (_, seq) in enumerate(batch):
            L = len(seq)
            out[start + j] = reps[j, 1:L + 1].mean(0).float().cpu().numpy()
        print(f"  embedded {min(start + batch_size, len(data))}/{len(data)}", flush=True)
    return out


@torch.no_grad()
def embed_esmc(sequences, ids, model_name, max_len, batch_size, device):
    """ESM-C embeddings via the EvolutionaryScale SDK (`pip install esm`, v3.x).

    The SDK exposes one protein at a time; `client.logits(..., return_embeddings=True)`
    returns a (1, L+2, D) tensor with a BOS/EOS pad on each end, so we mean-pool over
    [1:L+1] exactly like the ESM-2 path. batch_size is accepted for signature parity
    but the SDK is per-sequence; we just loop (fine for the few-thousand-seq subsets).
    """
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    # Guard: ESM-C on CPU is ~30x slower. A silent CPU fallback (e.g. torch CUDA
    # version newer than the node's driver) once turned a 15-min job into 10 hours.
    # Fail loudly instead, unless CPU is explicitly requested.
    if str(device) == "cpu" or not torch.cuda.is_available():
        raise SystemExit(
            "ESM-C embedding wants a working CUDA GPU but torch.cuda.is_available() is False "
            "(likely a torch-CUDA / node-driver mismatch -- see the CUDA UserWarning above). "
            "Refusing to embed on CPU (~30x slower). Fix the torch build for this node's driver, "
            "or exclude old-driver nodes.")

    print(f"Loading ESM-C {model_name} (EvolutionaryScale SDK) ...", flush=True)
    client = ESMC.from_pretrained(model_name).to(device).eval()
    cfg = LogitsConfig(sequence=True, return_embeddings=True)

    data = [(i, s[:max_len]) for i, s in zip(ids, sequences)]
    out = None
    for k, (_, seq) in enumerate(data):
        protein = ESMProtein(sequence=seq)
        pt = client.encode(protein)
        emb = client.logits(pt, cfg).embeddings  # (1, L+2, D)
        L = len(seq)
        vec = emb[0, 1:L + 1].mean(0).float().cpu().numpy()
        if out is None:
            out = np.zeros((len(data), vec.shape[0]), dtype=np.float32)
        out[k] = vec
        if (k + 1) % 100 == 0 or k + 1 == len(data):
            print(f"  embedded {k + 1}/{len(data)}", flush=True)
    return out


def save_set(outdir, tag, X, y):
    np.save(os.path.join(outdir, f"{tag}_X.npy"), X.astype(np.float32))
    np.save(os.path.join(outdir, f"{tag}_y.npy"), y.astype(np.int64))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--source_group", required=True, help="group value used as SOURCE")
    ap.add_argument("--target_group", required=True, help="group value used as TARGET")
    ap.add_argument("--ood_fracs", default="0.0,0.1,0.25,0.5,1.0",
                    help="Comma list of target fractions r in the adaptation pool (the sweep).")
    ap.add_argument("--pool_size", type=int, default=4000)
    ap.add_argument("--n_test", type=int, default=1000, help="held-out pure-target test size")
    ap.add_argument("--emb_cache", default=None,
                    help="Path to a .npy cache of ALL-sequence embeddings. If it exists, load it "
                         "and skip embedding; else embed and save it. Lets several target splits "
                         "(a taxonomic ladder) reuse ONE embedding pass over the same fasta.")
    ap.add_argument("--esm_model", default="esm2_t33_650M_UR50D",
                    help="ESM-2 (fair-esm) e.g. esm2_t33_650M_UR50D (1280-D), OR "
                         "ESM-C (EvolutionaryScale SDK) e.g. esmc_300m (960-D), esmc_600m "
                         "(1152-D). Backend auto-selected by name prefix ('esmc'->ESM-C). "
                         "ESM-C needs its own conda env; see precompute_swissprot_esmc.slurm.")
    ap.add_argument("--max_len", type=int, default=1022)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ids, sequences, families, groups = read_inputs(args.fasta, args.meta)
    print(f"Read {len(ids)} sequences "
          f"({sum(g == args.source_group for g in groups)} source / "
          f"{sum(g == args.target_group for g in groups)} target)")

    # shared label space: family -> int
    fam_list = sorted(set(families))
    fam2id = {f: i for i, f in enumerate(fam_list)}
    y_all = np.array([fam2id[f] for f in families], dtype=np.int64)
    groups = np.array(groups)

    if args.emb_cache and os.path.exists(args.emb_cache):
        X_all = np.load(args.emb_cache)
        if X_all.shape[0] != len(ids):
            raise SystemExit(f"emb_cache has {X_all.shape[0]} rows but fasta has {len(ids)} seqs "
                             f"-- stale cache? delete {args.emb_cache}")
        print(f"Loaded cached embeddings {X_all.shape} from {args.emb_cache} (skipped embedding)")
    else:
        X_all = embed(sequences, ids, args.esm_model, args.max_len, args.batch_size, args.device)
        if args.emb_cache:
            np.save(args.emb_cache, X_all.astype(np.float32))
            print(f"Cached embeddings -> {args.emb_cache}")

    src_idx = np.where(groups == args.source_group)[0]
    tgt_idx = np.where(groups == args.target_group)[0]
    if len(src_idx) == 0 or len(tgt_idx) == 0:
        raise SystemExit("source_group or target_group has no sequences -- check --groups names.")
    rng.shuffle(src_idx); rng.shuffle(tgt_idx)

    # hold out pure-target test; reserve a source pile for the pool's in-dist part
    n_test = min(args.n_test, len(tgt_idx) // 2)
    test_idx = tgt_idx[:n_test]
    tgt_pool_avail = tgt_idx[n_test:]

    # SOURCE training set = first 80% of source; remaining 20% feeds pool's (1-r) part
    n_src_train = int(0.8 * len(src_idx))
    src_train_idx = src_idx[:n_src_train]
    src_pool_avail = src_idx[n_src_train:]

    source_X, source_y = X_all[src_train_idx], y_all[src_train_idx]
    test_X, test_y = X_all[test_idx], y_all[test_idx]
    print(f"num_classes={len(fam_list)} | source_train={len(source_X)} | test={len(test_X)}")

    info = {
        "num_classes": len(fam_list),
        "family_map": fam2id,
        "dim": int(X_all.shape[1]),
        "source_group": args.source_group,
        "target_group": args.target_group,
        "counts": {"source_train": int(len(source_X)), "test": int(len(test_X))},
        "ood_fracs": [], "esm_model": args.esm_model,
    }

    for r in [float(x) for x in args.ood_fracs.split(",")]:
        n_tgt = int(round(r * args.pool_size))
        n_src = args.pool_size - n_tgt
        n_tgt = min(n_tgt, len(tgt_pool_avail))
        n_src = min(n_src, len(src_pool_avail))
        tgt_part = rng.choice(tgt_pool_avail, size=n_tgt, replace=False) if n_tgt else np.array([], int)
        src_part = rng.choice(src_pool_avail, size=n_src, replace=False) if n_src else np.array([], int)
        pool_idx = np.concatenate([tgt_part, src_part]).astype(int)
        rng.shuffle(pool_idx)

        tag = f"Shf{r}"
        save_set(args.outdir, f"source_{tag}", source_X, source_y)
        save_set(args.outdir, f"pool_{tag}", X_all[pool_idx], y_all[pool_idx])
        save_set(args.outdir, f"test_{tag}", test_X, test_y)
        print(f"r={r}: pool={len(pool_idx)} ({n_tgt} target + {n_src} source)")
        info["ood_fracs"].append({"r": r, "pool_target": n_tgt, "pool_source": n_src})

    with open(os.path.join(args.outdir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nDone. Set num_classes={len(fam_list)} in your config. Files in {args.outdir}")
