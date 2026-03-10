import numpy as np
import argparse
import os
from scipy.spatial.distance import cosine

def load_data(data_dir, n_train, seed):
    x_path = os.path.join(data_dir, f"source/source_X_{n_train}_s{seed}.npy")
    y_path = os.path.join(data_dir, f"source/source_y_{n_train}_s{seed}.npy")
    return np.load(x_path), np.load(y_path)

def get_conserved_motifs(X_family, variance_threshold=1e-5):
    """
    Calculates the variance of each dimension across all sequences in a family.
    Returns the indices of the dimensions that are perfectly conserved (variance near 0).
    """
    variances = np.var(X_family, axis=0)
    conserved_indices = np.where(variances < variance_threshold)[0]
    return conserved_indices

def quantify_relationships(X, y, anchor_fam, close_fam, far_fam):
    """
    Calculates the statistical similarities and motif overlaps between families.
    """
    dim = X.shape[1]
    
    # 1. Extract data for the three families
    X_anchor = X[y == anchor_fam]
    X_close = X[y == close_fam]
    X_far = X[y == far_fam]
    
    # 2. Verify Motif Conservation (The Zero-Variance Test)
    motifs_anchor = get_conserved_motifs(X_anchor)
    motifs_close = get_conserved_motifs(X_close)
    motifs_far = get_conserved_motifs(X_far)
    
    print("\n" + "="*60)
    print(f"1. INTRA-FAMILY MOTIF VERIFICATION (Expected ~30% of {dim} = {int(dim*0.3)})")
    print("="*60)
    print(f"Family {anchor_fam:<2} has {len(motifs_anchor)} perfectly conserved dimensions.")
    print(f"Family {close_fam:<2} has {len(motifs_close)} perfectly conserved dimensions.")
    print(f"Family {far_fam:<2} has {len(motifs_far)} perfectly conserved dimensions.")
    
    # 3. Centroid Cosine Similarity
    centroid_anchor = np.mean(X_anchor, axis=0)
    centroid_close = np.mean(X_close, axis=0)
    centroid_far = np.mean(X_far, axis=0)
    
    sim_close = 1 - cosine(centroid_anchor, centroid_close)
    sim_far = 1 - cosine(centroid_anchor, centroid_far)
    
    print("\n" + "="*60)
    print(f"2. INTER-FAMILY COSINE SIMILARITY (Continuous Space)")
    print("="*60)
    print(f"Similarity (Fam {anchor_fam} vs Close Fam {close_fam}): {sim_close:.4f}  <-- Should be higher")
    print(f"Similarity (Fam {anchor_fam} vs Far Fam {far_fam}): {sim_far:.4f}  <-- Should be lower")

    # 4. Motif Intersection (The Evolutionary Link)
    shared_with_close = np.intersect1d(motifs_anchor, motifs_close)
    shared_with_far = np.intersect1d(motifs_anchor, motifs_far)
    
    # Calculate Jaccard Similarity (Intersection over Union)
    iou_close = len(shared_with_close) / len(np.union1d(motifs_anchor, motifs_close))
    iou_far = len(shared_with_far) / len(np.union1d(motifs_anchor, motifs_far))

    print("\n" + "="*60)
    print(f"3. EVOLUTIONARY MOTIF OVERLAP (Jaccard Similarity of Indices)")
    print("="*60)
    print(f"Shared Motifs (Fam {anchor_fam} & Fam {close_fam}): {len(shared_with_close)} indices (IoU: {iou_close:.4f})")
    print(f"Shared Motifs (Fam {anchor_fam} & Fam {far_fam}): {len(shared_with_far)} indices (IoU: {iou_far:.4f})")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="results/phylogenetic_gmm/data")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--anchor", type=int, default=1)
    parser.add_argument("--close", type=int, default=7)
    parser.add_argument("--far", type=int, default=18)
    args = parser.parse_args()

    X, y = load_data(args.data_dir, args.n_train, args.seed)
    
    quantify_relationships(X, y, args.anchor, args.close, args.far)
