import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

print("Loading generated embeddings...")
X = np.load("source_train_X.npy")
y = np.load("source_train_y.npy")

print("\n--- LANDSCAPE VERIFICATION ---")
print(f"X shape (Embeddings): {X.shape} (Expected: 1000000, 1280)")
print(f"y shape (Labels):     {y.shape} (Expected: 1000000,)")
print(f"Total Unique Classes: {len(np.unique(y))} (Expected: 1000)")

# Check the long-tailed class distribution
class_counts = np.bincount(y)
max_class = np.max(class_counts)
min_class = np.min(class_counts[class_counts > 0])
print(f"Largest Class Size:   {max_class} sequences")
print(f"Smallest Class Size:  {min_class} sequences")

print("\nRunning PCA on a 10,000 sequence subsample for visualization...")
# Subsample to avoid crashing memory during PCA
idx = np.random.choice(X.shape[0], 10000, replace=False)
X_sub = X[idx]
y_sub = y[idx]

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_sub)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    X_pca[:, 0], X_pca[:, 1], c=y_sub, cmap="tab20", alpha=0.6, s=10
)
plt.title("2D PCA of Synthetic Protein Landscape")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.colorbar(scatter, label="Class Label")
plt.savefig("landscape_pca.png", dpi=300)
print("Saved 2D visualization to 'landscape_pca.png'!")
