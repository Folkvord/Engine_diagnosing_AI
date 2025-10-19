import os
import numpy as np
import matplotlib.pyplot as plt

# Optional: only if you also saved features
FEATURES_FILE = os.path.join("unsupervised_model", "outputs", "features.npy")
LABELS_FILE = os.path.join("unsupervised_model", "outputs", "dbscan_labels.npy")

def main():
    print("[DBSCAN FACT CHECKER]  Starting analysis...\n")

    # --- 1. Load cluster labels ---
    if not os.path.exists(LABELS_FILE):
        print(f" Could not find {LABELS_FILE}")
        return

    labels = np.load(LABELS_FILE)
    print(f" Loaded {len(labels)} cluster labels from: {LABELS_FILE}\n")

    # --- 2. Cluster statistics ---
    unique, counts = np.unique(labels, return_counts=True)
    print(" Cluster distribution:")
    for u, c in zip(unique, counts):
        if u == -1:
            print(f"   • Noise points (-1): {c}")
        else:
            print(f"   • Cluster {u}: {c} samples")

    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = (labels == -1).sum()

    print("\n Summary:")
    print(f"   → Total clusters found: {n_clusters}")
    print(f"   → Noise points: {n_noise}")
    print(f"   → Total samples: {len(labels)}")

    # --- 3. Optional visualization ---
    if os.path.exists(FEATURES_FILE):
        try:
            features = np.load(FEATURES_FILE)
            if features.ndim == 2 and features.shape[1] >= 2:
                print("\n Visualizing clusters...")
                plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=20)
                plt.title("DBSCAN Clusters (Fact Check Visualization)")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.show()
            else:
                print(" Features file found but not 2D, skipping visualization.")
        except Exception as e:
            print(f" Could not load features for visualization: {e}")
    else:
        print("\n(No features file found — skipping visualization.)")

    print("\n [DBSCAN FACT CHECKER] Done!")

if __name__ == "__main__":
    main()