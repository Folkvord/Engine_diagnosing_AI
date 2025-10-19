import os
import numpy as np
import matplotlib.pyplot as plt

# Optional:
FEATURES_FILE = os.path.join("unsupervised_model", "outputs", "features.npy")
LABELS_FILE = os.path.join("unsupervised_model", "outputs", "dbscan_labels.npy")

# Map cluster IDs
CLUSTER_MEANINGS = {
    0: "Good Engine",
    1: "Broken Engine",
    2: "Heavy Load"
}

def main():
    print("[DBSCAN FACT CHECKER] Starting analysis...\n")

    #1. Laster cluster labels
    if not os.path.exists(LABELS_FILE):
        print(f"Could not find {LABELS_FILE}")
        return

    labels = np.load(LABELS_FILE)
    print(f"Loaded {len(labels)} cluster labels from: {LABELS_FILE}\n")

    #2. Cluster statistikk
    unique, counts = np.unique(labels, return_counts=True)
    print("Cluster distribution:")
    for u, c in zip(unique, counts):
        if u == -1:
            print(f"  • Noise points (-1): {c}")
        else:
            meaning = CLUSTER_MEANINGS.get(u, "Unknown")
            print(f"  • Cluster {u} ({meaning}): {c} samples")

    n_clusters = len(unique) - (1 if -1 in unique else 0)
    n_noise = (labels == -1).sum()

    print("\nSummary:")
    print(f"  → Total clusters found: {n_clusters}")
    print(f"  → Noise points: {n_noise}")
    print(f"  → Total samples: {len(labels)}")

    # 3. Optional: Cluster tolkning
    print("\nCluster interpretation:")
    for cluster_id, meaning in CLUSTER_MEANINGS.items():
        if cluster_id in labels:
            count = list(labels).count(cluster_id)
            print(f"  Cluster {cluster_id}: {meaning} ({count} samples)")
        else:
            print(f"  Cluster {cluster_id}: {meaning} (not detected in this run)")

    # 4. Optional visualisering
    if os.path.exists(FEATURES_FILE):
        try:
            features = np.load(FEATURES_FILE)
            if features.ndim == 2 and features.shape[1] >= 2:
                print("\nVisualizing clusters...")
                plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis', s=20)
                plt.title("DBSCAN Clusters (Fact Check Visualization)")
                plt.xlabel("Feature 1")
                plt.ylabel("Feature 2")
                plt.show()
            else:
                print("Features file found but not 2D, skipping visualization.")
        except Exception as e:
            print(f"Could not load features for visualization: {e}")
    else:
        print("\n(No features file found — skipping visualization.)")

    print("\n[DBSCAN FACT CHECKER] Done!")


if __name__ == "__main__":
    main()