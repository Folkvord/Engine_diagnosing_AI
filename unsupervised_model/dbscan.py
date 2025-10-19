import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#utilities
from util.read_wav import get_formated_wav
from util.preprocessing import preprocess_data


def main():
    print("[DBSCAN ENGINE CLASSIFIER] Starting unsupervised clustering...")

    # 1. Laster data 
    print("\n[STEP 1] Loading .wav files ...")
    data = get_formated_wav(data_type="all", is_train=True)
    print(f"Loaded {len(data)} wav files.")

    #2. Preprocess data
    print("\n[STEP 2] Preprocessing audio data ...")
    features = preprocess_data(data, "unsupervised")

    if features is None or len(features) == 0:
        print("[ERROR] No features returned from preprocessing! Check audio files or preprocessing pipeline.")
        return

    features = np.array(features)
    if features.ndim == 1:
        features = features.reshape(-1, 1)

    print(f"Feature matrix shape: {features.shape}")

    #3. Skalerer funksjoner
    print("\n[STEP 3] Scaling features ...")
    X_scaled = StandardScaler().fit_transform(features)

    if np.isnan(X_scaled).any():
        print("[WARNING] NaN detected after scaling — replacing with zeros.")
        X_scaled = np.nan_to_num(X_scaled)

    print(f"Scaled features shape: {X_scaled.shape}")

    #4. kjører DBSCAN
    print("\n[STEP 4] Running DBSCAN clustering ...")
    dbscan = DBSCAN(eps=3.0, min_samples=3, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    # 5. Analyse av clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("\n[STEP 5] Clustering results:")
    print(f"Estimated clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    #5.5. cluster betydelser
    cluster_meanings = {
        0: "Good Engine",
        1: "Broken Engine",
        2: "Heavy Load"
    }

    print("\n[STEP 5.5] Cluster interpretation:")
    for cluster_id, meaning in cluster_meanings.items():
        if cluster_id in labels:
            count = list(labels).count(cluster_id)
            print(f"  Cluster {cluster_id}: {meaning} ({count} samples)")
        else:
            print(f"  Cluster {cluster_id}: {meaning} (not detected in this run)")

    #6. Visualiserer
    if X_scaled.shape[1] >= 2:
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=20)
        plt.title("DBSCAN clustering of engine sounds")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()
    else:
        print("[INFO] Not enough dimensions to plot (need at least 2 features).")

    #7. Lagrer resultater
    output_dir = os.path.join("unsupervised_model", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "dbscan_labels.npy"), labels)
    print(f"Saved cluster labels to: {os.path.join(output_dir, 'dbscan_labels.npy')}")

    print("\n[DBSCAN ENGINE CLASSIFIER] Done.")


if __name__ == "__main__":
    main()