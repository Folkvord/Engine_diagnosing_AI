import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#  Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utilities
from util.read_wav import get_formated_wav
from util.preprocessing import preprocess_data

# Main enté
def main():
    print("[DBSCAN ENGINE CLASSIFIER] Starting unsupervised clustering...")

    # --- 1. laster data ---
    print("\n[STEP 1] Loading .wav files ...")
    data = get_formated_wav(data_type="all", is_train=True)
    print(f"Loaded {len(data)} wav files.")

    #2. Preprocess data (denoise, filter, extract features, etc.)
    print("\n[STEP 2] Preprocessing audio data ...")
    features = preprocess_data(data)
    print(f"Feature matrix shape: {features.shape}")

    # 3. Scalerer funksjoner
    print("\n[STEP 3] Scaling features ...")
    print("Type of features:", type(features))
    print("Features shape:", np.shape(features))
    print("Example feature (if any):", features[0] if len(features) > 0 else "EMPTY")
    X_scaled = StandardScaler().fit_transform(features)

    # 4. Kjør DBSCAN 
    print("\n[STEP 4] Running DBSCAN clustering ...")
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    labels = dbscan.fit_predict(X_scaled)

    # 5. Analyserer clusters
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("\n[STEP 5] Clustering results:")
    print(f"Estimated clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")

    #6. Visuliserer clusters (må ikke ha)
    if X_scaled.shape[1] >= 2:
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', s=20)
        plt.title("DBSCAN clustering of engine sounds")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    #7. Lagrer resultat
    output_dir = os.path.join("unsupervised_model", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "dbscan_labels.npy"), labels)
    print(f"Saved cluster labels to: {os.path.join(output_dir, 'dbscan_labels.npy')}")

    print("\n[DBSCAN ENGINE CLASSIFIER] Done ✅")


if __name__ == "__main__":
    main()