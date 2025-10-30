import sys
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    pairwise_distances_argmin_min,
)
import matplotlib.pyplot as plt
import random

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.read_wav import get_formated_wav
from util.preprocessing import preprocess_data


def get_wav_from_folder(folder_path):
    combined = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            if f.endswith(".wav"):
                path = os.path.join(root, f)
                folder_name = os.path.basename(root)
                combined.append((None, None, path, folder_name))
    return combined


def plot_points(points, color, label=None):
    if len(points) == 0:
        return
    if points.shape[1] >= 2:
        plt.scatter(points[:, 0], points[:, 1], color=color, s=20, label=label, alpha=0.6)


def main():
    print("[DBSCAN ENGINE CLASSIFIER] Starting clustering...")

    # Step 1: Load training data
    print("Loading training data...")
    train_data = get_formated_wav("all", is_train=True)
    X_train = preprocess_data(train_data, "unsupervised")
    X_train = np.array(X_train)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    print(f"Training feature matrix shape: {X_train.shape}")

    # Step 2: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 3: DBSCAN clustering
    print("Running DBSCAN on training data...")
    dbscan = DBSCAN(eps=2.0, min_samples=3, n_jobs=-1)
    labels_train = dbscan.fit_predict(X_train_scaled)
    core_samples = X_train_scaled[dbscan.core_sample_indices_]
    core_labels = labels_train[dbscan.core_sample_indices_]
    n_clusters = len(set(labels_train)) - (1 if -1 in labels_train else 0)
    print(f"Found {n_clusters} clusters in training data")

    # Step 4: Load test data
    print("Loading test data...")
    test_dir = "data/test_cut"
    test_data_paths = get_wav_from_folder(test_dir)
    test_data = get_formated_wav("all", is_train=False)
    X_test = preprocess_data(test_data, "unsupervised")
    X_test = np.array(X_test)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)
    X_test_scaled = scaler.transform(X_test)

    # Step 5: Assign test samples to nearest DBSCAN cluster
    nearest, _ = pairwise_distances_argmin_min(X_test_scaled, core_samples)
    test_labels = core_labels[nearest]

    # Step 6: Auto-map clusters to folder names
    folder_names = [os.path.basename(os.path.dirname(path)) for _, _, path, _ in test_data_paths]
    cluster_mapping = {}
    for cluster_id in set(test_labels):
        indices = np.where(test_labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        folders_in_cluster = [folder_names[i] for i in indices]
        majority_label = max(set(folders_in_cluster), key=folders_in_cluster.count)
        cluster_mapping[cluster_id] = majority_label

    # Step 7: Print cluster results
    print("\nTest cluster distribution and assigned engine type:")
    for cluster_id in sorted(set(test_labels)):
        count = np.sum(test_labels == cluster_id)
        meaning = cluster_mapping.get(cluster_id, "Unknown")
        print(f"Cluster {cluster_id} ({meaning}): {count} samples")

    # Step 8: Random example per cluster
    print("\nRandom example per cluster:")
    for cluster_id in sorted(set(test_labels)):
        indices = np.where(test_labels == cluster_id)[0]
        if len(indices) == 0:
            continue
        chosen_idx = random.choice(indices)
        path = test_data_paths[chosen_idx][2]
        meaning = cluster_mapping.get(cluster_id, "Unknown")
        print(f"Cluster {cluster_id} ({meaning}): {path}")

    # Step 9: Plot cluster visualization
    if X_test_scaled.shape[1] >= 2:
        plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=test_labels, cmap="viridis", s=20)
        plt.title("DBSCAN clustering of test engine sounds")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.show()

    # Step 10: Assign fixed readable names
    readable_map = {0: "Good Engine", 1: "Broken Engine", 2: "Heavy Load"}
    print("\nCluster Mapping:")
    for k, v in readable_map.items():
        print(f"  Cluster {k}: {v}")

    # Step 11: Robust label detection
    def match_label(name):
        name = name.lower()
        if "good" in name:
            return 0
        elif "broken" in name:
            return 1
        elif "heavy" in name or "load" in name:
            return 2
        else:
            return -1

    y_true = np.array([match_label(name) for name in folder_names])
    y_pred = np.array([lbl if lbl in readable_map else -1 for lbl in test_labels])

    valid_idx = (y_true != -1)
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]

    if len(y_true) == 0:
        print("\n[ERROR] No valid labeled test samples found! Check your folder names.")
        return

    # Step 12: Evaluate model
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    # Step 13: Confusion Matrix
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Broken", "HeavyLoad"])
        disp.plot(cmap='Blues')
        plt.title("DBSCAN Confusion Matrix")
        plt.show()
    else:
        print("\n[WARNING] Not enough classes to display a confusion matrix.")

    # Step 14: Plot training vs test sets (optional visual)
    print("\nPlotting cluster distributions...")
    processed_train_good = X_train_scaled[labels_train == 0]
    processed_train_broken = X_train_scaled[labels_train == 1]
    processed_train_heavy = X_train_scaled[labels_train == 2]

    plt.title("Training Data Clusters")
    plot_points(processed_train_good, "blue", "Good")
    plot_points(processed_train_broken, "gray", "Broken")
    plot_points(processed_train_heavy, "red", "Heavy Load")
    plt.legend()
    plt.show()

    plt.title("Test Data Clusters")
    plot_points(X_test_scaled[y_pred == 0], "blue", "Good")
    plot_points(X_test_scaled[y_pred == 1], "gray", "Broken")
    plot_points(X_test_scaled[y_pred == 2], "red", "Heavy Load")
    plt.legend()
    plt.show()

    # Step 15: Save labels
    output_dir = os.path.join("unsupervised_model", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "dbscan_test_labels.npy"), test_labels)
    print(f"Saved cluster labels to: {os.path.join(output_dir, 'dbscan_test_labels.npy')}")

    print("\n[DBSCAN ENGINE CLASSIFIER] Done.")


if __name__ == "__main__":
    main()
