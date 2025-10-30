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


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.read_wav import get_formated_wav
from util.preprocessing import preprocess_data

#  Helper Functions 

def plot_points(points, color, label=None):
    if len(points) == 0:
        return
    if points.ndim >= 2 and points.shape[1] >= 2:
        plt.scatter(points[:, 0], points[:, 1], c=color, s=20, label=label, alpha=0.6)

def correct(cluster, prediction):
    return np.sum(prediction == cluster)

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

#  Main 

def main():
    print("[DBSCAN ENGINE CLASSIFIER] Starting clustering...")

    # Step 1: Load training data per class
    train_good = get_formated_wav("good", is_train=True)
    train_broken = get_formated_wav("broken", is_train=True)
    train_heavy = get_formated_wav("heavy load", is_train=True)

    X_train_good = preprocess_data(train_good, "unsupervised")
    X_train_broken = preprocess_data(train_broken, "unsupervised")
    X_train_heavy = preprocess_data(train_heavy, "unsupervised")

    X_train = np.vstack([X_train_good, X_train_broken, X_train_heavy])
    labels_train_true = np.array([0]*len(X_train_good) + [1]*len(X_train_broken) + [2]*len(X_train_heavy))

    print(f"Training data shape: {X_train.shape}")

    # Step 2: Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Step 3: DBSCAN clustering
    print("Running DBSCAN on training data...")
    dbscan = DBSCAN(eps=2.0, min_samples=3, n_jobs=-1)  #tune eps hvis trengs
    labels_train = dbscan.fit_predict(X_train_scaled)

    core_samples = X_train_scaled[dbscan.core_sample_indices_]
    core_labels = labels_train[dbscan.core_sample_indices_]

    #all classes get a cluster id
    if len(set(labels_train)) <= 1:
        print("[WARNING] DBSCAN failed to produce clusters. Try lowering eps or min_samples.")
        return

    n_clusters = len(set(labels_train)) - (1 if -1 in labels_train else 0)
    print(f"Found {n_clusters} clusters in training data")

    # Step 4: Load test data per class
    test_good = get_formated_wav("good", is_train=False)
    test_broken = get_formated_wav("broken", is_train=False)
    test_heavy = get_formated_wav("heavy load", is_train=False)

    X_test_good = preprocess_data(test_good, "unsupervised")
    X_test_broken = preprocess_data(test_broken, "unsupervised")
    X_test_heavy = preprocess_data(test_heavy, "unsupervised")

    X_test = np.vstack([X_test_good, X_test_broken, X_test_heavy])
    y_true = np.array([0]*len(X_test_good) + [1]*len(X_test_broken) + [2]*len(X_test_heavy))

    X_test_scaled = scaler.transform(X_test)

    # Step 5: Assign test samples to nearest DBSCAN cluster core
    nearest, _ = pairwise_distances_argmin_min(X_test_scaled, core_samples)
    test_labels = core_labels[nearest]

    # Step 6: Map DBSCAN majority vote clustering
    cluster_mapping = {}
    for cluster_id in set(labels_train):
        if cluster_id == -1:
            continue
        idx = np.where(labels_train == cluster_id)[0]
        majority = np.bincount(labels_train_true[idx]).argmax()
        cluster_mapping[cluster_id] = majority


    y_pred = np.array([cluster_mapping.get(lbl, -1) for lbl in test_labels])

    valid_idx = y_pred != -1
    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]

    # Step 7: Evaluate
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nEvaluation Metrics:")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  F1 Score:  {f1:.3f}")

    # Step 8: Confusion Matrix
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good", "Broken", "HeavyLoad"])
        disp.plot(cmap='Blues')
        plt.title("DBSCAN Confusion Matrix")
        plt.show()
    else:
        print("[WARNING] Not enough classes to display a confusion matrix.")

    # Step 9: Plot clusters
    plt.figure()
    plot_points(X_train_scaled[labels_train_true==0], "blue", "Good")
    plot_points(X_train_scaled[labels_train_true==1], "gray", "Broken")
    plot_points(X_train_scaled[labels_train_true==2], "red", "Heavy Load")
    plt.scatter(core_samples[:, 0], core_samples[:, 1], c="black", s=50, label="Core Samples")
    plt.legend()
    plt.title("Training Data with Core Samples")
    plt.show()

if __name__ == "__main__":
    main()
