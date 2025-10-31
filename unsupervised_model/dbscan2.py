import sys, os, random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, pairwise_distances_argmin_min

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.read_wav import get_formated_wav
from util.preprocessing import preprocess_data


def plot_points(points, color, label=None):
    if len(points) == 0:
        return
    if points.ndim >= 2 and points.shape[1] >= 2:
        plt.scatter(points[:, 0], points[:, 1], c=color, s=20, label=label, alpha=0.6)


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


def main():
    print("[DBSCAN] starting...")

    # load training stuff
    train_good = get_formated_wav("good", is_train=True)
    train_broken = get_formated_wav("broken", is_train=True)
    train_heavy = get_formated_wav("heavy load", is_train=True)

    Xg = preprocess_data(train_good, "unsupervised")
    Xb = preprocess_data(train_broken, "unsupervised")
    Xh = preprocess_data(train_heavy, "unsupervised")

    X_train = np.vstack([Xg, Xb, Xh])
    y_true_train = np.array([0]*len(Xg) + [1]*len(Xb) + [2]*len(Xh))

    print("training data:", X_train.shape)

    # scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # dbscan
    db = DBSCAN(eps=2.0, min_samples=3, n_jobs=-1)
    y_pred_train = db.fit_predict(X_train_scaled)

    if len(set(y_pred_train)) <= 1:
        print("no clusters found, tweak eps/min_samples")
        return

    core_samples = X_train_scaled[db.core_sample_indices_]
    core_labels = y_pred_train[db.core_sample_indices_]

    n_clusters = len(set(y_pred_train)) - (1 if -1 in y_pred_train else 0)
    print("clusters found:", n_clusters)

    # load test data
    test_good = get_formated_wav("good", is_train=False)
    test_broken = get_formated_wav("broken", is_train=False)
    test_heavy = get_formated_wav("heavy load", is_train=False)

    Xt_g = preprocess_data(test_good, "unsupervised")
    Xt_b = preprocess_data(test_broken, "unsupervised")
    Xt_h = preprocess_data(test_heavy, "unsupervised")

    X_test = np.vstack([Xt_g, Xt_b, Xt_h])
    y_true = np.array([0]*len(Xt_g) + [1]*len(Xt_b) + [2]*len(Xt_h))

    X_test_scaled = scaler.transform(X_test)

    nearest_idx, _ = pairwise_distances_argmin_min(X_test_scaled, core_samples)
    test_labels = core_labels[nearest_idx]

    # map clusters
    cluster_map = {}
    for c in set(y_pred_train):
        if c == -1:
            continue
        idx = np.where(y_pred_train == c)[0]
        majority = np.bincount(y_true_train[idx]).argmax()
        cluster_map[c] = majority

    y_pred = np.array([cluster_map.get(lbl, -1) for lbl in test_labels])

    # drop -1
    mask = y_pred != -1
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\nResults:")
    print("acc:", round(acc,3))
    print("prec:", round(prec,3))
    print("rec:", round(rec,3))
    print("f1:", round(f1,3))

    # confusion matrix
    if len(np.unique(y_true)) > 1:
        cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Good","Broken","HeavyLoad"])
        disp.plot(cmap="Blues")
        plt.title("DBSCAN Confusion Matrix")
        plt.show()
    else:
        print("not enough classes to show matrix")

    # plot
    plt.figure()
    plot_points(X_train_scaled[y_true_train==0], "blue", "Good")
    plot_points(X_train_scaled[y_true_train==1], "gray", "Broken")
    plot_points(X_train_scaled[y_true_train==2], "red", "Heavy Load")
    plt.scatter(core_samples[:, 0], core_samples[:, 1], c="black", s=50, label="Core Samples")
    plt.legend()
    plt.title("Training data + core samples")
    plt.show()


if __name__ == "__main__":
    main()
