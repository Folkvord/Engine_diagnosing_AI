import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import util.read_wav as read
import util.preprocessing as process

"""
    This file contains the K-means unsupervised model
    - Written by Kristoffer Folkvord
"""

# Prints the prediction result
def print_prediction_result(prediction):
    counter = Counter(prediction)
    for key, val in counter.items():
        print(f"CLUSTER {key}: {val}")

# Plots datapoints :)
def plot_points(data, colour):
    plt.scatter(data[:, 0], data[:, 1], c=colour)

def correct(cluster, prediction):
    n_correct = 0
    for i in range(len(prediction)):
        if prediction[i] == cluster:
            n_correct += 1
    return n_correct

# Model settings:
ALGORITHM = "elkan"
N_CLUSTERS = 3
MAX_ITER = 300

# Preprocess the data used
train_data = read.get_formated_wav(data_type="all", is_train=True)
processed_train_data = process.preprocess_data(train_data, "unsupervised")

train_good = read.get_formated_wav(data_type="good", is_train=True)
processed_train_good = process.preprocess_data(train_good, "unsupervised")
train_broken = read.get_formated_wav(data_type="broken", is_train=True)
processed_train_broken = process.preprocess_data(train_broken, "unsupervised")
train_heavyload = read.get_formated_wav(data_type="heavy load", is_train=True)
processed_train_heavyload = process.preprocess_data(train_heavyload, "unsupervised")


test_good = read.get_formated_wav(data_type="good", is_train=False)
processed_test_good = process.preprocess_data(test_good, "unsupervised")
test_broken = read.get_formated_wav(data_type="broken", is_train=False)
processed_test_broken = process.preprocess_data(test_broken, "unsupervised")
test_heavyload = read.get_formated_wav(data_type="heavy load", is_train=False)
processed_test_heavyload = process.preprocess_data(test_heavyload, "unsupervised")


# Train the model
kmeans_model = KMeans(N_CLUSTERS, max_iter=MAX_ITER, algorithm=ALGORITHM, random_state=42)
kmeans_model.fit(processed_train_data)
cluster_centres = kmeans_model.cluster_centers_
cluster_labels = kmeans_model.labels_
print(f"[K-MEANS MODEL]: Finished training with {kmeans_model.n_iter_} iterations.")

# Determine accuracy
good_cluster = kmeans_model.predict(processed_train_good)[0]
broken_cluster = kmeans_model.predict(processed_train_broken)[0]
heavyload_cluster = kmeans_model.predict(processed_train_heavyload)[0]

good_pred = kmeans_model.predict(processed_test_good)
broken_pred = kmeans_model.predict(processed_test_broken)
heavyload_pred = kmeans_model.predict(processed_test_heavyload)
n_correct = correct(good_cluster, good_pred) + correct(broken_cluster, broken_pred) + correct(heavyload_cluster, heavyload_pred)
n_total = len(good_pred) + len(broken_pred) + len(heavyload_pred)

answer = [good_cluster] * len(good_pred) + [broken_cluster] * len(broken_pred) + [heavyload_cluster] * len(heavyload_pred)
tot_pred = list(good_pred) + list(broken_pred) + list(heavyload_pred)

print(f"ACC: {n_correct / n_total}")
matdisp = ConfusionMatrixDisplay(confusion_matrix(answer, tot_pred))
matdisp.plot()

plot_points(processed_train_good, "blue")
plot_points(processed_train_broken, "gray")
plot_points(processed_train_heavyload, "red")
plot_points(cluster_centres, "black")
plt.show()

plot_points(processed_test_good, "blue")
plot_points(processed_test_broken, "gray")
plot_points(processed_test_heavyload, "red")
plot_points(cluster_centres, "black")
plt.show()