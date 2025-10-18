import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter

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

# Model settings:
N_CLUSTERS = 3
MAX_ITER = 300

# Preprocess the data used
train_data = read.get_formated_wav(data_type="all", is_train=True)
processed_train_data = process.preprocess_data(train_data, "unsupervised")

test_good = read.get_formated_wav(data_type="good", is_train=False)
processed_test_good = process.preprocess_data(test_good, "unsupervised")
test_broken = read.get_formated_wav(data_type="broken", is_train=False)
processed_test_broken = process.preprocess_data(test_broken, "unsupervised")
test_heavyload = read.get_formated_wav(data_type="heavy load", is_train=False)
processed_test_heavyload = process.preprocess_data(test_heavyload, "unsupervised")

# Train the model
kmeans_model = KMeans(N_CLUSTERS, max_iter=MAX_ITER, random_state=0)
kmeans_model.fit_predict(processed_train_data)
cluster_centres = kmeans_model.cluster_centers_
cluster_labels = kmeans_model.labels_
print(f"[K-MEANS MODEL]: Finished training with {kmeans_model.n_iter_} iterations.")

# Test the shit out of it
print(f"[K-MEANS MODEL]: Testing good motors...")
predicted_good = kmeans_model.predict(processed_test_good)
print("RESULTS FOR GOOD ENGINES:")
print_prediction_result(predicted_good)
print()

print(f"[K-MEANS MODEL]: Testing broken motors...")
predicted_broken = kmeans_model.predict(processed_test_broken)
print("RESULTS FOR BROKEN ENGINES:")
print_prediction_result(predicted_broken)
print()

print(f"[K-MEANS MODEL]: Testing heavyload motors...")
predicted_heavyload = kmeans_model.predict(processed_test_heavyload)
print("RESULTS FOR ENGINES UNDER HEAVY LOAD:")
print_prediction_result(predicted_heavyload)
print()
