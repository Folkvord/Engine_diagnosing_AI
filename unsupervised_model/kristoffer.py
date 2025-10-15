import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import util.read_wav as read
import util.preprocessing as process

"""
    This file will contain the K-means unsupervised model
    - Written by Kristoffer Folkvord
"""

# Model settings:
N_CLUSTERS = 3
MAX_ITER = 300

# Preprocess the data used
train_data = read.get_formated_wav(data_type="all", is_train=True)
processed_train_data = process.preprocess_data(train_data)
test_data = read.get_formated_wav(data_type="all", is_train=False)
processed_test_data = process.preprocess_data(test_data)

# Train the model
kmeans_model = KMeans(N_CLUSTERS, max_iter=MAX_ITER)
kmeans_model.fit_predict(processed_train_data)
cluster_centres = kmeans_model.cluster_centers_
cluster_labels = kmeans_model.labels_
print(f"[K-MEANS MODEL]: Finished training with {kmeans_model.n_iter_} iterations.");

# Test the shit out of it
