import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

import matplotlib.pyplot as plt
from collections import Counter
from sklearn.decomposition import PCA

import util.read_wav as read
import util.preprocessing as process

"""
    This file contains the K-means unsupervised model
    - Written by Kristoffer Folkvord
"""

# Model-settings:
MAX_ITERATIONS  = 30
K_CENTROIDS     =  3

# Preprocess the data used
#train_data = read.get_formated_wav(data_type="all", is_train=True)
#processed_train_data = process.preprocess_data(train_data, "unsupervised")

pca = PCA(n_components=2)

test_good = read.get_formated_wav(data_type="good", is_train=False)
processed_test_good = process.preprocess_data(test_good, "unsupervised")
x_pca = pca.fit_transform(processed_test_good)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c="blue")

test_broken = read.get_formated_wav(data_type="broken", is_train=False)
processed_test_broken = process.preprocess_data(test_broken, "unsupervised")
x_pca = pca.fit_transform(processed_test_broken)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c="gray")

test_heavyload = read.get_formated_wav(data_type="heavy load", is_train=False)
processed_test_heavyload = process.preprocess_data(test_heavyload, "unsupervised")
x_pca = pca.fit_transform(processed_test_heavyload)
plt.scatter(x_pca[:, 0], x_pca[:, 1], c="red")

#plt.plot()
plt.show()