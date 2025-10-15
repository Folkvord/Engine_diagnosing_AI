import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

import util.read_wav as read
import util.preprocessing as process

"""
    This file will contain the K-means unsupervised model
    - Written by Kristoffer Folkvord
"""

data = read.get_formated_wav(data_type="all", is_train=True)
processed_data = process.preprocess_data(data)


