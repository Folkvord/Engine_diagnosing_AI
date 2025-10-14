import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))     # Evil python environment hack >:)

import util.read_wav as read
import util.preprocessing as process

print(read.process_wav_files(read.get_training_data()))
process.reduce_noise()