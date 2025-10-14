from typing import Tuple
import numpy as np
import glob
import wave


def get_training_data() -> tuple:
    data = glob.glob("data/train_cut/*/*.wav")
    return data

def get_test_data() -> tuple:
    data = glob.glob("data/test_cut/*/*.wav")
    return data

def process_wav_files(data: tuple) -> tuple:
    processed_data = []
    for wav_file in data:
        processed_data.append(read_wav_with_sr(wav_file))
    return tuple(processed_data)


def read_wav_with_sr(file_as_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(file_as_path) as wav_file:
        sr = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array, sr
