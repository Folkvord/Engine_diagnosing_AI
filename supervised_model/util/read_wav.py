import numpy as np
import wave
from typing import Tuple

def get_train_path(type: str) -> str:
    if type == "good":
        return "data/train_cut/engine1_good/"
    elif type == "broken":
        return "data/train_cut/engine2_broken/"
    elif type == "heavy load":
        return "data/train_cut/engine3_heavyload/"
    else:
        return ""

def get_test_path(type: str) -> str:
    if type == "good":
        return "data/test_cut/engine1_good/"
    elif type == "broken":
        return "data/test_cut/engine2_broken/"
    elif type == "heavy load":
        return "data/test_cut/engine3_heavyload/"
    else:
        return ""

def read_wav(file_as_path: str) -> np.ndarray:
    with wave.open(file_as_path) as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array

def read_wav_with_sr(file_as_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(file_as_path) as wav_file:
        sr = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array, sr
