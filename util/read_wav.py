import numpy as np
import wave

def read_wav(file_as_path: str) -> np.ndarray:
    with wave.open(file_as_path) as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return audio_array;
