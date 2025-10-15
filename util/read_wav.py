from typing import Tuple
import numpy as np
import glob
import wave


# Returns a tuple of processable numpy arrays made from specified wav-files 
# is_train: bool specifying wether to get training- or test-data
# data_type: "all" processes all types of data
#            "good", "broken", "heavy load" processes this type
def get_formated_wav(data_type: str, is_train: bool):
    data = get_data(data_type, is_train);
    return process_wav_files(data)


# Returns a tuple of all cut wav-files of a spesified type
# is_train: bool specifying wether to get training- or test-data
# data_type: "all" processes all types of data
#            "good", "broken", "heavy load" processes this type
def get_data(type: str, is_train: bool) -> tuple:
    path = "data/train_cut/" if is_train else "data/test_cut/"
    if type == "all":
        path += "*/*.wav"
    elif type == "good":
        path += "engine1_good/*.wav"
    elif type == "broken":
        path += "engine2_broken/*.wav"
    elif type == "heavy load":
        path += "engine3_heavyload/*.wav"
    else:
        raise("BAD DATA TYPE GIVEN")
    
    data = glob.glob(path)
    return tuple(data)


# Takes a list of wav-files and turns them into processable arrays
def process_wav_files(data: tuple) -> tuple:
    processed_data = []
    for wav_file in data:
        processed_wav = read_wav_with_sample_rate(wav_file)
        processed_data.append(processed_wav)
    return tuple(processed_data)


# Takes one wav-file and processes it into processable arrays and sample_rate
def read_wav_with_sample_rate(file_as_path: str) -> Tuple[np.ndarray, int]:
    with wave.open(file_as_path) as wav_file:
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
        audio_array = np.frombuffer(frames, dtype=np.int16)
    return (audio_array, sample_rate)

