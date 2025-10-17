from scipy.signal import butter, filtfilt
import numpy as np
import librosa
import time

"""
    Preprocessing functions for preprocessing data with functions
    - Written by Edwina Larsen, Sofie <velg et etternavn>, Fredrik Bjune & Kristoffer Folkvord 
"""

# Reduces the noise in a singular dataentry 
def reduce_noise(audio_array: np.ndarray, sample_rate: int, noise_duration=0.5, prop_decrease=0.9) -> np.ndarray:
    
    # 1. Estimer støyprofil fra starten (f.eks. 0.5 sek)
    n_noise = int(noise_duration * sample_rate)
    noise_clip = audio_array[:n_noise]

    # 2. Beregn STFT (frekvensrepresentasjon)
    S_full, phase = librosa.magphase(librosa.stft(audio_array.astype(np.float32)))
    S_noise, _ = librosa.magphase(librosa.stft(noise_clip.astype(np.float32)))

    # 3. Beregn gjennomsnittlig støyenergi per frekvens
    noise_profile = np.mean(S_noise, axis=1, keepdims=True)

    # 4. Fjern støyprofil fra signalets spektrum
    S_denoised = S_full - prop_decrease * noise_profile
    S_denoised = np.maximum(S_denoised, 0.0)  # ingen negative verdier

    # 5. Gjør inverse STFT tilbake til tidsdomene
    audio_denoised = librosa.istft(S_denoised * phase)
    
    # 6. Normaliser og returner
    audio_denoised = audio_denoised / np.max(np.abs(audio_denoised) + 1e-8)
    return audio_denoised.astype(np.float32)


# Processes an array of dataentries
def preprocess_data(data: tuple, noise_duration=0.5, prop_decrease=0.5) -> np.ndarray:
    processed_data = []
    start = time.time()
    for i in range(len(data)):
        audio_array = data[i][0]
        sample_rate = data[i][1]
        processed_entry = preprocess_pipeline(audio_array, sample_rate)
        processed_data.append(processed_entry)
    end = time.time()
    time_taken = end - start
    print(f"[PREPROCESSOR]: Preprocessed {len(data)} entries in {time_taken:.2f} s.")
    return np.array(processed_data)



# The preprocessing pipeline
def preprocess_pipeline(audio_array: np.ndarray, sample_rate: int):

    # Normalize the audio array
    norm_audio = normalize((audio_array).astype(np.float32))

    # Reduce the background noise
    denoised_audio = reduce_noise_v2(norm_audio, sample_rate)

    # Filter outlying frequencies
    filtered_audio = filter_outlying_freq(denoised_audio, sample_rate, 5000)

    # Segmentize (wont work :( )
    segmentized_audio = segmentize(filtered_audio, sample_rate)

    # Select features
    selected_features = select_features(filtered_audio, sample_rate)

    return selected_features


# Normalizes an audio_array
def normalize(audio_array: np.ndarray):
    return audio_array / np.max(np.abs(audio_array) + 1e-8)


# Modified version of old reduce_noise
def reduce_noise_v2(audio_array: np.ndarray, sample_rate: int, noise_duration=0.5, prop_decrease=0.5):
    n_noise = int(noise_duration * sample_rate)
    noise_clip = audio_array[:n_noise]

    S_full, phase = librosa.magphase(librosa.stft(audio_array.astype(np.float32)))
    S_noise, _ = librosa.magphase(librosa.stft(noise_clip.astype(np.float32)))

    noise_profile = np.mean(S_noise, axis=1, keepdims=True)

    S_denoised = S_full - prop_decrease * noise_profile
    S_denoised = np.maximum(S_denoised, 0.0)  # ingen negative verdier

    audio_denoised = librosa.istft(S_denoised * phase)
    return audio_denoised


# Filters frequenies under *cutoff_freq* Hz
def filter_outlying_freq(audio_array: np.ndarray, sample_rate: int, cutoff_freq: int, order=4):
    half_sr = sample_rate / 2;
    norm_cutoff = cutoff_freq / half_sr;
    b, a = butter(order, norm_cutoff, btype="low", analog=False)
    filtered_audio = filtfilt(b, a, audio_array)
    return filtered_audio


# Segmentizes the audio_array
def segmentize(audio_array: np.ndarray, sample_rate: int):
    segment_size = int(0.2 * sample_rate)
    hop_len = int(0.1 * sample_rate)
    return librosa.util.frame(audio_array, frame_length=segment_size, hop_length=hop_len).T


# Selects features
def select_features(audio_array: np.ndarray, sample_rate: int):
    # "Lydsignatur"
    mfcc = librosa.feature.mfcc(y=audio_array, sr=sample_rate, n_mfcc=20)
    # Tyngepunktet i filen
    centroid = librosa.feature.spectral_centroid(y=audio_array, sr=sample_rate)
    # Bredden på frekvenseinnholdet
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_array, sr=sample_rate)
    # idk LOL
    rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=sample_rate, roll_percent=0.5)
    # Hvor kraftig signalet er
    rms = librosa.feature.rms(y=audio_array)
    # Hvor ofte lydfilen krysser null
    zcr = librosa.feature.zero_crossing_rate(y=audio_array)

    packed_features = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(centroid),
        np.mean(bandwidth),
        np.mean(rolloff),
        np.mean(rms),
        np.mean(zcr)
    ])

    return packed_features
