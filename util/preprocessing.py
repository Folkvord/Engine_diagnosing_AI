import numpy as np
import librosa

"""
    Preprocessing functions for preprocessing data with functions
    - Written by Edwina Larsen, Sofie <velg et etternavn>, Fredrik Bjune & Kristoffer Folkvord 
"""

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
    y_denoised = librosa.istft(S_denoised * phase)
    
    # 6. Normaliser og returner
    y_denoised = y_denoised / np.max(np.abs(y_denoised) + 1e-8)
    return y_denoised.astype(np.float32)


# Processes an array of dataentries
def preprocess_data(data: tuple) -> np.ndarray:
    processed_data = []
    for i in range(len(data)):
        audio_array = data[i][0]
        sample_rate = data[i][1]
        processed_entry = reduce_noise(audio_array, sample_rate)
        processed_data.append(processed_entry)
    return processed_data