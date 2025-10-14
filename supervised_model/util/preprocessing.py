import numpy as np
import librosa

def reduce_noise(y, sr, noise_duration=0.5, prop_decrease=0.9):
    
    # 1. Estimer støyprofil fra starten (f.eks. 0.5 sek)
    n_noise = int(noise_duration * sr)
    noise_clip = y[:n_noise]

    # 2. Beregn STFT (frekvensrepresentasjon)
    S_full, phase = librosa.magphase(librosa.stft(y))
    S_noise, _ = librosa.magphase(librosa.stft(noise_clip))

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