# supervised_model/debug_features.py
import os, sys, numpy as np, librosa
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.preprocessing import make_cnn_features

SR, N = 16000, 1  # N=1 fil per klasse er nok for sanity
CLASSES = ("engine1_good","engine2_broken","engine3_heavyload")
ROOT = Path(__file__).resolve().parents[1] / "data" / "train"

def center_crop_or_pad(y, L):
    cur = y.shape[-1]
    if cur == L: return y
    if cur < L:
        pad = L - cur; left = pad // 2; right = pad - left
        return np.pad(y,(left,right),mode="reflect")
    start = (cur - L)//2
    return y[start:start+L]

def main():
    L = int(2.0 * SR)
    for cname in CLASSES:
        folder = ROOT / cname
        wavs = sorted(folder.glob("*.wav"))[:N]
        if not wavs:
            print(f"[FEIL] Ingen filer i {folder}")
            continue
        print(f"\nKlasse: {cname}")
        for p in wavs:
            y, _ = librosa.load(p, sr=SR, mono=True)
            y = center_crop_or_pad(y, L)
            feat = make_cnn_features(y, SR)   # forventer [1, n_mels, T]
            print(f"- {p.name}: shape={feat.shape}, mean={feat.mean():.4f}, std={feat.std():.4f}, finite={np.isfinite(feat).all()}")
            if not np.isfinite(feat).all() or feat.std() < 1e-7:
                print("  -> UGYLDIG/KONSTANT FEATURE! Sjekk prepro (noise/lowpass).")
if __name__ == "__main__":
    main()