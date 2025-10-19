# supervised_model/diagnose_data.py
import os, sys, numpy as np, librosa, matplotlib.pyplot as plt
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.preprocessing import make_cnn_features

SR = 16000
CLASSES = ("engine1_good","engine2_broken","engine3_heavyload")
ROOT = Path(__file__).resolve().parents[1] / "data" / "test"
OUT  = Path(__file__).resolve().parents[1] / "diagnostics"
OUT.mkdir(exist_ok=True)

def center_crop_or_pad(y, L):
    cur = y.shape[-1]
    if cur == L: return y
    if cur < L:
        pad = L - cur; left = pad // 2; right = pad - left
        return np.pad(y, (left, right), mode="reflect")
    start = (cur - L)//2
    return y[start:start+L]

def save_mel(img, path):
    plt.figure(figsize=(6,3))
    plt.imshow(img, origin="lower", aspect="auto", cmap="magma")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def main():
    L = int(2.0 * SR)
    waves = []
    mels  = []
    names = []
    for cname in CLASSES:
        wavs = sorted((ROOT/cname).glob("*.wav"))
        if not wavs:
            print(f"[FEIL] Ingen wav i {ROOT/cname}")
            return
        p = wavs[0]
        y, _ = librosa.load(p, sr=SR, mono=True)
        y = center_crop_or_pad(y, L)
        mel = make_cnn_features(y, SR)[0]  # [64, T] for plotting
        waves.append(y); mels.append(mel); names.append(f"{cname}/{p.name}")

        # lagre mel-bilde
        save_mel(mel, OUT / f"mel_{cname}.png")

    waves = np.stack(waves)      # [3, L]
    print("\n=== Waveform forskjeller (L1-normalisert) ===")
    for i in range(3):
        for j in range(i+1, 3):
            d = np.mean(np.abs(waves[i] - waves[j])) / (np.mean(np.abs(waves[i])) + 1e-8)
            print(f"{CLASSES[i]} vs {CLASSES[j]}: {d:.6f}")

    # grov “likhet” i mel (cosine)
    print("\n=== Mel cosine-likhet (1 = identisk) ===")
    for i in range(3):
        for j in range(i+1, 3):
            a = mels[i].reshape(-1); b = mels[j].reshape(-1)
            c = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-8)
            print(f"{CLASSES[i]} vs {CLASSES[j]}: {c:.4f}")

    print(f"\nMel-figurer lagret i: {OUT}")

if __name__ == "__main__":
    main()