import os, sys, torch, librosa, soundfile as sf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from collections import Counter

# slik at vi kan importere treningsmodellen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supervised_model.train_cnn import SmallCNN
from util.preprocessing import make_cnn_features

# --- konfig ---
SR = 16000
DURATION = 2.0
N_MELS = 64
BATCH_SIZE = 32

# --- hjelpemetoder ---
def _target_len(): return int(SR * DURATION)

def _center_crop_or_pad(y, L):
    cur = y.shape[-1]
    if cur == L: return y
    if cur < L:
        pad = L - cur; left = pad // 2; right = pad - left
        return np.pad(y, (left, right), mode="reflect")
    start = (cur - L) // 2
    return y[start:start + L]

def _read_mid_segment(path: str, duration_s: float, target_sr: int) -> np.ndarray:
    info = sf.info(path)
    native_sr = info.samplerate
    seg_frames = int(duration_s * native_sr)
    start = max(0, (info.frames - seg_frames) // 2)
    y, sr = sf.read(path, start=start, frames=seg_frames, dtype="float32")
    if y.ndim > 1: y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return y

# --- dataset (gjør innlesing identisk med trening) ---
class EngineDataset(Dataset):
    def __init__(self, root, class_names):
        self.items = []
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.target_len = _target_len()

        root = Path(root)
        for cname in class_names:
            folder = root / cname
            files = sorted(folder.glob("*.wav"))
            if not files:
                raise FileNotFoundError(f"Ingen WAV i {folder}")
            self.items += [(str(f), self.class_to_idx[cname]) for f in files]

    def __len__(self): 
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]

        # Samme innlesing som i trening:
        y, _ = librosa.load(path, sr=SR, mono=True)

        # Samme lengdebehandling som i trening (center crop i val/test):
        y = _center_crop_or_pad(y, self.target_len)

        feat = make_cnn_features(y, SR)  # [1, N_MELS, T]
        feat = feat.astype(np.float32)

        # --- Sanitetssjekker (fanger feature-kollaps) ---
        if not np.isfinite(feat).all():
            raise RuntimeError(f"NaN/inf i features for {path}")
        if np.std(feat) < 1e-7:
            raise RuntimeError(f"Nesten konstant feature for {path} (std={np.std(feat):.2e})")

        x = torch.from_numpy(feat)  # [1, n_mels, T]
        return x, label

# --- plotting ---
def plot_confusion(cm, classes):
    plt.figure(figsize=(6,6))
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45, ha="right")
    plt.yticks(ticks, classes)
    thresh = cm.max()/2 if cm.max() > 0 else 1
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png") #må save fig pga problemer med Tcl og Windows python 13.3
    print("Saved confusion matrix as confusion_matrix.png")

def test_model(model_path, data_root):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    torch.set_grad_enabled(False)

    ckpt = torch.load(model_path, map_location=device)
    class_names = tuple(ckpt["class_names"])

    ds = EngineDataset(data_root, class_names)
    # num_workers=0 er tryggest på macOS/Windows
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallCNN(n_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred = [], []
    for xb, yb in dl:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        y_true.append(yb.numpy())
        y_pred.append(pred)

    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    # Rask sanity: per-klasse telling
    from collections import Counter
    print("Test fordeling:", Counter(y_true))
    print("Pred fordeling:", Counter(y_pred))

    acc = (y_true == y_pred).mean()
    print(f"\nTest accuracy: {acc:.4f}")

    # Confusion matrix + plot (som du hadde)
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    plot_confusion(cm, class_names)
    acc = (np.trace(cm) / np.sum(cm))
    print(f"Total accuracy: {acc*100:.2f}%")

if __name__ == "__main__":
    PROJECT  = Path(__file__).resolve().parents[1]
    MODEL    = PROJECT / "engine_cnn_best.pt"
    TEST_DATA = PROJECT / "data" / "test_cut"   # <-- bruk test_cut
    test_model(MODEL, TEST_DATA)
    