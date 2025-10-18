# supervised_model/test_cnn.py
import os, sys, random
from pathlib import Path
from typing import List, Dict, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # for util/

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------
# Konfig (må matche treningsskriptet)
# --------------------------
SR        = 16000
DURATION  = 2.0
N_MELS    = 64
BATCHSIZE = 32

# mappenavnene dine:
CLASS_NAMES = ("engine1_good", "engine2_broken", "engine3_heavyload")

# --------------------------
# Hent adapter fra deres prepro-fil
# --------------------------
from util.preprocessing import make_cnn_features

# --------------------------
# Hjelp
# --------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(42)

def pick_device() -> Tuple[str, bool]:
    if torch.cuda.is_available():
        return "cuda", True
    return "cpu", False

def _target_len() -> int:
    return int(SR * DURATION)

def _center_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    cur = y.shape[-1]
    if cur == target_len: return y
    if cur < target_len:
        pad = target_len - cur
        left = pad // 2; right = pad - left
        return np.pad(y, (left, right), mode="reflect")
    start = (cur - target_len) // 2
    return y[start:start + target_len]

def _read_segment_fast(path: str, duration_s: float, target_sr: int) -> np.ndarray:
    """Les KUN et fast segment fra fil (midt i), resample kun segmentet."""
    info = sf.info(path)
    native_sr = info.samplerate
    total_frames = info.frames
    seg_frames = max(1, int(duration_s * native_sr))
    start = max(0, (total_frames - seg_frames) // 2)
    y, file_sr = sf.read(path, start=start, frames=seg_frames, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if file_sr != target_sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=target_sr, res_type="soxr_hq")
    return y

# --------------------------
# Dataset for TEST (ingen augment)
# --------------------------
class EngineTestDataset(Dataset):
    def __init__(self, root, class_names=CLASS_NAMES):
        super().__init__()
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c:i for i,c in enumerate(class_names)}
        self.target_len = _target_len()

        per_class: Dict[str, List[str]] = {}
        for cname in class_names:
            p = self.root / cname
            if not p.is_dir():
                raise FileNotFoundError(f"Mangler mappe: {p}")
            files = [str(fp) for fp in p.iterdir() if fp.is_file() and fp.suffix.lower() == ".wav"]
            files.sort()
            if not files:
                raise FileNotFoundError(f"Ingen WAV i {p}")
            per_class[cname] = files

        self.items = [(f, self.class_to_idx[c]) for c, flist in per_class.items() for f in flist]

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        y = _read_segment_fast(path, DURATION, SR)
        y = _center_crop_or_pad(y, self.target_len)

        feat = make_cnn_features(y, SR)    # [1, N_MELS, T]
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3 or feat.shape[0] != 1 or feat.shape[1] != N_MELS:
            raise ValueError(f"make_cnn_features returnerte {feat.shape}, forventet [1,{N_MELS},T]")
        x = torch.from_numpy(feat)
        return x, label, path

# --------------------------
# Modell (samme som i treningen)
# --------------------------
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, k=(3,3), p=(1,1), pool=(2,2), drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, k, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, k, padding=p, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(pool),
            nn.Dropout(drop),
        )
    def forward(self, x): return self.net(x)

class SmallCNN(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.feat = nn.Sequential(
            ConvBlock(1, 32, drop=0.10),
            ConvBlock(32, 64, drop=0.15),
            ConvBlock(64, 128, drop=0.20),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )
    def forward(self, x):
        return self.head(self.feat(x))

# --------------------------
# Confusion matrix + rapport
# --------------------------
def plot_confusion(cm: np.ndarray, class_names: Tuple[str, ...], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2 if cm.max() > 0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    fp = out_dir / "test_confusion_matrix.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.show()

def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    # enkel egen rapport (precision/recall/f1 per klasse)
    eps = 1e-12
    rep = []
    for k in range(n_classes):
        tp = np.sum((y_true == k) & (y_pred == k))
        fp = np.sum((y_true != k) & (y_pred == k))
        fn = np.sum((y_true == k) & (y_pred != k))
        prec = tp / (tp + fp + eps)
        rec  = tp / (tp + fn + eps)
        f1   = 2 * prec * rec / (prec + rec + eps)
        rep.append((prec, rec, f1))
    return rep

# --------------------------
# Test-kjøring
# --------------------------
def run_test(test_root: Path, ckpt_path: Path, class_names=CLASS_NAMES):
    device, use_pinned = pick_device()

    # data
    ds = EngineTestDataset(test_root, class_names=class_names)
    dl = DataLoader(ds, batch_size=BATCHSIZE, shuffle=False, num_workers=0, pin_memory=use_pinned)

    # modell
    model = SmallCNN(n_classes=len(class_names)).to(device)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fant ikke modell-weights: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # loop
    y_true = []
    y_pred = []
    paths  = []

    with torch.no_grad():
        for xb, yb, pb in dl:
            xb = xb.to(device, non_blocking=False)
            logits = model(xb)
            pred = logits.argmax(1).cpu().numpy()
            y_true.append(yb.numpy())
            y_pred.append(pred)
            paths.extend(pb)

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # accuracy
    acc = (y_true == y_pred).mean()

    # confusion matrix
    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    # rapport
    rep = classification_report(y_true, y_pred, n_classes)
    rep_df = pd.DataFrame(rep, columns=["precision", "recall", "f1"], index=list(class_names))

    # predictions csv
    out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True, parents=True)
    pred_df = pd.DataFrame({
        "path": paths,
        "true_idx": y_true,
        "true_label": [class_names[i] for i in y_true],
        "pred_idx": y_pred,
        "pred_label": [class_names[i] for i in y_pred],
    })
    pred_df.to_csv(out_dir / "test_predictions.csv", index=False)

    # print & plots
    print("\n=== TEST RESULTATER ===")
    print(f"Accuracy: {acc:.4f}\n")
    print("Per-klasse (precision/recall/f1):")
    print(rep_df.to_string(float_format=lambda v: f"{v:.3f}"))

    plot_confusion(cm, class_names, out_dir)

# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    TEST_ROOT    = PROJECT_ROOT / "data" / "test"
    CKPT         = PROJECT_ROOT / "engine_cnn_best.pt"   # lagres av treningsskriptet

    run_test(TEST_ROOT, CKPT, class_names=CLASS_NAMES)