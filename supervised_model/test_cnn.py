import os, sys, random, argparse
from collections import Counter
from pathlib import Path
from typing import List, Dict, Tuple

# slik at vi kan importere util/ og treningsfila
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --- hent samme modelldefinisjon som i treningen ---
from supervised_model.SofieOgEdwina import SmallCNN  # <-- viktig

# --- må matche featurizer i treningen ---
SR        = 16000
DURATION  = 2.0
N_MELS    = 64
BATCHSIZE = 32

# --- adapter fra deres prepro-fil (samme som i treningen) ---
from util.preprocessing import make_cnn_features
if not callable(make_cnn_features):
    raise ImportError("Fant ikke make_cnn_features i util/preprocessing.py")

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
    info = sf.info(path)
    native_sr = info.samplerate
    total_frames = info.frames
    seg_frames = max(1, int(duration_s * native_sr))
    start = max(0, (total_frames - seg_frames) // 2)  # midtstilt segment
    y, file_sr = sf.read(path, start=start, frames=seg_frames, dtype="float32", always_2d=False)
    if y.ndim > 1: y = y.mean(axis=1)
    if file_sr != target_sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=target_sr, res_type="soxr_hq")
    return y

class EngineSplitDataset(Dataset):
    def __init__(self, root: Path, class_names: Tuple[str, ...]):
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
            files = [str(fp) for fp in p.iterdir() if fp.is_file() and fp.suffix.lower()==".wav"]
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
        feat = make_cnn_features(y, SR)  # [1, N_MELS, T]
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3 or feat.shape[0] != 1 or feat.shape[1] != N_MELS:
            raise ValueError(f"Feil feature-shape: {feat.shape}, forventet [1,{N_MELS},T]")
        x = torch.from_numpy(feat)
        return x, label, path

def plot_confusion(cm: np.ndarray, class_names: Tuple[str, ...], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion matrix")
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45, ha="right")
    plt.yticks(ticks, class_names)
    thresh = cm.max()/2 if cm.max()>0 else 1.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_dir/"confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

def classification_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int):
    eps = 1e-12
    rep = []
    for k in range(n_classes):
        tp = np.sum((y_true==k) & (y_pred==k))
        fp = np.sum((y_true!=k) & (y_pred==k))
        fn = np.sum((y_true==k) & (y_pred!=k))
        prec = tp / (tp+fp+eps)
        rec  = tp / (tp+fn+eps)
        f1   = 2*prec*rec/(prec+rec+eps)
        rep.append((prec, rec, f1))
    return rep

def run_eval(split_root: Path, ckpt_path: Path, split_name: str):
    device, use_pinned = pick_device()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Fant ikke modell-weights: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # KLASSER FRA CHECKPOINT (samme rekkefølge som i treningen SofieOgEdwina)
    class_names = tuple(ckpt.get("class_names",
                                 ("engine1_good","engine2_broken","engine3_heavyload")))
    print(f"[INFO] Klasser i checkpoint: {class_names}")

    ds = EngineSplitDataset(split_root, class_names=class_names)
    dl = DataLoader(ds, batch_size=BATCHSIZE, shuffle=False, num_workers=0, pin_memory=use_pinned)

    cnt = Counter([lbl for _, lbl, _ in ds])
    print(f"[INFO] {split_name}-filer per klasse:",
          {class_names[k]: v for k, v in cnt.items()})

    # bygg modell med riktig ut-dim og last weights
    model = SmallCNN(n_classes=len(class_names)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    y_true, y_pred, paths = [], [], []
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

    pred_cnt = Counter(y_pred.tolist())
    print(f"[INFO] Pred-fordeling ({split_name}):",
          {class_names[k]: v for k, v in pred_cnt.items()})

    acc = (y_true == y_pred).mean()
    print(f"\n=== {split_name.upper()} ACCURACY: {acc:.4f} ===\n")

    n_classes = len(class_names)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    rep = classification_report(y_true, y_pred, n_classes)
    rep_df = pd.DataFrame(rep, columns=["precision","recall","f1"],
                          index=list(class_names))
    print(rep_df.to_string(float_format=lambda v: f"{v:.3f}"))

    out_dir = PROJECT_ROOT/"outputs"; out_dir.mkdir(exist_ok=True, parents=True)
    plot_confusion(cm, class_names, out_dir)
    pd.DataFrame({
        "path": paths,
        "true_idx": y_true,
        "true_label": [class_names[i] for i in y_true],
        "pred_idx": y_pred,
        "pred_label": [class_names[i] for i in y_pred],
    }).to_csv(out_dir/f"{split_name}_predictions.csv", index=False)
    print(f"\nLagret: {out_dir/'confusion_matrix.png'} og {out_dir/(split_name + '_predictions.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train","test"], default="test",
                        help="Evaluer mot data/train eller data/test")
    args = parser.parse_args()

    CKPT = PROJECT_ROOT/"engine_cnn_best.pt"          # lagres av SofieOgEdwina
    SPLIT_ROOT = PROJECT_ROOT/"data"/args.split
    run_eval(SPLIT_ROOT, CKPT, split_name=args.split)