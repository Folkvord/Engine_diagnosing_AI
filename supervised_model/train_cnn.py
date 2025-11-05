import os, sys, random, math
from pathlib import Path
from typing import List, Dict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # for util/

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

"""Supervised model written and implemented by Sofie Emmelin Weber and Edwina Larsen"""

# ---------- Globals ----------
SR       = 16000
DURATION = 2.0
N_MELS   = 64
CLASSES  = ("engine1_good", "engine2_broken", "engine3_heavyload")

from util.preprocessing import supervised_preprocess_pipeline

def set_seed(seed=42):
    import time
    if seed is None:
        seed = int(time.time()) % (2**31-1)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Seed: {seed}")

set_seed(42)

def pick_device() -> torch.device:
    if torch.cuda.is_available():      return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

# ---------- helper functions :)  ----------
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

def _spec_mask(x: torch.Tensor):
    f = x.shape[1]
    fmask = np.random.randint(0, max(1, f // 10))
    fstart = np.random.randint(0, max(1, f - fmask + 1))
    x[:, fstart:fstart + fmask, :] = 0.0
    t = x.shape[-1]
    tmask = np.random.randint(0, max(1, t // 20))
    tstart = np.random.randint(0, max(1, t - tmask + 1))
    x[:, :, tstart:tstart + tmask] = 0.0
    return x

# ---------- Dataset! ----------
class EngineDataset(Dataset):
    def __init__(self, root, class_names=CLASSES, seed=42, augment=True):
        super().__init__()
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.augment = bool(augment)
        self.target_len = _target_len()
        rng = np.random.RandomState(seed)

        per_class: Dict[str, List[str]] = {}
        for cname in class_names:
            p = self.root / cname
            if not p.is_dir():
                raise FileNotFoundError(f"Mangler mappe: {p}")
            files = [str(fp) for fp in p.iterdir()
                     if fp.is_file() and fp.suffix.lower() == ".wav"]
            if not files:
                raise FileNotFoundError(f"Ingen WAV i {p}")
            rng.shuffle(files)
            per_class[cname] = files

        self.items = [(f, self.class_to_idx[c]) for c, flist in per_class.items() for f in flist]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        wav_path, label = self.items[idx]
        y, _ = librosa.load(wav_path, sr=SR, mono=True)
        y = _center_crop_or_pad(y, self.target_len)

        # feature extraction via util.preprocessing
        feat = supervised_preprocess_pipeline(y, SR)
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3 or feat.shape[0] != 1 or feat.shape[1] != N_MELS:
            raise ValueError(f"make_cnn_features returnerte {feat.shape}, forventet [1,{N_MELS},T]")
        x = torch.from_numpy(feat)

        # SpecAugment on the spectrogram
        if self.augment:
            x = _spec_mask(x)
        return x, label

# ---------- Model <3  ----------
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

# ---------- Training (updated with fixed epochs) ----------
def train(train_root,
          class_names=CLASSES, batch_size=32, epochs=25, lr=3e-4):
    device = pick_device()

    train_ds = EngineDataset(train_root, class_names=class_names, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SmallCNN(n_classes=len(class_names)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    save_path = Path(__file__).resolve().parents[1] / "engine_cnn_fixed.pt"
    best_train_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        running_loss, running_acc, n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            bs = xb.size(0)
            running_loss += loss.item() * bs
            running_acc  += (out.argmax(1) == yb).float().sum().item()
            n += bs

        avg_train_loss = running_loss / max(1, n)
        avg_train_acc  = running_acc  / max(1, n)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {avg_train_loss:.4f} | Train acc: {avg_train_acc:.3f}")

        if avg_train_loss < best_train_loss:
            best_train_loss = avg_train_loss
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names}, save_path)

    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    return model

# ---------- MAIN ----------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    TRAIN_ROOT = ROOT / "data" / "train_cut"
    print(f"[DEBUG] train root: {TRAIN_ROOT}")
    model = train(TRAIN_ROOT,
                  class_names=("engine1_good","engine2_broken","engine3_heavyload"),
                  epochs=10, batch_size=32, lr=1e-3)
    