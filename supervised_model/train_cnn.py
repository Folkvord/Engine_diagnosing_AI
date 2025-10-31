import os, sys, random
from pathlib import Path
from typing import List, Dict, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # for util/

import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Globale variables for simplicity 
SR       = 16000
DURATION = 2.0
N_MELS   = 64
CLASSES  = ("engine1_good", "engine2_broken", "engine3_heavyload")

# Getting the preprocess pipeline for supervised from util
from util.preprocessing import supervised_preprocess_pipeline

def set_seed(seed=None): #if using a random seed
    if seed is None:
        import time
        seed = int(time.time()) % (2**31-1)
    import random, numpy as np, torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"[INFO] Seed: {seed}")

set_seed(42)  

# tries so do cuda first, for nvidia gpu
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# Helpful/Support functions :) 

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

# since the clips of audio already all are the same length 
# this isnt nesaccery, but added as a safety 
"""def _random_crop_or_pad(y: np.ndarray, target_len: int) -> np.ndarray:
    cur = y.shape[-1]
    if cur == target_len: return y
    if cur < target_len:
        pad = target_len - cur
        left = np.random.randint(0, pad + 1); right = pad - left
        return np.pad(y, (left, right), mode="reflect")
    start = np.random.randint(0, cur - target_len + 1)
    return y[start:start + target_len]"""

def _spec_mask(x: torch.Tensor):
    # simple SpecAugment, simulates missing or disruptive parts
    # of the spectrograms - model learns to focus on the general
    # patterns exsisting
    f = x.shape[1]
    fmask = np.random.randint(0, max(1, f // 10))
    fstart = np.random.randint(0, max(1, f - fmask + 1))
    x[:, fstart:fstart + fmask, :] = 0.0
    t = x.shape[-1]
    tmask = np.random.randint(0, max(1, t // 20))
    tstart = np.random.randint(0, max(1, t - tmask + 1))
    x[:, :, tstart:tstart + tmask] = 0.0
    return x

# --------------------------------------------------------------
# Dataset

class EngineDataset(Dataset):
    def __init__(self, root, split="train",
                 class_names=CLASSES, augment=True, seed=42):
        super().__init__()
        assert split in ("train", "val")
        self.root = Path(root)
        self.class_names = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}
        self.augment = augment and split == "train"
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

    def __len__(self): return len(self.items)

    # Gets audio, fix length, extract features,
    # and return (x, label) for training
    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        # 
        y, _ = librosa.load(path, sr=SR, mono=True)
        # fast lengde
        #y = _random_crop_or_pad(y, self.target_len) if self.augment else _center_crop_or_pad(y, self.target_len)
        # features
        feat = supervised_preprocess_pipeline(y, SR)             
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim != 3 or feat.shape[0] != 1 or feat.shape[1] != N_MELS:
            raise ValueError(f"make_cnn_features returnerte {feat.shape}, forventet [1,{N_MELS},T]")
        x = torch.from_numpy(feat)
        if self.augment:
            x = _spec_mask(x)
        return x, label

# --------------------------------------------------------------------
# CNN-modell

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
        # feature extractor (convolution)
        self.feat = nn.Sequential(
            ConvBlock(1, 32, drop=0.10),
            ConvBlock(32, 64, drop=0.15),
            ConvBlock(64, 128, drop=0.20),
        )
        # takes the feature maps made, and "translates" them
        # into a classification
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, n_classes),
        )
    # sends the output to a lineær klassifikasjonslag
    def forward(self, x):
        return self.head(self.feat(x))

# --------------------------
# Training/evaluating

def _acc(logits, y): return (logits.argmax(1) == y).float().mean().item()
# adding no gradient so it doesnt save lots og unnessacery info, reducing
# performance
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    tot_l = 0.0; tot_a = 0.0; n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        out = model(xb); loss = loss_fn(out, yb)
        bs = xb.size(0)
        tot_l += loss.item()*bs; tot_a += _acc(out, yb)*bs; n += bs
    return tot_l/n, tot_a/n

def train(train_root, val_root,
          class_names=CLASSES, batch_size=32, epochs=25, lr=3e-4):
    device = pick_device()

    train_ds = EngineDataset(train_root, split="train", class_names=class_names, augment=True)
    val_ds   = EngineDataset(val_root,   split="val",   class_names=class_names, augment=False)

    # num_workers=0 for stability cross enviormment (Sofie has Mac :D)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(n_classes=len(class_names)).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    best_acc, bad, patience = 0.0, 0, 3
    save_path = Path(__file__).resolve().parents[1] / "engine_cnn_best.pt"

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running_loss += loss.item() * xb.size(0)

    # evaluate after each epoch, to see if we can stopp
        val_loss, val_acc = evaluate(model, val_loader, device)
        avg_train_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Train loss: {avg_train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | Val acc: {val_acc:.3f}")

    # early stopping logic if "patience have run out"
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            torch.save({"model_state": model.state_dict(),
                        "class_names": class_names}, save_path)
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping triggered.")
                break


    # loads beste weights back 
    if save_path.exists():
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
    return model

# --------------------------
# MAIN

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    TRAIN_ROOT = ROOT / "data" / "train_cut"   
    VAL_ROOT   = ROOT / "data" / "test_cut"
    print(f"[DEBUG] train root: {TRAIN_ROOT}")
    print(f"[DEBUG] val root:   {VAL_ROOT}")
    model = train(TRAIN_ROOT, VAL_ROOT,
                  class_names=("engine1_good","engine2_broken","engine3_heavyload"),
                  epochs=40, batch_size=32, lr=1e-3)