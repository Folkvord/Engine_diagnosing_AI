# supervised_model/test_cnn.py
import os, sys, torch, librosa
import numpy as np
import matplotlib
matplotlib.use('Agg')  
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# allow importing from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from supervised_model.train_cnn import SmallCNN
from util.preprocessing import supervised_preprocess_pipeline

if not callable(supervised_preprocess_pipeline):
    raise ImportError("Fant ikke supervised_preprocess_pipeline i util/preprocessing.py")

# Setting some global variabels for consitancy 
SR = 16000
DURATION = 2.0
N_MELS = 64
BATCH_SIZE = 32
SEED = 42

def _set_seed(s=SEED):
    import random
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
_set_seed()

# support functions <3 tihi 
def _target_len(): return int(SR * DURATION)

def _center_crop_or_pad(y, L):
    cur = y.shape[-1]
    if cur == L: return y
    if cur < L:
        pad = L - cur; left = pad // 2; right = pad - left
        return np.pad(y, (left, right), mode="reflect")
    start = (cur - L) // 2
    return y[start:start + L]

# dataset (mirror training IO path)
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
        # same loading policy as in  training
        y, _ = librosa.load(path, sr=SR, mono=True)
        # same length policy as validation: center crop/pad
        y = _center_crop_or_pad(y, self.target_len)

        # usin the supvervised adapter 
        feat = supervised_preprocess_pipeline(y, SR)  
        feat = np.asarray(feat, dtype=np.float32)

        # sanity checks:
        # find the three most commen fall throughs in preprossering 
        # of the spectograms: like catching collapsed features
        if feat.ndim != 3 or feat.shape[0] != 1 or feat.shape[1] != N_MELS:
            raise ValueError(f"Bad feature shape {feat.shape}, expected [1,{N_MELS},T]")
        if not np.isfinite(feat).all():
            raise RuntimeError(f"NaN/Inf i features for {path}")
        if np.std(feat) < 1e-7:
            raise RuntimeError(f"Nesten konstant feature for {path} (std={np.std(feat):.2e})")

        x = torch.from_numpy(feat)  # [1, n_mels, T]
        return x, label

# plotting for the graphics :)) love graphics
def plot_confusion(cm, classes, save_path="confusion_matrix.png", 
                   ui_font="Nunito", num_font="Inter",
                   title_size=22, tick_size=13, text_size=12):


    pastel_pink_purple = LinearSegmentedColormap.from_list(
        "pastel_pink_purple", ["#f9b4e9", "#e0b7e7", "#c9a7e7"], N=256
    )

    # font for the words
    matplotlib.rcParams.update({
        "font.family": ui_font,
        "font.size": tick_size,
        "axes.titleweight": "semibold",
    })
    # font for the numbers
    num_fp = FontProperties(family=num_font)  
    
    # 
    parsed = [str(c).split("_")[-1].lower() for c in classes]
    if set(parsed) >= {"good", "broken", "heavyload"}:
        display_labels = ["good", "broken", "heavyload"]
    # fallback
    else:
        display_labels = classes  

    n = len(display_labels)

    # The figure with padding for esthetics 
    fig, ax = plt.subplots(figsize=(7.0, 6.2), constrained_layout=False)
    im = ax.imshow(cm, interpolation="nearest", cmap=pastel_pink_purple, aspect="auto")

    ax.set_title("CONFUSION MATRIX", fontsize=title_size, fontweight="bold", pad=12)

    ticks = np.arange(n)
    ax.set_xticks(ticks, display_labels, rotation=0, ha="center", fontsize=tick_size)
    ax.set_yticks(ticks, display_labels, fontsize=tick_size)

    # Colorbar
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # the numbers for the matrix
    thresh = cm.max() / 2 if cm.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                fontsize=text_size,
                fontproperties=num_fp,            
                color="white" if cm[i, j] > thresh else "black",
            )

    ax.set_xlabel("PREDICTED", fontsize=tick_size+1, labelpad=8)
    ax.set_ylabel("TRUE", fontsize=tick_size+1, labelpad=8)

    # spacing 
    fig.subplots_adjust(left=0.20, right=0.95, bottom=0.20, top=0.88)

    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[INFO] Saved confusion matrix -> {save_path}")
    plt.close(fig)

# --- main test ---
def test_model(model_path, data_root):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")
    torch.set_grad_enabled(False)
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Model:  {model_path}")
    print(f"[INFO] Data:   {data_root}")

    ckpt = torch.load(model_path, map_location=device)
    class_names = tuple(ckpt["class_names"])
    print(f"[INFO] Classes: {class_names}")

    ds = EngineDataset(data_root, class_names)
    pin = bool(torch.cuda.is_available())  # pin memory only on CUDA
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=pin)

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

    # quick distribution sanity
    from collections import Counter
    print("[INFO] Test dist:", Counter(y_true))
    print("[INFO] Pred dist:", Counter(y_pred))

    # accuracy + confusion matrix
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    acc = (np.trace(cm) / np.sum(cm))
    print(f"[RESULT] Total accuracy: {acc*100:.2f}%")

    out_png = Path(model_path).with_suffix(".confmat.png")
    plot_confusion(cm, class_names, save_path=str(out_png))

if __name__ == "__main__":
    PROJECT   = Path(__file__).resolve().parents[1]
    MODEL     = PROJECT / "engine_cnn_best.pt"
    TEST_DATA = PROJECT / "data" / "test_cut"   # keep in sync with train
    test_model(MODEL, TEST_DATA)