"""
Shared helpers used by all five notebooks.
Keeping paths, class names, the data loader and the evaluation routine in ONE place
guarantees that every model is trained/tested on the same split and scored the same way.
"""
import os, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)

# --------------------------------------------------------------------------- paths
def _find_root():
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "src" / "common.py").exists():
            return p
    return here

ROOT       = _find_root()
RAW_DIR    = ROOT / "data" / "raw" / "natural-images"
PROC_DIR   = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
FIG_DIR    = ROOT / "results" / "figures"
MET_DIR    = ROOT / "results" / "metrics"
for _d in (PROC_DIR, MODELS_DIR, FIG_DIR, MET_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- settings
# Quick mode (set NI_QUICK=1 before launching Jupyter) = tiny subset + 2-3 epochs,
# useful to check that everything runs before the full training.
QUICK = os.environ.get("NI_QUICK", "0") == "1"

CLASSES   = ["airplane", "car", "cat", "dog", "flower", "fruit", "motorbike", "person"]
IMG_SIZE  = 64                       # every image is resized to 64 x 64
SEED      = 42
TEST_SIZE = 0.20
MAX_PER_CLASS = 40 if QUICK else None        # None = use ALL images (6,899 in total)
EPOCHS = dict(mlp=3, cnn=3, resnet=2, vgg=2) if QUICK else dict(mlp=30, cnn=25, resnet=8, vgg=8)
USE_PRETRAINED = not QUICK                   # ImageNet weights (downloaded on first use)
TL_SIZE = 32 if QUICK else 96                # input size fed to ResNet50 / VGG16
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# --------------------------------------------------------------------------- data
def class_dir(cls):
    """Return the folder of one class (also works if the zip created a nested folder)."""
    direct = RAW_DIR / cls
    if direct.is_dir():
        return direct
    for p in RAW_DIR.rglob(cls):
        if p.is_dir():
            return p
    raise FileNotFoundError(
        f"Folder for class '{cls}' not found in {RAW_DIR}.\n"
        "Download the Natural Images dataset and place the 8 class folders "
        "inside data/raw/natural-images/")

def list_images(cls, max_n=None):
    files = sorted(p for p in class_dir(cls).iterdir() if p.suffix.lower() in IMG_EXT)
    return files[:max_n] if max_n else files

def load_dataset(size=IMG_SIZE, max_per_class=MAX_PER_CLASS):
    """Load every image as RGB uint8 array (N, size, size, 3) + integer labels (N,)."""
    X, y, bad = [], [], 0
    for label, cls in enumerate(CLASSES):
        for f in list_images(cls, max_per_class):
            try:
                img = Image.open(f).convert("RGB").resize((size, size), Image.BILINEAR)
                X.append(np.asarray(img, dtype=np.uint8)); y.append(label)
            except Exception:
                bad += 1
    if bad:
        print(f"Skipped {bad} unreadable files")
    return np.stack(X), np.array(y)

def load_split():
    """Load the train/test split created in notebook 02."""
    f = PROC_DIR / "dataset_64.npz"
    if not f.exists():
        raise FileNotFoundError("Run 02_preprocessing.ipynb first (it creates the split).")
    d = np.load(f)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]

# --------------------------------------------------------------------------- figures
def save_fig(name, dpi=150):
    path = FIG_DIR / name
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print("saved ->", path.relative_to(ROOT))

def show_samples(X, y, n_per_class=5, title=None, fname=None):
    fig, axes = plt.subplots(len(CLASSES), n_per_class, figsize=(1.6 * n_per_class, 1.6 * len(CLASSES)))
    for r, cls in enumerate(CLASSES):
        idx = np.where(y == r)[0][:n_per_class]
        for c in range(n_per_class):
            ax = axes[r, c]; ax.axis("off")
            if c < len(idx):
                ax.imshow(X[idx[c]], cmap="gray" if X[idx[c]].ndim == 2 else None)
            if c == 0:
                ax.set_title(cls, fontsize=9, loc="left")
    if title: fig.suptitle(title, y=1.0)
    plt.tight_layout()
    if fname: save_fig(fname)
    plt.show()

# --------------------------------------------------------------------------- evaluation
def slug(name):
    return name.lower().replace(" ", "_").replace("/", "_")

def evaluate(name, family, y_true, y_pred, train_time=None, plot=True):
    """Compute metrics, print the report, plot + save the confusion matrix, store a json."""
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    res = dict(model=name, family=family, accuracy=acc, precision=p, recall=r, f1_score=f1,
               train_time_s=None if train_time is None else round(train_time, 1))
    report = classification_report(y_true, y_pred, labels=range(len(CLASSES)),
                                   target_names=CLASSES, zero_division=0)
    print(f"=== {name} ===   accuracy={acc:.4f}  precision={p:.4f}  recall={r:.4f}  F1={f1:.4f}")
    print(report)
    (MET_DIR / f"{slug(name)}.json").write_text(json.dumps(res, indent=2))
    (MET_DIR / f"{slug(name)}_report.txt").write_text(report)
    if plot:
        plot_confusion(name, y_true, y_pred)
    return res

def plot_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {name}")
    plt.colorbar(im, fraction=0.046, label="row-normalised")
    plt.tight_layout(); save_fig(f"cm_{slug(name)}.png"); plt.show()

def plot_history(history, name):
    h = history.history
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(h["loss"], label="train"); ax[0].plot(h["val_loss"], label="validation")
    ax[0].set_title(f"{name} - loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(h["accuracy"], label="train"); ax[1].plot(h["val_accuracy"], label="validation")
    ax[1].set_title(f"{name} - accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); save_fig(f"history_{slug(name)}.png"); plt.show()

def show_predictions(X, y_true, y_pred, name, n=12):
    idx = np.random.RandomState(0).choice(len(X), min(n, len(X)), replace=False)
    fig, axes = plt.subplots(2, len(idx) // 2, figsize=(2 * (len(idx) // 2), 4.6))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(X[i]); ax.axis("off")
        ok = y_true[i] == y_pred[i]
        ax.set_title(CLASSES[y_pred[i]], color="green" if ok else "red", fontsize=9)
    fig.suptitle(f"{name}: predictions (green = correct, red = wrong)")
    plt.tight_layout(); save_fig(f"predictions_{slug(name)}.png"); plt.show()

def compare():
    """Merge every saved metrics json into all_models_comparison.csv."""
    rows = [json.loads(p.read_text()) for p in MET_DIR.glob("*.json")]
    df = pd.DataFrame(rows).sort_values("f1_score", ascending=False).reset_index(drop=True)
    df.to_csv(ROOT / "results" / "all_models_comparison.csv", index=False)
    return df

def plot_comparison(df):
    d = df.sort_values("accuracy")
    colors = ["#4C78A8" if f == "Traditional ML" else "#F58518" for f in d["family"]]
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(d) + 1.5))
    ax.barh(d["model"], d["accuracy"], color=colors)
    for i, v in enumerate(d["accuracy"]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Test accuracy")
    ax.set_title("All models - test accuracy (blue = HOG + ML, orange = deep learning)")
    plt.tight_layout(); save_fig("all_models_comparison.png"); plt.show()
