"""Generates the five notebooks. Run:  python tools/build_notebooks.py"""
import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

def md(t):   return nbf.v4.new_markdown_cell(t.strip("\n"))
def code(t): return nbf.v4.new_code_cell(t.strip("\n"))

def save(name, cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nbf.write(nb, NB_DIR / name)
    print("written", name, len(cells), "cells")

HEAD = '''import sys; sys.path.append("..")          # lets the notebook import src/common.py
from src.common import *
import matplotlib.pyplot as plt, numpy as np, pandas as pd
print("Project root:", ROOT)
print("QUICK mode:", QUICK)'''

# ======================================================================== 01
nb1 = [
md('''# 01 - Data exploration
**Goal:** understand the *Natural Images* dataset before any modelling: how many images per class, what they look like, how big they are, and whether any file is broken.

![pipeline](../docs/00_project_pipeline.png)

| Step | What we do |
|---|---|
| 1 | Count images per class (is the dataset balanced?) |
| 2 | Look at random samples of every class |
| 3 | Check image sizes and colour modes |
| 4 | Check that every file can be opened |
| 5 | Compute the *average image* of each class |'''),
code(HEAD + '''
from PIL import Image'''),

md('## Step 1 - Class distribution'),
code('''counts = {c: len(list_images(c)) for c in CLASSES}
df_counts = pd.DataFrame({"class": list(counts), "images": list(counts.values())})
df_counts["share_%"] = (100 * df_counts["images"] / df_counts["images"].sum()).round(1)
display(df_counts)
print("Total images:", df_counts["images"].sum())

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(df_counts["class"], df_counts["images"], color=plt.cm.tab10.colors[:len(CLASSES)])
ax.bar_label(bars)
ax.set_title("Number of images per class"); ax.set_ylabel("images")
plt.xticks(rotation=30); plt.tight_layout()
save_fig("01_class_distribution.png"); plt.show()'''),

md('## Step 2 - Random samples of every class'),
code('''rng = np.random.RandomState(SEED)
n_show = 6
fig, axes = plt.subplots(len(CLASSES), n_show, figsize=(2 * n_show, 2 * len(CLASSES)))
for r, cls in enumerate(CLASSES):
    files = list_images(cls)
    for c, i in enumerate(rng.choice(len(files), n_show, replace=False)):
        ax = axes[r, c]; ax.imshow(Image.open(files[i]).convert("RGB")); ax.axis("off")
        if c == 0: ax.set_title(cls, loc="left", fontsize=11)
plt.tight_layout(); save_fig("01_random_samples.png"); plt.show()'''),

md('## Step 3 - Image sizes and colour modes\nThe CNN/MLP/HOG all need a *fixed* input size, so we must know how much the sizes vary.'),
code('''rows = []
for cls in CLASSES:
    for f in list_images(cls)[:300]:            # first 300 per class is enough for statistics
        with Image.open(f) as im:               # opening is lazy: only the header is read
            rows.append((cls, im.width, im.height, im.mode))
df_sizes = pd.DataFrame(rows, columns=["class", "width", "height", "mode"])
display(df_sizes.groupby("class")[["width", "height"]].agg(["min", "mean", "max"]).round(0))
print(df_sizes["mode"].value_counts())

fig, ax = plt.subplots(figsize=(6, 5))
for i, cls in enumerate(CLASSES):
    d = df_sizes[df_sizes["class"] == cls]
    ax.scatter(d["width"], d["height"], s=10, alpha=.5, label=cls, color=plt.cm.tab10.colors[i])
ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)"); ax.legend(fontsize=7)
ax.set_title("Image sizes"); plt.tight_layout(); save_fig("01_image_sizes.png"); plt.show()'''),

md('## Step 4 - Integrity check'),
code('''corrupt = []
for cls in CLASSES:
    for f in list_images(cls):
        try:
            with Image.open(f) as im: im.verify()
        except Exception:
            corrupt.append(f)
print("Unreadable files:", len(corrupt))
for f in corrupt[:10]: print("  ", f)'''),

md('## Step 5 - Average image of each class'),
code('''fig, axes = plt.subplots(1, len(CLASSES), figsize=(2 * len(CLASSES), 2.4))
for ax, cls in zip(axes, CLASSES):
    arr = [np.asarray(Image.open(f).convert("RGB").resize((64, 64)), dtype=np.float32)
           for f in list_images(cls)[:200]]
    ax.imshow(np.mean(arr, axis=0) / 255); ax.set_title(cls, fontsize=9); ax.axis("off")
plt.suptitle("Mean image per class (first 200 images)"); plt.tight_layout()
save_fig("01_mean_images.png"); plt.show()'''),

md('''## Conclusions
* The dataset has **8 classes**; check the table above for the balance (it is only mildly imbalanced).
* Images have **different sizes** -> they must be resized to one common size (64 x 64) in the next notebook.
* Mean images show that colour is already informative for some classes (e.g. blue sky for *airplane*), which is why the deep-learning models use RGB while HOG uses grayscale *shape* information.'''),
]
save("01_data_exploration.ipynb", nb1)

# ======================================================================== 02
nb2 = [
md('''# 02 - Preprocessing
**Goal:** turn the raw images of different sizes into clean arrays and create **one fixed train/test split** that *all ten models* will share.

| Step | Operation | Why |
|---|---|---|
| 1 | Show every operation on ONE image | understand what each step does |
| 2 | Load + resize all images to 64 x 64 | fixed input size |
| 3 | Grayscale version | HOG works on intensity/shape |
| 4 | Stratified 80 / 20 split | same class proportions in train and test |
| 5 | Save to `data/processed/dataset_64.npz` | notebooks 03-05 simply load it |

Pixel values are stored as `uint8` (0-255). Normalisation (/255) is done when a model needs it - this keeps the file small.'''),
code(HEAD + '''
from PIL import Image
from skimage.color import rgb2gray
from sklearn.model_selection import train_test_split'''),

md('## Step 1 - Every operation on one example image'),
code('''f = list_images("cat")[0]
raw = Image.open(f).convert("RGB")
resized = raw.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
arr_u8  = np.asarray(resized)                     # uint8, 0..255
arr_f   = arr_u8 / 255.0                          # float, 0..1   (normalisation)
gray    = rgb2gray(arr_f)                         # 64 x 64, one channel

fig, ax = plt.subplots(1, 5, figsize=(17, 3.4))
ax[0].imshow(raw);               ax[0].set_title(f"1. original {raw.size[0]}x{raw.size[1]} RGB")
ax[1].imshow(resized);           ax[1].set_title(f"2. resized {IMG_SIZE}x{IMG_SIZE} RGB")
ax[2].imshow(gray, cmap="gray"); ax[2].set_title("3. grayscale (for HOG)")
ax[3].hist(arr_u8.ravel(), bins=32, color="gray"); ax[3].set_title("pixel values 0-255 (uint8)")
ax[4].hist(arr_f.ravel(),  bins=32, color="steelblue"); ax[4].set_title("after /255 -> 0-1")
for a in ax[:3]: a.axis("off")
plt.tight_layout(); save_fig("02_preprocessing_steps.png"); plt.show()
print("array shape:", arr_u8.shape, " dtype:", arr_u8.dtype, " -> gray shape:", gray.shape)'''),

md('## Step 2 - Load and resize the whole dataset'),
code('''X, y = load_dataset()            # (N, 64, 64, 3) uint8 and (N,) labels
print("X:", X.shape, X.dtype, f"-> {X.nbytes/1e6:.0f} MB")
print("y:", y.shape, "classes:", np.bincount(y))'''),
code('''show_samples(X, y, 5, "Resized 64x64 RGB samples", "02_resized_samples.png")'''),

md('## Step 3 - Grayscale preview (this is what HOG will "see")'),
code('''Xg_preview = np.stack([rgb2gray(img / 255.0) for img in X[::10]])
yg_preview = y[::10]
show_samples(Xg_preview, yg_preview, 5, "Grayscale samples", "02_gray_samples.png")'''),

md('''## Step 4 - Stratified train/test split
`stratify=y` keeps the same class proportions in both parts; `random_state` makes the split reproducible.
Because **every notebook loads this same file**, the ten models are compared on exactly the same test images.'''),
code('''X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
print("train:", X_train.shape, " test:", X_test.shape)

tr = np.bincount(y_train, minlength=len(CLASSES)); te = np.bincount(y_test, minlength=len(CLASSES))
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.4; xs = np.arange(len(CLASSES))
ax.bar(xs - w/2, tr, w, label=f"train ({len(y_train)})"); ax.bar(xs + w/2, te, w, label=f"test ({len(y_test)})")
ax.set_xticks(xs); ax.set_xticklabels(CLASSES, rotation=30); ax.legend(); ax.set_title("Train / test class distribution")
plt.tight_layout(); save_fig("02_train_test_split.png"); plt.show()'''),

md('## Step 5 - Save'),
code('''np.savez_compressed(PROC_DIR / "dataset_64.npz",
                    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
d = np.load(PROC_DIR / "dataset_64.npz")
print({k: d[k].shape for k in d.files})
print("saved ->", (PROC_DIR / "dataset_64.npz").relative_to(ROOT))'''),
md('''## Next
`03_hog_traditional_ml.ipynb` converts these images to HOG features and trains the six traditional models.'''),
]
save("02_preprocessing.ipynb", nb2)

# ======================================================================== 03
def model_md(n, name, img, text):
    return md(f"## 4.{n} - {name}\n![{name}](../docs/model_steps/{img})\n\n{text}")

nb3 = [
md('''# 03 - HOG features + six traditional ML models
**Pipeline:** `64x64 RGB image -> grayscale -> HOG (1764 numbers) -> StandardScaler -> classifier`

All six classifiers receive **exactly the same features and the same train/test split**, so the comparison is fair.

![pipeline](../docs/00_project_pipeline.png)

| # | Model | Idea in one line |
|---|---|---|
| 1 | SVM | finds the maximum-margin boundary (RBF kernel) |
| 2 | KNN | votes among the k most similar training images |
| 3 | Random Forest | many decorrelated trees vote |
| 4 | Decision Tree | a single tree of if/else questions on features |
| 5 | Logistic Regression | linear scores + softmax probabilities |
| 6 | Naive Bayes | Bayes rule with independent Gaussian features |'''),
code(HEAD + '''
import time, joblib
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_fscore_support

X_train, X_test, y_train, y_test = load_split()
print("train", X_train.shape, "test", X_test.shape)'''),

md('''## Step 1 - What is HOG? (one image, step by step)
1. Convert to grayscale.
2. Compute the **gradient** (how fast intensity changes) in x and y -> magnitude + direction = edges.
3. Split the image into **8x8-pixel cells**; in each cell build a histogram of **9 gradient orientations**, weighted by magnitude.
4. Group cells into **2x2 blocks** and normalise (L2-Hys) -> robust to lighting.
5. Concatenate everything: 7x7 blocks x 4 cells x 9 bins = **1764 features** per image.

![HOG](../docs/00_hog_step_by_step.png)'''),
code('''HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")

img  = X_train[0]
gray = rgb2gray(img / 255.0)
gy, gx = np.gradient(gray)
magnitude   = np.hypot(gx, gy)
orientation = (np.rad2deg(np.arctan2(gy, gx)) % 180)
fd, hog_img = hog(gray, visualize=True, **HOG_PARAMS)

fig, ax = plt.subplots(1, 5, figsize=(17, 3.5))
ax[0].imshow(img);                                  ax[0].set_title(f"RGB 64x64 ({CLASSES[y_train[0]]})")
ax[1].imshow(gray, cmap="gray");                    ax[1].set_title("1. grayscale")
ax[2].imshow(magnitude, cmap="magma");              ax[2].set_title("2. gradient magnitude (edges)")
ax[3].imshow(orientation, cmap="hsv");              ax[3].set_title("2. gradient direction 0-180 deg")
ax[4].imshow(exposure.rescale_intensity(hog_img, in_range=(0, 10)), cmap="gray")
ax[4].set_title("3-4. HOG cells / blocks")
for a in ax: a.axis("off")
plt.tight_layout(); save_fig("03_hog_step_by_step.png"); plt.show()
print("HOG vector length:", fd.shape[0], "= 7 x 7 blocks x 2 x 2 cells x 9 orientations")'''),

md('## Step 2 - Extract HOG for every image (same code for train and test)'),
code('''def extract_hog(images, batch=500):
    """images: (N,64,64,3) uint8 -> (N,1764) float32. Batched to save RAM."""
    out = []
    for s in range(0, len(images), batch):
        g = rgb2gray(images[s:s + batch] / 255.0)
        out.extend(hog(im, **HOG_PARAMS) for im in g)
    return np.asarray(out, dtype=np.float32)

t = time.time()
H_train = extract_hog(X_train); H_test = extract_hog(X_test)
print("H_train", H_train.shape, " H_test", H_test.shape, f"({time.time()-t:.0f}s)")
np.savez_compressed(PROC_DIR / "hog_features.npz", H_train=H_train, H_test=H_test,
                    y_train=y_train, y_test=y_test)'''),

md('''## Step 3 - Standardise
SVM, KNN and Logistic Regression are sensitive to feature scale. The scaler is **fitted on the training set only** (fitting on test data would leak information) and then applied to both.'''),
code('''scaler = StandardScaler().fit(H_train)
S_train, S_test = scaler.transform(H_train), scaler.transform(H_test)
joblib.dump(scaler, MODELS_DIR / "hog_scaler.joblib")

fig, ax = plt.subplots(1, 2, figsize=(9, 3))
ax[0].hist(H_train.ravel(), bins=60, color="gray");      ax[0].set_title("HOG values (raw)")
ax[1].hist(S_train.ravel(), bins=60, color="steelblue"); ax[1].set_title("after StandardScaler"); ax[1].set_xlim(-6, 6)
plt.tight_layout(); save_fig("03_scaling.png"); plt.show()'''),

md('''## Step 4 - Train and evaluate the six models
Helper that trains, times, evaluates (accuracy / precision / recall / F1 + confusion matrix) and saves each model.'''),
code('''FAMILY = "Traditional ML"
results, preds = [], {}

def run(name, model):
    t = time.time(); model.fit(S_train, y_train); train_time = time.time() - t
    y_pred = model.predict(S_test); preds[name] = y_pred
    joblib.dump(model, MODELS_DIR / f"{slug(name)}.joblib")
    results.append(evaluate(name, FAMILY, y_test, y_pred, train_time))
    return model

def block_map(vec):
    """Sum a 1764-long HOG-sized vector over cells/orientations -> 7x7 spatial map."""
    return np.asarray(vec).reshape(7, 7, 2, 2, 9).sum(axis=(2, 3, 4))'''),

model_md(1, "SVM", "01_svm.png",
 "RBF-kernel SVM, `C=10`. Below: the 2-D PCA view of the HOG space and the number of support vectors per class (the images that define the boundary)."),
code('''svm = run("SVM", SVC(kernel="rbf", C=10, gamma="scale"))

pca = PCA(2, random_state=SEED).fit(S_train); P = pca.transform(S_train)
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for i, cls in enumerate(CLASSES):
    ax[0].scatter(*P[y_train == i].T, s=5, alpha=.5, label=cls, color=plt.cm.tab10.colors[i])
ax[0].scatter(*P[svm.support_].T, s=18, facecolors="none", edgecolors="k", linewidths=.3)
ax[0].set_title("HOG features in 2-D PCA (circled = support vectors)"); ax[0].legend(markerscale=3, fontsize=7)
ax[1].bar(CLASSES, svm.n_support_, color=plt.cm.tab10.colors[:8]); ax[1].set_title("support vectors per class")
plt.setp(ax[1].get_xticklabels(), rotation=30); plt.tight_layout(); save_fig("03_svm_process.png"); plt.show()'''),

model_md(2, "KNN", "02_knn.png",
 "The number of neighbours `k` is chosen by 3-fold cross-validation **on the training set only**."),
code('''ks = [1, 3, 5, 7, 9, 11, 15]
cv = [cross_val_score(KNeighborsClassifier(k), S_train, y_train, cv=3).mean() for k in ks]
best_k = ks[int(np.argmax(cv))]
plt.figure(figsize=(6, 3.5)); plt.plot(ks, cv, "o-"); plt.axvline(best_k, color="r", ls="--")
plt.xlabel("k"); plt.ylabel("CV accuracy"); plt.title(f"KNN: choosing k (best k = {best_k})")
plt.tight_layout(); save_fig("03_knn_k_selection.png"); plt.show()
knn = run("KNN", KNeighborsClassifier(n_neighbors=best_k))'''),

model_md(3, "Random Forest", "03_random_forest.png",
 "200 trees, each trained on a bootstrap sample with random feature subsets. The importance map shows *which image regions* the forest relies on."),
code('''rf = run("Random Forest", RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=SEED))

imp = block_map(rf.feature_importances_)
fig, ax = plt.subplots(1, 2, figsize=(9, 4))
ax[0].imshow(X_train[0]); ax[0].set_title("example image"); ax[0].axis("off")
im = ax[1].imshow(imp, cmap="hot"); ax[1].set_title("RF importance per HOG block (7x7)")
plt.colorbar(im, ax=ax[1], fraction=.046); plt.tight_layout(); save_fig("03_rf_importance.png"); plt.show()'''),

model_md(4, "Decision Tree", "04_decision_tree.png",
 "A single tree. Depth is selected by 3-fold CV on the training set (an unlimited tree memorises the training data)."),
code('''depths = [3, 5, 8, 12, 16, None]
cv = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), S_train, y_train, cv=3).mean() for d in depths]
best_d = depths[int(np.argmax(cv))]
plt.figure(figsize=(6, 3.5)); plt.plot([str(d) for d in depths], cv, "o-")
plt.xlabel("max_depth"); plt.ylabel("CV accuracy"); plt.title(f"Decision tree: choosing depth (best = {best_d})")
plt.tight_layout(); save_fig("03_dt_depth_selection.png"); plt.show()

dt = run("Decision Tree", DecisionTreeClassifier(max_depth=best_d, random_state=SEED))
plt.figure(figsize=(16, 6)); plot_tree(dt, max_depth=2, class_names=CLASSES, filled=True, fontsize=7, impurity=False)
plt.title("Top 3 levels of the tree (X[i] = i-th HOG feature)"); save_fig("03_dt_tree_top.png"); plt.show()'''),

model_md(5, "Logistic Regression", "05_logistic_regression.png",
 "Multinomial logistic regression: one weight vector per class, softmax turns the 8 scores into probabilities. The maps show where in the image each class's weights are large."),
code('''lr = run("Logistic Regression", LogisticRegression(max_iter=1000, C=1.0))

maps = np.abs(lr.coef_).reshape(len(CLASSES), 7, 7, 2, 2, 9).mean(axis=(3, 4, 5))
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for a, m, cls in zip(axes.ravel(), maps, CLASSES):
    a.imshow(m, cmap="viridis"); a.set_title(cls); a.axis("off")
plt.suptitle("|weights| per HOG block for each class"); plt.tight_layout()
save_fig("03_lr_weights.png"); plt.show()'''),

model_md(6, "Naive Bayes", "06_naive_bayes.png",
 "Gaussian Naive Bayes: for each class it learns a mean and variance per HOG feature, then multiplies the (assumed independent) likelihoods with the class prior."),
code('''nb = run("Naive Bayes", GaussianNB())

maps = nb.theta_.reshape(len(CLASSES), 7, 7, 2, 2, 9).mean(axis=(3, 4, 5))
fig, axes = plt.subplots(2, 4, figsize=(11, 5.5))
for a, m, cls in zip(axes.ravel(), maps, CLASSES):
    a.imshow(m, cmap="coolwarm", vmin=-.5, vmax=.5); a.set_title(cls); a.axis("off")
plt.suptitle("Mean (standardised) HOG energy per block learned for each class"); plt.tight_layout()
save_fig("03_nb_class_means.png"); plt.show()'''),

md('## Step 5 - Compare the six traditional models'),
code('''df = pd.DataFrame(results).sort_values("f1_score", ascending=False).reset_index(drop=True)
display(df.round(4))

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(df)); w = 0.2
for i, m in enumerate(["accuracy", "precision", "recall", "f1_score"]):
    ax.bar(x + (i - 1.5) * w, df[m], w, label=m)
ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=25); ax.set_ylim(0, 1); ax.legend(ncol=4, fontsize=8)
ax.set_title("Traditional ML (HOG) - test metrics"); plt.tight_layout(); save_fig("03_traditional_comparison.png"); plt.show()

# per-class F1
f1 = pd.DataFrame({n: precision_recall_fscore_support(y_test, p, labels=range(len(CLASSES)), zero_division=0)[2]
                   for n, p in preds.items()}, index=CLASSES)
plt.figure(figsize=(8, 4.5)); plt.imshow(f1.values, cmap="YlGn", vmin=0, vmax=1)
plt.xticks(range(f1.shape[1]), f1.columns, rotation=30, ha="right"); plt.yticks(range(len(CLASSES)), CLASSES)
for i in range(f1.shape[0]):
    for j in range(f1.shape[1]): plt.text(j, i, f"{f1.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("F1-score per class"); plt.colorbar(); plt.tight_layout(); save_fig("03_f1_per_class.png"); plt.show()'''),
md('''## Next
`04_deep_learning.ipynb` trains the MLP and the CNN directly on the images. All metrics are already stored in `results/metrics/`.'''),
]
save("03_hog_traditional_ml.ipynb", nb3)

# ======================================================================== 04
nb4 = [
md("""# 04 - Deep learning: MLP (ANN) and CNN  *(no TensorFlow needed)*
This version of the notebook works on PCs where **TensorFlow is blocked** (e.g. Windows App Control).

| Model | Library | Idea |
|---|---|---|
| MLP / ANN | scikit-learn (`MLPClassifier`) | flatten the RGB pixels -> fully-connected layers |
| CNN | PyTorch (**optional**) | convolution filters learn local patterns |

* The **MLP always runs** (scikit-learn is already installed).
* The **CNN section runs only if PyTorch can be imported**. If it is missing or blocked, that cell prints a message and is skipped - the rest of the project is unaffected.

Both use the **same train/test split** as notebook 03 and are scored with the same `evaluate()` function."""),
code(HEAD + """
import time, joblib
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = load_split()
Xtr = X_train.reshape(len(X_train), -1).astype("float32") / 255.0     # (N, 12288) pixels scaled to 0-1
Xte = X_test.reshape(len(X_test), -1).astype("float32") / 255.0
print("train", Xtr.shape, "test", Xte.shape, "| epochs (max):", EPOCHS)"""),

md("""## 1 - MLP / ANN
![MLP](../docs/model_steps/07_mlp_ann.png)

Flatten 64x64x3 = 12,288 numbers -> Dense 512 -> Dense 256 -> softmax(8). Optimiser Adam, early stopping on a 15 % validation part of the training set.
A MLP ignores the 2-D structure of the image, so it is usually clearly weaker than a CNN."""),
code("""mlp = MLPClassifier(hidden_layer_sizes=(512, 256), activation="relu", solver="adam",
                    alpha=1e-4, batch_size=64, learning_rate_init=1e-3,
                    max_iter=EPOCHS["mlp"], early_stopping=True, validation_fraction=0.15,
                    n_iter_no_change=5, random_state=SEED, verbose=True)
t = time.time(); mlp.fit(Xtr, y_train); tt = time.time() - t
print(f"trained {mlp.n_iter_} epochs in {tt:.0f}s")"""),
code("""fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
ax[0].plot(mlp.loss_curve_); ax[0].set_title("MLP - training loss"); ax[0].set_xlabel("epoch")
ax[1].plot(mlp.validation_scores_); ax[1].set_title("MLP - validation accuracy"); ax[1].set_xlabel("epoch")
plt.tight_layout(); save_fig("history_mlp.png"); plt.show()"""),
code("""y_pred = mlp.predict(Xte)
evaluate("MLP", "Deep Learning", y_test, y_pred, tt)
show_predictions(X_test, y_test, y_pred, "MLP")
joblib.dump(mlp, MODELS_DIR / "mlp.joblib")"""),

md("""## 2 - CNN (PyTorch, optional)
![CNN](../docs/model_steps/08_cnn.png)

Three blocks of *[Conv-BatchNorm-ReLU] x 2 -> MaxPool -> Dropout* with 32/64/128 filters, global average pooling and a small classifier,
plus light data augmentation (random horizontal flip) during training.

Install PyTorch (CPU) if you want to run it:  `python -m pip install torch`
If Windows blocks it, this cell just prints a message and you can continue."""),
code("""try:
    import torch, torch.nn as nn
    TORCH_OK = True
    print("PyTorch", torch.__version__)
except Exception as e:                       # ImportError, or a DLL blocked by Windows
    TORCH_OK = False
    print("PyTorch is not available here ->", type(e).__name__, "\\nThe CNN section is skipped.")"""),
code("""if TORCH_OK:
    torch.manual_seed(SEED); np.random.seed(SEED)

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.25))

    class CNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                      nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.4),
                                      nn.Linear(128, len(CLASSES)))
        def forward(self, x): return self.head(self.features(x))

    to_t = lambda X: torch.tensor(X.astype("float32") / 255.0).permute(0, 3, 1, 2)   # NHWC -> NCHW
    Ttr, Tte = to_t(X_train), to_t(X_test)
    ytr, yte = torch.tensor(y_train), torch.tensor(y_test)
    perm = torch.randperm(len(Ttr)); n_val = int(0.15 * len(Ttr))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    cnn = CNN(); opt = torch.optim.Adam(cnn.parameters(), 1e-3); lossf = nn.CrossEntropyLoss()
    hist = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    def run_eval(X, y):
        cnn.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 256): out.append(cnn(X[i:i + 256]))
        out = torch.cat(out); return lossf(out, y).item(), (out.argmax(1) == y).float().mean().item(), out

    t = time.time()
    for ep in range(EPOCHS["cnn"]):
        cnn.train(); idx = tr_idx[torch.randperm(len(tr_idx))]; tl = tc = 0
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; xb, yb = Ttr[b], ytr[b]
            flip = torch.rand(len(xb)) < 0.5; xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)   # augmentation
            opt.zero_grad(); out = cnn(xb); loss = lossf(out, yb); loss.backward(); opt.step()
            tl += loss.item() * len(b); tc += (out.argmax(1) == yb).sum().item()
        vl, va, _ = run_eval(Ttr[val_idx], ytr[val_idx])
        hist["loss"].append(tl / len(tr_idx)); hist["accuracy"].append(tc / len(tr_idx))
        hist["val_loss"].append(vl); hist["val_accuracy"].append(va)
        print(f"epoch {ep+1}/{EPOCHS['cnn']}  loss {hist['loss'][-1]:.3f}  acc {hist['accuracy'][-1]:.3f}  val_acc {va:.3f}")
    tt = time.time() - t

    class _H: pass
    h = _H(); h.history = hist; plot_history(h, "CNN")
    _, _, logits = run_eval(Tte, yte); y_pred = logits.argmax(1).numpy()
    evaluate("CNN", "Deep Learning", y_test, y_pred, tt)
    show_predictions(X_test, y_test, y_pred, "CNN")
    torch.save(cnn.state_dict(), MODELS_DIR / "cnn.pt")"""),
code("""if TORCH_OK:
    # feature maps of the first conv layer for one test image
    cnn.eval()
    with torch.no_grad():
        fm = cnn.features[0][0](Tte[:1])[0].numpy()          # first Conv2d -> (32, 64, 64)
    fig, axes = plt.subplots(2, 9, figsize=(14, 3.6))
    for a in axes.ravel(): a.axis("off")
    axes[0, 0].imshow(X_test[0]); axes[0, 0].set_title("input", fontsize=8)
    for c, a in enumerate(axes.ravel()[1:]): a.imshow(fm[c], cmap="viridis")
    plt.suptitle("CNN - first-layer feature maps"); plt.tight_layout(); save_fig("04_cnn_feature_maps.png"); plt.show()"""),

md("""## Summary of this notebook"""),
code("""df = compare(); display(df[df["family"] == "Deep Learning"].round(4))"""),
]
save("04_deep_learning.ipynb", nb4)

# ======================================================================== 05
nb5 = [
md('''# 05 - Transfer learning (ResNet50, VGG16) and final comparison
**Transfer learning:** instead of training a huge network from zero, we reuse a network pre-trained on ImageNet (1.2 M images) and only adapt it to our 8 classes.

**Two phases for each network**
1. **Feature extraction** - the pretrained base is *frozen*; only the new classification head is trained.
2. **Fine-tuning** - the last layers of the base are *unfrozen* and trained with a very small learning rate.

> The first run downloads the ImageNet weights (ResNet50 ~ 95 MB, VGG16 ~ 58 MB) - internet is required.
> If you are short of RAM/time, lower `EPOCHS` in `src/common.py` or set `MAX_PER_CLASS` to e.g. 300.'''),
code(HEAD),
md('''> **TensorFlow required for the next cells (ResNet50 / VGG16).** If TensorFlow is blocked on your PC, skip to *"3 - Final comparison"* - those cells only need the other notebooks' saved metrics.'''),
code('''import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50, VGG16
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_pre
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
tf.keras.utils.set_random_seed(SEED)

X_train, X_test, y_train, y_test = load_split()
Xtr = X_train.astype("float32"); Xte = X_test.astype("float32")     # keep 0..255: the networks' own preprocess_input expects this
WEIGHTS = "imagenet" if USE_PRETRAINED else None
FINE_TUNE_EPOCHS = 1 if QUICK else 3
print("pretrained weights:", WEIGHTS, "| input size:", TL_SIZE, "| epochs:", EPOCHS)'''),

md('''## Builder shared by both networks
`64x64 image -> Resizing(96x96) -> network-specific preprocessing -> pretrained base -> GlobalAveragePooling -> Dense(256) -> Dropout -> softmax(8)`'''),
code('''def build_transfer(base_cls, preprocess, name):
    base = base_cls(weights=WEIGHTS, include_top=False, input_shape=(TL_SIZE, TL_SIZE, 3))
    base.trainable = False                                   # phase 1: frozen
    inp = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = layers.Resizing(TL_SIZE, TL_SIZE)(inp)
    x = layers.Lambda(preprocess, name="preprocess")(x)
    x = base(x, training=False)                              # keep BatchNorm in inference mode
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(CLASSES), activation="softmax")(x)
    return keras.Model(inp, out, name=name), base

class _H: pass
def merge_history(h1, h2):
    m = _H(); m.history = {k: h1.history[k] + h2.history[k] for k in h1.history}; return m

def train_transfer(model, base, n_unfreeze, epochs, name):
    cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    # ---- phase 1: train the head only
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    t = time.time()
    h1 = model.fit(Xtr, y_train, validation_split=0.15, epochs=epochs, batch_size=64, callbacks=cb, verbose=2)
    # ---- phase 2: fine-tune the last layers with a tiny learning rate
    base.trainable = True
    for layer in base.layers[:-n_unfreeze]: layer.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(1e-5), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(Xtr, y_train, validation_split=0.15, epochs=FINE_TUNE_EPOCHS, batch_size=64, callbacks=cb, verbose=2)
    return merge_history(h1, h2), time.time() - t'''),

md('''## 1 - ResNet50
![ResNet50](../docs/model_steps/09_resnet50.png)

ResNet's **residual (skip) connections** let very deep networks (50 layers) train well. Phase 2 unfreezes the last 30 layers.'''),
code('''resnet, resnet_base = build_transfer(ResNet50, resnet_pre, "resnet50")
print("parameters:", f"{resnet.count_params():,}")
h_res, tt = train_transfer(resnet, resnet_base, n_unfreeze=30, epochs=EPOCHS["resnet"], name="ResNet50")
plot_history(h_res, "ResNet50")'''),
code('''y_pred = resnet.predict(Xte, verbose=0).argmax(1)
evaluate("ResNet50", "Deep Learning", y_test, y_pred, tt)
show_predictions(X_test, y_test, y_pred, "ResNet50")
resnet.save(MODELS_DIR / "resnet50.keras")      # reload with keras.models.load_model(path, safe_mode=False)'''),

md('''## 2 - VGG16
![VGG16](../docs/model_steps/10_vgg16.png)

VGG16 is a plain stack of 3x3 convolutions in 5 blocks. Phase 2 unfreezes the last convolution block (`block5`).'''),
code('''vgg, vgg_base = build_transfer(VGG16, vgg_pre, "vgg16")
print("parameters:", f"{vgg.count_params():,}")
n_block5 = len([l for l in vgg_base.layers if l.name.startswith("block5")])
h_vgg, tt = train_transfer(vgg, vgg_base, n_unfreeze=n_block5, epochs=EPOCHS["vgg"], name="VGG16")
plot_history(h_vgg, "VGG16")'''),
code('''y_pred = vgg.predict(Xte, verbose=0).argmax(1)
evaluate("VGG16", "Deep Learning", y_test, y_pred, tt)
show_predictions(X_test, y_test, y_pred, "VGG16")
vgg.save(MODELS_DIR / "vgg16.keras")'''),

md('''## 3 - Final comparison of all ten models
Everything is read from `results/metrics/*.json` (written by notebooks 03, 04, 05) and merged into **`results/all_models_comparison.csv`**.'''),
code('''df = compare()
expected = {"SVM", "KNN", "Random Forest", "Decision Tree", "Logistic Regression", "Naive Bayes",
            "MLP", "CNN", "ResNet50", "VGG16"}
missing = expected - set(df["model"])
if missing: print("WARNING - not trained yet:", sorted(missing))
display(df.round(4))
print("Saved ->", (ROOT / "results" / "all_models_comparison.csv").relative_to(ROOT))'''),
code('''plot_comparison(df)'''),
code('''# metrics side by side + training time
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5))
d = df.sort_values("f1_score", ascending=False); x = np.arange(len(d)); w = 0.2
for i, m in enumerate(["accuracy", "precision", "recall", "f1_score"]):
    ax[0].bar(x + (i - 1.5) * w, d[m], w, label=m)
ax[0].set_xticks(x); ax[0].set_xticklabels(d["model"], rotation=35, ha="right"); ax[0].set_ylim(0, 1.02)
ax[0].legend(ncol=4, fontsize=8); ax[0].set_title("Accuracy / precision / recall / F1")
ax[1].bar(d["model"], d["train_time_s"], color="gray"); ax[1].set_yscale("log")
ax[1].set_title("Training time (s, log scale)"); plt.setp(ax[1].get_xticklabels(), rotation=35, ha="right")
plt.tight_layout(); save_fig("05_metrics_and_time.png"); plt.show()'''),
md('''## How to read the results
* Compare **F1-score** rather than only accuracy if some classes are rarer.
* Look at each model's confusion matrix in `results/figures/cm_*.png` to see *which classes get confused* (e.g. cat vs dog).
* Typical pattern: HOG + SVM is the strongest traditional model; Naive Bayes and Decision Tree are the weakest; transfer learning (ResNet50 / VGG16) usually gives the best accuracy; MLP sits between the traditional models and the CNN.
  **Your own numbers are what count - report them from the table above.**
* Trade-off: deep models need more time/RAM; traditional models train in seconds-minutes.'''),
]
save("05_transfer_learning_evaluation.ipynb", nb5)