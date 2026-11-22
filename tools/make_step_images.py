"""
Creates the explanatory 'how it works, step by step' images in docs/ .
 - the photos are scikit-image sample images (cat, astronaut, coffee) and the HOG maps are REAL computations
 - the model panels use small 2-D toy data purely to illustrate the concept of each algorithm
Run:  python tools/make_step_images.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.colors import ListedColormap
from scipy.signal import convolve2d
from skimage import data, color, transform, feature, exposure
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = DOCS / "model_steps"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})

COLORS = ["#4C78A8", "#F58518", "#54A24B"]
CMAP3 = ListedColormap(COLORS)
STEP_COL = "#2F4B7C"

# ------------------------------------------------------------------ sample data
IMGS = {"cat": data.chelsea(), "person": data.astronaut(), "cup": data.coffee()}
SAMPLE = IMGS["cat"]
RS = {k: transform.resize(v, (64, 64), anti_aliasing=True) for k, v in IMGS.items()}
GRAY = color.rgb2gray(RS["cat"])
HOG_PARAMS = dict(orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
FD, HOG_IMG = feature.hog(GRAY, visualize=True, **HOG_PARAMS)

# toy 2-D data
XT, YT = make_blobs(n_samples=180, centers=[(-2, -1), (2, -1.5), (0, 2.5)], cluster_std=1.25, random_state=4)
XTR, XTE, YTR, YTE = train_test_split(XT, YT, test_size=.3, random_state=0, stratify=YT)

# ------------------------------------------------------------------ helpers
def title_step(ax, n, text):
    ax.set_title(f"STEP {n}  {text}", fontsize=9.5, color="white", fontweight="bold", loc="left",
                 bbox=dict(facecolor=STEP_COL, edgecolor="none", boxstyle="round,pad=0.3"), pad=8)

def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#bbbbbb")

def note(ax, text, y=-0.08, fs=8):
    ax.text(0.5, y, text, transform=ax.transAxes, ha="center", va="top", fontsize=fs, color="#333333", wrap=True)

def text_panel(ax, lines, fs=9, bullet="•"):
    ax.axis("off")
    y = 0.97
    for ln in lines:
        head = ln.startswith("#")
        t = ln.lstrip("#").strip()
        ax.text(0.0, y, t if head else f"{bullet} {t}", transform=ax.transAxes, va="top", fontsize=fs + (1 if head else 0),
                fontweight="bold" if head else "normal", color=STEP_COL if head else "black")
        y -= 0.135 if head else 0.118

# ---- common data-pipeline panels
def p_load(ax, n=1):
    m = np.hstack([transform.resize(IMGS[k], (150, 150), anti_aliasing=True) for k in IMGS])
    ax.imshow(m); clean(ax); title_step(ax, n, "Load images")
    ax.set_xlabel("cat · person · cup   (examples)\n8 classes, variable sizes", fontsize=8)

def p_resize(ax, n=2, gray=True, norm="normalise /255"):
    h, w = SAMPLE.shape[:2]
    ax.axis("off")
    a = ax.inset_axes([0.0, 0.2, 0.46, 0.7]); a.imshow(SAMPLE); clean(a); a.set_title(f"{w}×{h} RGB", fontsize=8)
    b = ax.inset_axes([0.54, 0.2, 0.46, 0.7])
    if gray: b.imshow(GRAY, cmap="gray")
    else: b.imshow(RS["cat"])
    clean(b); b.set_title("64×64 gray" if gray else "64×64 RGB", fontsize=8)
    ax.annotate("", xy=(0.52, 0.55), xytext=(0.47, 0.55), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="->", lw=2, color=STEP_COL))
    ax.set_title(f"STEP {n}  Resize" + (" + grayscale" if gray else (" + " + norm if norm else "")), fontsize=9.5, color="white",
                 fontweight="bold", loc="left", bbox=dict(facecolor=STEP_COL, edgecolor="none", boxstyle="round,pad=0.3"), pad=8)
    ax.text(0.5, 0.05, "all images → same size" + (" , 1 channel" if gray else (" , values 0-1" if norm else " , values 0-255")), ha="center", fontsize=8, transform=ax.transAxes)

def p_hog(ax, n=3):
    ax.imshow(exposure.rescale_intensity(HOG_IMG, in_range=(0, 10)), cmap="gray"); clean(ax)
    title_step(ax, n, "HOG features")
    ax.set_xlabel("gradients → 8×8 cells → 9 orientations\n→ 2×2 blocks  =  1764 numbers", fontsize=8)

def p_split(ax, n=4, scaler=True):
    ax.axis("off"); title_step(ax, n, "Train / test split" + (" + scale" if scaler else ""))
    ax.add_patch(Rectangle((0.02, 0.50), 0.78, 0.2, color=COLORS[0])); ax.add_patch(Rectangle((0.80, 0.50), 0.18, 0.2, color=COLORS[1]))
    ax.text(0.41, 0.60, "train 80 %  (≈5,519)", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(0.89, 0.60, "test\n20 %", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(0.5, 0.82, "6,899 images · stratified · seed 42", ha="center", fontsize=8.5)
    lines = ["StandardScaler fitted on TRAIN only", "→ same transform applied to test"] if scaler else \
            ["same split file for all 10 models", "validation = 15 % of train"]
    ax.text(0.5, 0.28, "\n".join(lines), ha="center", va="center", fontsize=8.5,
            bbox=dict(facecolor="#EEF3FA", edgecolor=STEP_COL, boxstyle="round,pad=0.4"))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

def toy_cm(clf):
    clf.fit(XTR, YTR); return confusion_matrix(YTE, clf.predict(XTE)), (clf.predict(XTE) == YTE).mean()

def p_eval(ax, clf, n=6, name=""):
    cm, acc = toy_cm(clf)
    pos = ax.get_position(); ax.set_position([pos.x0 + .02, pos.y0 + .03, pos.width * .85, pos.height * .75])
    ax.imshow(cm, cmap="Blues"); ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xlabel("predicted"); ax.set_ylabel("true", labelpad=1)
    for i in range(3):
        for j in range(3): ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    title_step(ax, n, "Evaluate on test set")
    ax.text(0.5, -0.32, "accuracy · precision · recall · F1\nconfusion matrix (toy example shown)", transform=ax.transAxes,
            ha="center", va="top", fontsize=8)

def regions(ax, clf, X=XTR, y=YTR, alpha=.25, pad=1.2):
    xx, yy = np.meshgrid(np.linspace(XT[:, 0].min() - pad, XT[:, 0].max() + pad, 220),
                         np.linspace(XT[:, 1].min() - pad, XT[:, 1].max() + pad, 220))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-.5, .5, 1.5, 2.5], colors=COLORS, alpha=alpha)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=CMAP3, vmin=0, vmax=2, edgecolor="k", s=16, linewidth=.4)
    ax.set_xlim(xx.min(), xx.max()); ax.set_ylim(yy.min(), yy.max()); clean(ax)

def sub_axes(fig, spec, n, wr=None):
    gs = GridSpecFromSubplotSpec(1, n, subplot_spec=spec, wspace=.12, width_ratios=wr)
    return [fig.add_subplot(gs[0, i]) for i in range(n)]

def concept_title(fig, spec, n, text):
    ax = fig.add_subplot(spec); ax.axis("off"); title_step(ax, n, text); return ax

# ------------------------------------------------------------------ traditional models
def trad_figure(idx, fname, name, concept, clf_factory, params, subtitle):
    fig = plt.figure(figsize=(17, 8.6))
    gs = GridSpec(2, 4, figure=fig, hspace=.55, wspace=.22)
    p_load(fig.add_subplot(gs[0, 0]), 1)
    p_resize(fig.add_subplot(gs[0, 1]), 2, gray=True)
    p_hog(fig.add_subplot(gs[0, 2]), 3)
    # step 4: feature vector
    ax = fig.add_subplot(gs[0, 3]); ax.plot(FD, lw=.5, color=STEP_COL); ax.set_xlim(0, len(FD)); ax.set_yticks([])
    ax.set_xlabel("feature index (0 … 1763)"); title_step(ax, 4, "Feature vector")
    ax.text(0.5, 0.85, f"{len(FD)} values / image", transform=ax.transAxes, ha="center", fontsize=9,
            bbox=dict(facecolor="white", edgecolor="#999999", boxstyle="round"))
    p_split(fig.add_subplot(gs[1, 0]), 5, True)
    holder = concept_title(fig, gs[1, 1:3], 6, f"Train {name}  –  {subtitle}")
    holder.set_position(holder.get_position())
    concept(fig, gs[1, 1:3])
    p_eval(fig.add_subplot(gs[1, 3]), clf_factory(), 7, name)
    fig.suptitle(f"{idx}. {name} on HOG features  –  step-by-step processing", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, -0.012, "Hyper-parameters used in the notebook:  " + params +
             "      |  photos: real samples · HOG maps: real computation · step 6 & 7 panels: toy 2-D data (concept illustration)",
             ha="center", fontsize=8.5, color="#444444")
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", fname)

def inner(fig, spec, n, wr=None):
    """Axes inside the concept area (leave room on top for the step title)."""
    gs = GridSpecFromSubplotSpec(1, n, subplot_spec=spec, wspace=.1, width_ratios=wr)
    axs = []
    for i in range(n):
        a = fig.add_subplot(gs[0, i]); axs.append(a)
        pos = a.get_position(); a.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.82])
    return axs

def c_svm(fig, spec):
    a, b = inner(fig, spec, 2)
    m = (YTR < 2); lin = SVC(kernel="linear", C=1).fit(XTR[m], YTR[m])
    a.scatter(XTR[m, 0], XTR[m, 1], c=YTR[m], cmap=CMAP3, vmin=0, vmax=2, edgecolor="k", s=16, linewidth=.4)
    xx = np.linspace(XT[:, 0].min() - 1, XT[:, 0].max() + 1, 2); w = lin.coef_[0]; bb = lin.intercept_[0]
    for off, ls in ((0, "-"), (1, "--"), (-1, "--")):
        a.plot(xx, (-w[0] * xx - bb + off) / w[1], "k" + ls, lw=1)
    a.scatter(*lin.support_vectors_.T, s=90, facecolors="none", edgecolors="red", linewidths=1.4)
    a.set_ylim(XT[:, 1].min() - 1, XT[:, 1].max() + 1); clean(a)
    a.set_title("a) maximum-margin line\n(red = support vectors)", fontsize=8.5)
    regions(b, SVC(kernel="rbf", C=10, gamma="scale").fit(XTR, YTR)); b.set_title("b) RBF kernel: curved boundaries\n(multi-class, C=10)", fontsize=8.5)

def c_knn(fig, spec):
    a, b, c = inner(fig, spec, 3)
    regions(a, KNeighborsClassifier(1).fit(XTR, YTR)); a.set_title("k = 1: jagged, noisy", fontsize=8.5)
    q = np.array([[0.4, 0.4]]); knn = KNeighborsClassifier(7).fit(XTR, YTR); d, idx = knn.kneighbors(q)
    b.scatter(XTR[:, 0], XTR[:, 1], c=YTR, cmap=CMAP3, vmin=0, vmax=2, s=14, edgecolor="k", linewidth=.3, alpha=.7)
    for i in idx[0]: b.plot([q[0, 0], XTR[i, 0]], [q[0, 1], XTR[i, 1]], "k-", lw=.8)
    b.add_patch(Circle(q[0], d.max(), fill=False, ls="--", color="k")); b.scatter(*q.T, marker="*", s=200, c="gold", edgecolor="k", zorder=5)
    b.set_xlim(-3.5, 4.5); b.set_ylim(-3.5, 4.5); clean(b)
    b.set_title(f"the 7 nearest neighbours vote\n→ class {knn.predict(q)[0]+1} (majority)", fontsize=8.5)
    regions(c, KNeighborsClassifier(15).fit(XTR, YTR)); c.set_title("k = 15: smooth", fontsize=8.5)

def c_rf(fig, spec):
    axs = inner(fig, spec, 4)
    rng = np.random.RandomState(1)
    for i, a in enumerate(axs[:3]):
        idx = rng.choice(len(XTR), len(XTR), replace=True)
        t = DecisionTreeClassifier(max_depth=4, max_features=1, random_state=i).fit(XTR[idx], YTR[idx])
        regions(a, t, XTR[idx], YTR[idx], alpha=.3); a.set_title(f"tree {i+1}\n(bootstrap sample)", fontsize=8.5)
    regions(axs[3], RandomForestClassifier(100, max_depth=5, random_state=0).fit(XTR, YTR)); axs[3].set_title("forest = majority\nvote of 200 trees", fontsize=8.5)

def c_dt(fig, spec):
    a, b = inner(fig, spec, 2, wr=[1, 1.5])
    t = DecisionTreeClassifier(max_depth=3, random_state=0).fit(XTR, YTR)
    regions(a, t); a.set_title("axis-parallel rectangles", fontsize=8.5)
    plot_tree(t, ax=b, filled=True, feature_names=["f₁", "f₂"], class_names=["A", "B", "C"], fontsize=6, impurity=False, rounded=True)
    b.set_title("sequence of if / else questions on features", fontsize=8.5)

def c_lr(fig, spec):
    a, b = inner(fig, spec, 2)
    lr = LogisticRegression(max_iter=500).fit(XTR, YTR); regions(a, lr); a.set_title("linear boundaries between classes", fontsize=8.5)
    z = np.linspace(-6, 6, 200); b.plot(z, 1 / (1 + np.exp(-z)), lw=2, color=STEP_COL); b.axhline(.5, color="gray", ls="--")
    b.set_xlabel("score  z = w·x + b"); b.set_ylabel("probability"); b.set_title("sigmoid / softmax turns scores\ninto probabilities", fontsize=8.5)
    b.text(-5.5, .8, "softmax:  P(c) = e^z_c / Σ e^z", fontsize=8)

def c_nb(fig, spec):
    a, b = inner(fig, spec, 2)
    xs = np.linspace(-6, 6, 300)
    for k in range(3):
        mu, sd = XTR[YTR == k, 0].mean(), XTR[YTR == k, 0].std()
        a.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), color=COLORS[k], lw=2)
        a.hist(XTR[YTR == k, 0], bins=12, density=True, color=COLORS[k], alpha=.25)
    a.set_yticks([]); a.set_title("per-class Gaussian of ONE feature\n(mean μ and variance σ² learned)", fontsize=8.5)
    regions(b, GaussianNB().fit(XTR, YTR)); b.set_title("P(class) × Π P(feature | class)", fontsize=8.5)

# ------------------------------------------------------------------ deep-learning figures
def draw_arch(ax, blocks, title_n, title):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); title_step(ax, title_n, title)
    n = len(blocks); w = min(0.07, 0.70 / n); xs = np.linspace(0.04 + w / 2, 0.96 - w / 2, n)
    for i, (label, size, col) in enumerate(blocks):
        h = 0.18 + 0.4 * size
        ax.add_patch(FancyBboxPatch((xs[i] - w / 2, 0.55 - h / 2), w, h, boxstyle="round,pad=0.005", fc=col, ec="#333333", lw=.8, alpha=.9))
        ax.text(xs[i], 0.55 - 0.58 / 2 - 0.04, label, ha="center", va="top", fontsize=7)
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - w / 2, 0.55), xytext=(xs[i] + w / 2, 0.55), arrowprops=dict(arrowstyle="->", lw=1))

IN, CONV, POOL, DENSE, OUTC, OTH = "#9ecae1", "#fdae6b", "#c7e9c0", "#bcbddc", "#fb6a4a", "#dddddd"

def dl_figure(idx, fname, name, arch, special, special_title, steps, hyper, scale01=True):
    fig = plt.figure(figsize=(17, 11.5))
    gs = GridSpec(3, 4, figure=fig, hspace=.6, wspace=.22, height_ratios=[1, .85, 1.05])
    p_load(fig.add_subplot(gs[0, 0]), 1)
    p_resize(fig.add_subplot(gs[0, 1]), 2, gray=False, norm=('normalise /255' if scale01 else ''))
    p_split(fig.add_subplot(gs[0, 2]), 3, scaler=False)
    ax = fig.add_subplot(gs[0, 3]); ax.axis("off"); title_step(ax, 4, "Input tensor")
    ax.add_patch(Rectangle((.1, .25), .5, .5, fc=IN, ec="k")); ax.text(.35, .5, "64 × 64\n× 3", ha="center", va="center", fontsize=11)
    ax.text(.8, .5, "RGB\n0 … 1" if scale01 else "RGB\n0 … 255", ha="center", va="center", fontsize=10)
    ax.text(.5, .12, "batch of 64 images per step", ha="center", fontsize=8.5); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    draw_arch(fig.add_subplot(gs[1, :]), arch, 5, f"{name} architecture (data flows left → right)")
    sp = fig.add_subplot(gs[2, :2]); special(sp, special_title)
    tx = fig.add_subplot(gs[2, 2:]); title_step(tx, 7, "Train → evaluate"); text_panel(tx, steps, fs=9)
    fig.suptitle(f"{idx}. {name}  –  step-by-step processing", fontsize=15, fontweight="bold", y=0.995)
    fig.text(0.5, -0.012, hyper + "      |  photos: real samples · diagrams: architecture used in the notebook", ha="center", fontsize=8.5, color="#444444")
    fig.savefig(OUT / fname, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", fname)

def sp_mlp(ax, ttl):
    ax.axis("off"); title_step(ax, 6, ttl); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    sizes = [5, 6, 5, 3]; xs = [.12, .38, .64, .9]
    pos = [[(x, .5 + (j - (s - 1) / 2) * .12) for j in range(s)] for x, s in zip(xs, sizes)]
    for l in range(3):
        for p in pos[l]:
            for q in pos[l + 1]: ax.plot([p[0], q[0]], [p[1], q[1]], color="#999999", lw=.4, zorder=1)
    for l, ps in enumerate(pos):
        for p in ps: ax.add_patch(Circle(p, .028, fc=[IN, DENSE, DENSE, OUTC][l], ec="k", zorder=3))
    for x, t in zip(xs, ["flattened\npixels", "Dense + ReLU", "Dense + ReLU", "softmax (8)"]): ax.text(x, .1, t, ha="center", fontsize=8)
    ax.text(.5, .93, "every neuron: output = ReLU( Σ wᵢ·xᵢ + b ) – weights learned by back-propagation", ha="center", fontsize=8.5)

def sp_cnn(fig, spec, ttl):
    holder = fig.add_subplot(spec); holder.axis("off"); title_step(holder, 6, ttl)
    gs = GridSpecFromSubplotSpec(1, 4, subplot_spec=spec, wspace=.15)
    kern = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
    conv = convolve2d(GRAY, kern, mode="same", boundary="symm"); relu = np.maximum(conv, 0)
    pool = relu.reshape(32, 2, 32, 2).max(axis=(1, 3))
    items = [(GRAY, "gray", "input patch"), (kern, "coolwarm", "3×3 filter\n(learned, here: edge)"),
             (relu, "magma", "feature map\nafter conv + ReLU"), (pool, "magma", "after 2×2\nMaxPooling")]
    for i, (im, cm, t) in enumerate(items):
        a = fig.add_subplot(gs[0, i]); pos = a.get_position(); a.set_position([pos.x0, pos.y0, pos.width, pos.height * .8])
        a.imshow(im, cmap=cm); clean(a); a.set_title(t, fontsize=8.5)
        if i == 1:
            for (r, c), v in np.ndenumerate(kern): a.text(c, r, int(v), ha="center", va="center", fontsize=9)

def sp_resnet(ax, ttl):
    ax.axis("off"); title_step(ax, 6, ttl); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, w, h, t, col, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01", fc=col, ec="k")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    box(.03, .36, .11, .16, "input\nx", IN, 9)
    box(.22, .50, .27, .12, "Conv 1×1 + BN + ReLU", CONV); box(.22, .33, .27, .12, "Conv 3×3 + BN + ReLU", CONV); box(.22, .16, .27, .12, "Conv 1×1 + BN", CONV)
    ax.add_patch(Circle((.62, .44), .045, fc="white", ec="k", zorder=3)); ax.text(.62, .44, "+", ha="center", va="center", fontsize=16, zorder=4)
    box(.76, .36, .13, .16, "ReLU\n→ output", OUTC, 9)
    ar = dict(arrowstyle="->", lw=1.2)
    ax.annotate("", xy=(.22, .56), xytext=(.14, .46), arrowprops=ar); ax.annotate("", xy=(.355, .45), xytext=(.355, .50), arrowprops=ar)
    ax.annotate("", xy=(.355, .28), xytext=(.355, .33), arrowprops=ar); ax.annotate("", xy=(.58, .42), xytext=(.49, .22), arrowprops=ar)
    ax.annotate("", xy=(.76, .44), xytext=(.665, .44), arrowprops=ar)
    ax.annotate("", xy=(.615, .485), xytext=(.085, .52), arrowprops=dict(arrowstyle="->", lw=1.8, color="red", connectionstyle="arc3,rad=-.45"))
    ax.text(.45, .86, "skip (identity) connection:  out = F(x) + x", color="red", fontsize=9.5, fontweight="bold", ha="center")
    ax.text(.5, .04, "the block only learns the residual F(x); gradients flow through the shortcut → 50 layers train well", ha="center", fontsize=8.5)

def sp_vgg(ax, ttl):
    ax.axis("off"); title_step(ax, 6, ttl); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    names = ["block1\n2×conv64", "block2\n2×conv128", "block3\n3×conv256", "block4\n3×conv512", "block5\n3×conv512", "new head\nGAP+Dense"]
    for ph, y, lab in ((1, .62, "PHASE 1 – feature extraction"), (2, .22, "PHASE 2 – fine-tuning")):
        ax.text(.0, y + .21, lab, fontsize=9, fontweight="bold", color=STEP_COL)
        for i, n in enumerate(names):
            frozen = (i < 5) if ph == 1 else (i < 4)
            col = "#BBBBBB" if frozen and i < 5 else ("#54A24B" if i == 5 else "#F58518")
            ax.add_patch(FancyBboxPatch((.02 + i * .16, y), .14, .17, boxstyle="round,pad=0.005", fc=col, ec="k")); ax.text(.09 + i * .16, y + .085, n, ha="center", va="center", fontsize=7.5)
    ax.text(.5, .02, "grey = frozen (ImageNet weights)   orange = trained, tiny learning rate 1e-5   green = new layers, lr 1e-3", ha="center", fontsize=8)

def sp_resnet_ft(ax, ttl): sp_resnet(ax, ttl)

# ------------------------------------------------------------------ build all
def build_all():
    # 01-06 traditional
    trad_figure("1", "01_svm.png", "SVM", c_svm, lambda: SVC(kernel="rbf", C=10), "SVC(kernel='rbf', C=10, gamma='scale')", "find the best separating boundary")
    trad_figure("2", "02_knn.png", "KNN", c_knn, lambda: KNeighborsClassifier(7), "KNeighborsClassifier(k = best of [1…15] by 3-fold CV on train)", "no training: store the data, vote at prediction")
    trad_figure("3", "03_random_forest.png", "Random Forest", c_rf, lambda: RandomForestClassifier(100, max_depth=5, random_state=0), "RandomForestClassifier(n_estimators=200, random_state=42)", "many de-correlated trees")
    trad_figure("4", "04_decision_tree.png", "Decision Tree", c_dt, lambda: DecisionTreeClassifier(max_depth=3, random_state=0), "DecisionTreeClassifier(max_depth = best of [3…None] by CV)", "learn if / else splits")
    trad_figure("5", "05_logistic_regression.png", "Logistic Regression", c_lr, lambda: LogisticRegression(max_iter=500), "LogisticRegression(C=1.0, max_iter=1000) – multinomial", "linear scores → probabilities")
    trad_figure("6", "06_naive_bayes.png", "Naive Bayes", c_nb, lambda: GaussianNB(), "GaussianNB()  (features independent given the class)", "Bayes rule with Gaussians")

    # 07 MLP
    dl_figure("7", "07_mlp_ann.png", "MLP / ANN",
      [("Input\n64×64×3", 1, IN), ("Flatten\n12,288", .9, OTH), ("Dense 512\nReLU", .75, DENSE), ("Dropout\n0.3", .75, OTH), ("Dense 256\nReLU", .55, DENSE), ("Dropout\n0.3", .55, OTH), ("Dense 8\nsoftmax", .25, OUTC)],
      sp_mlp, "Fully-connected layers",
      ["# Training", "loss = sparse categorical cross-entropy", "optimizer = Adam (lr 1e-3), batch 64", "30 epochs max, EarlyStopping + ReduceLROnPlateau", "15 % of train used as validation", "# Evaluation", "predict the test set → accuracy, precision,", "recall, F1, confusion matrix, saved .keras model"],
      "Layers: Flatten → Dense(512) → Dense(256) → Dense(8, softmax)")
    # 08 CNN
    dl_figure("8", "08_cnn.png", "CNN",
      [("Input\n64×64×3", 1, IN), ("Augment\nflip/rot/zoom", 1, OTH), ("2×Conv 32\n+BN+ReLU", .95, CONV), ("MaxPool\n32×32", .7, POOL), ("2×Conv 64\n+BN+ReLU", .7, CONV), ("MaxPool\n16×16", .5, POOL), ("2×Conv 128\n+BN+ReLU", .5, CONV), ("MaxPool\n8×8", .35, POOL), ("GAP\n128", .2, OTH), ("Dense 128\n+Drop .4", .2, DENSE), ("Softmax\n8", .12, OUTC)],
      lambda ax, t: sp_cnn(fig_holder[0], ax_spec[0], t), "Convolution → ReLU → pooling (real computation)",
      ["# Training", "loss = sparse categorical cross-entropy", "optimizer = Adam (lr 1e-3), batch 64", "25 epochs max, EarlyStopping + ReduceLROnPlateau", "augmentation active only while training", "# Evaluation", "test-set metrics + confusion matrix", "feature maps of each conv block are plotted"],
      "3 conv blocks (32/64/128 filters) · BatchNorm · Dropout · GlobalAveragePooling")
    # 09 ResNet / 10 VGG
    tl_steps = lambda n, l: ["# Transfer learning", "Resizing 64→96 px + network-specific preprocessing", "PHASE 1: base frozen, train new head (Adam 1e-3)", f"PHASE 2: unfreeze {l}, Adam 1e-5, 3 epochs", "weights downloaded from ImageNet on first run", "# Evaluation", "test-set metrics + confusion matrix", "saved .keras model + row in all_models_comparison.csv"]
    dl_figure("9", "09_resnet50.png", "ResNet50 (transfer learning)",
      [("Input\n64×64×3", 1, IN), ("Resize\n96×96", 1, OTH), ("conv1 7×7\n+MaxPool", .9, CONV), ("conv2_x\n3 res blocks", .75, CONV), ("conv3_x\n4 res blocks", .6, CONV), ("conv4_x\n6 res blocks", .45, CONV), ("conv5_x\n3 res blocks", .3, CONV), ("GAP\n2048", .2, OTH), ("Dense 256\n+Drop .3", .2, DENSE), ("Softmax\n8", .12, OUTC)],
      sp_resnet, "Residual block (the building block of ResNet)", tl_steps("ResNet50", "last 30 layers"),
      "Base: ResNet50(imagenet, include_top=False) + new head", scale01=False)
    dl_figure("10", "10_vgg16.png", "VGG16 (transfer learning)",
      [("Input\n64×64×3", 1, IN), ("Resize\n96×96", 1, OTH), ("block1\n2×conv 64", .95, CONV), ("Pool", .75, POOL), ("block2\n2×conv 128", .75, CONV), ("Pool", .6, POOL), ("block3\n3×conv 256", .6, CONV), ("Pool", .45, POOL), ("block4\n3×conv 512", .45, CONV), ("Pool", .35, POOL), ("block5\n3×conv 512", .3, CONV), ("GAP\n512", .2, OTH), ("Dense 256\n+Drop .3", .2, DENSE), ("Softmax 8", .12, OUTC)],
      sp_vgg, "Freeze → fine-tune", tl_steps("VGG16", "block5"),
      "Base: VGG16(imagenet, include_top=False) + new head", scale01=False)

fig_holder, ax_spec = [None], [None]

def dl_figure_cnn_patch():
    """CNN needs the figure + gridspec handle for its 4-panel special area -> monkeypatch dl_figure."""
    global dl_figure
    orig = dl_figure
    def patched(idx, fname, name, arch, special, special_title, steps, hyper, scale01=True):
        if idx != "8": return orig(idx, fname, name, arch, special, special_title, steps, hyper, scale01)
        fig = plt.figure(figsize=(17, 11.5)); gs = GridSpec(3, 4, figure=fig, hspace=.6, wspace=.22, height_ratios=[1, .85, 1.05])
        p_load(fig.add_subplot(gs[0, 0]), 1); p_resize(fig.add_subplot(gs[0, 1]), 2, gray=False, norm=('normalise /255' if scale01 else '')); p_split(fig.add_subplot(gs[0, 2]), 3, scaler=False)
        ax = fig.add_subplot(gs[0, 3]); ax.axis("off"); title_step(ax, 4, "Input tensor")
        ax.add_patch(Rectangle((.1, .25), .5, .5, fc=IN, ec="k")); ax.text(.35, .5, "64 × 64\n× 3", ha="center", va="center", fontsize=11)
        ax.text(.8, .5, "RGB\n0 … 1" if scale01 else "RGB\n0 … 255", ha="center", va="center", fontsize=10); ax.text(.5, .12, "batch of 64 images per step", ha="center", fontsize=8.5)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        draw_arch(fig.add_subplot(gs[1, :]), arch, 5, f"{name} architecture (data flows left → right)")
        sp_cnn(fig, gs[2, :2], special_title)
        tx = fig.add_subplot(gs[2, 2:]); title_step(tx, 7, "Train → evaluate"); text_panel(tx, steps, fs=9)
        fig.suptitle(f"{idx}. {name}  –  step-by-step processing", fontsize=15, fontweight="bold", y=0.995)
        fig.text(0.5, -0.012, hyper + "      |  photos: real samples · conv example: real computation · diagram: architecture used in the notebook", ha="center", fontsize=8.5, color="#444444")
        fig.savefig(OUT / fname, dpi=120, bbox_inches="tight"); plt.close(fig); print("saved", fname)
    dl_figure = patched

# ------------------------------------------------------------------ overview images
def hog_detail():
    gy, gx = np.gradient(GRAY); mag = np.hypot(gx, gy); ori = np.rad2deg(np.arctan2(gy, gx)) % 180
    fig, ax = plt.subplots(1, 6, figsize=(19, 3.9))
    ax[0].imshow(RS["cat"]); ax[0].set_title("1  RGB 64×64")
    ax[1].imshow(GRAY, cmap="gray"); ax[1].set_title("2  grayscale")
    ax[2].imshow(mag, cmap="magma"); ax[2].set_title("3  gradient magnitude\n(where are the edges?)")
    ax[3].imshow(ori, cmap="hsv"); ax[3].set_title("4  gradient direction\n(0–180°)")
    ax[4].imshow(GRAY, cmap="gray")
    for i in range(0, 65, 8): ax[4].axhline(i - .5, color="cyan", lw=.5); ax[4].axvline(i - .5, color="cyan", lw=.5)
    ax[4].add_patch(Rectangle((15.5, 15.5), 16, 16, fill=False, ec="red", lw=2))
    ax[4].set_title("5  8×8 cells (cyan)\nblock = 2×2 cells (red)")
    ax[5].imshow(exposure.rescale_intensity(HOG_IMG, in_range=(0, 10)), cmap="gray"); ax[5].set_title(f"6  HOG: 9 orientations/cell\n→ {len(FD)} features")
    for a in ax: a.axis("off")
    fig.suptitle("HOG feature extraction – step by step (real computation on a sample photo)", fontsize=14, fontweight="bold")
    fig.savefig(DOCS / "00_hog_step_by_step.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("saved 00_hog_step_by_step.png")

def pipeline_overview():
    fig, ax = plt.subplots(figsize=(17, 6.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    def box(x, y, w, h, t, col, fs=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008", fc=col, ec="#333333")); ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs)
    ar = dict(arrowstyle="->", lw=1.6, color="#333333")
    box(.01, .38, .11, .24, "Natural Images\n6,899 photos\n8 classes", "#9ecae1")
    box(.15, .38, .11, .24, "01 Explore\ncounts · samples\nsizes · mean img", "#c7e9c0")
    box(.29, .38, .12, .24, "02 Preprocess\nresize 64×64\n80/20 stratified split\n→ dataset_64.npz", "#c7e9c0")
    for a, b in ((.12, .15), (.26, .29)): ax.annotate("", xy=(b, .5), xytext=(a, .5), arrowprops=ar)
    ax.annotate("", xy=(.47, .75), xytext=(.41, .52), arrowprops=ar); ax.annotate("", xy=(.47, .25), xytext=(.41, .48), arrowprops=ar)
    box(.47, .62, .13, .26, "03 grayscale → HOG\n(1764 features)\n→ StandardScaler", "#fdd0a2")
    box(.64, .62, .17, .26, "SVM · KNN · Random Forest\nDecision Tree\nLogistic Regression\nNaive Bayes", "#fdae6b")
    ax.annotate("", xy=(.64, .75), xytext=(.60, .75), arrowprops=ar)
    box(.47, .12, .13, .26, "04 RGB pixels /255\n(64×64×3)", "#dadaeb")
    box(.64, .27, .17, .11, "04  MLP (ANN)   ·   CNN", "#bcbddc"); box(.64, .12, .17, .11, "05  ResNet50  ·  VGG16\n(transfer learning)", "#9e9ac8")
    ax.annotate("", xy=(.64, .325), xytext=(.60, .25), arrowprops=ar); ax.annotate("", xy=(.64, .175), xytext=(.60, .22), arrowprops=ar)
    box(.86, .38, .13, .24, "Metrics · confusion\nmatrices · saved models\nall_models_\ncomparison.csv", "#fb9a99")
    ax.annotate("", xy=(.86, .55), xytext=(.81, .75), arrowprops=ar); ax.annotate("", xy=(.86, .45), xytext=(.81, .22), arrowprops=ar)
    ax.text(.725, .94, "Traditional ML", ha="center", fontsize=11, fontweight="bold", color="#a63603")
    ax.text(.725, .02, "Deep learning", ha="center", fontsize=11, fontweight="bold", color="#54278f")
    fig.suptitle("Project pipeline: 5 notebooks, 10 models, one shared train/test split", fontsize=15, fontweight="bold")
    fig.savefig(DOCS / "00_project_pipeline.png", dpi=120, bbox_inches="tight"); plt.close(fig); print("saved 00_project_pipeline.png")

if __name__ == "__main__":
    pipeline_overview(); hog_detail()
    dl_figure_cnn_patch(); build_all()
