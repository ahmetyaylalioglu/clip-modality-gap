"""B3: modality gap visualization (PCA and UMAP) on 1,000 COCO test pairs.

1,000 test images are drawn with a fixed seed; each is paired with its
first caption. Top row: raw embeddings (M0, before any correction).
Bottom row: the same points after M1 centering with means from 10k COCO-train
calibration pairs (seed 0, as in scripts/c1_retrieval.py), for comparison.
PCA and UMAP are fit separately for each row, on images and texts jointly.
Thin grey lines connect 100 of the pairs.

Usage:  python scripts/b3_gap_plot.py --model clip_b16
Output: results/figures/gap_pca_umap_<model>.png
        results/reproduction/gap_stats_<model>.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from c1_retrieval import MAIN_N, calib_means, load, train_split  # noqa: E402
from src.gap import center, gap_stats  # noqa: E402

N_PAIRS = 1000
SEED = 0
N_LINES = 100
IMG_C, TXT_C = "#1f77b4", "#d62728"


def first_caption_pairs(model):
    img, txt, owner, _ = load(model, "mscoco_test")
    idx = np.sort(np.random.default_rng(SEED).choice(len(img), N_PAIRS, replace=False))
    _, first = np.unique(owner, return_index=True)          # first caption of every image
    return img[idx], txt[first[idx]], idx


def embed2d(x, how):
    if how == "PCA":
        p = PCA(n_components=2, random_state=SEED).fit(x)
        return p.transform(x), p.explained_variance_ratio_
    import umap
    return umap.UMAP(n_components=2, random_state=SEED, metric="cosine").fit_transform(x), None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    img, txt, idx = first_caption_pairs(m)
    mi, mt, _ = calib_means(train_split(m, "mscoco"), MAIN_N, 0)
    rows = {"M0 (raw)": (img, txt), "M1 (centered, COCO-train means)": center(img, txt, mi, mt, True)}

    stats = []
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.5))
    lines = np.random.default_rng(SEED + 1).choice(N_PAIRS, N_LINES, replace=False)
    for r, (label, (a, b)) in enumerate(rows.items()):
        s = gap_stats(a, b, paired=True)
        stats.append({"model": m, "variant": label, "n_pairs": N_PAIRS, **s})
        z = np.concatenate([a, b])
        for c, how in enumerate(["PCA", "UMAP"]):
            y, evr = embed2d(z, how)
            yi, yt = y[:N_PAIRS], y[N_PAIRS:]
            ax = axes[r, c]
            for j in lines:
                ax.plot([yi[j, 0], yt[j, 0]], [yi[j, 1], yt[j, 1]], color="0.75", lw=0.4, zorder=1)
            ax.scatter(yi[:, 0], yi[:, 1], s=5, c=IMG_C, label="image", zorder=2)
            ax.scatter(yt[:, 0], yt[:, 1], s=5, c=TXT_C, label="text", zorder=2)
            t = f"{how} — {label}"
            if evr is not None:
                t += f"\nexplained var. {evr[0]:.2f}, {evr[1]:.2f}"
            t += f"\n||Δ||={s['gap_norm']:.3f}, paired cos={s['cos_img_txt_paired']:.3f}"
            ax.set_title(t, fontsize=9)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0 and c == 0:
                ax.legend(loc="best", fontsize=8, markerscale=3)
    fig.suptitle(f"{m}: COCO test, {N_PAIRS} image–caption pairs (first caption)", fontsize=11)
    fig.tight_layout()
    out_fig = ROOT / "results" / "figures" / f"gap_pca_umap_{m}.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=150)

    df = pd.DataFrame(stats)
    df["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    df.to_csv(ROOT / "results" / "reproduction" / f"gap_stats_{m}.csv", index=False)
    print(df.drop(columns="git_commit").round(4).to_string(index=False))
    print("saved", out_fig)


if __name__ == "__main__":
    main()
