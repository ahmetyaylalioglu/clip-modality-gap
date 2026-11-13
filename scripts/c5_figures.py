"""Phase C figures, regenerated from the result CSVs (and, for the M4 test
curve, from cached embeddings).

  fig_calib_size_<model>.png    R@1 vs calibration size, I2T and T2I separately
  fig_val_curves_<model>.png    validation R@1 vs delta (M3) and alpha (M4);
                                the M4 test curve (seed 0) is drawn for reference
                                only and was not used for any selection
  fig_zero_shot_<model>.png     zero-shot accuracy before/after (95% CIs)
  fig_gini_vs_recall_<model>.png  dGini@10 vs dR@1 per method / setting (H4)

Usage:  python scripts/c5_figures.py --model clip_b16
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from c1_retrieval import MAIN_N, calib_means, load, r1, subset, train_split  # noqa: E402
from src.gap import ALPHA_GRID, partial_center  # noqa: E402

RES = ROOT / "results" / "improvement"
FIG = ROOT / "results" / "figures"
# fixed method -> color/marker mapping (same in every figure)
STYLE = {
    "M0": ("#6b6b6b", "o"), "M1": ("#2a78d6", "o"), "M2": ("#eb6834", "s"), "M3": ("#1baf7a", "^"),
    "M4": ("#eda100", "D"), "M1-OOD": ("#e87ba4", "v"), "M1-T": ("#008300", "P"), "C1": ("#6250d6", "X"),
}
SETTINGS = [("flickr30k", "5cap", "Flickr30k (5 captions)"), ("flickr30k", "first", "Flickr30k (first caption)"),
            ("mscoco", "5cap", "COCO 5k (5 captions)")]
DIRS = [("i2t", "Image→Text"), ("t2i", "Text→Image")]

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.9", "grid.linewidth": 0.6, "font.size": 9, "lines.linewidth": 2})


def calib_size(m, res):
    fig, axes = plt.subplots(len(SETTINGS), 2, figsize=(9, 9.5), sharex=True)
    for r, (ds, proto, title) in enumerate(SETTINGS):
        for c, (d, dname) in enumerate(DIRS):
            ax = axes[r, c]
            g = res[(res.dataset == ds) & (res.protocol == proto) & (res.direction == d)]
            m1 = g[(g.method == "M1") & (g.calib_source == f"{ds}_train")].groupby("calib_n")["R@1"]
            mean, sd = m1.mean() * 100, m1.std() * 100
            col, mk = STYLE["M1"]
            ax.errorbar(mean.index, mean.values, yerr=sd.values, color=col, marker=mk, ms=6, capsize=3,
                        label="M1 in-domain (mean ± sd, 3 seeds)")
            for meth, ls in [("M0", "-"), ("M4", "-"), ("M1-T", "--")]:
                v = g[g.method == meth]["R@1"].mean() * 100
                lab = {"M0": "M0 baseline", "M4": "M4 (10k, val-selected α)",
                       "M1-T": "M1-T (diagnostic, test stats)"}[meth]
                ax.axhline(v, color=STYLE[meth][0], ls=ls, lw=1.5, label=lab)
            ood = g[g.method == "M1-OOD"]["R@1"] * 100
            ax.errorbar([MAIN_N], [ood.mean()], yerr=[ood.std()], color=STYLE["M1-OOD"][0],
                        marker=STYLE["M1-OOD"][1], ms=7, ls="none", capsize=3, label="M1 out-of-domain (10k)")
            ax.set_xscale("log")
            ax.set_xlim(60, 16000)
            ax.set_xticks([100, 1000, 10000], ["100", "1k", "10k"])
            ax.set_title(f"{title} — {dname}", fontsize=9)
            if c == 0:
                ax.set_ylabel("Recall@1 (%)")
            if r == len(SETTINGS) - 1:
                ax.set_xlabel("calibration pairs (log scale)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f"{m}: Recall@1 vs calibration size", fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(FIG / f"fig_calib_size_{m}.png", dpi=150)
    plt.close(fig)


def val_curves(m, res):
    sel = pd.read_csv(RES / f"retrieval_selection_{m}.csv")
    fig, axes = plt.subplots(len(SETTINGS), 2, figsize=(9, 9.5))
    trains = {ds: train_split(m, ds) for ds in ["flickr30k", "mscoco"]}
    for r, (ds, proto, title) in enumerate(SETTINGS):
        te_img, te_txt, te_owner, _ = load(m, f"{ds}_test")
        I, T, O, _ = subset(te_img, te_txt, te_owner, np.arange(len(te_img)), proto == "first")
        mi, mt, _ = calib_means(trains[ds], MAIN_N, 0)
        test_m4 = np.array([r1(partial_center(I, mi, a), partial_center(T, mt, a), O) for a in ALPHA_GRID]) * 100
        for c, (meth, xl) in enumerate([("M3", "shift δ (M3)"), ("M4", "centering fraction α (M4)")]):
            ax = axes[r, c]
            g = sel[(sel.dataset == ds) & (sel.protocol == proto) & (sel.method == meth)]
            cur = g.groupby("param")[["val_i2t_R@1", "val_t2i_R@1"]].mean() * 100
            for j, (d, dname) in enumerate(DIRS):
                ls = "-" if d == "i2t" else "--"
                col = STYLE[meth][0]
                ax.plot(cur.index, cur[f"val_{d}_R@1"], color=col, ls=ls, label=f"val {dname}")
                chosen = res[(res.dataset == ds) & (res.protocol == proto) & (res.method == meth)
                             & (res.direction == d)]["param"]
                ps = sorted(set(chosen))
                ax.scatter(ps, cur.loc[ps, f"val_{d}_R@1"], marker="o" if d == "i2t" else "s", s=45,
                           facecolor=col if d == "i2t" else "white", edgecolor=col, linewidth=1.5, zorder=4,
                           label=f"selected ({dname})")
                if meth == "M4":
                    ax.plot(ALPHA_GRID, test_m4[:, j], color="0.45", ls=ls, lw=1,
                            label=f"test {dname} (seed 0, not used for selection)")
            ax.set_title(f"{title} — {xl}", fontsize=9)
            ax.set_xlabel(xl)
            if c == 0:
                ax.set_ylabel("Recall@1 (%)")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"{m}: validation curves (mean of 3 calibration seeds); markers = value selected on validation (per seed)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG / f"fig_val_curves_{m}.png", dpi=150)
    plt.close(fig)


def zero_shot(m):
    z = pd.read_csv(RES / f"zero_shot_{m}.csv")
    z = z[(z.method == "M0") | (z.text_mean == "prompts")]
    meths = ["M0", "M1", "M2", "M3", "M4"]
    dss = ["cifar10", "cifar100", "eurosat", "svhn"]
    names = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "eurosat": "EuroSAT (20% split)", "svhn": "SVHN"}
    fig, ax = plt.subplots(figsize=(9, 4))
    w = 0.16
    for k, meth in enumerate(meths):
        rows = z[z.method == meth].set_index("dataset").loc[dss]
        x = np.arange(len(dss)) + (k - 2) * w
        acc = rows.acc.to_numpy() * 100
        err = np.vstack([acc - rows.acc_lo * 100, rows.acc_hi * 100 - acc])
        ax.bar(x, acc, width=w - 0.02, color=STYLE[meth][0], label=meth, edgecolor="white", linewidth=1)
        ax.errorbar(x, acc, yerr=err, fmt="none", ecolor="0.2", elinewidth=0.8, capsize=2)
    ax.set_xticks(np.arange(len(dss)), [names[d] for d in dss])
    ax.set_ylabel("zero-shot top-1 accuracy (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=5, fontsize=8, frameon=False, loc="upper right")
    ax.set_title(f"{m}: zero-shot accuracy before/after (text mean = class prompts; 95% bootstrap CI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / f"fig_zero_shot_{m}.png", dpi=150)
    plt.close(fig)


def gini_vs_recall(m, k=10):
    hub = pd.read_csv(RES / f"retrieval_hubness_{m}.csv")
    sp = pd.read_csv(RES / f"hubness_spearman_{m}.csv")
    fig, axes = plt.subplots(len(SETTINGS), 2, figsize=(9, 10))
    for r, (ds, proto, title) in enumerate(SETTINGS):
        for c, (d, dname) in enumerate(DIRS):
            ax = axes[r, c]
            g = hub[(hub.dataset == ds) & (hub.protocol == proto) & (hub.direction == d)]
            base = g[g.method == "M0"][f"gini@{k}"].iloc[0]
            for meth in ["M1", "M2", "M3", "M4", "M1-OOD", "M1-T", "C1"]:
                h = g[g.method == meth]
                if len(h):
                    ax.scatter(h[f"gini@{k}"] - base, h["dR@1"] * 100, color=STYLE[meth][0], marker=STYLE[meth][1],
                               s=40, edgecolor="white", linewidth=0.8, label=meth, zorder=3)
            ax.axhline(0, color="0.5", lw=0.8)
            ax.axvline(0, color="0.5", lw=0.8)
            s = sp[(sp.dataset == ds) & (sp.protocol == proto) & (sp.direction == d) & (sp.k == k)].iloc[0]
            ax.set_title(f"{title} — {dname}\nSpearman ρ={s.spearman:.2f} [{s.ci_lo:.2f}, {s.ci_hi:.2f}], n={s.n}",
                         fontsize=9)
            ax.set_xlabel(f"ΔGini@{k} (vs M0)")
            if c == 0:
                ax.set_ylabel("ΔR@1 (pp, vs M0)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=7, fontsize=8, frameon=False)
    fig.suptitle(f"{m}: gallery-occupancy Gini vs Recall@1 change (H4)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / f"fig_gini_vs_recall_{m}.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(RES / f"retrieval_{m}.csv")
    calib_size(m, res)
    zero_shot(m)
    gini_vs_recall(m)
    val_curves(m, res)
    print("saved figures to", FIG)


if __name__ == "__main__":
    main()
