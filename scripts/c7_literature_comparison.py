"""Literature comparison: our M0/M1 retrieval numbers in the settings of
I0T (An et al. 2025, Table 2) and Liu et al. 2026 (Table 4).

Only our numbers are written to the CSV; the reported numbers stay in
results/paper_reported.json and are referenced by key (column
`reported_entry`). The figure shows both, with the reported values marked
as such. Nothing is tuned to match the papers.

  I0T setting:  Flickr30k 1k test, first caption per image   (paper: CLIP ViT-B/32)
  Liu setting:  COCO 5k test, 5 captions per image            (paper: backbone not named in the table)

Methods shown: M0, M1 (10k in-domain train pairs, mean over 3 seeds),
M1-T (test statistics, diagnostic -- the likely I0T condition), M4.

Output: results/improvement/literature_comparison.csv
        results/figures/fig_literature_comparison.png

Usage:  python scripts/c7_literature_comparison.py      (all models with retrieval results)
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "improvement"
FIG = ROOT / "results" / "figures"
MODELS = ["clip_b16", "clip_b32", "siglip_b16"]
SHORT = {"clip_b16": "CLIP B/16", "clip_b32": "CLIP B/32", "siglip_b16": "SigLIP B/16"}
MAIN_N = 10000
# setting -> (dataset, protocol, key in paper_reported.json, baseline row, corrected row, label)
SETTINGS = {
    "I0T": ("flickr30k", "first", "i0t_table2", "CLIP", "I0T_post", "Flickr30k 1k, first caption (I0T setting)"),
    "Liu": ("mscoco", "5cap", "liu2026_table4", "Original CLIP", "Mean-Centering", "COCO 5k, 5 captions (Liu setting)"),
}
DIRS = [("i2t", "Image→Text"), ("t2i", "Text→Image")]
METHODS = ["M0", "M1", "M1-T", "M4"]
# same method -> color/marker mapping as c5_figures.py
STYLE = {"M1": ("#2a78d6", "o"), "M1-T": ("#008300", "P"), "M4": ("#eda100", "D")}
REPORTED_COLOR = "#222222"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "0.9", "grid.linewidth": 0.6, "font.size": 9})


def ours(models):
    rows = []
    for m in models:
        r = pd.read_csv(RES / f"retrieval_{m}.csv")
        for setting, (ds, proto, key, *_rest) in SETTINGS.items():
            for d, _ in DIRS:
                g = r[(r.dataset == ds) & (r.protocol == proto) & (r.direction == d)]
                for meth in METHODS:
                    x = g[g.method == meth]
                    if meth in ("M1", "M4"):
                        x = x[(x.calib_n == MAIN_N) & (x.calib_source == f"{ds}_train")]
                    if not len(x):
                        continue
                    num = ["R@1", "R@1_lo", "R@1_hi", "dR@1", "dR@1_lo", "dR@1_hi"]
                    mean = x[num].mean()
                    rows.append({
                        "model": m, "setting": setting, "dataset": ds, "protocol": proto, "direction": d,
                        "method": meth, "calib_source": x.calib_source.iloc[0] if meth != "M0" else "",
                        "uses_test_statistics": bool(x.uses_test_statistics.iloc[0]),
                        "n_seeds": int(x.seed.notna().sum()) if meth in ("M1", "M4") else 0,
                        "param": "/".join(f"{v:g}" for v in x.param.dropna()),
                        **{c: float(mean[c]) for c in num},
                        "dR@1_sd_seeds": float(x["dR@1"].std()) if len(x) > 1 else float("nan"),
                        "reported_entry": key, "git_commit": x.git_commit.iloc[0],
                    })
    return pd.DataFrame(rows)


def reported():
    paper = json.loads((ROOT / "results" / "paper_reported.json").read_text(encoding="utf-8"))
    out = {}
    for setting, (_ds, _p, key, base, corr, _t) in SETTINGS.items():
        rows = {r["method"]: r for r in paper[key]["rows"]}
        out[setting] = {"model": paper[key]["model"] or "backbone not named",
                        **{d: (rows[base][f"{d}_R@1"], rows[corr][f"{d}_R@1"]) for d, _ in DIRS}}
    return out


def figure(df, rep):
    models = [m for m in MODELS if m in set(df.model)]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 1.2 + 0.55 * (1 + 3 * len(models)) * 2), sharex=True)
    for r, (setting, (*_x, title)) in enumerate(SETTINGS.items()):
        for c, (d, dname) in enumerate(DIRS):
            ax = axes[r, c]
            labels, y = [], 0
            base, corr = rep[setting][d]
            ax.plot(corr - base, y, marker="s", ms=7, mfc="white", mec=REPORTED_COLOR, mew=1.6, ls="none")
            labels.append(f"Reported ({setting}, {rep[setting]['model']})")
            y += 1
            for m in models:
                for meth in ["M1", "M1-T", "M4"]:
                    q = df[(df.model == m) & (df.setting == setting) & (df.direction == d) & (df.method == meth)]
                    if not len(q):
                        continue
                    q = q.iloc[0]
                    col, mk = STYLE[meth]
                    ax.errorbar(100 * q["dR@1"], y, xerr=[[100 * (q["dR@1"] - q["dR@1_lo"])], [100 * (q["dR@1_hi"] - q["dR@1"])]],
                                color=col, marker=mk, ms=6, capsize=2.5, lw=1.5, ls="none")
                    labels.append(f"Ours {SHORT[m]} {meth}" + (" (test stats)" if meth == "M1-T" else ""))
                    y += 1
            ax.axvline(0, color="0.4", lw=1)
            ax.set_yticks(range(len(labels)), labels if c == 0 else [""] * len(labels))
            ax.set_ylim(len(labels) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{title} — {dname}", fontsize=9)
            if r == 1:
                ax.set_xlabel("ΔRecall@1 vs. raw embeddings (pp)")
    fig.text(0.01, 0.005, "Hollow square: value reported by the paper (its own baseline, backbone and statistics).\n"
                          "Filled: our measurement, 95% paired-bootstrap CI (M1/M4: mean over 3 calibration seeds).",
             fontsize=7.5, color="0.3")
    fig.suptitle("Centering in the literature settings: reported vs. measured", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(FIG / "fig_literature_comparison.png", dpi=150)
    plt.close(fig)


def main():
    models = [m for m in MODELS if (RES / f"retrieval_{m}.csv").exists()]
    if not models:
        print("no retrieval results; skipped")
        return
    df = ours(models)
    df.to_csv(RES / "literature_comparison.csv", index=False)
    FIG.mkdir(parents=True, exist_ok=True)
    figure(df, reported())
    print("wrote results/improvement/literature_comparison.csv,", len(df), "rows; models:", ", ".join(models))


if __name__ == "__main__":
    main()
