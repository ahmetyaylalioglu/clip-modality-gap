"""Phase C, zero-shot classification cost of the gap corrections (RQ3, H5).

Same splits as scripts/b2_shift_zero_shot.py (calibration 5k / validation
5k from the train split; EuroSAT seeded 80/20). Methods:
  M0  raw embeddings
  M1  norm(x - mu_x)          image mean from calibration images
  M2  x - mu_x (no renorm)    dot-product scoring
  M3  Mind-the-Gap shift, direction from calibration images + class
      prompts, delta chosen on validation accuracy
  M4  norm(x - alpha*mu_x), alpha in {0,...,1} chosen on validation accuracy
Text mean (two variants, both reported):
  'prompts'     mean of the class-prompt embeddings (as in the original code)
  'coco_caps'   mean of 10k COCO-train calibration captions (retrieval calib.)
Also reported: Predicted-Class Gini (Sato et al. 2026, Sec. 5.1) on test.

Usage:  python scripts/c2_zero_shot.py --model clip_b16
Output: results/improvement/zero_shot_<model>.csv
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from b2_shift_zero_shot import DATASETS, splits  # noqa: E402
from c1_retrieval import MAIN_N, calib_means, train_split  # noqa: E402
from src.gap import ALPHA_GRID, SHIFT_GRID, center, gap_vector, partial_center, shift  # noqa: E402
from src.metrics import N_BOOT, gini  # noqa: E402

OUT = ROOT / "results" / "improvement"


def preds(img, txt):
    return (img @ txt.T).argmax(1)


def boot(x, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    m = x[idx].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def pick(grid, accs):
    cand = np.where(accs == accs.max())[0]
    return float(grid[cand[np.argmin(np.abs(np.asarray(grid)[cand]))]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    _, coco_mt, _ = calib_means(train_split(m, "mscoco"), MAIN_N, 0)
    rows = []
    for ds in DATASETS:
        s = splits(m, ds)
        txt = s["txt"]
        (te, yte), (cal, _), (val, yval) = s["test"], s["cal"], s["val"]
        n_cls = len(txt)
        mu_i = cal.mean(0)
        h0 = preds(te, txt) == yte

        def add(method, txt_mean, p_te, param=None):
            h = p_te == yte
            a = boot(h.astype(float))
            d = boot(h.astype(float) - h0.astype(float))
            rows.append({"model": m, "dataset": ds, "method": method, "text_mean": txt_mean, "param": param,
                         "n_test": len(yte), "acc": a[0], "acc_lo": a[1], "acc_hi": a[2],
                         "d_acc": d[0], "d_acc_lo": d[1], "d_acc_hi": d[2],
                         "pred_class_gini": gini(np.bincount(p_te, minlength=n_cls)),
                         "true_class_gini": gini(np.bincount(yte, minlength=n_cls))})

        add("M0", "", preds(te, txt))
        for tm_name, mu_t in [("prompts", txt.mean(0)), ("coco_caps", coco_mt)]:
            i1, t1 = center(te, txt, mu_i, mu_t, True)
            add("M1", tm_name, preds(i1, t1))
            i2, t2 = center(te, txt, mu_i, mu_t, False)
            add("M2", tm_name, preds(i2, t2))
            va = np.array([(preds(partial_center(val, mu_i, a), partial_center(txt, mu_t, a)) == yval).mean()
                           for a in ALPHA_GRID])
            a_star = pick(ALPHA_GRID, va)
            add("M4", tm_name, preds(partial_center(te, mu_i, a_star), partial_center(txt, mu_t, a_star)), a_star)
        dl = gap_vector(cal, txt)  # calibration images vs class prompts (B2b 'calib_train')
        vd = np.array([(preds(*shift(val, txt, d, dl)) == yval).mean() for d in SHIFT_GRID])
        d_star = pick(SHIFT_GRID, vd)
        add("M3", "prompts", preds(*shift(te, txt, d_star, dl)), d_star)
        for r in rows[-8:]:
            print(f"{ds:9s} {r['method']:3s} {r['text_mean']:9s} acc={r['acc']*100:6.2f} "
                  f"d={r['d_acc']*100:+.2f} [{r['d_acc_lo']*100:+.2f},{r['d_acc_hi']*100:+.2f}] "
                  f"gini={r['pred_class_gini']:.3f} param={r['param']}")

    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    df.to_csv(OUT / f"zero_shot_{m}.csv", index=False)


if __name__ == "__main__":
    main()
