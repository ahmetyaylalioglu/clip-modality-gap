"""Phase B2 + B2b: Mind-the-Gap embedding shift on zero-shot classification,
re-implemented on cached embeddings (env-main).

B2  (re-implementation, original protocol): same directions and shift grid
    as the original code; the "oracle" value is the max over the grid of the
    TEST accuracy (labeled test-selected). Used for the sanity check against
    results/reproduction/table1_original_code_summary.csv.
B2b (leakage-free, RQ1): gap direction from non-test data, delta chosen on a
    validation split, evaluated once on test. Paired bootstrap CIs.

Splits (seeded, see src/embed.py):
  CIFAR10/100, SVHN: test = official test; calibration = first 5k of the
      seeded 10k train subset, validation = remaining 5k.
  EuroSAT (single split): seeded 80/20 split of all 27k images
      (seed 0); test = 20%; calibration 5k and validation 5k from the 80%.

Directions:
  orig_test   : mean(test img) - mean(class prompts)   [original EuroSAT cell; uses test]
  orig_coco   : COCO val2017 pairs                       [original SVHN cell]
  calib_train : mean(calibration img) - mean(class prompts)   [leakage-free]
  coco_train  : COCO train shard pairs                   [leakage-free, out-of-domain]

Usage:  python scripts/b2_shift_zero_shot.py --model clip_b16
Output: results/reproduction/shift_zero_shot_curves_<model>.csv
        results/reproduction/shift_zero_shot_<model>.csv
        results/figures/shift_zero_shot_<model>.png
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gap import SHIFT_GRID, gap_vector, shift  # noqa: E402
from src.metrics import N_BOOT  # noqa: E402

CACHE = ROOT / "cache"
OUT = ROOT / "results" / "reproduction"
FIG = ROOT / "results" / "figures"
DATASETS = ["cifar10", "cifar100", "eurosat", "svhn"]
SPLIT_SEED = 0
N_CAL, N_VAL = 5000, 5000


def load(model, name):
    d = CACHE / model
    meta = json.loads((d / f"{name}_meta.json").read_text())
    return np.load(d / f"{name}_img.npy"), np.load(d / f"{name}_txt.npy"), meta


def splits(model, ds):
    """Return dict with test/cal/val (img, labels) and class-prompt txt."""
    if ds == "eurosat":
        img, txt, meta = load(model, "eurosat_all")
        y = np.array(meta["labels"])
        rng = np.random.default_rng(SPLIT_SEED)
        perm = rng.permutation(len(y))
        n_test = int(round(0.2 * len(y)))
        te, rest = perm[:n_test], perm[n_test:]
        cal, val = rest[:N_CAL], rest[N_CAL:N_CAL + N_VAL]
        return {"test": (img[te], y[te]), "test_all": (img, y), "cal": (img[cal], y[cal]),
                "val": (img[val], y[val]), "txt": txt}
    ti, txt, tm = load(model, f"{ds}_test")
    tr, _, trm = load(model, f"{ds}_train10k")
    ytr = np.array(trm["labels"])
    return {"test": (ti, np.array(tm["labels"])), "test_all": (ti, np.array(tm["labels"])),
            "cal": (tr[:N_CAL], ytr[:N_CAL]), "val": (tr[N_CAL:N_CAL + N_VAL], ytr[N_CAL:N_CAL + N_VAL]),
            "txt": txt}


def hits_curve(img, txt, y, direction):
    """(len(grid), n) boolean hit matrix over the shift grid."""
    out = np.empty((len(SHIFT_GRID), len(y)), dtype=bool)
    for k, d in enumerate(SHIFT_GRID):
        i2, t2 = shift(img, txt, d, direction)
        out[k] = (i2 @ t2.T).argmax(1) == y
    return out


def boot_diff(h_new, h_base, seed=0):
    rng = np.random.default_rng(seed)
    n = len(h_base)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    d = h_new[idx].mean(1) - h_base[idx].mean(1)
    return float(h_new.mean() - h_base.mean()), *map(float, np.quantile(d, [0.025, 0.975]))


def boot_oracle_gain(curve, seed=0):
    """Gain of the test-selected max, re-selected inside every bootstrap
    resample: estimates how large the oracle gain is just from selection."""
    rng = np.random.default_rng(seed)
    n = curve.shape[1]
    base_k = int(np.argmin(np.abs(SHIFT_GRID)))
    gains = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        acc = curve[:, idx].mean(1)
        gains.append(acc.max() - acc[base_k])
    return float(np.mean(gains)), *map(float, np.quantile(gains, [0.025, 0.975]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    base_k = int(np.argmin(np.abs(SHIFT_GRID)))
    assert SHIFT_GRID[base_k] == 0.0

    coco_v_i, coco_v_t, _ = load(m, "coco_val2017")
    coco_t_i, coco_t_t, coco_t_meta = load(m, "mscoco_train")
    # pair-level means for the COCO train shard: each caption with its image
    owner = np.array(coco_t_meta["txt_to_img"])
    dirs_global = {
        "orig_coco": gap_vector(coco_v_i, coco_v_t),
        "coco_train": gap_vector(coco_t_i[owner], coco_t_t),
    }

    curves, rows = [], []
    for ds in DATASETS:
        s = splits(m, ds)
        txt = s["txt"]
        dirs = {
            "orig_test": gap_vector(s["test_all"][0], txt),
            "orig_coco": dirs_global["orig_coco"],
            "calib_train": gap_vector(s["cal"][0], txt),
            "coco_train": dirs_global["coco_train"],
        }
        for dname, dvec in dirs.items():
            # ---- B2: original protocol on the full original test set
            img_all, y_all = s["test_all"]
            c_all = hits_curve(img_all, txt, y_all, dvec)
            acc_all = c_all.mean(1)
            # ---- B2b: our split (EuroSAT: 20% test; others identical)
            img_te, y_te = s["test"]
            c_te = c_all if ds != "eurosat" else hits_curve(img_te, txt, y_te, dvec)
            c_val = hits_curve(s["val"][0], txt, s["val"][1], dvec)
            acc_val = c_val.mean(1)
            k_val = int(np.argmax(acc_val))
            k_orc = int(np.argmax(c_te.mean(1)))
            for k, dl in enumerate(SHIFT_GRID):
                curves.append({"model": m, "dataset": ds, "direction": dname, "delta": float(dl),
                               "acc_test_original_set": float(acc_all[k]), "acc_test_split": float(c_te[k].mean()),
                               "acc_val": float(acc_val[k])})
            g_val = boot_diff(c_te[k_val], c_te[base_k])
            g_orc = boot_diff(c_te[k_orc], c_te[base_k])
            g_sel = boot_oracle_gain(c_te)
            rows.append({
                "model": m, "dataset": ds, "direction": dname,
                "uses_test_data_for_direction": dname == "orig_test",
                "n_test_original_set": len(y_all), "n_test_split": len(y_te),
                # B2 (original protocol, original test set)
                "B2_baseline_acc": float(acc_all[base_k]),
                "B2_oracle_acc_test_selected": float(acc_all.max()),
                "B2_oracle_delta": float(SHIFT_GRID[int(np.argmax(acc_all))]),
                # B2b (our split)
                "baseline_acc": float(c_te[base_k].mean()),
                "val_selected_delta": float(SHIFT_GRID[k_val]),
                "val_selected_acc": float(c_te[k_val].mean()),
                "val_gain": g_val[0], "val_gain_ci_lo": g_val[1], "val_gain_ci_hi": g_val[2],
                "oracle_delta_test_selected": float(SHIFT_GRID[k_orc]),
                "oracle_gain_test_selected": g_orc[0], "oracle_gain_ci_lo": g_orc[1], "oracle_gain_ci_hi": g_orc[2],
                "oracle_gain_reselected_boot_mean": g_sel[0], "oracle_gain_reselected_ci_lo": g_sel[1],
                "oracle_gain_reselected_ci_hi": g_sel[2],
                "gap_norm_direction": float(np.linalg.norm(dvec)),
            })
            r = rows[-1]
            print(f"{ds:9s} {dname:12s} base={r['B2_baseline_acc']:.4f} oracle={r['B2_oracle_acc_test_selected']:.4f}"
                  f" | split: val-sel gain={r['val_gain']*100:+.2f} [{r['val_gain_ci_lo']*100:+.2f},{r['val_gain_ci_hi']*100:+.2f}]"
                  f" oracle gain={r['oracle_gain_test_selected']*100:+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    df = pd.DataFrame(rows)
    df["git_commit"] = commit
    df.to_csv(OUT / f"shift_zero_shot_{m}.csv", index=False)
    cdf = pd.DataFrame(curves)
    cdf.to_csv(OUT / f"shift_zero_shot_curves_{m}.csv", index=False)
    plot(cdf, m)


def plot(cdf, m):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        for dname, sty in [("calib_train", "-"), ("orig_test", "--")]:
            sub = cdf[(cdf.dataset == ds) & (cdf.direction == dname)]
            ax.plot(sub.delta, sub.acc_test_split * 100, sty, label=f"test, {dname}")
            if dname == "calib_train":
                ax.plot(sub.delta, sub.acc_val * 100, ":", label="val, calib_train")
        ax.axvline(0, color="k", lw=0.6)
        ax.set_title(ds)
        ax.set_xlabel("shift δ (original code units)")
    axes[0].set_ylabel("top-1 accuracy (%)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"Mind-the-Gap shift, {m}")
    fig.tight_layout()
    fig.savefig(FIG / f"shift_zero_shot_{m}.png", dpi=150)


if __name__ == "__main__":
    main()
