"""Phase C, retrieval: training-free gap corrections on cached embeddings.

Methods (see results/improvement/HYPOTHESES.md; none of them is ours):
  M0    raw L2-normalized embeddings
  M1    norm(x - mu_x), mu from calibration pairs        (I0T_post form)
  M2    x - mu_x, no re-normalization, dot-product score (GR-CLIP form)
  M3    Mind-the-Gap shift, direction from calibration pairs,
        delta chosen on validation, separately for I2T and T2I
  M4    norm(x - alpha*mu_x), alpha in {0,0.1,..,1} chosen on validation,
        separately for I2T and T2I
  C1    control: shift by a random vector with ||r|| = ||Delta|| (5 seeds)
  M1-T  M1 with means from the TEST set (diagnostic only, uses test statistics)
  M1-OOD M1 with calibration pairs from the other dataset's train split

Splits (seeded, SPLIT_SEED):
  train images are permuted; the first N_VAL images (with their captions)
  form the validation set (1k for Flickr30k, 5k for COCO = test gallery
  size); calibration pairs are drawn from the remaining images. Test sets
  are only used for the final evaluation (and for M1-T, labeled).

Protocols: '5cap' (all captions, standard) and, for Flickr30k, 'first'
(first caption per image only, as in I0T).

Usage:  python scripts/c1_retrieval.py --model clip_b16
Output: results/improvement/retrieval_<model>.csv          (one row per method x direction)
        results/improvement/retrieval_hubness_<model>.csv
        results/improvement/retrieval_selection_<model>.csv (val curves for M3/M4)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.gap import ALPHA_GRID, SHIFT_GRID, center, gap_vector, l2n, partial_center, random_shift, shift  # noqa: E402
from src.metrics import N_BOOT, gini, retrieval_ranks, top_share, topk_occupancy  # noqa: E402

CACHE = ROOT / "cache"
OUT = ROOT / "results" / "improvement"
SPLIT_SEED = 0
N_VAL = {"flickr30k": 1000, "mscoco": 5000}
CAL_SIZES = [100, 1000, 10000]
CAL_SEEDS = [0, 1, 2]
MAIN_N = 10000
C1_SEEDS = [0, 1, 2, 3, 4]
KS = (1, 5, 10)
OTHER = {"flickr30k": "mscoco", "mscoco": "flickr30k"}


# ------------------------------------------------------------------ data
def load(model, name):
    d = CACHE / model
    meta = json.loads((d / f"{name}_meta.json").read_text())
    return np.load(d / f"{name}_img.npy"), np.load(d / f"{name}_txt.npy"), np.array(meta["txt_to_img"]), meta


def subset(img, txt, owner, img_idx, first_only=False):
    """Images img_idx with their captions; owner re-indexed."""
    remap = -np.ones(len(img), dtype=int)
    remap[img_idx] = np.arange(len(img_idx))
    keep = np.where(remap[owner] >= 0)[0]
    if first_only:
        _, first = np.unique(owner[keep], return_index=True)
        keep = keep[first]
    keep = keep[np.argsort(remap[owner[keep]], kind="stable")]
    return img[img_idx], txt[keep], remap[owner[keep]], keep


def train_split(model, ds):
    img, txt, owner, _ = load(model, f"{ds}_train")
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(img))
    val_idx, pool_idx = np.sort(perm[:N_VAL[ds]]), perm[N_VAL[ds]:]
    pool_pairs = np.where(np.isin(owner, pool_idx))[0]
    return {"img": img, "txt": txt, "owner": owner, "val_idx": val_idx, "pool_pairs": pool_pairs}


def calib_means(tr, n, seed):
    pairs = tr["pool_pairs"]
    if n > len(pairs):
        raise ValueError(f"calibration size {n} > pool {len(pairs)}")
    pick = np.random.default_rng(1000 + seed).choice(pairs, size=n, replace=False)
    ci, ct = tr["img"][tr["owner"][pick]], tr["txt"][pick]
    return ci.mean(0), ct.mean(0), gap_vector(ci, ct)


# ------------------------------------------------------------------ eval
_IDX = {}


def boot_idx(n):
    if n not in _IDX:
        _IDX[n] = np.random.default_rng(0).integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    return _IDX[n]


def ci(x):
    m = x.astype(np.float32)[boot_idx(len(x))].mean(1)
    lo, hi = np.quantile(m, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def run_eval(img, txt, owner, hub=False):
    i2t, t2i, sim = retrieval_ranks(img, txt, owner)
    out = {"i2t": {"rank": i2t}, "t2i": {"rank": t2i}}
    if hub:
        out["i2t"]["occ"] = topk_occupancy(sim, KS)      # gallery = captions
        out["t2i"]["occ"] = topk_occupancy(sim.T, KS)    # gallery = images
    out["gap_norm"] = float(np.linalg.norm(img.mean(0) - txt.mean(0)))
    return out


def r1(img, txt, owner):
    i2t, t2i, _ = retrieval_ranks(img, txt, owner)
    return float((i2t < 1).mean()), float((t2i < 1).mean())


def select(grid, fn, val, sel_rows, tag):
    """Pick the grid value maximizing validation R@1, per direction.
    Ties -> smallest |value| (closest to no change)."""
    scores = np.array([r1(*fn(val["img"], val["txt"], g), val["owner"]) for g in grid])
    for g, (a, b) in zip(grid, scores):
        sel_rows.append({**tag, "param": float(g), "val_i2t_R@1": a, "val_t2i_R@1": b})
    best = {}
    for j, d in enumerate(["i2t", "t2i"]):
        cand = np.where(scores[:, j] == scores[:, j].max())[0]
        best[d] = float(grid[cand[np.argmin(np.abs(np.asarray(grid)[cand]))]])
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    p.add_argument("--datasets", nargs="*", default=["flickr30k", "mscoco"])
    args = p.parse_args()
    m = args.model
    t0 = time.time()
    trains = {ds: train_split(m, ds) for ds in ["flickr30k", "mscoco"]}
    rows, hub_rows, sel_rows = [], [], []

    for ds in args.datasets:
        te_img, te_txt, te_owner, te_meta = load(m, f"{ds}_test")
        tr = trains[ds]
        protocols = ["5cap", "first"] if ds == "flickr30k" else ["5cap"]
        for proto in protocols:
            first = proto == "first"
            T = dict(zip(["img", "txt", "owner"], subset(te_img, te_txt, te_owner, np.arange(len(te_img)), first)[:3]))
            V = dict(zip(["img", "txt", "owner"], subset(tr["img"], tr["txt"], tr["owner"], tr["val_idx"], first)[:3]))
            base = run_eval(T["img"], T["txt"], T["owner"], hub=True)
            tag0 = {"model": m, "dataset": ds, "protocol": proto, "n_test_images": len(T["img"]),
                    "n_test_captions": len(T["txt"])}

            def record(method, res, extra, params=None):
                """res: dict direction -> eval output from the transform used for that direction."""
                for d in ["i2t", "t2i"]:
                    rk, rk0 = res[d][d]["rank"], base[d]["rank"]
                    row = {**tag0, "method": method, "direction": d, **extra,
                           "param": None if params is None else params[d],
                           "gap_norm_test": res[d]["gap_norm"]}
                    for k in KS:
                        h, h0 = (rk < k), (rk0 < k)
                        row[f"R@{k}"], row[f"R@{k}_lo"], row[f"R@{k}_hi"] = ci(h)
                        row[f"dR@{k}"], row[f"dR@{k}_lo"], row[f"dR@{k}_hi"] = ci(h.astype(np.int8) - h0.astype(np.int8))
                    rows.append(row)
                    if "occ" in res[d][d]:
                        hr = {**tag0, "method": method, "direction": d, **extra,
                              "param": row["param"], "dR@1": row["dR@1"], "R@1": row["R@1"]}
                        for k in KS:
                            occ = res[d][d]["occ"][k]
                            hr[f"gini@{k}"] = gini(occ)
                            hr[f"top1pct_share@{k}"] = top_share(occ, 0.01)
                            hr[f"max_occ@{k}"] = int(occ.max())
                            hr[f"zero_occ_frac@{k}"] = float((occ == 0).mean())
                        hub_rows.append(hr)

            def both(res):
                return {"i2t": res, "t2i": res}

            def ev(fn):
                i, t = fn(T["img"], T["txt"])
                return run_eval(i, t, T["owner"], hub=True)

            info = lambda **kw: {"calib_source": kw.get("src", ""), "calib_n": kw.get("n"), "seed": kw.get("seed"),
                                 "uses_test_statistics": kw.get("test", False)}

            record("M0", both(base), info())
            # M1 / M2 / M3 / M4 with in-domain train calibration
            for seed in CAL_SEEDS:
                for n in CAL_SIZES:
                    mi, mt, dl = calib_means(tr, n, seed)
                    record("M1", both(ev(lambda i, t: center(i, t, mi, mt, True))), info(src=f"{ds}_train", n=n, seed=seed))
                    if n != MAIN_N:
                        continue
                    record("M2", both(ev(lambda i, t: center(i, t, mi, mt, False))), info(src=f"{ds}_train", n=n, seed=seed))
                    tag = {**tag0, "calib_n": n, "seed": seed}
                    bd = select(SHIFT_GRID, lambda i, t, g: shift(i, t, g, dl), V, sel_rows, {**tag, "method": "M3"})
                    record("M3", {d: ev(lambda i, t: shift(i, t, bd[d], dl)) for d in ["i2t", "t2i"]},
                           info(src=f"{ds}_train", n=n, seed=seed), bd)
                    ba = select(ALPHA_GRID, lambda i, t, a: (partial_center(i, mi, a), partial_center(t, mt, a)),
                                V, sel_rows, {**tag, "method": "M4"})
                    record("M4", {d: ev(lambda i, t: (partial_center(i, mi, ba[d]), partial_center(t, mt, ba[d])))
                                  for d in ["i2t", "t2i"]}, info(src=f"{ds}_train", n=n, seed=seed), ba)
                    if seed == 0:
                        for cs in C1_SEEDS:
                            record("C1", both(ev(lambda i, t: random_shift(i, t, dl, cs))),
                                   info(src=f"{ds}_train", n=n, seed=cs))
            # M1-OOD: calibration from the other dataset's train split
            for seed in CAL_SEEDS:
                mi, mt, _ = calib_means(trains[OTHER[ds]], MAIN_N, seed)
                record("M1-OOD", both(ev(lambda i, t: center(i, t, mi, mt, True))),
                       info(src=f"{OTHER[ds]}_train", n=MAIN_N, seed=seed))
            # M1-T: diagnostic, test statistics (image mean over test images, text mean over test captions)
            record("M1-T", both(ev(lambda i, t: center(i, t, T["img"].mean(0), T["txt"].mean(0), True))),
                   info(src="TEST (diagnostic)", n=len(T["txt"]), test=True))
            print(f"[{ds}/{proto}] done ({time.time() - t0:.0f}s)", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for name, data in [(f"retrieval_{m}.csv", rows), (f"retrieval_hubness_{m}.csv", hub_rows),
                       (f"retrieval_selection_{m}.csv", sel_rows)]:
        df = pd.DataFrame(data)
        df["git_commit"] = commit
        df.to_csv(OUT / name, index=False)
    # compact console view: R@1 and delta to M0, main settings
    df = pd.DataFrame(rows)
    view = df[(df.calib_n.isna()) | (df.calib_n == MAIN_N) | (df.method == "M1-T")]
    g = view.groupby(["dataset", "protocol", "direction", "method"], sort=False)
    print(g[["R@1", "dR@1", "dR@1_lo", "dR@1_hi"]].mean().mul(100).round(2).to_string())


if __name__ == "__main__":
    main()
