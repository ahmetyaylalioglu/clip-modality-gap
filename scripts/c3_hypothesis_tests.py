"""Phase C: direct paired tests for the comparisons named in
results/improvement/HYPOTHESES.md that are not differences to M0
(H3: M1-T vs M1, M1 vs M1-OOD; H6: M4 vs M1), plus the H4 Spearman
correlation of dGini vs dR@1.

Reuses the splits, calibration draws and selected alphas of
scripts/c1_retrieval.py (alphas are read from its output CSV, i.e. chosen
on validation). Paired bootstrap over test queries, 1,000 resamples.

Usage:  python scripts/c3_hypothesis_tests.py --model clip_b16
Output: results/improvement/hypothesis_tests_<model>.csv
        results/improvement/hubness_spearman_<model>.csv
        results/improvement/hubness_spearman_noC1_<model>.csv (post-hoc, without C1)
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from c1_retrieval import CAL_SEEDS, MAIN_N, OTHER, calib_means, ci, load, subset, train_split  # noqa: E402
from src.gap import center, partial_center  # noqa: E402
from src.metrics import N_BOOT, retrieval_ranks  # noqa: E402

OUT = ROOT / "results" / "improvement"


def hits(img, txt, owner):
    i2t, t2i, _ = retrieval_ranks(img, txt, owner)
    return {"i2t": i2t < 1, "t2i": t2i < 1}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    p.add_argument("--spearman-only", action="store_true", help="skip the paired tests (reuse their CSV)")
    args = p.parse_args()
    m = args.model
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if not args.spearman_only:
        paired_tests(m, commit)
    spearman_all(m, commit)


def paired_tests(m, commit):
    res = pd.read_csv(OUT / f"retrieval_{m}.csv")
    trains = {ds: train_split(m, ds) for ds in ["flickr30k", "mscoco"]}
    rows = []
    for ds in ["flickr30k", "mscoco"]:
        te_img, te_txt, te_owner, _ = load(m, f"{ds}_test")
        for proto in (["5cap", "first"] if ds == "flickr30k" else ["5cap"]):
            I, T, O, _ = subset(te_img, te_txt, te_owner, np.arange(len(te_img)), proto == "first")
            h_t = hits(*center(I, T, I.mean(0), T.mean(0), True), O)            # M1-T (diagnostic)
            for seed in CAL_SEEDS:
                mi, mt, _ = calib_means(trains[ds], MAIN_N, seed)
                oi, ot, _ = calib_means(trains[OTHER[ds]], MAIN_N, seed)
                h_m1 = hits(*center(I, T, mi, mt, True), O)
                h_ood = hits(*center(I, T, oi, ot, True), O)
                sel = res[(res.dataset == ds) & (res.protocol == proto) & (res.method == "M4") & (res.seed == seed)]
                for d in ["i2t", "t2i"]:
                    a = float(sel[sel.direction == d].param.iloc[0])
                    h_m4 = hits(partial_center(I, mi, a), partial_center(T, mt, a), O)[d]
                    for name, hyp, x, y in [("M4 - M1", "H6", h_m4, h_m1[d]),
                                            ("M1-T - M1", "H3a", h_t[d], h_m1[d]),
                                            ("M1 - M1-OOD", "H3b", h_m1[d], h_ood[d])]:
                        est, lo, hi = ci(x.astype(np.int8) - y.astype(np.int8))
                        rows.append({"model": m, "dataset": ds, "protocol": proto, "direction": d, "seed": seed,
                                     "hypothesis": hyp, "comparison": name, "alpha_M4": a if hyp == "H6" else None,
                                     "dR@1": est, "dR@1_lo": lo, "dR@1_hi": hi,
                                     "significant": bool(lo > 0 or hi < 0)})
            print(f"[{ds}/{proto}] done", flush=True)

    df = pd.DataFrame(rows)
    df["git_commit"] = commit
    df.to_csv(OUT / f"hypothesis_tests_{m}.csv", index=False)
    print(df.groupby(["hypothesis", "comparison", "dataset", "protocol", "direction"])
          [["dR@1", "dR@1_lo", "dR@1_hi"]].mean().mul(100).round(2).to_string())


def spearman_all(m, commit):
    # H4 (pre-registered): all methods and calibration settings except M0.
    # Post-hoc sensitivity: the same without the C1 random-shift control rows,
    # which have the largest |dGini| and |dR@1| and can dominate the rank correlation.
    for tag, excl, fname in [("all non-M0 rows (pre-registered)", ["M0"], f"hubness_spearman_{m}.csv"),
                             ("without C1 (post-hoc sensitivity)", ["M0", "C1"], f"hubness_spearman_noC1_{m}.csv")]:
        sdf = spearman_table(m, excl)
        sdf["rows"] = tag
        sdf["git_commit"] = commit
        sdf.to_csv(OUT / fname, index=False)
        print(tag)
        print(sdf.drop(columns=["git_commit", "rows"]).round(3).to_string(index=False))


def spearman_table(m, exclude):
    """Spearman of dGini@k vs dR@1 across methods and calibration settings."""
    hub = pd.read_csv(OUT / f"retrieval_hubness_{m}.csv")
    base = hub[hub.method == "M0"].set_index(["dataset", "protocol", "direction"])
    sp = []
    for (ds, proto, d), g in hub[~hub.method.isin(exclude)].groupby(["dataset", "protocol", "direction"]):
        for k in (1, 5, 10):
            dg = (g[f"gini@{k}"] - base.loc[(ds, proto, d), f"gini@{k}"]).to_numpy()
            dr = g["dR@1"].to_numpy()
            rho = spearmanr(dg, dr).statistic
            rng = np.random.default_rng(0)
            bs = []
            for _ in range(N_BOOT):
                i = rng.integers(0, len(dg), len(dg))
                if np.unique(dg[i]).size > 1 and np.unique(dr[i]).size > 1:
                    bs.append(spearmanr(dg[i], dr[i]).statistic)
            lo, hi = np.quantile(bs, [0.025, 0.975])
            sp.append({"model": m, "dataset": ds, "protocol": proto, "direction": d, "k": k, "n": len(dg),
                       "spearman": float(rho), "ci_lo": float(lo), "ci_hi": float(hi)})
    return pd.DataFrame(sp)


if __name__ == "__main__":
    main()
