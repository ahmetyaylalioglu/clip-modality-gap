"""Write results/SUMMARY.md from the result files (facts only, no interpretation).

Every number is read from a CSV/JSON in results/; each section names its
source file. Sections follow the report structure of the assignment:
reproduction -> improvement -> controls/ablations -> uncertainty and
hypothesis tests -> resources -> failure analysis.
Models without result files are skipped.

Usage:  python scripts/make_summary.py
"""
import json
import subprocess
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP, IMP = ROOT / "results" / "reproduction", ROOT / "results" / "improvement"
MODELS = ["clip_b16", "clip_b32", "siglip_b16"]
NAMES = {"clip_b16": "CLIP ViT-B/16 (OpenAI)", "clip_b32": "CLIP ViT-B/32 (OpenAI)",
         "siglip_b16": "SigLIP ViT-B/16 (WebLI)"}
SETTINGS = [("flickr30k", "5cap", "Flickr30k 1k, 5 captions/image"),
            ("flickr30k", "first", "Flickr30k 1k, first caption only"),
            ("mscoco", "5cap", "COCO 5k, 5 captions/image")]
DIRS = {"i2t": "Image→Text", "t2i": "Text→Image"}
out = []


def w(s=""):
    out.append(s)


def table(header, rows):
    w("| " + " | ".join(header) + " |")
    w("|" + "|".join("---" for _ in header) + "|")
    for r in rows:
        w("| " + " | ".join(str(x) for x in r) + " |")
    w()


def pct(x, d=2):
    return f"{100 * x:.{d}f}"


def ci(est, lo, hi, d=2, sign=False):
    f = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    return f"{f.format(100 * est)} [{f.format(100 * lo)}, {f.format(100 * hi)}]"


def excl0(lo, hi):
    return "yes (>0)" if lo > 0 else ("yes (<0)" if hi < 0 else "no")


def seedmean(df, keys):
    num = df.select_dtypes("number").columns
    g = df.groupby(keys, sort=False)
    m = g[num].mean()
    m["R@1_sd"] = g["R@1"].std()
    m["n_seeds"] = g.size()
    return m.reset_index()


# ---------------------------------------------------------------- sections
def reproduction(models):
    w("## 1. Reproduction (Mind the Gap, Table 1)")
    w()
    paper = json.loads((ROOT / "results" / "paper_reported.json").read_text(encoding="utf-8"))
    pr = {r["dataset"]: r for r in paper["mind_the_gap_table1"]["rows"]}
    b1 = pd.read_csv(REP / "table1_original_code_summary.csv")
    w("### 1.1 Paper vs. original code vs. our re-implementation (CLIP ViT-B/16, zero-shot top-1, %)")
    w()
    w("Paper values: `results/paper_reported.json` (reported by the authors, not measured by us). "
      "Original code: `results/reproduction/table1_original_code_summary.csv` (env-orig). "
      "Re-implementation: `results/reproduction/shift_zero_shot_clip_b16.csv` (columns B2_*). "
      "\"Test-selected\" = maximum over the 40-value shift grid evaluated on the test set, as in the original code. "
      "EuroSAT: whole dataset (27,000 images), as in the original code.")
    w()
    b2 = pd.read_csv(REP / "shift_zero_shot_clip_b16.csv") if (REP / "shift_zero_shot_clip_b16.csv").exists() else None
    dmap = {"test": "orig_test", "coco_val2017": "orig_coco"}
    rows = []
    for _, r in b1.iterrows():
        p = pr[r.dataset]
        re_ = ""
        if b2 is not None:
            x = b2[(b2.dataset == r.dataset.lower()) & (b2.direction == dmap[r.direction_source])]
            if len(x):
                re_ = f"{pct(x.B2_baseline_acc.iloc[0])} → {pct(x.B2_oracle_acc_test_selected.iloc[0])}"
        rows.append([r.dataset, f"{pct(p['original'])} → {pct(p['modified'])}", r.direction_source,
                     f"{pct(r.baseline_hit1)} → {pct(r.oracle_hit1_test_selected)} (δ={r.oracle_delta:g})", re_])
    table(["Dataset", "Paper (orig. → modified)", "Gap direction", "Original code (δ=0 → test-selected)",
           "Re-implementation (δ=0 → test-selected)"], rows)

    for m in models:
        f = REP / f"shift_zero_shot_{m}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        w(f"### 1.2 Leakage-free re-test (RQ1) — {NAMES[m]}")
        w()
        w(f"Source: `results/reproduction/shift_zero_shot_{m}.csv`. δ chosen on a validation split; gap "
          "direction from calibration images + class prompts (`calib_train`) or COCO-train pairs (`coco_train`). "
          "Gain = accuracy − δ=0 accuracy, percentage points, 95% paired bootstrap CI. "
          "EuroSAT on a seeded 20% test split.")
        w()
        rows = []
        for _, r in d[d.direction.isin(["calib_train", "coco_train"])].iterrows():
            rows.append([r.dataset, r.direction, pct(r.baseline_acc), f"{r.val_selected_delta:g}",
                         ci(r.val_gain, r.val_gain_ci_lo, r.val_gain_ci_hi, sign=True),
                         excl0(r.val_gain_ci_lo, r.val_gain_ci_hi),
                         f"{ci(r.oracle_gain_test_selected, r.oracle_gain_ci_lo, r.oracle_gain_ci_hi, sign=True)} "
                         f"(δ={r.oracle_delta_test_selected:g})"])
        table(["Dataset", "Direction", "δ=0 acc", "val-selected δ", "Val-selected gain [95% CI]",
               "CI excludes 0", "Test-selected gain [95% CI] (oracle)"], rows)

    w("### 1.3 Gap geometry (COCO test, 1,000 pairs, first caption)")
    w()
    rows = []
    for m in models:
        f = REP / f"gap_stats_{m}.csv"
        if f.exists():
            for _, r in pd.read_csv(f).iterrows():
                rows.append([m, r.variant, f"{r.gap_norm:.3f}", f"{r.cos_img_img:.3f}", f"{r.cos_txt_txt:.3f}",
                             f"{r.cos_img_txt_all:.3f}", f"{r.cos_img_txt_paired:.3f}"])
    w("Source: `results/reproduction/gap_stats_<model>.csv`; figure `results/figures/gap_pca_umap_<model>.png`.")
    w()
    table(["Model", "Embeddings", "‖Δ‖", "mean cos img–img", "mean cos txt–txt", "mean cos img–txt (all)",
           "mean cos img–txt (paired)"], rows)

    ref = paper.get("openclip_retrieval_results")
    if ref:
        w("### 1.4 Baseline retrieval (M0) vs. open_clip reference table (R@1, %)")
        w()
        w("Reference: `results/paper_reported.json` → `openclip_retrieval_results` (published by open_clip, "
          "not measured by us). 5 captions per image.")
        w()
        key = {"clip_b16": ("ViT-B-16", "openai"), "clip_b32": ("ViT-B-32", "openai"),
               "siglip_b16": ("ViT-B-16-SigLIP", "webli")}
        rows = []
        for m in models:
            f = IMP / f"retrieval_{m}.csv"
            if not f.exists():
                continue
            r = pd.read_csv(f)
            r = r[(r.method == "M0") & (r.protocol == "5cap")]
            rr = [x for x in ref["rows"] if (x["name"], x["pretrained"]) == key[m]][0]
            for ds, nm in [("flickr30k", "flickr30k"), ("mscoco", "mscoco")]:
                for d in ["i2t", "t2i"]:
                    ours = r[(r.dataset == ds) & (r.direction == d)]["R@1"].iloc[0]
                    theirs = rr[f"{nm}_{d}_R@1/5/10"][0]
                    rows.append([m, ds, DIRS[d], pct(ours), pct(theirs), f"{100 * (ours - theirs):+.2f}"])
        table(["Model", "Dataset", "Direction", "Ours", "open_clip table", "Difference (pp)"], rows)


def retrieval(m):
    f = IMP / f"retrieval_{m}.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    w(f"### 2.1 Retrieval — {NAMES[m]}")
    w()
    w(f"Source: `results/improvement/retrieval_{m}.csv`. Calibration: 10k in-domain train pairs; "
      "values are the mean over 3 calibration seeds of the per-seed estimate and per-seed 95% bootstrap CI "
      "bounds (1,000 resamples over test queries); sd = standard deviation of R@1 across seeds. "
      "dR@1 = paired difference to M0. M3: δ, M4: α, chosen on validation (per seed, per direction).")
    w()
    main = r[r.method.isin(["M0", "M1", "M2", "M3", "M4"]) &
             ((r.method == "M0") | ((r.calib_n == 10000) & (r.calib_source == r.dataset + "_train")))]
    s = seedmean(main, ["dataset", "protocol", "direction", "method"])
    params = main.groupby(["dataset", "protocol", "direction", "method"])["param"].apply(
        lambda v: "/".join(f"{x:g}" for x in v.dropna()) or "")
    for d, dname in DIRS.items():
        w(f"**{dname}** (%)")
        w()
        rows = []
        for ds, proto, title in SETTINGS:
            for meth in ["M0", "M1", "M2", "M3", "M4"]:
                x = s[(s.dataset == ds) & (s.protocol == proto) & (s.direction == d) & (s.method == meth)]
                if not len(x):
                    continue
                x = x.iloc[0]
                sd = "" if meth == "M0" else f"{100 * x['R@1_sd']:.2f}"
                rows.append([title, meth, params.get((ds, proto, d, meth), ""),
                             ci(x["R@1"], x["R@1_lo"], x["R@1_hi"]), sd,
                             "" if meth == "M0" else ci(x["dR@1"], x["dR@1_lo"], x["dR@1_hi"], sign=True),
                             pct(x["R@5"]), pct(x["R@10"])])
        table(["Setting", "Method", "δ / α (seeds 0/1/2)", "R@1 [95% CI]", "sd", "dR@1 vs M0 [95% CI]", "R@5", "R@10"],
              rows)


def zero_shot(m):
    f = IMP / f"zero_shot_{m}.csv"
    if not f.exists():
        return
    z = pd.read_csv(f)
    w(f"### 2.2 Zero-shot classification — {NAMES[m]}")
    w()
    w(f"Source: `results/improvement/zero_shot_{m}.csv`. Image mean from 5k calibration images; text mean "
      "from the class prompts (`prompts`) or from 10k COCO-train captions (`coco_caps`). "
      "Accuracy and paired difference to M0 with 95% bootstrap CIs (%). Predicted-Class Gini as in Sato et al. 2026, §5.1. "
      "EuroSAT on a seeded 20% test split.")
    w()
    rows = []
    for _, r in z.iterrows():
        rows.append([r.dataset, r.method, "" if pd.isna(r.text_mean) else r.text_mean,
                     "" if pd.isna(r.param) else f"{r.param:g}", ci(r.acc, r.acc_lo, r.acc_hi),
                     "" if r.method == "M0" else ci(r.d_acc, r.d_acc_lo, r.d_acc_hi, sign=True),
                     f"{r.pred_class_gini:.3f}"])
    table(["Dataset", "Method", "Text mean", "δ / α", "Accuracy [95% CI]", "Δ vs M0 [95% CI]", "Pred.-class Gini"],
          rows)


def controls(m):
    f = IMP / f"retrieval_{m}.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    w(f"### 3.1 Controls and calibration-source ablation — {NAMES[m]}")
    w()
    w("C1: shift by a random vector with ‖r‖ = ‖Δ‖ (5 seeds; 10k-pair Δ from 2 calibration seeds for Flickr30k, "
      "1 for COCO, as stored). M1-OOD: means from the other dataset's train split. "
      "**M1-T is diagnostic only: it uses test-set statistics.** dR@1 vs M0 in pp, mean over seeds.")
    w()
    s = seedmean(r[r.method.isin(["C1", "M1-OOD", "M1-T"])], ["dataset", "protocol", "direction", "method"])
    rows = []
    for ds, proto, title in SETTINGS:
        for meth in ["C1", "M1-OOD", "M1-T"]:
            row = [title, meth]
            for d in DIRS:
                x = s[(s.dataset == ds) & (s.protocol == proto) & (s.direction == d) & (s.method == meth)]
                row += [pct(x["R@1"].iloc[0]), ci(x["dR@1"].iloc[0], x["dR@1_lo"].iloc[0], x["dR@1_hi"].iloc[0],
                                                  sign=True)] if len(x) else ["", ""]
            row.append(int(x["n_seeds"].iloc[0]) if len(x) else "")
            rows.append(row)
    table(["Setting", "Method", "I2T R@1", "I2T dR@1 [95% CI]", "T2I R@1", "T2I dR@1 [95% CI]", "rows averaged"], rows)

    w(f"### 3.2 Calibration-set size (M1, in-domain) — {NAMES[m]}")
    w()
    w("R@1 (%) mean ± sd over 3 calibration seeds. Figure: `results/figures/fig_calib_size_<model>.png`.")
    w()
    c = r[(r.method == "M1") & (r.calib_source == r.dataset + "_train")]
    rows = []
    for ds, proto, title in SETTINGS:
        for d, dname in DIRS.items():
            x = c[(c.dataset == ds) & (c.protocol == proto) & (c.direction == d)].groupby("calib_n")["R@1"]
            mm, sd = x.mean(), x.std()
            m0 = r[(r.method == "M0") & (r.dataset == ds) & (r.protocol == proto) & (r.direction == d)]["R@1"].iloc[0]
            rows.append([title, dname, pct(m0)] + [f"{100 * mm[n]:.2f} ± {100 * sd[n]:.2f}" for n in [100, 1000, 10000]])
    table(["Setting", "Direction", "M0", "n=100", "n=1k", "n=10k"], rows)


def hypotheses(m):
    fr, fz = IMP / f"retrieval_{m}.csv", IMP / f"zero_shot_{m}.csv"
    ft, fs = IMP / f"hypothesis_tests_{m}.csv", IMP / f"hubness_spearman_{m}.csv"
    if not all(f.exists() for f in [fr, fz, ft, fs]):
        return
    r, z, t = pd.read_csv(fr), pd.read_csv(fz), pd.read_csv(ft)
    w(f"### 4.1 Pre-registered hypotheses — {NAMES[m]}")
    w()
    w("Hypotheses: `results/improvement/HYPOTHESES.md` (committed before Phase C). Each row gives the "
      "difference (pp), its 95% paired-bootstrap CI and whether the CI excludes 0, per calibration seed "
      "(seeds 0/1/2) where applicable.")
    w()
    rows = []
    m1 = r[(r.method == "M1") & (r.calib_n == 10000) & (r.calib_source == r.dataset + "_train")]
    for hyp, ds, d in [("H1", "flickr30k", "t2i"), ("H2", "mscoco", "i2t")]:
        x = m1[(m1.dataset == ds) & (m1.protocol == "5cap") & (m1.direction == d)]
        for _, q in x.iterrows():
            rows.append([hyp, f"M1 − M0, {ds} 5cap {DIRS[d]}", int(q.seed), ci(q["dR@1"], q["dR@1_lo"], q["dR@1_hi"], sign=True),
                         excl0(q["dR@1_lo"], q["dR@1_hi"])])
    for name in ["M1-T - M1", "M1 - M1-OOD", "M4 - M1"]:
        x = t[t.comparison == name]
        for _, q in x.iterrows():
            rows.append([q.hypothesis, f"{name.replace('-', '−').replace('M1−T', 'M1-T').replace('M1−OOD', 'M1-OOD')}, "
                                       f"{q.dataset} {q.protocol} {DIRS[q.direction]}", int(q.seed),
                         ci(q["dR@1"], q["dR@1_lo"], q["dR@1_hi"], sign=True), excl0(q["dR@1_lo"], q["dR@1_hi"])])
    m4 = r[(r.method == "M4")]
    for _, q in m4.iterrows():
        rows.append(["H6", f"M4 − M0, {q.dataset} {q.protocol} {DIRS[q.direction]} (α={q.param:g})", int(q.seed),
                     ci(q["dR@1"], q["dR@1_lo"], q["dR@1_hi"], sign=True), excl0(q["dR@1_lo"], q["dR@1_hi"])])
    for _, q in z[(z.method == "M1")].iterrows():
        rows.append(["H5", f"M1 − M0 zero-shot, {q.dataset} (text mean: {q.text_mean})", "",
                     ci(q.d_acc, q.d_acc_lo, q.d_acc_hi, sign=True), excl0(q.d_acc_lo, q.d_acc_hi)])
    table(["Hyp.", "Comparison", "Seed", "Difference [95% CI] (pp)", "CI excludes 0"], rows)

    w(f"### 4.2 H4: Spearman of ΔGini@k vs ΔR@1 — {NAMES[m]}")
    w()
    w(f"Source: `results/improvement/hubness_spearman_{m}.csv` (pre-registered: all non-M0 rows) and "
      f"`hubness_spearman_noC1_{m}.csv` (post-hoc sensitivity: without the C1 control rows). "
      "95% CI from 1,000 bootstrap resamples of the rows.")
    w()
    rows = []
    for tag, fn in [("all non-M0 (pre-registered)", fs), ("without C1 (post-hoc)", IMP / f"hubness_spearman_noC1_{m}.csv")]:
        if not fn.exists():
            continue
        for _, q in pd.read_csv(fn).iterrows():
            rows.append([tag, q.dataset, q.protocol, DIRS[q.direction], q.k, q.n,
                         f"{q.spearman:.2f} [{q.ci_lo:.2f}, {q.ci_hi:.2f}]",
                         "yes" if (q.ci_lo > 0 or q.ci_hi < 0) else "no"])
    table(["Rows", "Dataset", "Protocol", "Direction", "k", "n", "Spearman ρ [95% CI]", "CI excludes 0"], rows)

    fh = IMP / f"retrieval_hubness_{m}.csv"
    if fh.exists():
        h = pd.read_csv(fh)
        h = h[h.method.isin(["M0", "M1", "M2", "M3", "M4"]) &
              ((h.method == "M0") | ((h.calib_n == 10000) & (h.calib_source == h.dataset + "_train")))]
        g = h.groupby(["dataset", "protocol", "direction", "method"], sort=False)[["gini@10", "top1pct_share@10"]].mean()
        w(f"### 4.3 Gallery-occupancy hubness at k=10 — {NAMES[m]}")
        w()
        w(f"Source: `results/improvement/retrieval_hubness_{m}.csv`; mean over calibration seeds. "
          "Top-1% share = share of top-10 slots taken by the 1% most retrieved gallery items.")
        w()
        rows = []
        for ds, proto, title in SETTINGS:
            for d, dname in DIRS.items():
                row = [title, dname]
                for meth in ["M0", "M1", "M2", "M3", "M4"]:
                    k = (ds, proto, d, meth)
                    row.append(f"{g.loc[k, 'gini@10']:.3f} / {100 * g.loc[k, 'top1pct_share@10']:.1f}%" if k in g.index else "")
                rows.append(row)
        table(["Setting", "Direction", "M0 Gini / top-1%", "M1", "M2", "M3", "M4"], rows)


def resources(models):
    w("## 5. Resources")
    w()
    rows = []
    for m in models:
        f = IMP / f"latency_{m}.csv"
        if f.exists():
            for _, q in pd.read_csv(f).iterrows():
                rows.append([m, q.device, DIRS[q.direction], q.operation, f"{q.median_us:.1f}", f"{q.p95_us:.1f}"])
    if rows:
        w("Per-query latency of the correction (µs, 1,000 queries after 100 warm-up calls); source "
          "`results/improvement/latency_<model>.csv` (hardware in the CSV).")
        w()
        table(["Model", "Device", "Direction", "Operation", "Median", "p95"], rows)
    f = ROOT / "results" / "resources" / "embedding_benchmark.csv"
    if f.exists():
        e = pd.read_csv(f)
        w("Embedding extraction: source `results/resources/embedding_benchmark.csv`.")
        w()
        table(list(e.columns), e.values.tolist())
    else:
        w("Embedding extraction benchmark: not yet run.")
        w()


def failures(models):
    w("## 6. Failure analysis")
    w()
    w("Rank changes of individual test queries relative to M0 (in-domain calibration, 10k pairs, seed 0, "
      "5 captions/image). The 20 most improved and 20 most worsened queries per setting, with captions and image "
      "IDs, are in `results/improvement/failure_cases_<model>.json`.")
    w()
    rows = []
    for m in models:
        f = IMP / f"rank_change_counts_{m}.csv"
        if f.exists():
            for _, q in pd.read_csv(f).iterrows():
                rows.append([m, q.dataset, q.method, "" if pd.isna(q.param) else f"{q.param:g}", DIRS[q.direction],
                             q.n_queries, q.improved, q.worsened, q.unchanged, q.top1_gained, q.top1_lost])
    table(["Model", "Dataset", "Method", "α", "Direction", "Queries", "Rank improved", "Rank worsened", "Unchanged",
           "Became top-1", "Lost top-1"], rows)


def main():
    models = [m for m in MODELS if (IMP / f"retrieval_{m}.csv").exists() or (REP / f"shift_zero_shot_{m}.csv").exists()]
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    w("# Results summary")
    w()
    w(f"Generated by `scripts/make_summary.py` from the files in `results/` (repository commit {commit} at "
      "generation time; each result file also records the commit that produced it). Facts only; no interpretation. "
      "Numbers reported by other papers are kept in `results/paper_reported.json` and are labeled as such.")
    w()
    w("Models with results: " + ", ".join(NAMES[m] for m in models) + ".")
    w()
    w("Methods: M0 raw; M1 mean-centering + re-normalization (I0T_post / GR-CLIP form, train-split means); "
      "M2 centering without re-normalization; M3 Mind-the-Gap shift with δ chosen on validation; "
      "M4 norm(x − α·μ) with α chosen on validation per direction; C1 random-shift control; "
      "M1-OOD out-of-domain calibration; M1-T test statistics (diagnostic only). None of the methods is ours.")
    w()
    reproduction(models)
    w("## 2. Improvement: gap corrections")
    w()
    for m in models:
        retrieval(m)
        zero_shot(m)
    w("## 3. Controls and ablations")
    w()
    for m in models:
        controls(m)
    w("## 4. Uncertainty and hypothesis tests")
    w()
    for m in models:
        hypotheses(m)
    resources(models)
    failures(models)
    (ROOT / "results" / "SUMMARY.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote results/SUMMARY.md,", len(out), "lines")


if __name__ == "__main__":
    main()
