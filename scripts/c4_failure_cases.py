"""Phase C failure analysis: the 20 test queries whose rank improved most and
the 20 that got worse most under a correction, relative to M0.

Corrections: M1 and M4 with in-domain calibration (10k train pairs,
calibration seed 0; the M4 alpha is the one chosen on validation, read from
results/improvement/retrieval_<model>.csv). Protocol: 5 captions per image.
Ranks are 0-based (0 = correct item retrieved first); for I2T the rank is
that of the best-ranked correct caption. Ties in the rank change are broken
by query index (deterministic).

Image IDs are the WebDataset sample keys of the pinned HF revisions
(REPRODUCE.md §5); the order of the keys is the order of the shards.

Usage:  python scripts/c4_failure_cases.py --model clip_b16
Output: results/improvement/failure_cases_<model>.json
        results/improvement/rank_change_counts_<model>.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
from c1_retrieval import MAIN_N, calib_means, load, train_split  # noqa: E402
from src.gap import center, partial_center  # noqa: E402
from src.metrics import retrieval_ranks  # noqa: E402

OUT = ROOT / "results" / "improvement"
N_SHOW = 20
SEED = 0


def top1(sim):
    return sim.argmax(1), sim.argmax(0)       # I2T: caption index per image; T2I: image index per caption


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    res = pd.read_csv(OUT / f"retrieval_{m}.csv")
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    out = {"model": m, "git_commit": commit, "protocol": "5cap", "calib_n": MAIN_N, "calib_seed": SEED,
           "rank_note": "0-based rank of the correct item; I2T uses the best-ranked of the 5 correct captions",
           "delta_note": "rank_change = rank_method - rank_M0 (negative = improved)", "cases": []}
    counts = []
    for ds in ["flickr30k", "mscoco"]:
        img, txt, owner, meta = load(m, f"{ds}_test")
        keys, caps = meta["keys"], meta["captions"]
        mi, mt, _ = calib_means(train_split(m, ds), MAIN_N, SEED)
        r0i, r0t, s0 = retrieval_ranks(img, txt, owner)
        b0i, b0t = top1(s0)
        sel = res[(res.dataset == ds) & (res.protocol == "5cap") & (res.method == "M4") & (res.seed == SEED)]
        alpha = {d: float(sel[sel.direction == d].param.iloc[0]) for d in ["i2t", "t2i"]}
        variants = {"M1": {d: center(img, txt, mi, mt, True) for d in ["i2t", "t2i"]},
                    "M4": {d: (partial_center(img, mi, alpha[d]), partial_center(txt, mt, alpha[d]))
                           for d in ["i2t", "t2i"]}}
        for meth, per_dir in variants.items():
            for d in ["i2t", "t2i"]:
                ri, rt, s1 = retrieval_ranks(*per_dir[d], owner)
                b1i, b1t = top1(s1)
                r0, r1, b0, b1 = (r0i, ri, b0i, b1i) if d == "i2t" else (r0t, rt, b0t, b1t)
                ch = r1.astype(int) - r0.astype(int)
                counts.append({"model": m, "dataset": ds, "method": meth, "direction": d,
                               "param": alpha[d] if meth == "M4" else None, "n_queries": len(ch),
                               "improved": int((ch < 0).sum()), "worsened": int((ch > 0).sum()),
                               "unchanged": int((ch == 0).sum()),
                               "top1_gained": int(((r0 > 0) & (r1 == 0)).sum()),
                               "top1_lost": int(((r0 == 0) & (r1 > 0)).sum()),
                               "median_rank_change": float(np.median(ch)), "git_commit": commit})
                ar = np.arange(len(ch))
                most_improved = np.lexsort((ar, ch))[:N_SHOW]        # most negative change first
                most_worsened = np.lexsort((ar, -ch))[:N_SHOW]       # most positive change first
                for kind, idx in [("improved", most_improved), ("worsened", most_worsened)]:
                    for q in idx:
                        q = int(q)
                        if (kind == "improved" and ch[q] >= 0) or (kind == "worsened" and ch[q] <= 0):
                            continue
                        if d == "i2t":
                            case = {"query_image_id": keys[q],
                                    "correct_captions": [caps[j] for j in np.where(owner == q)[0]],
                                    "top1_caption_M0": caps[int(b0[q])],
                                    f"top1_caption_{meth}": caps[int(b1[q])]}
                        else:
                            case = {"query_caption": caps[q], "correct_image_id": keys[int(owner[q])],
                                    "top1_image_id_M0": keys[int(b0[q])],
                                    f"top1_image_id_{meth}": keys[int(b1[q])],
                                    "top1_image_M0_first_caption": caps[int(np.where(owner == b0[q])[0][0])],
                                    f"top1_image_{meth}_first_caption": caps[int(np.where(owner == b1[q])[0][0])]}
                        out["cases"].append({"dataset": ds, "method": meth, "direction": d,
                                             "param": alpha[d] if meth == "M4" else None, "kind": kind,
                                             "query_index": q, "rank_M0": int(r0[q]), "rank_method": int(r1[q]),
                                             "rank_change": int(ch[q]), **case})
                print(f"{ds:9s} {meth} {d}: improved {counts[-1]['improved']}, worsened {counts[-1]['worsened']}, "
                      f"top1 +{counts[-1]['top1_gained']}/-{counts[-1]['top1_lost']}", flush=True)
    (OUT / f"failure_cases_{m}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(counts).to_csv(OUT / f"rank_change_counts_{m}.csv", index=False)


if __name__ == "__main__":
    main()
