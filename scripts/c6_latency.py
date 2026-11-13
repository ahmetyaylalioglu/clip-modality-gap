"""Phase C resources: added latency of the gap corrections per query.

For each correction the operation applied to ONE query embedding is timed
(1,000 queries = COCO test captions for T2I and COCO test images for I2T,
after 100 warm-up calls), on CPU (numpy) and on GPU (torch, synchronized).
For context the time of scoring one query against the full gallery
(COCO 5k images or 25k captions) is timed the same way.
The calibration means / directions are precomputed (one-off cost) and not
included. Reports median and p95 in microseconds. On the GPU, single-vector
operations are dominated by kernel-launch overhead, so they are slower than on CPU.

Usage:  python scripts/c6_latency.py --model clip_b16
Output: results/improvement/latency_<model>.csv
"""
import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from c1_retrieval import MAIN_N, calib_means, load, train_split  # noqa: E402

N_Q, N_WARM = 1000, 100


def time_calls(fn, queries, sync=None):
    for q in queries[:N_WARM]:
        fn(q)
    if sync:
        sync()
    ts = []
    for q in queries[:N_Q]:
        t0 = time.perf_counter_ns()
        fn(q)
        if sync:
            sync()
        ts.append(time.perf_counter_ns() - t0)
    ts = np.array(ts) / 1e3
    return float(np.median(ts)), float(np.quantile(ts, 0.95))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="clip_b16")
    args = p.parse_args()
    m = args.model
    img, txt, _, _ = load(m, "mscoco_test")
    mi, mt, gap = calib_means(train_split(m, "mscoco"), MAIN_N, 0)
    u = gap / np.linalg.norm(gap)
    alpha, delta = 0.5, 0.25
    rows = []
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        for d, Q, G, mu, sgn in [("t2i", txt, img, mt, +1), ("i2t", img, txt, mi, -1)]:
            if dev == "cpu":
                Qs, Gs = list(Q[:N_Q + N_WARM]), G
                mu_, u_ = mu, u
                nrm = lambda x: x / np.linalg.norm(x)  # noqa: E731
                sync = None
            else:
                Qs = list(torch.from_numpy(Q[:N_Q + N_WARM]).cuda())
                Gs = torch.from_numpy(G).cuda()
                mu_, u_ = torch.from_numpy(mu).cuda(), torch.from_numpy(u).cuda()
                nrm = lambda x: x / x.norm()  # noqa: E731
                sync = torch.cuda.synchronize
            ops = {
                "M1 norm(q - mu)": lambda q: nrm(q - mu_),
                "M2 q - mu": lambda q: q - mu_,
                "M3 norm(q +- 0.5*delta*u)": lambda q: nrm(q + sgn * 0.5 * delta * u_),
                "M4 norm(q - alpha*mu)": lambda q: nrm(q - alpha * mu_),
                f"(context) score vs gallery of {len(G)}": lambda q: Gs @ q,
            }
            for name, fn in ops.items():
                med, p95 = time_calls(fn, Qs, sync)
                rows.append({"model": m, "device": dev, "direction": d, "operation": name,
                             "n_queries": N_Q, "median_us": med, "p95_us": p95})
                print(f"{dev:4s} {d} {name:38s} median {med:8.2f} us  p95 {p95:8.2f} us", flush=True)
    df = pd.DataFrame(rows)
    df["cpu"] = platform.processor()
    df["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
    df["numpy"], df["torch"] = np.__version__, torch.__version__
    df["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                      capture_output=True, text=True).stdout.strip()
    df.to_csv(ROOT / "results" / "improvement" / f"latency_{m}.csv", index=False)


if __name__ == "__main__":
    main()
