"""Retrieval and classification metrics.

Retrieval follows the standard multi-caption protocol (as in
clip_benchmark): every image has several captions.
  - image->text (I2T): a query image is a hit at K if ANY of its captions
    is in the top-K texts.
  - text->image (T2I): a query caption is a hit at K if its own image is
    in the top-K images.
"""
from __future__ import annotations

import numpy as np


def recall_at_k(
    img: np.ndarray,
    txt: np.ndarray,
    txt_to_img: np.ndarray,
    ks=(1, 5, 10),
    return_hits: bool = False,
):
    """img: (N_img, d), txt: (N_txt, d), both already in the space to score.
    txt_to_img: (N_txt,) index of the image each caption belongs to.

    Returns dict like {'i2t_R@1': ..., 't2i_R@1': ...}. With return_hits,
    also returns per-query hit arrays (for bootstrap CIs) and per-query ranks.
    """
    sim = img @ txt.T  # (N_img, N_txt)
    n_img = img.shape[0]

    # I2T: rank of the best-ranked correct caption for each image
    order_i2t = np.argsort(-sim, axis=1)
    correct = txt_to_img[order_i2t] == np.arange(n_img)[:, None]
    i2t_rank = correct.argmax(axis=1)  # 0-based rank of first correct caption

    # T2I: rank of the correct image for each caption
    order_t2i = np.argsort(-sim.T, axis=1)
    t2i_rank = (order_t2i == txt_to_img[:, None]).argmax(axis=1)

    out, hits = {}, {}
    for k in ks:
        h_i2t = i2t_rank < k
        h_t2i = t2i_rank < k
        out[f"i2t_R@{k}"] = float(h_i2t.mean())
        out[f"t2i_R@{k}"] = float(h_t2i.mean())
        hits[f"i2t_R@{k}"] = h_i2t
        hits[f"t2i_R@{k}"] = h_t2i
    if return_hits:
        return out, hits, {"i2t": i2t_rank, "t2i": t2i_rank}
    return out


def zero_shot_accuracy(img: np.ndarray, cls_txt: np.ndarray, labels: np.ndarray, return_preds=False):
    preds = (img @ cls_txt.T).argmax(axis=1)
    acc = float((preds == labels).mean())
    return (acc, preds) if return_preds else acc


# ---------------------------------------------------------------- bootstrap
N_BOOT = 1000


def bootstrap_ci(hits: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """Percentile CI of the mean of a per-query 0/1 (or real) array."""
    rng = np.random.default_rng(seed)
    n = len(hits)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = hits[idx].mean(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(hits.mean()), float(lo), float(hi)


def paired_bootstrap_diff(hits_a: np.ndarray, hits_b: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, alpha: float = 0.05):
    """CI of mean(b) - mean(a), resampling the same queries for both."""
    assert hits_a.shape == hits_b.shape
    return bootstrap_ci(hits_b.astype(float) - hits_a.astype(float), n_boot, seed, alpha)


# ------------------------------------------------------------------ hubness
def gini(counts: np.ndarray) -> float:
    """Standard Gini coefficient (Sato et al. 2026 §5.1 form):
    G = 2*sum_r r*m_(r) / (C*sum m) - (C+1)/C, m sorted ascending, r=1..C."""
    m = np.sort(np.asarray(counts, dtype=float))
    c = len(m)
    if m.sum() == 0:
        return 0.0
    r = np.arange(1, c + 1)
    return float(2 * (r * m).sum() / (c * m.sum()) - (c + 1) / c)


def occupancy(sim: np.ndarray, k: int) -> np.ndarray:
    """How often each gallery item (column) appears in the top-k of a query (row)."""
    topk = np.argpartition(-sim, kth=min(k, sim.shape[1] - 1), axis=1)[:, :k]
    return np.bincount(topk.ravel(), minlength=sim.shape[1])


def top_share(counts: np.ndarray, frac: float = 0.01) -> float:
    """Share of top-k slots taken by the `frac` most-retrieved gallery items."""
    m = np.sort(counts)[::-1]
    n = max(1, int(round(len(m) * frac)))
    return float(m[:n].sum() / m.sum())
