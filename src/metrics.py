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
