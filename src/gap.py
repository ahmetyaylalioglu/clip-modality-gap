"""Modality-gap measurement and training-free corrections.

All inputs are L2-normalized embeddings (rows). Statistics (means, gap
vectors) are passed in explicitly so that the caller decides which split
they come from; nothing here looks at test data by itself.

Shift (Liang et al. 2022, official code shift_features.ipynb, move_features):
    u     = (mean(img) - mean(txt)) / ||mean(img) - mean(txt)||
    txt'  = norm(txt + 0.5 * delta * u)
    img'  = norm(img - 0.5 * delta * u)
    delta in np.arange(-5, 5, 0.25)
The paper text writes x' = Normalize(x - lambda * Delta) with the
unnormalized Delta; the code form is used here. Mapping:
lambda * ||Delta|| = 0.5 * delta.

Centering (I0T_post, An et al. 2025; GR-CLIP, Li et al. 2025 — attributed):
    M1: x' = norm(x - mean_x_calib)     (with re-normalization)
    M2: x' = x - mean_x_calib           (ablation, no re-normalization; cosine scoring
                                          is then NOT used, raw dot product is.
                                          GR-CLIP scores with cosine, so it corresponds to M1)
"""
from __future__ import annotations

import numpy as np

SHIFT_GRID = np.arange(-5, 5, 0.25)  # identical to the original code


def l2n(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def gap_vector(img: np.ndarray, txt: np.ndarray) -> np.ndarray:
    """Delta = mean(img) - mean(txt) (not normalized)."""
    return img.mean(0) - txt.mean(0)


def gap_stats(img: np.ndarray, txt: np.ndarray, paired: bool = True, max_n: int = 5000, seed: int = 0) -> dict:
    """||Delta|| and cone-effect statistics (mean cosine within / across)."""
    rng = np.random.default_rng(seed)
    i = img if len(img) <= max_n else img[rng.choice(len(img), max_n, replace=False)]
    t = txt if len(txt) <= max_n else txt[rng.choice(len(txt), max_n, replace=False)]

    def mean_offdiag(a):
        s = a @ a.T
        n = len(a)
        return float((s.sum() - np.trace(s)) / (n * (n - 1)))

    out = {
        "gap_norm": float(np.linalg.norm(gap_vector(img, txt))),
        "cos_img_img": mean_offdiag(i),
        "cos_txt_txt": mean_offdiag(t),
        "cos_img_txt_all": float((i @ t.T).mean()),
    }
    if paired and len(img) == len(txt):
        out["cos_img_txt_paired"] = float((img * txt).sum(1).mean())
    return out


def shift(img: np.ndarray, txt: np.ndarray, delta: float, direction: np.ndarray):
    """Original Mind-the-Gap shift. `direction` is normalized inside."""
    u = direction / np.linalg.norm(direction)
    return l2n(img - 0.5 * delta * u), l2n(txt + 0.5 * delta * u)


def center(img: np.ndarray, txt: np.ndarray, mu_img: np.ndarray, mu_txt: np.ndarray, renorm: bool = True):
    """M1 (renorm=True) / M2 (renorm=False)."""
    a, b = img - mu_img, txt - mu_txt
    return (l2n(a), l2n(b)) if renorm else (a, b)


def partial_center(x: np.ndarray, mu: np.ndarray, alpha: float) -> np.ndarray:
    """M4 building block: norm(x - alpha * mu). alpha=0 -> M0, alpha=1 -> M1."""
    return l2n(x - alpha * mu)


ALPHA_GRID = np.round(np.arange(0, 1.0001, 0.1), 1)  # fixed in HYPOTHESES.md


def random_direction_like(delta_vec: np.ndarray, seed: int) -> np.ndarray:
    """Control C1: random direction with the same norm as Delta."""
    rng = np.random.default_rng(seed)
    r = rng.standard_normal(delta_vec.shape)
    return r / np.linalg.norm(r) * np.linalg.norm(delta_vec)


def random_shift(img, txt, delta_vec, seed):
    """C1: shift by a random vector with ||r|| = ||Delta||, applied as the
    lambda=1 paper-text shift: img - r/2, txt + r/2, then re-normalize
    (same magnitude as fully closing the gap along Delta)."""
    r = random_direction_like(delta_vec, seed)
    return l2n(img - r / 2), l2n(txt + r / 2)
