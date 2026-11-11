"""Small end-to-end test (< 5 min on one GPU).

Loads every model in src/models.py, embeds the first 32 Flickr30k test
image-caption sets, and prints I2T / T2I Recall@1 and the gap norm on this
toy subset. Needs only the Flickr30k test shards:

    python scripts/download_data.py --only wds   # or just test/0.tar
    python scripts/smoke_test.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import models  # noqa: E402
from src.data import iter_wds  # noqa: E402
from src.metrics import recall_at_k  # noqa: E402

N = 32


@torch.no_grad()
def embed(model, preprocess, tokenizer, samples, device):
    imgs = torch.stack([preprocess(s.image) for s in samples]).to(device)
    caps, owner = [], []
    for i, s in enumerate(samples):
        caps += s.captions
        owner += [i] * len(s.captions)
    ie = model.encode_image(imgs).float()
    te = model.encode_text(tokenizer(caps).to(device)).float()
    ie = ie / ie.norm(dim=-1, keepdim=True)
    te = te / te.norm(dim=-1, keepdim=True)
    return ie.cpu().numpy(), te.cpu().numpy(), np.array(owner)


def main():
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples = list(iter_wds("flickr30k", "test", limit=N))
    n_caps = sum(len(s.captions) for s in samples)
    print(f"device={device}  images={len(samples)}  captions={n_caps}")
    for key in models.MODELS:
        model, preprocess, tokenizer = models.load(key, device)
        img, txt, owner = embed(model, preprocess, tokenizer, samples, device)
        r = recall_at_k(img, txt, owner, ks=(1,))
        gap = np.linalg.norm(img.mean(0) - txt.mean(0))
        print(f"{key:12s} I2T R@1={r['i2t_R@1']:.3f}  T2I R@1={r['t2i_R@1']:.3f}  ||gap||={gap:.3f}")
        del model
        torch.cuda.empty_cache()
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
