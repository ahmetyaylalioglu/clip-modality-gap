"""Embedding extraction + caching (env-main).

Writes L2-normalized float32 embeddings to cache/<model>/ :
  <name>_img.npy, <name>_txt.npy, <name>_meta.json
Retrieval sets  (name = flickr30k_test, flickr30k_train, mscoco_test, mscoco_train):
  meta: image keys, captions, txt_to_img (caption -> image index)
Classification  (name = cifar10_test, cifar10_train10k, ...):
  _img.npy = images, _txt.npy = class-prompt embeddings, meta: labels, classes, prompt
Every meta also records wall-clock seconds and peak GPU memory (resources).

Usage:
    python -m src.embed --model clip_b16             # everything
    python -m src.embed --model clip_b16 --only flickr30k_test cifar10_test
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from src import models
from src.data import DATA, iter_wds

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"

# EuroSAT class names from the CLIP repository (data/prompts.md); same
# assumption as scripts/run_original_table1.py (A3).
EUROSAT_CLASSES = [
    "annual crop land", "forest", "brushland or shrubland", "highway or road",
    "industrial buildings or commercial buildings", "pasture land",
    "permanent crop land", "residential buildings or homes or apartments",
    "river", "lake or sea",
]
# Prompts as in the original Mind-the-Gap code (shift_features.ipynb / utils.py)
PROMPTS = {
    "cifar10": "a photo of a {}.",
    "cifar100": "a photo of a {}.",
    "eurosat": "a centered satellite photo of {}.",
    "svhn": 'a street sign of the number: "{}".',
}
TRAIN_SUBSET = 10_000  # seeded subset of each classification train split
TRAIN_SUBSET_SEED = 0

RETRIEVAL_SETS = {
    "flickr30k_test": ("flickr30k", "test"),
    "flickr30k_train": ("flickr30k", "train"),
    "mscoco_test": ("mscoco", "test"),
    "mscoco_train": ("mscoco", "train"),
}
CLASSIF_SETS = [
    "cifar10_test", "cifar10_train10k", "cifar100_test", "cifar100_train10k",
    "svhn_test", "svhn_train10k", "eurosat_all",
]


class Encoder:
    def __init__(self, key: str, device: str = "cuda", batch_size: int = 256):
        self.key, self.device, self.bs = key, device, batch_size
        self.model, self.preprocess, self.tokenizer = models.load(key, device)
        self.pool = ThreadPoolExecutor(max_workers=8)

    @torch.no_grad()
    def images(self, pil_iter, total=None):
        out, batch, done = [], [], 0
        for im in pil_iter:
            batch.append(im)
            if len(batch) == self.bs:
                out.append(self._img_batch(batch))
                done += len(batch)
                batch = []
                if total:
                    print(f"\r  images {done}/{total}", end="", flush=True)
        if batch:
            out.append(self._img_batch(batch))
        print()
        return np.concatenate(out)

    def _img_batch(self, batch):
        x = torch.stack(list(self.pool.map(self.preprocess, batch))).to(self.device)
        with torch.autocast("cuda", enabled=False):
            f = self.model.encode_image(x).float()
        return (f / f.norm(dim=-1, keepdim=True)).cpu().numpy()

    @torch.no_grad()
    def texts(self, texts):
        out = []
        for i in range(0, len(texts), self.bs):
            t = self.tokenizer(texts[i:i + self.bs]).to(self.device)
            f = self.model.encode_text(t).float()
            out.append((f / f.norm(dim=-1, keepdim=True)).cpu().numpy())
        return np.concatenate(out)


def _save(dirp: Path, name: str, img, txt, meta):
    dirp.mkdir(parents=True, exist_ok=True)
    np.save(dirp / f"{name}_img.npy", img.astype(np.float32))
    np.save(dirp / f"{name}_txt.npy", txt.astype(np.float32))
    (dirp / f"{name}_meta.json").write_text(json.dumps(meta))


def embed_retrieval(enc: Encoder, name: str):
    dataset, split = RETRIEVAL_SETS[name]
    keys, captions, owner, images = [], [], [], []

    def gen():
        for i, s in enumerate(iter_wds(dataset, split)):
            keys.append(s.key)
            captions.extend(s.captions)
            owner.extend([i] * len(s.captions))
            yield s.image

    img = enc.images(gen())
    txt = enc.texts(captions)
    return img, txt, {"keys": keys, "captions": captions, "txt_to_img": owner}


def embed_coco_val2017(enc: Encoder, name: str):
    """COCO val2017 pairs built exactly like the original ImageCaptionDataset:
    one caption per image (the LAST annotation per image id wins, because the
    original builds a dict image_id -> caption). Used only for the original
    gap direction (SVHN cell) and the B2 sanity check — never for COCO-test
    evaluations (possible image overlap with the Karpathy test split)."""
    from PIL import Image

    base = DATA / "coco2017"
    data = json.loads((base / "annotations" / "captions_val2017.json").read_text(encoding="utf-8"))
    id2file = {it["id"]: it["coco_url"] for it in data["images"]}
    id2cap = {it["image_id"]: it["caption"] for it in data["annotations"]}
    pairs = [(id2file[i].replace("http://images.cocodataset.org/", ""), id2cap[i]) for i in id2cap]
    img = enc.images((Image.open(base / f).convert("RGB") for f, _ in pairs), total=len(pairs))
    caps = [c for _, c in pairs]
    txt = enc.texts(caps)
    return img, txt, {"files": [f for f, _ in pairs], "captions": caps, "txt_to_img": list(range(len(pairs)))}


def _classif_dataset(name: str):
    from torchvision import datasets as tvd

    root = DATA / "torchvision"
    base, part = name.split("_")
    if base == "cifar10":
        ds = tvd.CIFAR10(root, train=part != "test")
        classes = ds.classes
    elif base == "cifar100":
        ds = tvd.CIFAR100(root, train=part != "test")
        classes = ds.classes
    elif base == "svhn":
        ds = tvd.SVHN(root, split="test" if part == "test" else "train")
        classes = [str(i) for i in range(10)]  # original: ints 0..9
    elif base == "eurosat":
        ds = tvd.EuroSAT(root)
        classes = EUROSAT_CLASSES
    else:
        raise ValueError(name)
    idx = np.arange(len(ds))
    if part.startswith("train"):
        rng = np.random.default_rng(TRAIN_SUBSET_SEED)
        idx = np.sort(rng.choice(len(ds), size=min(TRAIN_SUBSET, len(ds)), replace=False))
    return ds, classes, idx, PROMPTS[base]


def embed_classification(enc: Encoder, name: str):
    ds, classes, idx, prompt = _classif_dataset(name)
    labels = []

    def gen():
        for i in idx:
            im, y = ds[int(i)]
            labels.append(int(y))
            yield im.convert("RGB")

    img = enc.images(gen(), total=len(idx))
    txt = enc.texts([prompt.format(c) for c in classes])
    return img, txt, {"labels": labels, "classes": list(map(str, classes)), "prompt": prompt,
                      "indices": idx.tolist()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(models.MODELS))
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    names = list(RETRIEVAL_SETS) + CLASSIF_SETS + ["coco_val2017"]
    if args.only:
        names = [n for n in names if n in args.only]
    dirp = CACHE / args.model
    enc = Encoder(args.model)
    for name in names:
        if (dirp / f"{name}_meta.json").exists() and not args.force:
            print(f"[skip] {args.model}/{name}")
            continue
        print(f"[embed] {args.model}/{name}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if name == "coco_val2017":
            fn = embed_coco_val2017
        elif name in RETRIEVAL_SETS:
            fn = embed_retrieval
        else:
            fn = embed_classification
        img, txt, meta = fn(enc, name)
        torch.cuda.synchronize()
        meta["resources"] = {
            "wall_clock_s": time.perf_counter() - t0,
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "n_images": int(img.shape[0]), "n_texts": int(txt.shape[0]),
            "batch_size": enc.bs, "gpu": torch.cuda.get_device_name(0),
        }
        meta["model"] = {"key": args.model, "arch_tag": models.MODELS[args.model]}
        _save(dirp, name, img, txt, meta)
        r = meta["resources"]
        print(f"  -> img {img.shape} txt {txt.shape} {r['wall_clock_s']:.0f}s peak {r['peak_gpu_mem_mb']:.0f} MB")


if __name__ == "__main__":
    main()
