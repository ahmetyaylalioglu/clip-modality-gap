"""Embedding-extraction resource benchmark (peak GPU memory and throughput).

The cache timings in cache/<model>/*_meta.json were taken on a thermally
limited laptop (pauses at 85 C), so they are not clean measurements. This
script measures each model under the same, fixed conditions:

  - Flickr30k test: the first 1,000 images and their 5,000 captions
  - images are decoded and preprocessed once on the CPU, outside the timing
  - timed separately: GPU image encoding (batches of 256) and GPU text
    encoding (tokenization included), fp32, after one warm-up pass
  - 3 repeats; median and range reported; GPU temperature before/after
  - before each model the script waits until the GPU is at or below
    --cooldown degrees C, so that every model starts from the same state
  - thermal pauses are disabled during the benchmark (EMBED_MAX_TEMP unset)

It also writes the end-to-end cache wall-clock (including JPEG decoding and
preprocessing, minus recorded thermal pauses) where the cache metadata has it.

Output: results/resources/embedding_benchmark.csv
Usage:  python scripts/c8_embedding_benchmark.py [--model clip_b16 clip_b32 siglip_b16]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.pop("EMBED_MAX_TEMP", None)
from src import models  # noqa: E402
from src.data import iter_wds  # noqa: E402
from src.embed import CACHE, Encoder  # noqa: E402

OUT = ROOT / "results" / "resources"
N_IMG, REPEATS = 1000, 3


def gpu_temp():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


def load_pairs():
    images, captions = [], []
    for i, s in enumerate(iter_wds("flickr30k", "test")):
        if i == N_IMG:
            break
        images.append(s.image)
        captions.extend(s.captions)
    return images, captions


@torch.no_grad()
def time_images(enc, x):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(x), enc.bs):
        f = enc.model.encode_image(x[i:i + enc.bs].to(enc.device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


@torch.no_grad()
def time_texts(enc, captions):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(0, len(captions), enc.bs):
        f = enc.model.encode_text(enc.tokenizer(captions[i:i + enc.bs]).to(enc.device)).float()
        f = f / f.norm(dim=-1, keepdim=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def cache_wall_clock(m):
    rows = []
    for f in sorted((CACHE / m).glob("*_meta.json")):
        r = json.loads(f.read_text()).get("resources")
        if not r:
            continue
        pause = r.get("thermal_pause_s")
        rows.append({"wall_s": r["wall_clock_s"], "pause_s": pause, "n_img": r["n_images"], "n_txt": r["n_texts"]})
    if not rows:
        return {}
    d = pd.DataFrame(rows)
    out = {"cache_total_wall_min": d.wall_s.sum() / 60, "cache_n_images": int(d.n_img.sum()),
           "cache_n_texts": int(d.n_txt.sum())}
    if d.pause_s.notna().all():
        out["cache_thermal_pause_min"] = d.pause_s.sum() / 60
        out["cache_wall_minus_pause_min"] = (d.wall_s - d.pause_s).sum() / 60
    else:
        out["cache_note"] = "thermal pauses not recorded for this run (wall-clock includes throttling)"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", nargs="*", default=list(models.MODELS))
    p.add_argument("--cooldown", type=int, default=60, help="start each model at or below this GPU temperature (C)")
    args = p.parse_args()
    images, captions = load_pairs()
    rows = []
    for m in args.model:
        enc = Encoder(m)
        x = torch.stack([enc.preprocess(im) for im in images])
        time_images(enc, x[:enc.bs])  # warm-up
        time_texts(enc, captions[:enc.bs])
        torch.cuda.reset_peak_memory_stats()
        while (t := gpu_temp()) is not None and t > args.cooldown:
            time.sleep(10)
        t_before = gpu_temp()
        ti = [time_images(enc, x) for _ in range(REPEATS)]
        tt = [time_texts(enc, captions) for _ in range(REPEATS)]
        rows.append({
            "model": m, "arch_tag": "/".join(models.MODELS[m]), "gpu": torch.cuda.get_device_name(0),
            "precision": "fp32", "batch_size": enc.bs, "n_images": len(x), "n_texts": len(captions),
            "image_encode_s_median": float(np.median(ti)), "image_encode_s_min": min(ti), "image_encode_s_max": max(ti),
            "images_per_s": len(x) / float(np.median(ti)),
            "text_encode_s_median": float(np.median(tt)), "text_encode_s_min": min(tt), "text_encode_s_max": max(tt),
            "texts_per_s": len(captions) / float(np.median(tt)),
            "peak_gpu_mem_mb": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_temp_before_c": t_before, "gpu_temp_after_c": gpu_temp(),
            **cache_wall_clock(m),
        })
        print(m, f"{rows[-1]['images_per_s']:.0f} img/s, {rows[-1]['texts_per_s']:.0f} txt/s, "
                 f"peak {rows[-1]['peak_gpu_mem_mb']:.0f} MB", flush=True)
        del enc, x
        torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "embedding_benchmark.csv", index=False)
    print("wrote results/resources/embedding_benchmark.csv")


if __name__ == "__main__":
    main()
