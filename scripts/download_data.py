"""Download all datasets used in this project into ./data (gitignored).

Usage:
    python scripts/download_data.py            # everything
    python scripts/download_data.py --only wds  # only HF WebDataset shards

Revisions of the HuggingFace datasets are pinned below; record any change
in REPRODUCE.md.
"""
import argparse
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Pinned HuggingFace dataset revisions (commit hashes, retrieved 2026-11-11).
WDS = {
    "flickr30k": {
        "repo": "clip-benchmark/wds_flickr30k",
        "revision": "f8af3ceee1b944bf761340682621dfafd23974d2",
        "files": [f"test/{i}.tar" for i in range(5)]
        + ["test/nshards.txt"]
        # 3 of 5 train shards (~21k images) -> calibration + validation pool
        + [f"train/{i}.tar" for i in range(3)],
    },
    "mscoco": {
        "repo": "clip-benchmark/wds_mscoco_captions",
        "revision": "3bf9ad5d280b03126293bcc691585015e29ce691",
        "files": [f"test/{i}.tar" for i in range(5)]
        + ["test/nshards.txt"]
        # 2 of 14 train shards -> calibration pool
        + [f"train/{i}.tar" for i in range(2)],
    },
}

# COCO val2017, used only by the original Mind-the-Gap code (gap direction
# for the SVHN cell). Never used for statistics in COCO-test evaluations.
COCO_VAL2017 = {
    "val2017.zip": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations_trainval2017.zip": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def download_wds():
    from huggingface_hub import hf_hub_download

    for name, spec in WDS.items():
        out = DATA / "wds" / name
        for f in spec["files"]:
            if (out / f).exists():
                print(f"[skip] {name}/{f}")
                continue
            print(f"[get ] {name}/{f}", flush=True)
            hf_hub_download(
                repo_id=spec["repo"],
                repo_type="dataset",
                filename=f,
                revision=spec["revision"],
                local_dir=out,
            )


def download_torchvision():
    from torchvision import datasets

    root = DATA / "torchvision"
    root.mkdir(parents=True, exist_ok=True)
    for train in (True, False):
        datasets.CIFAR10(root, train=train, download=True)
        datasets.CIFAR100(root, train=train, download=True)
    for split in ("train", "test"):
        datasets.SVHN(root, split=split, download=True)
    datasets.EuroSAT(root, download=True)


def download_coco_val2017():
    out = DATA / "coco2017"
    out.mkdir(parents=True, exist_ok=True)
    for fname, url in COCO_VAL2017.items():
        zpath = out / fname
        marker = out / (fname + ".extracted")
        if marker.exists():
            print(f"[skip] {fname}")
            continue
        if not zpath.exists():
            print(f"[get ] {url}", flush=True)
            urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        marker.touch()
        zpath.unlink()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--only", choices=["wds", "torchvision", "coco2017"])
    args = p.parse_args()
    steps = {
        "wds": download_wds,
        "torchvision": download_torchvision,
        "coco2017": download_coco_val2017,
    }
    for k, fn in steps.items():
        if args.only in (None, k):
            print(f"=== {k} ===", flush=True)
            fn()
    print("done")


if __name__ == "__main__":
    sys.exit(main())
