"""Dataset loaders.

Retrieval data come from the clip-benchmark WebDataset shards
(see scripts/download_data.py). Each sample holds one image and a text
field with all of its captions separated by newlines.
"""
from __future__ import annotations

import io
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


@dataclass
class PairSample:
    key: str                      # sample id inside the shard (image id)
    image: Image.Image
    captions: list[str] = field(default_factory=list)


def wds_shards(dataset: str, split: str) -> list[Path]:
    """Sorted shard paths, e.g. wds_shards('flickr30k', 'test')."""
    d = DATA / "wds" / dataset / split
    shards = sorted(d.glob("*.tar"), key=lambda p: int(p.stem))
    if not shards:
        raise FileNotFoundError(f"no shards in {d}; run scripts/download_data.py")
    return shards


def iter_wds(dataset: str, split: str, limit: int | None = None, shards: list[int] | None = None):
    """Yield PairSample in deterministic shard order.

    Reads tar members directly (no webdataset dependency) so the order is
    fully reproducible on every platform.
    """
    paths = wds_shards(dataset, split)
    if shards is not None:
        paths = [p for p in paths if int(p.stem) in shards]
    n = 0
    for path in paths:
        with tarfile.open(path) as tar:
            current: dict[str, bytes] = {}
            current_key = None
            for m in tar:
                if not m.isfile():
                    continue
                key, _, ext = m.name.partition(".")
                if current_key is not None and key != current_key:
                    yield _to_sample(current_key, current)
                    n += 1
                    if limit is not None and n >= limit:
                        return
                    current = {}
                current_key = key
                current[ext.lower()] = tar.extractfile(m).read()
            if current_key is not None and current:
                yield _to_sample(current_key, current)
                n += 1
                if limit is not None and n >= limit:
                    return


def _to_sample(key: str, parts: dict[str, bytes]) -> PairSample:
    img_bytes = next((parts[e] for e in IMAGE_EXTS if e in parts), None)
    if img_bytes is None:
        raise ValueError(f"sample {key} has no image ({list(parts)})")
    txt = parts.get("txt", b"").decode("utf-8")
    captions = [c.strip() for c in txt.split("\n") if c.strip()]
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return PairSample(key=key, image=image, captions=captions)
