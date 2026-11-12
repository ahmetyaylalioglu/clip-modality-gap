"""Run the ORIGINAL Mind-the-Gap Table 1 code (env-orig).

Executes cell 0 of
third_party/Modality-Gap/Table_1_Implications_CLIP_Zero_Shot/shifting/shift_features.ipynb
verbatim (evaluate_*, move_features, plot_metrics) together with the
original training/utils.py and training/datasets.py. The third-party files
are NOT modified. Only the following run-time adaptations are applied, each
documented in REPRODUCE.md §7:

  A1  `~/.cache` (torchvision root used by datasets.py) is redirected to
      data/torchvision via os.path.expanduser.
  A2  `vonmiseskde` (imported by utils.py, used only for density plots, not
      in the repository) is replaced by an empty stub module.
  A3  EuroSAT class names: datasets.py reads ~/.cache/eurosat/mapping.json,
      which is not in the repository. We write the EuroSAT class names from
      the CLIP repository's data/prompts.md (assumption).
  A4  ImageCaptionDataset base_dir: ANONYMOUS_ROOTDIR placeholder replaced by
      data/coco2017 (COCO val2017).
  A5  matplotlib Agg backend (no windows); DataLoader num_workers kept.

Gap direction per dataset:
  - EuroSAT: from the test features themselves (as in notebook cell 1)
  - SVHN:    from COCO val2017 image-caption pairs (as in notebook cell 2)
  - CIFAR10/100: no notebook cell exists -> both variants are run and
    labeled; the paper does not say which was used.

Usage (env-orig):
    env-orig/Scripts/python scripts/run_original_table1.py
Output: results/reproduction/table1_original_code.csv (full shift curves)
        results/reproduction/table1_original_code_summary.csv
"""
import csv
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MG = ROOT / "third_party" / "Modality-Gap" / "Table_1_Implications_CLIP_Zero_Shot"
TV_ROOT = ROOT / "data" / "torchvision"
COCO_ROOT = ROOT / "data" / "coco2017"
OUT = ROOT / "results" / "reproduction"

# EuroSAT class names, CLIP repository data/prompts.md (A3)
EUROSAT_CLASSES = [
    "annual crop land", "forest", "brushland or shrubland", "highway or road",
    "industrial buildings or commercial buildings", "pasture land",
    "permanent crop land", "residential buildings or homes or apartments",
    "river", "lake or sea",
]

PROMPTS = {  # from shift_features.ipynb cells 1-2 and utils.py default
    "CIFAR10": "a photo of a {}.",
    "CIFAR100": "a photo of a {}.",
    "EuroSAT": "a centered satellite photo of {}.",
    "SVHN": 'a street sign of the number: "{}".',
}
NOTEBOOK_DIRECTION = {"EuroSAT": "test", "SVHN": "coco_val2017"}

# Module level (not inside main) so that Windows DataLoader worker processes,
# which re-import this script, can also unpickle the original dataset classes.
sys.path.insert(0, str(MG / "training"))
_stub = types.ModuleType("vonmiseskde")  # A2
_stub.VonMisesKDE = None
sys.modules.setdefault("vonmiseskde", _stub)


def setup_environment():
    import matplotlib
    matplotlib.use("Agg")  # A5

    # A1: redirect ~/.cache
    real_expanduser = os.path.expanduser

    def expanduser(p):
        if isinstance(p, str) and p.startswith("~/.cache"):
            return str(TV_ROOT) + p[len("~/.cache"):]
        return real_expanduser(p)

    os.path.expanduser = expanduser

    # A3: EuroSAT mapping
    mp = TV_ROOT / "eurosat" / "mapping.json"
    if not mp.exists():
        mp.write_text(json.dumps(EUROSAT_CLASSES))


def load_notebook_cell0():
    nb = json.loads((MG / "shifting" / "shift_features.ipynb").read_text(encoding="utf-8"))
    src = "".join(nb["cells"][0]["source"])
    ns = {"__name__": "nb_cell0"}
    exec(compile(src, "shift_features.ipynb[cell0]", "exec"), ns)
    return ns


def main():
    setup_environment()
    import numpy as np
    import torch
    from functools import partial

    ns = load_notebook_cell0()
    load_clip, encode_clip, encode_clip_classification = ns["load_clip"], ns["encode_clip"], ns["encode_clip_classification"]
    ImageCaptionDataset, ClassificationDataset = ns["ImageCaptionDataset"], ns["ClassificationDataset"]
    evaluate_classification, move_features = ns["evaluate_classification"], ns["move_features"]

    torch.manual_seed(0)
    model = load_clip()

    # COCO val2017 direction (notebook cell 2), A4
    t0 = time.time()
    coco = ImageCaptionDataset(base_dir=str(COCO_ROOT))
    ci, ct = encode_clip(model, coco)
    coco_dir = ci.mean(axis=0) - ct.mean(axis=0)
    coco_gap = coco_dir.norm().item()
    coco_dir = coco_dir / coco_dir.norm()
    print(f"COCO val2017 pairs={len(coco)} gap={coco_gap:.4f} ({time.time()-t0:.0f}s)")

    rows, summary = [], []
    for name in ["CIFAR10", "CIFAR100", "EuroSAT", "SVHN"]:
        t0 = time.time()
        ds = ClassificationDataset(name=name)
        img, txt = encode_clip_classification(model, ds, prompt=PROMPTS[name])
        labels = [item[1] for item in ds]
        base = evaluate_classification(img, txt, labels)["Hit@1"]
        print(f"{name}: n={len(ds)} baseline Hit@1={base:.4f} ({time.time()-t0:.0f}s)")
        variants = {"test": None, "coco_val2017": coco_dir}
        if name in NOTEBOOK_DIRECTION:
            variants = {NOTEBOOK_DIRECTION[name]: variants[NOTEBOOK_DIRECTION[name]]}
        for vname, dvec in variants.items():
            allm = move_features(img, txt, partial(evaluate_classification, labels=labels), dvec)
            best_delta = max(allm, key=lambda d: allm[d][0]["Hit@1"])
            for delta in sorted(allm):
                m, gapd, _ = allm[delta]
                rows.append({"dataset": name, "direction_source": vname, "delta": float(delta),
                             "hit1": m["Hit@1"], "hit5": m["Hit@5"], "gap_distance": gapd})
            summary.append({
                "dataset": name, "direction_source": vname,
                "as_in_notebook": name in NOTEBOOK_DIRECTION,
                "n_test": len(ds), "baseline_hit1": base,
                "oracle_hit1_test_selected": allm[best_delta][0]["Hit@1"],
                "oracle_delta": float(best_delta), "gap_at_delta0": allm[0.0][1],
            })

    OUT.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for fname, data in [("table1_original_code.csv", rows), ("table1_original_code_summary.csv", summary)]:
        with open(OUT / fname, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)
    meta = {"git_commit": commit, "torch": torch.__version__, "coco_val2017_pairs": len(coco),
            "coco_val2017_gap": coco_gap, "env": "env-orig"}
    (OUT / "table1_original_code_meta.json").write_text(json.dumps(meta, indent=2))
    for s in summary:
        print(s)


if __name__ == "__main__":
    main()
