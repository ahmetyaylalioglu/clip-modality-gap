# Reproducibility record

Draft. Updated as experiments are run. Every number in `results/` must be
traceable to an entry here (commit, environment, data revision, command).

## 1. Hardware

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 3080 Ti Laptop GPU, 16 GB |
| Driver / CUDA (driver) | 596.71 / 13.2 |
| OS | Windows 11 Pro 10.0.26300 |

## 2. Environments

### `env-main` (all our experiments)

- Python 3.10.4, `python -m venv env-main`
- torch 2.6.0+cu124 (CUDA runtime 12.4, cuDNN 9.1.0), torchvision 0.21.0+cu124
- open_clip_torch 4.2.2, transformers 5.20.2 (SigLIP tokenizer), numpy 2.2.6
- Full pin list: `requirements-lock.txt`

```bash
python -m venv env-main
env-main/Scripts/python -m pip --isolated install torch torchvision --index-url https://download.pytorch.org/whl/cu124
env-main/Scripts/python -m pip --isolated install -r requirements-lock.txt
```

`--isolated` makes pip ignore machine-wide pip configuration files. On the
development machine a global config added an extra package index whose TLS
certificate could not be verified; isolation avoids it. It has no effect on
which package versions are installed.

### `env-orig` (only for running the original Table 1 code)

**Attempt 1 — original `environment.yml` (failed).**
`conda env create -f third_party/Modality-Gap/environment.yml -p ./env-orig`
(Anaconda 2022.10, Windows) did not finish solving within 5 minutes
(log: stuck at "Collecting package metadata"). The file is a Linux export:
it pins Linux build strings (e.g. `cudatoolkit=10.1.243=h036e899_10`,
`python=3.8.12=hf930737_2_cpython`) and a development build
`torchvision==0.13.0a0+7bb5e41` that is not published on any index, so it
cannot be resolved on Windows as-is.

**Minimal fix — venv with the closest released versions.**
| Package | `environment.yml` | Used | Why |
|---|---|---|---|
| Python | 3.8.12 | 3.9 | 3.8 not installed on the machine; 3.9 is the closest available |
| torchvision | 0.13.0a0+7bb5e41 (dev) | 0.13.0+cu113 | closest release of the same version; ≥0.12 is needed for `EuroSAT` |
| torch | 1.10.0 | 1.12.0+cu113 | the release paired with torchvision 0.13.0 |
| CLIP | `clip` (OpenAI) | `git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1` | pinned commit |
| ansicolors, ftfy, regex, scikit-learn, seaborn, umap-learn | 1.1.8, 6.0.3, 2021.11.10, 1.0.2, 0.11.2, 0.5.2 | same | |

Additional pins needed to install: `setuptools<70` (installed 58.1.0) and
`--no-build-isolation`, because the OpenAI CLIP `setup.py` imports
`pkg_resources`, which recent setuptools no longer ships. The first run
then failed with `AttributeError: module 'matplotlib.cm' has no attribute
'register_cmap'` (seaborn 0.11.2 vs. a recent matplotlib), so the plotting /
numeric stack was pinned to the exact `environment.yml` versions:
matplotlib 3.5.1, scipy 1.7.3, numba 0.55.0, pynndescent 0.5.6,
numpy 1.21.5, pandas 1.4.0 (`contourpy` removed; not used by matplotlib 3.5).
Full pin list: `requirements-orig-lock.txt`.

```bash
py -3.9 -m venv env-orig
env-orig/Scripts/python -m pip --isolated install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
env-orig/Scripts/python -m pip --isolated install "setuptools<70" wheel
env-orig/Scripts/python -m pip --isolated install --no-build-isolation -r requirements-orig-lock.txt
```

Code edits needed to run the original notebook are recorded in §7.

## 3. Source code

| Item | Value |
|------|-------|
| This repository | commit recorded per result file |
| Official Mind-the-Gap code | `third_party/Modality-Gap` (git submodule), commit `8e20cb24efa4c5f89aad694f2f65eb43ffc46d10` (2022-09-25), MIT license |

## 4. Models

Loaded with `open_clip.create_model_and_transforms(arch, pretrained=tag)`.
Weights are fetched from the HuggingFace Hub by open_clip.

| Key | open_clip arch / tag | HF weight repo | Snapshot revision |
|---|---|---|---|
| `clip_b16` | `ViT-B-16` / `openai` | `timm/vit_base_patch16_clip_224.openai` | `a6f597a30f7b82c51704746581f9a4e41421e878` |
| `clip_b32` | `ViT-B-32` / `openai` | `timm/vit_base_patch32_clip_224.openai` | `41f575766f40e752fdd1383e9565b7f02388c1c4` |
| `siglip_b16` | `ViT-B-16-SigLIP` / `webli` | `timm/ViT-B-16-SigLIP` | `977e3dd0ec55ab8da155f2fbeb6b5f54948b6e3d` |

`clip_b16` is the model used in the paper's Table 1 (the original code calls
`clip.load("ViT-B/16")` from the OpenAI `clip` package; open_clip loads the
same OpenAI weights).

## 5. Data

All data are downloaded into `data/` (gitignored) by
`python scripts/download_data.py`.

| Dataset | Source | Revision | Parts used |
|---|---|---|---|
| Flickr30k | HF `clip-benchmark/wds_flickr30k` | `f8af3ceee1b944bf761340682621dfafd23974d2` | `test` (all 5 shards, 1k images × 5 captions); `train` shards 0–2 |
| MS-COCO captions | HF `clip-benchmark/wds_mscoco_captions` | `3bf9ad5d280b03126293bcc691585015e29ce691` | `test` (all 5 shards, Karpathy 5k); `train` shards 0–1 |
| MS-COCO 2017 val | cocodataset.org `val2017.zip`, `annotations_trainval2017.zip` | — | only for the original code's gap direction (SVHN cell) |
| CIFAR-10 / CIFAR-100 | torchvision | — | train + test |
| SVHN | torchvision | — | train + test |
| EuroSAT | torchvision (RGB) | — | single split, our seeded 80/20 split |

**Skipped:** HatefulMemes (used in the paper's Table 1) — access requires
accepting a restricted license; not used.

**Leakage note:** COCO val2017 images can overlap the Karpathy COCO test
split. COCO val2017 statistics are therefore never used for any evaluation
on COCO test.

**Data licenses / access.** MS-COCO annotations are CC BY 4.0, images are
subject to Flickr terms of use. Flickr30k is provided for non-commercial
research use under the Flickr30k terms. CIFAR, SVHN and EuroSAT are
downloaded through torchvision from their original hosts.

## 6. Commands

```bash
python scripts/download_data.py          # ~8 GB
python scripts/smoke_test.py             # < 1 min after weights are cached
```

Smoke test output (2026-11-11, `env-main`, 32 Flickr30k test images / 160 captions;
**CLIP rows made with the plain GELU config, see §7 #2 — to be re-run**):

```
clip_b16     I2T R@1=0.969  T2I R@1=0.856  ||gap||=0.873
clip_b32     I2T R@1=0.969  T2I R@1=0.919  ||gap||=0.885
siglip_b16   I2T R@1=0.938  T2I R@1=0.931  ||gap||=1.021
total 38.8s
```

## 7. Deviations from the original setup

| # | Deviation | Likely effect |
|---|---|---|
| 1 | `env-orig` uses torch 1.12.0 / torchvision 0.13.0 / Python 3.9 instead of the original Linux environment (§2) | Possible tiny numeric differences in preprocessing/inference; to be checked against the paper's baseline accuracy |
| 2 | Our re-implementation uses open_clip instead of the OpenAI `clip` package. The OpenAI weights must be loaded with the `-quickgelu` configs (`ViT-B-16-quickgelu`, `ViT-B-32-quickgelu`); the plain `ViT-B-16` config uses GELU. A first run with the plain config failed the B2 sanity check (δ=0 accuracy CIFAR10 .8769 vs .9013 from the original code); all caches made with it were deleted. | With QuickGELU, CIFAR10 2k-image check: .9065 (GELU: .879) |
| 3 | HatefulMemes not evaluated | Table 1 reproduced on 4 of 5 datasets |
| 4 | Original code run through `scripts/run_original_table1.py`, which executes notebook cell 0 verbatim with run-time adaptations A1–A5 (listed in the script docstring): `~/.cache` → `data/torchvision`; stub for the missing `vonmiseskde` module (plots only); EuroSAT class names from CLIP `data/prompts.md` because `mapping.json` is not in the repository; COCO base dir; Agg backend. No third-party file is edited. | A3 is an assumption: if the authors used other EuroSAT class names, EuroSAT accuracy can differ |
| 5 | CIFAR10/100 have no cell in the notebook; the gap direction used for them in Table 1 is unknown. Both the test-feature direction and the COCO val2017 direction are run and labeled. | Which variant matches the paper is reported, not assumed |
| 6 | EuroSAT: the original code evaluates on the whole dataset (27,000 images, no split). B1/B2 do the same; B2b uses a seeded 80/20 split. | B2b EuroSAT numbers are on a 20% test subset |

## 8. Phase B/C commands (clip_b16)

```bash
python -m src.embed --model clip_b16                  # all caches (GPU)
python scripts/b2_shift_zero_shot.py --model clip_b16 # B2 / B2b (CPU, ~5 min)
python scripts/c1_retrieval.py --model clip_b16       # Phase C retrieval (CPU, ~9 min)
python scripts/c2_zero_shot.py --model clip_b16       # Phase C zero-shot (CPU, ~2 min)
python scripts/c3_hypothesis_tests.py --model clip_b16  # paired tests for H3/H6, H4 Spearman
python scripts/b3_gap_plot.py --model clip_b16        # B3 PCA/UMAP figure + gap stats (CPU, ~30 s)
python scripts/c4_failure_cases.py --model clip_b16   # 20 most improved / worsened queries (CPU, ~10 s)
```

**Thermal throttling on the development laptop.** Under sustained load the
GPU reached 92–98 °C. `src/embed.py` therefore supports two environment
variables that only add pauses (they do not change any embedding):
`EMBED_MAX_TEMP` (pause after a batch while the GPU is at/above this
temperature until it is 3 °C lower; 85 was used for `flickr30k_train`) and
`EMBED_SLEEP` (fixed pause per batch). The pause setting is stored in each
`*_meta.json` (`resources.embed_sleep_s_per_batch`). Wall-clock times of
caches produced under throttling or concurrent jobs are **not** used for the
resource table; a separate clean benchmark is used instead.

B2 sanity check (QuickGELU configs): δ=0 and test-selected max are identical
to the original code for CIFAR10 (.9013/.9081 test direction, .9126 COCO
direction), CIFAR100 (.6658/.6737), EuroSAT (.5494/.5685) and SVHN (.5389).
