# Modality Gap in Frozen Vision-Language Models

Research project for **YZM5110 – Multimodal Learning**.

> **Status:** Work in progress. Final submitted version will be pinned with the tag `v1.0-submission`.

## Research question

Does a training-free correction of the modality gap in frozen contrastive
vision-language models (CLIP, SigLIP) improve image–text retrieval, and
does it come at a cost to zero-shot classification?

## Reproduction target

Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, James Zou.
*Mind the Gap: Understanding the Modality Gap in Multi-modal Contrastive
Representation Learning.* NeurIPS 2022.
[arXiv:2203.02053](https://arxiv.org/abs/2203.02053)

## Planned scope

1. **Reproduction:** measure the modality gap in CLIP ViT-B/32 and reproduce
   the embedding-shift experiment on zero-shot classification.
2. **Improvement:** training-free gap correction (modality-wise centering
   and re-normalization), with statistics estimated on train/val data only.
3. **Evaluation:** image–text retrieval (Flickr30k, MS-COCO; Recall@1/5/10)
   and zero-shot classification, with ablations, a random-shift control,
   and resource measurements (latency, memory).

## Repository structure (planned)

| Path | Contents |
|------|----------|
| `configs/` | Experiment configurations |
| `scripts/` | Commands that produce reported tables and figures |
| `src/` | Implementation |
| `results/` | Saved outputs |
| `REPRODUCE.md` | Environment, versions, seeds, exact commands |
| `AI_USE.md` | AI-assistance disclosure |

## Setup

Setup instructions, pinned dependency versions, and dataset access
instructions will be added in `REPRODUCE.md`.

## Author

Ahmet Yaylalıoğlu
