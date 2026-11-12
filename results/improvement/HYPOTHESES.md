# Pre-registered hypotheses (Phase C)

Written and approved on 2026-11-12, before any Phase C experiment was run.
Not to be changed after results are seen.

Notation: M0 = raw L2-normalized embeddings. M1 = modality-wise mean-centering
followed by re-normalization (I0T_post form, An et al. 2025; GR-CLIP, Li et
al. 2025), with means estimated on a train/calibration split. M1-T = the same
with means from the test set itself (diagnostic only).

- **H1 (T2I):** With train-split calibration, M1 increases text→image
  Recall@1 over M0 on Flickr30k.
- **H2 (I2T):** With train-split calibration, M1 does not increase
  image→text Recall@1 on COCO 5K and may decrease it (Liu et al. 2026).
- **H3 (calibration):** Gains with test statistics (M1-T) are larger than
  with train-split statistics (M1), and in-domain calibration is better
  than out-of-domain.
- **H4 (hubness):** Where centering hurts retrieval, gallery-occupancy
  Gini increases.
- **H5 (classification):** The sign of (M1 − M0) zero-shot accuracy
  differs across CIFAR10/100, SVHN, EuroSAT: at least one dataset with a
  significant increase and one with a significant decrease (95% CI
  excluding 0). Otherwise H5 is rejected.

- **H6 (partial, direction-specific centering, M4):** M4 with α chosen on
  the validation split separately for I2T and T2I is not worse than M0 in
  either direction (95% CI of (M4 − M0) does not lie entirely below 0) and
  is better than M1 in at least one direction (95% CI of (M4 − M1) lies
  entirely above 0), on Flickr30k or COCO 5K.

M4 definition (fixed before running): x' = norm(x − α·μ_x) for each
modality, μ from the calibration split, α ∈ {0, 0.1, …, 1.0}. α = 0 is M0,
α = 1 is M1. α is selected by Recall@1 on a validation split (held-out
train pairs of the same dataset, disjoint from the calibration pairs),
once for I2T and once for T2I; test is evaluated once with the selected α.
M4 combines two existing ideas (mean-centering from I0T / GR-CLIP; a scaled
modification chosen on a grid, as in the Mind-the-Gap shift and the α of
Sato et al. 2026); it is not presented as a new method. Its own element in
this project is the leakage-free, per-direction selection of α.

Decision rule for H1/H2: a difference is called an increase (decrease) when
the 95% paired-bootstrap CI (1,000 resamples over test queries) of
(M1 − M0) lies entirely above (below) 0.
