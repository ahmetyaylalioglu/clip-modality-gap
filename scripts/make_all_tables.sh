#!/usr/bin/env bash
# Regenerate every table and figure in results/ from cached embeddings.
#
#   bash scripts/make_all_tables.sh                 # all models with a complete cache (CPU only)
#   bash scripts/make_all_tables.sh clip_b16        # one model
#   EMBED=1 bash scripts/make_all_tables.sh         # also build missing embedding caches (GPU)
#   ORIGINAL=1 bash scripts/make_all_tables.sh      # also re-run the original Table 1 code (env-orig, GPU)
#
# Python interpreters can be overridden with PY / PY_ORIG.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ -z "${PY:-}" ]]; then
  if [[ -x env-main/Scripts/python.exe ]]; then PY=env-main/Scripts/python.exe; else PY=env-main/bin/python; fi
fi
if [[ -z "${PY_ORIG:-}" ]]; then
  if [[ -x env-orig/Scripts/python.exe ]]; then PY_ORIG=env-orig/Scripts/python.exe; else PY_ORIG=env-orig/bin/python; fi
fi
MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=(clip_b16 clip_b32 siglip_b16)

# every cache file the analysis scripts read
NEEDED=(flickr30k_test flickr30k_train mscoco_test mscoco_train cifar10_test cifar10_train10k
        cifar100_test cifar100_train10k svhn_test svhn_train10k eurosat_all coco_val2017)

if [[ "${ORIGINAL:-0}" == "1" ]]; then
  echo "=== original Table 1 code (env-orig)"
  "$PY_ORIG" scripts/run_original_table1.py
fi

for m in "${MODELS[@]}"; do
  if [[ "${EMBED:-0}" == "1" ]]; then
    echo "=== embed $m (GPU)"
    "$PY" -m src.embed --model "$m"
  fi
  missing=()
  for n in "${NEEDED[@]}"; do [[ -f "cache/$m/${n}_meta.json" ]] || missing+=("$n"); done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "=== skip $m: cache incomplete (missing: ${missing[*]}); run with EMBED=1"
    continue
  fi
  echo "=== $m"
  "$PY" scripts/b2_shift_zero_shot.py  --model "$m"   # B2 / B2b (RQ1)
  "$PY" scripts/b3_gap_plot.py         --model "$m"   # B3 gap figure
  "$PY" scripts/c1_retrieval.py        --model "$m"   # retrieval, hubness, val curves (~10 min)
  "$PY" scripts/c2_zero_shot.py        --model "$m"   # zero-shot cost (RQ3, H5)
  "$PY" scripts/c3_hypothesis_tests.py --model "$m"   # paired tests H3/H6, H4 Spearman (~10 min)
  "$PY" scripts/c4_failure_cases.py    --model "$m"
  "$PY" scripts/c5_figures.py          --model "$m"
  "$PY" scripts/c6_latency.py          --model "$m"   # uses the GPU briefly if available
done

echo "=== literature comparison (all models)"
"$PY" scripts/c7_literature_comparison.py

echo "=== summary and resolved configuration"
"$PY" scripts/dump_config.py
"$PY" scripts/make_summary.py
echo "done"
