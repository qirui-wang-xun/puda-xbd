#!/usr/bin/env bash
set -euo pipefail
# Usage: bash run_wildfire.sh DATA_DIR WEIGHTS_FILE OUTPUT_ROOT [SEED ...]
if [ "$#" -lt 3 ]; then
  echo 'Usage: bash run_wildfire.sh DATA_DIR IMAGENET_WEIGHTS OUTPUT_ROOT [922 923 924]'
  exit 2
fi
DATA_DIR=$(realpath "$1")
WEIGHTS_FILE=$(realpath "$2")
OUTPUT_ROOT=$(realpath -m "$3")
shift 3
if [ "$#" -eq 0 ]; then set -- 922 923 924; fi
cd "$(dirname "$(realpath "$0")")"
mkdir -p "$OUTPUT_ROOT"
for SEED in "$@"; do
  STAGE1="$OUTPUT_ROOT/wildfire_seed${SEED}_stage1"
  STAGE2="$OUTPUT_ROOT/wildfire_seed${SEED}_stage2"
  COMMON=(--p-root "$DATA_DIR/xBD/santa-rosa-wildfire-train"
          --u-root "$DATA_DIR/santa-rosa-wildfire_building_patches_64"
          --test-root "$DATA_DIR/xBD/santa-rosa-wildfire-test"
          --imagenet-weights "$WEIGHTS_FILE" --batch-size 16)
  python -u train_xbd.py --dataset santa-rosa-wildfire --seed "$SEED" \
    --source-root "$DATA_DIR/xBD" --prior 0.7886 --p-count 400 --p-batch 16 \
    --epochs 50 --stage1-only --output-dir "$STAGE1" "${COMMON[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/wildfire_seed${SEED}_stage1.console.log"
  python -u train_stage2_from_last.py --checkpoint "$STAGE1/stage1_last.pt" \
    --p-manifest "$STAGE1/p_samples.json" --epochs 50 --workers 6 \
    --output-dir "$STAGE2" "${COMMON[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/wildfire_seed${SEED}_stage2.console.log"
done
