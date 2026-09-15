#!/usr/bin/env bash
# Visual-severity sweep for Auto-AVSR, full LRS3 test set, Permutation SHAP.
#
# 4 distortion types (GNC, CC, JPEG, GB) x 6 severity levels (0=clean, 1-5)
# = 24 runs, batched 8 at a time across GPUs 0-7 (3 clean waves), one
# eval_shap.py process per run. Requires the decode.vid_dist_type/
# decode.vid_dist_level fields just added to configs/decode/default.yaml.
#
# Mirrors the visual sweep scripts already used for usr2/Llama-AVSR/Omni-AVSR
# in this project, adapted to auto_avsr_shap's Hydra CLI (eval_shap.py).

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

GPUS=(0 1 2 3 4 5 6 7)
TYPES=(GNC CC JPEG GB)
LEVELS=(0 1 2 3 4 5)

ROOT_DIR=/ucappell/datasets
CKPT=/aa4825/models/autoavsr/vsr_trlrs2lrs3vox2avsp_base.pth
TEST_FILE=lrs3_test_transcript_lengths_seg24s_LLM_lowercase_micro_50.csv
OUT_DIR=output
WANDB_PROJECT=dr-shap-av-visual-autoavsr

mkdir -p "$OUT_DIR"
LOGDIR=logs/visual_sweep_autoavsr
mkdir -p "$LOGDIR"

COMMON_ARGS=(
  data.modality=audiovisual
  data.dataset.root_dir="$ROOT_DIR"
  data.dataset.test_file="$TEST_FILE"
  pretrained_model_path="$CKPT"
  decode.wandb_project="$WANDB_PROJECT"
  decode.shap_alg=permutation
  decode.num_samples_shap=2000
  decode.output_path="$OUT_DIR"
)

# Build the full (type, level) job list, then batch across GPUs (8 at a time).
JOBS=()
for type in "${TYPES[@]}"; do
  for level in "${LEVELS[@]}"; do
    JOBS+=("${type}:${level}")
  done
done

MAX_PARALLEL=${#GPUS[@]}
job_idx=0
total=${#JOBS[@]}

while [ $job_idx -lt $total ]; do
  echo ""
  echo "============================================================"
  echo "Launching batch starting at job $job_idx / $total"
  echo "============================================================"
  echo ""

  gpu_idx=0
  batch_end=$((job_idx + MAX_PARALLEL))
  if [ $batch_end -gt $total ]; then
    batch_end=$total
  fi

  for ((i=job_idx; i<batch_end; i++)); do
    job="${JOBS[$i]}"
    type="${job%%:*}"
    level="${job#*:}"

    gpu="${GPUS[$gpu_idx]}"
    gpu_idx=$((gpu_idx + 1))

    exp_name="autoavsr_shap_permutation_viddist-${type}-lvl${level}"

    echo "Launching $exp_name on GPU $gpu"
    echo "  decode.vid_dist_type=$type  decode.vid_dist_level=$level"
    echo "  Log: $LOGDIR/${exp_name}.log"

    CUDA_VISIBLE_DEVICES=$gpu python eval_shap.py \
      "${COMMON_ARGS[@]}" \
      decode.vid_dist_type="$type" \
      decode.vid_dist_level=$level \
      decode.exp_name="$exp_name" \
      > "$LOGDIR/${exp_name}.log" 2>&1 &

  done

  echo ""
  echo "Waiting for this batch to finish..."
  wait

  job_idx=$batch_end
done

echo "============================================================"
echo "ALL Auto-AVSR visual severity sweeps finished."
echo "Logs in $LOGDIR/"
echo "Shapley .npz files in $OUT_DIR/"
echo "============================================================"
