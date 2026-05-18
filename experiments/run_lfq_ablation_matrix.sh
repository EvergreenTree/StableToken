#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/data/venv/bin/python}"
STAMP="${STAMP:-20260518}"
EVAL_CONFIG="${EVAL_CONFIG:-experiments/configs/lfq_eval_matrix.yaml}"
POLL_SECONDS="${POLL_SECONDS:-30}"

cd "$ROOT"

mkdir -p experiments/matrix_runs
MATRIX_DIR="experiments/matrix_runs/lfq_ablation_matrix_${STAMP}"
mkdir -p "$MATRIX_DIR"
MASTER_LOG="$MATRIX_DIR/matrix_runner.log"

run_names=(
  "lfq_full_single_clean_${STAMP}"
  "lfq_full_single_aug_${STAMP}"
  "lfq_full_multi_clean_no_consensus_${STAMP}"
  "lfq_full_multi_aug_no_consensus_${STAMP}"
  "lfq_full_multi_aug_consensus_${STAMP}"
)
variants=(
  "single_clean"
  "single_aug"
  "multi_clean_no_consensus"
  "multi_aug_no_consensus"
  "multi_aug_consensus"
)
configs=(
  "experiments/configs/lfq_ablation_single_clean.yaml"
  "experiments/configs/lfq_ablation_single_aug.yaml"
  "experiments/configs/lfq_ablation_multi_clean_no_consensus.yaml"
  "experiments/configs/lfq_ablation_multi_aug_no_consensus.yaml"
  "experiments/configs/lfq_ablation_multi_aug_consensus.yaml"
)

log_msg() {
  printf '[%s] %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "$MASTER_LOG"
}

parse_loss() {
  local run_dir="$1"
  local title="$2"
  "$PYTHON" experiments/parse_lfq_loss_curve.py \
    --log "$run_dir/nohup_train.log" \
    --csv "$run_dir/loss_curve.csv" \
    --png "$run_dir/loss_curve.png" \
    --logging-steps 25 \
    --title "$title" >> "$run_dir/loss_monitor.log" 2>&1 || true
}

log_msg "starting LFQ ablation matrix stamp=${STAMP}"
printf '%s\n' "$$" > "$MATRIX_DIR/matrix_runner.pid"

for idx in "${!configs[@]}"; do
  config="${configs[$idx]}"
  run_name="${run_names[$idx]}"
  variant="${variants[$idx]}"
  run_dir="experiments/training_runs/${run_name}_${variant}"
  mkdir -p "$run_dir"

  if [[ -f "$run_dir/result.json" ]]; then
    log_msg "skip train ${variant}: existing ${run_dir}/result.json"
    parse_loss "$run_dir" "LFQ ${variant} training loss"
  else
    log_msg "train ${variant}: ${config}"
    : > "$run_dir/nohup_train.log"
    "$PYTHON" experiments/train_lfq_tokenizer.py \
      --config "$config" \
      --run-name "$run_name" \
      > "$run_dir/nohup_train.log" 2>&1 &
    train_pid=$!
    printf '%s\n' "$train_pid" > "$run_dir/train.pid"
    while kill -0 "$train_pid" 2>/dev/null; do
      parse_loss "$run_dir" "LFQ ${variant} training loss"
      sleep "$POLL_SECONDS"
    done
    wait "$train_pid"
    parse_loss "$run_dir" "LFQ ${variant} training loss"
    log_msg "finished train ${variant}"
  fi

  eval_dir="experiments/runs/lfq_matrix_${variant}_eval_${STAMP}"
  if [[ -f "$eval_dir/lfq_eval_summary.json" ]]; then
    log_msg "skip eval ${variant}: existing ${eval_dir}/lfq_eval_summary.json"
  else
    log_msg "eval ${variant}: ${run_dir}"
    "$PYTHON" experiments/eval_lfq_tokenizer.py \
      --config "$EVAL_CONFIG" \
      --checkpoint "$run_dir" \
      --output-dir "$eval_dir" \
      > "$eval_dir.tmp.log" 2>&1
    mv "$eval_dir.tmp.log" "$eval_dir/eval.log"
    log_msg "finished eval ${variant}"
  fi
done

log_msg "finished LFQ ablation matrix stamp=${STAMP}"
