#!/usr/bin/env bash
#SBATCH --job-name=hexabee-train
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
# GPU count, CPUs, memory, and time are explicit sbatch options at submission.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${HEXABEE_TRAIN_VENV:?Set HEXABEE_TRAIN_VENV to the LLaMA-Factory environment}/bin/activate"
stage="${1:?Stage required: pretrain, multilingual, merge-multilingual, translation, merge-final}"
run_dir="${2:-runs/experiment-01}"
case "$stage" in
  pretrain) yaml="$run_dir/configs/train_pretrain.yaml"; operation=train ;;
  multilingual) yaml="$run_dir/configs/train_multilingual.yaml"; operation=train ;;
  merge-multilingual) yaml="$run_dir/configs/merge_multilingual.yaml"; operation=export ;;
  translation) yaml="$run_dir/configs/train_translation.yaml"; operation=train ;;
  merge-final) yaml="$run_dir/configs/merge_final.yaml"; operation=export ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
test -f "$yaml"
test -n "${SLURM_GPUS_ON_NODE:-}"
export FORCE_TORCHRUN=1
export NPROC_PER_NODE="$SLURM_GPUS_ON_NODE"
printf 'job=%s commit=%s stage=%s yaml=%s gpus=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$stage" "$yaml" "$NPROC_PER_NODE"
srun llamafactory-cli "$operation" "$yaml"
