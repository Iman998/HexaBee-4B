#!/usr/bin/env bash
#SBATCH --job-name=metric+parallel
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
set -euo pipefail

variant="${1:?Use full, phase12, or phase3}"
case "$variant" in full|phase12|phase3) ;; *) exit 2 ;; esac
project=/mnt/ai_team/iman_barati/iman_31/paper/translate
repo="$project/pct4b-reproducibility"
python=/mnt/ai_team/iman_barati/iman_31/train/llama-factory-env/bin/python
output_dir="$project/ablation_eval/balanced6000/$variant"
test -s "$output_dir/predictions.jsonl"
export PYTHONPATH="$repo/src:$project/ablation_eval/deps"
printf 'JOB=%s VARIANT=%s INPUT=%s OUTPUT=%s\n' \
  "$SLURM_JOB_ID" "$variant" "$output_dir/predictions.jsonl" "$output_dir/metrics.json"
date -Is
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python" -u "$project/ablation_eval/scripts/ablation_parallel_metrics.py" \
  --input "$output_dir/predictions.jsonl" --output "$output_dir/metrics.json" \
  --workers "$SLURM_CPUS_PER_TASK" --per-direction 200
date -Is
