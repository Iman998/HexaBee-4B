#!/usr/bin/env bash
#SBATCH --job-name=eval+report
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
set -euo pipefail

project=/mnt/ai_team/iman_barati/iman_31/paper/translate
repo="$project/pct4b-reproducibility"
python=/mnt/ai_team/iman_barati/iman_31/train/llama-factory-env/bin/python
root="$project/ablation_eval/balanced6000"
for variant in phase12 phase3 full; do
  test -s "$root/$variant/predictions.jsonl"
  test -s "$root/$variant/metrics.json"
done
export PYTHONPATH="$repo/src"
printf 'JOB=%s ROOT=%s\n' "$SLURM_JOB_ID" "$root"
date -Is
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python" -u "$project/ablation_eval/scripts/ablation_finalize_report.py" --root "$root"
date -Is
