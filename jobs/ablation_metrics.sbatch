#!/usr/bin/env bash
#SBATCH --job-name=metric+phase
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
set -euo pipefail

phase="${1:?Use phase1, phase12, or phase3}"
case "$phase" in phase1|phase12|phase3) ;; *) exit 2 ;; esac
project=/mnt/ai_team/iman_barati/iman_31/paper/translate
repo="$project/pct4b-reproducibility"
python=/mnt/ai_team/iman_barati/iman_31/train/llama-factory-env/bin/python
output_dir="$project/ablation_eval/$phase"
test -s "$output_dir/predictions.jsonl"
export PYTHONPATH="$repo/src:$project/ablation_eval/deps"
printf 'JOB=%s PHASE=%s PREDICTIONS=%s\n' \
  "$SLURM_JOB_ID" "$phase" "$output_dir/predictions.jsonl"
date -Is
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python" -c 'from pct4b.cli import main; main()' \
  metrics --input "$output_dir/predictions.jsonl" \
  --metrics bleu chrf chrfpp ter --output "$output_dir/metrics.json"
date -Is
