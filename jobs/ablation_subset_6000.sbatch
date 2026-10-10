#!/usr/bin/env bash
#SBATCH --job-name=eval+subset
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail

project=/mnt/ai_team/iman_barati/iman_31/paper/translate
repo="$project/pct4b-reproducibility"
python=/mnt/ai_team/iman_barati/iman_31/train/llama-factory-env/bin/python
export PYTHONPATH="$repo/src"
date -Is
printf 'JOB=%s GIT_COMMIT=%s\n' "$SLURM_JOB_ID" "$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf unavailable)"
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python" -u "$project/ablation_eval/scripts/build_ablation_subset.py" \
  --test-dir "$project/huggingface-release/repositories/data-translation/data/aligned_pairs/test" \
  --final-dir "$project/huggingface-release/repositories/outputs/data/hexabee_full/test" \
  --output-root "$project/ablation_eval/balanced6000" \
  --per-direction 200 --seed hexabee-ablation-20261010
date -Is
