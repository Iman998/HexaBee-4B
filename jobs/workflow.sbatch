#!/usr/bin/env bash
#SBATCH --job-name=hexabee-workflow
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${HEXABEE_VENV:-.venv}/bin/activate"
printf 'job=%s commit=%s config=%s stage=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "${1:-configs/workflow.demo.json}" "${2:-all}"
srun python -u scripts/run_workflow.py --config "${1:-configs/workflow.demo.json}" --stage "${2:-all}"
