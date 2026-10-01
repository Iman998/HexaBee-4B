#!/usr/bin/env bash
#SBATCH --job-name=hexabee-cli
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "${HEXABEE_VENV:-.venv}/bin/activate"
test "$#" -gt 0
printf 'job=%s commit=%s command=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$1"
srun "$@"
