#!/usr/bin/env bash
#SBATCH --job-name=eval+phase
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
set -euo pipefail

phase="${1:?Use phase1, phase12, or phase3}"
project=/mnt/ai_team/iman_barati/iman_31/paper/translate
owner=/mnt/ai_team/iman_barati
repo="$project/pct4b-reproducibility"
python="$owner/iman_31/train/llama-factory-env/bin/python"
test_dir="$project/huggingface-release/repositories/data-translation/data/aligned_pairs/test"
base="${HEXABEE_GEMMA_BASE:-/mnt/ai_team/models/gemma-3-4b-it}"

case "$phase" in
  phase1)
    model_path="$base"
    adapter_path="$project/ablation_eval/adapters/phase1"
    processor_path="$project/huggingface-release/repositories/lora-pt"
    infer_script="$project/ablation_eval/scripts/ablation_infer_adapter.py"
    model_name='HexaBee phase 1'
    ;;
  phase12)
    model_path="$project/huggingface-release/repositories/multilingual-base"
    adapter_path=''
    processor_path=''
    infer_script="$project/ablation_eval/scripts/ablation_infer.py"
    model_name='HexaBee phases 1+2'
    ;;
  phase3)
    model_path="$base"
    adapter_path="$project/ablation_eval/adapters/phase3"
    processor_path="$owner/iman_12/lora_models/gemma-3-4b-translate-v2"
    infer_script="$project/ablation_eval/scripts/ablation_infer_adapter.py"
    model_name='HexaBee phase 3 only (v2)'
    ;;
  *)
    printf 'Unknown phase: %s\n' "$phase" >&2
    exit 2
    ;;
esac

if ! test -f "$model_path/model.safetensors.index.json" && \
   ! test -f "$model_path/model.safetensors"; then
  printf 'Base model weights unavailable: %s\n' "$model_path" >&2
  exit 2
fi
test -d "$test_dir"
test -x "$python"
output_dir="$project/ablation_eval/$phase"
mkdir -p "$output_dir"
cd "$project"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$project/ablation_eval/peftdeps:$repo/src"
printf 'JOB=%s PHASE=%s MODEL=%s ADAPTER=%s TEST=%s OUTPUT=%s\n' \
  "$SLURM_JOB_ID" "$phase" "$model_path" "${adapter_path:-none}" "$test_dir" "$output_dir"
printf 'GIT_COMMIT=%s\n' "$(git -C "$repo" rev-parse HEAD 2>/dev/null || printf unavailable)"
date -Is
args=(
  --test-dir "$test_dir"
  --model-path "$model_path"
  --output-dir "$output_dir"
  --model-name "$model_name"
  --batch-size 4
  --max-new-tokens 5000
)
if test -n "$adapter_path"; then
  args+=(--adapter-path "$adapter_path")
fi
if test -n "$processor_path"; then
  args+=(--processor-path "$processor_path")
fi
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" \
  "$python" -u "$infer_script" "${args[@]}"
date -Is
