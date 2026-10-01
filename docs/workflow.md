# End-to-end workflow on Maxwell

Every compute command in this guide runs within a Slurm allocation. Prepare the Python environments in a short CPU allocation, then submit one job per stage. The login node is used only to edit configuration, create the log directory, and submit/inspect jobs.

## 1. Inspect the cluster and install

Check the current partition and typed GPU inventory with `sinfo -N -l` and `sinfo -o "%P %a %D %c %m %G %T %E"`. Use an allocation for environment setup:

```bash
srun --partition=general --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=8G --time=00:30:00 \
  bash -lc 'python3 -m venv .venv && .venv/bin/python -m pip install -i https://nexus.boof-ai.com/repository/pypi/simple -e ".[metrics,plots,dev]"'
mkdir -p logs
```

LLaMA-Factory, PyTorch, CUDA, and optional COMET belong in a separate compatible GPU environment. Set `HEXABEE_TRAIN_VENV` to that environment's path before submitting training. Save its package versions and LLaMA-Factory commit with the experiment.

## 2. Prepare data and YAMLs

Edit a copy of `configs/workflow.example.json`. Each path is absolute or relative to the repository root:

- `pretrain_text`: JSON/JSONL/Parquet text files with a `text` column. The workflow links them into `runs/<name>/data/` and registers them for LLaMA-Factory.
- `wikipedia_qa`: multilingual Wikipedia QA export. The normalizer accepts the original flat language columns.
- `prompt_pools`: normalized prompt JSONL. Set null only for the small illustrative seed pool.
- `aligned_pairs`: source language, target language, source text, and reference. The new split groups identical normalized sources.
- `phase2_sft` and `translation_sft` (optional): existing `system`/`instruct`/`output` exports. Each skips rebuilding that stage; this is useful for the archived published corpora.
- `predictions`: matched full-test model outputs for automatic metrics.
- `judge_predictions` (optional): a separate matched subset containing model-judge results for subset metrics and paired bootstrap. Set `baseline_models` to `[]` when no judge scores are available.
- `focus_model`, `baseline_models`: exact model labels in the prediction files. These are not required for `prepare`.

```bash
sbatch jobs/workflow.sbatch runs/experiment.json prepare
```

The job writes phase-2 and translation SFT files plus `train_pretrain.yaml`, `train_multilingual.yaml`, `merge_multilingual.yaml`, `train_translation.yaml`, and `merge_final.yaml` under the configured run directory. It never launches training or a paid API call.

To build a **new** synthetic aligned corpus first, run the cached topic → native source → teacher translation commands in [generation.md](generation.md) through `jobs/cli.sbatch`, then point `aligned_pairs` at the output. The `--execute` flag is required for actual API calls. Keep credentials in environment variables or an external secret file.

## 3. Train and merge

Submit in order and wait for each successful completion before the next stage. On the currently discovered H100 partition, a typed resource request looks like:

```bash
export HEXABEE_TRAIN_VENV=/path/to/llamafactory-venv
sbatch --gres=gpu:nvidia_h100:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  jobs/train.sbatch pretrain runs/experiment-01
sbatch --gres=gpu:nvidia_h100:6 --cpus-per-task=24 --mem=192G --time=24:00:00 \
  jobs/train.sbatch multilingual runs/experiment-01
sbatch --gres=gpu:nvidia_h100:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  jobs/train.sbatch merge-multilingual runs/experiment-01
sbatch --gres=gpu:nvidia_h100:2 --cpus-per-task=8 --mem=64G --time=24:00:00 \
  jobs/train.sbatch translation runs/experiment-01
sbatch --gres=gpu:nvidia_h100:1 --cpus-per-task=4 --mem=32G --time=04:00:00 \
  jobs/train.sbatch merge-final runs/experiment-01
```

These resource values are starting requests, not measured runtime guarantees. The historical stage-1 YAML is unavailable; `configs/train_pretrain.yaml` is a labeled new-experiment recipe. Stages 2 and 3 preserve the recorded configuration and adapter lineage. Inspect `squeue`, `scontrol show job <id>`, and `logs/<job>-<id>.out` after each submission. A submitted job ID does not mean a completed stage.

## 4. Predict, judge, and evaluate

Generate candidate translations from the final model and baselines on the same stored held-out pairs. The CLI can call a configured model service; use `jobs/cli.sbatch` for each API generation job:

```bash
sbatch jobs/cli.sbatch hexabee translate --input data/test_pairs.jsonl \
  --prediction --model-name HexaBee-4B --api-config configs/inference.json \
  --cache runs/inference.sqlite --output runs/hexabee_predictions.jsonl --execute
```

Repeat with each baseline and then set `predictions` to the three full-test output files. Run `hexabee judge` on a **matched** subset and set `judge_predictions` to those separately; see [evaluation.md](evaluation.md). Full metrics and the paired judge bootstrap are separate analyses:

```bash
sbatch jobs/workflow.sbatch runs/experiment.json evaluate
```

The evaluation job rejects mismatched or duplicate example identities. If files have no judge scores, skip the bootstrap by setting `baseline_models` to `[]` or run judging first. COMET is optional and needs its explicitly pinned checkpoint; the workflow's default metric command computes BLEU, chrF, chrF++, and TER.

## 5. Verify archived numbers

```bash
sbatch jobs/workflow.sbatch runs/experiment.json verify
```

This reads `results/paper/` and rebuilds numeric tables and interval checks under the run directory. It does not re-score unavailable original text outputs. The separate `scripts/build_paper_figures.py` regenerates the versioned figures from committed numeric data. For exact historical boundaries, read [reproducibility.md](reproducibility.md).
