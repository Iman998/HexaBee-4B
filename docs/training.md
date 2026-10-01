# Training and checkpoint lineage

Training is performed by LLaMA-Factory in a separate GPU environment. This package writes configurations and data registration files. The explicit Slurm job in [workflow.md](workflow.md) launches a requested stage; data preparation never starts training on its own.

## Recorded sequence

| Stage | Initialization | Supervision | Recorded batch configuration |
|:--|:--|:--|:--|
| 1: Wikipedia adaptation | Gemma 3 4B IT; initial adapter | Wikipedia continued pretraining | 16 × 4 accumulation × 4 workers = 256 |
| 2: multilingual SFT | Base model + continued stage-1 adapter | All 22 selected datasets, 30,153,404 rows | 12 × 64 × 6 = 4,608 |
| Merge | Base model + stage-2 adapter | No training | Produces the stage-3 base |
| 3: translation SFT | Merged stage-2 model + fresh adapter | 658,274 forward/reverse examples | 12 × 8 × 2 = 192 |
| Final merge | Stage-3 base + translation adapter | No training | PCT-4B |

Recorded training uses learning rate $10^{-4}$, one epoch, cosine scheduling, 5% warmup, and AdamW (betas 0.9/0.999, epsilon $10^{-8}$). The stage-1 model card records seed 111; the subsequent runs use seed 42. Stage-2/3 cutoff length is 4,096 tokens. The original device model is not established by these files.

Stage 2 requests rank 256, dropout 0.001, `lora_target: all`, `freeze_vision_tower: false`, and `create_new_adapter: false`. A continued adapter also has its own saved PEFT configuration; requested YAML settings must not be mistaken for a newly initialized adapter configuration. The final translation adapter is independently verified as rank 256, alpha 512, dropout 0.005, without DoRA or RS-LoRA. Its saved module inventory is included as `configs/translation_adapter_config.json`; the training YAML uses the original `all` targeting with a frozen vision tower.

The historical validation fraction was 0.00001; the final stage had 658,267 training examples and only seven validation examples. The held-out 36,572-item translation benchmark is separate. Historical validation loss alone therefore provides little evidence of generalization.

## Prepare and register data

First use `pct4b export-sft` on a constructed dataset. A rendered configuration registers the `instruct`, `output`, and `system` columns explicitly; no user-specific absolute paths are embedded in the distributed templates. The PT renderer registers one or more files containing a `text` column, following the [LLaMA-Factory dataset format](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md#pre-training-dataset).

```bash
python scripts/render_training.py --stage pretrain \
  --model models/gemma-3-4b-it --dataset-dir data \
  --dataset-file wiki_fa.jsonl --dataset-file wiki_en.jsonl \
  --training-output adapters/wikipedia --config-output runs/train_pretrain.yaml

python scripts/render_training.py --stage multilingual \
  --model models/gemma-3-4b-it --adapter adapters/wikipedia \
  --dataset-dir data --dataset-file phase2_train.jsonl \
  --training-output adapters/multilingual --config-output runs/train_multilingual.yaml

python scripts/render_merge.py --model models/gemma-3-4b-it \
  --adapter adapters/multilingual --merged-output models/multilingual \
  --config-output runs/merge_multilingual.yaml

python scripts/render_training.py --stage translation \
  --model models/multilingual --dataset-dir data --dataset-file translation_train.jsonl \
  --training-output adapters/translation --config-output runs/train_translation.yaml

python scripts/render_merge.py --model models/multilingual \
  --adapter adapters/translation --merged-output models/pct-4b \
  --config-output runs/merge_final.yaml
```

Keep the two model/adapter merge pairs correct: the final adapter is merged into the already-merged multilingual model. Rendered files include an absolute path to a copied DeepSpeed ZeRO-0 configuration. The dataset registration updates preserve other existing dataset entries.

From a compatible LLaMA-Factory checkout/environment, launch each stage only after its required inputs/checkpoints exist. On Maxwell, submit [the typed-GPU Slurm jobs](workflow.md#3-train-and-merge). Their command sequence is:

```bash
FORCE_TORCHRUN=1 NPROC_PER_NODE=1 llamafactory-cli train runs/train_pretrain.yaml
FORCE_TORCHRUN=1 NPROC_PER_NODE=6 llamafactory-cli train runs/train_multilingual.yaml
llamafactory-cli export runs/merge_multilingual.yaml
FORCE_TORCHRUN=1 NPROC_PER_NODE=2 llamafactory-cli train runs/train_translation.yaml
llamafactory-cli export runs/merge_final.yaml
```

Those are GPU training/export commands, not part of the offline tests. Paths are interpreted relative to the launch directory, so keep the repository as the working directory or render absolute checkpoint/output paths.

## Environment and reconstruction limits

The stage-1 card records PEFT 0.15.2, Transformers 4.52.4, PyTorch 2.7.1+cu126, Datasets 3.6.0, and Tokenizers 0.21.1. The exact old LLaMA-Factory checkout and full stage-1 trainer YAML are not supplied in the public package; a current trainer may handle Gemma module targeting or templates differently. `configs/train_pretrain.yaml` is a **new-experiment starting point**, using the recorded seed and broad hyperparameters but a conservative per-device batch and explicit LoRA settings. It must not be presented as the historical stage-1 run. Preserve the original checkpoint/configuration to reproduce that stage exactly. The stage-2/3 templates represent the recorded configuration, with the saved adapter configuration taking precedence where a continued adapter is concerned.

For a new experiment, save the trainer commit, full resolved YAML, tokenizer revision, initial model revision, adapter configuration, exact split membership, and software environment with the run. Existing archived weights and data membership are needed for exact historical scoring; new hash-based construction is a reproducible reimplementation, not the historical random state.
