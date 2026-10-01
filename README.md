# HexaBee-4B research code

Code and configuration for the six-language HexaBee translation experiment. The repository follows the actual work: prepare Wikipedia text and multilingual answering data, build aligned translation supervision, render LoRA training configurations, evaluate matched predictions, and verify the reported numeric results.

**Languages:** Persian, English, Arabic, Chinese, Hebrew, and Spanish. **Task:** translation in all 30 non-identity directions. The released [final model](https://huggingface.co/Iman998/HexaBee-4B), [multilingual SFT data](https://huggingface.co/datasets/Iman998/HexaBee-Multilingual-SFT), and [evaluation archive](https://huggingface.co/datasets/Iman998/HexaBee-Evaluation) are separate from this code repository.

## Run the offline example

The example uses tiny hand-written records and artificial predictions. It checks the complete **CPU preparation and evaluation path**, and writes the five trainer/export YAMLs without launching GPU training or contacting an API.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[metrics,plots,dev]'
.venv/bin/python scripts/run_workflow.py --config configs/workflow.demo.json --stage all
```

On the ZharfaTech Maxwell cluster, run computation as a Slurm job after setting up the environment:

```bash
mkdir -p logs
sbatch jobs/workflow.sbatch configs/workflow.demo.json all
```

The output is under `runs/demo/`: prepared SFT files, registered pretraining text, training and merge configurations, evaluation metrics, paired bootstrap intervals, and an independent check of released numeric results. The example scores are **not** the paper's scores.

## Run your experiment

Copy [configs/workflow.example.json](configs/workflow.example.json) into `runs/experiment.json`, edit its input paths, and submit:

```bash
sbatch jobs/workflow.sbatch runs/experiment.json prepare
```

Paths in the JSON are absolute or relative to the repository root. The workflow links large pretraining source files into its ignored run directory, avoiding a second copy. It uses one canonical output directory for every phase. A full run requires your Wikipedia question/answer export, raw text sources, aligned pairs, and prompt pools. If you already have the original phase-2 or final translation SFT exports, set `phase2_sft` or `translation_sft` in the configuration to use those directly.

| Phase | Entry point | Output |
|:--|:--|:--|
| 1. Wikipedia text | `render_training.py --stage pretrain` | Registered raw text and a **new-experiment** PT LoRA YAML |
| 2. Multilingual answering and answer translation | `normalize-wiki`, `qa-build`, `export-sft` | Phase-2 SFT and continued-adapter YAML |
| 3. Aligned translation | `split`, `mt-build`, `export-sft` | Leak-aware split, forward/reverse SFT, fresh-adapter YAML |
| 4. Model adaptation | `jobs/train.sbatch` | PT adapter → multilingual adapter → merged base → translation adapter → final model |
| 5. Predictions and judging | `translate --prediction`, `judge` | Matched candidate and judge records |
| 6. Evaluation | `run_workflow.py --stage evaluate` | BLEU, chrF, chrF++, TER, reports, paired judge intervals |
| 7. Result check | `run_workflow.py --stage verify` | Rebuilt tables and checks from committed **numeric** evidence |

The preparation command renders all training YAMLs. Launch their GPU stages in order from [the Slurm guide](docs/workflow.md); it does not start a trainer automatically. Synthetic text and teacher translations can be produced with the explicit, cached commands in [generation.md](docs/generation.md). The evaluation stage consumes aligned prediction files supplied in the workflow config. New raw predictions and a model judge require the model or service named in that guide.

## Repository map

```text
configs/         Research recipes, training templates, workflow examples
examples/        Tiny offline fixtures; not research data
jobs/            Slurm CPU and typed-GPU job entry points
scripts/         Workflow, trainer/merge rendering, numeric verification
src/pct4b/       Streaming data, prompts, generation, evaluation, Hub tools
results/paper/   Numeric evidence and provenance for published tables
tests/           Offline regression and integration checks
docs/            Input formats, stages, evaluation, release instructions
```

The Python namespace `pct4b` and `pct4b` command retain the historical experiment label. `hexabee` is an equivalent CLI. Source data, checkpoints, model outputs, API caches, generated figures, the manuscript, and branding binaries are not stored in this Git repository.

## Verification and scope

```bash
ruff check src scripts tests
ruff format --check src scripts tests
pytest -q
python scripts/reproduce_paper.py
```

The saved numeric artifacts verify aggregate scores and paired intervals. They cannot replace the original source, reference, and candidate strings for a fresh full metric pass. The historical phase-1 trainer YAML, exact teacher revisions, and original unseeded split membership were not archived. This repository supplies a runnable **new** PT recipe and preserves the recorded stage-2/3 settings; it does not promise bit-identical checkpoint reconstruction. See [reproducibility.md](docs/reproducibility.md).

Full instructions: [workflow and Slurm](docs/workflow.md) · [data formats](docs/data.md) · [training](docs/training.md) · [evaluation](docs/evaluation.md) · [Hugging Face release](docs/huggingface.md).

Software license: [MIT](LICENSE). Dataset and model terms are described in [DATA_AND_MODELS.md](DATA_AND_MODELS.md). For citation, use [CITATION.cff](CITATION.cff) until final publication details are available.
