"""Run the offline HexaBee preparation, evaluation, and result-check stages.

The generated trainer YAMLs are plans. GPU training and hosted generation are
explicit separate jobs; this program never starts a paid API call or trainer.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def run(*args: object) -> None:
    command = [sys.executable, *(str(arg) for arg in args)]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def require_file(value: str, label: str) -> Path:
    source = path(value)
    if not source.is_file():
        raise ValueError(f"{label} does not exist: {source}")
    return source


def link_input(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        if destination.resolve() == source:
            return
        raise ValueError(f"Existing input link points elsewhere: {destination}")
    if destination.exists():
        raise ValueError(f"Input destination already exists: {destination}")
    destination.symlink_to(source)


def prepare(settings: dict, run_dir: Path) -> None:
    data, configs = run_dir / "data", run_dir / "configs"
    data.mkdir(parents=True, exist_ok=True)
    configs.mkdir(parents=True, exist_ok=True)
    base_model = settings["base_model"]
    pretrain = settings.get("pretrain_text", [])
    if not pretrain:
        raise ValueError("pretrain_text must list at least one raw-text file")
    pt_names = []
    for i, name in enumerate(pretrain):
        source = require_file(name, f"pretrain_text[{i}]")
        target = data / f"pretrain_{i:02d}{source.suffix}"
        link_input(source, target)
        pt_names.extend(["--dataset-file", target.name])
    run(
        "scripts/render_training.py",
        "--stage",
        "pretrain",
        "--model",
        base_model,
        "--dataset-dir",
        data,
        *pt_names,
        "--training-output",
        run_dir / "adapters/pretrain",
        "--config-output",
        configs / "train_pretrain.yaml",
    )

    phase2_name = "phase2_train.jsonl"
    if settings.get("phase2_sft"):
        source = require_file(settings["phase2_sft"], "phase2_sft")
        if source.suffix not in {".json", ".jsonl"}:
            raise ValueError("phase2_sft must be one JSON or JSONL file")
        phase2_name = "phase2_train" + source.suffix
        link_input(source, data / phase2_name)
    else:
        wiki = require_file(settings["wikipedia_qa"], "wikipedia_qa")
        pools = settings.get("prompt_pools")
        if pools:
            pool_file = require_file(pools, "prompt_pools")
        else:
            pool_file = data / "seed_prompts.jsonl"
            run("-m", "pct4b", "prompts", "seeds", "--output", pool_file)
        run("-m", "pct4b", "normalize-wiki", "--input", wiki, "--output", data / "wiki.jsonl")
        run(
            "-m",
            "pct4b",
            "qa-build",
            "--input",
            data / "wiki.jsonl",
            "--recipe",
            path(settings.get("qa_recipe", "configs/qa_multilingual.json")),
            "--pools",
            pool_file,
            "--output",
            data / "phase2.jsonl",
        )
        run(
            "-m",
            "pct4b",
            "export-sft",
            "--input",
            data / "phase2.jsonl",
            "--output",
            data / "phase2_train.jsonl",
        )
    run(
        "scripts/render_training.py",
        "--stage",
        "multilingual",
        "--model",
        base_model,
        "--adapter",
        run_dir / "adapters/pretrain",
        "--dataset-dir",
        data,
        "--dataset-file",
        phase2_name,
        "--training-output",
        run_dir / "adapters/multilingual",
        "--config-output",
        configs / "train_multilingual.yaml",
    )
    run(
        "scripts/render_merge.py",
        "--model",
        base_model,
        "--adapter",
        run_dir / "adapters/multilingual",
        "--merged-output",
        run_dir / "models/multilingual",
        "--config-output",
        configs / "merge_multilingual.yaml",
    )

    translation_name = "translation_train.jsonl"
    if settings.get("translation_sft"):
        source = require_file(settings["translation_sft"], "translation_sft")
        if source.suffix not in {".json", ".jsonl"}:
            raise ValueError("translation_sft must be one JSON or JSONL file")
        translation_name = "translation_train" + source.suffix
        link_input(
            source,
            data / translation_name,
        )
    else:
        pairs = require_file(settings["aligned_pairs"], "aligned_pairs")
        split = data / "split"
        run(
            "-m",
            "pct4b",
            "split",
            "--input",
            pairs,
            "--output",
            split,
            "--test-fraction",
            settings.get("test_fraction", 0.1),
            "--seed",
            settings.get("seed", 42),
        )
        run(
            "-m",
            "pct4b",
            "mt-build",
            "--input",
            split / "train.jsonl",
            "--output",
            data / "translation.jsonl",
        )
        run(
            "-m",
            "pct4b",
            "export-sft",
            "--input",
            data / "translation.jsonl",
            "--output",
            data / "translation_train.jsonl",
        )
    run(
        "scripts/render_training.py",
        "--stage",
        "translation",
        "--model",
        run_dir / "models/multilingual",
        "--dataset-dir",
        data,
        "--dataset-file",
        translation_name,
        "--training-output",
        run_dir / "adapters/translation",
        "--config-output",
        configs / "train_translation.yaml",
    )
    run(
        "scripts/render_merge.py",
        "--model",
        run_dir / "models/multilingual",
        "--adapter",
        run_dir / "adapters/translation",
        "--merged-output",
        run_dir / "models/final",
        "--config-output",
        configs / "merge_final.yaml",
    )
    print(f"Prepared data and five trainer/export configs in {run_dir}")


def evaluate(settings: dict, run_dir: Path) -> None:
    predictions = settings.get("predictions", [])
    if not predictions:
        raise ValueError("predictions must list aligned candidate files before evaluation")
    files = [require_file(value, "prediction") for value in predictions]
    output = run_dir / "evaluation"
    output.mkdir(parents=True, exist_ok=True)
    run("-m", "pct4b", "metrics", "--input", *files, "--output", output / "metrics.json")
    focus = settings["focus_model"]
    baselines = settings["baseline_models"]
    judge_files = [
        require_file(value, "judge_prediction") for value in settings.get("judge_predictions", [])
    ]
    if judge_files:
        run(
            "-m",
            "pct4b",
            "metrics",
            "--input",
            *judge_files,
            "--metrics",
            "chrfpp",
            "ter",
            "--output",
            output / "judge_metrics.json",
        )
        run(
            "-m",
            "pct4b",
            "report",
            "--input",
            output / "judge_metrics.json",
            "--output",
            output / "judge_report",
            "--focus",
            focus,
        )
    if baselines:
        run(
            "-m",
            "pct4b",
            "bootstrap",
            "--input",
            *(judge_files or files),
            "--system",
            focus,
            "--baselines",
            *baselines,
            "--replicates",
            settings.get("bootstrap_replicates", 10000),
            "--output",
            output / "bootstrap.json",
        )
    run(
        "-m",
        "pct4b",
        "report",
        "--input",
        output / "metrics.json",
        "--output",
        output / "report",
        "--focus",
        focus,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/workflow.demo.json")
    parser.add_argument("--stage", choices=["prepare", "evaluate", "verify", "all"], default="all")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    run_dir = path(settings["run_dir"])
    if args.stage in {"prepare", "all"}:
        prepare(settings, run_dir)
    if args.stage in {"evaluate", "all"}:
        evaluate(settings, run_dir)
    if args.stage in {"verify", "all"}:
        run("scripts/reproduce_paper.py", "--output", run_dir / "paper_verification")


if __name__ == "__main__":
    main()
