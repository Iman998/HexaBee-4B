"""Regression checks for reproducibility, artifact integrity, and CLI behavior."""

import collections
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pct4b.api import Client
from pct4b.artifacts import verify_paper
from pct4b.data import build_qa, sample_judge
from pct4b.evaluation import score_metrics
from pct4b.historical import replay
from pct4b.io import read_records, write_records
from pct4b.prompts import PromptPool, generate_paraphrases, import_pools, translation_weights
from pct4b.schema import normalize_wiki

ROOT = Path(__file__).resolve().parents[1]


def test_normalization_is_idempotent_and_variants_keep_identity():
    item = normalize_wiki({"id": 1, "variants": {"fa": {"question": "پرسش", "answer": "پاسخ"}}})
    assert normalize_wiki(item) == item
    item["variants"]["en"] = {"answer": "Answer"}
    assert normalize_wiki(item)["id"] == item["id"]
    assert normalize_wiki(item)["original_id"] == 1


def test_import_three_language_translation_pool_ignores_checkpoints(tmp_path):
    name = "system_prompt_translate_spanish_english_persian_combined.json"
    write_records(tmp_path / name, [{"text": "Translate English to Persian."}] * 2)
    write_records(tmp_path / ".ipynb_checkpoints" / name, [{"text": "Stale copy"}])
    rows = list(import_pools(tmp_path))
    assert len(rows) == 2
    assert rows[0] == {
        "component": "translate",
        "language": "es",
        "source": "en",
        "target": "fa",
        "text": "Translate English to Persian.",
    }
    pool = PromptPool(rows)
    assert pool.sample("translate", "en", "fa", np.random.default_rng(42))[1] == "es"
    with pytest.raises(ValueError):
        pool.sample("translate", "fa", "en", np.random.default_rng(42))


def test_historical_four_language_prompt_inventory():
    weights = translation_weights("fa", "en", ["fa", "en", "zh", "ar"])
    assert weights == pytest.approx({"fa": 0.3, "en": 0.55, "zh": 0.03, "ar": 0.12})
    for path in (ROOT / "configs/historical").glob("*_translation.json"):
        recipe = json.loads(path.read_text())
        for rule in recipe["rules"]:
            group = rule["prompt_groups"][0]
            assert sum(p["weight"] for p in group) == pytest.approx(1, abs=2e-6)
            assert len(group) in (4, 5, 6)


def test_replay_null_filter_and_preserved_empty_strings(tmp_path):
    write_records(tmp_path / "pool_combined.json", [{"text": "Answer in English."}])
    rows = [
        {"id": i, "instruction": "Question", "English_output": value}
        for i, value in enumerate(["Answer", "", None])
    ]
    rule = {
        "id": "test",
        "source": "fa",
        "target": "en",
        "task": "complete",
        "known_instruction_conflict": False,
        "prompt_groups": [[{"file": "pool_combined.json", "weight": 1}]],
        "composition": {"seed": 42, "n_sentences": 1},
        "sample": {"applied": False},
        "column_policy": {"instruct": {"instruction": 1}, "output": {"English_output": 1}},
    }
    counts = collections.Counter()
    result = list(replay(rows, {"rules": [rule]}, tmp_path, counts))
    assert [r["output"] for r in result] == ["Answer", ""]
    assert counts["null_removed"] == 1


def test_ambiguous_pool_filename_rejected(tmp_path):
    for folder in ("a", "b"):
        write_records(tmp_path / folder / "pool_combined.json", [{"text": "a"}])
    with pytest.raises(ValueError, match="Ambiguous"):
        list(replay([], {"rules": []}, tmp_path))


def test_paraphrase_placeholder_validation_and_uniqueness(tmp_path):
    seed = {"component": "language", "language": "en", "text": "Answer in {target}."}
    client = Client(
        {"model": "fake", "attempts": 1},
        tmp_path / "cache.sqlite",
        True,
        lambda _: '["Use {target}.", "Reply in {target}."]',
    )
    assert len(generate_paraphrases(client, [seed], 2)) == 2
    client.close()
    bad = Client(
        {"model": "fake", "attempts": 1},
        tmp_path / "bad.sqlite",
        True,
        lambda _: '["Reply in English."]',
    )
    with pytest.raises(RuntimeError):
        generate_paraphrases(bad, [seed], 1)
    assert bad.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    bad.close()


@pytest.mark.parametrize(
    "config",
    [
        {"model": "a", "attempts": 0},
        {"model": "a", "max_tokens": -1},
        {"model": "a", "timeout": 0},
        {"model": ""},
    ],
)
def test_invalid_client_configuration_has_no_cache(tmp_path, config):
    path = tmp_path / "cache.sqlite"
    with pytest.raises(ValueError):
        Client(config, path, transport=lambda _: "x")
    assert not path.exists()


def test_duplicate_rules_fail_before_writing_data():
    recipe = {"rules": [{"source": "fa", "target": "en"}] * 2}
    with pytest.raises(ValueError, match="Duplicate rule"):
        list(build_qa([], PromptPool.load(), recipe))


def test_sampling_rejects_insufficient_direction():
    with pytest.raises(ValueError, match="fewer than"):
        sample_judge(read_records(ROOT / "examples/pairs.jsonl"), 2)


def test_comet_wrapper_scale_and_truncation_with_mock(tmp_path, monkeypatch):
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"mock checkpoint, not neural weights")

    class FakeComet:
        encoder = SimpleNamespace(
            max_positions=512, tokenizer=lambda items, **kwargs: {"length": [600] * len(items)}
        )

        def eval(self):
            return self

        def predict(self, items, **kwargs):
            return SimpleNamespace(scores=[0.875] * len(items))

    monkeypatch.setitem(
        sys.modules, "comet", SimpleNamespace(load_from_checkpoint=lambda _: FakeComet())
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _: None))
    result = score_metrics([ROOT / "examples/demo_b.jsonl"], ["comet"], checkpoint)
    assert result["macro"]["demo_b"]["comet"] == 87.5
    assert result["pooled"]["demo_b"]["comet"] == 87.5
    assert result["comet"]["truncated_text_counts"]["demo_b"] == dict.fromkeys(
        ["src", "mt", "ref"], 30
    )


def test_paper_statistics_counts_and_bootstrap():
    result = verify_paper(ROOT / "results/paper")
    assert result["recomputed_statistic_scores"] == 270
    assert result["phase2_records"] == 30153404
    assert result["macro"]["PCT-4B"]["chrfpp"] == pytest.approx(73.7485782067)
    assert result["recomputed_paired_differences"]["Gemma 3 12B IT"]["reference"]["ci95"] == [
        0.34,
        2.08,
    ]


def test_cli_offline_pipeline_and_dry_run(tmp_path):
    def run(*args):
        return subprocess.run(
            [sys.executable, "-m", "pct4b", *map(str, args)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )

    run(
        "normalize-wiki",
        "--input",
        ROOT / "examples/wiki.jsonl",
        "--output",
        tmp_path / "wiki.jsonl",
    )
    run(
        "qa-build",
        "--input",
        tmp_path / "wiki.jsonl",
        "--recipe",
        ROOT / "configs/qa_demo.json",
        "--output",
        tmp_path / "qa.jsonl",
    )
    run("export-sft", "--input", tmp_path / "qa.jsonl", "--output", tmp_path / "train.jsonl")
    assert len(list(read_records(tmp_path / "train.jsonl"))) == 3
    result = run(
        "translate",
        "--input",
        ROOT / "examples/pairs.jsonl",
        "--api-config",
        ROOT / "configs/teacher_deepseek.json",
        "--output",
        tmp_path / "no.jsonl",
    )
    assert json.loads(result.stdout)["status"] == "dry-run"
    assert not (tmp_path / "no.jsonl").exists()


def test_training_render_preserves_registry_and_resolves_paths(tmp_path):
    write_records(tmp_path / "train.jsonl", [{"system": "s", "instruct": "q", "output": "a"}])
    (tmp_path / "dataset_info.json").write_text('{"existing":{"file_name":"old.jsonl"}}')
    output = tmp_path / "config/train.yaml"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/render_training.py"),
            "--stage",
            "translation",
            "--model",
            "models/multilingual",
            "--dataset-dir",
            str(tmp_path),
            "--dataset-file",
            "train.jsonl",
            "--training-output",
            "adapters/test",
            "--config-output",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    assert "existing" in info and "pct_translation" in info
    assert "DEEPSPEED_CONFIG" not in output.read_text() and "MODEL_PATH" not in output.read_text()
    assert (output.parent / "ds_z0_config.json").exists()
