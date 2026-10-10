"""Resumable, matched Gemma 3 inference on the archived HexaBee translation test.

Run this only inside a Slurm GPU allocation. It writes one SQLite progress file and
exports the complete prediction JSONL atomically after every expected item is done.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import transformers
from safetensors import safe_open
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from pct4b.generation import translation_messages
from pct4b.schema import evaluation_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--processor-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--reuse-progress", type=Path,
        help="Stopped full-test SQLite progress from the identical checkpoint and decoding setup",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-rows", type=int, default=36572)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--limit", type=int, default=0, help="Smoke test rows; zero means full test")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.limit < 0:
        parser.error("batch-size and max-new-tokens must be positive; limit must be nonnegative")
    return args


def test_files(test_dir: Path) -> list[Path]:
    files = sorted(test_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No Parquet test files under {test_dir}")
    return files


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows_from_parquet(files: list[Path], limit: int):
    count = 0
    for path in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256):
            for row in batch.to_pylist():
                if row["id"] != evaluation_id(row):
                    raise ValueError(f"Test identity mismatch in {path}, row {count}")
                yield row
                count += 1
                if limit and count >= limit:
                    return


def open_progress(path: Path, run_config: dict) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS predictions ("
        "id TEXT PRIMARY KEY, candidate TEXT NOT NULL, output_tokens INTEGER NOT NULL, "
        "hit_limit INTEGER NOT NULL)"
    )
    saved = dict(connection.execute("SELECT key, value FROM meta"))
    encoded = json.dumps(run_config, ensure_ascii=False, sort_keys=True)
    if saved and saved.get("run_config") != encoded:
        raise ValueError("Existing progress belongs to a different test/model/inference configuration")
    if not saved:
        connection.execute("INSERT INTO meta VALUES (?, ?)", ("run_config", encoded))
        connection.commit()
    return connection


def reuse_predictions(
    connection: sqlite3.Connection,
    selected_ids: set[str],
    source_path: Path,
    run_config: dict,
) -> int:
    """Copy only selected completed predictions from a stopped, compatible run."""
    if source_path.resolve() == Path(connection.execute("PRAGMA database_list").fetchone()[2]).resolve():
        raise ValueError("Reuse source must differ from the current progress database")
    with sqlite3.connect(f"file:{source_path}?mode=ro", uri=True) as source:
        saved = dict(source.execute("SELECT key, value FROM meta"))
        source_config = json.loads(saved["run_config"])
        keys = (
            "model_path", "adapter_path", "processor_path", "model_name", "prompt",
            "attention_implementation", "max_new_tokens", "transformers_version", "torch_version",
        )
        mismatched = [key for key in keys if source_config.get(key) != run_config.get(key)]
        if mismatched:
            raise ValueError(f"Reuse source has incompatible configuration: {mismatched}")
        before = connection.total_changes
        with connection:
            connection.executemany(
                "INSERT OR IGNORE INTO predictions VALUES (?, ?, ?, ?)",
                (
                    (identifier, candidate, output_tokens, hit_limit)
                    for identifier, candidate, output_tokens, hit_limit in source.execute(
                        "SELECT id, candidate, output_tokens, hit_limit FROM predictions"
                    )
                    if identifier in selected_ids
                ),
            )
        return connection.total_changes - before


def chat_batch(rows: list[dict]) -> list[list[dict]]:
    conversations = []
    for row in rows:
        plain = translation_messages(
            row["source_language"], row["target_language"], row["source"], "archived-short"
        )
        conversations.append(
            [
                {"role": item["role"], "content": [{"type": "text", "text": item["content"]}]}
                for item in plain
            ]
        )
    return conversations


def verify_adapter(model, adapter_path: Path) -> None:
    """Reject an adapter when any saved tensor failed to load into its LoRA module."""
    state = model.state_dict()
    errors = []
    matched = 0
    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt", device="cpu") as archive:
        keys = list(archive.keys())
        for key in keys:
            expected = key.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight"
            )
            tensor = state.get(expected)
            if tensor is None:
                errors.append(f"missing {expected}")
                continue
            saved = archive.get_tensor(key)
            loaded = tensor.detach().cpu()
            if saved.shape != loaded.shape or not torch.equal(saved.to(loaded.dtype), loaded):
                errors.append(f"value mismatch {expected}")
                continue
            matched += 1
    if errors or matched != 476:
        raise RuntimeError(
            f"Adapter validation failed: matched={matched}/{len(keys)}; first errors={errors[:3]}"
        )
    print(f"ADAPTER_VERIFIED tensors={matched} path={adapter_path}", flush=True)


@torch.inference_mode()
def generate_batch(model, processor, rows: list[dict], max_new_tokens: int):
    inputs = processor.apply_chat_template(
        chat_batch(rows),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    prompt_width = inputs["input_ids"].shape[1]
    try:
        sequences = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    except torch.cuda.OutOfMemoryError:
        del inputs
        torch.cuda.empty_cache()
        if len(rows) == 1:
            raise
        middle = len(rows) // 2
        return generate_batch(model, processor, rows[:middle], max_new_tokens) + generate_batch(
            model, processor, rows[middle:], max_new_tokens
        )
    outputs = []
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids or [])
    for sequence in sequences:
        tokens = sequence[prompt_width:].tolist()
        first_eos = next((index for index, token in enumerate(tokens) if token in eos_ids), None)
        if first_eos is not None:
            tokens = tokens[: first_eos + 1]
        hit_limit = first_eos is None and len(tokens) >= max_new_tokens
        candidate = processor.tokenizer.decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        outputs.append((candidate, len(tokens), int(hit_limit)))
    return outputs


def export_complete(connection: sqlite3.Connection, rows: list[dict], args: argparse.Namespace):
    output = args.output_dir / "predictions.jsonl"
    temporary = args.output_dir / "predictions.jsonl.tmp"
    inference = {
        "prompt": "archived-short",
        "temperature": 0,
        "max_new_tokens": args.max_new_tokens,
        "engine": "transformers-gemma3-generate",
        "attention_implementation": "eager",
        "transformers_version": transformers.__version__,
    }
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            result = connection.execute(
                "SELECT candidate, output_tokens, hit_limit FROM predictions WHERE id=?", (row["id"],)
            ).fetchone()
            if result is None:
                raise RuntimeError(f"Missing prediction for {row['id']}")
            record = {
                "id": row["id"],
                "model": args.model_name,
                "source_language": row["source_language"],
                "target_language": row["target_language"],
                "source": row["source"],
                "reference": row["reference"],
                "candidate": result[0],
                "output_tokens": result[1],
                "hit_token_limit": bool(result[2]),
                "inference": inference,
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output)
    print(f"COMPLETE rows={len(rows)} output={output}", flush=True)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("A Slurm GPU allocation is required")
    if not args.model_path.is_dir():
        raise FileNotFoundError(args.model_path)
    if args.adapter_path and not (args.adapter_path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(args.adapter_path / "adapter_model.safetensors")
    files = test_files(args.test_dir)
    rows = list(rows_from_parquet(files, args.limit))
    if len(rows) != (args.limit or args.expected_rows):
        raise ValueError(f"Expected {args.limit or args.expected_rows} test rows, found {len(rows)}")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate test identities")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "test_files": {path.name: file_sha256(path) for path in files},
        "model_path": str(args.model_path.resolve()),
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "processor_path": str(args.processor_path.resolve()) if args.processor_path else None,
        "model_name": args.model_name,
        "prompt": "archived-short",
        "attention_implementation": "eager",
        "max_new_tokens": args.max_new_tokens,
        "limit": args.limit,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    connection = open_progress(args.output_dir / "progress.sqlite", run_config)
    selected_ids = {row["id"] for row in rows}
    if args.reuse_progress:
        imported = reuse_predictions(connection, selected_ids, args.reuse_progress, run_config)
        print(f"REUSED predictions={imported} source={args.reuse_progress}", flush=True)
    completed = {identifier for (identifier,) in connection.execute("SELECT id FROM predictions")}
    if completed - selected_ids:
        raise ValueError("Progress contains IDs outside the selected test")
    pending = [row for row in rows if row["id"] not in completed]
    print(
        f"START job={os.environ['SLURM_JOB_ID']} model={args.model_name} "
        f"rows={len(rows)} done={len(completed)} pending={len(pending)} "
        f"device={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}",
        flush=True,
    )
    if pending:
        processor = AutoProcessor.from_pretrained(
            args.processor_path or args.adapter_path or args.model_path,
            local_files_only=True,
            use_fast=False,
        )
        processor.tokenizer.padding_side = "left"
        model = Gemma3ForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True,
            attn_implementation="eager",
        )
        if args.adapter_path:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
            verify_adapter(model, args.adapter_path)
        model.eval()
        started = time.monotonic()
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            outputs = generate_batch(model, processor, batch, args.max_new_tokens)
            with connection:
                connection.executemany(
                    "INSERT INTO predictions VALUES (?, ?, ?, ?)",
                    [
                        (row["id"], candidate, output_tokens, hit_limit)
                        for row, (candidate, output_tokens, hit_limit) in zip(batch, outputs)
                    ],
                )
            if (offset // args.batch_size) % 25 == 0 or offset + len(batch) == len(pending):
                done = len(completed) + offset + len(batch)
                elapsed = time.monotonic() - started
                print(f"PROGRESS {done}/{len(rows)} elapsed_sec={elapsed:.0f}", flush=True)
    export_complete(connection, rows, args)
    connection.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAILED {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
