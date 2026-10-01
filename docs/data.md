# Data construction

## Inputs and canonical records

Readers detect JSON arrays and JSONL from file contents, not filename extensions. The original Wikipedia export has a `.json` suffix but contains JSONL. Processing is streamed, except for historical replay (which loads a partition), judge sampling, audits with duplicate-ID tracking, and evaluation pairing.

`pct4b inspect --input PATH --limit 1` displays field names/types without printing text. `normalize-wiki` accepts the original flat fields, including `Persian_instruction`, `Persian_output`, `English_instruction`, `English_output`, and corresponding Arabic, Chinese, Hebrew, and Spanish columns. Persian aliases include `revise_instruction`, `revise_question`, `revise_output`, and `complete_answer`; style aliases are implemented in `schema.py`.

The normalized format is:

```json
{"schema_version":"wiki-v1","id":"content-derived identity","original_id":42,"title":"Water","context":"Upstream source context","variants":{"fa":{"question":"پرسش","answer":"پاسخ"},"en":{"question":"Question","answer":"Answer"}}}
```

The initial ID hashes the original ID, title, context, and available variants. Normalization is idempotent, and generating additional variants preserves that canonical identity. Language variants can additionally contain `formal_question`, `formal_answer`, `informal_question`, `informal_answer`, `moderate_answer`, and `short_answer`. Missing translations/styles remain missing until explicitly generated; the builder records skipped combinations.

## Build from a Wikipedia-style export

```bash
pct4b normalize-wiki --input data/wiki.json --output runs/wiki.jsonl
pct4b qa-build --input runs/wiki.jsonl --recipe configs/qa_multilingual.json \
  --pools data/prompt_pools.jsonl --output runs/phase2.jsonl
pct4b audit --input runs/phase2.jsonl --output runs/phase2_audit.json
pct4b export-sft --input runs/phase2.jsonl --output runs/phase2_train.jsonl
```

`--limit 100` is useful for first checking a large input. When a limit is supplied, provenance deliberately omits the full input hash to avoid scanning the complete source file. Otherwise a sidecar manifest records input/output SHA-256 hashes, selected settings, recipe hashes, and counters. API credentials/endpoints are not copied into manifests.

Each JSON recipe rule specifies `source`, `target`, `task`, `fraction`, and optionally an explicit `id`, field choices, components, or prompt-language weights. Tasks are `complete`, `moderate`, `formal`, `informal`, and `translate`. Different rules need different IDs. For QA, `instruct` is the question and `output` is the answer; for answer-text MT, both fields come from aligned answers. Context is upstream provenance, not an additional model input.

The default scalable recipe has all 36 question/answer language combinations for each of four QA styles and 30 answer-translation directions. QA combinations use fraction 1; MT uses 0.70 for Persian↔English, 0.25 for other directions touching Persian or English, and 0.08 otherwise. A fraction is stable per-item Bernoulli sampling based on SHA-256. It is order-independent and produces an expected count, not pandas' exact rounded sample size. Missing fields reduce the realized inventory.

## System prompts

```bash
pct4b prompts seeds --output runs/seeds.jsonl
pct4b prompts import --input data/original_prompt_directory --output runs/imported_pools.jsonl
```

Pool rows contain `component`, `language` (wording language), `text`, and optionally `source`/`target`. Imported translation filenames encode **wording language, source language, target language**. QA filenames encode wording and requested answer language. Hidden notebook checkpoint directories are ignored; duplicate paraphrase rows within a real pool are retained because they contribute sampling mass.

The six component types are response `language`, `complete`, `moderate`, `formal`, `informal`, and `translate`. Placeholders `{source}` and `{target}` use localized language names. Cross-language QA always includes a response-language directive in the default builder.

For MT, the target wording language receives mass 0.55 and the source wording language receives 0.30. The remaining 0.15 is allocated using the recorded script-group heuristic (Persian/Arabic/Hebrew, English/Spanish, Chinese), including residual redistribution. This is a script heuristic, not a linguistic language-family taxonomy. Historical notebooks have different active inventories of four, five, or six languages; their recipe files store the corresponding ordered weights. A provisional 0.10 residual cap is followed by redistribution and is not a final per-language bound.

Prompt composition randomizes component order and uses recorded separator weights. The release's 36 small seeds demonstrate the interface. They do not replace the original, much larger paraphrase inventory for historical replay. New paraphrases can be generated with the workflow in [generation.md](generation.md).

## Historical construction policies

```bash
pct4b historical-build --input data/ordered_qa_a_partition.jsonl \
  --recipe configs/historical/qa_a.json --pool-dir data/original_prompt_directory \
  --output runs/replayed_qa_a.jsonl
pct4b historical-build --input data/ordered_qa_a_partition.jsonl \
  --recipe configs/historical/qa_a_translation.json --pool-dir data/original_prompt_directory \
  --output runs/replayed_qa_a_mt.jsonl
```

Repeat for `gqa_a`, `gqa_b`, `qa_a`, `qa_b`, `title_a`, and `title_b`. These are neutral bookkeeping labels for the six construction groups. The 187 QA and 142 MT rules preserve sampled fractions, column policies, prompt pool order, composition settings, and audited retained counts. `--rule-id` restricts a run to one rule.

Replay uses NumPy's original random-generator sequence and pandas-compatible `RandomState` index sampling. Exact row reproduction requires the same original ordered source partition, exact original fields, and original prompt-pool row order. The unified export alone does not reconstruct all of that history. Canonical adapters do not invent missing legacy columns or fallback texts. Missing/null values are removed, while historical empty strings and known instruction conflicts remain flagged. The strict `export-sft` command rejects empty training fields; use retained historical exports directly if replaying those historical defects is required. See [reproducibility.md](reproducibility.md).

## Translation pairs and final SFT

Canonical aligned pairs use:

```json
{"source_language":"en","target_language":"fa","source":"Water is a liquid.","reference":"آب مایع است."}
```

The original `src_language`, `tgt_lang`, `text`, and `gold_translation` fields are accepted. A content identity hashes the normalized language names, exact source, and reference.

```bash
pct4b split --input data/pairs.jsonl --output runs/split --test-fraction 0.1 --seed 42
pct4b mt-build --input runs/split/train.jsonl --output runs/translation_sft.jsonl
pct4b export-sft --input runs/translation_sft.jsonl --output runs/translation_train.jsonl
pct4b judge-sample --input runs/split/test.jsonl --per-direction 10 \
  --output runs/judge_inputs.jsonl
```

Split **before reversal**. New splits group identical whitespace-normalized source strings and use a deterministic hash threshold. This avoids exact source leakage but does not detect paraphrases or semantic duplicates. The original split was unseeded; this new command intentionally does not claim the same membership. For paper-score reproduction, use the stored original held-out examples.

MT construction removes invalid/equal-text pairs and exact duplicate pair identities, then writes forward and reverse examples. Reverse augmentation swaps an existing aligned pair; it is not model-generated back-translation. Judge sampling is deterministic within every observed direction and fails when a direction has fewer than the requested number of items.

Training exports contain `system`, `instruct`, `output`; `--format messages` emits role-tagged chat messages. The construction output keeps identities, task labels, directions, and prompt-language metadata for analysis.
