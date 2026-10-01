# Model-assisted generation

All generation uses a chat-completions endpoint compatible with the request schema in `api.py`. Review the model identifier and decoding parameters in `configs/*.json`; endpoint deployments choose their own served names. The client does not load `.env` files automatically.

```bash
export PCT_API_BASE='https://your-server.example/v1'
read -rsp 'API key: ' PCT_API_KEY
export PCT_API_KEY
```

HTTPS is supported; HTTP is accepted for localhost endpoints. Never put real credentials in a config committed to GitHub. `.env.example` lists the variable names.

Commands with `--api-config` are **dry runs by default**: they show the selected model and write no outputs. Add `--execute` to run requests after configuring a service. Dry run is a configuration preview, not a full validation of a large dataset. Successful, validated completions are stored in a local SQLite cache keyed by endpoint, revision, and complete request payload. Rerunning after interruption reuses completed requests. Retries do not cache invalid JSON, missing rubric fields, empty completions, or token-limit truncation. Completed output files are replaced atomically; the cache preserves progress when a run fails.

## Multilingual variants

```bash
pct4b variants --input data/wiki.json --routes configs/variant_routes.json \
  --api-config configs/teacher_gemma.json --cache runs/gemma.sqlite \
  --output runs/wiki_variants.jsonl
```

The ordered routes are Persian→English, Persian→Arabic, then English→Chinese/Hebrew/Spanish. Each available field is translated separately. Existing target fields remain intact. New routes can be configured without changing source code. The recorded teacher is Gemma 3 **27B IT**, as identified by the original scripts. Generation parameters in portable examples should be reviewed for a new run; they are not a substitute for a missing historical request log.

## Prompt paraphrases

```bash
pct4b prompts seeds --output runs/seeds.jsonl
pct4b prompts generate --input runs/seeds.jsonl --count 20 \
  --api-config configs/paraphrases.json --cache runs/prompts.sqlite \
  --output runs/paraphrases.jsonl
```

`--count` is the number of unique paraphrases **per seed**. Each completion must preserve every source/target placeholder. Insufficient valid unique strings cause a clear error after a bounded number of batches. This is a new implementation of the generation procedure; generated strings are not claimed to be the exact original pools.

## Synthetic translation corpus

```bash
pct4b topics --input examples/topics.jsonl --depth 3 --branching 3 \
  --api-config configs/source_deepseek.json --output runs/topics.jsonl
pct4b sources --input runs/topics.jsonl --config configs/synthetic.json \
  --api-config configs/source_deepseek.json --output runs/sources.jsonl
pct4b translate --input runs/sources.jsonl --api-config configs/teacher_deepseek.json \
  --output runs/pairs.jsonl
```

Run each step with `--execute` before the next step needs its output. Topic depth counts the root node as level one. The source generator samples configurable genre, register, complexity, length, noise, and style attributes, and generates directly in each source language. Translation chooses a reproducible uniform non-self target when one is not supplied. The compact example attribute inventory is intended for new experiments; historical source texts and their sampled attributes must be retained to reproduce the original corpus exactly.

Historical source metadata identifies DeepSeek-V3.1. Translation and judge code records `deepseek-chat`, which is a service alias; the release leaves `revision: null` where an immutable backend revision is unavailable. Do not replace that missing provenance with a guessed version.

## Inference and model judging

```bash
pct4b translate --input data/test_pairs.jsonl --prediction --model-name PCT-4B \
  --prompt archived-short --api-config configs/inference.json \
  --cache runs/inference.sqlite --output runs/pct_predictions.jsonl
pct4b judge --input runs/pct_judge_predictions.jsonl --api-config configs/judge.json \
  --cache runs/judge.sqlite --output runs/pct_judged.jsonl
```

Prediction mode preserves references and adds `candidate`, `model`, and inference metadata. Use the same stored 300-item subset for every judged system. The archived prompt variants retain the historical wording and wrappers; `short`, `teacher`, and `baseline` provide concise portable alternatives. Historical prompt differences between systems are an evaluation limitation. For a new controlled comparison, choose the same template and decoding settings for all systems.

Judge outputs contain an overall integer score, a reason, and all eight integer rubric scores in [0, 100], both with and without a reference. Zero is a valid score. The original rubric texts are in `src/pct4b/assets/judge_*.txt`. Model outputs are data within the judge request; an LLM judge remains susceptible to bias and candidate-text prompt injection. Scores are not human assessment.
