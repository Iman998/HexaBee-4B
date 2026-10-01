# Translation evaluation

## Prediction format and strict pairing

A prediction is an aligned pair plus a string `model` and `candidate`. A null candidate is normalized to an empty string and still scored. No quotes, tags, preambles, or output wrappers are stripped.

```json
{"model":"PCT-4B","source_language":"en","target_language":"fa","source":"Water is a liquid.","reference":"آب مایع است.","candidate":"آب مایع است."}
```

Multiple files may represent different models or disjoint shards of the same model. Duplicate content identities within a model cause an error. Every model must have exactly the same identities, computed from source language, target language, exact source text, and reference. Row order and a numeric record ID do not establish a valid paired comparison.

Archived outputs are converted explicitly:

```bash
pct4b archive-import --input data/archived_outputs.json --model PCT-4B \
  --candidate-field translated_text --output runs/pct_predictions.jsonl
```

Replace `translated_text` with the actual archived candidate column. Archived judge responses are reparsed from `raw` when present, recovering valid zero scores rather than using truthiness fallbacks. The original retrospective audit recovered 11 reference-assisted and 13 reference-free zeros for Gemma 3 4B IT, and one reference-free zero for Gemma 3 12B IT.

## Automatic metrics

```bash
pct4b metrics --input data/gemma4.jsonl data/gemma12.jsonl data/pct.jsonl \
  --metrics bleu chrf chrfpp ter --output runs/full_metrics.json
pct4b report --input runs/full_metrics.json --output runs/full_report --focus PCT-4B --plots
```

| Metric | Configuration | Scope |
|:--|:--|:--|
| BLEU | SacreBLEU 2.5.1; Chinese-target `zh`, otherwise `13a`; mixed case, exponential smoothing | Per-direction corpus score; direction macro |
| chrF | Character order 6, word order 0, beta 2; mixed case, no whitespace | Per-direction, macro, pooled corpus |
| chrF++ | Character order 6, word order 2, beta 2; mixed case, no whitespace | Per-direction, macro, pooled corpus |
| TER | Normalization and Asian support enabled; case-insensitive, punctuation retained | Per-direction, macro, pooled corpus; lower is better |
| COMET | `Unbabel/wmt22-comet-da`, unbabel-comet 2.2.7, float32 | Per-direction, macro, pooled segment mean; scaled ×100 |

SacreBLEU signatures and sufficient statistics are written to JSON. The release uses the official SacreBLEU TER implementation; long examples can take substantial CPU time. The manuscript's full TER calculation used an independently validated acceleration, whose resulting sufficient statistics are included. Reconstructing scores from those statistics is fast; it is not a fresh check of the original token alignment. BLEU is not pooled across target tokenizers.

For direction set $D$, the macro is $M_{macro}=|D|^{-1}\sum_{d\in D}M_d$. Target-language macros average the incoming directions. A pooled corpus score applies the metric to all hypotheses/references together and generally differs from both a direction macro and an example-weighted mean of direction scores. TER counts can exceed 100 when edit costs exceed reference length.

### Optional COMET

Install `.[metrics,comet]` in a separate compatible environment and obtain the checkpoint explicitly from its official distribution. The known checkpoint is:

```text
Repository: Unbabel/wmt22-comet-da
Revision: 2760a223ac957f30acfb18c8aa649b01cf1d75f2
File: checkpoints/model.ckpt
SHA-256: e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af
```

```bash
pct4b metrics --input data/gemma4.jsonl data/gemma12.jsonl data/pct.jsonl \
  --metrics bleu chrf chrfpp ter comet --checkpoint models/comet.ckpt \
  --checkpoint-sha256 e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af \
  --batch-size 32 --gpus 1 --output runs/full_with_comet.json
```

The command does not select/download a COMET checkpoint implicitly. COMET's encoder/tokenizer also needs its upstream assets; cache them beforehand when running offline. The recorded encoder limit is 512 positions. Long texts are scored under that truncation, and source/candidate/reference truncation counts are reported. These counts describe truncation, not translation correctness.

## Judging and uncertainty

Generate both judge views as described in [generation.md](generation.md). The eight dimensions are semantic accuracy, terminology, fluency, cohesion, contextual appropriateness, target-language adherence, style, and formatting. The judge's overall score is a separately returned score, not an assumed average of the eight dimensions.

```bash
pct4b metrics --input data/gemma4_judged.jsonl data/gemma12_judged.jsonl data/pct_judged.jsonl \
  --metrics chrfpp ter --output runs/judge_metrics.json
pct4b bootstrap --input data/gemma4_judged.jsonl data/gemma12_judged.jsonl data/pct_judged.jsonl \
  --system PCT-4B --baselines 'Gemma 3 4B IT' 'Gemma 3 12B IT' \
  --replicates 10000 --seed 20260912 --output runs/judge_intervals.json
pct4b report --input runs/judge_metrics.json --output runs/judge_report --focus PCT-4B
```

For example $i$ in direction $d$, let $\delta_{d,i}=J_{system,d,i}-J_{baseline,d,i}$. Each bootstrap replicate samples with replacement **within every direction**, retaining system/baseline pairing, and computes

$$\Delta^{(b)}=\frac1{|D|}\sum_{d\in D}\frac1{n_d}\sum_{j=1}^{n_d}\delta_{d,I^{(b)}_{d,j}}.$$

The 2.5th and 97.5th percentiles form the interval. Equal direction weights are retained even if direction sample sizes differ. Win/tie/loss counts compare individual examples. These intervals quantify uncertainty in the stored judge scores; they are not uncertainty over training runs, judges, or synthetic-reference generation.

## Released results

`all_direction_metrics.csv` has 90 rows (three systems × 30 directions), automatic metrics for the full test, judged-subset metrics, both judge views, and all eight section means per view. `paired_scores.csv` contains only model identifiers, hashed content identities, directions, and two overall judge scores, allowing interval reproduction without releasing source texts. `scripts/reproduce_paper.py` validates aggregates and rebuilds separate full-test/judged-subset reports so their sample counts cannot be confused.
