"""Strict identity pairing, translation metrics, model judging and paired uncertainty."""

from __future__ import annotations

import collections
import json
import statistics
from importlib.resources import files

import numpy as np

from .api import parse_json
from .io import file_hash, read_records
from .schema import LANGUAGES, SECTIONS, normalize_pair


def parse_judge(raw):
    value = parse_json(raw) if isinstance(raw, str) else raw
    if not isinstance(value, dict) or type(value.get("score")) is not int:
        raise ValueError("Judge must supply an integer score (zero is valid)")
    if not 0 <= value["score"] <= 100:
        raise ValueError("Score outside [0, 100]")
    sections = value.get("section_scores")
    if not isinstance(sections, dict) or set(sections) != set(SECTIONS):
        raise ValueError("Judge must supply all eight section scores")
    if any(type(v) is not int or not 0 <= v <= 100 for v in sections.values()):
        raise ValueError("Invalid section score")
    if not isinstance(value.get("reason"), str):
        raise ValueError("Judge must supply a short reason")
    return value


def judge(rows, client):
    prompts = {"reference": "judge_ref.txt", "no_reference": "judge_noref.txt"}
    for raw in rows:
        row = normalize_pair(raw)
        if "candidate" not in row or "model" not in row:
            raise ValueError("Judge input requires candidate and model")
        out = dict(row)
        for view, file in prompts.items():
            content = {
                "source_language": LANGUAGES[row["source_language"]],
                "target_language": LANGUAGES[row["target_language"]],
                "source": row["source"],
                "candidate": row["candidate"],
            }
            if view == "reference":
                content["ground_truth"] = row["reference"]
            raw_answer = client.complete(
                [
                    {"role": "system", "content": files("pct4b.assets").joinpath(file).read_text()},
                    {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
                ],
                parse_judge,
            )
            out["judge_" + view] = {**parse_judge(raw_answer), "raw": raw_answer}
        out["judge_model"] = client.config["model"]
        yield out


def import_archive(rows, model, candidate_field):
    for raw in rows:
        row = normalize_pair(raw)
        if candidate_field not in raw:
            raise ValueError(f"Missing candidate column: {candidate_field}")
        out = {
            k: row[k] for k in ("id", "source_language", "target_language", "source", "reference")
        }
        candidate = raw[candidate_field]
        if candidate is not None and not isinstance(candidate, str):
            raise ValueError("Candidate column must contain strings or null")
        out.update(model=model, candidate="" if candidate is None else candidate)
        for view in ("reference", "no_reference"):
            key = "judge_" + view
            if key in raw:
                value = raw[key]
                # Never use `score or fallback`: 0 must remain 0.
                parsed = parse_judge(value["raw"] if value.get("raw") is not None else value)
                out[key] = {**parsed, **({"raw": value["raw"]} if value.get("raw") else {})}
        yield out


def paired_rows(paths):
    models = collections.defaultdict(dict)
    for path in paths:
        for raw in read_records(path):
            row = normalize_pair(raw)
            if not isinstance(row.get("model"), str) or "candidate" not in row:
                raise ValueError("Prediction rows require model and candidate")
            if row["id"] in models[row["model"]]:
                raise ValueError("Duplicate content identity within one model")
            models[row["model"]][row["id"]] = row
    if not models:
        raise ValueError("No predictions to evaluate")
    ids = set(next(iter(models.values())))
    if any(set(rows) != ids for rows in models.values()):
        raise ValueError("Models do not share exactly the same source/reference identities")
    return models


def score_metrics(
    paths,
    metrics=("bleu", "chrf", "chrfpp", "ter"),
    checkpoint=None,
    batch_size=32,
    gpus=0,
    checkpoint_sha256=None,
):
    from sacrebleu.metrics import BLEU, CHRF, TER

    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("Select one or more distinct metrics")
    if batch_size < 1 or gpus < 0:
        raise ValueError("batch_size must be positive and gpus nonnegative")
    models = paired_rows(paths)
    unknown = set(metrics) - {"bleu", "chrf", "chrfpp", "ter", "comet"}
    if unknown:
        raise ValueError(f"Unknown metrics: {unknown}")
    report = {
        "schema_version": 1,
        "scope": "raw stored outputs; no wrapper stripping",
        "paired_id_sets_equal": True,
        "metrics": list(metrics),
        "results": [],
        "macro": {},
        "pooled": {},
        "comet": None,
    }
    comet = None
    if "comet" in metrics:
        if not checkpoint:
            raise ValueError("COMET requires --checkpoint; weights are never downloaded implicitly")
        if checkpoint_sha256 and file_hash(checkpoint) != checkpoint_sha256:
            raise ValueError("COMET checkpoint SHA-256 mismatch")
        import torch
        from comet import load_from_checkpoint

        torch.set_num_threads(2)
        comet = load_from_checkpoint(checkpoint)
        comet.eval()
        report["comet"] = {
            "name": "wmt22-comet-da"
            if checkpoint_sha256
            == "e213091cde220f97b89f8bdfa750c458cfea741ad62affb455b59900210ff2af"
            else "user-supplied checkpoint",
            "sha256": file_hash(checkpoint),
            "precision": "float32",
            "batch_size": batch_size,
            "encoder_max_positions": comet.encoder.max_positions,
            "scale": 100,
        }
    for model, items in models.items():
        rows = [items[k] for k in sorted(items)]
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row["source_language"] + "-" + row["target_language"]].append(row)
        comet_scores = {}
        if comet is not None:
            samples = [
                {"src": r["source"], "mt": r["candidate"], "ref": r["reference"]} for r in rows
            ]
            pred = comet.predict(
                samples, batch_size=batch_size, gpus=gpus, num_workers=0, progress_bar=False
            )
            comet_scores = dict(zip([r["id"] for r in rows], pred.scores, strict=True))
            truncated = {}
            for field in ("src", "mt", "ref"):
                lengths = comet.encoder.tokenizer(
                    [r[field] for r in samples], truncation=False, padding=False, return_length=True
                )["length"]
                truncated[field] = sum(n > comet.encoder.max_positions for n in lengths)
            report["comet"].setdefault("truncated_text_counts", {})[model] = truncated
        model_results = []
        for pair, rr in sorted(groups.items()):
            hyp, ref = [r["candidate"] for r in rr], [[r["reference"] for r in rr]]
            scorers = {
                "bleu": BLEU(tokenize="zh" if pair.endswith("-zh") else "13a"),
                "chrf": CHRF(word_order=0),
                "chrfpp": CHRF(word_order=2),
                "ter": TER(normalized=True, asian_support=True),
            }
            record = {
                "model": model,
                "direction": pair,
                "n": len(rr),
                "scores": {},
                "signatures": {},
                "statistics": {},
            }
            for name in metrics:
                if name == "comet":
                    value = 100 * statistics.mean(float(comet_scores[r["id"]]) for r in rr)
                else:
                    scorer = scorers[name]
                    stats = scorer._extract_corpus_statistics(hyp, ref)
                    total = np.asarray(stats).sum(axis=0).tolist()
                    value = scorer._compute_score_from_stats(total).score
                    record["signatures"][name] = str(scorer.get_signature())
                    record["statistics"][name] = total
                record["scores"][name] = value
            for view in ("reference", "no_reference"):
                key = "judge_" + view
                if any(key in r for r in rr):
                    if not all(key in r for r in rr):
                        raise ValueError(
                            "Incomplete judged direction; do not mix full/judge scopes"
                        )
                    values = [parse_judge(r[key]) for r in rr]
                    record[key] = {
                        "n": len(values),
                        "score": statistics.mean(v["score"] for v in values),
                        "section_scores": {
                            k: statistics.mean(v["section_scores"][k] for v in values)
                            for k in SECTIONS
                        },
                    }
            model_results.append(record)
        report["results"].extend(model_results)
        report["macro"][model] = {
            k: statistics.mean(r["scores"][k] for r in model_results) for k in metrics
        }
        pooled = {"n": len(rows), "empty_candidates": sum(not r["candidate"] for r in rows)}
        for name in ("chrf", "chrfpp", "ter"):
            if name in metrics:
                total = (
                    np.asarray([r["statistics"][name] for r in model_results]).sum(axis=0).tolist()
                )
                pooled[name] = scorers[name]._compute_score_from_stats(total).score
        if comet_scores:
            pooled["comet"] = 100 * statistics.mean(map(float, comet_scores.values()))
        # BLEU is never pooled over different target tokenizers.
        report["pooled"][model] = pooled
    return report


def bootstrap(paths, system, baselines, replicates=10000, seed=20260912):
    models = paired_rows(paths)
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    for model in [system] + list(baselines):
        if model not in models:
            raise ValueError(f"Missing model: {model}")
    keys = sorted(models[system])
    pairs = [
        models[system][k]["source_language"] + "-" + models[system][k]["target_language"]
        for k in keys
    ]
    groups = [np.array([i for i, p in enumerate(pairs) if p == d]) for d in sorted(set(pairs))]
    rng = np.random.default_rng(seed)
    indices = np.concatenate(
        [rng.choice(g, size=(replicates, len(g)), replace=True) for g in groups], axis=1
    )
    # Equal-direction weights also handle unequal stratum sizes correctly.
    weights = np.concatenate([np.full(len(g), 1 / len(g) / len(groups)) for g in groups])
    out = {
        "system": system,
        "seed": seed,
        "replicates": replicates,
        "n": len(keys),
        "directions": len(groups),
        "aggregation": "direction-macro",
        "paired_differences": {},
    }
    for baseline in baselines:
        results = {}
        for view in ("reference", "no_reference"):

            def score(model, key, view=view):
                return parse_judge(models[model][key]["judge_" + view])["score"]

            diff = np.array([score(system, k) - score(baseline, k) for k in keys])
            samples = (diff[indices] * weights).sum(1)
            results[view] = {
                "mean": float(np.mean([diff[g].mean() for g in groups])),
                "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                "wins": int((diff > 0).sum()),
                "ties": int((diff == 0).sum()),
                "losses": int((diff < 0).sum()),
            }
        out["paired_differences"][baseline] = results
    return out
