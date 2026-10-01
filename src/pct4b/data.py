"""Streaming multilingual supervision and leakage-aware translation splitting."""

from __future__ import annotations

import collections
import json
import sqlite3
import tempfile
from pathlib import Path

import numpy as np

from .io import atomic_json, digest, read_records, write_records
from .prompts import translation_weights
from .schema import language, normalize_pair, normalize_wiki, text


def selected(source_id, rule_id, fraction, seed):
    if not 0 <= fraction <= 1:
        raise ValueError("Sampling fraction must lie in [0, 1]")
    # Stable per-record Bernoulli selection; independent of worker count and stream order.
    return int(digest([seed, source_id, rule_id])[:16], 16) / 2**64 < fraction


def build_qa(rows, pool, recipe, counts=None):
    counts = counts if counts is not None else collections.Counter()
    rule_ids = set()
    for rule in recipe["rules"]:
        s, t = language(rule["source"]), language(rule["target"])
        if (s, t) != (rule["source"], rule["target"]):
            raise ValueError("Recipe languages must use canonical two-letter codes")
        task = rule.get("task", "complete")
        if task not in ("complete", "moderate", "formal", "informal", "translate"):
            raise ValueError(f"Unknown QA task: {task}")
        key = rule.get("id", f"{task}:{s}:{t}")
        if key in rule_ids:
            raise ValueError(f"Duplicate rule id: {key}")
        rule_ids.add(key)
    for row in rows:
        item = normalize_wiki(row)
        for rule in recipe["rules"]:
            s, t = rule["source"], rule["target"]
            task = rule.get("task", "complete")
            key = rule.get("id", f"{task}:{s}:{t}")
            if not selected(item["id"], key, rule.get("fraction", 1), recipe.get("seed", 42)):
                counts["not_sampled"] += 1
                continue
            rng = np.random.default_rng(
                int(digest([recipe.get("seed", 42), item["id"], key])[:16], 16)
            )
            sv, tv = item["variants"].get(s, {}), item["variants"].get(t, {})
            if task == "translate":
                if s == t:
                    raise ValueError("MT rules cannot have identical languages")
                x, y = sv.get("answer"), tv.get("answer")
                components = ["translate"]
            else:
                xfield = rule.get("question_field", "question")
                yfield = rule.get(
                    "answer_field", {"complete": "answer"}.get(task, task + "_answer")
                )
                x, y = sv.get(xfield), tv.get(yfield)
                components = list(rule.get("components", [task] + (["language"] if s != t else [])))
                if s != t and "language" not in components:
                    raise ValueError(
                        "Cross-language answering requires an explicit language directive"
                    )
            if not text(x) or not text(y):
                counts["missing_variant"] += 1
                continue
            weights = rule.get("prompt_weights")
            if task == "translate" and weights is None:
                weights = translation_weights(s, t)
            system, langs = pool.compose(components, s, t, rng, weights)
            counts[task + ":" + s + "-" + t] += 1
            yield {
                "id": digest([item["id"], key]),
                "source_id": item["id"],
                "task": task,
                "source_language": s,
                "target_language": t,
                "prompt_languages": langs,
                "system": system,
                "instruct": x,
                "output": y,
            }


def build_mt(rows, pool, reverse=True, seed=42, counts=None):
    counts = counts if counts is not None else collections.Counter()
    # Disk-backed pair deduplication; a full production corpus need not fit in memory.
    with tempfile.TemporaryDirectory() as tmp:
        db = sqlite3.connect(str(Path(tmp) / "seen.sqlite"))
        db.execute("CREATE TABLE seen (id TEXT PRIMARY KEY)")
        try:
            for raw in rows:
                try:
                    row = normalize_pair(raw)
                except (KeyError, ValueError):
                    counts["invalid"] += 1
                    continue
                if row["source"].strip() == row["reference"].strip():
                    counts["equal_text"] += 1
                    continue
                if not db.execute("INSERT OR IGNORE INTO seen VALUES (?)", (row["id"],)).rowcount:
                    counts["duplicate_pair"] += 1
                    continue
                s, t = row["source_language"], row["target_language"]
                directions = [(s, t, row["source"], row["reference"], "forward")]
                if reverse:
                    directions.append((t, s, row["reference"], row["source"], "reverse"))
                for a, b, x, y, orientation in directions:
                    rng = np.random.default_rng(
                        int(digest([seed, row["id"], orientation])[:16], 16)
                    )
                    system, p = pool.sample("translate", a, b, rng, translation_weights(a, b))
                    counts[orientation] += 1
                    yield {
                        "id": digest([row["id"], orientation]),
                        "pair_id": row["id"],
                        "task": "translate",
                        "orientation": orientation,
                        "source_language": a,
                        "target_language": b,
                        "prompt_languages": [p],
                        "system": system,
                        "instruct": x,
                        "output": y,
                    }
        finally:
            db.close()


def export_sft(rows, format="alpaca"):
    for row in rows:
        if any(not text(row.get(k)) for k in ("system", "instruct", "output")):
            raise ValueError("SFT record requires nonempty system/instruct/output")
        if format == "alpaca":
            yield {k: row[k] for k in ("system", "instruct", "output")}
        else:
            yield {
                "messages": [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["instruct"]},
                    {"role": "assistant", "content": row["output"]},
                ]
            }


def split_pairs(rows, outdir, test_fraction=0.1, seed=42):
    """Partition aligned pairs BEFORE reversal; group identical normalized sources.

    Hash splitting is reproducible and avoids exact/whitespace source overlaps.
    It intentionally improves the historical unseeded row split, not its exact membership.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must lie strictly between 0 and 1")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    with tempfile.TemporaryDirectory() as tmp:
        train, test = Path(tmp) / "train.jsonl", Path(tmp) / "test.jsonl"
        with train.open("w") as tr, test.open("w") as te:
            seen = sqlite3.connect(str(Path(tmp) / "seen.sqlite"))
            seen.execute("CREATE TABLE pairs (id TEXT PRIMARY KEY)")
            try:
                for raw in rows:
                    row = normalize_pair(raw)
                    if not seen.execute(
                        "INSERT OR IGNORE INTO pairs VALUES (?)", (row["id"],)
                    ).rowcount:
                        counts["duplicate_pair"] += 1
                        continue
                    source_key = " ".join(row["source"].split())
                    is_test = selected(source_key, "split", test_fraction, seed)
                    part = "test" if is_test else "train"
                    (te if is_test else tr).write(json.dumps(row, ensure_ascii=False) + "\n")
                    counts[part] += 1
            finally:
                seen.close()
        write_records(outdir / "train.jsonl", read_records(train))
        write_records(outdir / "test.jsonl", read_records(test))
    atomic_json(
        outdir / "split.json",
        {
            "method": "normalized-source SHA-256 threshold",
            "seed": seed,
            "requested_test_fraction": test_fraction,
            "counts": dict(counts),
            "before_reversal": True,
            "historical_membership_reproduced": False,
        },
    )
    return counts


def sample_judge(rows, per_direction=10, seed=42):
    if per_direction < 1:
        raise ValueError("per_direction must be positive")
    groups = collections.defaultdict(list)
    seen = set()
    for raw in rows:
        row = normalize_pair(raw)
        if row["id"] in seen:
            raise ValueError("Duplicate content identity in judge sampling input")
        seen.add(row["id"])
        groups[row["source_language"] + "-" + row["target_language"]].append(row)
    result = []
    for direction, group in sorted(groups.items()):
        if len(group) < per_direction:
            raise ValueError(f"{direction} has fewer than {per_direction} examples")
        result.extend(sorted(group, key=lambda r: digest([seed, r["id"]]))[:per_direction])
    return result


def audit_records(rows):
    counts = collections.Counter()
    directions = collections.Counter()
    ids = set()
    for row in rows:
        counts["records"] += 1
        for key in ("system", "instruct", "output", "candidate"):
            if key in row and row[key] is None:
                counts["null_" + key] += 1
            elif key in row and isinstance(row[key], str) and not row[key].strip():
                counts["empty_" + key] += 1
        key = row.get("id", digest(row))
        if key in ids:
            counts["duplicate_ids"] += 1
        ids.add(key)
        if "source_language" in row and "target_language" in row:
            directions[row["source_language"] + "-" + row["target_language"]] += 1
    return {"counts": dict(counts), "directions": dict(sorted(directions.items()))}
