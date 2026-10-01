"""Replay audited QA column/sampling policies on a supplied source partition.

Unlike the scalable builder, this deliberately uses the original NumPy RNG order
and pandas-compatible row selection. Each partition is loaded in memory.
"""

import collections
from pathlib import Path

import numpy as np

from .io import read_records
from .prompts import weighted
from .schema import LANGUAGES, normalize_wiki


def legacy_columns(row):
    """Map the unified export to legacy column names without overriding supplied fields."""
    values = dict(row)
    variants = normalize_wiki(row)["variants"]
    for code, v in variants.items():
        full = LANGUAGES[code]
        for key, suffix in {
            "question": "instruction",
            "answer": "output",
            "formal_question": "formal_instruction",
            "formal_answer": "formal_output",
            "informal_question": "slang_instruction",
            "informal_answer": "slang_output",
        }.items():
            values.setdefault(full + "_" + suffix, v.get(key))
        if code == "fa":
            mapping = {
                "instruction": "question",
                "revise_instruction": "question",
                "revise_question": "question",
                "revise_output": "answer",
                "complete_answer": "answer",
                "formal_instruction": "formal_question",
                "formal_output": "formal_answer",
                "slang_instruction": "informal_question",
                "slang_output": "informal_answer",
                "normal_output": "moderate_answer",
            }
            for old, new in mapping.items():
                values.setdefault(old, v.get(new))
    return values


def replay(rows, recipe, pool_dir, counts=None, rule_id=None):
    counts = counts if counts is not None else collections.Counter()
    rows = [legacy_columns(row) for row in rows]
    paths = {}
    for path in sorted(Path(pool_dir).rglob("*_combined.json")):
        if any(part.startswith(".") for part in path.relative_to(pool_dir).parts):
            continue
        if path.name in paths:
            raise ValueError(f"Ambiguous prompt pool filename: {path.name}")
        paths[path.name] = path
    cache = {}
    rules = [r for r in recipe["rules"] if rule_id is None or r["id"] == rule_id]
    if not rules:
        raise ValueError("No matching historical rule")
    for rule in rules:
        cfg = rule["composition"]
        groups = []
        for group in rule["prompt_groups"]:
            choices = []
            for pool in group:
                if pool["file"] not in paths:
                    raise ValueError(f"Missing historical prompt pool: {pool['file']}")
                if pool["file"] not in cache:
                    cache[pool["file"]] = [r["text"] for r in read_records(paths[pool["file"]])]
                if cache[pool["file"]]:
                    choices.append((cache[pool["file"]], pool["weight"]))
            if not choices:
                raise ValueError("A historical prompt group is empty")
            groups.append(choices)
        spec = rule["sample"]
        # df without sample preserves its original order; df.sample(frac=1) does not.
        n = int(spec["n"]) if "n" in spec else round(float(spec.get("frac", 1)) * len(rows))
        if spec.get("applied", True):
            indexes = np.random.RandomState(spec.get("random_state", 42)).choice(
                len(rows), n, replace=spec.get("replace", False)
            )
        else:
            indexes = np.arange(len(rows))
        rr = [rows[int(i)] for i in indexes]
        rng = np.random.default_rng(cfg.get("seed", 42))
        columns = {}
        for outkey, policy in rule["column_policy"].items():
            columns[outkey] = [row.get(weighted(rng, policy)) for row in rr]
        for i, _row in enumerate(rr):
            k = cfg.get("n_sentences")
            if k is None:
                k = int(
                    rng.integers(
                        cfg.get("min_sentences", 1), cfg.get("max_sentences", len(groups)) + 1
                    )
                )
            chosen = (
                rng.choice(len(groups), size=k, replace=False)
                if cfg.get("random_groups")
                else np.arange(k)
            )
            sentences = []
            for g in chosen:
                choices = groups[int(g)]
                index = weighted(rng, {j: w for j, (_, w) in enumerate(choices)})
                strings = choices[index][0]
                sentences.append(str(strings[int(rng.integers(len(strings)))]))
            seps = cfg.get("sep_choices")
            system = sentences[0]
            for sentence in sentences[1:]:
                system += (
                    weighted(rng, seps) if seps else cfg.get("sep_between_sentences", " ")
                ) + sentence
            out = {key: values[i] for key, values in columns.items()}
            if any(value is None for value in out.values()):
                counts["null_removed"] += 1
                continue
            # Historical merger retained empty strings; do not silently repair the record.
            counts[rule["id"]] += 1
            yield {
                "system": system,
                **out,
                "rule_id": rule["id"],
                "task": rule["task"],
                "source_language": rule["source"],
                "target_language": rule["target"],
                "known_instruction_conflict": rule["known_instruction_conflict"],
            }
