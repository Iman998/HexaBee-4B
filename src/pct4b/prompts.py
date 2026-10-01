"""Typed prompt pools, imported paraphrases, and direction-conditioned sampling."""

from __future__ import annotations

import json
import math
from importlib.resources import files
from pathlib import Path

from .api import parse_json
from .io import read_records
from .schema import LANGUAGES, language

SEPARATORS = {
    " \n ": 0.60,
    " , ": 0.15,
    " - ": 0.05,
    " | ": 0.05,
    " \n\n ": 0.09,
    " -- ": 0.01,
    "    ": 0.05,
}


def weighted(rng, options):
    keys = list(options)
    weights = [float(options[k]) for k in keys]
    if not keys or any(not math.isfinite(x) or x < 0 for x in weights) or sum(weights) <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive mass")
    return keys[int(rng.choice(len(keys), p=[w / sum(weights) for w in weights]))]


def translation_weights(source, target, languages=None):
    source, target = language(source), language(target)
    if source == target:
        raise ValueError("Identity translation directions are not used")
    groups = {"fa": "rtl", "ar": "rtl", "he": "rtl", "en": "latin", "es": "latin", "zh": "cjk"}
    languages = list(LANGUAGES) if languages is None else [language(x) for x in languages]
    if (
        len(set(languages)) != len(languages)
        or len(languages) < 3
        or not {source, target}.issubset(languages)
    ):
        raise ValueError(
            "Prompt inventory must include source, target, and another distinct language"
        )
    rest = [p for p in languages if p not in (source, target)]
    same = [p for p in rest if groups[p] in (groups[source], groups[target])]
    other = [p for p in rest if p not in same]
    g = min(0.15 / 2, 0.01 * len(other))
    h = min(0.10, (0.15 - g) / len(same)) if same else 0
    residual = 0.15 - g - len(same) * h
    out = {
        p: 0.55
        if p == target
        else 0.30
        if p == source
        else (g / len(other) if p in other else 0) + (h if p in same else 0) + residual / len(rest)
        for p in languages
    }
    return out


class PromptPool:
    def __init__(self, rows):
        self.names = json.loads(
            files("pct4b.assets").joinpath("localized_languages.json").read_text()
        )
        self.rows = []
        self._index = {}
        self._candidates = {}
        for row in rows:
            row = dict(row)
            row["language"] = language(row["language"])
            if not isinstance(row.get("text"), str) or not row["text"].strip():
                raise ValueError("Prompt text must be nonempty")
            if row["component"] not in (
                "language",
                "complete",
                "moderate",
                "formal",
                "informal",
                "translate",
            ):
                raise ValueError("Unknown prompt component")
            # Target-independent extent/style prompts are legal and explicit.
            if row.get("target"):
                row["target"] = language(row["target"])
            if row.get("source"):
                row["source"] = language(row["source"])
            index_key = (row["component"], row.get("source"), row.get("target"))
            self._index.setdefault(index_key, []).append((len(self.rows), row))
            self.rows.append(row)
        if not self.rows:
            raise ValueError("Empty prompt pool")

    @classmethod
    def load(cls, path=None):
        if path:
            return cls(read_records(path))
        return cls(json.loads(files("pct4b.assets").joinpath("seed_pools.json").read_text()))

    def sample(self, component, source, target, rng, weights=None):
        key = (component, source, target)
        if key not in self._candidates:
            matches = []
            for s, t in ((None, None), (source, None), (None, target), (source, target)):
                matches.extend(self._index.get((component, s, t), []))
            candidates = {}
            # Preserve original row order and duplicate sampling mass across wildcard pools.
            for _, row in sorted(matches, key=lambda match: match[0]):
                candidates.setdefault(row["language"], []).append(row)
            self._candidates[key] = candidates
        candidates = self._candidates[key]
        weights = weights or {p: 1 for p in LANGUAGES}
        possible = {p: w for p, w in weights.items() if p in candidates and w > 0}
        p = weighted(rng, possible)
        row = candidates[p][int(rng.integers(len(candidates[p])))]
        names = self.names
        value = (
            row["text"].replace("{target}", names[p][target]).replace("{source}", names[p][source])
        )
        return value, p

    def compose(self, components, source, target, rng, weights=None):
        if not components:
            raise ValueError("At least one prompt component is required")
        order = list(rng.permutation(len(components)))
        selected = [self.sample(components[i], source, target, rng, weights) for i in order]
        result = selected[0][0]
        for value, _ in selected[1:]:
            result += weighted(rng, SEPARATORS) + value
        return result, [p for _, p in selected]


def generate_paraphrases(client, seeds, count, max_batches=5):
    """Generate exact requested *unique* paraphrases per seed; reject changed placeholders."""
    import re

    if count < 1 or max_batches < 1:
        raise ValueError("count and max_batches must be positive")
    result = []
    for seed in seeds:
        seen = set()
        wanted = sorted(re.findall(r"\{(?:source|target)\}", seed["text"]))

        def validate(raw, wanted=wanted):
            values = parse_json(raw)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(x, str) and x.strip() for x in values)
            ):
                raise ValueError("Expected a nonempty JSON array of strings")
            if any(sorted(re.findall(r"\{(?:source|target)\}", x)) != wanted for x in values):
                raise ValueError("Paraphrase changed or removed a placeholder")
            return values

        for batch in range(max_batches):
            if len(seen) >= count:
                break
            request = {
                "seed": seed["text"],
                "language": LANGUAGES[seed["language"]],
                "count": count - len(seen),
                "batch": batch,
                "exclude": sorted(seen),
            }
            raw = client.complete(
                [
                    {
                        "role": "system",
                        "content": "Paraphrase the instruction in its specified language. Preserve its meaning and every {source}/{target} placeholder exactly. Return only a JSON array of distinct strings.",
                    },
                    {"role": "user", "content": json.dumps(request, ensure_ascii=False)},
                ],
                validate,
            )
            seen.update(x.strip() for x in validate(raw))
        if len(seen) < count:
            raise RuntimeError("Not enough valid unique paraphrases; increase max_batches")
        result.extend(
            {**seed, "text": t, "generator_model": client.config["model"]}
            for t in sorted(seen)[:count]
        )
    return result


def import_pools(folder):
    """Import original combined pools, preserving duplicate rows and their sampling mass."""
    import re

    names = "persian|english|arabic|chinese|hebrew|spanish"
    for path in sorted(Path(folder).rglob("*_combined.json")):
        if any(part.startswith(".") for part in path.relative_to(folder).parts):
            continue
        mt = re.fullmatch(
            rf"system_prompt_translate_({names})_({names})_({names})_combined", path.stem
        )
        qa = re.fullmatch(
            rf"(?:system_prompt_)?(complete|balanced|balance|formal|informal)?_?({names})_({names})_combined",
            path.stem,
        )
        if mt:
            wording, source, target = mt.groups()
            meta = {
                "component": "translate",
                "language": language(wording),
                "source": language(source),
                "target": language(target),
            }
        elif qa:
            kind, wording, target = qa.groups()
            kind = {None: "language", "balanced": "moderate", "balance": "moderate"}.get(kind, kind)
            meta = {"component": kind, "language": language(wording), "target": language(target)}
        else:
            continue
        for row in read_records(path):
            yield {**meta, "text": row["text"]}
