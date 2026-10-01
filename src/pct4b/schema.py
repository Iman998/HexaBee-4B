"""Language labels and input adapters; no topic-based partition labels are required."""

from __future__ import annotations

import ast

from .io import digest

LANGUAGES = {
    "fa": "Persian",
    "en": "English",
    "ar": "Arabic",
    "zh": "Chinese",
    "he": "Hebrew",
    "es": "Spanish",
}
ALIASES = {
    **{k: k for k in LANGUAGES},
    **{v.lower(): k for k, v in LANGUAGES.items()},
    "farsi": "fa",
    "iw": "he",
    "zh-cn": "zh",
    "mandarin": "zh",
    "فارسی": "fa",
    "العربية": "ar",
}
SECTIONS = (
    "semantic_accuracy",
    "terminology",
    "fluency",
    "cohesion",
    "contextual_appropriateness",
    "target_language_adherence",
    "style",
    "formatting",
)


def language(value):
    try:
        return ALIASES[str(value).strip().lower()]
    except KeyError as e:
        raise ValueError(f"Unsupported language: {value!r}") from e


def unwrap(value):
    if isinstance(value, str) and value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                value = parsed
        except (SyntaxError, ValueError):
            pass
    if isinstance(value, dict):
        if value and all(str(k).isdigit() for k in value):
            return " ".join(str(value[k]) for k in sorted(value, key=lambda x: int(x)))
        if len(value) == 1:
            return next(iter(value.values()))
    return value


def text(value):
    value = unwrap(value)
    return value if isinstance(value, str) and value.strip() else None


def normalize_wiki(row):
    """Read unified Persian_* schema or original construction-column names."""
    if "variants" in row:
        variants = row["variants"]
        if not isinstance(variants, dict):
            raise ValueError("variants must be an object keyed by language code")
        variants = {language(k): dict(v) for k, v in variants.items()}
    else:
        variants = {}
        for code, full in LANGUAGES.items():
            aliases = {
                "question": [full + "_instruction"],
                "answer": [full + "_output"],
                "formal_question": [full + "_formal_instruction"],
                "formal_answer": [full + "_formal_output"],
                "informal_question": [full + "_slang_instruction"],
                "informal_answer": [full + "_slang_output"],
                "moderate_answer": [full + "_moderate_output"],
                "short_answer": [full + "_short_output"],
            }
            if code == "fa":
                for field, names in {
                    "question": ["revise_instruction", "revise_question", "instruction"],
                    "answer": ["revise_output", "complete_answer", "output"],
                    "formal_question": ["formal_instruction"],
                    "formal_answer": ["formal_output"],
                    "informal_question": ["slang_instruction"],
                    "informal_answer": ["slang_output"],
                    "moderate_answer": ["moderate_output", "balance_output", "normal_output"],
                }.items():
                    aliases[field] += names
            variants[code] = {
                field: next((text(row.get(k)) for k in keys if text(row.get(k))), None)
                for field, keys in aliases.items()
            }
            variants[code] = {k: v for k, v in variants[code].items() if v is not None}
        variants = {k: v for k, v in variants.items() if v}
    if not variants:
        raise ValueError("No usable language variants found; inspect the schema or supply variants")
    # Preserve identity after normalization or filling additional language variants.
    # Initial content hashing separates numeric ids reused across source partitions.
    canonical = row.get("schema_version") == "wiki-v1"
    source_id = (
        row["id"]
        if canonical
        else digest([row.get("id"), row.get("title"), row.get("context"), variants])
    )
    return {
        "schema_version": "wiki-v1",
        "id": source_id,
        "original_id": row.get("original_id") if canonical else row.get("id"),
        "title": row.get("title", ""),
        "context": row.get("context", ""),
        "variants": variants,
    }


def evaluation_id(row):
    """Identical canonical identity to the paper's retrospective evaluation."""
    s, t = language(row["source_language"]), language(row["target_language"])
    # digest sorts object keys; a list has no keys, matching the historical implementation.
    return digest([LANGUAGES[s], LANGUAGES[t], row["source"], row["reference"]])


def normalize_pair(row):
    out = dict(row)
    mapping = {
        "source_language": "src_language",
        "target_language": "tgt_lang",
        "source": "text",
        "reference": "gold_translation",
    }
    for key, legacy in mapping.items():
        if key not in out and legacy in row:
            out[key] = row[legacy]
    for key in ("source_language", "target_language"):
        out[key] = language(out[key])
    if out["source_language"] == out["target_language"]:
        raise ValueError("Translation pairs must have different source and target languages")
    if not text(out.get("source")) or not text(out.get("reference")):
        raise ValueError("Translation source and reference must be nonempty strings")
    out["id"] = evaluation_id(out)
    if "candidate" in out:
        if out["candidate"] is None:
            out["candidate"] = ""
        if not isinstance(out["candidate"], str):
            raise ValueError("candidate must be a string or null")
    return out
